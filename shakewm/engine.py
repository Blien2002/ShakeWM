"""Separate TF and fixed-origin rollout graphs, segmented TBPTT, reproducible state.

Truncation cuts only the autoregressive chain through predicted blocks. The history blocks
(visual history and IMU tokens) stay attached for the whole rollout, so the loss at every
horizon, including 0.5-1.0 s, trains the history and IMU pathway.
"""
from contextlib import nullcontext
import math
from pathlib import Path
import random
import numpy as np
import torch
from torch.utils.data import default_collate
from .model import KVCache


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def to_device(batch, device):
    return {key: value.to(device) if isinstance(value, torch.Tensor) else value for key, value in batch.items()}


def masked_l1_sum(predicted, target, valid):
    # Invalid values do not participate, even if a caller supplies NaN padding.
    errors = (predicted.float() - target.detach().float()).abs().mean(dim=(-1, -2))
    return torch.where(valid, errors, torch.zeros_like(errors)).sum()


def precision(device, enabled):
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16) if enabled else nullcontext()


def truncate_predicted(cache, history_kv, history_tokens):
    """Detach the K/V of predicted blocks while keeping the history K/V attached.

    ``history_kv`` holds the prefill tensors themselves (not slices of later concatenations), so
    the graph of the previous segment is not kept alive through them.
    """
    layers = []
    for (hk, hv), (k, v) in zip(history_kv, cache.layers):
        layers.append((torch.cat([hk, k[..., history_tokens:, :].detach()], -2),
                       torch.cat([hv, v[..., history_tokens:, :].detach()], -2)))
    return KVCache(layers, cache.blocks, cache.limit)


def train_microbatch(model, batch, config, device, boundary_hook=None):
    t = config.train
    batch = to_device(batch, device)
    history = batch["history"]
    dropped = torch.rand(len(history), device=device) < t.imu_dropout
    tf_count, roll_count = batch["tf_mask"].sum(), batch["target_mask"].sum()
    if tf_count == 0 or roll_count == 0:
        raise ValueError("batch has no valid supervised transitions")
    with precision(device, t.bf16):
        prediction, tf_cache = model(history, batch["short"], batch["long"], batch["eligible"], dropped,
                                     cache_limit=history.shape[1])
        tf_loss = masked_l1_sum(prediction, batch["tf_targets"], batch["tf_mask"]) / tf_count
        weighted_tf = t.tf_weight * tf_loss
    weighted_tf.backward()
    tf_value = tf_loss.detach().item()
    del prediction, tf_cache, tf_loss, weighted_tf

    h = config.data.horizon
    history_tokens = history.shape[1] * (config.model.grid ** 2 + 2)
    with precision(device, t.bf16):
        prefill, cache = model(history, batch["short"], batch["long"], batch["eligible"], dropped,
                               cache_limit=history.shape[1] + h - 1)
        last = prefill[:, -1:]
    del prefill
    # Prefill K/V of the history blocks (visual history + IMU tokens). They stay attached until the
    # final segment so that later segments also send gradients into the history and IMU encoder.
    history_kv = [(k, v) for k, v in cache.layers]
    segment_loss = None
    roll_value = 0.0
    for step in range(h):
        if step:
            with precision(device, t.bf16):
                last, cache = model(last, cache=cache)
        loss = t.rollout_weight * masked_l1_sum(last, batch["targets"][:, step:step + 1],
                                               batch["target_mask"][:, step:step + 1]) / roll_count
        roll_value += loss.detach().item()
        segment_loss = loss if segment_loss is None else segment_loss + loss
        del loss
        final = step == h - 1
        if (step + 1) % t.tbptt == 0 or final:
            # Backward each segment now; keep the history graph for later segments. No parameter
            # update happens until all segments finish (gradients accumulate).
            segment_loss.backward(retain_graph=not final)
            del segment_loss
            segment_loss = None
            last = last.detach()
            if final:
                cache = cache.detach()
            else:
                # Cut the chain through predicted blocks only; history stays attached.
                cache = truncate_predicted(cache, history_kv, history_tokens)
            if boundary_hook:
                boundary_hook(step + 1, last, cache)
    del last, cache, history_kv
    return {"tf_l1": tf_value, "weighted_rollout_l1": roll_value,
            "tf_valid": int(tf_count), "rollout_valid": int(roll_count),
            "tf_context_lengths": list(range(1, history.shape[1] + 1))}


def make_optimizer(model, config):
    t = config.train
    optimizer = torch.optim.AdamW(model.parameters(), lr=t.learning_rate, weight_decay=t.weight_decay)

    def factor(step):
        if step < t.warmup_steps:
            return (step + 1) / max(1, t.warmup_steps)
        progress = (step - t.warmup_steps) / max(1, t.steps - t.warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    return optimizer, torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def save_checkpoint(path, model, optimizer, scheduler, step, config, normalization, teacher, manifest_hash,
                    generator, device="cpu"):
    n = np.random.get_state()
    rng = {"python": random.getstate(), "numpy": [n[0], n[1].tolist(), n[2], n[3], n[4]],
           "torch": torch.get_rng_state(), "sampler": generator.get_state()}
    if torch.device(device).type == "cuda":
        rng["cuda"] = torch.cuda.get_rng_state_all()
    state = {"format": "shakewm.checkpoint.v1", "model": model.state_dict(), "optimizer": optimizer.state_dict(),
             "scheduler": scheduler.state_dict(), "global_step": step, "config": config.to_dict(),
             "normalization": normalization, "teacher": teacher, "manifest_hash": manifest_hash, "rng": rng}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.pt")
    torch.save(state, temporary)
    temporary.replace(path)


def load_checkpoint(path, model, config, manifest_hash, teacher, optimizer=None, scheduler=None,
                    generator=None, device="cpu"):
    state = torch.load(path, map_location="cpu", weights_only=True)
    if state["format"] != "shakewm.checkpoint.v1" or state["config"] != config.to_dict():
        raise ValueError("checkpoint format/config mismatch; resume must keep the training schedule")
    if state["manifest_hash"] != manifest_hash or state["teacher"] != teacher:
        raise ValueError("checkpoint split/teacher mismatch")
    model.load_state_dict(state["model"], strict=True)
    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        r = state["rng"]
        random.setstate(r["python"])
        n = r["numpy"]
        np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), n[2], n[3], n[4]))
        torch.set_rng_state(r["torch"])
        generator.set_state(r["sampler"])
        if torch.device(device).type == "cuda" and "cuda" in r:
            torch.cuda.set_rng_state_all(r["cuda"])
    return state


@torch.no_grad()
def evaluate(model, dataset, config, device, batch_size=2, no_imu=False):
    model.eval()
    horizon = config.data.horizon
    sums, copies, counts = torch.zeros(horizon), torch.zeros(horizon), torch.zeros(horizon, dtype=torch.long)
    records = []
    for start in range(0, len(dataset), batch_size):
        batch = to_device(default_collate([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))]), device)
        dropped = torch.ones(len(batch["history"]), dtype=torch.bool, device=device) if no_imu else None
        with precision(device, config.train.bf16):
            pred = model.rollout(batch["history"], batch["short"], batch["long"], batch["eligible"], horizon, dropped)
        errors = (pred.float() - batch["targets"].float()).abs().mean((-1, -2))
        copy_errors = (batch["history"][:, -1:].float() - batch["targets"].float()).abs().mean((-1, -2))
        mask = batch["target_mask"]
        sums += torch.where(mask, errors, 0).sum(0).cpu()
        copies += torch.where(mask, copy_errors, 0).sum(0).cpu()
        counts += mask.sum(0).cpu()
        for i in range(len(pred)):
            records.append({"episode_id": batch["episode_id"][i], "state_id": batch["state_id"][i],
                            "origin_time": float(batch["origin_time"][i]), "valid": mask[i].tolist(),
                            "l1": [float(errors[i, j]) if mask[i, j] else None for j in range(horizon)]})
    return {"protocol": "fixed_origin_open_loop", "mode": "masked_imu_diagnostic" if no_imu else model.config.imu_mode,
            "synthetic": dataset.manifest.get("synthetic", False), "teacher": dataset.teacher,
            "coverage": dataset.coverage, "per_horizon": [
                {"horizon": i + 1, "seconds": (i + 1) / config.data.visual_hz, "count": int(counts[i]),
                 "latent_l1": float(sums[i] / counts[i]) if counts[i] else None,
                 "copy_last_l1": float(copies[i] / counts[i]) if counts[i] else None} for i in range(horizon)],
            "windows": records}
