"""Single-GPU/CPU decoder-only training with complete-update recovery."""
import argparse
import json
import math
from pathlib import Path
import random
import time
import traceback
import numpy as np
import torch
from torch.utils.data import default_collate
from shakewm.config import digest_json
from .data import FrameDataset, check_split_isolation, epoch_batches
from .models import make_decoder, normalize_tokens
from .metrics import ErrorTotals
from .util import atomic_json, atomic_save, cpu_tree, rng_state, restore_rng, sync, memory


def learning_rate(step, total, warmup, peak, floor):
    if step < warmup: return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup - 1)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(model, cfg):
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        (no_decay if parameter.ndim < 2 or name.endswith('.bias') else decay).append(parameter)
    return torch.optim.AdamW([{'params': decay, 'weight_decay': cfg['weight_decay']},
        {'params': no_decay, 'weight_decay': 0.0}], lr=cfg['peak_lr'],
        betas=tuple(cfg['betas']), eps=cfg['epsilon'], foreach=False)


def finite_gradients(model):
    bad = [name for name, p in model.named_parameters() if p.grad is not None and not torch.isfinite(p.grad).all()]
    if bad: raise FloatingPointError('nonfinite gradients: ' + ','.join(bad[:10]))


def semantic_config(cfg):
    return {k: cfg[k] for k in ['architecture', 'train', 'roi', 'feature_space']}


def checkpoint(model, optimizer, cfg, train, val, epoch, cursor, step, best, stale, device):
    return {'format': 'shakewm.decoder.complete.v1', 'complete_update': True,
        'config': semantic_config(cfg), 'train_identity': train.index['identity'],
        'val_identity': val.index['identity'], 'teacher': train.index['teacher'],
        'model': cpu_tree(model.state_dict()), 'optimizer': cpu_tree(optimizer.state_dict()),
        'epoch': epoch, 'cursor': cursor, 'global_step': step, 'best_val_mse': best,
        'stale_epochs': stale, 'rng': rng_state(device),
        'runtime': {'torch': torch.__version__, 'device_type': device.type},
        'sampler': 'unique_valid_frame_permutation_seed_plus_epoch_no_prefetch',
        'wm_or_encoder_updated': False}


def restore(state, model, optimizer, cfg, train, val, device):
    if state.get('format') != 'shakewm.decoder.complete.v1' or state.get('complete_update') is not True:
        raise ValueError('ordinary resume rejects invalid diagnostic/WM checkpoints')
    if state['config'] != semantic_config(cfg) or state['train_identity'] != train.index['identity'] or state['val_identity'] != val.index['identity']:
        raise ValueError('resume configuration/data split differs')
    if state['teacher'] != train.index['teacher'] or state['runtime']['torch'] != torch.__version__ or state['runtime']['device_type'] != device.type:
        raise ValueError('resume teacher/runtime differs; no implicit conversion')
    if not all(torch.isfinite(p).all() for p in state['model'].values()): raise ValueError('nonfinite saved parameters')
    model.load_state_dict(state['model'], strict=True)
    optimizer.load_state_dict(state['optimizer']); restore_rng(state['rng'], device)


@torch.no_grad()
def reconstruct(model, dataset, device, batch_size, roi):
    was_training = model.training; model.eval()
    total, delta = ErrorTotals(roi), ErrorTotals(roi)
    previous = None
    try:
        for start in range(0, len(dataset), batch_size):
            batch = default_collate([dataset[i] for i in range(start, min(start + batch_size, len(dataset)))])
            target = batch['rgb'].to(device).float() / 255
            pred = model(normalize_tokens(batch['latent'].to(device)))
            total.add(pred, target)
            for i in range(len(pred)):
                episode, frame = int(batch['episode'][i]), int(batch['frame'][i])
                if previous is not None and previous[0] == episode and previous[1] + 1 == frame:
                    delta.add((pred[i] - previous[2])[None], (target[i] - previous[3])[None])
                previous = (episode, frame, pred[i].detach().clone(), target[i].detach().clone())
    finally:
        model.train(was_training)
    return {'GT_latent_to_RGB': total.report(), 'adjacent_valid_frame_delta': delta.report(),
        'roi_kind': 'predefined_rectangle_not_object_mask', 'roi_xyxy_normalized': roi,
        'outputs_clamped_for_metrics': False, 'split': dataset.index['split']}


def main(argv=None):
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument('--config', required=True); p.add_argument('--output', required=True)
    p.add_argument('--device', default='cuda:0'); p.add_argument('--resume')
    p.add_argument('--max-updates', type=int, help='bounded smoke; same complete state can explicitly continue')
    p.add_argument('--threads', type=int, default=2); a = p.parse_args(argv)
    cfg = json.loads(Path(a.config).read_text()); tc = cfg['train']
    if a.max_updates is not None and a.max_updates < 1: raise ValueError('max-updates must be positive')
    if tc['precision'] != 'float32' or tc['loss'] != 'mse': raise ValueError('v1 recipe is FP32 pure MSE')
    torch.set_num_threads(a.threads)
    device = torch.device(a.device); output = Path(a.output)
    if device.type == 'cuda' and torch.cuda.device_count() != 1: raise ValueError('exactly one guarded visible GPU required')
    if not a.resume:
        output.mkdir(parents=True, exist_ok=False)
    else:
        if not output.is_dir() or (output / 'failure.json').exists(): raise ValueError('resume requires own healthy run directory')
    train = FrameDataset(Path(cfg['inputs']['train_frames']) / 'index.json')
    val = FrameDataset(Path(cfg['inputs']['val_frames']) / 'index.json')
    check_split_isolation(train.index, val.index)
    for dataset, split, count in [(train, 'train', 360), (val, 'val', 45)]:
        if dataset.index['split'] != split or len(dataset.index['episodes']) != count:
            raise ValueError('complete real train360/val45 prepared inputs required')
    if not len(train) or not len(val): raise ValueError('empty real frame split')
    random.seed(tc['seed']); np.random.seed(tc['seed']); torch.manual_seed(tc['seed'])
    if device.type == 'cuda': torch.cuda.manual_seed(tc['seed'])
    model = make_decoder(cfg).to(device); optimizer = make_optimizer(model, tc)
    epoch = cursor = step = stale = 0; best = None
    if a.resume:
        state = torch.load(a.resume, map_location='cpu', weights_only=False)
        restore(state, model, optimizer, cfg, train, val, device)
        epoch, cursor, step, best, stale = (state[k] for k in ['epoch', 'cursor', 'global_step', 'best_val_mse', 'stale_epochs'])
        # Never silently rewind a log ahead of its last complete checkpoint.
        log = output / 'train.jsonl'
        if log.exists():
            rows = [json.loads(line) for line in log.read_text().splitlines() if line.strip()]
            if rows and rows[-1]['step'] > step: raise ValueError('log is ahead of checkpoint; preserve and report')
    steps_per_epoch = math.ceil(len(train) / tc['batch_size'])
    total = steps_per_epoch * tc['max_epochs']; warmup = min(tc['warmup_steps'], steps_per_epoch)
    stop = min(total, step + a.max_updates) if a.max_updates is not None else total
    if tc['early_stop_patience'] and epoch >= tc['min_epochs'] and stale >= tc['early_stop_patience']:
        stop = step
    atomic_json(output / 'config.json', cfg)
    atomic_json(output / 'provenance.json', {'kind': 'independent_decoder_training', 'feature_space': cfg['feature_space'],
        'train_frames': len(train), 'val_frames': len(val), 'train_episodes': len(train.index['episodes']),
        'val_episodes': len(val.index['episodes']), 'teacher': train.index['teacher'],
        'train_identity': train.index['identity'], 'val_identity': val.index['identity'],
        'parameters': sum(p.numel() for p in model.parameters()), 'max_frame_presentations': len(train) * tc['max_epochs'],
        'max_optimizer_updates': total, 'warmup_updates': warmup, 'trainable_modules': ['decoder'],
        'wm_or_encoder_loaded_in_training': False, 'private_data_upload': False})
    batch = None; phase = 'initialization'; pre_rng = None; update_started = False
    try:
        while epoch < tc['max_epochs'] and step < stop:
            for indices, next_cursor in epoch_batches(len(train), tc['batch_size'], tc['sampler_seed'], epoch, cursor):
                sync(device); started = time.perf_counter(); pre_rng = rng_state(device); update_started = False
                phase = 'read_batch'; batch = default_collate([train[i] for i in indices]); read_done = time.perf_counter()
                z = normalize_tokens(batch['latent'].to(device)); target = batch['rgb'].to(device).float() / 255
                optimizer.zero_grad(set_to_none=True); model.train(); phase = 'forward_backward'
                lr = learning_rate(step, total, warmup, tc['peak_lr'], tc['min_lr'])
                for group in optimizer.param_groups: group['lr'] = lr
                prediction = model(z); loss = (prediction - target).square().mean()
                if not torch.isfinite(loss): raise FloatingPointError('nonfinite MSE; optimizer not stepped')
                loss.backward(); finite_gradients(model)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc['grad_clip'], error_if_nonfinite=True)
                phase = 'optimizer'; update_started = True; optimizer.step()
                if any(not torch.isfinite(parameter).all() for parameter in model.parameters()):
                    raise FloatingPointError('nonfinite parameters after optimizer; diagnostic is not a pre-update state')
                step += 1; cursor = next_cursor; sync(device)
                record = {'step': step, 'epoch': epoch, 'cursor': cursor, 'valid_frames': len(indices), 'indices': indices,
                    'mse': float(loss.detach()), 'grad_norm_before_clip': float(norm), 'lr': lr,
                    'read_seconds': read_done - started, 'step_seconds': time.perf_counter() - started,
                    'finite': True, 'optimizer_updated': True, **memory(device)}
                with (output / 'train.jsonl').open('a') as f: f.write(json.dumps(record, allow_nan=False) + '\n'); f.flush()
                if step % tc['checkpoint_every_updates'] == 0 or step == stop:
                    atomic_save(output / 'latest.pt', checkpoint(model, optimizer, cfg, train, val, epoch, cursor, step, best, stale, device))
                if step % 10 == 0 or step == 1:
                    atomic_json(output / 'progress.json', {**record, 'phase': 'training', 'max_updates': total})
                    print(json.dumps(record), flush=True)
                if step >= stop: break
            if cursor == len(train):
                phase = 'validation'; validation = reconstruct(model, val, device, tc['eval_batch_size'], cfg['roi'])
                mse = validation['GT_latent_to_RGB']['mse']
                improved = best is None or mse < best * (1 - tc['min_relative_improvement'])
                best, stale = (mse, 0) if improved else (best, stale + 1)
                epoch += 1; cursor = 0
                atomic_json(output / 'eval' / f'epoch_{epoch:03d}.json', validation)
                state = checkpoint(model, optimizer, cfg, train, val, epoch, cursor, step, best, stale, device)
                atomic_save(output / f'epoch_{epoch:03d}.pt', state); atomic_save(output / 'latest.pt', state)
                if improved: atomic_save(output / 'best.pt', state)
                if tc['early_stop_patience'] and epoch >= tc['min_epochs'] and stale >= tc['early_stop_patience']:
                    break
        atomic_save(output / 'latest.pt', checkpoint(model, optimizer, cfg, train, val, epoch, cursor, step, best, stale, device))
        atomic_json(output / 'receipt.json', {'phase': 'completed', 'global_step': step, 'epoch': epoch, 'cursor': cursor,
            'complete_update': True, 'bounded_smoke': a.max_updates is not None, 'best_val_mse': best,
            'decoder_architecture': 'resize_conv', **memory(device)})
    except BaseException as error:
        # Publish the small failure receipt first, even if an OOM prevents copying
        # the invalid model/optimizer state. Never treat the last checkpoint as a
        # safe automatic restart after a numerical or memory failure.
        failure = {'phase': phase, 'step': step, 'error': repr(error),
            'optimizer_step_started': update_started, 'ordinary_resume_allowed': False,
            'traceback': traceback.format_exc(), 'capsule_saved': False, **memory(device)}
        atomic_json(output / 'failure.json', failure)
        diagnostic = {'format': 'shakewm.decoder.failure.v1', 'complete_update': False, 'ordinary_resume_allowed': False,
            'phase': phase, 'global_step': step, 'epoch': epoch, 'cursor': cursor, 'error': repr(error),
            'optimizer_step_started': update_started, 'pre_forward_rng': pre_rng, 'batch': batch,
            'config': semantic_config(cfg)}
        try:
            diagnostic.update(model=cpu_tree(model.state_dict()), optimizer=cpu_tree(optimizer.state_dict()))
            atomic_save(output / 'failure.pt', cpu_tree(diagnostic))
            failure['capsule_saved'] = True
        except BaseException as capture_error:
            failure['capsule_error'] = repr(capture_error)
        atomic_json(output / 'failure.json', failure)
        raise


if __name__ == '__main__': main()
