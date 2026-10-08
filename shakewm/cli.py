"""Run with python -m shakewm.cli (train/eval/cache/fixtures/fit-norm)."""
import argparse
import json
from pathlib import Path
import time
import torch
from torch.utils.data import default_collate
from .config import Config, digest_json
from .data import WindowDataset, build_cache, create_synthetic, fit_normalization, read_manifest
from .encoder import MockTeacher, OfficialTeacher
from .engine import (evaluate, load_checkpoint, make_optimizer, save_checkpoint, seed_all,
                     train_microbatch)
from .model import ShakeWM


def make_teacher(args, config):
    if args.encoder == "mock":
        return MockTeacher(config.model.grid, config.model.feature_dim, args.device)
    if (config.model.grid, config.model.feature_dim) != (16, 768):
        raise ValueError("official teacher requires production 256x768 feature dimensions")
    return OfficialTeacher(args.encoder_checkpoint, args.device, args.encoder_sha256)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("fixtures", help="Create explicitly synthetic sensor fixtures")
    f.add_argument("--output", required=True)
    f.add_argument("--seconds", type=float, default=2.0)
    f.add_argument("--image-size", type=int, default=32)
    n = sub.add_parser("fit-norm")
    n.add_argument("--manifest", required=True); n.add_argument("--output", required=True)
    for name in ["cache", "train", "eval"]:
        s = sub.add_parser(name)
        s.add_argument("--config", required=True)
        s.add_argument("--manifest", required=True)
        s.add_argument("--output", required=True)
        s.add_argument("--device", default="cpu")
        s.add_argument("--threads", type=int, default=2)
        s.add_argument("--encoder", choices=["official", "mock"], default="official")
        s.add_argument("--encoder-checkpoint")
        s.add_argument("--encoder-sha256")
        if name != "cache":
            s.add_argument("--cache", help="Frozen feature cache; omit for online encoding")
            s.add_argument("--normalization", required=True)
        if name == "train":
            s.add_argument("--resume")
            s.add_argument("--stop-after", type=int, help="Stop early without changing the configured schedule")
        if name == "eval":
            s.add_argument("--checkpoint", required=True)
            s.add_argument("--split", choices=["val", "test"], default="test")
            s.add_argument("--no-imu", action="store_true", help="Mask-at-test diagnostic; not independently trained RGB-only")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "fixtures":
        print(create_synthetic(args.output, args.seconds, args.image_size)); return
    if args.command == "fit-norm":
        value = fit_normalization(args.manifest)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(value, indent=2) + "\n")
        print(json.dumps({"count": value["count"], "source": value["source"]})); return
    torch.set_num_threads(args.threads)
    config = Config.load(args.config)
    seed_all(config.train.seed)
    if args.command == "cache":
        teacher = make_teacher(args, config)
        index = build_cache(args.manifest, args.output, config.data, teacher)
        print(json.dumps({"episodes": len(index["episodes"]), "teacher": index["teacher"]})); return
    normalization = json.loads(Path(args.normalization).read_text())
    teacher = None if args.cache else make_teacher(args, config)
    split = "train" if args.command == "train" else args.split
    dataset = WindowDataset(args.manifest, split, config.data, normalization, args.cache, teacher)
    if not len(dataset):
        raise ValueError(f"no eligible {split} windows: {dataset.coverage}")
    if dataset.teacher["patches"] != config.model.grid ** 2 or dataset.teacher["feature_dim"] != config.model.feature_dim:
        raise ValueError("teacher/model feature contract mismatch")
    if not dataset.manifest.get("synthetic", False) and dataset.teacher["kind"].startswith("SYNTHETIC"):
        raise ValueError("mock features require a manifest explicitly marked synthetic")
    model = ShakeWM(config.model).to(args.device)
    manifest_hash = digest_json(dataset.manifest)
    if args.command == "eval":
        saved = load_checkpoint(args.checkpoint, model, config, manifest_hash, dataset.teacher, device=args.device)
        if saved["normalization"] != normalization:
            raise ValueError("checkpoint normalization mismatch")
        report = evaluate(model, dataset, config, args.device, config.train.batch_size, args.no_imu)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        print(json.dumps({k: v for k, v in report.items() if k != "windows"}, allow_nan=False)); return

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "latest.pt").exists() and not args.resume:
        raise FileExistsError("run checkpoint exists; use --resume or another output directory")
    optimizer, scheduler = make_optimizer(model, config)
    generator = torch.Generator().manual_seed(config.train.seed + 1)
    step = 0
    if args.resume:
        saved = load_checkpoint(args.resume, model, config, manifest_hash, dataset.teacher,
                                optimizer, scheduler, generator, args.device)
        if saved["normalization"] != normalization:
            raise ValueError("checkpoint normalization mismatch")
        step = saved["global_step"]
    (output / "config.json").write_text(json.dumps(config.to_dict(), indent=2) + "\n")
    (output / "coverage.json").write_text(json.dumps(dataset.coverage, indent=2) + "\n")
    (output / "provenance.json").write_text(json.dumps({"teacher": dataset.teacher, "manifest_hash": manifest_hash,
        "synthetic": dataset.manifest.get("synthetic", False), "torch": str(torch.__version__),
        "device": args.device, "sampler": "uniform windows with replacement; RNG saved"}, indent=2) + "\n")
    finish = min(config.train.steps, args.stop_after) if args.stop_after else config.train.steps
    model.train()
    while step < finish:
        indices = torch.randint(len(dataset), (config.train.batch_size,), generator=generator).tolist()
        batch = default_collate([dataset[i] for i in indices])
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        metrics = train_microbatch(model, batch, config, args.device)
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.train.grad_clip, error_if_nonfinite=True)
        optimizer.step(); scheduler.step(); step += 1
        metrics.update(global_step=step, learning_rate=optimizer.param_groups[0]["lr"],
                       grad_norm=float(norm), seconds=time.perf_counter() - started)
        with (output / "train.jsonl").open("a") as f:
            f.write(json.dumps(metrics, allow_nan=False) + "\n")
        print(json.dumps(metrics, allow_nan=False), flush=True)
    save_checkpoint(output / "latest.pt", model, optimizer, scheduler, step, config, normalization,
                    dataset.teacher, manifest_hash, generator, args.device)


if __name__ == "__main__":
    main()
