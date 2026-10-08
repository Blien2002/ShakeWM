"""Check the official single-image 256px contract; never download weights implicitly."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from shakewm.encoder import OfficialTeacher, official_architecture, UPSTREAM_COMMIT


def main():
    p = argparse.ArgumentParser(description=__doc__)
    choice = p.add_mutually_exclusive_group(required=True)
    choice.add_argument("--checkpoint")
    choice.add_argument("--random-architecture", action="store_true")
    p.add_argument("--sha256")
    p.add_argument("--device", default="cpu")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    torch.set_num_threads(2); torch.manual_seed(7)
    if args.random_architecture:
        model = official_architecture().to(args.device).eval().requires_grad_(False)
        x = torch.randn(2, 3, 1, 256, 256, device=args.device)
    else:
        model = OfficialTeacher(args.checkpoint, args.device, args.sha256)
        x = torch.rand(2, 3, 256, 256, device=args.device)
    with torch.no_grad():
        actual = model(x)
        singles = torch.cat([model(x[i:i+1]) for i in range(2)])
    shape_ok = actual.shape == (2, 256, 768) and singles.shape == actual.shape
    finite = bool(torch.isfinite(actual).all() and torch.isfinite(singles).all())
    frozen = not actual.requires_grad and not singles.requires_grad and all(
        not p.requires_grad for p in model.parameters())
    atol, rtol = 2e-5, 2e-4
    batch_error = None
    if shape_ok and finite:
        delta = (actual - singles).float()
        close = torch.isclose(actual, singles, atol=atol, rtol=rtol)
        batch_error = {"atol": atol, "rtol": rtol,
                       "max_abs": float(delta.abs().max()),
                       "rms": float(delta.square().mean().sqrt()),
                       "mean_abs": float(delta.abs().mean()),
                       "mismatched_elements": int((~close).sum()),
                       "total_elements": actual.numel()}
    batch_consistent = batch_error is not None and batch_error["mismatched_elements"] == 0
    passed = shape_ok and finite and frozen and batch_consistent
    result = {"pretrained_checkpoint": not args.random_architecture, "shape": list(actual.shape),
              "parameters": sum(p.numel() for p in model.parameters()), "device": args.device,
              "max_batch_error": None if batch_error is None else batch_error["max_abs"],
              "upstream_commit": UPSTREAM_COMMIT, "passed": passed,
              "checks": {"shape": shape_ok, "finite": finite, "frozen": frozen,
                         "batch_consistency": batch_consistent,
                         "strict_checkpoint_load": True if not args.random_architecture else None},
              "batch_error": batch_error, "input_seed": 7,
              "torch": str(torch.__version__), "teacher": getattr(model, "contract", None)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
    if not passed:
        raise SystemExit("Official encoder gate failed; diagnostics saved to " + args.output)


if __name__ == "__main__":
    main()
