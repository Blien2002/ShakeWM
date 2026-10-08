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
    assert actual.shape == (2, 256, 768)
    torch.testing.assert_close(actual, singles, atol=2e-5, rtol=2e-4)
    assert not actual.requires_grad and torch.isfinite(actual).all()
    result = {"pretrained_checkpoint": not args.random_architecture, "shape": list(actual.shape),
              "parameters": sum(p.numel() for p in model.parameters()), "device": args.device,
              "max_batch_error": float((actual - singles).abs().max()), "upstream_commit": UPSTREAM_COMMIT,
              "torch": str(torch.__version__), "teacher": getattr(model, "contract", None)}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
