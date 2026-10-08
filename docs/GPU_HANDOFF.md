# Official checkpoint verification status

No ShakeWM GPU test has been run. Further testing and remote source upload were canceled at the user’s request. The code and CPU results do not establish production VRAM requirements, GPU optimizer success, GPU BF16 behavior or model quality.

## Verified checkpoint identity

The existing official checkpoint was inspected in place; no additional copy was downloaded for this task.

| Property | Verified value |
| --- | --- |
| Filename | `vjepa2_1_vitb_dist_vitG_384.pt` |
| File size | 1664223428 bytes |
| SHA256 | `848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d` |
| Upstream commit | `204698b45b3712590f06245fbfba32d3be539812` |
| Checked environment | Python 3.12, torch 2.7.1+cu126, CPU execution |
| Checkpoint encoder key | `ema_encoder`, 158 tensor entries |

The published checkpoint URL is [the official Meta file](https://dl.fbaipublicfiles.com/vjepa2/vjepa2_1_vitb_dist_vitG_384.pt). The loader requires an explicit local file and supports SHA256 validation; it does not silently fetch weights.

## Partial CPU result; numerical gate failed

Strict loading into the pinned official architecture succeeded, and normalized single-image inputs produced finite `[2,256,768]` outputs. The current batch-versus-single numerical gate did not pass: 2 of 393216 elements exceeded `atol=2e-5, rtol=2e-4`, with a maximum absolute difference of approximately `5.41e-5`. This records the previously observed partial CPU check; the revised diagnostic-writing script was not run against the checkpoint in this execution task. It must not be presented as a passed official-encoder acceptance gate.

`scripts/check_official_encoder.py` is written to save shape, finite/frozen checks, strict-load status and batch-error diagnostics before returning a failing exit code. The thresholds remain unchanged. Its revised diagnostic-writing behavior was not exercised against the official checkpoint in this execution task. No additional backend or FP64 reference experiment was performed.

No cause has been confirmed, and no new tolerance has been adopted. The discrepancy remains an explicit unresolved check.

The predictor configuration remains the intended 16-layer, width-1024, 16-head architecture with 256 visual patches and two IMU slots. Mock CPU results exercise the interface and causal contracts only. The two existing native clips are too short for production context and V1 history requirements. Real-encoder predictor training and GPU validation remain untested.
