# CLI reference and verification status

Run these interfaces from the repository root with your existing Python environment. The examples describe the supported interfaces; they are not evidence of additional tests. Keep recordings, feature caches, normalization files and checkpoints outside the source tree or in an ignored data directory. No weights are downloaded implicitly.

## Native import

Use an explicit native manifest, a recording directory, or a directory containing recording directories. Freeze the split plan before creating model windows. The importer writes a new directory and refuses to replace an existing converted dataset.

```bash
python3 -m shakewm.cli plan-native-splits --source /path/to/native-recordings --output /path/to/split_plan.json
python3 -m shakewm.cli import-native --source /path/to/native-recordings \
  --split-plan /path/to/split_plan.json --camera main --output /path/to/converted
python3 -m shakewm.cli fit-norm --manifest /path/to/converted/manifest.json \
  --output /path/to/normalization.json
```

`--camera wrist` selects the alternate RGB view. The configuration’s camera field must match the imported manifest. See [the native contract](NATIVE_IMPORT.md) for timing, physical-state grouping and termination rules.

## Frozen official features, training and evaluation

Supply a local official checkpoint and its verified SHA256. Replace the example paths with your own data and output paths. `configs/v0.json` is the production short-IMU configuration; do not substitute the small mock configuration for a production architecture.

```bash
python3 -m shakewm.cli cache --config configs/v0.json \
  --manifest /path/to/converted/manifest.json --encoder official \
  --encoder-checkpoint /path/to/vjepa2_1_vitb_dist_vitG_384.pt \
  --encoder-sha256 848a77c33cc9e6649ed2119c9bea1e2c569bcdab9539ff3e7c02ccc2959ddf4d \
  --device cpu --output /path/to/feature-cache
python3 -m shakewm.cli train --config configs/v0.json \
  --manifest /path/to/converted/manifest.json --normalization /path/to/normalization.json \
  --cache /path/to/feature-cache --device cpu --output /path/to/run
python3 -m shakewm.cli train --config configs/v0.json \
  --manifest /path/to/converted/manifest.json --normalization /path/to/normalization.json \
  --cache /path/to/feature-cache --device cpu --resume /path/to/run/latest.pt --output /path/to/run
python3 -m shakewm.cli eval --config configs/v0.json \
  --manifest /path/to/converted/manifest.json --normalization /path/to/normalization.json \
  --cache /path/to/feature-cache --device cpu --checkpoint /path/to/run/latest.pt \
  --split test --output /path/to/test.json
python3 -m shakewm.cli eval --config configs/v0.json \
  --manifest /path/to/converted/manifest.json --normalization /path/to/normalization.json \
  --cache /path/to/feature-cache --device cpu --checkpoint /path/to/run/latest.pt \
  --split test --no-imu --output /path/to/test-no-imu.json
```

These production commands require sufficient eligible data and compute; they have not been run as a production training experiment. Online encoding is supported by omitting `--cache` and supplying `--encoder official`, `--encoder-checkpoint` and `--encoder-sha256` to train/eval. A cached-feature checkpoint is not interchangeable with an online-feature checkpoint because their feature-source contracts differ. Checkpoint resume requires the same data, feature source and configuration.

`--stop-after` bounds the current training invocation. Training writes `latest.pt` on normal completion or at that boundary, using `latest.tmp.pt` for atomic replacement. Evaluation’s `--output` is a JSON filename and overwrites that file if it already exists. `--split train` supports an explicitly labeled diagnostic when no held-out states are available. Mock features on native recordings additionally require `--allow-mock-for-contract-test`; this is contract testing only.

## Verification record

The earlier CPU suite passed 34 tests. After the final static-review fixes, a separate execution task passed four targeted regressions: equivalent quaternions and pose labels, stale external split-plan summaries, modified/reindexed feature-cache checkpoint rejection, and raw-data-content checkpoint rejection. It also reran the native-recording mock cache/train/resume/eval interface successfully. This does not claim a fresh full-suite run or production model performance.

The pretrained official encoder strictly loaded and produced finite `[2,256,768]` CPU features. Its batch-versus-single numerical consistency gate remains failed: two of 393216 values exceeded the existing `atol=2e-5, rtol=2e-4`, with a maximum absolute difference of approximately `5.41e-5`. This is the previously recorded partial CPU result; the revised diagnostic-writing script was not run against the checkpoint in this execution task, so its expanded diagnostics remain unverified. The gate thresholds remain unchanged. See [the checkpoint status](GPU_HANDOFF.md) and [aggregate CPU evidence](NATIVE_CPU_REPORT.json).

No GPU test was performed. Further testing, numerical diagnostic experiments and source upload were canceled at the user’s request. The existing results are retained without claiming additional verification.
