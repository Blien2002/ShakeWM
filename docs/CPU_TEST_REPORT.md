# CPU acceptance — 2026-10-08

## Verified

`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q`: **23 passed in 1.34 s** on CPU. Main environment: Python 3.13.5, PyTorch 2.9.1+cu128, NumPy 1.26.4, pytest 8.3.4. CUDA was not used.

Coverage includes delivery-time alignment, negative reset prefill, missing-sample rejection, V1 full-context eligibility, state/seed split isolation, train-only normalization, variable termination masks, future RGB/IMU/GT/seed invariance, feature/cache shape and checksum contracts, same-block visibility, future-block exclusion, direct attention numerical reference, incremental-cache/full-recomputation agreement, cache capacity, detached cache/latent gradient boundaries, four-step backward with and without activation checkpointing, BF16 CPU backward, V1 long-branch gradients, anti-alias response, strict EMA key loading, frozen teacher behavior, and exact optimizer/scheduler/RNG/next-update restoration.

Production config was instantiated on the `meta` device to verify 16 layers / width 1024 / 16 heads / 256 patches / 768 feature dimensions without allocating the full model. This is a shape/parameter check, not production training.

## Synthetic executable loop

`bash scripts/smoke.sh runs/acceptance` completed. Four small, generated sensor episodes use disjoint state/seed splits. The smoke predictor has 2 layers / width 32 / 4 patches / 8 feature dimensions, C=3, H=6. It uses the explicitly labeled mock teacher, not V-JEPA features.

- Frozen feature-cache extraction completed for four episodes.
- V0 ran step 1, saved, restored, and completed steps 2–3.
- Independently initialized/trained RGB-only and V1 each completed three updates.
- All three ran fixed-origin evaluation with copy-last and horizon counts.
- Online mock encoding completed one train update.
- V0 test-time no-IMU diagnostic was also run separately on the initial 2-second fixture.

The four-second smoke test has 37 V0/RGB-only origins and 8 V1 origins; counts decrease across horizons due to termination masking. V1 coverage differs, so these smoke scores must not be used to claim V1 improvement. For scientific comparisons, retrain all models with the same V1 eligibility filter and frozen split.

V0 one-step test L1 was approximately **0.5598**, versus copy-last **0.007344**. The smoke model is worse than persistence, as expected for three tiny updates on simple synthetic frames. Successful execution is the only acceptance claim.

Machine-readable evidence: [cpu_smoke_results.json](cpu_smoke_results.json). Local raw fixtures, caches and model checkpoints are ignored by Git and are not published. The cache-reader memory refactor was checked against the saved V0 evaluation: the report matched exactly.

## Official encoder code path

Reused the existing `vla-adapter` environment solely for this CPU check (Python 3.10.16, PyTorch 2.2.0+cu121, timm 0.9.10); installed no new dependencies. The main runner's supported environment remains the version above.

```bash
CUDA_VISIBLE_DEVICES='' python3 scripts/check_official_encoder.py --random-architecture --device cpu --output docs/official_random_cpu.json
```

The exact pinned official V-JEPA 2.1 ViT-B encoder, **with random weights**, accepts `[2,3,1,256,256]` and returns `[2,256,768]`. Single-frame versus batched output agrees (maximum absolute difference 0 on this CPU run). Full instantiated encoder parameters: 86,833,152. All eight vendored files have verified SHA256 matches against the pinned source manifest. See [official_random_cpu.json](official_random_cpu.json).

This does **not** verify the released pretrained checkpoint's contents, strict full checkpoint compatibility, pretrained feature quality, or GPU behavior. Unit tests verify strict key handling with a small fake architecture; the real checkpoint still requires `scripts/check_official_encoder.py --checkpoint ...`.

## Remaining integration gates

1. Real checkpoint and official-weight online/cache smoke on a separately scheduled GPU.
2. Collector-native files to normalized NPZ adapter against an authorized real sample; camera identity, sensor units/extrinsics, filter warmup, timestamps and event boundary must be verified at export.
3. Formal state/seed split manifest and real train normalization after valid data arrive. Current repository does not invent or consume the pending 1350-slot collection.
4. Production C16/H10 forward/backward peak memory, throughput and checkpoint-save disk budget. No DDP or gradient-accumulation launcher is included in this initial single-device implementation.

No collection job, GPU job, existing ShakeBench project, old dataset or model weight was modified. Library DOCX materialization returned HTTP 403; the complete 12-page revised document was instead read through authenticated Library text access before implementation. The source design is not included in this Git repository.
