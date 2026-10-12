# Convolutional RGB decoder

`shakewm_decoder` trains a separate, deterministic convolutional decoder from
frozen visual tokens to RGB. `resize_conv` is the only supported architecture
and the default in `make_decoder`. Training loads neither the world model nor
the visual encoder; input tokens are detached and normalized with FP32,
non-affine per-token layer normalization.

The production contract is row-major `[B, 256, 768]` tokens from a 16×16 grid
and `[B, 3, 256, 256]` RGB output. Four nearest-neighbor resize/convolution
stages reconstruct the image. Metrics use unclamped RGB; only video display
clamps values to [0, 1].

## Prepare and train

Install the package with `pip install -e '.[decoder]'`. Edit
`configs/decoder_conv.json` to point to your real normalized train360/val45
manifests and their existing official V-JEPA2.1 indexed caches. Relative paths
are resolved from the command's working directory. The teacher is a frozen
ViT encoder; this package contains only a convolutional decoder.

```bash
python -m shakewm_decoder.data --config configs/decoder_conv.json
python -m shakewm_decoder.train --config configs/decoder_conv.json \
  --output decoder_runs/conv --device cuda:0
```

CPU preparation pairs 10 Hz cache frame `j` with 20 Hz RGB frame `2j`, preserves
validity/end-time masks, checks camera/teacher/source bindings, and rejects
overlapping train/validation state IDs. It performs no encoding, cropping or
data migration. Existing prepared frame indices may be reused only when their
input identity matches.

The recipe uses FP32 MSE, AdamW, warmup then cosine decay, seed 7, and a unique
valid-frame permutation per epoch. Only decoder parameters are optimized.
Validation includes RGB MSE/L1, a predefined rectangular ROI and adjacent
valid-frame differences. Temporal differences never cross gaps or episodes.

## Resume and evaluate

```bash
python -m shakewm_decoder.train --config configs/decoder_conv.json \
  --output decoder_runs/conv --resume decoder_runs/conv/latest.pt --device cuda:0
python -m shakewm_decoder.evaluate --config configs/decoder_conv.json \
  --checkpoint decoder_runs/conv/best.pt --output decoder_reports/conv \
  --device cuda:0
```

Complete checkpoints contain model, AdamW, RNG, sampler cursor, schedule step
and split/teacher identities. Resume rejects mismatched inputs, runtime or
configuration and rejects failure capsules. Nonfinite loss/gradients stop
before the optimizer update and retain diagnostic evidence; optimizer failures
also stop and record whether an update had started. `--max-updates` provides
an explicit bounded run with a complete checkpoint, without changing the recipe.

Standalone GT reconstruction works with the published ShakeWM package.
Optional `--wm-checkpoint` and `--wm-config` evaluate a frozen C10/H10 predictor
and generate true RGB, GT reconstruction, prediction reconstruction and
copy-last reconstruction. This optional path requires the native-v2 checkpoint
and validation interfaces in the supplied ShakeWM runtime; it fails explicitly
when those interfaces are absent. It never updates the predictor or encoder
and accepts only complete native-v2 checkpoints.

No private paths, data, caches, encoder weights, decoder weights or experiment
outputs are included in this code publication.
