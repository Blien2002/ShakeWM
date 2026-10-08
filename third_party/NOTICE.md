# V-JEPA 2.1 provenance

Official repository: https://github.com/facebookresearch/vjepa2

Pinned commit: `204698b45b3712590f06245fbfba32d3be539812`.

Only the image encoder's small import closure and `src/hub/backbones.py` reference factory are retained. Source bytes are unchanged and individually SHA256-pinned in `upstream.lock.json`. The predictor, IMU modules, data adapter and runners in `shakewm/` are new code.

V-JEPA is primarily MIT-licensed. The upstream README names three Apache-2.0 augmentation/worker files; none is used here. Both upstream license files are preserved for clarity, and original file headers remain intact.

The official ViT-B Hub entry is `vjepa2_1_vit_base_384`; published weights are `vjepa2_1_vitb_dist_vitG_384.pt` at the official `dl.fbaipublicfiles.com/vjepa2/` endpoint, key `ema_encoder`. The pinned factory's base URL is localhost; ShakeWM never executes that download route. No model weights are redistributed in this repository.
