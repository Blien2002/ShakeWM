"""Pinned official single-image EMA teacher, with explicit mock for CPU fixtures."""
import importlib
import json
from pathlib import Path
import sys
import torch
from torch import nn
from torch.nn import functional as F
from .config import file_sha256

UPSTREAM_COMMIT = "204698b45b3712590f06245fbfba32d3be539812"


class FrozenTeacher(nn.Module):
    def train(self, mode=True):
        return super().train(False)

    @torch.no_grad()
    def encode_numpy(self, rgb):
        x = torch.as_tensor(rgb.copy(), device=self.device).permute(0, 3, 1, 2).float() / 255
        return self(x).float().cpu()


class MockTeacher(FrozenTeacher):
    def __init__(self, grid=2, feature_dim=8, device="cpu"):
        super().__init__()
        self.device = torch.device(device)
        self.grid, self.feature_dim = grid, feature_dim
        projection = torch.sin(torch.arange(3 * feature_dim).float().reshape(3, feature_dim))
        self.register_buffer("projection", projection)
        self.to(self.device).eval()
        self.contract = {"kind": "SYNTHETIC_MOCK_NOT_VJEPA", "version": 1, "patches": grid ** 2,
                         "feature_dim": feature_dim, "grid": grid, "preprocessing": "RGB float [0,1]; adaptive average pool"}

    def forward(self, x):
        pooled = F.adaptive_avg_pool2d(x, (self.grid, self.grid)).flatten(2).transpose(1, 2)
        return pooled @ self.projection


def official_architecture(source_root=None):
    root = Path(source_root) if source_root else Path(__file__).resolve().parents[1] / "third_party/vjepa2"
    if not (root / "app/vjepa_2_1/models/vision_transformer.py").is_file():
        raise FileNotFoundError("official source unavailable; use an editable checkout installation")
    lock = json.loads((Path(__file__).resolve().parents[1] / "upstream.lock.json").read_text())
    if lock["commit"] != UPSTREAM_COMMIT or any(file_sha256(root / p) != sha for p, sha in lock["sha256"].items()):
        raise ValueError("pinned official source hash mismatch")
    sys.path.insert(0, str(root.resolve()))
    module = importlib.import_module("app.vjepa_2_1.models.vision_transformer")
    if not Path(module.__file__).resolve().is_relative_to(root.resolve()):
        raise RuntimeError("an unrelated app module shadows the pinned V-JEPA source")
    return module.vit_base(patch_size=16, img_size=(384, 384), num_frames=64, tubelet_size=2,
                           use_sdpa=True, use_SiLU=False, wide_SiLU=True, uniform_power=False,
                           use_rope=True, img_temporal_dim_size=1, interpolate_rope=True)


def clean_state(state):
    cleaned = {}
    for original, value in state.items():
        key = original
        for prefix in ["module.", "backbone."]:
            if key.startswith(prefix):
                key = key[len(prefix):]
        if key in cleaned:
            raise ValueError("checkpoint prefix cleaning produced duplicate keys")
        cleaned[key] = value
    return cleaned


class OfficialTeacher(FrozenTeacher):
    def __init__(self, checkpoint_path, device="cpu", expected_sha256=None):
        super().__init__()
        if checkpoint_path is None:
            raise ValueError("official teacher requires an explicit local official checkpoint")
        sha = file_sha256(checkpoint_path)
        if expected_sha256 is not None and sha != expected_sha256:
            raise ValueError("checkpoint checksum mismatch")
        self.device = torch.device(device)
        self.encoder = official_architecture()
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if "ema_encoder" not in state:
            raise ValueError("official ViT-B checkpoint must contain ema_encoder")
        self.encoder.load_state_dict(clean_state(state["ema_encoder"]), strict=True)
        self.requires_grad_(False)
        self.to(self.device).eval()
        self.contract = {"kind": "official_vjepa2_1_vit_base_384", "upstream_commit": UPSTREAM_COMMIT,
                         "weight_sha256": sha, "checkpoint_key": "ema_encoder", "input_size": 256,
                         "patches": 256, "feature_dim": 768, "grid": 16, "layer": "last_norm",
                         "extra_norm": False, "token_order": "row_major", "temporal_mode": "single_image_T1",
                         "preprocessing": "RGB [0,1]; require 256x256; ImageNet mean/std; no crop"}

    @torch.no_grad()
    def forward(self, x):
        if tuple(x.shape[1:]) != (3, 256, 256):
            raise ValueError("official image input must be [B,3,256,256]")
        if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
            raise ValueError("input must be finite RGB float [0,1]")
        mean = x.new_tensor([0.485, 0.456, 0.406])[None, :, None, None]
        std = x.new_tensor([0.229, 0.224, 0.225])[None, :, None, None]
        # T=1 selects the official image branch, never bidirectional video features.
        z = self.encoder(((x - mean) / std).unsqueeze(2))
        if tuple(z.shape) != (len(x), 256, 768) or not torch.isfinite(z).all():
            raise RuntimeError("official 256px teacher shape/numeric contract failed")
        return z
