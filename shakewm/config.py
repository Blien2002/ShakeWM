"""Explicit JSON configuration; smoke dimensions never replace production defaults."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass
class ModelConfig:
    feature_dim: int = 768
    grid: int = 16
    width: int = 1024
    depth: int = 16
    heads: int = 16
    mlp_ratio: int = 4
    imu_mode: str = "v0"
    activation_checkpoint: bool = True

    def __post_init__(self):
        if self.imu_mode not in {"v0", "v1", "none"}:
            raise ValueError("imu_mode must be v0, v1 or none")
        if self.width % self.heads or self.width // self.heads < 6 or self.width // self.heads % 2:
            raise ValueError("RoPE requires an even head dimension >= 6")
        if min(self.grid, self.depth, self.feature_dim) < 1:
            raise ValueError("positive model dimensions required")


@dataclass
class DataConfig:
    context: int = 10
    horizon: int = 10
    visual_hz: int = 10
    acquisition_hz: int = 20
    imu_hz: int = 200
    short_samples: int = 20
    long_samples: int = 600
    camera: str = "third_person"
    eligibility: str = "v0"
    window_stride: int = 1

    def __post_init__(self):
        if self.eligibility not in {"v0", "v1"}:
            raise ValueError("eligibility must be v0 or v1; use matched windows for RGB-only")
        if self.acquisition_hz % self.visual_hz or min(self.context, self.horizon, self.window_stride) < 1:
            raise ValueError("invalid time grid or window size")
        if self.short_samples != 20 or self.long_samples != 600 or self.imu_hz != 200:
            raise ValueError("v1 sensor contract is 200 Hz, 20 short and 600 long samples")


@dataclass
class TrainConfig:
    seed: int = 7
    batch_size: int = 2
    steps: int = 10000
    learning_rate: float = 1e-4
    weight_decay: float = 0.04
    warmup_steps: int = 1000
    tbptt: int = 4
    imu_dropout: float = 0.25
    tf_weight: float = 1.0
    rollout_weight: float = 1.0
    bf16: bool = True
    grad_clip: float = 1.0


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        if self.model.imu_mode == "v1" and self.data.eligibility != "v1":
            raise ValueError("V1 requires long-history eligible windows")
        if self.train.tbptt < 1 or self.train.steps < 1 or self.train.batch_size < 1:
            raise ValueError("positive training sizes required")
        if not 0 <= self.train.imu_dropout <= 1:
            raise ValueError("dropout probability out of range")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        d = json.loads(Path(path).read_text())
        return cls(ModelConfig(**d["model"]), DataConfig(**d["data"]), TrainConfig(**d["train"]))


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
