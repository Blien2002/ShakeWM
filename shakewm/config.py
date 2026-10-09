"""Explicit JSON configuration; smoke dimensions never replace production defaults."""
from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path


@dataclass
class ModelConfig:
    """Predictor dimensions and input mode for the ShakeWM transformer."""

    # `feature_dim` and `grid` must match the frozen image teacher's output.
    feature_dim: int = 768
    grid: int = 16
    # Transformer hidden size, block count, attention heads, and MLP expansion ratio.
    width: int = 1024
    depth: int = 16
    heads: int = 16
    mlp_ratio: int = 4
    # Select short+no-long IMU (v0), short+long IMU (v1), or RGB-only (none).
    imu_mode: str = "v0"
    # Recompute block activations during training to reduce peak memory.
    activation_checkpoint: bool = True

    def __post_init__(self):
        """Reject unsupported IMU modes and dimensions that cannot use 3D RoPE."""
        if self.imu_mode not in {"v0", "v1", "none"}:
            raise ValueError("imu_mode must be v0, v1 or none")
        if self.width % self.heads or self.width // self.heads < 6 or self.width // self.heads % 2:
            raise ValueError("RoPE requires an even head dimension >= 6")
        if min(self.grid, self.depth, self.feature_dim) < 1:
            raise ValueError("positive model dimensions required")


@dataclass
class DataConfig:
    """Sampling rates, history/forecast lengths, and sensor-window policy."""

    # C past visual steps and H future targets are sampled at `visual_hz`.
    context: int = 10
    horizon: int = 10
    visual_hz: int = 10
    # Original RGB and IMU acquisition rates, before model-rate subsampling.
    acquisition_hz: int = 20
    imu_hz: int = 200
    # IMU samples per visual step: recent 100 ms and recent 3 s, respectively.
    short_samples: int = 20
    long_samples: int = 600
    # Camera identifier used by the manifest and optional feature cache.
    camera: str = "third_person"
    # v0 requires short IMU; v1 additionally requires a complete long window.
    eligibility: str = "v0"
    # Advance this many model-rate frames between candidate training windows.
    window_stride: int = 1

    def __post_init__(self):
        """Validate the sampling grid and the fixed v1 IMU window contract."""
        if self.eligibility not in {"v0", "v1"}:
            raise ValueError("eligibility must be v0 or v1; use matched windows for RGB-only")
        if self.acquisition_hz % self.visual_hz or min(self.context, self.horizon, self.window_stride) < 1:
            raise ValueError("invalid time grid or window size")
        if self.short_samples != 20 or self.long_samples != 600 or self.imu_hz != 200:
            raise ValueError("v1 sensor contract is 200 Hz, 20 short and 600 long samples")


@dataclass
class TrainConfig:
    """Optimizer schedule, loss weights, precision, and TBPTT settings."""

    # Global RNG seed; the CLI derives a separate deterministic window-sampler seed from it.
    seed: int = 7
    # Microbatch size and total optimizer updates.
    batch_size: int = 2
    steps: int = 10000
    # AdamW schedule parameters.
    learning_rate: float = 1e-4
    weight_decay: float = 0.04
    warmup_steps: int = 1000
    # Number of autoregressive prediction steps between gradient truncations.
    tbptt: int = 4
    # Per-example probability of replacing both available IMU streams with no-IMU tokens.
    imu_dropout: float = 0.25
    # Independent weights for teacher-forced and autoregressive latent L1 losses.
    tf_weight: float = 1.0
    rollout_weight: float = 1.0
    # Enable bfloat16 autocast during model forward/backward.
    bf16: bool = True
    # Maximum global gradient norm before clipping.
    grad_clip: float = 1.0


@dataclass
class Config:
    """Complete run configuration, grouped by model, data, and training concerns."""

    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def __post_init__(self):
        """Validate settings that depend on more than one configuration section."""
        if self.model.imu_mode == "v1" and self.data.eligibility != "v1":
            raise ValueError("V1 requires long-history eligible windows")
        if self.train.tbptt < 1 or self.train.steps < 1 or self.train.batch_size < 1:
            raise ValueError("positive training sizes required")
        if not 0 <= self.train.imu_dropout <= 1:
            raise ValueError("dropout probability out of range")

    def to_dict(self):
        """Return a JSON-serializable snapshot of all run settings."""
        return asdict(self)

    @classmethod
    def load(cls, path):
        """Load the required `model`, `data`, and `train` sections from JSON."""
        settings = json.loads(Path(path).read_text())
        return cls(ModelConfig(**settings["model"]), DataConfig(**settings["data"]),
                   TrainConfig(**settings["train"]))


def digest_json(value):
    """Hash canonical JSON so key order and whitespace do not affect identity."""
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    """Compute a file's SHA-256 incrementally without reading it all into memory."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()
