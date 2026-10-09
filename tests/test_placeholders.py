"""Missing-IMU and future-step placeholders are separate learned tokens."""
import pytest
import torch
from shakewm.config import Config, ModelConfig
from shakewm.engine import train_microbatch
from shakewm.imu import IMUTokens
from shakewm.model import ShakeWM


def tiny(mode="v1"):
    return ModelConfig(feature_dim=8, grid=2, width=32, depth=2, heads=4, imu_mode=mode,
                       activation_checkpoint=False)


def test_missing_and_future_placeholders_are_distinct():
    tokens = IMUTokens(32, "v1")
    missing = tokens(2, 3)
    future = tokens(2, 3, future=True)
    torch.testing.assert_close(missing, tokens.missing[None, None].expand(2, 3, -1, -1))
    torch.testing.assert_close(future, tokens.future[None, None].expand(2, 3, -1, -1))
    assert not torch.allclose(missing, future)


def test_future_steps_reject_imu_and_require_cache():
    model = ShakeWM(tiny())
    x = torch.randn(1, 1, 4, 8)
    short, long = torch.randn(1, 1, 20, 6), torch.randn(1, 1, 600, 6)
    with pytest.raises(ValueError, match="history cache"):
        model(x, cache_limit=2, future=True)
    _, cache = model(x, cache_limit=3)
    with pytest.raises(ValueError, match="cannot receive IMU"):
        model(x, short, long, torch.ones(1, 1, 2, dtype=torch.bool), cache=cache, future=True)


def test_rollout_reads_future_not_missing_placeholder():
    torch.manual_seed(0)
    model = ShakeWM(tiny()).eval()
    history = torch.randn(1, 3, 4, 8)
    short, long = torch.randn(1, 3, 20, 6), torch.randn(1, 3, 600, 6)
    eligible = torch.ones(1, 3, 2, dtype=torch.bool)
    base = model.rollout(history, short, long, eligible, horizon=4)
    with torch.no_grad():
        model.imu.missing.add_(1.0)            # unused: every history window is real IMU
    torch.testing.assert_close(model.rollout(history, short, long, eligible, horizon=4), base)
    with torch.no_grad():
        model.imu.future[:, 0].add_(1.0)       # LayerNorm removes uniform shifts across all channels.
    changed = model.rollout(history, short, long, eligible, horizon=4)
    torch.testing.assert_close(changed[:, :1], base[:, :1])
    assert not torch.allclose(changed[:, 1:], base[:, 1:])


def test_training_updates_future_placeholder_only_when_rolling_out():
    config = Config.load("configs/smoke.json")
    config.model.imu_mode = "v1"; config.data.eligibility = "v1"
    config.train.imu_dropout = 0
    model = ShakeWM(config.model)
    b, c, h = 2, config.data.context, config.data.horizon
    batch = {"history": torch.randn(b, c, 4, 8), "targets": torch.randn(b, h, 4, 8),
             "tf_targets": torch.randn(b, c, 4, 8), "target_mask": torch.ones(b, h, dtype=torch.bool),
             "tf_mask": torch.ones(b, c, dtype=torch.bool), "short": torch.randn(b, c, 20, 6),
             "long": torch.randn(b, c, 600, 6), "eligible": torch.ones(b, c, 2, dtype=torch.bool)}
    train_microbatch(model, batch, config, "cpu")
    assert model.imu.future.grad is not None and model.imu.future.grad.abs().sum() > 0
    # With every history window real and no dropout, the missing placeholder is never used.
    assert model.imu.missing.grad is None or model.imu.missing.grad.abs().sum() == 0
