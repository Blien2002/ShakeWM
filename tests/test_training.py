import random
import numpy as np
import pytest
import torch
from shakewm.config import Config
from shakewm.model import ShakeWM
from shakewm.engine import (train_microbatch, masked_l1_sum, make_optimizer, save_checkpoint,
                            load_checkpoint, seed_all)
from shakewm.imu import LongEncoder


def batch(config):
    b, c, h = 2, config.data.context, config.data.horizon
    return {"history": torch.randn(b, c, 4, 8), "targets": torch.randn(b, h, 4, 8),
            "tf_targets": torch.randn(b, c, 4, 8), "target_mask": torch.ones(b, h, dtype=torch.bool),
            "tf_mask": torch.ones(b, c, dtype=torch.bool), "short": torch.randn(b, c, 20, 6),
            "long": torch.randn(b, c, 600, 6), "eligible": torch.ones(b, c, 2, dtype=torch.bool)}


@pytest.mark.parametrize("activation_checkpoint", [False, True])
def test_tbptt_backward_releases_segments(activation_checkpoint):
    config = Config.load("configs/smoke.json")
    config.model.activation_checkpoint = activation_checkpoint
    model = ShakeWM(config.model)
    boundaries = []
    def hook(step, last, cache):
        assert last.grad_fn is None
        assert all(k.grad_fn is None and v.grad_fn is None for k, v in cache.layers)
        boundaries.append(step)
    metrics = train_microbatch(model, batch(config), config, "cpu", hook)
    assert boundaries == [4, 6]
    assert metrics["tf_valid"] == 6 and metrics["rollout_valid"] == 12
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_resume_restores_optimizer_scheduler_rng_and_next_update(tmp_path):
    seed_all(23)
    config = Config.load("configs/smoke.json")
    model = ShakeWM(config.model)
    optimizer, scheduler = make_optimizer(model, config)
    generator = torch.Generator().manual_seed(99)
    data = batch(config)
    def update(m, o, s):
        o.zero_grad(set_to_none=True)
        train_microbatch(m, data, config, "cpu")
        o.step(); s.step()
    update(model, optimizer, scheduler)
    save_checkpoint(tmp_path / "resume.pt", model, optimizer, scheduler, 1, config, {}, {}, "split", generator)
    expected_rng = (random.random(), float(np.random.rand()), torch.rand(2), torch.rand(2, generator=generator))
    update(model, optimizer, scheduler)
    expected = {k: v.clone() for k, v in model.state_dict().items()}
    restored = ShakeWM(config.model)
    o2, s2 = make_optimizer(restored, config)
    g2 = torch.Generator()
    state = load_checkpoint(tmp_path / "resume.pt", restored, config, "split", {}, o2, s2, g2)
    assert state["global_step"] == 1
    actual_rng = (random.random(), float(np.random.rand()), torch.rand(2), torch.rand(2, generator=g2))
    assert expected_rng[:2] == actual_rng[:2]
    for a, b in zip(expected_rng[2:], actual_rng[2:]):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
    update(restored, o2, s2)
    for key, value in restored.state_dict().items():
        torch.testing.assert_close(value, expected[key], atol=0, rtol=0)
    assert scheduler.state_dict() == s2.state_dict()


def test_valid_mask_reduction():
    predicted = torch.zeros(1, 2, 4, 8)
    target = torch.ones_like(predicted); target[:, 1] = float("nan")
    assert masked_l1_sum(predicted, target, torch.tensor([[True, False]])).item() == 1


def test_long_antialias_filter_rejects_40hz():
    taps = LongEncoder().fir[0, 0]
    n = torch.arange(len(taps))
    gain = lambda hz: (taps.to(torch.complex64) * torch.exp(-2j * torch.pi * hz * n / 200)).sum().abs()
    assert gain(5) > 0.95 and gain(40) < 0.01


def test_bf16_cpu_step():
    config = Config.load("configs/smoke.json"); config.train.bf16 = True
    model = ShakeWM(config.model)
    train_microbatch(model, batch(config), config, "cpu")
    assert torch.isfinite(model.visual_out.weight.grad).all()


def test_v1_long_branch_has_gradients():
    config = Config.load("configs/smoke.json")
    config.model.imu_mode = "v1"; config.data.eligibility = "v1"
    config.train.imu_dropout = 0
    model = ShakeWM(config.model)
    train_microbatch(model, batch(config), config, "cpu")
    assert model.imu.long.input[0].weight.grad.abs().sum() > 0
