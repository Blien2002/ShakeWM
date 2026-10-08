import copy
import torch
import pytest
from shakewm.config import Config, ModelConfig
from shakewm.model import ShakeWM, block_causal_mask


def tiny(checkpoint=False):
    return ModelConfig(feature_dim=8, grid=2, width=32, depth=2, heads=4,
                       activation_checkpoint=checkpoint)


def test_block_visibility_and_direct_attention_reference():
    mask = block_causal_mask(3, 3, 6)
    assert mask[:6, :6].all() and not mask[:6, 6:].any()
    assert mask[6:12, :12].all() and not mask[6:12, 12:].any()
    torch.manual_seed(1)
    model = ShakeWM(tiny()).eval()
    x = torch.randn(2, 3, 4, 8)
    actual, _ = model(x, cache_limit=3)
    reference, _ = model(x, cache_limit=3, reference=True)
    torch.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)
    mutated = x.clone(); mutated[:, -1] += 99
    changed, _ = model(mutated, cache_limit=3)
    torch.testing.assert_close(actual[:, :-1], changed[:, :-1], atol=0, rtol=0)


def test_same_block_is_not_token_causal():
    model = ShakeWM(tiny()).eval()
    x = torch.randn(1, 1, 4, 8, requires_grad=True)
    output, _ = model(x, cache_limit=1)
    output[0, 0, 0, 0].backward()
    assert x.grad[0, 0, -1].abs().sum() > 0  # first patch reads last patch


def test_cache_matches_full_recomputation_and_is_bounded():
    model = ShakeWM(tiny()).eval()
    history = torch.randn(1, 3, 4, 8)
    y, cache = model(history, cache_limit=6)
    sequence = history
    last = y[:, -1:]
    for _ in range(3):
        sequence = torch.cat([sequence, last], 1)
        incremental, cache = model(last, cache=cache)
        full, _ = model(sequence, cache_limit=6)
        torch.testing.assert_close(incremental, full[:, -1:], atol=2e-6, rtol=2e-5)
        last = incremental
    with pytest.raises(ValueError, match="bounded"):
        model(last, cache=cache)


def test_detached_cache_and_latent_cut_all_old_gradients():
    model = ShakeWM(tiny())
    old = torch.randn(1, 3, 4, 8, requires_grad=True)
    pred, cache = model(old, cache_limit=4)
    detached = cache.detach()
    assert all(k.grad_fn is None and v.grad_fn is None for k, v in detached.layers)
    out, _ = model(pred[:, -1:].detach(), cache=detached)
    out.sum().backward()
    assert old.grad is None
    assert model.visual_in.weight.grad.abs().sum() > 0


def test_production_dimensions_without_allocating_large_weights():
    cfg = Config()
    assert (cfg.model.depth, cfg.model.width, cfg.model.heads, cfg.model.grid, cfg.model.feature_dim) == (16, 1024, 16, 16, 768)
    with torch.device("meta"):
        model = ShakeWM(cfg.model)
    backbone = sum(p.numel() for p in model.blocks.parameters())
    assert 200_000_000 < backbone < 203_000_000
    assert model.visual_in.in_features == 768 and model.visual_out.out_features == 768


def test_no_imu_slots_and_modes_have_same_parameter_budget():
    configs = [tiny() for _ in range(3)]
    for c, mode in zip(configs, ["v0", "v1", "none"]):
        c.imu_mode = mode
    models = [ShakeWM(c) for c in configs]
    assert len({sum(p.numel() for p in m.parameters()) for m in models}) == 1
    x = torch.randn(1, 2, 4, 8)
    model = models[0]
    eligibility = torch.ones(1, 2, 2, dtype=torch.bool)
    short, long = torch.randn(1, 2, 20, 6), torch.randn(1, 2, 600, 6)
    a, _ = model(x, short, long, eligibility, torch.ones(1, dtype=torch.bool), cache_limit=2)
    b, _ = model(x, cache_limit=2)
    torch.testing.assert_close(a, b, atol=0, rtol=0)
