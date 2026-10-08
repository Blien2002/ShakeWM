import json
import numpy as np
import pytest
import torch
from shakewm.config import Config, DataConfig
from shakewm.data import (create_synthetic, load_episode, imu_window, read_manifest, fit_normalization,
                           WindowDataset, build_cache)
from shakewm.encoder import MockTeacher


def fixture_data(tmp_path, seconds=2):
    manifest = create_synthetic(tmp_path / "raw", seconds)
    norm = fit_normalization(manifest)
    config = Config.load("configs/smoke.json")
    teacher = MockTeacher()
    build_cache(manifest, tmp_path / "cache", config.data, teacher)
    return manifest, norm, config, teacher


def test_delivery_alignment_reset_and_missing_history(tmp_path):
    manifest = create_synthetic(tmp_path / "raw", seconds=4)
    e = load_episode(manifest.parent / "synthetic_0.npz")
    assert not imu_window(e, 0.095, 20)[1]
    values, eligible = imu_window(e, 0.1, 20)
    assert eligible
    np.testing.assert_allclose(values[-1], e["imu"][39])  # acquisition=.095, delivery=.100
    assert not imu_window(e, 2.999, 600)[1]
    assert imu_window(e, 3.0, 600)[1]
    e["imu_live"][30] = False
    assert not imu_window(e, 0.1, 20)[1]


def test_split_state_and_seed_leaks_rejected(tmp_path):
    manifest = create_synthetic(tmp_path / "raw")
    original = json.loads(manifest.read_text())
    for key in ["state_id", "seed"]:
        d = json.loads(json.dumps(original))
        d["episodes"][-1][key] = d["episodes"][0][key]
        manifest.write_text(json.dumps(d))
        with pytest.raises(ValueError, match="leaks across splits"):
            read_manifest(manifest)


def test_variable_end_masks_and_caches(tmp_path):
    manifest, norm, config, teacher = fixture_data(tmp_path)
    cached = WindowDataset(manifest, "test", config.data, norm, tmp_path / "cache")
    online = WindowDataset(manifest, "test", config.data, norm, encoder=teacher)
    last = cached[-1]
    assert last["target_mask"].tolist() == [True, False, False, False, False, False]
    assert cached.coverage["long_eligible"] == 0
    for key in ["history", "targets"]:
        torch.testing.assert_close(last[key], online[-1][key], atol=5e-4, rtol=5e-4)


def test_train_only_statistics(tmp_path):
    manifest = create_synthetic(tmp_path / "raw")
    before = fit_normalization(manifest)
    path = manifest.parent / "synthetic_3.npz"
    e = load_episode(path); e["imu"][:] = 1e8
    np.savez_compressed(path, **e)
    assert before == fit_normalization(manifest)


def test_future_mutation_cannot_change_history_or_prediction(tmp_path):
    from shakewm.model import ShakeWM
    manifest, norm, config, teacher = fixture_data(tmp_path)
    dataset = WindowDataset(manifest, "test", config.data, norm, encoder=teacher)
    before = dataset[0]
    origin = before["origin_time"]
    path = manifest.parent / "synthetic_3.npz"
    e = load_episode(path)
    e["rgb"][e["rgb_time"] > origin] = 255
    e["imu"][e["imu_delivery_time"] > origin] = 1e6
    e["future_gt"] = np.ones(100) * 1e9
    np.savez_compressed(path, **e)
    m = json.loads(manifest.read_text())
    m["episodes"][-1]["scenario"] = "arbitrary_future_program"
    m["episodes"][-1]["seed"] = 999
    manifest.write_text(json.dumps(m))
    from shakewm.config import digest_json
    changed_norm = dict(norm, split_hash=digest_json(m))
    after = WindowDataset(manifest, "test", config.data, changed_norm, encoder=teacher)[0]
    for key in ["history", "short", "long", "eligible"]:
        torch.testing.assert_close(before[key], after[key], atol=0, rtol=0)
    model = ShakeWM(config.model).eval()
    def predict(item):
        return model.rollout(*(item[k][None] for k in ["history", "short", "long", "eligible"]), config.data.horizon)
    torch.testing.assert_close(predict(before), predict(after), atol=0, rtol=0)
    assert not torch.equal(before["targets"], after["targets"])


def test_v1_full_context_eligibility(tmp_path):
    manifest, norm, config, teacher = fixture_data(tmp_path, seconds=4)
    config.data.eligibility = "v1"
    data = WindowDataset(manifest, "test", config.data, norm, encoder=teacher)
    assert data[0]["origin_time"] == pytest.approx(3.2)
    assert data[0]["eligible"].all()


def test_changed_cache_contract_is_rejected(tmp_path):
    manifest, norm, config, teacher = fixture_data(tmp_path)
    index = tmp_path / "cache/index.json"
    d = json.loads(index.read_text()); d["teacher"]["feature_dim"] = 99
    index.write_text(json.dumps(d))
    with pytest.raises(ValueError, match="corrupt"):
        WindowDataset(manifest, "test", config.data, norm, tmp_path / "cache")
