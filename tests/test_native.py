"""Generated native-format fixtures, never redistributed real observations."""
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from shakewm.config import Config, file_sha256
from shakewm.data import WindowDataset, fit_normalization, read_manifest, create_synthetic, build_cache
from shakewm.encoder import MockTeacher
from shakewm.native import CHANNELS, UNITS, import_native, plan_splits, read_native, validate_plan


def write_native(root, seed=1, state_id="state-a", tick=2000, terminated=False):
    root.mkdir(parents=True)
    (root / "frames").mkdir(); (root / "imu").mkdir()
    times = np.arange(1, tick // 25 + 1, dtype=np.float64) * .005
    values = np.arange(len(times) * 6, dtype=np.float32).reshape(-1, 6)
    delivered = np.concatenate([np.zeros((1, 6), np.float32), values[:-1]])
    files = {}
    def save(name, **fields):
        np.savez_compressed(root / name, **fields)
        files[name] = file_sha256(root / name)
    save("sensor_initial.npz", window=np.zeros((10, 6), np.float32),
         acquisition_time_s=np.arange(-10, 0) * .005, is_live=np.zeros(10, bool))
    save("imu/000000.npz", acquisition_time_s=times, scheduled_delivery_time_s=times + .005,
         delivery_event_time_s=times.copy(), delivered_acquisition_time_s=times - .005,
         acquisition_is_live=np.ones(len(times), bool), delivered_is_live=times > .005 + 1e-12,
         acquired_measurement=values, delivered_measurement=delivered)
    def frame(t):
        count = int(round(t / .0002)) // 25
        window_times = count * .005 + np.arange(-10, 0) * .005
        pixels = np.zeros((2, 8, 8, 3), np.uint8); pixels[0] = int(t * 100); pixels[1] = 200
        return {"time_s": np.asarray(t), "rgb": pixels,
                "imu_window_acquisition_time_s": window_times,
                "imu_window_is_live": window_times > 1e-12,
                "imu_window": np.concatenate([np.zeros((10, 6)), delivered[:count]])[-10:].astype(np.float32)}
    frames = []
    for i in range(tick // 250 + 1):
        name = f"frames/{i:06d}.npz"; save(name, **frame(i / 20))
        frames.append({"index": i, "time_s": i / 20, "path": name})
    physical_pose = [seed / 100, 0, 0, 1, 0, 0, 0]
    state = {"state_id": state_id, "excitation_seed": seed,
             "object_pose_worktable": physical_pose, "task": {"object_id": "can"}, "basis": "upright"}
    meta = {"state": state, "original_state": state, "timing": {"control_hz": 20, "imu_hz": 200,
            "physics_dt_s": .0002, "total_steps": 8}, "imu": {"columns": CHANNELS, "units": UNITS,
            "mode": "canonical_noisy_v1", "mount": {}, "gravity_world_m_s2": [0, 0, -9.81],
            "profile": {"output": {"sample_rate_hz": 200, "delivery_delay_samples": 1}}},
            "cameras": {"order": ["main", "wrist"], "resolution_hw": [8, 8]},
            "excitation": {"excitation_seed": seed}, "provenance": {"source_revision": "generated-fixture"}}
    term = None
    if terminated:
        save("terminal_state.npz", **frame(tick * .0002))
        meta["termination_contract"] = {"version": 1}
        term = {"physics_tick": tick, "time_s": tick * .0002,
                "contract": "collision_convex_envelope_actual_table_v1", "terminal_is_regular_frame": tick % 250 == 0,
                "terminal_event_is_valid_response": True, "reason": "outside_table_support",
                "terminated": True, "truncated": False, "terminal_state_path": "terminal_state.npz"}
    m = {"schema_id": "shakebench.imu_wm.v1", "schema_version": 1, "purpose": "passive_response",
         "status": "COMPLETED", "metadata": meta, "termination": term, "files": files, "frames": frames,
         "imu_chunks": [{"path": "imu/000000.npz", "start_index": 0, "count": len(times)}],
         "valid_frame_count": len(frames), "valid_imu_count": len(times)}
    (root / "manifest.json").write_text(json.dumps(m))
    return root / "manifest.json"


def mutate_array(manifest, name, mutate):
    root = manifest.parent
    with np.load(root / name, allow_pickle=False) as f:
        d = dict(f)
    mutate(d); np.savez_compressed(root / name, **d)
    m = json.loads(manifest.read_text())
    m["files"][name] = file_sha256(root / name)
    manifest.write_text(json.dumps(m))


def test_native_delivery_pairing_and_main_camera(tmp_path):
    source = write_native(tmp_path / "raw")
    e, report, _ = read_native(source)
    assert len(e["rgb"]) == 9 and (e["rgb"][0] == 0).all()
    assert e["imu_acquisition_time"][0] == .005
    assert e["imu_delivery_time"][0] == .010
    assert report["pending_acquisitions_at_end"] == 1 and report["delivered_live"] == 79
    from shakewm.data import imu_window
    assert not imu_window(e, .1, 20)[1]  # only 19 live delivered samples
    assert imu_window(e, .105, 20)[1]


@pytest.mark.parametrize("kind", ["delay", "delivered_values", "prefill", "frame_window"])
def test_native_rejects_semantic_time_corruption_even_with_updated_hash(tmp_path, kind):
    source = write_native(tmp_path / "raw")
    if kind == "delay":
        mutate_array(source, "imu/000000.npz", lambda d: d["scheduled_delivery_time_s"].__iadd__(.005))
    elif kind == "delivered_values":
        mutate_array(source, "imu/000000.npz", lambda d: d["delivered_measurement"].__iadd__(1))
    elif kind == "prefill":
        mutate_array(source, "sensor_initial.npz", lambda d: d["is_live"].fill(True))
    else:
        mutate_array(source, "frames/000002.npz", lambda d: d["imu_window"].__iadd__(1))
    with pytest.raises(ValueError):
        read_native(source)


def test_offgrid_terminal_is_not_snapped_into_target(tmp_path):
    source = write_native(tmp_path / "raw", tick=1187, terminated=True)
    e, report, _ = read_native(source)
    assert float(e["end_time"]) == pytest.approx(.2374)
    assert len(e["rgb"]) == 5 and e["rgb_time"][-1] == .2
    assert len(e["imu"]) == 47 and report["off_grid_terminal_rgb_used_as_target"] is False
    import_native([source], tmp_path / "converted")
    manifest = tmp_path / "converted/manifest.json"
    config = Config.load("configs/smoke.json"); config.data.context = 1
    data = WindowDataset(manifest, "train", config.data, fit_normalization(manifest), encoder=MockTeacher())
    assert len(data) == 0  # no eligible .2 origin has a later actual 10 Hz target


def test_auto_split_same_state_seed_never_fake_holdout(tmp_path):
    a = write_native(tmp_path / "a")
    b = write_native(tmp_path / "b")
    plan = plan_splits([a, b])
    assert plan["actual_state_counts"] == {"train": 1}
    assert not plan["heldout_available"]
    plan["episodes"][1]["split"] = "test"
    with pytest.raises(ValueError, match="leaks"):
        validate_plan(plan)


def test_auto_split_deterministic_state_counts(tmp_path):
    paths = [write_native(tmp_path / f"s{i}", seed=i, state_id=f"s{i}") for i in range(30)]
    a, b = plan_splits(paths), plan_splits(list(reversed(paths)))
    assert a == b
    assert a["actual_state_counts"] == {"train": 24, "val": 3, "test": 3}
    assert a["heldout_available"]


def test_normalized_manifest_rejects_physical_state_relabeling(tmp_path):
    a = write_native(tmp_path / "a")
    b = write_native(tmp_path / "b")
    import_native([a, b], tmp_path / "converted")
    path = tmp_path / "converted/manifest.json"
    manifest = json.loads(path.read_text())
    manifest["episodes"][1].update(state_id="relabelled", seed=42, split="test")
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="state_fingerprint leaks"):
        read_manifest(path)


def test_equivalent_quaternion_and_pose_labels_remain_one_physical_state(tmp_path):
    sources = [write_native(tmp_path / "a", seed=1, state_id="a"),
               write_native(tmp_path / "b", seed=2, state_id="b")]
    for source, sign in zip(sources, [1, -1]):
        manifest = json.loads(source.read_text())
        for key in ["original_state", "state"]:
            if key not in manifest["metadata"]:
                continue
            state = manifest["metadata"][key]
            state["object_pose_worktable"] = [0, 0, 0, 0, sign, 0, 0]
            state["object_start_quat_wxyz"] = [0, sign, 0, 0]
            state["object_xy_m"] = [0, 0]
            state["object_yaw_rad"] = 0 if sign == 1 else 2 * np.pi
            state["basis"] = "arbitrary_label_" + str(sign)
        source.write_text(json.dumps(manifest))
    plan = plan_splits(sources)
    assert plan["actual_state_counts"] == {"train": 1}
    assert plan["state_id_counts"] == {"train": 2}
    assert plan["independent_components"] == 1 and not plan["heldout_available"]
    representative = min(plan["episodes"], key=lambda row: row["source"])
    assert plan["strata"] == {"/".join((representative["asset"], representative["basis"])): {"train": 1}}
    plan["episodes"][1]["split"] = "test"
    with pytest.raises(ValueError, match="state_fingerprint"):
        validate_plan(plan)


def test_external_plan_summaries_are_recomputed_before_import(tmp_path):
    sources = [write_native(tmp_path / "a"), write_native(tmp_path / "b")]
    plan = plan_splits(sources)
    plan.update(heldout_available=True, actual_state_counts={"test": 99},
                independent_components=99, strata={"bogus": {"test": 99}})
    path = tmp_path / "external_plan.json"
    path.write_text(json.dumps(plan))
    _, report = import_native(sources, tmp_path / "converted", path)
    frozen = json.loads((tmp_path / "converted/split_plan.json").read_text())
    assert report["heldout_available"] is False
    assert report["actual_state_counts"] == {"train": 1}
    assert frozen["independent_components"] == 1 and "bogus" not in frozen["strata"]


def test_import_rejects_changed_source_and_changed_normalized_data(tmp_path):
    source = write_native(tmp_path / "raw")
    plan = plan_splits([source]); plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(plan))
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="changed after split"):
        import_native([source], tmp_path / "bad", plan_path)
    import_native([source], tmp_path / "converted")
    manifest = tmp_path / "converted/manifest.json"
    e = tmp_path / "converted/episodes/000000.npz"
    with np.load(e, allow_pickle=False) as f:
        d = dict(f)
    d["rgb"].fill(255); np.savez_compressed(e, **d)
    with pytest.raises(ValueError, match="source_sha256"):
        read_manifest(manifest)


def test_data_content_fingerprint_blocks_rebuilt_cache_resume(tmp_path):
    from shakewm.model import ShakeWM
    from shakewm.engine import make_optimizer, save_checkpoint, load_checkpoint
    manifest = create_synthetic(tmp_path / "raw")
    config = Config.load("configs/smoke.json"); teacher = MockTeacher()
    before = WindowDataset(manifest, "train", config.data, fit_normalization(manifest), encoder=teacher)
    model = ShakeWM(config.model); optimizer, scheduler = make_optimizer(model, config)
    ckpt = tmp_path / "checkpoint.pt"
    save_checkpoint(ckpt, model, optimizer, scheduler, 0, config, before.normalization,
                    teacher.contract, before.manifest_hash, torch.Generator())
    path = manifest.parent / "synthetic_0.npz"
    with np.load(path, allow_pickle=False) as f:
        d = dict(f)
    d["rgb"].fill(255); np.savez_compressed(path, **d)
    with pytest.raises(ValueError, match="train data changed"):
        WindowDataset(manifest, "train", config.data, before.normalization, encoder=teacher)
    norm = fit_normalization(manifest)
    build_cache(manifest, tmp_path / "rebuilt", config.data, teacher)
    after = WindowDataset(manifest, "train", config.data, norm, tmp_path / "rebuilt")
    assert before.manifest_hash != after.manifest_hash
    with pytest.raises(ValueError, match="split/teacher"):
        load_checkpoint(ckpt, model, config, after.manifest_hash, teacher.contract)


def test_cache_content_fingerprint_blocks_reindexed_feature_resume(tmp_path):
    from shakewm.config import file_sha256
    from shakewm.model import ShakeWM
    from shakewm.engine import make_optimizer, save_checkpoint, load_checkpoint
    manifest = create_synthetic(tmp_path / "raw")
    config = Config.load("configs/smoke.json"); teacher = MockTeacher()
    norm = fit_normalization(manifest)
    cache = tmp_path / "cache"
    build_cache(manifest, cache, config.data, teacher)
    before = WindowDataset(manifest, "train", config.data, norm, cache)
    model = ShakeWM(config.model); optimizer, scheduler = make_optimizer(model, config)
    ckpt = tmp_path / "checkpoint.pt"
    save_checkpoint(ckpt, model, optimizer, scheduler, 0, config, norm,
                    teacher.contract, before.manifest_hash, torch.Generator())
    index_path = cache / "index.json"
    index = json.loads(index_path.read_text())
    entry = next(iter(index["episodes"].values()))
    feature_path = cache / entry["path"]
    features = np.load(feature_path, allow_pickle=False).copy() + 100
    np.save(feature_path, features)
    entry["sha256"] = file_sha256(feature_path)
    index_path.write_text(json.dumps(index))
    after = WindowDataset(manifest, "train", config.data, norm, cache)
    assert before.manifest_hash != after.manifest_hash
    with pytest.raises(ValueError, match="split/teacher"):
        load_checkpoint(ckpt, model, config, after.manifest_hash, teacher.contract)
