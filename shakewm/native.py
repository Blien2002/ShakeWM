"""Read-only ingestion of ShakeBench's lossless IMU-WM v1 directories.

No simulator imports, rerendering, video transcoding or privileged model inputs.
"""
from collections import Counter, defaultdict
import json
from pathlib import Path
import numpy as np
from .config import digest_json, file_sha256

CHANNELS = ["specific_force_x", "specific_force_y", "specific_force_z",
            "angular_velocity_x", "angular_velocity_y", "angular_velocity_z"]
UNITS = ["m/s2"] * 3 + ["rad/s"] * 3


def require(condition, message):
    """Raise `ValueError` when an imported-data contract is not satisfied."""
    if not condition:
        raise ValueError(message)


def close(actual, expected, message):
    """Require equal array shapes and values within the strict numeric tolerance."""
    a, b = np.asarray(actual), np.asarray(expected)
    require(a.shape == b.shape and np.allclose(a, b, atol=1e-9, rtol=0), message)


def safe_file(root, name):
    """Resolve a manifest-relative file while rejecting absolute/path-escape paths."""
    require(not Path(name).is_absolute(), "absolute native path refused")
    p = (root / name).resolve()
    require(p.is_relative_to(root.resolve()) and p.is_file(), f"unsafe/missing native file: {name}")
    return p


def canonical_quaternion(values):
    """Normalize a WXYZ quaternion and choose one stable sign/rounding form.

    A rotation is unchanged by negating all four quaternion components, so the
    first nonzero component is made positive before fingerprinting the pose.
    """
    q = np.asarray(values, dtype=np.float64)
    require(q.shape == (4,) and np.isfinite(q).all(), "invalid physical quaternion")
    norm = np.linalg.norm(q)
    require(np.isfinite(norm) and norm > 0, "invalid physical quaternion norm")
    q = q / norm
    # q and -q describe the same rotation, including rotations with qw == 0.
    first = np.flatnonzero(q != 0)[0]
    if q[first] < 0:
        q = -q
    q = np.round(q, decimals=12)
    q[q == 0] = 0.0
    return q.tolist()


def identity(manifest):
    """Extract split identity and a canonical fingerprint of the physical state.

    Pose is authoritative for physical identity. The returned audit hash still
    retains the complete original state record, including non-physical labels.
    """
    meta = manifest["metadata"]
    original = meta.get("original_state", meta["state"])
    state = meta["state"]
    require("state_id" in original and "object_pose_worktable" in original, "state identity/pose missing")
    # The seven pose values are xyz position followed by a WXYZ quaternion.
    pose = np.asarray(original["object_pose_worktable"], dtype=np.float64)
    require(pose.shape == (7,) and np.isfinite(pose).all(), "invalid physical object pose")
    quaternion = canonical_quaternion(pose[3:])
    if "object_start_quat_wxyz" in original:
        require(np.allclose(canonical_quaternion(original["object_start_quat_wxyz"]),
                            quaternion, atol=1e-6, rtol=0), "redundant physical quaternion mismatch")
    # Pose is authoritative. Redundant xy/yaw/quaternion and state/program labels
    # must not manufacture independent physical states. Keep task/velocity/grasp
    # conditions and retain the full original state separately as immutable audit.
    position = [float(v) if v != 0 else 0.0 for v in pose[:3]]
    velocity = original.get("object_initial_velocity")
    if velocity is not None:
        velocity = np.asarray(velocity, dtype=np.float64)
        require(velocity.shape == (6,) and np.isfinite(velocity).all(), "invalid physical initial velocity")
        velocity = [float(v) if v != 0 else 0.0 for v in velocity]
    # Only physical/task conditions define grouping; the full record is hashed separately for audit.
    physical = {"object_pose_worktable": position + quaternion,
                "task": original.get("task", {}), "object_initial_velocity": velocity,
                "grasp_region": original.get("grasp_region")}
    seed = meta["excitation"].get("excitation_seed", state.get("excitation_seed"))
    require(seed is not None and not isinstance(seed, bool), "excitation seed missing")
    return {"state_id": str(original["state_id"]), "state_fingerprint": digest_json(physical),
            "original_state_sha256": digest_json(original), "seed": int(seed),
            "asset": original.get("task", {}).get("object_id", "unknown"),
            "basis": original.get("stable_basis", original.get("basis", "unspecified")),
            "scenario": meta["excitation"].get("mode_params", {}).get("scenario", "unspecified")}


def discover(sources):
    """Find native manifests and separate completed episodes from failure receipts.

    Each returned row includes a checksum-bound source path and canonical state,
    seed, asset, basis, and scenario identity. Only immediate child directories
    are searched when a source is a directory without its own manifest.
    """
    paths = set()
    for source in sources:
        p = Path(source).resolve()
        if p.is_file():
            paths.add(p)
        elif (p / "manifest.json").is_file():
            paths.add(p / "manifest.json")
        else:
            paths.update(p.glob("*/manifest.json"))
    require(bool(paths), "no native manifest.json found (direct sources or immediate child directories)")
    rows, excluded = [], []
    for p in sorted(paths):
        m = json.loads(p.read_text())
        require(m.get("schema_id") == "shakebench.imu_wm.v1" and m.get("schema_version") == 1,
                f"unsupported native schema: {p}")
        if m.get("status") != "COMPLETED":
            excluded.append({"source": str(p), "status": m.get("status"), "reason": m.get("failure_reason")})
            continue
        rows.append({"source": str(p), "source_manifest_sha256": file_sha256(p), **identity(m)})
    require(bool(rows), "no COMPLETED native episodes; failure receipts are not training examples")
    return rows, excluded


def physical_representatives(rows):
    """Keep one deterministic row per physical-state fingerprint.

    The lexicographically first source path is retained if audit labels differ.
    """
    unique = {}
    for row in sorted(rows, key=lambda r: r["source"]):
        unique.setdefault(row["state_fingerprint"], row)
    return unique


def plan_splits(sources):
    """Freeze train/val/test groups before window creation.

    Episodes connected by state ID, physical fingerprint, or seed stay in one
    split. The deterministic grouped allocation targets 80/10/10 by unique
    physical states and reports when an independent held-out split is unavailable.
    """
    rows, excluded = discover(sources)
    # `parent` is the union-find table used to join episodes sharing any identity key.
    parent = list(range(len(rows)))
    def find(i):
        """Return an episode's component root and compress the traversed path."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    for key in ["state_id", "state_fingerprint", "seed"]:
        # Union repeated identities; transitive links then keep the whole component in one split.
        seen = {}
        for i, row in enumerate(rows):
            value = row[key]
            if value in seen:
                parent[find(i)] = find(seen[value])
            seen[value] = i
    components = defaultdict(list)
    for i, row in enumerate(rows):
        components[find(i)].append(row)
    groups = sorted(components.values(), key=lambda g: (-len({r["state_fingerprint"] for r in g}),
                                                        digest_json(sorted(r["state_fingerprint"] for r in g))))
    total = len({r["state_fingerprint"] for r in rows})
    # `targets` are desired unique-state counts; `counts` and `strata` track assignments so far.
    targets = {"train": total - 2 * (total // 10), "val": total // 10, "test": total // 10}
    counts = Counter()
    strata = defaultdict(Counter)
    for group in groups:
        unique = physical_representatives(group)
        stratum = Counter((r["asset"], r["basis"]) for r in unique.values())
        # Largest remaining state quota, with stratum coverage as a deterministic tie-break.
        split = max(targets, key=lambda s: (targets[s] - counts[s],
                      -sum(strata[k][s] for k in stratum), s == "train"))
        counts[split] += len(unique)
        for k, n in stratum.items():
            strata[k][split] += n
        for row in group:
            row["split"] = split
    plan = {"schema": "shakewm.native-split.v1", "policy": "state+physical-fingerprint+seed connected components",
            "target_state_counts": targets, "actual_state_counts": dict(counts), "independent_components": len(groups),
            "heldout_available": bool(counts["val"] and counts["test"]),
            "strata": {"/".join(k): dict(v) for k, v in strata.items()}, "episodes": rows, "excluded": excluded}
    validate_plan(plan)
    return plan


def validate_plan(plan):
    """Reject split leakage and recompute every summary from episode assignments.

    This also permits a caller to reassign whole groups without trusting stale
    counts, held-out flags, or stratum summaries in an external plan file.
    """
    require(plan.get("schema") == "shakewm.native-split.v1", "unsupported native split plan")
    # Each map records the first split assigned to a state/seed identity.
    seen = {k: {} for k in ["state_id", "state_fingerprint", "seed"]}
    paths = set()
    for row in plan["episodes"]:
        require(row["split"] in {"train", "val", "test"}, "invalid split")
        require(row["source"] not in paths, "duplicate native source")
        paths.add(row["source"])
        for key in seen:
            old = seen[key].setdefault(str(row[key]), row["split"])
            require(old == row["split"], f"native split leaks {key}")
    # External plans may legally reassign whole components; recompute all derived
    # summaries instead of trusting stale heldout/count/stratum declarations.
    physical_states, state_labels = defaultdict(set), defaultdict(set)
    strata = defaultdict(Counter)
    parent = list(range(len(plan["episodes"])))
    def find(i):
        """Return and compress the union-find root for split-connected episodes."""
        while parent[i] != i:
            parent[i] = parent[parent[i]]; i = parent[i]
        return i
    previous = {k: {} for k in seen}
    unique_physical = physical_representatives(plan["episodes"])
    for i, row in enumerate(plan["episodes"]):
        split = row["split"]
        physical_states[split].add(row["state_fingerprint"])
        state_labels[split].add(row["state_id"])
        for key in previous:
            value = str(row[key])
            if value in previous[key]:
                parent[find(i)] = find(previous[key][value])
            previous[key][value] = i
    for row in unique_physical.values():
        strata[(row["asset"], row["basis"])][row["split"]] += 1
    plan["actual_state_counts"] = {split: len(values) for split, values in physical_states.items()}
    plan["state_id_counts"] = {split: len(values) for split, values in state_labels.items()}
    plan["heldout_available"] = bool(physical_states.get("val") and physical_states.get("test"))
    plan["independent_components"] = len({find(i) for i in range(len(parent))})
    plan["strata"] = {"/".join(k): dict(v) for k, v in strata.items()}
    plan["physical_identity_contract"] = "pose-wxyz-canonical-v1"


def read_native(path, camera="main"):
    """Validate one lossless native recording and convert it to normalized arrays.

    `camera` selects the RGB view. The result is `(episode, audit, identity)`;
    validation checks file hashes, sensor timing/delivery, frame windows, and
    terminal evidence without importing collector code or rerunning simulation.
    """
    path = Path(path).resolve()
    # `m` is the native manifest; `meta` contains the timing, sensor, and camera contracts.
    root, m = path.parent, json.loads(path.read_text())
    require(m.get("schema_id") == "shakebench.imu_wm.v1" and m.get("schema_version") == 1, "unsupported native schema")
    require(m.get("status") == "COMPLETED" and not m.get("failure_reason"), "only completed valid responses are accepted")
    meta = m["metadata"]
    require(meta["timing"]["control_hz"] == 20 and meta["timing"]["imu_hz"] == 200, "native rate mismatch")
    close(meta["timing"]["physics_dt_s"], .0002, "physics clock mismatch")
    require(meta["imu"]["columns"] == CHANNELS and meta["imu"]["units"] == UNITS, "sensor units/channel order mismatch")
    require(meta["imu"]["mode"] == "canonical_noisy_v1", "main model requires canonical noisy IMU")
    profile = meta["imu"]["profile"]["output"]
    require(profile["sample_rate_hz"] == 200 and profile["delivery_delay_samples"] == 1, "IMU profile delay/rate mismatch")
    order = meta["cameras"]["order"]
    require(camera in order and len(set(order)) == len(order), "missing/ambiguous camera")
    camera_index = order.index(camera)
    height, width = meta["cameras"]["resolution_hw"]
    # Verify all immutable native evidence bytes; do not import or execute collector code.
    for name, expected in m["files"].items():
        require(file_sha256(safe_file(root, name)) == expected, f"native checksum mismatch: {name}")
    def arrays(name):
        """Load one checksum-listed NPZ file as a name-to-array mapping."""
        require(name in m["files"], f"native array lacks a manifest checksum: {name}")
        with np.load(safe_file(root, name), allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    initial = arrays("sensor_initial.npz")
    close(initial["acquisition_time_s"], np.arange(-10, 0) * .005, "reset prefill time mismatch")
    require(initial["is_live"].dtype == bool and not initial["is_live"].any(), "reset prefill cannot be live")
    require(initial["window"].shape == (10, 6), "reset window shape mismatch")
    # Chunk arrays are concatenated by these fields after each chunk's ordering is verified.
    keys = ["acquisition_time_s", "scheduled_delivery_time_s", "delivery_event_time_s",
            "delivered_acquisition_time_s", "acquisition_is_live", "delivered_is_live",
            "acquired_measurement", "delivered_measurement"]
    chunks = {k: [] for k in keys}
    # `cursor` is the next expected global IMU acquisition index across chunks.
    cursor = 0
    for row in m["imu_chunks"]:
        d = arrays(row["path"])
        n = row["count"]
        require(row["start_index"] == cursor and n > 0, "native IMU chunk gap/order mismatch")
        # Native chunks begin after the reset prefill; index zero is the first 5 ms acquisition.
        t = (np.arange(n) + cursor + 1) * .005
        close(d["acquisition_time_s"], t, "native IMU acquisition gap/order mismatch")
        close(d["scheduled_delivery_time_s"], t + .005, "scheduled IMU delivery is not +5ms")
        close(d["delivery_event_time_s"], t, "actual IMU delivery event mismatch")
        close(d["delivered_acquisition_time_s"], t - .005, "delivered IMU acquisition is not event-5ms")
        require(d["acquisition_is_live"].dtype == bool and d["acquisition_is_live"].all(), "invalid acquired live mask")
        require(np.array_equal(d["delivered_is_live"], t > .005 + 1e-12), "delivered reset mask mismatch")
        for key in keys:
            expected = (n, 6) if key.endswith("measurement") else (n,)
            require(d[key].shape == expected and np.isfinite(d[key]).all(), f"invalid native IMU field: {key}")
            chunks[key].append(d[key])
        cursor += n
    require(cursor > 0 and cursor == m["valid_imu_count"], "native IMU count mismatch")
    imu = {k: np.concatenate(v) for k, v in chunks.items()}
    require(np.array_equal(imu["delivered_measurement"][1:], imu["acquired_measurement"][:-1]),
            "actual delivered values disagree with preceding acquired values")

    # `term` is optional terminal-state evidence; absent evidence means legacy fixed duration.
    term = m.get("termination")
    planned_steps = meta["timing"]["total_steps"]
    if term:
        tick = term["physics_tick"]
        require(isinstance(tick, int) and not isinstance(tick, bool) and 0 <= tick <= planned_steps * 250,
                "invalid terminal physics tick")
        close(term["time_s"], tick * .0002, "termination time/tick mismatch")
        require(term["contract"] == "collision_convex_envelope_actual_table_v1", "unsupported departure contract")
        require(term["terminal_is_regular_frame"] == (tick % 250 == 0), "terminal frame classification mismatch")
        require(term["terminal_event_is_valid_response"] is True, "invalid event cannot become a training response")
        require(term["reason"] in {"time_limit", "outside_table_support", "below_table_top"}, "unknown terminal reason")
        departed = term["reason"] != "time_limit"
        require(term["terminated"] == departed and term["truncated"] != departed, "termination flags mismatch")
        require(departed or tick == planned_steps * 250, "time limit ended prematurely")
        end_time = term["time_s"]
    else:
        require(not meta.get("termination_contract"), "departure-enabled native record lacks terminal evidence")
        tick = planned_steps * 250
        end_time = planned_steps / 20
    require(m["valid_frame_count"] == len(m["frames"]) == tick // 250 + 1, "native frame count/termination mismatch")
    require(cursor == tick // 25, "native IMU count/termination mismatch")

    rgb, times = [], []
    def validate_frame(d, t):
        """Check a frame's timestamp, RGB payload, and delivered-IMU history window."""
        close(d["time_s"], t, "frame timestamp mismatch")
        require(d["rgb"].shape == (len(order), height, width, 3) and d["rgb"].dtype == np.uint8,
                "native RGB shape/dtype mismatch")
        count = int(round(t / .0002)) // 25
        expected_times = count * .005 + np.arange(-10, 0) * .005
        close(d["imu_window_acquisition_time_s"], expected_times, "frame delivery window time mismatch")
        require(np.array_equal(d["imu_window_is_live"], expected_times > 1e-12), "frame reset live mask mismatch")
        # Rebuild the frame's last ten delivered samples from reset prefill plus causal deliveries.
        expected_window = np.concatenate([initial["window"], imu["delivered_measurement"][:count]])[-10:]
        require(np.array_equal(d["imu_window"], expected_window), "frame values disagree with actual delivery trace")
    for i, row in enumerate(m["frames"]):
        require(row["index"] == i, "native frame index mismatch")
        close(row["time_s"], i / 20, "native RGB time grid mismatch")
        d = arrays(row["path"])
        validate_frame(d, i / 20)
        rgb.append(d["rgb"][camera_index]); times.append(i / 20)
    if term:
        terminal = arrays(term["terminal_state_path"])
        validate_frame(terminal, end_time)
        if term["terminal_is_regular_frame"]:
            require(np.array_equal(terminal["rgb"][camera_index], rgb[-1]), "terminal regular RGB disagrees with final frame")
        # Off-grid terminal images remain native audit evidence; never snap them onto 10/20 Hz targets.

    output = {"rgb": np.stack(rgb), "rgb_time": np.asarray(times, np.float64),
              "rgb_valid": np.ones(len(rgb), bool), "imu": imu["acquired_measurement"].astype(np.float32),
              "imu_acquisition_time": imu["acquisition_time_s"],
              "imu_delivery_time": imu["scheduled_delivery_time_s"],
              "imu_live": imu["acquisition_is_live"], "end_time": np.asarray(end_time, np.float64)}
    audit = {"native_manifest_sha256": file_sha256(path), "native_files_checked": len(m["files"]),
             "frames": len(rgb), "acquisitions": cursor,
             "delivered_live": int(imu["delivered_is_live"].sum()),
             "pending_acquisitions_at_end": int((imu["scheduled_delivery_time_s"] > end_time + 1e-9).sum()),
             "reset_prefill_discarded": 10, "camera": camera, "end_time": end_time,
             "termination": term or {"reason": "legacy_fixed_duration", "time_s": end_time},
             "off_grid_terminal_rgb_used_as_target": False,
             "provenance_sha256": digest_json(meta["provenance"]),
             "sensor_contract": {k: meta["imu"][k] for k in ["columns", "units", "mode", "mount", "gravity_world_m_s2", "profile"]},
             "camera_contract": meta["cameras"], "timing": meta["timing"]}
    return output, audit, identity(m)


def import_native(sources, output, plan_path=None, camera="main"):
    """Convert validated native recordings into the normalized episode-manifest format.

    A supplied split plan is revalidated against the discovered source set before
    any arrays are written. `output` must be a new directory; the return value is
    the normalized manifest and an audit report, not evidence of physical validity.
    """
    plan = json.loads(Path(plan_path).read_text()) if plan_path else plan_splits(sources)
    validate_plan(plan)
    if plan_path:
        discovered, _ = discover(sources)
        require({r["source"] for r in discovered} == {r["source"] for r in plan["episodes"]}, "split plan source set mismatch")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "episodes").mkdir()
    # Freeze the exact grouping before reading windows or fitting any statistics.
    (output / "split_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    manifest = {"schema": "shakewm.manifest.v1", "source_schema": "shakebench.imu_wm.v1",
                "synthetic": False, "data_origin": "native_simulator_recordings", "adapter_version": 1,
                "camera": "third_person" if camera == "main" else "wrist", "split_plan_hash": digest_json(plan),
                "episodes": []}
    audits = []
    for i, row in enumerate(plan["episodes"]):
        require(file_sha256(row["source"]) == row["source_manifest_sha256"], "native manifest changed after split freeze")
        arrays, audit, actual = read_native(row["source"], camera)
        require(all(actual[k] == row[k] for k in actual), "native identity changed after split freeze")
        name = f"episodes/{i:06d}.npz"
        np.savez_compressed(output / name, **arrays)
        episode_id = f"native_{i:06d}_{row['source_manifest_sha256'][:12]}"
        manifest["episodes"].append({"id": episode_id, "path": name, "source_sha256": file_sha256(output / name),
                                     "native_manifest_sha256": row["source_manifest_sha256"],
                                     **actual, "split": row["split"], "valid": True,
                                     "termination": audit["termination"]["reason"]})
        audits.append({"episode_id": episode_id, **audit})
    report = {"schema": "shakewm.native-import-report.v1", "episodes": audits,
              "heldout_available": plan.get("heldout_available", False),
              "actual_state_counts": plan.get("actual_state_counts"), "excluded": plan.get("excluded", []),
              "claims": "input contract validation only; no simulation rerun or physical outcome validation"}
    (output / "import_report.json").write_text(json.dumps(report, indent=2) + "\n")
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, report
