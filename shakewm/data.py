"""Validated normalized IMU-WM v1 adapter, state splits, causal windows and caches."""
from pathlib import Path
import json
import numpy as np
import torch
from torch.utils.data import Dataset
from .config import digest_json, file_sha256


def read_manifest(path):
    """Load a normalized manifest and validate split isolation and episode paths.

    Returns the parsed manifest and its resolved parent directory. Episode paths
    must stay under that directory; state IDs, seeds, and physical fingerprints
    may not cross train/validation/test splits.
    """
    path = Path(path).resolve()
    manifest = json.loads(path.read_text())
    if manifest.get("schema") != "shakewm.manifest.v1":
        raise ValueError("expected shakewm.manifest.v1")
    if manifest.get("source_schema") != "shakebench.imu_wm.v1":
        raise ValueError("source schema must explicitly identify IMU-WM v1")
    ids, states, seeds, fingerprints = set(), {}, {}, {}
    for e in manifest["episodes"]:
        if e["id"] in ids or e["split"] not in {"train", "val", "test"}:
            raise ValueError("duplicate episode id or invalid split")
        ids.add(e["id"])
        split_keys = [("state_id", states), ("seed", seeds)]
        if "state_fingerprint" in e:
            split_keys.append(("state_fingerprint", fingerprints))
        for key, groups in split_keys:
            value = str(e[key])
            if value in groups and groups[value] != e["split"]:
                raise ValueError(f"{key} leaks across splits: {value}")
            groups[value] = e["split"]
        if e.get("valid", True) is not True:
            raise ValueError("invalid attempts belong in the audit ledger, not a training manifest")
        if not (path.parent / e["path"]).is_file():
            raise FileNotFoundError(e["path"])
        resolved = (path.parent / e["path"]).resolve()
        if not resolved.is_relative_to(path.parent):
            raise ValueError("episode path escapes manifest directory")
        if "source_sha256" in e and file_sha256(resolved) != e["source_sha256"]:
            raise ValueError("normalized episode source_sha256 mismatch")
    return manifest, path.parent


def load_episode(path):
    """Read one normalized NPZ episode and check its RGB, IMU, and time arrays."""
    with np.load(path, allow_pickle=False) as z:
        e = {k: z[k] for k in z.files}
    required = {"rgb", "rgb_time", "rgb_valid", "imu", "imu_acquisition_time", "imu_delivery_time", "imu_live", "end_time"}
    if not required <= e.keys():
        raise ValueError(f"missing normalized fields: {sorted(required - e.keys())}")
    n, m = len(e["rgb_time"]), len(e["imu"])
    if e["rgb"].dtype != np.uint8 or e["rgb"].ndim != 4 or e["rgb"].shape[0] != n or e["rgb"].shape[-1] != 3:
        raise ValueError("rgb must be uint8 [frames,height,width,3]")
    if e["imu"].shape != (m, 6) or e["rgb_valid"].shape != (n,):
        raise ValueError("invalid sensor shapes")
    if e["rgb_valid"].dtype != bool or e["imu_live"].dtype != bool:
        raise ValueError("valid/live arrays must be boolean")
    for key in ["imu_acquisition_time", "imu_delivery_time", "imu_live"]:
        if e[key].shape != (m,):
            raise ValueError(f"invalid {key} shape")
    for key in ["rgb_time", "imu_acquisition_time", "imu_delivery_time"]:
        if not np.isfinite(e[key]).all() or (np.diff(e[key]) <= 0).any():
            raise ValueError(f"non-monotonic/non-finite {key}")
    if (e["imu_delivery_time"] < e["imu_acquisition_time"]).any() or not np.isfinite(e["imu"]).all():
        raise ValueError("invalid IMU delivery or values")
    if (e["imu_live"] & (e["imu_acquisition_time"] < 0)).any():
        raise ValueError("negative reset prefill must be invalid")
    if not np.isfinite(e["end_time"]).all() or np.asarray(e["end_time"]).size != 1:
        raise ValueError("end_time must be a finite scalar")
    return e


def visual_indices(episode, config):
    """Select the regular RGB frames used at the model's visual sampling rate."""
    # Require acquisition-grid timestamps; no interpolation that can read future pixels.
    times = episode["rgb_time"]
    if len(times) > 1 and not np.allclose(np.diff(times), 1 / config.acquisition_hz, atol=1e-5, rtol=0):
        raise ValueError("RGB acquisition grid mismatch; mark missing frames invalid explicitly")
    return np.arange(0, len(times), config.acquisition_hz // config.visual_hz)


def imu_window(episode, cutoff, samples, imu_hz=200):
    """Build a complete IMU window using only samples delivered by `cutoff`.

    Returns a float32 `[samples, 6]` array and a flag indicating whether the
    window is complete, recent enough, and uniformly sampled at `imu_hz`.
    """
    acquisition = episode["imu_acquisition_time"]
    delivery = episode["imu_delivery_time"]
    live = episode["imu_live"]
    # Delivery field is authoritative; 5 ms is recorded rather than added again.
    eligible = np.flatnonzero(live & (delivery <= cutoff + 1e-9) &
                             (acquisition >= 0) & (delivery <= float(episode["end_time"]) + 1e-9))
    chosen = eligible[-samples:]
    complete = len(chosen) == samples
    if complete:
        complete = (cutoff - delivery[chosen[-1]] < 1 / imu_hz + 1e-7 and
                    np.allclose(np.diff(acquisition[chosen]), 1 / imu_hz, atol=1e-6, rtol=0))
    out = np.zeros((samples, 6), np.float32)
    if complete:
        out[:] = episode["imu"][chosen]
    return out, bool(complete)


def fit_normalization(manifest_path):
    """Fit per-channel IMU mean/std from live train-split samples only."""
    manifest, root = read_manifest(manifest_path)
    total = np.zeros(6, np.float64)
    squares = total.copy()
    count = 0
    train_sources = {}
    for row in manifest["episodes"]:
        if row["split"] != "train":
            continue
        e = load_episode(root / row["path"])
        train_sources[row["id"]] = file_sha256(root / row["path"])
        keep = e["imu_live"] & (e["imu_acquisition_time"] >= 0) & (e["imu_delivery_time"] <= float(e["end_time"]))
        x = e["imu"][keep].astype(np.float64)
        total += x.sum(0)
        squares += (x * x).sum(0)
        count += len(x)
    if not count:
        raise ValueError("no live train IMU for normalization")
    mean = total / count
    std = np.sqrt(np.maximum(squares / count - mean * mean, 1e-12))
    return {"mean": mean.tolist(), "std": std.tolist(), "count": count,
            "split_hash": digest_json(manifest), "source": "train_live_only", "train_sources": train_sources}


class WindowDataset(Dataset):
    """Create causal history/target windows for one manifest split.

    Exactly one visual-feature source is required: a verified disk cache or an
    online encoder. The dataset also records which short/long IMU windows are
    complete at each historical visual timestamp.
    """

    def __init__(self, manifest_path, split, config, normalization, cache_dir=None, encoder=None):
        """Validate contracts and index eligible origins without copying RGB/features.

        `normalization` must match the manifest and its train-file hashes. Supply
        either `cache_dir` or `encoder`; supplying both or neither is an error.
        """
        self.manifest, self.root = read_manifest(manifest_path)
        self.config, self.normalization = config, normalization
        if normalization["split_hash"] != digest_json(self.manifest):
            raise ValueError("normalization belongs to a different split manifest")
        sources = {r["id"]: file_sha256(self.root / r["path"]) for r in self.manifest["episodes"]}
        train_sources = {r["id"]: sources[r["id"]] for r in self.manifest["episodes"] if r["split"] == "train"}
        if "train_sources" not in normalization:
            raise ValueError("normalization predates data-content binding; rerun fit-norm")
        if normalization["train_sources"] != train_sources:
            raise ValueError("train data changed since normalization")
        if self.manifest.get("camera", config.camera) != config.camera:
            raise ValueError("manifest camera differs from configured input camera")
        if (cache_dir is None) == (encoder is None):
            raise ValueError("select exactly one of feature cache or online encoder")
        self.encoder, self.episodes, self.windows = encoder, [], []
        cache = None
        if cache_dir is not None:
            cache = json.loads((Path(cache_dir) / "index.json").read_text())
            if cache["manifest_hash"] != digest_json(self.manifest) or cache["visual_hz"] != config.visual_hz or cache["camera"] != config.camera:
                raise ValueError("cache manifest/rate/camera mismatch")
            if digest_json(cache["teacher"]) != cache["teacher_hash"]:
                raise ValueError("cache teacher contract is corrupt")
        self.teacher = cache["teacher"] if cache else encoder.contract
        # Bind the exact feature-source contract as well as raw episode content.
        # A reindexed modified cache must not silently resume an older run.
        # Online/cache mode changes require a new run rather than weakening provenance.
        self.manifest_hash = digest_json({"manifest": self.manifest, "episode_sha256": sources,
                                          "feature_source": cache if cache else {
                                              "kind": "online", "teacher": self.teacher}})
        # Coverage counters: eligible episodes, candidate origins, and complete short/long contexts.
        seen = candidates = short_ok = long_ok = 0
        for row in self.manifest["episodes"]:
            if row["split"] != split:
                continue
            e = load_episode(self.root / row["path"])
            # `idx` selects model-rate frames; the remaining arrays track frame and IMU validity.
            idx = visual_indices(e, config)
            times = e["rgb_time"][idx]
            valid = e["rgb_valid"][idx] & (times <= float(e["end_time"]) + 1e-9)
            eligibility = []
            for t in times:
                s, a = imu_window(e, t, config.short_samples, config.imu_hz)
                l, b = imu_window(e, t, config.long_samples, config.imu_hz)
                eligibility.append([a, b])
            eligibility = np.asarray(eligibility, bool)
            features = None
            if cache:
                entry = cache["episodes"][row["id"]]
                if entry["source_sha256"] != file_sha256(self.root / row["path"]):
                    raise ValueError("raw source changed since cache extraction")
                feature_path = Path(cache_dir) / entry["path"]
                if file_sha256(feature_path) != entry["sha256"]:
                    raise ValueError("cache feature checksum mismatch")
                features = np.load(feature_path, mmap_mode="r", allow_pickle=False)
                if tuple(features.shape) != (len(idx), self.teacher["patches"], self.teacher["feature_dim"]):
                    raise ValueError("cache shape mismatch")
            episode_id = len(self.episodes)
            # Cache mode must not retain the RGB dataset or duplicate 600-sample
            # windows per frame in RAM. Keep compact sensors and materialize only C windows.
            sensors = {k: e[k] for k in ["rgb_time", "imu", "imu_acquisition_time",
                        "imu_delivery_time", "imu_live", "end_time"]}
            self.episodes.append((row, sensors, idx, valid, eligibility, features))
            used = 0
            for origin in range(config.context - 1, len(idx) - 1, config.window_stride):
                left = origin - config.context + 1
                if not valid[left:origin + 1].all() or not valid[origin + 1:min(len(idx), origin + config.horizon + 1)].any():
                    continue
                candidates += 1
                # Eligibility must hold at every history step; V1 additionally needs long IMU.
                s_ok, l_ok = eligibility[left:origin + 1].all(0)
                short_ok += int(s_ok); long_ok += int(l_ok and s_ok)
                if s_ok and (config.eligibility == "v0" or l_ok):
                    self.windows.append((episode_id, origin)); used += 1
            seen += int(used > 0)
        self.coverage = {"episodes_total": len(self.episodes), "episodes_eligible": seen,
                         "candidate_windows": candidates, "short_eligible": short_ok,
                         "long_eligible": long_ok, "selected_windows": len(self.windows),
                         "eligibility": config.eligibility}

    def __len__(self):
        """Return the number of eligible episode/origin pairs in this split."""
        return len(self.windows)

    def __getitem__(self, index):
        """Materialize one training example at an episode's fixed forecast origin.

        The returned `history` and `targets` are visual features. `short` and
        `long` contain normalized `[C, 20, 6]` and `[C, 600, 6]` IMU windows;
        masks mark eligible IMU inputs and valid future targets. Teacher-forcing
        targets are the next visual feature at each of the C context positions.
        """
        # A dataset index resolves to one episode and its final history-frame index.
        episode_id, origin = self.windows[index]
        # Stored tuple: manifest row, compact episode arrays, frame indices,
        # validity/eligibility masks, and optional cached features.
        row, e, idx, valid, eligible, features = self.episodes[episode_id]
        # `c` and `h` are the configured context and forecast lengths.
        c, h = self.config.context, self.config.horizon
        # Slice from the first history frame through the available future horizon.
        start, stop = origin - c + 1, min(len(idx), origin + h + 1)
        if features is None:
            with np.load(self.root / row["path"], allow_pickle=False) as raw:
                rgb = raw["rgb"][idx[start:stop]]
            values = self.encoder.encode_numpy(rgb)
        else:
            values = torch.from_numpy(np.array(features[start:stop], dtype=np.float32))
        history = values[:c]
        targets = torch.zeros((h,) + tuple(history.shape[1:]), dtype=history.dtype)
        targets[:stop-origin-1] = values[c:]
        # `mask[j]` is true only when a real future visual target exists at horizon j.
        mask = torch.zeros(h, dtype=torch.bool)
        mask[:stop-origin-1] = torch.from_numpy(valid[origin + 1:stop].copy())
        # TF uses C true blocks to predict C next frames, with actual context lengths 1..C.
        tf_targets = torch.cat([history[1:], targets[:1]])
        tf_mask = torch.cat([torch.ones(c - 1, dtype=torch.bool), mask[:1]])
        mean = np.asarray(self.normalization["mean"], np.float32)
        std = np.maximum(np.asarray(self.normalization["std"], np.float32), 1e-6)
        sl = slice(start, origin + 1)
        times = e["rgb_time"][idx[sl]]
        short = np.stack([imu_window(e, t, self.config.short_samples, self.config.imu_hz)[0] for t in times])
        long = np.stack([imu_window(e, t, self.config.long_samples, self.config.imu_hz)[0] for t in times])
        return {"history": history, "targets": targets, "target_mask": mask,
                "tf_targets": tf_targets, "tf_mask": tf_mask,
                "short": torch.from_numpy((short - mean) / std),
                "long": torch.from_numpy((long - mean) / std),
                "eligible": torch.from_numpy(eligible[sl].copy()),
                "episode_id": row["id"], "state_id": row["state_id"],
                "origin_time": float(e["rgb_time"][idx[origin]])}


def build_cache(manifest_path, output, config, encoder, batch_size=8):
    """Encode selected RGB frames and stream their frozen features into an indexed cache.

    Cache arrays are FP16 and each index entry binds both the source episode and
    generated feature file by SHA-256.
    """
    manifest, root = read_manifest(manifest_path)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    index = {"schema": "shakewm.cache.v1", "manifest_hash": digest_json(manifest),
             "teacher": encoder.contract, "teacher_hash": digest_json(encoder.contract),
             "visual_hz": config.visual_hz, "camera": config.camera, "episodes": {}}
    for i, row in enumerate(manifest["episodes"]):
        e = load_episode(root / row["path"])
        selected = visual_indices(e, config)
        path = output / f"{i:06d}.npy"
        # Stream features to disk rather than keeping a dataset-wide tensor in memory.
        shape = (len(selected), encoder.contract["patches"], encoder.contract["feature_dim"])
        array = np.lib.format.open_memmap(path, mode="w+", dtype=np.float16, shape=shape)
        for j in range(0, len(selected), batch_size):
            array[j:j + batch_size] = encoder.encode_numpy(e["rgb"][selected[j:j + batch_size]]).numpy()
        array.flush(); del array
        index["episodes"][row["id"]] = {"path": path.name, "source_sha256": file_sha256(root / row["path"]),
                                        "sha256": file_sha256(path), "frames": len(selected)}
    (output / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    return index


def create_synthetic(output, seconds=2.0, image_size=32):
    """Create small synthetic RGB/IMU episodes for pipeline checks, never real-data evidence.

    The four episodes have disjoint synthetic state/seed identities and fixed
    train/validation/test assignments. The output directory must not exist.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {"schema": "shakewm.manifest.v1", "source_schema": "shakebench.imu_wm.v1",
                "synthetic": True, "episodes": []}
    for i, split in enumerate(["train", "train", "val", "test"]):
        end = seconds - (0.25 if i == 1 else 0)
        time = np.arange(round(end * 20) + 1, dtype=np.float64) / 20
        it = np.arange(-20, int(end * 200), dtype=np.float64) / 200
        signal = np.stack([np.sin(2 * np.pi * (1 + k / 10) * it + i) for k in range(6)], -1).astype(np.float32)
        yy, xx = np.mgrid[:image_size, :image_size]
        rgb = np.stack([np.stack([(xx * 3 + t * 20 + i * 17) % 255,
                                  (yy * 4 + t * 30) % 255,
                                  ((xx + yy) * 2 + t * 10) % 255], -1).astype(np.uint8) for t in time])
        path = output / f"synthetic_{i}.npz"
        np.savez_compressed(path, rgb=rgb, rgb_time=time, rgb_valid=np.ones(len(time), bool),
                            imu=signal, imu_acquisition_time=it, imu_delivery_time=it + 0.005,
                            imu_live=it >= 0, end_time=np.asarray(end))
        manifest["episodes"].append({"id": f"synthetic_{i}", "path": path.name, "state_id": f"synthetic_state_{i}",
                                     "seed": 100 + i, "split": split, "asset": "synthetic", "basis": "synthetic",
                                     "scenario": "synthetic_sine", "termination": "fixture_end", "valid": True})
    path = output / "manifest.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    return path
