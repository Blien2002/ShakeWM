# Normalized dataset adapter v1

This is a training adapter for `shakebench.imu_wm.v1`, not a redefinition of the collector schema. The collector's native layout has not yet been tested here. Export its exact timestamps, masks, sensor-frame values, camera selection and termination semantics into the following format; record the collector revision/config hash in the manifest. Do not rename fields while silently changing their meanings.

One NPZ per episode, loaded with `allow_pickle=False`:

| Key | Shape / dtype | Meaning |
| --- | --- | --- |
| `rgb` | `[T,H,W,3]` uint8 | Selected third-person RGB; production H=W=256, original 20 Hz grid |
| `rgb_time` | `[T]` float64 seconds | Shared clock, strictly increasing; retain missing grid frames with `rgb_valid=false` |
| `rgb_valid` | `[T]` bool | Valid observations only; no invented post-termination frames |
| `imu` | `[M,6]` float | Sensor-frame specific force xyz (m/s²) and angular velocity xyz (rad/s) |
| `imu_acquisition_time` | `[M]` float64 seconds | Actual 200 Hz sample time |
| `imu_delivery_time` | `[M]` float64 seconds | Actual delivery time, nominal acquisition+0.005; never add latency twice |
| `imu_live` | `[M]` bool | False for reset prefill / invalid observations |
| `end_time` | scalar float64 | Valid episode boundary, event timestamp or right censoring at limit |

Sensor clock origin, units, extrinsics, actual gravity, filter revision/warmup, source view, original schema, frozen scenario revision and terminal event definition belong in the manifest's audit metadata. The adapter uses no zero-phase filtering or future interpolation. A missing IMU sample invalidates windows spanning its gap. Initial FIR/TCN zero padding is computation, not observed history. Collector warmup invalidity must be reflected in `imu_live`; it is not inferred from a label.

The real geometry-based off-table detector belongs to collection. This library consumes its end time and validity; it does not pretend to reconstruct geometry from RGB. Preserve the second wrist camera and physical replay/GT in the source dataset; only the selected input camera is exported here.

Minimal manifest (paths are relative to the JSON file; add all episodes):

```json
{
  "schema": "shakewm.manifest.v1",
  "source_schema": "shakebench.imu_wm.v1",
  "synthetic": false,
  "collector_revision": "record-the-frozen-revision",
  "episodes": [
    {
      "id": "state001_S1_seed101",
      "path": "episodes/state001_S1_seed101.npz",
      "state_id": "exact-initial-state-fingerprint",
      "seed": 101,
      "split": "train",
      "asset": "mug",
      "basis": "upright",
      "scenario": "canonical_S1",
      "termination": "time_limit",
      "valid": true
    }
  ]
}
```

Split assignment is explicit and frozen before window construction; this library validates it rather than silently repairing it. For the planned 150 states, prepare a stratified 120/15/15 state assignment over the nine asset/basis strata with balanced rounding, and independent seed pools. Every rerender/replay of a state remains in its original split. Reject failed attempts from training; keep them in a separate attempt ledger with their original slot identity and failure reason. A valid off-table episode is retained, not replaced by a surviving episode.

The manifest and train normalization are hashed into checkpoints. Cache entries additionally hash raw NPZ and feature bytes. For online experiments keep source files immutable and archive their hashes with the manifest. Schema/audit fields are not network inputs. Changing actual pixel/sensor values changes the data version even if split assignment is unchanged.

At 10 Hz, each window has C history frames and up to H future targets. Every history frame requires 20 real, consecutive, already delivered samples for V0; V1 requires all 600 samples as well. C=10 gives earliest nominal origins around 1.0 s for V0 and 3.9 s for V1. Exact eligibility uses timestamps, not those rounded estimates. Future target masks can be partial. An origin is kept only if history is valid and at least one valid target exists. Report per-horizon counts and coverage; overlapping windows are not independent trials.

Use `configs/rgb_only.json` with V0 eligibility for matched V0/RGB-only comparison. To compare V1 fairly, set eligibility to V1 for every participating model before training. Retain separate coverage reports rather than masking away early terminations from the experiment narrative.
