# Changelog

## v4.0.0 — the deep optimization release ("low specifications, high output")

Performance:
- Voting LR cross-check (disparity voting) replaces the second full
  cost+SGM pass — ~45% of v3 stereo time removed, same rejection semantics.
- Exact two-smallest (min1/min2) SGM penalties — faster and closer to exact
  SGM than v3's `min + P2` approximation.
- Census transform: fused single-pass fine+ring, row-parallel, reusable
  buffers (`census_into`).
- Medians row-parallel, no per-pixel allocation.
- `StereoBuffers`: all matcher scratch caller-owned — zero steady-state
  large allocation; peak RSS measured per config in `avc-bench` (new).
- Release profile: fat LTO, codegen-units 1.
- Measured: VGA accuracy-mode stereo 1127 → 670 ms (1.7×), QVGA stereo
  274 → 36.5 ms (7.5×), QVGA end-to-end 3.6 → 17.8 FPS.

New architecture:
- **Pyramid matcher** (fast mode): dense census+SGM at half resolution +
  full-res ±3 px local refinement with a coarse-estimate fallback. Three
  GT-measured fixes: cost-consistency trust test for the fallback (SGM
  boundary ramps), full-range census rescue when the window holds no
  confident match, and a `coarse` mask on DisparityMap (honest map values
  with σ_d 0.8, excluded from object geometry — boundary smearing bridges
  objects).
- **Temporal depth fusion** (`stereo::temporal`): motion-compensated
  (through the VO pose) inverse-σ EMA with a 3σ gate; moving objects keep
  fresh measurements; fused σ floored at 0.75× (systematic error components
  do not average away). Depth median 4.1 → 3.9 cm on the validation world.
- **Occlusion identity continuity** (tracker): Coasting tracks rejoin the
  Hungarian pool with prediction-aware Mahalanobis gating; fragment
  stitching transfers occluded identities onto fragments that won the
  assignment during the dark period, with motion-continuity gates.
  ID switches 3 → 1 (VGA) / 0 (QVGA); person height 1.23 → 1.34 m.
- `EngineConfig::temporal_fusion` flag; demo/validation ablation flags
  (`--full-res`, `--no-fusion`, `--stride`, `--qvga`); `AVC_TRACE` per-frame
  object trace.

Validation: 85 tests (82 unit + 3 integration, up from 75). VGA accuracy
mode 39/43 checks (v3: 37/41); QVGA real-time mode 39/45 @ 17.8 FPS.
Failure root causes + the coupled-knob experiment log (ground-band caps,
σ_d recalibration attempts — all reverted with reasons) in docs/VALIDATION.md.

## v3.0.0 — the Rust edition

Full engine as one hermetic crate (4 pure-Rust deps, vendored, offline
build, prebuilt binaries): SE(3) with MC-verified uncertainty propagation,
Fusiello rectification, two-scale census + 8-dir SGM on a half-pixel slice
grid, geometric detection (gravity-prior ground RANSAC, voxel clustering,
yaw-OBB), 6-DoF Kalman tracking with anchored coasting, σ-weighted Kabsch
VO, voxel cloud, relations/measurement engines, 7 export formats,
60-frame GT validation (37/41), 75 tests.

## v2.0.0 (Python) — the competitive upgrade

Benchmark harness vs OpenCV SGBM (won 6/8 metrics, losses documented);
two-scale census + half-pixel cost grid; JointKalmanBank skeleton;
calibration v2 with honest σ floors; 42/42 GT validation, 46 tests.

## v1.0-v1.1 (Python) — the original

28 modules, YOLOv8n ONNX backends, NCC subpixel + SGBM, 3D Kalman tracking,
VO + pose graph, TSDF mesh, 8 exports; 33/42 → 38/42 GT validation.
