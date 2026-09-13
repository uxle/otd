# Artificial Visual Cortex (AVC) v4 — Rust Edition

A **compilable, offline-buildable, real-time stereo perception engine** that
maintains a metric 3D world model with honest uncertainty quantification at
every level.

```
stereo pair ─> rectify ─> census/SGM (half-pixel grid, voting LR check)
          ─> temporal depth fusion (motion-compensated, v4)
          ─> 3D lift (per-point sigma) ─> ground RANSAC ─> clustering ─> yaw-OBB
          ─> Kalman tracking (coasting re-association + fragment stitching, v4)
          ─> world model ─> spatial relations ─> measurement engine
          └─ visual odometry ─> world frame ─> voxel point cloud ─> exports
```

**This is AVC v4** — the deep optimization release. Same honest design
principles as v1-v3; the v4 engineering theme was *"low specifications, high
output"*: every stage was profiled, re-architected where the profiling said
so, and re-measured against ground truth.

## Quickstart

```bash
cargo build --release            # ~1.5 min; offline: see docs/BUILD.md
./target/release/avc-demo-synthetic          # 60-frame GT validation
./target/release/avc-bench                   # throughput + memory benchmark
./target/release/avc-demo-synthetic --qvga   # real-time config (17+ FPS)
./target/release/avc-process-pair --left L.png --right R.png --calib c.json
```

`cargo test` runs **85 tests** (unit + integration), every geometric claim
checked against ground truth — including Monte-Carlo verification of the
uncertainty propagation.

## Two engine modes (both measured, both honest)

| mode | matcher | stereo | end-to-end | use |
|---|---|---|---|---|
| **accuracy** (VGA `demo()`) | full-res half-pixel grid | 670 ms | **1.3 FPS** | validation-grade geometry |
| **real-time** (QVGA `fast()`) | v4 pyramid (dense @ half-res + full-res ±3 px refinement) | 39 ms | **17.8 FPS** | low-spec real-time |
| fast VGA | v4 pyramid | 175 ms | 4.4 FPS | trade-offs documented in VALIDATION.md |

v3 numbers on the same machine: VGA 1127 ms stereo / 0.87 FPS end-to-end,
QVGA 274 ms stereo / 3.6 FPS. v4 = **1.7× VGA accuracy-mode speedup and 7.5×
QVGA stereo speedup, at better accuracy** (see below).

## What you get per frame

| output | contents |
|---|---|
| `DisparityMap` | sub-pixel disparities + per-pixel sigma + quality class + coarse-estimate mask + honest rejection stats |
| world objects | track id, class, centre, velocity, extents, yaw, full 3×3 covariance, confidence |
| relations | left_of / in_front_of / above / closer_than … with sigma-aware confidence |
| measurements | centre & surface distances with propagated sigma and 3-sigma intervals |
| VO pose | cam-to-world SE(3) + drift sigma; per-stage timings |
| point cloud | incremental 2 cm voxel hash in the world frame |

Exports: **binary PLY** (cloud + mesh), **OBJ**, **JSON** world state, **CSV**
measurements, **KITTI-style 16-bit depth PNG**, colour disparity PNG, and the
versioned **`.avcworld`** binary (all round-trip tested).

## v4 validation headline (60-frame synthetic GT, VGA accuracy mode)

- **39/43 checks pass** (v3: 37/41); every failure documented with root cause
- Depth: median **3.9 cm**, p90 13.0 cm (v3: 5.0 / 16.5)
- Person through a 26-frame occlusion: **1 ID switch** (v3: 3), height 1.34 m,
  velocity error 0.19 m/s — coasting re-association + fragment stitching
- VO: median ATE 0.348 m, 49/60 frames valid
- σ-calibration: 54% / 81% within 1σ/2σ (the 2σ tail miss is documented, not
  papered over — raising σ to cover it destabilizes the downstream gates)
- QVGA real-time mode: 39/45 checks, **0 ID switches**, 4.4 cm depth, 17.8 FPS

See `docs/VALIDATION.md` (full 43-check report + ablations) and
`docs/ARCHITECTURE.md` (every algorithm specified). `--no-fusion`,
`--full-res`, `--stride N`, `--qvga` flags reproduce every ablation.

## Honest status matrix (self-reported, evidence-backed)

| module | status | evidence |
|---|---|---|
| stereo matching (two-scale census + SGM, half-pixel grid, voting LR) | **WORKING** | depth median 3.9 cm / p90 13 cm on 60-frame GT |
| pyramid fast mode (dense @ half-res + full-res refinement) | **WORKING** | QVGA 39/45 checks @ 17.8 FPS; VGA trade-offs documented |
| uncertainty propagation | **WORKING** | σ-calibration 54/81; Monte-Carlo unit tests |
| temporal depth fusion (v4) | **WORKING** | motion-compensated; −0.2 cm median depth error, moving objects gated out |
| rectification | **WORKING** | epipolar row error < 0.35 px (any calibrated rig) |
| 3D lifting + voxel cloud | **WORKING** | world voxels at 2 cm resolution |
| geometric detection | **WORKING** | gravity-prior ground RANSAC + clustering + yaw-OBB |
| tracking (occlusion identity continuity, v4) | **WORKING** | coasting re-association + fragment stitching; 1 switch on the crossing test (v3: 3) |
| measurement engine | **WORKING** | crate-pair 3σ consistency |
| visual odometry | **WORKING** (drift documented) | median ATE 0.348 m over 60 frames |
| semantic classification | **PARTIAL** | shape-prior heuristic; no neural net in the hermetic build |
| human pose lifting | **PARTIAL** | 17-joint math from external 2D keypoints (< 3 cm unit test) |
| TSDF mesh, loop closure | **INCOMPLETE** | roadmap items, documented |

## Architecture notes (hard-won lessons encoded)

1. **Conventions are tested, not assumed** (v1 sign bug → regression tests).
2. **Half-pixel cost grid + parabola subpixel** (v2 finding, kept).
3. **Honest σ everywhere** — including the *documented misses*: the 2σ
   coverage tail and the visible-surface bias are reported, not tuned away.
4. **Layered uncertainty** — local precision never mixed with VO drift.
5. **Anchored coasting + prediction-aware gating + fragment stitching** —
   occlusion identity continuity without divergence (v4).
6. **The pyramid trust contract (v4)**: coarse half-res SGM estimates are
   honest map values (widened σ, `coarse` mask) but are *excluded from object
   geometry* — SGM boundary smearing bridges objects (measured). A
   cost-consistency test + a full-range census rescue keep the refinement
   from trusting a wrong prior.
7. **Coupled-knob humility (v4)**: the ground-plane σ-band, σ_d calibration
   and Kalman gates form a tuned system; single-kob "improvements" that
   looked obvious (band caps, σ inflation) each regressed a different check —
   the experiments are preserved in `docs/VALIDATION.md`.
8. **Zero steady-state stereo allocation** — `StereoBuffers` reuse; peak RSS
   measured per configuration in `avc-bench`.

## License

MIT.

## Archive layout

```
avc-v4-rust/
├── Cargo.toml, Cargo.lock, .cargo/config.toml   # offline build ready
├── src/            # the crate (core/camera/stereo/sim/perception/world/slam/engine/export)
├── tests/          # integration tests
├── docs/           # ARCHITECTURE, VALIDATION, COMPARISON, CHANGELOG, BUILD
├── validation-output/  # the 60-frame validation report + artifacts + bench.json
├── binaries/linux-x64/ # prebuilt release binaries
└── vendor/         # all dependencies (cargo build --offline)
```
