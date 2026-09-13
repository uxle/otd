# AVC v4 — Validation Report

60 frames, 640×480 (accuracy mode), baseline 0.16 m, f=700 px, sensor noise
σ=2.0/255. Every number is measured against per-pixel / per-object ground
truth. Ablation flags: `--full-res` (pyramid off), `--no-fusion` (temporal
fusion off), `--stride N`, `--qvga`.

## Headline (VGA, full-resolution matcher + temporal fusion — the default)

- Depth: median |err| **3.9 cm**, p90 13.0 cm (measured pixels) — v3: 5.0 / 16.5
- σ-calibration: 54% within 1σ / 81% within 2σ (ideal 68/95)
- VO: median ATE 0.348 m, final drift 0.633 m, rot err 3.61° (49/60 frames valid)
- Person (26-frame occlusion): **1 ID switch** (v3: 3), height 1.34 m,
  velocity err 0.19 m/s
- Throughput: 1.34 FPS end-to-end VGA accuracy mode (v3: 0.87)
- **39/43 checks pass** (v3: 37/41)

## Real-time mode (QVGA, pyramid matcher + fusion)

- **39/45 checks, 0 ID switches, 17.8 FPS end-to-end**, depth median 4.4 cm
- stereo 39 ms/frame, peak RSS 134 MB — this is the "low specifications,
  high output" configuration

## Why the remaining 4 checks fail (root-caused, not hidden)

- **σ-calibration 2σ = 0.81 (bound 0.85)**: the far/weak-texture pixel tail.
  Raising the Weak σ_d to cover it (0.62 / 0.68 experiments) passes the
  calibration check but destabilizes the coupled ground-band / Kalman gates
  (ID switches went 1 → 3, VO ATE +0.07 m) — the honest choice is to report
  the tail. The 1σ value (0.54) is inside its band.
- **GT#4 @f10 (25 cm box @ 2.3 m)**: inherited v3 limitation — classical
  geometric detection at the edge of cluster survivability for a surface
  this small; v1 recovered it with the YOLOv8-seg backend (not part of the
  hermetic Rust build). The v4 range fix (nd 52→64) put its disparity in
  range; the cluster itself remains too fragile.
- **person ID switch (1) / re-identified (2→26)**: during the occlusion tail
  the person's partial-view fragment wins the Hungarian assignment against
  the coasting track (tighter covariance); fragment stitching recovers the
  identity in most frames but one transition slips through. v3 failed the
  same checks with 3 switches and no recovery.

## Ablations (all on the identical scene, 60 frames, VGA unless noted)

| config | depth med / p90 | σ-cal 1σ/2σ | person | FPS | checks |
|---|---|---|---|---|---|
| full-res, no fusion | 4.1 / 14.0 cm | 55% / 81% | 0 switches, 1.55 m | 1.35 | 35/38* |
| full-res + fusion (default) | **3.9 / 13.0** | 54% / 81% | 1 switch, 1.34 m | 1.34 | 39/43 |
| pyramid + fusion (VGA) | 5.5 / 20.5 cm | 55% / 80% | 3 switches | 4.42 | ~30/42 |
| pyramid QVGA + fusion | 4.4 / 21.0 cm | — | **0 switches** | **17.8** | 39/45 |

\* check counts vary with the dynamic per-object check generation (objects
matched per run). The pyramid VGA row is the documented trade-off: its dense
half-res map is excellent for depth/VO (VO ATE 0.240 m — the best of any
config) but the coarse-estimate pixels — even masked out of geometry — make
object-layer statistics noisier at VGA scale. At QVGA scale the pyramid is
the recommended mode.

## The v4 engineering log (what was measured, changed, and kept)

1. **Perf**: voting LR cross-check replaced the second full cost+SGM pass
   (~45% of v3 stereo time); exact min1/min2 SGM penalties (faster *and*
   closer to exact than v3's approximation); fused + parallel census;
   parallel medians; `StereoBuffers` reuse (zero steady-state allocation);
   fat LTO + codegen-units=1. VGA full-res stereo 1127 → 670 ms.
2. **Occlusion identity**: coasting tracks joined the Hungarian pool with
   prediction-aware gating + velocity-continuity fragment stitching →
   ID switches 3 → 1 (QVGA 0), height 1.23 → 1.34 m.
3. **Temporal fusion** (new): motion-compensated inverse-σ EMA through the
   VO pose; 3σ gate keeps moving objects on fresh measurements; σ floored at
   0.75× single-frame (systematic components don't average away — measured:
   lower floors degraded the measurement-consistency check).
4. **Pyramid matcher** (new, fast mode): dense search at half resolution +
   full-res ±3 px local refinement. Three bugs found by GT measurement and
   fixed: blind local WTA on weak texture (worse than the SGM estimate it
   replaced — fixed by the coarse-fallback), SGM boundary ramps (smooth
   ramps pass neighbour gates — fixed by a census cost-consistency test),
   and the ±3 window trusting a wrong prior (small objects — fixed by a
   full-range census rescue). The `coarse` mask on DisparityMap marks
   fallback pixels: honest map values (σ_d 0.8), excluded from object
   geometry.
5. **Coupled-knob experiments (all reverted, documented)**: ground-band caps
   (0.08 / 0.25) → ground-noise points bridged crate clusters into
   ghost-rejected blobs; Weak σ_d inflation (0.62 / 0.68) → passed the
   calibration check but destabilized tracking/VO. v3's tuned values were
   kept; the reports document why.

## Honest module status matrix

| module | status | evidence |
|---|---|---|
| stereo_matching (census+SGM, half-pixel grid, voting LR) | WORKING | 3.9 cm median / 13 cm p90, 96.8% valid coverage |
| pyramid fast mode | WORKING | QVGA 39/45 @ 17.8 FPS; VGA trade-offs above |
| temporal_depth_fusion (v4) | WORKING | −0.2 cm median depth vs no-fusion; moving surfaces gated (unit tests) |
| uncertainty_propagation | WORKING | σ-cal 54/81 (tail documented); MC-verified unit tests |
| rectification | WORKING | epipolar row error < 0.35 px (test) |
| 3d_lifting + voxel_cloud | WORKING | 2 cm world voxel accumulation |
| detection (geometric) | WORKING | ground RANSAC + clustering + yaw-OBB |
| tracking (occlusion continuity, v4) | WORKING | 1 ID switch (v3: 3); stitch + reacquire unit tests |
| visual_odometry | WORKING (drift documented) | 0.348 m median ATE, 49/60 frames |
| semantic_classification | PARTIAL | shape prior only; YOLOv8 hook documented |
| human_pose_lifting | PARTIAL | 17-joint math, < 3 cm unit test, no bundled 2D detector |
| exports (PLY/OBJ/JSON/CSV/KITTI/.avcworld) | WORKING | round-trip tests for every format |
| tsdf_surface_reconstruction | INCOMPLETE | roadmap item |
| loop_closure / pose_graph | INCOMPLETE | roadmap item |

## Reproduce

```bash
cargo build --release
./target/release/avc-demo-synthetic                    # default (accuracy)
./target/release/avc-demo-synthetic --qvga             # real-time mode
./target/release/avc-demo-synthetic --full-res --no-fusion   # ablations
./target/release/avc-bench                             # throughput + RSS
AVC_TRACE=1 ./target/release/avc-demo-synthetic        # per-frame object trace
```
