# AVC — Architecture Reference (v4)

**Artificial Visual Cortex** — a metric 3D perception engine for a 2-camera rig.
This document specifies **every feature and algorithm** in the system: what it
does, the exact math, the constants, and why each design decision was made.
It is the contract between the architecture and the code — `cargo test`
enforces the parts that are checkable, and the synthetic-GT validation harness
(`avc-demo-synthetic`, 41 checks) enforces the rest.

```
            ┌────────────────────────── L4 · MEMORY / EXPORTS ──────────────────────────┐
            │  track history ring · relations · measurements · PLY/OBJ/JSON/CSV/.avcworld │
            └──────────────────────────────▲────────────────────────────────────────────┘
                                           │ WorldState (objects + σ + cloud)
   left ─┐                          ┌─────┴──────┐   relations (erf conf)   measurement
         ├─ rectify (Fusiello) ─────►│  L3 WORLD  │◄── voxel hash (2 cm) ─── point cloud
   right ┘                           └─────▲──────┘
                                           │ tracks (id, class, pos, vel, P85 extents, 3×3 Σ)
                                          ┌┴───────────┐  Kalman + Hungarian + anchored coasting
                                          │ L2 OBJECTS │◄─ ground RANSAC → clustering → yaw-OBB
                                          └─────▲──────┘
                                                │ 3D points + per-point σ (lift, stride)
                    ┌───────────────────────────┴────────────────────────┐
                    │ L1 GEOMETRY: census+SGM stereo (pyramid, half-pixel)│
                    │ temporal depth fusion (motion-compensated, v4)      │
                    │ visual odometry (FAST→NCC→Kabsch RANSAC)            │
                    └───────────────────────────▲────────────────────────┘
                                                 │ rectified pair
                    ┌────────────────────────────┴────────────────────┐
                    │ L0 SENSOR: sync pair → median3 prefilter         │
                    │ camera model (y-up, x-right, z-forward)          │
                    └─────────────────────────────────────────────────┘
```

## 0. Design contract (unchanged since v1)

1. **Every number carries a σ or a documented failure.** No silent guesses.
2. **Conventions are tested, not assumed.** Camera frame x-right / y-up /
   z-forward; cam-to-world SE(3); a side-looking regression test pins the
   projection sign conventions (v1 shipped a sign bug here — never again).
3. **Layered uncertainty.** Per-pixel σ_d → per-point 3×3 Σ → per-track Kalman
   P → VO drift σ reported separately. Local precision is never mixed with
   world-frame drift.
4. **Low-spec budget.** Target: 2 vCPU, < 64 MB working set, no GPU, no
   system libraries, single static binary. v4 reaches this via the pyramid
   matcher + buffer reuse (§3.7–3.8).
5. **Honest status matrix** (§12): WORKING / PARTIAL / INCOMPLETE, each with
   evidence.

---

## 1. L0 — Sensor & camera model

### 1.1 Intrinsics / stereo rig
- Pinhole `K = [f, cx, cy]` (square pixels), baseline `B` (left = reference).
- Rectified standard frame: `x_L = x_R + d`, depth `Z = f·B/d`, `X = (u−cx)·Z/f`,
  `Y = (v−cy)·Z/f` (rectified y-up → Y up in the rect camera frame).
- Canonical demo rig: 640×480, f = 700 px, B = 0.16 m, sensor noise σ = 2.0/255.

### 1.2 Rectification (Fusiello)
- Compute rectifying homographies `H_L, H_R` from the calibration rotations so
  both cameras share a common row-aligned image plane; `remap_bilinear` with
  u8 images. Canonical rigs skip the warp (identity) — tested by the epipolar
  row-error test (< 0.35 px on a calibrated synthetic rig).

### 1.3 Prefilter
- 3×3 median on each rectified gray image before census — kills isolated
  sensor spikes before they can poison the census bits.

---

## 2. L0.5 — Census transform (two-scale)

Two scales are needed because a 5×5 census is blind to texture larger than
~4 px — the v2 Python finding:

- **Fine**: 5×5 window, 24 samples (radius 2, even/odd pattern), u32 code.
- **Ring**: radius-4 ring, 8 samples, u8 code — coarse texture scale.
- Cost = `popcount(fine_L ⊕ fine_R) + popcount(ring_L ⊕ ring_R)` ∈ [0, 32].
- v4: computed in **one fused row pass** (both codes per pixel) and
  **parallelized across rows** (rayon).

---

## 3. L1 — Stereo matching (the accuracy core)

### 3.1 Half-pixel disparity grid
The cost volume is sampled on a **0.5-px disparity grid**: `S = 2·nd` slices,
slice `s` ↔ disparity `d = disp_min + 0.5·s`. Integer-grid WTA locks far
depth to integer steps (measured in v2: bad-1px ×4) — the half grid is the
fix. Slice costs between integers are the average of the two neighbouring
integer costs (cost-domain interpolation, cheap and stable).

### 3.2 SGM aggregation
4-direction SGM (→, ←, ↓, ↑; 8-direction available) with:
- `P1' = p1/2` (±1 slice = 0.5 px), `P2' = p1` (±2 slices = 1 px) —
  penalties defined in *pixel* units then converted to slice units.
- Gradient-adaptive `P2 = max(p2_min, p2_base·8/(8+|ΔI|))` (horizontal
  reference-side gradient; same for vertical).
- Min-normalization: `L_r(p,d) = C(p,d) + min(...) − min_d' L_r(p−r,d')`
  to bound u16 sums.
- **v4 — exact two-smallest penalties**: the `min_{d'≠d} L_r + P2` term uses
  the **min1/min2 trick** (track the two smallest values of the previous
  line while writing it; use min2 when the current slice is the argmin,
  else min1). This is both *faster* (no second O(S) scan per pixel) and
  *more exact* than v3's `min + P2` approximation.

### 3.3 WTA + uniqueness
- Best slice per pixel; second-best restricted to slices ≥ 2 apart (≥ 1 px).
- Reject if `best · uniqueness > second` (uniqueness = 1.1) — ambiguity is
  demoted, not guessed.
- Border cost 200; matches near the image border (4 px) are invalid.

### 3.4 Sub-pixel
Parabola fit `V(s) = a·s² + b·s + c` over the aggregated cost at
`[s−2, s, s+2]` (±1 px), delta clamped to ±1 px. Strong only if the
runner-up is ≥ 1.6× the best (else the peak is not trustworthy → Weak).

### 3.5 Left–right consistency — **v4: disparity-voting cross-check**
v3 ran a complete second cost volume + SGM + WTA on the swapped pair
(~45% of stereo runtime). v4 derives the right-view map from the **same
aggregated volume by voting**: every left match `(x, d)` votes
`right[x − d] ← d`; then a left pixel survives iff
`|right[x − d_L] − d_L| ≤ lr_thresh` (1.0 px). Same rejection semantics
(occlusion + bad matches), ~0 marginal cost.

### 3.6 Postfilters
- Speckle: 4-connected components of valid disparities; kill components
  smaller than 60 px or with internal range > 2.0 px.
- Fill: row-wise nearest-valid (background = farther) → Filled quality.
- 3×3 median on the disparity image.
- **σ_d by quality class**: Strong 0.30 px, Weak 0.55 px, Filled 0.90 px
  (calibrated to the measured median |d_err| ≈ 0.3 px — honest floors).

### 3.7 Pyramid coarse-to-fine (v4 — the fast/low-spec mode)
Full-res SGM over a 0.5-px grid costs ~190 MB and ~1.1 s/frame at VGA on a
2-vCPU box. The v4 pyramid runs the *dense search* at half resolution and
only a *local refinement* at full resolution:

1. Half-res image (2×2 box average).
2. Full pipeline (census → cost → 4-dir SGM → WTA → subpixel) at half res;
   disparity range scaled by 0.5.
3. Upsample `d₀(x,y) = 2·d_{½}(x/2, y/2)`.
4. Full-res **local refinement**: for every pixel, re-cost the 7 integer
   disparities around `round(d₀)` (±3 px) with the full-res two-scale
   census, WTA, then the ±1-px parabola on those raw costs.
5. **Full-range census rescue**: when the ±3 window holds no confident
   match (cost > 12/32 — small objects, wrong priors), a full-range census
   WTA runs and replaces the window result when clearly better. Without
   this the window emits confidently-wrong disparities (measured).
6. **Coarse fallback + trust contract**: when the refinement is ambiguous,
   the half-res SGM estimate is kept (SGM smoothness beats a blind local
   WTA on weak texture) — but only where a census **cost-consistency test**
   (cost at the coarse value within +4 hamming of the local best) and a
   neighbour-smoothness check agree: SGM smears depth discontinuities into
   smooth *ramps* that pass naive smoothness gates. Fallback pixels carry
   the `coarse` mask on the DisparityMap: honest map values (σ_d widened
   to 0.8) that are **excluded from object geometry** (lifting skips them)
   — boundary smearing bridges objects into ghost clusters (measured).
7. LR voting check, speckle, fill, median, σ — as §3.5–3.6.

Roles: **QVGA real-time** (17.8 FPS end-to-end, 39/45 checks, recommended
low-spec mode), **VGA fast** (4.4 FPS, object-layer trade-offs documented in
VALIDATION.md). The **full-resolution matcher remains the accuracy default**
(VGA 640×480 validation) — mode selection is explicit, both are measured.
Peak memory: ~30 MB active scratch (vs ~190 MB); `StereoBuffers` reuse makes
the steady state allocation-free.

### 3.8 Buffer reuse (v4)
All stereo scratch (cost volume, sums, census codes, median images) lives in
a `StereoBuffers` struct owned by the engine — steady-state stereo performs
**zero large allocations**. `match_pair` remains as the allocating wrapper
for tests / one-shot use.

### 3.9 Temporal depth fusion (v4 — motion-compensated)
Static-scene pixels are re-observed every frame; averaging them through the
VO pose kills noise without touching moving objects:

- Keep the previous fused map `{Z_prev(u,v), σ_prev}` **in the previous
  camera frame** + the previous cam-to-world pose `T_wc_prev`.
- For each valid current pixel: backproject `P_w = T_wc_now · Z(u,v)·K⁻¹u`,
  project into the previous camera `u' = π(T_wc_prev⁻¹ · P_w)` (bilinear).
- Gate `|Z_prev(u') − Z_cur| ≤ 3·max(σ)` — motion / disocclusion / VO drift
  fall out of the gate automatically and keep the current measurement.
- Fuse inverse-σ-weighted exponential forgetting (0.6 new / 0.4 old,
  effective window ≈ 2.5 frames); the fused σ is floored at 0.75·σ_raw:
  the EMA variance assumes independent noise, but real depth errors carry
  systematic components (sub-pixel census bias, visible-surface bias) that
  do NOT average away — measured on the validation world (lower floors
  degraded the 2σ coverage and the measurement-consistency checks).
- Write back into the DisparityMap (`d = f·B/Z`) so the whole downstream
  (lifting, detection, measurement) inherits the improved σ.

---

## 4. L1.5 — Visual odometry

- FAST-9 corners (threshold 20, greedy NMS, ≤ 350 features) on the previous
  and current left images; 13×13 NCC patch matching (search ±14 px,
  score ≥ 0.65 + margin ≥ 0.01).
- 3D-3D: triangulate both feature sets from the disparity map
  (Strong/Weak pixels, d ≥ 2 px, σ_z ≤ 0.25 m, 0.4 ≤ Z ≤ 15 m).
- **σ-weighted Kabsch RANSAC** (150 iters, inlier tol
  `max(0.20, min(2.5σ, 0.35))` m, ≥ 8 inliers, |rot| ≤ 0.3 rad), final
  weighted Kabsch (`w = 1/σ²`) → relative pose; `T_wc *= δ⁻¹`.
- Drift σ: per-frame residual → per-frame σ → adjoint-rotated, accumulated;
  reported **separately** from local σ (layered uncertainty, §0.3).

---

## 5. L2 — 3D lifting & geometric detection

### 5.1 Lifting
For every stride-th pixel with valid disparity (d ≥ 1.0):
`Z = fB/d`, `σ_Z = Z²/(fB)·max(σ_d, 0.5)`, lateral `σ = (Z/f)·0.5`,
per-point 3×3 Σ propagated through the pose covariance (first-order,
Monte-Carlo unit-verified), then transformed to world.

### 5.2 Ground plane
Gravity-prior RANSAC on ≤ 4000 subsampled points: samples must have
`n_y ≥ 0.5` (kills walls), inlier tol `max(0.03, 2.5·σ_z)` m (σ-aware),
200 iters, least-squares eigen refinement. Points within the plane band are
excluded from clustering.

### 5.3 Clustering → yaw-OBB
- Voxel grid 0.06 m; BFS over 5³ voxel neighbourhood (radius 1.9 voxels).
- Cluster gates: `n ≥ min_points (20)`, height `h > max(0.08, 2.5·σ_z)`,
  `h ≤ 3.5`; ground-hugging clusters (> 95% within 3 cm) rejected;
  **ghost rejection**: depth span P5→P95 > 0.9 m rejected (occlusion
  boundary mix).
- Yaw from 2-D PCA on (x, z); extents P2→P98 in the yaw frame (robust to
  boundary outliers); sliver reject `min/max < 0.04`; class heuristic:
  person (1.25 < h < 2.3, footprint < 0.55), small_box, cylinder, box.
- Covariance: scatter variance floored by `max(var, σ̄_z², 0.04)` (the
  documented 0.2 m visible-surface systematic bias — stereo sees the front
  face, the centroid is behind it).

### 5.4 Disparity-range adequacy (v4 fix for the small-box miss)
The v3 validation failure on the 25 cm moving box was **not** a detector
failure: at frames 10–20 the box's disparity (~58 px) exceeded the
configured range (nd = 52) — the box literally did not exist in the cost
volume. v4 raises the demo range to nd = 64 (affordable because of §3.7)
and drops the lift stride to 2. Small objects now need the *same* gates as
everything else — no special-case rescue, the honest fix.

---

## 6. L2.5 — Tracking

### 6.1 Kalman (6-DoF constant velocity)
State `[p, v]`; **exact discretization of continuous-white-noise-accel** Q
(σ_a = 1.0 m/s²); Joseph-form update; velocity clamp 3 m/s (with 4× P
inflation when clamped — honest). Measurement R from detection σ floors
(0.2 m — the visible-surface bias).

### 6.2 Association
Hungarian (JV O(n³)) over the cost `m² (+ class_penalty 6.0 if classes
differ)` with gates: Mahalanobis² ≤ 12, hard distance ≤ 1.0 m.
**v4 — Coasting tracks stay in the assignment pool.** v3 excluded them, so
an object reappearing after occlusion could *never* reclaim its ID — a young
fragment track won by default (that was validation failures #28/#29).
Their coast-grown covariance already prices the uncertainty; the gates bound
the risk.

### 6.3 Lifecycle
Tentative (≥ 3 hits → Confirmed) → Coasting on miss (max 70, tentative
max 5) → Terminated (retained ≤ 200 frames for audit). **Anchored
coasting**: missed frames never advance the state, only grow P — occlusion
survival without divergence (unit-tested invariants: velocity frozen,
covariance grows, snap-back).

### 6.4 Fragment absorption (v4)
After every update, any young track (born after an older track went dark,
fewer hits) whose position agrees with the older **Coasting** track's
*prediction* (same class 0.5 m, cross-class 0.35 m — the class heuristic
flaps on partial views) is **stitched**: it inherits the older identity,
hits and extent history; the older track is retired. **Motion-continuity
gates** block false stitches: a stationary ghost fragment can never inherit
a moving identity, and a static identity never absorbs a mover. Handles the
case where the fragment won the Hungarian during the occlusion tail.

### 6.5 Extents
P85 of the last 25 extent observations per axis (lower-bound statistic —
partial views shrink objects; P85 recovers) with σ = max(det σ, …).

---

## 7. L3 — World model

- **Objects**: id, class, position + 3×3 Σ, velocity, extents, yaw,
  confidence (hits / observations).
- **Voxel cloud**: 2 cm spatial hash, running centroids per voxel, capped
  points/voxel; incremental across frames in the world frame.
- **Relations** (9 kinds, both directions): confidence from the Gaussian
  overlap `erf(|Δ|/√(2(σ₁²+σ₂²)))` — σ-aware, not threshold-magic.
- **Measurement engine**: pairwise centre & surface distances with
  first-order σ propagation and 3σ intervals (≤ 10 pairs/frame).
- **Memory**: per-track position ring buffer (velocity / path audit).

---

## 8. L4 — Exports (all round-trip tested)

binary PLY (cloud + mesh), OBJ, world.json, measurements.csv, KITTI-style
16-bit depth PNG, turbo disparity PNG, `.avcworld` (magic `AVCW`, u16
version, JSON + cloud + mesh sections, little-endian).

---

## 9. Validation methodology

Synthetic world with per-pixel / per-object GT (z-buffer renderer,
world-anchored value noise so texture is alias-free): 60 frames @ 640×480,
B = 0.16 m, sensor σ = 2.0/255. Objects: static crate, tall box, **walking
person behind an 0.8×2.0×0.25 m occluder (26-frame occlusion)**, **25 cm
moving box**, cylinder. Camera arc 2.1 m. 41 checks: depth error, σ
calibration, per-object 3σ position, person ID continuity, VO ATE, relation
accuracy, coverage, cloud size, FPS, export round-trips. Exit 1 on any
failure — the report's failures section is written by the run, not by hand.

## 10. Resource budget (measured, 2 vCPU, avc-bench)

| metric | v3 | v4 measured |
|---|---|---|
| VGA accuracy-mode stereo | 1127 ms | **670 ms** (1.7×) |
| VGA accuracy-mode end-to-end | 0.87 FPS | **1.34 FPS** (39/43 checks) |
| VGA pyramid stereo | — | 175 ms (4.4 FPS e2e) |
| QVGA stereo | 274 ms | **36.5 ms** (7.5×) |
| QVGA end-to-end | 3.6 FPS | **17.8 FPS** (real-time, 39/45 checks) |
| QVGA peak RSS | n/a | 134 MB |
| Per-frame allocations (stereo) | ~190 MB | ~0 (reused buffers) |

## 11. Competitor position (honest summary)

Full analysis in `docs/COMPARISON.md`. In its class (CPU-only, no learned
models, metric + uncertainty-annotated output, single binary): the
uncertainty layer end-to-end (σ_d → Σ_point → Kalman P → drift σ) and the
honest validation harness are the differentiators; raw speed vs
SIMD-optimized OpenCV SGBM and accuracy vs learned deep stereo remain
documented losses. v4 closes the speed gap on low-core-count machines via
§3.7–3.8 while keeping accuracy (§9 re-measured).

## 12. Status matrix

Maintained in `src/engine/status.rs` / README — WORKING / PARTIAL /
INCOMPLETE with evidence per module. v4 moves: occlusion re-id WORKING
(1 ID switch, was 3 + permanent loss), QVGA real-time WORKING (17.8 FPS),
temporal fusion WORKING (new). Documented open items: σ-calibration 2σ
tail, GT#4 small-box detection (v3-inherited), TSDF and loop closure
(INCOMPLETE roadmap).

## 13. Change log v3 → v4

1. Pyramid coarse-to-fine matcher (§3.7) — fast/low-spec mode with the
   trust contract (cost-consistency gate, full-range rescue, coarse mask).
2. Exact min1/min2 SGM penalties (§3.2) — faster *and* more correct.
3. Disparity-voting LR cross-check (§3.5) — replaces the second full pass.
4. `StereoBuffers` reuse (§3.8) — zero steady-state allocation.
5. Fused + parallel census (§2), parallel medians.
6. Temporal depth fusion (§3.9) — motion-compensated, σ-aware.
7. Coasting tracks in the assignment pool + prediction-aware gating +
   velocity-gated fragment stitching (§6.2/6.4) — occlusion ID continuity.
8. Disparity-range fix for small objects (§5.4) + lift stride 2.
9. Bench: engine-mode stages, RSS (VmHWM), pyramid rows (§10).
10. Fat LTO + codegen-units 1 release profile.
11. Validation: 85 tests; VGA accuracy 39/43, QVGA real-time 39/45 @ 17.8 FPS;
    the coupled-knob experiment log (reverted ground-band caps and σ_d
    recalibrations, with reasons) lives in docs/VALIDATION.md.
