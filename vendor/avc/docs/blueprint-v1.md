# AVC v1 (Python) — Complete Algorithm Blueprint for the Rust Port

Source of truth: `/home/z/my-project/avc-project` (AVC Python v2.0.0 — the codebase
referred to as "AVC v1" in the porting effort; it contains the v1.0/v1.1/v2.0
layered history). Every constant below is verbatim from the code. File:line
citations refer to that tree. This document is the porting contract: an
engineer should be able to re-implement the system in Rust without reading
the Python.

---

## 1. Global conventions (avc/core/types.py:1-11)

- **World frame**: X right, Y up, Z forward (right-handed). Units: meters, seconds.
- **Camera frame (OpenCV)**: X right, Y down, Z forward (optical axis).
- **Camera pose convention**: a `Camera.pose` is the **camera-to-world** transform
  `T_c2w` (4x4 homogeneous): `P_w = R @ P_c + t`. Therefore
  `P_c = R^T (P_w - t)` (types.py:91-98, `world_to_cam`).
- **Projection**: `uv = K[:2,:2]^T @ (p_cam[:2]/z) + K[:2,2]`, valid iff `z > 1e-6`
  (types.py:100-109). Distortion coefficients stored but only applied for sparse
  point projection via OpenCV elsewhere.
- **Unproject** (types.py:111-119): `x=(u-cx)/fx, y=(v-cy)/fy`,
  `P_cam = (x·Z, y·Z, Z)`, `P_w = P_cam @ R^T + t`.
- Confidence floats in [0,1]. Every metric value carries a 1-sigma uncertainty
  `sigma` (meters / radians / m/s). Errors reported as ±1σ.
- **Intrinsics**: `fx, fy, cx, cy` + 5 OpenCV distortion coeffs (zeros default) +
  per-param 1-sigma calibration uncertainties `sigma_fx..sigma_cy`. `fov_deg =
  2·atan(width/(2 fx))`, `2·atan(height/(2 fy))` (types.py:73-75).
- StereoRig: `baseline = ||T||` m, `sigma_baseline`, rectification R, quality [0,1].

### Core data structures
- `Frame{camera_id, camera_name, timestamp, image(uint8 BGR), sequence, metadata}`
- `FrameGroup{timestamp, frames[], max_skew}` — frames captured within tolerance.
- `Detection2D{label, box=(x,y,w,h) px top-left, confidence[0,1], mask(HxW uint8)?, camera_name, source}`
- `ObjectState{id, label, position(3,), sigma_position(3,), dimensions(3,) w/h/d,
  sigma_dimensions(3,), velocity(3,), rotation(3x3), confidence,
  confidence_breakdown{stage->conf}, visibility, appearance(HSV hist)?, color,
  first_seen, last_seen, n_observations, is_dynamic, history[(t, pos)],
  source, is_person, pose_skeleton?}`
- `MeasurementResult{quantity, value, unit, sigma, confidence, source, method}`;
  `ci95 = 1.96·sigma`; print precision derived from sigma (types.py:384-388).
- `Confidence.combine(*items, mode)`: geometric mean
  `exp(mean(log(clip(v,1e-6,1))))` (strict) or noisy-OR `1-Π(1-v)` (lenient)
  (types.py:342-354).
- 17-joint COCO skeleton (types.py:271-285):
  `JOINT_NAMES = [nose, left_eye, right_eye, left_ear, right_ear, left_shoulder,
  right_shoulder, left_elbow, right_elbow, left_wrist, right_wrist, left_hip,
  right_hip, left_knee, right_knee, left_ankle, right_ankle]`
  `LIMBS = [(5,7),(7,9),(6,8),(8,10),(5,6),(5,11),(6,12),(11,12),(11,13),(13,15),
  (12,14),(14,16),(0,5),(0,6)]`
- `Skeleton3D{joints(17,3) world, sigma(17,3), confidence(17,), height,
  sigma_height, shoulder_width, sigma_shoulder_width, source,
  ankles_measured}`; body parts: head=j[0..4].mean, neck=(j5+j6)/2,
  torso=(j5+j6+j11+j12)/4, arms mean of triples, hands=j9/j10, legs mean (types.py:302-319).
- Lifecycle enum: `VISIBLE / OCCLUDED / PREDICTED / LOST` (types.py:27-33).

---

## 2. core/geometry.py — SE(3), estimation, triangulation

### so3_exp (geometry.py:18-25)
```
theta = ||w||; if theta < 1e-12 -> I
k = w/theta; K = skew(k)
R = I + sin(theta)·K + (1-cos(theta))·K²
```

### so3_log (geometry.py:28-45)
```
cos = clip((tr(R)-1)/2, -1, 1); theta = acos(cos)
if theta < 1e-10 -> 0
if |pi - theta| < 1e-6: A=(R+I)/2; axis = sqrt(clip(diag(A),0)); k=argmax;
   axis = A[:,k]/A[k,k]; normalize; return axis·theta
else: vee = [R21-R12, R02-R20, R10-R01]; return vee · theta/(2 sin theta)
```

### se3_exp (geometry.py:48-67) — twist [w; v], Barfoot convention
```
R = so3_exp(w); theta = ||w||
if theta < 1e-12: t = v
else: k=w/theta; K=skew(k)
  V = I + ((1-cos θ)/θ²)·K + ((θ - sin θ)/θ³)·K²
  t = V @ v
T = [R t; 0 1]
```

### se3_log (geometry.py:70-91)
```
w = so3_log(R); θ = ||w||; t = T[:3,3]
if θ < 1e-12: v = t
else: V as above; Vinv = inv(V)  (fallback: I - (θ/2)·K on LinAlgError)
v = Vinv @ t;  return [w; v]
```
(Note: the doc comment mentions the closed form
`V^{-1} = I - (θ/2)K + (1-(θ/2)cot(θ/2))/θ² K²` but the implementation
inverts V numerically.)

### rigid_inverse (geometry.py:94-100)
`inv = [R^T, -R^T t; 0 1]`.

### look_at (geometry.py:110-121) — OpenCV camera pose (c2w)
```
forward = normalize(target-eye)
right = normalize(cross(forward, up))     (up default [0,1,0];
                                           fallback up=[0,0,1] if degenerate)
down = cross(forward, right)
R = [right down forward] as COLUMNS  (= camera axes expressed in world)
```
Regression test (tests/test_geometry.py:25-33): camera at origin looking +Z with
world Y up must give cam-x = -worldX, cam-y = -worldY, cam-z = +worldZ.

### umeyama / Kabsch (geometry.py:129-148) — least squares `dst = R·src + t`
```
mu_s, mu_d = means; sc,dc = centered
cov = dc^T @ sc / n
U,D,Vt = svd(cov); S = I; if det(U)·det(Vt) < 0: S[2,2] = -1   # reflection fix
R = U @ S @ Vt
scale (optional) = tr(diag(D)@S)/var_s
t = mu_d - scale·(R @ mu_s)
```

### ransac_rigid (geometry.py:151-179)
`threshold=0.05 m, iters=200, min_inliers=6`, 3-point minimal sets, final
re-fit on inliers. Returns (R, t, inlier_mask, ratio).

### triangulate_dlt (geometry.py:187-216)
Per point builds A (4x4) rows `u·P[2]-P[0], v·P[2]-P[1]` for both views, SVD,
X = Vt[-1] dehomogenized (guard |X[3]|>1e-12). Reprojection RMS per point =
sqrt((e1²+e2²)/2).

### triangulate_midpoint (geometry.py:219-235)
Closest point between two rays: solve 2x2 normal equations for ray params s,t
(clip [0,100]), midpoint of the two closest points.

### fit_plane (geometry.py:248-280) — RANSAC plane `n·x + d = 0`
`iterations=200, threshold=0.03` default; 3 random points per hypothesis;
refinement: centroid + covariance eigen, normal = smallest-eigenvector,
**oriented so n[1] > 0** (world Y up); final inlier mask at 2·threshold.

### rotation_matrix_to_euler_zyx (geometry.py:287-298)
Intrinsic Z-Y-X: pitch=asin(-R[2,0]); yaw=atan2(R[1,0],R[0,0]);
roll=atan2(R[2,1],R[2,2]); gimbal-lock branch yaw=atan2(-R[0,1], R[0,1]),
roll=0.

---

## 3. core/uncertainty.py — the error model

### DepthErrorModel (uncertainty.py:24-58)
```
Z = f·B/d
σ_Z_disp = Z²·σ_d/(f·B)                (random; ÷√N for N independent samples)
σ_Z_f    = Z/f · σ_f                   (systematic)
σ_Z_B    = Z/B · σ_B                   (systematic)
σ_Z = sqrt( (σ_Z_disp/√N)² + σ_Z_f² + σ_Z_B² )
```
Defaults: `sigma_disparity = 0.15 px` (subpixel matchers), `sigma_fx = 0`,
`sigma_baseline = 0`. `sigma_z_from_disparity(d, σ_d)` computes Z then σ_Z.

### propagate_a_b (uncertainty.py:60-70)
σ of `||a-b||` with independent per-axis sigmas:
`grad = (b-a)/||b-a||; σ = sqrt(Σ grad_i² (σa_i² + σb_i²))`; degenerate →
`sqrt(Σ(σa²+σb²))`.

### propagate_extent (uncertainty.py:73-83) — sigma of (max-min)
```
N<2: 2σ.  else: ev = 2·sqrt(ln N);
σ_extent = min(2·ev, sqrt(N)) · σ · 2 / sqrt(max(N//4, 1))
```
(extreme-value upper bound, honest).

### combine_sources (uncertainty.py:86-90)
quadrature (independent) or linear sum (correlated worst case).

### sigma_to_confidence (uncertainty.py:93-103)
```
rel = σ/|value|  (if value≈0: 1 if σ>0 else 0)
conf = floor / (1 + (rel/k)²)     with floor=0.95, k=0.10; clip [0,1]
```
Monotone, deliberately conservative (tests/test_geometry.py:133-137).

---

## 4. calibration/calibrator.py

### Corner detection (calibrator.py:74-100)
- `cv2.findChessboardCorners(gray, (9,6), ADAPTIVE_THRESH|NORMALIZE_IMAGE)`.
- **v2 subpixel refinement**: upsample the gray image 4x (INTER_CUBIC), scale
  corner coords ×4, `cornerSubPix` on the 4x image with window (7·4, 7·4),
  criteria (EPS+MAX_ITER, 60, 1e-4), divide results by 4. `supersample=1`
  recovers v1 behavior (40 iters, window 7x7).
- Board default: pattern (9,6) inner corners, square 0.08 m.

### calibrate_intrinsics (calibrator.py:120-183)
- `cv2.calibrateCamera` (flags=0) on planar objp grid `objp[x,y,0] = (i·s, j·s)`.
- Per-view reprojection RMS; mean RMS.
- **Statistical sigma**: `px = sqrt(2/n_obs)·max(mean_rms, 0.05)`;
  `sigma_fx = sigma_fy = px·√2`, `sigma_cx = sigma_cy = px`.
- **Systematic floors (v2, calibrator.py:151-163)**:
  `sigma_fx,fy ≥ 1.2 px`, `sigma_cx,cy ≥ 2.0 px` (planar-calibration
  reproducibility; v1 reporting 0.01 px on cx was indefensible).
- Coverage: fraction of frames having corners in the outer 20% image band
  (x<0.2w or x>0.8w or y<0.2h or y>0.8h).
- Quality (calibrator.py:186-191):
  `q = 0.5·q_rms + 0.25·q_n + 0.25·q_cov` where
  `q_rms = 1/(1+(rms/0.35)²)` (0.35px → 0.89), `q_n = 1 - exp(-n/8)`,
  `q_cov = 0.6 + 0.4·coverage`; clip [0,1].

### sampson_error (calibrator.py:199-207)
`|x2ᵀ F x1| / sqrt((Fx1)₁²+(Fx1)₂²+(Fx2)₁²+(Fx2)₂²)` per match.

### calibrate_stereo (calibrator.py:210-258)
- `cv2.stereoCalibrate` with `CALIB_FIX_INTRINSIC`, criteria (MAX_ITER+EPS,
  100, 1e-6); inputs are the already-calibrated K and 5-coeff distortion.
- Sampson RMS over all concatenated corners; baseline = ||T||.
- Rectified row alignment: `stereoRectify(alpha=0)`, `initUndistortRectifyMap`,
  `undistortPoints(R=R1,P=P1)` both sides, mean |Δy| of matched corners.
- Quality (calibrator.py:261-266):
  `0.35·q_rms + 0.30·q_row + 0.20·q_n + 0.15·q_b`,
  `q_rms = 1/(1+(rms/0.4)²)`, `q_row = 1/(1+(row/0.5)²)`,
  `q_n = 1-exp(-n/8)`, `q_b = 1/(1+sqrt(0.02/baseline))`.

### Rectification (calibrator.py:274-321)
`cv2.stereoRectify(alpha=0)` -> R1,R2,P1,P2,Q; `initUndistortRectifyMap`
CV_32FC1 maps; `rectify()` remaps INTER_LINEAR. Derived:
`baseline = |P2[0,3]/P2[0,0]|`, `fx_rect = P1[0,0]`.

### Calibration JSON (calibrator.py:329-360)
```
{"intrinsics": {name: {width,height,fx,fy,cx,cy,distortion[5],
                        sigma_fx,sigma_fy,sigma_cx,sigma_cy}},
 "stereo": {pair: {R:3x3, T:3, baseline, quality}},
 "rig_quality": float}
```

---

## 5. Stereo matching

### 5.1 avc/stereo/matcher.py — NCC block matcher (v1)

Class `NCCStereoMatcher(block=9, num_disparities=64, min_disparity=0,
texture_thresh=5.0, ratio_thresh=0.6, lr_thresh=1.0)`.

**Shift semantics (matcher.py:42-65)** — L→R matching compares, at left pixel x,
the right pixel x−d (`_shift_right`: `out[y,x] = img[y,x-d]`, edge-padded left).
R→L matching (`_right_best`, matcher.py:188-208) compares at right pixel x the
left pixel x+d (`_shift_left`). **Disparity sign: d ≥ 0, right camera is
offset to the left; left pixel x ↔ right pixel x−d.**

**NCC cost (matcher.py:92-98)**, window statistics via `cv2.boxFilter` with
`BORDER_REFLECT`, block 9x9 (r = 4):
```
mean_b = box(b_shifted); cross = box(a·b) - mean_a·mean_b
var_b  = box(b²) - mean_b²
cost   = clip(1 - cross / sqrt(max(var_a,1e-6)·max(var_b,1e-6)), 0, 2)
```
Streaming WTA over d in [min_d, min_d+64): border validity
`y∈[r,H-r), x∈[r+d, W-r)` else INF_COST=2.5. `second` tracks 2nd best,
`c_lo/c_hi` cache costs at best_d∓1 for the parabola.

**Subpixel parabola (matcher.py:141-148)**:
```
denom = c_lo - 2·c0 + c_hi
delta = clip((c_lo - c_hi)/(2·denom), -0.75, +0.75)   (needs |denom|>1e-9)
disp  = d0 + delta
```
**sigma_d from curvature (matcher.py:149-153)**:
```
curvature = |denom|
sigma_d = clip(0.25/sqrt(curvature), 0.05, 0.6)   if curvature > 4e-3, else 0.5
```

**Confidence (matcher.py:155-161)**:
```
ratio = (second - best)/max(second,1e-6)   (second==INF -> ratio 0)
conf_ratio = clip(ratio/0.25, 0, 1)
conf_tex   = clip((texture - 5.0)/12.0, 0, 1)
conf = clip(0.65·conf_ratio + 0.35·conf_tex, 0, 1)
```
texture = sqrt(var_l) (local std in 9x9 window).

**Validity (matcher.py:163)**: `d0 > 0 & best < 1.9 & texture ≥ 2.5`
(= texture_thresh·0.5).

**LR check (matcher.py:166-173)**: integer-d right-image WTA map `d_r`;
for left pixel (x,y) look up `d_r[y, clip(x - round(disp))]`; consistent iff
`|disp - d_r_at| ≤ 1.0`; inconsistent → `occluded` and invalid.

### 5.2 avc/stereo/matcher2.py — SGM2 (v2, the accuracy backend)

Class `SGM2StereoMatcher(block=13, num_disparities=64, min_disparity=0,
texture_thresh=5.0, census_weight=0.3, ncc_weight=0.7, p1=0.06, p2=0.9,
p2_grad_cap=6.0, uniqueness=1.4, lr_thresh=1.0, speckle_area=12,
interp_conf=0.35, sigma_floor=0.12, prefilter_median=True, post_median=True,
delta_clip=0.75, half_grid=True, hard_uniqueness=True)`.

Pipeline: median prefilter (3x3) → two-scale census + windowed NCC cost
volume → 4-path SGM aggregation → WTA + uniqueness → subpixel parabola on the
RAW NCC cost → LR cross-check → speckle filter → row-gap interpolation →
confidence/sigma.

**Census transform (matcher2.py:43-69)**: uint32 code, offsets =
- fine: 5x5 minus center (24 bits) — sharp edges;
- coarse ring: (±4,0),(0,±4),(±4,±4) (8 bits) — survives coarse textures.
32 bits total. bit_k = (padded[dy,dx] > img).
**Hamming (matcher2.py:72-77)**: popcount(a^b)/n_bits as float [0,1]
(n_bits = 32 here).

**Cost volume (matcher2.py:132-185)**: with `half_grid` the disparity axis is
2·D slices, index k ↔ disparity `min_d + 0.5·k` (k even = integer d). Fused
cost `cost = 0.3·hamming + 0.7·ncc` where
`ncc = clip(0.5·(1 - NCC), 0, 1)` (matcher2.py:129) — note the 0.5 rescale
into [0,1]. Half-grid costs: NCC from the 0.5-lerp of the right image shifts
`0.5·(shift(d) + shift(d+1))`; census from the average of the two neighbouring
Hamming costs. Border cost = 1.2. A separate `ncc_vol` keeps the raw NCC cost
for subpixel/σ (also 1.0 at borders).

**SGM aggregation (matcher2.py:188-258)**: 4 scanline paths: L→R, R→L
(state (K,H) per column x), T→B, B→T (state (K,W) per row y). Recurrence per
pixel (standard Hirschmüller):
```
p2_eff = p2 / (1 + |I(x) - I(xn)|)          # gradient-adaptive P2
Lnew(k) = C(x,k) + min( Lprev(k),
                        Lprev(k±1) + p1,     # ±1 grid step in disparity
                        min_k' Lprev(k') + p2_eff ) - min_k' Lprev(k')
```
`p1` is scaled by `p1_scale = 0.5` when half_grid (grid step 0.5 px).
Accumulator = mean of the 4 paths (`acc·0.25`). P1 = 0.06, P2 = 0.9 absolute
on the [0,1.2] cost scale.

**WTA + uniqueness (matcher2.py:261-304)**: best & k0 over all K slices;
second-best excludes grid neighbours `|k-k0| ≥ excl` with
`excl = max(1, ceil(1/step))` (=2 on the half grid → ≥1 px disparity away,
SGBM uniqueness semantics). `p0 = min_d + step·k0`.
Right-image WTA (`_right_wta`): integer slices only (stride 2); right pixel
x_r ↔ left pixel x_r + d, cost slice `agg[ke, :, d:]`, updates best/d_r over
right pixels [0, W-d).

**Subpixel (matcher2.py:380-403)** — fitted on the RAW NCC cost curve at grid
neighbours (NOT the SGM-aggregated curve; aggregated-parabola inflated
bad-1px ~4x):
```
c_mid = ncc_vol[k0]; c_lo = ncc_vol[k0-1]; c_hi = ncc_vol[k0+1]
denom = c_lo - 2·c_mid + c_hi
delta = clip(step·(c_lo - c_hi)/(2·denom), ±0.75)
disp = p0 + delta;   disp[k0==0] = 0
```
**σ_d (matcher2.py:413-417)**: `σ_d = clip(0.25/sqrt(curvature), sigma_floor
0.12, 0.6)` when `curvature > 1e-3`, else 0.35.

**Uniqueness gating (matcher2.py:419-436)**:
```
ratio = best/second (fallback 1.0);  thresh_u = 1/max(uniqueness, 1.01) ≈ 0.714
conf_uniq = clip((thresh_u - ratio)/thresh_u, 0, 1)
valid = (p0 ≥ 1.0) & border(r=block//2 margin) & (best < 1.1) & finite
if hard_uniqueness: valid &= ratio < thresh_u      # ambiguous -> hole
```

**Confidence (matcher2.py:425-459)**:
```
conf_curv = clip(curvature/0.08, 0, 1)
conf_tex  = clip((texture - 2.5)/12.0, 0, 1)     # texture_thresh·0.5
conf = 0.45·conf_uniq + 0.25·conf_curv + 0.30·conf_tex     (valid only)
```

**Speckle filter (matcher2.py:307-317)**: `cv2.connectedComponentsWithStats`
(8-connectivity) on the valid mask; drop components with area < 12.

**Row-gap interpolation (matcher2.py:320-350)**: forward pass fills invalid
pixels with the nearest valid disparity to the left; backward pass fills with
nearest valid to the right **only if smaller** (farther background).
Interpolated mask = filled & ~valid, restricted to the interior border.
Interpolated pixels: `conf = min(conf·0.6, 0.35)` and `σ ×2` — flagged
estimates, never measurements (`DisparityResult.interpolated`).

**Discontinuity σ inflation + post-median (matcher2.py:466-476)**:
`med = medianBlur(disp,3)`; if `|disp - med| > 2.0` → `σ = max(σ·1.5, 0.25)`;
post_median: where `|disp - med| ≤ 1.0` replace disp by med (conservative).

### 5.3 SGBM wrapper (matcher.py:211-243)
```
minDisparity=0, numDisparities=64, blockSize=7,
P1 = 8·3·block²,  P2 = 32·3·block²,
disp12MaxDiff = 1 (lr_check) else -1, uniquenessRatio=8,
speckleWindowSize=80, speckleRange=2, preFilterCap=31,
mode = STEREO_SGBM_MODE_SGBM_3WAY
```
Disparity = output/16. `σ_d = 0.3 px` constant, `conf = 0.7` (documented
conservative limitation). Backend string "sgbm".

### 5.4 avc/stereo/pipeline.py — StereoEngine

`StereoEngine(left, right, R, T, backend∈{ncc,sgbm,sgm2/avc2},
num_disparities=64, block=9, sigma_baseline=0.001, lr_check=True)`.
sgm2 uses `block = max(7, block)`. Error model =
`DepthErrorModel(fx=fx_rect, baseline=rect.baseline, sigma_fx=left.σ_fx,
sigma_baseline=0.001)`.

`process(img_l, img_r, t, left_world_pose?)`:
1. rectify both via remap maps;
2. grayscale; matcher.compute(gl, gr, lr_check);
3. **border mask**: `(gl>1)&(gr>1)` eroded with kernel k = max(3, block//2);
   applied to disp.valid (kills remap black borders) (pipeline.py:81-87);
4. `_disparity_to_depth` (pipeline.py:96-142):
```
f = P1[0,0]; cx,cy = P1[0,2],P1[1,2]; B = |P2[0,3]/P2[0,0]|
valid &= d > 0.5
Z = f·B/max(d, 0.5);  X = (x-cx)/f·Z;  Y = (y-cy)/f·Z     (rectified frame)
p_cam  = R1^T @ p_rect          (implemented as p_rect @ R1)
p_world = R_w @ p_cam + t_w     (left camera c2w pose, possibly VO-updated)
σ_z  = error_model.sigma_z(Z, disp.σ, n_samples=1)
σ_xy = σ_z · 1.2 / f
pose_rect_c2w = pose_left_c2w @ make_pose(R1^T, 0)
```
The `DepthMap` therefore lives in the WORLD frame; `_cx,_cy,_fy` stored from P1.

`triangulate_points` (pipeline.py:145-163): P1 = K_l @ w2l[:3,:], P2 =
K_r @ w2r[:3,:] (world→pix via inverse poses), DLT per point.

### 5.5 avc/depth/depth.py — DepthMap
Fields: `points(H,W,3) world (NaN invalid), σ_z, σ_xy, valid, confidence,
disparity, fx, baseline, pose_left_c2w, pose_rect_c2w`.
`world_to_rect(pts) = (pts - t) @ R` (using pose_rect_c2w).
`sample_uv(uv, radius=1)` (depth.py:61-87): per query pixel a (2r+1)² window;
robust center = per-axis median; σ = max(median σ_z, std/√n); conf = mean.
`colored()`: TURBO colormap of z/6.0, invalid black.

---

## 6. avc/retina/retina.py — low-level preprocessing

`Retina(pyramid_levels=4, corner_max=600, corner_quality=0.08,
temporal_alpha=0.6)`.
- Pyramid: `pyrDown` chain, level 0 full res.
- brightness/contrast: 9x9 boxFilter mean & std (BORDER_REFLECT).
- gradients: Sobel 3x3 (CV_32F) x/y; magnitude hypot.
- edges: Canny with `lo = 0.33·median(gray)+20`, `hi = 0.66·median(gray)+60`.
- corners: `cornerMinEigenVal(gray, 5)` normalized by max (response);
  `goodFeaturesToTrack(maxCorners=600, qualityLevel=0.08, minDistance=8)`.
- temporal: `absdiff` vs previous gray; motion energy = GaussianBlur(7x7) of
  it, EMA `m = 0.6·new + 0.4·prev_ema`.
- feature_confidence = `clip(0.6·clip(contrast/12,0,1) + 0.4·clip(grad/60,0,1), 0, 1)`.
- `debug_panel`: 8 panels (gray, brightness, contrast, gradient, edges,
  conf, temporal, blank).

---

## 7. avc/features/features.py + avc/motion/motion.py

**FeatureExtractor** `n_features=1200` (VO uses 1400), ORB
(scaleFactor=1.2, nlevels=8), optional SIFT. **FeatureMatcher**: BFMatcher
NORM_HAMMING, knn k=2, Lowe ratio 0.75, `findFundamentalMat(FM_RANSAC,
2.0 px, 0.999, 10000)`. **track_points** (LK): `calcOpticalFlowPyrLK`
winSize (21,21), maxLevel 3, criteria (EPS|COUNT, 30, 0.01).

**MotionAnalyzer** (motion.py:30-51): Farneback flow
`pyr_scale=0.5, levels=3, winsize=15, iterations=3, poly_n=5, poly_sigma=1.2`;
moving mask = magnitude > 1.2 px, 3x3 open. Flow viz: HSV hue=angle, V=mag·8.

---

## 8. avc/detection — detectors

### 8.1 ClassicalDetector (detector.py:40-102)
HOG (`getDefaultPeopleDetector`) `detectMultiScale(winStride=(8,8),
padding=(16,16), scale=1.05)`; person conf =
`clip((weight-0.3)/1.4, 0.05, 0.9)`. Motion blobs: MOG2
(history=80, varThreshold=30, no shadows), learningRate 0.05 for the first 40
frames then 0; open 3x3, close 9x9; connected components; `min_area=900`;
label "unknown_object" conf 0.35; suppressed when IoU > 0.4 with a HOG person.

### 8.2 YOLOv8ONNX (yolo_onnx.py)
Input 1x3x640x640 RGB float /255 (letterbox: scale = min(640/w, 640/h),
canvas 114, centered). Output (1,84,8400) = [cx,cy,w,h, 80 class scores]
(NMS-free head). conf_thresh 0.35, iou_thresh 0.45; NMS per class (IoU
greedy). Box un-letterbox: `x=(cx-bw/2-ox)/scale` etc. COCO class list
(80 names) inline (yolo_onnx.py:33-46).

### 8.3 YOLOv8SegONNX (yolo_seg_onnx.py)
Outputs: `output0 (1,116,8400)` = [box(4), 80 scores, 32 mask coeffs];
`output1 (1,32,160,160)` prototypes. Mask decode (yolo_seg_onnx.py:91-127):
```
m = sigmoid(coeffs @ protos.reshape(32,-1)).reshape(160,160)
box (letterbox px) -> region on proto grid via s = 160/640
patch = m[my1:my2, mx1:mx2] resized to box size (INTER_LINEAR)
mask_lb[box] = (patch > 0.5)·255
undo letterbox: crop [oy:oy+nh, ox:ox+nw], resize INTER_NEAREST to (W,H)
```
Validated vs ultralytics: 6/6 same detections, mask IoU 0.90-0.96.

### 8.4 Validation-only GT detectors (detector.py:110-251)
- `GroundTruthDetector`: projects 8 box corners, 2D bbox + N(0,2px) noise on
  each edge, conf ~ U(0.85,0.99); miss_prob; rng seed 11.
- `OcclusionAwareGTDetector`: uses renderer obj-id buffer remapped
  (INTER_NEAREST) into rectified-left coordinates via the rect maps;
  visible mask = isin(id_rect, item ids); min_pixels 60; box from mask
  extents + N(0,2px) noise; mask·255 attached; source "groundtruth".
- Selection chain `best_available_detector`: yolov8n-seg.onnx > yolov8n.onnx
  > classical (explicit, never silent).

---

## 9. avc/segmentation

- `Segmenter` (segmentation.py:19-48): MOG2 history=60, varThreshold=24,
  learningRate 0.05 (30 frames) then 0; open 3x3, close 7x7; min_area 180.
- `color_histogram` (segmentation.py:51-57): HSV, `calcHist([0,1], mask,
  bins [16,8], ranges [0,180,0,256])`, L1-normalized, raveled — the reid
  appearance descriptor.
- `histogram_similarity`: Bhattacharyya.
- `BoxMaskRefiner` (refine.py:23-106) — GrabCut box→mask fallback:
  `min_box_px=1200 (and w,h ≥ 8), iterations=3, margin=0.06,
  fg_range=(0.05, 0.95)`. Crop with 6% margin, `cv2.grabCut(GC_INIT_WITH_RECT)`
  3 iterations; fg = (GC_FGD|GC_PR_FGD) AND box ROI; degenerate fraction
  (<5% or >95%) → fallback eroded box (6% inner erosion); else largest
  connected component. Counts n_refined / n_fallback (honesty stat).

---

## 10. avc/detection/lifting.py — 2D→3D lifting (v1.1 rewritten)

`ObjectLifter(inner_margin=0.08, min_points=24, outlier_sigmas=2.5,
cluster_gap=0.30 (kept for API compat, unused), fill_holes=True,
max_fill_frac=0.85)`. The demo pipeline constructs it with
`min_points=150` (EngineConfig.lifting_min_points, engine/pipeline.py:57).

**Gather (lifting.py:74-112)**: if detection has a full-size mask → erode 3x3
(fallback to raw mask if < min_points/2 pixels survive); else inner-margin box
(8% inset each side; falls back to full box if <3px). Requires ≥ max(4,
min_points//4) valid depth pixels.

**Sliver rejection (lifting.py:213-214)**: `keep.sum() < max(8, min_points)`
→ return None (occlusion-boundary fragments corrupt tracking).

**1. Object-aware Z-envelope (lifting.py:221-240)** — replaces the old hard
Z-cluster split (which amputated a walking person's forward shin ~0.35 m
ahead of the torso and the table's far edge):
```
z_ref = median(Z of kept points)
s_min = max(min(box_w, box_h)·z_ref/fx, 0.05)      # smallest box axis, metres
env   = max(0.40, 0.60·s_min)
keep points with |Z - z_ref| < env   (if ≥ max(8, min_points//2) survive)
```
**2. MAD outlier window, floored at the envelope (lifting.py:243-253)**:
```
med_z = median Z; spread = max(1.4826·MAD(Z), 0.05)
window = max(2.5·spread, env)          # never cuts deeper than the envelope
keep |Z - med_z| < window
```
**3. Mask-hole fill (lifting.py:163-202, 255-271)**: unproject mask pixels
lacking stereo depth using **row-band medians computed in the RECTIFIED
camera frame** (world-Z vs rect-Z conflation was a measured bug on tall
panels). Row medians need ≥3 points/row; NaN rows borrow the nearest row
within 12 rows. Fill σ = `s_ref·2 + 0.02` with `s_ref = max(0.04,
0.5·std(z_ok))`; fill conf 0.35. Accepted only if `n_filled ≤ 0.85·n_measured`.
`n_eff = max((n_pts − 0.75·n_filled)//4, 1)`.

**Statistics (lifting.py:273-296)**:
```
position = per-axis median of points
spread_xyz = max(1.4826·MAD(p), 1e-3) per axis
syst_xy = max(z_median/480, 0.012)          # ~1 px calibration error
syst_z  = max(median(σ_d)·0.15, 0.02)
σ_pos = sqrt( (spread/√n_eff)² + [syst_xy², syst_xy², syst_z²]
              + [median(σ_xy)², median(σ_xy)², median(σ_z)²] )
σ_pos = max(σ_pos, [0.010, 0.010, 0.025])   # hard floors
dims = P99.5(p) − P0.5(p) per axis
σ_dims = propagate_extent(p, σ_pos·1.5+0.01, axis) per axis, clip [0.03, 0.5]
```
**Plausibility (lifting.py:298-308)**: backproject the 2D box at the median
range: `exp_w = max(box_w·dist/fx, 0.05)`, `exp_h` likewise; plausible iff
`dims[0] < 3·exp_w + 0.15 and dims[1] < 3·exp_h + 0.15 and max(dims) < 3.5`.
**Confidence**: `clip(det_conf·0.6 + mean_depth_conf·0.4)`; ×0.3 if not
plausible. Appearance: HSV color_histogram over the mask/box region of the
rectified image. Subsampled points cap 3000 (rng seed 7).

**merge_lifted_views (lifting.py:350-456)** — multi-rig fusion:
- Group by (label, proximity): same object iff
  `dist < gate_m + 0.5·max(max_dim_a, max_dim_b)` (gate_m=1.0).
- `require_primary` (pipeline sets True with primary_view = first pair):
  drop groups with no primary-view observation.
- Label conflicts across groups: prefer the group containing the primary view,
  else the most points.
- Per group: drop implausible views; reference = primary-view object, else
  max points. Position-agreement filter: keep views within 0.6 m of ref.
  Dimension-agreement: keep views with `|dims−ref.dims|/max(ref.dims,0.05) <
  0.6` per axis.
- **PRIMARY-AUTHORITY POSITION POLICY (lifting.py:434-442)**: merged
  `position = ref.position` and `σ_pos = ref.σ_pos` exactly; secondary rigs
  contribute their points for DIMENSION refinement only
  (`σ_pos = max(σ_pos, spread(all points)/√n_eff · 0.5)`); dims from the
  union P0.5..P99.5; `σ_dims = mean(view σ_dims)·0.8`; conf = mean+0.05;
  points cap 6000. (A secondary rig lost the person's legs to thin-structure
  stereo and dragged the merged centroid +0.12 m in Y before this policy.)

`lifted_to_state` converts to `ObjectState` (source "stereo+detection").

---

## 11. avc/tracking/tracker.py — 3D Kalman + Hungarian

### Kalman3D (tracker.py:78-125)
State `x = [px,py,pz, vx,vy,vz]` (6). `sigma_acc = 2.0` (m/s²), sigma_meas0 0.05.
```
F = I6; F[0,3]=F[1,4]=F[2,5]=dt
Q (piecewise white accel): per axis i:
  Q[i,i]=q·dt⁴/4; Q[i,i+3]=Q[i+3,i]=q·dt³/2; Q[i+3,i+3]=q·dt²   (q=σ_acc²)
H = [I3 0] (3x6)
R = diag(max(σ_z, [0.030, 0.030, 0.060])²)      # honest floors: ~3 cm lateral,
                                                 # 6 cm depth scatter at 3-5 m
Update: K = P Hᵀ S⁻¹; Joseph form P = (I-KH)P(I-KH)ᵀ + KRKᵀ; symmetrize.
```

### MultiObjectTracker (tracker.py:128-332)
`confirm_hits=3, max_misses=14 (EngineConfig demo: 40), gate_m=3.2,
appearance_weight=0.35, size_weight=0.3, dims_alpha=0.25 (legacy),
keep_history=600`.

**Anchored-state coasting (tracker.py:188-202, 313-332)**: `tr.x/tr.P` are the
last UPDATED state at `tr.t_last`. Per-frame predictions
`preds[id] = kf.predict(tr.x, tr.P, dt)` are derived and NEVER written back —
the old cumulative predict-to-t made coasted positions diverge quadratically.

**Cost matrix (tracker.py:152-185)** — gates first, then cost:
```
gates (all must hold): det.label == tr.label AND Mahalanobis m² ≤ 25.0
                       (χ², 3 dof ≈ 3σ... 5σ) AND ||Δpos|| ≤ 2.5 m
m² = Δxᵀ S⁻¹ Δx with S = H P_pred Hᵀ
app = 1 − Bhattacharyya(hist_tr, hist_det)   (0.5 if either missing)
size_d = ||(det.dims − tr.dims)/(tr.σ_dims + det.σ_dims + 0.05)||
rs = reid_score(...)                         (see §12)
cost = 0.45·m²/25 + 0.25·(1−app) + 0.15·min(size_d,3)/3 + 0.15·(1−rs)
```
Assignment: `scipy linear_sum_assignment` on `where(valid, cost, 1e6+1)`;
pairs with `cost > 5.0` are discarded.

**Update on match (tracker.py:211-263)**:
- Partial-occlusion model: `ratio = det.dims/max(tr.dims,1e-6)`; partial iff
  `min(ratio) < 0.75`; then `missing = max(tr.dims − det.dims, 0)` (max over
  axes), `bias = 0.5·missing`, `σ_pos = sqrt(det.σ² + bias²)` (occluded
  centroids are biased toward the visible part by up to half the missing extent).
- KF update with (possibly inflated) σ_pos.
- **Velocity clamp**: if `||v|| > 4.0 m/s` scale v to 4.0 (physical guard;
  clamping, not rejection — rejection would fragment assignment).
- dims_history (last 64): `tr.dims = P85(dims_history)` per axis (extents are
  LOWER bounds under occlusion; the 85th percentile estimates true extent);
  `tr.σ_dims = max(det.σ_dims, min(0.5·MAD(hist), 0.25))`.
- Appearance EMA: `0.7·old + 0.3·new`. history (t, position) last 600.
- `track_conf = 0.5·last_conf + 0.3·min(hits/8,1) + 0.2·coast_c`, coast_c = 1
  if VISIBLE else `max(0, 1 − misses/max_misses)`.

**Lifecycle**: unmatched → misses+1; if hits ≥ confirm_hits → OCCLUDED, and
PREDICTED once misses > 2; else (unconfirmed) → LOST; misses > max_misses →
LOST. Unconfirmed tracks with misses > 2 are dropped from the output.
**New tracks**: x=[pos,0], P = diag([0.01·I3, 0.5·I3]); id `{label}_{n:03d}`.
`tracks_as_states(t)` re-derives states at t via anchored predict; σ_position
= sqrt(diag(P)[:3]).

---

## 12. avc/prediction/predict.py — reid & coasting

**reid_score (predict.py:21-34)**:
```
z = ||(obs − pred)/σ||                (σ per axis)
pos_score  = exp(−0.5·z²)
app_score  = clip(appearance_sim, 0, 1)
size_score = exp(−0.5·min(size_mismatch, 6)²)
reid = 0.45·pos + 0.35·app + 0.20·size
```
**OcclusionPredictor**: `reid_accept = 0.55`. `predict_position`:
`p = pos + v·dt; σ = σ_pos + 0.35·dt` (0.35 m/s diffusion). `search_gate`:
center + `k·σ` (k=3). `try_reidentify`: label match, reid ≥ accept.
**predict_trajectory**: horizon 2.0 s, step 0.2 s; σ grows as
`σ = σ_pos + t·(0.35 + ||σ_pos/max(speed,0.5)·0.5||)`.

---

## 13. avc/pose/human.py — 17-joint 3D skeleton

### 2D adapters
- `MediaPipePose2D` (33 landmarks → COCO17 via MAP {coco: mp} =
  {0:0, 2:1, 5:2, 7:3, 8:4, 11:5, 12:6, 13:7, 14:8, 15:9, 16:10, 23:11,
  24:12, 25:13, 26:14, 27:15, 28:16}); conf = landmark.visibility.
- `ProportionalPose2D` — anthropometric prior from the person box
  (human.py:55-64), fractions (height-from-head, x-offset in half-widths):
  nose (0.03,0.00), eyes (0.045,±0.03), ears (0.05,±0.07), shoulders
  (0.13,±0.16), elbows (0.28,±0.21), wrists (0.42,±0.19), hips (0.50,±0.09),
  knees (0.72,±0.08), ankles (0.95,±0.06); conf 0.55 for
  shoulders/hips/ankles else 0.4; source "estimated (anthropometric prior)".
- `GroundTruthPose2D`: project GT joints + N(0,2px) noise, clip to image;
  conf 0.95 where in front; rng seed 5.

### StereoSkeletonLifter (human.py:173-349)
`max_joint_gap=0.45, anchor_tol=0.45`.
**Person anchor (human.py:193-216)**: depth support inside the 2D box;
≥30 valid; `med = median(Z)`; `mad = 1.4826·median|Z−med|`; inliers
`|Z−med| < max(3·mad, 0.15)`; re-median; `σ_ref = max(std/√n, 0.02)`;
`conf_ref = mean depth confidence`.
**Per joint (human.py:243-271)**: 5x5 window around the joint pixel;
local measurement = per-axis median of valid pts, σ from median σ_xy/σ_z.
**Thin-limb decision**: if no local measurement OR `|p_z − z_ref| > 0.45` →
backproject (u,v) at `z_ref` via the rectified camera:
`p_rect = ((u−cx)/fx·Z, (v−cy)/fy·Z, Z)`, `p_w = R·p_rect + t` (pose_rect_c2w);
σ = `[σ_ref·0.8, σ_ref·0.8, σ_ref·1.5]`; conf = `min(pose2d, conf_ref)·0.85`;
flagged anchored (never hidden).
**Torso cross-check (human.py:273-281)**: median Z of shoulders+hips; joints
with `|Z − z_torso| > 0.6` are NaN'd.
**Fill missing (human.py:311-349)**: FILL_ORDER = symmetric pairs first
[(5,6),(11,12),(5,11),(6,12),(0,5),(0,6),(5,7),(7,9),(6,8),(8,10),(11,13),
(13,15),(12,14),(14,16)], 2 passes; filled joint = counterpart position,
σ = 2σ_other+0.05, conf = 0.4·conf_other; last resort body center (σ 0.5,
conf 0.1). NaN-sanitized.
**Derived (human.py:288-301)**: height = |max(Y of head joints 0..4) −
min(Y of ankles)|; ankles "measured" iff conf > 0.45; shoulder_width =
||j5−j6||; σ_shoulder = sqrt(mean(σ[5,6]²))/√2 + 0.005.

### JointKalmanBank (human.py:356-531) — v2 temporal filtering
Per-joint CV Kalman (17 × state [x,y,z,vx,vy,vz]). `sigma_acc=1.0,
gate_sigma=4.0, bone_lo=0.6, bone_hi=1.4`.
- Init: x=[z,0], P = 0.02·I6 with pos-var `max(R, 0.02²)`, R = max(σ,0.015)².
- Update: F/Q as Kalman3D (q = 1.0²); measurement used iff conf > 0.15;
  gate `m² ≤ (gate_sigma²)·3 = 48`; outlier → reset state to z, v=0,
  P = 0.05·I, conf ×0.8; else conf = 0.6·new + 0.4·old.
- Occluded joint: pure predict; conf decay `×max(0, 1 − dt/3)`.
- `predict(t)` (occlusion): CV predict, σ from P+Q diag, conf decay
  `×max(0, 1 − dt/2.5)`, source "predicted (joint kalman)".
- **Anthropometric constraints**: BONES (14 pairs: upper arms, forearms,
  thighs, shins, shoulder_width, hip_width, torso sides, neck links —
  human.py:162-170); running bone-length median (history 16); after ≥4
  samples clamp bone length to `[0.6, 1.4]×median` by moving the DISTAL joint
  along the bone direction; σ of moved joint floored at 0.05.
- σ_height = `sqrt(Σ σ[[0,15,16]][:,1]²)/1.7 + 0.01`.
- Pipeline attaches skeletons to nearest person track (hip mean), and uses
  the bank's prediction for occluded persons (≥3 updates, height > 0.3).

---

## 14. avc/slam/vo.py — stereo VO + keyframes + loop closure

### StereoVisualOdometry (vo.py:62-165)
`min_inliers=40, ransac_thresh=0.03, keyframe_min_motion=0.12,
keyframe_min_angle=0.08, loop_min_inliers=90, loop_index_gap=10`;
FeatureExtractor n_features=1400; rng seed 4.

**process(rect_left_gray, depth, t)**:
1. ORB detect (≥20 needed); round keypoints to pixels, look up
   `depth.points` (world frame) where valid.
2. If previous points exist and ≥12 current: LK-track previous keypoints into
   this frame; map tracked positions to current feature indices by NN lookup
   with `nn_d < 4.0 px²` (squared distance); need ≥12 pairs.
3. Convert both point sets to camera frames using the CURRENT pose estimate:
   `cam_prev = (pts_prev − t_prev)·R_prev`, `cam_curr = (pts_curr − t_est)·R_est`.
4. `ransac_rigid(cam_curr, cam_prev, threshold=0.03, iters=300)` — solves
   `cam_prev ≈ R·cam_curr + t`, i.e. **p_prev = T_rel @ p_t**.
5. Accept iff inliers ≥ 40: `pose_c2w ← pose_c2w @ T_rel`;
   `σ_t = 0.004 + 0.004·||t|| + 0.6·0.05/n_in`; `σ_rot = 0.002 + 0.5/n_in`;
   total_path += ||t||.
6. **Keyframe**: first frame or `_motion_since_kf() > 0.12`, requires ≥30
   features. `_motion_since_kf = ||Δt|| + 0.5·||so3_log(pose @ inv(kf_pose))||`.
   Consecutive-keyframe edge `T_ij = inv(prev.pose) @ kf.pose`, weight 1.0.

**Loop closure (vo.py:184-220)**: when keyframes ≥ loop_gap+3 (13): match
last keyframe vs every older one (gap ≥ 10) with BFMatcher(NORM_HAMMING) knn
k=2, Lowe ratio 0.7; need ≥90 good matches and ≥30 finite 3D pairs on both
sides; `ransac_rigid(pi, pj, threshold=0.10, iters=200)` (constraint from
matching WORLD points — identity if no drift); ≥30 inliers; skip if drift
`(||t|| + ||so3_log(T_corr)||) < 0.05`; else add loop edge (weight 0.5) and
run `optimize_pose_graph()`.

**Pose-graph Gauss-Newton (vo.py:223-264)**: 12 iterations over all keyframe
poses (c2w). Edge residual `err = se3_log(inv(Tj) @ Ti @ inv(T_ij))` (6x1).
Jacobians are **numeric**, central-free (forward) with eps=1e-6 per dof
(`J_i` wrt left-multiplication perturbation `se3_exp(d) @ Ti`, `J_j` wrt
`se3_exp(d) @ Tj`). H = Σ w·[Jᵢ;Jⱼ] blocks, b = Σ w·Jᵀ err. Anchor: first 6
dofs pinned (`H[:6,:] = 0; H[:6,:6] = 1e6·I; b[:6] = 0`). Solve
`(H + 1e-6·I)·dx = −b`; update `poses[i] = se3_exp(dx_i) @ poses[i]`; live
camera pose ← last keyframe pose.

---

## 15. avc/pointcloud/fusion.py — incremental voxel hash

`VoxelFuser(voxel=0.02, min_conf=0.25, stride=2, dynamic_radius=0.55)`.
- Voxel record (flat array layout, fusion.py:22): `[xyz(3), rgb(3), conf(7),
  count(8), dynamic(9), last_t(10)]`; key = `floor(p/voxel)` int64 triple.
- `integrate(depth, image, t)`: sample grid every 2 px; keep conf ≥ 0.25 and
  finite points; colors BGR from the rectified image; per-voxel running mean
  of position & color (mean of means, n-weighted), conf = max, count += m,
  last_t = t. One `np.add.at`-free grouped pass via lexsort.
- `invalidate_region(center, radius=0.55)`: dynamic_counter += 1 for voxels
  within radius (called by the pipeline for dynamic objects with
  `radius = 0.8·max(dims) + 0.2`).
- `sweep_stale(now, max_dynamic_marks=2, stale_seconds=0.6)`: drop voxels
  with `(dynamic ≥ 2 AND count < 3) OR (dynamic > 0 AND now−last_t > 0.6)`.
- `statistical_outlier_removal(cloud, k=12, sigma=1.5)`: cKDTree mean kNN
  distance vs mean+1.5σ (pipeline applies it when cloud > 200 points).
- `merge_clouds`: concatenation.

---

## 16. avc/mesh/tsdf.py — TSDF + marching cubes

`TSDFVolume(origin=(-2.6, -0.15, -0.4), size=(5.2, 2.9, 6.3), voxel=0.05,
trunc=0.08)`; dims = ceil(size/voxel); tsdf init +1 (far), weight 0.
**integrate (tsdf.py:65-107)**:
```
project each voxel center into the RECTIFIED camera: u = round(x/z·fx+cx),
v = round(y/z·fy+cy); z > 0.05; inside image; depth.valid there
sdf = obs_z(=world z of the depth pixel, == rect z) − vox_z
use = valid & sdf > −trunc & |obs_z| < 30
val = clip(sdf/trunc, −1, 1);  w = confidence + 0.2
EMA updates: weight += w; tsdf += w·(val − tsdf); color += w·(c − color)
weight capped at 100
```
**extract_mesh (tsdf.py:110-142)**: mask weight ≥ 0.4; volume = where(mask,
tsdf, 1.0); scikit-image `marching_cubes(level=0.0)`; verts_world =
`verts·voxel + origin`; colors from nearest voxel (floor(verts)); dim colors
to 140 where weight < 0.2. **texture_project**: reproject vertices into a
depth map's rectified image and overwrite vertex colors where valid.

Pipeline: `mesh_every=10` frames, TSDF integrates all pairs; mesh cached
until dirty.

---

## 17. avc/world — model, relations, memory

### WorldModel (model.py:30-152)
`static_speed_thresh=0.08 (m/s), static_min_obs=12, forget_after=30.0 s`.
`update(states, t)`: merge by id (replace position/σ/dims/velocity/conf/
visibility/history (last 2000)/appearance/skeleton); `is_dynamic = speed >
0.08`; objects unseen > 30 s → LOST; version++.
`fit_structure(cloud, max_planes=4)`: needs ≥500 pts per iteration;
`fit_plane(iterations=300, threshold=0.04)`; conf =
`clip(2·inliers/remaining, 0, 0.98)`; plane classify (model.py:104-109):
`n_y > 0.9 floor; n_y < −0.9 ceiling; else wall` (normals pre-oriented +Y by
fit_plane); remaining points minus inliers each round.

### Relations (relations.py)
Predicates: near, on, in, behind, in_front_of, facing, contains, beside,
left_of, right_of. `near_thresh=1.5`. Per ordered pair (a,b):
- **near**: d < 1.5; conf = `exp(−0.5(σ/0.15)²)` capped 0.95; σ =
  sqrt(Σ(σa²+σb²)).
- **on (a rests on b)**: XZ-footprint overlap (half-extents) `min overlap >
  0.02` and all > 0; `b_top < a_bot + 0.35`; `a_bot − b_top > −0.3`;
  `a.h < 0.8·b.h`; `|gap| < 0.22`; conf = `exp(−gap²/(2(σ+0.05)²))·0.9`.
- **contains (b ⊃ a)**: a.dims < 0.7·b.dims (all axes); overlap > 0;
  `ha ≤ hb + 0.05`; b_bot < a_bot and a_top < b_top; conf 0.85.
- **facing (person a → b)**: `facing = head − hipcenter`; cos(facing, b−a) >
  0.65; conf = min(0.9, cos).
- **behind (a behind b)**: `db < da − 0.2` and `dir_a·dir_b > 0.985`; conf 0.8.
- **left_of/right_of**: `right = cross([0,1,0], dir_a)`; `(a−cam)·right <
  −0.25 / > 0.25`; conf 0.7.
- **on floor**: bottom = pos_y − h/2; gap = |bottom − (−d/ny)| when |ny| >
  0.9; |gap| < 0.25 → on "floor", conf min(0.9, floor.conf), distance gap.
Output dicts: `{subject, predicate, obj, confidence, distance}`. Rendering:
`relations_to_text` (sorted by conf) and `relations_to_tree` ("WORLD" tree).

### SpatialMemory (memory.py)
Per-object deques (maxlen 400) of (t, position) and (t, dims).
`trajectory_stats(window=3.0 s)`: least-squares `p = c0 + c1·t` per axis
(velocity = c1); if ≥3 samples a second-order fit `+0.5·c2·t²` gives
acceleration; path = Σ||Δp||; direction = v/|v|. `position_at(t)` linear
interpolation. `movement_report`: up to 5 sampled waypoints + velocity lines.
`record_history` ingests the track history (last 200).

---

## 18. avc/measurement/measure.py — measurement engine

`systematic_floor = 0.008 m`.
- **distance(a, σa, b, σb)**: propagate_a_b; σ = quadrature(σ, 0.008);
  conf = sigma_to_confidence (floor 0.95, k 0.1); method
  "euclidean 3D, error-propagated".
- **height/width/depth**: value = dims[axis]; σ = quadrature(σ_dims[axis],
  0.008); conf floor 0.9; source "calibrated stereo (+ tracking)" if
  n_observations > 1.
- **person_height**: skeleton head-to-feet iff `ankles_measured` (source =
  skeleton source), else the 3D box height with method note.
- **shoulder_width**: skeleton ||j5−j6|| else 0.7× box-width proxy.
- **limb_lengths**: 8 segments (arms/thighs/shins), σ =
  sqrt(Σ(σa²+σb²))/√2 floored at 0.008.
- **footprint_area** = w·d; σ = value·hypot(σw/w, σd/d) + 1e-4.
- **volume** = Π dims; σ = value·sqrt(Σ(σi/di)²) + 1e-5.
- **angle** σ_deg = 1.5.
- **velocity** σ = ||σ_pos||/0.3 + 0.02 (0.3 s window).
- **measure_points**: user two-point distance with σ_xy 0.03, σ_z 0.06.
Formatting: digits chosen from sigma magnitude (2 significant-ish).

---

## 19. avc/export/exporters.py — 8 formats + .avcworld

- **PLY points** (binary_little_endian): header `ply/format … 1.0/comment
  AVC point cloud/element vertex N/property float x,y,z/property uchar
  red,green,blue/end_header`; record dtype [(x,f4),(y,f4),(z,f4),(r,u1),
  (g,u1),(b,u1)]. ASCII variant with %.5f.
- **PLY mesh**: + `element face F`, `property list uchar int vertex_indices`;
  face records dtype [(n,u1)=3, (v,i4,3)].
- **PCD 0.7**: header `FIELDS x y z rgb / SIZE 4 4 4 4 / TYPE F F F U /
  COUNT 1 1 1 1 / WIDTH N / HEIGHT 1 / VIEWPOINT 0 0 0 1 0 0 0 / POINTS N /
  DATA binary`; rgb packed `(r<<16)|(g<<8)|b` reinterpreted as f32.
- **OBJ**: `v x y z r g b` (colors /255), `vn`, `f i j k` (1-based).
- **STL binary**: 80-byte header "AVC binary STL", u32 count, per triangle
  normal (cross product normalized), 3 vertices f32, u16 attr 0.
- **GLB** (glTF 2.0): magic `0x46546C67`, version 2, total length; JSON chunk
  `0x4E4F534A` (space-padded to 4), BIN chunk `0x004E4942` (zero-padded).
  Accessors: POSITION 5126 VEC3 (with min/max), NORMAL 5126 VEC3, COLOR_0
  5121 VEC4 normalized, indices 5125 SCALAR; mesh primitive mode 4 + optional
  point-cloud primitive mode 0; material pbr baseColor [1,1,1,1], metallic 0,
  roughness 0.9; bufferViews 4-byte aligned (target 34962/34963).
- **gltf**: same content split into .gltf JSON + .bin sidecar.
- **XYZ**: `x y z r g b` text.
- **.avcworld binary (v1)** (exporters.py:308-396):
```
magic    4s  b"AVCW"
version  u16 1
json_len u32  + utf-8 json:
   {objects: {id: ObjectState.to_json()
      + history: [[t,[x,y,z]],…] (last 400)
      + skeleton: {joints 17x3, confidence 17, source}},
   relations: [...], planes: [{label, normal, d, confidence, inliers}],
   cameras: [Camera.json()], world_version, last_update}
cloud_n  u32;  points N×3 f32 (world);  colors N×3 u8
mesh_vn  u32;  mesh_fn u32
verts mesh_vn×3 f32; faces mesh_fn×3 i32; colors mesh_vn×3 u8
```
All little-endian; `import_world` round-trips (tested).
Dispatcher `export_scene(path, mesh?, cloud?, world?, cameras?)` by extension.

`ObjectState.to_json` schema (types.py:245-262): id, label, position,
sigma_position, dimensions, sigma_dimensions, velocity, speed, confidence,
confidence_breakdown, visibility, first_seen, last_seen, n_observations,
is_dynamic, source, history_n. `Camera.json` (types.py:121-136): id, name,
kind, frame_rate, intrinsics{...incl sigmas, distortion}, pose_c2w 4x4.

---

## 20. avc/render — snapshots & viewport

`SnapshotRenderer` (matplotlib Agg, figsize (11,8), dpi 110): scatter cloud
(≤60k points, painter's order by −Z), mesh Poly3DCollection (≤30k tris,
alpha 0.35, color 0.72), object boxes (12 edges; green person / blue other /
orange non-visible; label `id\nconf% visibility`), skeletons (LIMBS crimson),
camera frustums (f 0.22, depth 0.55, red), floor grid x∈[−2.4,2.4] z∈[0,5.6];
views iso(elev 28, azim −60), top(89,−90), front(8,−90), side(8,0);
box aspect (4.8, 2.6, 5.6). `render_debug_panel`: (title, image) grid, 3 cols.
`render/viewport.py`: optional Open3D live viewport (import-guarded).

---

## 21. avc/camera + avc/capture

`CaptureSource` (cv2.VideoCapture; sets 640x480 & fps; resize if mismatch;
reconnect on failure). `ReplaySource` (video or image dir, fps pacing, loop).
`SyntheticSource` (renders the sim world at wall-clock fps; returns
(t, image, {t_scene, gt_objects?, gt_joints?})). `make_camera(kind, source,
intrinsics?, pose?, name, fps)` default intrinsics 640x480 f=500 c=(320,240).
`SynchronizedCapture(cameras, sync_tolerance=0.05, scene?, fps=10)`: per-camera
threads push into lock-protected **latest-wins** buffers (stale dropped);
`latest_group` assembles frames when max pairwise skew ≤ 0.05 s (synthetic:
always). FrameGroup timestamp = mean.

---

## 22. avc/engine/pipeline.py + avc/api/engine.py

### EngineConfig defaults (pipeline.py:44-78)
```
backend="ncc", num_disparities=64, block=9, lr_check=True,
detector="auto", yolo_model="models/yolov8n.onnx",
yolo_seg_model="models/yolov8n-seg.onnx", detection_conf=0.35,
refine_masks=True, lifting_min_points=150,
confirm_hits=3, max_misses=40, pose_backend="auto",
multi_view_detection=True, slam_enabled=False,
voxel_size=0.02, cloud_min_conf=0.25, cloud_stride=2,
mesh_enabled=True, mesh_voxel=0.05, mesh_every=10,
mesh_origin=(-2.6,-0.15,-0.4), mesh_size=(5.2,2.9,6.3), near_thresh=1.5
```

### PerceptionPipeline.step (pipeline.py:232-399) — 10 stages
1. Stereo on every pair (left_world_pose from `camera_poses` dict).
2. Retina + Farneback motion on the primary raw left.
3. VO (primary pair, rectified-left gray + depth) if slam_enabled; on accept,
   propagate: `camera_poses[cam] = vo.pose_c2w @ rigid_inverse(base_left) @
   cam.pose` (static rig extrinsics composed).
4. Detection+lifting on every pair (gt_hook / detector.detect(rect_left);
   GrabCut refine masks for box-only detectors); `lo.view = pair_name`;
   multi-view merge via `merge_lifted_views(primary_view=primary,
   require_primary=True)`.
5. (lifting folded into 4)
6. Human pose on the primary pair only (person detections; gt_pose_hook or
   pose2d adapter) → StereoSkeletonLifter.
7. `tracker.update(lifted, t)`; `tracks_as_states(t)`; attach skeletons to
   nearest free person track (by hip mean distance) via per-track
   JointKalmanBank; occluded persons get bank.predict(t) if ≥3 updates and
   predicted height > 0.3.
8. `world.update(states, t)`; memory.record; `compute_relations(world,
   cam_pos, near_thresh)`.
9. Cloud fusion: invalidate regions of dynamic objects
   (`radius = 0.8·max(dims) + 0.2`, n_observations > 2); integrate every
   pair's depth; sweep_stale(t).
10. TSDF every 10th frame (all pairs). Timings EMA 0.8/0.2.

`rectified_camera(pair)`: virtual Camera with P1 intrinsics and pose
`camera_poses[left] @ make_pose(R1ᵀ, 0)` — used by GT pose injection.

### VisionEngine (api/engine.py)
- `VisionEngine.demo()`: Scene cameras; pairs computed from world poses:
  `R = l.Rᵀ r.R`, `T = l.Rᵀ (r.t − l.t)` (R,T of right in left coords).
- `start(threaded, fps=10)`: builds pipeline + SynchronizedCapture; loop
  thread steps on latest groups (t = metadata t_scene for sim).
- `step_offline(t)`: renders every camera directly (Renderer add_noise=2.0),
  attaches gt_render to metadata, steps pipeline — deterministic, no threads.
- `camera.calibrate(path | image_dirs, pattern (9,6), square 0.08)`:
  loads JSON or calibrates from folders (≥8 detected boards per camera).
- Sub-APIs: world (get_objects/get_object/relations tree/movement_report),
  measure (distance/height/width/depth/volume/area/velocity/
  distance_to_camera), scene (export by extension; snapshot).
- Results buffer last 120.

---

## 23. avc/sim — the synthetic validation world (dev/test only)

### Renderer (renderer.py:193-311)
Numpy z-buffer rasterizer. `LIGHT_DIR = normalize([0.35, 0.8, 0.48])`,
ambient 0.55, diffuse `0.45·max(n·L, 0)`, optional glow floor. Per-triangle:
back-face culling none (double-sided default); near clip z > 0.05;
barycentric coverage with 1e-4 tolerance; **perspective-correct** depth
`1/(Σ wᵢ/zᵢ)` and world position `Σ wᵢ·(1/zᵢ)·Pᵢ / Σ wᵢ·(1/zᵢ)`;
texture evaluated per-pixel on interpolated world position (noise textures)
or interpolated uv (checker/chessboard); sensor noise ±2 int uniform
(rng seed 7). Outputs image (uint8 BGR), depth (float32 camera Z),
obj_id (int32 item index, −1 bg).
`render_supersampled(factor=2)`: renders at 2x intrinsics and INTER_AREA
downsampling — anti-aliased edges (fixes chessboard corner bias); depth/obj
buffers subsampled `[::2, ::2]`.

**Textures (deterministic)**:
- `_hash01(p)`: `q = floor(p·48); h = sin(qx·12.9898 + qy·78.233 + qz·37.719)`;
  `h − floor(h)`.
- `tex_checker(base, scale=0.5)` (uv-space): `base·(0.55 + 0.45·checker) +
  (hash−0.5)·14`.
- `tex_noise(base, amp)`: `base + (hash−0.5)·2·amp`.
- `tex_chessboard(square=0.08, cols=9, rows=6)` (uv): white 245 / black 10,
  1-square white border, checker `((i+j)%2) > 0.5`.

**Geometry helpers**: `box_corners(center, size, R)` — 8 corners, local
layout [ (+hx,−hy,−hz), (+hx,−hy,+hz), (−hx,−hy,+hz), (−hx,−hy,−hz),
(+hx,+hy,−hz), (+hx,+hy,+hz), (−hx,+hy,+hz), (−hx,+hy,−hz) ] (renderer.py:98-99)
— the historical "degenerate 8th corner" bug is fixed here and pinned by
`test_box_corners_well_formed` (8 distinct, mean = center).
`BOX_FACES = [(0,1,3,2),(4,6,7,5),(0,4,5,1),(2,3,7,6),(0,2,6,4),(1,5,7,3)]`;
`box_to_tris` → 12 triangles + 6 face normals. `segment_box(a, b, thick)`:
oriented box along a→b (+Z along the segment; up [0,1,0] fallback [0,0,1]);
size `[thick, thick, L + 0.6·thick]`. `plane_to_tris`: quad + uv in metres.

### Scene (scene.py)
Room: x ∈ [−2.4, 2.4], z ∈ [0, 5.6], height 2.6 (ROOM dict, scene.py:211).
**Cameras (default_cameras, scene.py:214-246)**: 640x480, fov 65° →
`fx = fy = 0.5·width/tan(32.5°)` (= 502.30 px), c = (320, 240). Front rig at
(0.0, 1.55, 0.30) looking at (0.0, 0.9, 3.4), baseline 0.12 m along the rig
right axis (cam_left − B/2, cam_right + B/2). Side rig at (1.55, 1.55, 1.75)
looking at (0.0, 0.55, 3.3) (cam_side_a/b, same 0.12 baseline). 30 fps.
**Static objects**:
```
table  pos (0.15, 0, 3.05) size (1.50, 0.74, 0.85)   [top slab 0.06 thick at
                                                     y+0.71; base slab inset]
chair  pos (−1.90, 0, 2.60) size (0.46, 0.92, 0.46)  [seat y+0.45; back at
                                                     z−0.20; legs box]
cup    pos (0.25, 0.74, 2.92) size (0.085, 0.105, 0.085)
panel  pos (−0.27, 0, 2.15) size (0.45, 1.92, 0.18)  [the occluder]
floor/walls: tex_checker 0.5 m scale
```
**Dynamics**: moving box path `x = 1.80 − 0.05·t, y=0.09, z=4.6`, size
(0.30, 0.18, 0.30). Person path `x = −1.6 + 0.40·t, y=0, z=4.45`, yaw 0,
duration 6.0 s.
**PersonSpec (scene.py:31-46)**: height 1.78, shoulder_width 0.42, hip_width
0.30, thigh 0.44, shin 0.43, torso 0.52, upper_arm 0.30, forearm 0.27,
head 0.23, stride 0.62, walk_speed 0.40 m/s; colors shirt (60,130,70),
pants (50,55,90), skin (190,160,130), shoe (40,35,30).
`hip_height = thigh+shin = 0.87; shoulder_height = 1.39; nose_height =
shoulder + neck + 0.12` (neck = height − shoulder − head/2).
**Gait (scene.py:89-144)**: phase `φ = 2π·t·speed/0.62`; vertical bob
`0.025·sin(2φ)`; ankle swing `sin(φ)` (right leg φ+π) along forward, stride/2
amplitude, lift `0.09·max(sin,0)`; **two_link_ik** (scene.py:67-86):
`a = (L²+l1²−l2²)/(2L)`, `h = sqrt(max(l1²−a²,0))`, knee bends toward
forward; arms counter-swing (elbow fwd 0.16·swing, wrist 0.22·swing) hanging
−upper_arm/−forearm in Y; eyes/ears offsets ±0.045/±0.085.
`person_render_items`: per-part boxes (torso, head, neck, thighs/shins
thick 0.15/0.12, upper arms/forearms 0.11/0.09, feet), all gt_id "person".
**GT objects** (scene.py:353-374): table/chair/cup/panel static centers
(dims/2 lifted in Y), box on its path, person center = path + height/2,
dims (0.45, 1.78, 0.30). `gt_person_joints(t)` → (17,3) COCO.
**Calibration boards**: `chessboard_items(center, yaw, pitch)` vertical
plane (cols+3)x(rows+3) squares with white border;
`calibration_poses(n=16 default / 24 in demo, cam_pos=(0,1.55,0.30),
rng_seed=3)`: arc angle `−0.95 + 1.9·i/(n−1) ± 0.08`, distance U(1.1, 3.2),
height cam_y + U(−0.60, +0.35), yaw = bearing ± 0.25, pitch U(−0.45, +0.45) —
v2's wide protocol (v1 narrow arc left cx weakly observable).

---

## 24. avc/apps/benchmark.py — competitor comparison harness

- **Rectified virtual pair** (benchmark.py:62-73): right camera = left pose +
  `R·[baseline, 0, 0]` (pure X offset, same rotation) → truly rectified input.
- **GT disparity** (benchmark.py:99-115): from left depth, project into right
  (`Xr = X − B`), z-buffer visibility check `|z_r_at − z_l| < 0.03`;
  `d_gt = fx·B/z`, valid when `1 ≤ d ≤ 62`; eval mask inset
  `[64+8 : H−8, 8 : W−8]`.
- **Cases**: standard t ∈ {0, 2, 4, 5.5} (noise 2), noise8 (8/255),
  photometric (right ×1.25 +12 bias), textureless (flat-color walls/floor/
  table/box/person-limbs).
- **Methods**: OpenCV StereoBM (blockSize 15, uniqueness 10, speckle 80/2,
  disp12MaxDiff 1, texture 10), OpenCV SGBM (blockSize 7, P1 = 8·3·49,
  P2 = 32·3·49, uniqueness 8, speckle 80/2, preFilterCap 31, 3WAY),
  AVC v1 NCC (block 9), AVC v2 SGM2 (block 9 in bench).
- **Metrics (KITTI-style)** (benchmark.py:224-269): D1-all =
  `(|e|>3px AND |e|>5%·d_gt)` on matched; D1-dense = (invalid + D1)/n_base;
  bad 0.5/1 px; MAE/RMSE px; med |e| m (via Z = fx·B/d); coverage; runtime
  (median of 3 after warmup); σ calibration 1σ/2σ/3σ; measured-only D1 &
  coverage (excluding interpolated).
- **Tracking head-to-head** (benchmark.py:369-...): GreedyNN (gate 0.8 m,
  coast at last position, dims EMA 0.7/0.3), KF+Greedy (same KF, greedy
  Mahalanobis), vs the AVC tracker, on identical recorded observation
  streams; CLEAR-MOT-style Hungarian evaluation (0 ID switches, occlusion
  prediction 43 cm vs 69-85 cm).

---

## 25. Validation (avc/apps/demo_synthetic.py) — stages & results

Stages: (1) calibration from 24 2x-supersampled chessboards; (2) stereo depth
vs renderer GT (sgm2 backend, block 13); (3) full pipeline 60 frames @ 10 fps
(backend sgbm block 7, GT detector + GT pose, mesh_every 10, max_misses 40);
(4) identity/occlusion; (5) measurement report vs **visible-surface GT**
(`gt_surface_stats`: per-object world points from the renderer id+depth
buffers; median center, P0.5-P99.5 dims — surfaces, not box volumes);
(5b) segmentation ablation (mask vs box-only lifting, same depth maps,
t ∈ {0.4, 0.8, 4.6, 5.0, 5.4}); (6) world/relations/memory; (7)
reconstruction; (8) SLAM (rig slides 0.08·i for i=0..9 = 0.72 m);
(9) exports + round-trip; (10) visualizations. Report rows:
`zscore = |measured − gt|/σ`, PASS iff ≤ 3.0. Eval at the final frame time
`t_eval = (n_frames−1)·0.1` (the v1 timestamp mismatch — world at 5.9 vs GT
at 5.5 — is fixed).

**Measured results (documented, verbatim)**:
- v1.0: 33/42 checks; v1.1: 38/42; **v2.0: 42/42 within 3σ**.
- Depth (2–6 m): median |err| 2.9 cm (v1.1 NCC) → **2.4 cm** (v2 sgm2);
  p95 ~55 cm; σ calibration 81/92/94% (v1.1) → **90/95/97%** within
  1σ/2σ/3σ (v2); coverage after rectification of the verged rig 46.9%
  (100% on rectified pairs, 85% true measurements).
- Calibration: fx true 502.30 → 499.92 ± 1.20 (2.0σ); cx 320 → 318.01 ± 2.0;
  cy 240 → 237.78 ± 2.0; baseline 12.00 → 11.97 cm; Sampson RMS 0.017 px
  (v1: 0.066); rect row error 0.073 px; reprojection RMS 0.170 px.
- Objects (t=5.9 s, vs visible-surface GT): cup 0.2/0.5/0.7σ; panel
  0.1/0.0/0.1σ; table 0.2/0.4/0.3σ; box 0.4/0.3/1.0σ; person 0.3/0.2/0.1σ;
  all dims ≤ 2.6σ.
- Person: speed 0.456 vs 0.40 m/s GT; reappearance after 2.5 s occlusion
  19.7 cm; skeleton height median 1.854 m (0.2σ; was 0.955 m in v1.1);
  single track id.
- VO: 0.643 m estimated vs 0.72 GT (0.8σ), 110–136 inliers/frame.
- Reconstruction: 665,690-voxel cloud (2 cm), 19,373-vertex / 36,320-face
  TSDF mesh (5 cm).
- Segmentation ablation: median extent error 0.030 m (mask) vs 0.089 m
  (box) — 66% reduction (panel 88%, table 66%, person 51%).
- Stereo benchmark vs SGBM (v2): D1-dense 1.76% vs 2.11%; measured-pixel D1
  0.2–0.7% vs 1.12%; coverage 100% vs 99%; photometric 1.43% vs 2.78%;
  textureless 34.6% vs 39.9%; **loses** bad-1px (4.93% vs 2.69%) and runtime
  (4.5 s vs 47 ms).

**Honest status matrix** (avc/__init__.py:24-133, runtime-audited): with
onnxruntime + local models + no mediapipe/open3d: 23 WORKING / 5 PARTIAL
(loop closure+pose graph "no global BA"; TSDF "CPU numpy"; MediaPipe absent;
hardware backends; viewport-on-server) / 1 INCOMPLETE (AI spatial reasoning).
Full list of rows: core, camera sources, sync capture, calibration, retina,
stereo rect+disparity, LR consistency, metric depth, detection classical,
detection ONNX (WORKING/PARTIAL/INCOMPLETE by availability), instance
segmentation (same), 3D lifting, 3D tracking, re-identification, human pose
2D (mediapipe or PARTIAL), human pose 3D, VO, loop closure (PARTIAL), point
cloud fusion, multi-camera fusion, TSDF (PARTIAL), spatial memory, world
model + relations, measurement, export, snapshot rendering, live viewport
(PARTIAL/INCOMPLETE), hardware acceleration (PARTIAL), AI spatial reasoning
(INCOMPLETE).

---

## 26. Historical bug register (must be encoded as regression tests in Rust)

1. **world_to_cam sign error** — was `R^T·P + c` instead of
   `R^T·(P − c)`; fixed at core/types.py:91-98
   (`return (pts_world - t) @ R`). Caught by synthetic GT validation.
2. **Degenerate 8th box corner** — box_corners produced a repeated corner;
   fixed layout at sim/renderer.py:95-100, pinned by
   tests/test_pipeline.py:58-64 (8 distinct, symmetric mean).
3. **L→R matching direction** — left pixel x corresponds to right pixel x−d;
   implemented by `_shift_right` (stereo/matcher.py:42-52) for L→R cost and
   `_shift_left` (matcher.py:55-65) + `_right_best` (matcher.py:188-208) for
   the R→L cross-check; `_shift_right_u32` for census (matcher2.py:493-499).
4. **SE3 exp/log correctness** — V-matrix (Barfoot) formulas at
   core/geometry.py:48-91; round-trip test tests/test_geometry.py:11-17
   (atol 1e-6); near-π branch in so3_log.
5. **Umeyama reflection** — `S[2,2] = −1` when det(U)·det(Vt) < 0
   (core/geometry.py:142-143); test test_umeyama_rigid.
6. **Cumulative coasting divergence** — predict-to-t was applied repeatedly
   to missed tracks (quadratic divergence, masked by exploding covariance);
   fixed with anchored states never written back
   (tracking/tracker.py:188-202, 313-332; same policy in
   pose/human.py JointKalmanBank).
7. **Multi-view position drag** — secondary-rig support lost the person's
   legs; merge now uses primary-authoritative position
   (detection/lifting.py:428-442).
8. **Subpixel parabola on SGM-aggregated cost biased** — fit moved to the
   raw NCC cost curve (stereo/matcher2.py:380-403); aggregated-parabola
   inflated bad-1px ~4x.
9. **Integer-grid cost jaggedness** — half-pixel disparity grid (128 slices)
   with lerp-shifted NCC + cost-averaged census (matcher2.py:136-185).
10. **Census 5x5 too small for texture scale** — two-scale census (fine
    24-bit + radius-4 ring 8-bit) (matcher2.py:44-48).
11. **World-Z vs rect-Z conflation in mask-hole fill** — row medians are
    computed in the RECTIFIED camera frame (detection/lifting.py:163-184).
12. **Hard Z-cluster split amputated multi-depth objects** (0.30 m gap cut
    the forward shin / table far edge) — replaced by the object-aware
    Z-envelope (lifting.py:221-240).
13. **MAD window amputated flat objects** (±0.15 m floor cut the table top)
    — window floored at the envelope (lifting.py:243-253).
14. **Extent EMA shrinkage under partial occlusion** — extents now P85 of
    dims history, σ floored by observational MAD
    (tracker.py:238-250).
15. **Kalman R over-confidence** — measurement noise floors
    [0.030, 0.030, 0.060] (tracker.py:111-125) reflect real surface-median
    scatter, else the filter rejects valid matches during gait.
16. **Aliased chessboard edges biased intrinsics** (fx +5.4 px, cy +9.2 px
    at 1x) — 2x supersampled board rendering + 4x corner refinement +
    systematic σ floors (calibrator.py:74-100, 151-163;
    renderer.py:228-258); protocol widened to 24 poses.
17. **Eval timestamp mismatch** (world at t=5.9 vs GT at 5.5) — eval at the
    final frame time (apps/demo_synthetic.py:803).
18. **SGBM uniqueness semantics for ambiguous matches** — hard uniqueness
    gate demotes them to interpolated+flagged, never measurements
    (matcher2.py:429-436).
19. **Latent person-skeleton failure** (0.955 m height): thin-limb stereo
    bleeding to background — person-anchored depth + per-joint Kalman +
    anthropometric bone clamping (pose/human.py:173-531).
20. **Threading/capture**: latest-wins buffers (no queue growth)
    (capture/sync.py:72-118).

---

## 27. Numeric constants quick reference

| constant | value | location |
|---|---|---|
| world/camera frames | X-right Y-up Z-fwd / X-right Y-down Z-fwd | types.py:4-5 |
| NCC block / disparities | 9 / 64 (min 0) | matcher.py:76 |
| NCC INF_COST / cost clip | 2.5 / [0, 2] | matcher.py:98,117 |
| parabola delta clip | ±0.75 | matcher.py:147 |
| σ_d clip (NCC) | [0.05, 0.6] px, 0.25/√curv | matcher.py:151-153 |
| NCC conf weights | 0.65 ratio + 0.35 texture | matcher.py:161 |
| NCC validity | d0>0, best<1.9, texture≥2.5 | matcher.py:163 |
| LR threshold | 1.0 px | matcher.py:78 |
| SGBM params | P1=8·3·b², P2=32·3·b², uniq 8, speckle 80/2, preFilterCap 31, 3WAY, /16 | matcher.py:220-233 |
| SGBM σ_d / conf | 0.3 px / 0.7 | matcher.py:237-238 |
| census offsets | 5x5−center (24) + ring r=4 (8) = 32 bits | matcher2.py:45-48 |
| census/NCC weights | 0.3 / 0.7 | matcher2.py:93 |
| SGM2 p1 / p2 | 0.06 / 0.9 (p2/(1+|∇I|)), p1_scale 0.5 on half grid | matcher2.py:94,196-197 |
| SGM2 uniqueness | 1.4 (thresh_u = 1/1.4) | matcher2.py:95 |
| SGM2 speckle area | 12 px | matcher2.py:96 |
| interp conf cap / σ mult | 0.35 / ×2 | matcher2.py:97,461-464 |
| σ_d floor (sgm2) | 0.12 (else 0.35 default, cap 0.6) | matcher2.py:97,413-417 |
| border cost (sgm2) | 1.2 (valid: best<1.1, p0≥1.0) | matcher2.py:169,431 |
| conf weights (sgm2) | 0.45 uniq + 0.25 curv + 0.30 tex; curv/0.08 | matcher2.py:425,458 |
| σ discontinuity | jump>2 px → max(σ·1.5, 0.25) | matcher2.py:466-469 |
| post-median | 3x3 where |Δ|≤1 px | matcher2.py:470-476 |
| DepthErrorModel σ_d default | 0.15 px | uncertainty.py:32 |
| σ_Z | sqrt((Z²σ_d/(fB)/√N)² + (Zσ_f/f)² + (Zσ_B/B)²) | uncertainty.py:34-51 |
| σ_xy | σ_z·1.2/f | stereo/pipeline.py:124 |
| disparity→depth | Z=fB/d (d≥0.5), X=(u−cx)Z/f, p_cam=R1ᵀp_rect | pipeline.py:108-115 |
| border mask erosion | k = max(3, block//2) | pipeline.py:82 |
| σ_baseline (engine) | 0.001 m | pipeline.py:44 |
| lifting inner margin | 0.08 (8% inset) | lifting.py:63 |
| lifting min_points | 24 (demo: 150) | lifting.py:63; pipeline.py:57 |
| sliver reject | keep < max(8, min_points) | lifting.py:213 |
| Z-envelope | max(0.40, 0.60·s_min); s_min=max(min(w,h)·z/f, 0.05) | lifting.py:228-232 |
| MAD window | max(2.5·(1.4826·MAD, floor 0.05), envelope) | lifting.py:247-249 |
| hole fill | row medians ≥3 pts, reach ≤12 rows; ≤85% of support; conf 0.35; σ 2·s_ref+0.02 | lifting.py:135-139,199-201,263 |
| σ_pos floors | [0.010, 0.010, 0.025] m | lifting.py:286 |
| σ_dims clip | [0.03, 0.5] m | lifting.py:296 |
| plausibility | dims < 3·boxBackproj + 0.15; max dim < 3.5 | lifting.py:307-308 |
| lift conf | 0.6·det + 0.4·depth; ×0.3 implausible | lifting.py:325-327 |
| points caps | 3000 per view / 6000 merged | lifting.py:330,453 |
| merge gates | 1.0 m + 0.5·dim; 0.6 m agreement; 60% dim agreement | lifting.py:343-344,417,421 |
| Kalman σ_acc | 2.0 m/s² (tracker) / 1.0 (joints) | tracker.py:87; human.py:372 |
| R floors | [0.030, 0.030, 0.060] m | tracker.py:117 |
| new-track P | pos 0.01, vel 0.5 | tracker.py:286-287 |
| assignment gates | label equal; m² ≤ 25; dist ≤ 2.5 m; cost ≤ 5.0 | tracker.py:174,211 |
| cost weights | 0.45 m²/25 + 0.25 app + 0.15 size + 0.15 reid | tracker.py:182-183 |
| velocity clamp | 4.0 m/s | tracker.py:235-236 |
| partial-view | min dims ratio < 0.75; R += (0.5·missing)² | tracker.py:223-229 |
| dims estimate | P85 of last 64; σ = max(det, min(0.5·MAD, 0.25)) | tracker.py:239-250 |
| confirm / coast / delete | 3 hits / >2 misses PREDICTED / 14 (demo 40) misses | tracker.py:131-134 |
| track_conf | 0.5 conf + 0.3 hits/8 + 0.2 coast | tracker.py:307-311 |
| appearance EMA | 0.7/0.3; history 600 positions | tracker.py:254,262 |
| reid weights / accept | 0.45/0.35/0.20; 0.55; σ growth 0.35 m/s; gate 3σ | predict.py:23-24,40,47-54 |
| skeleton anchor_tol | 0.45 m; torso check 0.6 m | human.py:188,279 |
| joint KF gate | m² ≤ 48 (=(4σ)²·3); outlier reset P 0.05 | human.py:372,432,441 |
| bone clamp | [0.6, 1.4]× running median (16 hist, ≥4) | human.py:373,506-531 |
| joint conf decay | ×(1 − dt/3) update / (1 − dt/2.5) predict | human.py:446,463 |
| VO | ransac 0.03 m, 300 iters; ≥40 inliers; ≥12 corr; NN<4 px² | vo.py:65-66,127,141 |
| VO σ | σ_t=0.004+0.004|t|+0.03/n; σ_rot=0.002+0.5/n | vo.py:147-149 |
| keyframe | motion 0.12 m + 0.5·angle; angle thresh 0.08; ≥30 feats | vo.py:66-67,154-156 |
| loop closure | gap≥10 idx; Lowe 0.7; ≥90 matches; RANSAC 0.10/200; ≥30 inliers; drift≥0.05; edge w 0.5 | vo.py:67,184-220 |
| pose-graph GN | 12 iters; numeric J ε=1e-6; anchor 1e6; damp 1e-6 | vo.py:223-264 |
| voxel fuser | 0.02 m; stride 2; conf ≥0.25; dyn radius 0.55; stale 0.6 s / 2 marks / count<3 | fusion.py:39-41,108-116 |
| SOR | k=12, 1.5σ (only >200 pts) | fusion.py:133-145 |
| TSDF | origin (−2.6,−0.15,−0.4); size (5.2,2.9,6.3); voxel 0.05; trunc 0.08; w=conf+0.2; cap 100; mesh every 10; min_weight 0.4 | tsdf.py:37-39,66,96,110 |
| world | static 0.08 m/s; forget 30 s; planes: 4 max, ≥500 pts, thresh 0.04, 300 iters | model.py:33-35,81-99 |
| relations | near 1.5; on gap 0.22/0.35/0.3/0.8; contains 0.7/+0.05; facing 0.65; behind 0.2/0.985; side 0.25; floor 0.25 | relations.py:35,66-120 |
| memory | deque 400; window 3.0 s; history ingest 200 | memory.py:36,72 |
| measurement floor | 0.008 m; conf floor 0.95, k 0.1; angle 1.5°; velocity σ/0.3+0.02 | measure.py:25; uncertainty.py:94 |
| retina | 4 levels; 600 corners 0.08/8 px; box 9; Canny 0.33/0.66·median +20/+60; EMA 0.6; conf 0.6 tex + 0.4 edg (tex/12, edg/60) | retina.py:44-90 |
| YOLO | 640; conf 0.35; IoU 0.45; letterbox 114; mask thresh 0.5; proto 160² | yolo_onnx.py:53-54; yolo_seg_onnx.py:37-39 |
| classical det | HOG stride (8,8) pad (16,16) scale 1.05; conf map (w−0.3)/1.4→[0.05,0.9]; MOG2 80/30; blob ≥900 px, conf 0.35; IoU suppress 0.4 | detector.py:52-91 |
| GrabCut refiner | ≥1200 px; 3 iter; margin 0.06; frac (0.05, 0.95); eroded box 6% | refine.py:26-28,99 |
| histogram | HSV H16xS8, ranges 0-180/0-256, L1 | segmentation.py:51-57 |
| capture | sync tolerance 0.05 s; fps 10 (demo); MOG2 learning 0.05→0 @40 frames (detector) / @30 (segmenter) | sync.py:31; sources.py |
| sim renderer | light (0.35,0.8,0.48)ᵀ; ambient 0.55; diffuse 0.45; noise ±2; near 0.05; hash 48/m, 12.9898/78.233/37.719 | renderer.py:24,36,196-225 |
| sim scene | room [−2.4,2.4]×[0,5.6]×h2.6; front rig (0,1.55,0.30)→(0,0.9,3.4); side rig (1.55,1.55,1.75)→(0,0.55,3.3); B=0.12; fov 65°; fx=502.30 | scene.py:211-246 |
| person | 1.78 m; walk 0.40 m/s; stride 0.62; phase 2πt·v/0.62; bob 0.025; lift 0.09 | scene.py:31-47,103-113 |
| demo | 60 frames @0.1 s; eval t=(n−1)·0.1; VO slide 0.72 m; renderer noise 2 | demo_synthetic.py:801-803,629-630 |

---

## 28. Rust porting notes (derived from the above)

- Reproduce the **anchored-state** pattern in both the object tracker and the
  joint Kalman bank; it is load-bearing for occlusion identity.
- The **half-pixel SGM grid** + raw-NCC parabola + hard-uniqueness + flagged
  interpolation is the differentiating stereo design; keep the honesty flags
  (interpolated ≠ measured) in the type system.
- Keep the **primary-rig authority** and P85 extents as explicit policies,
  not accidents.
- Every σ needs its floor(s); the floors are the calibration of the whole
  system (42/42 depends on them, not on gate widening).
- The renderer's perspective-correct interpolation and the deterministic
  hash textures make the GT validation reproducible bit-for-bit; port them
  first (they are the test harness).
- Port the regression tests enumerated in §26 alongside the modules.
