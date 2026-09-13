//! Stereo visual odometry: FAST + NCC patch tracking + RANSAC-Kabsch.
//!
//! Estimates the frame-to-frame motion of the rectified LEFT camera by
//! matching sparse features across frames, triangulating them with the
//! current disparity map, and robustly registering the 3D-3D pairs
//! (moving-object points and mismatches are RANSAC outliers).
//!
//! Integration: `T_world_cam(k) = T_world_cam(k-1) * delta^-1` where delta
//! maps previous-cam coordinates into current-cam coordinates.

use image::GrayImage;
use nalgebra::{Matrix6, Vector3};

use crate::camera::rectify::RectifiedRig;
use crate::core::rng::GaussRng;
use crate::core::se3::Se3;
use crate::core::{kabsch, wrap_pi};
use crate::slam::fast::{fast9, nms};
use crate::stereo::map::{DisparityMap, Quality};

#[derive(Debug, Clone)]
pub struct VoParams {
    pub fast_threshold: u8,
    pub max_features: usize,
    /// NCC patch radius (patch is (2r+1)^2)
    pub patch_radius: usize,
    /// search radius around the previous position (px)
    pub search_radius: usize,
    /// minimum NCC score to accept a match
    pub min_ncc: f64,
    /// minimum NCC uniqueness margin (best minus second best)
    pub ncc_margin: f64,
    /// maximum feature depth-sigma (m)
    pub max_sigma_z: f64,
    /// RANSAC inlier gate (m)
    pub ransac_thresh: f64,
    pub ransac_iters: usize,
    /// reject implausible per-frame rotations (rad)
    pub max_rotation: f64,
    /// minimum inliers to accept the motion estimate
    pub min_inliers: usize,
}

impl Default for VoParams {
    fn default() -> Self {
        VoParams {
            fast_threshold: 20,
            max_features: 350,
            patch_radius: 6,
            search_radius: 14,
            min_ncc: 0.65,
            ncc_margin: 0.01,
            max_sigma_z: 0.25,
            ransac_thresh: 0.20,
            ransac_iters: 150,
            max_rotation: 0.3,
            min_inliers: 8,
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct StoredPoint {
    /// pixel in the frame where detected
    u: f64,
    v: f64,
    /// 3D position in THAT camera frame
    p: Vector3<f64>,
}

#[derive(Debug, Clone)]
pub struct VoState {
    /// cam-to-world pose of the rectified left camera.
    pub t_world_cam: Se3,
    /// 6x6 body-frame pose covariance (grows when VO degrades).
    pub cov: Matrix6<f64>,
    prev_gray: Option<GrayImage>,
    prev_pts: Vec<StoredPoint>,
    pub frames: usize,
    pub last_inliers: usize,
    pub last_matched: usize,
    /// total path length estimated (m)
    pub path_len: f64,
}

#[derive(Debug, Clone)]
pub struct VoResult {
    pub t_world_cam: Se3,
    /// previous-cam -> current-cam transform (identity when VO failed)
    pub delta: Se3,
    pub inliers: usize,
    pub matched: usize,
    pub features: usize,
    /// false when the motion estimate was rejected this frame
    pub ok: bool,
}

impl Default for VoState {
    fn default() -> Self {
        Self::new()
    }
}

impl VoState {
    pub fn new() -> Self {
        let mut cov = Matrix6::zeros();
        for i in 0..6 {
            cov[(i, i)] = 1e-8;
        }
        VoState {
            t_world_cam: Se3::identity(),
            cov,
            prev_gray: None,
            prev_pts: Vec::new(),
            frames: 0,
            last_inliers: 0,
            last_matched: 0,
            path_len: 0.0,
        }
    }

    /// Process one rectified left frame + its disparity map.
    pub fn update(
        &mut self,
        left_rect: &GrayImage,
        disp: &DisparityMap,
        rect: &RectifiedRig,
        params: &VoParams,
        rng: &mut GaussRng,
    ) -> VoResult {
        self.frames += 1;

        // 1. current features
        let corners = fast9(left_rect, params.fast_threshold);
        let feats = nms(&corners, 5, params.max_features);

        // 2. triangulate current features
        let mut cur_pts: Vec<StoredPoint> = Vec::with_capacity(feats.len());
        for &(u, v, _) in &feats {
            if let Some(p) = triangulate_pixel(u as f64, v as f64, disp, rect, params) {
                cur_pts.push(StoredPoint { u: u as f64, v: v as f64, p });
            }
        }

        // 3. match previous points into the current frame (NCC search)
        let mut delta = Se3::identity();
        let mut inliers = 0usize;
        let mut matched = 0usize;
        let mut ok = false;

        if let (Some(prev_img), false) = (&self.prev_gray, self.prev_pts.is_empty()) {
            // unreachable, kept for clarity
            let _ = prev_img;
        }
        if let Some(prev_img) = &self.prev_gray {
            let mut pairs_a: Vec<Vector3<f64>> = Vec::new();
            let mut pairs_b: Vec<Vector3<f64>> = Vec::new();
            let mut pair_sigma: Vec<f64> = Vec::new();
            for sp in &self.prev_pts {
                // skip points whose pixel is too close to the border for NCC
                let r = (params.patch_radius + params.search_radius) as f64;
                if sp.u < r || sp.v < r || sp.u > (left_rect.width() - 1) as f64 - r || sp.v > (left_rect.height() - 1) as f64 - r {
                    continue;
                }
                if let Some((nu, nv, score, margin)) = ncc_match(prev_img, left_rect, sp.u, sp.v, params) {
                    if score >= params.min_ncc && margin >= params.ncc_margin {
                        if let Some(p_cur) = triangulate_pixel(nu, nv, disp, rect, params) {
                            let sa = (sp.p[2] * sp.p[2] / (rect.k.fx * rect.baseline)) * 0.3;
                            let sb = (p_cur[2] * p_cur[2] / (rect.k.fx * rect.baseline)) * 0.3;
                            pairs_a.push(sp.p); // previous cam frame
                            pairs_b.push(p_cur); // current cam frame
                            pair_sigma.push((sa * sa + sb * sb).sqrt());
                            matched += 1;
                        }
                    }
                }
            }
            // 4. robust rigid registration (sigma-scaled per-pair tolerance)
            if pairs_a.len() >= params.min_inliers + 5 {
                if let Some(fit) = ransac_rigid_sigma(&pairs_a, &pairs_b, &pair_sigma, params.ransac_thresh, params.ransac_iters, rng) {
                    let rot_ok = fit.r.angle() <= params.max_rotation;
                    if fit.n_inliers >= params.min_inliers && rot_ok {
                        // delta maps prev-cam -> cur-cam
                        delta = Se3::from_parts(fit.r, fit.t);
                        inliers = fit.n_inliers;
                        ok = true;
                        // pose update: T_wc_cur = T_wc_prev * delta^-1
                        self.t_world_cam = self.t_world_cam.compose(&delta.inverse());
                        self.path_len += delta.t.norm();
                        // honest covariance growth from the fit quality
                        let n = fit.n_inliers.max(1) as f64;
                        let rms = fit.rms.max(0.005);
                        let sig_t = (rms * 2.0).max(0.004) / n.sqrt();
                        let sig_r = (rms / 2.0).max(0.001) / n.sqrt();
                        let mut dq = Matrix6::zeros();
                        for i in 0..3 {
                            dq[(i, i)] = sig_r * sig_r;
                            dq[(i + 3, i + 3)] = sig_t * sig_t;
                        }
                        // rotate the per-frame delta covariance into the world
                        // accumulated frame and add it
                        let ad = self.t_world_cam.adjoint();
                        let dw = ad * dq * ad.transpose();
                        for i in 0..6 {
                            for j in 0..6 {
                                self.cov[(i, j)] += dw[(i, j)];
                            }
                        }
                    }
                }
            }
        }
        if !ok {
            // VO failed this frame: inflate covariance honestly (2 cm / frame)
            for i in 3..6 {
                self.cov[(i, i)] += 0.02 * 0.02;
            }
            for i in 0..3 {
                self.cov[(i, i)] += 0.002 * 0.002;
            }
        }

        self.last_inliers = inliers;
        self.last_matched = matched;
        // 5. store current frame state
        self.prev_gray = Some(left_rect.clone());
        self.prev_pts = cur_pts;

        VoResult {
            t_world_cam: self.t_world_cam,
            delta,
            inliers,
            matched,
            features: feats.len(),
            ok,
        }
    }

    /// Cumulative drift statistics (honest reporting).
    pub fn pose_sigma(&self) -> (f64, f64) {
        let sig_rot = (self.cov[(0, 0)] + self.cov[(1, 1)] + self.cov[(2, 2)]).sqrt();
        let sig_trans = (self.cov[(3, 3)] + self.cov[(4, 4)] + self.cov[(5, 5)]).sqrt();
        (sig_rot, sig_trans)
    }
}

/// Triangulate a pixel via the disparity map (strong/weak quality only).
fn triangulate_pixel(
    u: f64,
    v: f64,
    disp: &DisparityMap,
    rect: &RectifiedRig,
    params: &VoParams,
) -> Option<Vector3<f64>> {
    let xi = u.round() as i64;
    let yi = v.round() as i64;
    if xi < 0 || yi < 0 || xi >= disp.width as i64 || yi >= disp.height as i64 {
        return None;
    }
    let i = yi as usize * disp.width + xi as usize;
    let d = disp.disp[i];
    if !d.is_finite() || d < 2.0 {
        return None;
    }
    match disp.quality[i] {
        q if q == Quality::Strong as u8 || q == Quality::Weak as u8 => {}
        _ => return None,
    }
    let sd = disp.sigma[i];
    let z = rect.k.fx * rect.baseline / d as f64;
    let sz = z * z / (rect.k.fx * rect.baseline) * sd.max(0.15) as f64;
    if sz > params.max_sigma_z || z > 15.0 || z < 0.4 {
        return None;
    }
    Some(Vector3::new(
        (u - rect.k.cx) * z / rect.k.fx,
        -(v - rect.k.cy) * z / rect.k.fy,
        z,
    ))
}

/// Find the best NCC match of the patch at (u0, v0) in `from` within a
/// window of the same location in `to`. Returns (u, v, score).
fn ncc_match(from: &GrayImage, to: &GrayImage, u0: f64, v0: f64, params: &VoParams) -> Option<(f64, f64, f64, f64)> {
    let pr = params.patch_radius as i64;
    let sr = params.search_radius as i64;
    let (w, h) = (to.width() as i64, to.height() as i64);
    let cx = u0.round() as i64;
    let cy = v0.round() as i64;
    if cx - pr - sr < 0 || cy - pr - sr < 0 || cx + pr + sr >= w || cy + pr + sr >= h {
        return None;
    }
    // reference patch statistics
    let mut ref_mean = 0.0;
    let n = ((2 * pr + 1) * (2 * pr + 1)) as f64;
    let mut ref_px: Vec<f64> = Vec::with_capacity(n as usize);
    for dy in -pr..=pr {
        for dx in -pr..=pr {
            let v = from.get_pixel((cx + dx) as u32, (cy + dy) as u32)[0] as f64;
            ref_px.push(v);
            ref_mean += v;
        }
    }
    ref_mean /= n;
    let mut ref_var = 0.0;
    for v in &ref_px {
        ref_var += (v - ref_mean) * (v - ref_mean);
    }
    if ref_var < 1.0 {
        return None; // textureless patch: useless for tracking
    }
    let ref_std = ref_var.sqrt();

    let mut best = f64::NEG_INFINITY;
    let mut second = f64::NEG_INFINITY;
    let mut best_ox = 0i64;
    let mut best_oy = 0i64;
    for oy in -sr..=sr {
        for ox in -sr..=sr {
            // candidate patch mean
            let mut cand_mean = 0.0;
            for dy in -pr..=pr {
                for dx in -pr..=pr {
                    cand_mean += to.get_pixel((cx + ox + dx) as u32, (cy + oy + dy) as u32)[0] as f64;
                }
            }
            cand_mean /= n;
            let mut cov = 0.0;
            let mut var = 0.0;
            let mut k = 0usize;
            for dy in -pr..=pr {
                for dx in -pr..=pr {
                    let cv = to.get_pixel((cx + ox + dx) as u32, (cy + oy + dy) as u32)[0] as f64 - cand_mean;
                    cov += cv * (ref_px[k] - ref_mean);
                    var += cv * cv;
                    k += 1;
                }
            }
            if var < 1.0 {
                continue;
            }
            let ncc = cov / (ref_std * var.sqrt());
            if ncc > best {
                second = best;
                best = ncc;
                best_ox = ox;
                best_oy = oy;
            } else if ncc > second {
                second = ncc;
            }
        }
    }
    let margin = best - second.max(0.0);
    Some(((cx + best_ox) as f64, (cy + best_oy) as f64, best, margin))
}

/// Weighted Kabsch: pairs weighted by w_i (use 1/sigma^2). Near, accurate
/// points dominate the translation; far points still constrain rotation.
fn kabsch_weighted(a: &[Vector3<f64>], b: &[Vector3<f64>], w: &[f64]) -> Option<(nalgebra::Rotation3<f64>, Vector3<f64>)> {
    let n = a.len();
    if n < 3 || b.len() != n || w.len() != n {
        return None;
    }
    let wsum: f64 = w.iter().sum();
    if wsum <= 1e-12 {
        return None;
    }
    let mut ca = Vector3::zeros();
    let mut cb = Vector3::zeros();
    for i in 0..n {
        ca += a[i].scale(w[i]);
        cb += b[i].scale(w[i]);
    }
    ca /= wsum;
    cb /= wsum;
    let mut h = nalgebra::Matrix3::zeros();
    for i in 0..n {
        let da = a[i] - ca;
        let db = b[i] - cb;
        h += da.scale(w[i]) * db.transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    let mut r = vt.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        let d = nalgebra::Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0));
        r = vt.transpose() * d * u.transpose();
    }
    let rot = nalgebra::Rotation3::from_matrix(&r);
    let t = cb - rot.transform_vector(&ca);
    Some((rot, t))
}

/// RANSAC rigid registration with per-pair sigma-scaled inlier tolerance:
/// inlier if |R a + t - b| <= max(base_thresh, 2.5 * sigma_pair).
fn ransac_rigid_sigma(
    a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    sigma: &[f64],
    base_thresh: f64,
    iters: usize,
    rng: &mut GaussRng,
) -> Option<crate::core::geo::RigidFit> {
    let n = a.len();
    if n < 6 {
        return kabsch(a, b).map(|(r, t)| crate::core::geo::RigidFit { r, t, n_inliers: n, rms: 0.0 });
    }
    let tol = |i: usize| base_thresh.max((2.5 * sigma[i]).min(0.35));
    let mut best: Option<(usize, Vec<usize>)> = None;
    for _ in 0..iters {
        let i0 = rng.uniform_usize(n);
        let mut i1 = rng.uniform_usize(n);
        while i1 == i0 {
            i1 = rng.uniform_usize(n);
        }
        let mut i2 = rng.uniform_usize(n);
        while i2 == i0 || i2 == i1 {
            i2 = rng.uniform_usize(n);
        }
        let idx = [i0, i1, i2];
        let sa: Vec<Vector3<f64>> = idx.iter().map(|&i| a[i]).collect();
        let sb: Vec<Vector3<f64>> = idx.iter().map(|&i| b[i]).collect();
        let (r, t) = match kabsch(&sa, &sb) {
            Some(x) => x,
            None => continue,
        };
        let inliers: Vec<usize> = (0..n)
            .filter(|&i| (r.transform_vector(&a[i]) + t - b[i]).norm() <= tol(i))
            .collect();
        let better = best.as_ref().map(|(c, _)| inliers.len() > *c).unwrap_or(true);
        if inliers.len() >= 6 && better {
            best = Some((inliers.len(), inliers));
        }
    }
    let (_, inliers) = best?;
    let ia: Vec<Vector3<f64>> = inliers.iter().map(|&i| a[i]).collect();
    let ib: Vec<Vector3<f64>> = inliers.iter().map(|&i| b[i]).collect();
    let iw: Vec<f64> = inliers.iter().map(|&i| 1.0 / (sigma[i] * sigma[i] + 1e-6)).collect();
    let (r, t) = kabsch_weighted(&ia, &ib, &iw)?;
    let rms = (0..n)
        .map(|i| (r.transform_vector(&a[i]) + t - b[i]).norm_squared())
        .sum::<f64>()
        / n.max(1) as f64;
    Some(crate::core::geo::RigidFit { r, t, n_inliers: inliers.len(), rms: rms.sqrt() })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ncc_finds_shifted_patch() {
        // random texture with a distinctive patch, shifted by (5, 3)
        let mut rng = crate::core::rng::GaussRng::new(31);
        let (w, h) = (120usize, 90usize);
        let mut from = GrayImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let v = (rng.uniform() * 255.0) as u8;
                from.put_pixel(x as u32, y as u32, image::Luma([v]));
            }
        }
        let mut to = GrayImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                let sx = x as i64 - 5;
                let sy = y as i64 - 3;
                let v = if sx >= 0 && sy >= 0 && sx < w as i64 && sy < h as i64 {
                    from.get_pixel(sx as u32, sy as u32)[0]
                } else {
                    0
                };
                to.put_pixel(x as u32, y as u32, image::Luma([v]));
            }
        }
        // patch in `from` at (50, 45) appears in `to` at (55, 48)
        let params = VoParams::default();
        let (mu, mv, score, margin) = ncc_match(&from, &to, 50.0, 45.0, &params).unwrap();
        assert!(score > 0.9, "ncc score {score:.3}");
        assert!(margin > 0.3, "ncc margin {margin:.3}");
        assert!((mu - 55.0).abs() < 1.5 && (mv - 48.0).abs() < 1.5, "match at ({mu},{mv})");
    }

    #[test]
    fn ncc_rejects_textureless() {
        let img = GrayImage::from_pixel(100, 100, image::Luma([77u8]));
        let params = VoParams::default();
        assert!(ncc_match(&img, &img, 50.0, 50.0, &params).is_none());
    }

    #[test]
    fn vo_covariance_grows_on_failures() {
        // build a disparity-like map is complex; here we only check the
        // covariance bookkeeping through repeated identity updates
        let mut vo = VoState::new();
        let k = crate::camera::model::Intrinsics::new(700.0, 700.0, 319.5, 239.5, 320, 240);
        let rig = crate::camera::model::StereoRig::canonical(
            k,
            Se3::identity(),
            0.16,
        );
        let rect = rig.rectify().unwrap();
        // blank frame: no features -> VO fails -> cov grows
        let img = GrayImage::from_pixel(320, 240, image::Luma([128u8]));
        let disp = DisparityMap {
            width: 320,
            height: 240,
            disp: vec![f32::NAN; 320 * 240],
            sigma: vec![f32::NAN; 320 * 240],
            quality: vec![0; 320 * 240],
            coarse: vec![false; 320 * 240],
            disp_min: 0.0,
            stats: Default::default(),
        };
        let mut rng = GaussRng::new(1);
        let s0 = vo.pose_sigma().1;
        for _ in 0..5 {
            vo.update(&img, &disp, &rect, &VoParams::default(), &mut rng);
        }
        let s1 = vo.pose_sigma().1;
        assert!(s1 > s0, "sigma {s1} must grow on VO failure (was {s0})");
        // wrap_pi sanity used by callers
        assert!((wrap_pi(3.5) + 2.78).abs() < 0.01);
    }
}



