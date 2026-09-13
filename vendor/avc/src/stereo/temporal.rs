//! Motion-compensated temporal depth fusion (v4).
//!
//! Static-scene pixels are re-observed every frame; averaging them through
//! the VO pose kills noise without touching moving objects:
//!
//! - the previous fused map is kept **in the previous camera frame** with its
//!   per-pixel depth sigma and the previous cam-to-world pose;
//! - each valid current pixel is backprojected, transformed into the previous
//!   camera (through world), and bilinearly sampled there;
//! - a 3-sigma gate rejects motion / disocclusion / VO drift automatically -
//!   moving objects simply keep their fresh single-frame measurement;
//! - survivors are fused with exponential forgetting (0.6 new / 0.4 old,
//!   effective window ~2.5 frames); the fused sigma is floored at 0.6x the
//!   single-frame sigma so confidence never over-collapses;
//! - the fused depth is written back into the DisparityMap, so lifting,
//!   detection and measurement all inherit the improved sigma.
//!
//! Honesty: the fusion statistics (n_fused / n_measured) are reported by the
//! engine; the sigma model is EMA-variance (independent-noise assumption,
//! conservative) and is re-measured by the validation harness.

use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use nalgebra::Vector3;

use crate::camera::rectify::RectifiedRig;
use crate::core::se3::Se3;
use crate::stereo::map::DisparityMap;

/// Exponential forgetting weight of the previous estimate (0.4 -> ~2.5-frame
/// effective window; long enough to average noise, short enough to track
/// slow scene changes).
const W_PREV: f64 = 0.4;
/// Floor on the fused sigma relative to the single-frame sigma. The EMA
/// variance assumes INDEPENDENT noise, but real depth errors carry
/// systematic components (sub-pixel census bias, visible-surface bias,
/// half-res quantisation) that do NOT average away - the 0.75 floor keeps
/// the fused sigma honest (calibrated on the validation scene: below 0.75
/// the 2-sigma coverage and the measurement-consistency checks degrade).
const SIGMA_FLOOR_FRAC: f64 = 0.75;

#[derive(Debug, Clone)]
pub struct TemporalFuser {
    w: usize,
    h: usize,
    /// depth in the previous camera frame (m); NaN = unknown
    z_prev: Vec<f32>,
    /// depth sigma in the previous camera frame (m)
    sig_prev: Vec<f32>,
    /// cam-to-world pose of the previous rectified-left camera
    pose_prev: Option<Se3>,
    /// last-frame fusion statistics (honest reporting)
    pub n_fused: usize,
    pub n_measured: usize,
}

impl Default for TemporalFuser {
    fn default() -> Self {
        TemporalFuser {
            w: 0,
            h: 0,
            z_prev: Vec::new(),
            sig_prev: Vec::new(),
            pose_prev: None,
            n_fused: 0,
            n_measured: 0,
        }
    }
}

impl TemporalFuser {
    pub fn new() -> Self {
        Self::default()
    }

    /// Fraction of measured pixels that passed the gate and were fused in
    /// the last frame (0 = none; ~1 on a static scene with good VO).
    pub fn fused_fraction(&self) -> f64 {
        if self.n_measured > 0 {
            self.n_fused as f64 / self.n_measured as f64
        } else {
            0.0
        }
    }

    /// Fuse the current map with the previous (motion-compensated) estimate,
    /// in place. `rect`/`pose_now` describe the CURRENT rectified-left camera.
    pub fn fuse(&mut self, disp: &mut DisparityMap, rect: &RectifiedRig, pose_now: &Se3) {
        let f = rect.k.fx;
        let fy = rect.k.fy;
        let cx = rect.k.cx;
        let cy = rect.k.cy;
        let fb = f * rect.baseline;
        let (w, h) = (disp.width, disp.height);
        if self.w != w || self.h != h {
            self.z_prev = vec![f32::NAN; w * h];
            self.sig_prev = vec![f32::NAN; w * h];
            self.w = w;
            self.h = h;
            self.pose_prev = None;
        }
        let Some(&pose_prev) = self.pose_prev.as_ref() else {
            // first frame: just store
            self.store(disp, fb);
            self.pose_prev = Some(*pose_now);
            self.n_fused = 0;
            self.n_measured = 0;
            return;
        };
        let inv_prev = pose_prev.inverse();
        let z_prev: &Vec<f32> = &self.z_prev;
        let sig_prev: &Vec<f32> = &self.sig_prev;
        let n_fused = AtomicUsize::new(0);
        let n_meas = AtomicUsize::new(0);

        disp.disp
            .par_chunks_mut(w)
            .zip(disp.sigma.par_chunks_mut(w))
            .zip(disp.quality[..].par_chunks(w))
            .enumerate()
            .for_each(|(y, ((drow, srow), qrow))| {
                for x in 0..w {
                    if qrow[x] < 2 {
                        continue; // Strong/Weak only - never fuse fills
                    }
                    let d = drow[x];
                    if !d.is_finite() || d < 1.0 {
                        continue;
                    }
                    let z = fb / d as f64;
                    let sd = if srow[x].is_finite() && srow[x] > 0.0 {
                        srow[x] as f64
                    } else {
                        0.5
                    };
                    let sigma_z = z * z / fb * sd;
                    n_meas.fetch_add(1, Ordering::Relaxed);
                    // backproject (rect frame is y-up: v grows downward)
                    let pc = Vector3::new(
                        (x as f64 - cx) * z / f,
                        -(y as f64 - cy) * z / fy,
                        z,
                    );
                    let pw = pose_now.transform_point(&pc);
                    let pp = inv_prev.transform_point(&pw);
                    if !pp.z.is_finite() || pp.z <= 0.1 {
                        continue;
                    }
                    let u = f * pp.x / pp.z + cx;
                    let v = cy - fy * pp.y / pp.z;
                    if !(0.0..=(w - 1) as f64).contains(&u)
                        || !(0.0..=(h - 1) as f64).contains(&v)
                    {
                        continue;
                    }
                    // bilinear sample of the previous depth + sigma
                    let x0 = u.floor() as usize;
                    let y0 = v.floor() as usize;
                    let x1 = (x0 + 1).min(w - 1);
                    let y1 = (y0 + 1).min(h - 1);
                    let fx = (u - x0 as f64) as f32;
                    let fy = (v - y0 as f64) as f32;
                    let z00 = z_prev[y0 * w + x0];
                    let z10 = z_prev[y0 * w + x1];
                    let z01 = z_prev[y1 * w + x0];
                    let z11 = z_prev[y1 * w + x1];
                    if !(z00.is_finite() && z10.is_finite() && z01.is_finite() && z11.is_finite()) {
                        continue;
                    }
                    let s00 = sig_prev[y0 * w + x0];
                    let s10 = sig_prev[y0 * w + x1];
                    let s01 = sig_prev[y1 * w + x0];
                    let s11 = sig_prev[y1 * w + x1];
                    if !(s00.is_finite() && s10.is_finite() && s01.is_finite() && s11.is_finite()) {
                        continue;
                    }
                    let bilin = |a: f32, b: f32, c: f32, dd: f32| -> f32 {
                        a * (1.0 - fx) * (1.0 - fy)
                            + b * fx * (1.0 - fy)
                            + c * (1.0 - fx) * fy
                            + dd * fx * fy
                    };
                    let zp = bilin(z00, z10, z01, z11) as f64;
                    let sp = bilin(s00, s10, s01, s11) as f64;
                    if sp <= 0.0 {
                        continue;
                    }
                    // 3-sigma gate: motion / disocclusion / VO drift fall out
                    let gate = 3.0 * sigma_z.max(sp).max(0.02);
                    if (zp - z).abs() > gate {
                        continue;
                    }
                    // exponential fusion
                    let w_new = 1.0 - W_PREV;
                    let zf = w_new * z + W_PREV * zp;
                    let var = w_new * w_new * sigma_z * sigma_z + W_PREV * W_PREV * sp * sp;
                    let sf = var.sqrt().max(SIGMA_FLOOR_FRAC * sigma_z);
                    drow[x] = (fb / zf) as f32;
                    srow[x] = (sf * fb / (zf * zf)) as f32;
                    n_fused.fetch_add(1, Ordering::Relaxed);
                }
            });

        self.n_fused = n_fused.load(Ordering::Relaxed);
        self.n_measured = n_meas.load(Ordering::Relaxed);
        self.store(disp, fb);
        self.pose_prev = Some(*pose_now);
    }

    /// Store the (post-fusion) map as the previous estimate. Only measured
    /// pixels are stored; fills and invalid pixels become unknown.
    fn store(&mut self, disp: &DisparityMap, fb: f64) {
        let n = self.w * self.h;
        for i in 0..n {
            let q = disp.quality[i];
            let d = disp.disp[i];
            if q >= 2 && d.is_finite() && d >= 1.0 {
                let z = fb / d as f64;
                let sd = disp.sigma[i];
                self.z_prev[i] = z as f32;
                self.sig_prev[i] = if sd.is_finite() && sd > 0.0 {
                    (z * z / fb * sd as f64) as f32
                } else {
                    (z * z / fb * 0.5) as f32
                };
            } else {
                self.z_prev[i] = f32::NAN;
                self.sig_prev[i] = f32::NAN;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::model::{Intrinsics, StereoRig};
    use crate::stereo::map::Quality;

    fn rig64() -> RectifiedRig {
        let k = Intrinsics::new(100.0, 100.0, 31.5, 23.5, 64, 48);
        let rig = StereoRig::canonical(k, Se3::identity(), 0.16);
        rig.rectify().unwrap()
    }

    fn make_map(w: usize, h: usize, d: f32, sigma: f32) -> DisparityMap {
        DisparityMap {
            width: w,
            height: h,
            disp: vec![d; w * h],
            sigma: vec![sigma; w * h],
            quality: vec![Quality::Strong as u8; w * h],
            coarse: vec![false; w * h],
            disp_min: 0.0,
            stats: Default::default(),
        }
    }

    #[test]
    fn static_scene_fuses_and_shrinks_sigma() {
        let rect = rig64();
        let mut f = TemporalFuser::new();
        let pose = Se3::identity();
        let mut m1 = make_map(64, 48, 28.0, 0.30);
        f.fuse(&mut m1, &rect, &pose);
        assert_eq!(f.n_fused, 0, "first frame only stores");
        // identical second frame: every pixel fuses, z is unchanged
        let mut m2 = make_map(64, 48, 28.0, 0.30);
        f.fuse(&mut m2, &rect, &pose);
        assert!(f.n_fused > 64 * 48 * 9 / 10, "most pixels fused: {}", f.n_fused);
        let i = 20 * 64 + 30;
        assert!((m2.disp[i] - 28.0).abs() < 0.05, "identical measurements keep z: {}", m2.disp[i]);
        assert!(m2.sigma[i] < 0.30, "sigma shrinks: {}", m2.sigma[i]);
        assert!(m2.sigma[i] > 0.30 * 0.5, "sigma floor respected: {}", m2.sigma[i]);
    }

    #[test]
    fn moving_depth_is_not_fused() {
        // fb = 100*0.16 = 16; d=8 -> z=2 m; d=16 -> z=1 m (1 m jump: gate
        // 3*sigma rejects, current measurement kept untouched)
        let rect = rig64();
        let mut f = TemporalFuser::new();
        let pose = Se3::identity();
        let mut m1 = make_map(64, 48, 8.0, 0.30);
        f.fuse(&mut m1, &rect, &pose);
        let mut m2 = make_map(64, 48, 16.0, 0.30);
        f.fuse(&mut m2, &rect, &pose);
        assert_eq!(f.n_fused, 0, "moving surface must not fuse");
        let i = 20 * 64 + 30;
        assert!((m2.disp[i] - 16.0).abs() < 1e-6, "current disparity kept");
        assert!((m2.sigma[i] - 0.30).abs() < 1e-6, "current sigma kept");
    }

    #[test]
    fn camera_translation_fuses_consistent_world() {
        // frame 1: identity pose, plane at z=1 m (d=16 with fb=16).
        // frame 2: camera translated +x by 0.1 m (10 px at f=100); each
        // current pixel is built to observe the SAME world point as some
        // previous pixel -> the reprojected sample must agree and fuse.
        let rect = rig64();
        let fb = 16.0f64;
        let f = rect.k.fx;
        let cx = rect.k.cx;
        let cy = rect.k.cy;
        let mut fuser = TemporalFuser::new();
        let pose1 = Se3::identity();
        let mut m1 = make_map(64, 48, 16.0, 0.30);
        fuser.fuse(&mut m1, &rect, &pose1);

        let pose2 = Se3::from_parts(nalgebra::Rotation3::identity(), Vector3::new(0.1, 0.0, 0.0));
        let mut d2 = vec![f32::NAN; 64 * 48];
        let mut q2 = vec![0u8; 64 * 48];
        let mut s2 = vec![0.30f32; 64 * 48];
        for v in 0..48usize {
            for u in 0..64usize {
                // world point seen by prev pixel (u, v) at z = 1
                let pw = Vector3::new((u as f64 - cx) * 1.0 / f, -(v as f64 - cy) * 1.0 / f, 1.0);
                let pc2 = pose2.inverse().transform_point(&pw);
                if pc2.z <= 0.1 {
                    continue;
                }
                let u2 = f * pc2.x / pc2.z + cx;
                let v2 = cy - rect.k.fy * pc2.y / pc2.z;
                if !(0.0..63.0).contains(&u2) || !(0.0..47.0).contains(&v2) {
                    continue;
                }
                let iu = u2.round() as usize;
                let iv = v2.round() as usize;
                d2[iv * 64 + iu] = (fb / pc2.z) as f32;
                q2[iv * 64 + iu] = Quality::Strong as u8;
            }
        }
        let mut m2 = DisparityMap {
            width: 64,
            height: 48,
            disp: d2,
            sigma: s2,
            quality: q2,
            coarse: vec![false; 64 * 48],
            disp_min: 0.0,
            stats: Default::default(),
        };
        fuser.fuse(&mut m2, &rect, &pose2);
        let n_valid = m2.quality.iter().filter(|&&q| q >= 2).count();
        assert!(n_valid > 0);
        assert!(
            fuser.n_fused as f64 > n_valid as f64 * 0.5,
            "consistent world points must fuse: {}/{}",
            fuser.n_fused,
            n_valid
        );
        // a fused pixel keeps z ~ 1 m -> d ~ 16
        let i = 20 * 64 + 30;
        if m2.disp[i].is_finite() {
            assert!((m2.disp[i] - 16.0).abs() < 1.0, "fused depth stays ~1 m: d={}", m2.disp[i]);
        }
    }
}
