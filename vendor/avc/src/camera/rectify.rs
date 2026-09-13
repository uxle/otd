//! Calibrated stereo rectification (Fusiello-style) + bilinear remapping.
//!
//! Both cameras keep their centres; a shared new orientation `R_n` is chosen
//! so the rectified x-axis aligns with the baseline. The induced image
//! homographies `H_i = K_n R_n^T R_i K_i^-1` are **exact** for calibrated
//! cameras (every 3D point is remapped consistently, not just the plane at
//! infinity - that caveat only applies to uncalibrated rectification).
//!
//! After rectification: same intrinsics for both cameras, canonical geometry,
//! `disparity = u_L - u_R`, `Z = f B / disparity`.

use image::GrayImage;
use nalgebra::{Matrix3, Rotation3, Vector3};

use crate::camera::model::{Camera, Intrinsics, StereoRig};
use crate::core::se3::Se3;

/// Rectification product: everything needed to run the stereo matcher and
/// lift disparities back to 3D.
#[derive(Debug, Clone)]
pub struct RectifiedRig {
    /// Shared intrinsics of the rectified pair.
    pub k: Intrinsics,
    /// Baseline in metres.
    pub baseline: f64,
    /// cam-to-world pose of the rectified LEFT camera.
    pub pose_left: Se3,
    /// cam-to-world pose of the rectified RIGHT camera.
    pub pose_right: Se3,
    /// Homography: original left pixel -> rectified left pixel.
    pub h_left: Matrix3<f64>,
    /// Homography: original right pixel -> rectified right pixel.
    pub h_right: Matrix3<f64>,
}

impl StereoRig {
    /// Compute the rectification. Fails only for a degenerate (zero) baseline.
    pub fn rectify(&self) -> Option<RectifiedRig> {
        let cl = self.left.pose.t;
        let cr = self.right.pose.t;
        let base_vec = cr - cl;
        let baseline = base_vec.norm();
        if baseline < 1e-9 {
            return None;
        }
        let x_n = base_vec / baseline;

        // choose y so that z stays as close as possible to the left camera's
        // forward direction
        let z_l = self.left.forward();
        let mut y_n = z_l.cross(&x_n);
        if y_n.norm() < 1e-9 {
            // camera looks along the baseline: fall back to the world up axis
            y_n = Vector3::new(0.0, 1.0, 0.0).cross(&x_n);
            if y_n.norm() < 1e-9 {
                y_n = Vector3::new(0.0, 0.0, 1.0).cross(&x_n);
            }
        }
        let y_n = y_n.normalize();
        let z_n = x_n.cross(&y_n);

        let r_n = Rotation3::from_matrix(&Matrix3::from_columns(&[x_n, y_n, z_n]));

        // new shared intrinsics: square pixels, focal = max of both (avoid
        // aliasing when shrinking), principal point at the image centre
        let kl = &self.left.k;
        let kr = &self.right.k;
        let f = kl.fx.max(kl.fy).max(kr.fx).max(kr.fy);
        let k_n = Intrinsics::new(f, f, kl.cx, kl.cy, kl.width, kl.height);

        // H_i = K_n' R_n^T R_i K_i'^{-1}   (y-up pixel convention)
        let h_left = k_n.matrix_yup() * r_n.matrix().transpose() * self.left.pose.r.matrix() * kl.inverse_matrix_yup();
        let h_right = k_n.matrix_yup() * r_n.matrix().transpose() * self.right.pose.r.matrix() * kr.inverse_matrix_yup();

        Some(RectifiedRig {
            k: k_n,
            baseline,
            pose_left: Se3::from_parts(r_n, cl),
            pose_right: Se3::from_parts(r_n, cr),
            h_left,
            h_right,
        })
    }
}

impl RectifiedRig {
    /// True when the rig is already canonical (remap can be skipped).
    pub fn is_canonical(&self) -> bool {
        let eye = Matrix3::identity();
        (self.h_left - &eye).abs().max() < 1e-9 && (self.h_right - &eye).abs().max() < 1e-9
    }

    /// Disparity (px) -> depth (m) in the rectified left camera frame.
    #[inline]
    pub fn depth_of_disparity(&self, d: f64) -> f64 {
        if d <= 1e-6 {
            f64::INFINITY
        } else {
            self.k.fx * self.baseline / d
        }
    }

    /// Disparity uncertainty -> depth uncertainty.
    #[inline]
    pub fn depth_sigma_of_disparity_sigma(&self, z: f64, sigma_d: f64) -> f64 {
        z * z / (self.k.fx * self.baseline) * sigma_d
    }

    /// Triangulate a rectified pixel pair to a point in the RECTIFIED left
    /// camera frame (y-up convention: image v grows downward).
    pub fn triangulate_cam(&self, u_l: f64, v: f64, u_r: f64) -> Vector3<f64> {
        let d = u_l - u_r;
        let z = self.depth_of_disparity(d);
        Vector3::new(
            (u_l - self.k.cx) * z / self.k.fx,
            -(v - self.k.cy) * z / self.k.fy,
            z,
        )
    }
}

/// Apply a homography to a pixel.
pub fn apply_h(h: &Matrix3<f64>, u: f64, v: f64) -> (f64, f64) {
    let p = h * Vector3::new(u, v, 1.0);
    if p[2].abs() < 1e-12 {
        return (f64::NAN, f64::NAN);
    }
    (p[0] / p[2], p[1] / p[2])
}

/// Warp `img` by the homography `h` (source -> target). Target pixel (u, v)
/// samples the source at h^-1 (u, v) bilinearly; out-of-bounds becomes 0.
pub fn remap_bilinear(img: &GrayImage, h: &Matrix3<f64>) -> GrayImage {
    let (w, hgt) = img.dimensions();
    let h_inv = h.try_inverse().unwrap_or_else(Matrix3::identity);
    let src = img;
    let mut out = GrayImage::new(w, hgt);
    for y in 0..hgt {
        for x in 0..w {
            let (sx, sy) = apply_h(&h_inv, x as f64, y as f64);
            if sx.is_finite() && sy.is_finite() && sx >= 0.0 && sy >= 0.0 && sx <= (w - 1) as f64 && sy <= (hgt - 1) as f64 {
                let x0 = sx.floor() as i64;
                let y0 = sy.floor() as i64;
                let x1 = (x0 + 1).min((w - 1) as i64);
                let y1 = (y0 + 1).min((hgt - 1) as i64);
                let fx = sx - x0 as f64;
                let fy = sy - y0 as f64;
                let v00 = src.get_pixel(x0.max(0) as u32, y0.max(0) as u32)[0] as f64;
                let v10 = src.get_pixel(x1.max(0) as u32, y0.max(0) as u32)[0] as f64;
                let v01 = src.get_pixel(x0.max(0) as u32, y1.max(0) as u32)[0] as f64;
                let v11 = src.get_pixel(x1.max(0) as u32, y1.max(0) as u32)[0] as f64;
                let val = v00 * (1.0 - fx) * (1.0 - fy)
                    + v10 * fx * (1.0 - fy)
                    + v01 * (1.0 - fx) * fy
                    + v11 * fx * fy;
                out.put_pixel(x, y, image::Luma([val.round().clamp(0.0, 255.0) as u8]));
            }
        }
    }
    out
}

/// Rectify an image pair (skips the warp when the rig is already canonical).
pub fn rectify_pair(rect: &RectifiedRig, left: &GrayImage, right: &GrayImage) -> (GrayImage, GrayImage) {
    if rect.is_canonical() {
        return (left.clone(), right.clone());
    }
    (remap_bilinear(left, &rect.h_left), remap_bilinear(right, &rect.h_right))
}

/// Convenience: build a Camera from intrinsics + pose.
pub fn camera(k: Intrinsics, pose: Se3) -> Camera {
    Camera::new(k, pose)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::model::look_at;
    use crate::core::rng::GaussRng;

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!((a - b).abs() <= tol, "{msg}: {a} vs {b}");
    }

    #[test]
    fn canonical_rig_has_identity_homographies() {
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let pose = Se3::from_parts(
            look_at(Vector3::new(0.0, 1.6, 0.0), Vector3::new(0.0, 1.6, 4.0), Vector3::y()),
            Vector3::new(0.0, 1.6, 0.0),
        );
        let rig = StereoRig::canonical(k, pose, 0.16);
        let rect = rig.rectify().unwrap();
        assert!(rect.is_canonical(), "canonical rig must rectify to identity");
        assert_close(rect.baseline, 0.16, 1e-12, "baseline");
    }

    /// The core rectification guarantee: after remapping, corresponding
    /// pixels share the same row (epipolar lines are horizontal), and the
    /// disparity recovers the rectified-frame depth.
    #[test]
    fn rectified_epipolar_geometry() {
        let k = Intrinsics::new(600.0, 600.0, 319.5, 239.5, 640, 480);
        // left camera with a pitch/roll, right camera translated + rotated
        let rl = look_at(Vector3::new(0.1, 1.6, -0.05), Vector3::new(0.0, 1.5, 4.0), Vector3::y());
        let left = Camera::new(k, Se3::from_parts(rl, Vector3::new(0.1, 1.6, -0.05)));
        // right: rotated by a few degrees about y and x, centre offset
        let rr = look_at(Vector3::new(0.28, 1.62, 0.0), Vector3::new(0.0, 1.5, 4.0), Vector3::y());
        let right = Camera::new(k, Se3::from_parts(rr, Vector3::new(0.28, 1.62, 0.0)));
        let rig = StereoRig { left, right };
        let rect = rig.rectify().unwrap();

        let mut rng = GaussRng::new(4);
        let mut max_dy = 0.0f64;
        let mut max_dz_err = 0.0f64;
        let mut checked = 0;
        while checked < 200 {
            let p = Vector3::new(rng.gauss(0.0, 1.0), 1.3 + rng.gauss(0.0, 0.4), 2.5 + rng.gauss(0.0, 0.8));
            let (ul, vl) = match rig.left.project(&p) { Some(x) => x, None => continue };
            let (ur, vr) = match rig.right.project(&p) { Some(x) => x, None => continue };
            let (ul_r, vl_r) = apply_h(&rect.h_left, ul, vl);
            let (ur_r, vr_r) = apply_h(&rect.h_right, ur, vr);
            max_dy = max_dy.max((vl_r - vr_r).abs());
            // depth from disparity vs depth in the rectified left frame
            let d = ul_r - ur_r;
            if d > 1.0 {
                let z_est = rect.depth_of_disparity(d);
                let p_rect = rect.pose_left.inverse().transform_point(&p);
                max_dz_err = max_dz_err.max((z_est - p_rect.z).abs() / p_rect.z);
            }
            checked += 1;
        }
        assert!(max_dy < 0.35, "epipolar row mismatch {max_dy} px");
        assert!(max_dz_err < 0.02, "rectified depth error {max_dz_err}");
    }

    #[test]
    fn remap_translates_image() {
        let mut img = GrayImage::new(64, 32);
        for y in 0..32 {
            for x in 0..64 {
                let v = if (10..14).contains(&x) { 200u8 } else { 30u8 };
                img.put_pixel(x, y, image::Luma([v]));
            }
        }
        // target pixel (u,v) samples source at (u-2, v): stripe moves +2
        let h = Matrix3::new(1.0, 0.0, 2.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0);
        let out = remap_bilinear(&img, &h);
        assert_eq!(out.get_pixel(12, 10)[0], 200, "stripe centre moves to 12");
        assert_eq!(out.get_pixel(8, 10)[0], 30, "background before stripe");
        assert_eq!(out.get_pixel(20, 10)[0], 30, "background after stripe");
    }
}
