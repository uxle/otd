//! Disparity -> world 3D lifting with per-point uncertainty.
//!
//! Each point carries a per-axis sigma propagated from the disparity sigma
//! (quantisation + match quality), pixel localisation, and the current VO
//! pose covariance. Downstream consumers (detection, measurement) use these
//! sigmas - never point outputs without error bars.

use nalgebra::{Matrix6, Vector3};

use crate::camera::rectify::RectifiedRig;
use crate::stereo::map::{DisparityMap, Quality};

#[derive(Debug, Clone, Copy)]
pub struct Point3 {
    pub p: Vector3<f64>,
    /// per-axis standard deviations (world frame)
    pub sigma: Vector3<f64>,
    pub quality: Quality,
}

/// Lift a disparity map to world-frame points (subsampled by `stride`).
///
/// `pose_cov` is the 6x6 covariance of the rectified-left-camera pose used
/// for the lift (VO covariance, or zeros when the pose is exact/GT).
pub fn lift_to_world(
    disp: &DisparityMap,
    rect: &RectifiedRig,
    pose_cov: &Matrix6<f64>,
    stride: usize,
) -> Vec<Point3> {
    let k = &rect.k;
    let f = k.fx;
    let b = rect.baseline;
    let mut pts = Vec::new();
    for y in (0..disp.height).step_by(stride.max(1)) {
        for x in (0..disp.width).step_by(stride.max(1)) {
            let i = y * disp.width + x;
            let d = disp.disp[i];
            if !d.is_finite() || d < 1.0 {
                continue;
            }
            // v4: coarse half-res SGM estimates are honest MAP values but
            // excluded from object geometry - SGM boundary smearing bridges
            // objects (measured in the v4 validation; see docs/ARCHITECTURE)
            if disp.coarse[i] {
                continue;
            }
            let sd = disp.sigma[i];
            let sd = if sd.is_finite() && sd > 0.0 { sd as f64 } else { 0.5 };
            let z = f * b / d as f64;
            let sigma_z = z * z / (f * b) * sd;
            // pixel localisation uncertainty (~0.5 px) -> lateral sigma
            let sigma_lat = z / f * 0.5;
            // cam-frame point (y-up convention)
            let p_cam = Vector3::new(
                (x as f64 - k.cx) * z / f,
                -(y as f64 - k.cy) * z / k.fy,
                z,
            );
            let sigma_cam = Vector3::new(sigma_lat, sigma_lat, sigma_z);
            let cov = crate::core::uncertainty::point_covariance_through_pose(
                &rect.pose_left,
                pose_cov,
                &p_cam,
                &sigma_cam,
            );
            let p_world = rect.pose_left.transform_point(&p_cam);
            pts.push(Point3 {
                p: p_world,
                sigma: Vector3::new(cov[(0, 0)].sqrt(), cov[(1, 1)].sqrt(), cov[(2, 2)].sqrt()),
                quality: disp.quality(x, y),
            });
        }
    }
    pts
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::model::{Intrinsics, StereoRig};
    use crate::core::se3::Se3;
    use crate::camera::model::look_at;

    #[test]
    fn lifting_recovers_known_ground_depth() {
        // canonical rig at (0, 1.6, 0) looking +z; a synthetic disparity map
        // for a ground plane point at Z = 4 m, pixel (320, 400)
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let pose = Se3::from_parts(
            look_at(Vector3::new(0.0, 1.6, 0.0), Vector3::new(0.0, 1.6, 4.0), Vector3::y()),
            Vector3::new(0.0, 1.6, 0.0),
        );
        let rig = StereoRig::canonical(k, pose, 0.16);
        let rect = rig.rectify().unwrap();
        let z_true = 4.0;
        let d = k.fx * 0.16 / z_true;
        let mut disp = DisparityMap {
            width: 640,
            height: 480,
            disp: vec![f32::NAN; 640 * 480],
            sigma: vec![f32::NAN; 640 * 480],
            quality: vec![0; 640 * 480],
            coarse: vec![false; 640 * 480],
            disp_min: 0.0,
            stats: Default::default(),
        };
        // fill a 10x10 patch around (320, 400)
        for y in 395..405 {
            for x in 315..325 {
                disp.disp[y * 640 + x] = d as f32;
                disp.sigma[y * 640 + x] = 0.15;
                disp.quality[y * 640 + x] = Quality::Strong as u8;
            }
        }
        let pts = lift_to_world(&disp, &rect, &Matrix6::zeros(), 1);
        assert_eq!(pts.len(), 100);
        // Z in cam frame ~ 4; check world position: ray through (320, 400)
        let p = pts[0].p;
        // cam frame: x = 0.5*4/700*... (320-319.5)/700*4 ~ 0.003; y = -(400-239.5)/700*4 = -0.917
        assert!((p[2] - 4.0).abs() < 0.01, "world z {}", p[2]);
        assert!((p[1] - (1.6 - 0.9)).abs() < 0.05, "world y {}", p[1]);
        // sigma_z = Z^2/(fB) * 0.15 = 16/(700*0.16)*0.15 = 0.0214
        assert!(pts[0].sigma[2] > 0.015 && pts[0].sigma[2] < 0.03, "sigma z {}", pts[0].sigma[2]);
        assert!(pts[0].sigma[0] < 0.01, "sigma x {}", pts[0].sigma[0]);
    }
}
