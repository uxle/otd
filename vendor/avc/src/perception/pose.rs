//! 17-joint human pose lifting (COCO keypoint set).
//!
//! Honest scope: this module lifts EXTERNAL 2D keypoints (e.g. from a
//! YOLO-pose / OpenPose style detector running outside this crate) to
//! metric 3D using the disparity map, with per-joint depth sigma and
//! occlusion flags. v1's YOLOv8n ONNX backend is not part of the Rust
//! edition's dependency set (keeps the build hermetic); the lifting math
//! and API are identical.

use crate::camera::rectify::RectifiedRig;
use crate::stereo::map::{DisparityMap, Quality};

pub const JOINT_NAMES: [&str; 17] = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
];

/// Skeleton bones (index pairs) for visualisation / export.
pub const SKELETON_EDGES: [(usize, usize); 18] = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
    (0, 5), (0, 6),
];

#[derive(Debug, Clone, Copy)]
pub struct Keypoint2 {
    pub joint: usize,
    pub u: f64,
    pub v: f64,
    pub conf: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct Joint3 {
    /// world-frame position
    pub p: [f64; 3],
    /// per-axis sigma
    pub sigma: [f64; 3],
    /// true when the depth was interpolated from neighbours
    pub filled: bool,
    pub conf: f64,
}

#[derive(Debug, Clone, Default)]
pub struct Skeleton3D {
    /// 17 slots in COCO order; `None` = joint not liftable (occluded / no depth).
    pub joints: Vec<Option<Joint3>>,
}

impl Skeleton3D {
    pub fn joint(&self, name: &str) -> Option<&Joint3> {
        let i = JOINT_NAMES.iter().position(|&n| n == name)?;
        self.joints.get(i)?.as_ref()
    }
    pub fn height(&self) -> Option<f64> {
        // nose to the lowest ankle
        let nose = self.joint("nose")?;
        let ankles: Vec<&Joint3> = ["left_ankle", "right_ankle"]
            .iter()
            .filter_map(|&n| self.joint(n))
            .collect();
        let min_y = ankles.iter().map(|a| a.p[1]).fold(f64::INFINITY, f64::min);
        Some(nose.p[1] - min_y)
    }
}

/// Sample the disparity at (u, v); fall back to the 5x5 neighbourhood.
fn sample_disparity(disp: &DisparityMap, u: f64, v: f64) -> Option<(f64, f64, bool)> {
    let xi = u.round() as i64;
    let yi = v.round() as i64;
    if xi < 0 || yi < 0 || xi >= disp.width as i64 || yi >= disp.height as i64 {
        return None;
    }
    let direct = disp.disp[yi as usize * disp.width + xi as usize];
    if direct.is_finite() {
        let s = disp.sigma[yi as usize * disp.width + xi as usize];
        let filled = disp.quality(xi as usize, yi as usize) == Quality::Filled;
        return Some((direct as f64, s.max(0.15) as f64, filled));
    }
    // 5x5 neighbourhood fallback
    let mut best: Option<(f64, f64)> = None;
    for dy in -2i64..=2 {
        for dx in -2i64..=2 {
            let x = xi + dx;
            let y = yi + dy;
            if x < 0 || y < 0 || x >= disp.width as i64 || y >= disp.height as i64 {
                continue;
            }
            let d = disp.disp[y as usize * disp.width + x as usize];
            if d.is_finite() {
                let s = disp.sigma[y as usize * disp.width + x as usize].max(0.3) as f64;
                match best {
                    Some((_, bs)) if s >= bs => {}
                    _ => best = Some((d as f64, s)),
                }
            }
        }
    }
    best.map(|(d, s)| (d, s + 0.2, true))
}

/// Lift 2D keypoints to metric 3D via the disparity map.
pub fn lift_skeleton(
    kps: &[Keypoint2],
    disp: &DisparityMap,
    rect: &RectifiedRig,
) -> Skeleton3D {
    let mut joints: Vec<Option<Joint3>> = vec![None; 17];
    for kp in kps {
        if let Some((d, sd, filled)) = sample_disparity(disp, kp.u, kp.v) {
            let z = rect.k.fx * rect.baseline / d;
            let sigma_z = z * z / (rect.k.fx * rect.baseline) * sd;
            let sigma_lat = z / rect.k.fx * 1.0; // 2D detector localisation ~ 1 px
            let p_cam = crate::camera::rectify::RectifiedRig::triangulate_cam(
                rect, kp.u, kp.v, kp.u - d,
            );
            let p_world = rect.pose_left.transform_point(&p_cam);
            let sig = crate::core::uncertainty::point_sigma_through_pose(
                &rect.pose_left,
                &nalgebra::Matrix6::zeros(),
                &p_cam,
                &nalgebra::Vector3::new(sigma_lat, sigma_lat, sigma_z),
            );
            joints[kp.joint.min(16)] = Some(Joint3 {
                p: [p_world[0], p_world[1], p_world[2]],
                sigma: [sig[0], sig[1], sig[2]],
                filled,
                conf: kp.conf,
            });
        }
    }
    Skeleton3D { joints }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::model::{Intrinsics, StereoRig, look_at};
    use crate::core::se3::Se3;
    use crate::stereo::map::{DisparityMap, Quality};
    use nalgebra::Vector3;

    #[test]
    fn skeleton_lift_roundtrip() {
        // a "person" at (0, 0, 4): nose y=1.65, hips 0.9, ankles 0.05
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let pose = Se3::from_parts(
            look_at(Vector3::new(0.0, 1.6, 0.0), Vector3::new(0.0, 1.6, 4.0), Vector3::y()),
            Vector3::new(0.0, 1.6, 0.0),
        );
        let rig = StereoRig::canonical(k, pose, 0.16);
        let rect = rig.rectify().unwrap();

        let true_joints: Vec<(&str, [f64; 3])> = vec![
            ("nose", [0.0, 1.65, 5.0]),
            ("left_shoulder", [-0.18, 1.42, 5.0]),
            ("right_shoulder", [0.18, 1.42, 5.0]),
            ("left_hip", [-0.11, 0.9, 5.0]),
            ("right_hip", [0.11, 0.9, 5.0]),
            ("left_ankle", [-0.11, 0.05, 5.0]),
            ("right_ankle", [0.11, 0.05, 5.0]),
        ];
        // build a consistent disparity map: every pixel at the depth of the
        // person plane Z = 4 (constant disparity)
        let d_person = k.fx * 0.16 / 5.0;
        let mut disp = DisparityMap {
            width: 640,
            height: 480,
            disp: vec![d_person as f32; 640 * 480],
            sigma: vec![0.15; 640 * 480],
            quality: vec![Quality::Strong as u8; 640 * 480],
            coarse: vec![false; 640 * 480],
            disp_min: 0.0,
            stats: Default::default(),
        };
        disp.disp[0] = f32::NAN; // some invalid pixel elsewhere

        let kps: Vec<Keypoint2> = true_joints
            .iter()
            .map(|(name, p)| {
                // project world -> pixel; joint index = COCO slot
                let joint = JOINT_NAMES.iter().position(|&n| n == *name).unwrap();
                let pw = Vector3::new(p[0], p[1], p[2]);
                let pc = rig.left.to_cam(&pw);
                let (u, v) = k.project_cam(&pc).unwrap();
                Keypoint2 { joint, u, v, conf: 0.9 }
            })
            .collect();
        let sk = lift_skeleton(&kps, &disp, &rect);
        assert_eq!(sk.joints.len(), 17);
        for (name, p) in true_joints.iter() {
            let j = sk.joint(name).expect("joint lifted");
            let err = (j.p[0] - p[0]).abs().max((j.p[1] - p[1]).abs()).max((j.p[2] - p[2]).abs());
            assert!(err < 0.03, "joint {name} error {err:.3} m: {:?} vs {:?}", j.p, p);
            assert!(j.sigma[2] > 0.01, "depth sigma present");
        }
        let h = sk.height().unwrap();
        assert!((h - 1.60).abs() < 0.05, "skeleton height {h:.3} (nose to ankle 1.60)");
    }
}
