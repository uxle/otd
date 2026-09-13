//! Demo scene: a deterministic synthetic world with per-pixel GT.
//!
//! Layout mirrors the v1/v2 validation world: a checkerboard ground, crates,
//! a pillar, a moving person (occluded by a dark panel mid-run), a sliding
//! small box, and a camera on a slow arc trajectory.

use image::GrayImage;
use nalgebra::Vector3;

use crate::camera::model::{Intrinsics, StereoRig, look_at};
use crate::core::rng::GaussRng;
use crate::core::se3::Se3;
use crate::sim::renderer::{RenderResult, TriMesh, add_sensor_noise, render};

#[derive(Debug, Clone)]
pub struct ObjectSpec {
    pub id: i32,
    pub class: &'static str,
    /// centre at frame 0
    pub start: [f64; 3],
    /// m / frame
    pub velocity: [f64; 3],
    /// full dimensions (w, h, d); cylinders: (2r, h, 2r)
    pub dims: [f64; 3],
    pub base_gray: u8,
    pub kind: ObjKind,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObjKind {
    Box,
    Cylinder,
}

#[derive(Debug, Clone)]
pub struct SceneSpec {
    pub objects: Vec<ObjectSpec>,
    pub n_frames: usize,
    pub baseline: f64,
    pub fps: f64,
    pub noise_sigma: f64,
}

#[derive(Debug, Clone)]
pub struct GtObjectState {
    pub id: i32,
    pub class: &'static str,
    pub center: [f64; 3],
    pub dims: [f64; 3],
    pub velocity: [f64; 3],
}

pub struct GtFrame {
    pub frame: usize,
    /// cam-to-world pose of the LEFT camera.
    pub cam_pose_left: Se3,
    /// per-pixel (left image) camera-space depth, 0 = invalid.
    pub depth: Vec<f32>,
    /// per-pixel object id, -1 = invalid, 0 = ground.
    pub obj_id: Vec<i32>,
    /// per-pixel world position.
    pub world: Vec<[f32; 3]>,
    pub objects: Vec<GtObjectState>,
}

pub struct RenderedPair {
    pub left: GrayImage,
    pub right: GrayImage,
    pub gt: GtFrame,
}

impl SceneSpec {
    /// The demo world (v1-parity: 60 frames, occlusion + motion tests).
    pub fn demo() -> SceneSpec {
        SceneSpec {
            objects: vec![
                ObjectSpec { id: 1, class: "box", start: [1.3, 0.3, 3.3], velocity: [0.0; 3], dims: [0.6, 0.6, 0.6], base_gray: 172, kind: ObjKind::Box },
                ObjectSpec { id: 2, class: "box", start: [-0.95, 0.45, 2.9], velocity: [0.0; 3], dims: [0.5, 0.9, 0.5], base_gray: 140, kind: ObjKind::Box },
                ObjectSpec { id: 3, class: "person", start: [-0.9, 0.0, 4.2], velocity: [0.025, 0.0, 0.0], dims: [0.32, 1.72, 0.32], base_gray: 95, kind: ObjKind::Cylinder },
                ObjectSpec { id: 4, class: "box", start: [0.55, 0.125, 2.3], velocity: [0.008, 0.0, 0.0], dims: [0.25, 0.25, 0.25], base_gray: 205, kind: ObjKind::Box },
                ObjectSpec { id: 5, class: "cylinder", start: [2.3, 0.0, 4.9], velocity: [0.0; 3], dims: [0.5, 1.3, 0.5], base_gray: 118, kind: ObjKind::Cylinder },
                ObjectSpec { id: 6, class: "box", start: [0.1, 1.0, 3.4], velocity: [0.0; 3], dims: [0.8, 2.0, 0.25], base_gray: 110, kind: ObjKind::Box }, // occluder (tall enough to block the person from the swinging camera)
            ],
            n_frames: 60,
            baseline: 0.16,
            fps: 30.0,
            noise_sigma: 2.0,
        }
    }

    /// Camera pose (cam-to-world) at frame t: a slow arc with forward drift.
    pub fn camera_pose(&self, t: usize) -> Se3 {
        let tf = t as f64;
        let eye = Vector3::new(1.2 * (0.03 * tf).sin(), 1.6, 0.4 + 0.035 * tf);
        let target = Vector3::new(0.0, 1.25, 3.5);
        Se3::from_parts(look_at(eye, target, Vector3::y()), eye)
    }

    /// Left camera at frame t (canonical rig).
    pub fn rig_at(&self, t: usize, k: Intrinsics) -> StereoRig {
        StereoRig::canonical(k, self.camera_pose(t), self.baseline)
    }

    fn object_states(&self, t: usize) -> Vec<GtObjectState> {
        self.objects
            .iter()
            .map(|o| {
                // cylinders are specified by their BASE; the reported GT
                // center is the geometric middle (what a detector reports)
                let y_off = if o.kind == ObjKind::Cylinder { o.dims[1] / 2.0 } else { 0.0 };
                let c = [
                    o.start[0] + o.velocity[0] * t as f64,
                    o.start[1] + o.velocity[1] * t as f64 + y_off,
                    o.start[2] + o.velocity[2] * t as f64,
                ];
                GtObjectState { id: o.id, class: o.class, center: c, dims: o.dims, velocity: o.velocity }
            })
            .collect()
    }

    fn mesh_at(&self, t: usize) -> TriMesh {
        let mut mesh = TriMesh::default();
        mesh.add_ground(14.0);
        for o in &self.objects {
            let c = [
                o.start[0] + o.velocity[0] * t as f64,
                o.start[1] + o.velocity[1] * t as f64,
                o.start[2] + o.velocity[2] * t as f64,
            ];
            match o.kind {
                ObjKind::Box => {
                    mesh.add_box(c, [o.dims[0] / 2.0, o.dims[1] / 2.0, o.dims[2] / 2.0], 0.0, o.id, o.base_gray)
                }
                ObjKind::Cylinder => {
                    mesh.add_cylinder(c, o.dims[0] / 2.0, o.dims[1], 14, o.id, o.base_gray)
                }
            }
        }
        mesh
    }

    /// Render one stereo frame with GT (deterministic noise per frame).
    pub fn render_frame(&self, t: usize, k: Intrinsics, seed: u64) -> RenderedPair {
        let rig = self.rig_at(t, k);
        let mesh = self.mesh_at(t);
        let mut rng = GaussRng::new(seed.wrapping_mul(0x9E3779B97F4A7C15).wrapping_add(t as u64));
        let mut left = render(&mesh, &rig.left);
        let mut right = render(&mesh, &rig.right);
        add_sensor_noise(&mut left, self.noise_sigma, &mut rng);
        add_sensor_noise(&mut right, self.noise_sigma, &mut rng);
        RenderedPair {
            left: left.gray,
            right: right.gray,
            gt: GtFrame {
                frame: t,
                cam_pose_left: rig.left.pose,
                depth: left.depth,
                obj_id: left.obj_id,
                world: left.world,
                objects: self.object_states(t),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn demo_scene_person_occluded_midway() {
        // The person (id 3) walks behind the occluder (id 6) mid-run and
        // re-emerges at the end - verify with rendered pixel counts.
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let scene = SceneSpec::demo();
        let visible_pixels = |t: usize| -> usize {
            let mesh = scene.mesh_at(t);
            let rig = scene.rig_at(t, k);
            let res = render(&mesh, &rig.left);
            res.obj_id.iter().filter(|&&id| id == 3).count()
        };
        let v0 = visible_pixels(0);
        let v_mid = visible_pixels(35);
        let v_end = visible_pixels(59);
        assert!(v0 > 200, "person visible at start: {v0} px");
        assert!(v_mid < 20, "person occluded mid-run: {v_mid} px");
        assert!(v_end > 100, "person re-emerges: {v_end} px");
    }

    #[test]
    fn gt_depth_matches_geometry() {
        // The ground directly below the camera start position should render
        // at depth ~ horizontal distance.
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let scene = SceneSpec::demo();
        let pair = scene.render_frame(0, k, 42);
        // pixel (220, 470): ray passes between crate2 and the occluder,
        // hits ground at z ~ 3.9
        let i = 470 * 640 + 220;
        assert_eq!(pair.gt.obj_id[i], 0, "ground between crate2 and occluder");
        assert!(pair.gt.depth[i] > 2.5 && pair.gt.depth[i] < 5.0, "depth {}", pair.gt.depth[i]);
        assert!((pair.gt.world[i][1]).abs() < 0.05, "ground world y {}", pair.gt.world[i][1]);
    }
}
