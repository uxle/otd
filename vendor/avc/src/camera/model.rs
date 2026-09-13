//! Pinhole camera model.
//!
//! ## Pose convention (the v1 sign-bug regression test lives here)
//!
//! `Camera::pose` is **cam-to-world**: `p_world = pose.r * p_cam + pose.t`
//! where `pose.t` is the camera *centre* in world coordinates. Projection
//! therefore uses the *inverse*: `p_cam = pose.inverse() * p_world`.
//! The side-looking test below fails if the convention is violated.

use nalgebra::{Matrix3, Rotation3, Unit, Vector3};

use crate::core::se3::Se3;

#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Intrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
    pub width: u32,
    pub height: u32,
}

impl Intrinsics {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64, width: u32, height: u32) -> Self {
        Intrinsics { fx, fy, cx, cy, width, height }
    }

    /// Square-pixel intrinsics from a horizontal field of view.
    pub fn from_fov(hfov_deg: f64, width: u32, height: u32) -> Self {
        let fx = (width as f64 - 1.0) / 2.0 / (hfov_deg.to_radians() / 2.0).tan();
        Intrinsics { fx, fy: fx, cx: (width - 1) as f64 / 2.0, cy: (height - 1) as f64 / 2.0, width, height }
    }

    pub fn matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            self.fx, 0.0, self.cx,
            0.0, self.fy, self.cy,
            0.0, 0.0, 1.0,
        )
    }

    pub fn inverse_matrix(&self) -> Matrix3<f64> {
        Matrix3::new(
            1.0 / self.fx, 0.0, -self.cx / self.fx,
            0.0, 1.0 / self.fy, -self.cy / self.fy,
            0.0, 0.0, 1.0,
        )
    }

    /// K matrix for the y-up / v-down pixel convention: maps cam coords
    /// `(x, y, z)` to homogeneous pixels `[fx x + cx z, -fy y + cy z, z]`.
    /// Used by rectification homographies.
    pub fn matrix_yup(&self) -> Matrix3<f64> {
        Matrix3::new(
            self.fx, 0.0, self.cx,
            0.0, -self.fy, self.cy,
            0.0, 0.0, 1.0,
        )
    }

    /// Inverse of [`Self::matrix_yup`].
    pub fn inverse_matrix_yup(&self) -> Matrix3<f64> {
        Matrix3::new(
            1.0 / self.fx, 0.0, -self.cx / self.fx,
            0.0, -1.0 / self.fy, self.cy / self.fy,
            0.0, 0.0, 1.0,
        )
    }

    /// Project cam-frame point to pixel (returns None if behind the camera).
    /// Convention: cam frame x right, y UP, z forward (right-handed, OpenCV
    /// style); image v grows downward, hence `v = cy - fy*y/z`.
    pub fn project_cam(&self, p_cam: &Vector3<f64>) -> Option<(f64, f64)> {
        if p_cam.z <= 1e-9 {
            return None;
        }
        Some((
            self.fx * p_cam.x / p_cam.z + self.cx,
            self.cy - self.fy * p_cam.y / p_cam.z,
        ))
    }

    /// Back-project a pixel to a unit ray direction in the camera frame.
    pub fn ray_dir(&self, u: f64, v: f64) -> Vector3<f64> {
        Vector3::new(
            (u - self.cx) / self.fx,
            -(v - self.cy) / self.fy,
            1.0,
        ).normalize()
    }
}

/// A camera: intrinsics + cam-to-world pose.
#[derive(Debug, Clone, Copy)]
pub struct Camera {
    pub k: Intrinsics,
    /// cam-to-world: `p_world = pose.r * p_cam + pose.t` (t = centre in world).
    pub pose: Se3,
}

impl Camera {
    pub fn new(k: Intrinsics, pose: Se3) -> Self {
        Camera { k, pose }
    }

    /// Project a world point; returns (u, v) or None if behind.
    pub fn project(&self, p_world: &Vector3<f64>) -> Option<(f64, f64)> {
        let p_cam = self.pose.inverse().transform_point(p_world);
        self.k.project_cam(&p_cam)
    }

    /// World point -> cam-frame coordinates.
    pub fn to_cam(&self, p_world: &Vector3<f64>) -> Vector3<f64> {
        self.pose.inverse().transform_point(p_world)
    }

    /// Back-project pixel to a world-frame ray (origin = camera centre).
    pub fn ray_world(&self, u: f64, v: f64) -> (Vector3<f64>, Vector3<f64>) {
        let d_cam = self.k.ray_dir(u, v);
        (self.pose.t, self.pose.transform_dir(&d_cam))
    }

    /// Camera "forward" (+z of cam frame) in world coordinates.
    pub fn forward(&self) -> Vector3<f64> {
        self.pose.transform_dir(&Vector3::new(0.0, 0.0, 1.0))
    }
}

/// A stereo camera pair.
#[derive(Debug, Clone, Copy)]
pub struct StereoRig {
    pub left: Camera,
    pub right: Camera,
}

impl StereoRig {
    /// Canonical rectified rig: parallel cameras, baseline along cam +x,
    /// right camera translated from the left by `baseline` metres.
    pub fn canonical(k: Intrinsics, pose_left: Se3, baseline: f64) -> Self {
        let x_axis_world = pose_left.transform_dir(&Vector3::new(1.0, 0.0, 0.0));
        let right_pose = Se3::from_parts(
            pose_left.r,
            pose_left.t + x_axis_world.scale(baseline),
        );
        StereoRig { left: Camera::new(k, pose_left), right: Camera::new(k, right_pose) }
    }

    /// Baseline distance between the two camera centres.
    pub fn baseline(&self) -> f64 {
        (self.right.pose.t - self.left.pose.t).norm()
    }
}

/// Build a rotation from the "look at" convention: z forward, y up-ish.
pub fn look_at(eye: Vector3<f64>, target: Vector3<f64>, up_hint: Vector3<f64>) -> Rotation3<f64> {
    let z = (target - eye).normalize();
    let x = up_hint.cross(&z).normalize();
    let y = z.cross(&x);
    let m = Matrix3::from_columns(&[x, y, z]);
    Rotation3::from_matrix(&m)
}

/// Unit helper re-exported for camera math.
pub fn unitv(v: Vector3<f64>) -> Unit<Vector3<f64>> {
    Unit::new_normalize(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!((a - b).abs() <= tol, "{msg}: {a} vs {b}");
    }

    #[test]
    fn project_center_pixel() {
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let cam = Camera::new(k, Se3::identity());
        let (u, v) = cam.project(&Vector3::new(0.0, 0.0, 5.0)).unwrap();
        assert_close(u, 319.5, 1e-9, "center u");
        assert_close(v, 239.5, 1e-9, "center v");
    }

    #[test]
    fn project_offset_point() {
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let cam = Camera::new(k, Se3::identity());
        // world point 0.5 m right, 0.25 m up, 5 m deep
        let (u, v) = cam.project(&Vector3::new(0.5, 0.25, 5.0)).unwrap();
        assert_close(u, 319.5 + 700.0 * 0.5 / 5.0, 1e-9, "offset u");
        assert_close(v, 239.5 - 700.0 * 0.25 / 5.0, 1e-9, "offset v (y up -> smaller v)");
    }

    /// The v1 killer test: a camera looking sideways (90 deg yaw). If the
    /// cam-to-world convention is violated the projection lands on the wrong side.
    #[test]
    fn side_looking_camera() {
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        // camera at origin, rotated +90 deg about y: cam +z now points along world +x
        let pose = Se3::from_parts(
            Rotation3::from_axis_angle(&unitv(Vector3::y()), std::f64::consts::FRAC_PI_2),
            Vector3::zeros(),
        );
        let cam = Camera::new(k, pose);
        // world point 5 m along +x, 0.5 m along world -z (cam +x after yaw = world -z)
        let p = Vector3::new(5.0, 0.0, -0.5);
        let (u, _) = cam.project(&p).unwrap();
        // p in cam frame: x_cam = p . (cam x axis in world) = 5*0 + 0 + (-0.5)(-1) = +0.5
        assert_close(u, 319.5 + 700.0 * 0.5 / 5.0, 1e-6, "side-looking u");
    }

    #[test]
    fn canonical_rig_disparity_geometry() {
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let pose = Se3::from_parts(
            look_at(Vector3::new(0.0, 1.6, 0.0), Vector3::new(0.0, 1.6, 4.0), Vector3::y()),
            Vector3::new(0.0, 1.6, 0.0),
        );
        let rig = StereoRig::canonical(k, pose, 0.16);
        let p = Vector3::new(0.3, 1.2, 3.5);
        let pl = rig.left.to_cam(&p);
        let pr = rig.right.to_cam(&p);
        let (ul, _) = rig.left.k.project_cam(&pl).unwrap();
        let (ur, _) = rig.right.k.project_cam(&pr).unwrap();
        let d = ul - ur;
        assert_close(d, k.fx * rig.baseline() / pl.z, 1e-9, "disparity = fB/Z");
        // triangulate back
        let z = k.fx * rig.baseline() / d;
        assert_close(z, pl.z, 1e-9, "Z recovery");
    }

    #[test]
    fn ray_backprojection_roundtrip() {
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let cam = Camera::new(k, Se3::identity());
        let p = Vector3::new(0.4, -0.2, 4.0);
        let (u, v) = cam.project(&p).unwrap();
        let (o, d) = cam.ray_world(u, v);
        // p should lie on the ray
        let t = (p - o).dot(&d);
        let closest = o + d.scale(t);
        assert_close((closest - p).norm(), 0.0, 1e-9, "point on ray");
        assert_close(t, p.norm(), 1e-9, "ray parameter = euclidean distance");
    }
}
