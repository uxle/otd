//! P0600 — orbit camera (Y-up world, perspective, auto-fit, view presets).

use crate::math3::{Aabb, M4, V3};

#[derive(Clone, Copy, Debug)]
pub struct Camera {
    /// degrees around Y (0 = looking from +Z)
    pub yaw: f64,
    /// degrees above the horizon
    pub pitch: f64,
    /// distance from target
    pub dist: f64,
    /// look-at point (mm)
    pub target: V3,
    /// vertical field of view, degrees
    pub fov: f64,
}

impl Default for Camera {
    fn default() -> Camera {
        Camera { yaw: 40.0, pitch: 25.0, dist: 300.0, target: V3::ZERO, fov: 42.0 }
    }
}

impl Camera {
    /// Auto-fit an iso view to a bounding box.
    pub fn fit(bb: &Aabb) -> Camera {
        if bb.is_empty() {
            return Camera {
                yaw: 40.0,
                pitch: 28.0,
                dist: 200.0,
                target: V3::ZERO,
                fov: 42.0,
            };
        }
        let c = bb.center();
        let r = bb.radius().max(1.0);
        Camera {
            yaw: 40.0,
            pitch: 28.0,
            dist: r / (42.0f64.to_radians() / 2.0).tan() * 1.15,
            target: c,
            fov: 42.0,
        }
    }

    pub fn preset(view: &str, bb: &Aabb) -> Camera {
        let mut c = Camera::fit(bb);
        match view {
            "front" => { c.yaw = 0.0; c.pitch = 5.0; }
            "back" => { c.yaw = 180.0; c.pitch = 5.0; }
            "side" => { c.yaw = 90.0; c.pitch = 5.0; }
            "top" => { c.yaw = 0.0; c.pitch = 88.5; }
            "iso" | _ => {}
        }
        c
    }

    pub fn eye(&self) -> V3 {
        let (sy, cy) = self.yaw.to_radians().sin_cos();
        let (sp, cp) = self.pitch.to_radians().sin_cos();
        self.target.add(&V3::new(
            self.dist * cp * sy,
            self.dist * sp,
            self.dist * cp * cy,
        ))
    }

    /// View matrix (world → camera space; camera looks down −Z).
    pub fn view(&self) -> M4 {
        let eye = self.eye();
        let fwd = self.target.sub(&eye).norm();
        let world_up = V3::new(0.0, 1.0, 0.0);
        let up_hint = if fwd.y().abs() > 0.99 { V3::new(0.0, 0.0, 1.0) } else { world_up };
        let right = fwd.cross(&up_hint).norm();
        let up = right.cross(&fwd);
        // rows of the rotation, then translation
        let m = [
            right.x(), right.y(), right.z(), -right.dot(&eye),
            up.x(), up.y(), up.z(), -up.dot(&eye),
            -fwd.x(), -fwd.y(), -fwd.z(), fwd.dot(&eye),
            0.0, 0.0, 0.0, 1.0,
        ];
        // our M4 is column-major (m[col*4+row]); the above is row-major
        let mut cm = [0.0; 16];
        for col in 0..4 {
            for row in 0..4 {
                cm[col * 4 + row] = m[row * 4 + col];
            }
        }
        M4(cm)
    }

    /// Perspective project a view-space point: returns (screen_x, screen_y, view_z)
    /// in f32, or None if behind the near plane.
    pub fn project(&self, v: V3, w: f64, h: f64) -> Option<(f32, f32, f32)> {
        if v.z() > -0.5 {
            return None;
        }
        let aspect = w / h;
        let t = (self.fov / 2.0).to_radians().tan();
        let x_ndc = (v.x() / -v.z()) / (t * aspect);
        let y_ndc = (v.y() / -v.z()) / t;
        let sx = (x_ndc + 1.0) * 0.5 * w;
        let sy = (1.0 - y_ndc) * 0.5 * h;
        Some((sx as f32, sy as f32, (-v.z()) as f32))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn eye_orbits_the_target() {
        let c = Camera { yaw: 0.0, pitch: 0.0, dist: 100.0, target: V3::ZERO, fov: 42.0 };
        let e = c.eye();
        assert!((e.z() - 100.0).abs() < 1e-9 && e.y().abs() < 1e-9);
    }

    #[test]
    fn projection_maps_center() {
        let c = Camera { yaw: 0.0, pitch: 0.0, dist: 100.0, target: V3::ZERO, fov: 42.0 };
        // a point at the target projects to screen center
        let v = c.view().apply(&V3::ZERO);
        let p = c.project(v, 640.0, 480.0).unwrap();
        assert!((p.0 - 320.0).abs() < 0.01, "{}", p.0);
        assert!((p.1 - 240.0).abs() < 0.01, "{}", p.1);
    }
}
