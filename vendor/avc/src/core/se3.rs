//! SE(3) rigid transforms with a correct exponential / logarithm map.
//!
//! ## Conventions (verified by tests below - v1 shipped a sign bug here)
//!
//! `Se3 { r, t }` maps points of frame **b** into frame **a**:
//! `p_a = r * p_b + t`. The camera module stores *camera-to-world* poses as
//! `Se3 { r, c }` (i.e. `t` equals the camera centre in world coordinates),
//! so `p_world = pose * p_cam` and `p_cam = pose.inverse() * p_world`.
//!
//! The exponential map follows Barfoot, *State Estimation for Robotics*:
//! `exp(w, v) = (Rodrigues(w), A(w) v)` with
//! `A = I + ((1-cos t)/t^2) [w]x + ((t-sin t)/t^3) [w]x^2`, and
//! `log` inverts it with `A^-1 = I - 1/2 [w]x + (1/t^2 - (1+cos t)/(2 t sin t)) [w]x^2`.

use nalgebra::{Matrix3, Matrix6, Rotation3, Unit, Vector3, Vector6};

/// Rigid transform: `p_a = r * p_b + t`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Se3 {
    pub r: Rotation3<f64>,
    pub t: Vector3<f64>,
}

impl Default for Se3 {
    fn default() -> Self {
        Self::identity()
    }
}

impl Se3 {
    /// The identity transform.
    pub fn identity() -> Self {
        Se3 { r: Rotation3::identity(), t: Vector3::zeros() }
    }

    /// Build from rotation + translation.
    pub fn from_parts(r: Rotation3<f64>, t: Vector3<f64>) -> Self {
        Se3 { r, t }
    }

    /// Build from a rotation matrix (validated: orthonormal, det +1).
    pub fn from_rotmat(m: Matrix3<f64>, t: Vector3<f64>) -> Option<Self> {
        let id = m * m.transpose();
        for i in 0..3 {
            for j in 0..3 {
                let expect = if i == j { 1.0 } else { 0.0 };
                if (id[(i, j)] - expect).abs() > 1e-6 {
                    return None;
                }
            }
        }
        if (m.determinant() - 1.0).abs() > 1e-6 {
            return None;
        }
        Some(Se3 { r: Rotation3::from_matrix(&m), t })
    }

    /// Inverse transform: maps a -> b.
    pub fn inverse(&self) -> Se3 {
        Se3 { r: self.r.inverse(), t: -self.r.inverse().transform_vector(&self.t) }
    }

    /// Compose: `self * other` maps other's frame into self's frame.
    /// `(R1 t1) * (R2 t2) = (R1 R2, R1 t2 + t1)`.
    pub fn compose(&self, other: &Se3) -> Se3 {
        Se3 { r: self.r * other.r, t: self.r.transform_vector(&other.t) + self.t }
    }

    /// Transform a point.
    pub fn transform_point(&self, p: &Vector3<f64>) -> Vector3<f64> {
        self.r.transform_vector(p) + self.t
    }

    /// Transform a direction (no translation).
    pub fn transform_dir(&self, d: &Vector3<f64>) -> Vector3<f64> {
        self.r.transform_vector(d)
    }

    /// 4x4 homogeneous matrix (for exports / debugging).
    pub fn to_homogeneous(&self) -> [[f64; 4]; 4] {
        let m = self.r.matrix();
        [
            [m[(0, 0)], m[(0, 1)], m[(0, 2)], self.t[0]],
            [m[(1, 0)], m[(1, 1)], m[(1, 2)], self.t[1]],
            [m[(2, 0)], m[(2, 1)], m[(2, 2)], self.t[2]],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }

    /// Adjoint: `T exp(xi) = exp(Ad_T xi) T` for body-frame twists.
    /// `Ad = [[R, 0], [[t]x R, R]]`.
    pub fn adjoint(&self) -> Matrix6<f64> {
        let mut ad = Matrix6::zeros();
        let m = self.r.matrix();
        // top-left: R, top-right: 0
        for i in 0..3 {
            for j in 0..3 {
                ad[(i, j)] = m[(i, j)];
                ad[(i + 3, j + 3)] = m[(i, j)];
            }
        }
        // bottom-left: [t]x * R
        let tr = skew(&self.t) * m;
        for i in 0..3 {
            for j in 0..3 {
                ad[(i + 3, j)] = tr[(i, j)];
            }
        }
        ad
    }

    /// SE(3) exponential map from twist (w, v) (body frame).
    pub fn exp_se3(w: &Vector3<f64>, v: &Vector3<f64>) -> Se3 {
        let theta = w.norm();
        let r = if theta < 1e-10 {
            Rotation3::from_scaled_axis(*w)
        } else {
            let axis = Unit::new_normalize(*w);
            Rotation3::from_axis_angle(&axis, theta)
        };
        let t = if theta < 1e-10 {
            // A -> I in the limit
            *v + 0.5 * w.cross(v) // first-order correction (keeps log/exp consistent)
        } else {
            let wx = skew(w);
            let wx2 = wx * wx;
            let a = Matrix3::identity()
                + wx.scale((1.0 - theta.cos()) / (theta * theta))
                + wx2.scale((theta - theta.sin()) / (theta * theta * theta));
            a * v
        };
        Se3 { r, t }
    }

    /// SE(3) logarithm: returns (w, v) in the body frame.
    pub fn log_se3(&self) -> (Vector3<f64>, Vector3<f64>) {
        let w = so3_log(&self.r);
        let theta = w.norm();
        let v = if theta < 1e-10 {
            self.t - 0.5 * w.cross(&self.t)
        } else {
            let wx = skew(&w);
            let wx2 = wx * wx;
            let c = theta.cos();
            let s = theta.sin();
            let coeff = 1.0 / (theta * theta) - (1.0 + c) / (2.0 * theta * s);
            let a_inv =
                Matrix3::identity() - wx.scale(0.5) + wx2.scale(coeff);
            a_inv * self.t
        };
        (w, v)
    }

    /// Uncertainty of this pose in its own body frame (6x6, order [w; v]).
    /// See `uncertainty::point_sigma_through_pose` for how it is propagated.
    pub fn pose_cov_identity() -> Matrix6<f64> {
        Matrix6::identity()
    }
}

/// so(3) logarithm with robust handling near 0 and pi.
pub fn so3_log(r: &Rotation3<f64>) -> Vector3<f64> {
    let theta = r.angle();
    if theta < 1e-10 {
        // small angle: 0.5 * vee(R - R^T)
        let m = r.matrix();
        let mut w = Vector3::zeros();
        w[0] = 0.5 * (m[(2, 1)] - m[(1, 2)]);
        w[1] = 0.5 * (m[(0, 2)] - m[(2, 0)]);
        w[2] = 0.5 * (m[(1, 0)] - m[(0, 1)]);
        return w;
    }
    if (theta - std::f64::consts::PI).abs() < 1e-6 {
        // near pi: axis from the eigenvector of R with eigenvalue +1
        // (column of R + I with the largest norm)
        let m = r.matrix();
        let mut best = Vector3::zeros();
        let mut best_norm = -1.0;
        for k in 0..3 {
            let mut col = Vector3::zeros();
            for i in 0..3 {
                col[i] = m[(i, k)] + if i == k { 1.0 } else { 0.0 };
            }
            let n = col.norm();
            if n > best_norm {
                best_norm = n;
                best = col;
            }
        }
        let axis = best.normalize();
        // sign: choose so that rotation is near +pi (either is valid modulo 2pi)
        return axis.scale(theta);
    }
    // general case: theta / (2 sin theta) * vee(R - R^T)
    let m = r.matrix();
    let mut vee = Vector3::zeros();
    vee[0] = m[(2, 1)] - m[(1, 2)];
    vee[1] = m[(0, 2)] - m[(2, 0)];
    vee[2] = m[(1, 0)] - m[(0, 1)];
    vee.scale(theta / (2.0 * theta.sin()))
}

/// 3x3 skew-symmetric matrix [v]x.
pub fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v[2], v[1],
        v[2], 0.0, -v[0],
        -v[1], v[0], 0.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f64, b: f64, tol: f64, msg: &str) {
        assert!((a - b).abs() <= tol, "{msg}: {a} vs {b} (tol {tol})");
    }

    #[test]
    fn exp_log_roundtrip_general() {
        for &(wx, wy, wz, vx, vy, vz) in &[
            (0.3, -0.2, 0.5, 1.0, -2.0, 0.5),
            (-1.2, 0.1, 0.0, 0.0, 0.3, -1.0),
            (0.05, 0.02, -0.03, 0.1, 0.2, 0.3),
            (2.5, 1.0, -1.5, -3.0, 2.0, 1.0),
        ] {
            let w = Vector3::new(wx, wy, wz);
            let v = Vector3::new(vx, vy, vz);
            let t = Se3::exp_se3(&w, &v);
            let (w2, v2) = t.log_se3();
            assert_close((w2 - w).norm(), 0.0, 1e-9, "w roundtrip");
            assert_close((v2 - v).norm(), 0.0, 1e-9, "v roundtrip");
        }
    }

    #[test]
    fn exp_log_roundtrip_near_pi() {
        // 179 degrees around z
        let axis = Unit::new_normalize(Vector3::new(0.1, 0.2, 1.0));
        let r = Rotation3::from_axis_angle(&axis, 3.124);
        let t = Se3 { r, t: Vector3::new(0.5, -0.25, 1.0) };
        let (w, v) = t.log_se3();
        assert_close(w.norm(), 3.124, 1e-6, "angle near pi");
        let t2 = Se3::exp_se3(&w, &v);
        // compare as transforms (angle 3.124 vs -3.124+2pi both valid; use transform action)
        let p = Vector3::new(1.0, 2.0, -1.0);
        let pa = t.transform_point(&p);
        let pb = t2.transform_point(&p);
        assert_close((pa - pb).norm(), 0.0, 1e-6, "transform equivalence near pi");
    }

    #[test]
    fn exp_small_angle_matches_limit() {
        let w = Vector3::new(1e-12, 0.0, 0.0);
        let v = Vector3::new(1.0, 2.0, 3.0);
        let t = Se3::exp_se3(&w, &v);
        assert_close((t.t - v).norm(), 0.0, 1e-9, "small angle translation");
    }

    #[test]
    fn compose_and_inverse() {
        let a = Se3::exp_se3(&Vector3::new(0.1, 0.2, -0.05), &Vector3::new(1.0, 0.0, 0.0));
        let b = Se3::exp_se3(&Vector3::new(-0.3, 0.0, 0.4), &Vector3::new(0.0, 2.0, -1.0));
        let c = a.compose(&b);
        let p = Vector3::new(0.5, -0.5, 2.0);
        // a(b(p)) == (a*b)(p)
        let pa = a.transform_point(&b.transform_point(&p));
        let pc = c.transform_point(&p);
        assert_close((pa - pc).norm(), 0.0, 1e-12, "compose action");
        // inverse undoes
        let ai = a.inverse();
        let roundtrip = ai.transform_point(&a.transform_point(&p));
        assert_close((roundtrip - p).norm(), 0.0, 1e-12, "inverse action");
        // inv via matrix: (A B)^-1 = B^-1 A^-1
        let ci = c.inverse();
        let cb = b.inverse().compose(&a.inverse());
        assert_close(
            (ci.transform_point(&p) - cb.transform_point(&p)).norm(),
            0.0,
            1e-12,
            "inverse of composition",
        );
    }

    #[test]
    fn adjoint_identity_property() {
        // T exp(xi) == exp(Ad_T xi) T
        let t = Se3::exp_se3(&Vector3::new(0.2, -0.4, 0.1), &Vector3::new(1.0, -1.0, 2.0));
        let w = Vector3::new(0.01, 0.02, -0.015);
        let v = Vector3::new(0.05, -0.03, 0.02);
        let lhs = t.compose(&Se3::exp_se3(&w, &v));
        let xi = Vector6::new(w[0], w[1], w[2], v[0], v[1], v[2]);
        let ad_xi = t.adjoint() * xi;
        let w2 = Vector3::new(ad_xi[0], ad_xi[1], ad_xi[2]);
        let v2 = Vector3::new(ad_xi[3], ad_xi[4], ad_xi[5]);
        let rhs = Se3::exp_se3(&w2, &v2).compose(&t);
        let p = Vector3::new(0.3, 0.7, -1.1);
        assert_close(
            (lhs.transform_point(&p) - rhs.transform_point(&p)).norm(),
            0.0,
            1e-9,
            "adjoint identity",
        );
    }

    #[test]
    fn homogeneous_matrix_roundtrip() {
        let t = Se3::exp_se3(&Vector3::new(0.1, 0.1, 0.1), &Vector3::new(0.4, -0.2, 0.9));
        let h = t.to_homogeneous();
        let p = Vector3::new(1.0, 2.0, 3.0);
        let mut ph = [p[0], p[1], p[2], 1.0];
        let mut out = [0.0; 4];
        for i in 0..4 {
            out[i] = h[i][0] * ph[0] + h[i][1] * ph[1] + h[i][2] * ph[2] + h[i][3] * ph[3];
        }
        let pt = t.transform_point(&p);
        assert_close(out[0], pt[0], 1e-12, "homog x");
        assert_close(out[1], pt[1], 1e-12, "homog y");
        assert_close(out[2], pt[2], 1e-12, "homog z");
        ph = out; // silence unused warning pattern
        let _ = ph;
    }
}
