//! 6-DoF constant-velocity Kalman filter (position + velocity per axis).
//!
//! v1/v2 lessons baked in:
//! - Coasting never writes predicted states back (anchored state; only the
//!   covariance grows). Prevents quadratic divergence during occlusions.
//! - Velocity clamped to a physical prior with covariance inflation.
//! - R floors: callers must pass a measurement covariance that already
//!   includes depth-sigma floors (detection does this).

use nalgebra::{Matrix3, Matrix6, Matrix3x6, Vector3, Vector6};

#[derive(Debug, Clone)]
pub struct Kalman6 {
    pub x: Vector6<f64>,
    pub p: Matrix6<f64>,
    pub sigma_acc: f64,
    /// time of the last update (seconds)
    pub t: f64,
    pub initialized: bool,
}

impl Kalman6 {
    pub fn new(pos: Vector3<f64>, pos_cov: Matrix3<f64>, sigma_acc: f64, t: f64) -> Self {
        let mut x = Vector6::zeros();
        x.fixed_rows_mut::<3>(0).copy_from(&pos);
        let mut p = Matrix6::zeros();
        for i in 0..3 {
            p[(i, i)] = pos_cov[(i, i)];
            p[(i + 3, i + 3)] = 4.0; // 2 m/s velocity prior
        }
        Kalman6 { x, p, sigma_acc, t, initialized: true }
    }

    fn transition(&self, dt: f64) -> (Matrix6<f64>, Matrix6<f64>) {
        let mut f = Matrix6::identity();
        for i in 0..3 {
            f[(i, i + 3)] = dt;
        }
        // exact discretisation of continuous white-noise acceleration
        // (per axis, paired i / i+3): Q = sigma_a^2 [[dt^3/3, dt^2/2],[dt^2/2, dt]]
        let mut q = Matrix6::zeros();
        let sa = self.sigma_acc * self.sigma_acc;
        for i in 0..3 {
            q[(i, i)] = dt * dt * dt / 3.0 * sa;
            q[(i, i + 3)] = dt * dt / 2.0 * sa;
            q[(i + 3, i)] = dt * dt / 2.0 * sa;
            q[(i + 3, i + 3)] = dt * sa;
        }
        (f, q)
    }

    /// Advance the state to time `t` (predict step, state IS written).
    pub fn predict_to(&mut self, t: f64) {
        let dt = (t - self.t).clamp(0.0, 1.0);
        if dt <= 0.0 {
            return;
        }
        let (f, q) = self.transition(dt);
        self.x = f * self.x;
        self.p = f * self.p * f.transpose() + q;
        self.t = t;
    }

    /// Grow the covariance for elapsed time WITHOUT moving the state
    /// (anchored coasting for missed detections). The full F P F^T + Q
    /// propagation is applied so velocity uncertainty correctly inflates
    /// the position uncertainty during long occlusions.
    pub fn coast_covariance(&mut self, t: f64) {
        let dt = (t - self.t).clamp(0.0, 2.0);
        if dt <= 0.0 {
            return;
        }
        let (f, q) = self.transition(dt);
        self.p = f * self.p * f.transpose() + q;
        self.t = t;
    }

    /// Position measurement update.
    pub fn update_pos(&mut self, z: &Vector3<f64>, r: &Matrix3<f64>, t: f64) {
        self.predict_to(t);
        // H = [I 0]
        let mut h = Matrix3x6::zeros();
        for i in 0..3 {
            h[(i, i)] = 1.0;
        }
        let hp = h * self.p;
        let s = hp * h.transpose() + r;
        let s_inv = match s.try_inverse() {
            Some(inv) => inv,
            None => return,
        };
        let k = self.p * h.transpose() * s_inv;
        let y = *z - h * self.x;
        self.x += k * y;
        // Joseph form for numerical stability
        let ikh = Matrix6::identity() - k * h;
        self.p = ikh * self.p * ikh.transpose() + k * r * k.transpose();
        self.t = t;
    }

    /// Mahalanobis distance^2 of a position measurement (at current state).
    pub fn mahalanobis2(&self, z: &Vector3<f64>) -> f64 {
        self.mahalanobis_impl(z, self.position())
    }

    /// v4: position the state would have at time `t` WITHOUT mutating it
    /// (anchored: coasting never writes predictions back, but gating must
    /// compare against the prediction, not the stale anchor).
    pub fn predicted_position(&self, t: f64) -> Vector3<f64> {
        let dt = (t - self.t).clamp(0.0, 2.0);
        self.position() + self.velocity() * dt
    }

    /// v4: Mahalanobis distance^2 of a measurement against the state
    /// predicted (mean only; P already coast-grown by the caller) at `t`.
    pub fn mahalanobis2_at(&self, z: &Vector3<f64>, t: f64) -> f64 {
        self.mahalanobis_impl(z, self.predicted_position(t))
    }

    fn mahalanobis_impl(&self, z: &Vector3<f64>, mean: Vector3<f64>) -> f64 {
        let d = *z - mean;
        let mut s = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                s[(i, j)] = self.p[(i, j)];
            }
        }
        // minimum measurement noise to avoid degenerate gates
        s[(0, 0)] += 1e-4;
        s[(1, 1)] += 1e-4;
        s[(2, 2)] += 1e-4;
        match s.try_inverse() {
            Some(inv) => {
                let tmp = inv * d;
                (d.dot(&tmp)).max(0.0)
            }
            None => f64::INFINITY,
        }
    }

    pub fn position(&self) -> Vector3<f64> {
        Vector3::new(self.x[0], self.x[1], self.x[2])
    }

    pub fn velocity(&self) -> Vector3<f64> {
        Vector3::new(self.x[3], self.x[4], self.x[5])
    }

    pub fn pos_cov(&self) -> Matrix3<f64> {
        let mut c = Matrix3::zeros();
        for i in 0..3 {
            for j in 0..3 {
                c[(i, j)] = self.p[(i, j)];
            }
        }
        c
    }

    /// Clamp the velocity to `vmax` (m/s), inflating covariance when applied.
    pub fn clamp_velocity(&mut self, vmax: f64) {
        let v = self.velocity();
        let s = v.norm();
        if s > vmax {
            let scale = vmax / s;
            for i in 3..6 {
                self.x[i] *= scale;
                self.p[(i, i)] *= 4.0; // honest: we intervened, trust less
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converges_on_constant_velocity() {
        let mut rng = crate::core::rng::GaussRng::new(3);
        let mut kf = Kalman6::new(
            Vector3::zeros(),
            Matrix3::identity() * 0.04,
            0.5,
            0.0,
        );
        let dt = 1.0 / 30.0;
        let v_true = Vector3::new(0.5, 0.0, 0.0);
        for k in 1..90 {
            let t = k as f64 * dt;
            let z = v_true.scale(t) + Vector3::new(rng.gauss(0.0, 0.02), 0.0, 0.0);
            let r = Matrix3::identity() * 0.02 * 0.02;
            kf.update_pos(&z, &r, t);
        }
        let verr = (kf.velocity() - v_true).norm();
        assert!(verr < 0.05, "velocity error {verr}");
        let perr = (kf.position() - v_true.scale(90.0 * dt)).norm();
        assert!(perr < 0.05, "position error {perr}");
        // covariance shrank
        assert!(kf.p[(0, 0)] < 0.01, "cov {}", kf.p[(0, 0)]);
    }

    #[test]
    fn anchored_coasting_does_not_diverge() {
        let mut kf = Kalman6::new(
            Vector3::new(1.0, 0.0, 3.0),
            Matrix3::identity() * 0.01,
            0.5,
            0.0,
        );
        // teach it a velocity
        let dt = 1.0 / 30.0;
        for k in 1..30 {
            let t = k as f64 * dt;
            let z = Vector3::new(1.0 + 0.3 * t, 0.0, 3.0);
            kf.update_pos(&z, &(Matrix3::identity() * 0.0004), t);
        }
        let v_before = kf.velocity();
        // coast 60 frames WITHOUT updates: state must stay anchored
        let mut t = kf.t;
        for _ in 0..60 {
            t += dt;
            kf.coast_covariance(t);
        }
        assert!((kf.velocity() - v_before).norm() < 1e-9, "velocity frozen during coast");
        // covariance grew
        let c = kf.p[(0, 0)];
        assert!(c > 0.01, "covariance must grow during coasting: {c}");
        // a late measurement snaps it back
        let t_end = t;
        kf.update_pos(&Vector3::new(1.0 + 0.3 * t_end, 0.0, 3.0), &(Matrix3::identity() * 0.0004), t_end);
        assert!((kf.position() - Vector3::new(1.0 + 0.3 * t_end, 0.0, 3.0)).norm() < 0.1);
    }

    #[test]
    fn velocity_clamp() {
        let mut kf = Kalman6::new(Vector3::zeros(), Matrix3::identity() * 0.01, 0.5, 0.0);
        kf.x[3] = 10.0;
        kf.clamp_velocity(3.0);
        assert!(kf.velocity().norm() <= 3.01);
        assert!(kf.p[(3, 3)] > 4.0, "covariance inflated after intervention");
    }
}
