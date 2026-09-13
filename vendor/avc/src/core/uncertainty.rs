//! First-order uncertainty propagation.
//!
//! v1 taught us that hiding error bars is worse than admitting them: every
//! position, velocity and measurement in the world model carries a sigma
//! propagated through the actual geometry of the pipeline. All formulas are
//! the standard first-order (Jacobian) approximations; tests verify them
//! against Monte-Carlo ground truth.

use nalgebra::{Matrix3, Matrix6, Matrix3x6, Vector3};

use crate::core::se3::{skew, Se3};

/// Transform a 3D point with per-axis measurement sigma through a pose that
/// itself has a 6x6 body-frame covariance (twist order [w; v]).
///
/// `sigma_point` are the per-axis standard deviations of the point in the
/// source frame. Returns the transformed point's 3x3 covariance in the
/// destination frame.
///
/// Model: `p' = T exp(delta) p` with `delta ~ N(0, pose_cov)`,
/// `J_delta = [-[R p]x R | R]`, hence
/// `Sigma' = J Sigma_delta J^T + R Sigma_p R^T`.
pub fn point_covariance_through_pose(
    pose: &Se3,
    pose_cov: &Matrix6<f64>,
    p_src: &Vector3<f64>,
    sigma_point: &Vector3<f64>,
) -> Matrix3<f64> {
    let rp = pose.r.transform_vector(p_src); // rotated point
    let j_w = -skew(&rp) * pose.r.matrix();
    // J = [ j_w | R ]  (3x6)
    let r = pose.r.matrix();
    let mut jac = [[0.0f64; 6]; 3];
    for i in 0..3 {
        for k in 0..3 {
            jac[i][k] = j_w[(i, k)];
            jac[i][k + 3] = r[(i, k)];
        }
    }
    let mut jac_m = Matrix3x6::zeros();
    for i in 0..3 {
        for k in 0..6 {
            jac_m[(i, k)] = jac[i][k];
        }
    }
    let cov = jac_m * pose_cov * jac_m.transpose();
    // add the point's own covariance, rotated
    let mut sp = Matrix3::zeros();
    for i in 0..3 {
        sp[(i, i)] = sigma_point[i] * sigma_point[i];
    }
    cov + r * sp * r.transpose()
}

/// Convenience: per-axis sigmas of a transformed point.
pub fn point_sigma_through_pose(
    pose: &Se3,
    pose_cov: &Matrix6<f64>,
    p_src: &Vector3<f64>,
    sigma_point: &Vector3<f64>,
) -> Vector3<f64> {
    let cov = point_covariance_through_pose(pose, pose_cov, p_src, sigma_point);
    Vector3::new(cov[(0, 0)].sqrt(), cov[(1, 1)].sqrt(), cov[(2, 2)].sqrt())
}

/// Standard deviation of the distance between two uncertain points
/// (first order): `sigma_d^2 = u^T (Sigma_a + Sigma_b) u`,
/// `u = (p_a - p_b)/|p_a - p_b|`.
pub fn sigma_of_distance(p_a: &Vector3<f64>, cov_a: &Matrix3<f64>, p_b: &Vector3<f64>, cov_b: &Matrix3<f64>) -> f64 {
    let d = p_a - p_b;
    let n = d.norm();
    if n < 1e-9 {
        return 0.0;
    }
    let u = d / n;
    let mut var = 0.0;
    for i in 0..3 {
        for j in 0..3 {
            var += u[i] * (cov_a[(i, j)] + cov_b[(i, j)]) * u[j];
        }
    }
    var.max(0.0).sqrt()
}

/// Disparity -> depth sigma: `sigma_Z = Z^2 / (f B) * sigma_d`.
#[inline]
pub fn depth_sigma(z: f64, f: f64, baseline: f64, sigma_d: f64) -> f64 {
    (z * z) / (f * baseline) * sigma_d
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rng::GaussRng;
    use nalgebra::{Rotation3, Unit};

    /// Monte-Carlo check: linearised covariance must match sample covariance
    /// within ~25% for small sigmas (first-order validity regime).
    #[test]
    fn point_cov_matches_monte_carlo() {
        let mut rng = GaussRng::new(21);
        let axis = Unit::new_normalize(Vector3::new(0.3, 0.5, 1.0));
        let pose = Se3 {
            r: Rotation3::from_axis_angle(&axis, 0.4),
            t: Vector3::new(0.5, -0.3, 1.2),
        };
        // pose covariance: 0.01 m on translation axes, 0.002 rad on rotation axes
        let mut pc = Matrix6::zeros();
        for i in 0..3 {
            pc[(i, i)] = 0.002f64 * 0.002;
        }
        for i in 3..6 {
            pc[(i, i)] = 0.01 * 0.01;
        }
        let p = Vector3::new(0.4, 0.6, 3.0);
        let sig = Vector3::new(0.02, 0.02, 0.05);

        let cov = point_covariance_through_pose(&pose, &pc, &p, &sig);

        // Monte Carlo using the same model: p' = T exp(delta) p, delta ~ N(0, pc)
        let n = 40000;
        let mut sum = Vector3::zeros();
        let mut samples = Vec::with_capacity(n);
        for _ in 0..n {
            let mut xi = [0.0f64; 6];
            for i in 0..6 {
                xi[i] = rng.gauss(0.0, pc[(i, i)].sqrt());
            }
            let w = Vector3::new(xi[0], xi[1], xi[2]);
            let v = Vector3::new(xi[3], xi[4], xi[5]);
            let perturbed = pose.compose(&Se3::exp_se3(&w, &v));
            let p_noise = p + Vector3::new(rng.gauss(0.0, sig[0]), rng.gauss(0.0, sig[1]), rng.gauss(0.0, sig[2]));
            let q = perturbed.transform_point(&p_noise);
            samples.push(q);
            sum += q;
        }
        let mean = sum / n as f64;
        let mut sample_cov = Matrix3::zeros();
        for q in &samples {
            let d = q - mean;
            sample_cov += d * d.transpose();
        }
        sample_cov /= (n - 1) as f64;

        for i in 0..3 {
            let lin = cov[(i, i)].sqrt();
            let mc = sample_cov[(i, i)].sqrt();
            let rel = (lin - mc).abs() / mc.max(1e-9);
            assert!(rel < 0.25, "axis {i}: linear {lin:.4} vs MC {mc:.4} (rel {rel:.2})");
        }
    }

    #[test]
    fn distance_sigma_matches_monte_carlo() {
        let mut rng = GaussRng::new(33);
        let pa = Vector3::new(1.0, 0.2, 3.0);
        let pb = Vector3::new(-0.5, 0.4, 2.0);
        let mut ca = Matrix3::zeros();
        ca[(0, 0)] = 0.04;
        ca[(1, 1)] = 0.01;
        ca[(2, 2)] = 0.09;
        let mut cb = Matrix3::zeros();
        cb[(0, 0)] = 0.01;
        cb[(1, 1)] = 0.01;
        cb[(2, 2)] = 0.04;
        let sd = sigma_of_distance(&pa, &ca, &pb, &cb);
        let n = 40000;
        let mut ds = Vec::with_capacity(n);
        for _ in 0..n {
            let qa = pa + Vector3::new(rng.gauss(0.0, ca[(0,0)].sqrt()), rng.gauss(0.0, ca[(1,1)].sqrt()), rng.gauss(0.0, ca[(2,2)].sqrt()));
            let qb = pb + Vector3::new(rng.gauss(0.0, cb[(0,0)].sqrt()), rng.gauss(0.0, cb[(1,1)].sqrt()), rng.gauss(0.0, cb[(2,2)].sqrt()));
            ds.push((qa - qb).norm());
        }
        ds.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let mc = ds[n / 2] - ds[0]; // crude scale check below uses std instead
        let _ = mc;
        let mean = ds.iter().sum::<f64>() / n as f64;
        let var = ds.iter().map(|d| (d - mean) * (d - mean)).sum::<f64>() / n as f64;
        let mc_std = var.sqrt();
        let rel = (sd - mc_std).abs() / mc_std;
        assert!(rel < 0.2, "distance sigma {sd:.4} vs MC {mc_std:.4}");
    }

    #[test]
    fn depth_sigma_scaling() {
        // 1 px disparity error at Z=5 m, f=700, B=0.16
        let s = depth_sigma(5.0, 700.0, 0.16, 1.0);
        let expect = 25.0 / (700.0 * 0.16);
        assert!((s - expect).abs() < 1e-12);
    }
}
