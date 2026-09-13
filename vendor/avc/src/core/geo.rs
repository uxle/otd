//! Geometric estimation helpers: Kabsch / Umeyama (fixed scale=1) for 3D-3D
//! rigid registration (used by visual odometry), and RANSAC plane fitting
//! (used by ground segmentation).

use nalgebra::{Matrix3, Rotation3, Vector3};

use crate::core::rng::GaussRng;

/// Result of a rigid 3D-3D registration: `b ~ r * a + t`.
#[derive(Debug, Clone, Copy)]
pub struct RigidFit {
    pub r: Rotation3<f64>,
    pub t: Vector3<f64>,
    pub n_inliers: usize,
    pub rms: f64,
}

/// Kabsch algorithm (rigid, scale fixed to 1) on paired points
/// `a_i -> b_i`. Returns (R, t) minimising sum |R a + t - b|^2.
pub fn kabsch(a: &[Vector3<f64>], b: &[Vector3<f64>]) -> Option<(Rotation3<f64>, Vector3<f64>)> {
    let n = a.len();
    if n < 3 || b.len() != n {
        return None;
    }
    let mut ca = Vector3::zeros();
    let mut cb = Vector3::zeros();
    for i in 0..n {
        ca += a[i];
        cb += b[i];
    }
    ca /= n as f64;
    cb /= n as f64;

    // H = sum (a - ca)(b - cb)^T
    let mut h = Matrix3::zeros();
    for i in 0..n {
        let da = a[i] - ca;
        let db = b[i] - cb;
        h += da * db.transpose();
    }
    let svd = h.svd(true, true);
    let u = svd.u?;
    let vt = svd.v_t?;
    // R = V * D * U^T  (Kabsch), D = diag(1,1,det(V U^T))
    let mut r = vt.transpose() * u.transpose();
    if r.determinant() < 0.0 {
        // reflection: flip the least significant axis
        let d = Matrix3::from_diagonal(&Vector3::new(1.0, 1.0, -1.0));
        r = vt.transpose() * d * u.transpose();
    }
    let rot = Rotation3::from_matrix(&r);
    let t = cb - rot.transform_vector(&ca);
    Some((rot, t))
}

/// RANSAC over 3D-3D pairs. Residual = |R a + t - b|.
/// `thresh` is the inlier gate in metres.
pub fn ransac_rigid(
    a: &[Vector3<f64>],
    b: &[Vector3<f64>],
    thresh: f64,
    iters: usize,
    rng: &mut crate::core::rng::GaussRng,
) -> Option<RigidFit> {
    let n = a.len();
    if n < 6 {
        return kabsch(a, b).map(|(r, t)| RigidFit { r, t, n_inliers: n, rms: rms_of(&a, &b, &r, &t) });
    }
    let mut best: Option<RigidFit> = None;
    for _ in 0..iters {
        let i0 = rng.uniform_usize(n);
        let mut i1 = rng.uniform_usize(n);
        while i1 == i0 {
            i1 = rng.uniform_usize(n);
        }
        let mut i2 = rng.uniform_usize(n);
        while i2 == i0 || i2 == i1 {
            i2 = rng.uniform_usize(n);
        }
        let idx = [i0, i1, i2];
        let sa: Vec<Vector3<f64>> = idx.iter().map(|&i| a[i]).collect();
        let sb: Vec<Vector3<f64>> = idx.iter().map(|&i| b[i]).collect();
        let (r, t) = match kabsch(&sa, &sb) {
            Some(x) => x,
            None => continue,
        };
        // count inliers
        let mut inl_a = Vec::new();
        let mut inl_b = Vec::new();
        for i in 0..n {
            let d = (r.transform_vector(&a[i]) + t - b[i]).norm();
            if d <= thresh {
                inl_a.push(a[i]);
                inl_b.push(b[i]);
            }
        }
        let score = inl_a.len();
        if score >= 3 {
            let fit = kabsch(&inl_a, &inl_b).map(|(r2, t2)| {
                let rms = rms_of(&a, &b, &r2, &t2);
                RigidFit { r: r2, t: t2, n_inliers: score, rms }
            });
            if let Some(f) = fit {
                let better = best.as_ref().map(|b| f.n_inliers > b.n_inliers).unwrap_or(true);
                if better {
                    best = Some(f);
                }
            }
        }
    }
    best
}

fn rms_of(a: &[Vector3<f64>], b: &[Vector3<f64>], r: &Rotation3<f64>, t: &Vector3<f64>) -> f64 {
    let mut s = 0.0;
    for i in 0..a.len() {
        let d = r.transform_vector(&a[i]) + t - b[i];
        s += d.norm_squared();
    }
    (s / a.len().max(1) as f64).sqrt()
}

/// A fitted plane: `n . p + d = 0`, `n` unit.
#[derive(Debug, Clone, Copy)]
pub struct Plane {
    pub n: Vector3<f64>,
    pub d: f64,
}

impl Plane {
    pub fn signed_distance(&self, p: &Vector3<f64>) -> f64 {
        self.n.dot(p) + self.d
    }
}

/// RANSAC plane fit. Returns the plane with most inliers within `thresh`.
pub fn ransac_plane(
    pts: &[Vector3<f64>],
    thresh: f64,
    iters: usize,
    rng: &mut crate::core::rng::GaussRng,
) -> Option<Plane> {
    let n = pts.len();
    if n < 3 {
        return None;
    }
    let mut best: Option<(Plane, usize)> = None;
    for _ in 0..iters {
        let i0 = rng.uniform_usize(n);
        let mut i1 = rng.uniform_usize(n);
        while i1 == i0 {
            i1 = rng.uniform_usize(n);
        }
        let mut i2 = rng.uniform_usize(n);
        while i2 == i0 || i2 == i1 {
            i2 = rng.uniform_usize(n);
        }
        let p1 = pts[i0];
        let p2 = pts[i1];
        let p3 = pts[i2];
        let nrm = (p2 - p1).cross(&(p3 - p1));
        if nrm.norm() < 1e-9 {
            continue;
        }
        let nvec = nrm.normalize();
        let d = -nvec.dot(&p1);
        let mut count = 0usize;
        for p in pts {
            if (nvec.dot(p) + d).abs() <= thresh {
                count += 1;
            }
        }
        let better = best.map(|(_, c)| count > c).unwrap_or(true);
        if better {
            best = Some((Plane { n: nvec, d }, count));
        }
    }
    // refine on inliers with least squares (normal = smallest eigenvector)
    let (plane, _) = best?;
    let mut inl: Vec<Vector3<f64>> = pts
        .iter()
        .filter(|p| plane.signed_distance(p).abs() <= thresh)
        .copied()
        .collect();
    if inl.len() >= 3 {
        let mut c = Vector3::zeros();
        for p in &inl {
            c += p;
        }
        c /= inl.len() as f64;
        let mut cov = Matrix3::zeros();
        for p in &inl {
            let d = *p - c;
            cov += d * d.transpose();
        }
        let eig = cov.symmetric_eigen();
        // smallest eigenvalue = last (ascending order in nalgebra)
        let col = eig.eigenvectors.column(2);
        let n = Vector3::new(col[0], col[1], col[2]);
        if n.norm() > 1e-9 {
            let n = n.normalize();
            let d = -n.dot(&c);
            inl.clear();
            return Some(Plane { n, d });
        }
    }
    Some(plane)
}

/// Unit vector helper.
pub fn unit(v: Vector3<f64>) -> Vector3<f64> {
    let n = v.norm();
    if n < 1e-12 {
        Vector3::zeros()
    } else {
        v / n
    }
}

/// Wrap an angle to [-pi, pi].
pub fn wrap_pi(a: f64) -> f64 {
    let mut a = a;
    while a > std::f64::consts::PI {
        a -= 2.0 * std::f64::consts::PI;
    }
    while a < -std::f64::consts::PI {
        a += 2.0 * std::f64::consts::PI;
    }
    a
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Unit;

    #[test]
    fn kabsch_recovers_known_transform() {
        let mut rng = GaussRng::new(42);
        let axis = unit(Vector3::new(0.2, 1.0, 0.3));
        let r = Rotation3::from_axis_angle(&Unit::new_normalize(axis), 0.35);
        let t = Vector3::new(0.4, -0.2, 1.1);
        let a: Vec<Vector3<f64>> = (0..30)
            .map(|_| Vector3::new(rng.gauss(0.0, 1.5), rng.gauss(0.0, 1.5), rng.gauss(2.0, 1.5)))
            .collect();
        let b: Vec<Vector3<f64>> = a
            .iter()
            .map(|p| {
                let noisy = p + Vector3::new(rng.gauss(0.0, 0.002), rng.gauss(0.0, 0.002), rng.gauss(0.0, 0.002));
                r.transform_vector(&noisy) + t
            })
            .collect();
        let (re, te) = kabsch(&a, &b).unwrap();
        let p = Vector3::new(0.5, 0.5, 0.5);
        let err = (re.transform_vector(&p) + te - (r.transform_vector(&p) + t)).norm();
        assert!(err < 1e-3, "kabsch error {err}");
    }

    #[test]
    fn kabsch_reflection_guard() {
        // degenerate collinear points must not produce a reflection silently
        let a = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0), Vector3::new(2.0, 0.0, 0.0)];
        let b = a.clone();
        let fit = kabsch(&a, &b);
        assert!(fit.is_some());
        let (r, _) = fit.unwrap();
        assert!(r.matrix().determinant() > 0.99, "det {}", r.matrix().determinant());
    }

    #[test]
    fn ransac_rigid_with_outliers() {
        let mut rng = GaussRng::new(7);
        let r = Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::z()), 0.05);
        let t = Vector3::new(0.1, 0.0, 0.05);
        let mut a = Vec::new();
        let mut b = Vec::new();
        for i in 0..40 {
            let p = Vector3::new(
                rng.gauss(0.0, 0.5),
                rng.gauss(0.0, 0.5),
                3.0 + rng.gauss(0.0, 0.3),
            );
            let _ = i;
            a.push(p);
            b.push(r.transform_vector(&(p + Vector3::new(rng.gauss(0.0, 0.003), 0.0, 0.0))) + t);
        }
        // add gross outliers
        for _ in 0..8 {
            a.push(Vector3::new(5.0, 5.0, 5.0));
            b.push(Vector3::new(-5.0, -2.0, 1.0));
        }
        let fit = ransac_rigid(&a, &b, 0.05, 200, &mut rng).unwrap();
        assert!(fit.n_inliers >= 38, "inliers {}", fit.n_inliers);
        let p = Vector3::new(0.2, 0.1, 2.5);
        let err = (fit.r.transform_vector(&p) + fit.t - (r.transform_vector(&p) + t)).norm();
        assert!(err < 0.02, "ransac rigid err {err}");
    }

    #[test]
    fn ransac_plane_finds_ground() {
        let mut rng = GaussRng::new(11);
        let mut pts = Vec::new();
        for _ in 0..600 {
            pts.push(Vector3::new(rng.gauss(0.0, 2.0), 0.0 + rng.gauss(0.0, 0.005), rng.gauss(0.0, 2.0)));
        }
        for _ in 0..60 {
            pts.push(Vector3::new(rng.gauss(0.0, 2.0), rng.gauss(1.0, 0.3), rng.gauss(0.0, 2.0)));
        }
        let plane = ransac_plane(&pts, 0.02, 300, &mut rng).unwrap();
        assert!((plane.n[1].abs() - 1.0).abs() < 1e-3, "normal {:?}", plane.n);
        assert!(plane.d.abs() < 0.02, "d {}", plane.d);
    }
}

/// Ground-plane RANSAC with (a) a gravity prior - candidate planes must be
/// within ~60 degrees of world +y (kills vertical-surface competitors) and
/// (b) sigma-scaled inlier tolerance - far points are noisier and must not
/// be excluded by a fixed threshold. `sample` holds (point, depth-sigma).
pub fn ransac_ground(
    sample: &[(Vector3<f64>, f64)],
    base_tol: f64,
    iters: usize,
    rng: &mut crate::core::rng::GaussRng,
) -> Option<Plane> {
    let n = sample.len();
    if n < 3 {
        return None;
    }
    // sigma-scaled inlier tolerance (v3 design, kept: the fat band absorbs
    // ground measurement noise so ground remnants never bridge object bases
    // into >0.9 m ghost clusters - a v4 cap experiment regressed exactly that)
    let tol = |i: usize| (base_tol).max(2.5 * sample[i].1);
    let mut best: Option<(Plane, usize)> = None;
    for _ in 0..iters {
        let i0 = rng.uniform_usize(n);
        let mut i1 = rng.uniform_usize(n);
        while i1 == i0 {
            i1 = rng.uniform_usize(n);
        }
        let mut i2 = rng.uniform_usize(n);
        while i2 == i0 || i2 == i1 {
            i2 = rng.uniform_usize(n);
        }
        let p1 = sample[i0].0;
        let p2 = sample[i1].0;
        let p3 = sample[i2].0;
        let nrm = (p2 - p1).cross(&(p3 - p1));
        if nrm.norm() < 1e-9 {
            continue;
        }
        let mut nv = nrm.normalize();
        if nv[1] < 0.0 {
            nv = -nv;
        }
        // gravity prior: ground-like planes only
        if nv[1] < 0.5 {
            continue;
        }
        let d = -nv.dot(&p1);
        let mut count = 0usize;
        for (i, (p, _)) in sample.iter().enumerate() {
            if (nv.dot(p) + d).abs() <= tol(i) {
                count += 1;
            }
        }
        let better = best.as_ref().map(|(_, c)| count > *c).unwrap_or(true);
        if better {
            best = Some((Plane { n: nv, d }, count));
        }
    }
    let (plane, _) = best?;
    // least-squares refinement on inliers (smallest-eigenvalue normal)
    let inl: Vec<(Vector3<f64>, f64)> = sample
        .iter()
        .enumerate()
        .filter(|(i, _)| plane.signed_distance(&sample[*i].0).abs() <= tol(*i))
        .map(|(_, (p, s))| (*p, *s))
        .collect();
    if inl.len() >= 3 {
        let mut c = Vector3::zeros();
        for (p, _) in &inl {
            c += p;
        }
        c /= inl.len() as f64;
        let mut cov = Matrix3::zeros();
        for (p, _) in &inl {
            let dd = *p - c;
            cov += dd * dd.transpose();
        }
        let eig = cov.symmetric_eigen();
        let col = eig.eigenvectors.column(2);
        let mut nrm = Vector3::new(col[0], col[1], col[2]);
        if nrm.norm() > 1e-9 {
            let mut nn = nrm.normalize();
            if nn[1] < 0.0 {
                nn = -nn;
            }
            if nn[1] >= 0.5 {
                let d = -nn.dot(&c);
                return Some(Plane { n: nn, d });
            }
        }
    }
    Some(plane)
}

#[cfg(test)]
mod ground_tests {
    use super::*;
    use crate::core::rng::GaussRng;

    #[test]
    fn ransac_ground_beats_vertical_competitor() {
        // ground (noisy, far) + a large vertical panel that would otherwise
        // win a fixed-threshold RANSAC
        let mut rng = GaussRng::new(4);
        let mut sample = Vec::new();
        // near ground: 300 pts, sigma 0.01
        for _ in 0..300 {
            sample.push((Vector3::new(rng.gauss(0.0, 2.0), rng.gauss(0.0, 0.01), rng.gauss(4.0, 0.5)), 0.01));
        }
        // far ground: 500 pts, sigma 0.12
        for _ in 0..500 {
            sample.push((Vector3::new(rng.gauss(0.0, 3.0), rng.gauss(0.0, 0.12), rng.gauss(9.0, 1.5)), 0.12));
        }
        // vertical panel: 600 pts at z = 3.3
        for _ in 0..600 {
            sample.push((Vector3::new(rng.gauss(0.0, 0.4), rng.gauss(1.0, 0.6), 3.3 + rng.gauss(0.0, 0.01)), 0.01));
        }
        let plane = ransac_ground(&sample, 0.03, 200, &mut rng).unwrap();
        assert!(plane.n[1] > 0.9, "ground normal found, got {:?}", plane.n);
        assert!(plane.d.abs() < 0.05, "ground d {}", plane.d);
    }
}
