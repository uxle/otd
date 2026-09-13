//! Object detection from lifted 3D points (classical, NN-free).
//!
//! Pipeline: ground-plane RANSAC -> height gating -> voxel-grid Euclidean
//! clustering -> robust yaw-aligned oriented bounding box + class heuristic.
//!
//! Status honesty: geometric detection is WORKING (validated against GT);
//! semantic class labels are a heuristic (box/person/cylinder shape prior).
//! A YOLO-style detector can be plugged in at the `Detection` boundary.

use std::collections::HashMap;

use nalgebra::{Matrix3, Rotation3, Vector3};

use super::lifting::Point3;
use crate::core::rng::GaussRng;
use crate::core::{ransac_ground, wrap_pi};

#[derive(Debug, Clone)]
pub struct Detection {
    pub class: &'static str,
    pub center: Vector3<f64>,
    /// world-frame position covariance
    pub cov: Matrix3<f64>,
    /// full dimensions (w, h, d) in the yaw-aligned frame
    pub extents: Vector3<f64>,
    pub yaw: f64,
    pub n_points: usize,
    pub median_sigma_z: f64,
    pub frame: usize,
}

#[derive(Debug, Clone)]
pub struct DetectParams {
    /// RANSAC ground inlier distance (m)
    pub ground_thresh: f64,
    /// clustering voxel size (m)
    pub voxel: f64,
    /// neighbour radius in voxels for cluster adjacency
    pub neighbor_radius: f64,
    /// minimum points per cluster
    pub min_points: usize,
    /// minimum height above ground (m)
    pub min_height: f64,
    /// maximum height above ground (m)
    pub max_height: f64,
    /// maximum object dimension (m) - rejects walls / ceiling fragments
    pub max_extent: f64,
    pub ransac_iters: usize,
}

impl Default for DetectParams {
    fn default() -> Self {
        DetectParams {
            ground_thresh: 0.03,
            voxel: 0.06,
            neighbor_radius: 1.9,
            min_points: 20,
            min_height: 0.08,
            max_height: 3.5,
            max_extent: 3.0,
            ransac_iters: 200,
        }
    }
}

pub fn detect(points: &[Point3], params: &DetectParams, frame: usize, rng: &mut GaussRng) -> Vec<Detection> {
    if points.len() < params.min_points * 3 {
        return Vec::new();
    }
    // 1. ground plane: gravity-prior RANSAC, sigma-scaled inlier tolerance.
    // Samples carry their depth sigma; far noisy points still count as
    // inliers at their honest tolerance.
    let mut sample: Vec<(Vector3<f64>, f64)> = Vec::with_capacity(points.len());
    let step = (points.len() / 4000).max(1);
    for p in points.iter().step_by(step) {
        sample.push((p.p, p.sigma[2]));
    }
    if sample.len() < 300 {
        // fall back: include everything (starved sampler)
        sample.clear();
        for p in points.iter() {
            sample.push((p.p, p.sigma[2]));
        }
    }
    let plane = match ransac_ground(&sample, params.ground_thresh, params.ransac_iters, rng) {
        Some(p) => p,
        None => return Vec::new(),
    };
    // orient the normal "up" (positive y-ish)
    let mut n = plane.n;
    if n[1] < 0.0 {
        n = -n;
    }
    let d = plane.d * if plane.n[1] < 0.0 { -1.0 } else { 1.0 };
    // recompute d with flipped normal: n.p + d = 0 -> (-n).p - d = 0
    let d = if plane.n[1] < 0.0 { -plane.d } else { plane.d };

    // 2. height gating - sigma-aware: a point only counts as "above ground"
    // when its height exceeds the honest noise floor (far ground points with
    // large sigma-Z cannot masquerade as objects). Uncapped on purpose: the
    // fat band absorbs ground noise so it never bridges object bases (a v4
    // cap experiment regressed the crate clusters - ghost rejection ate them).
    let object_pts: Vec<&Point3> = points
        .iter()
        .filter(|p| {
            let h = n.dot(&p.p) + d;
            h > params.min_height.max(2.5 * p.sigma[2]) && h < params.max_height
        })
        .collect();
    if object_pts.len() < params.min_points {
        return Vec::new();
    }

    // 3. voxel grid
    let inv = 1.0 / params.voxel;
    let mut grid: HashMap<(i32, i32, i32), Vec<usize>> = HashMap::new();
    for (i, p) in object_pts.iter().enumerate() {
        let key = (
            (p.p[0] * inv).floor() as i32,
            (p.p[1] * inv).floor() as i32,
            (p.p[2] * inv).floor() as i32,
        );
        grid.entry(key).or_default().push(i);
    }

    // 4. connected components over voxel adjacency (radius)
    let r2 = params.neighbor_radius * params.neighbor_radius;
    let mut visited: HashMap<(i32, i32, i32), bool> = HashMap::new();
    let mut detections = Vec::new();
    let keys: Vec<(i32, i32, i32)> = grid.keys().copied().collect();
    for start in keys {
        if *visited.get(&start).unwrap_or(&false) {
            continue;
        }
        // BFS
        let mut comp_keys = Vec::new();
        let mut stack = vec![start];
        visited.insert(start, true);
        while let Some(k) = stack.pop() {
            comp_keys.push(k);
            for dx in -2i32..=2 {
                for dy in -2i32..=2 {
                    for dz in -2i32..=2 {
                        if dx == 0 && dy == 0 && dz == 0 {
                            continue;
                        }
                        let nb = (k.0 + dx, k.1 + dy, k.2 + dz);
                        if (dx as f64 * dx as f64 + dy as f64 * dy as f64 + dz as f64 * dz as f64) as f64 <= r2 + 1e-9
                            && !*visited.get(&nb).unwrap_or(&false)
                            && grid.contains_key(&nb)
                        {
                            visited.insert(nb, true);
                            stack.push(nb);
                        }
                    }
                }
            }
        }
        // gather points
        let mut idxs: Vec<usize> = Vec::new();
        for k in &comp_keys {
            idxs.extend(&grid[k]);
        }
        if idxs.len() < params.min_points {
            continue;
        }
        // ground-hugging rejection: if nearly all points are within 3 cm of
        // the plane, this is a ground remnant, not an object
        let near_ground = idxs
            .iter()
            .filter(|&&i| (n.dot(&object_pts[i].p) + d).abs() < 0.03)
            .count();
        if near_ground * 100 > idxs.len() * 95 {
            continue;
        }
        // depth-consistency rejection: clusters spanning a large depth
        // range are occlusion-boundary ghosts (mixed-surface matches), not
        // physical objects. Real objects here span <= 0.6 m in depth.
        {
            let mut zs: Vec<f64> = idxs.iter().map(|&i| object_pts[i].p[2]).collect();
            zs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let lo_idx = ((zs.len() as f64) * 0.05) as usize;
            let hi_idx = (((zs.len() as f64) * 0.95) as usize).min(zs.len() - 1);
            let z_lo = zs[lo_idx];
            let z_hi = zs[hi_idx];
            if z_hi - z_lo > 0.9 {
                continue;
            }
        }

        if let Some(det) = fit_detection(&idxs, &object_pts, params, frame) {
            detections.push(det);
        }
    }
    detections.sort_by_key(|d| std::cmp::Reverse(d.n_points));
    detections
}

/// Robust yaw-aligned OBB + class heuristic for one cluster.
fn fit_detection(
    idxs: &[usize],
    pts: &[&Point3],
    params: &DetectParams,
    frame: usize,
) -> Option<Detection> {
    let n = idxs.len();
    // median center (robust)
    let mut xs: Vec<f64> = Vec::with_capacity(n);
    let mut ys: Vec<f64> = Vec::with_capacity(n);
    let mut zs: Vec<f64> = Vec::with_capacity(n);
    let mut szs: Vec<f64> = Vec::with_capacity(n);
    for &i in idxs {
        xs.push(pts[i].p[0]);
        ys.push(pts[i].p[1]);
        zs.push(pts[i].p[2]);
        szs.push(pts[i].sigma[2]);
    }
    let med = |v: &mut Vec<f64>| {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let (cx, cy, cz) = (med(&mut xs), med(&mut ys), med(&mut zs));
    let median_sigma_z = med(&mut szs);

    // yaw from 2D PCA on (x, z)
    let mut sxx = 0.0;
    let mut szz = 0.0;
    let mut sxz = 0.0;
    for &i in idxs {
        let dx = pts[i].p[0] - cx;
        let dz = pts[i].p[2] - cz;
        sxx += dx * dx;
        szz += dz * dz;
        sxz += dx * dz;
    }
    let yaw = 0.5 * (2.0 * sxz).atan2(sxx - szz); // principal axis angle in xz

    // robust extents via percentiles in the yaw frame
    let rot = Rotation3::new(nalgebra::Vector3::new(0.0, yaw, 0.0));
    let mut along = Vec::with_capacity(n);
    let mut up = Vec::with_capacity(n);
    let mut across = Vec::with_capacity(n);
    for &i in idxs {
        let local = rot.inverse().transform_vector(&(pts[i].p - Vector3::new(cx, cy, cz)));
        along.push(local[0]);
        up.push(local[1]);
        across.push(local[2]);
    }
    let extent = |v: &mut Vec<f64>| -> f64 {
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let lo = v[(v.len() as f64 * 0.02) as usize];
        let hi = v[(v.len() as f64 * 0.98) as usize];
        hi - lo
    };
    let ex = extent(&mut along);
    let ey = extent(&mut up);
    let ez = extent(&mut across);
    let max_e = ex.max(ey).max(ez);
    if max_e > params.max_extent {
        return None; // wall / ceiling fragment
    }
    // sliver rejection
    let min_e = ex.min(ey).min(ez);
    if max_e > 0.0 && min_e / max_e < 0.04 {
        return None;
    }

    // class heuristic (honest: shape prior, not semantics)
    let class: &'static str = if ey > 1.25 && ey < 2.3 && ex.max(ez) < 0.55 {
        "person"
    } else if ey < 0.35 && ex < 0.9 && ez < 0.9 {
        "small_box"
    } else if ex.max(ez) < 0.9 && (ex / ez.max(0.01) - 1.0).abs() < 0.35 {
        "cylinder"
    } else {
        "box"
    };

    // covariance: point scatter (statistical) floored by (a) the median
    // depth sigma and (b) a systematic 0.2 m visible-surface bias - the
    // classical geometric centre of a boxy object sits on its visible
    // surface, not its volumetric centre. The covariance models this
    // honestly instead of hiding it.
    let nn = n as f64;
    let var = |v: &[f64], m: f64| v.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / nn.max(1.0);
    let vx = var(&xs, cx).max(median_sigma_z * median_sigma_z).max(0.04);
    let vy = var(&ys, cy).max(median_sigma_z * median_sigma_z * 0.25).max(0.04);
    let vz = var(&zs, cz).max(median_sigma_z * median_sigma_z).max(0.04);
    let mut cov = Matrix3::zeros();
    cov[(0, 0)] = vx;
    cov[(1, 1)] = vy;
    cov[(2, 2)] = vz;

    Some(Detection {
        class,
        center: Vector3::new(cx, cy, cz),
        cov,
        extents: Vector3::new(ex, ey, ez),
        yaw: wrap_pi(yaw),
        n_points: n,
        median_sigma_z,
        frame,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::rng::GaussRng;

    fn synthetic_scene_points(rng: &mut GaussRng) -> Vec<Point3> {
        let mut pts = Vec::new();
        // ground plane points
        for _ in 0..4000 {
            pts.push(Point3 {
                p: Vector3::new(rng.gauss(0.0, 2.5), rng.gauss(0.0, 0.005), rng.gauss(2.0, 1.2)),
                sigma: Vector3::new(0.01, 0.01, 0.02),
                quality: crate::stereo::map::Quality::Strong,
            });
        }
        // box object: 0.6 x 0.9 x 0.6 at (1.0, 0.45, 3.2)
        for _ in 0..800 {
            pts.push(Point3 {
                p: Vector3::new(
                    rng.uniform_range(0.72, 1.28),
                    rng.uniform_range(0.05, 0.88),
                    rng.uniform_range(2.92, 3.48),
                ),
                sigma: Vector3::new(0.02, 0.02, 0.03),
                quality: crate::stereo::map::Quality::Strong,
            });
        }
        // person object: 0.32 x 1.72 x 0.32 at (-0.5, 0.86, 4.2)
        for _ in 0..800 {
            pts.push(Point3 {
                p: Vector3::new(
                    rng.uniform_range(-0.65, -0.35),
                    rng.uniform_range(0.05, 1.7),
                    rng.uniform_range(4.05, 4.35),
                ),
                sigma: Vector3::new(0.02, 0.02, 0.04),
                quality: crate::stereo::map::Quality::Strong,
            });
        }
        pts
    }

    #[test]
    fn detects_two_objects_with_accuracy() {
        let mut rng = GaussRng::new(17);
        let pts = synthetic_scene_points(&mut rng);
        let dets = detect(&pts, &DetectParams::default(), 0, &mut rng);
        assert!(dets.len() >= 2, "expected >= 2 detections, got {}", dets.len());
        // match detections to expected objects by center proximity
        let box_det = dets.iter().find(|d| (d.center - Vector3::new(1.0, 0.45, 3.2)).norm() < 0.3);
        let person_det = dets.iter().find(|d| (d.center - Vector3::new(-0.5, 0.86, 4.2)).norm() < 0.3);
        let bd = box_det.expect("box detected");
        let pd = person_det.expect("person detected");
        assert_eq!(pd.class, "person", "person class: {}", pd.class);
        assert!((pd.extents[1] - 1.72).abs() < 0.25, "person height {} (true 1.72)", pd.extents[1]);
        assert!((bd.center - Vector3::new(1.0, 0.45, 3.2)).norm() < 0.15, "box center err");
        assert!((bd.extents[0] - 0.6).abs() < 0.2, "box extent x {} (true 0.6)", bd.extents[0]);
        // every detection carries a real covariance
        for d in &dets {
            assert!(d.cov[(0, 0)] > 0.0 && d.cov[(2, 2)] > 0.0);
            assert!(d.median_sigma_z > 0.0);
        }
    }

    #[test]
    fn rejects_ground_fragments() {
        // a thin layer of points just above the ground: must NOT be an object
        let mut rng = GaussRng::new(5);
        let mut pts = Vec::new();
        for _ in 0..5000 {
            pts.push(Point3 {
                p: Vector3::new(rng.gauss(0.0, 2.0), rng.gauss(0.0, 0.005), rng.gauss(2.5, 1.0)),
                sigma: Vector3::new(0.01, 0.01, 0.02),
                quality: crate::stereo::map::Quality::Strong,
            });
        }
        let dets = detect(&pts, &DetectParams::default(), 0, &mut rng);
        assert!(dets.is_empty(), "ground fragments must not become objects: {}", dets.len());
    }
}

