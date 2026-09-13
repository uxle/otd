//! Incremental voxel-hash point cloud (world frame).
//!
//! Accumulates lifted points across frames into 1-2 cm voxels with running
//! centroids; acts as the world's geometric memory and the PLY export source.

use std::collections::HashMap;

use super::lifting::Point3;

#[derive(Debug, Clone, Copy)]
struct VoxelAcc {
    count: u32,
    cx: f64,
    cy: f64,
    cz: f64,
    first_frame: u32,
    last_frame: u32,
}

pub struct VoxelCloud {
    pub voxel: f64,
    map: HashMap<(i32, i32, i32), VoxelAcc>,
    pub n_points_total: usize,
}

impl VoxelCloud {
    pub fn new(voxel: f64) -> Self {
        VoxelCloud { voxel: voxel.max(0.005), map: HashMap::new(), n_points_total: 0 }
    }

    pub fn len_voxels(&self) -> usize {
        self.map.len()
    }

    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// Insert one frame of world points.
    pub fn insert_frame(&mut self, points: &[Point3], frame: usize) {
        let inv = 1.0 / self.voxel;
        for pt in points {
            let key = (
                (pt.p[0] * inv).floor() as i32,
                (pt.p[1] * inv).floor() as i32,
                (pt.p[2] * inv).floor() as i32,
            );
            let e = self.map.entry(key).or_insert(VoxelAcc {
                count: 0,
                cx: 0.0,
                cy: 0.0,
                cz: 0.0,
                first_frame: frame as u32,
                last_frame: frame as u32,
            });
            e.count += 1;
            e.cx += pt.p[0];
            e.cy += pt.p[1];
            e.cz += pt.p[2];
            e.last_frame = frame as u32;
            self.n_points_total += 1;
        }
    }

    /// Voxel centroids as (position, count, last_frame).
    pub fn voxels(&self) -> impl Iterator<Item = ([f64; 3], u32, u32)> + '_ {
        self.map.values().map(|v| {
            let n = v.count as f64;
            ([v.cx / n, v.cy / n, v.cz / n], v.count, v.last_frame)
        })
    }

    /// PLY-ready flat point list (centroids).
    pub fn to_points(&self) -> Vec<[f32; 3]> {
        self.voxels().map(|(p, _, _)| [p[0] as f32, p[1] as f32, p[2] as f32]).collect()
    }

    /// Remove voxels seen only in a single frame before `frame` (temporal
    /// consistency filter against spurious matches).
    pub fn prune_transient(&mut self, frame: usize, min_frames: u32) -> usize {
        let before = self.map.len();
        self.map.retain(|_, v| v.last_frame as usize + 2 >= frame.saturating_sub(0) && v.count >= min_frames as u32 * 3);
        before - self.map.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;

    fn pt(x: f64, y: f64, z: f64) -> Point3 {
        Point3 { p: Vector3::new(x, y, z), sigma: Vector3::new(0.01, 0.01, 0.02), quality: crate::stereo::map::Quality::Strong }
    }

    #[test]
    fn voxel_accumulation() {
        let mut cloud = VoxelCloud::new(0.05);
        // two clusters of 50 points, 1 m apart
        for i in 0..50 {
            let j = i as f64;
            cloud.insert_frame(&[pt(0.02 + 0.0004 * (j - 25.0), 0.0, 3.0), pt(1.02 + 0.0004 * (j - 25.0), 0.0, 3.0)], 0);
        }
        assert_eq!(cloud.len_voxels(), 2, "two voxels");
        assert_eq!(cloud.n_points_total, 100);
        let pts = cloud.to_points();
        assert!(pts.iter().any(|p| (p[0] - 0.02).abs() < 0.05), "centroid 0");
        assert!(pts.iter().any(|p| (p[0] - 1.02).abs() < 0.05), "centroid 1");
    }

    #[test]
    fn world_frame_consistency() {
        // insert the same point across frames -> single voxel, centroid exact
        let mut cloud = VoxelCloud::new(0.02);
        for f in 0..10 {
            cloud.insert_frame(&[pt(0.5, 0.1, 2.0)], f);
        }
        assert_eq!(cloud.len_voxels(), 1);
        let v = cloud.to_points();
        assert_eq!(v[0], [0.5, 0.1, 2.0]);
    }
}
