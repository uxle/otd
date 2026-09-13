//! The measurement engine: metric distances between tracked objects with
//! propagated sigmas. This is the product the whole pipeline exists for:
//! "how far is A from B, and how much do we trust it?"

use nalgebra::Vector3;

use super::model::WorldModel;
use crate::core::uncertainty::sigma_of_distance;

#[derive(Debug, Clone, Copy)]
pub struct Measurement {
    pub a: u64,
    pub b: u64,
    /// centre-to-centre distance (m)
    pub center_distance: f64,
    /// 1-sigma of the centre distance
    pub sigma: f64,
    /// approximate surface-to-surface distance (m), centre distance minus
    /// both effective radii (honest: bounding-sphere approximation)
    pub surface_distance: f64,
    /// 3-sigma interval
    pub interval_3sigma: (f64, f64),
    pub frame: usize,
}

#[derive(Debug, Clone, Copy)]
pub struct MeasurementEngine {
    /// measurements are only offered for objects seen recently
    pub max_age_frames: usize,
}

impl Default for MeasurementEngine {
    fn default() -> Self {
        MeasurementEngine { max_age_frames: 5 }
    }
}

impl MeasurementEngine {
    /// Measure centre and surface distance between two tracked objects.
    pub fn measure(&self, world: &WorldModel, a: u64, b: u64) -> Option<Measurement> {
        let oa = world.get(a)?;
        let ob = world.get(b)?;
        self.measure_objects(oa, ob, world.frame)
    }

    pub fn measure_objects(&self, oa: &super::model::WorldObject, ob: &super::model::WorldObject, frame: usize) -> Option<Measurement> {
        let d = (oa.center - ob.center).norm();
        let s = sigma_of_distance(&oa.center, &oa.cov, &ob.center, &ob.cov).max(0.01);
        let surf = (d - oa.effective_radius() - ob.effective_radius()).max(0.0);
        Some(Measurement {
            a: oa.track_id,
            b: ob.track_id,
            center_distance: d,
            sigma: s,
            surface_distance: surf,
            interval_3sigma: ((d - 3.0 * s).max(0.0), d + 3.0 * s),
            frame,
        })
    }

    /// Measure the distance from the camera to an object.
    pub fn measure_from_camera(&self, world: &WorldModel, id: u64) -> Option<(f64, f64)> {
        let o = world.get(id)?;
        let cam = world.cam_pose?;
        let d = (o.center - cam.t).norm();
        // camera position is exact (GT) or VO-uncertain; use a 2% + 2 cm model
        let s = (d * 0.02 + 0.02 + o.cov[(2, 2)].sqrt()).max(0.01);
        Some((d, s))
    }

    /// All pairwise measurements among confident objects (bounded: at most
    /// the 10 closest pairs).
    pub fn all_measurements(&self, world: &WorldModel) -> Vec<Measurement> {
        let mut out = Vec::new();
        let objs: Vec<_> = world
            .objects
            .iter()
            .filter(|o| world.frame.saturating_sub(o.last_frame) <= self.max_age_frames && o.confidence > 0.5)
            .collect();
        let mut pairs: Vec<(f64, usize, usize)> = Vec::new();
        for i in 0..objs.len() {
            for j in i + 1..objs.len() {
                let d = (objs[i].center - objs[j].center).norm();
                pairs.push((d, i, j));
            }
        }
        pairs.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap());
        for (_, i, j) in pairs.into_iter().take(10) {
            if let Some(m) = self.measure_objects(objs[i], objs[j], world.frame) {
                out.push(m);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::model::WorldObject;
    use nalgebra::{Matrix3, Vector3};

    fn obj(id: u64, center: Vector3<f64>, sigma: f64) -> WorldObject {
        WorldObject {
            track_id: id,
            class: "box",
            center,
            velocity: Vector3::zeros(),
            cov: Matrix3::identity() * sigma * sigma,
            extents: Vector3::new(0.4, 0.4, 0.4),
            yaw: 0.0,
            confidence: 0.9,
            first_frame: 0,
            last_frame: 0,
            n_hits: 10,
        }
    }

    #[test]
    fn measurement_matches_truth_and_carries_sigma() {
        let mut world = WorldModel { frame: 5, ..Default::default() };
        world.objects = vec![
            obj(1, Vector3::new(0.0, 0.5, 3.0), 0.02),
            obj(2, Vector3::new(3.0, 0.5, 3.0), 0.03),
        ];
        let eng = MeasurementEngine::default();
        let m = eng.measure(&world, 1, 2).unwrap();
        assert!((m.center_distance - 3.0).abs() < 1e-9, "distance {}", m.center_distance);
        // sigma ~ sqrt(0.02^2 + 0.03^2) = 0.036 along the axis
        assert!(m.sigma > 0.02 && m.sigma < 0.06, "sigma {}", m.sigma);
        // 3-sigma interval covers the truth
        assert!(m.interval_3sigma.0 < 3.0 && m.interval_3sigma.1 > 3.0);
        // surface distance = 3 - 0.2 - 0.2
        assert!((m.surface_distance - 2.6).abs() < 1e-9, "surface {}", m.surface_distance);
    }
}
