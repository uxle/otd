//! World objects and the frame-by-frame world state.

use nalgebra::{Matrix3, Vector3};

use crate::perception::tracking::{Track, TrackState};

#[derive(Debug, Clone)]
pub struct WorldObject {
    pub track_id: u64,
    pub class: &'static str,
    /// world-frame centre
    pub center: Vector3<f64>,
    /// world-frame velocity (m/s)
    pub velocity: Vector3<f64>,
    /// position covariance
    pub cov: Matrix3<f64>,
    /// full dimensions (w, h, d) in the yaw frame
    pub extents: Vector3<f64>,
    pub yaw: f64,
    pub confidence: f64,
    pub first_frame: usize,
    pub last_frame: usize,
    pub n_hits: usize,
}

impl WorldObject {
    /// Equivalent-sphere radius for surface-distance approximations.
    pub fn effective_radius(&self) -> f64 {
        (self.extents[0].min(self.extents[1]).min(self.extents[2])) / 2.0
    }

    pub fn speed(&self) -> f64 {
        self.velocity.norm()
    }
}

#[derive(Debug, Clone, Default)]
pub struct WorldModel {
    pub objects: Vec<WorldObject>,
    pub frame: usize,
    pub t: f64,
    /// cam-to-world pose of the (rectified left) camera this frame.
    pub cam_pose: Option<crate::core::se3::Se3>,
}

impl WorldModel {
    /// Build the world state from live (non-terminated) tracks.
    pub fn from_tracks(tracks: &[Track], frame: usize, t: f64, cam_pose: crate::core::se3::Se3) -> Self {
        let objects = tracks
            .iter()
            .filter(|tr| tr.state != TrackState::Terminated && tr.hits >= 2)
            .map(|tr| WorldObject {
                track_id: tr.id,
                class: tr.class,
                center: tr.position(),
                velocity: tr.velocity(),
                cov: tr.pos_cov(),
                extents: tr.extents,
                yaw: tr.yaw,
                confidence: tr.confidence(),
                first_frame: tr.first_frame,
                last_frame: tr.last_frame,
                n_hits: tr.hits,
            })
            .collect();
        WorldModel { objects, frame, t, cam_pose: Some(cam_pose) }
    }

    pub fn get(&self, id: u64) -> Option<&WorldObject> {
        self.objects.iter().find(|o| o.track_id == id)
    }

    pub fn object_ids(&self) -> Vec<u64> {
        self.objects.iter().map(|o| o.track_id).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn world_objects_carry_everything() {
        // constructed indirectly in integration tests; here just sanity
        let o = WorldObject {
            track_id: 7,
            class: "box",
            center: Vector3::new(1.0, 0.5, 3.0),
            velocity: Vector3::new(0.2, 0.0, 0.0),
            cov: Matrix3::identity() * 0.01,
            extents: Vector3::new(0.6, 0.6, 0.6),
            yaw: 0.0,
            confidence: 0.9,
            first_frame: 0,
            last_frame: 10,
            n_hits: 11,
        };
        assert!((o.speed() - 0.2).abs() < 1e-9);
        assert!((o.effective_radius() - 0.3).abs() < 1e-9);
    }
}
