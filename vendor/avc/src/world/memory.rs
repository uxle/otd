//! Object memory: where was each object at time t? (bounded history ring)

use std::collections::HashMap;

use super::model::WorldObject;

#[derive(Debug, Clone, Copy)]
struct Snapshot {
    t: f64,
    center: [f64; 3],
    velocity: [f64; 3],
}

#[derive(Debug, Default)]
pub struct ObjectMemory {
    history: HashMap<u64, Vec<Snapshot>>,
    max_len: usize,
}

impl ObjectMemory {
    pub fn new(max_len: usize) -> Self {
        ObjectMemory { history: HashMap::new(), max_len: max_len.max(4) }
    }

    pub fn observe(&mut self, objs: &[WorldObject], t: f64) {
        for o in objs {
            let h = self.history.entry(o.track_id).or_default();
            h.push(Snapshot {
                t,
                center: [o.center[0], o.center[1], o.center[2]],
                velocity: [o.velocity[0], o.velocity[1], o.velocity[2]],
            });
            if h.len() > self.max_len {
                let drain = h.len() - self.max_len;
                h.drain(0..drain);
            }
        }
    }

    /// Best estimate of an object's position at time `t` (linear interp /
    /// extrapolation with velocity from the nearest snapshot).
    pub fn position_at(&self, id: u64, t: f64) -> Option<[f64; 3]> {
        let h = self.history.get(&id)?;
        if h.is_empty() {
            return None;
        }
        // nearest snapshot
        let (i, snap) = h
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                (a.t - t).abs().partial_cmp(&(b.t - t).abs()).unwrap()
            })
            .unwrap();
        let dt = t - snap.t;
        let extrapolate = dt.abs() < 2.0;
        if extrapolate {
            Some([
                snap.center[0] + snap.velocity[0] * dt,
                snap.center[1] + snap.velocity[1] * dt,
                snap.center[2] + snap.velocity[2] * dt,
            ])
        } else {
            let _ = i;
            Some(snap.center)
        }
    }

    pub fn tracked_ids(&self) -> Vec<u64> {
        self.history.keys().copied().collect()
    }

    pub fn n_snapshots(&self, id: u64) -> usize {
        self.history.get(&id).map(|h| h.len()).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};

    fn obj(id: u64, center: Vector3<f64>, vel: Vector3<f64>) -> WorldObject {
        WorldObject {
            track_id: id,
            class: "box",
            center,
            velocity: vel,
            cov: Matrix3::identity() * 0.01,
            extents: Vector3::new(0.5, 0.5, 0.5),
            yaw: 0.0,
            confidence: 0.9,
            first_frame: 0,
            last_frame: 0,
            n_hits: 1,
        }
    }

    #[test]
    fn memory_reconstructs_past_positions() {
        let mut mem = ObjectMemory::new(50);
        for k in 0..30 {
            let t = k as f64 / 30.0;
            let x = 0.5 * t;
            mem.observe(&[obj(1, Vector3::new(x, 0.5, 3.0), Vector3::new(0.5, 0.0, 0.0))], t);
        }
        // mid-history query
        let p = mem.position_at(1, 0.5).unwrap();
        assert!((p[0] - 0.25).abs() < 0.06, "x at t=0.5: {}", p[0]);
        // near end
        let p2 = mem.position_at(1, 0.97).unwrap();
        assert!((p2[0] - 0.485).abs() < 0.08, "x at t=0.97: {}", p2[0]);
    }

    #[test]
    fn history_is_bounded() {
        let mut mem = ObjectMemory::new(10);
        for k in 0..100 {
            let t = k as f64;
            mem.observe(&[obj(1, Vector3::new(t * 0.01, 0.0, 0.0), Vector3::zeros())], t);
        }
        assert_eq!(mem.n_snapshots(1), 10);
    }
}
