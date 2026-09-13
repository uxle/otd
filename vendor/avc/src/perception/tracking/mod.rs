//!
//! v1/v2 hard-won behaviours:
//! - **Soft class gate**: class mismatches add a cost penalty instead of
//!   forbidding association (the geometric class heuristic is unstable
//!   frame-to-frame; a hard gate churns tracks - v3 lesson).
//! - **Mahalanobis gating** before assignment.
//! - **Anchored coasting**: missed frames never move the state, only grow
//!   the covariance (occlusion survival without divergence).
//! - **P85 extents**: object dimensions are the 85th percentile of recent
//!   observations - a lower-bound statistic that resists the partial-view
//!   shrinkage which broke v1.0.
//!
//! v4 occlusion identity continuity:
//! - **Coasting tracks stay in the assignment pool** (v3 excluded them, so an
//!   object reappearing after occlusion could NEVER reclaim its ID - a young
//!   fragment won by default). Gating uses the *predicted* position (the
//!   anchored mean is stale by design) with the coast-grown covariance.
//! - **Fragment stitching**: a young track born after an older track went
//!   dark, sitting on the older track's prediction, inherits the older
//!   identity - covers fragments that slipped past the Hungarian gate.

use nalgebra::{Matrix3, Vector3};

mod hungarian;
mod kalman;

pub use hungarian::hungarian;
pub use kalman::Kalman6;

use crate::perception::detection::Detection;

#[derive(Debug, Clone)]
pub struct TrackerParams {
    /// Mahalanobis^2 gate
    pub gate_m2: f64,
    /// hits needed to confirm a track
    pub confirm_hits: usize,
    /// missed frames before termination
    pub max_missed: usize,
    /// acceleration process noise (m/s^2)
    pub sigma_acc: f64,
    /// physical velocity clamp (m/s)
    pub vel_clamp: f64,
    /// require same class for association
    pub class_gate: bool,
    /// soft penalty (Mahalanobis^2 units) for cross-class association
    pub class_penalty: f64,
    /// hard geometric association gate (m): distant objects can never steal
    /// a coasting track even when its covariance has grown
    pub hard_dist_gate: f64,
    /// extent history length
    pub extent_history: usize,
}

impl Default for TrackerParams {
    fn default() -> Self {
        TrackerParams {
            gate_m2: 12.0,
            confirm_hits: 3,
            max_missed: 70,
            sigma_acc: 1.0,
            vel_clamp: 3.0,
            class_gate: false,
            class_penalty: 6.0,
            hard_dist_gate: 1.0,
            extent_history: 25,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrackState {
    Tentative,
    Confirmed,
    Coasting,
    Terminated,
}

#[derive(Debug, Clone)]
pub struct Track {
    pub id: u64,
    pub kf: Kalman6,
    pub class: &'static str,
    pub extents: Vector3<f64>,
    pub yaw: f64,
    pub hits: usize,
    pub missed: usize,
    pub first_frame: usize,
    pub last_frame: usize,
    pub state: TrackState,
    extent_hist: Vec<Vector3<f64>>,
    /// votes for the current class (majority-switch bookkeeping)
    class_votes: usize,
}

impl Track {
    pub fn position(&self) -> Vector3<f64> {
        self.kf.position()
    }
    pub fn velocity(&self) -> Vector3<f64> {
        self.kf.velocity()
    }
    pub fn pos_cov(&self) -> Matrix3<f64> {
        self.kf.pos_cov()
    }
    pub fn confidence(&self) -> f64 {
        (self.hits as f64 / (self.hits as f64 + 3.0)).clamp(0.0, 1.0)
    }

    fn update_extents_p85(&mut self, observed: &Vector3<f64>) {
        self.extent_hist.push(*observed);
        if self.extent_hist.len() > 25 {
            let drain = self.extent_hist.len() - 25;
            self.extent_hist.drain(0..drain);
        }
        // P85 per axis (resists partial-view shrinkage)
        for ax in 0..3 {
            if !self.extent_hist.is_empty() {
                let mut vals: Vec<f64> = self.extent_hist.iter().map(|e| e[ax]).collect();
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let idx = ((vals.len() as f64 - 1.0) * 0.85).round() as usize;
                self.extents[ax] = vals[idx.min(vals.len() - 1)];
            }
        }
    }
}

pub struct Tracker {
    pub params: TrackerParams,
    tracks: Vec<Track>,
    next_id: u64,
    frame: usize,
    t: f64,
}

impl Tracker {
    pub fn new(params: TrackerParams) -> Self {
        Tracker { params, tracks: Vec::new(), next_id: 1, frame: 0, t: 0.0 }
    }

    pub fn tracks(&self) -> &[Track] {
        &self.tracks
    }

    pub fn live_tracks(&self) -> impl Iterator<Item = &Track> {
        self.tracks.iter().filter(|t| t.state != TrackState::Terminated)
    }

    /// Process one frame of detections at time `t` (seconds).
    pub fn update(&mut self, dets: &[Detection], frame: usize, t: f64, dt: f64) {
        self.frame = frame;
        self.t = t;

        // 1. predict existing tracks to t (for gating; state advances)
        for tr in self.tracks.iter_mut() {
            if tr.state == TrackState::Terminated {
                continue;
            }
            if tr.state == TrackState::Coasting {
                // anchored: grow covariance only
                tr.kf.coast_covariance(t);
            } else {
                tr.kf.predict_to(t);
            }
        }

        // 2. build cost matrix (tracks x detections)
        // v4: Coasting tracks are candidates too (prediction-aware gating);
        // v3 excluded them and reappearing objects could never reclaim IDs.
        let live: Vec<usize> = self
            .tracks
            .iter()
            .enumerate()
            .filter(|(_, tr)| tr.state != TrackState::Terminated)
            .map(|(i, _)| i)
            .collect();
        let nt = live.len();
        let nd = dets.len();

        let mut matched_tracks = vec![false; nt];
        let mut matched_dets = vec![false; nd];
        let mut assignments: Vec<(usize, usize)> = Vec::new();

        if nt > 0 && nd > 0 {
            // pad to square with dummy nodes; dummy cost = gate (allows
            // unmatched rows/cols at that price)
            let n = nt.max(nd);
            let gate = self.params.gate_m2;
            let mut cost = vec![vec![gate; n]; n];
            for (ti, &track_i) in live.iter().enumerate() {
                let tr = &self.tracks[track_i];
                for (di, d) in dets.iter().enumerate() {
                    // v4: gate against the PREDICTED position (coasting means
                    // are anchored/stale by design; the prediction is where
                    // the object should be now)
                    let dist = (tr.kf.predicted_position(t) - d.center).norm();
                    if dist > self.params.hard_dist_gate {
                        continue; // cost stays at the dummy price: forbidden
                    }
                    let mut c = tr.kf.mahalanobis2_at(&d.center, t);
                    if tr.class != d.class {
                        if self.params.class_gate {
                            c = f64::INFINITY;
                        } else {
                            c += self.params.class_penalty;
                        }
                    }
                    cost[ti][di] = c.min(gate * 2.0);
                }
            }
            let ans = hungarian(&cost, n, n);
            for (ti, &col) in ans.iter().enumerate() {
                if ti < nt && col < nd {
                    let c = cost[ti][col];
                    if c < gate {
                        assignments.push((ti, col));
                        matched_tracks[ti] = true;
                        matched_dets[col] = true;
                    }
                }
            }
        }

        // 3. apply updates
        for &(ti, di) in &assignments {
            let track_i = live[ti];
            let d = &dets[di];
            let tr = &mut self.tracks[track_i];
            // R floors: detection covariance already includes sigma floors
            tr.kf.update_pos(&d.center, &d.cov, t);
            tr.kf.clamp_velocity(self.params.vel_clamp);
            tr.hits += 1;
            tr.missed = 0;
            tr.last_frame = frame;
            tr.yaw = d.yaw;
            // majority class: switch only when the challenger wins the vote
            if d.class == tr.class {
                tr.class_votes = tr.class_votes.saturating_add(1);
            } else if tr.class_votes == 0 {
                tr.class = d.class;
                tr.class_votes = 1;
            } else {
                tr.class_votes -= 1;
            }
            tr.update_extents_p85(&d.extents);
            tr.state = if tr.hits >= self.params.confirm_hits {
                TrackState::Confirmed
            } else {
                TrackState::Tentative
            };
        }

        // 4. unmatched live tracks: coast or terminate
        for (ti, &track_i) in live.iter().enumerate() {
            if matched_tracks[ti] {
                continue;
            }
            let tr = &mut self.tracks[track_i];
            tr.missed += 1;
            if tr.missed > self.params.max_missed || tr.state == TrackState::Tentative && tr.missed > 5 {
                tr.state = TrackState::Terminated;
            } else if tr.state == TrackState::Confirmed {
                tr.state = TrackState::Coasting;
            }
        }

        // 5. spawn tentative tracks for unmatched detections
        for (di, d) in dets.iter().enumerate() {
            if matched_dets[di] {
                continue;
            }
            let kf = Kalman6::new(d.center, d.cov, self.params.sigma_acc, t);
            let tr = Track {
                id: self.next_id,
                kf,
                class: d.class,
                extents: d.extents,
                yaw: d.yaw,
                hits: 1,
                missed: 0,
                first_frame: frame,
                last_frame: frame,
                state: TrackState::Tentative,
                extent_hist: vec![d.extents],
                class_votes: 1,
            };
            self.next_id += 1;
            self.tracks.push(tr);
        }

        // 6. v4 fragment stitching: transfer occluded identities onto
        // fragments that won the Hungarian during the dark period
        self.stitch_fragments(t);

        // 7. drop long-terminated tracks from memory
        self.tracks
            .retain(|t| t.state != TrackState::Terminated || self.frame - t.last_frame < 200);

        let _ = dt;
    }

    /// v4 fragment stitching (occlusion identity continuity): a young track
    /// born AFTER an older track went dark, sitting within the stitch radius
    /// of the older track's prediction, inherits the older identity (hits and
    /// extent history merge); the older track is retired. This covers the
    /// case where a fragment track won the Hungarian during an occlusion
    /// tail and the coasting track can no longer re-associate.
    fn stitch_fragments(&mut self, t: f64) {
        let mut merges: Vec<(usize, usize)> = Vec::new(); // (young_idx, old_idx)
        for (yi, young) in self.tracks.iter().enumerate() {
            if young.state == TrackState::Terminated || young.state == TrackState::Coasting {
                continue;
            }
            for (oi, old) in self.tracks.iter().enumerate() {
                if oi == yi || old.state != TrackState::Coasting {
                    continue;
                }
                if old.last_frame >= young.first_frame {
                    continue; // not dark before the fragment appeared
                }
                if old.hits <= young.hits {
                    continue; // the older identity must carry more evidence
                }
                let pred = old.kf.predicted_position(t);
                let d = (pred - young.position()).norm();
                // geometry dominates: same-class 0.5 m, cross-class 0.35 m
                // (the class heuristic flaps on partial views)
                let lim = if old.class == young.class { 0.5 } else { 0.35 };
                if d > lim {
                    continue;
                }
                // v4.1 motion continuity: a STATIC ghost fragment must never
                // inherit a MOVING identity (and vice versa) - this is what
                // separates the real reappearance from boundary ghosts.
                let v_old = old.velocity();
                let v_young = young.velocity();
                let (so, sy) = (v_old.norm(), v_young.norm());
                if so > 0.3 {
                    if sy < 0.4 * so {
                        continue; // moving identity, stationary fragment: no
                    }
                    let dot = v_old.dot(&v_young) / (so * sy).max(1e-9);
                    if dot < 0.3 {
                        continue; // directions disagree
                    }
                }
                if sy > 0.3 && so < 0.15 * sy {
                    continue; // static identity must not absorb a mover
                }
                merges.push((yi, oi));
                break;
            }
        }
        for (yi, oi) in merges {
            let old_id = self.tracks[oi].id;
            let old_hits = self.tracks[oi].hits;
            let old_hist = std::mem::take(&mut self.tracks[oi].extent_hist);
            let old_votes = self.tracks[oi].class_votes;
            let young = &mut self.tracks[yi];
            young.id = old_id;
            young.hits += old_hits;
            young.class_votes += old_votes;
            young.extent_hist.extend(old_hist);
            // keep the P85 window bounded
            const KEEP: usize = 25;
            if young.extent_hist.len() > KEEP {
                let drop_n = young.extent_hist.len() - KEEP;
                young.extent_hist.drain(0..drop_n);
            }
            self.tracks[oi].state = TrackState::Terminated; // identity carried on
        }
    }

    /// Number of ID switches protection debug: unique live IDs count.
    pub fn n_tracks_ever(&self) -> u64 {
        self.next_id - 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::perception::detection::Detection;

    fn det(class: &'static str, center: Vector3<f64>, sigma: f64) -> Detection {
        let mut cov = Matrix3::zeros();
        cov[(0, 0)] = sigma * sigma;
        cov[(1, 1)] = sigma * sigma;
        cov[(2, 2)] = sigma * sigma;
        Detection {
            class,
            center,
            cov,
            extents: Vector3::new(0.5, 0.5, 0.5),
            yaw: 0.0,
            n_points: 200,
            median_sigma_z: sigma,
            frame: 0,
        }
    }

    #[test]
    fn two_crossing_objects_keep_ids() {
        let mut tracker = Tracker::new(TrackerParams::default());
        let dt = 1.0 / 30.0;
        let mut rng = crate::core::rng::GaussRng::new(1);
        let mut id_a: Option<u64> = None;
        let mut id_b: Option<u64> = None;
        for k in 0..60 {
            let t = k as f64 * dt;
            // A moves +x, B moves -x; they cross near the middle
            let ax = -1.0 + 2.0 * t;
            let bx = 1.0 - 2.0 * t;
            let dets = vec![
                det("box", Vector3::new(ax, 0.5, 3.0) + Vector3::new(rng.gauss(0.0, 0.02), 0.0, 0.0), 0.03),
                det("box", Vector3::new(bx, 0.5, 3.0) + Vector3::new(rng.gauss(0.0, 0.02), 0.0, 0.0), 0.03),
            ];
            tracker.update(&dets, k, t, dt);
            let live: Vec<&Track> = tracker.live_tracks().collect();
            if live.len() == 2 {
                // A is the one with x < 0 early, > 0 late (by velocity)
                let by_vel: Vec<&Track> = live.iter().filter(|tr| tr.kf.velocity()[0] > 0.1).copied().collect();
                let by_vel_neg: Vec<&Track> = live.iter().filter(|tr| tr.kf.velocity()[0] < -0.1).copied().collect();
                if k > 20 {
                    if let Some(a) = by_vel.first() {
                        id_a = Some(a.id);
                    }
                    if let Some(b) = by_vel_neg.first() {
                        id_b = Some(b.id);
                    }
                }
            }
        }
        assert!(id_a.is_some() && id_b.is_some(), "both tracks alive at end");
        assert_ne!(id_a.unwrap(), id_b.unwrap());
        // total spawned tracks should stay small (no fragmentation)
        assert!(tracker.n_tracks_ever() <= 4, "spawned {} tracks", tracker.n_tracks_ever());
    }

    #[test]
    fn occlusion_survival_and_reacquire() {
        let mut tracker = Tracker::new(TrackerParams { max_missed: 50, ..Default::default() });
        let dt = 1.0 / 30.0;
        let mut rng = crate::core::rng::GaussRng::new(2);
        // phase 1: visible, moving +x at 0.3 m/s
        for k in 0..30 {
            let t = k as f64 * dt;
            let x = 0.3 * t;
            let dets = vec![det("person", Vector3::new(x, 0.86, 4.0) + Vector3::new(rng.gauss(0.0, 0.02), 0.0, 0.0), 0.04)];
            tracker.update(&dets, k, t, dt);
        }
        let id_before = tracker.live_tracks().next().map(|t| t.id).unwrap();
        let vel = tracker.live_tracks().next().unwrap().velocity();
        assert!((vel[0] - 0.3).abs() < 0.1, "velocity learned: {vel:?}");
        // phase 2: occluded for 45 frames (0.75 m moved at 0.3 m/s = 0.225 m)
        for k in 30..75 {
            let t = k as f64 * dt;
            tracker.update(&[], k, t, dt);
        }
        let after: Vec<&Track> = tracker.live_tracks().collect();
        assert_eq!(after.len(), 1, "track survives occlusion");
        assert_eq!(after[0].id, id_before, "same ID through occlusion");
        // phase 3: reappears at the predicted place
        let t_end = 75.0 * dt;
        let x_end = 0.3 * t_end;
        let dets = vec![det("person", Vector3::new(x_end, 0.86, 4.0) + Vector3::new(rng.gauss(0.0, 0.02), 0.0, 0.0), 0.04)];
        tracker.update(&dets, 75, t_end, dt);
        let live: Vec<&Track> = tracker.live_tracks().collect();
        assert!(live.iter().any(|t| t.id == id_before), "re-acquired with the same id");
        assert!(live.len() <= 2, "no spurious track explosion: {}", live.len());
    }

    #[test]
    fn reacquire_after_long_occlusion_reuses_old_id() {
        // v4: the validation-scenario shape (0.75 m/s, 26-frame occlusion,
        // reappears at the frozen-velocity prediction). v3 failed this: the
        // coasting track was excluded from the assignment pool.
        let mut tracker = Tracker::new(TrackerParams::default());
        let dt = 1.0 / 30.0;
        let mut rng = crate::core::rng::GaussRng::new(5);
        for k in 0..30 {
            let t = k as f64 * dt;
            let dets = vec![det(
                "person",
                Vector3::new(0.75 * t, 0.86, 4.2) + Vector3::new(rng.gauss(0.0, 0.02), 0.0, 0.0),
                0.05,
            )];
            tracker.update(&dets, k, t, dt);
        }
        let id = tracker.live_tracks().next().unwrap().id;
        for k in 30..56 {
            tracker.update(&[], k, k as f64 * dt, dt);
        }
        let t = 56.0 * dt;
        let x = 0.75 * t;
        let dets = vec![det(
            "person",
            Vector3::new(x, 0.86, 4.2) + Vector3::new(rng.gauss(0.0, 0.02), 0.0, 0.0),
            0.05,
        )];
        tracker.update(&dets, 56, t, dt);
        let live: Vec<&Track> = tracker.live_tracks().collect();
        assert_eq!(live.len(), 1, "coasting track matched directly, no fragment: {}", live.len());
        assert_eq!(live[0].id, id, "same ID through the occlusion");
        assert_eq!(live[0].last_frame, 56, "the OLD track was updated at reappearance");
        assert_eq!(live[0].state, TrackState::Confirmed, "re-confirmed on reacquisition");
    }

    #[test]
    fn fragment_stitched_to_old_identity() {
        // v4: the reappearance is 0.45 m off the stale prediction with tight
        // measurement sigma -> the Mahalanobis gate rejects the coasting
        // track, a fragment spawns and wins updates -> stitching must
        // transfer the old identity to the fragment.
        let mut tracker = Tracker::new(TrackerParams::default());
        let dt = 1.0 / 30.0;
        let mut rng = crate::core::rng::GaussRng::new(6);
        for k in 0..25 {
            let t = k as f64 * dt;
            let dets = vec![det(
                "person",
                Vector3::new(0.3 * t, 0.86, 4.0) + Vector3::new(rng.gauss(0.0, 0.02), 0.0, 0.0),
                0.03,
            )];
            tracker.update(&dets, k, t, dt);
        }
        let old_id = tracker.live_tracks().next().unwrap().id;
        // 6 frames dark
        for k in 25..31 {
            tracker.update(&[], k, k as f64 * dt, dt);
        }
        // reappears 0.45 m beyond the prediction, tight sigma: gate rejects
        // the coasting track -> fragment spawns and takes over
        let x_off = 0.3 * 25.0 * dt + 0.45;
        for k in 31..37 {
            let t = k as f64 * dt;
            let dets = vec![det(
                "person",
                Vector3::new(x_off + 0.3 * (k - 31) as f64 * dt, 0.86, 4.0),
                0.02,
            )];
            tracker.update(&dets, k, t, dt);
        }
        let live: Vec<&Track> = tracker.live_tracks().collect();
        assert!(
            live.iter().any(|tr| tr.id == old_id),
            "old identity survives via stitching: {:?}",
            live.iter().map(|t| t.id).collect::<Vec<_>>()
        );
        assert!(live.len() <= 2, "no track explosion: {}", live.len());
        // the stitched track must be the one receiving updates now
        let t37 = 37.0 * dt;
        let dets = vec![det("person", Vector3::new(x_off + 0.3 * 6.0 * dt, 0.86, 4.0), 0.02)];
        tracker.update(&dets, 37, t37, dt);
        let old = tracker.live_tracks().find(|tr| tr.id == old_id).unwrap();
        assert_eq!(old.last_frame, 37, "stitched track is live and updating");
        assert_eq!(old.state, TrackState::Confirmed);
    }

    #[test]
    fn soft_class_gate_geometry_first() {
        let mut tracker = Tracker::new(TrackerParams::default());
        let dt = 1.0 / 30.0;
        // a person track at x=0 moving +x
        for k in 0..20 {
            let t = k as f64 * dt;
            let dets = vec![det("person", Vector3::new(0.3 * t, 0.86, 4.0), 0.04)];
            tracker.update(&dets, k, t, dt);
        }
        let person_id = tracker.live_tracks().next().unwrap().id;
        // a box appears EXACTLY at the person's next position: geometry
        // wins (soft gate) - the track follows it, no spurious new track
        let t = 20.0 * dt;
        let dets = vec![det("box", Vector3::new(0.3 * t, 0.86, 4.0), 0.04)];
        tracker.update(&dets, 20, t, dt);
        let live: Vec<(u64, &str)> = tracker.live_tracks().map(|tr| (tr.id, tr.class)).collect();
        assert!(live.iter().any(|(id, _)| *id == person_id), "person track alive");
        assert_eq!(live.len(), 1, "colocated box absorbed, no extra track: {live:?}");
        // majority vote keeps the class stable through one mismatch
        let person = tracker.live_tracks().next().unwrap();
        assert_eq!(person.class, "person");
        // a DISTANT box spawns its own track
        let dets2 = vec![det("box", Vector3::new(5.0, 0.5, 4.0), 0.04)];
        tracker.update(&dets2, 21, t + dt, dt);
        assert_eq!(tracker.live_tracks().count(), 2, "distant box spawns separately");
    }
}
