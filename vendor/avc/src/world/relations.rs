//! Spatial relations between objects (and the camera), each reported with an
//! uncertainty-aware confidence.

use nalgebra::Vector3;

use super::model::WorldObject;
use crate::core::se3::Se3;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RelationKind {
    LeftOf,
    RightOf,
    InFrontOf,
    Behind,
    Above,
    Below,
    CloserThan,
    FartherThan,
    Near,
}

impl RelationKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            RelationKind::LeftOf => "left_of",
            RelationKind::RightOf => "right_of",
            RelationKind::InFrontOf => "in_front_of",
            RelationKind::Behind => "behind",
            RelationKind::Above => "above",
            RelationKind::Below => "below",
            RelationKind::CloserThan => "closer_than",
            RelationKind::FartherThan => "farther_than",
            RelationKind::Near => "near",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Relation {
    pub subject: u64,
    pub object: u64,
    pub kind: RelationKind,
    /// 0..1 - from sigma overlap (0 = uncertain, 1 = certain)
    pub confidence: f64,
    /// margin in metres (signed separation along the comparison axis)
    pub margin_m: f64,
}

/// Compare two uncertain 1D values: returns (signed difference, confidence).
/// Confidence = P(difference has the observed sign), approximated from the
/// combined sigma. ~0.5 means indistinguishable.
fn uncertain_diff(a: f64, sa: f64, b: f64, sb: f64) -> (f64, f64) {
    let d = a - b;
    let s = (sa * sa + sb * sb).sqrt().max(1e-6);
    // P(D > 0) for D ~ N(d, s): confidence in the positive sign
    let p = 0.5 * (1.0 + erf(d / (s * std::f64::consts::SQRT_2)));
    (d, (2.0 * (p - 0.5)).abs()) // 0..1 distance from coin-flip
}

fn erf(x: f64) -> f64 {
    // Abramowitz-Stegun 7.1.26
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

fn axis_sigma(cov: &nalgebra::Matrix3<f64>, axis: &Vector3<f64>) -> f64 {
    let v = cov * axis;
    (axis.dot(&v)).sqrt().max(0.01)
}

/// Compute pairwise relations among objects and relative to the camera.
/// Only relations with confidence >= 0.5 are emitted.
pub fn compute_relations(objects: &[WorldObject], cam_pose: &Se3) -> Vec<Relation> {
    let mut out = Vec::new();
    let cam_right = cam_pose.transform_dir(&Vector3::new(1.0, 0.0, 0.0));
    let cam_fwd = cam_pose.transform_dir(&Vector3::new(0.0, 0.0, 1.0));
    let up = Vector3::new(0.0, 1.0, 0.0);

    for (i, a) in objects.iter().enumerate() {
        for b in objects.iter().skip(i + 1) {
            // lateral (camera-relative) - both directions
            let (dl, cl) = uncertain_diff(
                a.center.dot(&cam_right), axis_sigma(&a.cov, &cam_right),
                b.center.dot(&cam_right), axis_sigma(&b.cov, &cam_right),
            );
            if cl >= 0.5 {
                let (left, right) = if dl < 0.0 { (a, b) } else { (b, a) };
                out.push(Relation {
                    subject: left.track_id,
                    object: right.track_id,
                    kind: RelationKind::LeftOf,
                    confidence: cl,
                    margin_m: dl.abs(),
                });
                out.push(Relation {
                    subject: right.track_id,
                    object: left.track_id,
                    kind: RelationKind::RightOf,
                    confidence: cl,
                    margin_m: dl.abs(),
                });
            }
            // depth (camera-relative) - both directions
            let (df, cf) = uncertain_diff(
                a.center.dot(&cam_fwd), axis_sigma(&a.cov, &cam_fwd),
                b.center.dot(&cam_fwd), axis_sigma(&b.cov, &cam_fwd),
            );
            if cf >= 0.5 {
                let (front, back) = if df < 0.0 { (a, b) } else { (b, a) };
                out.push(Relation {
                    subject: front.track_id,
                    object: back.track_id,
                    kind: RelationKind::InFrontOf,
                    confidence: cf,
                    margin_m: df.abs(),
                });
                out.push(Relation {
                    subject: back.track_id,
                    object: front.track_id,
                    kind: RelationKind::Behind,
                    confidence: cf,
                    margin_m: df.abs(),
                });
            }
            // vertical (world) - both directions
            let (du, cu) = uncertain_diff(
                a.center[1], axis_sigma(&a.cov, &up),
                b.center[1], axis_sigma(&b.cov, &up),
            );
            if cu >= 0.5 {
                let (top, bottom) = if du > 0.0 { (a, b) } else { (b, a) };
                out.push(Relation {
                    subject: top.track_id,
                    object: bottom.track_id,
                    kind: RelationKind::Above,
                    confidence: cu,
                    margin_m: du.abs(),
                });
                out.push(Relation {
                    subject: bottom.track_id,
                    object: top.track_id,
                    kind: RelationKind::Below,
                    confidence: cu,
                    margin_m: du.abs(),
                });
            }
            // radial range from camera - both directions
            let ra = (a.center - cam_pose.t).norm();
            let rb = (b.center - cam_pose.t).norm();
            let (dr, cr) = uncertain_diff(ra, ra * 0.02 + 0.02, rb, rb * 0.02 + 0.02);
            if cr >= 0.5 {
                let (near, far) = if dr < 0.0 { (a, b) } else { (b, a) };
                out.push(Relation {
                    subject: near.track_id,
                    object: far.track_id,
                    kind: RelationKind::CloserThan,
                    confidence: cr,
                    margin_m: dr.abs(),
                });
                out.push(Relation {
                    subject: far.track_id,
                    object: near.track_id,
                    kind: RelationKind::FartherThan,
                    confidence: cr,
                    margin_m: dr.abs(),
                });
            }
            // proximity
            let dist = (a.center - b.center).norm();
            if dist < 1.5 {
                out.push(Relation {
                    subject: a.track_id,
                    object: b.track_id,
                    kind: RelationKind::Near,
                    confidence: 0.9,
                    margin_m: dist,
                });
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Rotation3, Unit};

    fn obj(id: u64, center: Vector3<f64>) -> WorldObject {
        WorldObject {
            track_id: id,
            class: "box",
            center,
            velocity: Vector3::zeros(),
            cov: Matrix3::identity() * 0.0004,
            extents: Vector3::new(0.5, 0.5, 0.5),
            yaw: 0.0,
            confidence: 0.9,
            first_frame: 0,
            last_frame: 0,
            n_hits: 10,
        }
    }

    #[test]
    fn relations_match_geometry() {
        let cam = Se3::from_parts(
            Rotation3::from_axis_angle(&Unit::new_normalize(Vector3::y()), 0.0),
            Vector3::new(0.0, 1.6, 0.0),
        );
        // camera looks along +z (identity rotation); A left+front, B right+back
        let objs = vec![
            obj(1, Vector3::new(-1.0, 0.5, 3.0)),
            obj(2, Vector3::new(1.0, 1.0, 5.0)),
        ];
        let rels = compute_relations(&objs, &cam);
        let has = |k: RelationKind, s: u64, o: u64| rels.iter().any(|r| r.kind == k && r.subject == s && r.object == o);
        assert!(has(RelationKind::LeftOf, 1, 2), "1 left of 2");
        assert!(has(RelationKind::RightOf, 2, 1), "2 right of 1");
        assert!(has(RelationKind::InFrontOf, 1, 2), "1 in front of 2");
        assert!(has(RelationKind::Behind, 2, 1));
        assert!(has(RelationKind::CloserThan, 1, 2));
        assert!(has(RelationKind::FartherThan, 2, 1));
        assert!(has(RelationKind::Above, 2, 1), "2 above 1 (y=1.0 vs 0.5)");
        assert!(has(RelationKind::Below, 1, 2));
    }

    #[test]
    fn uncertain_positions_yield_low_confidence() {
        let cam = Se3::identity();
        let mut a = obj(1, Vector3::new(0.0, 0.0, 3.0));
        let mut b = obj(2, Vector3::new(0.05, 0.0, 3.0));
        a.cov = Matrix3::identity() * 0.04; // 20 cm sigma
        b.cov = Matrix3::identity() * 0.04;
        let rels = compute_relations(&[a, b], &cam);
        // 5 cm separation with 28 cm combined sigma: no confident lateral/depth relation
        let lateral = rels.iter().any(|r| r.kind == RelationKind::LeftOf || r.kind == RelationKind::RightOf);
        assert!(!lateral, "ambiguous lateral position must not yield a confident relation");
    }
}
