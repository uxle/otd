//! JSON world-state export (serde).

use serde::Serialize;

use crate::core::se3::Se3;
use crate::world::measurement::Measurement;
use crate::world::model::WorldModel;
use crate::world::relations::Relation;

#[derive(Serialize)]
pub struct WorldJson {
    pub version: String,
    pub frame: usize,
    pub t: f64,
    pub camera: CameraJson,
    pub objects: Vec<ObjectJson>,
    pub relations: Vec<RelationJson>,
    pub measurements: Vec<MeasurementJson>,
}

#[derive(Serialize)]
pub struct CameraJson {
    /// cam-to-world 4x4 (row major)
    pub pose: [f64; 16],
    pub sigma_rot: f64,
    pub sigma_trans: f64,
}

#[derive(Serialize)]
pub struct ObjectJson {
    pub track_id: u64,
    pub class: String,
    pub center: [f64; 3],
    pub velocity: [f64; 3],
    pub sigma: [f64; 3],
    pub cov: [f64; 9],
    pub extents: [f64; 3],
    pub yaw: f64,
    pub confidence: f64,
    pub first_frame: usize,
    pub last_frame: usize,
    pub speed: f64,
}

#[derive(Serialize)]
pub struct RelationJson {
    pub subject: u64,
    pub object: u64,
    pub kind: String,
    pub confidence: f64,
    pub margin_m: f64,
}

#[derive(Serialize)]
pub struct MeasurementJson {
    pub a: u64,
    pub b: u64,
    pub center_distance: f64,
    pub surface_distance: f64,
    pub sigma: f64,
    pub interval_3sigma: [f64; 2],
}

pub fn world_to_json(
    world: &WorldModel,
    relations: &[Relation],
    measurements: &[Measurement],
    vo_sigma: (f64, f64),
) -> WorldJson {
    let pose = world.cam_pose.unwrap_or_default();
    WorldJson {
        version: crate::VERSION.to_string(),
        frame: world.frame,
        t: world.t,
        camera: CameraJson {
            pose: pose_to_row_major(&pose),
            sigma_rot: vo_sigma.0,
            sigma_trans: vo_sigma.1,
        },
        objects: world
            .objects
            .iter()
            .map(|o| ObjectJson {
                track_id: o.track_id,
                class: o.class.to_string(),
                center: [o.center[0], o.center[1], o.center[2]],
                velocity: [o.velocity[0], o.velocity[1], o.velocity[2]],
                sigma: [o.cov[(0, 0)].sqrt(), o.cov[(1, 1)].sqrt(), o.cov[(2, 2)].sqrt()],
                cov: cov_row_major(&o.cov),
                extents: [o.extents[0], o.extents[1], o.extents[2]],
                yaw: o.yaw,
                confidence: o.confidence,
                first_frame: o.first_frame,
                last_frame: o.last_frame,
                speed: o.speed(),
            })
            .collect(),
        relations: relations
            .iter()
            .map(|r| RelationJson {
                subject: r.subject,
                object: r.object,
                kind: r.kind.as_str().to_string(),
                confidence: r.confidence,
                margin_m: r.margin_m,
            })
            .collect(),
        measurements: measurements
            .iter()
            .map(|m| MeasurementJson {
                a: m.a,
                b: m.b,
                center_distance: m.center_distance,
                surface_distance: m.surface_distance,
                sigma: m.sigma,
                interval_3sigma: [m.interval_3sigma.0, m.interval_3sigma.1],
            })
            .collect(),
    }
}

fn pose_to_row_major(p: &Se3) -> [f64; 16] {
    let h = p.to_homogeneous();
    let mut out = [0f64; 16];
    let mut k = 0;
    for row in h.iter() {
        for v in row.iter() {
            out[k] = *v;
            k += 1;
        }
    }
    out
}

fn cov_row_major(c: &nalgebra::Matrix3<f64>) -> [f64; 9] {
    let mut out = [0f64; 9];
    let mut k = 0;
    for i in 0..3 {
        for j in 0..3 {
            out[k] = c[(i, j)];
            k += 1;
        }
    }
    out
}

pub fn write_world_json(path: &std::path::Path, wj: &WorldJson) -> std::io::Result<()> {
    let s = serde_json::to_string_pretty(wj).map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
    std::fs::write(path, s)
}

/// Measurements as CSV rows (id_a, id_b, frame, center_dist, surface_dist, sigma, lo3s, hi3s).
pub fn measurements_csv(measurements: &[Measurement]) -> String {
    let mut s = String::from("frame,id_a,id_b,center_distance_m,surface_distance_m,sigma_m,lo_3sigma_m,hi_3sigma_m\n");
    for m in measurements {
        s.push_str(&format!(
            "{},{},{},{:.4},{:.4},{:.4},{:.4},{:.4}\n",
            m.frame, m.a, m.b, m.center_distance, m.surface_distance, m.sigma, m.interval_3sigma.0, m.interval_3sigma.1
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix3, Vector3};
    use crate::world::model::WorldObject;
    use crate::world::relations::RelationKind;

    #[test]
    fn json_export_contains_everything() {
        let mut world = WorldModel { frame: 12, t: 0.4, ..Default::default() };
        world.cam_pose = Some(Se3::identity());
        world.objects = vec![WorldObject {
            track_id: 1,
            class: "box",
            center: Vector3::new(1.0, 0.5, 3.0),
            velocity: Vector3::new(0.1, 0.0, 0.0),
            cov: Matrix3::identity() * 0.01,
            extents: Vector3::new(0.6, 0.6, 0.6),
            yaw: 0.0,
            confidence: 0.9,
            first_frame: 0,
            last_frame: 12,
            n_hits: 10,
        }];
        let rels = vec![Relation {
            subject: 1,
            object: 2,
            kind: RelationKind::LeftOf,
            confidence: 0.8,
            margin_m: 1.2,
        }];
        let meas = vec![Measurement {
            a: 1,
            b: 2,
            center_distance: 2.5,
            sigma: 0.05,
            surface_distance: 2.0,
            interval_3sigma: (2.35, 2.65),
            frame: 12,
        }];
        let wj = world_to_json(&world, &rels, &meas, (0.01, 0.05));
        let s = serde_json::to_string_pretty(&wj).unwrap();
        assert!(s.contains("\"track_id\": 1"));
        assert!(s.contains("\"class\": \"box\""));
        assert!(s.contains("\"left_of\""));
        assert!(s.contains("\"center_distance\": 2.5"));
        assert!(s.contains("\"sigma\""));
        // parses back
        let v: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["objects"].as_array().unwrap().len(), 1);
        assert_eq!(v["frame"].as_u64().unwrap(), 12);
    }
}
