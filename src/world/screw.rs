//! P2150 — OTD3 THE SCREW — the nut-and-bolt machine, animated.
//!
//! A screw is rotation and translation LOCKED TOGETHER: one full turn
//! advances it exactly one pitch. That coupling is the whole invention —
//! Archimedes' 2300-year-old machine that trades turns for force (a 5 N
//! twist on an M6 wrench becomes ~500 N of clamp: the lever advantage of
//! the long spiral).
//!
//! The animation API (CLI `--screw`):
//!
//!   otd --png nut-bolt.otd out.png --screw nut=6@y:5mm --frames 96
//!        --cams 8 --video nut-bolt.mp4
//!
//! means: part `nut`, 6 turns, around Y, 5 mm pitch per turn — 96 frames,
//! rendered by 8 cameras from 8 directions and encoded into a real
//! H.264/AVC video. Each frame advances rotation by 360/96·k degrees and
//! translation by pitch·k/360 mm — the machine's own law.
//!
//! Righty-tighty: viewed from +Y down, positive degrees turn clockwise and
//! the part DESCENDS (negative Y translation) — exactly how a nut rides a
//! bolt in the real world.

use super::eval::{Part, World};
use crate::math3::V3;

/// Rotate point about an axis through a pivot (like anim::rot_about, but
/// with an explicit pivot so translation can compose).
fn rot_about(p: &V3, c: &V3, axis: char, rad: f64) -> V3 {
    let d = p.sub(c);
    let (s, co) = (rad.sin(), rad.cos());
    let (x, y, z) = (d.x(), d.y(), d.z());
    let q = match axis {
        'x' | 'X' => V3::new(x, y * co - z * s, y * s + z * co),
        'z' | 'Z' => V3::new(x * co - y * s, x * s + y * co, z),
        _ => V3::new(x * co + z * s, y, -x * s + z * co), // 'y'
    };
    c.add(&q)
}

/// THE screw coupling: rotate a part around its own axis center by `deg`
/// AND translate along the axis by `pitch_mm * deg / 360`.
///
/// Returns the axial travel applied (mm) — negative = descending (righty-
/// tighty with positive deg).
pub fn screw_part(part: &mut Part, axis: char, deg: f64, pitch_mm: f64) -> f64 {
    if deg.abs() < 1e-12 || part.mesh.verts.is_empty() {
        return 0.0;
    }
    // rotation happens about the part's own bbox centre
    let c = part.mesh.bbox().center();
    let rad = deg.to_radians();
    // righty-tighty: clockwise (negative mathematical rotation about +Y)
    // descends. Viewed from above (+Y), clockwise = negative angle in the
    // x-z plane; and descend = -Y translation.
    let cw = -rad; // clockwise when viewed from the +axis end
    for v in part.mesh.verts.iter_mut() {
        *v = rot_about(v, &c, axis, cw);
    }
    // translation: one pitch per full turn, descending for positive deg
    let travel = -pitch_mm * deg / 360.0;
    let t = match axis {
        'x' | 'X' => V3::new(travel, 0.0, 0.0),
        'z' | 'Z' => V3::new(0.0, 0.0, travel),
        _ => V3::new(0.0, travel, 0.0),
    };
    for v in part.mesh.verts.iter_mut() {
        *v = v.add(&t);
    }
    part.centroid = Some(part.mesh.bbox().center());
    travel
}

/// Screw every part matching `target` (exact or substring, like --spin).
/// Returns (parts moved, total axial travel mm).
pub fn screw_named(world: &mut World, target: &str, axis: char, deg: f64, pitch_mm: f64) -> (usize, f64) {
    let mut moved = 0;
    let mut travel = 0.0;
    for part in world.parts.iter_mut() {
        if part.name == target || part.name.contains(target) {
            travel += screw_part(part, axis, deg, pitch_mm);
            moved += 1;
        }
    }
    if moved > 0 {
        super::anim::refresh_bbox(world);
    }
    (moved, travel)
}

/// The mechanical advantage of a screw: the force multiplication from the
/// spiral. W = F·d at the wrench radius vs clamp force at the pitch:
/// MA = 2π·r_wrench / pitch. An M6 bolt (1 mm pitch) with a 100 mm wrench
/// gives ~628× — minus friction eats ~90% of it (which is why it HOLDS).
pub fn mechanical_advantage(wrench_radius_mm: f64, pitch_mm: f64) -> f64 {
    if pitch_mm <= 0.0 { return f64::INFINITY; }
    2.0 * std::f64::consts::PI * wrench_radius_mm / pitch_mm
}

/// Standard ISO metric coarse thread table: (nominal mm, pitch mm).
pub const METRIC_THREADS: &[(f64, f64)] = &[
    (1.6, 0.35),   // M1.6 — watch screws
    (2.0, 0.4),    // M2
    (2.5, 0.45),   // M2.5
    (3.0, 0.5),    // M3 — eyeglass hinge class
    (4.0, 0.7),    // M4
    (5.0, 0.8),    // M5
    (6.0, 1.0),    // M6 — the world's workhorse
    (8.0, 1.25),   // M8
    (10.0, 1.5),   // M10
    (12.0, 1.75),  // M12
    (16.0, 2.0),   // M16 — structural
    (20.0, 2.5),   // M20
    (24.0, 3.0),   // M24 — bridge bolts
    (30.0, 3.5),   // M30
    (36.0, 4.0),   // M36 — wind-turbine foundation class
    (42.0, 4.5),   // M42
    (48.0, 5.0),   // M48
];

/// Nearest standard thread for a diameter (snaps to the table).
pub fn nearest_metric(diameter_mm: f64) -> (f64, f64) {
    *METRIC_THREADS.iter()
        .min_by(|a, b| {
            (a.0 - diameter_mm).abs().partial_cmp(&(b.0 - diameter_mm).abs()).unwrap()
        })
        .unwrap_or(&(6.0, 1.0))
}

/// ISO V-thread geometry: depth = 0.541·pitch (truncated triangle with a
/// flat crest). The 60° flank angle is the reason bolts self-lock: the
/// friction ramp is too shallow to slide back.
pub fn thread_depth(pitch_mm: f64) -> f64 {
    0.541 * pitch_mm
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::eval::compile;

    #[test]
    fn one_turn_is_one_pitch() {
        let mut w = compile("scene \"t\"\nnut = cube 2cm at (0, 10cm, 0) material: steel");
        let y0 = w.parts[0].mesh.bbox().center().y();
        let (moved, travel) = screw_named(&mut w, "nut", 'y', 360.0, 5.0);
        assert_eq!(moved, 1);
        let y1 = w.parts[0].mesh.bbox().center().y();
        assert!((travel + 5.0).abs() < 1e-9, "one turn travels one pitch (down)");
        assert!((y0 - y1 - 5.0).abs() < 1e-6, "nut descended 5 mm");
    }

    #[test]
    fn half_turn_is_half_pitch() {
        let mut w = compile("scene \"t\"\nnut = cube 2cm at (0, 10cm, 0)");
        let (_, travel) = screw_named(&mut w, "nut", 'y', 180.0, 4.0);
        assert!((travel + 2.0).abs() < 1e-9);
    }

    #[test]
    fn rotation_and_translation_stay_coupled() {
        // after N quarter turns, total rotation = N·90° and travel = N·pitch/4
        let mut w = compile("scene \"t\"\nnut = cylinder r: 1cm h: 1cm at (0, 10cm, 0)");
        let mut total = 0.0;
        for _ in 0..7 {
            total += screw_named(&mut w, "nut", 'y', 90.0, 2.0).1;
        }
        assert!((total + 7.0 * 2.0 / 4.0).abs() < 1e-6);
    }

    #[test]
    fn mass_survives_the_screw() {
        let mut w = compile("scene \"t\"\nnut = cube 2cm at (0, 10cm, 0) material: steel");
        let m0 = w.parts[0].mass_g;
        let v0 = w.parts[0].volume_mm3;
        screw_named(&mut w, "nut", 'y', 720.0, 3.0);
        assert!((w.parts[0].mass_g - m0).abs() < 1e-9);
        assert!((w.parts[0].volume_mm3 - v0).abs() < 1e-6);
    }

    #[test]
    fn m6_is_one_millimetre_pitch() {
        assert!((nearest_metric(6.0).1 - 1.0).abs() < 1e-9);
        assert!((nearest_metric(6.2).0 - 6.0).abs() < 1e-9); // snaps
        assert!(nearest_metric(100.0).1 > 1.0);
    }

    #[test]
    fn thread_depth_follows_iso() {
        assert!((thread_depth(1.0) - 0.541).abs() < 1e-9);
    }

    #[test]
    fn mechanical_advantage_is_the_lever() {
        // 100 mm wrench on M6: 2π·100/1 ≈ 628×
        let ma = mechanical_advantage(100.0, 1.0);
        assert!((ma - 628.3).abs() < 1.0);
        assert!(mechanical_advantage(100.0, 5.0) < ma);
    }

    #[test]
    fn thread_mesh_builds() {
        let m = crate::geo::builders::thread(3.0, 1.0, 8.0, 0.541);
        assert!(!m.verts.is_empty());
        assert!(!m.tris.is_empty());
        // crest must be outside root
        let bb = m.bbox();
        assert!(bb.size().x() > 6.0, "major diameter > root diameter 6mm");
    }
}
