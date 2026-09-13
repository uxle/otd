//! P1420 — animation: spin a named part (motor rotor, flywheel, gear) or
//! turn the whole scene like a turntable. The CLI drives it:
//!
//!   otd render motor.otd f.png --spin rotor=90        (one phase)
//!   otd render motor.otd anim.png --spin rotor --frames 8
//!   otd render engine.otd side.png --turn 30          (turntable view)
//!
//! Rotation is real geometry: vertices rotate about the part's own
//! bounding-box center, so the mass, volume, and physics stay identical —
//! only the orientation changes. That is exactly what a motor does: same
//! part, new angle every frame.

use crate::math3::{Aabb, V3};
use super::eval::{Part, World};

/// Rotate point `p` about center `c` around the given axis.
fn rot_about(p: &V3, c: &V3, axis: char, rad: f64) -> V3 {
    let d = p.sub(c);
    let (s, co) = (rad.sin(), rad.cos());
    let (x, y, z) = (d.x(), d.y(), d.z());
    let q = match axis {
        'x' | 'X' => V3::new(x, y * co - z * s, y * s + z * co),
        'z' | 'Z' => V3::new(x * co - y * s, x * s + y * co, z),
        _ => V3::new(x * co + z * s, y, -x * s + z * co), // 'y' — the shaft axis
    };
    c.add(&q)
}

/// Spin ONE part around its own center (the motor-shaft case).
pub fn spin_part(part: &mut Part, axis: char, deg: f64) {
    if deg.abs() < 1e-9 || part.mesh.verts.is_empty() {
        return;
    }
    let c = part.mesh.bbox().center();
    for v in part.mesh.verts.iter_mut() {
        *v = rot_about(v, &c, axis, deg.to_radians());
    }
    part.centroid = Some(c);
}

/// Spin every part whose name matches `target` (exact, or substring).
/// Returns how many parts moved.
pub fn spin_named(world: &mut World, target: &str, axis: char, deg: f64) -> usize {
    let mut moved = 0;
    for part in world.parts.iter_mut() {
        let hit = part.name == target || part.name.contains(target);
        if hit {
            spin_part(part, axis, deg);
            moved += 1;
        }
    }
    moved
}

/// Turntable: rotate the WHOLE scene around the vertical axis through the
/// scene center — the way an engineer walks around a machine on a bench.
pub fn turn_world(world: &mut World, deg: f64) {
    if deg.abs() < 1e-9 {
        return;
    }
    let c = world.stats.bbox.center();
    for part in world.parts.iter_mut() {
        for v in part.mesh.verts.iter_mut() {
            *v = rot_about(v, &c, 'y', deg.to_radians());
        }
        part.centroid = Some(part.mesh.bbox().center());
    }
    refresh_bbox(world);
}

/// Recompute the scene bbox after any geometry move (camera reframes).
pub fn refresh_bbox(world: &mut World) {
    let mut bb = Aabb::empty();
    for part in &world.parts {
        if part.hidden {
            continue;
        }
        bb.grow_box(&part.mesh.bbox());
    }
    if !bb.is_empty() {
        world.stats.bbox = bb;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::eval::compile;

    #[test]
    fn spin_moves_vertices_not_mass() {
        let mut w = compile("scene \"t\"\nwheel = cube 10cm at (5cm, 5cm, 5cm)");
        let mass_before = w.stats.total_mass_g;
        let vol_before = w.stats.total_volume_mm3;
        let moved = spin_named(&mut w, "wheel", 'y', 37.0);
        assert_eq!(moved, 1);
        assert!((w.stats.total_mass_g - mass_before).abs() < 1e-9);
        assert!((w.stats.total_volume_mm3 - vol_before).abs() < 1e-6);
    }

    #[test]
    fn spin_360_returns_home() {
        let mut w = compile("scene \"t\"\nball = sphere 3cm at (2cm, 4cm, 6cm)");
        let p0 = w.parts[0].mesh.verts[0];
        spin_named(&mut w, "ball", 'y', 360.0);
        let p1 = w.parts[0].mesh.verts[0];
        assert!(p0.sub(&p1).len() < 1e-6, "full turn must return home");
    }

    #[test]
    fn turn_world_reframes_bbox() {
        let mut w = compile("scene \"t\"\ncube 10cm at (0, 5cm, 0)");
        let before = w.stats.bbox.size();
        turn_world(&mut w, 45.0);
        let after = w.stats.bbox.size();
        // a cube turned 45° spans wider in x/z but the same height
        assert!((before.y() - after.y()).abs() < 1e-6);
        assert!(after.x() > before.x() - 1e-6);
    }
}
