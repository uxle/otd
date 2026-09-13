//! P1190 — the `rope` word: a cable hung between two points, *solved* by
//! XPBD under the scene's gravity, then swept into a tube mesh. The sag is
//! real physics — a hanging chain converges to the catenary, and `ask
//! "cable sag?"` reports the measured dip.

use crate::geo::builders;
use crate::math3::V3;
use crate::world::xpbd;

/// Build a rope from `from` to `to` (mm). `thickness` is the tube radius.
/// `sag`: Some(target) — rest length derived from the catenary equation;
/// None — 6 % longer than the span (a gentle natural hang).
/// `gravity` in mm/ms² (engine canonical). Returns (mesh, measured sag).
pub fn rope(
    from: V3,
    to: V3,
    thickness: f64,
    sag: Option<f64>,
    gravity: f64,
) -> (crate::geo::mesh::Mesh, f64) {
    let span = to.sub(&from).len();
    if span < 1e-6 {
        return (crate::geo::mesh::Mesh::new(), 0.0);
    }
    let rest = match sag {
        Some(s) if s > 1e-6 => xpbd::length_for_sag(span, s.min(span * 0.9)),
        _ => span * 1.06,
    };
    let pts = xpbd::solve_rope(from, to, 48, rest, gravity);
    let measured = xpbd::sag_of(&pts);
    // tube along the solved polyline — smooth scaled to thickness
    let smooth = if thickness > 2.0 { 10 } else { 8 };
    let mesh = builders::tube(&pts, thickness.max(0.2), smooth);
    (mesh, measured)
}

#[cfg(test)]
mod tests {
    use super::*;

    const G: f64 = 0.00981; // mm/ms²

    #[test]
    fn rope_hangs_and_ends_pinned() {
        let (m, sag) = rope(V3::ZERO, V3::new(100.0, 0.0, 0.0), 2.0, Some(10.0), G);
        assert!(sag > 5.0 && sag < 15.0, "sag {}", sag);
        assert!(!m.tris.is_empty());
        assert!(m.volume_signed() > 0.0);
        // the mesh bbox must dip below the endpoints' y
        let bb = m.bbox();
        assert!(bb.min.y() < -5.0, "rope must visibly sag, bbox min y {}", bb.min.y());
        // and span the full width
        assert!(bb.max.x() > 99.0 && bb.min.x() < 1.0);
    }

    #[test]
    fn no_target_sag_is_gentle() {
        let (_, sag) = rope(V3::ZERO, V3::new(100.0, 0.0, 0.0), 2.0, None, G);
        // 6 % extra length → small sag (catenary ≈ √(1.06·d·s) ish)
        assert!(sag > 0.5 && sag < 25.0, "natural sag {}", sag);
    }

    #[test]
    fn shape_is_gravity_independent() {
        // ideal cables: the catenary follows the LENGTH; g sets only tension
        let (_, earth) = rope(V3::ZERO, V3::new(80.0, 0.0, 0.0), 2.0, Some(8.0), G);
        let (_, moon) = rope(V3::ZERO, V3::new(80.0, 0.0, 0.0), 2.0, Some(8.0), G * 1.62 / 9.81);
        assert!((earth - moon).abs() / earth < 0.12, "moon {} vs earth {}", moon, earth);
    }
}
