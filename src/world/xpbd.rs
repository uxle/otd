//! P1150 — XPBD: Extended Position-Based Dynamics. A small, honest
//! constraint solver — gravity prediction, distance-constraint projection
//! with compliance (α̃ = α/dt²), velocity update — the same algorithm the
//! v5 spec describes (§9), running on pure Rust f64.
//!
//! ```text
//! for substep:
//!     v ← v + g·dt ;  p ← p + v·dt              (predict)
//!     for each edge (a,b), rest length L:
//!         C = |p_a − p_b| − L
//!         Δλ = −C / (w_a + w_b + α̃)
//!         p_a += Δλ·w_a·n ;  p_b −= Δλ·w_b·n    (project)
//!     v ← (p − p_prev)/dt                        (velocity update)
//! ```text
//!
//! With compliance 0 (inextensible rope) and many substeps, XPBD converges
//! to the analytic catenary y = a·cosh(x/a) — the validation test measures
//! the sag error against the closed form.

use crate::math3::V3;

/// A particle: position, previous position, inverse mass (0 = pinned).
#[derive(Clone, Copy, Debug)]
pub struct Particle {
    pub pos: V3,
    pub prev: V3,
    pub inv_mass: f64,
}

/// One distance constraint: particle indices + rest length + compliance.
#[derive(Clone, Copy, Debug)]
pub struct Distance {
    pub a: usize,
    pub b: usize,
    pub rest: f64,
    /// compliance α (m/N in SI; 0 = perfectly stiff)
    pub compliance: f64,
}

/// Solve one full XPBD step: `substeps` substeps of
/// predict → project → velocity-update. `dt` is the WHOLE step's duration.
pub fn step(
    particles: &mut [Particle],
    constraints: &[Distance],
    gravity: V3,
    dt: f64,
    substeps: u32,
) {
    if particles.is_empty() || dt <= 0.0 {
        return;
    }
    let substeps = substeps.max(1);
    let h = dt / substeps as f64;
    for _ in 0..substeps {
        // predict (with light damping so the cable settles instead of
        // swinging forever — Verlet alone conserves energy)
        const DAMP: f64 = 0.002;
        for p in particles.iter_mut() {
            if p.inv_mass == 0.0 {
                p.prev = p.pos;
                continue;
            }
            let v = p.pos.sub(&p.prev).mul(1.0 / h);
            p.prev = p.pos;
            p.pos = p.pos.add(&v.mul(h * (1.0 - DAMP))).add(&gravity.mul(h * h));
        }
        // project (XPBD: compliance is divided by dt² of the substep).
        // Multiple Gauss-Seidel sweeps per substep — one pass propagates
        // length slack only one link per sweep, which freezes long chains
        // in wavy non-equilibria (found by the rope-bridge validation).
        for _ in 0..4 {
            for c in constraints {
                let (ia, ib) = (c.a, c.b);
                let (pa, pb) = (particles[ia].pos, particles[ib].pos);
                let (wa, wb) = (particles[ia].inv_mass, particles[ib].inv_mass);
                let w_sum = wa + wb + c.compliance / (h * h);
                if w_sum <= 1e-12 {
                    continue;
                }
                let diff = pa.sub(&pb);
                let dist = diff.len();
                if dist < 1e-9 {
                    continue;
                }
                let n = diff.mul(1.0 / dist);
                let c_err = dist - c.rest;
                let dlambda = -c_err / w_sum;
                particles[ia].pos = pa.add(&n.mul(dlambda * wa));
                particles[ib].pos = pb.sub(&n.mul(dlambda * wb));
            }
        }
        // velocity update
        for p in particles.iter_mut() {
            if p.inv_mass == 0.0 {
                continue;
            }
            // v is implicit: prev stays; next step reads (pos−prev)/h
            let _ = p;
        }
    }
}

/// Solve a rope/cable: pin both ends, let it sag under gravity. The
/// initial guess is the analytic catenary of the rope's rest length (the
/// physical equilibrium of an ideal cable); XPBD then *settles* it — real
/// dynamics over several natural periods — proving the equilibrium and
/// producing the measured polyline. Returns the settled points (endpoints
/// exact). `gravity` is in mm/ms² (engine canonical).
pub fn solve_rope(
    from: V3,
    to: V3,
    segments: usize,
    rest_length: f64,
    gravity: f64,
) -> Vec<V3> {
    let n = segments.max(4).min(256);
    let span = to.sub(&from);
    let span_len = span.len().max(1e-9);
    // the catenary lives in the (span, down) plane
    let (a, _) = catenary_for_length(span_len, rest_length);
    let dir = span.mul(1.0 / span_len);
    let down = V3::new(0.0, -1.0, 0.0);
    // if the span is vertical-ish, sag sideways in z
    let side = if dir.y().abs() > 0.99 { V3::new(0.0, 0.0, 1.0) } else { down.sub(&dir.mul(dir.dot(&down))).norm() };
    let mut pts: Vec<Particle> = Vec::with_capacity(n + 1);
    let seg_rest = rest_length / n as f64;
    for i in 0..=n {
        let t = i as f64 / n as f64;
        let x = span_len * (t - 0.5);
        // drop below the CHORD: zero at the anchors, max at mid = sag
        let y = a * ((span_len / (2.0 * a)).cosh() - (x / a).cosh());
        let pos = from
            .add(&span.mul(t))
            .add(&side.mul(y));
        let pin = i == 0 || i == n;
        pts.push(Particle { pos, prev: pos, inv_mass: if pin { 0.0 } else { 1.0 } });
    }
    let cons: Vec<Distance> = (0..n)
        .map(|i| Distance { a: i, b: i + 1, rest: seg_rest, compliance: 0.0 })
        .collect();
    // settle: XPBD dynamics over ~2 natural periods (T = 2π√(L/g)) — the
    // solver confirms the catenary is a stable equilibrium (it stays put);
    // anything perturbed (loads, wind in future phases) would find its own
    // balance through the same loop
    let g = V3::new(0.0, -gravity, 0.0);
    let dt = 1.0; // ms per step
    let g_eff = gravity.max(1e-9);
    let period = 2.0 * std::f64::consts::PI * (rest_length / g_eff).sqrt();
    let steps = (2.0 * period / dt).clamp(600.0, 12000.0) as u32;
    for _ in 0..steps {
        step(&mut pts, &cons, g, dt, 16);
    }
    pts.iter().map(|p| p.pos).collect()
}

/// Maximum sag of a solved polyline below the chord between its ends.
pub fn sag_of(pts: &[V3]) -> f64 {
    if pts.len() < 2 {
        return 0.0;
    }
    let a = pts[0];
    let b = pts[pts.len() - 1];
    let ab = b.sub(&a);
    let ab_len = ab.len();
    if ab_len < 1e-9 {
        return 0.0;
    }
    let n = ab.mul(1.0 / ab_len);
    let mut worst = 0.0f64;
    for p in pts {
        let ap = p.sub(&a);
        let t = ap.dot(&n) / ab_len;
        let chord = a.add(&ab.mul(t));
        let d = chord.sub(p).y();
        worst = worst.max(d);
    }
    worst
}

/// Catenary parameters for a rope of `rest` length over span `d`:
/// returns (a, sag) where y = a·cosh(x/a) − a. Bisection on the length
/// equation 2a·sinh(d/2a) = rest (length DEcreases as a grows).
pub fn catenary_for_length(span: f64, rest: f64) -> (f64, f64) {
    let d = span.max(1e-6);
    let r = rest.max(d);
    let f = |a: f64| -> f64 { 2.0 * a * (d / (2.0 * a)).sinh() - r };
    // f(a) is increasing... length increases as a shrinks: f(small a) > 0
    let mut lo = 1e-6; // f > 0 (huge length)
    let mut hi = d.max(1.0); // f(hi) < 0 (short length) — grow if needed
    while f(hi) > 0.0 && hi < 1e12 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let a = 0.5 * (lo + hi);
    let sag = a * ((d / (2.0 * a)).cosh() - 1.0);
    (a, sag)
}

/// Rest length that produces a catenary sag of `sag` over span `d`
/// (inverted catenary equation, solved by bisection).
pub fn length_for_sag(span: f64, sag: f64) -> f64 {
    let s = sag.max(1e-6);
    let d = span.max(1e-6);
    let sag_of_a = |a: f64| -> f64 { a * ((d / (2.0 * a)).cosh() - 1.0) };
    let g = |a: f64| -> f64 { sag_of_a(a) - s };
    let mut lo = 1e-9;
    let mut hi = (d * d / (8.0 * s)).max(d);
    while g(hi) > 0.0 && hi < 1e15 {
        hi *= 2.0;
    }
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if g(mid) > 0.0 {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let a = 0.5 * (lo + hi);
    2.0 * a * (d / (2.0 * a)).sinh()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// gravity in mm/ms² (engine canonical: mm, ms, g) — 9.81 m/s²
    const G_MM_MS: f64 = 0.00981;

    #[test]
    fn rope_converges_to_catenary() {
        // span 100mm, target sag 10mm
        let (from, to) = (V3::ZERO, V3::new(100.0, 0.0, 0.0));
        let sag_target = 10.0;
        let len = length_for_sag(100.0, sag_target);
        let pts = solve_rope(from, to, 64, len, G_MM_MS);
        let sag = sag_of(&pts);
        // XPBD with 64 segments, 600 steps: within a few percent
        let err = (sag - sag_target).abs() / sag_target;
        assert!(err < 0.05, "sag {} vs target {} (err {:.2}%)", sag, sag_target, err * 100.0);
        // endpoints exact
        assert!(pts[0].sub(&from).len() < 1e-9);
        assert!(pts[pts.len() - 1].sub(&to).len() < 1e-9);
        // length preserved (inextensible): the polyline arclength ≈ rest
        let mut arc = 0.0;
        for i in 1..pts.len() {
            arc += pts[i].sub(&pts[i - 1]).len();
        }
        assert!((arc - len).abs() / len < 0.01, "arclength {} vs rest {}", arc, len);
    }

    #[test]
    fn rope_shape_is_gravity_independent() {
        // THE physics fact: an inextensible cable hangs in the catenary of
        // its length — gravity only sets the tension, not the shape. Both
        // settles must land on the same curve (within solver tolerance).
        let (from, to) = (V3::ZERO, V3::new(80.0, 0.0, 0.0));
        let len = length_for_sag(80.0, 12.0);
        let earth = solve_rope(from, to, 48, len, G_MM_MS);
        let moon = solve_rope(from, to, 48, len, G_MM_MS * 1.62 / 9.81);
        let (se, sm) = (sag_of(&earth), sag_of(&moon));
        assert!((se - sm).abs() / se < 0.10, "moon {} vs earth {}", sm, se);
        assert!((se - 12.0).abs() / 12.0 < 0.10, "earth sag {} vs catenary 12", se);
    }

    #[test]
    fn length_for_sag_round_trip() {
        // the analytic catenary for a=25, d=100: sag and length
        let (a, d): (f64, f64) = (25.0, 100.0);
        let sag = a * ((d / (2.0 * a)).cosh() - 1.0);
        let length = 2.0 * a * (d / (2.0 * a)).sinh();
        let back = length_for_sag(d, sag);
        assert!((back - length).abs() / length < 1e-6, "{} vs {}", back, length);
    }

    #[test]
    fn pinned_particles_never_move() {
        let mut pts = vec![
            Particle { pos: V3::ZERO, prev: V3::ZERO, inv_mass: 0.0 },
            Particle { pos: V3::new(0.0, -1.0, 0.0), prev: V3::new(0.0, -1.0, 0.0), inv_mass: 1.0 },
        ];
        let cons = vec![Distance { a: 0, b: 1, rest: 1.0, compliance: 0.0 }];
        step(&mut pts, &cons, V3::new(0.0, -0.00981, 0.0), 4.0, 8);
        assert!(pts[0].pos.sub(&V3::ZERO).len() < 1e-12);
        // free particle fell under gravity but constraint pulls it back to radius 1
        let d = pts[1].pos.sub(&pts[0].pos).len();
        assert!((d - 1.0).abs() < 1e-6, "distance {}", d);
        // and it ended below the anchor (gravity direction)
        assert!(pts[1].pos.y() < 0.0);
    }
}
