//! P0900 — Discrete Exterior Calculus: the cotangent Laplacian and Taubin
//! λ|μ smoothing (the `smooth` word). The DEC view: a triangle mesh is a
//! simplicial 2-complex; the cotan weights are the *discrete Dirichlet
//! energy* weights, so one smoothing step is gradient flow of surface
//! energy — real differential geometry, not a blur.
//!
//!   Δφ(vᵢ) = Σⱼ wᵢⱼ (φⱼ − φᵢ),   wᵢⱼ = ½(cot αᵢⱼ + cot βᵢⱼ)
//!
//! Pure Laplacian flow shrinks surfaces (mean-curvature flow). Taubin's
//! alternating λ / μ pair with **μ = −λ** gives the per-iteration gain
//! G(α) = (1+λ(α−1))(1−λ(α−1)) = 1 − λ²(α−1)² ≤ 1 — unconditionally
//! stable, kills the high frequencies (noise, sharp corners) while the
//! low frequencies (overall shape) survive. That is why `smooth` rounds a
//! cube without collapsing it.
//!
//! Weights are computed once per `smooth` call from the input surface
//! (fixed-operator smoothing — standard practice, keeps O(n) per step).

use crate::geo::halfedge::{HalfEdges, UNSET};
use crate::geo::mesh::Mesh;
use crate::math3::V3;

const GUARD: u64 = 8_000_000;

/// Cotangent edge weights: w[e] for the undirected edge of half-edge e,
/// ½(cot α + cot β) with α, β opposite the edge in its two faces. Boundary
/// edges use their single angle. Clamped ≥ 0 (stability on obtuse meshes).
pub fn cotan_weights(he: &HalfEdges) -> Vec<f64> {
    let n = he.target.len();
    let mut w = vec![0.0f64; n];
    for e in 0..n as u32 {
        let t = he.twin[e as usize];
        if t != UNSET && t < e {
            continue; // mirrored copy — already stored
        }
        let a = he.verts[he.origin(e) as usize];
        let b = he.verts[he.target[e as usize] as usize];
        let c = he.verts[he.apex(e) as usize];
        let mut weight = 0.5 * cotan(a, b, c);
        if t != UNSET {
            let d = he.verts[he.apex(t) as usize];
            weight += 0.5 * cotan(a, b, d);
        }
        let weight = weight.max(0.0);
        w[e as usize] = weight;
        if t != UNSET {
            w[t as usize] = weight;
        }
    }
    w
}

/// cot(angle at apex c) in the triangle (a, b, c), clamped ≥ 0.
fn cotan(a: V3, b: V3, c: V3) -> f64 {
    let u = a.sub(&c);
    let v = b.sub(&c);
    let area2 = u.cross(&v).len(); // |u||v| sin θ
    let dot = u.dot(&v); // |u||v| cos θ
    if area2 < 1e-12 {
        return 0.0;
    }
    (dot / area2).max(0.0)
}

/// One Laplacian step over positions: p ← p + λ · Σw(pⱼ−pᵢ)/Σw.
/// Boundary vertices are pinned (open meshes keep their rims). Falls back
/// to the uniform umbrella when all weights vanish (degenerate triangles).
pub fn laplacian_step(verts: &[V3], he: &HalfEdges, w: &[f64], lambda: f64) -> Vec<V3> {
    let mut out = verts.to_vec();
    for v in 0..verts.len() {
        if he.vert_out[v] == UNSET {
            continue; // isolated vertex
        }
        if he.is_boundary_vertex(v) {
            continue; // pinned rim
        }
        let pairs = ring_edges(he, v);
        if pairs.is_empty() {
            continue;
        }
        let mut sum_w = 0.0;
        let mut target = V3::ZERO;
        let mut count = 0usize;
        for (nbr, e) in &pairs {
            count += 1;
            let wi = w[*e as usize];
            if wi <= 0.0 {
                continue;
            }
            sum_w += wi;
            target = target.add(&verts[*nbr as usize].mul(wi));
        }
        let centroid = if sum_w > 1e-12 {
            target.mul(1.0 / sum_w)
        } else {
            // uniform umbrella fallback
            let mut c = V3::ZERO;
            for (nbr, _) in &pairs {
                c = c.add(&verts[*nbr as usize]);
            }
            c.mul(1.0 / count as f64)
        };
        let p = verts[v];
        out[v] = p.add(&centroid.sub(&p).mul(lambda));
    }
    out
}

/// Walk the one-ring of v collecting (neighbor, half-edge v→neighbor) pairs.
#[inline]
fn ring_edges(he: &HalfEdges, v: usize) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let start = he.vert_out[v];
    if start == UNSET {
        return out;
    }
    // CCW walk
    let mut e = start;
    let mut guard = 0u64;
    loop {
        out.push((he.target[e as usize], e));
        let t = he.twin[e as usize];
        if t == UNSET {
            break;
        }
        e = he.next[t as usize];
        if e == start {
            return out;
        }
        guard += 1;
        if guard > GUARD {
            return out;
        }
    }
    // CW walk (open fan)
    let mut e = start;
    let mut guard = 0u64;
    loop {
        let p = he.prev[e as usize];
        let t = he.twin[p as usize];
        if t == UNSET {
            break;
        }
        e = t;
        if e == start {
            break;
        }
        out.push((he.target[e as usize], e));
        guard += 1;
        if guard > GUARD {
            return out;
        }
    }
    out
}

/// Taubin λ|μ smoothing, the engine behind `smooth(n, strength)`.
/// The half-edge build welds the mesh (point-built prims duplicate corners
/// — one-rings need shared vertices). Same connectivity, new positions,
/// volume tracked by the caller.
pub fn smooth(mesh: &Mesh, iterations: u32, strength: f64) -> Mesh {
    if iterations == 0 || mesh.tris.is_empty() {
        return mesh.clone();
    }
    let he = HalfEdges::build(mesh);
    let tris = he.tris();
    let w = cotan_weights(&he);
    // gentle by design: the language default strength 0.5 → λ 0.175. One
    // Taubin pair per iteration damps every non-DC mode by λ²(α−1)² — small
    // meshes (a cube is 8 verts) melt fast, so λ stays conservative.
    let lambda = (0.35 * strength).clamp(0.02, 0.35);
    let mu = -lambda; // μ = −λ: per-iteration gain 1−λ²(α−1)² ≤ 1, stable
    let mut verts = he.verts.clone();
    for it in 0..iterations {
        verts = laplacian_step(&verts, &he, &w, lambda);
        if it + 1 < iterations {
            // μ pass every iteration except the last (end on a λ pass)
            verts = laplacian_step(&verts, &he, &w, mu);
        }
    }
    let mut out = Mesh { verts, tris };
    out.ensure_outward();
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn welded_cube() -> Mesh {
        // point-built cube (known-good winding from mesh.rs tests) + weld
        let mut m = Mesh::new();
        let (x0, x1) = (0.0, 2.0);
        let (y0, y1) = (0.0, 2.0);
        let (z0, z1) = (0.0, 2.0);
        m.add_quad(V3::new(x0, y1, z0), V3::new(x0, y1, z1), V3::new(x1, y1, z1), V3::new(x1, y1, z0));
        m.add_quad(V3::new(x0, y0, z0), V3::new(x1, y0, z0), V3::new(x1, y0, z1), V3::new(x0, y0, z1));
        m.add_quad(V3::new(x0, y0, z1), V3::new(x1, y0, z1), V3::new(x1, y1, z1), V3::new(x0, y1, z1));
        m.add_quad(V3::new(x0, y0, z0), V3::new(x0, y1, z0), V3::new(x1, y1, z0), V3::new(x1, y0, z0));
        m.add_quad(V3::new(x1, y0, z0), V3::new(x1, y1, z0), V3::new(x1, y1, z1), V3::new(x1, y0, z1));
        m.add_quad(V3::new(x0, y0, z0), V3::new(x0, y0, z1), V3::new(x0, y1, z1), V3::new(x0, y1, z0));
        m.welded()
    }

    #[test]
    fn cotan_weights_finite_nonnegative() {
        let (m, _) = crate::geo::prims::sphere(10.0, 16);
        let he = HalfEdges::build(&m);
        let w = cotan_weights(&he);
        assert_eq!(w.len(), he.target.len());
        for wi in &w {
            assert!(*wi >= 0.0 && wi.is_finite(), "weight {}", wi);
        }
    }

    #[test]
    fn smoothing_sphere_keeps_volume_class() {
        let (m, _) = crate::geo::prims::sphere(10.0, 16);
        let v0 = m.volume_signed();
        let s = smooth(&m, 3, 0.5);
        let v1 = s.volume_signed();
        assert!(v1 > 0.0);
        assert!((v1 - v0).abs() / v0 < 0.05, "volume {} → {}", v0, v1);
        assert_eq!(s.tris.len(), m.tris.len());
    }

    #[test]
    fn smoothing_a_cube_rounds_it() {
        let m = welded_cube();
        let v0 = m.volume_signed();
        // the language-realistic call: 4 iterations at default strength
        let s = smooth(&m, 4, 0.5);
        let v1 = s.volume_signed();
        assert!(v1 > 0.0);
        assert!(v1 < v0, "corners must melt: {} → {}", v0, v1);
        assert!(v1 > v0 * 0.25, "Taubin must not collapse: {}", v1 / v0);
        let b0 = m.bbox().size();
        let b1 = s.bbox().size();
        assert!(b1.x() < b0.x() && b1.x() > b0.x() * 0.8, "bbox x {} → {}", b0.x(), b1.x());
    }

    #[test]
    fn smoothing_soup_welds_first() {
        // point-built cube (per-face duplicated verts) — weld must merge 24 → 8
        let mut m = Mesh::new();
        let (x0, x1) = (0.0, 2.0);
        let (y0, y1) = (0.0, 2.0);
        let (z0, z1) = (0.0, 2.0);
        m.add_quad(V3::new(x0, y1, z0), V3::new(x0, y1, z1), V3::new(x1, y1, z1), V3::new(x1, y1, z0));
        m.add_quad(V3::new(x0, y0, z0), V3::new(x1, y0, z0), V3::new(x1, y0, z1), V3::new(x0, y0, z1));
        m.add_quad(V3::new(x0, y0, z1), V3::new(x1, y0, z1), V3::new(x1, y1, z1), V3::new(x0, y1, z1));
        m.add_quad(V3::new(x0, y0, z0), V3::new(x0, y1, z0), V3::new(x1, y1, z0), V3::new(x1, y0, z0));
        m.add_quad(V3::new(x1, y0, z0), V3::new(x1, y1, z0), V3::new(x1, y1, z1), V3::new(x1, y0, z1));
        m.add_quad(V3::new(x0, y0, z0), V3::new(x0, y0, z1), V3::new(x0, y1, z1), V3::new(x0, y1, z0));
        let w = m.welded();
        assert_eq!(w.verts.len(), 8, "24 soup verts → 8 welded");
        assert_eq!(w.tris.len(), 12);
        assert!((w.volume_signed() - 8.0).abs() < 1e-9);
        // and smoothing now acts on corners
        let v0 = w.volume_signed();
        let s = smooth(&w, 4, 0.5);
        assert!(s.volume_signed() < v0, "{} -> {}", v0, s.volume_signed());
    }

    #[test]
    fn zero_iterations_is_identity() {
        let (m, _) = crate::geo::prims::sphere(10.0, 16);
        let s = smooth(&m, 0, 0.5);
        assert_eq!(s.verts.len(), m.verts.len());
    }
}
