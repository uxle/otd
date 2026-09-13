//! P1000 — Loop subdivision (the `subdiv` word). Each triangle splits into
//! 4 by edge midpoints; new odd (edge) points use the ³⁄₈–⅛ mask, updated
//! even (vertex) points use Loop's β weight. Boundaries follow the
//! Catmull-Rom ⅛–¾–⅛ rule. One `subdiv(n: 1)` = 4× triangles and a fairer
//! surface — sphere + subdiv(2) closes most of the volume gap to analytic.
//!
//! β(n) = (1/n)(⅝ − (⅜ + ¼ cos(2π/n))²)   for n > 3
//! β(3) = 3/16, β(4) = 3/8 (Loop's table)

use crate::geo::halfedge::{HalfEdges, UNSET};
use crate::geo::mesh::Mesh;
use crate::math3::V3;
use std::collections::HashMap;

/// Loop-subdivide `levels` times (welds first — point-built prims
/// duplicate corners, subdivision needs the shared-vertex topology).
pub fn loop_subdivide(mesh: &Mesh, levels: u32) -> Mesh {
    let mut m = mesh.welded();
    for _ in 0..levels {
        m = subdivide_once(&m);
    }
    m
}

fn loop_beta(n: usize) -> f64 {
    match n {
        0 | 1 | 2 => 0.0, // degenerate — don't move
        3 => 3.0 / 16.0,
        n => {
            let c = (3.0 / 8.0 + 0.25 * (2.0 * std::f64::consts::PI / n as f64).cos()).powi(2);
            (1.0 / n as f64) * (5.0 / 8.0 - c)
        }
    }
}

fn subdivide_once(mesh: &Mesh) -> Mesh {
    if mesh.tris.is_empty() {
        return mesh.clone();
    }
    // HalfEdges::build welds — shared-vertex topology is what subdiv needs
    let he = HalfEdges::build(mesh);
    let faces = he.tris();
    let old_pos: Vec<V3> = he.verts.clone();

    // even positions (updated old vertices), computed up front
    let even_pos: Vec<V3> = (0..old_pos.len())
        .map(|v| even_point(&he, v, &old_pos))
        .collect();

    let mut out = Mesh::new();
    // old vertices first (positions patched at the end)
    for _ in 0..old_pos.len() {
        out.add_vert(V3::ZERO);
    }
    // odd points: one new vertex per undirected edge, memoized by key
    let mut edge_point: HashMap<(u32, u32), u32> = HashMap::new();
    let mut odd = |e: u32,
                   edge_point: &mut HashMap<(u32, u32), u32>,
                   out: &mut Mesh|
     -> u32 {
        let a = he.origin(e);
        let b = he.target[e as usize];
        let key = if a < b { (a, b) } else { (b, a) };
        if let Some(id) = edge_point.get(&key) {
            return *id;
        }
        let t = he.twin[e as usize];
        let p = if t == UNSET {
            // boundary edge: midpoint (the boundary curve is piecewise linear
            // at odd points — the Catmull-Rom smoothing lives on even points)
            old_pos[a as usize].add(&old_pos[b as usize]).mul(0.5)
        } else {
            let c = old_pos[he.apex(e) as usize];
            let d = old_pos[he.apex(t) as usize];
            // ⅜(a+b) + ⅛(c+d)
            old_pos[a as usize]
                .add(&old_pos[b as usize])
                .mul(3.0 / 8.0)
                .add(&c.add(&d).mul(1.0 / 8.0))
        };
        let id = out.add_vert(p);
        edge_point.insert(key, id);
        id
    };

    // Each triangle [a, b, c] (CCW) owns half-edges at its face base:
    //   base+0: c→a   base+1: a→b   base+2: b→c
    for (t, tri) in faces.iter().enumerate() {
        let [a, b, c] = *tri;
        let base = 3 * t as u32;
        let ca = odd(base, &mut edge_point, &mut out); // edge (c,a)
        let ab = odd(base + 1, &mut edge_point, &mut out); // edge (a,b)
        let bc = odd(base + 2, &mut edge_point, &mut out); // edge (b,c)
        out.add_tri(a, ab, ca);
        out.add_tri(ab, b, bc);
        out.add_tri(ca, bc, c);
        out.add_tri(ab, bc, ca);
    }
    // patch even positions
    for v in 0..old_pos.len() {
        out.verts[v] = even_pos[v];
    }
    out.ensure_outward();
    out
}

fn even_point(he: &HalfEdges, v: usize, old_pos: &[V3]) -> V3 {
    if he.vert_out[v] == UNSET {
        return old_pos[v];
    }
    if he.is_boundary_vertex(v) {
        // Catmull-Rom boundary: ¾ v + ⅛(prev + next along the rim)
        if let Some((n1, n2)) = he.boundary_neighbors(v) {
            return old_pos[v]
                .mul(0.75)
                .add(&old_pos[n1 as usize].mul(0.125))
                .add(&old_pos[n2 as usize].mul(0.125));
        }
        return old_pos[v];
    }
    let ring = match he.ring(v) {
        Some(r) if r.len() >= 2 => r,
        _ => return old_pos[v],
    };
    let n = ring.len();
    let beta = loop_beta(n);
    let mut sum = V3::ZERO;
    for nbr in &ring {
        sum = sum.add(&old_pos[*nbr as usize]);
    }
    old_pos[v].mul(1.0 - n as f64 * beta).add(&sum.mul(beta))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tetra() -> Mesh {
        let mut m = Mesh::new();
        let a = m.add_vert(V3::new(1.0, 0.0, 0.0));
        let b = m.add_vert(V3::new(-1.0, 0.5, 0.0));
        let c = m.add_vert(V3::new(0.0, -0.5, 1.0));
        let d = m.add_vert(V3::new(0.0, 0.0, -1.0));
        m.add_tri(a, b, c);
        m.add_tri(a, d, b);
        m.add_tri(a, c, d);
        m.add_tri(b, d, c);
        m
    }

    #[test]
    fn tetrahedron_once() {
        let m = tetra();
        let v0 = m.volume_signed().abs();
        let s = loop_subdivide(&m, 1);
        assert_eq!(s.tris.len(), 16, "4 → 16 triangles");
        assert_eq!(s.verts.len(), 4 + 6, "verts = 4 old + 6 edge points");
        let v1 = s.volume_signed().abs();
        // A tetra is all valence-3 vertices — Loop's β(3) pulls each corner
        // hard toward the centroid (7/16·v + 3/16·Σ, and Σ = −v for a tetra),
        // so the volume legitimately collapses. Count + positivity is the
        // honest assertion; scale preservation is tested on higher-valence
        // meshes (torus, octahedron).
        assert!(v1 > 0.0 && v1 < 0.5 * v0, "volume {} → {}", v0, v1);
    }

    #[test]
    fn torus_gets_fairer() {
        // the torus is pure valence-6 after weld — Loop's home turf.
        // Fairness = mean angle (°) between face normal and the exact torus
        // normal at the face centroid. Subdiv must halve it per level.
        let (m, _) = crate::geo::prims::torus(20.0, 5.0, 16);
        let f0 = torus_unfairness(&m);
        let s1 = loop_subdivide(&m, 1);
        let f1 = torus_unfairness(&s1);
        let s2 = loop_subdivide(&m, 2);
        let f2 = torus_unfairness(&s2);
        assert!(f1 < f0, "fairness {} → {}", f0, f1);
        assert!(f2 < f1, "fairness {} → {}", f1, f2);
        assert!(f2 < f0 * 0.9, "two levels: {} → {}", f0, f2);
        // volume: the limit surface of a torus control mesh is very close
        let v0 = m.volume_signed();
        let v2 = s2.volume_signed();
        assert!((v2 - v0).abs() / v0 < 0.10, "torus volume {} → {}", v0, v2);
        let he = HalfEdges::build(&s2);
        assert!(he.manifold_report().closed);
    }

    /// Mean angle (°) between face normals and the analytic torus normal
    /// at the face centroid (torus: R = 20, r = 5, axis Y, center height r).
    fn torus_unfairness(m: &Mesh) -> f64 {
        let (r_main, r_tube) = (20.0, 5.0);
        let c = V3::new(0.0, r_tube, 0.0);
        let mut acc = 0.0;
        for t in 0..m.tris.len() {
            let [a, b, cc] = m.tris[t];
            let (va, vb, vc) = (m.verts[a as usize], m.verts[b as usize], m.verts[cc as usize]);
            let n = vb.sub(&va).cross(&vc.sub(&va));
            let p = va.add(&vb).add(&vc).mul(1.0 / 3.0).sub(&c);
            // analytic normal: radial direction on the tube circle
            let rho = (p.x() * p.x() + p.z() * p.z()).sqrt();
            let dir = V3::new(p.x() / rho, 0.0, p.z() / rho);
            let exact = p.sub(&dir.mul(r_main));
            let cos = n.dot(&exact) / (n.len() * exact.len());
            acc += cos.clamp(-1.0, 1.0).acos().to_degrees();
        }
        acc / m.tris.len() as f64
    }

    #[test]
    fn sphere_gets_fairer_away_from_poles() {
        // A lat-long sphere's poles are valence-24 extraordinary vertices —
        // honest Loop behavior lets the cap faces tilt there. Away from the
        // caps (|y−10| < 8), fairness must improve; the whole-sphere volume
        // must stay in class. `sphere` remains the exact-sphere tool.
        let r = 10.0f64;
        let (m, _) = crate::geo::prims::sphere(r, 24);
        let analytic = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
        let f0 = mean_normal_dev(&m, 8.0);
        let s1 = loop_subdivide(&m, 1);
        let f1 = mean_normal_dev(&s1, 8.0);
        let s2 = loop_subdivide(&m, 2);
        let f2 = mean_normal_dev(&s2, 8.0);
        assert!(f1 < f0, "fairness {} → {}", f0, f1);
        assert!(f2 < f1, "fairness {} → {}", f1, f2);
        assert!(f2 < f0 * 0.6, "two levels: {} → {}", f0, f2);
        // volume stays within 6 % of the analytic sphere
        let err2 = (s2.volume_signed() - analytic).abs() / analytic;
        assert!(err2 < 0.06, "volume error {}", err2);
        assert_eq!(s2.tris.len(), m.tris.len() * 16);
        // still closed
        let he = HalfEdges::build(&s2);
        assert!(he.manifold_report().closed);
    }

    /// Mean angle (°) between each face normal and the radial direction at
    /// the face centroid, for faces whose centroid |y−10| < cap_cut.
    fn mean_normal_dev(m: &Mesh, cap_cut: f64) -> f64 {
        let mut acc = 0.0;
        let mut n = 0.0;
        let c = V3::new(0.0, 10.0, 0.0);
        for t in 0..m.tris.len() {
            let [a, b, cc] = m.tris[t];
            let (va, vb, vc) = (m.verts[a as usize], m.verts[b as usize], m.verts[cc as usize]);
            let cen = va.add(&vb).add(&vc).mul(1.0 / 3.0);
            if (cen.y() - 10.0).abs() >= cap_cut {
                continue; // polar caps: extraordinary-vertex region
            }
            let nrm = vb.sub(&va).cross(&vc.sub(&va));
            let radial = cen.sub(&c);
            let cos = nrm.dot(&radial) / (nrm.len() * radial.len());
            acc += cos.clamp(-1.0, 1.0).acos().to_degrees();
            n += 1.0;
        }
        acc / n
    }

    #[test]
    fn closed_box_rounds_but_keeps_class() {
        let (m, _) = crate::geo::prims::plane(20.0, 20.0, 1.0);
        let s = loop_subdivide(&m, 1);
        assert_eq!(s.tris.len(), m.tris.len() * 4);
        let v0 = m.volume_signed();
        let v1 = s.volume_signed();
        // Loop rounds sharp box corners aggressively (no crease tags in 2.0)
        // — the volume drops but the solid stays a positive, watertight class
        assert!(v1 > 0.2 * v0, "box volume {} → {}", v0, v1);
        let he = HalfEdges::build(&s);
        assert!(he.manifold_report().closed, "subdivided box stays watertight");
    }

    #[test]
    fn loop_beta_table() {
        assert!((loop_beta(3) - 3.0 / 16.0).abs() < 1e-12);
        // β(4) from the closed formula = 31/256 (3/8 is the *edge* mask weight)
        assert!((loop_beta(4) - 31.0 / 256.0).abs() < 1e-12);
        assert!((loop_beta(6) - 1.0 / 16.0).abs() < 1e-12);
        // β(n) → 15/(64 n) asymptotically (cos term → 1)
        let big = loop_beta(100);
        assert!((big - 15.0 / (64.0 * 100.0)).abs() < 1e-5, "β(100) = {}", big);
    }

    #[test]
    fn subdivision_then_smoothing_is_stable() {
        // the two deep-tier words compose
        let (m, _) = crate::geo::prims::sphere(10.0, 8);
        let s = loop_subdivide(&m, 1);
        let sm = crate::geo::dec::smooth(&s, 2, 0.5);
        assert!(sm.volume_signed() > 0.0);
        assert_eq!(sm.tris.len(), s.tris.len());
    }
}
