//! P1050 — implicit-field blending (the `blend` word). Two solids become
//! signed-distance fields (BVH closest-point + ray-parity sign), fuse with
//! the exponential smooth-min — the "solder" fillet — and re-surface by
//! marching tetrahedra over a lattice. Volume is measured after, so mass
//! stays honest.
//!
//!   sd(p)   = ±dist_to_surface(p)         (− inside, + outside)
//!   smin(a, b, k) = −ln(e^(−ka) + e^(−kb)) / k,   k = 1/gap
//!
//! `k` controls the fillet radius: gap → ∞ reproduces the hard union,
//! gap → 0 melts the parts together. The default is 5 mm.

use crate::geo::bvh::Bvh;
use crate::geo::mesh::Mesh;
use crate::math3::{Aabb, V3};

/// Marching-tetrahedra extractor shared with the metaball builder: samples
/// `field` on a res³ lattice over `bb`, emits the iso-surface field = 0
/// with outward orientation (field > 0 counts as inside).
pub fn march_tets(bb: &Aabb, res: u32, field: &dyn Fn(V3) -> f64) -> Mesh {
    let res = res.clamp(10, 96);
    let step = bb.size().mul(1.0 / res as f64);
    let corner = |i: u32, j: u32, k: u32| -> V3 {
        bb.min.add(&V3::new(
            step.x() * i as f64,
            step.y() * j as f64,
            step.z() * k as f64,
        ))
    };
    let cube_c: [[u32; 3]; 8] = [
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ];
    // Freudenthal-Kuhn 6-tet triangulation of the cube around body diagonal
    // 0-6 — the *globally conforming* lattice triangulation: neighboring
    // cubes agree on every shared-face diagonal, so the isosurface cannot
    // crack. (The 1.0 table tiled the cube incorrectly — uncovered cells and
    // overlaps — which is why metaball meshes were never watertight.)
    let tets: [[usize; 4]; 6] = [
        [0, 1, 2, 6], [0, 1, 5, 6], [0, 3, 2, 6], [0, 3, 7, 6], [0, 4, 5, 6], [0, 4, 7, 6],
    ];
    let mut m = Mesh::new();
    for k in 0..res {
        for j in 0..res {
            for i in 0..res {
                let mut cv = [V3::ZERO; 8];
                let mut cf = [0.0f64; 8];
                for (c, cc) in cube_c.iter().enumerate() {
                    let p = corner(i + cc[0], j + cc[1], k + cc[2]);
                    cv[c] = p;
                    cf[c] = field(p);
                }
                for t in tets {
                    // classify tet corners: above/below iso 0
                    let above = [
                        cf[t[0]] > 0.0,
                        cf[t[1]] > 0.0,
                        cf[t[2]] > 0.0,
                        cf[t[3]] > 0.0,
                    ];
                    let n_above = above.iter().filter(|a| **a).count();
                    if n_above == 0 || n_above == 4 {
                        continue;
                    }
                    // crossings on the sign-changing tet edges: (corner_i, corner_j, point)
                    let mut cr: Vec<(usize, usize, V3)> = Vec::new();
                    for a in 0..4 {
                        for b in (a + 1)..4 {
                            if above[a] != above[b] {
                                let (va, fa) = (cv[t[a]], cf[t[a]]);
                                let (vb, fb) = (cv[t[b]], cf[t[b]]);
                                let da = fa;
                                let db = fb;
                                let t_frac =
                                    if (da - db).abs() > 1e-12 { da / (da - db) } else { 0.5 };
                                cr.push((a, b, va.lerp(&vb, t_frac)));
                            }
                        }
                    }
                    // order the crossings as a closed cycle (consecutive share a corner)
                    let cycle = cycle_order(&cr);
                    if cycle.len() < 3 {
                        continue;
                    }
                    // inside ("above") centroid — the surface faces away from it
                    let mut ca = V3::ZERO;
                    let mut cnt = 0.0;
                    for (slot, is_ab) in above.iter().enumerate() {
                        if *is_ab {
                            ca = ca.add(&cv[t[slot]]);
                            cnt += 1.0;
                        }
                    }
                    ca = ca.mul(1.0 / cnt);
                    // 3 crossings -> 1 triangle; 4 -> 2 triangles split along
                    // the (0,2) diagonal of the CYCLE (the naive index order
                    // leaves one quad edge uncovered -> holes; fixed in 2.0)
                    let pts: Vec<V3> = cycle.iter().map(|(_, _, p)| *p).collect();
                    let tri_out = |m: &mut Mesh, x: V3, y: V3, z: V3| {
                        let nrm = y.sub(&x).cross(&z.sub(&x));
                        let outward = x.add(&y).add(&z).mul(1.0 / 3.0).sub(&ca);
                        if nrm.dot(&outward) < 0.0 {
                            m.add_tri_pts(x, z, y);
                        } else {
                            m.add_tri_pts(x, y, z);
                        }
                    };
                    tri_out(&mut m, pts[0], pts[1], pts[2]);
                    if pts.len() == 4 {
                        tri_out(&mut m, pts[0], pts[2], pts[3]);
                    }
                }
            }
        }
    }
    m.ensure_outward();
    m
}

/// Order the tet's crossings as a closed cycle: consecutive crossings share
/// a tet corner. In the 2-2 case the four crossings form a 4-cycle in this
/// adjacency (each corner appears in exactly two crossings); for 1-3 / 3-1
/// cases the three crossings form a triangle — order irrelevant (the
/// emitter fixes orientation).
fn cycle_order(cr: &[(usize, usize, V3)]) -> Vec<(usize, usize, V3)> {
    if cr.len() != 4 {
        return cr.to_vec();
    }
    let shares = |x: &(usize, usize, V3), y: &(usize, usize, V3)| -> bool {
        x.0 == y.0 || x.0 == y.1 || x.1 == y.0 || x.1 == y.1
    };
    let mut order = vec![0usize];
    let mut used = [false; 4];
    used[0] = true;
    let mut next = None;
    for i in 1..4 {
        if shares(&cr[0], &cr[i]) {
            next = Some(i);
            break;
        }
    }
    let mut cur = match next {
        Some(i) => {
            used[i] = true;
            order.push(i);
            i
        }
        None => return cr.to_vec(),
    };
    while order.len() < 4 {
        let mut stepped = false;
        for i in 0..4 {
            if !used[i] && shares(&cr[cur], &cr[i]) {
                used[i] = true;
                order.push(i);
                cur = i;
                stepped = true;
                break;
            }
        }
        if !stepped {
            break;
        }
    }
    order.into_iter().map(|i| cr[i]).collect()
}

/// A mesh as a signed-distance field: BVH closest point for |sd|, ray
/// parity for the sign. Closed meshes only (OTD solids guarantee this).
pub struct Sdf<'a> {
    mesh: &'a Mesh,
    bvh: Bvh,
}

impl<'a> Sdf<'a> {
    pub fn new(mesh: &'a Mesh) -> Sdf<'a> {
        let bvh = Bvh::build(mesh);
        Sdf { mesh, bvh }
    }

    /// Signed distance at p (mm; negative inside). The sign costs one
    /// jittered ray cast — BVH-accelerated.
    pub fn at(&self, p: &V3) -> f64 {
        let d = self.bvh.distance(self.mesh, p);
        if self.bvh.contains(self.mesh, p) {
            -d
        } else {
            d
        }
    }
}

/// Exponential smooth-min (the fillet). Clamped exponents keep e^x finite.
pub fn smooth_min(a: f64, b: f64, k: f64) -> f64 {
    if k <= 1e-9 {
        return a.min(b);
    }
    let h = (-k * a).max(-60.0);
    let g = (-k * b).max(-60.0);
    -(h.exp() + g.exp()).ln() / k
}

/// Blend two closed solids with a fillet of radius ≈ `gap` mm.
/// `res` is the lattice resolution (default 36).
pub fn blend(a: &Mesh, b: &Mesh, gap: f64, res: u32) -> Mesh {
    if a.tris.is_empty() {
        return b.clone();
    }
    if b.tris.is_empty() {
        return a.clone();
    }
    let sdf_a = Sdf::new(a);
    let sdf_b = Sdf::new(b);
    let k = 1.0 / gap.max(0.01);
    // bounds: union bbox grown by gap + 2 lattice steps
    let mut bb = a.bbox();
    bb.grow_box(&b.bbox());
    let step_len = bb.size().x().max(bb.size().y()).max(bb.size().z()) / res as f64;
    let margin = V3::new(gap + 2.0 * step_len, gap + 2.0 * step_len, gap + 2.0 * step_len);
    bb.min = bb.min.sub(&margin);
    bb.max = bb.max.add(&margin);
    // skip the far-outside fast: if min(sdA, sdB) > gap*2 + step, the smin is
    // just the min and outside anyway — the field still answers correctly.
    let field = |p: V3| -> f64 {
        let da = sdf_a.at(&p);
        let db = sdf_b.at(&p);
        -smooth_min(da, db, k)
    };
    let mesh = march_tets(&bb, res, &field);
    mesh
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::prims;

    #[test]
    fn smooth_min_math() {
        // k → 0 approaches plain min
        assert!((smooth_min(1.0, 2.0, 1e-9) - 1.0).abs() < 1e-6);
        // symmetric point
        let s = smooth_min(1.0, 1.0, 0.5);
        // −ln(2e^−0.5)/0.5 = 0.5·(ln2 + ... ) = (ln2)/0.5 − 1... compute:
        let expect = -(2.0 * (-0.5f64).exp()).ln() / 0.5;
        assert!((s - expect).abs() < 1e-12);
        // smooth-min is ≤ min
        assert!(smooth_min(3.0, 4.0, 0.25) <= 3.0 + 1e-12);
        // and never below min by more than ln2/k
        assert!(smooth_min(3.0, 4.0, 0.25) >= 3.0 - 0.7 / 0.25);
    }

    #[test]
    fn sdf_sign() {
        // sphere sits at center (0, 10, 0) — r = 10
        let (m, _) = prims::sphere(10.0, 24);
        let sdf = Sdf::new(&m);
        let inside = sdf.at(&V3::new(0.0, 10.0, 0.0));
        assert!(inside < 0.0 && inside > -11.0, "inside sd = {}", inside);
        let outside = sdf.at(&V3::new(0.0, 25.0, 0.0));
        assert!((outside - 5.0).abs() < 1e-6, "outside sd = {}", outside);
        let edge = sdf.at(&V3::new(0.0, 20.0, 0.0)); // on the north pole
        assert!(edge.abs() < 0.5, "on-surface sd = {}", edge);
    }

    #[test]
    fn blend_two_cubes_volume() {
        // two 10mm cubes overlapping by 5mm; blended volume ≈ union + fillet
        let (a, _) = prims::cube(10.0, 10.0, 10.0);
        let (b, _) = prims::cube(10.0, 10.0, 10.0);
        let mut b = b;
        b.transform(&crate::math3::M4::translate(5.0, 0.0, 0.0));
        let union_v = a.volume_signed() + b.volume_signed() - 5.0 * 10.0 * 10.0; // 1000+1000−500 = 1500
        let blended = blend(&a, &b, 1.0, 26);
        let v = blended.volume_signed();
        assert!(v > union_v * 0.9, "blend volume {} vs union {}", v, union_v);
        // the fillet adds material — but bounded by the fillet size
        assert!(v < union_v * 1.6, "but not wild: {} vs {}", v, union_v);
        // and the blend is closed
        let he = crate::geo::halfedge::HalfEdges::build(&blended);
        assert!(he.manifold_report().closed, "blend must be watertight");
    }

    #[test]
    fn blend_disjoint_is_unionish() {
        // far apart with a tiny gap value: result ≈ both solids
        let (a, _) = prims::cube(2.0, 2.0, 2.0);
        let (b, _) = prims::cube(2.0, 2.0, 2.0);
        let mut b = b;
        b.transform(&crate::math3::M4::translate(10.0, 0.0, 0.0));
        let blended = blend(&a, &b, 0.5, 24);
        let v = blended.volume_signed();
        // two separate blobs (smin never fully disconnects, but 4mm apart
        // with 0.5mm fillet leaves a thin neck) — volume close to 16
        assert!(v > 14.0 && v < 20.0, "volume {}", v);
    }

    #[test]
    fn march_tets_sphere_field() {
        // field = r − |p−c| (inside positive) → a sphere of radius 10
        let bb = Aabb::from_center_extent(&V3::new(0.0, 10.0, 0.0), &V3::new(24.0, 24.0, 24.0));
        let m = march_tets(&bb, 20, &|p| 10.0 - p.sub(&V3::new(0.0, 10.0, 0.0)).len());
        let v = m.volume_signed();
        let analytic = 4.0 / 3.0 * std::f64::consts::PI * 1000.0;
        assert!((v - analytic).abs() / analytic < 0.15, "volume {} vs {}", v, analytic);
    }
}
