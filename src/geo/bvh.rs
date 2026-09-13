//! P1050 — the bounding-volume hierarchy. Median-split binary tree over a
//! mesh's triangles, answering two queries in O(log n):
//!   · closest_point(p)  — signed-distance fields for `blend`
//!   · ray_hit(o, d)     — soft shadows, ambient occlusion, inside/outside
//! f64 millimeters throughout, like the rest of the kernel.

use crate::geo::mesh::Mesh;
use crate::math3::{Aabb, V3};

/// A BVH built from a mesh. The mesh is *not* owned — queries take it again
/// by reference, so building is cheap and memory stays flat.
pub struct Bvh {
    /// interior nodes: (aabb, axis, left child, right child)
    /// leaf nodes:     (aabb, axis = 3, tri_start, tri_end) into `order`
    nodes: Vec<Node>,
    /// triangle indices, reordered so each leaf owns a contiguous range
    order: Vec<u32>,
    /// root node index (nodes are pushed post-order — the root is NOT 0)
    root: u32,
}

#[derive(Clone, Copy)]
struct Node {
    bb: Aabb,
    /// 0/1/2 = split axis, 3 = leaf
    axis: u8,
    a: u32,
    b: u32,
}

const LEAF_MAX: usize = 8;

impl Bvh {
    /// Build a BVH over a closed triangle soup. Empty meshes give an empty tree.
    pub fn build(mesh: &Mesh) -> Bvh {
        let n = mesh.tris.len();
        if n == 0 {
            return Bvh { nodes: Vec::new(), order: Vec::new(), root: 0 };
        }
        let mut order: Vec<u32> = (0..n as u32).collect();
        // per-triangle centroids + bounds
        let mut cents = vec![V3::ZERO; n];
        let mut bbs = vec![Aabb::empty(); n];
        for (t, o) in order.iter().enumerate() {
            let (a, b, c) = mesh.tri(*o as usize);
            cents[t] = a.add(&b).add(&c).mul(1.0 / 3.0);
            let mut bb = Aabb::empty();
            bb.grow_pt(&a);
            bb.grow_pt(&b);
            bb.grow_pt(&c);
            bbs[t] = bb;
        }
        let mut nodes = Vec::with_capacity(2 * n / LEAF_MAX + 1);
        let root = build_rec(mesh, &mut order, &cents, &bbs, 0, n, &mut nodes);
        Bvh { nodes, order, root }
    }

    pub fn is_empty(&self) -> bool { self.nodes.is_empty() }

    /// Closest point on the mesh surface to `p` (f64). Descends nearest
    /// child first and prunes with point-to-AABB distance — exact, not
    /// approximate.
    pub fn closest_point(&self, mesh: &Mesh, p: &V3) -> Option<V3> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best_d2 = f64::INFINITY;
        let mut best_pt = V3::ZERO;
        closest_rec(self, mesh, p, self.root, &mut best_d2, &mut best_pt);
        Some(best_pt)
    }

    /// Closest distance from `p` to the surface (mm).
    pub fn distance(&self, mesh: &Mesh, p: &V3) -> f64 {
        self.closest_point(mesh, p).map_or(f64::INFINITY, |q| q.sub(p).len())
    }

    /// First intersection of a ray with the mesh, as (t, triangle index).
    /// `dir` need not be normalized — t scales with |dir|.
    pub fn ray_hit(&self, mesh: &Mesh, o: &V3, dir: &V3) -> Option<(f64, usize)> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = (f64::INFINITY, 0usize);
        if !ray_rec(self, mesh, o, dir, self.root, &mut best) {
            return None;
        }
        if best.0.is_finite() { Some(best) } else { None }
    }

    /// All intersections along a ray, sorted by t (parity counting for
    /// inside/outside tests).
    pub fn ray_all(&self, mesh: &Mesh, o: &V3, dir: &V3) -> Vec<f64> {
        let mut out = Vec::new();
        if self.nodes.is_empty() {
            return out;
        }
        ray_all_rec(self, mesh, o, dir, self.root, &mut out);
        out.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        out
    }

    /// Inside/outside classification by ray parity. Shared-edge and shared-
    /// vertex hits intersect several triangles at once — hits at (nearly)
    /// the same t are clustered into a single crossing so parity survives.
    /// Closed, consistently-wound meshes only — `Mesh::ensure_outward`
    /// guarantees this for OTD solids.
    pub fn contains(&self, mesh: &Mesh, p: &V3) -> bool {
        // deterministic jittered ray
        let dir = V3::new(0.5257311, 0.8506508, 0.4472136);
        let hits = self.ray_all(mesh, p, &dir);
        // cluster by t: consecutive hits closer than eps are one crossing
        let mut crossings = 0usize;
        let mut last_t = f64::NAN;
        for t in hits.into_iter().filter(|t| *t > 1e-9) {
            if last_t.is_nan() || (t - last_t).abs() > 1e-7 {
                crossings += 1;
            }
            last_t = t;
        }
        crossings % 2 == 1
    }

    /// nodes visited — for the perf nerd in all of us (`ask "bvh depth?"`)
    pub fn node_count(&self) -> usize { self.nodes.len() }

    /// Height of the tree (1 = single leaf).
    pub fn depth(&self) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        depth_rec(self, self.root)
    }
}

// ---------- recursive builders & walkers (iterative stack, no recursion depth blowup risk) ----------

fn build_rec(mesh: &Mesh, order: &mut [u32], cents: &[V3], bbs: &[Aabb], lo: usize, hi: usize, nodes: &mut Vec<Node>) -> u32 {
    // NOTE: bbs/cents are indexed by *triangle id*; the id at position k is
    // order[k] (the array is sorted in-place as we descend).
    let mut bb = Aabb::empty();
    for k in lo..hi {
        bb.grow_box(&bbs[order[k] as usize]);
    }
    let size = bb.size();
    let axis = if size.x() >= size.y() && size.x() >= size.z() {
        0
    } else if size.y() >= size.z() {
        1
    } else {
        2
    };
    if hi - lo <= LEAF_MAX {
        let id = nodes.len() as u32;
        nodes.push(Node { bb, axis: 3, a: lo as u32, b: hi as u32 });
        return id;
    }
    // median split along the longest axis on centroid
    let part = order_lo_hi(order, lo, hi);
    part.sort_by(|x, y| {
        let cx = cents[*x as usize].0[axis];
        let cy = cents[*y as usize].0[axis];
        cx.partial_cmp(&cy).unwrap_or(std::cmp::Ordering::Equal)
    });
    let mid = (lo + hi) / 2;
    let left = build_rec(mesh, order, cents, bbs, lo, mid, nodes);
    let right = build_rec(mesh, order, cents, bbs, mid, hi, nodes);
    let id = nodes.len() as u32;
    nodes.push(Node { bb, axis: axis as u8, a: left, b: right });
    id
}

fn order_lo_hi(order: &mut [u32], lo: usize, hi: usize) -> &mut [u32] {
    if lo < hi && hi <= order.len() { &mut order[lo..hi] } else { &mut order[0..0] }
}

/// Squared distance from point to AABB (0 inside).
fn pt_aabb_d2(p: &V3, bb: &Aabb) -> f64 {
    let d = V3::new(
        (bb.min.x() - p.x()).max(0.0).max(p.x() - bb.max.x()),
        (bb.min.y() - p.y()).max(0.0).max(p.y() - bb.max.y()),
        (bb.min.z() - p.z()).max(0.0).max(p.z() - bb.max.z()),
    );
    d.dot(&d)
}

fn closest_rec(bvh: &Bvh, mesh: &Mesh, p: &V3, node: u32, best_d2: &mut f64, best_pt: &mut V3) {
    let n = bvh.nodes[node as usize];
    // prune: if the whole node box is farther than our best, skip
    if pt_aabb_d2(p, &n.bb) >= *best_d2 {
        return;
    }
    if n.axis == 3 {
        for k in n.a..n.b {
            let ti = bvh.order[k as usize] as usize;
            let (a, b, c) = mesh.tri(ti);
            let q = closest_point_on_tri(p, &a, &b, &c);
            let d2 = q.sub(p).dot(&q.sub(p));
            if d2 < *best_d2 {
                *best_d2 = d2;
                *best_pt = q;
            }
        }
        return;
    }
    // nearest child first
    let (first, second) = {
        let l = &bvh.nodes[n.a as usize];
        let r = &bvh.nodes[n.b as usize];
        if pt_aabb_d2(p, &l.bb) <= pt_aabb_d2(p, &r.bb) { (n.a, n.b) } else { (n.b, n.a) }
    };
    closest_rec(bvh, mesh, p, first, best_d2, best_pt);
    closest_rec(bvh, mesh, p, second, best_d2, best_pt);
}

/// Exact closest point on triangle (a,b,c) to p — Ericson, Real-Time
/// Collision Detection, barycentric regions. f64.
pub fn closest_point_on_tri(p: &V3, a: &V3, b: &V3, c: &V3) -> V3 {
    let ab = b.sub(a);
    let ac = c.sub(a);
    let ap = p.sub(a);
    let d1 = ab.dot(&ap);
    let d2 = ac.dot(&ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return *a;
    }
    let bp = p.sub(b);
    let d3 = ab.dot(&bp);
    let d4 = ac.dot(&bp);
    if d3 >= 0.0 && d4 <= d3 {
        return *b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return a.add(&ab.mul(v));
    }
    let cp = p.sub(c);
    let d5 = ab.dot(&cp);
    let d6 = ac.dot(&cp);
    if d6 >= 0.0 && d5 <= d6 {
        return *c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return a.add(&ac.mul(w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return b.add(&c.sub(b).mul(w));
    }
    // interior
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    a.add(&ab.mul(v)).add(&ac.mul(w))
}

/// Slab test: does the ray (o, d) hit the box within (tmin, tmax)?
fn ray_aabb(o: &V3, d: &V3, bb: &Aabb, tmin: f64, tmax: f64) -> bool {
    let mut lo = tmin;
    let mut hi = tmax;
    for k in 0..3 {
        let dk = d.0[k];
        let ok = o.0[k];
        if dk.abs() < 1e-12 {
            if ok < bb.min.0[k] || ok > bb.max.0[k] {
                return false;
            }
        } else {
            let inv = 1.0 / dk;
            let (mut t1, mut t2) = ((bb.min.0[k] - ok) * inv, (bb.max.0[k] - ok) * inv);
            if t1 > t2 {
                std::mem::swap(&mut t1, &mut t2);
            }
            lo = lo.max(t1);
            hi = hi.min(t2);
            if lo > hi {
                return false;
            }
        }
    }
    true
}

/// Möller–Trumbore triangle intersection, f64. Returns t along dir.
fn ray_tri(o: &V3, d: &V3, a: &V3, b: &V3, c: &V3) -> Option<f64> {
    let e1 = b.sub(a);
    let e2 = c.sub(a);
    let pvec = d.cross(&e2);
    let det = e1.dot(&pvec);
    if det.abs() < 1e-12 {
        return None; // parallel
    }
    let inv = 1.0 / det;
    let tvec = o.sub(a);
    let u = tvec.dot(&pvec) * inv;
    if u < -1e-12 || u > 1.0 + 1e-12 {
        return None;
    }
    let qvec = tvec.cross(&e1);
    let v = d.dot(&qvec) * inv;
    if v < -1e-12 || u + v > 1.0 + 1e-12 {
        return None;
    }
    let t = e2.dot(&qvec) * inv;
    if t > 1e-9 {
        Some(t)
    } else {
        None
    }
}

fn ray_rec(bvh: &Bvh, mesh: &Mesh, o: &V3, d: &V3, node: u32, best: &mut (f64, usize)) -> bool {
    let n = bvh.nodes[node as usize];
    let tmax = if best.0.is_finite() { best.0 } else { f64::INFINITY };
    if !ray_aabb(o, d, &n.bb, 0.0, tmax) {
        return best.0.is_finite();
    }
    if n.axis == 3 {
        for k in n.a..n.b {
            let ti = bvh.order[k as usize] as usize;
            let (a, b, c) = mesh.tri(ti);
            if let Some(t) = ray_tri(o, d, &a, &b, &c) {
                if t < best.0 {
                    *best = (t, ti);
                }
            }
        }
        return best.0.is_finite();
    }
    ray_rec(bvh, mesh, o, d, n.a, best);
    ray_rec(bvh, mesh, o, d, n.b, best);
    best.0.is_finite()
}

fn ray_all_rec(bvh: &Bvh, mesh: &Mesh, o: &V3, d: &V3, node: u32, out: &mut Vec<f64>) {
    let n = bvh.nodes[node as usize];
    if !ray_aabb(o, d, &n.bb, 0.0, f64::INFINITY) {
        return;
    }
    if n.axis == 3 {
        for k in n.a..n.b {
            let ti = bvh.order[k as usize] as usize;
            let (a, b, c) = mesh.tri(ti);
            if let Some(t) = ray_tri(o, d, &a, &b, &c) {
                out.push(t);
            }
        }
        return;
    }
    ray_all_rec(bvh, mesh, o, d, n.a, out);
    ray_all_rec(bvh, mesh, o, d, n.b, out);
}

fn depth_rec(bvh: &Bvh, node: u32) -> usize {
    let n = bvh.nodes[node as usize];
    if n.axis == 3 {
        1
    } else {
        1 + depth_rec(bvh, n.a).max(depth_rec(bvh, n.b))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cube_mesh() -> Mesh {
        // 0..2 cube (same winding as mesh tests: outward, volume 8)
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
        m
    }

    #[test]
    fn closest_point_on_cube() {
        let m = cube_mesh();
        let bvh = Bvh::build(&m);
        // outside +x face
        let q = bvh.closest_point(&m, &V3::new(5.0, 1.0, 1.0)).unwrap();
        assert!((q.x() - 2.0).abs() < 1e-9 && (q.y() - 1.0).abs() < 1e-9 && (q.z() - 1.0).abs() < 1e-9);
        // above center → top face
        let q = bvh.closest_point(&m, &V3::new(1.0, 5.0, 1.0)).unwrap();
        assert!((q.y() - 2.0).abs() < 1e-9);
        // inside → nearest face at distance 1
        let d = bvh.distance(&m, &V3::new(1.0, 1.0, 1.0));
        assert!((d - 1.0).abs() < 1e-9, "d={}", d);
        // far corner query → the corner itself
        let q = bvh.closest_point(&m, &V3::new(-3.0, -3.0, -3.0)).unwrap();
        assert!((q.x() - 0.0).abs() < 1e-9 && (q.y() - 0.0).abs() < 1e-9 && (q.z() - 0.0).abs() < 1e-9);
    }

    #[test]
    fn closest_point_brute_force_parity() {
        // random points: BVH answer must equal brute force over all triangles
        let mut rng = crate::rng::Rng::new(1234);
        let m = crate::geo::prims::sphere(10.0, 24);
        let bvh = Bvh::build(&m.0);
        for _ in 0..200 {
            let p = V3::new(
                (rng.next_f64() - 0.5) * 40.0,
                (rng.next_f64() - 0.5) * 40.0 + 10.0,
                (rng.next_f64() - 0.5) * 40.0,
            );
            let q_bvh = bvh.closest_point(&m.0, &p).unwrap();
            let mut best = V3::ZERO;
            let mut bd2 = f64::INFINITY;
            for t in 0..m.0.tris.len() {
                let (a, b, c) = m.0.tri(t);
                let q = closest_point_on_tri(&p, &a, &b, &c);
                let d2 = q.sub(&p).dot(&q.sub(&p));
                if d2 < bd2 {
                    bd2 = d2;
                    best = q;
                }
            }
            let dv = q_bvh.sub(&best).len();
            assert!(dv < 1e-9, "bvh {} vs brute {} (Δ {})", q_bvh.sub(&p).len(), best.sub(&p).len(), dv);
        }
    }

    #[test]
    fn ray_hits_and_misses() {
        let m = cube_mesh();
        let bvh = Bvh::build(&m);
        // straight down through the top (off the face diagonals)
        let hit = bvh.ray_hit(&m, &V3::new(0.5, 9.0, 0.5), &V3::new(0.0, -1.0, 0.0)).unwrap();
        assert!((hit.0 - 7.0).abs() < 1e-9, "t={}", hit.0);
        // miss
        assert!(bvh.ray_hit(&m, &V3::new(9.0, 9.0, 9.0), &V3::new(0.0, -1.0, 0.0)).is_none());
        // through and through: parity 2 — entry (0.25, 0.75) is off both
        // face diagonals (z = x), so exactly one triangle per face is hit
        let all = bvh.ray_all(&m, &V3::new(0.25, 9.0, 0.75), &V3::new(0.0, -1.0, 0.0));
        assert_eq!(all.len(), 2);
    }

    #[test]
    fn inside_by_parity() {
        let m = crate::geo::prims::sphere(10.0, 24);
        let bvh = Bvh::build(&m.0);
        assert!(bvh.contains(&m.0, &V3::new(0.0, 10.0, 0.0)));
        assert!(bvh.contains(&m.0, &V3::new(3.0, 10.0, -2.0)));
        assert!(bvh.contains(&m.0, &V3::new(9.0, 10.0, 0.0)), "r=9 of 10 is inside");
        assert!(!bvh.contains(&m.0, &V3::new(0.0, 30.0, 0.0)));
        // dead-center cast — poles/vertices are clustered crossings, parity holds
        assert!(bvh.contains(&m.0, &V3::new(0.0, 10.0, 0.0)));
    }

    #[test]
    fn tree_shape() {
        let m = crate::geo::prims::sphere(10.0, 24);
        let bvh = Bvh::build(&m.0);
        assert!(bvh.node_count() > 1);
        assert!(bvh.depth() >= 2 && bvh.depth() < 40, "depth {}", bvh.depth());
        // empty mesh → empty tree
        let e = Bvh::build(&Mesh::new());
        assert!(e.is_empty());
        assert!(e.ray_hit(&Mesh::new(), &V3::ZERO, &V3::new(0.0, 1.0, 0.0)).is_none());
    }
}
