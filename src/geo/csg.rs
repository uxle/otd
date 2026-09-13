//! P0300–P0310 — BSP-tree CSG booleans (the classic csg.js algorithm,
//! reimplemented from scratch in Rust with f64 millimeter coordinates).
//!
//! Design notes:
//! - Plane and polygon data are VALUES (Rust clones) — the "shared plane
//!   double-flip" bug class from reference implementations is impossible here.
//! - Split-plane choice: the largest-area polygon (better tree balance).
//! - AABB fast paths skip BSP entirely for disjoint solids.

use crate::geo::mesh::Mesh;
use crate::math3::{Aabb, V3};

/// Plane epsilon in mm: above f64 rounding noise (≈1e-10 for values ~1000),
/// far below any real feature (walls are ≥ 0.1 mm).
const EPS: f64 = 1e-4;

#[derive(Clone, Copy, Debug)]
pub struct Plane {
    pub n: V3,
    pub w: f64,
}

impl Plane {
    pub fn from_points(a: &V3, b: &V3, c: &V3) -> Plane {
        let n = b.sub(a).cross(&c.sub(a));
        let l = n.len();
        if l < 1e-12 {
            return Plane { n: V3::new(0.0, 1.0, 0.0), w: 0.0 };
        }
        let n = n.mul(1.0 / l);
        Plane { w: n.dot(a), n }
    }
    pub fn flip(&mut self) {
        self.n = self.n.mul(-1.0);
        self.w = -self.w;
    }
    pub fn dist(&self, p: &V3) -> f64 { self.n.dot(p) - self.w }
}

#[derive(Clone, Debug)]
pub struct Poly {
    pub verts: Vec<V3>,
    pub plane: Plane,
}

impl Poly {
    pub fn from_tri(a: V3, b: V3, c: V3) -> Poly {
        let plane = Plane::from_points(&a, &b, &c);
        Poly { verts: vec![a, b, c], plane }
    }
    pub fn flip(&mut self) {
        self.verts.reverse();
        self.plane.flip();
    }
}

#[derive(Default)]
pub struct Node {
    pub plane: Option<Plane>,
    pub front: Option<Box<Node>>,
    pub back: Option<Box<Node>>,
    pub coplanar: Vec<Poly>,
}

fn mesh_to_polys(m: &Mesh) -> Vec<Poly> {
    let mut out = Vec::with_capacity(m.tris.len());
    for t in 0..m.tris.len() {
        let (a, b, c) = m.tri(t);
        out.push(Poly::from_tri(a, b, c));
    }
    out
}

fn polys_to_mesh(polys: &[Poly]) -> Mesh {
    let mut m = Mesh::new();
    for p in polys {
        if p.verts.len() < 3 {
            continue;
        }
        for i in 1..p.verts.len() - 1 {
            let (a, b, c) = (&p.verts[0], &p.verts[i], &p.verts[i + 1]);
            m.add_tri_pts(*a, *b, *c);
        }
    }
    m
}

impl Node {
    pub fn new(polys: Vec<Poly>) -> Box<Node> {
        let mut n = Box::new(Node::default());
        n.build(polys);
        n
    }

    /// Insert polygons into the tree, splitting as needed.
    pub fn build(&mut self, mut polys: Vec<Poly>) {
        if polys.is_empty() {
            return;
        }
        if self.plane.is_none() {
            // largest-area polygon as split plane
            let mut best = 0usize;
            let mut best_area = -1.0;
            for (i, p) in polys.iter().enumerate() {
                let area = plane_poly_area(p);
                if area > best_area {
                    best_area = area;
                    best = i;
                }
            }
            self.plane = Some(polys[best].plane);
        }
        let plane = self.plane.unwrap();
        let mut front = Vec::new();
        let mut back = Vec::new();
        for p in polys.drain(..) {
            for piece in split_poly(&plane, p) {
                match piece {
                    Piece::CoplanarFront(p) | Piece::CoplanarBack(p) => self.coplanar.push(p),
                    Piece::Front(p) => front.push(p),
                    Piece::Back(p) => back.push(p),
                }
            }
        }
        if !front.is_empty() {
            if self.front.is_none() {
                self.front = Some(Box::new(Node::default()));
            }
            self.front.as_mut().unwrap().build(front);
        }
        if !back.is_empty() {
            if self.back.is_none() {
                self.back = Some(Box::new(Node::default()));
            }
            self.back.as_mut().unwrap().build(back);
        }
    }

    pub fn invert(&mut self) {
        for p in self.coplanar.iter_mut() {
            p.flip();
        }
        if let Some(f) = self.front.as_mut() {
            f.invert();
        }
        if let Some(b) = self.back.as_mut() {
            b.invert();
        }
        std::mem::swap(&mut self.front, &mut self.back);
        if let Some(pl) = self.plane.as_mut() {
            pl.flip();
        }
    }

    /// Remove all polygons in `polys` that are inside this tree.
    pub fn clip_polys(&self, polys: Vec<Poly>) -> Vec<Poly> {
        match self.plane {
            None => polys,
            Some(plane) => {
                let mut front = Vec::new();
                let mut back = Vec::new();
                for p in polys {
                    for piece in split_poly(&plane, p) {
                        // (matches csg.js: coplanar-front joins the front list,
                        //  coplanar-back joins the back list)
                        match piece {
                            Piece::CoplanarFront(p) | Piece::Front(p) => front.push(p),
                            Piece::CoplanarBack(p) | Piece::Back(p) => back.push(p),
                        }
                    }
                }
                let mut out = match self.front.as_ref() {
                    Some(f) => f.clip_polys(front),
                    None => front,
                };
                let back_out = match self.back.as_ref() {
                    Some(b) => b.clip_polys(back),
                    // closed solid: everything behind a leaf plane is inside → drop
                    None => Vec::new(),
                };
                out.extend(back_out);
                out
            }
        }
    }

    /// Remove all polygons in this tree that are inside `other`.
    pub fn clip_to(&mut self, other: &Node) {
        let mine = std::mem::take(&mut self.coplanar);
        self.coplanar = other.clip_polys(mine);
        if let Some(f) = self.front.as_mut() {
            f.clip_to(other);
        }
        if let Some(b) = self.back.as_mut() {
            b.clip_to(other);
        }
    }

    pub fn all_polys(&self) -> Vec<Poly> {
        let mut out = self.coplanar.clone();
        if let Some(f) = self.front.as_ref() {
            out.extend(f.all_polys());
        }
        if let Some(b) = self.back.as_ref() {
            out.extend(b.all_polys());
        }
        out
    }
}

fn plane_poly_area(p: &Poly) -> f64 {
    if p.verts.len() < 3 {
        return 0.0;
    }
    let n = p.verts[1].sub(&p.verts[0]).cross(&p.verts[2].sub(&p.verts[0]));
    n.len() * 0.5
}

const COPLANAR: u8 = 0;
const FRONT: u8 = 1;
const BACK: u8 = 2;
const SPANNING: u8 = 3;

/// Classification of one polygon against a plane.
enum Piece {
    CoplanarFront(Poly),
    CoplanarBack(Poly),
    Front(Poly),
    Back(Poly),
}

/// Route one polygon against `plane`, splitting spanning polygons.
fn split_poly(plane: &Plane, poly: Poly) -> Vec<Piece> {
    let mut ptype: u8 = COPLANAR;
    for v in &poly.verts {
        let t = plane.dist(v);
        let t2 = if t < -EPS { BACK } else if t > EPS { FRONT } else { COPLANAR };
        ptype |= t2;
    }
    match ptype {
        COPLANAR => {
            let same_side = plane.n.dot(&poly.plane.n) > 0.0;
            if same_side {
                vec![Piece::CoplanarFront(poly)]
            } else {
                vec![Piece::CoplanarBack(poly)]
            }
        }
        FRONT => vec![Piece::Front(poly)],
        BACK => vec![Piece::Back(poly)],
        _ => {
            // SPANNING: split into front and back pieces
            let mut f: Vec<V3> = Vec::new();
            let mut b: Vec<V3> = Vec::new();
            let n = poly.verts.len();
            for i in 0..n {
                let vi = poly.verts[i];
                let vj = poly.verts[(i + 1) % n];
                let ti = plane.dist(&vi);
                let tj = plane.dist(&vj);
                if ti > -EPS {
                    f.push(vi);
                }
                if ti < EPS {
                    b.push(vi);
                }
                if (ti < -EPS && tj > EPS) || (ti > EPS && tj < -EPS) {
                    let d = plane.n.dot(&vj.sub(&vi));
                    if d.abs() > 1e-12 {
                        let t = (plane.w - plane.n.dot(&vi)) / d;
                        let v = vi.lerp(&vj, t);
                        f.push(v);
                        b.push(v);
                    }
                }
            }
            let plane_p = poly.plane;
            let mut out = Vec::new();
            if f.len() >= 3 {
                out.push(Piece::Front(Poly { verts: f, plane: plane_p }));
            }
            if b.len() >= 3 {
                out.push(Piece::Back(Poly { verts: b, plane: plane_p }));
            }
            out
        }
    }
}

// ---- mesh-level boolean operations ----

fn boxes_overlap(a: &Mesh, b: &Mesh) -> bool {
    if a.is_empty() || b.is_empty() {
        return false;
    }
    a.bbox().intersects(&b.bbox())
}

fn merge_two(a: &Mesh, b: &Mesh) -> Mesh {
    let mut m = a.clone();
    m.merge(b);
    m
}

/// `a + b` — union.
pub fn mesh_union(a: &Mesh, b: &Mesh) -> Mesh {
    if a.is_empty() {
        return b.clone();
    }
    if b.is_empty() {
        return a.clone();
    }
    if !boxes_overlap(a, b) {
        return merge_two(a, b);
    }
    let mut an = Node::new(mesh_to_polys(a));
    let mut bn = Node::new(mesh_to_polys(b));
    an.clip_to(&bn);
    bn.clip_to(&an);
    bn.invert();
    bn.clip_to(&an);
    bn.invert();
    let bpolys = bn.all_polys();
    an.build(bpolys);
    polys_to_mesh(&an.all_polys())
}

/// `a - b` — subtract.
pub fn mesh_subtract(a: &Mesh, b: &Mesh) -> Mesh {
    if a.is_empty() || b.is_empty() {
        return a.clone();
    }
    if !boxes_overlap(a, b) {
        return a.clone();
    }
    let mut an = Node::new(mesh_to_polys(a));
    let mut bn = Node::new(mesh_to_polys(b));
    an.invert();
    an.clip_to(&bn);
    bn.clip_to(&an);
    bn.invert();
    bn.clip_to(&an);
    bn.invert();
    let bpolys = bn.all_polys();
    an.build(bpolys);
    an.invert();
    polys_to_mesh(&an.all_polys())
}

/// `a & b` — intersect.
pub fn mesh_intersect(a: &Mesh, b: &Mesh) -> Mesh {
    if a.is_empty() || b.is_empty() {
        return Mesh::new();
    }
    if !boxes_overlap(a, b) {
        return Mesh::new();
    }
    let mut an = Node::new(mesh_to_polys(a));
    let mut bn = Node::new(mesh_to_polys(b));
    an.invert();
    bn.clip_to(&an);
    bn.invert();
    an.clip_to(&bn);
    bn.clip_to(&an);
    let bpolys = bn.all_polys();
    an.build(bpolys);
    an.invert();
    polys_to_mesh(&an.all_polys())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::prims::cube;

    #[test]
    fn union_overlapping_cubes() {
        // two 20mm cubes offset by 10 → overlap 10×20×20 = 4000 → union 12000
        let a = cube(20.0, 20.0, 20.0).0;
        let b = cube(20.0, 20.0, 20.0).0.translated(V3::new(10.0, 0.0, 0.0));
        let u = mesh_union(&a, &b);
        let v = u.volume_signed();
        assert!((v - 12000.0).abs() / 12000.0 < 0.01, "volume {}", v);
    }

    #[test]
    fn union_disjoint_is_concat() {
        let a = cube(10.0, 10.0, 10.0).0;
        let b = cube(10.0, 10.0, 10.0).0.translated(V3::new(100.0, 0.0, 0.0));
        let u = mesh_union(&a, &b);
        assert!((u.volume_signed() - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn subtract_makes_a_hollow_box() {
        // 20mm cube minus a 16mm cube inset 2mm on every side → walls 2mm
        let a = cube(20.0, 20.0, 20.0).0;
        let b = cube(16.0, 16.0, 16.0).0.translated(V3::new(0.0, 2.0, 0.0));
        let d = mesh_subtract(&a, &b);
        let v = d.volume_signed().abs();
        // expected 20³ − 16³ = 8000 − 4096 = 3904
        assert!((v - 3904.0).abs() / 3904.0 < 0.02, "volume {}", v);
    }

    #[test]
    fn subtract_half_cube() {
        // cube 20 minus cube 20 shifted x+10 → 8000 − 4000 = 4000
        let a = cube(20.0, 20.0, 20.0).0;
        let b = cube(20.0, 20.0, 20.0).0.translated(V3::new(10.0, 0.0, 0.0));
        let d = mesh_subtract(&a, &b);
        let v = d.volume_signed().abs();
        assert!((v - 4000.0).abs() / 4000.0 < 0.02, "volume {}", v);
    }

    #[test]
    fn intersect_half_cube() {
        let a = cube(20.0, 20.0, 20.0).0;
        let b = cube(20.0, 20.0, 20.0).0.translated(V3::new(10.0, 0.0, 0.0));
        let i = mesh_intersect(&a, &b);
        let v = i.volume_signed().abs();
        assert!((v - 4000.0).abs() / 4000.0 < 0.02, "volume {}", v);
    }

    #[test]
    fn identities() {
        let a = cube(10.0, 10.0, 10.0).0;
        // A − A = ∅
        let d = mesh_subtract(&a, &a);
        assert!(d.is_empty() || d.volume_signed().abs() < 1e-6, "{}", d.volume_signed());
        // A & A = A
        let i = mesh_intersect(&a, &a);
        assert!((i.volume_signed() - 1000.0).abs() / 1000.0 < 0.01);
    }

    #[test]
    fn union_sphere_cube_roundtrip_volume() {
        // sphere r=10 at center of cube 20 → union ≈ 8000 + 4188 − overlap
        use crate::geo::prims::sphere;
        let c = cube(20.0, 20.0, 20.0).0;
        let s = sphere(10.0, 32).0.translated(V3::new(0.0, 10.0, 0.0));
        let u = mesh_union(&c, &s);
        // sphere() rests with center at y=r → after +10 translate, center is at
        // the cube TOP face → union = cube + half sphere ≈ 8000 + 2094
        let v = u.volume_signed();
        assert!((v - 10094.4).abs() / 10094.4 < 0.02, "volume {}", v);
    }
}
