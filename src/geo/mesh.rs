//! P0200 — the mesh container: indexed triangle soup, f64 millimeters.
//! Closed manifolds guaranteed by `ensure_outward` (divergence-theorem sign).

use crate::math3::{Aabb, M4, V3};

#[derive(Clone, Default, Debug)]
pub struct Mesh {
    pub verts: Vec<V3>,
    pub tris: Vec<[u32; 3]>,
}

impl Mesh {
    pub fn new() -> Mesh { Mesh { verts: Vec::new(), tris: Vec::new() } }

    pub fn add_vert(&mut self, p: V3) -> u32 {
        self.verts.push(p);
        (self.verts.len() - 1) as u32
    }

    pub fn add_tri(&mut self, a: u32, b: u32, c: u32) {
        self.tris.push([a, b, c]);
    }

    /// Triangle from points (allocates 3 verts).
    pub fn add_tri_pts(&mut self, a: V3, b: V3, c: V3) {
        let i = self.add_vert(a);
        let j = self.add_vert(b);
        let k = self.add_vert(c);
        self.add_tri(i, j, k);
    }

    /// Quad given CCW-from-outside; fan-triangulated.
    pub fn add_quad(&mut self, a: V3, b: V3, c: V3, d: V3) {
        let i = self.add_vert(a);
        let j = self.add_vert(b);
        let k = self.add_vert(c);
        let l = self.add_vert(d);
        self.add_tri(i, j, k);
        self.add_tri(i, k, l);
    }

    pub fn tri(&self, t: usize) -> (V3, V3, V3) {
        let [a, b, c] = self.tris[t];
        (self.verts[a as usize], self.verts[b as usize], self.verts[c as usize])
    }

    /// Face normal (not normalized sign guaranteed by winding).
    pub fn face_normal(&self, t: usize) -> V3 {
        let (a, b, c) = self.tri(t);
        b.sub(&a).cross(&c.sub(&a))
    }

    /// Flip all triangles (reverse winding).
    pub fn flip_all(&mut self) {
        for t in self.tris.iter_mut() {
            t.swap(1, 2);
        }
    }

    /// Signed volume via the divergence theorem: V = Σ det(a,b,c)/6.
    /// Positive ⇔ outward winding. Watertightness ⇔ stable under refinement.
    pub fn volume_signed(&self) -> f64 {
        let mut v = 0.0;
        for t in 0..self.tris.len() {
            let (a, b, c) = self.tri(t);
            v += a.dot(&b.cross(&c));
        }
        v / 6.0
    }

    /// Centroid of a closed mesh (volume-weighted).
    pub fn centroid(&self) -> Option<V3> {
        let mut cx = [0.0f64; 3];
        let mut vol = 0.0;
        for t in 0..self.tris.len() {
            let (a, b, c) = self.tri(t);
            let det = a.dot(&b.cross(&c));
            vol += det;
            for k in 0..3 {
                cx[k] += det * (a.0[k] + b.0[k] + c.0[k]);
            }
        }
        if vol.abs() < 1e-12 {
            return None;
        }
        // centroid = Σ det·(a+b+c)/4 / (Σ det/6·2)?? — exact: C = Σ det·(a+b+c) / (4·Σdet)
        Some(V3([
            cx[0] / (4.0 * vol),
            cx[1] / (4.0 * vol),
            cx[2] / (4.0 * vol),
        ]))
    }

    /// Surface area: Σ |cross| / 2.
    pub fn area(&self) -> f64 {
        let mut a = 0.0;
        for t in 0..self.tris.len() {
            a += self.face_normal(t).len();
        }
        a * 0.5
    }

    /// Guarantee outward winding for a closed mesh: if the signed volume is
    /// negative, flip everything. (Volume sign is global for consistent winding.)
    pub fn ensure_outward(&mut self) {
        if self.volume_signed() < 0.0 {
            self.flip_all();
        }
    }

    pub fn transform(&mut self, m: &M4) {
        for v in self.verts.iter_mut() {
            *v = m.apply(v);
        }
    }

    pub fn translated(&self, t: V3) -> Mesh {
        let mut m = self.clone();
        m.transform(&M4::translate(t.x(), t.y(), t.z()));
        m
    }

    pub fn merge(&mut self, other: &Mesh) {
        let base = self.verts.len() as u32;
        for v in &other.verts {
            self.verts.push(*v);
        }
        for t in &other.tris {
            self.tris.push([t[0] + base, t[1] + base, t[2] + base]);
        }
    }

    /// Merge multiple meshes into one (multi-component solid).
    pub fn from_meshes(meshes: &[Mesh]) -> Mesh {
        let mut out = Mesh::new();
        for m in meshes {
            out.merge(m);
        }
        out
    }

    pub fn bbox(&self) -> Aabb {
        let mut bb = Aabb::empty();
        for v in &self.verts {
            bb.grow_pt(v);
        }
        bb
    }

    pub fn is_empty(&self) -> bool { self.tris.is_empty() }

    /// Weld coincident vertices (within 10⁻⁶ mm) and re-index triangles.
    /// Point-based builders (add_quad/add_tri_pts) duplicate corners; the
    /// topology tiers (smooth, subdiv) need shared vertices to see one-rings.
    /// Exact duplicates — which is what all OTD builders produce — merge
    /// perfectly; near-duplicates below a micron are the same point anyway.
    pub fn welded(&self) -> Mesh {
        if self.verts.is_empty() {
            return self.clone();
        }
        let mut map: std::collections::HashMap<(i64, i64, i64), u32> =
            std::collections::HashMap::with_capacity(self.verts.len());
        let mut new_verts: Vec<V3> = Vec::with_capacity(self.verts.len());
        let mut remap: Vec<u32> = vec![u32::MAX; self.verts.len()];
        for (i, v) in self.verts.iter().enumerate() {
            // quantize to a 10⁻⁶ mm lattice
            let key = (
                (v.x() * 1e6).round() as i64,
                (v.y() * 1e6).round() as i64,
                (v.z() * 1e6).round() as i64,
            );
            match map.get(&key) {
                Some(rep) => remap[i] = *rep,
                None => {
                    let id = new_verts.len() as u32;
                    new_verts.push(*v);
                    map.insert(key, id);
                    remap[i] = id;
                }
            }
        }
        let mut out = Mesh { verts: new_verts, tris: Vec::with_capacity(self.tris.len()) };
        for t in &self.tris {
            let [a, b, c] = *t;
            // drop degenerate triangles created by welding
            let (a, b, c) = (remap[a as usize], remap[b as usize], remap[c as usize]);
            if a != b && b != c && a != c {
                out.tris.push([a, b, c]);
            }
        }
        out
    }

    /// Vertex normals (area-weighted average of adjacent face normals).
    /// For CSG results this gives smooth shading across shared edges.
    pub fn vertex_normals(&self) -> Vec<V3> {
        let mut n = vec![V3::ZERO; self.verts.len()];
        for t in 0..self.tris.len() {
            let fn_ = self.face_normal(t); // area-weighted (length = 2×area)
            let [a, b, c] = self.tris[t];
            for idx in [a, b, c] {
                let idx = idx as usize;
                n[idx] = n[idx].add(&fn_);
            }
        }
        for v in n.iter_mut() {
            let l = v.len();
            if l > 1e-20 {
                *v = v.mul(1.0 / l);
            }
        }
        n
    }
}

/// What a mesh was built from — lets `hollow` use exact analytic inner shapes
/// instead of approximations (P0320).
#[derive(Clone, Debug)]
pub enum Kind {
    Box { w: f64, d: f64, h: f64 },
    Frustum { top: f64, bottom: f64, h: f64 },
    Sphere { r: f64 },
    Torus { r_main: f64, tube: f64 },
    Capsule { r: f64, h: f64 },
    Prism { sides: u32, r: f64, h: f64 },
    Revolve { profile: Vec<(f64, f64)>, angle: f64 },
    Generic,
}

impl Default for Kind {
    fn default() -> Self { Kind::Generic }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_cube() -> Mesh {
        // cube 0..2 (w=d=h=2), built with add_quad
        let mut m = Mesh::new();
        let (x0, x1) = (0.0, 2.0);
        let (y0, y1) = (0.0, 2.0);
        let (z0, z1) = (0.0, 2.0);
        // +y top
        m.add_quad(V3::new(x0, y1, z0), V3::new(x0, y1, z1), V3::new(x1, y1, z1), V3::new(x1, y1, z0));
        // -y bottom
        m.add_quad(V3::new(x0, y0, z0), V3::new(x1, y0, z0), V3::new(x1, y0, z1), V3::new(x0, y0, z1));
        // +z front
        m.add_quad(V3::new(x0, y0, z1), V3::new(x1, y0, z1), V3::new(x1, y1, z1), V3::new(x0, y1, z1));
        // -z back
        m.add_quad(V3::new(x0, y0, z0), V3::new(x0, y1, z0), V3::new(x1, y1, z0), V3::new(x1, y0, z0));
        // +x right
        m.add_quad(V3::new(x1, y0, z0), V3::new(x1, y1, z0), V3::new(x1, y1, z1), V3::new(x1, y0, z1));
        // -x left
        m.add_quad(V3::new(x0, y0, z0), V3::new(x0, y0, z1), V3::new(x0, y1, z1), V3::new(x0, y1, z0));
        m
    }

    #[test]
    fn cube_volume_and_winding() {
        let mut m = unit_cube();
        assert_eq!(m.tris.len(), 12);
        assert!((m.volume_signed() - 8.0).abs() < 1e-9, "volume {}", m.volume_signed());
        m.ensure_outward();
        assert!(m.volume_signed() > 0.0);
        // flipped cube auto-corrects
        let mut f = unit_cube();
        f.flip_all();
        f.ensure_outward();
        assert!(f.volume_signed() > 0.0);
    }

    #[test]
    fn cube_centroid() {
        let mut m = unit_cube();
        m.ensure_outward();
        let c = m.centroid().unwrap();
        assert!((c.x() - 1.0).abs() < 1e-9 && (c.y() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn merge_multi_component() {
        let a = unit_cube();
        let b = unit_cube().translated(V3::new(10.0, 0.0, 0.0));
        let mut m = Mesh::new();
        m.merge(&a);
        m.merge(&b);
        assert!((m.volume_signed() - 16.0).abs() < 1e-9);
    }
}
