//! P0800 — the half-edge mesh structure (the v5 `HalfEdgeStandard` layout,
//! pure Rust). Built from any triangle soup in O(n): one half-edge per
//! directed edge, twins glued across faces. Powers the one-ring walks that
//! `smooth` (DEC) and `subdiv` (Loop) need, and validates manifoldness.
//!
//! Walk conventions (CCW face winding, outward normals):
//!   · outgoing edge at v:  origin(e) = target(prev(e)) = v
//!   · next outgoing CCW:   e ← next(twin(e))
//!   · next outgoing CW:    e ← twin(prev(e))
//! Boundary = twinless half-edge. A closed mesh has none.

use crate::geo::mesh::Mesh;
use crate::math3::V3;

pub const UNSET: u32 = 0xFFFF_FFFF;
const GUARD: u64 = 8_000_000;

/// A half-edge structure over a mesh, **welded** (coincident vertices
/// merged — point-built prims duplicate corners and seams). Owns the welded
/// vertex positions + topology; `tris()` reconstructs the triangle list.
#[derive(Clone, Debug)]
pub struct HalfEdges {
    /// welded vertex positions
    pub verts: Vec<V3>,
    /// half-edge → vertex it points at
    pub target: Vec<u32>,
    /// half-edge → opposite half-edge (twin), UNSET on boundary
    pub twin: Vec<u32>,
    /// half-edge → next in the same face (CCW)
    pub next: Vec<u32>,
    /// half-edge → previous in the same face
    pub prev: Vec<u32>,
    /// half-edge → face (triangle) index
    pub face: Vec<u32>,
    /// vertex → one outgoing half-edge (UNSET for isolated verts)
    pub vert_out: Vec<u32>,
}

impl HalfEdges {
    /// Build topology for a triangle mesh (welding first). Winding is
    /// whatever the mesh has; twins are found by directed-edge matching,
    /// so consistent winding gives every interior half-edge a twin.
    pub fn build(mesh: &Mesh) -> HalfEdges {
        let welded = mesh.welded();
        let nv = welded.verts.len();
        let nt = welded.tris.len();
        let nhe = 3 * nt;
        let mut he = HalfEdges {
            verts: welded.verts,
            target: vec![UNSET; nhe],
            twin: vec![UNSET; nhe],
            next: vec![UNSET; nhe],
            prev: vec![UNSET; nhe],
            face: vec![UNSET; nhe],
            vert_out: vec![UNSET; nv],
        };
        if nt == 0 {
            he.target.clear();
            he.twin.clear();
            he.next.clear();
            he.prev.clear();
            he.face.clear();
            return he;
        }
        // pass 1: targets + face + next/prev within each triangle
        for (t, tri) in welded.tris.iter().enumerate() {
            let base = (3 * t) as u32;
            for k in 0..3 {
                let e = base + k as u32;
                he.target[e as usize] = tri[k];
                he.face[e as usize] = t as u32;
                he.next[e as usize] = base + ((k + 1) % 3) as u32;
                he.prev[e as usize] = base + ((k + 2) % 3) as u32;
            }
        }
        // pass 2: twins — a twin is the *reverse* directed edge (b→a for
        // a→b). Map every directed edge to its half-edge, then pair them.
        let mut dir_map: std::collections::HashMap<(u32, u32), u32> =
            std::collections::HashMap::with_capacity(nhe);
        for e in 0..nhe as u32 {
            let o = he.origin(e);
            let d = he.target[e as usize];
            dir_map.insert((o, d), e);
        }
        for e in 0..nhe as u32 {
            let o = he.origin(e);
            let d = he.target[e as usize];
            if o < d {
                // set once per undirected edge (o < d canonical order)
                if let Some(&t) = dir_map.get(&(d, o)) {
                    he.twin[e as usize] = t;
                    he.twin[t as usize] = e;
                }
                // duplicated directed edges (o→o twice) stay twinless —
                // non-manifold, flagged by open rings in the report
            }
        }
        // pass 3: one outgoing half-edge per vertex — prefer an interior one
        // (has a twin) so ring walks start away from the boundary
        for e in 0..nhe as u32 {
            let o = he.origin(e) as usize;
            if he.vert_out[o] == UNSET || he.twin[e as usize] != UNSET {
                he.vert_out[o] = e;
            }
        }
        he
    }

    /// origin vertex of a half-edge (the vertex it starts from).
    pub fn origin(&self, e: u32) -> u32 {
        self.target[self.prev[e as usize] as usize]
    }

    /// The apex (opposite) vertex of the face containing half-edge e — the
    /// third vertex of the triangle, across the undirected edge (e, twin).
    /// Cotangent weights need it from both sides. For e = a→b in face
    /// (a, b, c): next(e) = b→c, so apex = target[next(e)] = c. ✓
    pub fn apex(&self, e: u32) -> u32 {
        self.target[self.next[e as usize] as usize]
    }

    /// Iterate the one-ring (neighbors) of vertex v. Returns None only if
    /// the walk cannot close within the guard (damaged topology).
    pub fn ring(&self, v: usize) -> Option<Vec<u32>> {
        let start = self.vert_out[v];
        if start == UNSET {
            return Some(Vec::new());
        }
        let mut out = Vec::new();
        // CCW walk: push target, advance over twins
        let mut e = start;
        let mut guard = 0u64;
        loop {
            out.push(self.target[e as usize]);
            let t = self.twin[e as usize];
            if t == UNSET {
                break; // boundary — now walk the other way
            }
            e = self.next[t as usize];
            if e == start {
                return Some(out); // closed cycle
            }
            guard += 1;
            if guard > GUARD {
                return None;
            }
        }
        // CW walk: e ← twin(prev(e)); stop at a twinless incoming edge p and
        // take origin(p) as the final neighbor
        let mut e = start;
        let mut guard = 0u64;
        loop {
            let p = self.prev[e as usize];
            let t = self.twin[p as usize];
            if t == UNSET {
                // p is a boundary half-edge INTO v; its origin is a neighbor
                out.push(self.target[self.prev[p as usize] as usize]);
                break;
            }
            e = t;
            if e == start {
                break;
            }
            out.push(self.target[e as usize]);
            guard += 1;
            if guard > GUARD {
                return None;
            }
        }
        Some(out)
    }

    /// Is v on the boundary (touching a twinless edge)?
    pub fn is_boundary_vertex(&self, v: usize) -> bool {
        let start = self.vert_out[v];
        if start == UNSET {
            return false;
        }
        let mut e = start;
        let mut guard = 0u64;
        loop {
            if self.twin[e as usize] == UNSET {
                return true;
            }
            e = self.next[self.twin[e as usize] as usize];
            if e == start {
                return false;
            }
            guard += 1;
            if guard > GUARD {
                return false;
            }
        }
    }

    /// The two boundary neighbors of a boundary vertex v (the ends of its
    /// open fan) — used by the Catmull-Rom boundary rule in subdivision.
    pub fn boundary_neighbors(&self, v: usize) -> Option<(u32, u32)> {
        let start = self.vert_out[v];
        if start == UNSET || !self.is_boundary_vertex(v) {
            return None;
        }
        // CCW end: first twinless outgoing edge
        let mut e = start;
        let mut guard = 0u64;
        while self.twin[e as usize] != UNSET {
            e = self.next[self.twin[e as usize] as usize];
            guard += 1;
            if guard > GUARD || e == start {
                return None; // not actually boundary
            }
        }
        let ccw_end = self.target[e as usize];
        // CW end: walk twin(prev(·)) until the incoming edge is twinless
        let mut e2 = start;
        let mut guard = 0u64;
        loop {
            let p = self.prev[e2 as usize];
            let t = self.twin[p as usize];
            if t == UNSET {
                let cw_end = self.target[self.prev[p as usize] as usize];
                return Some((ccw_end, cw_end));
            }
            e2 = t;
            if e2 == start {
                return None;
            }
            guard += 1;
            if guard > GUARD {
                return None;
            }
        }
    }

    /// Reconstruct the welded triangle list (consistent with `face`).
    pub fn tris(&self) -> Vec<[u32; 3]> {
        let mut out = Vec::with_capacity(self.face.len() / 3);
        for t in 0..self.face.len() / 3 {
            let base = 3 * t;
            out.push([self.target[base], self.target[base + 1], self.target[base + 2]]);
        }
        out
    }

    /// The welded mesh this topology was built from.
    pub fn to_mesh(&self) -> Mesh {
        Mesh { verts: self.verts.clone(), tris: self.tris() }
    }

    /// Manifoldness summary — `ask "watertight?"` prints this. Boundary
    /// edges are legal (open meshes); what matters is closed ⇔ no boundary.
    pub fn manifold_report(&self) -> ManifoldReport {
        let mut boundary = 0usize;
        for e in 0..self.twin.len() {
            if self.twin[e] == UNSET {
                boundary += 1;
            }
        }
        // a boundary edge appears exactly once as a directed half-edge
        let mut open_rings = 0usize;
        for v in 0..self.verts.len() {
            if self.vert_out[v] == UNSET {
                continue;
            }
            if self.ring(v).is_none() {
                open_rings += 1;
            }
        }
        ManifoldReport {
            boundary_edges: boundary,
            open_vertex_rings: open_rings,
            closed: boundary == 0,
        }
    }
}

/// Result of the manifold check.
#[derive(Clone, Copy, Debug, Default)]
pub struct ManifoldReport {
    pub boundary_edges: usize,
    pub open_vertex_rings: usize,
    pub closed: bool,
}

/// Angle (radians) at apex c of the triangle (a, b, c).
pub fn angle_at(a: V3, b: V3, c: V3) -> f64 {
    let u = a.sub(&c);
    let v = b.sub(&c);
    let d = u.dot(&v) / (u.len() * v.len());
    d.clamp(-1.0, 1.0).acos()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tri_mesh(a: V3, b: V3, c: V3) -> Mesh {
        let mut m = Mesh::new();
        m.add_tri_pts(a, b, c);
        m
    }

    #[test]
    fn single_triangle() {
        let m = tri_mesh(V3::ZERO, V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0));
        let he = HalfEdges::build(&m);
        assert_eq!(he.target.len(), 3);
        assert_eq!(he.twin.iter().filter(|t| **t == UNSET).count(), 3);
        // corner 0 (the origin) has neighbors 1 and 2
        let ring = he.ring(0).unwrap();
        assert_eq!(ring.len(), 2);
        assert!(he.is_boundary_vertex(0));
        let (n1, n2) = he.boundary_neighbors(0).unwrap();
        // both other vertices are "boundary neighbors"
        assert!(n1 != n2);
    }

    #[test]
    fn two_triangles_glued() {
        // quad made of 2 triangles sharing edge 0-1
        let mut m = Mesh::new();
        let a = m.add_vert(V3::ZERO);
        let b = m.add_vert(V3::new(1.0, 0.0, 0.0));
        let c = m.add_vert(V3::new(1.0, 1.0, 0.0));
        let d = m.add_vert(V3::new(0.0, 1.0, 0.0));
        m.add_tri(a, b, c);
        m.add_tri(a, c, d);
        let he = HalfEdges::build(&m);
        // interior edge (a,c): directed edges a→c and c→a are twins
        let rep = he.manifold_report();
        assert!(!rep.closed);
        // 4 boundary edges: a→b, b→c, c→d… wait: edges of both triangles
        // minus the shared diagonal = 4 boundary edges
        assert_eq!(rep.boundary_edges, 4, "boundary {:?}", rep);
        // vertex b ring = {a, c}
        let ring = he.ring(b as usize).unwrap();
        assert_eq!(ring.len(), 2);
    }

    #[test]
    fn sphere_closed_and_rings() {
        let (m, _) = crate::geo::prims::sphere(10.0, 16);
        let he = HalfEdges::build(&m);
        let rep = he.manifold_report();
        assert!(rep.closed, "sphere must be closed {:?}", rep);
        // north pole ring = meridians = 16
        let ring = he.ring(0).unwrap();
        assert_eq!(ring.len(), 16, "pole ring");
        // grid interior vertex ring = 6 (hex neighborhood)
        let ring1 = he.ring(17).unwrap();
        assert_eq!(ring1.len(), 6, "interior ring, got {:?}", ring1.len());
        let mut s = ring1.clone();
        s.sort();
        s.dedup();
        assert_eq!(s.len(), ring1.len(), "ring must be a simple cycle");
        assert!(!he.is_boundary_vertex(0));
    }

    #[test]
    fn plane_slab_is_closed() {
        // `plane` is a thin closed slab — the topology must say watertight
        let (m, _) = crate::geo::prims::plane(20.0, 20.0, 1.0);
        let he = HalfEdges::build(&m);
        let rep = he.manifold_report();
        assert!(rep.closed, "plane is a closed slab: {:?}", rep);
    }

    #[test]
    fn open_fan_has_boundary() {
        // genuinely open mesh: a strip of 3 triangles
        let mut m = Mesh::new();
        let a = m.add_vert(V3::ZERO);
        let b = m.add_vert(V3::new(1.0, 0.0, 0.0));
        let c = m.add_vert(V3::new(1.0, 1.0, 0.0));
        let d = m.add_vert(V3::new(2.0, 0.0, 0.0));
        let e = m.add_vert(V3::new(2.0, 1.0, 0.0));
        m.add_tri(a, b, c);
        m.add_tri(b, d, e);
        m.add_tri(b, e, c);
        let he = HalfEdges::build(&m);
        let rep = he.manifold_report();
        assert!(!rep.closed);
        assert!(rep.boundary_edges > 0);
    }

    #[test]
    fn angles() {
        let a = angle_at(V3::new(1.0, 0.0, 0.0), V3::new(0.0, 1.0, 0.0), V3::ZERO);
        assert!((a - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        let e = angle_at(V3::new(1.0, 0.0, 0.0), V3::new(0.5, 0.86602540378, 0.0), V3::ZERO);
        assert!((e - std::f64::consts::FRAC_PI_3).abs() < 1e-9);
    }
}
