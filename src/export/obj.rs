//! P0640b — OBJ exporter with vertex colors (widely supported extension).

use crate::geo::mesh::Mesh;
use crate::world::colors::Color;

pub fn to_obj(m: &Mesh, name: &str, color: Option<Color>) -> String {
    let mut s = String::new();
    s.push_str(&format!("# OTD export — {}\n# units: millimeters\n", name));
    let (r, g, b) = color.map(|c| c.rgb01()).unwrap_or((0.8, 0.8, 0.8));
    for v in &m.verts {
        s.push_str(&format!(
            "v {:.6} {:.6} {:.6} {:.4} {:.4} {:.4}\n",
            v.x(), v.y(), v.z(), r, g, b
        ));
    }
    // vertex normals per face (flat) — keeps importers happy
    for t in 0..m.tris.len() {
        let n = m.face_normal(t).norm();
        s.push_str(&format!("vn {:.6} {:.6} {:.6}\n", n.x(), n.y(), n.z()));
    }
    for (ti, t) in m.tris.iter().enumerate() {
        // OBJ is 1-indexed; use the same normal index for flat shading
        s.push_str(&format!(
            "f {0}//{3} {1}//{3} {2}//{3}\n",
            t[0] + 1, t[1] + 1, t[2] + 1, ti + 1
        ));
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::prims::cube;

    #[test]
    fn obj_has_v_and_f() {
        let (c, _) = cube(10.0, 10.0, 10.0);
        let s = to_obj(&c, "cube", None);
        assert!(s.lines().filter(|l| l.starts_with("v ")).count() == c.verts.len());
        assert!(s.lines().filter(|l| l.starts_with("f ")).count() == c.tris.len());
    }
}
