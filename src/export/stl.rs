//! P0640a — binary STL exporter (millimeters). 84-byte header + N×50 bytes.

use crate::geo::mesh::Mesh;
use crate::math3::V3;

pub fn to_binary_stl(m: &Mesh, name: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(84 + m.tris.len() * 50);
    // header: 80 bytes, must not start with "solid"
    let mut header = format!("OTD {} — binary STL, mm", name).into_bytes();
    header.resize(80.min(header.len().max(1)), b' ');
    if header.len() < 80 {
        let mut h = vec![b' '; 80];
        h[..header.len()].copy_from_slice(&header);
        header = h;
    }
    // ensure it does not start with "solid" (confuses some parsers)
    let start: String = header.iter().take(5).map(|b| *b as char).collect();
    if start.to_lowercase() == "solid" {
        header[0] = b'O';
    }
    out.extend_from_slice(&header);
    out.extend_from_slice(&(m.tris.len() as u32).to_le_bytes());
    for t in 0..m.tris.len() {
        let (a, b, c) = m.tri(t);
        let n = b.sub(&a).cross(&c.sub(&a)).norm();
        for v in [n, a, b, c] {
            for k in 0..3 {
                out.extend_from_slice(&(v.0[k] as f32).to_le_bytes());
            }
        }
        out.extend_from_slice(&0u16.to_le_bytes()); // attribute byte count
    }
    out
}

/// ASCII STL (for debugging / human reading).
#[allow(dead_code)]
pub fn to_ascii_stl(m: &Mesh, name: &str) -> String {
    let mut s = String::new();
    s.push_str(&format!("solid {}\n", name));
    for t in 0..m.tris.len() {
        let (a, b, c) = m.tri(t);
        let n = b.sub(&a).cross(&c.sub(&a)).norm();
        s.push_str(&format!("  facet normal {:.6} {:.6} {:.6}\n", n.x(), n.y(), n.z()));
        s.push_str("    outer loop\n");
        for v in [a, b, c] {
            s.push_str(&format!("      vertex {:.6} {:.6} {:.6}\n", v.x(), v.y(), v.z()));
        }
        s.push_str("    endloop\n  endfacet\n");
    }
    s.push_str(&format!("endsolid {}\n", name));
    s
}

/// Sanity helper used by tests: file size must equal 84 + 50·triangles.
pub fn stl_len_for_tris(n: usize) -> usize { 84 + n * 50 }

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::prims::cube;

    #[test]
    fn stl_structure() {
        let (c, _) = cube(10.0, 10.0, 10.0);
        let bytes = to_binary_stl(&c, "cube");
        assert_eq!(bytes.len(), stl_len_for_tris(c.tris.len()));
        assert_eq!(&bytes[80..84], &(c.tris.len() as u32).to_le_bytes());
    }
}
