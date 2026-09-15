//! P0640c — GLB exporter (glTF 2.0 binary). One buffer, interleaved meshes per
//! part, pbrMetallicRoughness from OTD materials. Written from scratch: JSON
//! chunk + BIN chunk, little-endian, own alignment padding.

use crate::geo::mesh::Mesh;

#[derive(Clone, Copy)]
pub struct GlbMaterial {
    /// linear base color RGBA (0..1)
    pub base_color: [f32; 4],
    pub metallic: f32,
    pub roughness: f32,
}

pub struct GlbPart<'a> {
    pub name: &'a str,
    pub mesh: &'a Mesh,
    pub material: usize,
}

struct Buffer {
    data: Vec<u8>,
}

impl Buffer {
    fn push(&mut self, bytes: &[u8]) -> u32 {
        let off = self.data.len() as u32;
        self.data.extend_from_slice(bytes);
        off
    }
    fn pad4(&mut self) {
        while self.data.len() % 4 != 0 {
            self.data.push(0);
        }
    }
}

fn f32s(v: &[f64]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&(*x as f32).to_le_bytes());
    }
    b
}

/// Build a GLB from parts. Returns the .glb file bytes.
pub fn to_glb(title: &str, parts: &[GlbPart], materials: &[GlbMaterial]) -> Vec<u8> {
    let mut buf = Buffer { data: Vec::new() };
    let mut accessors: Vec<String> = Vec::new();
    let mut mesh_defs: Vec<String> = Vec::new();
    let mut node_defs: Vec<String> = Vec::new();

    for (pi, p) in parts.iter().enumerate() {
        // positions accessor
        let mut pos: Vec<f64> = Vec::with_capacity(p.mesh.verts.len() * 3);
        let bb = p.mesh.bbox();
        for v in &p.mesh.verts {
            pos.push(v.x());
            pos.push(v.y());
            pos.push(v.z());
        }
        // normals: area-weighted vertex normals
        let normals = p.mesh.vertex_normals();
        let mut nor: Vec<f64> = Vec::with_capacity(normals.len() * 3);
        for n in &normals {
            nor.push(n.x());
            nor.push(n.y());
            nor.push(n.z());
        }
        // indices
        let _idx: Vec<u8> = Vec::with_capacity(p.mesh.tris.len() * 6);
        // always u32 indices
        let mut idx32: Vec<u8> = Vec::with_capacity(p.mesh.tris.len() * 12);
        for t in &p.mesh.tris {
            for i in 0..3 {
                idx32.extend_from_slice(&t[i].to_le_bytes());
            }
        }

        buf.pad4();
        let pos_off = buf.push(&f32s(&pos));
        buf.pad4();
        let pos_min = [bb.min.x() as f32, bb.min.y() as f32, bb.min.z() as f32];
        let pos_max = [bb.max.x() as f32, bb.max.y() as f32, bb.max.z() as f32];
        let pos_count = p.mesh.verts.len();
        let pos_acc = accessors.len();
        accessors.push(format!(
            "{{\"bufferView\":0,\"componentType\":5126,\"count\":{},\"type\":\"VEC3\",\"byteOffset\":{},\"min\":{:?},\"max\":{:?}}}",
            pos_count, pos_off, arr_json(&pos_min), arr_json(&pos_max)
        ));

        buf.pad4();
        let nor_off = buf.push(&f32s(&nor));
        let nor_count = normals.len();
        let nor_acc = accessors.len();
        accessors.push(format!(
            "{{\"bufferView\":0,\"componentType\":5126,\"count\":{},\"type\":\"VEC3\",\"byteOffset\":{}}}",
            nor_count, nor_off
        ));

        buf.pad4();
        let idx_off = buf.push(&idx32);
        let idx_count = p.mesh.tris.len() * 3;
        let idx_acc = accessors.len();
        accessors.push(format!(
            "{{\"bufferView\":0,\"componentType\":5125,\"count\":{},\"type\":\"SCALAR\",\"byteOffset\":{}}}",
            idx_count, idx_off
        ));

        mesh_defs.push(format!(
            "{{\"name\":\"{}\",\"primitives\":[{{\"attributes\":{{\"POSITION\":{},\"NORMAL\":{}}},\"indices\":{},\"material\":{}}}]}}",
            p.name, pos_acc, nor_acc, idx_acc, p.material
        ));
        node_defs.push(format!("{{\"mesh\":{},\"name\":\"{}\"}}", pi, p.name));
    }
    buf.pad4();

    let mat_defs: Vec<String> = materials
        .iter()
        .map(|m| {
            format!(
                "{{\"pbrMetallicRoughness\":{{\"baseColorFactor\":{},\"metallicFactor\":{},\"roughnessFactor\":{}}}}}",
                arr4_json(&m.base_color), m.metallic, m.roughness
            )
        })
        .collect();

    let scene = if node_defs.is_empty() {
        "{\"scenes\":[{\"nodes\":[]}]}}".to_string()
    } else {
        format!("{{\"scenes\":[{{\"nodes\":[{}]}}]}}", (0..node_defs.len()).map(|i| i.to_string()).collect::<Vec<_>>().join(","))
    };

    // glTF JSON (single shared bufferView 0 → whole BIN chunk)
    let mut json = String::new();
    json.push_str("{");
    json.push_str("\"asset\":{\"version\":\"2.0\",\"generator\":\"OTD 1.0 (pure Rust)\"},");
    json.push_str(&format!("\"scene\":0,"));
    json.push_str(&format!("\"scenes\":[{{\"nodes\":[{}]}}],", (0..node_defs.len()).map(|i| i.to_string()).collect::<Vec<_>>().join(",")));
    let _ = scene;
    json.push_str(&format!("\"nodes\":[{}],", node_defs.join(",")));
    json.push_str(&format!("\"meshes\":[{}],", mesh_defs.join(",")));
    json.push_str(&format!("\"materials\":[{}],", mat_defs.join(",")));
    json.push_str(&format!("\"accessors\":[{}],", accessors.join(",")));
    json.push_str(&format!(
        "\"bufferViews\":[{{\"buffer\":0,\"byteOffset\":0,\"byteLength\":{}}}],",
        buf.data.len()
    ));
    json.push_str(&format!("\"buffers\":[{{\"byteLength\":{}}}]}}", buf.data.len()));

    // pad JSON to 4
    let mut json_bytes = json.into_bytes();
    while json_bytes.len() % 4 != 0 {
        json_bytes.push(b' ');
    }

    // GLB container
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(b"glTF"); // magic
    out.extend_from_slice(&2u32.to_le_bytes()); // version
    let total = 12 + 8 + json_bytes.len() + 8 + buf.data.len();
    out.extend_from_slice(&(total as u32).to_le_bytes());
    // JSON chunk
    out.extend_from_slice(&(json_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(b"JSON");
    out.extend_from_slice(&json_bytes);
    // BIN chunk
    out.extend_from_slice(&(buf.data.len() as u32).to_le_bytes());
    out.extend_from_slice(b"BIN\0");
    out.extend_from_slice(&buf.data);
    let _ = title;
    out
}

fn arr_json(a: &[f32]) -> String {
    "[".to_string() + &a.iter().map(|v| format!("{:.5}", v)).collect::<Vec<_>>().join(",") + "]"
}
fn arr4_json(a: &[f32; 4]) -> String {
    "[".to_string() + &a.iter().map(|v| format!("{:.5}", v)).collect::<Vec<_>>().join(",") + "]"
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::prims::cube;

    #[test]
    fn glb_structure() {
        let (c, _) = cube(10.0, 10.0, 10.0);
        let parts = [GlbPart { name: "cube", mesh: &c, material: 0 }];
        let mats = [GlbMaterial { base_color: [0.8, 0.2, 0.2, 1.0], metallic: 0.0, roughness: 0.5 }];
        let bytes = to_glb("test", &parts, &mats);
        // magic + version
        assert_eq!(&bytes[0..4], b"glTF");
        assert_eq!(u32::from_le_bytes([bytes[4], bytes[5], bytes[6], bytes[7]]), 2);
        // total length field matches
        let total = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]) as usize;
        assert_eq!(total, bytes.len());
        // JSON chunk
        assert_eq!(&bytes[16..20], b"JSON");
    }
}
