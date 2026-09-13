//! Binary PLY writer/reader for point clouds and meshes.

use std::io::{Read, Write};

/// A mesh for export (vertices + triangle indices).
#[derive(Debug, Clone, Default)]
pub struct Mesh {
    pub verts: Vec<[f32; 3]>,
    pub faces: Vec<[u32; 3]>,
}

/// Write a point cloud as binary little-endian PLY (x, y, z, intensity).
pub fn write_point_cloud_ply(path: &std::path::Path, points: &[[f32; 3]], intensity: Option<&[u8]>) -> std::io::Result<()> {
    let has_i = intensity.is_some();
    let mut header = String::new();
    header.push_str("ply\n");
    header.push_str("format binary_little_endian 1.0\n");
    header.push_str(&format!("comment AVC v{} world point cloud\n", crate::VERSION));
    header.push_str(&format!("element vertex {}\n", points.len()));
    header.push_str("property float x\nproperty float y\nproperty float z\n");
    if has_i {
        header.push_str("property uchar intensity\n");
    }
    header.push_str("end_header\n");
    let mut file = std::fs::File::create(path)?;
    file.write_all(header.as_bytes())?;
    for (i, p) in points.iter().enumerate() {
        file.write_all(&p[0].to_le_bytes())?;
        file.write_all(&p[1].to_le_bytes())?;
        file.write_all(&p[2].to_le_bytes())?;
        if let Some(ints) = intensity {
            file.write_all(&[ints[i]])?;
        }
    }
    Ok(())
}

/// Write a mesh as binary PLY.
pub fn write_mesh_ply(path: &std::path::Path, mesh: &Mesh) -> std::io::Result<()> {
    let mut header = String::new();
    header.push_str("ply\n");
    header.push_str("format binary_little_endian 1.0\n");
    header.push_str(&format!("comment AVC v{} reconstructed surface\n", crate::VERSION));
    header.push_str(&format!("element vertex {}\n", mesh.verts.len()));
    header.push_str("property float x\nproperty float y\nproperty float z\n");
    header.push_str(&format!("element face {}\n", mesh.faces.len()));
    header.push_str("property list uchar int vertex_indices\n");
    header.push_str("end_header\n");
    let mut file = std::fs::File::create(path)?;
    file.write_all(header.as_bytes())?;
    for v in &mesh.verts {
        file.write_all(&v[0].to_le_bytes())?;
        file.write_all(&v[1].to_le_bytes())?;
        file.write_all(&v[2].to_le_bytes())?;
    }
    for f in &mesh.faces {
        file.write_all(&[3u8])?;
        for idx in f {
            file.write_all(&idx.to_le_bytes())?;
        }
    }
    Ok(())
}

/// Minimal PLY reader for verification (vertex + face counts, first verts).
pub fn read_ply_info(path: &std::path::Path) -> std::io::Result<(usize, usize, Vec<[f32; 3]>)> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    let header_end = bytes
        .windows(11)
        .position(|w| w == b"end_header\n")
        .ok_or_else(|| std::io::Error::new(std::io::ErrorKind::InvalidData, "no end_header"))?;
    let header = String::from_utf8_lossy(&bytes[..header_end + 11]).to_string();
    let mut n_vert = 0;
    let mut n_face = 0;
    let mut has_intensity = false;
    let mut binary = false;
    for line in header.lines() {
        let parts: Vec<&str> = line.split_whitespace().collect();
        if parts.len() >= 3 && parts[0] == "element" && parts[1] == "vertex" {
            n_vert = parts[2].parse().unwrap_or(0);
        }
        if parts.len() >= 3 && parts[0] == "element" && parts[1] == "face" {
            n_face = parts[2].parse().unwrap_or(0);
        }
        if line.contains("intensity") {
            has_intensity = true;
        }
        if parts.len() >= 3 && parts[0] == "format" && parts[1].starts_with("binary") {
            binary = true;
        }
    }
    if !binary {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "only binary PLY supported by reader"));
    }
    // read first 3 vertices
    let stride = if has_intensity { 13 } else { 12 };
    let data = &bytes[header_end + 11..];
    let mut verts = Vec::new();
    for k in 0..3.min(n_vert) {
        let off = k * stride;
        if off + 12 > data.len() {
            break;
        }
        let x = f32::from_le_bytes([data[off], data[off + 1], data[off + 2], data[off + 3]]);
        let y = f32::from_le_bytes([data[off + 4], data[off + 5], data[off + 6], data[off + 7]]);
        let z = f32::from_le_bytes([data[off + 8], data[off + 9], data[off + 10], data[off + 11]]);
        verts.push([x, y, z]);
    }
    Ok((n_vert, n_face, verts))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ply_point_cloud_roundtrip() {
        let dir = std::env::temp_dir().join("avc_ply_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("cloud.ply");
        let pts = vec![[0.1f32, 0.2, 3.0], [1.0, 0.5, 4.0], [-0.5, 0.1, 2.0]];
        let intensity = vec![100u8, 150, 200];
        write_point_cloud_ply(&path, &pts, Some(&intensity)).unwrap();
        let (n, f, v) = read_ply_info(&path).unwrap();
        assert_eq!(n, 3);
        assert_eq!(f, 0);
        assert_eq!(v.len(), 3);
        assert_eq!(v[0], pts[0]);
        assert_eq!(v[2], pts[2]);
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn ply_mesh_roundtrip() {
        let dir = std::env::temp_dir().join("avc_ply_mesh_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mesh.ply");
        let mesh = Mesh {
            verts: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2]],
        };
        write_mesh_ply(&path, &mesh).unwrap();
        let (n, f, v) = read_ply_info(&path).unwrap();
        assert_eq!(n, 3);
        assert_eq!(f, 1);
        assert_eq!(v[1], mesh.verts[1]);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
