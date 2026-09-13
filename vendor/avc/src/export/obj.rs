//! OBJ mesh export (ASCII, widely supported).

use super::ply::Mesh;

pub fn write_mesh_obj(path: &std::path::Path, mesh: &Mesh) -> std::io::Result<()> {
    let mut out = String::new();
    out.push_str(&format!("# AVC v{} reconstructed surface\n", crate::VERSION));
    for v in &mesh.verts {
        out.push_str(&format!("v {:.4} {:.4} {:.4}\n", v[0], v[1], v[2]));
    }
    for f in &mesh.faces {
        out.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
    }
    std::fs::write(path, out)
}

/// Minimal OBJ reader for verification.
pub fn read_obj_info(path: &std::path::Path) -> std::io::Result<(usize, usize)> {
    let s = std::fs::read_to_string(path)?;
    let n_v = s.lines().filter(|l| l.starts_with("v ")).count();
    let n_f = s.lines().filter(|l| l.starts_with("f ")).count();
    Ok((n_v, n_f))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn obj_roundtrip() {
        let dir = std::env::temp_dir().join("avc_obj_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("mesh.obj");
        let mesh = Mesh {
            verts: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [1.0, 1.0, 0.0]],
            faces: vec![[0, 1, 2], [1, 3, 2]],
        };
        write_mesh_obj(&path, &mesh).unwrap();
        let (v, f) = read_obj_info(&path).unwrap();
        assert_eq!((v, f), (4, 2));
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
