//! `.avcworld` binary world-state format (versioned, round-trip tested).

use std::io::{Read, Write};

use nalgebra::{Matrix3, Vector3};

use crate::world::model::WorldObject;

const MAGIC: &[u8; 5] = b"AVCW3";
const VERSION: u16 = 3;

#[derive(Debug, Clone, Default)]
pub struct WorldFile {
    pub frame: u32,
    pub t: f64,
    pub objects: Vec<WorldObject>,
}

impl WorldFile {
    pub fn from_world(frame: u32, t: f64, objects: &[WorldObject]) -> Self {
        WorldFile { frame, t, objects: objects.to_vec() }
    }
}

fn write_u64(w: &mut impl Write, v: u64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_u32(w: &mut impl Write, v: u32) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_u16(w: &mut impl Write, v: u16) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}
fn write_f64(w: &mut impl Write, v: f64) -> std::io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

pub fn write_avcworld(path: &std::path::Path, wf: &WorldFile) -> std::io::Result<()> {
    let mut w = std::fs::File::create(path)?;
    w.write_all(MAGIC)?;
    write_u16(&mut w, VERSION)?;
    write_u32(&mut w, wf.frame)?;
    write_f64(&mut w, wf.t)?;
    write_u32(&mut w, wf.objects.len() as u32)?;
    for o in &wf.objects {
        write_u64(&mut w, o.track_id)?;
        let class = o.class.as_bytes();
        write_u16(&mut w, class.len() as u16)?;
        w.write_all(class)?;
        for v in o.center.iter() {
            write_f64(&mut w, *v)?;
        }
        for v in o.velocity.iter() {
            write_f64(&mut w, *v)?;
        }
        // covariance upper triangle
        for i in 0..3 {
            for j in i..3 {
                write_f64(&mut w, o.cov[(i, j)])?;
            }
        }
        for v in o.extents.iter() {
            write_f64(&mut w, *v)?;
        }
        write_f64(&mut w, o.yaw)?;
        write_f64(&mut w, o.confidence)?;
        write_u32(&mut w, o.first_frame as u32)?;
        write_u32(&mut w, o.last_frame as u32)?;
    }
    Ok(())
}

pub fn read_avcworld(path: &std::path::Path) -> std::io::Result<WorldFile> {
    let mut bytes = Vec::new();
    std::fs::File::open(path)?.read_to_end(&mut bytes)?;
    let mut r = &bytes[..];
    let mut magic = [0u8; 5];
    read_exact(&mut r, &mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad magic"));
    }
    let version = read_u16(&mut r)?;
    if version != VERSION {
        return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("version {version} != {VERSION}")));
    }
    let frame = read_u32(&mut r)?;
    let t = read_f64(&mut r)?;
    let n = read_u32(&mut r)? as usize;
    let mut objects = Vec::with_capacity(n);
    for _ in 0..n {
        let track_id = read_u64(&mut r)?;
        let clen = read_u16(&mut r)? as usize;
        if clen > 32 {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "class too long"));
        }
        let mut cbuf = [0u8; 32];
        read_exact(&mut r, &mut cbuf[..clen])?;
        let class: &'static str = if &cbuf[..clen] == b"box" {
            "box"
        } else if &cbuf[..clen] == b"person" {
            "person"
        } else if &cbuf[..clen] == b"cylinder" {
            "cylinder"
        } else if &cbuf[..clen] == b"small_box" {
            "small_box"
        } else {
            "object"
        };
        let mut center = [0f64; 3];
        for v in center.iter_mut() {
            *v = read_f64(&mut r)?;
        }
        let mut velocity = [0f64; 3];
        for v in velocity.iter_mut() {
            *v = read_f64(&mut r)?;
        }
        let mut cov = Matrix3::zeros();
        let mut k = 0;
        let mut tri = [0f64; 6];
        for v in tri.iter_mut() {
            *v = read_f64(&mut r)?;
        }
        cov[(0, 0)] = tri[0];
        cov[(0, 1)] = tri[1];
        cov[(0, 2)] = tri[2];
        cov[(1, 1)] = tri[3];
        cov[(1, 2)] = tri[4];
        cov[(2, 2)] = tri[5];
        cov[(1, 0)] = tri[1];
        cov[(2, 0)] = tri[2];
        cov[(2, 1)] = tri[4];
        k += 1;
        let _ = k;
        let mut extents = [0f64; 3];
        for v in extents.iter_mut() {
            *v = read_f64(&mut r)?;
        }
        let yaw = read_f64(&mut r)?;
        let confidence = read_f64(&mut r)?;
        let first_frame = read_u32(&mut r)? as usize;
        let last_frame = read_u32(&mut r)? as usize;
        objects.push(WorldObject {
            track_id,
            class,
            center: Vector3::new(center[0], center[1], center[2]),
            velocity: Vector3::new(velocity[0], velocity[1], velocity[2]),
            cov,
            extents: Vector3::new(extents[0], extents[1], extents[2]),
            yaw,
            confidence,
            first_frame,
            last_frame,
            n_hits: 0,
        });
    }
    Ok(WorldFile { frame, t, objects })
}

fn read_exact<'a>(r: &mut &'a [u8], buf: &mut [u8]) -> std::io::Result<()> {
    if r.len() < buf.len() {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "eof"));
    }
    let (head, tail) = r.split_at(buf.len());
    buf.copy_from_slice(head);
    *r = tail;
    Ok(())
}
fn read_u16(r: &mut &[u8]) -> std::io::Result<u16> {
    let mut b = [0u8; 2];
    read_exact(r, &mut b)?;
    Ok(u16::from_le_bytes(b))
}
fn read_u32(r: &mut &[u8]) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    read_exact(r, &mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut &[u8]) -> std::io::Result<u64> {
    let mut b = [0u8; 8];
    read_exact(r, &mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f64(r: &mut &[u8]) -> std::io::Result<f64> {
    let mut b = [0u8; 8];
    read_exact(r, &mut b)?;
    Ok(f64::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn avcworld_roundtrip() {
        let dir = std::env::temp_dir().join("avc_worldfile_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("world.avcworld");
        let o = WorldObject {
            track_id: 42,
            class: "person",
            center: Vector3::new(0.3, 0.86, 4.2),
            velocity: Vector3::new(0.25, 0.0, 0.0),
            cov: Matrix3::identity() * 0.01,
            extents: Vector3::new(0.32, 1.72, 0.32),
            yaw: 0.1,
            confidence: 0.87,
            first_frame: 3,
            last_frame: 59,
            n_hits: 55,
        };
        let wf = WorldFile::from_world(59, 1.97, &[o]);
        write_avcworld(&path, &wf).unwrap();
        let back = read_avcworld(&path).unwrap();
        assert_eq!(back.frame, 59);
        assert!((back.t - 1.97).abs() < 1e-12);
        assert_eq!(back.objects.len(), 1);
        let b = &back.objects[0];
        assert_eq!(b.track_id, 42);
        assert_eq!(b.class, "person");
        assert!((b.center - wf.objects[0].center).norm() < 1e-12);
        assert!((b.velocity - wf.objects[0].velocity).norm() < 1e-12);
        assert!((b.extents - wf.objects[0].extents).norm() < 1e-12);
        assert!((b.cov - wf.objects[0].cov).abs().max() < 1e-12);
        assert_eq!(b.first_frame, 3);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
