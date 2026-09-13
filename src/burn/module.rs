//! The `Module` trait — burn-core's contract. A module owns `Var` parameters
//! (via `Rc` interior mutability, so `forward(&self, …)` can run while the
//! optimizer still updates weights through the same handles), can enumerate
//! them for optimizers, and can save/load itself as a flat checkpoint file.

use std::path::Path;

use super::autodiff::Var;
use super::backend::NdArray;
use super::tensor::Tensor;

/// A neural network module: `forward` maps inputs to outputs, `params`
/// exposes every trainable leaf. Checkpoints round-trip through `save`/`load`.
pub trait Module {
    /// Run the forward pass.
    fn forward(&self, x: &Var) -> Var;

    /// All trainable leaves in a stable order (the optimizer's view).
    fn params(&self) -> Vec<Var>;

    /// Human-readable architecture line, e.g. "linear 3→5 relu".
    fn arch(&self) -> String {
        format!("{} params", self.num_params())
    }

    fn num_params(&self) -> usize {
        self.params().iter().map(|p| p.val().len()).sum()
    }

    /// Reset gradients of every leaf.
    fn zero_grad(&self) {
        for p in self.params() {
            p.zero_grad();
        }
    }

    /// Write every parameter to a checkpoint file (little-endian f32).
    fn save(&self, path: &Path) -> std::io::Result<()> {
        let mut bytes: Vec<u8> = b"OTDBURN1".to_vec();
        let params = self.params();
        push_u32(&mut bytes, params.len() as u32);
        for p in &params {
            let v = p.val();
            push_u32(&mut bytes, v.shape.len() as u32);
            for d in &v.shape {
                push_u32(&mut bytes, *d as u32);
            }
            push_u32(&mut bytes, v.data.len() as u32);
            for f in &v.data {
                bytes.extend_from_slice(&f.to_le_bytes());
            }
        }
        std::fs::write(path, bytes)
    }

    /// Load parameters back in the same order `params()` yields them.
    fn load(&self, path: &Path) -> std::io::Result<()> {
        let bytes = std::fs::read(path)?;
        if &bytes[0..8] != b"OTDBURN1" {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "not an OTD-Burn checkpoint"));
        }
        let mut off = 8usize;
        let n = read_u32(&bytes, &mut off)? as usize;
        let params = self.params();
        if n != params.len() {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!("checkpoint has {} blocks, module has {}", n, params.len()),
            ));
        }
        for p in &params {
            let ndim = read_u32(&bytes, &mut off)? as usize;
            let mut shape = Vec::with_capacity(ndim);
            for _ in 0..ndim {
                shape.push(read_u32(&bytes, &mut off)? as usize);
            }
            let len = read_u32(&bytes, &mut off)? as usize;
            let mut data = Vec::with_capacity(len);
            for _ in 0..len {
                if off + 4 > bytes.len() {
                    return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated checkpoint"));
                }
                let b: [u8; 4] = bytes[off..off + 4].try_into().unwrap();
                data.push(f32::from_le_bytes(b));
                off += 4;
            }
            if shape.iter().product::<usize>() != len {
                return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "shape/length mismatch"));
            }
            p.set_value(Tensor::<NdArray>::from_vec(&shape, data));
        }
        Ok(())
    }
}

fn push_u32(bytes: &mut Vec<u8>, v: u32) {
    bytes.extend_from_slice(&v.to_le_bytes());
}

fn read_u32(bytes: &[u8], off: &mut usize) -> std::io::Result<u32> {
    if *off + 4 > bytes.len() {
        return Err(std::io::Error::new(std::io::ErrorKind::UnexpectedEof, "truncated checkpoint"));
    }
    let b: [u8; 4] = bytes[*off..*off + 4].try_into().unwrap();
    *off += 4;
    Ok(u32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::autodiff::{backward, vmean, vmul};
    use crate::burn::nn::Linear;

    #[test]
    fn checkpoint_roundtrip() {
        let dir = std::env::temp_dir().join("otd_burn_ckpt_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("lin.ckpt");
        let mut rng = crate::rng::Rng::new(77);
        let lin = Linear::new(3, 2, &mut rng);
        let before = lin.params()[0].val().data.clone();
        lin.save(&path).unwrap();
        // perturb, then load back
        lin.params()[0].set_value(Tensor::<NdArray>::from_vec(&[3, 2], vec![9.0; 6]));
        lin.load(&path).unwrap();
        assert_eq!(lin.params()[0].val().data, before);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn module_zero_grad_clears_leaves() {
        let mut rng = crate::rng::Rng::new(5);
        let lin = Linear::new(4, 3, &mut rng);
        let x = Var::constant(Tensor::<NdArray>::from_vec(&[2, 4], vec![1.0; 8]));
        let y = lin.forward(&x);
        let loss = vmean(&vmul(&y, &y));
        backward(&loss);
        assert!(lin.params().iter().any(|p| p.grad().data.iter().any(|&g| g != 0.0)));
        lin.zero_grad();
        assert!(lin.params().iter().all(|p| p.grad().data.iter().all(|&g| g == 0.0)));
    }
}
