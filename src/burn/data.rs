//! Datasets and batching — the burn-data analogue. A `Supervised` dataset is
//! a pair of aligned tensors; `train_split` carves out a deterministic
//! train/test split; the batcher shuffles with the seeded PRNG so runs are
//! reproducible.

use crate::rng::Rng;

use super::backend::NdArray;
use super::tensor::Tensor;

/// Aligned X/Y tensors: X (n, din), Y (n, dout) or Y (n,) for classifiers.
pub struct Supervised {
    pub x: Tensor<NdArray>,
    pub y: Tensor<NdArray>,
}

impl Supervised {
    pub fn new(x: Tensor<NdArray>, y: Tensor<NdArray>) -> Supervised {
        assert_eq!(x.shape[0], y.shape[0], "x/y row counts must match");
        Supervised { x, y }
    }

    pub fn len(&self) -> usize {
        self.x.shape[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Deterministic (train, test) split by row shuffle with `seed`.
    pub fn train_split(&self, frac: f64, seed: u64) -> (Supervised, Supervised) {
        let n = self.len();
        let mut rng = Rng::new(seed);
        let mut idx: Vec<usize> = (0..n).collect();
        for i in (1..idx.len()).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            idx.swap(i, j);
        }
        let cut = ((n as f64) * frac.clamp(0.1, 0.9)) as usize;
        let din = self.x.shape[1];
        let dout = *self.y.shape.last().unwrap_or(&1);
        let ycols = dout;
        let build = |ids: &[usize]| -> Supervised {
            let mut x = Tensor::<NdArray>::zeros(&[ids.len(), din]);
            let mut y = Tensor::<NdArray>::zeros(&[ids.len(), ycols]);
            for (r, &i) in ids.iter().enumerate() {
                for j in 0..din {
                    x.data[r * din + j] = self.x.data[i * din + j];
                }
                for j in 0..ycols {
                    y.data[r * ycols + j] = self.y.data[i * ycols + j];
                }
            }
            Supervised::new(x, y)
        };
        (build(&idx[..cut]), build(&idx[cut..]))
    }

    /// One shuffled epoch of (start, end) index ranges into `batches`.
    pub fn epoch_order(&self, seed: u64) -> Vec<usize> {
        let mut rng = Rng::new(seed);
        let mut idx: Vec<usize> = (0..self.len()).collect();
        for i in (1..idx.len()).rev() {
            let j = (rng.next_f64() * (i + 1) as f64) as usize;
            idx.swap(i, j);
        }
        idx
    }

    /// Gather rows `[s, e)` as fresh tensors.
    pub fn rows(&self, s: usize, e: usize, order: &[usize]) -> (Tensor<NdArray>, Tensor<NdArray>) {
        let ids = &order[s..e.min(order.len())];
        let din = self.x.shape[1];
        let ycols = *self.y.shape.last().unwrap_or(&1);
        let mut x = Tensor::<NdArray>::zeros(&[ids.len(), din]);
        let mut y = Tensor::<NdArray>::zeros(&[ids.len(), ycols]);
        for (r, &i) in ids.iter().enumerate() {
            for j in 0..din {
                x.data[r * din + j] = self.x.data[i * din + j];
            }
            for j in 0..ycols {
                y.data[r * ycols + j] = self.y.data[i * ycols + j];
            }
        }
        (x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn split_is_deterministic_and_disjoint() {
        let x = Tensor::<NdArray>::from_vec(&[10, 2], (0..20).map(|i| i as f32).collect());
        let y = Tensor::<NdArray>::from_vec(&[10, 1], (0..10).map(|i| i as f32).collect());
        let ds = Supervised::new(x, y);
        let (t1, e1) = ds.train_split(0.8, 5);
        let (t2, e2) = ds.train_split(0.8, 5);
        assert_eq!(t1.x.data, t2.x.data, "same seed ⇒ same split");
        assert_eq!(t1.len(), 8);
        assert_eq!(e1.len(), 2);
        // disjoint: every original row appears exactly once across both
        let mut seen = vec![false; 10];
        for &v in t1.y.data.iter().chain(e1.y.data.iter()) {
            seen[v as usize] = true;
        }
        assert!(seen.iter().all(|&s| s));
    }

    #[test]
    fn rows_gather_in_order() {
        let x = Tensor::<NdArray>::from_vec(&[4, 1], vec![10.0, 20.0, 30.0, 40.0]);
        let y = Tensor::<NdArray>::from_vec(&[4, 1], vec![1.0, 2.0, 3.0, 4.0]);
        let ds = Supervised::new(x, y);
        // rows(1, 3, order) gathers order[1..3] = [2, 1] → original rows 2 and 1
        let (bx, by) = ds.rows(1, 3, &[3, 2, 1, 0]);
        assert_eq!(bx.data, vec![30.0, 20.0]);
        assert_eq!(by.data, vec![3.0, 2.0]);
    }
}
