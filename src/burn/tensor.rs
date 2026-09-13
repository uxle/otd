//! Shape-checked tensors over a backend — the burn-tensor analogue. The data
//! kernels (matmul, softmax, …) come from `crate::nn` — one kernel library,
//! two frameworks on top, exactly how a healthy ecosystem shares wheels.

use core::marker::PhantomData;
use super::backend::Backend;

/// A dense f32 tensor bound to a backend `B`.
///
/// In real Burn this type is `Tensor<B, const D: usize, K>` with compile-time
/// dimensionality; OTD keeps runtime shapes (the world module already speaks
/// runtime shapes everywhere) and checks them on every op.
#[derive(Clone, Debug)]
pub struct Tensor<B: Backend> {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
    _backend: PhantomData<B>,
}

impl<B: Backend> Tensor<B> {
    pub fn zeros(shape: &[usize]) -> Tensor<B> {
        let n: usize = shape.iter().product();
        Tensor { shape: shape.to_vec(), data: vec![0.0; n], _backend: PhantomData }
    }

    pub fn full(shape: &[usize], v: f32) -> Tensor<B> {
        let n: usize = shape.iter().product();
        Tensor { shape: shape.to_vec(), data: vec![v; n], _backend: PhantomData }
    }

    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Tensor<B> {
        debug_assert_eq!(data.len(), shape.iter().product::<usize>(), "shape/data mismatch");
        Tensor { shape: shape.to_vec(), data, _backend: PhantomData }
    }

    /// From the shared kernel library's tensor (same layout, any backend).
    pub fn from_plain(t: crate::nn::Tensor) -> Tensor<B> {
        Tensor { shape: t.shape, data: t.data, _backend: PhantomData }
    }

    pub fn to_plain(&self) -> crate::nn::Tensor {
        crate::nn::Tensor { shape: self.shape.clone(), data: self.data.clone() }
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.data.is_empty()
    }

    /// (rows, cols) for a 2-D tensor.
    pub fn dims2(&self) -> (usize, usize) {
        assert_eq!(self.shape.len(), 2, "expected 2-D, got {:?}", self.shape);
        (self.shape[0], self.shape[1])
    }

    fn same_shape(&self, o: &Tensor<B>, what: &str) {
        assert_eq!(self.shape, o.shape, "{}: shapes {:?} vs {:?}", what, self.shape, o.shape);
    }

    pub fn add(&self, o: &Tensor<B>) -> Tensor<B> {
        self.same_shape(o, "add");
        Tensor::from_vec(&self.shape, self.data.iter().zip(&o.data).map(|(a, b)| a + b).collect())
    }

    pub fn sub(&self, o: &Tensor<B>) -> Tensor<B> {
        self.same_shape(o, "sub");
        Tensor::from_vec(&self.shape, self.data.iter().zip(&o.data).map(|(a, b)| a - b).collect())
    }

    pub fn mul(&self, o: &Tensor<B>) -> Tensor<B> {
        self.same_shape(o, "mul");
        Tensor::from_vec(&self.shape, self.data.iter().zip(&o.data).map(|(a, b)| a * b).collect())
    }

    pub fn scale(&self, s: f32) -> Tensor<B> {
        Tensor::from_vec(&self.shape, self.data.iter().map(|x| x * s).collect())
    }

    pub fn map(&self, f: impl Fn(f32) -> f32) -> Tensor<B> {
        Tensor::from_vec(&self.shape, self.data.iter().map(|&x| f(x)).collect())
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn mean(&self) -> f32 {
        if self.data.is_empty() { 0.0 } else { self.sum() / self.data.len() as f32 }
    }

    pub fn t(&self) -> Tensor<B> {
        let (r, c) = self.dims2();
        let mut out = Tensor::zeros(&[c, r]);
        for i in 0..r {
            for j in 0..c {
                out.data[j * r + i] = self.data[i * c + j];
            }
        }
        out
    }

    /// Matmul through the shared blocked kernel.
    pub fn matmul(&self, o: &Tensor<B>) -> Tensor<B> {
        let (m, k) = self.dims2();
        let (k2, n) = o.dims2();
        assert_eq!(k, k2, "matmul: inner dims {} vs {}", k, k2);
        let out = self.to_plain().matmul(&o.to_plain());
        let _ = (m, n);
        Tensor::from_vec(&out.shape, out.data)
    }

    pub fn argmax_rows(&self) -> Vec<usize> {
        let (n, c) = self.dims2();
        (0..n)
            .map(|i| {
                let row = &self.data[i * c..(i + 1) * c];
                let mut best = 0usize;
                for j in 1..c {
                    if row[j] > row[best] {
                        best = j;
                    }
                }
                best
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::burn::backend::NdArray;

    #[test]
    fn tensor_ops_shape_checked() {
        let a = Tensor::<NdArray>::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
        let b = Tensor::<NdArray>::from_vec(&[2, 3], vec![6., 5., 4., 3., 2., 1.]);
        assert_eq!(a.add(&b).data, vec![7.0; 6]);
        assert_eq!(a.mul(&b).sum(), 1. * 6. + 2. * 5. + 3. * 4. + 4. * 3. + 5. * 2. + 6. * 1.);
    }

    #[test]
    fn tensor_matmul_through_shared_kernel() {
        let a = Tensor::<NdArray>::from_vec(&[2, 2], vec![1., 0., 0., 1.]);
        let b = Tensor::<NdArray>::from_vec(&[2, 2], vec![1., 2., 3., 4.]);
        assert_eq!(a.matmul(&b).data, b.data); // identity
    }

    #[test]
    fn argmax_rows_picks_the_winner() {
        let t = Tensor::<NdArray>::from_vec(&[3, 3], vec![
            0.1, 0.9, 0.0,
            0.5, 0.2, 0.3,
            0.0, 0.0, 1.0,
        ]);
        assert_eq!(t.argmax_rows(), vec![1, 0, 2]);
    }
}
