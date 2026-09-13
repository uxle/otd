//! The backend axis — where computation happens. In real Burn a backend is a
//! swappable trait (ndarray CPU, CUDA, ROCm, WASM, tch); here there is one
//! honest implementation, the CPU, plus the autodiff *wrapper* backend that
//! marks a tensor as gradient-tracked — the same wrapper trick burn-autodiff
//! uses to turn any backend into a differentiating one.

/// A compute backend. `name()` is what `--train` prints next to the arch.
pub trait Backend {
    /// Stable backend name, e.g. "ndarray-cpu".
    fn name(&self) -> String;
}

/// The reference CPU backend (Burn's burn-ndarray analogue).
#[derive(Clone, Copy, Debug, Default)]
pub struct NdArray;

impl Backend for NdArray {
    fn name(&self) -> String {
        "ndarray-cpu".into()
    }
}

/// The autodiff wrapper "backend" — wraps any backend with gradient tracking
/// (Burn's burn-autodiff analogue). Backends are cheap handles, never
/// resources: `Clone` is free.
pub struct Autodiff<B: Backend> {
    inner: B,
}

impl<B: Backend> Autodiff<B> {
    /// Wrap a backend with gradient tracking.
    pub fn new(inner: B) -> Autodiff<B> {
        Autodiff { inner }
    }
}

impl<B: Backend + Default> Default for Autodiff<B> {
    fn default() -> Self {
        Self::new(B::default())
    }
}

impl<B: Backend + Clone> Clone for Autodiff<B> {
    fn clone(&self) -> Self {
        Self::new(self.inner.clone())
    }
}

impl<B: Backend> Backend for Autodiff<B> {
    fn name(&self) -> String {
        format!("autodiff<{}>", self.inner.name())
    }
}
