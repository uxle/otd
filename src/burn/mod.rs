//! P2210 — OTD-Burn: a self-made deep learning framework in the image of
//! [Burn](https://burn.dev) (Tracel, Apache-2.0), rebuilt under this project's
//! P0000 zero-dependency constraint. The real Burn could not be vendored —
//! it drags ~50 dependency crates (matrixmultiply, num-traits, derive-new,
//! rand, …) behind it, which would break the guarantee that `Cargo.toml`
//! ships with an empty `[dependencies]`. So Burn's *architecture* was ported
//! instead, and every layer is written from nothing:
//!
//! | real Burn crate   | OTD-Burn module        | what it does |
//! |-------------------|------------------------|--------------|
//! | burn-core         | `burn::module`         | the `Module` trait, params, checkpoints |
//! | burn-tensor       | `burn::tensor`         | shape-checked `Tensor<B>` ops |
//! | burn-ndarray      | `burn::backend::NdArray` | the CPU reference backend |
//! | burn-autodiff     | `burn::autodiff`       | reverse-mode tape (`Var`), gradcheck'd |
//! | burn-train        | `burn::train`          | `Learner` loop, metrics, early stop |
//! | burn-optim        | `burn::optim`          | Sgd, Adam, AdamW, LR schedules |
//! | burn-data         | `burn::data`           | dataset + shuffled batching |
//!
//! The signature Burn move — *autodiff as a backend wrapper* — is preserved in
//! spirit: a plain `Tensor<NdArray>` becomes a differentiable value by living
//! inside a `Var` (≈ `Tensor<Autodiff<NdArray>>`), and the same `Module`
//! code runs under either. `Learner::fit` then trains modules on data —
//! including OTD's own physics laws (see `train::learn_freefall_burn`).

pub mod backend;
pub mod tensor;
pub mod autodiff;
pub mod module;
pub mod nn;
pub mod optim;
pub mod data;
pub mod train;

pub use backend::{Autodiff, Backend, NdArray};
pub use tensor::Tensor;
pub use autodiff::Var;
pub use module::Module;
pub use nn::{Dropout, Embedding, Linear, Sequential};
pub use optim::{Adam, LrScheduler, Optimizer as OptimTrait, Sgd};
pub use data::Supervised;
pub use train::{fit, learn_freefall_burn, learn_xor_burn, TrainReport};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn burn_backend_names() {
        assert_eq!(NdArray.name(), "ndarray-cpu");
        assert_eq!(Autodiff::new(NdArray).name(), "autodiff<ndarray-cpu>");
    }

    #[test]
    fn backend_handles_are_cheap() {
        let a = NdArray;
        let b = a;
        assert_eq!(a.name(), b.name());
        let ad = Autodiff::new(NdArray);
        let ad2 = ad.clone();
        assert_eq!(ad.name(), ad2.name());
    }
}
