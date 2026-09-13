//! Reasoning AI — tensor crate (Rust port of the Python `tensor` package):
//! the from-scratch N-D tensor abstraction of Phase 001.

pub mod tensor;

pub use tensor::{NestedVal, Number, ShapeError, Tensor, TensorError, TensorIndex};
