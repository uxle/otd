//! Reasoning AI — neural crate (Rust port of the Python `neural` package):
//! the Phase 017 neural process reward model (a small Transformer encoder
//! over tokenized state text) and its Phase 018 training loop.
//!
//! The Python original was written against PyTorch ("cannot be run or
//! verified in this sandbox"); this port replaces torch with a from-scratch
//! forward/backward over plain `Vec<f64>` parameter blocks — same layer
//! shapes (token embedding -> learned positional embedding -> 2-layer
//! post-LN Transformer encoder, 4 heads, d_model 64, FFN 128 -> masked
//! mean-pool -> linear head -> sigmoid) and manual gradients, so the whole
//! pipeline finally runs and can be verified.

pub mod neural_prm;
pub mod train_prm;

pub use neural_prm::{state_to_text_equation, state_to_text_number_target, NeuralPrm};
pub use train_prm::{build_tensor_dataset, train_neural_prm, Adam, TensorDataset, TrainHistory};
