//! Shared helpers for the Rust port of reasoning-ai: Python-compatible
//! float semantics, exact rational arithmetic, and rational matrices
//! (nullspace / determinant / solve / multiply) used by both the symbolic
//! CAS and the chemistry balancing domain.

mod pyfloat;
mod rat;
mod ratmat;

pub use pyfloat::{py_float_str, py_round};
pub use rat::Rat;
pub use ratmat::{rat_matrix_from_rows, RatMatrix};
