//! Verifier crate (Rust port of python/verifier/): the ground-truth signal
//! the whole system depends on.
//!
//! - `symbolic_verifier`: sympy-style parse + verify over the mini-CAS in
//!   `reasoning-symbolic`; parsing never evaluates model output as code.
//! - `units`: dimensional-analysis consistency checks (Phase 043).
//! - `disagreement`: cross-checks independent verification methods and
//!   flags when they disagree (Phase 033).
//! - `code_sandbox`: AST-whitelisted interpreter for executing generated
//!   code safely (Phase 031).
//!
//! No neural component here at all, by design: search and self-evolution
//! only work if verification is independent of the policy model.

pub mod code_sandbox;
pub mod disagreement;
pub mod symbolic_verifier;
pub mod units;

pub use code_sandbox::{run_sandboxed, SandboxError, SandboxValue};
pub use disagreement::{cross_check_equality_claim, DisagreementReport};
pub use symbolic_verifier::{
    safe_parse, safe_parse_with_locals, solve_equation, verify_algebraic_equivalence,
    verify_equation_solution, verify_numeric_equality, verify_numeric_equality_with_tol,
    VerificationError, VerificationResult,
};
pub use units::{
    add, check_expression_units, divide, multiply, quantity, Dimension, UnitQuantity,
};
