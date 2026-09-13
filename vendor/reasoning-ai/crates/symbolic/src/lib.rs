//! Minimal computer-algebra engine for the Reasoning AI Rust port — the
//! standing replacement for the Python project's `sympy` usage (which has
//! no mature Rust equivalent). Covers exactly the surface this project
//! uses: expression parsing with implicit multiplication and `^`/`**`
//! powers, exact rational arithmetic, polynomial expansion and
//! canonicalization, algebraic-equivalence checking (structural + numeric
//! cross-check), equation solving (linear/quadratic), symbolic
//! differentiation, exact trigonometry at standard angles, `nsimplify`-style
//! float→exact conversion, and rational matrices (re-exported from
//! `reasoning-common`).
//!
//! Design notes:
//! - Numbers are ALWAYS exact rationals (`Rat`); floats never enter symbolic
//!   trees, so equivalence decisions are exact where sympy's would be.
//! - `equiv` falls back to numeric sampling at several sample points for
//!   expressions structural normalization can't cancel (e.g. trig
//!   identities), mirroring sympy's `simplify(a - b) == 0` outcomes for the
//!   well-separated candidates this project tests.

pub mod diff;
pub mod evalf;
pub mod expr;
pub mod norm;
pub mod nsimplify;
pub mod parser;
pub mod solve;
pub mod trig;

pub use diff::diff;
pub use evalf::{equiv, eval_f64, is_zero};
pub use expr::{Expr, FuncKind};
pub use norm::{expand, simplify};
pub use nsimplify::nsimplify;
pub use parser::{parse_expr, parse_expr_with_locals, ParseError};
pub use solve::solve_equation;
