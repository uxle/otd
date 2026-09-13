//! Reasoning AI — engine crate (Rust port of the Python `apps/` package +
//! `nlp/solver.py` + `uncertainty/audit_log.py` +
//! `curriculum/adversarial.py`).
//!
//! The integration layer: the unified typed solve() API (Phase 030), the
//! expanded science/puzzle routes (Phases 110-123), the natural-language
//! ask() front door (Phase 125), audit logging (Phase 044), adversarial
//! stress tests (Phase 039), memory integration (Phase 098), and input
//! validation (Phase 045). Python's import cycles (apps <-> nlp /
//! curriculum / uncertainty) are broken here: this crate sits above all
//! domain crates and below the app crate's CLI binaries.

pub mod audit_log;
pub mod curriculum_adversarial;
pub mod integrated_solve;
pub mod nlp_solver;
pub mod science_solve;
pub mod solve;
pub mod validation;

pub use nlp_solver::ask;
pub use solve::{solve, Problem};

pub use integrated_solve::{IntegratedSolver, SolverStats};
