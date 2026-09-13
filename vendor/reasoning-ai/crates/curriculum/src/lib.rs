//! Reasoning AI — curriculum crate (Rust port of the Python `curriculum`
//! package): verified problem generation (Phase 005), adaptive progression
//! (Phase 014), template-based word problems (Phase 028), composite
//! two-step word problems (Phase 037) and adaptive compute (Phase 008).
//!
//! `adversarial.py` is NOT ported here: it depends on `apps.solve`, which
//! lives in the engine crate per the workspace layering (see worklog).

pub mod adaptive_compute;
pub mod composite_word_problems;
pub mod generator;
pub mod progression;
pub mod word_problems;

pub use adaptive_compute::estimate_budget;
pub use composite_word_problems::{solve_composite_word_problem, CompositeResult};
pub use generator::{
    generate_curriculum, gen_arithmetic, gen_linear_equation, gen_polynomial_identity, Problem,
};
pub use progression::{run_curriculum, CurriculumController, HistoryEntry};
pub use word_problems::{parse_word_problem, solve_word_problem, ParsedProblem};
