//! Reasoning AI — selfplay crate (Rust port of the Python `selfplay`
//! package): the Phase 009 self-evolution loop and its Phase 013 retry
//! with root-oversampling calibration.

pub mod self_evolution;
pub mod self_evolution_v2;

pub use self_evolution::{evaluate, run_self_evolution, Problem};
pub use self_evolution_v2::{
    evaluate as evaluate_v2, make_eval_set, make_round_problems, run_self_evolution_v2,
};
