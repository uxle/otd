//! Reasoning AI — evaluation crate (Rust port of the Python
//! `evaluation/` package): benchmark harnesses for the search/PRM/RL stack
//! and the full-system solve() scorecard.

pub mod benchmark;
pub mod full_system_benchmark;
pub mod rl_benchmark;
pub mod rl_benchmark_low_budget;
pub mod rl_benchmark_v2;
pub mod rl_benchmark_v3;

pub use benchmark::{
    eval_bezout, eval_combinatorics, eval_linear_equation, eval_logic, eval_number_target,
    eval_prm_calibration, eval_quadratic_factoring, eval_statistics, eval_word_problems,
    format_full_report, format_phase_1_30_report, format_phase_1_40_report, format_report,
    run_benchmark, run_full_benchmark, run_phase_1_30_benchmark, run_phase_1_40_benchmark,
    BenchmarkReport, CalibrationBucket, DomainEval, FullReport, Phase130Report, Phase140Report,
    PrmCalibration,
};

pub use full_system_benchmark::{
    run_full_system_benchmark, DomainResult, FullSystemReport, DEFAULT_KINDS, GroundTruth,
};

pub use rl_benchmark::{
    run_three_way_comparison, ThreeWayReport,
};

pub use rl_benchmark_low_budget::run_low_budget_comparison;

pub use rl_benchmark_v2::{
    run_four_way_comparison, ConditionResult, FourWayReport,
};

pub use rl_benchmark_v3::{
    run_five_way_comparison, FiveWayReport,
};
