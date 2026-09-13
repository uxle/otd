//! Port of python/tests/test_full_integration.py (TestFullIntegration).
//!
//! Originally ported in the engine crate's tests with the whole
//! `evaluation/benchmark.py` logic inlined (the evaluation crate didn't
//! exist then); moved here once this crate landed so it calls the real
//! `run_full_benchmark` / `format_full_report` like the Python does.

mod common;

use common::build_default_prm;
use reasoning_evaluation::{format_full_report, run_full_benchmark};

#[test]
fn test_full_integration_full_pipeline_runs_and_produces_sane_report() {
    let prm = build_default_prm();
    let report = run_full_benchmark(&prm, 42).unwrap();

    // Python's key list maps to the report struct's fields.
    let domains = [
        (&report.level1_number_target, "level1_number_target"),
        (&report.level2_linear_equation, "level2_linear_equation"),
        (&report.level3_quadratic_factoring, "level3_quadratic_factoring"),
        (
            &report.level5_number_theory_bezout,
            "level5_number_theory_bezout",
        ),
    ];
    for (r, key) in domains {
        assert!(
            (0.0..=1.0).contains(&r.accuracy),
            "{} accuracy {}",
            key,
            r.accuracy
        );
        assert!(r.n_problems > 0, "{} n_problems", key);
    }

    let text = format_full_report(&report);
    assert!(text.contains("Full Benchmark Report"));
    assert!(text.contains("Level 5"));
}
