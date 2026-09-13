//! Port of python/tests/test_capstone_benchmark.py (TestCapstoneBenchmark).

mod common;

use common::build_default_prm;
use reasoning_evaluation::{format_phase_1_40_report, run_phase_1_40_benchmark};

#[test]
fn test_capstone_benchmark_capstone_covers_every_domain() {
    let prm = build_default_prm();
    let report = run_phase_1_40_benchmark(&prm, 42).unwrap();

    // Python's expected_keys dict checks, field by field.
    let accuracies = [
        &report.level1_number_target,
        &report.level2_linear_equation,
        &report.level3_quadratic_factoring,
        &report.level5_number_theory_bezout,
        &report.level3_combinatorics,
        &report.level4_word_problems,
        &report.level5_logic,
        &report.level5_statistics,
    ];
    for r in accuracies {
        assert!((0.0..=1.0).contains(&r.accuracy), "accuracy out of range");
    }

    let text = format_phase_1_40_report(&report);
    assert!(text.contains("logic"));
    assert!(text.contains("statistics"));
}
