//! Port of python/tests/test_expanded_benchmark.py (TestExpandedBenchmark).

mod common;

use common::build_default_prm;
use reasoning_evaluation::{format_phase_1_30_report, run_phase_1_30_benchmark};

#[test]
fn test_expanded_benchmark_expanded_report_covers_all_domains() {
    let prm = build_default_prm();
    let report = run_phase_1_30_benchmark(&prm, 42).unwrap();

    // Python's expected_keys dict checks, field by field.
    let accuracies = [
        &report.level1_number_target,
        &report.level2_linear_equation,
        &report.level3_quadratic_factoring,
        &report.level5_number_theory_bezout,
        &report.level3_combinatorics,
        &report.level4_word_problems,
    ];
    for r in accuracies {
        assert!((0.0..=1.0).contains(&r.accuracy), "accuracy out of range");
    }

    let text = format_phase_1_30_report(&report);
    assert!(text.contains("combinatorics"));
    assert!(text.contains("word problems"));
}
