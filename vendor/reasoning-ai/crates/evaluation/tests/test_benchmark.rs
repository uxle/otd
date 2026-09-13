//! Port of python/tests/test_benchmark.py (TestBenchmarkHarness).

mod common;

use common::train_a_prm;
use reasoning_evaluation::{format_report, run_benchmark};
use reasoning_prm::LinearPRM;

#[test]
fn test_benchmark_harness_report_structure_is_sane() {
    let prm = train_a_prm(0);
    let report = run_benchmark(&prm, 42).unwrap();

    assert!(report.level1_number_target.n_problems > 0);
    assert!(
        (0.0..=1.0).contains(&report.level1_number_target.accuracy),
        "accuracy out of range"
    );
    assert!(report.level1_number_target.avg_nodes_expanded > 0.0);

    assert!(report.level2_linear_equation.n_problems > 0);
    assert!((0.0..=1.0).contains(&report.level2_linear_equation.accuracy));

    assert!(!report.prm_calibration.buckets.is_empty());
    for b in &report.prm_calibration.buckets {
        assert!((0.0..=1.0).contains(&b.avg_predicted));
        assert!((0.0..=1.0).contains(&b.actual_success_rate));
    }
}

#[test]
fn test_benchmark_harness_report_formats_without_crashing() {
    let prm = LinearPRM::new();
    let report = run_benchmark(&prm, 1).unwrap();
    let text = format_report(&report);
    assert!(text.contains("Benchmark Report"));
    assert!(text.contains("Level 1"));
    assert!(text.contains("Level 2"));
}
