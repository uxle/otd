//! Port of python/tests/test_rl_phase_96.py (TestFullSystemBenchmark).

use reasoning_evaluation::{run_full_system_benchmark, DEFAULT_KINDS};

#[test]
fn test_full_system_benchmark_runs_all_default_kinds_with_tiny_params() {
    let report = run_full_system_benchmark(None, 2, 0, Some(40)).unwrap();
    assert_eq!(report.domains.len(), DEFAULT_KINDS.len());
    for d in &report.domains {
        assert_eq!(d.n, 2);
        assert!((0.0..=1.0).contains(&d.solve_rate()));
        assert_eq!(
            d.wrong_verified, 0,
            "{}: a verified answer failed independent re-check -- real bug",
            d.kind
        );
    }
}

#[test]
fn test_full_system_benchmark_overall_wrong_verified_rate_is_zero() {
    // The one number this whole project's correctness claim rests on:
    // nothing marked verified should ever be wrong.
    let report = run_full_system_benchmark(
        Some(&["matrix_determinant", "trig_evaluate", "linear_system"]),
        4,
        1,
        Some(50),
    )
    .unwrap();
    assert_eq!(report.overall_wrong_verified_rate(), 0.0);
}
