//! Port of python/tests/test_multi_path.py (TestMultiPathReasoning).

use reasoning_search::{make_initial_state, solve_number_target_multi_path, NumberTargetDomain};
use reasoning_verifier::symbolic_verifier::verify_numeric_equality;

fn vne_passed(expr: &str, expected: f64) -> bool {
    verify_numeric_equality(expr, expected)
        .map(|r| r.passed)
        .unwrap_or(false)
}

#[test]
fn test_multi_path_at_least_one_strategy_solves_easy_problem() {
    let domain = NumberTargetDomain::new(24.0);
    let state = make_initial_state(&[4.0, 7.0, 8.0, 8.0]);
    let result = solve_number_target_multi_path(&domain, &state, 24.0, 1);

    assert!(result.consensus_answer.is_some());
    assert!(!result.contradiction_detected);
    // independently re-verify the consensus one more time, outside the module
    let consensus = result.consensus_answer.clone().unwrap();
    assert!(vne_passed(&consensus, 24.0));
}

#[test]
fn test_multi_path_reports_per_strategy_outcomes() {
    let domain = NumberTargetDomain::new(4.0);
    let state = make_initial_state(&[2.0, 2.0]);
    let result = solve_number_target_multi_path(&domain, &state, 4.0, 2);
    assert_eq!(result.paths.len(), 3);
    let strategy_names: std::collections::HashSet<&str> =
        result.paths.iter().map(|p| p.strategy.as_str()).collect();
    let expected: std::collections::HashSet<&str> =
        ["mcts", "best_first", "beam"].into_iter().collect();
    assert_eq!(strategy_names, expected);
}

#[test]
fn test_multi_path_honestly_reports_no_consensus_when_all_strategies_fail() {
    let domain = NumberTargetDomain::new(1e9);
    let state = make_initial_state(&[1.0, 1.0, 1.0]);
    let result = solve_number_target_multi_path(&domain, &state, 1e9, 3);
    assert!(result.consensus_answer.is_none());
    assert!(!result.contradiction_detected);
    assert!(result.paths.iter().all(|p| !p.found));
}
