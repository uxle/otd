//! Port of python/tests/test_abstention.py (TestAbstention).

use reasoning_search::{make_initial_state, NumberTargetDomain};
use reasoning_uncertainty::solve_with_abstention;
use reasoning_verifier::symbolic_verifier::verify_numeric_equality;

#[test]
fn test_confident_answer_on_solvable_problem() {
    let domain = NumberTargetDomain::new(24.0);
    let state = make_initial_state(&[4.0, 7.0, 8.0, 8.0]);
    let ans = solve_with_abstention(&domain, &state, 24.0, 1);

    assert!(ans.verified);
    assert_eq!(ans.confidence, 1.0);
    assert!(ans.final_answer().is_some());
    assert!(verify_numeric_equality(ans.final_answer().unwrap(), 24.0)
        .map(|r| r.passed)
        .unwrap_or(false));
}

#[test]
fn test_abstains_on_unsolvable_problem() {
    let domain = NumberTargetDomain::new(1e9);
    let state = make_initial_state(&[1.0, 1.0, 1.0]);
    let ans = solve_with_abstention(&domain, &state, 1e9, 2);

    assert!(!ans.verified);
    assert!(
        ans.final_answer().is_none(),
        "final_answer() must never return a value when unverified"
    );
    assert_eq!(ans.confidence, 0.0);
    assert!(ans.explanation.contains("Abstaining"));
}

#[test]
fn test_final_answer_is_the_only_sanctioned_path_even_with_a_guess_present() {
    // critical structural guarantee: even when unverified_best_guess IS
    // populated (for transparency), final_answer() still returns None.
    let domain = NumberTargetDomain::new(1e9);
    let state = make_initial_state(&[1.0, 1.0, 1.0]);
    let ans = solve_with_abstention(&domain, &state, 1e9, 3);

    // the guess field may or may not be populated depending on search, but
    // regardless, final_answer() must be None whenever verified=false
    assert!(!ans.verified);
    assert!(ans.final_answer().is_none());
}
