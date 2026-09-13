//! Port of python/tests/test_calculus_domain.py (TestDerivativeDomain).

use reasoning_search::{DerivativeDomain, Domain, Mcts};

#[test]
fn test_calculus_finds_correct_derivative_of_polynomial() {
    // d/dx(x^3 + 2x) = 3x^2 + 2
    let candidates = vec![
        "3*x**2 + 2".to_string(),
        "x**2 + 2".to_string(),
        "3*x**2".to_string(),
        "2*x + 2".to_string(),
        "3*x**2 + 2*x".to_string(),
    ];
    let domain = DerivativeDomain::try_new("x**3 + 2*x", candidates.clone()).unwrap();
    assert_eq!(domain.ground_truth, "3*x**2 + 2");

    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 1, 1);
    let result = mcts.search(initial, 100);
    assert!(result.found_verified_solution);
    assert_eq!(
        candidates[result.best_terminal_state.as_ref().unwrap().choice.unwrap()],
        "3*x**2 + 2"
    );
}

#[test]
fn test_calculus_finds_correct_derivative_via_product_rule() {
    // d/dx(x * sin(x)) = sin(x) + x*cos(x)
    let candidates = vec![
        "sin(x) + x*cos(x)".to_string(),
        "cos(x)".to_string(),
        "x*cos(x)".to_string(),
        "sin(x)*cos(x)".to_string(),
    ];
    let domain = DerivativeDomain::try_new("x*sin(x)", candidates.clone()).unwrap();
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 1, 2);
    let result = mcts.search(initial, 100);
    assert!(result.found_verified_solution);
    let chosen = &candidates[result.best_terminal_state.as_ref().unwrap().choice.unwrap()];
    assert_eq!(chosen, "sin(x) + x*cos(x)");
}

#[test]
fn test_calculus_honestly_fails_when_no_candidate_is_correct() {
    let candidates = vec!["x**2".to_string(), "5*x".to_string(), "x + 1".to_string()];
    // correct answer 3*x**2 not among candidates
    let domain = DerivativeDomain::try_new("x**3", candidates).unwrap();
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 1, 3);
    let result = mcts.search(initial, 50);
    assert!(!result.found_verified_solution);
}
