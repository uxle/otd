//! Port of python/tests/test_quadratic_factoring.py
//! (TestQuadraticFactoringDomain).

use reasoning_search::{Domain, Mcts, QuadraticFactoringDomain};
use reasoning_verifier::symbolic_verifier::verify_algebraic_equivalence;

#[test]
fn test_quadratic_factoring_factors_x2_minus_5x_plus_6() {
    // x^2 - 5x + 6 = (x-2)(x-3)
    let domain = QuadraticFactoringDomain::new(-5, 6, 10);
    let b = domain.b;
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 2, 1);
    let result = mcts.search(initial, 200);
    assert!(result.found_verified_solution);
    let p = result.best_terminal_state.as_ref().unwrap().p.unwrap();
    let q = b - p;
    // independently re-verify with the Phase 002 verifier directly
    let check = verify_algebraic_equivalence(&format!("(x+{})*(x+{})", p, q), "x**2 - 5*x + 6")
        .expect("verifier call must succeed");
    assert!(check.passed);
    let mut roots = vec![p, q];
    roots.sort_unstable();
    assert_eq!(roots, vec![-3, -2]);
}

#[test]
fn test_quadratic_factoring_factors_x2_plus_7x_plus_10() {
    // (x+2)(x+5)
    let domain = QuadraticFactoringDomain::new(7, 10, 10);
    let b = domain.b;
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 2, 2);
    let result = mcts.search(initial, 200);
    assert!(result.found_verified_solution);
    let p = result.best_terminal_state.as_ref().unwrap().p.unwrap();
    let q = b - p;
    let mut roots = vec![p, q];
    roots.sort_unstable();
    assert_eq!(roots, vec![2, 5]);
}

#[test]
fn test_quadratic_factoring_honestly_fails_on_irrational_roots() {
    // x^2 - 3 has no integer factorization at all (roots +-sqrt(3))
    let domain = QuadraticFactoringDomain::new(0, -3, 15);
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 2, 3);
    let result = mcts.search(initial, 200);
    assert!(!result.found_verified_solution);
}
