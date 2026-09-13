//! Port of python/tests/test_linear_equation_domain.py
//! (TestLinearEquationDomain).

use reasoning_search::{Domain, LinearEquationDomain, Mcts};

#[test]
fn test_linear_equation_solves_simple_equation() {
    let domain = LinearEquationDomain::try_new("2*x + 4 = 10", 6).unwrap(); // x = 3
    let mut mcts = Mcts::new(domain.clone(), 6, 1);
    let result = mcts.search(domain.initial_state(), 1500);
    assert!(result.found_verified_solution);
    let rhs = result
        .best_terminal_state
        .as_ref()
        .unwrap()
        .rhs
        .as_rat()
        .map(|r| r.to_f64())
        .unwrap();
    assert!((rhs - 3.0).abs() < 1e-7); // assertAlmostEqual default places=7
}

#[test]
fn test_linear_equation_solves_negative_solution() {
    let domain = LinearEquationDomain::try_new("3*x - 9 = -18", 6).unwrap(); // x = -3
    let mut mcts = Mcts::new(domain.clone(), 6, 2);
    let result = mcts.search(domain.initial_state(), 1500);
    assert!(result.found_verified_solution);
    let rhs = result
        .best_terminal_state
        .as_ref()
        .unwrap()
        .rhs
        .as_rat()
        .map(|r| r.to_f64())
        .unwrap();
    assert!((rhs - (-3.0)).abs() < 1e-7);
}

#[test]
fn test_linear_equation_step_trace_is_real_algebra() {
    // confirm the path found is an actual sequence of legal add/sub/mul/div
    // steps, not a shortcut
    let domain = LinearEquationDomain::try_new("5*x + 1 = 16", 6).unwrap(); // x = 3
    let mut mcts = Mcts::new(domain.clone(), 6, 3);
    let result = mcts.search(domain.initial_state(), 1500);
    assert!(result.found_verified_solution);
    let mut state = domain.initial_state();
    for action in result.verified_path.as_ref().unwrap() {
        assert!(domain.legal_actions(&state).contains(action));
        state = domain.apply(&state, action);
    }
    assert!(domain.is_terminal(&state));
    let rhs = state.rhs.as_rat().map(|r| r.to_f64()).unwrap();
    assert!((rhs - 3.0).abs() < 1e-7);
}
