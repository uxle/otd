//! Port of python/tests/test_critic.py (TestCritic).

use reasoning_search::{Domain, LinearEquationDomain, Mcts};
use reasoning_uncertainty::{critique, INTEGER_VALUED, NON_NEGATIVE, POSITIVE};
use reasoning_verifier::symbolic_verifier::verify_equation_solution;

fn solve_and_verify(equation: &str, seed: u64) -> (bool, Option<f64>) {
    let domain = LinearEquationDomain::try_new(equation, 6).unwrap();
    let mut mcts = Mcts::new(domain.clone(), 6, seed);
    let result = mcts.search(domain.initial_state(), 1000);
    if !result.found_verified_solution {
        return (false, None);
    }
    // Python: float(result.best_terminal_state.rhs)
    let value = result
        .best_terminal_state
        .as_ref()
        .and_then(|s| s.rhs.as_rat())
        .map(|r| r.to_f64());
    let check = verify_equation_solution(equation, "x", value.unwrap(), 1e-9);
    (check.map(|c| c.passed).unwrap_or(false), value)
}

#[test]
fn test_critic_rejects_a_genuinely_verified_but_negative_answer() {
    // "2x = -6" is a perfectly valid, solver-verifiable equation (x=-3),
    // but if this represents "how many apples," negative is nonsense.
    let (verified, value) = solve_and_verify("2*x = -6", 1);
    assert!(verified); // the SOLVER is right that x=-3 solves the equation
    assert!((value.unwrap() - (-3.0)).abs() < 1e-9);

    let result = critique(verified, value, &[NON_NEGATIVE]);
    assert!(result.solver_verified);
    assert!(!result.final_accept); // but the CRITIC correctly rejects it
    assert!(result.violated_constraints.contains(&"non_negative".to_string()));
}

#[test]
fn test_critic_accepts_a_valid_and_constraint_satisfying_answer() {
    let (verified, value) = solve_and_verify("2*x = 6", 2);
    assert!(verified);
    assert!((value.unwrap() - 3.0).abs() < 1e-9);

    let result = critique(verified, value, &[NON_NEGATIVE, POSITIVE]);
    assert!(result.final_accept);
    assert_eq!(result.violated_constraints, Vec::<String>::new());
}

#[test]
fn test_critic_catches_non_integer_when_integer_required() {
    let (verified, value) = solve_and_verify("2*x = 5", 3); // x = 2.5
    assert!(verified);
    let result = critique(verified, value, &[INTEGER_VALUED]);
    assert!(!result.final_accept);
    assert!(result
        .violated_constraints
        .contains(&"integer_valued".to_string()));
}

#[test]
fn test_critic_never_accepts_an_unverified_answer_regardless_of_constraints() {
    // sanity: even with zero constraints, an unverified solver result must
    // never be accepted
    let result = critique(false, None, &[]);
    assert!(!result.final_accept);
}
