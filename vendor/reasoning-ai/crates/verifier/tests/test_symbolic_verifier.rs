//! Port of python/tests/test_symbolic_verifier.py.

use reasoning_verifier::symbolic_verifier::{
    solve_equation, verify_algebraic_equivalence, verify_equation_solution,
    verify_numeric_equality, verify_numeric_equality_with_tol,
};

// ---- TestNumericEquality ----

#[test]
fn test_numeric_equality_correct_arithmetic() {
    let r = verify_numeric_equality("2 + 2", 4.0).unwrap();
    assert!(r.passed);
}

#[test]
fn test_numeric_equality_wrong_arithmetic_is_rejected() {
    let r = verify_numeric_equality("2 + 2", 5.0).unwrap();
    assert!(!r.passed);
}

#[test]
fn test_numeric_equality_order_of_operations() {
    let r = verify_numeric_equality("2 + 3 * 4", 14.0).unwrap(); // not 20
    assert!(r.passed);
}

#[test]
fn test_numeric_equality_irrational_within_tolerance() {
    let r = verify_numeric_equality_with_tol("sqrt(2)", 1.41421356237, 1e-6).unwrap();
    assert!(r.passed);
}

#[test]
fn test_numeric_equality_garbage_input_raises_not_silently_passes() {
    assert!(verify_numeric_equality("this is not math @#$", 1.0).is_err());
}

// ---- TestAlgebraicEquivalence ----

#[test]
fn test_algebraic_equivalence_expansion_equivalent() {
    let r = verify_algebraic_equivalence("(x+1)**2", "x**2 + 2*x + 1").unwrap();
    assert!(r.passed);
}

#[test]
fn test_algebraic_equivalence_non_equivalent_rejected() {
    let r = verify_algebraic_equivalence("(x+1)**2", "x**2 + 2*x + 2").unwrap();
    assert!(!r.passed);
}

#[test]
fn test_algebraic_equivalence_factoring() {
    let r = verify_algebraic_equivalence("x**2 - 9", "(x-3)*(x+3)").unwrap();
    assert!(r.passed);
}

// ---- TestEquationSolution ----

#[test]
fn test_equation_solution_correct_root() {
    let r = verify_equation_solution("2*x + 4 = 0", "x", -2.0, 1e-9).unwrap();
    assert!(r.passed);
}

#[test]
fn test_equation_solution_wrong_root_rejected() {
    let r = verify_equation_solution("2*x + 4 = 0", "x", 2.0, 1e-9).unwrap();
    assert!(!r.passed);
}

#[test]
fn test_equation_solution_quadratic_root() {
    // x^2 - 5x + 6 = 0 -> roots 2, 3
    let r2 = verify_equation_solution("x**2 - 5*x + 6 = 0", "x", 2.0, 1e-9).unwrap();
    let r3 = verify_equation_solution("x**2 - 5*x + 6 = 0", "x", 3.0, 1e-9).unwrap();
    let r_wrong = verify_equation_solution("x**2 - 5*x + 6 = 0", "x", 4.0, 1e-9).unwrap();
    assert!(r2.passed);
    assert!(r3.passed);
    assert!(!r_wrong.passed);
}

// ---- TestSolveEquation ----

#[test]
fn test_solve_equation_linear_solve() {
    let roots = solve_equation("3*x - 9 = 0", "x").unwrap();
    assert_eq!(roots.len(), 1);
    assert_eq!(roots[0].as_int(), Some(3));
}

#[test]
fn test_solve_equation_quadratic_solve_matches_verifier() {
    let roots = solve_equation("x**2 - 5*x + 6 = 0", "x").unwrap();
    let ints: std::collections::HashSet<i64> =
        roots.iter().filter_map(|r| r.as_int()).collect();
    assert_eq!(ints, [2, 3].into_iter().collect());
    // cross-check: every root returned by the solver must itself pass
    // the independent equation-solution verifier (consistency check)
    for r in roots {
        let v = r.as_rat().map(|x| x.to_f64()).unwrap_or(0.0);
        assert!(verify_equation_solution("x**2 - 5*x + 6 = 0", "x", v, 1e-9)
            .unwrap()
            .passed);
    }
}
