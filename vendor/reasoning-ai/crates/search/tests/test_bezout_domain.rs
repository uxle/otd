//! Port of python/tests/test_bezout_domain.py (TestBezoutDomain).

use reasoning_search::{BezoutIdentityDomain, Domain, Mcts};
use reasoning_verifier::symbolic_verifier::verify_numeric_equality;

/// math.gcd parity (test's ground truth cross-check).
fn math_gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[test]
fn test_bezout_finds_bezout_coefficients_coprime() {
    // gcd(35, 12) = 1
    let domain = BezoutIdentityDomain::new(35, 12, 15);
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 3, 1);
    let result = mcts.search(initial, 1000);
    assert!(result.found_verified_solution);
    let x = result.best_terminal_state.as_ref().unwrap().x.unwrap();
    let y = result.best_terminal_state.as_ref().unwrap().y.unwrap();
    // independent re-check
    let check = verify_numeric_equality(&format!("35*({}) + 12*({})", x, y), 1.0)
        .expect("verifier call must succeed");
    assert!(check.passed);
}

#[test]
fn test_bezout_finds_bezout_coefficients_with_common_factor() {
    // gcd(24, 18) = 6
    let domain = BezoutIdentityDomain::new(24, 18, 10);
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 3, 2);
    let result = mcts.search(initial, 1000);
    assert!(result.found_verified_solution);
    let x = result.best_terminal_state.as_ref().unwrap().x.unwrap();
    let y = result.best_terminal_state.as_ref().unwrap().y.unwrap();
    let check = verify_numeric_equality(&format!("24*({}) + 18*({})", x, y), 6.0)
        .expect("verifier call must succeed");
    assert!(check.passed);
}

#[test]
fn test_bezout_gcd_matches_python_stdlib() {
    // sanity: our target (self.gcd) must agree with math.gcd, since that's
    // the ground truth the verifier's checking against
    let domain = BezoutIdentityDomain::new(35, 12, 200);
    assert_eq!(domain.gcd, math_gcd(35, 12));
}
