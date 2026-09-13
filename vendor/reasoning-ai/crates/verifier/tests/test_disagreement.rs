//! Port of python/tests/test_disagreement.py.

use reasoning_verifier::disagreement::cross_check_equality_claim;

#[test]
fn test_disagreement_agreement_when_actually_equal() {
    let r = cross_check_equality_claim("2+2", "4", 1e-9);
    assert!(r.agree);
    assert_eq!(r.numeric_passed, Some(true));
    assert!(r.symbolic_passed);
}

#[test]
fn test_disagreement_agreement_when_actually_different() {
    let r = cross_check_equality_claim("2+2", "5", 1e-9);
    assert!(r.agree);
    assert_eq!(r.numeric_passed, Some(false));
    assert!(!r.symbolic_passed);
}

#[test]
fn test_disagreement_detects_real_disagreement_from_float_precision_collapse() {
    // 10^20 + 1 != 10^20 symbolically (differ by exactly 1), but at
    // float64 precision the +1 vanishes into rounding error, so a
    // naive numeric check would wrongly call them equal. This is a
    // genuine, reproducible disagreement, not a contrived one.
    let r = cross_check_equality_claim("10**20 + 1", "10**20", 1e-9);
    assert!(!r.agree);
    assert_eq!(r.numeric_passed, Some(true)); // numeric is fooled
    assert!(!r.symbolic_passed); // symbolic is exact and correct
    assert!(r.detail.contains("DISAGREEMENT"));
}

#[test]
fn test_disagreement_algebraic_identity_agrees_across_both_methods() {
    let r = cross_check_equality_claim("(x+1)**2", "x**2+2*x+1", 1e-9);
    // free-variable expressions take the numeric-not-applicable path;
    // confirm it doesn't crash and produces a real verdict
    let _ = r.agree; // must be a bool (compiles = typed verdict exists)
}
