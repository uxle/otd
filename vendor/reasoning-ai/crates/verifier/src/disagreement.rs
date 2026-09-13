//! Phase 033 — Verifier disagreement detection (design doc section 10)
//! (Rust port of python/verifier/disagreement.py).
//!
//! Runs multiple independent verification methods on the same claim and
//! flags cases where they disagree, rather than silently trusting whichever
//! one ran first. Real disagreement source: floating-point numeric checks
//! can pass on values that aren't EXACTLY equal (within tolerance) while
//! exact symbolic checks fail, or vice versa for expressions the CAS can't
//! fully simplify. This module makes that visible instead of hiding it.

use crate::symbolic_verifier::{
    safe_parse, verify_algebraic_equivalence, verify_numeric_equality_with_tol,
};
use reasoning_symbolic::eval_f64;
use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct DisagreementReport {
    pub claim: String,
    pub numeric_passed: Option<bool>,
    pub symbolic_passed: bool,
    pub agree: bool,
    pub detail: String,
}

fn fmt_opt_bool(v: Option<bool>) -> &'static str {
    match v {
        Some(true) => "True",
        Some(false) => "False",
        None => "None",
    }
}

fn fmt_bool(v: bool) -> &'static str {
    if v {
        "True"
    } else {
        "False"
    }
}

/// Checks `expr_a == expr_b` two independent ways: numerically (evaluate
/// both, compare within tolerance) and symbolically (simplify the
/// difference to exactly zero). Reports whether they agree.
pub fn cross_check_equality_claim(
    expr_a: &str,
    expr_b: &str,
    tolerance: f64,
) -> DisagreementReport {
    // numeric: evaluate expr_b and compare expr_a against its numeric value.
    // Python wraps this in `except (VerificationError, TypeError)`: an
    // expr_b with free variables can't be reduced to a float (TypeError) —
    // the numeric method genuinely doesn't apply, not a pass or a fail, so
    // fall back to symbolic-only rather than failing the whole check.
    let numeric_passed: Option<bool> = (|| {
        let b_expr = safe_parse(expr_b).ok()?;
        let b_value = eval_f64(&b_expr, &HashMap::new()).ok()?;
        Some(verify_numeric_equality_with_tol(expr_a, b_value, tolerance).ok()?.passed)
    })();

    let symbolic_passed = match verify_algebraic_equivalence(expr_a, expr_b) {
        Ok(r) => r.passed,
        Err(_) => false,
    };

    let agree = numeric_passed.is_none() || numeric_passed == Some(symbolic_passed);
    let mut detail = format!(
        "numeric={}, symbolic={}",
        fmt_opt_bool(numeric_passed),
        fmt_bool(symbolic_passed)
    );
    if !agree {
        detail.push_str(" -- DISAGREEMENT, do not trust either verdict blindly");
    }
    if numeric_passed.is_none() {
        detail.push_str(" (numeric check not applicable: free variables)");
    }
    DisagreementReport {
        claim: format!("{} == {}", expr_a, expr_b),
        numeric_passed,
        symbolic_passed,
        agree,
        detail,
    }
}
