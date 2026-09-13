//! Verifier — the ground-truth signal the whole system depends on
//! (Rust port of python/verifier/symbolic_verifier.py).
//!
//! No neural component here at all, by design: search and self-evolution
//! (design doc sections 2.7-2.9) only work if verification is *independent*
//! of the policy model. If the verifier could be fooled by confident-looking
//! but wrong reasoning, the self-evolution loop would train the model to
//! produce confident wrong answers.
//!
//! Parsing goes through the mini-CAS parser in `reasoning-symbolic`
//! (implicit multiplication + `^`/`**` powers), never anything like `eval`
//! on model output. Security parity with the Python original's two-layer
//! defense: the parser rejects attribute access (`x.__class__`) with
//! "attribute access (.x) is not permitted in math expressions", which is
//! surfaced here as a `VerificationError` exactly like Python's
//! `_reject_attribute_access` layer, and unknown/other syntax fails closed.

use reasoning_common::{py_float_str, Rat};
use reasoning_symbolic::{
    equiv, eval_f64, parse_expr_with_locals, simplify, solve_equation as sym_solve, Expr,
};
use std::collections::HashMap;
use std::fmt;

/// Result of a verification run: `passed` = the claim held, `method` =
/// which independent check ran, `detail` = human-readable evidence.
/// `confidence`: 1.0 = exact symbolic/numeric proof, <1.0 = approximate.
#[derive(Debug, Clone, PartialEq)]
pub struct VerificationResult {
    pub passed: bool,
    pub method: String,
    pub detail: String,
    pub confidence: f64,
}

/// Raised when the input can't even be parsed as math — distinct from
/// "parsed fine but the answer is wrong" (that's `passed=false`, not an
/// error).
#[derive(Debug, Clone)]
pub struct VerificationError(pub String);

impl fmt::Display for VerificationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for VerificationError {}

impl From<String> for VerificationError {
    fn from(msg: String) -> Self {
        VerificationError(msg)
    }
}

/// Safe parse of a math expression (mirrors Python `_safe_parse` with no
/// local symbol table). Any parse failure — including attempted attribute
/// access, which the parser rejects outright — becomes a
/// `VerificationError` with the same "Could not parse ...: ..." message.
pub fn safe_parse(expr_str: &str) -> Result<Expr, VerificationError> {
    safe_parse_with_locals(expr_str, &[])
}

/// Safe parse with an explicit local symbol table (e.g. `&["x"]` when
/// verifying an equation in x — mirrors Python's `local_dict`).
pub fn safe_parse_with_locals(expr_str: &str, locals: &[&str]) -> Result<Expr, VerificationError> {
    parse_expr_with_locals(expr_str, locals)
        .map_err(|e| VerificationError(format!("Could not parse {:?}: {}", expr_str, e)))
}

/// Check that `expr_str` evaluates to `expected`, within 1e-9.
pub fn verify_numeric_equality(
    expr_str: &str,
    expected: f64,
) -> Result<VerificationResult, VerificationError> {
    verify_numeric_equality_with_tol(expr_str, expected, 1e-9)
}

/// Check that `expr_str` evaluates to `expected`, within `tolerance`.
pub fn verify_numeric_equality_with_tol(
    expr_str: &str,
    expected: f64,
    tolerance: f64,
) -> Result<VerificationResult, VerificationError> {
    let expr = safe_parse(expr_str)?;
    // Python wraps N() in complex() and rejects a non-zero imaginary part;
    // the CAS evaluates over the reals (complex values such as sqrt(-1)
    // fail at evaluation and surface as VerificationError instead).
    let value = eval_f64(&expr, &HashMap::new()).map_err(|e| {
        VerificationError(format!("Could not numerically evaluate {:?}: {}", expr_str, e))
    })?;
    let diff = (value - expected).abs();
    let passed = diff <= tolerance;
    Ok(VerificationResult {
        passed,
        method: "numeric".to_string(),
        detail: format!(
            "{} = {} vs expected {} (diff={})",
            expr_str,
            py_float_str(value),
            py_float_str(expected),
            py_float_str(diff)
        ),
        confidence: 1.0,
    })
}

/// Check that two expressions are the same function, e.g. confirming a
/// 'simplify' or 'factor' step didn't change the underlying expression.
pub fn verify_algebraic_equivalence(
    expr_a: &str,
    expr_b: &str,
) -> Result<VerificationResult, VerificationError> {
    let a = safe_parse(expr_a)?;
    let b = safe_parse(expr_b)?;
    let diff = simplify(&(a.clone() - b.clone()));
    let passed = equiv(&a, &b);
    Ok(VerificationResult {
        passed,
        method: "symbolic_equivalence".to_string(),
        detail: format!("simplify(({}) - ({})) = {}", expr_a, expr_b, diff),
        confidence: 1.0,
    })
}

/// Check that substituting `candidate` for `variable` satisfies an
/// equation given as 'lhs = rhs' or 'lhs - rhs' (implicitly = 0).
pub fn verify_equation_solution(
    equation_str: &str,
    variable: &str,
    candidate: f64,
    tolerance: f64,
) -> Result<VerificationResult, VerificationError> {
    let expr = if let Some((lhs_str, rhs_str)) = equation_str.split_once('=') {
        let lhs = safe_parse_with_locals(lhs_str, &[variable])?;
        let rhs = safe_parse_with_locals(rhs_str, &[variable])?;
        lhs - rhs
    } else {
        safe_parse_with_locals(equation_str, &[variable])?
    };

    let residual = expr.subs(variable, Expr::num(Rat::from_f64(candidate)));
    let residual_val = eval_f64(&residual, &HashMap::new())
        .map_err(|e| VerificationError(format!("Could not evaluate residual: {}", e)))?;

    let passed = residual_val.abs() <= tolerance;
    Ok(VerificationResult {
        passed,
        method: "equation_solution".to_string(),
        detail: format!(
            "residual at {}={}: {}",
            variable,
            py_float_str(candidate),
            py_float_str(residual_val)
        ),
        confidence: 1.0,
    })
}

/// Ground-truth solver used to auto-label self-play rollouts (design doc
/// 2.9): if the model's final answer matches a root returned here, the
/// trajectory is verified-correct with no human labeling needed.
/// Re-exports the CAS solver, mapping its parse failures into
/// `VerificationError` (the CAS already formats them
/// "Could not parse '...': ...").
pub fn solve_equation(equation_str: &str, variable: &str) -> Result<Vec<Expr>, VerificationError> {
    sym_solve(equation_str, variable).map_err(VerificationError)
}
