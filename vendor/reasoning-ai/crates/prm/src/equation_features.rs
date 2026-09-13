//! Phase 012 — Extend PRM to the algebra (linear equation) domain (Rust
//! port of `python/prm/equation_features.py`).
//!
//! Phase 004's features only knew how to read NumberTargetDomain states
//! (tuples of (value, expr)). This adds a feature extractor for
//! LinearEquationDomain's EqState (lhs/rhs expressions), so the same
//! LinearPRM can be trained and used for guiding real algebraic-step search
//! too — proving the PRM mechanism generalizes across domains (design doc
//! section 3).

use crate::features::FeatureMap;
use reasoning_search::EqState;
use reasoning_symbolic::{expand, is_zero, Expr};

pub fn extract_equation_features(state: &EqState, max_depth: usize) -> FeatureMap {
    let (lhs, rhs) = (&state.lhs, &state.rhs);
    let lhs_has_x = lhs.has_sym("x");
    let rhs_has_x = rhs.has_sym("x");
    let x = Expr::sym("x");

    // crude "how isolated is x" signal: 1.0 if x is alone on one side with
    // no x on the other, degrading as more terms/complexity remain.
    // (Python guarded `len(Add.make_args(expand(...)))` with try/except
    // defaulting to 5 — the Rust `expand` is total, so the fallback is
    // unreachable.)
    let isolation = if lhs_has_x && !rhs_has_x {
        let lhs_terms = expand(lhs).terms().len();
        1.0 / lhs_terms as f64
    } else if rhs_has_x && !lhs_has_x {
        let rhs_terms = expand(rhs).terms().len();
        1.0 / rhs_terms as f64
    } else {
        0.0 // x on both sides, or x fully gone (degenerate/wrong)
    };

    // Python: sympy.simplify(lhs - X) == 0 and not rhs_has_x — the domain's
    // expand() collapses every equivalent-to-x form it can produce.
    let is_solved = if is_zero(&expand(&(lhs.clone() - x))) && !rhs_has_x {
        1.0
    } else {
        0.0
    };

    let mut feats = FeatureMap::new();
    feats.insert("bias".to_string(), 1.0);
    feats.insert("isolation".to_string(), isolation);
    feats.insert("depth_norm".to_string(), state.depth as f64 / max_depth.max(1) as f64);
    feats.insert("is_solved".to_string(), is_solved);
    feats.insert(
        "both_sides_have_x".to_string(),
        if lhs_has_x && rhs_has_x { 1.0 } else { 0.0 },
    );
    feats
}
