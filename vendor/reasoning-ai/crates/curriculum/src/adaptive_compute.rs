//! Phase 008 — Adaptive compute (design doc section 13) (Rust port of
//! python/curriculum/adaptive_compute.py).
//!
//! Uses the PRM's own prediction at the root state as a cheap difficulty
//! signal: if the PRM already thinks the starting position looks promising,
//! spend less search; if it looks hopeless/uncertain, spend more. This is
//! deliberately simple (no separate "difficulty estimator" network) — the
//! PRM we already trained is repurposed rather than adding a new component
//! that would itself need training data.

use reasoning_prm::{extract_features, LinearPRM};
use reasoning_search::NtState;

/// Python `estimate_budget(prm, root_state, target, max_depth, min_sims=60,
/// max_sims=400)`.
///
/// Lower PRM confidence at the root -> bigger budget. Confidence near 0.5
/// (maximally uncertain) gets the largest budget; confidence near 0 or 1
/// (PRM thinks it clearly knows the outcome) gets the smallest.
pub fn estimate_budget(
    prm: &LinearPRM,
    root_state: &NtState,
    target: f64,
    max_depth: usize,
    min_sims: usize,
    max_sims: usize,
) -> usize {
    let feats = extract_features(root_state, target, max_depth, 0);
    let p = prm.predict(&feats);
    // 0 at p=0 or p=1, 1 at p=0.5
    let uncertainty = 1.0 - (2.0 * p - 1.0).abs();
    // Python int() truncation
    let budget = (min_sims as f64 + uncertainty * (max_sims - min_sims) as f64) as usize;
    budget.max(min_sims).min(max_sims)
}
