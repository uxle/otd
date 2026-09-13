//! Phase 048 — Graduated confidence (design doc section 23: confidence
//! scoring, calibration) (Rust port of
//! `python/uncertainty/graded_confidence.py`).
//!
//! Phase 024's Answer used binary confidence (1.0 verified, 0.0 abstained).
//! That's honest but throws away information: a verified answer found after
//! exhausting most of the search budget is less "comfortably" verified than
//! one found immediately. This adds a graduated confidence signal from the
//! calibrated PRM (Phase 011) ALONGSIDE the binary verified flag — the
//! binary flag still gates final_answer() (never weakened), confidence is
//! now just richer metadata about HOW verified, not a replacement for
//! verification itself.

use reasoning_prm::{extract_features, LinearPRM};
use reasoning_search::NtState;

#[derive(Debug, Clone, PartialEq)]
pub struct GradedConfidence {
    pub verified: bool,
    /// 1.0 or 0.0, same as Phase 024 — never removed.
    pub binary_confidence: f64,
    /// PRM's own predicted P(success) at the root, informational.
    pub graduated_confidence: Option<f64>,
    /// fraction of budget consumed before finding a solution.
    pub search_effort_used: Option<f64>,
}

/// Python `grade_number_target_confidence(prm, root_state, target,
/// max_depth, verified, nodes_expanded, budget)`.
///
/// The binary `verified` flag (from the real verifier) is passed in and
/// NEVER overridden by the PRM's opinion — graduated_confidence is purely
/// additional context, and a caller that only checks `verified` behaves
/// exactly as before. (Python wrapped the feature extraction/prediction in
/// try/except -> None; neither step can fail here, so the value is always
/// present.)
pub fn grade_number_target_confidence(
    prm: &LinearPRM,
    root_state: &NtState,
    target: f64,
    max_depth: usize,
    verified: bool,
    nodes_expanded: usize,
    budget: usize,
) -> GradedConfidence {
    let feats = extract_features(root_state, target, max_depth, 0);
    let graduated = Some(prm.predict(&feats));

    // Python: min(nodes_expanded / budget, 1.0) if budget > 0 else None
    let effort = if budget > 0 {
        Some(((nodes_expanded as f64) / (budget as f64)).min(1.0))
    } else {
        None
    };

    GradedConfidence {
        verified,
        binary_confidence: if verified { 1.0 } else { 0.0 },
        graduated_confidence: graduated,
        search_effort_used: effort,
    }
}
