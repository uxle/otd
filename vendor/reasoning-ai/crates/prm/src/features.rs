//! Feature extraction for the NumberTargetDomain (Phase 003), used to train
//! a Process Reward Model on partial-solution states (Rust port of
//! `python/prm/features.py`).
//!
//! This is intentionally hand-engineered and domain-specific for now — the
//! neural policy model in a later phase learns features directly from the
//! expression text instead of needing them handed to it.

use reasoning_search::NtState;
use std::collections::HashMap;

pub type FeatureMap = HashMap<String, f64>;

/// The five feature keys `extract_features` produces.
pub const FEATURE_KEYS: [&str; 5] = [
    "bias",
    "num_remaining",
    "min_diff_to_target_norm",
    "depth_norm",
    "has_exact_match",
];

pub fn extract_features(
    state: &NtState,
    target: f64,
    max_depth: usize,
    depth: usize,
) -> FeatureMap {
    let values: Vec<f64> = state.iter().map(|(v, _)| *v).collect();
    let n = values.len();
    let diffs: Vec<f64> = values.iter().map(|v| (v - target).abs()).collect();
    let min_diff = if diffs.is_empty() {
        target.abs()
    } else {
        diffs.into_iter().fold(f64::INFINITY, f64::min)
    };
    // normalize the "closeness" feature so it's roughly in [0,1] regardless
    // of target magnitude
    let denom = target.abs() + 1.0;
    let mut feats = HashMap::with_capacity(FEATURE_KEYS.len());
    feats.insert("bias".to_string(), 1.0);
    feats.insert("num_remaining".to_string(), n as f64);
    // clip outliers (Python `min(min_diff / denom, 5.0)`)
    feats.insert("min_diff_to_target_norm".to_string(), (min_diff / denom).min(5.0));
    // Python true division `depth / max(max_depth, 1)`
    feats.insert("depth_norm".to_string(), depth as f64 / max_depth.max(1) as f64);
    feats.insert(
        "has_exact_match".to_string(),
        if min_diff < 1e-9 { 1.0 } else { 0.0 },
    );
    feats
}
