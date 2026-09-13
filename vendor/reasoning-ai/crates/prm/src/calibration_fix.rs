//! Phase 011 — Fix PRM calibration (root-depth oversampling) (Rust port of
//! `python/prm/calibration_fix.py`).
//!
//! Phase 010 found the PRM badly miscalibrated at root states (mean gap
//! 0.66): only ~2% of its Phase-004 training examples were near depth=0,
//! because tree-walk extraction naturally yields far more deep-node examples
//! than root examples (one root per tree, many descendants). Fix: run
//! searches specifically to harvest (root_features, final_outcome) pairs and
//! oversample them into the training set so depth=0 is no longer a
//! practically-unseen region for the model.

use crate::data_extraction::extract_training_examples;
use crate::features::{extract_features, FeatureMap};
use reasoning_search::{make_initial_state, Mcts, NumberTargetDomain};

pub fn collect_root_examples(
    problems: &[(NumberTargetDomain, Vec<f64>, usize)],
    budget: usize,
    seed: u64,
) -> Vec<(FeatureMap, f64)> {
    let mut examples = Vec::new();
    for (i, (domain, nums, max_depth)) in problems.iter().enumerate() {
        let state = make_initial_state(nums);
        let mut mcts = Mcts::new(domain.clone(), *max_depth, seed + i as u64);
        let result = mcts.search(state.clone(), budget);
        let feats = extract_features(&state, domain.target, *max_depth, 0);
        let outcome = if result.found_verified_solution { 1.0 } else { 0.0 };
        examples.push((feats, outcome));
    }
    examples
}

pub fn build_calibrated_training_set(
    tree_problems: &[(NumberTargetDomain, Vec<f64>, usize)],
    root_oversample_factor: usize,
    budget: usize,
    seed: u64,
) -> Vec<(FeatureMap, f64)> {
    // Combine the original deep-node tree-walk examples with oversampled
    // root-outcome examples, so root states are no longer <5% of the data.
    let mut tree_examples: Vec<(FeatureMap, f64)> = Vec::new();
    for (i, (domain, nums, max_depth)) in tree_problems.iter().enumerate() {
        let state = make_initial_state(nums);
        let mut mcts = Mcts::new(domain.clone(), *max_depth, seed + i as u64);
        let result = mcts.search(state, budget);
        // Python default min_visits=2
        tree_examples.extend(extract_training_examples(&result, domain.target, *max_depth, 2));
    }

    let root_examples = collect_root_examples(tree_problems, budget, seed + 5000);
    let mut out = tree_examples;
    // Python `root_examples * root_oversample_factor`
    for _ in 0..root_oversample_factor {
        out.extend(root_examples.iter().cloned());
    }
    out
}
