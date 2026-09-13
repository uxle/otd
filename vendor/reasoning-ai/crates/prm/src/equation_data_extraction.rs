//! Rust port of `python/prm/equation_data_extraction.py`: walks a completed
//! MCTS tree over the linear-equation domain and extracts (equation
//! features, Q-value) training pairs for the PRM.
//!
//! (The Python module has no docstring; it mirrors data_extraction.py with
//! the equation feature extractor.)

use crate::equation_features::extract_equation_features;
use crate::features::FeatureMap;
use reasoning_search::{EqState, LinEqAction, SearchResult, TreeNode};

/// Recursive walk of the arena tree starting at `node_idx`.
pub fn extract_equation_training_examples_from_arena(
    tree: &[TreeNode<EqState, LinEqAction>],
    node_idx: usize,
    max_depth: usize,
    min_visits: usize,
) -> Vec<(FeatureMap, f64)> {
    let mut examples = Vec::new();
    walk(tree, node_idx, max_depth, min_visits, &mut examples);
    examples
}

fn walk(
    tree: &[TreeNode<EqState, LinEqAction>],
    node_idx: usize,
    max_depth: usize,
    min_visits: usize,
    out: &mut Vec<(FeatureMap, f64)>,
) {
    let node = &tree[node_idx];
    if node.visit_count >= min_visits {
        let feats = extract_equation_features(&node.state, max_depth);
        out.push((feats, node.value()));
    }
    for &child in &node.children {
        walk(tree, child, max_depth, min_visits, out);
    }
}

/// Thin wrapper over a completed search result (Python
/// `extract_equation_training_examples(result.root, max_depth,
/// min_visits=2)`).
pub fn extract_equation_training_examples(
    result: &SearchResult<EqState, LinEqAction>,
    max_depth: usize,
    min_visits: usize,
) -> Vec<(FeatureMap, f64)> {
    extract_equation_training_examples_from_arena(&result.tree, result.root, max_depth, min_visits)
}
