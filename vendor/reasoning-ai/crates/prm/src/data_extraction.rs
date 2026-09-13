//! Walks a completed MCTS tree (Phase 003) and extracts (features, Q-value)
//! training pairs for the PRM (Rust port of
//! `python/prm/data_extraction.py`).
//!
//! The label for every node is its own backpropagated `value` — an average
//! of *verifier-scored* rollouts, so every label ultimately traces back to
//! the symbolic verifier, never to a human annotation or the policy's own
//! confidence.
//!
//! The Rust MCTS tree is an arena (`Vec<TreeNode>`) with index-based
//! children, so the walk is over `tree[node_idx]` + its child indices.

use crate::features::{extract_features, FeatureMap};
use reasoning_search::{NtAction, NtState, SearchResult, TreeNode};

/// Recursive walk of the arena tree starting at `node_idx`.
pub fn extract_training_examples_from_arena(
    tree: &[TreeNode<NtState, NtAction>],
    node_idx: usize,
    target: f64,
    max_depth: usize,
    min_visits: usize,
) -> Vec<(FeatureMap, f64)> {
    let mut examples = Vec::new();
    walk(tree, node_idx, target, max_depth, min_visits, &mut examples);
    examples
}

fn walk(
    tree: &[TreeNode<NtState, NtAction>],
    node_idx: usize,
    target: f64,
    max_depth: usize,
    min_visits: usize,
    out: &mut Vec<(FeatureMap, f64)>,
) {
    let node = &tree[node_idx];
    if node.visit_count >= min_visits {
        let feats = extract_features(&node.state, target, max_depth, node.depth);
        out.push((feats, node.value()));
    }
    for &child in &node.children {
        walk(tree, child, target, max_depth, min_visits, out);
    }
}

/// Thin wrapper over a completed search result — Python's
/// `extract_training_examples(result.root, target, max_depth, min_visits=2)`
/// walked from the root; here the arena + root index live on the result.
pub fn extract_training_examples(
    result: &SearchResult<NtState, NtAction>,
    target: f64,
    max_depth: usize,
    min_visits: usize,
) -> Vec<(FeatureMap, f64)> {
    extract_training_examples_from_arena(&result.tree, result.root, target, max_depth, min_visits)
}
