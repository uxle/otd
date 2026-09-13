//! Phase 003 — Monte Carlo Tree Search over reasoning steps (Rust port of
//! `search/mcts.py`).
//!
//! Domain-agnostic: MCTS knows nothing about math specifically, only the
//! `Domain` contract. The tree is stored as an arena (`Vec<TreeNode>`) with
//! index-based parent/child links instead of Python's parent pointers; the
//! arena is returned in `SearchResult` so later phases (PRM training,
//! MCTS-sourced episode generation) can walk it exactly like the Python
//! `result.root` tree walk.

use crate::domain::{Domain, RolloutPolicy, UniformRollout};
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

#[derive(Debug, Clone)]
pub struct TreeNode<S, A> {
    pub state: S,
    pub parent: Option<usize>,
    pub action_from_parent: Option<A>,
    pub depth: usize,
    /// P(s,a) — uniform until a policy model exists.
    pub prior: f64,
    pub children: Vec<usize>,
    pub visit_count: usize,
    /// Sum of backpropagated rewards.
    pub value_sum: f64,
    /// Set only for terminal nodes.
    pub reward: Option<f64>,
    /// visit_count-derived confidence, filled in during search.
    pub confidence: f64,
}

impl<S, A> TreeNode<S, A> {
    pub fn value(&self) -> f64 {
        if self.visit_count == 0 {
            0.0
        } else {
            self.value_sum / self.visit_count as f64
        }
    }

    pub fn is_leaf(&self) -> bool {
        self.children.is_empty()
    }
}

/// UCB(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))  (design doc 2.8)
pub fn ucb_score<S, A>(child: &TreeNode<S, A>, parent_visits: usize, c: f64) -> f64 {
    let exploit = child.value();
    let explore =
        c * child.prior * (parent_visits as f64).sqrt() / (1.0 + child.visit_count as f64);
    exploit + explore
}

#[derive(Debug, Clone)]
pub struct SearchResult<S: Clone, A: Clone> {
    pub found_verified_solution: bool,
    pub best_terminal_state: Option<S>,
    pub best_reward: f64,
    pub nodes_expanded: usize,
    /// Sequence of actions to the best verified terminal, if solved.
    pub verified_path: Option<Vec<A>>,
    /// Tree arena (index-based; `root` is the root's index) — exposed so
    /// later phases (PRM training, MCTS episode sources) can walk the tree.
    pub tree: Vec<TreeNode<S, A>>,
    pub root: usize,
}

impl<S: Clone, A: Clone> SearchResult<S, A> {
    pub fn node(&self, idx: usize) -> &TreeNode<S, A> {
        &self.tree[idx]
    }
}

pub struct Mcts<D: Domain> {
    pub domain: D,
    pub max_depth: usize,
    rng: StdRng,
    rollout: Box<dyn RolloutPolicy<D::State, D::Action>>,
}

impl<D: Domain> Mcts<D> {
    /// Python parity: `MCTS(domain, max_depth=..., rng=random.Random(seed))`
    pub fn new(domain: D, max_depth: usize, seed: u64) -> Self {
        Mcts {
            domain,
            max_depth,
            rng: StdRng::seed_from_u64(seed),
            rollout: Box::new(UniformRollout),
        }
    }

    /// Attach a custom rollout policy (PRM-guided / trainable policies).
    pub fn with_rollout_policy(
        mut self,
        policy: Box<dyn RolloutPolicy<D::State, D::Action>>,
    ) -> Self {
        self.rollout = policy;
        self
    }

    pub fn search(
        &mut self,
        root_state: D::State,
        num_simulations: usize,
    ) -> SearchResult<D::State, D::Action> {
        let mut tree: Vec<TreeNode<D::State, D::Action>> = vec![TreeNode {
            state: root_state.clone(),
            parent: None,
            action_from_parent: None,
            depth: 0,
            prior: 1.0,
            children: Vec::new(),
            visit_count: 0,
            value_sum: 0.0,
            reward: None,
            confidence: 0.0,
        }];
        let root_idx = 0usize;

        let mut best_reward = -1.0f64;
        let mut best_node_idx: Option<usize> = None;
        let mut best_path: Option<Vec<D::Action>> = None;
        let mut nodes_expanded = 0usize;

        for _ in 0..num_simulations {
            let mut node_idx = root_idx;

            // 1. Selection: descend via UCB until a leaf
            while !tree[node_idx].is_leaf() {
                let pv = tree[node_idx].visit_count;
                let children = tree[node_idx].children.clone();
                // Python `max(children, key=ucb_score)`: ties break toward the
                // FIRST child (Rust's `max_by` returns the last max, which
                // reverses the unvisited-children sweep order the Python
                // search's outcomes depend on).
                let mut best = children[0];
                let mut best_score = ucb_score(&tree[best], pv, 1.4);
                for &c in &children[1..] {
                    let s = ucb_score(&tree[c], pv, 1.4);
                    if s > best_score {
                        best = c;
                        best_score = s;
                    }
                }
                node_idx = best;
            }

            // 2. Expansion (if not terminal and not at depth limit)
            if !self.domain.is_terminal(&tree[node_idx].state)
                && tree[node_idx].depth < self.max_depth
            {
                if tree[node_idx].is_leaf() {
                    let parent = node_idx;
                    let state = tree[parent].state.clone();
                    let depth = tree[parent].depth + 1;
                    let actions = self.domain.legal_actions(&state);
                    let children_before = tree[parent].children.len();
                    for action in actions {
                        let child_state = self.domain.apply(&state, &action);
                        let child_idx = tree.len();
                        tree[parent].children.push(child_idx);
                        tree.push(TreeNode {
                            state: child_state,
                            parent: Some(parent),
                            action_from_parent: Some(action),
                            depth,
                            prior: 1.0,
                            children: Vec::new(),
                            visit_count: 0,
                            value_sum: 0.0,
                            reward: None,
                            confidence: 0.0,
                        });
                    }
                    nodes_expanded += tree[parent].children.len() - children_before;
                }
                if !tree[node_idx].children.is_empty() {
                    let kids = tree[node_idx].children.clone();
                    node_idx = *kids
                        .choose(&mut self.rng)
                        .expect("children non-empty (checked)");
                }
            }

            // 3. Simulation (rollout to terminal, scored by verifier)
            let is_term = self.domain.is_terminal(&tree[node_idx].state);
            let reward = if is_term {
                self.domain.terminal_reward(&tree[node_idx].state)
            } else {
                self.rollout_from(node_idx, &tree)
            };
            if is_term {
                tree[node_idx].reward = Some(reward);
            }

            // 4. Backpropagation
            let mut cur = Some(node_idx);
            while let Some(i) = cur {
                tree[i].visit_count += 1;
                tree[i].value_sum += reward;
                tree[i].confidence = tree[i].visit_count as f64 / (tree[i].visit_count as f64 + 5.0);
                cur = tree[i].parent;
            }

            if is_term && reward > best_reward {
                best_reward = reward;
                best_node_idx = Some(node_idx);
                let mut path: Vec<D::Action> = Vec::new();
                let mut i = node_idx;
                while let Some(p) = tree[i].parent {
                    path.push(tree[i].action_from_parent.clone().expect("child has action"));
                    i = p;
                }
                path.reverse();
                best_path = Some(path);
            }
        }

        let best_state = best_node_idx.map(|i| tree[i].state.clone());
        let found = best_reward >= 0.999; // verifier returns ~1.0 for exact match
        let best_reward = best_reward.max(0.0);
        SearchResult {
            found_verified_solution: found,
            best_terminal_state: best_state,
            best_reward,
            nodes_expanded,
            verified_path: if found { best_path } else { None },
            tree,
            root: root_idx,
        }
    }

    /// Random rollout to a terminal state (or max_depth), scored by the
    /// verifier via terminal_reward. Returns 0.0 for non-terminal cutoff —
    /// never fabricates a reward for an unfinished trajectory.
    fn rollout_from(&mut self, start: usize, tree: &[TreeNode<D::State, D::Action>]) -> f64 {
        let mut state = tree[start].state.clone();
        let mut depth = tree[start].depth;
        while !self.domain.is_terminal(&state) && depth < self.max_depth {
            let legal = self.domain.legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let action = self.rollout.choose(&mut self.rng, &state, &legal);
            state = self.domain.apply(&state, &action);
            depth += 1;
        }
        if self.domain.is_terminal(&state) {
            self.domain.terminal_reward(&state)
        } else {
            0.0
        }
    }
}
