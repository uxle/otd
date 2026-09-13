//! Phase 009 — Self-evolution loop (design doc 2.9) (Rust port of
//! `python/selfplay/self_evolution.py`).
//!
//! Each round: use the *current* PRM to guide search over a batch of
//! problems, keep only verifier-confirmed trajectories, fold that data into
//! the training set, retrain the PRM, and re-evaluate on a FIXED held-out
//! set at a FIXED low budget (so improvement, if any, is measured on an
//! apples-to-apples basis across rounds, not on the ever-easier problems
//! the PRM picks for itself).

use rand::rngs::StdRng;
use rand::SeedableRng;
use reasoning_prm::{extract_training_examples, FeatureMap, LinearPRM, PrmGuidedPolicy};
use reasoning_search::{make_initial_state, Mcts, NtState, NumberTargetDomain};
use std::rc::Rc;

/// One problem: (domain, starting numbers, max search depth) — Python's
/// `Tuple[NumberTargetDomain, List[float], int]`.
pub type Problem = (NumberTargetDomain, Vec<f64>, usize);

/// Python `_depth_fn(max_depth)`: a state's depth = merges already done
/// (`max_depth - len(state) + 1`).
fn depth_fn(max_depth: usize) -> Rc<dyn Fn(&NtState) -> usize> {
    Rc::new(move |state: &NtState| max_depth - state.len() + 1)
}

/// Python `evaluate(prm, eval_problems, budget, seed)` — how many of the
/// held-out problems the PRM-guided search solves at the fixed budget.
pub fn evaluate(prm: &LinearPRM, eval_problems: &[Problem], budget: usize, seed: u64) -> usize {
    let mut solved = 0usize;
    for (i, (domain, nums, max_depth)) in eval_problems.iter().enumerate() {
        let state = make_initial_state(nums);
        let policy = PrmGuidedPolicy::new(
            prm.clone(),
            domain.clone(),
            domain.target,
            *max_depth,
            depth_fn(*max_depth),
            0.15, // Python: epsilon=0.15 for evaluation
        );
        let mut mcts = Mcts::new(domain.clone(), *max_depth, seed + i as u64)
            .with_rollout_policy(Box::new(policy));
        let result = mcts.search(state, budget);
        solved += result.found_verified_solution as usize;
    }
    solved
}

/// Python `run_self_evolution(rounds, round_problem_fn, eval_problems,
/// round_budget=200, eval_budget=40, epochs=60, lr=0.3, seed=0) ->
/// Tuple[LinearPRM, List[int]]`: the PRM and the per-round solved-count
/// history (index 0 = the untrained round-0 baseline).
#[allow(clippy::too_many_arguments)]
pub fn run_self_evolution(
    rounds: usize,
    round_problem_fn: &dyn Fn(usize) -> Vec<Problem>,
    eval_problems: &[Problem],
    round_budget: usize,
    eval_budget: usize,
    epochs: usize,
    lr: f64,
    seed: u64,
) -> (LinearPRM, Vec<usize>) {
    // untrained: all weights 0 -> predicts 0.5 everywhere, ~uniform policy
    let mut prm = LinearPRM::new();
    let mut all_examples: Vec<(FeatureMap, f64)> = Vec::new();
    let mut history = vec![evaluate(&prm, eval_problems, eval_budget, 9000)]; // round-0 baseline

    for round_idx in 0..rounds {
        let problems = round_problem_fn(round_idx);
        for (i, (domain, nums, max_depth)) in problems.iter().enumerate() {
            let state = make_initial_state(nums);
            let policy = PrmGuidedPolicy::new(
                prm.clone(),
                domain.clone(),
                domain.target,
                *max_depth,
                depth_fn(*max_depth),
                0.2, // Python: epsilon=0.2 during data generation
            );
            let mut mcts = Mcts::new(
                domain.clone(),
                *max_depth,
                seed + round_idx as u64 * 1000 + i as u64,
            )
            .with_rollout_policy(Box::new(policy));
            let result = mcts.search(state, round_budget);
            let examples = extract_training_examples(&result, domain.target, *max_depth, 2);
            all_examples.extend(examples);
        }

        let mut new_prm = LinearPRM::new();
        let mut rng = StdRng::seed_from_u64(seed + round_idx as u64);
        new_prm.train(&all_examples, epochs, lr, &mut rng);
        prm = new_prm;

        let score = evaluate(&prm, eval_problems, eval_budget, 9000);
        history.push(score);
    }

    (prm, history)
}
