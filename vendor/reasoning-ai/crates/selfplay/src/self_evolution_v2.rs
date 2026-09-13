//! Phase 013 — Self-evolution retry, applying Phase 009/011 lessons (Rust
//! port of `python/selfplay/self_evolution_v2.py`).
//!
//! Phase 009's flat result was diagnosed as: 3-number problems terminate in
//! 1-2 merges, leaving no room for rollout guidance to matter. This retry
//! uses 4-5 number problems (more merges = more room for guidance) and folds
//! in Phase 011's root-oversampling fix so the PRM used for guidance is
//! actually calibrated at the states search visits most.

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use reasoning_prm::{
    collect_root_examples, extract_training_examples, FeatureMap, LinearPRM, PrmGuidedPolicy,
};
use reasoning_search::{make_initial_state, Mcts, NtState, NumberTargetDomain};
use std::rc::Rc;

use crate::self_evolution::Problem;

/// Python imports `_OPS` from `search.number_target_domain` (a private
/// detail there); the Rust port keeps the const private, so the same list
/// lives here.
const _OPS: [char; 4] = ['+', '-', '*', '/'];

/// Python `_depth_fn(max_depth)`.
fn depth_fn(max_depth: usize) -> Rc<dyn Fn(&NtState) -> usize> {
    Rc::new(move |state: &NtState| max_depth - state.len() + 1)
}

/// Python `_make_guaranteed_solvable(rng, count, max_depth)`: construct a
/// problem by combining `count` random numbers with random valid operators
/// down to a single value, then using that value as the target. Guarantees
/// at least one solution exists (the path just taken), unlike picking an
/// arbitrary target and hoping it's reachable.
fn make_guaranteed_solvable(
    rng: &mut StdRng,
    count: usize,
    max_depth: usize,
) -> (NumberTargetDomain, Vec<f64>, usize) {
    let nums: Vec<f64> = (0..count).map(|_| rng.gen_range(1..=9) as f64).collect();
    let mut values = nums.clone();
    while values.len() > 1 {
        let n = values.len();
        // Python rng.sample(range(n), 2): two distinct indices
        let i = rng.gen_range(0..n);
        let j = (i + rng.gen_range(1..n)) % n;
        let mut op = _OPS[rng.gen_range(0.._OPS.len())];
        let (a, b) = (values[i], values[j]);
        if op == '/' && b == 0.0 {
            op = '+';
        }
        let v = match op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            _ => {
                if b != 0.0 {
                    a / b
                } else {
                    a + b
                }
            }
        };
        let mut remaining: Vec<f64> = values
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != i && *k != j)
            .map(|(_, x)| *x)
            .collect();
        remaining.push(v);
        values = remaining;
    }
    let target = values[0];
    (NumberTargetDomain::new(target), nums, max_depth)
}

/// Python `make_round_problems(round_idx, n=8, seed_base=200)`: count=4
/// (deeper than Phase 009's 3 -> more merges -> more room for guidance),
/// max_depth=6.
pub fn make_round_problems(round_idx: usize, n: usize, seed_base: u64) -> Vec<Problem> {
    let mut rng = StdRng::seed_from_u64(seed_base + round_idx as u64);
    let count = 4;
    (0..n)
        .map(|_| make_guaranteed_solvable(&mut rng, count, 6))
        .collect()
}

/// Python `make_eval_set(n=20, seed=777)`: guaranteed-solvable 4-number
/// problems at max_depth 6.
pub fn make_eval_set(n: usize, seed: u64) -> Vec<Problem> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| make_guaranteed_solvable(&mut rng, 4, 6))
        .collect()
}

/// Python `evaluate(prm, eval_problems, budget, seed)`.
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
            0.15,
        );
        let mut mcts = Mcts::new(domain.clone(), *max_depth, seed + i as u64)
            .with_rollout_policy(Box::new(policy));
        let result = mcts.search(state, budget);
        solved += result.found_verified_solution as usize;
    }
    solved
}

/// Python `run_self_evolution_v2(rounds, round_problem_fn, eval_problems,
/// round_budget=250, eval_budget=60, root_oversample=8, epochs=60, lr=0.3,
/// seed=0)`: like Phase 009's loop but with 4-number problems and the
/// Phase 011 calibration fix — root (depth=0) examples are harvested from
/// a separate search pass and oversampled `root_oversample`× into the
/// training set, so the PRM is calibrated at the states search actually
/// visits most.
#[allow(clippy::too_many_arguments)]
pub fn run_self_evolution_v2(
    rounds: usize,
    round_problem_fn: &dyn Fn(usize) -> Vec<Problem>,
    eval_problems: &[Problem],
    round_budget: usize,
    eval_budget: usize,
    root_oversample: usize,
    epochs: usize,
    lr: f64,
    seed: u64,
) -> (LinearPRM, Vec<usize>) {
    let mut prm = LinearPRM::new();
    let mut all_tree_examples: Vec<(FeatureMap, f64)> = Vec::new();
    let mut all_root_examples: Vec<(FeatureMap, f64)> = Vec::new();
    let mut history = vec![evaluate(&prm, eval_problems, eval_budget, 9500)];

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
                0.2,
            );
            let mut mcts = Mcts::new(
                domain.clone(),
                *max_depth,
                seed + round_idx as u64 * 1000 + i as u64,
            )
            .with_rollout_policy(Box::new(policy));
            let result = mcts.search(state, round_budget);
            all_tree_examples.extend(extract_training_examples(
                &result,
                domain.target,
                *max_depth,
                2,
            ));
        }

        // Phase 011 calibration fix: separate root-outcome harvest, then
        // Python's `all_root_examples.extend(root_examples * root_oversample)`
        let root_examples =
            collect_root_examples(&problems, round_budget, seed + round_idx as u64 * 1000 + 500);
        for _ in 0..root_oversample {
            all_root_examples.extend(root_examples.iter().cloned());
        }

        let mut new_prm = LinearPRM::new();
        let mut rng = StdRng::seed_from_u64(seed + round_idx as u64);
        let mut train_set: Vec<(FeatureMap, f64)> = all_tree_examples.clone();
        train_set.extend(all_root_examples.iter().cloned());
        new_prm.train(&train_set, epochs, lr, &mut rng);
        prm = new_prm;
        history.push(evaluate(&prm, eval_problems, eval_budget, 9500));
    }

    (prm, history)
}
