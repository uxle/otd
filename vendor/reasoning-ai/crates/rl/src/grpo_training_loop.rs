//! Phase 067 — Multi-round GRPO training loop, with honest before/after
//! evaluation (Promot section 29 / project convention: report what the
//! experiment actually found, not what we hoped it would find).
//!
//! Mirrors the shape of selfplay/self_evolution_v2.run_self_evolution_v2,
//! but the thing being trained each round is the Phase 061 policy's own
//! weights via GRPO (Phase 066), not a PRM via supervised regression.

use std::rc::Rc;

use reasoning_search::{Mcts, NtState};
use serde::{Deserialize, Serialize};

use crate::grpo_self_play::run_grpo_round;
use crate::policy::SoftmaxPolicy;
use crate::trainable_rollout::TrainableRollout;

use crate::Problem;

/// Python `_depth_fn(max_depth)`: depth = max_depth - len(state) + 1.
/// (Saturates at 0 for the impossible len > max_depth + 1 case, where
/// Python would produce a negative int.)
pub fn _depth_fn(max_depth: usize) -> Rc<dyn Fn(&NtState) -> usize> {
    Rc::new(move |state: &NtState| {
        (max_depth as i64 - state.len() as i64 + 1).max(0) as usize
    })
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct TrainingHistory {
    pub round_solve_rates: Vec<f64>,
    pub round_mean_rewards: Vec<f64>,
    pub round_grad_norms: Vec<f64>,
    pub eval_solve_rate_before: f64,
    pub eval_solve_rate_after: f64,
}

/// Held-out evaluation: run MCTS *guided by the trained policy* (via the
/// Phase 065 bridge) on a fixed eval set, report solve rate. Using MCTS
/// (not plain policy sampling) for eval gives the policy the same search
/// budget the PRM-guided baseline gets in self_evolution_v2, so the two
/// are comparable on equal footing (Phase 069).
///
/// Python created two separately-seeded `random.Random(seed + i)`
/// objects (one captured by the rollout closure, one for MCTS); this
/// port drives both from the MCTS-owned RNG stream seeded with
/// `seed + i` — same information flow, different (uncorrelated) stream.
pub fn evaluate_policy(
    policy: &SoftmaxPolicy,
    eval_problems: &[Problem],
    num_simulations: usize,
    seed: u64,
) -> f64 {
    let mut solved = 0usize;
    for (i, (domain, initial_state, max_depth)) in eval_problems.iter().enumerate() {
        let depth_fn = _depth_fn(*max_depth);
        let rollout = TrainableRollout::new(
            policy.clone(),
            domain.clone(),
            domain.target,
            *max_depth,
            depth_fn,
            0.1,
        );
        let mut mcts = Mcts::new(domain.clone(), *max_depth, seed + i as u64)
            .with_rollout_policy(Box::new(rollout));
        let result = mcts.search(initial_state.clone(), num_simulations);
        solved += result.found_verified_solution as usize;
    }
    solved as f64 / eval_problems.len() as f64
}

/// Runs `rounds` GRPO rounds, evaluating before round 0 and after the
/// final round. Returns `(final_policy, history)` — history is populated
/// with REAL measured numbers, whatever they turn out to be.
///
/// Python defaults: n_samples_per_problem=6, lr=0.2, epsilon=0.2,
/// eval_num_simulations=60, seed=0.
#[allow(clippy::too_many_arguments)]
pub fn run_grpo_training(
    rounds: usize,
    round_problem_fn: &dyn Fn(usize) -> Vec<Problem>,
    eval_problems: &[Problem],
    n_samples_per_problem: usize,
    lr: f64,
    epsilon: f64,
    eval_num_simulations: usize,
    seed: u64,
) -> Result<(SoftmaxPolicy, TrainingHistory), String> {
    let mut policy = SoftmaxPolicy::new();
    let mut history = TrainingHistory::default();
    history.eval_solve_rate_before =
        evaluate_policy(&policy, eval_problems, eval_num_simulations, 9000);

    for round_idx in 0..rounds {
        let problems = round_problem_fn(round_idx);
        let (new_policy, diag) = run_grpo_round(
            &policy,
            &problems,
            &_depth_fn,
            n_samples_per_problem,
            lr,
            epsilon,
            seed + round_idx as u64 * 137,
        )?;
        policy = new_policy;
        history.round_solve_rates.push(diag.solve_rate);
        history.round_mean_rewards.push(diag.mean_reward);
        history.round_grad_norms.push(diag.grad_norm);
    }

    history.eval_solve_rate_after =
        evaluate_policy(&policy, eval_problems, eval_num_simulations, 9000);
    Ok((policy, history))
}
