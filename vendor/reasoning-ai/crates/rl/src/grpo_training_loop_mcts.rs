//! Phase 073 — GRPO self-play round + multi-round training loop using
//! Phase 71's MCTS-sourced episodes instead of Phase 66/67's raw policy
//! sampling. Same GRPO update math (rl/grpo_update) — only the episode
//! *source* changes, which is exactly the variable Phase 70's diagnosis
//! isolated.

use std::rc::Rc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::grpo_training_loop::{_depth_fn, evaluate_policy, TrainingHistory};
use crate::grpo_update::{grpo_weight_update, GRPOEpisode, GRPOEpisodeStep};
use crate::mcts_episode_source::sample_episode_via_mcts;
use crate::policy::SoftmaxPolicy;

use crate::Problem;
use reasoning_search::NtState;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MCTSRoundDiagnostics {
    pub mean_reward: f64,
    pub reward_std: f64,
    pub solve_rate: f64,
    pub grad_norm: f64,
    pub fraction_gradient_zeroed_by_clip: f64,
    pub fraction_groups_with_nonzero_variance: f64,
}

/// One GRPO round whose episodes come from MCTS (Phase 071).
///
/// Python defaults: depth_fn_factory=_depth_fn,
/// n_samples_per_problem=6, train_num_simulations=150, lr=0.2,
/// epsilon=0.2, seed=0.
#[allow(clippy::too_many_arguments)]
pub fn run_grpo_round_mcts(
    policy: &SoftmaxPolicy,
    problems: &[Problem],
    depth_fn_factory: &dyn Fn(usize) -> Rc<dyn Fn(&NtState) -> usize>,
    n_samples_per_problem: usize,
    train_num_simulations: usize,
    lr: f64,
    epsilon: f64,
    seed: u64,
) -> Result<(SoftmaxPolicy, MCTSRoundDiagnostics), String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut policy = policy.clone();
    let mut total_reward = 0.0f64;
    let mut total_solved = 0usize;
    let mut total_episodes = 0usize;
    let mut groups_with_variance = 0usize;
    let mut grad_norm_accum = 0.0f64;
    let mut zeroed_accum = 0.0f64;
    let mut n_groups = 0usize;

    for (domain, initial_state, max_depth) in problems {
        let depth_fn = depth_fn_factory(*max_depth);
        let mut episodes: Vec<GRPOEpisode> = Vec::new();
        let mut rewards: Vec<f64> = Vec::new();
        for _ in 0..n_samples_per_problem {
            // Python: rng=random.Random(rng.randint(0, 10**9)).
            let mut ep_rng = StdRng::seed_from_u64(rng.gen_range(0..=1_000_000_000u64));
            let (ep, reward, passed) = sample_episode_via_mcts(
                &policy,
                domain,
                initial_state,
                domain.target,
                *max_depth,
                depth_fn.clone(),
                &mut ep_rng,
                train_num_simulations,
                0.1,
            );
            // skip degenerate zero-step episodes (nothing to update on)
            let steps: Vec<GRPOEpisodeStep> = ep
                .states
                .iter()
                .zip(ep.actions.iter())
                .zip(ep.legals.iter().zip(ep.log_probs.iter()))
                .filter(|(_, (l, _))| !l.is_empty())
                .map(|((s, a), (l, lp))| GRPOEpisodeStep {
                    state: s.clone(),
                    action: a.clone(),
                    legal: l.clone(),
                    old_log_prob: *lp,
                })
                .collect();
            if !steps.is_empty() {
                episodes.push(GRPOEpisode {
                    steps,
                    reward,
                });
                rewards.push(reward);
            }
            total_reward += reward;
            total_solved += passed as usize;
            total_episodes += 1;
        }

        if episodes.len() >= 2 {
            let max_r = rewards.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let min_r = rewards.iter().cloned().fold(f64::INFINITY, f64::min);
            if max_r - min_r > 1e-9 {
                groups_with_variance += 1;
            }
            let (new_policy, diag) = grpo_weight_update(
                &policy,
                &episodes,
                domain,
                domain.target,
                *max_depth,
                depth_fn.as_ref(),
                lr,
                epsilon,
            )?;
            policy = new_policy;
            grad_norm_accum += diag.grad_norm;
            zeroed_accum += diag.fraction_gradient_zeroed_by_clip;
            n_groups += 1;
        }
    }

    let round_diag = MCTSRoundDiagnostics {
        mean_reward: total_reward / (total_episodes.max(1)) as f64,
        reward_std: 0.0,
        solve_rate: total_solved as f64 / (total_episodes.max(1)) as f64,
        grad_norm: grad_norm_accum / (n_groups.max(1)) as f64,
        fraction_gradient_zeroed_by_clip: zeroed_accum / (n_groups.max(1)) as f64,
        fraction_groups_with_nonzero_variance: groups_with_variance as f64
            / (problems.len().max(1)) as f64,
    };
    Ok((policy, round_diag))
}

/// Multi-round MCTS-sourced GRPO training. Python defaults:
/// n_samples_per_problem=6, train_num_simulations=150, lr=0.2,
/// epsilon=0.2, eval_num_simulations=60, seed=0.
#[allow(clippy::too_many_arguments)]
pub fn run_grpo_training_mcts(
    rounds: usize,
    round_problem_fn: &dyn Fn(usize) -> Vec<Problem>,
    eval_problems: &[Problem],
    n_samples_per_problem: usize,
    train_num_simulations: usize,
    lr: f64,
    epsilon: f64,
    eval_num_simulations: usize,
    seed: u64,
) -> Result<(SoftmaxPolicy, TrainingHistory, Vec<MCTSRoundDiagnostics>), String> {
    let mut policy = SoftmaxPolicy::new();
    let mut history = TrainingHistory::default();
    history.eval_solve_rate_before =
        evaluate_policy(&policy, eval_problems, eval_num_simulations, 9000);

    let mut round_diags: Vec<MCTSRoundDiagnostics> = Vec::new();
    for round_idx in 0..rounds {
        let problems = round_problem_fn(round_idx);
        let (new_policy, diag) = run_grpo_round_mcts(
            &policy,
            &problems,
            &_depth_fn,
            n_samples_per_problem,
            train_num_simulations,
            lr,
            epsilon,
            seed + round_idx as u64 * 137,
        )?;
        policy = new_policy;
        history.round_solve_rates.push(diag.solve_rate);
        history.round_mean_rewards.push(diag.mean_reward);
        history.round_grad_norms.push(diag.grad_norm);
        round_diags.push(diag);
    }

    history.eval_solve_rate_after =
        evaluate_policy(&policy, eval_problems, eval_num_simulations, 9000);
    Ok((policy, history, round_diags))
}
