//! Phase 066 — One GRPO self-play round.
//!
//! For each problem in a batch: sample N full episodes directly from the
//! current policy (Phase 065), verify each terminal state, score with
//! Phase 051's reward composition, then take one GRPO weight-update step
//! (Phase 064) pooling every step from every episode. The first place in
//! the whole project a policy's own parameters change as a *direct
//! consequence* of a policy-gradient RL update.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reasoning_search::{Domain, NtState, NumberTargetDomain};

use crate::grpo_update::{grpo_weight_update, GRPOEpisode, GRPOEpisodeStep};
use crate::policy::SoftmaxPolicy;
use crate::reward_model::reward_from_verification_result;
use crate::trainable_rollout::sample_episode;

use crate::Problem;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RoundDiagnostics {
    pub mean_reward: f64,
    pub reward_std: f64,
    pub solve_rate: f64,
    pub grad_norm: f64,
    pub fraction_gradient_zeroed_by_clip: f64,
}

/// Runs one full GRPO round over `problems`, returns
/// `(updated_policy, diagnostics)`. Each problem contributes its own
/// group of N episodes with its own group-relative baseline — group
/// statistics never mix across different problems (that would compare
/// rewards on an easy problem to a hard one, which is exactly what
/// GRPO's per-problem grouping avoids).
///
/// `depth_fn_factory` is Python's `depth_fn_factory: Callable[[int],
/// Callable[[state], int]]`. Python defaults:
/// n_samples_per_problem=6, lr=0.2, epsilon=0.2, seed=0.
#[allow(clippy::too_many_arguments)]
pub fn run_grpo_round(
    policy: &SoftmaxPolicy,
    problems: &[Problem],
    depth_fn_factory: &dyn Fn(usize) -> std::rc::Rc<dyn Fn(&NtState) -> usize>,
    n_samples_per_problem: usize,
    lr: f64,
    epsilon: f64,
    seed: u64,
) -> Result<(SoftmaxPolicy, RoundDiagnostics), String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut policy = policy.clone();
    let mut total_reward = 0.0f64;
    let mut total_solved = 0usize;
    let mut total_episodes = 0usize;

    let mut per_problem_updates: Vec<Vec<GRPOEpisode>> = Vec::with_capacity(problems.len());

    for (domain, initial_state, max_depth) in problems {
        let depth_fn = depth_fn_factory(*max_depth);
        let mut episodes: Vec<GRPOEpisode> = Vec::new();
        for _ in 0..n_samples_per_problem {
            // Python: rng=random.Random(rng.randint(0, 10**9)) — a fresh
            // RNG stream per episode, seeded from the round RNG.
            let mut ep_rng = StdRng::seed_from_u64(rng.gen_range(0..=1_000_000_000u64));
            let ep = sample_episode(
                &policy,
                domain,
                initial_state,
                domain.target,
                *max_depth,
                depth_fn.as_ref(),
                &mut ep_rng,
            );
            let terminal_state = _final_state(domain, initial_state, &ep.actions);
            let passed = domain.is_terminal(&terminal_state)
                && domain.terminal_reward(&terminal_state) >= 0.999;
            let reward = reward_from_verification_result(passed);
            let steps: Vec<GRPOEpisodeStep> = ep
                .states
                .iter()
                .zip(ep.actions.iter())
                .zip(ep.legals.iter().zip(ep.log_probs.iter()))
                .map(|((s, a), (l, lp))| GRPOEpisodeStep {
                    state: s.clone(),
                    action: a.clone(),
                    legal: l.clone(),
                    old_log_prob: *lp,
                })
                .collect();
            episodes.push(GRPOEpisode {
                steps,
                reward,
            });
            total_reward += reward;
            total_solved += passed as usize;
            total_episodes += 1;
        }
        per_problem_updates.push(episodes);
    }

    // Apply GRPO update per-problem-group, sequentially, accumulating into
    // the same policy — keeps each group's own baseline intact rather
    // than averaging baselines across problems of different difficulty.
    let mut grad_norm_accum = 0.0f64;
    let mut zeroed_accum = 0.0f64;
    let mut n_groups = 0usize;
    for ((domain, _initial_state, max_depth), episodes) in
        problems.iter().zip(per_problem_updates.iter())
    {
        let depth_fn = depth_fn_factory(*max_depth);
        let (new_policy, diag) = grpo_weight_update(
            &policy,
            episodes,
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

    let round_diag = RoundDiagnostics {
        mean_reward: total_reward / total_episodes as f64,
        reward_std: 0.0, // computed at the multi-round level (Phase 067/068)
        solve_rate: total_solved as f64 / total_episodes as f64,
        grad_norm: grad_norm_accum / n_groups.max(1) as f64,
        fraction_gradient_zeroed_by_clip: zeroed_accum / n_groups.max(1) as f64,
    };
    Ok((policy, round_diag))
}

fn _final_state(
    domain: &NumberTargetDomain,
    initial_state: &NtState,
    actions: &[reasoning_search::NtAction],
) -> NtState {
    let mut state = initial_state.clone();
    for a in actions {
        state = domain.apply(&state, a);
    }
    state
}
