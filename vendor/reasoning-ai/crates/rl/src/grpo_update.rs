//! Phase 064 — GRPO weight-update step for the Phase 061 policy.
//!
//! Identical mechanics to Phase 063's PPO update, except the advantage
//! for each sample comes from Phase 056's group-relative normalization
//! instead of a value-function baseline. Given N sampled full episodes
//! for the SAME problem with rewards r_1..r_N, one weight update pools
//! every (state, action) pair from every episode, using that episode's
//! *single* group-relative advantage A_i as the credit for every step it
//! took (standard REINFORCE-with-baseline credit assignment for sparse,
//! episode-terminal-only rewards).

use crate::grpo::group_advantages;
use crate::policy::{SoftmaxPolicy, FEATURE_KEYS};
use crate::policy_grad::grad_log_prob;
use crate::ppo_update::ppo_gradient_coefficient;
use reasoning_search::{NtAction, NtState, NumberTargetDomain};

/// One step of a sampled episode: the state the action was taken in, the
/// action, the legal set at that state, and the old policy's log-prob.
#[derive(Debug, Clone)]
pub struct GRPOEpisodeStep {
    pub state: NtState,
    pub action: NtAction,
    pub legal: Vec<NtAction>,
    pub old_log_prob: f64,
}

/// One sampled full episode: its steps plus its verifier-grounded reward.
#[derive(Debug, Clone)]
pub struct GRPOEpisode {
    pub steps: Vec<GRPOEpisodeStep>,
    pub reward: f64,
}

/// Diagnostics returned by [`grpo_weight_update`] (Python dict fields).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GRPOUpdateDiagnostics {
    pub grad_norm: f64,
    pub fraction_gradient_zeroed_by_clip: f64,
    pub mean_reward: f64,
    pub reward_std: f64,
}

/// One SGD ascent step on the mean GRPO objective across a group of
/// episodes sampled for the same problem.
pub fn grpo_weight_update(
    policy: &SoftmaxPolicy,
    episodes: &[GRPOEpisode],
    domain: &NumberTargetDomain,
    target: f64,
    max_depth: usize,
    depth_fn: &dyn Fn(&NtState) -> usize,
    lr: f64,
    epsilon: f64,
) -> Result<(SoftmaxPolicy, GRPOUpdateDiagnostics), String> {
    if episodes.len() < 2 {
        return Err("GRPO requires at least 2 episodes (group needs variance)".to_string());
    }

    let rewards: Vec<f64> = episodes.iter().map(|ep| ep.reward).collect();
    let advantages = group_advantages(&rewards, 1e-8)?;

    let mut total_grad = zero_weights();
    let mut total_steps = 0usize;
    let mut zeroed = 0usize;
    for (episode, &adv) in episodes.iter().zip(advantages.iter()) {
        for step in &episode.steps {
            let new_log_prob = policy.log_prob(
                &step.state,
                &step.action,
                &step.legal,
                domain,
                target,
                max_depth,
                depth_fn,
            );
            let ratio = (new_log_prob - step.old_log_prob).exp();
            let coeff = ppo_gradient_coefficient(ratio, adv, epsilon);
            if coeff == 0.0 {
                zeroed += 1;
            }
            let lp_grad = grad_log_prob(
                policy,
                &step.state,
                &step.action,
                &step.legal,
                domain,
                target,
                max_depth,
                depth_fn,
            );
            for k in FEATURE_KEYS {
                *total_grad.get_mut(k).expect("key present") +=
                    coeff * adv * lp_grad.get(k).copied().unwrap_or(0.0);
            }
            total_steps += 1;
        }
    }

    if total_steps == 0 {
        return Err("no steps found across episodes (all episodes empty)".to_string());
    }

    let mut mean_grad = zero_weights();
    for k in FEATURE_KEYS {
        mean_grad.insert(
            k.to_string(),
            total_grad.get(k).copied().unwrap_or(0.0) / total_steps as f64,
        );
    }
    let mut new_weights = policy.weights.clone();
    for k in FEATURE_KEYS {
        let w = new_weights.get_mut(k).expect("key present");
        *w += lr * mean_grad.get(k).copied().unwrap_or(0.0);
    }
    let new_policy = SoftmaxPolicy {
        weights: new_weights,
    };

    let grad_norm = FEATURE_KEYS
        .iter()
        .map(|k| {
            let v = mean_grad.get(*k).copied().unwrap_or(0.0);
            v * v
        })
        .sum::<f64>()
        .sqrt();
    let mean_reward = rewards.iter().sum::<f64>() / rewards.len() as f64;
    let reward_std = (rewards
        .iter()
        .map(|&r| {
            let d = r - mean_reward;
            d * d
        })
        .sum::<f64>()
        / rewards.len() as f64)
        .sqrt();
    let diagnostics = GRPOUpdateDiagnostics {
        grad_norm,
        fraction_gradient_zeroed_by_clip: zeroed as f64 / total_steps as f64,
        mean_reward,
        reward_std,
    };
    Ok((new_policy, diagnostics))
}

fn zero_weights() -> std::collections::HashMap<String, f64> {
    FEATURE_KEYS
        .iter()
        .map(|k| (k.to_string(), 0.0))
        .collect()
}
