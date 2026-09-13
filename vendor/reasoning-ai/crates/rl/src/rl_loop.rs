//! Phase 060 — RL training loop integration (design doc section 2.9,
//! training loop diagram in Loop file).
//!
//! Wires together every Phase 051-059 component into one verified
//! end-to-end demonstration: MCTS rollout -> verifier reward ->
//! discounted returns -> GAE advantages -> advantage normalization ->
//! PPO loss -> GRPO loss -> KL penalty -> entropy bonus -> diagnostics.
//! No weight update happens here — a pure arithmetic integration test
//! that proves every component composes correctly on real data.
//!
//! The 'policy' in this phase is a stand-in: a uniform random
//! distribution over legal MCTS actions, producing log-probabilities of
//! -log(n_actions); ratios pi_new/pi_old are exactly 1.0 everywhere.

use serde::{Deserialize, Serialize};

use reasoning_search::{Domain, Mcts, NtAction, NtState, NumberTargetDomain};
use reasoning_search::make_initial_state;

use crate::entropy_reg::apply_entropy_bonus;
use crate::gae::gae_returns;
use crate::grpo::grpo_loss;
use crate::kl_penalty::apply_kl_penalty;
use crate::ppo::{clip_fraction, ppo_clip_loss};
use crate::reward_model::reward_from_verification_result;
use crate::reward_norm::{clip_rewards, normalize_advantages};
use crate::returns::discounted_returns;

#[derive(Debug, Clone)]
pub struct RolloutStep {
    pub state: NtState,
    /// number of legal actions at this state
    pub n_actions: usize,
    pub action_taken: Option<NtAction>,
    /// Phase 051 composed reward
    pub reward: f64,
    /// placeholder: 0.0 until a value head exists
    pub value_estimate: f64,
}

impl RolloutStep {
    /// Uniform policy: log(1/n_actions).
    pub fn log_prob(&self) -> f64 {
        if self.n_actions == 0 {
            return f64::NEG_INFINITY;
        }
        -((self.n_actions as f64).ln())
    }
}

/// All computed quantities from one RL mini-batch, for testing/logging.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RLBatchResult {
    pub rewards: Vec<f64>,
    pub returns: Vec<f64>,
    pub advantages_gae: Vec<f64>,
    pub advantages_norm: Vec<f64>,
    pub ppo_loss: f64,
    pub ppo_ratios: Vec<f64>,
    pub ppo_clip_frac: f64,
    pub grpo_loss: f64,
    pub grpo_advantages: Vec<f64>,
    pub kl_value: f64,
    pub entropy_value: f64,
    pub total_loss: f64,
    pub all_finite: bool,
}

/// Run one MCTS episode and extract a trajectory of (state, action,
/// reward) triples. The trajectory ends at the terminal state (good or
/// bad); if search found no verified path the episode is one failed
/// step. Python signature: `run_rollout(numbers, target, seed=0,
/// num_simulations=300)`.
pub fn run_rollout(numbers: &[f64], target: f64, seed: u64, num_simulations: usize) -> Vec<RolloutStep> {
    let domain = NumberTargetDomain::new(target);
    let state = make_initial_state(numbers);
    let mut mcts = Mcts::new(domain.clone(), 8, seed);
    let result = mcts.search(state.clone(), num_simulations);

    // Reconstruct trajectory from MCTS tree root via the best verified
    // path (or the best-reward path even if not fully verified).
    let mut steps: Vec<RolloutStep> = Vec::new();
    let mut cur_state = state;
    let path: Vec<NtAction> = result.verified_path.clone().unwrap_or_default();

    // If search found no path, treat the episode as one step (failed attempt)
    if path.is_empty() {
        let r = reward_from_verification_result(false);
        let legal = domain.legal_actions(&cur_state);
        steps.push(RolloutStep {
            state: cur_state,
            n_actions: legal.len(),
            action_taken: None,
            reward: r,
            value_estimate: 0.0,
        });
        return steps;
    }

    for action in &path {
        let legal = domain.legal_actions(&cur_state);
        let next_state = domain.apply(&cur_state, action);
        let is_last = domain.is_terminal(&next_state);
        let r = if is_last {
            let passed = domain.terminal_reward(&next_state) >= 0.999;
            reward_from_verification_result(passed)
        } else {
            0.0 // intermediate steps get 0 until a PRM scores them
        };
        steps.push(RolloutStep {
            state: cur_state.clone(),
            n_actions: legal.len(),
            action_taken: Some(action.clone()),
            reward: r,
            value_estimate: 0.0,
        });
        cur_state = next_state;
    }

    steps
}

/// Given a list of rollout trajectories, compute all RL quantities.
/// Multiple trajectories are concatenated for PPO; GRPO uses per-problem
/// groups (here: each rollout is treated as one 'problem', using the
/// single terminal reward as the group element — a real GRPO run would
/// sample N rollouts per problem and pass them together).
///
/// Python defaults: gamma=0.99, lam=0.95, ppo_epsilon=0.2, kl_beta=0.01,
/// ent_beta=0.01.
pub fn compute_rl_batch(
    rollouts: &[Vec<RolloutStep>],
    gamma: f64,
    lam: f64,
    ppo_epsilon: f64,
    kl_beta: f64,
    ent_beta: f64,
) -> Result<RLBatchResult, String> {
    // Flatten all steps for PPO
    let mut all_rewards: Vec<f64> = Vec::new();
    let mut all_values: Vec<f64> = Vec::new();
    let mut all_log_probs: Vec<f64> = Vec::new();
    for traj in rollouts {
        for step in traj {
            all_rewards.push(step.reward);
            all_values.push(step.value_estimate);
            all_log_probs.push(step.log_prob());
        }
    }

    if all_rewards.is_empty() {
        return Err("No steps in rollouts".to_string());
    }

    // Phase 052: discounted returns
    let returns_flat = discounted_returns(&all_rewards, gamma)?;

    // Phase 054: GAE (value estimates are all 0 in this stand-in)
    let (advs_gae, _) = gae_returns(&all_rewards, &all_values, 0.0, gamma, lam)?;

    // Phase 059: normalize advantages
    let advs_norm = normalize_advantages(&advs_gae, 1e-8)?;

    // Phase 059: clip rewards (sanity check no extreme values slip through)
    let _clipped_rewards = clip_rewards(&all_rewards, 10.0)?;

    // Phase 055: PPO loss (old == new since stand-in policy is fixed)
    let log_probs_old = all_log_probs.clone(); // same stand-in policy
    let (ppo_l, ratios) =
        ppo_clip_loss(&all_log_probs, &log_probs_old, &advs_norm, ppo_epsilon)?;
    let clip_frac = clip_fraction(&ratios, ppo_epsilon)?;

    // Phase 056: GRPO over terminal rewards of each trajectory
    let mut terminal_rewards: Vec<f64> = rollouts
        .iter()
        .filter(|t| !t.is_empty())
        .map(|t| t[t.len() - 1].reward)
        .collect();
    let mut terminal_log_new: Vec<f64> = rollouts
        .iter()
        .filter(|t| !t.is_empty())
        .map(|t| t[t.len() - 1].log_prob())
        .collect();
    let mut terminal_log_old = terminal_log_new.clone();
    if terminal_rewards.len() < 2 {
        // GRPO requires group size >= 2; pad with a second identical sample
        terminal_rewards = [terminal_rewards.clone(), terminal_rewards].concat();
        terminal_log_new = [terminal_log_new.clone(), terminal_log_new].concat();
        terminal_log_old = [terminal_log_old.clone(), terminal_log_old].concat();
    }
    let (grpo_l, grpo_advs, _) =
        grpo_loss(&terminal_log_new, &terminal_log_old, &terminal_rewards, ppo_epsilon, 1e-8)?;

    // Phase 057: KL penalty (ratio = 1 everywhere -> KL = 0 for this stand-in)
    let (total_loss_no_ent, kl_val) =
        apply_kl_penalty(ppo_l, &all_log_probs, &log_probs_old, kl_beta)?;

    // Phase 058: entropy bonus
    let (total_loss, ent_val) =
        apply_entropy_bonus(total_loss_no_ent, &all_log_probs, ent_beta)?;

    // Sanity: all computed floats must be finite
    let mut all_finite = returns_flat.iter().all(|x| x.is_finite())
        && advs_gae.iter().all(|x| x.is_finite())
        && advs_norm.iter().all(|x| x.is_finite())
        && ratios.iter().all(|x| x.is_finite());
    for x in [ppo_l, grpo_l, kl_val, ent_val, total_loss] {
        all_finite = all_finite && x.is_finite();
    }

    Ok(RLBatchResult {
        rewards: all_rewards,
        returns: returns_flat,
        advantages_gae: advs_gae,
        advantages_norm: advs_norm,
        ppo_loss: ppo_l,
        ppo_ratios: ratios,
        ppo_clip_frac: clip_frac,
        grpo_loss: grpo_l,
        grpo_advantages: grpo_advs,
        kl_value: kl_val,
        entropy_value: ent_val,
        total_loss,
        all_finite,
    })
}
