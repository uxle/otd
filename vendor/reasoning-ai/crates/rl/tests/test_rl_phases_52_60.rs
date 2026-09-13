//! Tests for Phases 052–060 (Rust port of python/tests/test_rl_phases_52_60.py):
//!   052 — discounted returns
//!   053 — advantage estimation
//!   054 — GAE
//!   055 — PPO
//!   056 — GRPO
//!   057 — KL penalty
//!   058 — entropy regularization
//!   059 — reward normalization / clipping
//!   060 — end-to-end RL loop on real MCTS rollouts
// Box-Muller stand-in for Python's random.gauss (rand 0.8 has no
// StandardNormal without the rand_distr crate)
fn gauss(rng: &mut StdRng, mean: f64, std: f64) -> f64 {
    use rand::Rng;
    let u1: f64 = rng.gen::<f64>().max(1e-12);
    let u2: f64 = rng.gen::<f64>();
    mean + std * (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}
use rand::rngs::StdRng;
use rand::SeedableRng;

use reasoning_rl::advantage::{advantages_from_returns, advantages_from_rewards, td_errors};
use reasoning_rl::entropy_reg::{
    apply_entropy_bonus, entropy_from_logprobs, max_entropy,
};
use reasoning_rl::gae::{gae, gae_returns};
use reasoning_rl::grpo::{
    grpo_all_correct_zero_gradient, grpo_loss, group_advantages,
};
use reasoning_rl::kl_penalty::{apply_kl_penalty, kl_divergence_from_logprobs};
use reasoning_rl::ppo::{clip_fraction, ppo_clip_loss, ppo_clip_loss_single};
use reasoning_rl::reward_norm::{
    clip_advantages, clip_rewards, normalize_advantages, RunningMeanStd,
};
use reasoning_rl::returns::{discounted_returns, discounted_returns_with_bootstrap};
use reasoning_rl::rl_loop::{compute_rl_batch, run_rollout};

// ── Phase 052 ───────────────────────────────────────────────────────────
#[test]
fn test_returns_single_step_return_equals_reward() {
    let g = discounted_returns(&[5.0], 0.99).unwrap();
    assert!((g[0] - 5.0).abs() < 1e-7);
}

#[test]
fn test_returns_two_step_backward_recursion() {
    // G_0 = r_0 + gamma*r_1, G_1 = r_1
    let rewards = [1.0, 2.0];
    let gamma = 0.9;
    let g = discounted_returns(&rewards, gamma).unwrap();
    assert!((g[1] - 2.0).abs() < 1e-7);
    assert!((g[0] - (1.0 + 0.9 * 2.0)).abs() < 1e-7);
}

#[test]
fn test_returns_gamma_1_gives_undiscounted_sum() {
    let rewards = [1.0, 1.0, 1.0];
    let g = discounted_returns(&rewards, 1.0).unwrap();
    assert!((g[0] - 3.0).abs() < 1e-7);
    assert!((g[1] - 2.0).abs() < 1e-7);
    assert!((g[2] - 1.0).abs() < 1e-7);
}

#[test]
fn test_returns_bootstrap_increases_last_return() {
    let rewards = [0.0, 0.0, 0.0];
    let last_val = 10.0;
    let g_boot = discounted_returns_with_bootstrap(&rewards, last_val, 0.99).unwrap();
    let g_no = discounted_returns(&rewards, 0.99).unwrap();
    assert!(g_boot[0] > g_no[0]);
}

#[test]
fn test_returns_empty_rewards_raises() {
    assert_eq!(
        discounted_returns(&[], 0.99).unwrap_err(),
        "rewards must be non-empty"
    );
}

#[test]
fn test_returns_gamma_out_of_range_raises() {
    assert!(discounted_returns(&[1.0], 0.0).is_err());
    assert!(discounted_returns(&[1.0], 1.1).is_err());
}

// ── Phase 053 ───────────────────────────────────────────────────────────
#[test]
fn test_advantage_td_errors_simple() {
    // r=1, V(s)=0, V(s')=0 -> delta=1
    let deltas = td_errors(&[1.0], &[0.0], 0.0, 0.99).unwrap();
    assert!((deltas[0] - 1.0).abs() < 1e-7);
}

#[test]
fn test_advantage_td_error_with_nonzero_value() {
    // r=0, V(s)=1, V(s')=0 -> delta=0+0-1 = -1
    let deltas = td_errors(&[0.0], &[1.0], 0.0, 0.99).unwrap();
    assert!((deltas[0] - -1.0).abs() < 1e-7);
}

#[test]
fn test_advantage_advantages_are_returns_minus_values() {
    let returns = [3.0, 2.0, 1.0];
    let values = [1.0, 1.0, 1.0];
    let advs = advantages_from_returns(&returns, &values).unwrap();
    assert_eq!(advs, vec![2.0, 1.0, 0.0]);
}

#[test]
fn test_advantage_length_mismatch_raises() {
    assert!(td_errors(&[1.0, 2.0], &[1.0], 0.0, 0.99).is_err());
}

#[test]
fn test_advantage_advantages_from_rewards_convenience() {
    // Should not raise and should return a list of the right length
    let advs = advantages_from_rewards(&[1.0, 0.0, 1.0], &[0.5, 0.5, 0.5], 0.0, 0.99).unwrap();
    assert_eq!(advs.len(), 3);
}

// ── Phase 054 ───────────────────────────────────────────────────────────
#[test]
fn test_gae_lambda_0_recovers_td_error() {
    let rewards = [1.0, 0.0, 1.0];
    let values = [0.5, 0.5, 0.5];
    let gae_advs = gae(&rewards, &values, 0.0, 0.99, 0.0).unwrap();
    let td = td_errors(&rewards, &values, 0.0, 0.99).unwrap();
    for (a, d) in gae_advs.iter().zip(td.iter()) {
        assert!((a - d).abs() < 1e-10);
    }
}

#[test]
fn test_gae_lambda_1_recovers_returns_minus_values() {
    let rewards = [1.0, 1.0, 1.0];
    let values = [0.0, 0.0, 0.0];
    let gae_advs = gae(&rewards, &values, 0.0, 1.0, 1.0).unwrap();
    let returns = discounted_returns(&rewards, 1.0).unwrap();
    for (a, g) in gae_advs.iter().zip(returns.iter()) {
        assert!((a - g).abs() < 1e-9);
    }
}

#[test]
fn test_gae_returns_targets_equal_adv_plus_value() {
    let rewards = [0.5, 0.5];
    let values = [0.3, 0.3];
    let (advs, targets) = gae_returns(&rewards, &values, 0.0, 0.99, 0.95).unwrap();
    for ((adv, val), tgt) in advs.iter().zip(values.iter()).zip(targets.iter()) {
        assert!((adv + val - tgt).abs() < 1e-10);
    }
}

#[test]
fn test_gae_invalid_gamma_raises() {
    assert!(gae(&[1.0], &[0.0], 0.0, 1.5, 0.95).is_err());
}

#[test]
fn test_gae_invalid_lambda_raises() {
    assert!(gae(&[1.0], &[0.0], 0.0, 0.99, -0.1).is_err());
}

// ── Phase 055 ───────────────────────────────────────────────────────────
#[test]
fn test_ppo_ratio_1_gives_unclipped_loss() {
    // ratio=1 -> clip is never active -> loss = -1*advantage
    let (loss, ratio) = ppo_clip_loss_single((0.5f64).ln(), (0.5f64).ln(), 2.0, 0.2);
    assert!((ratio - 1.0).abs() < 1e-10);
    assert!((loss - -2.0).abs() < 1e-10);
}

#[test]
fn test_ppo_clip_fires_on_large_positive_advantage() {
    // ratio=2.0, eps=0.2, A=1.0: unclipped=2.0, clipped=1.2*1.0=1.2
    // min(2.0, 1.2) = 1.2  -> loss = -1.2
    let (loss, ratio) = ppo_clip_loss_single((0.8f64).ln(), (0.4f64).ln(), 1.0, 0.2);
    assert!((ratio - 2.0).abs() < 1e-6);
    assert!((loss - -1.2).abs() < 1e-6);
}

#[test]
fn test_ppo_batch_returns_mean() {
    let lp = vec![(0.5f64).ln(); 3];
    let advs = [1.0, 1.0, 1.0];
    let (loss, ratios) = ppo_clip_loss(&lp, &lp, &advs, 0.2).unwrap();
    assert!((loss - -1.0).abs() < 1e-10);
    assert!(ratios.iter().all(|r| (r - 1.0).abs() < 1e-9));
}

#[test]
fn test_ppo_clip_fraction_all_within() {
    let ratios = [1.0, 1.05, 0.98];
    assert!((clip_fraction(&ratios, 0.2).unwrap() - 0.0).abs() < 1e-7);
}

#[test]
fn test_ppo_clip_fraction_some_outside() {
    let ratios = [2.0, 0.5, 1.0]; // 2 out of 3 clipped
    assert!((clip_fraction(&ratios, 0.2).unwrap() - 2.0 / 3.0).abs() < 1e-7);
}

#[test]
fn test_ppo_mismatched_lengths_raises() {
    assert!(ppo_clip_loss(&[1.0], &[1.0, 2.0], &[1.0], 0.2).is_err());
}

// ── Phase 056 ───────────────────────────────────────────────────────────
#[test]
fn test_grpo_group_advantages_zero_mean() {
    let rewards = [0.2, 0.5, 0.8, 1.0];
    let advs = group_advantages(&rewards, 1e-8).unwrap();
    let mean = advs.iter().sum::<f64>() / advs.len() as f64;
    assert!(mean.abs() < 1e-10);
}

#[test]
fn test_grpo_group_advantages_unit_variance() {
    let rewards = [0.0, 1.0, 0.5, 0.75];
    let advs = group_advantages(&rewards, 1e-8).unwrap();
    let var = advs.iter().map(|a| a * a).sum::<f64>() / advs.len() as f64; // mean already 0
    assert!((var - 1.0).abs() < 1e-5);
}

#[test]
fn test_grpo_all_correct_gives_near_zero_gradient() {
    // all rewards=1.0 → std≈0 → advs≈0 → gradient≈0
    assert!(grpo_all_correct_zero_gradient(&[1.0, 1.0, 1.0, 1.0], 1e-8).unwrap());
}

#[test]
fn test_grpo_all_wrong_gives_near_zero_gradient() {
    assert!(grpo_all_correct_zero_gradient(&[0.0, 0.0, 0.0, 0.0], 1e-8).unwrap());
}

#[test]
fn test_grpo_mixed_correct_wrong_gives_nonzero_gradient() {
    let rewards = [1.0, 0.0, 1.0, 0.0];
    let advs = group_advantages(&rewards, 1e-8).unwrap();
    let max_abs = advs.iter().fold(0.0f64, |m, a| m.max(a.abs()));
    assert!(max_abs > 0.1);
}

#[test]
fn test_grpo_loss_returns_finite() {
    let lp = vec![(0.25f64).ln(); 4];
    let rewards = [1.0, 0.0, 1.0, 0.0];
    let (loss, advs, _ratios) = grpo_loss(&lp, &lp, &rewards, 0.2, 1e-8).unwrap();
    assert!(loss.is_finite());
    assert_eq!(advs.len(), 4);
}

#[test]
fn test_grpo_requires_min_2_candidates() {
    let lp = vec![(0.5f64).ln()];
    assert!(grpo_loss(&lp, &lp, &[1.0], 0.2, 1e-8).is_err());
}

// ── Phase 057 ───────────────────────────────────────────────────────────
#[test]
fn test_kl_identical_distributions_zero_kl() {
    let log_p = vec![(0.25f64).ln(); 4];
    let kl = kl_divergence_from_logprobs(&log_p, &log_p).unwrap();
    assert!(kl.abs() < 1e-10);
}

#[test]
fn test_kl_nonnegative() {
    // KL divergence is always >= 0 (Gibbs' inequality)
    let log_p = vec![(0.7f64).ln(), (0.2f64).ln(), (0.1f64).ln()];
    let log_q = vec![(0.3f64).ln(), (0.4f64).ln(), (0.3f64).ln()];
    let kl = kl_divergence_from_logprobs(&log_p, &log_q).unwrap();
    assert!(kl >= 0.0);
}

#[test]
fn test_kl_penalty_increases_loss() {
    let lp = vec![(0.5f64).ln(), (0.3f64).ln(), (0.2f64).ln()];
    let policy_loss = 1.0;
    let (total, _kl) = apply_kl_penalty(policy_loss, &lp, &lp, 0.01).unwrap();
    // KL of identical dists = 0, so total == policy_loss
    assert!((total - policy_loss).abs() < 1e-10);
}

#[test]
fn test_kl_negative_beta_raises() {
    let lp = vec![(0.5f64).ln()];
    assert!(apply_kl_penalty(0.0, &lp, &lp, -0.1).is_err());
}

// ── Phase 058 ───────────────────────────────────────────────────────────
#[test]
fn test_entropy_uniform_distribution_maximum_entropy() {
    let n = 4usize;
    let log_p = vec![(1.0 / n as f64).ln(); n];
    let ent = entropy_from_logprobs(&log_p).unwrap();
    assert!((ent - max_entropy(n).unwrap()).abs() < 1e-8);
}

#[test]
fn test_entropy_deterministic_distribution_zero_entropy() {
    // Almost deterministic: one action prob = ~1, rest = ~0
    let mut log_p = vec![(1.0 - 1e-9f64).ln()];
    log_p.extend(vec![(1e-9f64 / 3.0).ln(); 3]);
    let ent = entropy_from_logprobs(&log_p).unwrap();
    assert!(ent < 0.1);
}

#[test]
fn test_entropy_nonnegative() {
    let log_p = vec![(0.6f64).ln(), (0.3f64).ln(), (0.1f64).ln()];
    assert!(entropy_from_logprobs(&log_p).unwrap() >= 0.0);
}

#[test]
fn test_entropy_bonus_reduces_loss() {
    let lp = vec![(0.5f64).ln(), (0.5f64).ln()];
    let policy_loss = 2.0;
    let (total, _ent) = apply_entropy_bonus(policy_loss, &lp, 0.1).unwrap();
    assert!(total < policy_loss);
}

#[test]
fn test_entropy_negative_beta_raises() {
    let lp = vec![(0.5f64).ln()];
    assert!(apply_entropy_bonus(0.0, &lp, -0.1).is_err());
}

// ── Phase 059 ───────────────────────────────────────────────────────────
#[test]
fn test_reward_norm_clip_caps_large_rewards() {
    let rewards = [100.0, -50.0, 5.0];
    let clipped = clip_rewards(&rewards, 10.0).unwrap();
    assert_eq!(clipped, vec![10.0, -10.0, 5.0]);
    // clip_advantages is the same operation
    let clipped_advs = clip_advantages(&rewards, 10.0).unwrap();
    assert_eq!(clipped_advs, vec![10.0, -10.0, 5.0]);
}

#[test]
fn test_reward_norm_clip_zero_range_raises() {
    assert!(clip_rewards(&[1.0], 0.0).is_err());
}

#[test]
fn test_reward_norm_normalize_advantages_zero_mean_unit_std() {
    let advs = [1.0, 2.0, 3.0, 4.0, 5.0];
    let normed = normalize_advantages(&advs, 1e-8).unwrap();
    let mean = normed.iter().sum::<f64>() / normed.len() as f64;
    let std = (normed.iter().map(|a| a * a).sum::<f64>() / normed.len() as f64).sqrt(); // mean=0
    assert!(mean.abs() < 1e-10);
    assert!((std - 1.0).abs() < 1e-5);
}

#[test]
fn test_reward_norm_running_mean_std_converges() {
    let mut rms = RunningMeanStd::new(1e-8);
    let mut rng = StdRng::seed_from_u64(42);
    for _ in 0..10 {
        let batch: Vec<f64> = (0..50)
            .map(|_| gauss(&mut rng, 5.0, 2.0))
            .collect();
        rms.update(&batch);
    }
    // loose tolerances: small sample (Python delta=0.5)
    assert!((rms.mean - 5.0).abs() < 0.5);
    assert!((rms.std() - 2.0).abs() < 0.5);
}

#[test]
fn test_reward_norm_running_normalize_finite() {
    let mut rms = RunningMeanStd::new(1e-8);
    let data = [1.0, 2.0, 3.0, 4.0];
    let normed = rms.normalize_and_update(&data);
    assert!(normed.iter().all(|x| x.is_finite()));
}

// ── Phase 060 ───────────────────────────────────────────────────────────
#[test]
fn test_rl_loop_rollout_returns_at_least_one_step() {
    let steps = run_rollout(&[2.0, 2.0], 4.0, 0, 100);
    assert!(!steps.is_empty());
}

#[test]
fn test_rl_loop_rollout_rewards_are_in_valid_range() {
    let steps = run_rollout(&[4.0, 7.0, 8.0, 8.0], 24.0, 7, 300);
    for step in &steps {
        assert!(
            -3.0 <= step.reward && step.reward <= 3.0,
            "reward {} out of expected range",
            step.reward
        );
    }
}

#[test]
fn test_rl_loop_full_rl_batch_all_finite() {
    let rollouts: Vec<Vec<_>> = (0..4)
        .map(|i| run_rollout(&[2.0, 2.0], 4.0, i, 100))
        .collect();
    let result = compute_rl_batch(&rollouts, 0.99, 0.95, 0.2, 0.01, 0.01).unwrap();
    assert!(
        result.all_finite,
        "Some RL quantity was NaN or Inf — pipeline bug"
    );
}

#[test]
fn test_rl_loop_returns_same_length_as_rewards() {
    let rollouts = vec![run_rollout(&[2.0, 2.0], 4.0, 0, 100)];
    let result = compute_rl_batch(&rollouts, 0.99, 0.95, 0.2, 0.01, 0.01).unwrap();
    assert_eq!(result.returns.len(), result.rewards.len());
}

#[test]
fn test_rl_loop_advantages_same_length_as_rewards() {
    let rollouts = vec![run_rollout(&[2.0, 2.0], 4.0, 0, 100)];
    let result = compute_rl_batch(&rollouts, 0.99, 0.95, 0.2, 0.01, 0.01).unwrap();
    assert_eq!(result.advantages_gae.len(), result.rewards.len());
}

#[test]
fn test_rl_loop_ppo_ratios_near_one_for_standin_policy() {
    // Stand-in policy is the same uniform distribution for old and new,
    // so every ratio r_t = pi_new / pi_old must be exactly 1.0.
    let rollouts = vec![run_rollout(&[2.0, 2.0], 4.0, 0, 100)];
    let result = compute_rl_batch(&rollouts, 0.99, 0.95, 0.2, 0.01, 0.01).unwrap();
    for r in &result.ppo_ratios {
        assert!((r - 1.0).abs() < 1e-8);
    }
}

#[test]
fn test_rl_loop_kl_near_zero_for_identical_policies() {
    // Same stand-in policy for old and new -> KL = 0.
    let rollouts = vec![run_rollout(&[2.0, 2.0], 4.0, 0, 100)];
    let result = compute_rl_batch(&rollouts, 0.99, 0.95, 0.2, 0.01, 0.01).unwrap();
    assert!(result.kl_value.abs() < 1e-8);
}

#[test]
fn test_rl_loop_entropy_positive() {
    let rollouts = vec![run_rollout(&[2.0, 2.0], 4.0, 0, 100)];
    let result = compute_rl_batch(&rollouts, 0.99, 0.95, 0.2, 0.01, 0.01).unwrap();
    assert!(result.entropy_value > 0.0);
}

#[test]
fn test_rl_loop_grpo_advantages_sum_near_zero() {
    // Group advantages are zero-meaned by construction.
    let rollouts: Vec<Vec<_>> = (0..4)
        .map(|i| run_rollout(&[2.0, 2.0], 4.0, i, 100))
        .collect();
    let result = compute_rl_batch(&rollouts, 0.99, 0.95, 0.2, 0.01, 0.01).unwrap();
    let mean_adv = result.grpo_advantages.iter().sum::<f64>() / result.grpo_advantages.len() as f64;
    assert!(mean_adv.abs() < 1e-8);
}
