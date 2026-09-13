//! Rust port of python/tests/test_rl_phases_64_68.py:
//!   064 — GRPO weight update
//!   065 — trainable rollout bridge
//!   066 — GRPO self-play round
//!   067 — GRPO training loop
//!   068 — training diagnostics
use std::rc::Rc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use reasoning_search::{make_initial_state, NtState, NumberTargetDomain};

use reasoning_rl::grpo_self_play::run_grpo_round;
use reasoning_rl::grpo_training_loop::{evaluate_policy, run_grpo_training};
use reasoning_rl::grpo_update::{grpo_weight_update, GRPOEpisode, GRPOEpisodeStep};
use reasoning_rl::policy::{SoftmaxPolicy, FEATURE_KEYS};
use reasoning_rl::trainable_rollout::{sample_episode, TrainableRollout};
use reasoning_rl::training_diagnostics::{
    detect_entropy_collapse, detect_policy_stagnation, kl_from_reference,
    policy_entropy_on_probe_states,
};
use reasoning_rl::{Problem, _depth_fn};
use reasoning_search::Domain;

fn _mk_problem(numbers: &[f64], target: f64) -> Problem {
    let max_depth = 6;
    (
        NumberTargetDomain::new(target),
        make_initial_state(numbers),
        max_depth,
    )
}

fn weights(pairs: &[(&str, f64)]) -> std::collections::HashMap<String, f64> {
    FEATURE_KEYS
        .iter()
        .map(|k| {
            (
                k.to_string(),
                pairs
                    .iter()
                    .find(|(n, _)| n == k)
                    .map(|(_, v)| *v)
                    .unwrap_or(0.0),
            )
        })
        .collect()
}

// ── Phase 064 ───────────────────────────────────────────────────────────
struct GrpoUpdateSetup {
    domain: NumberTargetDomain,
    state: NtState,
    legal: Vec<reasoning_search::NtAction>,
    max_depth: usize,
    depth_fn: Rc<dyn Fn(&NtState) -> usize>,
}

impl GrpoUpdateSetup {
    fn new() -> Self {
        let domain = NumberTargetDomain::new(10.0);
        let state = make_initial_state(&[2.0, 3.0, 5.0]);
        let legal = domain.legal_actions(&state);
        let max_depth = 4;
        GrpoUpdateSetup {
            domain,
            state,
            legal,
            max_depth,
            depth_fn: _depth_fn(max_depth),
        }
    }

    fn make_episode(&self, policy: &SoftmaxPolicy, reward: f64, action_idx: usize) -> GRPOEpisode {
        let action = self.legal[action_idx].clone();
        let lp = policy.log_prob(
            &self.state,
            &action,
            &self.legal,
            &self.domain,
            10.0,
            self.max_depth,
            self.depth_fn.as_ref(),
        );
        GRPOEpisode {
            steps: vec![GRPOEpisodeStep {
                state: self.state.clone(),
                action,
                legal: self.legal.clone(),
                old_log_prob: lp,
            }],
            reward,
        }
    }

    /// The 5-feature representation is deliberately coarse, so distinct
    /// actions CAN alias to identical feature vectors — this helper finds
    /// two actions the policy can actually tell apart.
    fn find_two_actions_with_different_features(
        &self,
        policy: &SoftmaxPolicy,
    ) -> (usize, usize) {
        let (_probs, feats) = policy.action_distribution(
            &self.state,
            &self.legal,
            &self.domain,
            10.0,
            self.max_depth,
            self.depth_fn.as_ref(),
        );
        for i in 0..self.legal.len() {
            for j in i + 1..self.legal.len() {
                if feats[i] != feats[j] {
                    return (i, j);
                }
            }
        }
        panic!("no two legal actions had distinguishable features -- test setup needs a richer state");
    }
}

#[test]
fn test_grpo_update_requires_at_least_two_episodes() {
    let s = GrpoUpdateSetup::new();
    let policy = SoftmaxPolicy::new();
    let episodes = vec![s.make_episode(&policy, 1.0, 0)];
    assert!(grpo_weight_update(
        &policy,
        &episodes,
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
        0.1,
        0.2
    )
    .is_err());
}

#[test]
fn test_grpo_update_mixed_rewards_on_different_actions_moves_weights() {
    // Two actions the policy can actually distinguish, with different
    // rewards -> gradients shouldn't cancel (unlike two episodes that
    // both take the same action, where opposite-signed advantages on an
    // identical (state,action) pair exactly cancel by construction).
    let s = GrpoUpdateSetup::new();
    let policy = SoftmaxPolicy::new();
    let (i, j) = s.find_two_actions_with_different_features(&policy);
    let episodes = vec![
        s.make_episode(&policy, 1.0, i),
        s.make_episode(&policy, 0.0, j),
    ];
    let (new_policy, diag) = grpo_weight_update(
        &policy,
        &episodes,
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
        0.1,
        0.2,
    )
    .unwrap();
    assert!(diag.grad_norm > 0.0);
    assert_ne!(new_policy.weights, policy.weights);
}

#[test]
fn test_grpo_update_identical_rewards_gives_near_zero_gradient() {
    let s = GrpoUpdateSetup::new();
    let policy = SoftmaxPolicy::new();
    let episodes = vec![
        s.make_episode(&policy, 1.0, 0),
        s.make_episode(&policy, 1.0, 1),
        s.make_episode(&policy, 1.0, 2),
    ];
    let (_new_policy, diag) = grpo_weight_update(
        &policy,
        &episodes,
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
        0.1,
        0.2,
    )
    .unwrap();
    assert!(diag.grad_norm < 1e-6);
}

#[test]
fn test_grpo_update_identical_action_opposite_reward_cancels() {
    // Same (state, action) taken by two episodes with opposite
    // group-relative advantage -> gradient contributions cancel exactly.
    let s = GrpoUpdateSetup::new();
    let policy = SoftmaxPolicy::new();
    let episodes = vec![
        s.make_episode(&policy, 1.0, 0),
        s.make_episode(&policy, 0.0, 0),
    ];
    let (_new_policy, diag) = grpo_weight_update(
        &policy,
        &episodes,
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
        0.1,
        0.2,
    )
    .unwrap();
    assert!(diag.grad_norm < 1e-9);
}

// ── Phase 065 ───────────────────────────────────────────────────────────
#[test]
fn test_rollout_bridge_rollout_policy_returns_legal_action() {
    let domain = NumberTargetDomain::new(10.0);
    let state = make_initial_state(&[2.0, 3.0, 5.0]);
    let max_depth = 4;
    let policy = SoftmaxPolicy::new();
    let mut rollout = TrainableRollout::new(
        policy,
        domain.clone(),
        10.0,
        max_depth,
        _depth_fn(max_depth),
        0.0, // epsilon=0.0 (Python test)
    );
    let mut rng = StdRng::seed_from_u64(1);
    let legal = domain.legal_actions(&state);
    let action = rollout.choose_with(&mut rng, &state, &legal);
    assert!(legal.contains(&action));
}

#[test]
fn test_rollout_bridge_sample_episode_terminates() {
    let domain = NumberTargetDomain::new(10.0);
    let state = make_initial_state(&[2.0, 3.0, 5.0]);
    let max_depth = 4;
    let policy = SoftmaxPolicy::new();
    let mut rng = StdRng::seed_from_u64(3);
    let ep = sample_episode(
        &policy,
        &domain,
        &state,
        10.0,
        max_depth,
        _depth_fn(max_depth).as_ref(),
        &mut rng,
    );
    assert_eq!(ep.states.len(), ep.actions.len());
    assert_eq!(ep.actions.len(), ep.log_probs.len());
    assert!(ep.actions.len() > 0);
    // 3 numbers -> exactly 2 merges to reach a single number
    assert_eq!(ep.actions.len(), 2);
}

#[test]
fn test_rollout_bridge_sample_episode_log_probs_are_valid() {
    let domain = NumberTargetDomain::new(10.0);
    let state = make_initial_state(&[2.0, 3.0, 5.0]);
    let max_depth = 4;
    let policy = SoftmaxPolicy::new();
    let mut rng = StdRng::seed_from_u64(5);
    let ep = sample_episode(
        &policy,
        &domain,
        &state,
        10.0,
        max_depth,
        _depth_fn(max_depth).as_ref(),
        &mut rng,
    );
    for lp in &ep.log_probs {
        assert!(*lp <= 0.0);
        assert!(lp.is_finite());
    }
}

// ── Phase 066 ───────────────────────────────────────────────────────────
#[test]
fn test_grpo_self_play_one_round_runs_and_returns_diagnostics() {
    let policy = SoftmaxPolicy::new();
    let problems = vec![
        _mk_problem(&[2.0, 3.0, 5.0], 10.0),
        _mk_problem(&[1.0, 4.0, 6.0], 9.0),
    ];
    let (_new_policy, diag) =
        run_grpo_round(&policy, &problems, &_depth_fn, 4, 0.2, 0.2, 1).unwrap();
    assert!(0.0 <= diag.solve_rate && diag.solve_rate <= 1.0);
    assert!(diag.mean_reward.is_finite());
    assert!(diag.grad_norm.is_finite());
}

#[test]
fn test_grpo_self_play_weights_change_after_a_round() {
    let policy = SoftmaxPolicy::new();
    let problems = vec![
        _mk_problem(&[2.0, 3.0, 5.0], 10.0),
        _mk_problem(&[1.0, 4.0, 6.0], 9.0),
        _mk_problem(&[2.0, 2.0, 8.0], 12.0),
    ];
    let (new_policy, _diag) =
        run_grpo_round(&policy, &problems, &_depth_fn, 6, 0.3, 0.2, 2).unwrap();
    assert_ne!(new_policy.weights, policy.weights);
}

// ── Phase 067 ───────────────────────────────────────────────────────────
#[test]
fn test_grpo_training_loop_short_training_run_produces_history() {
    let round_problems = |round_idx: usize| -> Vec<Problem> {
        let mut rng = StdRng::seed_from_u64(500 + round_idx as u64);
        let mut out = Vec::new();
        for _ in 0..4 {
            let nums: Vec<f64> = (0..3).map(|_| rng.gen_range(1..=9) as f64).collect();
            let target = nums[0] + nums[1] + nums[2];
            out.push(_mk_problem(&nums, target));
        }
        out
    };

    let eval_problems = vec![
        _mk_problem(&[2.0, 3.0, 5.0], 10.0),
        _mk_problem(&[1.0, 1.0, 8.0], 10.0),
    ];
    let (_policy, history) = run_grpo_training(
        2,
        &round_problems,
        &eval_problems,
        4,
        0.2,
        0.2,
        30,
        42,
    )
    .unwrap();
    assert_eq!(history.round_solve_rates.len(), 2);
    assert!(0.0 <= history.eval_solve_rate_before && history.eval_solve_rate_before <= 1.0);
    assert!(0.0 <= history.eval_solve_rate_after && history.eval_solve_rate_after <= 1.0);
}

#[test]
fn test_grpo_training_loop_evaluate_policy_runs_on_untrained_policy() {
    let policy = SoftmaxPolicy::new();
    let eval_problems = vec![_mk_problem(&[2.0, 2.0], 4.0)];
    let rate = evaluate_policy(&policy, &eval_problems, 50, 1);
    assert!(0.0 <= rate && rate <= 1.0);
}

// ── Phase 068 ───────────────────────────────────────────────────────────
#[allow(dead_code)]
struct DiagSetup {
    domain: NumberTargetDomain,
    state: NtState,
    legal: Vec<reasoning_search::NtAction>,
    max_depth: usize,
    depth_fn: Rc<dyn Fn(&NtState) -> usize>,
    probes: Vec<(NtState, Vec<reasoning_search::NtAction>)>,
}

impl DiagSetup {
    fn new() -> Self {
        let domain = NumberTargetDomain::new(10.0);
        let state = make_initial_state(&[2.0, 3.0, 5.0]);
        let legal = domain.legal_actions(&state);
        let max_depth = 4;
        DiagSetup {
            probes: vec![(state.clone(), legal.clone())],
            domain,
            state,
            legal,
            max_depth,
            depth_fn: _depth_fn(max_depth),
        }
    }
}

#[test]
fn test_training_diagnostics_uniform_policy_has_near_max_normalized_entropy() {
    let s = DiagSetup::new();
    let policy = SoftmaxPolicy::new();
    let ent = policy_entropy_on_probe_states(
        &policy,
        &s.probes,
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
    )
    .unwrap();
    assert!(ent > 0.99);
}

#[test]
fn test_training_diagnostics_extreme_weights_have_low_normalized_entropy() {
    let s = DiagSetup::new();
    let policy = SoftmaxPolicy {
        weights: weights(&[("has_exact_match", 100.0)]),
    };
    let ent = policy_entropy_on_probe_states(
        &policy,
        &s.probes,
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
    )
    .unwrap();
    assert!(ent < 0.5);
}

#[test]
fn test_training_diagnostics_kl_from_self_is_zero() {
    let s = DiagSetup::new();
    let policy = SoftmaxPolicy {
        weights: weights(&[("bias", 0.5)]),
    };
    let kl = kl_from_reference(
        &policy,
        &policy,
        &s.probes,
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
    )
    .unwrap();
    assert!(kl.abs() < 1e-8);
}

#[test]
fn test_training_diagnostics_kl_from_different_policy_is_positive() {
    let s = DiagSetup::new();
    let p1 = SoftmaxPolicy::new();
    let p2 = SoftmaxPolicy {
        weights: weights(&[("has_exact_match", 5.0)]),
    };
    let kl = kl_from_reference(
        &p2,
        &p1,
        &s.probes,
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
    )
    .unwrap();
    assert!(kl > 0.0);
}

#[test]
fn test_training_diagnostics_detect_entropy_collapse() {
    assert!(detect_entropy_collapse(&[0.8, 0.5, 0.02], 0.05).unwrap());
    assert!(!detect_entropy_collapse(&[0.8, 0.5, 0.3], 0.05).unwrap());
}

#[test]
fn test_training_diagnostics_detect_policy_stagnation() {
    assert!(detect_policy_stagnation(&[0.5, 1e-6, 1e-7, 1e-8], 1e-4, 3));
    assert!(!detect_policy_stagnation(&[0.5, 0.4, 0.3], 1e-4, 3));
    assert!(!detect_policy_stagnation(&[1e-8], 1e-4, 3)); // not enough history
}
