//! Rust port of python/tests/test_rl_phases_61_63.py:
//!   061 — SoftmaxPolicy
//!   062 — analytic vs numerical policy gradient
//!   063 — PPO weight update
use std::collections::HashMap;

use rand::rngs::StdRng;
use rand::SeedableRng;

use reasoning_search::{make_initial_state, NtAction, NtState, NumberTargetDomain};

use reasoning_rl::policy::{FeatureMap, SoftmaxPolicy, FEATURE_KEYS};
use reasoning_rl::policy_grad::{grad_log_prob, max_abs_grad_diff, numerical_grad_log_prob};
use reasoning_rl::ppo_update::{
    ppo_gradient_coefficient, ppo_weight_update, PPOSample,
};

fn _depth_fn(max_depth: usize) -> impl Fn(&NtState) -> usize {
    move |state: &NtState| (max_depth as i64 - state.len() as i64 + 1).max(0) as usize
}

/// Python `SoftmaxPolicy(weights={...})` with a partial weights dict
/// (missing keys default to 0.0 over FEATURE_KEYS).
fn weights(pairs: &[(&str, f64)]) -> FeatureMap {
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

struct Setup {
    domain: NumberTargetDomain,
    state: NtState,
    legal: Vec<NtAction>,
    max_depth: usize,
}

impl Setup {
    fn new(numbers: &[f64], target: f64, max_depth: usize) -> Self {
        let domain = NumberTargetDomain::new(target);
        let state = make_initial_state(numbers);
        use reasoning_search::Domain;
        let legal = domain.legal_actions(&state);
        Setup {
            domain,
            state,
            legal,
            max_depth,
        }
    }
}

// ── Phase 061 ───────────────────────────────────────────────────────────
#[test]
fn test_softmax_policy_probs_sum_to_one() {
    let s = Setup::new(&[2.0, 3.0, 5.0], 10.0, 4);
    let policy = SoftmaxPolicy::new();
    let depth_fn = _depth_fn(s.max_depth);
    let (probs, _) = policy.action_distribution(
        &s.state,
        &s.legal,
        &s.domain,
        10.0,
        s.max_depth,
        &depth_fn,
    );
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-8);
}

#[test]
fn test_softmax_policy_zero_weights_gives_uniform_distribution() {
    let s = Setup::new(&[2.0, 3.0, 5.0], 10.0, 4);
    let policy = SoftmaxPolicy {
        weights: weights(&[]),
    };
    let depth_fn = _depth_fn(s.max_depth);
    let (probs, _) = policy.action_distribution(
        &s.state,
        &s.legal,
        &s.domain,
        10.0,
        s.max_depth,
        &depth_fn,
    );
    let expected = 1.0 / s.legal.len() as f64;
    for p in &probs {
        assert!((p - expected).abs() < 1e-8);
    }
}

#[test]
fn test_softmax_policy_extreme_weight_concentrates_probability() {
    let s = Setup::new(&[2.0, 3.0, 5.0], 10.0, 4);
    let policy = SoftmaxPolicy {
        weights: weights(&[("has_exact_match", 50.0)]),
    };
    let depth_fn = _depth_fn(s.max_depth);
    let (probs, feats) = policy.action_distribution(
        &s.state,
        &s.legal,
        &s.domain,
        10.0,
        s.max_depth,
        &depth_fn,
    );
    // whichever action(s) achieve exact match should now dominate probability mass
    let exact_idx: Vec<usize> = feats
        .iter()
        .enumerate()
        .filter(|(_, f)| f.get("has_exact_match").copied().unwrap_or(0.0) == 1.0)
        .map(|(i, _)| i)
        .collect();
    if !exact_idx.is_empty() {
        let mass: f64 = exact_idx.iter().map(|&i| probs[i]).sum();
        assert!(mass > 0.9);
    }
}

#[test]
fn test_softmax_policy_sample_action_is_legal() {
    let s = Setup::new(&[2.0, 3.0, 5.0], 10.0, 4);
    let policy = SoftmaxPolicy::new();
    let mut rng = StdRng::seed_from_u64(0);
    let depth_fn = _depth_fn(s.max_depth);
    let (action, log_p, _feats) = policy.sample_action(
        &s.state,
        &s.legal,
        &s.domain,
        10.0,
        s.max_depth,
        &depth_fn,
        &mut rng,
    );
    assert!(s.legal.contains(&action));
    assert!(log_p <= 0.0);
}

#[test]
fn test_softmax_policy_log_prob_matches_distribution() {
    let s = Setup::new(&[2.0, 3.0, 5.0], 10.0, 4);
    let policy = SoftmaxPolicy {
        weights: weights(&[("depth_norm", 1.5)]),
    };
    let depth_fn = _depth_fn(s.max_depth);
    let (probs, _) = policy.action_distribution(
        &s.state,
        &s.legal,
        &s.domain,
        10.0,
        s.max_depth,
        &depth_fn,
    );
    for (i, action) in s.legal.iter().enumerate() {
        let lp = policy.log_prob(
            &s.state,
            action,
            &s.legal,
            &s.domain,
            10.0,
            s.max_depth,
            &depth_fn,
        );
        assert!((lp - probs[i].max(1e-12).ln()).abs() < 1e-8);
    }
}

// ── Phase 062 ───────────────────────────────────────────────────────────
#[test]
fn test_policy_gradient_analytic_matches_numeric_at_zero_weights() {
    let s = Setup::new(&[2.0, 3.0, 5.0, 7.0], 17.0, 4);
    let policy = SoftmaxPolicy::new();
    let action = s.legal[3].clone();
    let depth_fn = _depth_fn(s.max_depth);
    let analytic = grad_log_prob(
        &policy,
        &s.state,
        &action,
        &s.legal,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
    );
    let numeric = numerical_grad_log_prob(
        &policy,
        &s.state,
        &action,
        &s.legal,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
        1e-5,
    );
    assert!(max_abs_grad_diff(&analytic, &numeric) < 1e-4);
}

#[test]
fn test_policy_gradient_analytic_matches_numeric_at_nonzero_weights() {
    let s = Setup::new(&[2.0, 3.0, 5.0, 7.0], 17.0, 4);
    let policy = SoftmaxPolicy {
        weights: weights(&[
            ("bias", 0.3),
            ("num_remaining", -0.2),
            ("min_diff_to_target_norm", 0.7),
            ("depth_norm", -0.5),
            ("has_exact_match", 1.1),
        ]),
    };
    let depth_fn = _depth_fn(s.max_depth);
    // sample a handful of actions (Python legal[::7])
    for action in s.legal.iter().step_by(7) {
        let analytic = grad_log_prob(
            &policy,
            &s.state,
            action,
            &s.legal,
            &s.domain,
            17.0,
            s.max_depth,
            &depth_fn,
        );
        let numeric = numerical_grad_log_prob(
            &policy,
            &s.state,
            action,
            &s.legal,
            &s.domain,
            17.0,
            s.max_depth,
            &depth_fn,
            1e-5,
        );
        assert!(
            max_abs_grad_diff(&analytic, &numeric) < 1e-4,
            "mismatch for action {:?}",
            action
        );
    }
}

#[test]
fn test_policy_gradient_of_taken_action_points_toward_increasing_its_prob() {
    // Sanity: nudging weights in the analytic-gradient direction for
    // action a should increase pi(a|s).
    let s = Setup::new(&[2.0, 3.0, 5.0, 7.0], 17.0, 4);
    let policy = SoftmaxPolicy::new();
    let action = s.legal[0].clone();
    let depth_fn = _depth_fn(s.max_depth);
    let g = grad_log_prob(
        &policy,
        &s.state,
        &action,
        &s.legal,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
    );
    let before = policy.log_prob(
        &s.state,
        &action,
        &s.legal,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
    );
    let nudged = SoftmaxPolicy {
        weights: FEATURE_KEYS
            .iter()
            .map(|k| {
                (
                    k.to_string(),
                    policy.weights.get(*k).copied().unwrap_or(0.0)
                        + 0.05 * g.get(*k).copied().unwrap_or(0.0),
                )
            })
            .collect::<HashMap<String, f64>>(),
    };
    let after = nudged.log_prob(
        &s.state,
        &action,
        &s.legal,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
    );
    assert!(after > before);
}

// ── Phase 063 ───────────────────────────────────────────────────────────
#[test]
fn test_ppo_update_coefficient_is_ratio_inside_band() {
    let c = ppo_gradient_coefficient(1.05, 1.0, 0.2);
    assert!((c - 1.05).abs() < 1e-7);
}

#[test]
fn test_ppo_update_coefficient_zeroed_when_clipped_and_out_of_band_positive_adv() {
    // ratio way above 1+eps, A>0 -> clipped branch is the min -> gradient killed
    let c = ppo_gradient_coefficient(5.0, 1.0, 0.2);
    assert_eq!(c, 0.0);
}

#[test]
fn test_ppo_update_coefficient_zeroed_when_clipped_and_out_of_band_negative_adv() {
    // check against direct definition instead of assuming sign intuition
    let (ratio, a, eps) = (0.1, -1.0, 0.2);
    let unclipped = ratio * a;
    let clipped = eps.clamp_lo_hi(ratio) * a;
    let expect_zero = (ratio < 1.0 - eps || ratio > 1.0 + eps) && clipped <= unclipped;
    let c = ppo_gradient_coefficient(ratio, a, eps);
    assert_eq!(c, if expect_zero { 0.0 } else { ratio });
}

#[test]
fn test_ppo_update_weight_update_increases_probability_of_high_advantage_action() {
    let s = Setup::new(&[2.0, 3.0, 5.0, 7.0], 17.0, 4);
    let policy = SoftmaxPolicy::new();
    let action = s.legal[2].clone();
    let depth_fn = _depth_fn(s.max_depth);
    let old_log_prob = policy.log_prob(
        &s.state,
        &action,
        &s.legal,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
    );
    let samples = vec![PPOSample {
        state: s.state.clone(),
        action: action.clone(),
        legal: s.legal.clone(),
        advantage: 2.0,
        old_log_prob,
    }];
    let (new_policy, _diag) = ppo_weight_update(
        &policy,
        &samples,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
        0.5,
        0.2,
    )
    .unwrap();
    let new_log_prob = new_policy.log_prob(
        &s.state,
        &action,
        &s.legal,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
    );
    assert!(new_log_prob > old_log_prob);
}

#[test]
fn test_ppo_update_weight_update_decreases_probability_of_negative_advantage_action() {
    let s = Setup::new(&[2.0, 3.0, 5.0, 7.0], 17.0, 4);
    let policy = SoftmaxPolicy::new();
    let action = s.legal[2].clone();
    let depth_fn = _depth_fn(s.max_depth);
    let old_log_prob = policy.log_prob(
        &s.state,
        &action,
        &s.legal,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
    );
    let samples = vec![PPOSample {
        state: s.state.clone(),
        action: action.clone(),
        legal: s.legal.clone(),
        advantage: -2.0,
        old_log_prob,
    }];
    let (new_policy, _diag) = ppo_weight_update(
        &policy,
        &samples,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
        0.5,
        0.2,
    )
    .unwrap();
    let new_log_prob = new_policy.log_prob(
        &s.state,
        &action,
        &s.legal,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
    );
    assert!(new_log_prob < old_log_prob);
}

#[test]
fn test_ppo_update_empty_samples_raises() {
    let s = Setup::new(&[2.0, 3.0, 5.0, 7.0], 17.0, 4);
    let policy = SoftmaxPolicy::new();
    let depth_fn = _depth_fn(s.max_depth);
    assert!(ppo_weight_update(
        &policy,
        &[],
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
        0.1,
        0.2
    )
    .is_err());
}

#[test]
fn test_ppo_update_diagnostics_reports_clip_fraction() {
    let s = Setup::new(&[2.0, 3.0, 5.0, 7.0], 17.0, 4);
    let policy = SoftmaxPolicy::new();
    let action = s.legal[0].clone();
    let depth_fn = _depth_fn(s.max_depth);
    let old_log_prob = policy.log_prob(
        &s.state,
        &action,
        &s.legal,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
    );
    // Force an extreme ratio by pretending old_log_prob was very different
    let samples = vec![PPOSample {
        state: s.state.clone(),
        action: action.clone(),
        legal: s.legal.clone(),
        advantage: 1.0,
        old_log_prob: old_log_prob - 10.0,
    }];
    let (_new_policy, diag) = ppo_weight_update(
        &policy,
        &samples,
        &s.domain,
        17.0,
        s.max_depth,
        &depth_fn,
        0.1,
        0.2,
    )
    .unwrap();
    assert_eq!(diag.fraction_gradient_zeroed_by_clip, 1.0);
}

/// Python `max(1 - eps, min(1 + eps, ratio))` used by the
/// direct-definition cross-check above.
trait ClampLoHi {
    fn clamp_lo_hi(self, ratio: f64) -> f64;
}
impl ClampLoHi for f64 {
    fn clamp_lo_hi(self, ratio: f64) -> f64 {
        let hi = 1.0 + self;
        let lo = 1.0 - self;
        let inner = if ratio < hi { ratio } else { hi };
        if inner > lo {
            inner
        } else {
            lo
        }
    }
}
