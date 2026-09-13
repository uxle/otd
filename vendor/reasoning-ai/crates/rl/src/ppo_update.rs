//! Phase 063 — One PPO-clip weight-update step for the Phase 061 policy.
//!
//! Combines Phase 055 (PPO clip formula) and Phase 062 (analytic
//! d/d theta [log pi]) into an actual SGD step:
//! theta <- theta + lr * grad(objective).
//!
//! Gradient of the PPO-clip objective w.r.t. theta, for one sample:
//! inside the ratio band both min branches are differentiable
//! (d(objective)/d theta = ratio * A * d(log pi_new)/d theta); outside
//! the band, when the clipped branch is the min, clip(ratio) is pinned
//! to the boundary so its derivative is exactly 0 — PPO's whole point
//! is to kill the gradient once the policy has moved "far enough".
//! Implemented directly as:
//!
//! active = not ( (ratio < 1-eps or ratio > 1+eps) and clipped <= unclipped )
//! coefficient = ratio if active else 0.0
//! d(objective)/d theta = coefficient * A * d(log pi_new)/d theta
use crate::policy::{SoftmaxPolicy, FEATURE_KEYS};
use crate::policy_grad::grad_log_prob;
use reasoning_search::{NtAction, NtState, NumberTargetDomain};

/// One PPO training sample: a (state, action, legal set) triple with its
/// advantage and the behavior (old) policy's log-prob of the action.
#[derive(Debug, Clone)]
pub struct PPOSample {
    pub state: NtState,
    pub action: NtAction,
    pub legal: Vec<NtAction>,
    pub advantage: f64,
    pub old_log_prob: f64,
}

/// Diagnostics returned by [`ppo_weight_update`] (Python dict fields).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PPOUpdateDiagnostics {
    pub grad_norm: f64,
    pub fraction_gradient_zeroed_by_clip: f64,
}

/// Returns the scalar multiplier on (advantage * d(log pi_new)/d theta)
/// for one sample's contribution to the PPO-clip objective gradient.
/// See module docstring for the derivation of the three cases.
pub fn ppo_gradient_coefficient(ratio: f64, advantage: f64, epsilon: f64) -> f64 {
    let clipped_ratio = py_clamp(ratio, 1.0 - epsilon, 1.0 + epsilon);
    let unclipped = ratio * advantage;
    let clipped = clipped_ratio * advantage;
    let out_of_band = ratio < 1.0 - epsilon || ratio > 1.0 + epsilon;
    let clipped_is_min = clipped <= unclipped;
    if out_of_band && clipped_is_min {
        return 0.0;
    }
    ratio
}

/// One SGD ascent step on the mean PPO-clip objective over `samples`.
/// Returns `(new_policy, diagnostics)`: the mean gradient norm and the
/// fraction of samples whose gradient was zeroed by clipping (a "how much
/// of the batch got clipped away" signal, complementary to Phase 055's
/// clip_fraction on the loss side).
pub fn ppo_weight_update(
    policy: &SoftmaxPolicy,
    samples: &[PPOSample],
    domain: &NumberTargetDomain,
    target: f64,
    max_depth: usize,
    depth_fn: &dyn Fn(&NtState) -> usize,
    lr: f64,
    epsilon: f64,
) -> Result<(SoftmaxPolicy, PPOUpdateDiagnostics), String> {
    if samples.is_empty() {
        return Err("samples must be non-empty".to_string());
    }
    let mut total_grad = zero_weights();
    let mut zeroed = 0usize;
    for s in samples {
        let new_log_prob = policy.log_prob(
            &s.state,
            &s.action,
            &s.legal,
            domain,
            target,
            max_depth,
            depth_fn,
        );
        let ratio = (new_log_prob - s.old_log_prob).exp();
        let coeff = ppo_gradient_coefficient(ratio, s.advantage, epsilon);
        if coeff == 0.0 {
            zeroed += 1;
        }
        let lp_grad = grad_log_prob(
            policy,
            &s.state,
            &s.action,
            &s.legal,
            domain,
            target,
            max_depth,
            depth_fn,
        );
        for k in FEATURE_KEYS {
            *total_grad.get_mut(k).expect("key present") +=
                coeff * s.advantage * lp_grad.get(k).copied().unwrap_or(0.0);
        }
    }

    let n = samples.len();
    let mut mean_grad = zero_weights();
    for k in FEATURE_KEYS {
        mean_grad.insert(
            k.to_string(),
            total_grad.get(k).copied().unwrap_or(0.0) / n as f64,
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

    // grad_norm = sqrt(sum(v^2)) over FEATURE_KEYS (Python dict order).
    let grad_norm = FEATURE_KEYS
        .iter()
        .map(|k| {
            let v = mean_grad.get(*k).copied().unwrap_or(0.0);
            v * v
        })
        .sum::<f64>()
        .sqrt();
    let diagnostics = PPOUpdateDiagnostics {
        grad_norm,
        fraction_gradient_zeroed_by_clip: zeroed as f64 / n as f64,
    };
    Ok((new_policy, diagnostics))
}

fn zero_weights() -> std::collections::HashMap<String, f64> {
    FEATURE_KEYS
        .iter()
        .map(|k| (k.to_string(), 0.0))
        .collect()
}

/// Python `max(lo, min(hi, x))` for plain (non-NaN) floats.
#[inline]
fn py_clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if hi < x {
        hi
    } else if lo > x {
        lo
    } else {
        x
    }
}
