//! Phase 062 — Analytic policy gradient for the softmax policy, verified
//! against numerical (finite-difference) gradients (Promot section 29:
//! "Never assume mathematical code is correct without numerical
//! verification").
//!
//! d/d theta_k [ log pi(a|s) ] = phi(s,a)_k - E_{a' ~ pi(.|s)}[ phi(s,a')_k ]
//! — the feature of the action taken, minus the policy's expected feature
//! under its own current distribution.

use std::collections::HashMap;

use crate::policy::{FeatureMap, SoftmaxPolicy, FEATURE_KEYS};
use reasoning_search::{NtAction, NtState, NumberTargetDomain};

/// Returns the gradient dict: d(log pi(action|state)) / d theta_k for
/// each feature key k, using the closed-form identity above.
pub fn grad_log_prob(
    policy: &SoftmaxPolicy,
    state: &NtState,
    action: &NtAction,
    legal: &[NtAction],
    domain: &NumberTargetDomain,
    target: f64,
    max_depth: usize,
    depth_fn: &dyn Fn(&NtState) -> usize,
) -> FeatureMap {
    let (probs, feats_list) =
        policy.action_distribution(state, legal, domain, target, max_depth, depth_fn);
    let idx = legal
        .iter()
        .position(|a| a == action)
        .expect("action must be in legal");
    let taken_feats = &feats_list[idx];

    let mut expected_feats: FeatureMap = FEATURE_KEYS
        .iter()
        .map(|k| (k.to_string(), 0.0))
        .collect();
    for (p, feats) in probs.iter().zip(feats_list.iter()) {
        for k in FEATURE_KEYS {
            let e = expected_feats.get_mut(k).expect("key present");
            *e += p * feats.get(k).copied().unwrap_or(0.0);
        }
    }

    let mut out: FeatureMap = std::collections::HashMap::new();
    for k in FEATURE_KEYS {
        out.insert(
            k.to_string(),
            taken_feats.get(k).copied().unwrap_or(0.0)
                - expected_feats.get(k).copied().unwrap_or(0.0),
        );
    }
    out
}

/// Central-difference numerical gradient, for testing grad_log_prob
/// against: (f(theta+h) - f(theta-h)) / (2h) for each theta_k
/// independently, all other weights held fixed.
pub fn numerical_grad_log_prob(
    policy: &SoftmaxPolicy,
    state: &NtState,
    action: &NtAction,
    legal: &[NtAction],
    domain: &NumberTargetDomain,
    target: f64,
    max_depth: usize,
    depth_fn: &dyn Fn(&NtState) -> usize,
    h: f64,
) -> FeatureMap {
    let mut grads: FeatureMap = HashMap::new();
    for k in FEATURE_KEYS {
        let mut w_plus = policy.weights.clone();
        *w_plus.entry(k.to_string()).or_insert(0.0) += h;
        let mut w_minus = policy.weights.clone();
        *w_minus.entry(k.to_string()).or_insert(0.0) -= h;

        let p_plus = SoftmaxPolicy { weights: w_plus };
        let p_minus = SoftmaxPolicy { weights: w_minus };

        let lp_plus = p_plus.log_prob(state, action, legal, domain, target, max_depth, depth_fn);
        let lp_minus = p_minus.log_prob(state, action, legal, domain, target, max_depth, depth_fn);

        grads.insert(k.to_string(), (lp_plus - lp_minus) / (2.0 * h));
    }
    grads
}

/// Diagnostic: largest per-key |analytic - numeric| difference.
pub fn max_abs_grad_diff(analytic: &FeatureMap, numeric: &FeatureMap) -> f64 {
    let mut best = f64::NEG_INFINITY;
    for k in FEATURE_KEYS {
        let d = (analytic.get(k).copied().unwrap_or(0.0)
            - numeric.get(k).copied().unwrap_or(0.0))
            .abs();
        if d > best {
            best = d;
        }
    }
    best
}
