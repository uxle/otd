//! Phase 092 — One distillation SGD step for the Phase 061 policy.
//!
//! Chain rule from Phase 091's per-action gradient to policy weights:
//! logit(s,a) = theta . phi(s,a), so
//!     d(L)/d(theta_k) = sum_a (p_student(a) - p_teacher(a)) * phi(s,a)_k
//! This is descent (not ascent, unlike the PPO/GRPO updates):
//! theta <- theta - lr * grad(L).

use crate::distillation::{distillation_grad_log_probs, soft_target_cross_entropy};
use crate::policy::{SoftmaxPolicy, FEATURE_KEYS};
use reasoning_search::{NtAction, NtState, NumberTargetDomain};

/// One supervised distillation sample: a state, its legal action set,
/// and the teacher's distribution aligned with `legal`.
#[derive(Debug, Clone)]
pub struct DistillSample {
    pub state: NtState,
    pub legal: Vec<NtAction>,
    /// aligned with `legal`
    pub teacher_distribution: Vec<f64>,
}

/// Diagnostics returned by [`distill_weight_update`] (Python dict fields).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistillUpdateDiagnostics {
    pub mean_loss_before_update: f64,
    pub grad_norm: f64,
}

/// One gradient-descent step on mean soft-target cross-entropy across
/// `samples`. Returns `(new_policy, diagnostics)` with the mean loss
/// reported before this update (so callers can plot a real loss curve).
pub fn distill_weight_update(
    policy: &SoftmaxPolicy,
    samples: &[DistillSample],
    domain: &NumberTargetDomain,
    target: f64,
    max_depth: usize,
    depth_fn: &dyn Fn(&NtState) -> usize,
    lr: f64,
) -> Result<(SoftmaxPolicy, DistillUpdateDiagnostics), String> {
    if samples.is_empty() {
        return Err("samples must be non-empty".to_string());
    }

    let mut total_grad = zero_weights();
    let mut total_loss = 0.0;
    for s in samples {
        let (p_student, feats_list) =
            policy.action_distribution(&s.state, &s.legal, domain, target, max_depth, depth_fn);
        if s.teacher_distribution.len() != s.legal.len() {
            return Err("teacher_distribution must align with legal actions".to_string());
        }
        total_loss += soft_target_cross_entropy(&s.teacher_distribution, &p_student)?;

        let per_action_grad =
            distillation_grad_log_probs(&s.teacher_distribution, &p_student)?;
        for (g_a, feats) in per_action_grad.iter().zip(feats_list.iter()) {
            for k in FEATURE_KEYS {
                *total_grad.get_mut(k).expect("key present") +=
                    g_a * feats.get(k).copied().unwrap_or(0.0);
            }
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
        *w -= lr * mean_grad.get(k).copied().unwrap_or(0.0);
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
    let diagnostics = DistillUpdateDiagnostics {
        mean_loss_before_update: total_loss / n as f64,
        grad_norm,
    };
    Ok((new_policy, diagnostics))
}

fn zero_weights() -> std::collections::HashMap<String, f64> {
    FEATURE_KEYS
        .iter()
        .map(|k| (k.to_string(), 0.0))
        .collect()
}
