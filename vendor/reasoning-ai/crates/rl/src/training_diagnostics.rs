//! Phase 068 — Training diagnostics: entropy collapse and KL-drift
//! monitoring (Promot section 13/23; rl/entropy_reg.py's own docstring
//! flagged this as a real risk: "Too low too early -> premature
//! collapse").
//!
//! Small-model RL with a 5-parameter linear-softmax policy can collapse
//! to a near-deterministic policy very quickly. These functions turn
//! that risk into something measured every round, not assumed away.

// NOTE: the Python source for `policy_entropy_on_probe_states` and
// `kl_from_reference` has transcription typos in the list comprehensions
// ("ath.log(max(p, 1e-12)) for p in probs]"); the intended
// `[math.log(max(p, 1e-12)) for p in probs]` is what's ported here.

use crate::entropy_reg::{entropy_from_logprobs, max_entropy};
use crate::kl_penalty::kl_divergence_from_logprobs;
use crate::policy::SoftmaxPolicy;
use reasoning_search::{NtAction, NtState, NumberTargetDomain};

/// Mean entropy of the policy's action distribution across a fixed set
/// of probe `(state, legal_actions)` pairs, normalized by each state's
/// max possible entropy (log(num legal actions)) so results are
/// comparable across states with different branching factors. Returns a
/// value in [0, 1].
pub fn policy_entropy_on_probe_states(
    policy: &SoftmaxPolicy,
    probe_states: &[(NtState, Vec<NtAction>)],
    domain: &NumberTargetDomain,
    target: f64,
    max_depth: usize,
    depth_fn: &dyn Fn(&NtState) -> usize,
) -> Result<f64, String> {
    if probe_states.is_empty() {
        return Err("probe_states must be non-empty".to_string());
    }
    let mut normalized_entropies = Vec::with_capacity(probe_states.len());
    for (state, legal) in probe_states {
        let (probs, _) = policy.action_distribution(state, legal, domain, target, max_depth, depth_fn);
        let log_probs: Vec<f64> = probs.iter().map(|&p| p.max(1e-12).ln()).collect();
        let ent = entropy_from_logprobs(&log_probs)?;
        let max_ent = max_entropy(legal.len())?;
        normalized_entropies.push(if max_ent > 0.0 { ent / max_ent } else { 1.0 });
    }
    Ok(normalized_entropies.iter().sum::<f64>() / normalized_entropies.len() as f64)
}

/// Mean KL(current || reference) across probe states — how far the
/// policy has drifted from a reference (e.g. its round-0 initial
/// weights).
pub fn kl_from_reference(
    current_policy: &SoftmaxPolicy,
    reference_policy: &SoftmaxPolicy,
    probe_states: &[(NtState, Vec<NtAction>)],
    domain: &NumberTargetDomain,
    target: f64,
    max_depth: usize,
    depth_fn: &dyn Fn(&NtState) -> usize,
) -> Result<f64, String> {
    if probe_states.is_empty() {
        return Err("probe_states must be non-empty".to_string());
    }
    let mut kls = Vec::with_capacity(probe_states.len());
    for (state, legal) in probe_states {
        let (probs_new, _) =
            current_policy.action_distribution(state, legal, domain, target, max_depth, depth_fn);
        let (probs_ref, _) = reference_policy.action_distribution(
            state,
            legal,
            domain,
            target,
            max_depth,
            depth_fn,
        );
        let log_new: Vec<f64> = probs_new.iter().map(|&p| p.max(1e-12).ln()).collect();
        let log_ref: Vec<f64> = probs_ref.iter().map(|&p| p.max(1e-12).ln()).collect();
        kls.push(kl_divergence_from_logprobs(&log_new, &log_ref)?);
    }
    Ok(kls.iter().sum::<f64>() / kls.len() as f64)
}

/// True if normalized entropy has dropped below `threshold` (i.e. the
/// policy has become nearly deterministic) at the most recent
/// measurement.
pub fn detect_entropy_collapse(entropy_history: &[f64], threshold: f64) -> Result<bool, String> {
    if entropy_history.is_empty() {
        return Err("entropy_history must be non-empty".to_string());
    }
    Ok(*entropy_history.last().expect("non-empty") < threshold)
}

/// True if the last `window` gradient norms are all below `threshold` —
/// the policy has stopped changing, whether because it converged or
/// because updates are being fully clipped away.
pub fn detect_policy_stagnation(
    grad_norm_history: &[f64],
    threshold: f64,
    window: usize,
) -> bool {
    if grad_norm_history.len() < window {
        return false;
    }
    grad_norm_history[grad_norm_history.len() - window..]
        .iter()
        .all(|&g| g < threshold)
}
