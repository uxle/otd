//! Phase 056 — GRPO: Group Relative Policy Optimization (design doc 2.5,
//! Promot sections 11 and 43).
//!
//! GRPO removes the need for a separate value-function critic by using
//! *group statistics* as the baseline: A_i = (r_i - mean(r)) / (std(r) +
//! eps), then the same PPO-clip objective over the N grouped candidates.
//! The verifier still provides the reward signal.

/// A_i = (r_i - mean(r)) / (std(r) + eps)
///
/// The group is a set of N sampled solutions to the *same* problem.
/// Normalizing by group std keeps advantage magnitudes comparable across
/// problems of different difficulty.
pub fn group_advantages(rewards: &[f64], eps: f64) -> Result<Vec<f64>, String> {
    if rewards.is_empty() {
        return Err("rewards must be non-empty".to_string());
    }
    let n = rewards.len() as f64;
    let mean_r = rewards.iter().sum::<f64>() / n;
    let var_r = rewards.iter().map(|&r| (r - mean_r) * (r - mean_r)).sum::<f64>() / n;
    let std_r = var_r.sqrt();
    Ok(rewards
        .iter()
        .map(|&r| (r - mean_r) / (std_r + eps))
        .collect())
}

/// Full GRPO objective for one problem with N grouped candidates.
///
/// Returns `(mean_loss, advantages, ratios)`:
/// mean_loss is the scalar to minimise (negated mean GRPO objective),
/// advantages the group-normalized advantage per candidate, ratios the
/// policy ratio r_t_i per candidate (for diagnostics).
pub fn grpo_loss(
    log_probs_new: &[f64],
    log_probs_old: &[f64],
    rewards: &[f64],
    epsilon: f64,
    eps_norm: f64,
) -> Result<(f64, Vec<f64>, Vec<f64>), String> {
    let n = rewards.len();
    if !(log_probs_new.len() == log_probs_old.len() && log_probs_new.len() == n) {
        return Err(
            "log_probs_new, log_probs_old, rewards must be same length".to_string()
        );
    }
    if n < 2 {
        return Err(
            "GRPO requires at least 2 candidates (group statistics need variance)".to_string()
        );
    }
    if !(0.0 < epsilon && epsilon < 1.0) {
        return Err(format!(
            "epsilon must be in (0,1), got {}",
            reasoning_common::py_float_str(epsilon)
        ));
    }

    let advs = group_advantages(rewards, eps_norm)?;
    let mut losses = Vec::with_capacity(n);
    let mut ratios = Vec::with_capacity(n);
    for i in 0..n {
        let ratio = (log_probs_new[i] - log_probs_old[i]).exp();
        let unclipped = ratio * advs[i];
        let clipped =
            py_clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advs[i];
        losses.push(-unclipped.min(clipped)); // negate: we minimise loss
        ratios.push(ratio);
    }

    let mean = losses.iter().sum::<f64>() / n as f64;
    Ok((mean, advs, ratios))
}

/// When every candidate in a group is correct (all rewards = 1.0) or all
/// wrong (all = 0.0), group std ~ 0 and all advantages ~ 0, so the
/// gradient is ~0 — correct behavior: nothing to learn from a batch
/// already solved 100% (or 0%) of the time.
pub fn grpo_all_correct_zero_gradient(rewards: &[f64], eps: f64) -> Result<bool, String> {
    let advs = group_advantages(rewards, eps)?;
    Ok(advs.iter().all(|&a| a.abs() < 1.0)
        && (advs.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
            - advs.iter().cloned().fold(f64::INFINITY, f64::min))
            < 1e-4)
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
