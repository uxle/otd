//! Phase 057 — KL divergence penalty (design doc 2.5, Promot section 45).
//!
//! Forward KL(pi_theta || pi_ref) computed from log-probabilities, plus
//! the common single-action trajectory-level KL estimate used in
//! PPO-style RL.

/// D_KL(pi_new || pi_ref) = sum_a exp(log_p_new_a) * (log_p_new_a -
/// log_p_ref_a). The lists are over discrete actions in a distribution
/// (not trajectory steps); they should sum to 1 in probability space.
pub fn kl_divergence_from_logprobs(log_p_new: &[f64], log_p_ref: &[f64]) -> Result<f64, String> {
    if log_p_new.len() != log_p_ref.len() {
        return Err("log_p_new and log_p_ref must have the same length".to_string());
    }
    Ok(log_p_new
        .iter()
        .zip(log_p_ref.iter())
        .map(|(&lp_new, &lp_ref)| lp_new.exp() * (lp_new - lp_ref))
        .sum())
}

/// Per-sample KL estimate summed over a trajectory (common in PPO):
/// KL_traj = sum_t exp(log_new_t) * (log_new_t - log_ref_t). An estimate,
/// not an exact KL — unbiased in expectation and cheap.
pub fn kl_sample_trajectory(
    log_probs_new: &[f64],
    log_probs_ref: &[f64],
) -> Result<f64, String> {
    if log_probs_new.len() != log_probs_ref.len() {
        return Err("log_probs_new and log_probs_ref must have the same length".to_string());
    }
    Ok(log_probs_new
        .iter()
        .zip(log_probs_ref.iter())
        .map(|(&lp_new, &lp_ref)| lp_new.exp() * (lp_new - lp_ref))
        .sum())
}

/// L_total = L_policy - beta * KL(pi_new || pi_ref). The subtraction
/// because L_policy is already negated (we minimise it). Returns
/// `(total_loss, kl_value)` so callers can log the KL.
pub fn apply_kl_penalty(
    policy_loss: f64,
    log_probs_new: &[f64],
    log_probs_ref: &[f64],
    beta: f64,
) -> Result<(f64, f64), String> {
    if beta < 0.0 {
        return Err(format!(
            "beta must be >= 0, got {}",
            reasoning_common::py_float_str(beta)
        ));
    }
    let kl = kl_sample_trajectory(log_probs_new, log_probs_ref)?;
    Ok((policy_loss + beta * kl, kl))
}
