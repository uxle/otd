//! Phase 058 — Entropy regularization (design doc 2.5, Promot section 46).
//!
//! H(pi) = -sum_a exp(log_p_a) * log_p_a, added to the total objective as
//! a bonus (subtracted from the loss) so the policy stays exploratory
//! rather than collapsing prematurely.

/// H(pi) = -sum_a pi(a) * log(pi(a)) from action log-probabilities.
pub fn entropy_from_logprobs(log_probs: &[f64]) -> Result<f64, String> {
    if log_probs.is_empty() {
        return Err("log_probs must be non-empty".to_string());
    }
    Ok(-log_probs.iter().map(|&lp| lp.exp() * lp).sum::<f64>())
}

/// Maximum possible entropy for n_actions: H(uniform) = log(n_actions).
/// Useful for normalizing entropy to [0,1] in dashboards.
pub fn max_entropy(n_actions: usize) -> Result<f64, String> {
    if n_actions < 1 {
        return Err(format!("n_actions must be >= 1, got {}", n_actions));
    }
    Ok(if n_actions > 1 {
        (n_actions as f64).ln()
    } else {
        0.0
    })
}

/// Mean entropy over all steps in a trajectory — one diagnostic scalar
/// for "how exploratory was this trajectory on average?"
pub fn entropy_trajectory_mean(log_probs_per_step: &[Vec<f64>]) -> Result<f64, String> {
    if log_probs_per_step.is_empty() {
        return Err("log_probs_per_step must be non-empty".to_string());
    }
    let total: f64 = log_probs_per_step
        .iter()
        .map(|lp| entropy_from_logprobs(lp))
        .sum::<Result<f64, String>>()?;
    Ok(total / log_probs_per_step.len() as f64)
}

/// L_total = L_policy - beta_ent * H(pi). We subtract: minimising the
/// loss means maximising H (exploration bonus). Returns
/// `(total_loss, entropy_value)` so callers can log entropy.
pub fn apply_entropy_bonus(
    policy_loss: f64,
    log_probs: &[f64],
    beta_ent: f64,
) -> Result<(f64, f64), String> {
    if beta_ent < 0.0 {
        return Err(format!(
            "beta_ent must be >= 0, got {}",
            reasoning_common::py_float_str(beta_ent)
        ));
    }
    let ent = entropy_from_logprobs(log_probs)?;
    Ok((policy_loss - beta_ent * ent, ent))
}
