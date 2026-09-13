//! Phase 053 — Advantage estimation (design doc section 2.5).
//!
//! A_t = G_t - V(s_t) (baseline subtraction) and the TD error
//! delta_t = r_t + gamma*V(s_{t+1}) - V(s_t), the building block for GAE.

use crate::returns::discounted_returns_with_bootstrap;

/// delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)   (design doc 2.5)
///
/// `values` must have the same length as `rewards`; `last_value` = V(s_T)
/// is used only in the final step's bootstrap.
pub fn td_errors(
    rewards: &[f64],
    values: &[f64],
    last_value: f64,
    gamma: f64,
) -> Result<Vec<f64>, String> {
    if rewards.len() != values.len() {
        return Err(format!(
            "rewards length {} != values length {}",
            rewards.len(),
            values.len()
        ));
    }
    let t = rewards.len();
    let mut deltas = Vec::with_capacity(t);
    for i in 0..t {
        let v_next = if i + 1 < t { values[i + 1] } else { last_value };
        deltas.push(rewards[i] + gamma * v_next - values[i]);
    }
    Ok(deltas)
}

/// A_t = G_t - V(s_t) — baseline subtraction. No normalization applied
/// here; that's a separate, optional step (reward_norm).
pub fn advantages_from_returns(returns: &[f64], values: &[f64]) -> Result<Vec<f64>, String> {
    if returns.len() != values.len() {
        return Err(format!(
            "returns length {} != values length {}",
            returns.len(),
            values.len()
        ));
    }
    Ok(returns
        .iter()
        .zip(values.iter())
        .map(|(&g, &v)| g - v)
        .collect())
}

/// Convenience: compute G_t, then A_t = G_t - V(s_t), in one call.
pub fn advantages_from_rewards(
    rewards: &[f64],
    values: &[f64],
    last_value: f64,
    gamma: f64,
) -> Result<Vec<f64>, String> {
    let g = discounted_returns_with_bootstrap(rewards, last_value, gamma)?;
    advantages_from_returns(&g, values)
}
