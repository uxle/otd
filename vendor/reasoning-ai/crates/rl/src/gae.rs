//! Phase 054 — Generalized Advantage Estimation (design doc section 2.5).
//!
//! A_t^GAE = sum_l (gamma*lambda)^l * delta_{t+l}, computed in O(T) via
//! backward recursion. lambda=0 -> pure 1-step TD; lambda=1 -> MC return
//! minus baseline.

use crate::advantage::td_errors;

/// Compute GAE advantages for a single trajectory.
///
/// `rewards[i]` = R_{t+1}, `values[i]` = V(s_t), `last_value` = V(s_T)
/// (0.0 if the trajectory is terminal).
pub fn gae(
    rewards: &[f64],
    values: &[f64],
    last_value: f64,
    gamma: f64,
    lam: f64,
) -> Result<Vec<f64>, String> {
    if rewards.is_empty() {
        return Err("rewards must be non-empty".to_string());
    }
    if !(0.0 < gamma && gamma <= 1.0) {
        return Err(format!(
            "gamma must be in (0, 1], got {}",
            reasoning_common::py_float_str(gamma)
        ));
    }
    if !(0.0 <= lam && lam <= 1.0) {
        return Err(format!(
            "lam must be in [0, 1], got {}",
            reasoning_common::py_float_str(lam)
        ));
    }

    let deltas = td_errors(rewards, values, last_value, gamma)?;
    let t = deltas.len();
    let mut a = vec![0.0; t];
    a[t - 1] = deltas[t - 1];
    let coef = gamma * lam;
    for i in (0..t - 1).rev() {
        a[i] = deltas[i] + coef * a[i + 1];
    }
    Ok(a)
}

/// Returns `(advantages, value_targets)` in one call.
/// value_targets[t] = A_t + V(s_t) — the regression target for the value
/// head; more numerically stable than computing discounted returns again.
pub fn gae_returns(
    rewards: &[f64],
    values: &[f64],
    last_value: f64,
    gamma: f64,
    lam: f64,
) -> Result<(Vec<f64>, Vec<f64>), String> {
    let advs = gae(rewards, values, last_value, gamma, lam)?;
    let targets = advs
        .iter()
        .zip(values.iter())
        .map(|(&a, &v)| a + v)
        .collect();
    Ok((advs, targets))
}
