//! Phase 052 — Discounted return calculation (design doc section 2.5).
//!
//! G_t = sum_{k=0}^{T-t-1} gamma^k * R_{t+k+1}, computed in O(T) via
//! backward recursion. Pure arithmetic — no neural component.

/// Compute G_t for each timestep t in [0, T-1]. `rewards[i]` is the
/// reward *received after* taking action i.
///
/// Errors (matching Python ValueError messages): empty rewards, or gamma
/// outside (0, 1].
pub fn discounted_returns(rewards: &[f64], gamma: f64) -> Result<Vec<f64>, String> {
    if rewards.is_empty() {
        return Err("rewards must be non-empty".to_string());
    }
    if !(0.0 < gamma && gamma <= 1.0) {
        return Err(format!(
            "gamma must be in (0, 1], got {}",
            reasoning_common::py_float_str(gamma)
        ));
    }
    let t = rewards.len();
    let mut g = vec![0.0; t];
    g[t - 1] = rewards[t - 1];
    for i in (0..t - 1).rev() {
        g[i] = rewards[i] + gamma * g[i + 1];
    }
    Ok(g)
}

/// Like [`discounted_returns`] but bootstraps from `last_value` V(s_T)
/// instead of assuming G_T = 0 — used when the trajectory is truncated
/// (search budget ran out) rather than truly finished.
pub fn discounted_returns_with_bootstrap(
    rewards: &[f64],
    last_value: f64,
    gamma: f64,
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
    let t = rewards.len();
    let mut g = vec![0.0; t];
    g[t - 1] = rewards[t - 1] + gamma * last_value;
    for i in (0..t - 1).rev() {
        g[i] = rewards[i] + gamma * g[i + 1];
    }
    Ok(g)
}
