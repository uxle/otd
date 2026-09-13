//! Phase 055 — PPO clipped objective (design doc section 2.5, Promot
//! section 11).
//!
//! L_CLIP = E[ min( r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t ) ] with the
//! ratio r_t = exp(log_prob_new - log_prob_old) (log-space to avoid
//! overflow). Works with plain floats so it is fully testable in-sandbox.

/// Compute PPO-clip loss for one (state, action) sample.
///
/// Returns `(loss, ratio)`: the scalar loss contribution (negated
/// objective), and r_t = pi_new/pi_old for diagnostics/clipping stats.
pub fn ppo_clip_loss_single(
    log_prob_new: f64,
    log_prob_old: f64,
    advantage: f64,
    epsilon: f64,
) -> (f64, f64) {
    let ratio = (log_prob_new - log_prob_old).exp();
    let unclipped = ratio * advantage;
    let clipped = clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * advantage;
    // Take the minimum: when A > 0 we cap upside; when A < 0 we cap downside.
    let objective = unclipped.min(clipped);
    // Loss = -objective so gradient descent on the loss is ascent on the
    // objective.
    (-objective, ratio)
}

/// Batch version: mean PPO-clip loss over a trajectory.
/// Returns `(mean_loss, ratios)`.
pub fn ppo_clip_loss(
    log_probs_new: &[f64],
    log_probs_old: &[f64],
    advantages: &[f64],
    epsilon: f64,
) -> Result<(f64, Vec<f64>), String> {
    if !(log_probs_new.len() == log_probs_old.len() && log_probs_new.len() == advantages.len()) {
        return Err(
            "log_probs_new, log_probs_old, advantages must be same length".to_string()
        );
    }
    if advantages.is_empty() {
        return Err("advantages must be non-empty".to_string());
    }
    if !(0.0 < epsilon && epsilon < 1.0) {
        return Err(format!(
            "epsilon must be in (0,1), got {}",
            reasoning_common::py_float_str(epsilon)
        ));
    }

    let mut losses = Vec::with_capacity(advantages.len());
    let mut ratios = Vec::with_capacity(advantages.len());
    for i in 0..advantages.len() {
        let (loss, r) = ppo_clip_loss_single(log_probs_new[i], log_probs_old[i], advantages[i], epsilon);
        losses.push(loss);
        ratios.push(r);
    }
    let mean = losses.iter().sum::<f64>() / losses.len() as f64;
    Ok((mean, ratios))
}

/// Fraction of policy-ratio samples that were clipped. Near 0 means the
/// policy isn't moving; near 1 means clipping is doing all the work.
pub fn clip_fraction(ratios: &[f64], epsilon: f64) -> Result<f64, String> {
    if ratios.is_empty() {
        return Err("ratios must be non-empty".to_string());
    }
    let clipped = ratios
        .iter()
        .filter(|&&r| r < 1.0 - epsilon || r > 1.0 + epsilon)
        .count();
    Ok(clipped as f64 / ratios.len() as f64)
}

/// Python `max(lo, min(hi, x))` (order matters for NaN, keep it).
#[inline]
fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    if hi < x {
        hi
    } else if lo > x {
        lo
    } else {
        x
    }
}
