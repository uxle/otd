//! Phase 059 — Reward normalization and reward clipping (design doc 2.5,
//! Promot sections 47–48).
//!
//! Two separate concepts kept as separate functions: (1) hard reward
//! clipping to [-clip_range, +clip_range]; (2) running-statistics
//! normalization (Welford, one batch at a time) as used in PPO
//! implementations like OpenAI Baselines.

// NOTE: the Python source for `clip_rewards` contains a transcription
// typo ("return ax(-clip_range, min(clip_range, r)) for r in rewards]"),
// which is not even syntactically valid Python; the intended body
// `[max(-clip_range, min(clip_range, r)) for r in rewards]` (matching the
// docstring: "Hard-clip each reward to [-clip_range, +clip_range]") is
// what's ported here.

/// Hard-clip each reward to [-clip_range, +clip_range].
/// clip_range must be positive.
pub fn clip_rewards(rewards: &[f64], clip_range: f64) -> Result<Vec<f64>, String> {
    if clip_range <= 0.0 {
        return Err(format!(
            "clip_range must be > 0, got {}",
            reasoning_common::py_float_str(clip_range)
        ));
    }
    Ok(rewards
        .iter()
        .map(|&r| py_max(-clip_range, py_min(clip_range, r)))
        .collect())
}

/// Same operation applied to advantage estimates — sometimes also clipped
/// before passing into the PPO loss to tame outlier gradients.
pub fn clip_advantages(advantages: &[f64], clip_range: f64) -> Result<Vec<f64>, String> {
    clip_rewards(advantages, clip_range)
}

/// Online (Welford's) algorithm for tracking mean and variance of a
/// scalar stream. Updates one batch at a time.
#[derive(Debug, Clone)]
pub struct RunningMeanStd {
    /// running mean
    pub mean: f64,
    /// running variance
    pub var: f64,
    /// total number of samples seen
    pub count: f64,
    /// floor for std to avoid division by zero
    pub epsilon: f64,
}

impl Default for RunningMeanStd {
    fn default() -> Self {
        RunningMeanStd::new(1e-8)
    }
}

impl RunningMeanStd {
    pub fn new(epsilon: f64) -> Self {
        RunningMeanStd {
            mean: 0.0,
            var: 1.0,
            count: 0.0,
            epsilon,
        }
    }

    pub fn std(&self) -> f64 {
        (self.var + self.epsilon).sqrt()
    }

    /// Update running stats with a new batch of scalars (no-op on empty).
    pub fn update(&mut self, values: &[f64]) {
        if values.is_empty() {
            return;
        }
        let n = values.len() as f64;
        let batch_mean = values.iter().sum::<f64>() / n;
        let batch_var = values
            .iter()
            .map(|&v| (v - batch_mean) * (v - batch_mean))
            .sum::<f64>()
            / (n.max(1.0));

        let total = self.count + n;
        let delta = batch_mean - self.mean;
        let new_mean = self.mean + delta * n / total;
        // Parallel variance formula (Chan et al.):
        let new_var =
            (self.var * self.count + batch_var * n + delta * delta * self.count * n / total)
                / total;
        self.mean = new_mean;
        self.var = new_var;
        self.count = total;
    }

    /// Subtract running mean, divide by running std.
    pub fn normalize(&self, values: &[f64]) -> Vec<f64> {
        let (mu, sigma) = (self.mean, self.std());
        values.iter().map(|&v| (v - mu) / sigma).collect()
    }

    /// Update stats from this batch, then normalize. Update-first order is
    /// intentional — avoids the first batch being normalized against
    /// stale zeros.
    pub fn normalize_and_update(&mut self, values: &[f64]) -> Vec<f64> {
        self.update(values);
        self.normalize(values)
    }
}

/// Normalize a mini-batch of advantages to zero mean, unit variance —
/// the simpler *batch-level* normalization (no running stats) common in
/// PPO for the advantages specifically.
pub fn normalize_advantages(advantages: &[f64], eps: f64) -> Result<Vec<f64>, String> {
    if advantages.is_empty() {
        return Err("advantages must be non-empty".to_string());
    }
    let n = advantages.len() as f64;
    let mean = advantages.iter().sum::<f64>() / n;
    let var = advantages
        .iter()
        .map(|&a| (a - mean) * (a - mean))
        .sum::<f64>()
        / n;
    let std = (var + eps).sqrt();
    Ok(advantages.iter().map(|&a| (a - mean) / std).collect())
}

#[inline]
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

#[inline]
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}
