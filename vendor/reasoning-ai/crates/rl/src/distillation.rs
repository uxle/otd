//! Phase 091 — Knowledge distillation math (design doc section 19:
//! "Train using hard targets, soft targets, temperature scaling, KL
//! divergence, reasoning trajectories, verified solutions. Only distill
//! verified useful behavior.").
//!
//! The teacher is MCTS's own visit-count distribution (already a strictly
//! *better* policy than its raw rollout prior at every searched state).
//! Loss: temperature-scaled soft-target cross-entropy,
//!
//! L = -sum_a p_teacher(a) * log(p_student(a))
//!
//! with T=1 recovering ordinary cross-entropy; T>1 softens a very peaked
//! teacher (e.g. a visit-count distribution with one dominant action).
// NOTE: the Python source's `softmax` body has a transcription typo
// ("exps = ath.exp(x - m) for x in scaled]"); the intended
// `[math.exp(x - m) for x in scaled]` (per the standard softmax and the
// tests) is what's ported here.

pub fn softmax(logits: &[f64], temperature: f64) -> Result<Vec<f64>, String> {
    if temperature <= 0.0 {
        return Err(format!(
            "temperature must be > 0, got {}",
            reasoning_common::py_float_str(temperature)
        ));
    }
    let scaled: Vec<f64> = logits.iter().map(|&x| x / temperature).collect();
    let m = scaled
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let exps: Vec<f64> = scaled.iter().map(|&x| (x - m).exp()).collect();
    let total: f64 = exps.iter().sum();
    Ok(exps.iter().map(|&e| e / total).collect())
}

/// MCTS visit counts -> a probability distribution, the standard
/// AlphaZero "improved policy" target. Uses counts directly — the
/// temperature exponentiates the *counts*: pi(a) ~ N(a)^(1/T), exactly
/// the AlphaZero convention (T=1 proportional to visits, T->0 approaches
/// argmax/greedy).
///
/// Note: counts are `i64` (not `usize`) so the Python
/// "visit_counts must be non-negative" error path stays testable.
pub fn visit_counts_to_teacher_distribution(
    visit_counts: &[i64],
    temperature: f64,
) -> Result<Vec<f64>, String> {
    if visit_counts.is_empty() {
        return Err("visit_counts must be non-empty".to_string());
    }
    if temperature <= 0.0 {
        return Err(format!(
            "temperature must be > 0, got {}",
            reasoning_common::py_float_str(temperature)
        ));
    }
    if visit_counts.iter().any(|&v| v < 0) {
        return Err("visit_counts must be non-negative".to_string());
    }
    let powered: Vec<f64> = visit_counts
        .iter()
        .map(|&v| if v > 0 { (v as f64).powf(1.0 / temperature) } else { 0.0 })
        .collect();
    let total: f64 = powered.iter().sum();
    if total == 0.0 {
        let n = visit_counts.len();
        return Ok(vec![1.0 / n as f64; n]); // no visits at all -> uniform, not undefined
    }
    Ok(powered.iter().map(|&p| p / total).collect())
}

/// L = -sum_a p_teacher(a) * log(p_student(a)). Floors the log-argument
/// at 1e-12 to avoid -inf when the student assigns near-zero probability
/// to something the teacher wants weight on — exactly the situation this
/// loss is meant to push against.
pub fn soft_target_cross_entropy(p_teacher: &[f64], p_student: &[f64]) -> Result<f64, String> {
    if p_teacher.len() != p_student.len() {
        return Err("p_teacher and p_student must be the same length".to_string());
    }
    Ok(-p_teacher
        .iter()
        .zip(p_student.iter())
        .map(|(&pt, &ps)| pt * ps.max(1e-12).ln())
        .sum::<f64>())
}

/// d(L)/d(student_logit_a) for a softmax student = p_student(a) -
/// p_teacher(a). Standard softmax-cross-entropy gradient identity (same
/// family as Phase 062's log-derivative trick, derived for a target
/// distribution instead of a single one-hot action). Verified against
/// finite differences in the test file, per house convention.
pub fn distillation_grad_log_probs(p_teacher: &[f64], p_student: &[f64]) -> Result<Vec<f64>, String> {
    if p_teacher.len() != p_student.len() {
        return Err("p_teacher and p_student must be the same length".to_string());
    }
    Ok(p_teacher
        .iter()
        .zip(p_student.iter())
        .map(|(&pt, &ps)| ps - pt)
        .collect())
}
