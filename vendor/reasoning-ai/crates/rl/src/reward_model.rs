//! Phase 051 — Modular reward composition (design doc section 2.6, Promot
//! section 12).
//!
//! A single scalar reward assembled from named, independently-inspectable
//! components — never an opaque number. Deliberately no field anywhere
//! for trajectory length / step count / token count: the schema itself
//! makes a length-based reward impossible to express. This module does
//! not decide what counts as "correct" — that stays with the verifier and
//! disagreement detector; this only combines already-graded signals.

/// Every weight is a non-negative magnitude; compose_reward applies the
/// sign (bonus vs. penalty), the caller never does. Defaults are chosen
/// so a single serious failure mode (reward_hacking_detected) outweighs
/// the entire positive side of an otherwise-perfect trajectory.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardWeights {
    pub correctness: f64,
    pub partial_credit: f64,
    pub verification_pass: f64,
    pub logical_consistency: f64,
    pub tool_use_bonus: f64,
    pub contradiction_penalty: f64,
    pub hallucination_penalty: f64,
    pub invalid_reasoning_penalty: f64,
    pub reward_hacking_penalty: f64,
}

impl Default for RewardWeights {
    fn default() -> Self {
        DEFAULT_WEIGHTS
    }
}

impl RewardWeights {
    /// Python `RewardWeights(...)` with validation from `__post_init__`
    /// (all weights must be >= 0). Struct literals skip validation, so
    /// use this constructor (or `validate`) when weights come from
    /// untrusted input.
    pub fn checked(
        correctness: f64,
        partial_credit: f64,
        verification_pass: f64,
        logical_consistency: f64,
        tool_use_bonus: f64,
        contradiction_penalty: f64,
        hallucination_penalty: f64,
        invalid_reasoning_penalty: f64,
        reward_hacking_penalty: f64,
    ) -> Result<Self, String> {
        let w = RewardWeights {
            correctness,
            partial_credit,
            verification_pass,
            logical_consistency,
            tool_use_bonus,
            contradiction_penalty,
            hallucination_penalty,
            invalid_reasoning_penalty,
            reward_hacking_penalty,
        };
        w.validate()?;
        Ok(w)
    }

    /// `__post_init__` check: every weight must be non-negative.
    pub fn validate(&self) -> Result<(), String> {
        for (name, v) in [
            ("correctness", self.correctness),
            ("partial_credit", self.partial_credit),
            ("verification_pass", self.verification_pass),
            ("logical_consistency", self.logical_consistency),
            ("tool_use_bonus", self.tool_use_bonus),
            ("contradiction_penalty", self.contradiction_penalty),
            ("hallucination_penalty", self.hallucination_penalty),
            ("invalid_reasoning_penalty", self.invalid_reasoning_penalty),
            ("reward_hacking_penalty", self.reward_hacking_penalty),
        ] {
            if v < 0.0 {
                return Err(format!(
                    "RewardWeights.{} must be >= 0, got {}",
                    name,
                    reasoning_common::py_float_str(v)
                ));
            }
        }
        Ok(())
    }
}

pub const DEFAULT_WEIGHTS: RewardWeights = RewardWeights {
    correctness: 1.0,
    partial_credit: 0.3,
    verification_pass: 0.2,
    logical_consistency: 0.2,
    tool_use_bonus: 0.05,
    contradiction_penalty: 0.5,
    hallucination_penalty: 0.5,
    invalid_reasoning_penalty: 0.3,
    reward_hacking_penalty: 1.6,
};

/// Already-graded signals for ONE trajectory or step. Produced by
/// upstream verifiers/critics — never fabricated inside this module.
/// Note what is intentionally absent: num_steps, num_tokens, length, or
/// anything duration-shaped — that absence is the enforcement mechanism
/// for "don't reward length".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RewardComponents {
    pub correct: bool,
    /// fraction of sub-answers right, in [0,1]
    pub partial_credit: f64,
    pub verification_passed: bool,
    pub logically_consistent: bool,
    pub used_tool_usefully: bool,
    pub contradiction_detected: bool,
    pub hallucination_detected: bool,
    pub invalid_reasoning_detected: bool,
    pub reward_hacking_detected: bool,
}

impl Default for RewardComponents {
    fn default() -> Self {
        RewardComponents {
            correct: false,
            partial_credit: 0.0,
            verification_passed: false,
            logically_consistent: true,
            used_tool_usefully: false,
            contradiction_detected: false,
            hallucination_detected: false,
            invalid_reasoning_detected: false,
            reward_hacking_detected: false,
        }
    }
}

impl RewardComponents {
    /// `__post_init__` check: partial_credit must be in [0,1]. Struct
    /// literals skip validation; use this when components come from
    /// untrusted input.
    pub fn validate(&self) -> Result<(), String> {
        if !(0.0 <= self.partial_credit && self.partial_credit <= 1.0) {
            return Err(format!(
                "partial_credit must be in [0,1], got {}",
                reasoning_common::py_float_str(self.partial_credit)
            ));
        }
        Ok(())
    }
}

/// Full form: `compose_reward(components, weights, clip_range)`.
///
/// reward = w1*correct + w2*partial_credit + w3*verification_pass
///        + w4*logical_consistency + w5*tool_use
///        - w6*contradiction - w7*hallucination - w8*invalid_reasoning
///        - w9*reward_hacking
/// then clipped to `clip_range` (Python default `(-2.0, 2.0)`,
/// `None` disables clipping).
pub fn compose_reward_full(
    components: &RewardComponents,
    weights: &RewardWeights,
    clip_range: Option<(f64, f64)>,
) -> f64 {
    let b = |x: bool| if x { 1.0 } else { 0.0 };
    let mut r = 0.0;
    r += weights.correctness * b(components.correct);
    r += weights.partial_credit * components.partial_credit;
    r += weights.verification_pass * b(components.verification_passed);
    r += weights.logical_consistency * b(components.logically_consistent);
    r += weights.tool_use_bonus * b(components.used_tool_usefully);
    r -= weights.contradiction_penalty * b(components.contradiction_detected);
    r -= weights.hallucination_penalty * b(components.hallucination_detected);
    r -= weights.invalid_reasoning_penalty * b(components.invalid_reasoning_detected);
    r -= weights.reward_hacking_penalty * b(components.reward_hacking_detected);
    if let Some((lo, hi)) = clip_range {
        r = py_max(lo, py_min(hi, r));
    }
    r
}

/// `compose_reward(components)` — default weights and clip range.
pub fn compose_reward(components: &RewardComponents) -> f64 {
    compose_reward_full(components, &DEFAULT_WEIGHTS, Some((-2.0, 2.0)))
}

/// `compose_reward(components, weights=..., clip_range=...)` variants.
pub fn compose_reward_with_weights(
    components: &RewardComponents,
    weights: &RewardWeights,
) -> f64 {
    compose_reward_full(components, weights, Some((-2.0, 2.0)))
}

/// `compose_reward(components, clip_range=...)` — default weights.
pub fn compose_reward_with_clip(
    components: &RewardComponents,
    clip_range: Option<(f64, f64)>,
) -> f64 {
    compose_reward_full(components, &DEFAULT_WEIGHTS, clip_range)
}

/// Adapter for the common case: nothing fancy detected, just plug in the
/// verifier's `.passed` boolean directly.
pub fn reward_from_verification_result(verification_passed: bool) -> f64 {
    let components = RewardComponents {
        correct: verification_passed,
        verification_passed,
        ..Default::default()
    };
    compose_reward(&components)
}

/// Weights-parameterized form of [`reward_from_verification_result`].
pub fn reward_from_verification_result_with_weights(
    verification_passed: bool,
    weights: &RewardWeights,
) -> f64 {
    let components = RewardComponents {
        correct: verification_passed,
        verification_passed,
        ..Default::default()
    };
    compose_reward_with_weights(&components, weights)
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
