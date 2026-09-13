//! Phase 077 — Adaptive exploration (design doc 13/23; motivated directly
//! by rl/entropy_reg.py's own warning: "Too low too early -> premature
//! collapse", and rl/training_diagnostics.detect_entropy_collapse, which
//! until now nothing actually *acted on*).
//!
//! A small controller that watches the entropy/stagnation history Phase
//! 068 already computes and raises the rollout epsilon when the policy
//! has collapsed or stalled — then lowers it back down once entropy
//! recovers. Deliberately simple (proportional response, not a tuned
//! schedule) so its behavior is easy to unit-test exactly.

use crate::training_diagnostics::{detect_entropy_collapse, detect_policy_stagnation};

#[derive(Debug, Clone)]
pub struct AdaptiveExplorationController {
    pub base_epsilon: f64,
    pub boosted_epsilon: f64,
    pub entropy_collapse_threshold: f64,
    pub stagnation_window: usize,
    pub stagnation_grad_threshold: f64,

    pub entropy_history: Vec<f64>,
    pub grad_norm_history: Vec<f64>,
    pub epsilon_history: Vec<f64>,
}

impl Default for AdaptiveExplorationController {
    fn default() -> Self {
        AdaptiveExplorationController {
            base_epsilon: 0.1,
            boosted_epsilon: 0.4,
            entropy_collapse_threshold: 0.05,
            stagnation_window: 3,
            stagnation_grad_threshold: 1e-4,
            entropy_history: Vec::new(),
            grad_norm_history: Vec::new(),
            epsilon_history: Vec::new(),
        }
    }
}

/// Python `summary()` dict fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExplorationSummary {
    pub rounds_recorded: usize,
    pub rounds_boosted: usize,
    pub fraction_boosted: f64,
}

impl AdaptiveExplorationController {
    /// Feed in this round's measured entropy and gradient norm, get back
    /// the rollout epsilon to use for the NEXT round.
    pub fn record_and_get_epsilon(&mut self, entropy: f64, grad_norm: f64) -> f64 {
        self.entropy_history.push(entropy);
        self.grad_norm_history.push(grad_norm);

        // Both detectors take non-empty histories (we just appended).
        let collapsed = detect_entropy_collapse(&self.entropy_history, self.entropy_collapse_threshold)
            .expect("entropy_history non-empty after append");
        let stagnant = detect_policy_stagnation(
            &self.grad_norm_history,
            self.stagnation_grad_threshold,
            self.stagnation_window,
        );
        let epsilon = if collapsed || stagnant {
            self.boosted_epsilon
        } else {
            self.base_epsilon
        };
        self.epsilon_history.push(epsilon);
        epsilon
    }

    pub fn summary(&self) -> ExplorationSummary {
        let n_boosted = self
            .epsilon_history
            .iter()
            .filter(|&&e| e == self.boosted_epsilon)
            .count();
        ExplorationSummary {
            rounds_recorded: self.epsilon_history.len(),
            rounds_boosted: n_boosted,
            fraction_boosted: if self.epsilon_history.is_empty() {
                0.0
            } else {
                n_boosted as f64 / self.epsilon_history.len() as f64
            },
        }
    }
}
