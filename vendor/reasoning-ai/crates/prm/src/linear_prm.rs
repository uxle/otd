//! Phase 004 — Process Reward Model, v0 (pure Rust, no tensors) (Rust port
//! of `python/prm/linear_prm.py`).
//!
//! A linear model with a sigmoid squash, trained by hand-rolled SGD on
//! (features -> Q-value in [0,1]) pairs harvested from MCTS trees (design
//! doc 2.7: "no human step-labeling required — the symbolic verifier is the
//! label source"). Deliberately the simplest model that could work, to
//! prove the *training signal* (verified rollouts -> useful step scores) is
//! sound before spending neural-net time on a bigger model.

use crate::features::FeatureMap;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;

/// The exact sigmoid from the Python original.
fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        let z = (-x).exp();
        1.0 / (1.0 + z)
    } else {
        let z = x.exp();
        z / (1.0 + z)
    }
}

#[derive(Debug, Clone, Default, PartialEq)]
pub struct LinearPRM {
    pub weights: FeatureMap,
}

impl LinearPRM {
    /// Python `LinearPRM()` (empty weights dict).
    pub fn new() -> Self {
        LinearPRM {
            weights: FeatureMap::new(),
        }
    }

    fn score_raw(&self, features: &FeatureMap) -> f64 {
        features
            .iter()
            .map(|(k, v)| self.weights.get(k).copied().unwrap_or(0.0) * v)
            .sum()
    }

    /// Predicted P(this state leads to a verified-correct answer), in [0,1].
    pub fn predict(&self, features: &FeatureMap) -> f64 {
        sigmoid(self.score_raw(features))
    }

    /// One SGD step on binary-cross-entropy-style gradient. Returns the
    /// squared error before the update (for monitoring).
    pub fn train_step(&mut self, features: &FeatureMap, target: f64, lr: f64) -> f64 {
        let pred = self.predict(features);
        let error = pred - target;
        for (k, v) in features {
            // dL/dw for logistic loss with sigmoid output
            let g = error * v;
            let w = self.weights.get(k).copied().unwrap_or(0.0) - lr * g;
            self.weights.insert(k.clone(), w);
        }
        error * error
    }

    /// Python `train(examples, epochs=50, lr=0.1, rng=None)` — shuffles the
    /// examples each epoch and returns the per-epoch mean squared-error
    /// history.
    pub fn train(
        &mut self,
        examples: &[(FeatureMap, f64)],
        epochs: usize,
        lr: f64,
        rng: &mut StdRng,
    ) -> Vec<f64> {
        let mut history = Vec::with_capacity(epochs);
        for _ in 0..epochs {
            let mut shuffled = examples.to_vec();
            shuffled.shuffle(rng);
            let mut total_sq_err = 0.0;
            for (features, target) in &shuffled {
                total_sq_err += self.train_step(features, *target, lr);
            }
            history.push(total_sq_err / shuffled.len().max(1) as f64);
        }
        history
    }
}
