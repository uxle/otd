//! Phase 007 — PRM-guided search (Rust port of
//! `python/prm/guided_policy.py`).
//!
//! Phase 003's MCTS used a uniform-random rollout policy — proof that
//! search+verification alone can solve problems, with zero learned guidance.
//! This replaces it with a policy that uses the trained PRM (Phase 004) to
//! prefer actions whose resulting state it predicts is more likely to reach
//! a verified answer — the first place a "trained" component actually
//! changes search behavior.

use crate::features::extract_features;
use crate::linear_prm::LinearPRM;
use rand::rngs::StdRng;
use rand::Rng;
use reasoning_search::{Domain, NtAction, NtState, NumberTargetDomain, RolloutPolicy};
use std::rc::Rc;

/// Python `make_prm_guided_policy(...)` returned a closure
/// `(state, legal) -> action` for MCTS; the Rust MCTS takes a
/// `RolloutPolicy` impl, so the same state lives here as a struct.
///
/// With probability epsilon, act uniformly at random (keeps some
/// exploration so a mediocre early PRM can't permanently trap search in a
/// bad region). Otherwise, softmax over PRM-predicted value of the
/// resulting states. `depth_fn` computes a state's depth since not every
/// domain's state object carries it directly (e.g. tuple-based states).
pub struct PrmGuidedPolicy {
    prm: LinearPRM,
    /// Holds a domain clone so `apply` works like Python's bound
    /// `domain.apply` argument.
    domain: NumberTargetDomain,
    target: f64,
    max_depth: usize,
    depth_fn: Rc<dyn Fn(&NtState) -> usize>,
    epsilon: f64,
}

impl PrmGuidedPolicy {
    /// Python `make_prm_guided_policy(prm, apply_fn, target, max_depth,
    /// depth_fn, epsilon=0.15, rng=None)` — the apply_fn came from the
    /// domain, so this takes the domain itself.
    pub fn new(
        prm: LinearPRM,
        domain: NumberTargetDomain,
        target: f64,
        max_depth: usize,
        depth_fn: Rc<dyn Fn(&NtState) -> usize>,
        epsilon: f64,
    ) -> Self {
        PrmGuidedPolicy {
            prm,
            domain,
            target,
            max_depth,
            depth_fn,
            epsilon,
        }
    }
}

impl RolloutPolicy<NtState, NtAction> for PrmGuidedPolicy {
    fn choose(&mut self, rng: &mut StdRng, state: &NtState, legal: &[NtAction]) -> NtAction {
        // callers guarantee legal is non-empty
        if rng.gen::<f64>() < self.epsilon || legal.len() == 1 {
            return legal[rng.gen_range(0..legal.len())].clone();
        }
        let mut scores: Vec<f64> = Vec::with_capacity(legal.len());
        for action in legal {
            let next_state = self.domain.apply(state, action);
            let depth = (self.depth_fn)(&next_state);
            let feats = extract_features(&next_state, self.target, self.max_depth, depth);
            scores.push(self.prm.predict(&feats));
        }
        // softmax selection (temperature=1) over predicted scores
        let m = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let total: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|e| e / total).collect();
        let r = rng.gen::<f64>();
        let mut cum = 0.0;
        for (action, p) in legal.iter().zip(probs.iter()) {
            cum += p;
            if r <= cum {
                return action.clone();
            }
        }
        legal[legal.len() - 1].clone()
    }
}
