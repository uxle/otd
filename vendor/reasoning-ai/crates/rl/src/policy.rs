//! Phase 061 — Trainable softmax policy (design doc 2.5 policy pi_theta).
//!
//! A genuine parametric policy pi_theta(a|s) with theta a plain weight
//! dict over the SAME feature names prm/features.py produces (bias,
//! num_remaining, min_diff_to_target_norm, depth_norm, has_exact_match) —
//! mechanically a 5-parameter softmax policy, small enough to train on
//! CPU and unit-test exhaustively:
//!
//! score(s, a)  = theta . phi(s, a)            (linear in theta)
//! pi(a|s)      = softmax_a( score(s, a) )
//!
//! Uses the PRM crate's feature extractor so it plugs into the existing
//! MCTS rollout_policy interface for free (Phase 065).
use rand::rngs::StdRng;
use rand::Rng;
use reasoning_prm::extract_features;
use reasoning_search::{Domain, NtAction, NtState, NumberTargetDomain};

/// Feature-vector type (same as the PRM crate: `HashMap<String, f64>`).
pub type FeatureMap = reasoning_prm::FeatureMap;

/// The feature names `prm/features.py` produces, in its emission order.
pub const FEATURE_KEYS: [&str; 5] = [
    "bias",
    "num_remaining",
    "min_diff_to_target_norm",
    "depth_norm",
    "has_exact_match",
];

#[derive(Debug, Clone)]
pub struct SoftmaxPolicy {
    pub weights: FeatureMap,
}

impl Default for SoftmaxPolicy {
    fn default() -> Self {
        SoftmaxPolicy::new()
    }
}

impl SoftmaxPolicy {
    /// Zero weights over FEATURE_KEYS (Python default_factory).
    pub fn new() -> Self {
        SoftmaxPolicy {
            weights: FEATURE_KEYS
                .iter()
                .map(|k| (k.to_string(), 0.0))
                .collect(),
        }
    }

    /// theta . phi — linear score. Iterated in FEATURE_KEYS order to match
    /// Python's dict iteration order (insertion order) for bit-stable sums.
    pub fn score(&self, feats: &FeatureMap) -> f64 {
        let mut s = 0.0;
        for k in FEATURE_KEYS {
            s += self.weights.get(k).copied().unwrap_or(0.0)
                * feats.get(k).copied().unwrap_or(0.0);
        }
        s
    }

    /// Returns `(probs, feats_per_action)`, both aligned with `legal`.
    /// Exposing feats_per_action lets Phase 062 compute the policy
    /// gradient without re-deriving features from scratch.
    ///
    /// Python passes `apply_fn = domain.apply`; here the domain itself is
    /// passed so features can be computed on `next_state` exactly like
    /// the Python code did.
    pub fn action_distribution(
        &self,
        state: &NtState,
        legal: &[NtAction],
        domain: &NumberTargetDomain,
        target: f64,
        max_depth: usize,
        depth_fn: &dyn Fn(&NtState) -> usize,
    ) -> (Vec<f64>, Vec<FeatureMap>) {
        // Python raises ValueError("legal actions must be non-empty").
        assert!(
            !legal.is_empty(),
            "legal actions must be non-empty"
        );
        let mut feats_list: Vec<FeatureMap> = Vec::with_capacity(legal.len());
        let mut scores: Vec<f64> = Vec::with_capacity(legal.len());
        for action in legal {
            let next_state = domain.apply(state, action);
            let depth = depth_fn(&next_state);
            let feats = extract_features(&next_state, target, max_depth, depth);
            scores.push(self.score(&feats));
            feats_list.push(feats);
        }
        let m = scores
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let total: f64 = exps.iter().sum();
        let probs: Vec<f64> = exps.iter().map(|e| e / total).collect();
        (probs, feats_list)
    }

    /// log pi(action|state) with the probability floored at 1e-12 to
    /// avoid log(0).
    pub fn log_prob(
        &self,
        state: &NtState,
        action: &NtAction,
        legal: &[NtAction],
        domain: &NumberTargetDomain,
        target: f64,
        max_depth: usize,
        depth_fn: &dyn Fn(&NtState) -> usize,
    ) -> f64 {
        let (probs, _) =
            self.action_distribution(state, legal, domain, target, max_depth, depth_fn);
        let idx = legal
            .iter()
            .position(|a| a == action)
            .expect("action must be in legal");
        let p = probs[idx].max(1e-12); // floor to avoid log(0)
        p.ln()
    }

    /// Returns `(action, log_prob_of_that_action, feats_per_action)`.
    pub fn sample_action(
        &self,
        state: &NtState,
        legal: &[NtAction],
        domain: &NumberTargetDomain,
        target: f64,
        max_depth: usize,
        depth_fn: &dyn Fn(&NtState) -> usize,
        rng: &mut StdRng,
    ) -> (NtAction, f64, Vec<FeatureMap>) {
        let (probs, feats_list) =
            self.action_distribution(state, legal, domain, target, max_depth, depth_fn);
        let r: f64 = rng.gen();
        let mut cum = 0.0;
        for (i, p) in probs.iter().enumerate() {
            cum += p;
            if r <= cum {
                return (legal[i].clone(), probs[i].max(1e-12).ln(), feats_list);
            }
        }
        let last = probs.len() - 1;
        (
            legal[last].clone(),
            probs[last].max(1e-12).ln(),
            feats_list,
        )
    }

    /// Python `copy()`.
    pub fn copy(&self) -> SoftmaxPolicy {
        self.clone()
    }
}
