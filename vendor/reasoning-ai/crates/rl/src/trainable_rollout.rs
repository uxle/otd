//! Phase 065 — Trainable-policy <-> MCTS bridge.
//!
//! `prm/guided_policy.py` already defines the interface MCTS expects:
//! `rollout_policy(state, legal_actions) -> action`. This phase provides
//! the same shape for a Phase 061 SoftmaxPolicy, so search can use a
//! *trained* policy for rollouts instead of the PRM's regression-based
//! softmax. Also provides a log-prob-tracking episode sampler for
//! building genuine trajectory objects.

use std::rc::Rc;

use rand::rngs::StdRng;
use rand::Rng;
use reasoning_search::{Domain, NtAction, NtState, NumberTargetDomain, RolloutPolicy};

use crate::policy::SoftmaxPolicy;

/// The standard GRPO/RLHF rollout shape: four parallel lists, one entry
/// per step taken (`states[i]` is the state `actions[i]` was taken FROM).
#[derive(Debug, Clone)]
pub struct Episode {
    pub states: Vec<NtState>,
    pub actions: Vec<NtAction>,
    pub legals: Vec<Vec<NtAction>>,
    pub log_probs: Vec<f64>,
}

impl Episode {
    pub fn new() -> Self {
        Episode {
            states: Vec::new(),
            actions: Vec::new(),
            legals: Vec::new(),
            log_probs: Vec::new(),
        }
    }
}

impl Default for Episode {
    fn default() -> Self {
        Episode::new()
    }
}

/// Rust port of Python `make_trainable_rollout_policy(...)`'s returned
/// closure: the same epsilon-greedy-over-softmax shape as
/// prm.guided_policy's `make_prm_guided_policy`, so it's a drop-in for
/// `Mcts::with_rollout_policy`. Python shared the policy object between
/// closure and caller; here callers pass a clone — training updates use
/// the log_probs returned by `sample_episode`/`sample_episode_via_mcts`,
/// which matches Python's information flow.
pub struct TrainableRollout {
    policy: SoftmaxPolicy,
    domain: NumberTargetDomain,
    target: f64,
    max_depth: usize,
    depth_fn: Rc<dyn Fn(&NtState) -> usize>,
    epsilon: f64,
}

impl TrainableRollout {
    /// Python `make_trainable_rollout_policy(policy, apply_fn, target,
    /// max_depth, depth_fn, epsilon=0.1, rng=None)` — the rng is supplied
    /// per-call by the MCTS harness in this port (the Python closure
    /// captured one; MCTS drives the stream either way).
    pub fn new(
        policy: SoftmaxPolicy,
        domain: NumberTargetDomain,
        target: f64,
        max_depth: usize,
        depth_fn: Rc<dyn Fn(&NtState) -> usize>,
        epsilon: f64,
    ) -> Self {
        TrainableRollout {
            policy,
            domain,
            target,
            max_depth,
            depth_fn,
            epsilon,
        }
    }

    /// Python closure called directly (outside MCTS).
    pub fn choose_with(&mut self, rng: &mut StdRng, state: &NtState, legal: &[NtAction]) -> NtAction {
        self.choose(rng, state, legal)
    }

    pub fn policy(&self) -> &SoftmaxPolicy {
        &self.policy
    }
}

impl RolloutPolicy<NtState, NtAction> for TrainableRollout {
    fn choose(&mut self, rng: &mut StdRng, state: &NtState, legal: &[NtAction]) -> NtAction {
        if rng.gen::<f64>() < self.epsilon || legal.len() == 1 {
            return legal[rng.gen_range(0..legal.len())].clone();
        }
        let (action, _log_p, _feats) = self.policy.sample_action(
            state,
            legal,
            &self.domain,
            self.target,
            self.max_depth,
            self.depth_fn.as_ref(),
            rng,
        );
        action
    }
}

/// Roll out ONE full episode by direct policy sampling (no tree search):
/// repeatedly sample_action from `policy` until the domain reaches a
/// terminal state or legal_actions is empty (or the safety cap of
/// max_depth + 2 steps fires). Sample directly from the current policy —
/// the standard GRPO/RLHF rollout shape.
pub fn sample_episode(
    policy: &SoftmaxPolicy,
    domain: &NumberTargetDomain,
    initial_state: &NtState,
    target: f64,
    max_depth: usize,
    depth_fn: &dyn Fn(&NtState) -> usize,
    rng: &mut StdRng,
) -> Episode {
    let mut ep = Episode::new();
    let mut state = initial_state.clone();
    let mut steps_taken = 0usize;
    while !domain.is_terminal(&state) && steps_taken < max_depth + 2 {
        let legal = domain.legal_actions(&state);
        if legal.is_empty() {
            break;
        }
        let (action, log_p, _feats) =
            policy.sample_action(&state, &legal, domain, target, max_depth, depth_fn, rng);
        ep.states.push(state.clone());
        ep.actions.push(action.clone());
        ep.legals.push(legal);
        ep.log_probs.push(log_p);
        state = domain.apply(&state, &action);
        steps_taken += 1;
    }
    ep
}
