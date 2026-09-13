//! The `Domain` trait — the search contract every domain implements
//! (Rust port of the Python `search.mcts.Domain` Protocol).
//!
//! A domain knows: what actions are legal in a state, how to apply them,
//! when a state is terminal, and what reward a terminal state deserves
//! (grounded in an external verifier — never a learned/self-reported
//! score).

/// Anything MCTS / heuristic search can search over.
pub trait Domain {
    type State: Clone;
    type Action: Clone + PartialEq;

    /// The state search starts from.
    fn initial_state(&self) -> Self::State;
    /// All legal next actions in `state` (empty = dead end / terminal).
    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action>;
    /// The successor state after taking `action` in `state`.
    fn apply(&self, state: &Self::State, action: &Self::Action) -> Self::State;
    /// Is this a terminal state?
    fn is_terminal(&self, state: &Self::State) -> bool;
    /// Terminal reward in [0, 1], grounded in an external verifier.
    fn terminal_reward(&self, state: &Self::State) -> f64;
}

/// Rollout policy for MCTS's simulation phase: `(state, legal) -> action`.
/// The uniform-random default lives in `mcts::UniformRollout`.
pub trait RolloutPolicy<S, A> {
    fn choose(&mut self, rng: &mut rand::rngs::StdRng, state: &S, legal: &[A]) -> A;
}

use rand::Rng;

/// Uniform random choice over legal actions (used as the default rollout
/// policy and for MCTS's in-expansion child sampling).
pub struct UniformRollout;

impl<S, A: Clone> RolloutPolicy<S, A> for UniformRollout {
    fn choose(&mut self, rng: &mut rand::rngs::StdRng, _state: &S, legal: &[A]) -> A {
        legal[rng.gen_range(0..legal.len())].clone()
    }
}

/// Pick a uniformly random element (Python `rng.choice`).
pub fn rng_choice<T: Clone>(rng: &mut rand::rngs::StdRng, items: &[T]) -> T {
    items[rng.gen_range(0..items.len())].clone()
}
