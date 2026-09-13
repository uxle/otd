//! Phase 025 — Failure memory (Rust port of
//! `python/memory/failure_memory.py`).
//!
//! Wraps a Domain so that once a specific terminal state has been checked
//! and found wrong, it's remembered — repeat encounters (a real possibility
//! in MCTS, which can revisit the same state via different paths) skip the
//! verifier call entirely and return the cached failure. This never changes
//! *which* states are correct/incorrect (the wrapped domain's own
//! terminal_reward is still the source of truth on first encounter) — it
//! only saves redundant verifier calls.

use reasoning_search::Domain;
use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::rc::Rc;

#[derive(Debug, Default)]
pub struct FailureMemory {
    known_failures: HashMap<String, HashSet<String>>,
    pub verifier_calls: usize,
    pub memory_hits: usize,
}

impl FailureMemory {
    pub fn new() -> Self {
        FailureMemory {
            known_failures: HashMap::new(),
            verifier_calls: 0,
            memory_hits: 0,
        }
    }

    pub fn is_known_failure(&self, problem_key: &str, state_repr: &str) -> bool {
        self.known_failures
            .get(problem_key)
            .map(|s| s.contains(state_repr))
            .unwrap_or(false)
    }

    pub fn record_failure(&mut self, problem_key: &str, state_repr: &str) {
        self.known_failures
            .entry(problem_key.to_string())
            .or_default()
            .insert(state_repr.to_string());
    }
}

/// Wraps any Domain (matching the MCTS protocol) with failure-memory-aware
/// terminal_reward. Delegates everything else unchanged.
///
/// The Python version shares the `FailureMemory` object by reference; the
/// Rust version shares it via `Rc<RefCell<...>>` so both the wrapped domain
/// and the caller can observe the counters (the Python version also passed
/// through `inner_domain.target` when present — Rust's generic layering
/// can't reflect over arbitrary domains, and no caller used it).
pub struct FailureMemoryDomain<D: Domain> {
    pub inner: D,
    pub memory: Rc<RefCell<FailureMemory>>,
    pub problem_key: String,
    pub state_repr_fn: Box<dyn Fn(&D::State) -> String>,
}

impl<D: Domain> FailureMemoryDomain<D> {
    pub fn new(
        inner: D,
        memory: Rc<RefCell<FailureMemory>>,
        problem_key: &str,
        state_repr_fn: Box<dyn Fn(&D::State) -> String>,
    ) -> Self {
        FailureMemoryDomain {
            inner,
            memory,
            problem_key: problem_key.to_string(),
            state_repr_fn,
        }
    }
}

impl<D: Domain> Domain for FailureMemoryDomain<D> {
    type State = D::State;
    type Action = D::Action;

    fn initial_state(&self) -> Self::State {
        self.inner.initial_state()
    }

    fn legal_actions(&self, state: &Self::State) -> Vec<Self::Action> {
        self.inner.legal_actions(state)
    }

    fn apply(&self, state: &Self::State, action: &Self::Action) -> Self::State {
        self.inner.apply(state, action)
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        self.inner.is_terminal(state)
    }

    fn terminal_reward(&self, state: &Self::State) -> f64 {
        let repr = (self.state_repr_fn)(state);
        if self.memory.borrow().is_known_failure(&self.problem_key, &repr) {
            self.memory.borrow_mut().memory_hits += 1;
            return 0.0;
        }
        self.memory.borrow_mut().verifier_calls += 1;
        let reward = self.inner.terminal_reward(state);
        if reward < 0.999 {
            self.memory
                .borrow_mut()
                .record_failure(&self.problem_key, &repr);
        }
        reward
    }
}
