//! Phase 035 — Propositional logic domain (Rust port of
//! `search/logic_domain.py`, design doc section 3: logic).
//!
//! Task: find a truth assignment satisfying a propositional formula.
//! Ground truth is brute-force truth-table enumeration (feasible for small
//! variable counts) — the same "exhaustive enumeration IS the proof"
//! pattern as Phase 027's combinatorics domain.

use crate::domain::Domain;
use std::collections::HashMap;
use std::rc::Rc;

/// A propositional formula over a truth assignment (Python
/// `Callable[[Dict[str, bool]], bool]`). Shared via `Rc` so the domain and
/// the tests can hold the same closure.
pub type LogicFormula = Rc<dyn Fn(&HashMap<String, bool>) -> bool>;

/// Python `brute_force_satisfying_assignments` — full truth table.
pub fn brute_force_satisfying_assignments(
    formula: &LogicFormula,
    variables: &[String],
) -> Vec<HashMap<String, bool>> {
    let n = variables.len();
    let mut results = Vec::new();
    // itertools.product([False, True], repeat=n) as a bitmask enumeration
    // (Python would take 2^n iterations for huge n — we stop enumerating
    // beyond 2^62 which is equally unreachable in practice)
    if n < 62 {
        for mask in 0u64..(1u64 << n) {
            let assignment: HashMap<String, bool> = variables
                .iter()
                .enumerate()
                .map(|(i, v)| (v.clone(), (mask >> i) & 1 == 1))
                .collect();
            if formula(&assignment) {
                results.push(assignment);
            }
        }
    }
    results
}

#[derive(Debug, Clone, PartialEq)]
pub struct LogicState {
    /// None = unassigned, indexed by variable order.
    pub assignment: Vec<Option<bool>>,
}

#[derive(Clone)]
pub struct SatisfiabilityDomain {
    pub formula: LogicFormula,
    pub variables: Vec<String>,
    pub satisfying: Vec<HashMap<String, bool>>,
    pub is_satisfiable: bool,
}

impl SatisfiabilityDomain {
    pub fn new(formula: LogicFormula, variables: Vec<String>) -> Self {
        let satisfying = brute_force_satisfying_assignments(&formula, &variables);
        let is_satisfiable = !satisfying.is_empty();
        SatisfiabilityDomain {
            formula,
            variables,
            satisfying,
            is_satisfiable,
        }
    }
}

impl Domain for SatisfiabilityDomain {
    type State = LogicState;
    type Action = bool;

    fn initial_state(&self) -> LogicState {
        LogicState {
            assignment: self.variables.iter().map(|_| None).collect(),
        }
    }

    fn legal_actions(&self, state: &LogicState) -> Vec<bool> {
        if state.assignment.iter().all(|v| v.is_some()) {
            return Vec::new();
        }
        vec![false, true]
    }

    fn apply(&self, state: &LogicState, action: &bool) -> LogicState {
        let idx = state
            .assignment
            .iter()
            .position(|v| v.is_none())
            .expect("apply called on a fully-assigned state");
        let mut new_assignment = state.assignment.clone();
        new_assignment[idx] = Some(*action);
        LogicState {
            assignment: new_assignment,
        }
    }

    fn is_terminal(&self, state: &LogicState) -> bool {
        state.assignment.iter().all(|v| v.is_some())
    }

    fn terminal_reward(&self, state: &LogicState) -> f64 {
        assert!(self.is_terminal(state));
        let assignment: HashMap<String, bool> = self
            .variables
            .iter()
            .zip(state.assignment.iter())
            .map(|(v, b)| (v.clone(), b.expect("is_terminal checked")))
            .collect();
        if (self.formula)(&assignment) {
            1.0
        } else {
            0.0
        }
    }
}
