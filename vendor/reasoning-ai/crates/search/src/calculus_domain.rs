//! Phase 041 — Calculus domain (Rust port of
//! `search/calculus_domain.py`, design doc section 9: calculus).
//!
//! Search proposes a candidate derivative expression; ground truth is the
//! CAS's own `diff()` — not re-implemented by hand. What's being tested is
//! the SEARCH's ability to find/recognize the right candidate among
//! plausible-looking wrong ones (off-by-a-constant, missing chain rule
//! term), verified by algebraic equivalence, exactly the same
//! verification pattern as every other domain.

use crate::domain::Domain;
use reasoning_symbolic::diff;
use reasoning_verifier::symbolic_verifier::{safe_parse_with_locals, verify_algebraic_equivalence};

#[derive(Debug, Clone, PartialEq)]
pub struct CalcState {
    pub choice: Option<usize>,
}

pub struct DerivativeDomain {
    pub expr_str: String,
    pub candidates: Vec<String>,
    pub ground_truth: String,
}

impl DerivativeDomain {
    /// Python `__init__` can raise (parse failure) — `try_new` propagates.
    pub fn try_new(expr_str: &str, candidates: Vec<String>) -> Result<Self, String> {
        let parsed = safe_parse_with_locals(expr_str, &["x"]).map_err(|e| format!("{:?}", e))?;
        let ground_truth = diff(&parsed, "x").to_string();
        Ok(DerivativeDomain {
            expr_str: expr_str.to_string(),
            candidates,
            ground_truth,
        })
    }

    /// Panicking constructor for callers with statically-valid input.
    pub fn new(expr_str: &str, candidates: Vec<String>) -> Self {
        Self::try_new(expr_str, candidates).expect("valid expression")
    }
}

impl Domain for DerivativeDomain {
    type State = CalcState;
    type Action = usize;

    fn initial_state(&self) -> CalcState {
        CalcState { choice: None }
    }

    fn legal_actions(&self, state: &CalcState) -> Vec<usize> {
        if state.choice.is_some() {
            return Vec::new();
        }
        (0..self.candidates.len()).collect()
    }

    fn apply(&self, _state: &CalcState, action: &usize) -> CalcState {
        CalcState {
            choice: Some(*action),
        }
    }

    fn is_terminal(&self, state: &CalcState) -> bool {
        state.choice.is_some()
    }

    fn terminal_reward(&self, state: &CalcState) -> f64 {
        assert!(self.is_terminal(state));
        let candidate = &self.candidates[state.choice.expect("is_terminal checked")];
        match verify_algebraic_equivalence(candidate, &self.ground_truth) {
            Ok(result) => {
                if result.passed {
                    1.0
                } else {
                    0.0
                }
            }
            Err(_) => 0.0,
        }
    }
}
