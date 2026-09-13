//! Phase 085 — Trigonometry domain (Rust port of
//! `search/trigonometry_domain.py`, design doc section 9: "Trigonometry").
//!
//! Same candidate-selection pattern as calculus_domain.py (Phase 041): the
//! CAS computes ground truth (never re-implemented by hand — trig identity
//! simplification is exactly the kind of thing worth trusting a mature,
//! independently-tested CAS for), search picks among candidates, verified
//! by algebraic equivalence via the verifier.
//!
//! Two problem kinds:
//!   - simplify: reduce a trig expression to its simplest equivalent form
//!     (e.g. sin(x)^2 + cos(x)^2 -> 1)
//!   - evaluate: exact value of a trig function at a "nice" angle
//!     (e.g. sin(pi/6) -> 1/2), where floating point would introduce
//!     spurious inexactness exact rationals avoid.

use crate::domain::Domain;
use reasoning_symbolic::{nsimplify, simplify};
use reasoning_verifier::symbolic_verifier::{safe_parse_with_locals, verify_algebraic_equivalence};

// Python's _TRIG_LOCALS: {"x": X, "sin": sin, "cos": cos, "tan": tan,
// "pi": pi} — the parser has sin/cos/tan/pi built in, so only "x" is a
// local symbol here.
const TRIG_LOCALS: &[&str] = &["x"];

#[derive(Debug, Clone, PartialEq)]
pub struct TrigState {
    pub choice: Option<usize>,
}

pub struct TrigSimplifyDomain {
    pub expr_str: String,
    pub candidates: Vec<String>,
    pub ground_truth: String,
}

impl TrigSimplifyDomain {
    /// Python `__init__` can raise (parse failure) — `try_new` propagates.
    pub fn try_new(expr_str: &str, candidates: Vec<String>) -> Result<Self, String> {
        let parsed = safe_parse_with_locals(expr_str, TRIG_LOCALS).map_err(|e| format!("{:?}", e))?;
        let ground_truth = simplify(&parsed).to_string();
        Ok(TrigSimplifyDomain {
            expr_str: expr_str.to_string(),
            candidates,
            ground_truth,
        })
    }

    /// Panicking constructor for statically-valid input.
    pub fn new(expr_str: &str, candidates: Vec<String>) -> Self {
        Self::try_new(expr_str, candidates).expect("valid trig expression")
    }
}

impl Domain for TrigSimplifyDomain {
    type State = TrigState;
    type Action = usize;

    fn initial_state(&self) -> TrigState {
        TrigState { choice: None }
    }

    fn legal_actions(&self, state: &TrigState) -> Vec<usize> {
        if state.choice.is_some() {
            return Vec::new();
        }
        (0..self.candidates.len()).collect()
    }

    fn apply(&self, _state: &TrigState, action: &usize) -> TrigState {
        TrigState {
            choice: Some(*action),
        }
    }

    fn is_terminal(&self, state: &TrigState) -> bool {
        state.choice.is_some()
    }

    fn terminal_reward(&self, state: &TrigState) -> f64 {
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

pub struct TrigEvaluateDomain {
    pub expr_str: String,
    pub candidates: Vec<String>,
    pub ground_truth: String,
}

impl TrigEvaluateDomain {
    /// Python `__init__` can raise (parse failure) — `try_new` propagates.
    pub fn try_new(expr_str: &str, candidates: Vec<String>) -> Result<Self, String> {
        let parsed = safe_parse_with_locals(expr_str, TRIG_LOCALS).map_err(|e| format!("{:?}", e))?;
        // Python: str(sympy.nsimplify(parsed, [sympy.pi], rational=False))
        let ground_truth = nsimplify(&parsed).to_string();
        Ok(TrigEvaluateDomain {
            expr_str: expr_str.to_string(),
            candidates,
            ground_truth,
        })
    }

    /// Panicking constructor for statically-valid input.
    pub fn new(expr_str: &str, candidates: Vec<String>) -> Self {
        Self::try_new(expr_str, candidates).expect("valid trig expression")
    }
}

impl Domain for TrigEvaluateDomain {
    type State = TrigState;
    type Action = usize;

    fn initial_state(&self) -> TrigState {
        TrigState { choice: None }
    }

    fn legal_actions(&self, state: &TrigState) -> Vec<usize> {
        if state.choice.is_some() {
            return Vec::new();
        }
        (0..self.candidates.len()).collect()
    }

    fn apply(&self, _state: &TrigState, action: &usize) -> TrigState {
        TrigState {
            choice: Some(*action),
        }
    }

    fn is_terminal(&self, state: &TrigState) -> bool {
        state.choice.is_some()
    }

    fn terminal_reward(&self, state: &TrigState) -> f64 {
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
