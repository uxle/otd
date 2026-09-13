//! Phase 015 — Quadratic factoring domain (Rust port of
//! `search/quadratic_factoring_domain.py`, design doc 3.5 level 3:
//! polynomial algebra).
//!
//! Given x^2 + b*x + c, search over candidate integer pairs (p, q) with
//! p + q = b (only p is guessed; q = b - p is determined), and verify each
//! hypothesis via the algebraic-equivalence checker: (x+p)(x+q) must
//! expand to exactly x^2 + b*x + c. Real multi-path reasoning applied to
//! factoring: generate candidates, verify each, keep the one that checks
//! out.

use crate::domain::Domain;
use reasoning_verifier::symbolic_verifier::verify_algebraic_equivalence;

/// p = None = root/unsolved state, an int once a hypothesis is chosen.
#[derive(Debug, Clone, PartialEq)]
pub struct QFState {
    pub p: Option<i64>,
}

pub struct QuadraticFactoringDomain {
    pub b: i64,
    pub c: i64,
    pub search_radius: i64,
}

impl QuadraticFactoringDomain {
    /// Python `__init__(self, b, c, search_radius=20)` (never fails).
    pub fn new(b: i64, c: i64, search_radius: i64) -> Self {
        QuadraticFactoringDomain {
            b,
            c,
            search_radius,
        }
    }
}

impl Domain for QuadraticFactoringDomain {
    type State = QFState;
    type Action = i64;

    fn initial_state(&self) -> QFState {
        QFState { p: None }
    }

    fn legal_actions(&self, state: &QFState) -> Vec<i64> {
        if state.p.is_some() {
            return Vec::new(); // already terminal
        }
        (-self.search_radius..=self.search_radius).collect()
    }

    fn apply(&self, _state: &QFState, action: &i64) -> QFState {
        QFState { p: Some(*action) }
    }

    fn is_terminal(&self, state: &QFState) -> bool {
        state.p.is_some()
    }

    fn terminal_reward(&self, state: &QFState) -> f64 {
        assert!(self.is_terminal(state));
        let p = state.p.expect("is_terminal checked");
        let q = self.b - p;
        let lhs = format!("(x+{})*(x+{})", p, q);
        let rhs = format!("x**2 + {}*x + {}", self.b, self.c);
        match verify_algebraic_equivalence(&lhs, &rhs) {
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
