//! Phase 019 / Phase 101 — Number theory domain (Rust port of
//! `search/bezout_domain.py`, design doc 3.5 level 5).
//!
//! After live-testing found the original two-dimensional (x, y) search
//! could abstain on solvable problems, this is a single-level search over
//! x only: for any fixed x, at most one y satisfies a*x + b*y = gcd(a,b),
//! given in closed form y = (gcd - a*x) / b, valid only when b divides
//! (gcd - a*x) evenly. The result is still independently re-verified by
//! the Phase 002 verifier before being trusted.

use crate::domain::Domain;
use reasoning_verifier::symbolic_verifier::verify_numeric_equality;

#[derive(Debug, Clone, PartialEq)]
pub struct BezoutState {
    pub x: Option<i64>,
    /// None until x is chosen; also None if no integer y exists for this x.
    pub y: Option<i64>,
}

/// Python `math.gcd` parity: non-negative gcd of the absolute values.
pub fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

pub struct BezoutIdentityDomain {
    pub a: i64,
    pub b: i64,
    pub gcd: i64,
    pub search_radius: i64,
}

impl BezoutIdentityDomain {
    /// Python `__init__(self, a, b, search_radius=200)` (never fails).
    pub fn new(a: i64, b: i64, search_radius: i64) -> Self {
        BezoutIdentityDomain {
            a,
            b,
            gcd: gcd(a, b),
            search_radius,
        }
    }
}

impl Domain for BezoutIdentityDomain {
    type State = BezoutState;
    type Action = i64;

    fn initial_state(&self) -> BezoutState {
        BezoutState { x: None, y: None }
    }

    fn legal_actions(&self, state: &BezoutState) -> Vec<i64> {
        if state.x.is_some() {
            return Vec::new();
        }
        (-self.search_radius..=self.search_radius).collect()
    }

    fn apply(&self, _state: &BezoutState, action: &i64) -> BezoutState {
        let x = *action;
        let y = if self.b == 0 {
            // a*x + 0*y = gcd(a,0) = |a| -- solvable with x = sign(a), any y
            if self.a * x == self.gcd {
                Some(0)
            } else {
                None
            }
        } else {
            // y determined in closed form; divisibility is sign-independent,
            // so Rust's remainder matches Python's floored `%` here
            let numerator = self.gcd - self.a * x;
            if numerator % self.b == 0 {
                Some(numerator / self.b)
            } else {
                None
            }
        };
        BezoutState {
            x: Some(x),
            y,
        }
    }

    fn is_terminal(&self, state: &BezoutState) -> bool {
        state.x.is_some()
    }

    fn terminal_reward(&self, state: &BezoutState) -> f64 {
        assert!(self.is_terminal(state));
        let (Some(x), Some(y)) = (state.x, state.y) else {
            // no valid y for this x — not a near-miss, just invalid
            return 0.0;
        };
        let expr = format!("{}*({}) + {}*({})", self.a, x, self.b, y);
        match verify_numeric_equality(&expr, self.gcd as f64) {
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
