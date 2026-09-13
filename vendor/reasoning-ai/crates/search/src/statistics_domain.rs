//! Phase 034 — Statistics domain (Rust port of
//! `search/statistics_domain.py`, design doc section 9: probability,
//! statistics; expected value, variance).
//!
//! Search proposes a candidate numeric answer for E[X] or Var(X) given a
//! discrete distribution, verified against an exact fraction-based ground
//! truth (Python `fractions.Fraction` -> `reasoning_common::Rat`, not
//! floats, so there's no precision ambiguity in what "correct" means).

use crate::domain::Domain;
use reasoning_common::Rat;

/// Exact E[X] = sum(Fraction(v) * p).
pub fn exact_expected_value(outcomes: &[(f64, Rat)]) -> Rat {
    let mut sum = Rat::from_int(0);
    for (v, p) in outcomes {
        sum = sum + Rat::from_f64(*v) * *p;
    }
    sum
}

/// Exact Var(X) = sum(p * (Fraction(v) - mean)^2).
pub fn exact_variance(outcomes: &[(f64, Rat)]) -> Rat {
    let mean = exact_expected_value(outcomes);
    let mut sum = Rat::from_int(0);
    for (v, p) in outcomes {
        let d = Rat::from_f64(*v) - mean;
        sum = sum + *p * d * d;
    }
    sum
}

#[derive(Debug, Clone, PartialEq)]
pub struct StatsState {
    pub guess_numerator: Option<i64>,
    pub guess_denominator: Option<i64>,
}

/// Python tuple actions ("denom", d) / ("num", n) as an enum.
#[derive(Debug, Clone, PartialEq)]
pub enum StatsAction {
    Denom(i64),
    Num(i64),
}

pub struct ExpectedValueDomain {
    /// list of (value, probability as Fraction)
    pub outcomes: Vec<(f64, Rat)>,
    pub metric: String,
    /// computed exactly, never approximated — verification is exact
    /// equality, not tolerance
    pub ground_truth: Rat,
    pub denom_range: i64,
}

impl ExpectedValueDomain {
    /// Python `__init__` asserts the probabilities sum to exactly 1 (kept
    /// as a panicking `assert!`, mirroring the Python AssertionError).
    pub fn new(outcomes: Vec<(f64, Rat)>, metric: &str, denom_range: i64) -> Self {
        let p_sum: Rat = outcomes
            .iter()
            .map(|(_, p)| *p)
            .fold(Rat::from_int(0), |a, b| a + b);
        assert!(p_sum == Rat::from_int(1), "probabilities must sum to exactly 1");
        let ground_truth = if metric == "expectation" {
            exact_expected_value(&outcomes)
        } else {
            exact_variance(&outcomes)
        };
        ExpectedValueDomain {
            outcomes,
            metric: metric.to_string(),
            ground_truth,
            denom_range,
        }
    }

    /// Python defaults: `metric="expectation", denom_range=20`.
    pub fn with_defaults(outcomes: Vec<(f64, Rat)>) -> Self {
        Self::new(outcomes, "expectation", 20)
    }
}

impl Domain for ExpectedValueDomain {
    type State = StatsState;
    type Action = StatsAction;

    fn initial_state(&self) -> StatsState {
        StatsState {
            guess_numerator: None,
            guess_denominator: None,
        }
    }

    fn legal_actions(&self, state: &StatsState) -> Vec<StatsAction> {
        if state.guess_denominator.is_none() {
            // first pick denominator
            return (1..=self.denom_range).map(StatsAction::Denom).collect();
        }
        if state.guess_numerator.is_none() {
            let max_num = self.denom_range * 10;
            return (-max_num..=max_num).map(StatsAction::Num).collect();
        }
        Vec::new()
    }

    fn apply(&self, state: &StatsState, action: &StatsAction) -> StatsState {
        match *action {
            StatsAction::Denom(v) => StatsState {
                guess_numerator: None,
                guess_denominator: Some(v),
            },
            StatsAction::Num(v) => StatsState {
                guess_numerator: Some(v),
                guess_denominator: state.guess_denominator,
            },
        }
    }

    fn is_terminal(&self, state: &StatsState) -> bool {
        state.guess_numerator.is_some() && state.guess_denominator.is_some()
    }

    fn terminal_reward(&self, state: &StatsState) -> f64 {
        assert!(self.is_terminal(state));
        let num = state.guess_numerator.expect("is_terminal checked");
        let den = state.guess_denominator.expect("is_terminal checked");
        let guess = Rat::new(num, den);
        if guess == self.ground_truth {
            1.0
        } else {
            0.0
        }
    }
}
