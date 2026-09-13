//! Phase 042 — Geometry domain (Rust port of
//! `search/geometry_domain.py`, design doc section 9: geometry).
//!
//! Search proposes a candidate numeric answer for area/perimeter given
//! shape parameters; ground truth is exact formula evaluation (rational
//! arithmetic via the expr strings, no float rounding ambiguity for the
//! geometry itself, though transcendental cases like circles get a
//! tolerance since pi is irrational).

use crate::domain::Domain;
use reasoning_common::py_float_str;
use std::collections::HashMap;

/// Python `exact_ground_truth(shape, **params)` — returns
/// (expr_str, numeric_value). Expr strings are built exactly like the
/// Python f-strings, with params rendered via Python float repr
/// (`py_float_str`).
pub fn exact_ground_truth(shape: &str, params: &HashMap<String, f64>) -> Result<(String, f64), String> {
    let get = |k: &str| -> Result<f64, String> {
        params
            .get(k)
            .copied()
            .ok_or_else(|| format!("missing parameter {:?}", k))
    };
    match shape {
        "rectangle_area" => {
            let (w, h) = (get("w")?, get("h")?);
            Ok((
                format!("{}*{}", py_float_str(w), py_float_str(h)),
                w * h,
            ))
        }
        "rectangle_perimeter" => {
            let (w, h) = (get("w")?, get("h")?);
            Ok((
                format!("2*({}+{})", py_float_str(w), py_float_str(h)),
                2.0 * (w + h),
            ))
        }
        "triangle_area" => {
            let (b, h) = (get("b")?, get("h")?);
            Ok((
                format!("{}*{}/2", py_float_str(b), py_float_str(h)),
                b * h / 2.0,
            ))
        }
        "circle_area" => {
            let r = get("r")?;
            Ok((
                format!("pi*{}**2", py_float_str(r)),
                std::f64::consts::PI * r * r,
            ))
        }
        "circle_circumference" => {
            let r = get("r")?;
            Ok((
                format!("2*pi*{}", py_float_str(r)),
                2.0 * std::f64::consts::PI * r,
            ))
        }
        _ => Err(format!("unknown shape/quantity {}", shape)),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct GeometryState {
    pub guess_numerator: Option<i64>,
    pub guess_denominator: Option<i64>,
}

/// Python tuple actions ("denom", d) / ("num", n) as an enum.
#[derive(Debug, Clone, PartialEq)]
pub enum GeometryAction {
    Denom(i64),
    Num(i64),
}

pub struct GeometryDomain {
    pub shape: String,
    pub params: HashMap<String, f64>,
    pub ground_truth_expr: String,
    pub ground_truth_value: f64,
    pub denom_range: i64,
    pub tolerance: f64,
}

impl GeometryDomain {
    /// Python `__init__` can raise (unknown shape) — `try_new` propagates.
    pub fn try_new(
        shape: &str,
        params: HashMap<String, f64>,
        denom_range: i64,
        tolerance: f64,
    ) -> Result<Self, String> {
        let (ground_truth_expr, ground_truth_value) = exact_ground_truth(shape, &params)?;
        Ok(GeometryDomain {
            shape: shape.to_string(),
            params,
            ground_truth_expr,
            ground_truth_value,
            denom_range,
            tolerance,
        })
    }

    /// Python defaults: `denom_range=10, tolerance=1e-4`.
    pub fn new(shape: &str, params: HashMap<String, f64>) -> Self {
        Self::try_new(shape, params, 10, 1e-4).expect("valid geometry problem")
    }

    /// Python `GeometryDomain(shape, params, denom_range=...)` with the
    /// default tolerance.
    pub fn with_denom_range(shape: &str, params: HashMap<String, f64>, denom_range: i64) -> Self {
        Self::try_new(shape, params, denom_range, 1e-4).expect("valid geometry problem")
    }
}

impl Domain for GeometryDomain {
    type State = GeometryState;
    type Action = GeometryAction;

    fn initial_state(&self) -> GeometryState {
        GeometryState {
            guess_numerator: None,
            guess_denominator: None,
        }
    }

    fn legal_actions(&self, state: &GeometryState) -> Vec<GeometryAction> {
        if state.guess_denominator.is_none() {
            return (1..=self.denom_range).map(GeometryAction::Denom).collect();
        }
        if state.guess_numerator.is_none() {
            // bound derived from the raw INPUT parameters, not the answer
            // itself (deriving it from ground_truth_value would leak the
            // answer's scale into the search space, as happened in an
            // earlier version of the combinatorics domain, Phase 027)
            let param_scale: f64 = self.params.values().map(|v| v.abs()).sum::<f64>() + 1.0;
            let bound = (param_scale * param_scale * self.denom_range as f64 * 4.0) as i64;
            return (0..=bound).map(GeometryAction::Num).collect();
        }
        Vec::new()
    }

    fn apply(&self, state: &GeometryState, action: &GeometryAction) -> GeometryState {
        match *action {
            GeometryAction::Denom(v) => GeometryState {
                guess_numerator: None,
                guess_denominator: Some(v),
            },
            GeometryAction::Num(v) => GeometryState {
                guess_numerator: Some(v),
                guess_denominator: state.guess_denominator,
            },
        }
    }

    fn is_terminal(&self, state: &GeometryState) -> bool {
        state.guess_numerator.is_some() && state.guess_denominator.is_some()
    }

    fn terminal_reward(&self, state: &GeometryState) -> f64 {
        assert!(self.is_terminal(state));
        let num = state.guess_numerator.expect("is_terminal checked");
        let den = state.guess_denominator.expect("is_terminal checked");
        let guess = num as f64 / den as f64;
        if (guess - self.ground_truth_value).abs() <= self.tolerance {
            1.0
        } else {
            0.0
        }
    }
}
