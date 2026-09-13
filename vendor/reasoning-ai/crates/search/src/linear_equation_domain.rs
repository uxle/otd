//! Phase 006 — Linear-equation-solving Domain (Rust port of
//! `search/linear_equation_domain.py`).
//!
//! Real step-by-step algebra: the state is a live equation, actions are
//! legal algebraic manipulations (add/subtract/multiply/divide both sides),
//! and the episode is terminal only once the equation is reduced to
//! `x = <number>`. Reward checks the found x against the symbolic
//! ground-truth solver (Phase 002) — the search never declares victory.

use crate::domain::Domain;
use reasoning_common::Rat;
use reasoning_symbolic::{eval_f64, expand, is_zero, Expr};
use reasoning_verifier::symbolic_verifier::safe_parse_with_locals;
use std::collections::HashMap;

/// A live equation (Python `@dataclass(frozen=True) EqState`).
#[derive(Debug, Clone, PartialEq)]
pub struct EqState {
    pub lhs: Expr,
    pub rhs: Expr,
    pub depth: usize,
}

/// ("add" | "sub" | "mul" | "div", constant)
pub type LinEqAction = (String, f64);

#[derive(Debug, Clone)]
pub struct LinearEquationDomain {
    pub initial: EqState,
    pub max_depth: usize,
    /// Ground-truth root from `solve_equation` (Phase 002).
    pub true_root: f64,
}

impl LinearEquationDomain {
    /// Python `__init__` can raise (parse failure, unsolvable equation, no
    /// root at all) — exposed as `try_new` so callers handle it panic-free.
    pub fn try_new(equation_str: &str, max_depth: usize) -> Result<Self, String> {
        let (lhs_str, rhs_str) = equation_str.split_once('=').ok_or_else(|| {
            format!("equation {:?} must contain a '=' separator", equation_str)
        })?;
        let lhs = safe_parse_with_locals(lhs_str, &["x"]).map_err(|e| format!("{:?}", e))?;
        let rhs = safe_parse_with_locals(rhs_str, &["x"]).map_err(|e| format!("{:?}", e))?;
        // Python: self._true_root = solve_equation(equation_str, "x")[0]
        // (IndexError if no roots — propagated as an Err here)
        let roots = reasoning_symbolic::solve_equation(equation_str, "x")?;
        let root = roots
            .first()
            .cloned()
            .ok_or_else(|| format!("solve_equation found no roots for {:?}", equation_str))?;
        let true_root = eval_f64(&root, &HashMap::new())?;
        Ok(LinearEquationDomain {
            initial: EqState {
                lhs,
                rhs,
                depth: 0,
            },
            max_depth,
            true_root,
        })
    }

    /// Python `LinearEquationDomain(equation_str, max_depth=6)` with the
    /// default depth.
    pub fn new(equation_str: &str) -> Self {
        Self::try_new(equation_str, 6).expect("linear equation must parse and solve")
    }

    fn candidate_constants(&self, state: &EqState) -> Vec<f64> {
        // legal move constants derived from the equation's own coefficients
        // (a real solver doesn't need to try arbitrary numbers — it reacts
        // to what's actually in the equation)
        let mut consts: Vec<f64> = Vec::new();
        for expr in [&state.lhs, &state.rhs] {
            for term in expr.terms() {
                let (coeff, _rest) = term.as_coeff_mul();
                if !coeff.is_zero() {
                    let c = coeff.to_f64();
                    if !consts.contains(&c) {
                        consts.push(c);
                    }
                }
            }
        }
        consts.retain(|&c| c != 0.0);
        if consts.is_empty() {
            consts.push(1.0);
        }
        consts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        consts
    }
}

impl Domain for LinearEquationDomain {
    type State = EqState;
    type Action = LinEqAction;

    fn initial_state(&self) -> EqState {
        self.initial.clone()
    }

    fn legal_actions(&self, state: &EqState) -> Vec<LinEqAction> {
        if state.depth >= self.max_depth {
            return Vec::new();
        }
        let mut actions: Vec<LinEqAction> = Vec::new();
        for c in self.candidate_constants(state) {
            actions.push(("add".to_string(), c));
            actions.push(("sub".to_string(), c));
            if c != 0.0 {
                actions.push(("mul".to_string(), 1.0 / c));
                actions.push(("div".to_string(), c));
            }
        }
        actions
    }

    fn apply(&self, state: &EqState, action: &LinEqAction) -> EqState {
        let (op, c) = action.clone();
        let c_expr = Expr::num(Rat::from_f64(c));
        let (mut lhs, mut rhs) = (state.lhs.clone(), state.rhs.clone());
        match op.as_str() {
            "add" => {
                lhs = lhs + c_expr.clone();
                rhs = rhs + c_expr;
            }
            "sub" => {
                lhs = lhs - c_expr.clone();
                rhs = rhs - c_expr;
            }
            "mul" => {
                lhs = lhs * c_expr.clone();
                rhs = rhs * c_expr;
            }
            "div" => {
                if c == 0.0 {
                    // no-op, wastes depth (Python returns before expanding)
                    return EqState {
                        lhs,
                        rhs,
                        depth: state.depth + 1,
                    };
                }
                lhs = lhs / c_expr.clone();
                rhs = rhs / c_expr;
            }
            _ => {}
        }
        EqState {
            lhs: expand(&lhs),
            rhs: expand(&rhs),
            depth: state.depth + 1,
        }
    }

    fn is_terminal(&self, state: &EqState) -> bool {
        // terminal iff lhs is equivalent to bare "x" (e.g. 1.0*x counts) and
        // rhs has no x in it (fully isolated). expand() alone collapses every
        // equivalent-to-x form this domain can produce — dramatically cheaper
        // than full simplify() (Phase 097 profiling).
        if state.rhs.has_sym("x") {
            return false;
        }
        is_zero(&expand(&(state.lhs.clone() - Expr::sym("x"))))
    }

    fn terminal_reward(&self, state: &EqState) -> f64 {
        assert!(self.is_terminal(state));
        // Python `float(state.rhs)` raises TypeError on non-numbers -> 0.0
        let Some(found) = state.rhs.as_rat().map(|r| r.to_f64()) else {
            return 0.0;
        };
        if (found - self.true_root).abs() < 1e-6 {
            1.0
        } else {
            0.0
        }
    }
}
