//! A concrete Domain for MCTS (Rust port of `search/number_target_domain.py`):
//! given a multiset of starting numbers, combine them pairwise with
//! +,-,*,/ until one remains. Terminal reward comes from the symbolic
//! verifier, not a hand-rolled equality check.

use crate::domain::Domain;
use reasoning_common::py_float_str;

pub type Number = (f64, String); // (value, expression string that produced it)
pub type NtState = Vec<Number>;
pub type NtAction = (usize, usize, char); // (index_a, index_b, operator)

const OPS: [char; 4] = ['+', '-', '*', '/'];

/// Python parity: `make_initial_state(numbers)`.
pub fn make_initial_state(numbers: &[f64]) -> NtState {
    numbers
        .iter()
        .map(|&n| {
            let s = if n == n.trunc() && n.abs() < 1e15 {
                format!("{}", n as i64)
            } else {
                py_float_str(n)
            };
            (n, s)
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct NumberTargetDomain {
    pub target: f64,
    pub tolerance: f64,
}

impl NumberTargetDomain {
    pub fn new(target: f64) -> Self {
        NumberTargetDomain {
            target,
            tolerance: 1e-6,
        }
    }

    pub fn with_tolerance(target: f64, tolerance: f64) -> Self {
        NumberTargetDomain { target, tolerance }
    }
}

impl Domain for NumberTargetDomain {
    type State = NtState;
    type Action = NtAction;

    fn initial_state(&self) -> NtState {
        Vec::new() // callers pass their own root state (Python parity)
    }

    fn legal_actions(&self, state: &NtState) -> Vec<NtAction> {
        let mut actions = Vec::new();
        let n = state.len();
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                for &op in &OPS {
                    if op == '/' && state[j].0.abs() < 1e-12 {
                        continue; // no division by zero
                    }
                    actions.push((i, j, op));
                }
            }
        }
        actions
    }

    fn apply(&self, state: &NtState, action: &NtAction) -> NtState {
        let (i, j, op) = *action;
        let (a_val, a_expr) = &state[i];
        let (b_val, b_expr) = &state[j];
        let val = match op {
            '+' => a_val + b_val,
            '-' => a_val - b_val,
            '*' => a_val * b_val,
            '/' => a_val / b_val,
            _ => panic!("Unknown op {}", op),
        };
        let new_expr = format!("({}{}{})", a_expr, op, b_expr);
        let mut remaining: Vec<Number> = Vec::with_capacity(state.len() - 1);
        for (k, item) in state.iter().enumerate() {
            if k != i && k != j {
                remaining.push(item.clone());
            }
        }
        remaining.push((val, new_expr));
        remaining
    }

    fn is_terminal(&self, state: &NtState) -> bool {
        state.len() == 1
    }

    fn terminal_reward(&self, state: &NtState) -> f64 {
        debug_assert!(self.is_terminal(state));
        let expr = &state[0].1;
        match reasoning_symbolic::parse_expr(expr) {
            Ok(e) => match reasoning_symbolic::eval_f64(&e, &Default::default()) {
                Ok(v) => {
                    if (v - self.target).abs() <= self.tolerance {
                        1.0
                    } else {
                        0.0
                    }
                }
                Err(_) => 0.0,
            },
            Err(_) => 0.0,
        }
    }
}
