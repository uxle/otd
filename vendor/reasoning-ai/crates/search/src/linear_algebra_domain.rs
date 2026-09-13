//! Phase 086 — Linear algebra domain (Rust port of
//! `search/linear_algebra_domain.py`, design doc section 9).
//!
//! Same candidate-selection pattern again. Three problem kinds, each
//! backed by exact rational matrices (`reasoning_common::RatMatrix`,
//! the sympy.Matrix subset — matrix arithmetic is not re-derived by hand
//! any more than calculus_domain re-derives differentiation rules):
//!
//!   - determinant: det(M) for a 2x2 or 3x3 integer matrix
//!   - multiply: A @ B for compatible integer matrices
//!   - solve: solve a small linear system A x = b for x
//!
//! Ground truth is computed once at construction time; matrices are
//! compared by converting to a canonical tuple-of-tuples form for exact
//! equality, integers/rationals throughout so there's no floating-point
//! tolerance question to get wrong.

use crate::domain::Domain;
use reasoning_common::{rat_matrix_from_rows, Rat, RatMatrix};
use reasoning_symbolic::{nsimplify, Expr};

/// Python `_canonical(m)` — exact, order-independent-of-representation
/// form for equality checking: every entry as an nsimplify'd string, so
/// Rational(1,2) and "1/2" compare equal regardless of how the candidate
/// was typed.
fn canonical(m: &RatMatrix) -> Vec<Vec<String>> {
    (0..m.rows)
        .map(|i| {
            (0..m.cols)
                .map(|j| nsimplify(&Expr::num(m.at(i, j))).to_string())
                .collect()
        })
        .collect()
}

#[derive(Debug, Clone, PartialEq)]
pub struct LinAlgState {
    pub choice: Option<usize>,
}

pub struct MatrixDeterminantDomain {
    pub matrix: RatMatrix,
    pub candidates: Vec<i64>,
    pub ground_truth: i64,
}

impl MatrixDeterminantDomain {
    /// Python `__init__` raises ValueError for non-square matrices.
    pub fn try_new(matrix_rows: &[Vec<i64>], candidates: Vec<i64>) -> Result<Self, String> {
        let matrix = rat_matrix_from_rows(matrix_rows);
        if matrix.rows != matrix.cols {
            return Err("determinant requires a square matrix".to_string());
        }
        let d = matrix.det();
        // Python int(self.matrix.det()) truncates toward zero
        let ground_truth = d.num / d.den;
        Ok(MatrixDeterminantDomain {
            matrix,
            candidates,
            ground_truth,
        })
    }
}

impl Domain for MatrixDeterminantDomain {
    type State = LinAlgState;
    type Action = usize;

    fn initial_state(&self) -> LinAlgState {
        LinAlgState { choice: None }
    }

    fn legal_actions(&self, state: &LinAlgState) -> Vec<usize> {
        if state.choice.is_some() {
            return Vec::new();
        }
        (0..self.candidates.len()).collect()
    }

    fn apply(&self, _state: &LinAlgState, action: &usize) -> LinAlgState {
        LinAlgState {
            choice: Some(*action),
        }
    }

    fn is_terminal(&self, state: &LinAlgState) -> bool {
        state.choice.is_some()
    }

    fn terminal_reward(&self, state: &LinAlgState) -> f64 {
        assert!(self.is_terminal(state));
        let candidate = self.candidates[state.choice.expect("is_terminal checked")];
        if candidate == self.ground_truth {
            return 1.0;
        }
        // Phase 101: matching the Bezout-domain lesson -- a flat 0.0 for
        // every non-exact candidate gives a bandit-style search over
        // thousands of candidate integers zero gradient toward the right
        // region. Distance-shaped partial credit only affects which
        // candidate gets EXPLORED more; result.found_verified_solution
        // (elsewhere, thresholded at reward>=0.999) still requires an
        // exact match, so this cannot turn a wrong guess into a false
        // "verified" answer.
        let scale = (self.ground_truth.abs() + 10) as f64;
        (1.0 - (candidate - self.ground_truth).abs() as f64 / scale).max(0.0)
    }
}

pub struct MatrixMultiplyDomain {
    pub a: RatMatrix,
    pub b: RatMatrix,
    pub candidates: Vec<Vec<Vec<i64>>>,
    pub ground_truth: Vec<Vec<String>>,
}

impl MatrixMultiplyDomain {
    /// Python `__init__` raises ValueError on incompatible shapes.
    pub fn try_new(
        a_rows: &[Vec<i64>],
        b_rows: &[Vec<i64>],
        candidates: Vec<Vec<Vec<i64>>>,
    ) -> Result<Self, String> {
        let a = rat_matrix_from_rows(a_rows);
        let b = rat_matrix_from_rows(b_rows);
        if a.cols != b.rows {
            // Python: f"incompatible shapes for multiplication: {A.shape} x {B.shape}"
            return Err(format!(
                "incompatible shapes for multiplication: ({}, {}) x ({}, {})",
                a.rows, a.cols, b.rows, b.cols
            ));
        }
        let ground_truth = canonical(&a.mat_mul(&b));
        Ok(MatrixMultiplyDomain {
            a,
            b,
            candidates,
            ground_truth,
        })
    }
}

impl Domain for MatrixMultiplyDomain {
    type State = LinAlgState;
    type Action = usize;

    fn initial_state(&self) -> LinAlgState {
        LinAlgState { choice: None }
    }

    fn legal_actions(&self, state: &LinAlgState) -> Vec<usize> {
        if state.choice.is_some() {
            return Vec::new();
        }
        (0..self.candidates.len()).collect()
    }

    fn apply(&self, _state: &LinAlgState, action: &usize) -> LinAlgState {
        LinAlgState {
            choice: Some(*action),
        }
    }

    fn is_terminal(&self, state: &LinAlgState) -> bool {
        state.choice.is_some()
    }

    fn terminal_reward(&self, state: &LinAlgState) -> f64 {
        assert!(self.is_terminal(state));
        let rows = &self.candidates[state.choice.expect("is_terminal checked")];
        // Python wraps candidate construction in try/except: a malformed
        // (ragged) candidate matrix scores 0 instead of crashing
        let width = rows.first().map(|r| r.len()).unwrap_or(0);
        if rows.iter().any(|r| r.len() != width) {
            return 0.0;
        }
        let candidate = canonical(&rat_matrix_from_rows(rows));
        if candidate == self.ground_truth {
            1.0
        } else {
            0.0
        }
    }
}

pub struct LinearSystemDomain {
    pub a: RatMatrix,
    pub b: RatMatrix,
    pub candidates: Vec<Vec<i64>>,
    pub ground_truth: Vec<String>,
}

impl LinearSystemDomain {
    /// Python `__init__` raises on shape mismatch and on non-unique
    /// solutions (A.solve never silently guesses).
    pub fn try_new(a_rows: &[Vec<i64>], b_col: &[i64], candidates: Vec<Vec<i64>>) -> Result<Self, String> {
        let a = rat_matrix_from_rows(a_rows);
        let b = RatMatrix {
            rows: b_col.len(),
            cols: 1,
            data: b_col.iter().map(|&v| Rat::from_int(v)).collect(),
        };
        if a.rows != b.rows {
            return Err("A and b row counts must match".to_string());
        }
        // raises if not uniquely solvable — never silently guesses
        let solution = a.solve(&b)?;
        let ground_truth = (0..solution.rows)
            .map(|i| nsimplify(&Expr::num(solution.at(i, 0))).to_string())
            .collect();
        Ok(LinearSystemDomain {
            a,
            b,
            candidates,
            ground_truth,
        })
    }
}

impl Domain for LinearSystemDomain {
    type State = LinAlgState;
    type Action = usize;

    fn initial_state(&self) -> LinAlgState {
        LinAlgState { choice: None }
    }

    fn legal_actions(&self, state: &LinAlgState) -> Vec<usize> {
        if state.choice.is_some() {
            return Vec::new();
        }
        (0..self.candidates.len()).collect()
    }

    fn apply(&self, _state: &LinAlgState, action: &usize) -> LinAlgState {
        LinAlgState {
            choice: Some(*action),
        }
    }

    fn is_terminal(&self, state: &LinAlgState) -> bool {
        state.choice.is_some()
    }

    fn terminal_reward(&self, state: &LinAlgState) -> f64 {
        assert!(self.is_terminal(state));
        let candidate = &self.candidates[state.choice.expect("is_terminal checked")];
        let candidate_strs: Vec<String> = candidate
            .iter()
            .map(|v| nsimplify(&Expr::num(Rat::from_int(*v))).to_string())
            .collect();
        if candidate_strs == self.ground_truth {
            1.0
        } else {
            0.0
        }
    }
}
