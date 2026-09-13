//! Phase 038 — Critic role (design doc section 22: Solver / Critic /
//! Verifier / ... roles) (Rust port of `python/uncertainty/critic.py`).
//!
//! A subtle, real gap the Solver alone doesn't catch: an equation can be
//! correctly solved and verified, and STILL be the wrong answer to the
//! actual word problem, if the problem implies a constraint the equation
//! doesn't encode (e.g. "how many apples" implies a non-negative integer,
//! even though "2x = -6" is a perfectly valid, verifiable equation with
//! x=-3). The Critic re-checks the Solver's verified answer against
//! constraints stated separately, and can reject an answer the Solver and
//! Verifier both accepted.
//!
//! (The Python `check` field holds an arbitrary lambda; the project only
//! ever ships the three plain-function constraints below, so the Rust field
//! is a plain `fn(f64) -> bool` pointer — same names, descriptions and
//! thresholds.)

use reasoning_common::py_round;

/// (No `PartialEq`: the `check` field is a fn pointer, and comparing
/// function pointers is meaningless.)
#[derive(Debug, Clone, Copy)]
pub struct Constraint {
    pub name: &'static str,
    pub check: fn(f64) -> bool,
    pub description: &'static str,
}

fn non_negative_check(x: f64) -> bool {
    x >= 0.0
}

fn integer_valued_check(x: f64) -> bool {
    // Python: abs(x - round(x)) < 1e-9 (banker's rounding)
    (x - py_round(x, 0)).abs() < 1e-9
}

fn positive_check(x: f64) -> bool {
    x > 0.0
}

pub const NON_NEGATIVE: Constraint = Constraint {
    name: "non_negative",
    check: non_negative_check,
    description: "answer must not be negative",
};

pub const INTEGER_VALUED: Constraint = Constraint {
    name: "integer_valued",
    check: integer_valued_check,
    description: "answer must be a whole number",
};

pub const POSITIVE: Constraint = Constraint {
    name: "positive",
    check: positive_check,
    description: "answer must be strictly positive",
};

#[derive(Debug, Clone, PartialEq)]
pub struct CritiqueResult {
    pub solver_verified: bool,
    pub all_constraints_satisfied: bool,
    pub violated_constraints: Vec<String>,
    pub final_accept: bool,
}

/// The Critic's job: even a verified-correct answer to the EQUATION can be
/// rejected here if it violates a real-world constraint the equation itself
/// doesn't know about. `final_accept` is the actual gate a caller should
/// use — `solver_verified` alone is not enough.
pub fn critique(
    solver_verified: bool,
    answer: Option<f64>,
    constraints: &[Constraint],
) -> CritiqueResult {
    if !solver_verified || answer.is_none() {
        return CritiqueResult {
            solver_verified: false,
            all_constraints_satisfied: false,
            violated_constraints: Vec::new(),
            final_accept: false,
        };
    }

    let x = answer.expect("checked above");
    let violated: Vec<String> = constraints
        .iter()
        .filter(|c| !(c.check)(x))
        .map(|c| c.name.to_string())
        .collect();
    let all_ok = violated.is_empty();
    CritiqueResult {
        solver_verified: true,
        all_constraints_satisfied: all_ok,
        violated_constraints: violated,
        final_accept: all_ok,
    }
}
