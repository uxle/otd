//! Phase 037 — Composite (two-step) word problems (Rust port of
//! python/curriculum/composite_word_problems.py).
//!
//! Extends Phase 028's single-equation word problems to ones requiring TWO
//! chained facts, e.g.: "A number doubled is 14. That result plus 5 equals
//! what?" Step 1 solves for the number (real algebra search, Phase 006).
//! Step 2 uses the VERIFIED result from working memory (Phase 036) — if
//! step 1 isn't verified, step 2 refuses to proceed rather than silently
//! carrying forward a guess.

use regex::Regex;
use std::sync::OnceLock;

use reasoning_memory::working_memory::{WorkingMemory, WorkingMemoryEntry, WmValue};
use reasoning_search::{Mcts, LinearEquationDomain, Domain};
use reasoning_verifier::symbolic_verifier::{verify_equation_solution, verify_numeric_equality};
use reasoning_common::py_float_str;

fn step1_pattern() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)a number doubled is\s+(-?\d+)").unwrap())
}

fn step2_pattern() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)that result plus\s+(-?\d+)\s+equals what").unwrap())
}

/// Python `@dataclass CompositeResult` (trace = the working-memory entries).
#[derive(Debug, Clone, PartialEq)]
pub struct CompositeResult {
    pub step1_equation: Option<String>,
    pub step1_answer: Option<f64>,
    pub step2_answer: Option<f64>,
    pub trace: Vec<WorkingMemoryEntry>,
}

/// Python `solve_composite_word_problem(text, budget=800, seed=0)`.
pub fn solve_composite_word_problem(
    text: &str,
    budget: usize,
    seed: u64,
) -> CompositeResult {
    let mut wm = WorkingMemory::new();
    let m1 = step1_pattern().captures(text);
    let m2 = step2_pattern().captures(text);
    let (m1, m2) = match (m1, m2) {
        (Some(a), Some(b)) => (a, b),
        _ => {
            return CompositeResult {
                step1_equation: None,
                step1_answer: None,
                step2_answer: None,
                trace: Vec::new(),
            }
        }
    };

    let target1: i64 = m1[1].parse().expect("regex guarantees an integer");
    let equation1 = format!("2*x = {}", target1);

    // Python's LinearEquationDomain.__init__ could raise; "2*x = <int>"
    // always parses and solves, so an Err here is reported as unsolved
    // instead of crashing the caller.
    let domain = match LinearEquationDomain::try_new(&equation1, 6) {
        Ok(d) => d,
        Err(_) => {
            wm.set(
                "step1",
                WmValue::PyNone,
                false,
                &format!("unsolved: {}", equation1),
            );
            return CompositeResult {
                step1_equation: Some(equation1),
                step1_answer: None,
                step2_answer: None,
                trace: wm.trace().into_iter().cloned().collect(),
            };
        }
    };
    let mut mcts = Mcts::new(domain.clone(), 6, seed);
    let result = mcts.search(domain.initial_state(), budget);

    if !result.found_verified_solution {
        wm.set(
            "step1",
            WmValue::PyNone,
            false,
            &format!("unsolved: {}", equation1),
        );
        return CompositeResult {
            step1_equation: Some(equation1),
            step1_answer: None,
            step2_answer: None,
            trace: wm.trace().into_iter().cloned().collect(),
        };
    }

    // Python: float(result.best_terminal_state.rhs)
    let step1_value = result
        .best_terminal_state
        .as_ref()
        .and_then(|s| s.rhs.as_rat())
        .map(|r| r.to_f64());
    let step1_value = match step1_value {
        Some(v) => v,
        None => {
            // unreachable: a verified solution implies a numeric rhs
            wm.set(
                "step1",
                WmValue::PyNone,
                false,
                &format!("unsolved: {}", equation1),
            );
            return CompositeResult {
                step1_equation: Some(equation1),
                step1_answer: None,
                step2_answer: None,
                trace: wm.trace().into_iter().cloned().collect(),
            };
        }
    };
    // Python would raise if verification itself failed to parse (it can't
    // for "2*x = <int>"); this port fails closed to not-verified.
    let check1_passed = verify_equation_solution(&equation1, "x", step1_value, 1e-9)
        .map(|r| r.passed)
        .unwrap_or(false);
    wm.set(
        "step1",
        WmValue::Num(step1_value),
        check1_passed,
        &format!("solved {}", equation1),
    );

    // step 2 REQUIRES a verified step 1 -- get_verified_value enforces this
    let verified_step1 = match wm.get_verified_value("step1") {
        Ok(WmValue::Num(v)) => v,
        // Python: except ValueError -> bail out with step1 only. (An entry
        // that is verified but not a number can't occur on this path.)
        _ => {
            return CompositeResult {
                step1_equation: Some(equation1),
                step1_answer: Some(step1_value),
                step2_answer: None,
                trace: wm.trace().into_iter().cloned().collect(),
            }
        }
    };

    let addend: i64 = m2[1].parse().expect("regex guarantees an integer");
    let step2_value = verified_step1 + addend as f64;
    // Python f"{verified_step1} + {addend}" -> float repr + int
    let step2_expr = format!("{} + {}", py_float_str(verified_step1), addend);
    let check2_passed = verify_numeric_equality(&step2_expr, step2_value)
        .map(|r| r.passed)
        .unwrap_or(false);
    wm.set(
        "step2",
        WmValue::Num(step2_value),
        check2_passed,
        &step2_expr,
    );

    CompositeResult {
        step1_equation: Some(equation1),
        step1_answer: Some(verified_step1),
        step2_answer: Some(step2_value),
        trace: wm.trace().into_iter().cloned().collect(),
    }
}
