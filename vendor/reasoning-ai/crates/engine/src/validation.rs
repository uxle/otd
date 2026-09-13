//! Phase 045 — Input validation & resource limits (design doc section 27)
//! (Rust port of `python/apps/validation.py`).
//!
//! A caller (or an attacker) could pass a negative simulation budget, an
//! absurdly large one meant to hang the process, malformed payload dicts,
//! or an unknown domain kind. This validates BEFORE any search starts,
//! with tests that specifically try to break it, not just check the happy
//! path. (Python's `InputValidationError` maps to `Err(String)` with the
//! identical message.)

use serde_json::Value;

use crate::solve::{solve, Problem};
use reasoning_uncertainty::Answer;

pub const MAX_ALLOWED_BUDGET: usize = 20_000;
pub const MIN_ALLOWED_BUDGET: usize = 1;

/// Python `_REQUIRED_PAYLOAD_KEYS` (order fixed for deterministic error
/// messages; the Python set's repr order was arbitrary anyway).
const REQUIRED_PAYLOAD_KEYS: [(&str, &[&str]); 6] = [
    ("number_target", &["numbers", "target"]),
    ("linear_equation", &["equation"]),
    ("quadratic_factoring", &["b", "c"]),
    ("gcd_bezout", &["a", "b"]),
    ("combinatorics", &["ctype", "n", "r"]),
    ("word_problem", &["text"]),
];

/// Python `validate_problem` — `Err` carries the InputValidationError
/// message, byte-identical where the tests can observe it.
pub fn validate_problem(problem: &Problem, budget: Option<usize>) -> Result<(), String> {
    let Some((kind, required)) = REQUIRED_PAYLOAD_KEYS
        .iter()
        .find(|(k, _)| *k == problem.kind)
    else {
        return Err(format!("unknown problem kind: '{}'", problem.kind));
    };

    let missing: Vec<&str> = required
        .iter()
        .copied()
        .filter(|key| !problem.payload.contains_key(*key))
        .collect();
    if !missing.is_empty() {
        let rendered = missing
            .iter()
            .map(|k| format!("'{}'", k))
            .collect::<Vec<_>>()
            .join(", ");
        return Err(format!(
            "missing required payload keys for '{}': {{{}}}",
            kind, rendered
        ));
    }

    if problem.kind == "number_target" {
        let numbers = problem.payload.get("numbers");
        let ok = numbers
            .and_then(Value::as_array)
            .map(|arr| (1..=8).contains(&arr.len()) && arr.iter().all(|v| v.is_number()))
            .unwrap_or(false);
        if !ok {
            return Err(
                "numbers must be a list of 1-8 values (larger is a resource-exhaustion risk)"
                    .to_string(),
            );
        }
    }

    if problem.kind == "linear_equation" {
        let eq = problem.payload.get("equation");
        let ok = eq
            .and_then(Value::as_str)
            .map(|s| s.chars().count() <= 200)
            .unwrap_or(false);
        if !ok {
            return Err("equation must be a string under 200 characters".to_string());
        }
        if !eq.and_then(Value::as_str).unwrap_or("").contains('=') {
            return Err("equation must contain '='".to_string());
        }
    }

    if problem.kind == "word_problem" {
        let text = problem.payload.get("text");
        let ok = text
            .and_then(Value::as_str)
            .map(|s| s.chars().count() <= 2000)
            .unwrap_or(false);
        if !ok {
            return Err("word problem text must be a string under 2000 characters".to_string());
        }
    }

    if let Some(budget) = budget {
        if budget < MIN_ALLOWED_BUDGET {
            return Err(format!("budget must be >= {}, got {}", MIN_ALLOWED_BUDGET, budget));
        }
        if budget > MAX_ALLOWED_BUDGET {
            return Err(format!(
                "budget {} exceeds MAX_ALLOWED_BUDGET={} (protects against \
                 resource-exhaustion via an oversized search)",
                budget, MAX_ALLOWED_BUDGET
            ));
        }
    }
    Ok(())
}

/// Validates first, then delegates to Phase 030's solve(). Fails BEFORE any
/// search resources are spent, rather than letting a bad input run partway
/// through and fail confusingly.
pub fn solve_validated(
    problem: &Problem,
    budget: Option<usize>,
    seed: u64,
) -> Result<Answer, String> {
    validate_problem(problem, budget)?;
    Ok(solve(problem, budget, seed))
}
