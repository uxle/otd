//! Port of python/tests/test_validation.py — Phase 045 input validation
//! & resource limits. A caller (or an attacker) could pass a negative
//! simulation budget, an absurdly large one, malformed payload dicts, or
//! an unknown domain kind; these tests specifically try to break it.
//!
//! Deviation: Python `budget=-100` cannot be represented in Rust's
//! `Option<usize>` — negative budgets are now a *type-level* guarantee
//! against exactly this attack. The smallest representable invalid value
//! (0, below MIN_ALLOWED_BUDGET=1) exercises the same rejection branch.

use std::time::Instant;

use reasoning_engine::solve::Problem;
use reasoning_engine::validation::{solve_validated, validate_problem};
use serde_json::{json, Value};

fn p(kind: &str, payload: Value) -> Problem {
    Problem::new(kind, serde_json::from_value(payload).unwrap())
}

#[test]
fn test_valid_problem_passes() {
    let problem = p("linear_equation", json!({"equation": "2*x = 4"}));
    assert!(validate_problem(&problem, None).is_ok()); // should not raise
}

#[test]
fn test_unknown_kind_rejected() {
    let problem = p("nonexistent_kind", json!({}));
    assert!(validate_problem(&problem, None).is_err());
}

#[test]
fn test_missing_required_key_rejected() {
    let problem = p("linear_equation", json!({})); // missing "equation"
    assert!(validate_problem(&problem, None).is_err());
}

#[test]
fn test_oversized_number_list_rejected() {
    let numbers: Vec<i64> = (1..50).collect();
    let problem = p("number_target", json!({"numbers": numbers, "target": 5}));
    assert!(validate_problem(&problem, None).is_err());
}

#[test]
fn test_negative_budget_rejected() {
    // Python passed budget=-100; Rust's Option<usize> cannot hold a
    // negative, so the type already prevents that. 0 is the closest
    // representable attack: below MIN_ALLOWED_BUDGET.
    let problem = p("linear_equation", json!({"equation": "2*x=4"}));
    assert!(validate_problem(&problem, Some(0)).is_err());
}

#[test]
fn test_absurdly_large_budget_rejected() {
    let problem = p("linear_equation", json!({"equation": "2*x=4"}));
    assert!(validate_problem(&problem, Some(1_000_000_000)).is_err());
}

#[test]
fn test_oversized_equation_string_rejected() {
    let huge_equation = format!("{}1 = 0", "x + ".repeat(1000));
    let problem = p("linear_equation", json!({"equation": huge_equation}));
    assert!(validate_problem(&problem, None).is_err());
}

#[test]
fn test_non_numeric_values_in_numbers_rejected() {
    let problem = p("number_target", json!({"numbers": [1, "two", 3], "target": 5}));
    assert!(validate_problem(&problem, None).is_err());
}

#[test]
fn test_solve_validated_rejects_before_spending_any_search_time() {
    let numbers: Vec<i64> = (0..100).collect();
    let problem = p("number_target", json!({"numbers": numbers, "target": 5}));
    let start = Instant::now();
    assert!(solve_validated(&problem, Some(5000), 0).is_err());
    let elapsed = start.elapsed();
    assert!(
        elapsed.as_secs_f64() < 0.5,
        "validation should reject instantly, before any search runs (took {:?})",
        elapsed
    );
}

#[test]
fn test_solve_validated_still_works_for_valid_input() {
    let problem = p("linear_equation", json!({"equation": "2*x + 4 = 10"}));
    let ans = solve_validated(&problem, None, 1).unwrap();
    assert!(ans.verified);
}
