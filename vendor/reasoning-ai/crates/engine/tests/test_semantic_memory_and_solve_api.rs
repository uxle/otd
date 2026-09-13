//! Port of python/tests/test_semantic_memory_and_solve_api.py — ONLY the
//! deferred TestSolveAPINewDomains class (the typed solve() API for the
//! Phase 085-097 domains). The TestSemanticMemory class already lives in
//! crates/memory/tests/ (ported with the memory crate in an earlier wave).

use reasoning_engine::solve::{solve, Problem};
use serde_json::{json, Value};

fn p(kind: &str, payload: Value) -> Problem {
    Problem::new(kind, serde_json::from_value(payload).unwrap())
}

#[test]
fn test_trig_simplify_via_solve() {
    let answer = solve(&p("trig_simplify", json!({"expr": "sin(x)**2 + cos(x)**2"})), None, 0);
    assert!(answer.verified);
    assert_eq!(answer.final_answer(), Some("1"));
}

#[test]
fn test_trig_evaluate_via_solve() {
    let answer = solve(&p("trig_evaluate", json!({"expr": "sin(pi/6)"})), None, 0);
    assert!(answer.verified);
    assert_eq!(answer.final_answer(), Some("1/2"));
}

#[test]
fn test_matrix_determinant_via_solve() {
    let answer = solve(&p("matrix_determinant", json!({"matrix": [[3, 8], [4, 6]]})), None, 0);
    assert!(answer.verified);
    assert_eq!(answer.final_answer(), Some("-14"));
}

#[test]
fn test_matrix_multiply_via_solve() {
    let answer = solve(
        &p("matrix_multiply", json!({"a": [[1, 2], [3, 4]], "b": [[5, 6], [7, 8]]})),
        None,
        0,
    );
    assert!(answer.verified);
    assert!(answer.final_answer().unwrap().contains("19"));
}

#[test]
fn test_linear_system_via_solve() {
    let answer = solve(
        &p("linear_system", json!({"a": [[1, 1], [1, -1]], "b": [5, 1]})),
        None,
        0,
    );
    assert!(answer.verified);
    assert!(answer.final_answer().unwrap().contains("3"));
}

#[test]
fn test_unsolvable_trig_evaluate_abstains_not_guesses() {
    // An angle/value not in the candidate pool -> must abstain, not fabricate
    let answer = solve(&p("trig_evaluate", json!({"expr": "sin(pi/7)"})), None, 0);
    assert!(!answer.verified);
    assert!(answer.final_answer().is_none());
}

#[test]
fn test_unknown_kind_abstains() {
    let answer = solve(&p("not_a_real_kind", json!({})), None, 0);
    assert!(!answer.verified);
    assert!(answer.final_answer().is_none());
}
