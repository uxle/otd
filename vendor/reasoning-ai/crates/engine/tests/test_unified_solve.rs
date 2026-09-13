//! Port of python/tests/test_unified_solve.py — Phase 030's unified solve()
//! API exercised problem-by-problem through `Problem` + `solve`.

use reasoning_engine::solve::{solve, Problem};
use serde_json::{json, Value};

fn p(kind: &str, payload: Value) -> Problem {
    Problem::new(kind, serde_json::from_value(payload).unwrap())
}

#[test]
fn test_number_target() {
    let ans = solve(&p("number_target", json!({"numbers": [4, 7, 8, 8], "target": 24})), None, 1);
    assert!(ans.verified);
    assert!(ans.final_answer().is_some());
}

#[test]
fn test_linear_equation() {
    let ans = solve(&p("linear_equation", json!({"equation": "2*x + 4 = 10"})), None, 2);
    assert!(ans.verified);
    let v: f64 = ans.final_answer().unwrap().parse().unwrap();
    // Python assertAlmostEqual default: 7 decimal places
    assert!((v - 3.0).abs() < 1e-7);
}

#[test]
fn test_quadratic_factoring() {
    let ans = solve(&p("quadratic_factoring", json!({"b": -5, "c": 6})), None, 3);
    assert!(ans.verified);
    assert!(ans.final_answer().is_some());
}

#[test]
fn test_gcd_bezout() {
    let ans = solve(&p("gcd_bezout", json!({"a": 35, "b": 12})), None, 4);
    assert!(ans.verified);
}

#[test]
fn test_combinatorics() {
    let ans = solve(
        &p("combinatorics", json!({"ctype": "combinations", "n": 5, "r": 2})),
        None,
        5,
    );
    assert!(ans.verified);
    assert_eq!(ans.final_answer(), Some("10"));
}

#[test]
fn test_word_problem() {
    let ans = solve(
        &p("word_problem", json!({"text": "4 more than 3 times a number is 19"})),
        None,
        6,
    );
    assert!(ans.verified);
    let v: f64 = ans.final_answer().unwrap().parse().unwrap();
    assert!((v - 5.0).abs() < 1e-7);
}

#[test]
fn test_unknown_kind_abstains_cleanly() {
    let ans = solve(&p("something_undefined", json!({})), None, 7);
    assert!(!ans.verified);
    assert!(ans.final_answer().is_none());
    assert!(ans.explanation.contains("Unknown problem kind"));
}

#[test]
fn test_unparseable_word_problem_abstains() {
    let ans = solve(&p("word_problem", json!({"text": "Tell me about your day"})), None, 8);
    assert!(!ans.verified);
    assert!(ans.final_answer().is_none());
}

#[test]
fn test_unreachable_number_target_abstains() {
    let ans = solve(
        &p("number_target", json!({"numbers": [1, 1, 1], "target": 1000000000})),
        Some(300),
        9,
    );
    assert!(!ans.verified);
    assert!(ans.final_answer().is_none());
}

#[test]
fn test_injection_via_public_api_is_blocked() {
    // Phase 046 found and fixed a real code-execution vulnerability in
    // the shared parser. This confirms the fix holds at the actual
    // public entry point (solve()), not just in isolated unit tests.
    let ans = solve(
        &p(
            "linear_equation",
            json!({"equation": "__import__(\"os\").system(\"echo x\") + x = 0"}),
        ),
        None,
        10,
    );
    assert!(!ans.verified);
    assert!(ans.final_answer().is_none());
}
