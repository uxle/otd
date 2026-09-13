//! Rust port of python/tests/test_word_problems.py.

use reasoning_curriculum::word_problems::{parse_word_problem, solve_word_problem};
use reasoning_verifier::symbolic_verifier::verify_equation_solution;

// ── TestWordProblemParsing ──────────────────────────────────────────────
#[test]
fn test_parses_more_than_times_template() {
    let p = parse_word_problem("4 more than 3 times a number is 19");
    assert!(p.is_some());
    assert_eq!(p.unwrap().equation, "3*x + 4 = 19");
}

#[test]
fn test_parses_sum_template() {
    let p = parse_word_problem("The sum of a number and 7 is 15");
    assert!(p.is_some());
    assert_eq!(p.unwrap().equation, "x + 7 = 15");
}

#[test]
fn test_parses_minus_template() {
    let p = parse_word_problem("A number minus 5 is 10");
    assert_eq!(p.unwrap().equation, "x - 5 = 10");
}

#[test]
fn test_honestly_fails_on_unrecognized_phrasing() {
    let p = parse_word_problem("What is the airspeed velocity of an unladen swallow?");
    assert!(p.is_none());
}

// ── TestWordProblemFullPipeline ─────────────────────────────────────────
#[test]
fn test_solves_end_to_end() {
    let (equation, answer) = solve_word_problem("4 more than 3 times a number is 19", 800, 1);
    assert_eq!(equation.as_deref(), Some("3*x + 4 = 19"));
    assert!(answer.is_some());
    let answer = answer.unwrap();
    // independently re-verify the final numeric answer
    let check = verify_equation_solution("3*x + 4 = 19", "x", answer, 1e-9).unwrap();
    assert!(check.passed);
    assert!((answer - 5.0).abs() < 1e-9);
}

#[test]
fn test_end_to_end_matches_hand_solved_answer() {
    // "the sum of a number and 7 is 15" -> x = 8
    let (_equation, answer) = solve_word_problem("The sum of a number and 7 is 15", 800, 2);
    assert!((answer.unwrap() - 8.0).abs() < 1e-9);
}

#[test]
fn test_unparseable_problem_returns_none_not_a_fabricated_answer() {
    let (equation, answer) = solve_word_problem("Tell me a story about a dragon", 800, 3);
    assert!(equation.is_none());
    assert!(answer.is_none());
}
