//! Rust port of python/tests/test_composite_word_problems.py.

use reasoning_curriculum::composite_word_problems::solve_composite_word_problem;

// ── TestCompositeWordProblems ───────────────────────────────────────────
#[test]
fn test_solves_two_step_problem_end_to_end() {
    let text = "A number doubled is 14. That result plus 5 equals what?";
    let result = solve_composite_word_problem(text, 800, 1);
    assert_eq!(result.step1_equation.as_deref(), Some("2*x = 14"));
    assert!((result.step1_answer.unwrap() - 7.0).abs() < 1e-9);
    assert!((result.step2_answer.unwrap() - 12.0).abs() < 1e-9);
}

#[test]
fn test_working_memory_trace_shows_both_steps() {
    let text = "A number doubled is 14. That result plus 5 equals what?";
    let result = solve_composite_word_problem(text, 800, 2);
    let names: Vec<&str> = result.trace.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["step1", "step2"]);
    assert!(result.trace.iter().all(|e| e.verified));
}

#[test]
fn test_different_numbers_produce_correct_chained_answer() {
    let text = "A number doubled is 20. That result plus 3 equals what?";
    let result = solve_composite_word_problem(text, 800, 3);
    assert!((result.step1_answer.unwrap() - 10.0).abs() < 1e-9);
    assert!((result.step2_answer.unwrap() - 13.0).abs() < 1e-9);
}

#[test]
fn test_unparseable_text_returns_empty_result() {
    let result = solve_composite_word_problem("nonsense text with no structure", 800, 4);
    assert!(result.step1_equation.is_none());
    assert!(result.step2_answer.is_none());
}
