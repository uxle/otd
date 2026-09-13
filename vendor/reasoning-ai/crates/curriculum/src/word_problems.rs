//! Phase 028 — Word problem -> equation extraction (design doc 3.5 level 4)
//! (Rust port of python/curriculum/word_problems.py).
//!
//! Template-based, not NLP — deliberately honest about scope: this matches
//! a small set of common phrasings ("N more than X times a number is Y",
//! "the sum of a number and N is Y", etc.) to a linear equation, then hands
//! off to the already-verified LinearEquationDomain (Phase 006) and MCTS
//! solver. Anything outside these templates is correctly reported as
//! unparseable, not guessed at.

use regex::Regex;
use std::sync::OnceLock;

use reasoning_search::{Mcts, LinearEquationDomain, Domain};

/// Python `@dataclass ParsedProblem`.
#[derive(Debug, Clone, PartialEq)]
pub struct ParsedProblem {
    pub equation: String,
    pub variable: String,
}

impl ParsedProblem {
    pub fn new(equation: String) -> ParsedProblem {
        ParsedProblem {
            equation,
            variable: "x".to_string(),
        }
    }
}

// "N more than K times a number is M"  ->  K*x + N = M
fn tpl_more_than_times() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)(-?\d+)\s+more than\s+(-?\d+)\s+times a number is\s+(-?\d+)").unwrap()
    })
}

// "K times a number plus N equals M"
fn tpl_times_plus() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?i)(-?\d+)\s+times a number plus\s+(-?\d+)\s+equals\s+(-?\d+)").unwrap()
    })
}

// "the sum of a number and N is M"  ->  x + N = M
fn tpl_sum() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)the sum of a number and\s+(-?\d+)\s+is\s+(-?\d+)").unwrap())
}

// "a number minus N is M"  ->  x - N = M
fn tpl_minus() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)a number minus\s+(-?\d+)\s+is\s+(-?\d+)").unwrap())
}

// "K times a number is M"  ->  K*x = M
fn tpl_times_is() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?i)(-?\d+)\s+times a number is\s+(-?\d+)").unwrap())
}

/// Python `parse_word_problem`: honestly reports "couldn't parse" (None),
/// never guesses.
pub fn parse_word_problem(text: &str) -> Option<ParsedProblem> {
    let text = text.trim().trim_end_matches('.');
    if let Some(m) = tpl_more_than_times().captures(text) {
        return Some(ParsedProblem::new(format!(
            "{}*x + {} = {}",
            &m[2], &m[1], &m[3]
        )));
    }
    if let Some(m) = tpl_times_plus().captures(text) {
        return Some(ParsedProblem::new(format!(
            "{}*x + {} = {}",
            &m[1], &m[2], &m[3]
        )));
    }
    if let Some(m) = tpl_sum().captures(text) {
        return Some(ParsedProblem::new(format!("x + {} = {}", &m[1], &m[2])));
    }
    if let Some(m) = tpl_minus().captures(text) {
        return Some(ParsedProblem::new(format!("x - {} = {}", &m[1], &m[2])));
    }
    if let Some(m) = tpl_times_is().captures(text) {
        return Some(ParsedProblem::new(format!("{}*x = {}", &m[1], &m[2])));
    }
    None
}

/// Python `solve_word_problem(text, budget=800, seed=0)`.
///
/// Full pipeline: parse -> solve via the proven algebra domain -> verify.
/// Returns (parsed_equation_or_None, verified_answer_or_None).
///
/// Deviation from Python: `LinearEquationDomain.__init__` could raise (an
/// unsolvable template like "0 times a number is 5"); Python would crash
/// the caller, this port reports it as "parsed but unsolved" — unreachable
/// through the template shapes the tests exercise.
pub fn solve_word_problem(text: &str, budget: usize, seed: u64) -> (Option<String>, Option<f64>) {
    let parsed = match parse_word_problem(text) {
        Some(p) => p,
        None => return (None, None),
    };

    let domain = match LinearEquationDomain::try_new(&parsed.equation, 6) {
        Ok(d) => d,
        Err(_) => return (Some(parsed.equation), None),
    };
    let mut mcts = Mcts::new(domain.clone(), 6, seed);
    let result = mcts.search(domain.initial_state(), budget);
    if !result.found_verified_solution {
        return (Some(parsed.equation), None);
    }
    // Python: float(result.best_terminal_state.rhs)
    let answer = result
        .best_terminal_state
        .as_ref()
        .and_then(|s| s.rhs.as_rat())
        .map(|r| r.to_f64());
    (Some(parsed.equation), answer)
}
