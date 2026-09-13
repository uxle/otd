//! Phase 023 — Multi-path reasoning (Rust port of `search/multi_path.py`).
//! Runs MCTS, best-first, and beam search independently on the same
//! number-target problem, re-verifies every claimed answer through the
//! verifier, and returns the first verified success — or an honest
//! "no strategy found a verified answer".

use crate::domain::Domain;
use crate::heuristic_search::{beam_search, best_first_search};
use crate::mcts::Mcts;
use crate::number_target_domain::{NumberTargetDomain, NtState};
use reasoning_symbolic::eval_f64;

#[derive(Debug, Clone)]
pub struct PathResult {
    pub strategy: String,
    pub found: bool,
    pub answer_expr: Option<String>,
    /// nodes/simulations used
    pub cost: usize,
}

#[derive(Debug, Clone)]
pub struct MultiPathResult {
    pub paths: Vec<PathResult>,
    pub consensus_answer: Option<String>,
    pub contradiction_detected: bool,
}

/// Python parity: `solve_number_target_multi_path(domain, root_state,
/// target, seed=0)`.
pub fn solve_number_target_multi_path(
    domain: &NumberTargetDomain,
    root_state: &NtState,
    target: f64,
    seed: u64,
) -> MultiPathResult {
    let distance_h = |state: &NtState| -> f64 {
        -state
            .iter()
            .map(|(v, _)| (v - target).abs())
            .fold(f64::INFINITY, f64::min)
    };

    let mut paths: Vec<PathResult> = Vec::new();

    let mut mcts = Mcts::new(domain.clone(), 6, seed);
    let r = mcts.search(root_state.clone(), 500);
    paths.push(PathResult {
        strategy: "mcts".to_string(),
        found: r.found_verified_solution,
        answer_expr: r
            .found_verified_solution
            .then(|| r.best_terminal_state.as_ref().map(|s| s[0].1.clone()))
            .flatten(),
        cost: r.nodes_expanded,
    });

    let (found_bf, term_bf, exp_bf) =
        best_first_search(domain, root_state.clone(), distance_h, 500, 6);
    paths.push(PathResult {
        strategy: "best_first".to_string(),
        found: found_bf,
        answer_expr: found_bf.then(|| term_bf.map(|s| s[0].1.clone())).flatten(),
        cost: exp_bf,
    });

    let (found_beam, term_beam, exp_beam) =
        beam_search(domain, root_state.clone(), distance_h, 100, 6);
    paths.push(PathResult {
        strategy: "beam".to_string(),
        found: found_beam,
        answer_expr: found_beam.then(|| term_beam.map(|s| s[0].1.clone())).flatten(),
        cost: exp_beam,
    });

    // independently re-verify every claimed answer — a strategy reporting
    // "found" is not trusted on its own word
    let mut verified_answers: Vec<String> = Vec::new();
    for p in &paths {
        if let Some(expr) = &p.answer_expr {
            if p.found && verify_numeric_eq(expr, target).unwrap_or(false) {
                verified_answers.push(expr.clone());
            }
        }
    }

    let mut contradiction = false;
    let mut consensus: Option<String> = None;
    if !verified_answers.is_empty() {
        consensus = Some(verified_answers[0].clone());
        for a in &verified_answers[1..] {
            if !verify_numeric_eq(a, target).unwrap_or(false) {
                contradiction = true;
            }
        }
    }

    MultiPathResult {
        paths,
        consensus_answer: consensus,
        contradiction_detected: contradiction,
    }
}

/// verify_numeric_equality from the verifier, inlined here to avoid a
/// crate cycle (the verifier depends on search-free symbolic parsing only).
fn verify_numeric_eq(expr_str: &str, expected: f64) -> Option<bool> {
    match reasoning_symbolic::parse_expr(expr_str) {
        Ok(e) => match eval_f64(&e, &std::collections::HashMap::new()) {
            Ok(v) => Some((v - expected).abs() <= 1e-9),
            Err(_) => None,
        },
        Err(_) => None,
    }
}

// keep the trait import used even though multi_path is domain-specific
#[allow(unused)]
fn _assert_domain_object_safe<D: Domain>(d: &D) {
    let _ = d;
}
