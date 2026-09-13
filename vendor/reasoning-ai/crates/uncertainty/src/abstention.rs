//! Phase 024 — Uncertainty & abstention (design doc section 23: "the system
//! should know when it is uncertain"; section 18: "DO NOT automatically
//! fabricate an answer") (Rust port of `python/uncertainty/abstention.py`).
//!
//! Wraps multi-path search (Phase 023) with an explicit answer type that
//! separates "verified answer" from "unverified guess" at the type level —
//! `final_answer()` only ever returns something when `verified=true`, so
//! calling code structurally cannot accidentally present an unverified guess
//! as a real answer.

use reasoning_search::{
    solve_number_target_multi_path, Mcts, NumberTargetDomain, NtState,
};

#[derive(Debug, Clone, PartialEq)]
pub struct Answer {
    pub verified: bool,
    pub answer_expr: Option<String>,
    pub confidence: f64,
    /// informational only, never a substitute answer
    pub unverified_best_guess: Option<String>,
    pub explanation: String,
}

impl Answer {
    /// Python positional dataclass construction:
    /// `Answer(True, "x", 1.0, None, "msg")`.
    pub fn new(
        verified: bool,
        answer_expr: Option<&str>,
        confidence: f64,
        guess: Option<&str>,
        explanation: &str,
    ) -> Self {
        Answer {
            verified,
            answer_expr: answer_expr.map(|s| s.to_string()),
            confidence,
            unverified_best_guess: guess.map(|s| s.to_string()),
            explanation: explanation.to_string(),
        }
    }

    /// The only sanctioned way to extract an answer. Returns `None`
    /// whenever nothing was verified — there is no code path that lets an
    /// unverified guess masquerade as a real answer.
    pub fn final_answer(&self) -> Option<&str> {
        self.answer_expr.as_deref().filter(|_| self.verified)
    }
}

/// Python `solve_with_abstention(domain, root_state, target, seed=0)`.
pub fn solve_with_abstention(
    domain: &NumberTargetDomain,
    root_state: &NtState,
    target: f64,
    seed: u64,
) -> Answer {
    let result = solve_number_target_multi_path(domain, root_state, target, seed);

    if result.consensus_answer.is_some() && !result.contradiction_detected {
        return Answer::new(
            true,
            result.consensus_answer.as_deref(),
            1.0, // exact symbolic/numeric verification, not a guess
            None,
            "Verified by independent symbolic re-check across multiple search strategies.",
        );
    }

    // Nothing verified. Still run one more MCTS pass with generous budget
    // purely to surface the *closest* attempt for transparency — labeled
    // clearly as unverified, never returned by final_answer().
    let mut mcts = Mcts::new(domain.clone(), 6, seed + 999);
    let r = mcts.search(root_state.clone(), 800);
    // Python: best_guess = r.best_terminal_state[0][1] in a
    // try/except (TypeError, IndexError) -> None
    let best_guess: Option<String> = r
        .best_terminal_state
        .as_ref()
        .and_then(|s| s.first())
        .map(|(_, expr)| expr.clone());

    Answer::new(
        false,
        None,
        0.0,
        best_guess.as_deref(),
        if !result.contradiction_detected {
            "No search strategy produced a verifier-confirmed solution. Abstaining rather than guessing."
        } else {
            "Strategies produced contradictory verified answers -- abstaining pending review."
        },
    )
}
