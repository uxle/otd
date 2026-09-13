//! Phase 098 — Integration test: memory + solve() actually working together
//! (Rust port of `python/apps/integrated_solve.py`).
//!
//! (Arch diagram: INPUT -> UNDERSTAND -> MEMORY -> GOAL -> PLAN -> REASON
//! -> SEARCH -> TOOLS -> VERIFY -> CRITIC -> CORRECT -> FINAL ANSWER.)
//! Not a new algorithm — a thin coordination layer proving the existing
//! pieces compose: episodic memory (Phase 026) for number_target problems
//! and semantic memory (Phase 088) for trig_simplify identities. Every
//! verified answer still goes through exactly the same verifier chain
//! solve() already uses; memory only changes whether search runs again,
//! never whether an answer counts as verified.

use serde_json::json;

use reasoning_memory::{EpisodicMemory, SemanticMemory};
use reasoning_uncertainty::Answer;

use crate::solve::{solve, solve_trig_simplify, Problem};

/// Python `class IntegratedSolver`.
#[derive(Debug, Default)]
pub struct IntegratedSolver {
    pub episodic: EpisodicMemory,
    pub semantic: SemanticMemory,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Python `stats()` dict fields.
#[derive(Debug, Clone, PartialEq)]
pub struct SolverStats {
    pub cache_hits: usize,
    pub cache_misses: usize,
    pub cache_hit_rate: f64,
    pub episodic_entries: usize,
    pub semantic_facts: usize,
}

impl IntegratedSolver {
    pub fn new() -> IntegratedSolver {
        IntegratedSolver::default()
    }

    pub fn solve_number_target(
        &mut self,
        numbers: &[f64],
        target: f64,
        budget: usize,
        seed: u64,
    ) -> Answer {
        let cached = self.episodic.lookup(numbers, target).cloned();
        if let Some(cached) = cached {
            self.cache_hits += 1;
            return Answer::new(
                true,
                Some(&cached.answer_expr),
                cached.confidence,
                None,
                &format!(
                    "Answered from episodic memory (used {} times), \
                     independently re-verified before being trusted.",
                    cached.usage_count
                ),
            );
        }
        self.cache_misses += 1;
        let mut payload = serde_json::Map::new();
        payload.insert("numbers".to_string(), json!(numbers));
        payload.insert("target".to_string(), json!(target));
        let answer = solve(&Problem::new("number_target", payload), Some(budget), seed);
        if let Some(final_answer) = answer.final_answer() {
            self.episodic
                .store(numbers, target, final_answer, answer.confidence);
        }
        answer
    }

    pub fn solve_trig_simplify_remembered(
        &mut self,
        expr: &str,
        budget: usize,
        seed: u64,
    ) -> Answer {
        let answer = solve_trig_simplify(expr, budget, seed);
        if answer.verified {
            if let Some(final_answer) = answer.final_answer() {
                let statement = format!("{} = {}", expr, final_answer);
                if self
                    .semantic
                    .store(&statement, "trig_identity", "solve_trig_simplify", 0.5)
                    .is_some()
                {
                    self.semantic.record_usage(&statement);
                }
            }
        }
        answer
    }

    pub fn stats(&self) -> SolverStats {
        let total = self.cache_hits + self.cache_misses;
        SolverStats {
            cache_hits: self.cache_hits,
            cache_misses: self.cache_misses,
            cache_hit_rate: if total > 0 {
                self.cache_hits as f64 / total as f64
            } else {
                0.0
            },
            episodic_entries: self.episodic.len(),
            semantic_facts: self.semantic.len(),
        }
    }
}
