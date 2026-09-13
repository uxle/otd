//! Phase 088 — Semantic memory (design doc section 14) (Rust port of
//! `python/memory/semantic_memory.py`).
//!
//! Distinct from episodic memory (Phase 026), which caches *specific problem
//! instances* keyed by signature. Semantic memory stores *general
//! facts/identities* learned across many problems ("sin(x)**2 + cos(x)**2 =
//! 1"), reusable regardless of which specific problem surfaced it.
//!
//! "Prevent incorrect memories from becoming permanent knowledge" (14's
//! closing requirement) is enforced structurally: `store()` requires an
//! equation-form statement ("lhs = rhs") and independently re-verifies it
//! via the algebraic verifier before accepting — there is no code path to
//! insert an unverified fact.

use reasoning_verifier::symbolic_verifier::verify_algebraic_equivalence;
use std::time::{SystemTime, UNIX_EPOCH};

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct SemanticFact {
    /// e.g. "sin(x)**2 + cos(x)**2 = 1"
    pub statement: String,
    /// e.g. "trig_identity", "matrix_property"
    pub category: String,
    /// Which domain/phase verified this, for provenance.
    pub source: String,
    pub confidence: f64,
    pub timestamp: f64,
    pub usage_count: usize,
    /// Caller-assigned, e.g. how often this identity recurs (Python default
    /// 0.5).
    pub importance: f64,
}

#[derive(Debug, Default, Clone)]
pub struct SemanticMemory {
    /// Python `_store: Dict[statement, fact]` — kept insertion-ordered
    /// (Vec of pairs) so `retrieve`'s stable ranking matches Python's
    /// dict-order + stable `sorted`.
    pub store: Vec<(String, SemanticFact)>,
}

impl SemanticMemory {
    pub fn new() -> Self {
        SemanticMemory { store: Vec::new() }
    }

    /// Only accepts equation-form statements ("lhs = rhs") and only if the
    /// equivalence independently re-verifies. Returns `None` (and stores
    /// nothing) if verification fails — callers must check for `None`
    /// rather than assume storage succeeded (never-silently-trust).
    pub fn store(
        &mut self,
        statement: &str,
        category: &str,
        source: &str,
        importance: f64,
    ) -> Option<SemanticFact> {
        if !statement.contains('=') {
            return None;
        }
        // Python `statement.partition("=")` splits at the first '='.
        let (lhs, rhs) = statement.split_once('=').expect("'=' checked above");
        let check = match verify_algebraic_equivalence(lhs.trim(), rhs.trim()) {
            Ok(c) => c,
            Err(_) => return None, // Python `except Exception: return None`
        };
        if !check.passed {
            return None;
        }

        let fact = SemanticFact {
            statement: statement.to_string(),
            category: category.to_string(),
            source: source.to_string(),
            confidence: check.confidence,
            timestamp: now_secs(),
            usage_count: 0,
            importance,
        };
        // dict assignment: replaces an existing key in place (keeps position)
        match self.store.iter_mut().find(|(s, _)| s == statement) {
            Some(slot) => slot.1 = fact.clone(),
            None => self.store.push((statement.to_string(), fact.clone())),
        }
        Some(fact)
    }

    /// Ranked by (importance, usage_count) descending — facts that matter
    /// more and have proven useful more often surface first.
    pub fn retrieve(&self, category: Option<&str>, min_confidence: f64) -> Vec<SemanticFact> {
        let mut facts: Vec<SemanticFact> = self
            .store
            .iter()
            .map(|(_, f)| f.clone())
            .filter(|f| {
                f.confidence >= min_confidence && (category.is_none() || Some(f.category.as_str()) == category)
            })
            .collect();
        // Python: sorted(key=lambda f: (f.importance, f.usage_count), reverse=True)
        // — stable descending.
        facts.sort_by(|a, b| {
            b.importance
                .partial_cmp(&a.importance)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(b.usage_count.cmp(&a.usage_count))
        });
        facts
    }

    pub fn record_usage(&mut self, statement: &str) {
        if let Some((_, f)) = self.store.iter_mut().find(|(s, _)| s == statement) {
            f.usage_count += 1;
        }
    }

    pub fn forget(&mut self, statement: &str) -> bool {
        let before = self.store.len();
        self.store.retain(|(s, _)| s != statement);
        self.store.len() != before
    }

    /// Forget low-value facts: never used, and old enough that "never used
    /// yet" isn't just "used yet" timing noise. Returns count forgotten.
    pub fn prune_unused(&mut self, min_usage: usize, older_than_seconds: f64) -> usize {
        let now = now_secs();
        let to_forget: Vec<String> = self
            .store
            .iter()
            .filter(|(_, f)| f.usage_count < min_usage && (now - f.timestamp) > older_than_seconds)
            .map(|(s, _)| s.clone())
            .collect();
        let n = to_forget.len();
        self.store.retain(|(s, _)| !to_forget.contains(s));
        n
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Mutable access to the stored facts (the Python tests reach into
    /// `mem._store.values()` to backdate timestamps).
    pub fn facts_mut(&mut self) -> impl Iterator<Item = &mut SemanticFact> {
        self.store.iter_mut().map(|(_, f)| f)
    }
}
