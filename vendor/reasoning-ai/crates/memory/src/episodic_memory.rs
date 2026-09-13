//! Phase 026 — Episodic memory (design doc section 14) (Rust port of
//! `python/memory/episodic_memory.py`).
//!
//! Caches (problem signature -> verified solution) so an identical problem
//! never needs to be searched twice. Every cache hit is re-verified against
//! the *current* problem before being returned (protects against a subtle
//! bug: two different problems accidentally hashing to the same signature
//! would otherwise silently return a wrong cached answer for one of them).

use reasoning_common::py_float_str;
use reasoning_verifier::symbolic_verifier::verify_numeric_equality;
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[derive(Debug, Clone, PartialEq)]
pub struct EpisodicMemoryEntry {
    pub answer_expr: String,
    pub target: f64,
    pub confidence: f64,
    pub timestamp: f64,
    pub usage_count: usize,
}

#[derive(Debug, Default)]
pub struct EpisodicMemory {
    store: HashMap<String, EpisodicMemoryEntry>,
}

impl EpisodicMemory {
    pub fn new() -> Self {
        EpisodicMemory {
            store: HashMap::new(),
        }
    }

    /// Python `f"{sorted(numbers)}|{target}"` — elements rendered with
    /// Python `str(float)` semantics via `py_float_str`.
    fn signature(&self, numbers: &[f64], target: f64) -> String {
        let mut sorted = numbers.to_vec();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let nums_str = sorted
            .iter()
            .map(|v| py_float_str(*v))
            .collect::<Vec<_>>()
            .join(", ");
        format!("[{}]|{}", nums_str, py_float_str(target))
    }

    pub fn lookup(&mut self, numbers: &[f64], target: f64) -> Option<&EpisodicMemoryEntry> {
        let sig = self.signature(numbers, target);
        let passed = match self.store.get(&sig) {
            None => return None,
            Some(entry) => verify_numeric_equality(&entry.answer_expr, target)
                .map(|r| r.passed)
                // Python lets a parse failure propagate; nothing in the
                // reachable flow stores an unparseable expr, so fail closed
                // (treat as failed verification) rather than panic.
                .unwrap_or(false),
        };
        if !passed {
            // corrupt/stale entry, evict it
            self.store.remove(&sig);
            return None;
        }
        let entry = self.store.get_mut(&sig).expect("entry exists (checked)");
        entry.usage_count += 1;
        Some(&*entry)
    }

    /// Python `store(numbers, target, answer_expr, confidence=1.0)`.
    pub fn store(&mut self, numbers: &[f64], target: f64, answer_expr: &str, confidence: f64) {
        let sig = self.signature(numbers, target);
        self.store.insert(
            sig,
            EpisodicMemoryEntry {
                answer_expr: answer_expr.to_string(),
                target,
                confidence,
                timestamp: now_secs(),
                usage_count: 0,
            },
        );
    }

    pub fn len(&self) -> usize {
        self.store.len()
    }

    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }
}
