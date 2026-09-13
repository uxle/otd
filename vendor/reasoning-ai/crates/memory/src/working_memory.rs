//! Phase 036 — Working memory (design doc section 14, distinct from episodic
//! memory: this is per-trajectory scratch state, discarded after the
//! problem is solved, not persisted across problems) (Rust port of
//! `python/memory/working_memory.py`).
//!
//! A simple typed scratchpad: push subgoals, record intermediate verified
//! values, retrieve them by name. Used to solve two-step word problems where
//! the second equation depends on the first step's verified result.

use std::collections::HashMap;

/// The Python scratchpad stored `Any` — the Python codebase only ever stores
/// floats and `None` (composite word problems store `None` for unsolved
/// steps), so the Rust surface is those two.
#[derive(Debug, Clone, PartialEq)]
pub enum WmValue {
    Num(f64),
    /// Python `None`.
    PyNone,
}

#[derive(Debug, Clone, PartialEq)]
pub struct WorkingMemoryEntry {
    pub name: String,
    pub value: WmValue,
    pub verified: bool,
    /// What produced this value, for traceability.
    pub source: String,
}

#[derive(Debug, Clone, Default)]
pub struct WorkingMemory {
    entries: HashMap<String, WorkingMemoryEntry>,
    order: Vec<String>,
}

impl WorkingMemory {
    pub fn new() -> Self {
        WorkingMemory {
            entries: HashMap::new(),
            order: Vec::new(),
        }
    }

    /// Python `set(name, value, verified, source)` — updating an existing
    /// name replaces the entry but does not duplicate it in the trace order.
    pub fn set(&mut self, name: &str, value: WmValue, verified: bool, source: &str) {
        if !self.entries.contains_key(name) {
            self.order.push(name.to_string());
        }
        self.entries.insert(
            name.to_string(),
            WorkingMemoryEntry {
                name: name.to_string(),
                value,
                verified,
                source: source.to_string(),
            },
        );
    }

    pub fn get(&self, name: &str) -> Option<&WorkingMemoryEntry> {
        self.entries.get(name)
    }

    /// Errors if the entry doesn't exist (Python `KeyError`) or isn't
    /// verified (Python `ValueError`) — callers that need a trustworthy
    /// intermediate result can't accidentally pull an unverified guess out
    /// of the scratchpad. Same message strings as Python.
    pub fn get_verified_value(&self, name: &str) -> Result<WmValue, String> {
        let entry = self.entries.get(name).ok_or_else(|| {
            // Python f"{name!r}" -> single-quoted repr
            format!("no working-memory entry named '{}'", name)
        })?;
        if !entry.verified {
            return Err(format!(
                "working-memory entry '{}' is not verified, refusing to use it as a fact",
                name
            ));
        }
        Ok(entry.value.clone())
    }

    pub fn trace(&self) -> Vec<&WorkingMemoryEntry> {
        self.order.iter().map(|n| &self.entries[n]).collect()
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
    }
}
