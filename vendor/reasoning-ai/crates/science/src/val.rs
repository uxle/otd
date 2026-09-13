//! Shared heterogeneous result-value helpers (Rust-only aid; no direct
//! Python counterpart).
//!
//! Several science-domain Python functions return heterogeneous tuples or
//! dicts, e.g. density's `(rho, floats)` pair or pH neutralization's
//! `{"status": "acid in excess", "excess_equivalents": ...}` dict. `Val`
//! mirrors those payloads; `ValMap` is an insertion-ordered String -> Val
//! map that mirrors a Python dict (re-inserting an existing key replaces
//! the value in place, preserving first-insert position).

/// A single heterogeneous result value (Python float / bool / str).
#[derive(Debug, Clone, PartialEq)]
pub enum Val {
    Num(f64),
    Bool(bool),
    Str(String),
}

impl Val {
    /// Numeric payload; panics on a non-numeric value (the Rust analogue of
    /// the TypeError Python itself would raise on the equivalent misuse).
    pub fn num(&self) -> f64 {
        match self {
            Val::Num(x) => *x,
            _ => panic!("Val::num() called on a non-numeric value"),
        }
    }

    /// Boolean payload; panics on a non-boolean value.
    pub fn boolean(&self) -> bool {
        match self {
            Val::Bool(b) => *b,
            _ => panic!("Val::boolean() called on a non-boolean value"),
        }
    }

    /// String payload; panics on a non-string value.
    pub fn string(&self) -> &str {
        match self {
            Val::Str(s) => s,
            _ => panic!("Val::string() called on a non-string value"),
        }
    }

    /// `None` on a non-numeric value (mirrors a Python try/float() check).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Val::Num(x) => Some(*x),
            _ => None,
        }
    }
}

impl From<f64> for Val {
    fn from(x: f64) -> Val {
        Val::Num(x)
    }
}

impl From<bool> for Val {
    fn from(b: bool) -> Val {
        Val::Bool(b)
    }
}

impl From<&str> for Val {
    fn from(s: &str) -> Val {
        Val::Str(s.to_string())
    }
}

impl From<String> for Val {
    fn from(s: String) -> Val {
        Val::Str(s)
    }
}

/// Insertion-ordered String -> Val map mirroring a Python dict.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ValMap {
    entries: Vec<(String, Val)>,
}

impl ValMap {
    pub fn new() -> ValMap {
        ValMap { entries: Vec::new() }
    }

    /// Insert (or replace in place, keeping the original position) —
    /// Python dict assignment semantics.
    pub fn insert(&mut self, key: impl Into<String>, value: impl Into<Val>) {
        let key = key.into();
        let value = value.into();
        if let Some(slot) = self.entries.iter_mut().find(|(k, _)| *k == key) {
            slot.1 = value;
        } else {
            self.entries.push((key, value));
        }
    }

    pub fn get(&self, key: &str) -> Option<&Val> {
        self.entries.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    /// Numeric lookup; panics on a missing key (Python KeyError) or on a
    /// non-numeric value (Python TypeError).
    pub fn num(&self, key: &str) -> f64 {
        match self.get(key) {
            Some(v) => v.num(),
            None => panic!("ValMap::num(): missing key '{:?}'", key),
        }
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn iter(&self) -> std::slice::Iter<'_, (String, Val)> {
        self.entries.iter()
    }
}

impl<K: Into<String>, V: Into<Val>> FromIterator<(K, V)> for ValMap {
    fn from_iter<I: IntoIterator<Item = (K, V)>>(iter: I) -> ValMap {
        let mut m = ValMap::new();
        for (k, v) in iter {
            m.insert(k, v);
        }
        m
    }
}
