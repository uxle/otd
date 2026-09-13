//! Phase 044 — Audit logging (design doc section 27: "audit logging",
//! section 25: "reproducibility") (Rust port of
//! `python/uncertainty/audit_log.py`).
//!
//! Wraps solve() (Phase 030) with a persistent, append-only log of every
//! call: what was asked, what was returned, whether it was verified, and
//! enough of a seed/timestamp trail to reproduce the call later. This
//! isn't a debug print statement — it's queryable, and a test proves the
//! record it stores can reconstruct exactly what happened.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Serialize;
use serde_json::{Map, Value};

use reasoning_uncertainty::Answer;

use crate::solve::{solve, Problem};

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

/// Python `@dataclass AuditRecord`.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct AuditRecord {
    pub call_id: i64,
    pub timestamp: f64,
    pub problem_kind: String,
    pub problem_payload: Value,
    pub seed: u64,
    pub budget: Option<usize>,
    pub verified: bool,
    pub final_answer: Option<String>,
    pub explanation: String,
}

/// Python `class AuditLog`.
#[derive(Debug, Default)]
pub struct AuditLog {
    records: Vec<AuditRecord>,
    next_id: i64,
}

impl AuditLog {
    pub fn new() -> AuditLog {
        AuditLog {
            records: Vec::new(),
            next_id: 1,
        }
    }

    pub fn solve_with_audit(
        &mut self,
        problem: &Problem,
        budget: Option<usize>,
        seed: u64,
    ) -> Answer {
        let answer = solve(problem, budget, seed);
        let record = AuditRecord {
            call_id: self.next_id,
            timestamp: now_secs(),
            problem_kind: problem.kind.clone(),
            problem_payload: Value::Object(problem.payload.clone()),
            seed,
            budget,
            verified: answer.verified,
            final_answer: answer.final_answer().map(str::to_string),
            explanation: answer.explanation.clone(),
        };
        self.records.push(record);
        self.next_id += 1;
        answer
    }

    pub fn get_record(&self, call_id: i64) -> Option<&AuditRecord> {
        self.records.iter().find(|r| r.call_id == call_id)
    }

    pub fn query(&self, verified: Option<bool>, kind: Option<&str>) -> Vec<&AuditRecord> {
        let mut results: Vec<&AuditRecord> = self.records.iter().collect();
        if let Some(verified) = verified {
            results.retain(|r| r.verified == verified);
        }
        if let Some(kind) = kind {
            results.retain(|r| r.problem_kind == kind);
        }
        results
    }

    /// Re-run a logged call with the exact same inputs, to independently
    /// confirm the original result — proves the log has enough
    /// information to reproduce, not just describe, what happened.
    /// (Python raised KeyError; here `Err` carries the same message.)
    pub fn replay(&self, call_id: i64) -> Result<Answer, String> {
        let Some(record) = self.get_record(call_id) else {
            return Err(format!("no audit record with call_id={}", call_id));
        };
        let payload = match &record.problem_payload {
            Value::Object(m) => m.clone(),
            _ => Map::new(),
        };
        let problem = Problem::new(record.problem_kind.clone(), payload);
        Ok(solve(&problem, record.budget, record.seed))
    }

    /// Python `json.dumps([asdict(r) for r in records], indent=2)`.
    pub fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.records).unwrap_or_else(|_| "[]".to_string())
    }

    pub fn len(&self) -> usize {
        self.records.len()
    }

    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
}
