//! Port of python/tests/test_audit_log.py — Phase 044 audit logging: a
//! persistent, append-only, queryable log of every solve() call, with
//! replay proving the record can reproduce (not just describe) a result.

use reasoning_engine::audit_log::AuditLog;
use reasoning_engine::solve::Problem;
use serde_json::{json, Value};

fn p(kind: &str, payload: Value) -> Problem {
    Problem::new(kind, serde_json::from_value(payload).unwrap())
}

#[test]
fn test_records_a_solved_call() {
    let mut log = AuditLog::new();
    let ans = log.solve_with_audit(&p("linear_equation", json!({"equation": "2*x + 4 = 10"})), None, 1);
    assert_eq!(log.len(), 1);
    let record = log.get_record(1).unwrap();
    assert!(record.verified);
    assert_eq!(record.problem_kind, "linear_equation");
    assert_eq!(record.final_answer.as_deref(), ans.final_answer());
}

#[test]
fn test_records_an_abstained_call_too() {
    let mut log = AuditLog::new();
    log.solve_with_audit(
        &p("number_target", json!({"numbers": [1, 1, 1], "target": 1000000000})),
        Some(200),
        2,
    );
    let record = log.get_record(1).unwrap();
    assert!(!record.verified);
    assert!(record.final_answer.is_none());
}

#[test]
fn test_query_by_verified_status() {
    let mut log = AuditLog::new();
    log.solve_with_audit(&p("linear_equation", json!({"equation": "2*x + 4 = 10"})), None, 1);
    log.solve_with_audit(
        &p("number_target", json!({"numbers": [1, 1, 1], "target": 1000000000})),
        Some(200),
        2,
    );
    let verified_records = log.query(Some(true), None);
    let failed_records = log.query(Some(false), None);
    assert_eq!(verified_records.len(), 1);
    assert_eq!(failed_records.len(), 1);
}

#[test]
fn test_replay_reproduces_the_original_result() {
    let mut log = AuditLog::new();
    let original = log.solve_with_audit(&p("linear_equation", json!({"equation": "3*x - 9 = 0"})), None, 5);
    let replayed = log.replay(1).unwrap();
    assert_eq!(original.verified, replayed.verified);
    assert_eq!(original.final_answer(), replayed.final_answer());
}

#[test]
fn test_replay_raises_on_unknown_call_id() {
    // Python: `with self.assertRaises(KeyError)` — here `Err(String)`.
    let log = AuditLog::new();
    assert!(log.replay(999).is_err());
}

#[test]
fn test_to_json_produces_valid_parseable_json() {
    let mut log = AuditLog::new();
    log.solve_with_audit(&p("linear_equation", json!({"equation": "2*x = 8"})), None, 1);
    let parsed: Value = serde_json::from_str(&log.to_json()).unwrap();
    let records = parsed.as_array().unwrap();
    assert_eq!(records.len(), 1);
    assert!(records[0].get("call_id").is_some());
    assert!(records[0].get("timestamp").is_some());
}
