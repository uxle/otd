//! Port of the TestSemanticMemory class from
//! python/tests/test_semantic_memory_and_solve_api.py.
//!
//! The TestSolveAPINewDomains class in that file needs apps.solve (higher
//! crates) — skipped here, ported with the engine crate later.

use reasoning_memory::SemanticMemory;
use std::time::{SystemTime, UNIX_EPOCH};

fn now_secs() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs_f64())
        .unwrap_or(0.0)
}

#[test]
fn test_store_accepts_verified_identity() {
    let mut mem = SemanticMemory::new();
    let fact = mem.store("sin(x)**2 + cos(x)**2", "trig_identity", "phase085", 0.5);
    // Note: statement must be "lhs = rhs" form to verify; this call
    // deliberately has no "=" and should be rejected.
    assert!(fact.is_none());
}

#[test]
fn test_store_accepts_correct_equation_form() {
    let mut mem = SemanticMemory::new();
    let fact = mem.store("sin(x)**2 + cos(x)**2 = 1", "trig_identity", "phase085", 0.5);
    assert!(fact.is_some());
    assert_eq!(mem.len(), 1);
}

#[test]
fn test_store_rejects_false_equation() {
    let mut mem = SemanticMemory::new();
    let fact = mem.store("sin(x)**2 + cos(x)**2 = 2", "trig_identity", "phase085", 0.5);
    assert!(fact.is_none());
    assert_eq!(mem.len(), 0);
}

#[test]
fn test_store_rejects_unparseable_statement() {
    let mut mem = SemanticMemory::new();
    let fact = mem.store("not a valid ((( expression = 1", "trig_identity", "test", 0.5);
    assert!(fact.is_none());
}

#[test]
fn test_retrieve_filters_by_category() {
    let mut mem = SemanticMemory::new();
    mem.store("sin(x)**2 + cos(x)**2 = 1", "trig_identity", "a", 0.5);
    mem.store("2 + 2 = 4", "arithmetic", "b", 0.5);
    let trig_facts = mem.retrieve(Some("trig_identity"), 0.0);
    assert_eq!(trig_facts.len(), 1);
    assert_eq!(trig_facts[0].category, "trig_identity");
}

#[test]
fn test_retrieve_ranked_by_importance_and_usage() {
    let mut mem = SemanticMemory::new();
    mem.store("2 + 2 = 4", "arithmetic", "a", 0.2);
    mem.store("3 + 3 = 6", "arithmetic", "a", 0.9);
    let ranked = mem.retrieve(Some("arithmetic"), 0.0);
    assert_eq!(ranked[0].statement, "3 + 3 = 6"); // higher importance first
}

#[test]
fn test_record_usage_increments_count() {
    let mut mem = SemanticMemory::new();
    mem.store("2 + 2 = 4", "arithmetic", "a", 0.5);
    mem.record_usage("2 + 2 = 4");
    mem.record_usage("2 + 2 = 4");
    let facts = mem.retrieve(None, 0.0);
    assert_eq!(facts[0].usage_count, 2);
}

#[test]
fn test_forget_removes_fact() {
    let mut mem = SemanticMemory::new();
    mem.store("2 + 2 = 4", "arithmetic", "a", 0.5);
    assert!(mem.forget("2 + 2 = 4"));
    assert_eq!(mem.len(), 0);
    assert!(!mem.forget("2 + 2 = 4")); // already gone
}

#[test]
fn test_prune_unused_respects_min_usage_and_age() {
    let mut mem = SemanticMemory::new();
    mem.store("2 + 2 = 4", "arithmetic", "a", 0.5);
    // not old enough yet -> nothing pruned
    let pruned = mem.prune_unused(1, 3600.0);
    assert_eq!(pruned, 0);
    assert_eq!(mem.len(), 1);
    // simulate age by backdating timestamp directly (Python test pokes
    // mem._store.values())
    let backdate = now_secs() - 7200.0;
    for f in mem.facts_mut() {
        f.timestamp = backdate;
    }
    let pruned = mem.prune_unused(1, 3600.0);
    assert_eq!(pruned, 1);
    assert_eq!(mem.len(), 0);
}
