//! Port of python/tests/test_working_memory.py (TestWorkingMemory).

use reasoning_memory::{WorkingMemory, WmValue};

#[test]
fn test_set_and_get() {
    let mut wm = WorkingMemory::new();
    wm.set(
        "step1",
        WmValue::Num(5.0),
        true,
        "solved 2*x+4=10 -> wait actually x=3",
    );
    let entry = wm.get("step1").unwrap();
    assert_eq!(entry.value, WmValue::Num(5.0));
    assert!(entry.verified);
}

#[test]
fn test_get_verified_value_returns_value_when_verified() {
    let mut wm = WorkingMemory::new();
    wm.set("x", WmValue::Num(3.0), true, "algebra solve");
    assert_eq!(wm.get_verified_value("x").unwrap(), WmValue::Num(3.0));
}

#[test]
fn test_get_verified_value_refuses_unverified_entries() {
    let mut wm = WorkingMemory::new();
    wm.set("guess", WmValue::Num(42.0), false, "unverified guess");
    // Python raises ValueError
    let err = wm.get_verified_value("guess").unwrap_err();
    assert!(err.contains("is not verified"));
}

#[test]
fn test_get_verified_value_raises_on_missing_key() {
    let wm = WorkingMemory::new();
    // Python raises KeyError
    let err = wm.get_verified_value("does_not_exist").unwrap_err();
    assert!(err.contains("no working-memory entry named"));
}

#[test]
fn test_trace_preserves_insertion_order() {
    let mut wm = WorkingMemory::new();
    wm.set("a", WmValue::Num(1.0), true, "s1");
    wm.set("b", WmValue::Num(2.0), true, "s2");
    wm.set("a", WmValue::Num(10.0), true, "s3"); // update, shouldn't duplicate order
    let trace = wm.trace();
    let names: Vec<&str> = trace.iter().map(|e| e.name.as_str()).collect();
    assert_eq!(names, vec!["a", "b"]);
    assert_eq!(trace[0].value, WmValue::Num(10.0)); // latest value for "a"
}

#[test]
fn test_clear() {
    let mut wm = WorkingMemory::new();
    wm.set("a", WmValue::Num(1.0), true, "s");
    wm.clear();
    assert_eq!(wm.trace().len(), 0);
    assert!(wm.get("a").is_none());
}
