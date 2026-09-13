//! Port of python/tests/test_code_sandbox.py.

use reasoning_verifier::code_sandbox::{run_sandboxed, SandboxError, SandboxValue};
use std::time::Instant;

fn violation(err: &SandboxError) -> bool {
    matches!(err, SandboxError::Violation(_))
}

fn timeout(err: &SandboxError) -> bool {
    matches!(err, SandboxError::Timeout(_))
}

// ---- TestSandboxHappyPath ----

#[test]
fn test_sandbox_simple_function_runs_correctly() {
    let code = "def add(a, b):\n    return a + b\n";
    let result = run_sandboxed(code, "add(3, 4)", 100_000).unwrap();
    assert_eq!(result, SandboxValue::Int(7));
}

#[test]
fn test_sandbox_loop_and_conditional() {
    let code = "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n";
    let result = run_sandboxed(code, "fib(10)", 100_000).unwrap();
    assert_eq!(result, SandboxValue::Int(55));
}

#[test]
fn test_sandbox_uses_allowed_builtins() {
    let code = "def total(xs):\n    return sum(xs)\n";
    let result = run_sandboxed(code, "total([1,2,3,4,5])", 100_000).unwrap();
    assert_eq!(result, SandboxValue::Int(15));
}

// ---- TestSandboxBlocksRealAttacks ----

#[test]
fn test_sandbox_blocks_import() {
    let err = run_sandboxed("import os\ndef f():\n    return 1\n", "f()", 100_000).unwrap_err();
    assert!(violation(&err));
}

#[test]
fn test_sandbox_blocks_dunder_globals_escape() {
    let code = "def f():\n    return f.__globals__\n";
    let err = run_sandboxed(code, "f()", 100_000).unwrap_err();
    assert!(violation(&err));
}

#[test]
fn test_sandbox_blocks_eval_call() {
    let code = "def f():\n    return eval('1+1')\n";
    let err = run_sandboxed(code, "f()", 100_000).unwrap_err();
    assert!(violation(&err));
}

#[test]
fn test_sandbox_blocks_open_call() {
    let code = "def f():\n    return open('/etc/passwd').read()\n";
    let err = run_sandboxed(code, "f()", 100_000).unwrap_err();
    assert!(violation(&err));
}

#[test]
fn test_sandbox_blocks_attribute_access_generally() {
    let code = "def f(x):\n    return x.__class__\n";
    let err = run_sandboxed(code, "f(1)", 100_000).unwrap_err();
    assert!(violation(&err));
}

#[test]
fn test_sandbox_blocks_class_definitions() {
    let err = run_sandboxed("class Evil:\n    pass\n", "1", 100_000).unwrap_err();
    assert!(violation(&err));
}

#[test]
fn test_sandbox_infinite_loop_hits_step_limit_not_hang_forever() {
    let code = "def f():\n    x = 0\n    while True:\n        x = x + 1\n    return x\n";
    let start = Instant::now();
    let err = run_sandboxed(code, "f()", 10_000).unwrap_err();
    assert!(timeout(&err), "expected timeout, got {:?}", err);
    assert!(start.elapsed().as_secs() < 5, "infinite loop hung the sandbox");
}

#[test]
fn test_sandbox_syntax_error_reported_not_silently_ignored() {
    let err = run_sandboxed("def f(:\n    pass", "f()", 100_000).unwrap_err();
    // syntax problems surface as violations in this port (same guarantee:
    // never silently ignored)
    assert!(!err.to_string().is_empty());
}

#[test]
fn test_sandbox_blocks_list_comprehension() {
    // regression test: an earlier version of the sandbox defined an
    // allowed-node whitelist but never actually enforced it
    let code = "def f():\n    return [x for x in range(5)]\n";
    let err = run_sandboxed(code, "f()", 100_000).unwrap_err();
    assert!(violation(&err));
}

#[test]
fn test_sandbox_blocks_fstrings() {
    let code = "def f():\n    return f\"{1+1}\"\n";
    let err = run_sandboxed(code, "f()", 100_000).unwrap_err();
    assert!(violation(&err));
}

#[test]
fn test_sandbox_blocks_assert() {
    let code = "def f():\n    assert 1 == 1\n    return 1\n";
    let err = run_sandboxed(code, "f()", 100_000).unwrap_err();
    assert!(violation(&err));
}
