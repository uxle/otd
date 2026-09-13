//! Port of python/tests/test_codegen_domain.py (TestCodeGenDomain).

use reasoning_search::{CodeGenDomain, Domain, Mcts};
use reasoning_verifier::code_sandbox::SandboxValue;

#[test]
fn test_codegen_finds_correct_implementation_among_wrong_ones() {
    // target: f(n) = sum of first n integers = n*(n+1)/2
    let templates = vec![
        "def f(n):\n    return n * (n + 1) // 2\n".to_string(), // correct
        "def f(n):\n    return n * n\n".to_string(),             // wrong
        "def f(n):\n    return n + 1\n".to_string(),             // wrong
        // off by one, wrong
        "def f(n):\n    total = 0\n    for i in range(n):\n        total = total + i\n    return total\n".to_string(),
        // correct, different implementation
        "def f(n):\n    total = 0\n    for i in range(n + 1):\n        total = total + i\n    return total\n".to_string(),
    ];
    let test_cases: Vec<(i64, SandboxValue)> = vec![
        (1, SandboxValue::Int(1)),
        (2, SandboxValue::Int(3)),
        (5, SandboxValue::Int(15)),
        (10, SandboxValue::Int(55)),
    ];
    let domain = CodeGenDomain::new(templates.clone(), test_cases);
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 1, 1);
    let result = mcts.search(initial, 100);

    assert!(result.found_verified_solution);
    let chosen =
        &templates[result.best_terminal_state.as_ref().unwrap().template_choice.unwrap()];
    assert!(chosen == &templates[0] || chosen == &templates[4]); // either correct implementation
}

#[test]
fn test_codegen_unsafe_candidate_is_rejected_not_crashes_search() {
    let templates = vec![
        "import os\ndef f(n):\n    return n\n".to_string(), // unsafe, should be caught and scored 0
        "def f(n):\n    return n * 2\n".to_string(),            // correct for this test
    ];
    let test_cases: Vec<(i64, SandboxValue)> = vec![
        (1, SandboxValue::Int(2)),
        (2, SandboxValue::Int(4)),
        (3, SandboxValue::Int(6)),
    ];
    let domain = CodeGenDomain::new(templates, test_cases);
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 1, 2);
    let result = mcts.search(initial, 50);
    assert!(result.found_verified_solution);
    assert_eq!(
        result.best_terminal_state.as_ref().unwrap().template_choice,
        Some(1)
    );
}

#[test]
fn test_codegen_honestly_fails_when_no_candidate_passes() {
    let templates = vec![
        "def f(n):\n    return n + 999\n".to_string(),
        "def f(n):\n    return -n\n".to_string(),
    ];
    let test_cases: Vec<(i64, SandboxValue)> = vec![
        (1, SandboxValue::Int(1)),
        (2, SandboxValue::Int(2)),
    ];
    let domain = CodeGenDomain::new(templates, test_cases);
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 1, 3);
    let result = mcts.search(initial, 50);
    assert!(!result.found_verified_solution);
}
