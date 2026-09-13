//! Port of python/tests/test_failure_memory.py (TestFailureMemory).

use reasoning_memory::{FailureMemory, FailureMemoryDomain};
use reasoning_search::{make_initial_state, Mcts, NtState, NumberTargetDomain};
use std::cell::RefCell;
use std::rc::Rc;

fn state_repr(state: &NtState) -> String {
    state[0].1.clone() // the expression string of the (only) remaining number
}

#[test]
fn test_reduces_redundant_verifier_calls() {
    let base_domain = NumberTargetDomain::new(24.0);
    let memory = Rc::new(RefCell::new(FailureMemory::new()));
    let wrapped = FailureMemoryDomain::new(
        base_domain,
        Rc::clone(&memory),
        "problem_24_4788",
        Box::new(state_repr),
    );

    let state = make_initial_state(&[4.0, 7.0, 8.0, 8.0]);
    let mut mcts = Mcts::new(wrapped, 6, 1);
    let result = mcts.search(state, 1500);

    println!(
        "\n[Phase 025] verifier_calls={}, memory_hits={}",
        memory.borrow().verifier_calls,
        memory.borrow().memory_hits
    );

    assert!(result.found_verified_solution);
    assert!(
        memory.borrow().memory_hits > 0,
        "expected MCTS to revisit at least one repeated terminal state"
    );
}

#[test]
fn test_correctness_unaffected_by_memory() {
    // same search, same seed, with vs without failure memory -- must both
    // find a verified solution (memory changes efficiency, never correctness)
    let state = make_initial_state(&[4.0, 7.0, 8.0, 8.0]);

    let mut plain = Mcts::new(NumberTargetDomain::new(24.0), 6, 5);
    let r_plain = plain.search(state.clone(), 1500);

    let memory = Rc::new(RefCell::new(FailureMemory::new()));
    let wrapped = FailureMemoryDomain::new(
        NumberTargetDomain::new(24.0),
        memory,
        "p",
        Box::new(state_repr),
    );
    let mut with_mem = Mcts::new(wrapped, 6, 5);
    let r_mem = with_mem.search(state, 1500);

    assert!(r_plain.found_verified_solution);
    assert!(r_mem.found_verified_solution);
}

#[test]
fn test_never_caches_a_success_as_a_failure() {
    // sanity: only failures get recorded, never the winning state
    let base_domain = NumberTargetDomain::new(24.0);
    let memory = Rc::new(RefCell::new(FailureMemory::new()));
    let wrapped = FailureMemoryDomain::new(
        base_domain,
        Rc::clone(&memory),
        "p2",
        Box::new(state_repr),
    );
    let state = make_initial_state(&[4.0, 7.0, 8.0, 8.0]);
    let mut mcts = Mcts::new(wrapped, 6, 7);
    let result = mcts.search(state, 1500);
    let winning_repr = state_repr(result.best_terminal_state.as_ref().unwrap());
    assert!(!memory.borrow().is_known_failure("p2", &winning_repr));
}
