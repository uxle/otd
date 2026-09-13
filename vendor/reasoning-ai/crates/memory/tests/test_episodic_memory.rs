//! Port of python/tests/test_episodic_memory.py (TestEpisodicMemory).

use reasoning_memory::EpisodicMemory;
use reasoning_search::{make_initial_state, Mcts, NumberTargetDomain};
use reasoning_verifier::symbolic_verifier::verify_numeric_equality;
use std::time::Instant;

fn solve_with_memory(
    memory: &mut EpisodicMemory,
    numbers: &[f64],
    target: f64,
    seed: u64,
) -> (Option<String>, bool) {
    if let Some(cached) = memory.lookup(numbers, target) {
        return (Some(cached.answer_expr.clone()), true); // (answer, was_cache_hit)
    }

    let domain = NumberTargetDomain::new(target);
    let state = make_initial_state(numbers);
    let mut mcts = Mcts::new(domain, 6, seed);
    let result = mcts.search(state, 1500);
    if result.found_verified_solution {
        let expr = result.best_terminal_state.as_ref().unwrap()[0].1.clone();
        memory.store(numbers, target, &expr, 1.0);
        return (Some(expr), false);
    }
    (None, false)
}

#[test]
fn test_second_solve_is_a_cache_hit_and_much_faster() {
    let mut memory = EpisodicMemory::new();
    let (numbers, target) = (vec![4.0, 7.0, 8.0, 8.0], 24.0);

    let t0 = Instant::now();
    let (ans1, hit1) = solve_with_memory(&mut memory, &numbers, target, 1);
    let first_duration = t0.elapsed();

    let t2 = Instant::now();
    let (ans2, hit2) = solve_with_memory(&mut memory, &numbers, target, 1);
    let second_duration = t2.elapsed();

    assert!(!hit1);
    assert!(hit2);
    assert_eq!(ans1, ans2);
    assert!(
        second_duration < first_duration / 5,
        "cache hit should be dramatically faster than a fresh search"
    );
}

#[test]
fn test_cached_answer_is_always_independently_correct() {
    let mut memory = EpisodicMemory::new();
    let (ans, _) = solve_with_memory(&mut memory, &[4.0, 7.0, 8.0, 8.0], 24.0, 2);
    assert!(ans.is_some());
    let cached = memory.lookup(&[4.0, 7.0, 8.0, 8.0], 24.0).unwrap();
    assert!(verify_numeric_equality(&cached.answer_expr, 24.0)
        .map(|r| r.passed)
        .unwrap_or(false));
}

#[test]
fn test_corrupted_entry_is_detected_and_evicted_not_trusted() {
    let mut memory = EpisodicMemory::new();
    // deliberately inject a WRONG cached answer to prove the re-verify
    // safety net actually works, not just trusts whatever is stored
    memory.store(&[4.0, 7.0, 8.0, 8.0], 24.0, "99", 1.0); // wrong on purpose
    assert_eq!(memory.len(), 1);

    let result = memory.lookup(&[4.0, 7.0, 8.0, 8.0], 24.0);
    assert!(
        result.is_none(),
        "a wrong cached answer must be rejected on lookup, not trusted"
    );
    assert_eq!(memory.len(), 0, "the bad entry should have been evicted");
}

#[test]
fn test_different_problems_dont_collide() {
    let mut memory = EpisodicMemory::new();
    memory.store(&[2.0, 2.0], 4.0, "(2+2)", 1.0);
    assert!(memory.lookup(&[3.0, 3.0], 6.0).is_none()); // different problem, no entry yet
}
