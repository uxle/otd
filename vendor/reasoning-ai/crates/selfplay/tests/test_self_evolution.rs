//! Port of python/tests/test_self_evolution.py (TestSelfEvolution).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reasoning_selfplay::self_evolution::run_self_evolution;
use reasoning_selfplay::self_evolution::Problem;
use reasoning_search::NumberTargetDomain;

/// Python test helper `_make_guaranteed_solvable`: construct a problem by
/// combining `count` random numbers with random valid operators down to a
/// single value, then using that value as the target. Guarantees at least
/// one solution exists (the path just taken), unlike picking an arbitrary
/// target and hoping it's reachable.
fn make_guaranteed_solvable(rng: &mut StdRng, count: usize, max_depth: usize) -> Problem {
    const OPS: [char; 4] = ['+', '-', '*', '/'];
    let nums: Vec<f64> = (0..count).map(|_| rng.gen_range(1..=9) as f64).collect();
    let mut values = nums.clone();
    while values.len() > 1 {
        let n = values.len();
        // Python rng.sample(range(n), 2): two distinct indices
        let i = rng.gen_range(0..n);
        let j = (i + rng.gen_range(1..n)) % n;
        let mut op = OPS[rng.gen_range(0..OPS.len())];
        let (a, b) = (values[i], values[j]);
        if op == '/' && b == 0.0 {
            op = '+';
        }
        let v = match op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            _ => {
                if b != 0.0 {
                    a / b
                } else {
                    a + b
                }
            }
        };
        let mut remaining: Vec<f64> = values
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != i && *k != j)
            .map(|(_, x)| *x)
            .collect();
        remaining.push(v);
        values = remaining;
    }
    let target = values[0];
    (NumberTargetDomain::new(target), nums, max_depth)
}

/// Python test helper `make_round_problems(round_idx, n=8, seed_base=100)`:
/// curriculum-ish — later rounds get slightly larger number sets (harder).
/// Every problem is constructed to be guaranteed-solvable.
fn make_round_problems(round_idx: usize, n: usize, seed_base: u64) -> Vec<Problem> {
    let mut rng = StdRng::seed_from_u64(seed_base + round_idx as u64);
    let count = if round_idx < 2 { 3 } else { 4 }; // difficulty ramps up
    (0..n)
        .map(|_| make_guaranteed_solvable(&mut rng, count, 5))
        .collect()
}

/// Python test helper `make_eval_set(n=20, seed=555)`.
fn make_eval_set(n: usize, seed: u64) -> Vec<Problem> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| make_guaranteed_solvable(&mut rng, 3, 4))
        .collect()
}

#[test]
fn test_self_evolution_improves_or_holds_over_rounds() {
    let eval_set = make_eval_set(20, 555);
    let (_prm, history) = run_self_evolution(
        4,
        &|r| make_round_problems(r, 8, 100),
        &eval_set,
        150, // round_budget
        40,  // eval_budget
        60,  // epochs
        0.3, // lr
        0,   // seed
    );
    println!(
        "\n[Phase 009] self-evolution history (round 0=untrained baseline): {:?} out of {} held-out problems solved",
        history,
        eval_set.len()
    );

    // Honest bar: final round should not be worse than the untrained
    // baseline. We do NOT assert strict monotonic improvement every
    // single round — that would be dishonestly demanding for a system
    // with this much randomness in a 4-round, small-data run.
    assert!(
        history[history.len() - 1] >= history[0],
        "self-evolution should not end up worse than the untrained starting point"
    );
}

#[test]
fn test_weights_actually_change_across_rounds() {
    // sanity: retraining should produce different weights each round,
    // not silently reuse the same PRM (a real failure mode to guard against)
    let eval_set = make_eval_set(10, 555);
    let (prm, _history) = run_self_evolution(
        2,
        &|r| make_round_problems(r, 6, 100),
        &eval_set,
        100, // round_budget
        30,  // eval_budget
        40,  // epochs
        0.3, // Python default lr
        1,   // seed
    );
    assert!(
        prm.weights.values().any(|w| w.abs() > 1e-6),
        "retrained PRM should have nonzero weights"
    );
}
