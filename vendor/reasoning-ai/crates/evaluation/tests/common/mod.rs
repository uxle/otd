//! Shared test helpers for the evaluation crate's integration tests.
//!
//! Python imported these cross-file: `train_a_prm` from
//! tests/test_prm_guided_search.py and `build_default_prm` from
//! apps/evaluate.py. Rust integration tests are separate binaries, so
//! local copies live here (same situation as the prm/uncertainty crates
//! before this one).

use rand::rngs::StdRng;
use rand::SeedableRng;
use reasoning_prm::{build_calibrated_training_set, extract_training_examples, LinearPRM};
use reasoning_search::{make_initial_state, Mcts, NumberTargetDomain};

/// Port of `train_a_prm` in python/tests/test_prm_guided_search.py (same
/// helper the prm crate's own tests use).
#[allow(dead_code)]
pub fn train_a_prm(seed: u64) -> LinearPRM {
    let problems = vec![
        (NumberTargetDomain::new(4.0), vec![2.0, 2.0], 4usize),
        (NumberTargetDomain::new(24.0), vec![4.0, 7.0, 8.0, 8.0], 6),
        (NumberTargetDomain::new(10.0), vec![2.0, 3.0, 5.0], 4),
        (NumberTargetDomain::new(13.0), vec![1.0, 2.0, 6.0], 4),
        (NumberTargetDomain::new(1.0), vec![3.0, 3.0], 4),
    ];
    let mut examples = Vec::new();
    for (i, (domain, nums, max_depth)) in problems.iter().enumerate() {
        let state = make_initial_state(nums);
        let mut mcts = Mcts::new(domain.clone(), *max_depth, seed + i as u64);
        let result = mcts.search(state, 800);
        examples.extend(extract_training_examples(&result, domain.target, *max_depth, 2));
    }
    let mut prm = LinearPRM::new();
    let mut rng = StdRng::seed_from_u64(seed);
    prm.train(&examples, 60, 0.3, &mut rng);
    prm
}

/// Port of `build_default_prm` in python/apps/evaluate.py (Phase 020 — the
/// calibrated-PRM entry point; the app crate owns the runnable script, the
/// function itself is inlined here until that port lands).
#[allow(dead_code)]
pub fn build_default_prm() -> LinearPRM {
    let train_problems = vec![
        (NumberTargetDomain::new(4.0), vec![2.0, 2.0], 4usize),
        (NumberTargetDomain::new(24.0), vec![4.0, 7.0, 8.0, 8.0], 6),
        (NumberTargetDomain::new(10.0), vec![2.0, 3.0, 5.0], 4),
        (NumberTargetDomain::new(13.0), vec![1.0, 2.0, 6.0], 4),
        (NumberTargetDomain::new(1.0), vec![3.0, 3.0], 4),
        (NumberTargetDomain::new(100.0), vec![1.0, 1.0, 1.0, 1.0], 6),
    ];
    let examples = build_calibrated_training_set(&train_problems, 15, 800, 0);
    let mut prm = LinearPRM::new();
    let mut rng = StdRng::seed_from_u64(0);
    prm.train(&examples, 60, 0.3, &mut rng);
    prm
}
