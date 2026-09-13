//! Port of python/tests/test_graded_confidence.py
//! (TestGradedConfidence).
//!
//! Python imports `train_a_prm` from tests/test_prm_guided_search.py — that
//! helper lives in the prm crate's test binaries, so an identical copy is
//! inlined here (cross-crate test sharing isn't possible in Rust).

use rand::rngs::StdRng;
use rand::SeedableRng;
use reasoning_prm::{extract_training_examples, LinearPRM};
use reasoning_search::{make_initial_state, Mcts, NumberTargetDomain};
use reasoning_uncertainty::grade_number_target_confidence;

/// Port of `train_a_prm` from python/tests/test_prm_guided_search.py.
fn train_a_prm(seed: u64) -> LinearPRM {
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

#[test]
fn test_verified_case_has_binary_confidence_one() {
    let prm = train_a_prm(0);
    let state = make_initial_state(&[4.0, 7.0, 8.0, 8.0]);
    let result =
        grade_number_target_confidence(&prm, &state, 24.0, 6, true, 50, 300);
    assert!(result.verified);
    assert_eq!(result.binary_confidence, 1.0);
}

#[test]
fn test_unverified_case_has_binary_confidence_zero_regardless_of_prm_opinion() {
    // critical: even if the PRM is (wrongly) confident, an unverified result
    // must still report binary_confidence=0.0 -- the PRM's opinion never
    // overrides the real verifier's word
    let prm = train_a_prm(0);
    let state = make_initial_state(&[1.0, 1.0, 1.0]);
    let result =
        grade_number_target_confidence(&prm, &state, 1e9, 6, false, 300, 300);
    assert!(!result.verified);
    assert_eq!(result.binary_confidence, 0.0);
    // graduated_confidence can be anything -- it's informational, and the
    // test explicitly does NOT assert it matches verified status
}

#[test]
fn test_search_effort_used_is_a_valid_fraction() {
    let prm = train_a_prm(0);
    let state = make_initial_state(&[2.0, 2.0]);
    let result = grade_number_target_confidence(&prm, &state, 4.0, 4, true, 10, 100);
    let effort = result.search_effort_used.unwrap();
    assert!((0.0..=1.0).contains(&effort));
    assert!((effort - 0.1).abs() < 1e-9);
}

#[test]
fn test_effort_is_capped_at_one_even_if_nodes_exceed_budget() {
    let prm = train_a_prm(0);
    let state = make_initial_state(&[2.0, 2.0]);
    let result = grade_number_target_confidence(&prm, &state, 4.0, 4, true, 500, 100);
    assert_eq!(result.search_effort_used, Some(1.0));
}
