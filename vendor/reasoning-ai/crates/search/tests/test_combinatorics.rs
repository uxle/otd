//! Port of python/tests/test_combinatorics.py
//! (TestCombinatoricsDomain).

use reasoning_search::{brute_force_count, CombinatoricsDomain, Domain, Mcts};

/// math.comb parity for the test's cross-check.
fn math_comb(n: i64, r: i64) -> i64 {
    if r < 0 || r > n {
        return 0;
    }
    let mut v: i64 = 1;
    for i in 0..r {
        v = v * (n - i) / (i + 1);
    }
    v
}

/// math.perm parity for the test's cross-check.
fn math_perm(n: i64, r: i64) -> i64 {
    if r < 0 || r > n {
        return 0;
    }
    let mut v: i64 = 1;
    for i in 0..r {
        v *= n - i;
    }
    v
}

#[test]
fn test_combinatorics_brute_force_matches_math_comb() {
    assert_eq!(brute_force_count("combinations", 5, 2).unwrap(), math_comb(5, 2));
    assert_eq!(brute_force_count("permutations", 5, 2).unwrap(), math_perm(5, 2));
}

#[test]
fn test_combinatorics_search_space_does_not_leak_the_answer() {
    let domain = CombinatoricsDomain::try_new("combinations", 5, 2, 300).unwrap();
    let actions = domain.legal_actions(&domain.initial_state());
    // candidates must span from 0, not be centered on the answer
    let min = *actions.iter().min().unwrap();
    let max = *actions.iter().max().unwrap();
    assert_eq!(min, 0);
    // not suspiciously centered
    assert!(domain.ground_truth != min && domain.ground_truth != max);
}

#[test]
fn test_combinatorics_mcts_finds_verified_combination_count() {
    // C(5,2) = 10
    let domain = CombinatoricsDomain::try_new("combinations", 5, 2, 30).unwrap();
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 1, 1);
    let result = mcts.search(initial, 200);
    assert!(result.found_verified_solution);
    assert_eq!(result.best_terminal_state.as_ref().unwrap().guess, Some(10));
}

#[test]
fn test_combinatorics_mcts_finds_verified_permutation_count() {
    // P(5,2) = 20
    let domain = CombinatoricsDomain::try_new("permutations", 5, 2, 30).unwrap();
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 1, 2);
    let result = mcts.search(initial, 200);
    assert!(result.found_verified_solution);
    assert_eq!(result.best_terminal_state.as_ref().unwrap().guess, Some(20));
}
