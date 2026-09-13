//! Port of python/tests/test_statistics_domain.py
//! (TestStatisticsGroundTruth, TestExpectedValueDomainSearch).

use reasoning_common::Rat;
use reasoning_search::Domain;
use reasoning_search::{exact_expected_value, exact_variance, ExpectedValueDomain, Mcts};

fn fair_die_outcomes() -> Vec<(f64, Rat)> {
    (1..=6).map(|v| (v as f64, Rat::new(1, 6))).collect()
}

// ---- TestStatisticsGroundTruth ----

#[test]
fn test_statistics_fair_die_expectation_is_exactly_seven_halves() {
    let outcomes = fair_die_outcomes();
    assert_eq!(exact_expected_value(&outcomes), Rat::new(7, 2));
}

#[test]
fn test_statistics_coin_flip_variance() {
    // X=1 w.p. 1/2, X=0 w.p. 1/2 -> Var = 1/4
    let outcomes = vec![(1.0, Rat::new(1, 2)), (0.0, Rat::new(1, 2))];
    assert_eq!(exact_variance(&outcomes), Rat::new(1, 4));
}

// ---- TestExpectedValueDomainSearch ----

#[test]
fn test_statistics_mcts_finds_exact_expectation_of_fair_die() {
    let outcomes = fair_die_outcomes();
    let domain = ExpectedValueDomain::new(outcomes, "expectation", 10);
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 2, 1);
    let result = mcts.search(initial, 2000);
    assert!(result.found_verified_solution);
    let s = result.best_terminal_state.as_ref().unwrap();
    let found = Rat::new(s.guess_numerator.unwrap(), s.guess_denominator.unwrap());
    assert_eq!(found, Rat::new(7, 2));
}

#[test]
fn test_statistics_mcts_finds_exact_variance_of_coin_flip() {
    let outcomes = vec![(1.0, Rat::new(1, 2)), (0.0, Rat::new(1, 2))];
    let domain = ExpectedValueDomain::new(outcomes, "variance", 8);
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 2, 2);
    let result = mcts.search(initial, 1000);
    assert!(result.found_verified_solution);
    let s = result.best_terminal_state.as_ref().unwrap();
    let found = Rat::new(s.guess_numerator.unwrap(), s.guess_denominator.unwrap());
    assert_eq!(found, Rat::new(1, 4));
}

#[test]
#[should_panic(expected = "probabilities must sum to exactly 1")]
fn test_statistics_rejects_bad_probability_distribution() {
    // doesn't sum to 1 (Python: assertRaises(AssertionError))
    let bad_outcomes = vec![(1.0, Rat::new(1, 2)), (2.0, Rat::new(1, 3))];
    let _ = ExpectedValueDomain::with_defaults(bad_outcomes);
}
