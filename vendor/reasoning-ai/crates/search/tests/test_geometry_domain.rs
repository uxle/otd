//! Port of python/tests/test_geometry_domain.py
//! (TestGeometryGroundTruth, TestGeometryDomainSearch).

use reasoning_search::{exact_ground_truth, Domain, GeometryAction, GeometryDomain, Mcts};
use std::collections::HashMap;

fn params(pairs: &[(&str, f64)]) -> HashMap<String, f64> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect()
}

// ---- TestGeometryGroundTruth ----

#[test]
fn test_geometry_ground_truth_rectangle_area() {
    let (expr, value) = exact_ground_truth("rectangle_area", &params(&[("w", 4.0), ("h", 5.0)]))
        .expect("valid shape");
    assert!(!expr.is_empty());
    assert!((value - 20.0).abs() < 1e-9);
}

#[test]
fn test_geometry_ground_truth_circle_area_matches_math_pi() {
    let (_expr, value) = exact_ground_truth("circle_area", &params(&[("r", 3.0)]))
        .expect("valid shape");
    assert!((value - std::f64::consts::PI * 9.0).abs() < 1e-9);
}

// ---- TestGeometryDomainSearch ----

#[test]
fn test_geometry_finds_rectangle_area() {
    let domain = GeometryDomain::with_denom_range("rectangle_area", params(&[("w", 4.0), ("h", 5.0)]), 4);
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 2, 1);
    let result = mcts.search(initial, 500);
    assert!(result.found_verified_solution);
    let s = result.best_terminal_state.as_ref().unwrap();
    let guess = s.guess_numerator.unwrap() as f64 / s.guess_denominator.unwrap() as f64;
    assert!((guess - 20.0).abs() < 1e-3); // assertAlmostEqual places=3
}

#[test]
fn test_geometry_finds_triangle_area_fraction() {
    // base=5, height=4 -> area = 10 exactly
    let domain = GeometryDomain::with_denom_range("triangle_area", params(&[("b", 5.0), ("h", 4.0)]), 4);
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 2, 2);
    let result = mcts.search(initial, 500);
    assert!(result.found_verified_solution);
    let s = result.best_terminal_state.as_ref().unwrap();
    let guess = s.guess_numerator.unwrap() as f64 / s.guess_denominator.unwrap() as f64;
    assert!((guess - 10.0).abs() < 1e-3);
}

#[test]
fn test_geometry_finds_circle_area_within_tolerance() {
    let domain = GeometryDomain::try_new(
        "circle_area",
        params(&[("r", 2.0)]),
        6,    // denom_range
        0.05, // tolerance
    )
    .unwrap();
    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 2, 3);
    let result = mcts.search(initial, 1500);
    assert!(result.found_verified_solution);
    let s = result.best_terminal_state.as_ref().unwrap();
    let guess = s.guess_numerator.unwrap() as f64 / s.guess_denominator.unwrap() as f64;
    assert!((guess - std::f64::consts::PI * 4.0).abs() < 0.05); // delta=0.05
}

#[test]
fn test_geometry_action_bound_does_not_scale_with_computed_answer() {
    // regression guard: legal_actions must derive its bound from the INPUT
    // params, not from ground_truth_value -- verify two problems with
    // identical params but where we tamper with a wildly different ground
    // truth still get the SAME action bound
    let mut d1 = GeometryDomain::with_denom_range("rectangle_area", params(&[("w", 3.0), ("h", 3.0)]), 5);
    let s = d1.apply(&d1.initial_state(), &GeometryAction::Denom(1));
    let actions1 = d1.legal_actions(&s);
    d1.ground_truth_value = 999999.0; // tamper with the computed answer only
    let actions2 = d1.legal_actions(&s);
    assert_eq!(actions1.len(), actions2.len());
}
