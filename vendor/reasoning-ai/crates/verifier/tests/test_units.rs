//! Port of python/tests/test_units.py.

use reasoning_verifier::units::{add, check_expression_units, divide, multiply, quantity};

fn q(v: f64, u: &str) -> reasoning_verifier::units::UnitQuantity {
    quantity(v, u).unwrap()
}

#[test]
fn test_units_adding_same_units_works() {
    let result = add(q(5.0, "m"), q(3.0, "m")).unwrap();
    assert_eq!(result.value, 8.0);
    assert_eq!(result.unit_str, "m");
}

#[test]
fn test_units_adding_mismatched_units_is_caught() {
    // "5 + 3 = 8" is numerically fine but dimensionally nonsense, and a
    // pure numeric verifier would never catch it
    let err = add(q(5.0, "m"), q(3.0, "s")).unwrap_err();
    assert!(err.contains("dimension mismatch"));
}

#[test]
fn test_units_speed_is_meters_per_second() {
    let distance = q(100.0, "m");
    let time = q(20.0, "s");
    let speed = divide(distance, time).unwrap();
    assert_eq!(speed.value, 5.0);
    let expected: Vec<(String, i32)> = vec![
        ("m".to_string(), 1),
        ("s".to_string(), -1),
    ];
    assert_eq!(speed.dimension, expected);
}

#[test]
fn test_units_area_is_meters_squared() {
    let w = q(4.0, "m");
    let h = q(5.0, "m");
    let area = multiply(w, h).unwrap();
    assert_eq!(area.value, 20.0);
    let expected: Vec<(String, i32)> = vec![("m".to_string(), 2)];
    assert_eq!(area.dimension, expected);
}

#[test]
fn test_units_check_expression_units_flags_mismatch_without_raising() {
    assert!(!check_expression_units("add", &q(5.0, "m"), &q(3.0, "s")));
    assert!(check_expression_units("add", &q(5.0, "m"), &q(3.0, "m")));
    assert!(check_expression_units("multiply", &q(5.0, "m"), &q(3.0, "s")));
}

#[test]
fn test_units_unknown_unit_rejected() {
    assert!(quantity(5.0, "smoots").is_err());
}
