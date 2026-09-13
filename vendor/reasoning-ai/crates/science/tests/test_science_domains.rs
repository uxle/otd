//! Port of python/tests/test_science_domains.py (the physics + chemistry
//! classes; TestGenetics belongs to the biology crate/port).
//!
//! Every expected value is hand-computed; error paths are tested because
//! "refuses rather than guesses" is part of the contract.

use reasoning_science::chemistry_domain::{
    molar_mass, parse_formula, stoichiometry_moles,
};
use reasoning_science::physics_domain::{
    relative_speed, solve_kinematics, solve_speed_distance_time, KinematicsInputs,
};

// unittest assertAlmostEqual (default places=7)
fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-7
}

// ---- class TestKinematics ----

#[test]
fn test_kinematics_solve_v_from_u_a_t() {
    // Hand-known: u=0, a=10, t=5 -> v=0+10*5=50
    let r = solve_kinematics(
        "v",
        &KinematicsInputs {
            u: Some(0.0),
            a: Some(10.0),
            t: Some(5.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(approx(r.value, 50.0));
}

#[test]
fn test_kinematics_solve_s_from_u_a_t() {
    // Hand-known: s = 0*5 + 0.5*10*25 = 125
    let r = solve_kinematics(
        "s",
        &KinematicsInputs {
            u: Some(0.0),
            a: Some(10.0),
            t: Some(5.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(approx(r.value, 125.0));
}

#[test]
fn test_kinematics_solve_v_from_u_a_s_matches_time_based() {
    // Cross-check: v computed via v^2=u^2+2as should match v=u+at
    // for the SAME physical scenario (u=0,a=10,t=5,s=125)
    let r1 = solve_kinematics(
        "v",
        &KinematicsInputs {
            u: Some(0.0),
            a: Some(10.0),
            t: Some(5.0),
            ..Default::default()
        },
    )
    .unwrap();
    let r2 = solve_kinematics(
        "v",
        &KinematicsInputs {
            u: Some(0.0),
            a: Some(10.0),
            s: Some(125.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert!((r1.value - r2.value).abs() < 1e-6); // places=6
}

#[test]
fn test_kinematics_solve_a_from_v_u_t() {
    // Hand-known: a=(20-0)/4=5
    let r = solve_kinematics(
        "a",
        &KinematicsInputs {
            v: Some(20.0),
            u: Some(0.0),
            t: Some(4.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert!(approx(r.value, 5.0));
}

#[test]
fn test_kinematics_insufficient_inputs_raises() {
    let r = solve_kinematics(
        "v",
        &KinematicsInputs {
            u: Some(0.0),
            ..Default::default()
        },
    );
    assert!(r.is_err());
}

#[test]
fn test_kinematics_unknown_find_raises() {
    let r = solve_kinematics(
        "q",
        &KinematicsInputs {
            u: Some(0.0),
            a: Some(1.0),
            t: Some(1.0),
            ..Default::default()
        },
    );
    assert!(r.is_err());
}

// ---- class TestSpeedDistanceTime ----

#[test]
fn test_speed_distance_time_speed_hand_computed() {
    // 120 km in 2 hours = 60 km/h
    assert!(approx(
        solve_speed_distance_time("speed", None, Some(120.0), Some(2.0)).unwrap(),
        60.0
    ));
}

#[test]
fn test_speed_distance_time_distance_hand_computed() {
    assert!(approx(
        solve_speed_distance_time("distance", Some(60.0), None, Some(2.0)).unwrap(),
        120.0
    ));
}

#[test]
fn test_speed_distance_time_time_hand_computed() {
    assert!(approx(
        solve_speed_distance_time("time", Some(60.0), Some(120.0), None).unwrap(),
        2.0
    ));
}

#[test]
fn test_speed_distance_time_zero_time_raises() {
    assert!(solve_speed_distance_time("speed", None, Some(10.0), Some(0.0)).is_err());
}

#[test]
fn test_speed_distance_time_relative_speed_same_direction() {
    // car A 60km/h, car B 40km/h same direction -> closes at 20km/h
    assert!(approx(relative_speed(60.0, 40.0, true), 20.0));
}

#[test]
fn test_speed_distance_time_relative_speed_opposite_direction() {
    // trains approaching each other at 60 and 40 -> combined 100km/h
    assert!(approx(relative_speed(60.0, 40.0, false), 100.0));
}

// ---- class TestChemistry ----

#[test]
fn test_chemistry_water_formula_and_mass() {
    let counts = parse_formula("H2O").unwrap();
    assert_eq!(
        counts,
        vec![("H".to_string(), 2), ("O".to_string(), 1)]
    );
    // Hand-known: 2*1.01 + 16.00 = 18.02
    assert!((molar_mass("H2O").unwrap() - 18.02).abs() < 1e-2); // places=2
}

#[test]
fn test_chemistry_calcium_hydroxide_nested_parens() {
    let counts = parse_formula("Ca(OH)2").unwrap();
    assert_eq!(
        counts,
        vec![
            ("Ca".to_string(), 1),
            ("O".to_string(), 2),
            ("H".to_string(), 2)
        ]
    );
    // Hand-known: 40.08 + 2*(16.00+1.01) = 40.08 + 34.02 = 74.10
    assert!((molar_mass("Ca(OH)2").unwrap() - 74.10).abs() < 1e-2); // places=2
}

#[test]
fn test_chemistry_glucose_formula() {
    // C6H12O6: 6*12.01 + 12*1.01 + 6*16.00 = 72.06+12.12+96.00 = 180.18
    assert!((molar_mass("C6H12O6").unwrap() - 180.18).abs() < 1e-2); // places=2
}

#[test]
fn test_chemistry_unrecognized_element_raises() {
    assert!(parse_formula("Xx2").is_err());
}

#[test]
fn test_chemistry_unbalanced_parens_raises() {
    assert!(parse_formula("Ca(OH2").is_err());
}

#[test]
fn test_chemistry_stoichiometry_hand_computed() {
    // 2 H2 + O2 -> 2 H2O. Given 4 mol H2, moles H2O = 4 * (2/2) = 4 mol
    // mass = 4 * 18.02 = 72.08 g
    let result = stoichiometry_moles(4.0, 2, 2, "H2O").unwrap();
    assert!(approx(result.moles_product, 4.0));
    assert!((result.mass_product - 72.08).abs() < 1e-2); // places=2
}

#[test]
fn test_chemistry_stoichiometry_uneven_ratio() {
    // N2 + 3 H2 -> 2 NH3. Given 6 mol H2, moles NH3 = 6 * (2/3) = 4 mol
    let result = stoichiometry_moles(6.0, 3, 2, "NH3").unwrap();
    assert!(approx(result.moles_product, 4.0));
}
