//! Port of python/tests/test_science_expansion.py (the physics + chemistry
//! classes; the biology classes belong to another port).
//!
//! Every expected value is hand-computed (textbook constants: g = 9.8,
//! R = 0.0821 L*atm/mol/K, Kw = 1e-14), cross-checked against INDEPENDENT
//! ground truth, not just the module's own consistency. Error paths are
//! tested because "refuses rather than guesses" is part of the contract.

use reasoning_science::chem_balance_domain::{balance_equation, limiting_reagent};
use reasoning_science::chem_gas_domain::{boyle_law, charles_law, combined_gas_law, ideal_gas};
use reasoning_science::chem_ph_domain::{
    concentration_from_ph, neutralization, ph_from_concentration, ph_poh_pair,
};
use reasoning_science::chem_solutions_domain::{
    dilution, molarity, percent_composition, percent_yield,
};
use reasoning_science::density_domain::{
    density, float_test, hydrostatic_pressure, pressure_from_force,
};
use reasoning_science::electricity_domain::{
    electrical_power, ohms_law, parallel_resistance, series_parallel_current, series_resistance,
};
use reasoning_science::energy_domain::{
    height_for_speed, impact_speed_from_height, kinetic_energy, potential_energy,
    power_from_force_velocity, power_from_work, work_done,
};
use reasoning_science::forces_domain::{
    solve_force, solve_friction, solve_net_acceleration, solve_weight,
};
use reasoning_science::momentum_domain::{
    elastic_collision, impulse, inelastic_collision, momentum,
};
use reasoning_science::Val;

// unittest assertAlmostEqual (default places=7)
fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-7
}

// ---- class TestForces ----

#[test]
fn test_forces_newtons_second_law_hand_computed() {
    // F = 2 kg * 10 m/s^2 = 20 N
    assert!(approx(
        solve_force("force", Some(2.0), Some(10.0), None, None)
            .unwrap()
            .value,
        20.0
    ));
    // m = 49 N / 9.8 = 5 kg
    assert!(approx(
        solve_force("mass", None, Some(9.8), Some(49.0), None)
            .unwrap()
            .value,
        5.0
    ));
    // a = 10 N / 2 kg = 5
    assert!(approx(
        solve_force("acceleration", Some(2.0), None, Some(10.0), None)
            .unwrap()
            .value,
        5.0
    ));
}

#[test]
fn test_forces_weight_hand_computed() {
    assert!(approx(solve_weight(5.0, None).unwrap().value, 49.0)); // 5 * 9.8
    assert!(approx(solve_weight(10.0, Some(10.0)).unwrap().value, 100.0)); // custom g
}

#[test]
fn test_forces_friction_hand_computed() {
    // f = mu * m * g = 0.3 * 4 * 9.8 = 11.76
    assert!(approx(
        solve_friction("friction", Some(4.0), Some(0.3), None, None, None)
            .unwrap()
            .value,
        11.76
    ));
    // mu from friction and normal: 8 N over 40 N -> 0.2
    assert!(approx(
        solve_friction("mu", None, None, Some(40.0), Some(8.0), None)
            .unwrap()
            .value,
        0.2
    ));
}

#[test]
fn test_forces_net_acceleration_with_friction() {
    // (10 - 0.2*2*9.8)/2 = (10 - 3.92)/2 = 3.04
    assert!(approx(
        solve_net_acceleration(2.0, 10.0, Some(0.2), None)
            .unwrap()
            .value,
        3.04
    ));
}

#[test]
fn test_forces_error_paths() {
    assert!(solve_force("force", Some(2.0), None, None, None).is_err()); // missing acceleration
    assert!(solve_friction("friction", Some(1.0), Some(-1.0), None, None, None).is_err()); // negative mu
    assert!(solve_net_acceleration(0.0, 10.0, None, None).is_err()); // zero mass
}

// ---- class TestEnergy ----

#[test]
fn test_energy_kinetic_energy_hand_computed() {
    // 0.5 * 2 * 9 = 9 J
    assert!(approx(kinetic_energy(2.0, 3.0).unwrap().value, 9.0));
}

#[test]
fn test_energy_potential_energy_hand_computed() {
    assert!(approx(potential_energy(3.0, 5.0, None).unwrap().value, 147.0)); // 3*9.8*5
}

#[test]
fn test_energy_work_and_power_hand_computed() {
    assert!(approx(work_done(10.0, 4.0).unwrap().value, 40.0));
    assert!(approx(power_from_work(100.0, 20.0).unwrap().value, 5.0));
    assert!(approx(power_from_force_velocity(10.0, 3.0).unwrap().value, 30.0));
}

#[test]
fn test_energy_impact_speed_cross_checked_two_ways() {
    // sqrt(2*9.8*20) = sqrt(392) ~ 19.799; the function itself re-checks
    // via the kinematics equation from physics_domain before returning
    let r = impact_speed_from_height(20.0, None).unwrap();
    assert!((r.value - (2.0f64 * 9.8 * 20.0).powf(0.5)).abs() < 1e-6); // places=6
}

#[test]
fn test_energy_height_for_speed_inverse() {
    // h for 14 m/s: 196/19.6 = 10 m (also cross-checked internally)
    assert!((height_for_speed(14.0, None).unwrap().value - 10.0).abs() < 1e-6); // places=6
}

// ---- class TestMomentum ----

#[test]
fn test_momentum_hand_computed() {
    assert!(approx(momentum(1000.0, 20.0).unwrap().values[0], 20000.0));
}

#[test]
fn test_momentum_impulse_hand_computed() {
    // F=10 N for 4 s on 2 kg from rest -> v = 10*4/2 = 20 m/s
    let r = impulse(10.0, 4.0, 2.0, Some(0.0)).unwrap();
    assert!(approx(r.values[0], 20.0));
}

#[test]
fn test_momentum_inelastic_conserves_momentum_loses_energy() {
    // (2*3 + 3*(-1))/5 = 0.6; KE before 9+1.5=10.5, after 0.5*5*0.36=0.9 -> lost 9.6
    let r = inelastic_collision(2.0, 3.0, 3.0, -1.0).unwrap();
    assert!(approx(r.values[0], 0.6));
    assert!(approx(r.values[1], 9.6));
}

#[test]
fn test_momentum_elastic_equal_masses_swap() {
    let r = elastic_collision(2.0, 3.0, 2.0, -1.0).unwrap();
    assert!(approx(r.values[0], -1.0));
    assert!(approx(r.values[1], 3.0));
}

#[test]
fn test_momentum_elastic_hard_case_conserves_both() {
    // m1=1 v1=5, m2=2 v2=-1 (hand-derived expectations)
    let r = elastic_collision(1.0, 5.0, 2.0, -1.0).unwrap();
    let v1p = ((1.0 - 2.0) * 5.0 + 2.0 * 2.0 * (-1.0)) / 3.0;
    let v2p = ((2.0 - 1.0) * (-1.0) + 2.0 * 1.0 * 5.0) / 3.0;
    assert!(approx(r.values[0], v1p));
    assert!(approx(r.values[1], v2p));
    assert!(approx(1.0 * 5.0 + 2.0 * (-1.0), 1.0 * v1p + 2.0 * v2p)); // p conserved
    // KE conserved
    assert!(approx(
        0.5 * 25.0 + 0.5 * 2.0 * 1.0,
        0.5 * v1p.powi(2) + 0.5 * 2.0 * v2p.powi(2)
    ));
}

// ---- class TestElectricity ----

#[test]
fn test_electricity_ohms_law_hand_computed() {
    assert!(approx(
        ohms_law("voltage", None, Some(2.0), Some(5.0)).unwrap().values[0],
        10.0
    ));
    assert!(approx(
        ohms_law("current", Some(12.0), None, Some(4.0)).unwrap().values[0],
        3.0
    ));
    assert!(approx(
        ohms_law("resistance", Some(12.0), Some(3.0), None).unwrap().values[0],
        4.0
    ));
}

#[test]
fn test_electricity_power_three_forms_agree() {
    // V=12, I=3, R=4 -> 36 W by all three forms
    let r = electrical_power(Some(12.0), Some(3.0), Some(4.0)).unwrap();
    assert!(approx(r.values[0], 36.0));
    assert_eq!(r.verify_values.len(), 3);
}

#[test]
fn test_electricity_series_parallel_hand_computed() {
    assert!(approx(series_resistance(&[2.0, 3.0, 5.0]).unwrap().values[0], 10.0));
    // 1/(1/4+1/6+1/12) = 1/(0.5) = 2
    assert!(approx(parallel_resistance(&[4.0, 6.0, 12.0]).unwrap().values[0], 2.0));
}

#[test]
fn test_electricity_series_current_kvl() {
    // 12 V over series 2+3+1 -> I = 2 A; KVL re-checked inside
    let r = series_parallel_current(12.0, &[2.0, 3.0, 1.0]).unwrap();
    assert!(approx(r.values[0], 2.0));
}

#[test]
fn test_electricity_error_paths() {
    assert!(ohms_law("current", Some(12.0), None, Some(0.0)).is_err()); // short circuit
    assert!(parallel_resistance(&[0.0, 4.0]).is_err()); // zero resistance
    assert!(electrical_power(Some(12.0), None, None).is_err()); // only one input
}

// ---- class TestDensityPressure ----

#[test]
fn test_density_pressure_density_hand_computed() {
    let r = density(Some(5.0), Some(2.0), None).unwrap();
    assert!(approx(r.values[0].num(), 2.5));
    assert_eq!(r.values[1], Val::Bool(true)); // floats on water
}

#[test]
fn test_density_pressure_volume_from_density() {
    // 1000 kg/m^3 and 5 kg -> 0.005 m^3
    assert!(approx(
        density(Some(5.0), None, Some(1000.0)).unwrap().values[0].num(),
        0.005
    ));
}

#[test]
fn test_density_pressure_pressure_force_area_hand_computed() {
    assert!(approx(
        pressure_from_force(100.0, 2.0).unwrap().values[0].num(),
        50.0
    ));
}

#[test]
fn test_density_pressure_hydrostatic_cross_checked_against_force_over_area() {
    // rho*g*h = 1000*9.8*10 = 98000; also weight/area = 1000*(2*10)*9.8/2 = 98000
    let r = hydrostatic_pressure(1000.0, 10.0, Some(2.0), None).unwrap();
    assert!(approx(r.values[0].num(), 98000.0));
    assert!(approx(r.values[1].num(), 98000.0));
}

#[test]
fn test_density_pressure_float_test_archimedes() {
    let r = float_test(600.0, None).unwrap(); // wood on water
    assert_eq!(r.values[0], Val::Bool(true));
    assert!(approx(r.values[1].num(), 0.6));
}

// ---- class TestChemBalance ----

#[test]
fn test_chem_balance_simple_balance() {
    let eq = balance_equation("H2 + O2 -> H2O").unwrap();
    assert_eq!(eq.reactant_coeffs, vec![2, 1]);
    assert_eq!(eq.product_coeffs, vec![2]);
}

#[test]
fn test_chem_balance_propane_balance_hand_known() {
    assert_eq!(
        balance_equation("C3H8 + O2 -> CO2 + H2O").unwrap().formatted(),
        "C3H8 + 5 O2 -> 3 CO2 + 4 H2O"
    );
}

#[test]
fn test_chem_balance_iron_oxide_balance() {
    assert_eq!(
        balance_equation("Fe + O2 -> Fe2O3").unwrap().formatted(),
        "4 Fe + 3 O2 -> 2 Fe2O3"
    );
}

#[test]
fn test_chem_balance_unbalanceable_raises() {
    // element appears out of nowhere
    assert!(balance_equation("H2 -> H2 + O2").is_err());
}

#[test]
fn test_chem_balance_limiting_reagent_hand_computed() {
    // N2 + 3 H2 -> 2 NH3 with 2 mol N2 and 3 mol H2:
    // extents 2/1=2 vs 3/3=1 -> H2 limiting, NH3 = 1*2 = 2 mol
    let r = limiting_reagent("N2 + H2 -> NH3", &[("N2", 2.0), ("H2", 3.0)], "NH3").unwrap();
    assert_eq!(r.limiting_reactant, "H2");
    assert!(approx(r.moles_product, 2.0));
    assert!(approx(r.leftover("N2").unwrap(), 1.0));
    assert!(approx(r.leftover("H2").unwrap(), 0.0));
}

// ---- class TestGasLaws ----

#[test]
fn test_gas_laws_ideal_gas_hand_computed() {
    // V = 1*0.0821*273/1 = 22.41 L (classic molar volume)
    assert!((ideal_gas("volume", Some(1.0), None, Some(1.0), Some(273.0), None)
        .unwrap()
        .value
        - 22.4133)
        .abs()
        < 1e-3); // places=3
}

#[test]
fn test_gas_laws_ideal_gas_solve_moles() {
    // n = PV/(RT) = 1*22.4133/(0.0821*273) ~ 1
    assert!((ideal_gas("moles", Some(1.0), Some(22.4133), None, Some(273.0), None)
        .unwrap()
        .value
        - 1.0)
        .abs()
        < 1e-3); // places=3
}

#[test]
fn test_gas_laws_boyle_hand_computed() {
    assert!(approx(boyle_law(2.0, 6.0, "v2", Some(3.0), None, true).unwrap().value, 4.0));
    assert!(approx(boyle_law(2.0, 6.0, "p2", None, Some(4.0), true).unwrap().value, 3.0));
}

#[test]
fn test_gas_laws_charles_hand_computed() {
    // V1/T1 = V2/T2: 1 L at 273 K -> at 546 K doubles to 2 L
    assert!(approx(charles_law(1.0, 273.0, "v2", None, Some(546.0)).unwrap().value, 2.0));
}

#[test]
fn test_gas_laws_combined_hand_computed() {
    // P1V1/T1 = P2V2/T2 with everything doubled/consistent
    let r = combined_gas_law(2.0, 3.0, 300.0, "v2", Some(1.0), None, Some(300.0)).unwrap();
    assert!(approx(r.value, 6.0));
}

#[test]
fn test_gas_laws_absolute_zero_rejected() {
    assert!(ideal_gas("volume", Some(1.0), None, Some(1.0), Some(0.0), None).is_err());
}

// ---- class TestSolutions ----

#[test]
fn test_solutions_molarity_from_mass_hand_computed() {
    // 4 g NaOH (40 g/mol) = 0.1 mol in 0.5 L -> 0.2 M
    let r = molarity(None, Some(0.5), None, Some(4.0), Some("NaOH")).unwrap();
    assert!(approx(r.values.num("molarity"), 0.2));
}

#[test]
fn test_solutions_dilution_conserves_moles() {
    let r = dilution(Some(2.0), Some(0.5), Some(0.5), None).unwrap(); // -> v2 = 2 L
    assert!(approx(r.values.num("value"), 2.0));
}

#[test]
fn test_solutions_percent_composition_sums_to_100() {
    let r = percent_composition("H2O").unwrap();
    assert!((r.values.num("O") + r.values.num("H") - 100.0).abs() < 1e-2); // places=2
    assert!((r.values.num("H") - 11.21).abs() < 1e-1); // places=1
}

#[test]
fn test_solutions_percent_yield_range_enforced() {
    assert!(approx(
        percent_yield(8.0, 10.0).unwrap().values.num("percent_yield"),
        80.0
    ));
    assert!(percent_yield(11.0, 10.0).is_err()); // >100% is inconsistent
}

// ---- class TestPH ----

#[test]
fn test_ph_roundtrip() {
    assert!(approx(
        ph_from_concentration(1e-3).unwrap().values.num("pH"),
        3.0
    ));
    assert!((concentration_from_ph(3.0).unwrap().values.num("h+") - 1e-3).abs() < 1e-12); // places=12
}

#[test]
fn test_ph_pair_and_kw_identity() {
    let r = ph_poh_pair(Some(3.0), None).unwrap();
    assert!(approx(r.values.num("pOH"), 11.0));
    assert!((r.values.num("h+") * r.values.num("oh-") - 1e-14).abs() < 1e-20); // places=20
}

#[test]
fn test_ph_neutralization() {
    let r = neutralization(1.0, 1, 1.0, 1).unwrap();
    assert!(approx(r.values.num("excess_equivalents"), 0.0));
    assert_eq!(r.values.get("status").unwrap().string(), "neutralized exactly");
    let r2 = neutralization(2.0, 1, 1.0, 1).unwrap();
    assert!(approx(r2.values.num("excess_equivalents"), 1.0));
    assert_eq!(r2.values.get("status").unwrap().string(), "acid in excess");
}
