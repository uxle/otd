//! Phase 111 — Physics: work, energy, and power (Rust port of
//! python/science/energy_domain.py)
//!
//! Ground truth is the standard textbook definitions (KE = 0.5*m*v^2,
//! PE = m*g*h, W = F*d, P = W/t). Independent cross-checks: the impact
//! speed from energy conservation is re-derived through the kinematics
//! identity from physics_domain, and every other result is re-checked
//! through a rearrangement of its own formula.

use reasoning_common::py_float_str;

use crate::forces_domain::DEFAULT_G;
use crate::physics_domain::{solve_kinematics, KinematicsInputs};

/// Result of an energy-domain computation with its independent re-check.
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyResult {
    pub value: f64,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_value: f64,
}

/// KE = 0.5*m*v^2. Re-check: KE*2/m must reproduce v^2, and
/// sqrt(2*KE/m) must reproduce |v|.
pub fn kinetic_energy(mass: f64, velocity: f64) -> Result<EnergyResult, String> {
    if mass <= 0.0 {
        return Err("mass must be positive".to_string());
    }
    let ke = 0.5 * mass * velocity.powi(2);
    let v_back = (2.0 * ke / mass).powf(0.5);
    if (v_back - velocity.abs()).abs() > 1e-9 {
        return Err("independent re-check failed for KE = 0.5*m*v^2".to_string());
    }
    Ok(EnergyResult {
        value: ke,
        equation_used: "KE = 0.5*m*v^2".to_string(),
        verify_method: "reconstruct |v| = sqrt(2*KE/m)".to_string(),
        verify_value: v_back,
    })
}

/// PE = m*g*h. Re-check: PE/(m*g) must reproduce h and PE/(g*h) the mass.
pub fn potential_energy(mass: f64, height: f64, g: Option<f64>) -> Result<EnergyResult, String> {
    let g = g.unwrap_or(DEFAULT_G);
    if mass <= 0.0 {
        return Err("mass must be positive".to_string());
    }
    let pe = mass * g * height;
    if (pe / (mass * g) - height).abs() > 1e-9 {
        return Err("independent re-check failed for PE = m*g*h".to_string());
    }
    Ok(EnergyResult {
        value: pe,
        equation_used: "PE = m*g*h".to_string(),
        verify_method: "reconstruct h = PE/(m*g)".to_string(),
        verify_value: pe / (mass * g),
    })
}

/// W = F*d (force along the direction of motion). Re-check: W/F == d.
pub fn work_done(force: f64, distance: f64) -> Result<EnergyResult, String> {
    let w = force * distance;
    if (w / force - distance).abs() > 1e-9 {
        return Err("independent re-check failed for W = F*d".to_string());
    }
    Ok(EnergyResult {
        value: w,
        equation_used: "W = F*d".to_string(),
        verify_method: "reconstruct d = W/F".to_string(),
        verify_value: w / force,
    })
}

/// P = W/t. Re-check: P*t must reproduce W.
pub fn power_from_work(work: f64, time: f64) -> Result<EnergyResult, String> {
    if time <= 0.0 {
        return Err("time must be positive".to_string());
    }
    let p = work / time;
    if (p * time - work).abs() > 1e-9 {
        return Err("independent re-check failed for P = W/t".to_string());
    }
    Ok(EnergyResult {
        value: p,
        equation_used: "P = W/t".to_string(),
        verify_method: "reconstruct W = P*t".to_string(),
        verify_value: p * time,
    })
}

/// P = F*v. Re-check: P/F == v and P/v == F.
pub fn power_from_force_velocity(force: f64, velocity: f64) -> Result<EnergyResult, String> {
    let p = force * velocity;
    if (p / force - velocity).abs() > 1e-9 || (p / velocity - force).abs() > 1e-9 {
        return Err("independent re-check failed for P = F*v".to_string());
    }
    Ok(EnergyResult {
        value: p,
        equation_used: "P = F*v".to_string(),
        verify_method: "reconstruct v = P/F".to_string(),
        verify_value: p / force,
    })
}

/// Speed after falling from rest through height h.
///
/// PRIMARY computation: energy conservation m*g*h = 0.5*m*v^2 -> v =
/// sqrt(2*g*h) (mass cancels). INDEPENDENT re-check: the kinematics
/// equation v^2 = u^2 + 2*a*s from physics_domain with u=0, a=g, s=h -- a
/// different physical law that must agree.
pub fn impact_speed_from_height(height: f64, g: Option<f64>) -> Result<EnergyResult, String> {
    let g = g.unwrap_or(DEFAULT_G);
    if height < 0.0 {
        return Err("height cannot be negative".to_string());
    }
    let v_energy = (2.0 * g * height).powf(0.5);
    let v_kinematics = solve_kinematics(
        "v",
        &KinematicsInputs {
            u: Some(0.0),
            a: Some(g),
            s: Some(height),
            ..Default::default()
        },
    )?
    .value;
    if (v_energy - v_kinematics).abs() > 1e-6 {
        return Err(format!(
            "cross-check failed: energy gives {}, kinematics gives {}",
            py_float_str(v_energy),
            py_float_str(v_kinematics)
        ));
    }
    Ok(EnergyResult {
        value: v_energy,
        equation_used: "v = sqrt(2*g*h) (energy conservation)".to_string(),
        verify_method: "kinematics v^2 = u^2 + 2*a*s from physics_domain".to_string(),
        verify_value: v_kinematics,
    })
}

/// Height needed to reach `speed` when dropped from rest -- the inverse of
/// impact_speed_from_height, same double-check structure.
///
/// PRIMARY computation: v^2 = 2*g*h -> h = v^2/(2*g). INDEPENDENT re-check:
/// a two-step kinematics path (t = (v-u)/a then s = (u+v)/2 * t), both
/// from physics_domain.solve_kinematics.
pub fn height_for_speed(speed: f64, g: Option<f64>) -> Result<EnergyResult, String> {
    let g = g.unwrap_or(DEFAULT_G);
    if speed < 0.0 {
        return Err("speed cannot be negative".to_string());
    }
    let h_energy = speed.powi(2) / (2.0 * g);
    let t = solve_kinematics(
        "t",
        &KinematicsInputs {
            v: Some(speed),
            u: Some(0.0),
            a: Some(g),
            ..Default::default()
        },
    )?
    .value;
    let h_kinematics = solve_kinematics(
        "s",
        &KinematicsInputs {
            u: Some(0.0),
            v: Some(speed),
            t: Some(t),
            ..Default::default()
        },
    )?
    .value;
    if (h_energy - h_kinematics).abs() > 1e-6 {
        return Err(format!(
            "cross-check failed: energy gives {}, kinematics gives {}",
            py_float_str(h_energy),
            py_float_str(h_kinematics)
        ));
    }
    Ok(EnergyResult {
        value: h_energy,
        equation_used: "h = v^2/(2*g) (energy conservation)".to_string(),
        verify_method: "kinematics t=(v-u)/a then s=(u+v)/2*t".to_string(),
        verify_value: h_kinematics,
    })
}

// ---------------------------------------------------------------------------
// Phase 130b (OTD3) — the COMPLETE ENERGY TAXONOMY: two categories, nine
// forms. Energy is either stored (potential) or moving (kinetic):
//
//   KINETIC (5): mechanical, thermal, radiant, electrical, sound
//   POTENTIAL (4): chemical, gravitational, nuclear, elastic
//
// Every function re-verifies through an independent rearrangement.
// ---------------------------------------------------------------------------

/// Thermal energy (internal): U = m·c·ΔT above a reference.
pub fn thermal_energy(mass: f64, specific_heat: f64, delta_t: f64) -> Result<EnergyResult, String> {
    if mass <= 0.0 || specific_heat <= 0.0 {
        return Err("mass and specific heat must be positive".to_string());
    }
    let u = mass * specific_heat * delta_t;
    if (u / (mass * specific_heat) - delta_t).abs() > 1e-9 {
        return Err("independent re-check failed for U = m·c·ΔT".to_string());
    }
    Ok(EnergyResult {
        value: u,
        equation_used: "U = m·c·ΔT".to_string(),
        verify_method: "reconstruct ΔT = U/(m·c)".to_string(),
        verify_value: u / (mass * specific_heat),
    })
}

/// Radiant energy of one photon: E = h·f (h = 6.626e-34 J·s).
pub fn photon_energy(frequency_hz: f64) -> Result<EnergyResult, String> {
    if frequency_hz <= 0.0 {
        return Err("frequency must be positive".to_string());
    }
    const H: f64 = 6.62607015e-34;
    let e = H * frequency_hz;
    if (e / H - frequency_hz).abs() > 1e-6 {
        return Err("independent re-check failed for E = h·f".to_string());
    }
    Ok(EnergyResult {
        value: e,
        equation_used: "E = h·f (Planck)".to_string(),
        verify_method: "reconstruct f = E/h".to_string(),
        verify_value: e / H,
    })
}

/// Electrical energy: E = P·t = V·I·t.
pub fn electrical_energy(voltage: f64, current: f64, time_s: f64) -> Result<EnergyResult, String> {
    if voltage < 0.0 || current < 0.0 || time_s < 0.0 {
        return Err("voltage, current, time must be non-negative".to_string());
    }
    let e = voltage * current * time_s;
    if (e / (voltage.max(1e-12) * time_s.max(1e-12)) - current).abs() > 1e-9 * current.abs().max(1e-12) {
        return Err("independent re-check failed for E = V·I·t".to_string());
    }
    Ok(EnergyResult {
        value: e,
        equation_used: "E = V·I·t".to_string(),
        verify_method: "reconstruct I = E/(V·t)".to_string(),
        verify_value: e / (voltage.max(1e-12) * time_s.max(1e-12)),
    })
}

/// Sound intensity level of a point source at distance d:
/// I = P/(4π·d²) — the inverse-square law of every loudspeaker.
pub fn sound_intensity(power_w: f64, distance_m: f64) -> Result<EnergyResult, String> {
    if power_w < 0.0 || distance_m <= 0.0 {
        return Err("power non-negative; distance positive".to_string());
    }
    let i = power_w / (4.0 * std::f64::consts::PI * distance_m * distance_m);
    // re-check: I·4π·d² = P
    if (i * 4.0 * std::f64::consts::PI * distance_m * distance_m - power_w).abs() > 1e-9 * power_w {
        return Err("independent re-check failed for I = P/(4π·d²)".to_string());
    }
    Ok(EnergyResult {
        value: i,
        equation_used: "I = P/(4π·d²)".to_string(),
        verify_method: "I·4π·d² = P".to_string(),
        verify_value: i * 4.0 * std::f64::consts::PI * distance_m * distance_m,
    })
}

/// Chemical energy: E = m·e_d (energy density of the fuel/food).
pub fn chemical_energy(mass: f64, energy_density_j_kg: f64) -> Result<EnergyResult, String> {
    if mass <= 0.0 || energy_density_j_kg < 0.0 {
        return Err("mass positive; energy density non-negative".to_string());
    }
    let e = mass * energy_density_j_kg;
    if (e / mass - energy_density_j_kg).abs() > 1e-9 {
        return Err("independent re-check failed for E = m·e_d".to_string());
    }
    Ok(EnergyResult {
        value: e,
        equation_used: "E = m·e_d".to_string(),
        verify_method: "E/m = e_d".to_string(),
        verify_value: e / mass,
    })
}

/// Nuclear energy: E = m·c²·f (f = mass-to-energy conversion fraction;
/// fission ~0.0009, fusion ~0.004).
pub fn nuclear_energy(mass: f64, fraction: f64) -> Result<EnergyResult, String> {
    if mass <= 0.0 {
        return Err("mass must be positive".to_string());
    }
    if !(0.0..=1.0).contains(&fraction) {
        return Err("conversion fraction must be 0–1 (fission ≈ 0.0009, fusion ≈ 0.004)".to_string());
    }
    const C: f64 = 299_792_458.0;
    let e = mass * C * C * fraction;
    if (e / (mass * C * C) - fraction).abs() > 1e-12 {
        return Err("independent re-check failed for E = mc²·f".to_string());
    }
    Ok(EnergyResult {
        value: e,
        equation_used: "E = mc²·f".to_string(),
        verify_method: "E/(mc²) = f".to_string(),
        verify_value: e / (mass * C * C),
    })
}

/// Elastic potential energy: U = ½·k·x².
pub fn elastic_energy(spring_const: f64, displacement: f64) -> Result<EnergyResult, String> {
    if spring_const <= 0.0 {
        return Err("spring constant must be positive".to_string());
    }
    let u = 0.5 * spring_const * displacement.powi(2);
    let x_back = (2.0 * u / spring_const).powf(0.5);
    if (x_back - displacement.abs()).abs() > 1e-9 {
        return Err("independent re-check failed for U = ½·k·x²".to_string());
    }
    Ok(EnergyResult {
        value: u,
        equation_used: "U = ½·k·x²".to_string(),
        verify_method: "reconstruct |x| = sqrt(2U/k)".to_string(),
        verify_value: x_back,
    })
}

/// The taxonomy classifier: name a form of energy from its physics.
pub fn classify_energy_form(form: &str) -> Result<(&'static str, &'static str, &'static str), String> {
    let (category, formula, example) = match form.to_lowercase().as_str() {
        "mechanical" => ("kinetic", "KE = ½mv² (plus PE of the same machine)", "a rolling ball, a spinning turbine, a flying bird"),
        "thermal" | "heat" => ("kinetic", "U = m·c·ΔT (jiggling atoms)", "the faster they move, the hotter it is"),
        "radiant" | "light" => ("kinetic", "E = h·f (electromagnetic waves)", "sunlight, X-rays, radio — all one spectrum"),
        "electrical" => ("kinetic", "E = V·I·t (drifting electrons)", "what powers lights, TVs and computers"),
        "sound" => ("kinetic", "I = P/(4π·d²) (vibrating matter)", "pressure waves through air, water, steel"),
        "chemical" => ("potential", "E = m·e_d (bonds between atoms)", "food, wood, batteries, gasoline"),
        "gravitational" => ("potential", "PE = mgh (height above a field)", "a rock resting at the top of a hill"),
        "nuclear" => ("potential", "E = mc²·f (the nucleus itself)", "fission of uranium, fusion of hydrogen"),
        "elastic" => ("potential", "U = ½kx² (stretch and squeeze)", "a pulled rubber band, a squished spring"),
        _ => return Err(format!("'{}' is not one of the nine forms — mechanical, thermal, radiant, electrical, sound, chemical, gravitational, nuclear, elastic", form)),
    };
    Ok((category, formula, example))
}

#[cfg(test)]
mod taxonomy_tests {
    use super::*;

    #[test]
    fn nine_forms_classify() {
        for form in ["mechanical", "thermal", "radiant", "electrical", "sound",
                     "chemical", "gravitational", "nuclear", "elastic"] {
            let (cat, _, _) = classify_energy_form(form).unwrap();
            let kinetic = ["mechanical", "thermal", "radiant", "electrical", "sound"].contains(&form);
            assert_eq!(cat, if kinetic { "kinetic" } else { "potential" }, "{} misfiled", form);
        }
        assert!(classify_energy_form("dark").is_err());
    }

    #[test]
    fn thermal_of_a_kettle() {
        let r = thermal_energy(1.0, 4186.0, 80.0).unwrap();
        assert!((r.value - 334880.0).abs() < 1e-6);
    }

    #[test]
    fn photon_of_green_light() {
        // 545 THz → ~3.6e-19 J
        let r = photon_energy(5.45e14).unwrap();
        assert!((r.value / 1e-19 - 3.61).abs() < 0.01);
    }

    #[test]
    fn one_kilowatt_hour() {
        // 230 V × 4.3478 A × 3600 s = 3.6 MJ = 1 kWh exactly
        let r = electrical_energy(230.0, 4.3478, 3600.0).unwrap();
        assert!((r.value / 3.6e6 - 1.0).abs() < 1e-3, "1 kWh = 3.6 MJ");
    }

    #[test]
    fn fission_fraction() {
        // 1 kg at 0.0009: 0.0009 × 8.98755e16 = 8.089e13 J
        let r = nuclear_energy(1.0, 0.0009).unwrap();
        assert!((r.value / 8.0888e13 - 1.0).abs() < 1e-3);
    }

    #[test]
    fn toy_spring() {
        // k=100 N/m, x=10 cm → 0.5 J
        let r = elastic_energy(100.0, 0.1).unwrap();
        assert!((r.value - 0.5).abs() < 1e-12);
    }

    #[test]
    fn gasoline_beats_batteries() {
        let gas = chemical_energy(1.0, 46e6).unwrap().value;
        let li = chemical_energy(1.0, 0.9e6).unwrap().value;
        assert!(gas > li * 50.0, "gasoline carries ~50× lithium per kilo");
    }
}
