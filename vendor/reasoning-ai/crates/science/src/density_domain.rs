//! Phase 114 — Physics: density and pressure (Rust port of
//! python/science/density_domain.py)
//!
//! Textbook ground truth: rho = m/V, P = F/A, P = rho*g*h, float iff
//! rho_object < rho_fluid. Independent cross-checks: density recomputed
//! both ways, and hydrostatic pressure cross-checked against the
//! force/area definition via the weight of the liquid column.

use reasoning_common::py_float_str;

use crate::forces_domain::DEFAULT_G;
use crate::val::Val;

/// kg/m^3 at ~4 degrees C, standard textbook value.
pub const WATER_DENSITY: f64 = 1000.0;

/// Result of a density/pressure computation. `values` may mix floats and
/// bools (Python returned heterogeneous tuples like `(rho, floats)`).
#[derive(Debug, Clone, PartialEq)]
pub struct DensityResult {
    pub values: Vec<Val>,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: Vec<Val>,
}

/// rho = m/V for any one of the three given the other two (which one is
/// solved is decided by which parameters are None, exactly like Python).
pub fn density(
    mass: Option<f64>,
    volume: Option<f64>,
    rho: Option<f64>,
) -> Result<DensityResult, String> {
    if rho.is_none() && (mass.is_none() || volume.is_none()) {
        return Err("solving for density needs mass and volume".to_string());
    }
    if let Some(rho) = rho {
        if let Some(volume) = volume {
            let m = rho * volume;
            if (m / volume - rho).abs() > 1e-9 {
                return Err("independent re-check failed for rho = m/V".to_string());
            }
            return Ok(DensityResult {
                values: vec![Val::Num(m)],
                equation_used: "m = rho*V".to_string(),
                verify_method: "reconstruct rho = m/V".to_string(),
                verify_values: vec![Val::Num(m / volume)],
            });
        }
        if let Some(mass) = mass {
            if rho <= 0.0 {
                return Err("density must be positive".to_string());
            }
            let v = mass / rho;
            if (rho * v - mass).abs() > 1e-9 {
                return Err("independent re-check failed for rho = m/V".to_string());
            }
            return Ok(DensityResult {
                values: vec![Val::Num(v)],
                equation_used: "V = m/rho".to_string(),
                verify_method: "reconstruct m = rho*V".to_string(),
                verify_values: vec![Val::Num(rho * v)],
            });
        }
        return Err("solving for mass or volume needs rho and the other one".to_string());
    }
    let mass = mass.unwrap();
    let volume = volume.unwrap();
    if volume == 0.0 {
        return Err("volume 0 makes density undefined".to_string());
    }
    if mass < 0.0 || volume < 0.0 {
        return Err("mass and volume cannot be negative".to_string());
    }
    let d = mass / volume;
    if (d * volume - mass).abs() > 1e-9 {
        return Err("independent re-check failed for rho = m/V".to_string());
    }
    let floats = d < WATER_DENSITY;
    Ok(DensityResult {
        values: vec![Val::Num(d), Val::Bool(floats)],
        equation_used: "rho = m/V".to_string(),
        verify_method: "reconstruct m = rho*V + float test vs water".to_string(),
        verify_values: vec![Val::Num(d * volume), Val::Bool(floats)],
    })
}

/// P = F/A. Re-check: F = P*A reproduces the input force.
pub fn pressure_from_force(force: f64, area: f64) -> Result<DensityResult, String> {
    if area <= 0.0 {
        return Err("area must be positive".to_string());
    }
    let p = force / area;
    if (p * area - force).abs() > 1e-9 {
        return Err("independent re-check failed for P = F/A".to_string());
    }
    Ok(DensityResult {
        values: vec![Val::Num(p)],
        equation_used: "P = F/A".to_string(),
        verify_method: "reconstruct F = P*A".to_string(),
        verify_values: vec![Val::Num(p * area)],
    })
}

/// P = rho*g*h at the base of a liquid column of height h.
///
/// INDEPENDENT re-check via the force/area definition (a different law):
/// the liquid in the column weighs m*g = rho*(A*h)*g, and weight/area
/// must reproduce the same pressure. Requires an area to run the check.
pub fn hydrostatic_pressure(
    density_fluid: f64,
    height: f64,
    area: Option<f64>,
    g: Option<f64>,
) -> Result<DensityResult, String> {
    let g = g.unwrap_or(DEFAULT_G);
    if density_fluid <= 0.0 || height < 0.0 {
        return Err("density must be positive and height non-negative".to_string());
    }
    let p = density_fluid * g * height;
    if let Some(area) = area {
        if area > 0.0 {
            let volume = area * height;
            let weight = density_fluid * volume * g;
            let p_check = weight / area;
            if (p_check - p).abs() > 1e-6 * f64::max(1.0, p.abs()) {
                return Err(format!(
                    "cross-check failed: rho*g*h gives {}, F/A gives {}",
                    py_float_str(p),
                    py_float_str(p_check)
                ));
            }
            return Ok(DensityResult {
                values: vec![Val::Num(p), Val::Num(p_check)],
                equation_used: "P = rho*g*h".to_string(),
                verify_method: "weight-of-column / area (P = F/A definition)".to_string(),
                verify_values: vec![Val::Num(p), Val::Num(p_check)],
            });
        }
    }
    Ok(DensityResult {
        values: vec![Val::Num(p)],
        equation_used: "P = rho*g*h".to_string(),
        verify_method: "(supply area for the F/A cross-check)".to_string(),
        verify_values: vec![Val::Num(p)],
    })
}

/// An object floats iff its density is below the fluid's. Re-check via the
/// Archimedes framing: submerged fraction = rho_obj/rho_fluid, which must
/// be < 1 exactly when it floats (two formulations of the same physical
/// condition agreeing is the check).
pub fn float_test(density_object: f64, density_fluid: Option<f64>) -> Result<DensityResult, String> {
    let density_fluid = density_fluid.unwrap_or(WATER_DENSITY);
    if density_object <= 0.0 || density_fluid <= 0.0 {
        return Err("densities must be positive".to_string());
    }
    let floats = density_object < density_fluid;
    let submerged_fraction = density_object / density_fluid;
    let consistent = (submerged_fraction < 1.0) == floats;
    if !consistent {
        return Err(
            "Archimedes re-check failed: float test disagrees with submerged fraction".to_string(),
        );
    }
    Ok(DensityResult {
        values: vec![Val::Bool(floats), Val::Num(submerged_fraction)],
        equation_used: "floats iff rho_obj < rho_fluid".to_string(),
        verify_method: "Archimedes submerged-fraction < 1 test".to_string(),
        verify_values: vec![Val::Num(submerged_fraction)],
    })
}
