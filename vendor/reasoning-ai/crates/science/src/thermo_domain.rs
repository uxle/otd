//! Phase 132 — Physics: thermodynamics (Rust port).
//!
//! Temperature is kinetic energy per atom; heat is energy on the move.
//! The four laws: zeroth (touching equalises), first (energy conserved),
//! second (heat flows hot→cold), third (absolute zero unreachable).

/// Result with its independent re-check.
#[derive(Debug, Clone, PartialEq)]
pub struct ThermoResult {
    pub values: Vec<f64>,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: Vec<f64>,
}

/// Temperature conversion among C, K, F (solve for the missing one).
pub fn convert_temp(c: Option<f64>, k: Option<f64>, f: Option<f64>) -> Result<ThermoResult, String> {
    let known = [c.is_some(), k.is_some(), f.is_some()].iter().filter(|x| **x).count();
    if known != 2 {
        return Err("give exactly two of celsius, kelvin, fahrenheit".into());
    }
    let (cc, kk, _ff) = match (c, k, f) {
        (Some(c), Some(k), _) => (c, k, c * 9.0 / 5.0 + 32.0),
        (Some(c), None, Some(f)) => (c, c + 273.15, f),
        (None, Some(k), Some(f)) => (k - 273.15, k, f),
        _ => unreachable!(),
    };
    if kk < 0.0 {
        return Err("below absolute zero (0 K = −273.15 °C)".into());
    }
    // re-check: K = C + 273.15 and F = 1.8·C + 32
    if (kk - cc - 273.15).abs() > 1e-9 {
        return Err("re-check failed for K = C + 273.15".into());
    }
    Ok(ThermoResult {
        values: vec![cc, kk, cc * 9.0 / 5.0 + 32.0],
        equation_used: "K = C + 273.15; F = 1.8·C + 32".into(),
        verify_method: "K − 273.15 = C and (F − 32)/1.8 = C".into(),
        verify_values: vec![kk - 273.15, (cc * 9.0 / 5.0 + 32.0 - 32.0) / 1.8],
    })
}

/// Sensible heat: Q = m·c·ΔT, solved for any missing one.
pub fn sensible_heat(q: Option<f64>, m: Option<f64>, c: Option<f64>, dt: Option<f64>) -> Result<ThermoResult, String> {
    let known = [q, m, c, dt].iter().filter(|x| x.is_some()).count();
    if known != 3 {
        return Err("give exactly three of Q, m, c, ΔT".into());
    }
    let (qq, mm, cc, dd) = match (q, m, c, dt) {
        (None, Some(m), Some(c), Some(dt)) => (m * c * dt, m, c, dt),
        (Some(q), None, Some(c), Some(dt)) => (q, q / (c * dt), c, dt),
        (Some(q), Some(m), None, Some(dt)) => (q, m, q / (m * dt), dt),
        (Some(q), Some(m), Some(c), None) => (q, m, c, q / (m * c)),
        _ => unreachable!(),
    };
    if mm <= 0.0 || cc <= 0.0 {
        return Err("mass and specific heat must be positive".into());
    }
    // re-check: Q = m·c·ΔT
    if (mm * cc * dd - qq).abs() > 1e-6 * qq.abs().max(1e-9) {
        return Err("re-check failed for Q = m·c·ΔT".into());
    }
    Ok(ThermoResult {
        values: vec![qq],
        equation_used: "Q = m·c·ΔT".into(),
        verify_method: "m·c·ΔT reproduces Q".into(),
        verify_values: vec![mm * cc * dd],
    })
}

/// Latent heat: Q = m·L (phase changes at constant temperature).
pub fn latent_heat(m_kg: f64, l_j_kg: f64) -> Result<ThermoResult, String> {
    if m_kg <= 0.0 || l_j_kg <= 0.0 {
        return Err("mass and latent heat must be positive".into());
    }
    let q = m_kg * l_j_kg;
    if (q / m_kg - l_j_kg).abs() > 1e-9 {
        return Err("re-check failed for Q = m·L".into());
    }
    Ok(ThermoResult {
        values: vec![q],
        equation_used: "Q = m·L".into(),
        verify_method: "Q/m = L".into(),
        verify_values: vec![q / m_kg],
    })
}

/// Ideal gas law: P·V = n·R·T, solved for the missing one.
pub fn ideal_gas(p: Option<f64>, v: Option<f64>, n: Option<f64>, t: Option<f64>) -> Result<ThermoResult, String> {
    const R: f64 = 8.314462618;
    let known = [p, v, n, t].iter().filter(|x| x.is_some()).count();
    if known != 3 {
        return Err("give exactly three of P (Pa), V (m³), n (mol), T (K)".into());
    }
    let (pp, vv, nn, tt) = match (p, v, n, t) {
        (None, Some(v), Some(n), Some(t)) => (n * R * t / v, v, n, t),
        (Some(p), None, Some(n), Some(t)) => (p, n * R * t / p, n, t),
        (Some(p), Some(v), None, Some(t)) => (p, v, p * v / (R * t), t),
        (Some(p), Some(v), Some(n), None) => (p, v, n, p * v / (n * R)),
        _ => unreachable!(),
    };
    if tt <= 0.0 {
        return Err("temperature must be above absolute zero".into());
    }
    if vv <= 0.0 || pp <= 0.0 {
        return Err("pressure and volume must be positive".into());
    }
    // re-check: P·V = n·R·T
    if (pp * vv - nn * R * tt).abs() > 1e-6 * (nn * R * tt).abs() {
        return Err("re-check failed for P·V = n·R·T".into());
    }
    Ok(ThermoResult {
        values: vec![pp],
        equation_used: "P·V = n·R·T".into(),
        verify_method: "n·R·T reproduces P·V".into(),
        verify_values: vec![nn * R * tt],
    })
}

/// Fourier conduction: P = k·A·ΔT / d (heat through a wall).
pub fn conduction(k_wmk: f64, area_m2: f64, dt_c: f64, thickness_m: f64) -> Result<ThermoResult, String> {
    if thickness_m <= 0.0 || area_m2 <= 0.0 {
        return Err("thickness and area must be positive".into());
    }
    if k_wmk < 0.0 {
        return Err("thermal conductivity non-negative".into());
    }
    let p = k_wmk * area_m2 * dt_c / thickness_m;
    // re-check: P·d/(A·ΔT) = k
    let k_back = p * thickness_m / (area_m2 * dt_c);
    if (k_back - k_wmk).abs() > 1e-9 * k_wmk.abs().max(1e-12) {
        return Err("re-check failed for P = k·A·ΔT/d".into());
    }
    Ok(ThermoResult {
        values: vec![p],
        equation_used: "P = k·A·ΔT/d".into(),
        verify_method: "P·d/(A·ΔT) = k".into(),
        verify_values: vec![k_back],
    })
}

/// Stefan–Boltzmann radiant power: P = εσA·T⁴ (σ = 5.67e-8).
pub fn stefan_boltzmann(area_m2: f64, t_kelvin: f64, emissivity: f64) -> Result<ThermoResult, String> {
    if area_m2 <= 0.0 || t_kelvin <= 0.0 {
        return Err("area and temperature must be positive".into());
    }
    let e = emissivity.clamp(0.0, 1.0);
    let sigma = 5.670374419e-8;
    let p = e * sigma * area_m2 * t_kelvin.powi(4);
    // re-check: P/(εσA) = T⁴
    let t4 = p / (e * sigma * area_m2);
    if (t4.powf(0.25) - t_kelvin).abs() > 1e-3 {
        return Err("re-check failed for P = εσA·T⁴".into());
    }
    Ok(ThermoResult {
        values: vec![p],
        equation_used: "P = εσA·T⁴".into(),
        verify_method: "(P/(εσA))^(1/4) = T".into(),
        verify_values: vec![t4.powf(0.25)],
    })
}

/// Linear thermal expansion: ΔL = α·L₀·ΔT.
pub fn thermal_expansion(alpha_1k: f64, l0_m: f64, dt_c: f64) -> Result<ThermoResult, String> {
    if l0_m <= 0.0 {
        return Err("original length must be positive".into());
    }
    let dl = alpha_1k * l0_m * dt_c;
    // re-check: ΔL/(L₀·ΔT) = α
    let a_back = dl / (l0_m * dt_c);
    if (a_back - alpha_1k).abs() > 1e-12 * alpha_1k.abs().max(1e-12) {
        return Err("re-check failed for ΔL = α·L₀·ΔT".into());
    }
    Ok(ThermoResult {
        values: vec![dl, l0_m + dl],
        equation_used: "ΔL = α·L₀·ΔT".into(),
        verify_method: "ΔL/(L₀·ΔT) = α".into(),
        verify_values: vec![a_back],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn water_boils_at_373k() {
        let r = convert_temp(Some(100.0), Some(373.15), None).unwrap();
        assert!((r.values[1] - 373.15).abs() < 1e-9);
        assert!((r.values[2] - 212.0).abs() < 1e-9);
    }

    #[test]
    fn absolute_zero_rejected() {
        assert!(convert_temp(Some(-300.0), Some(-26.85), None).is_err());
    }

    #[test]
    fn boiling_a_kettle() {
        // 1 kg water 20→100 °C: Q = 1×4186×80 = 334,880 J
        let r = sensible_heat(None, Some(1.0), Some(4186.0), Some(80.0)).unwrap();
        assert!((r.values[0] - 334880.0).abs() < 1e-6);
    }

    #[test]
    fn melting_ice() {
        // 0.5 kg ice: Q = 0.5×334000 = 167 kJ
        let r = latent_heat(0.5, 334000.0).unwrap();
        assert!((r.values[0] - 167000.0).abs() < 1e-6);
    }

    #[test]
    fn one_mole_at_stp() {
        // 1 mol at 273.15 K in 22.4 L → ~101 kPa
        let r = ideal_gas(None, Some(0.0224), Some(1.0), Some(273.15)).unwrap();
        assert!((r.values[0] / 1000.0 - 101.3).abs() < 0.1);
    }

    #[test]
    fn conduction_through_glass_window() {
        // 1 m², 4 mm glass, 20° difference: P = 1.05×1×20/0.004 ≈ 5.25 kW
        let r = conduction(1.05, 1.0, 20.0, 0.004).unwrap();
        assert!((r.values[0] - 5250.0).abs() < 1.0);
    }

    #[test]
    fn sun_surface_radiance() {
        // 1 m² at 5778 K, ε=1: P ≈ 63.1 MW (the solar constant's mother)
        let r = stefan_boltzmann(1.0, 5778.0, 1.0).unwrap();
        assert!((r.values[0] / 1e6 - 63.2).abs() < 0.05);
    }

    #[test]
    fn bridge_expands_in_summer() {
        // 100 m steel bridge, +40 K: ΔL = 12e-6×100×40 = 48 mm
        let r = thermal_expansion(12e-6, 100.0, 40.0).unwrap();
        assert!((r.values[0] - 0.048).abs() < 1e-9);
    }
}
