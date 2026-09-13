//! Phase 133 — Physics: relativity — time, speed, and Einstein's limits.
//!
//! The faster you move through space, the slower you move through time.
//! GPS satellites at 3.9 km/s run 7 μs/day slow and maps would drift
//! kilometres without the correction. c is not a speed, it is THE speed.

/// Result with its independent re-check.
#[derive(Debug, Clone, PartialEq)]
pub struct RelativityResult {
    pub values: Vec<f64>,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: Vec<f64>,
}

pub const C_LIGHT: f64 = 299_792_458.0;

/// Lorentz factor γ = 1/√(1 − v²/c²). The stretch factor of moving time.
pub fn lorentz_factor(v_mps: f64) -> Result<RelativityResult, String> {
    if v_mps < 0.0 {
        return Err("speed is a magnitude — non-negative".into());
    }
    if v_mps >= C_LIGHT {
        return Err("nothing massive reaches c — the universe's speed limit".into());
    }
    let beta = v_mps / C_LIGHT;
    let gamma = 1.0 / (1.0 - beta * beta).sqrt();
    // re-check: γ²·(1 − v²/c²) = 1
    let lhs = gamma * gamma * (1.0 - beta * beta);
    if (lhs - 1.0).abs() > 1e-12 {
        return Err("re-check failed for γ = 1/√(1 − v²/c²)".into());
    }
    Ok(RelativityResult {
        values: vec![gamma],
        equation_used: "γ = 1/√(1 − v²/c²)".into(),
        verify_method: "γ²·(1 − v²/c²) = 1".into(),
        verify_values: vec![lhs],
    })
}

/// Time dilation: Δt = γ·Δt₀. Solve for either given the other + v.
pub fn time_dilation(v_mps: f64, proper_s: Option<f64>, dilated_s: Option<f64>) -> Result<RelativityResult, String> {
    let g = lorentz_factor(v_mps)?.values[0];
    let (t0, t) = match (proper_s, dilated_s) {
        (Some(t0), None) => (t0, g * t0),
        (None, Some(t)) => (t / g, t),
        _ => return Err("give exactly one of proper time, dilated time".into()),
    };
    if t0 < 0.0 {
        return Err("time non-negative".into());
    }
    // re-check: t/t0 = γ
    if t0 > 0.0 && (t / t0 - g).abs() > 1e-9 {
        return Err("re-check failed for Δt = γ·Δt₀".into());
    }
    Ok(RelativityResult {
        values: vec![t, t0],
        equation_used: "Δt = γ·Δt₀".into(),
        verify_method: "Δt/Δt₀ = γ".into(),
        verify_values: vec![if t0 > 0.0 { t / t0 } else { g }],
    })
}

/// Length contraction: L = L₀/γ.
pub fn length_contraction(v_mps: f64, proper_m: f64) -> Result<RelativityResult, String> {
    let g = lorentz_factor(v_mps)?.values[0];
    if proper_m < 0.0 {
        return Err("length non-negative".into());
    }
    let l = proper_m / g;
    // re-check: L·γ = L₀
    if (l * g - proper_m).abs() > 1e-9 * proper_m {
        return Err("re-check failed for L = L₀/γ".into());
    }
    Ok(RelativityResult {
        values: vec![l],
        equation_used: "L = L₀/γ".into(),
        verify_method: "L·γ = L₀".into(),
        verify_values: vec![l * g],
    })
}

/// Rest-mass energy: E = mc².
pub fn rest_energy(mass_kg: f64) -> Result<RelativityResult, String> {
    if mass_kg < 0.0 {
        return Err("mass non-negative".into());
    }
    let e = mass_kg * C_LIGHT * C_LIGHT;
    // re-check: E/c² = m
    if (e / (C_LIGHT * C_LIGHT) - mass_kg).abs() > 1e-9 * mass_kg {
        return Err("re-check failed for E = mc²".into());
    }
    Ok(RelativityResult {
        values: vec![e],
        equation_used: "E = mc²".into(),
        verify_method: "E/c² = m".into(),
        verify_values: vec![e / (C_LIGHT * C_LIGHT)],
    })
}

/// Relativistic velocity addition: u' = (u+v)/(1 + uv/c²) — why c is the limit.
pub fn velocity_addition(u: f64, v: f64) -> Result<RelativityResult, String> {
    let denom = 1.0 + u * v / (C_LIGHT * C_LIGHT);
    if denom.abs() < 1e-15 {
        return Err("degenerate addition".into());
    }
    let w = (u + v) / denom;
    if w.abs() > C_LIGHT + 1e-6 {
        return Err("velocity addition must never exceed c".into());
    }
    // re-check: invert the transform — u = (w − v)/(1 − wv/c²)
    let u_back = (w - v) / (1.0 - w * v / (C_LIGHT * C_LIGHT));
    if (u_back - u).abs() > 1e-6 * u.abs().max(1.0) {
        return Err("re-check failed for relativistic addition".into());
    }
    Ok(RelativityResult {
        values: vec![w],
        equation_used: "u' = (u + v)/(1 + uv/c²)".into(),
        verify_method: "inverse transform recovers u".into(),
        verify_values: vec![u_back],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standing_still_is_newtonian() {
        assert!((lorentz_factor(0.0).unwrap().values[0] - 1.0).abs() < 1e-15);
        let slow = velocity_addition(10.0, 20.0).unwrap();
        assert!((slow.values[0] - 30.0).abs() < 1e-6);
    }

    #[test]
    fn half_light_stretches_15_percent() {
        let g = lorentz_factor(0.5 * C_LIGHT).unwrap().values[0];
        assert!((g - 1.1547).abs() < 1e-3);
    }

    #[test]
    fn the_speed_limit_holds() {
        let w = velocity_addition(0.9 * C_LIGHT, 0.9 * C_LIGHT).unwrap();
        assert!(w.values[0] < C_LIGHT);
        assert!((w.values[0] / C_LIGHT - 0.9945).abs() < 1e-3);
    }

    #[test]
    fn light_itself_is_rejected() {
        assert!(lorentz_factor(C_LIGHT).is_err());
        assert!(lorentz_factor(1.5 * C_LIGHT).is_err());
    }

    #[test]
    fn paperclip_energy() {
        // 1 g → 9e13 J ≈ 25 GWh
        let r = rest_energy(0.001).unwrap();
        assert!((r.values[0] - 8.987551787e13).abs() < 1e6);
    }

    #[test]
    fn muons_reach_the_ground() {
        // muon lifetime 2.2 μs at 0.999c: γ ≈ 22.4 → 49 μs in the lab
        let v = 0.999 * C_LIGHT;
        let r = time_dilation(v, Some(2.2e-6), None).unwrap();
        assert!(r.values[0] > 40e-6, "dilated lifetime ≈ 49 μs, got {}", r.values[0]);
    }

    #[test]
    fn gps_satellite_clocks() {
        // 3.9 km/s: γ−1 ≈ 8.3e-11 → per day (86400 s): ~7.2 μs
        let g = lorentz_factor(3874.0).unwrap().values[0];
        let drift = (g - 1.0) * 86400.0;
        assert!((drift * 1e6 - 7.2).abs() < 0.2, "GPS drift ≈ 7.2 μs/day, got {:.2}", drift * 1e6);
    }
}
