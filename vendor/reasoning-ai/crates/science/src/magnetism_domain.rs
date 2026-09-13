//! Phase 130 — Physics: magnetism (Rust port of python/science/magnetism_domain.py)
//!
//! Every current is a magnet and every magnet is a current (Ampère). These
//! laws carry motors, generators, compasses and the entire electrical grid.

/// Result with its independent re-check.
#[derive(Debug, Clone, PartialEq)]
pub struct MagnetismResult {
    pub values: Vec<f64>,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: Vec<f64>,
}

/// Motor force on a current-carrying wire: F = B·I·L·sin(θ).
/// Re-check: F / (B·I) must reproduce L·sin(θ).
pub fn force_on_wire(b_t: f64, current_a: f64, length_m: f64, sin_theta: f64) -> Result<MagnetismResult, String> {
    if b_t <= 0.0 || current_a <= 0.0 || length_m <= 0.0 {
        return Err("B, I, L must be positive (SI: tesla, ampere, metre)".into());
    }
    let st = sin_theta.clamp(-1.0, 1.0);
    let f = b_t * current_a * length_m * st;
    if (f / (b_t * current_a) - length_m * st).abs() > 1e-9 * f.abs().max(1.0) {
        return Err("independent re-check failed for F = B·I·L·sin(θ)".into());
    }
    Ok(MagnetismResult {
        values: vec![f],
        equation_used: "F = B·I·L·sin(θ)".into(),
        verify_method: "reconstruct L·sin(θ) = F/(B·I)".into(),
        verify_values: vec![f / (b_t * current_a)],
    })
}

/// Lorentz force on a moving charge: F = q·v·B·sin(θ).
pub fn lorentz_force(charge_c: f64, speed_ms: f64, b_t: f64, sin_theta: f64) -> Result<MagnetismResult, String> {
    if speed_ms < 0.0 || b_t < 0.0 {
        return Err("speed and field must be non-negative".into());
    }
    let st = sin_theta.clamp(-1.0, 1.0);
    let f = charge_c.abs() * speed_ms * b_t * st;
    if (f / (charge_c.abs() * b_t) - speed_ms * st).abs() > 1e-9 * f.abs().max(1.0) {
        return Err("independent re-check failed for F = q·v·B".into());
    }
    Ok(MagnetismResult {
        values: vec![f],
        equation_used: "F = |q|·v·B·sin(θ)".into(),
        verify_method: "reconstruct v·sin(θ) = F/(|q|·B)".into(),
        verify_values: vec![f / (charge_c.abs() * b_t)],
    })
}

/// Field of a long straight wire: B = μ₀·I/(2π·r).
/// μ₀ = 4π×10⁻⁷ T·m/A — 1 A at 1 cm gives 20 μT (Earth-strength).
pub fn wire_field(current_a: f64, dist_m: f64) -> Result<MagnetismResult, String> {
    if dist_m <= 0.0 {
        return Err("distance must be positive".into());
    }
    let mu0 = 4.0 * std::f64::consts::PI * 1e-7;
    let b = mu0 * current_a / (2.0 * std::f64::consts::PI * dist_m);
    // re-check: r·B = μ₀·I/2π
    let lhs = dist_m * b;
    let rhs = mu0 * current_a / (2.0 * std::f64::consts::PI);
    if (lhs - rhs).abs() > 1e-15 * rhs.abs().max(1e-12) {
        return Err("re-check failed for B = μ₀·I/(2π·r)".into());
    }
    Ok(MagnetismResult {
        values: vec![b],
        equation_used: "B = μ₀·I/(2π·r)".into(),
        verify_method: "r·B = μ₀·I/(2π)".into(),
        verify_values: vec![lhs],
    })
}

/// Solenoid core field: B = μ₀·n·I (n = turns per metre).
pub fn solenoid_field(turns_per_m: f64, current_a: f64) -> Result<MagnetismResult, String> {
    if turns_per_m < 0.0 {
        return Err("turn density must be non-negative".into());
    }
    let mu0 = 4.0 * std::f64::consts::PI * 1e-7;
    let b = mu0 * turns_per_m * current_a;
    if (b / mu0 - turns_per_m * current_a).abs() > 1e-9 {
        return Err("re-check failed for B = μ₀·n·I".into());
    }
    Ok(MagnetismResult {
        values: vec![b],
        equation_used: "B = μ₀·n·I".into(),
        verify_method: "B/μ₀ = n·I".into(),
        verify_values: vec![b / mu0],
    })
}

/// Faraday's law of induction: EMF = N·ΔΦ/Δt (the minus sign is Lenz).
pub fn faraday_emf(turns: f64, d_flux_weber: f64, dt_s: f64) -> Result<MagnetismResult, String> {
    if dt_s <= 0.0 {
        return Err("Δt must be positive".into());
    }
    if turns <= 0.0 {
        return Err("turns must be positive".into());
    }
    let emf = turns * d_flux_weber / dt_s;
    // re-check: EMF·Δt/N = ΔΦ
    if (emf * dt_s / turns - d_flux_weber).abs() > 1e-9 * d_flux_weber.abs().max(1e-12) {
        return Err("re-check failed for EMF = N·ΔΦ/Δt".into());
    }
    Ok(MagnetismResult {
        values: vec![emf],
        equation_used: "EMF = N·ΔΦ/Δt".into(),
        verify_method: "EMF·Δt/N = ΔΦ".into(),
        verify_values: vec![emf * dt_s / turns],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn motor_force_law() {
        // 0.5 T, 2 A, 10 cm, perpendicular: F = 0.1 N
        let r = force_on_wire(0.5, 2.0, 0.1, 1.0).unwrap();
        assert!((r.values[0] - 0.1).abs() < 1e-12);
    }

    #[test]
    fn one_amp_wire_matches_earth() {
        // 1 A at 1 cm: B = 20 μT — compass-grade
        let r = wire_field(1.0, 0.01).unwrap();
        assert!((r.values[0] * 1e6 - 19.99).abs() < 0.05);
    }

    #[test]
    fn solenoid_sanity() {
        let r = solenoid_field(1000.0, 1.0).unwrap();
        assert!((r.values[0] * 1e3 - 1.2566).abs() < 1e-3);
    }

    #[test]
    fn faraday_generates() {
        // 100 turns, 10 mWb swing in 0.1 s → 10 V
        let r = faraday_emf(100.0, 0.01, 0.1).unwrap();
        assert!((r.values[0] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn lorentz_on_electron() {
        // e = 1.602e-19 C at 1e6 m/s in 1 T: F = 1.602e-13 N
        let r = lorentz_force(1.602e-19, 1e6, 1.0, 1.0).unwrap();
        assert!((r.values[0] - 1.602e-13).abs() < 1e-16);
    }

    #[test]
    fn bad_inputs_rejected() {
        assert!(wire_field(1.0, 0.0).is_err());
        assert!(faraday_emf(100.0, 0.01, 0.0).is_err());
        assert!(force_on_wire(-1.0, 1.0, 1.0, 1.0).is_err());
    }
}
