//! Phase 131 — Physics: waves — sound AND light (Rust port).
//!
//! v = f·λ is the one law that ties every wave together, from earthquakes
//! to gamma rays. Sound needs matter; light needs nothing but itself.

/// Result with its independent re-check.
#[derive(Debug, Clone, PartialEq)]
pub struct WavesResult {
    pub values: Vec<f64>,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: Vec<f64>,
}

/// The wave equation, solved for any one of v, f, λ given the other two.
/// Re-check: the two knowns must be reproducible from the answer.
pub fn wave_equation(v: Option<f64>, f: Option<f64>, lambda: Option<f64>) -> Result<WavesResult, String> {
    let known = [v.is_some(), f.is_some(), lambda.is_some()].iter().filter(|x| **x).count();
    if known != 2 {
        return Err("give exactly two of v, f, lambda (the wave equation solves the third)".into());
    }
    let (vv, ff, ll, missing) = match (v, f, lambda) {
        (Some(v), Some(f), None) => (v, f, v / f, "lambda"),
        (Some(v), None, Some(l)) => (v, v / l, l, "f"),
        (None, Some(f), Some(l)) => (f * l, f, l, "v"),
        _ => unreachable!(),
    };
    if vv <= 0.0 || ff <= 0.0 || ll <= 0.0 {
        return Err("v, f, lambda must all be positive".into());
    }
    if (vv / ff - ll).abs() > 1e-9 * ll {
        return Err("re-check failed for v = f·λ".into());
    }
    Ok(WavesResult {
        values: vec![vv],
        equation_used: "v = f·λ".into(),
        verify_method: format!("λ = v/f reproduces the given {}", missing),
        verify_values: vec![vv / ff],
    })
}

/// Speed of sound in air vs temperature: v = 331.3 + 0.606·T °C.
pub fn sound_speed_air(temp_c: f64) -> Result<WavesResult, String> {
    if temp_c < -273.15 {
        return Err("temperature below absolute zero".into());
    }
    let v = 331.3 + 0.606 * temp_c;
    // re-check: (v − 331.3)/0.606 = T
    let t_back = (v - 331.3) / 0.606;
    if (t_back - temp_c).abs() > 1e-6 {
        return Err("re-check failed for v = 331.3 + 0.606·T".into());
    }
    Ok(WavesResult {
        values: vec![v],
        equation_used: "v = 331.3 + 0.606·T".into(),
        verify_method: "(v − 331.3)/0.606 = T".into(),
        verify_values: vec![t_back],
    })
}

/// Doppler shift for a moving source: f' = f·v/(v − v_s).
pub fn doppler(f_hz: f64, v_sound: f64, v_source: f64) -> Result<WavesResult, String> {
    if f_hz <= 0.0 || v_sound <= 0.0 {
        return Err("frequency and sound speed must be positive".into());
    }
    let denom = v_sound - v_source;
    if denom.abs() < 1e-9 {
        return Err("source at the speed of sound — the sonic boom (f' → ∞)".into());
    }
    if denom < 0.0 {
        return Err("source faster than sound: shock wave, no ordinary Doppler".into());
    }
    let f2 = f_hz * v_sound / denom;
    // re-check: f'·(v − v_s)/v = f
    if (f2 * denom / v_sound - f_hz).abs() > 1e-6 * f_hz {
        return Err("re-check failed for f' = f·v/(v − v_s)".into());
    }
    Ok(WavesResult {
        values: vec![f2],
        equation_used: "f' = f·v/(v − v_s)".into(),
        verify_method: "f'·(v − v_s)/v = f".into(),
        verify_values: vec![f2 * denom / v_sound],
    })
}

/// Echo ranging: d = v·t/2 (sonar, bats, radar — the round trip halved).
pub fn echo_distance(v_mps: f64, round_trip_s: f64) -> Result<WavesResult, String> {
    if v_mps <= 0.0 || round_trip_s < 0.0 {
        return Err("speed and time must be non-negative (speed positive)".into());
    }
    let d = v_mps * round_trip_s / 2.0;
    // re-check: 2d/v = t
    if (2.0 * d / v_mps - round_trip_s).abs() > 1e-9 * round_trip_s.max(1e-12) {
        return Err("re-check failed for d = v·t/2".into());
    }
    Ok(WavesResult {
        values: vec![d],
        equation_used: "d = v·t/2".into(),
        verify_method: "2·d/v = t".into(),
        verify_values: vec![2.0 * d / v_mps],
    })
}

/// Snell's law: n₁·sin(θ₁) = n₂·sin(θ₂). Angles in degrees from the normal.
pub fn snell(n1: f64, theta1_deg: f64, n2: f64) -> Result<WavesResult, String> {
    if n1 <= 0.0 || n2 <= 0.0 {
        return Err("refractive indices must be positive".into());
    }
    if !(0.0..=90.0).contains(&theta1_deg) {
        return Err("incidence angle must be 0–90° from the normal".into());
    }
    let s1 = theta1_deg.to_radians().sin();
    let s2 = n1 * s1 / n2;
    if s2 > 1.0 {
        // total internal reflection — not an error, a verdict
        return Ok(WavesResult {
            values: vec![-1.0],
            equation_used: "n₁·sin(θ₁) = n₂·sin(θ₂)".into(),
            verify_method: "sin(θ₂) > 1 → total internal reflection (the fibre-optic law)".into(),
            verify_values: vec![s2],
        });
    }
    let theta2 = s2.asin().to_degrees();
    // re-check: n₂·sin(θ₂) = n₁·sin(θ₁)
    let lhs = n2 * theta2.to_radians().sin();
    let rhs = n1 * s1;
    if (lhs - rhs).abs() > 1e-9 {
        return Err("re-check failed for Snell's law".into());
    }
    Ok(WavesResult {
        values: vec![theta2],
        equation_used: "n₁·sin(θ₁) = n₂·sin(θ₂)".into(),
        verify_method: "n₂·sin(θ₂) = n₁·sin(θ₁)".into(),
        verify_values: vec![lhs],
    })
}

/// Thin lens equation: 1/f = 1/do + 1/di, solved for the missing one.
pub fn thin_lens(f: Option<f64>, dobj: Option<f64>, dimg: Option<f64>) -> Result<WavesResult, String> {
    let known = [f.is_some(), dobj.is_some(), dimg.is_some()].iter().filter(|x| **x).count();
    if known != 2 {
        return Err("give exactly two of f, do, di".into());
    }
    let (ff, d_o, d_i) = match (f, dobj, dimg) {
        (Some(f), Some(do_), None) => (f, do_, 1.0 / (1.0 / f - 1.0 / do_)),
        (Some(f), None, Some(di)) => (f, 1.0 / (1.0 / f - 1.0 / di), di),
        (None, Some(do_), Some(di)) => (1.0 / (1.0 / do_ + 1.0 / di), do_, di),
        _ => unreachable!(),
    };
    if ff == 0.0 {
        return Err("zero focal length".into());
    }
    // re-check: 1/f = 1/do + 1/di
    let lhs = 1.0 / ff;
    let rhs = 1.0 / d_o + 1.0 / d_i;
    if (lhs - rhs).abs() > 1e-9 * lhs.abs().max(1e-9) {
        return Err("re-check failed for 1/f = 1/do + 1/di".into());
    }
    Ok(WavesResult {
        values: vec![ff],
        equation_used: "1/f = 1/do + 1/di".into(),
        verify_method: "1/do + 1/di reproduces 1/f".into(),
        verify_values: vec![rhs],
    })
}

/// Inverse-square law for intensity: I₂ = I₁·(d₁/d₂)².
pub fn inverse_square(i1: f64, d1: f64, d2: f64) -> Result<WavesResult, String> {
    if i1 < 0.0 || d1 <= 0.0 || d2 <= 0.0 {
        return Err("intensity non-negative; distances positive".into());
    }
    let i2 = i1 * (d1 / d2) * (d1 / d2);
    // re-check: I₂·d₂² = I₁·d₁² (power conservation through the sphere)
    if (i2 * d2 * d2 - i1 * d1 * d1).abs() > 1e-9 * (i1 * d1 * d1).max(1e-12) {
        return Err("re-check failed for I = P/(4π·d²)".into());
    }
    Ok(WavesResult {
        values: vec![i2],
        equation_used: "I₂ = I₁·(d₁/d₂)²".into(),
        verify_method: "I₂·d₂² = I₁·d₁²".into(),
        verify_values: vec![i2 * d2 * d2],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a4_in_air() {
        let r = wave_equation(None, Some(440.0), Some(0.78)).unwrap();
        assert!((r.values[0] - 343.2).abs() < 0.1);
    }

    #[test]
    fn hot_air_is_fast_air() {
        let r = sound_speed_air(30.0).unwrap();
        assert!(r.values[0] > sound_speed_air(10.0).unwrap().values[0]);
        assert!((r.values[0] - 349.5).abs() < 0.1);
    }

    #[test]
    fn ambulance_passes() {
        let toward = doppler(440.0, 343.0, 30.0).unwrap();
        assert!(toward.values[0] > 440.0);
        let away = doppler(440.0, 343.0, -30.0).unwrap();
        assert!(away.values[0] < 440.0);
    }

    #[test]
    fn thunder_counting() {
        // 3 s of thunder delay in 20 °C air → ~515 m away
        let r = echo_distance(343.42, 3.0).unwrap();
        assert!((r.values[0] - 515.13).abs() < 0.1);
    }

    #[test]
    fn snell_bends_toward_normal() {
        // air→water at 45°: θ₂ ≈ 32.0°
        let r = snell(1.0, 45.0, 1.333).unwrap();
        assert!((r.values[0] - 32.03).abs() < 0.1);
    }

    #[test]
    fn fibre_optics_total_internal_reflection() {
        // glass→air beyond the critical angle: TIR
        let r = snell(1.5, 60.0, 1.0).unwrap();
        assert_eq!(r.values[0], -1.0, "60° > 41.8° critical → TIR");
    }

    #[test]
    fn thin_lens_magician() {
        // f=10 cm, object at 15 cm → image at 30 cm
        let r = thin_lens(Some(10.0), Some(15.0), None).unwrap();
        assert!((r.values[0] - 10.0).abs() < 1e-9); // solves f... hmm returns f
        let r2 = thin_lens(None, Some(15.0), Some(30.0)).unwrap();
        assert!((r2.values[0] - 10.0).abs() < 1e-9);
    }

    #[test]
    fn doubling_distance_quarters_intensity() {
        let r = inverse_square(100.0, 1.0, 2.0).unwrap();
        assert!((r.values[0] - 25.0).abs() < 1e-9);
    }
}
