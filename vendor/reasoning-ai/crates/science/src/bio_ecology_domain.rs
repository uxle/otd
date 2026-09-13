//! Phase 122 — Biology: population ecology (exponential and logistic growth,
//! trophic energy transfer) (Rust port of python/science/bio_ecology_domain.py).
//!
//! Formulas (textbook ground truth):
//!     Exponential (discrete):  N_t = N0 * (1 + r)^t
//!     Exponential (continuous): N_t = N0 * e^(r*t)
//!     Logistic:  N_t = K / (1 + A * e^(-r*t)),  A = (K - N0)/N0
//!     Doubling time (continuous): t2 = ln(2) / r
//!     10% rule: energy at trophic level n = producer energy * 0.1^(n-1)
//!
//! Independent cross-checks: equivalent-rate reformulation for exponential
//! growth (discrete vs continuous), logistic boundary checks, doubling
//! re-grown to exactly 2*N0, and the 10% rule walked level-by-level vs the
//! closed form.

use std::collections::HashMap;

#[derive(Debug, Clone, PartialEq)]
pub struct EcologyResult {
    pub values: HashMap<String, f64>,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: HashMap<String, f64>,
}

/// `math.log1p` (ln(1+x), accurate for tiny x): Rust std has no log1p, so
/// use the exact identity log1p(x) = 2*atanh(x/(2+x)) with a series
/// (|x| <= 0.5 -> |t| <= 1/3, 26 terms give < 1e-25 truncation error).
/// Raises "math domain error" like Python for x <= -1.
fn _log1p(x: f64) -> Result<f64, String> {
    if x <= -1.0 {
        return Err("math domain error".to_string());
    }
    if x > 0.5 || x < -0.5 {
        return Ok((1.0 + x).ln());
    }
    let t = x / (2.0 + x);
    let t2 = t * t;
    let mut term = t;
    let mut sum = t;
    for n in 1..=26 {
        term *= t2;
        sum += term / (2 * n + 1) as f64;
    }
    Ok(2.0 * sum)
}

/// N_t = N0*(1+r)^t (discrete generations) or N0*e^(r*t) (continuous).
/// Cross-check: the other formulation with the equivalent rate
/// (r_cont = ln(1+r_disc), or r_disc = e^(r_cont)-1) must give the same
/// population -- a genuinely independent formula that must agree.
/// (`continuous` defaults to False in Python.)
pub fn exponential_growth(
    n0: f64,
    rate: f64,
    time: f64,
    continuous: bool,
) -> Result<EcologyResult, String> {
    if n0 <= 0.0 {
        return Err("initial population must be positive".to_string());
    }
    let (n_t, n_check) = if continuous {
        let n_t = n0 * (rate * time).exp();
        let r_disc = rate.exp() - 1.0;
        let n_check = n0 * (1.0 + r_disc).powf(time);
        (n_t, n_check)
    } else {
        let n_t = n0 * (1.0 + rate).powf(time);
        let r_cont = _log1p(rate)?;
        let n_check = n0 * (r_cont * time).exp();
        (n_t, n_check)
    };
    if (n_check - n_t).abs() > 1e-6 * f64::max(1.0, n_t.abs()) {
        return Err("discrete/continuous cross-check failed".to_string());
    }
    let mut values = HashMap::new();
    values.insert("population".to_string(), n_t);
    let mut verify_values = HashMap::new();
    verify_values.insert("population".to_string(), n_check);
    Ok(EcologyResult {
        values,
        equation_used: if !continuous {
            "N0*(1+r)^t".to_string()
        } else {
            "N0*e^(r*t)".to_string()
        },
        verify_method: "equivalent-rate reformulation".to_string(),
        verify_values,
    })
}

/// t2 = ln(2)/r (continuous) or the discrete solve of (1+r)^t = 2.
/// Cross-check: one doubling period of growth must yield exactly 2*N0.
/// (`continuous` defaults to True in Python.)
pub fn doubling_time(rate: f64, continuous: bool) -> Result<EcologyResult, String> {
    if rate <= 0.0 {
        return Err("growth rate must be positive for a doubling time".to_string());
    }
    let (t2, grown) = if continuous {
        let t2 = std::f64::consts::LN_2 / rate;
        // verify: N0 * e^(r*t2) == 2*N0 for N0=1000 (scale-free check)
        let n0 = 1000.0;
        let grown = n0 * (rate * t2).exp();
        (t2, grown)
    } else {
        let t2 = std::f64::consts::LN_2 / _log1p(rate)?;
        let n0 = 1000.0;
        let grown = n0 * (1.0 + rate).powf(t2);
        (t2, grown)
    };
    let n0 = 1000.0;
    if (grown - 2.0 * n0).abs() > 1e-6 * n0 {
        return Err(
            "doubling re-check failed: population did not exactly double".to_string()
        );
    }
    let mut values = HashMap::new();
    values.insert("doubling_time".to_string(), t2);
    let mut verify_values = HashMap::new();
    verify_values.insert("population_after".to_string(), grown);
    Ok(EcologyResult {
        values,
        equation_used: "t2 = ln(2)/r".to_string(),
        verify_method: "grow one t2 and require exactly 2*N0".to_string(),
        verify_values,
    })
}

/// N_t = K / (1 + A*e^(-r*t)), A = (K-N0)/N0.
/// Boundary verification: N(0) must reproduce N0, and 0 < N_t < K when
/// N0 < K (population approaches but never exceeds carrying capacity).
pub fn logistic_growth(
    n0: f64,
    carrying_capacity: f64,
    rate: f64,
    time: f64,
) -> Result<EcologyResult, String> {
    if n0 <= 0.0 || carrying_capacity <= 0.0 {
        return Err("population and carrying capacity must be positive".to_string());
    }
    if n0 >= carrying_capacity {
        return Err(
            "this model assumes N0 < K (population below carrying capacity)".to_string()
        );
    }
    let a = (carrying_capacity - n0) / n0;
    let n_t = carrying_capacity / (1.0 + a * (-rate * time).exp());
    // boundary check 1: N(0) == N0
    let n_zero = carrying_capacity / (1.0 + a);
    if (n_zero - n0).abs() > 1e-9 * n0 {
        return Err("boundary check failed: model does not reproduce N0 at t=0".to_string());
    }
    // boundary check 2: stays strictly below K
    if !(n0 <= n_t && n_t < carrying_capacity) {
        return Err("boundary check failed: population left [N0, K) range".to_string());
    }
    let mut values = HashMap::new();
    values.insert("population".to_string(), n_t);
    values.insert("carrying_capacity".to_string(), carrying_capacity);
    let mut verify_values = HashMap::new();
    verify_values.insert("n_at_zero".to_string(), n_zero);
    Ok(EcologyResult {
        values,
        equation_used: "N = K / (1 + A*e^(-r*t)), A=(K-N0)/N0".to_string(),
        verify_method: "boundary checks N(0)==N0 and N_t < K".to_string(),
        verify_values,
    })
}

/// Energy reaching trophic level n under the 10% rule:
/// level 1 = producers, level 2 = primary consumers (10%), etc.
/// Verified by walking level-by-level (each step divides the PREVIOUS
/// level by 10) instead of using the closed-form 0.1^(n-1) formula --
/// then the two must agree.
pub fn energy_transfer_10_percent(
    producer_energy: f64,
    trophic_level: i32,
) -> Result<EcologyResult, String> {
    if producer_energy <= 0.0 {
        return Err("producer energy must be positive".to_string());
    }
    if trophic_level < 1 {
        return Err("trophic levels start at 1 (producers)".to_string());
    }
    let mut walked = producer_energy;
    for _ in 0..(trophic_level - 1) {
        walked /= 10.0;
    }
    let closed_form = producer_energy * 0.1f64.powf((trophic_level - 1) as f64);
    if (walked - closed_form).abs() > 1e-12 * f64::max(1.0, closed_form.abs()) {
        return Err("walked vs closed-form energy transfer disagree".to_string());
    }
    let mut values = HashMap::new();
    values.insert("energy".to_string(), closed_form);
    let mut verify_values = HashMap::new();
    verify_values.insert("energy".to_string(), walked);
    Ok(EcologyResult {
        values,
        equation_used: "E_n = E1 * 0.1^(n-1)".to_string(),
        verify_method: "level-by-level walk (each level = previous / 10)".to_string(),
        verify_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn log1p_matches_python() {
        // pinned against Python: math.log1p(0.1) = 0.09531017980432487
        assert!((_log1p(0.1).unwrap() - 0.09531017980432487).abs() < 1e-16);
        assert!(_log1p(0.0).unwrap() == 0.0);
        assert!(_log1p(-1.0).unwrap_err() == "math domain error");
        assert!(_log1p(-2.0).unwrap_err() == "math domain error");
    }
}
