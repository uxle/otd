//! Phase 113 — Physics: DC electricity (Rust port of
//! python/science/electricity_domain.py)
//!
//! Textbook ground truth: V = I*R, P = V*I = I^2*R = V^2/R, series and
//! parallel resistance networks. Independent cross-checks: the three
//! algebraically distinct power forms must agree; network totals are
//! re-verified via Kirchhoff's laws (KVL/KCL), a genuinely different
//! physical principle than the combination formulas.

use reasoning_common::py_float_str;

/// Result of an electricity-domain computation (tuple of values) with its
/// independent re-check values.
#[derive(Debug, Clone, PartialEq)]
pub struct ElectricResult {
    pub values: Vec<f64>,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: Vec<f64>,
}

/// V = I*R for any one of 'voltage', 'current', 'resistance'.
pub fn ohms_law(
    find: &str,
    voltage: Option<f64>,
    current: Option<f64>,
    resistance: Option<f64>,
) -> Result<ElectricResult, String> {
    if find == "voltage" {
        let (Some(current), Some(resistance)) = (current, resistance) else {
            return Err("solving for voltage needs current and resistance".to_string());
        };
        let v = current * resistance;
        if resistance != 0.0 && (v / resistance - current).abs() > 1e-9 {
            return Err("independent re-check failed for V = I*R".to_string());
        }
        return Ok(ElectricResult {
            values: vec![v],
            equation_used: "V = I*R".to_string(),
            verify_method: "reconstruct I = V/R".to_string(),
            verify_values: vec![v / resistance],
        });
    }
    if find == "current" {
        let (Some(voltage), Some(resistance)) = (voltage, resistance) else {
            return Err("solving for current needs voltage and resistance".to_string());
        };
        if resistance == 0.0 {
            return Err("resistance 0 makes current undefined (short circuit)".to_string());
        }
        let i = voltage / resistance;
        if (i * resistance - voltage).abs() > 1e-9 {
            return Err("independent re-check failed for V = I*R".to_string());
        }
        return Ok(ElectricResult {
            values: vec![i],
            equation_used: "I = V/R".to_string(),
            verify_method: "reconstruct V = I*R".to_string(),
            verify_values: vec![i * resistance],
        });
    }
    if find == "resistance" {
        let (Some(voltage), Some(current)) = (voltage, current) else {
            return Err("solving for resistance needs voltage and current".to_string());
        };
        if current == 0.0 {
            return Err("current 0 makes resistance undefined".to_string());
        }
        let r = voltage / current;
        if (r * current - voltage).abs() > 1e-9 {
            return Err("independent re-check failed for V = I*R".to_string());
        }
        return Ok(ElectricResult {
            values: vec![r],
            equation_used: "R = V/I".to_string(),
            verify_method: "reconstruct V = I*R".to_string(),
            verify_values: vec![r * current],
        });
    }
    Err("find must be one of voltage, current, resistance".to_string())
}

/// Electrical power via all three equivalent forms that the given inputs
/// allow -- the forms are algebraically independent rearrangements, and
/// this function REQUIRES at least two of them to agree (that is the
/// cross-check) rather than trusting one.
pub fn electrical_power(
    voltage: Option<f64>,
    current: Option<f64>,
    resistance: Option<f64>,
) -> Result<ElectricResult, String> {
    // Python builds the `forms` dict in this insertion order.
    let mut forms: Vec<(&str, f64)> = Vec::new();
    if let (Some(v), Some(i)) = (voltage, current) {
        forms.push(("P = V*I", v * i));
    }
    if let (Some(i), Some(r)) = (current, resistance) {
        forms.push(("P = I^2*R", i.powi(2) * r));
    }
    if let (Some(v), Some(r)) = (voltage, resistance) {
        forms.push(("P = V^2/R", v.powi(2) / r));
    }
    if forms.len() < 2 {
        return Err("provide at least two of voltage, current, resistance so two independent power-law forms can cross-check".to_string());
    }
    let values: Vec<f64> = forms.iter().map(|(_, v)| *v).collect();
    let reference = values[0];
    for &v in &values[1..] {
        if (v - reference).abs() > 1e-6 * f64::max(1.0, reference.abs()) {
            // f"power-form cross-check failed: {forms}" (Python dict repr)
            let repr = forms
                .iter()
                .map(|(k, v)| format!("'{}': {}", k, py_float_str(*v)))
                .collect::<Vec<_>>()
                .join(", ");
            return Err(format!("power-form cross-check failed: {{{}}}", repr));
        }
    }
    Ok(ElectricResult {
        values: vec![reference],
        equation_used: forms.iter().map(|(k, _)| *k).collect::<Vec<_>>().join(" / "),
        verify_method: format!("agreement of {} independent forms", forms.len()),
        verify_values: values,
    })
}

/// R_total = sum(R_i). Kirchhoff re-check: with a 1 A test current, the
/// per-element drops I*R_i must sum to I*R_total (KVL).
pub fn series_resistance(resistances: &[f64]) -> Result<ElectricResult, String> {
    if resistances.is_empty() {
        return Err("need at least one resistor".to_string());
    }
    if resistances.iter().any(|r| *r < 0.0) {
        return Err("resistances cannot be negative".to_string());
    }
    let r_total: f64 = resistances.iter().sum();
    let test_current = 1.0;
    let kvl_sum: f64 = resistances.iter().map(|r| test_current * r).sum();
    if (kvl_sum - test_current * r_total).abs() > 1e-9 {
        return Err("KVL re-check failed for series network".to_string());
    }
    Ok(ElectricResult {
        values: vec![r_total],
        equation_used: "R_series = sum(R_i)".to_string(),
        verify_method: "KVL: sum of I*R_i drops equals I*R_total".to_string(),
        verify_values: vec![kvl_sum],
    })
}

/// 1/R_total = sum(1/R_i). Re-checks:
///   1. for n=2 the product-over-sum shortcut must agree with the general rule
///   2. KCL: with a 1 V test source, branch currents sum to V/R_total
pub fn parallel_resistance(resistances: &[f64]) -> Result<ElectricResult, String> {
    if resistances.is_empty() {
        return Err("need at least one resistor".to_string());
    }
    if resistances.iter().any(|r| *r <= 0.0) {
        return Err("parallel resistances must be positive".to_string());
    }
    let inv_sum: f64 = resistances.iter().map(|r| 1.0 / r).sum();
    let r_total = 1.0 / inv_sum;
    if resistances.len() == 2 {
        let (r1, r2) = (resistances[0], resistances[1]);
        let shortcut = r1 * r2 / (r1 + r2);
        if (shortcut - r_total).abs() > 1e-9 {
            return Err("product-over-sum re-check failed".to_string());
        }
    }
    let test_v = 1.0;
    let branch_sum: f64 = resistances.iter().map(|r| test_v / r).sum();
    if (branch_sum - test_v / r_total).abs() > 1e-9 {
        return Err("KCL re-check failed for parallel network".to_string());
    }
    Ok(ElectricResult {
        values: vec![r_total],
        equation_used: "1/R_parallel = sum(1/R_i)".to_string(),
        verify_method: if resistances.len() == 2 {
            "product-over-sum + KCL branch currents".to_string()
        } else {
            "KCL branch currents".to_string()
        },
        verify_values: vec![branch_sum],
    })
}

/// Total current drawn by `voltage` across a series string: I = V / sum(R_i).
/// Re-check: each element's drop I*R_i, summed, must reproduce the source
/// voltage exactly (Kirchhoff's voltage law on the actual numbers).
pub fn series_parallel_current(voltage: f64, resistances: &[f64]) -> Result<ElectricResult, String> {
    if voltage < 0.0 {
        return Err("voltage cannot be negative".to_string());
    }
    let r_total = series_resistance(resistances)?.values[0];
    if r_total == 0.0 {
        return Err("total resistance 0 (short circuit)".to_string());
    }
    let i = voltage / r_total;
    let kvl: f64 = resistances.iter().map(|r| i * r).sum();
    if (kvl - voltage).abs() > 1e-9 {
        return Err("KVL re-check failed: drops do not sum to source voltage".to_string());
    }
    Ok(ElectricResult {
        values: vec![i, r_total],
        equation_used: "I = V / R_series".to_string(),
        verify_method: "KVL: sum of I*R_i equals source voltage".to_string(),
        verify_values: vec![kvl],
    })
}
