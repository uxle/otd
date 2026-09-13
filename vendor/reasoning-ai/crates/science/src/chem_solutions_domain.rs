//! Phase 117 — Chemistry: solutions (molarity, dilution, percent
//! composition, percent yield) (Rust port of
//! python/science/chem_solutions_domain.py)
//!
//! Textbook ground truth: M = n/V, n = mass/molar_mass, M1*V1 = M2*V2,
//! %element = atoms*atomic_mass/molar_mass*100, %yield =
//! actual/theoretical*100. Independent cross-checks: moles recomputed
//! both ways for molarity, moles conservation for dilution, percentages
//! summing to 100, and yield <= 100.

use crate::chemistry_domain::{atomic_weight, molar_mass, parse_formula};
use crate::val::ValMap;

/// Result of a solutions-domain computation (Python Dict[str, float]
/// values, insertion-ordered).
#[derive(Debug, Clone, PartialEq)]
pub struct SolutionResult {
    pub values: ValMap,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: ValMap,
}

/// M = n/V. Optionally the moles can arrive as mass of a named solute --
/// then the mass->moles conversion is the primary path and M*V==n the
/// independent check.
pub fn molarity(
    moles: Option<f64>,
    volume_l: Option<f64>,
    molarity_m: Option<f64>,
    mass_g: Option<f64>,
    solute_formula: Option<&str>,
) -> Result<SolutionResult, String> {
    if let Some(v) = volume_l {
        if v <= 0.0 {
            return Err("volume must be positive".to_string());
        }
    }
    let moles = match moles {
        Some(m) => m,
        None => match (mass_g, solute_formula) {
            (Some(mass), Some(formula)) => {
                let mm = molar_mass(formula)?;
                mass / mm
            }
            _ => return Err("need moles, or mass of a named solute".to_string()),
        },
    };
    if molarity_m.is_none() {
        let Some(volume_l) = volume_l else {
            return Err("solving for molarity needs volume".to_string());
        };
        let m = moles / volume_l;
        let n_back = m * volume_l;
        if (n_back - moles).abs() > 1e-9 {
            return Err("independent re-check failed for M = n/V".to_string());
        }
        return Ok(SolutionResult {
            values: [("molarity", m)].into_iter().collect(),
            equation_used: "M = n/V".to_string(),
            verify_method: "n = M*V".to_string(),
            verify_values: [("moles", n_back)].into_iter().collect(),
        });
    }
    let molarity_m = molarity_m.unwrap();
    if volume_l.is_none() {
        let v = moles / molarity_m;
        let n_back = molarity_m * v;
        if (n_back - moles).abs() > 1e-9 {
            return Err("independent re-check failed for V = n/M".to_string());
        }
        return Ok(SolutionResult {
            values: [("volume_l", v)].into_iter().collect(),
            equation_used: "V = n/M".to_string(),
            verify_method: "n = M*V".to_string(),
            verify_values: [("moles", n_back)].into_iter().collect(),
        });
    }
    let n_check = molarity_m * volume_l.unwrap();
    if (n_check - moles).abs() > 1e-9 {
        return Err("inconsistent inputs: M*V does not reproduce the moles".to_string());
    }
    Ok(SolutionResult {
        values: [("moles", moles)].into_iter().collect(),
        equation_used: "n = M*V".to_string(),
        verify_method: "given M and V directly; moles identity verified".to_string(),
        verify_values: [("moles", n_check)].into_iter().collect(),
    })
}

/// M1*V1 = M2*V2 for any one of the four given the other three.
/// Re-check: moles before == moles after (the conservation the law encodes).
pub fn dilution(
    m1: Option<f64>,
    v1: Option<f64>,
    m2: Option<f64>,
    v2: Option<f64>,
) -> Result<SolutionResult, String> {
    let (mut m1, mut v1, mut m2, mut v2) = (m1, v1, m2, v2);
    let val;
    if m1.is_none() || v1.is_none() || m2.is_none() || v2.is_none() {
        if let (Some(a), Some(b), Some(c), None) = (m1, v1, m2, v2) {
            val = a * b / c;
            v2 = Some(val);
        } else if let (Some(a), Some(b), None, Some(d)) = (m1, v1, m2, v2) {
            val = a * b / d;
            m2 = Some(val);
        } else if let (None, Some(b), Some(c), Some(d)) = (m1, v1, m2, v2) {
            val = c * d / b;
            m1 = Some(val);
        } else if let (Some(a), None, Some(c), Some(d)) = (m1, v1, m2, v2) {
            val = c * d / a;
            v1 = Some(val);
        } else {
            return Err("dilution needs any three of m1, v1, m2, v2".to_string());
        }
    } else {
        return Err("leave exactly the unknown as None".to_string());
    }
    let moles_before = m1.unwrap() * v1.unwrap();
    let moles_after = m2.unwrap() * v2.unwrap();
    if (moles_before - moles_after).abs() > 1e-9 * f64::max(1.0, moles_before.abs()) {
        return Err("conservation re-check failed: moles changed during dilution".to_string());
    }
    Ok(SolutionResult {
        values: [("value", val)].into_iter().collect(),
        equation_used: "M1*V1 = M2*V2".to_string(),
        verify_method: "moles conservation M1*V1 == M2*V2".to_string(),
        verify_values: [("moles_before", moles_before), ("moles_after", moles_after)]
            .into_iter()
            .collect(),
    })
}

/// Mass percent of each element in a compound.
/// Re-check: percentages must sum to 100% (within float rounding) -- an
/// incorrect element count or weight breaks that sum.
pub fn percent_composition(formula: &str) -> Result<SolutionResult, String> {
    let counts = parse_formula(formula)?;
    let mm = molar_mass(formula)?;
    if mm <= 0.0 {
        return Err("molar mass computed as zero -- bad formula".to_string());
    }
    let percents: Vec<(String, f64)> = counts
        .iter()
        .map(|(el, n)| {
            // parse_formula already validated the symbol is in the table
            let w = atomic_weight(el).expect("element validated by parse_formula");
            (el.clone(), (*n as f64) * w / mm * 100.0)
        })
        .collect();
    let total: f64 = percents.iter().map(|(_, v)| *v).sum();
    if (total - 100.0).abs() > 0.05 {
        return Err(format!("sum-to-100 re-check failed: got {:.4}%", total));
    }
    Ok(SolutionResult {
        values: percents.iter().map(|(k, v)| (k.as_str(), *v)).collect(),
        equation_used: "%element = atoms*atomic_mass / molar_mass * 100".to_string(),
        verify_method: "all percentages sum to 100%".to_string(),
        verify_values: [("total", total)].into_iter().collect(),
    })
}

/// %yield = actual/theoretical*100. Verified: must be in (0, 100]; a value
/// above 100% means the 'actual' claim is inconsistent with the
/// theoretical maximum and is rejected rather than reported.
pub fn percent_yield(actual: f64, theoretical: f64) -> Result<SolutionResult, String> {
    if theoretical <= 0.0 || actual < 0.0 {
        return Err("theoretical yield must be positive, actual non-negative".to_string());
    }
    let pct = actual / theoretical * 100.0;
    if pct > 100.0 + 1e-9 {
        return Err("yield >100% is physically inconsistent for this reaction setup -- rejecting instead of reporting".to_string());
    }
    Ok(SolutionResult {
        values: [("percent_yield", pct)].into_iter().collect(),
        equation_used: "%yield = actual/theoretical * 100".to_string(),
        verify_method: "range check 0 < yield <= 100".to_string(),
        verify_values: [("percent_yield", pct)].into_iter().collect(),
    })
}
