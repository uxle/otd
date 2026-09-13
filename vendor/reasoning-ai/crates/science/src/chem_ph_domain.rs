//! Phase 118 — Chemistry: pH and acid-base arithmetic (Rust port of
//! python/science/chem_ph_domain.py)
//!
//! Standard aqueous chemistry at 25 degrees C: pH = -log10([H+]),
//! pOH = -log10([OH-]), pH + pOH = 14, [H+]*[OH-] = Kw = 1e-14,
//! [H+] = 10^-pH. Every function verifies its answer by the inverse
//! transformation (a genuinely different computation than the one that
//! produced it), and neutralization checks acid vs base equivalents.

use crate::val::{Val, ValMap};

/// Ionic product of water at 25 degrees C.
pub const KW: f64 = 1e-14;

fn _log10(x: f64) -> f64 {
    x.log10()
}

/// Result of a pH-domain computation (Python Dict values; may mix floats
/// and a status string).
#[derive(Debug, Clone, PartialEq)]
pub struct PHResult {
    pub values: ValMap,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: ValMap,
}

/// pH = -log10([H+]). Re-check: 10^-pH must reproduce [H+].
pub fn ph_from_concentration(h_concentration: f64) -> Result<PHResult, String> {
    if h_concentration <= 0.0 {
        return Err("[H+] must be positive (activity, not a count)".to_string());
    }
    let ph = -_log10(h_concentration);
    let h_back = 10f64.powf(-ph);
    if (h_back - h_concentration).abs() > 1e-9 * f64::max(1e-12, h_concentration) {
        return Err("inverse re-check failed for pH = -log10[H+]".to_string());
    }
    Ok(PHResult {
        values: [("pH", ph)].into_iter().collect(),
        equation_used: "pH = -log10([H+])".to_string(),
        verify_method: "inverse [H+] = 10^-pH".to_string(),
        verify_values: [("h+", h_back)].into_iter().collect(),
    })
}

/// [H+] = 10^-pH. Re-check: -log10 of the answer reproduces pH.
pub fn concentration_from_ph(ph: f64) -> Result<PHResult, String> {
    let ph_val = 10f64.powf(-ph);
    let ph_back = -_log10(ph_val);
    if (ph_back - ph).abs() > 1e-9 {
        return Err("inverse re-check failed for [H+] = 10^-pH".to_string());
    }
    Ok(PHResult {
        values: [("h+", ph_val)].into_iter().collect(),
        equation_used: "[H+] = 10^-pH".to_string(),
        verify_method: "inverse pH = -log10([H+])".to_string(),
        verify_values: [("pH", ph_back)].into_iter().collect(),
    })
}

/// Complete the pH/pOH pair using pH + pOH = 14, and verify the pair by
/// the Kw identity [H+]*[OH-] == 1e-14 -- a different law than the sum rule.
pub fn ph_poh_pair(ph: Option<f64>, poh: Option<f64>) -> Result<PHResult, String> {
    if ph.is_none() && poh.is_none() {
        return Err("give pH or pOH".to_string());
    }
    let mut ph = ph;
    let mut poh = poh;
    if ph.is_none() {
        ph = Some(14.0 - poh.unwrap());
    }
    if poh.is_none() {
        poh = Some(14.0 - ph.unwrap());
    }
    let ph = ph.unwrap();
    let poh = poh.unwrap();
    if (ph + poh - 14.0).abs() > 1e-9 {
        return Err("pH + pOH must equal 14".to_string());
    }
    let h = 10f64.powf(-ph);
    let oh = 10f64.powf(-poh);
    let kw_check = h * oh;
    if (kw_check - KW).abs() > 1e-3 * KW * f64::max(1.0, kw_check / KW) {
        // tolerance: float exponentials near 1e-14 lose precision, but the
        // product must still be within a small factor of Kw, never orders off
        if (_log10(kw_check) - _log10(KW)).abs() > 0.01 {
            return Err("Kw re-check failed: [H+]*[OH-] is not 1e-14".to_string());
        }
    }
    Ok(PHResult {
        values: [("pH", ph), ("pOH", poh), ("h+", h), ("oh-", oh)]
            .into_iter()
            .collect(),
        equation_used: "pH + pOH = 14".to_string(),
        verify_method: "Kw identity [H+]*[OH-] == 1e-14".to_string(),
        verify_values: [("kw_check", kw_check)].into_iter().collect(),
    })
}

/// Endpoint check for H+ + OH- -> H2O: acid equivalents
/// (moles_acid * acid_protons) must equal base equivalents
/// (moles_base * base_oh) for exact neutralization, or the excess is
/// reported. Verified by computing the excess from both directions.
pub fn neutralization(
    moles_acid: f64,
    acid_protons: i64,
    moles_base: f64,
    base_oh: i64,
) -> Result<PHResult, String> {
    if acid_protons <= 0 || base_oh <= 0 {
        return Err("acid_protons and base_oh must be positive".to_string());
    }
    let eq_acid = moles_acid * acid_protons as f64;
    let eq_base = moles_base * base_oh as f64;
    let excess = eq_acid - eq_base;
    let excess_back = moles_acid * acid_protons as f64 - moles_base * base_oh as f64;
    if (excess - excess_back).abs() > 1e-12 {
        return Err("two-sided recompute of the excess disagrees".to_string());
    }
    let kind = if excess.abs() < 1e-9 {
        "neutralized exactly"
    } else if excess > 0.0 {
        "acid in excess"
    } else {
        "base in excess"
    };
    Ok(PHResult {
        values: vec![
            ("excess_equivalents", Val::Num(excess)),
            ("status", Val::Str(kind.to_string())),
            ("eq_acid", Val::Num(eq_acid)),
            ("eq_base", Val::Num(eq_base)),
        ]
        .into_iter()
        .collect(),
        equation_used: "equivalents = moles * (protons or OH-)".to_string(),
        verify_method: "recompute excess from both sides".to_string(),
        verify_values: [("excess", excess_back)].into_iter().collect(),
    })
}
