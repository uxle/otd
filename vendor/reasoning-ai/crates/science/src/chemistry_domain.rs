//! Phase 103 — Chemistry: stoichiometry and molar mass (Rust port of
//! python/science/chemistry_domain.py)
//!
//! Ground truth is a fixed table of standard atomic weights (IUPAC,
//! rounded to 2 decimal places) plus straightforward mole-ratio
//! arithmetic from a balanced equation. Formula parsing is a small,
//! explicit recursive-descent parser (handles nesting like Ca(OH)2),
//! tested against hand-computed molar masses for common compounds.

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;

use reasoning_common::py_round;

/// Standard atomic weights, IUPAC 2021 abridged values, rounded to 2dp
/// (the Python module-level `ATOMIC_WEIGHTS` dict). Covers what
/// intro-level stoichiometry problems actually use.
pub fn atomic_weights() -> &'static HashMap<&'static str, f64> {
    static W: OnceLock<HashMap<&'static str, f64>> = OnceLock::new();
    W.get_or_init(|| {
        let entries = [
            ("H", 1.01), ("He", 4.00), ("Li", 6.94), ("Be", 9.01), ("B", 10.81), ("C", 12.01),
            ("N", 14.01), ("O", 16.00), ("F", 19.00), ("Ne", 20.18), ("Na", 22.99), ("Mg", 24.31),
            ("Al", 26.98), ("Si", 28.09), ("P", 30.97), ("S", 32.07), ("Cl", 35.45), ("Ar", 39.95),
            ("K", 39.10), ("Ca", 40.08), ("Fe", 55.85), ("Cu", 63.55), ("Zn", 65.38), ("Ag", 107.87),
            ("I", 126.90), ("Ba", 137.33), ("Au", 196.97), ("Hg", 200.59), ("Pb", 207.2),
        ];
        let mut m = HashMap::with_capacity(entries.len());
        for (k, v) in entries {
            m.insert(k, v);
        }
        m
    })
}

/// Single-element lookup (Python: `ATOMIC_WEIGHTS[el]`).
pub fn atomic_weight(symbol: &str) -> Option<f64> {
    atomic_weights().get(symbol).copied()
}

// re.findall(r"[A-Z][a-z]?|\d+|\(|\)", formula)
fn tokenize_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"[A-Z][a-z]?|\d+|\(|\)").unwrap())
}

// re.fullmatch(r"[A-Z][a-z]?", tok) -- anchored for the regex crate.
fn element_fullmatch_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[A-Z][a-z]?$").unwrap())
}

fn _tokenize_formula(formula: &str) -> Vec<String> {
    tokenize_re()
        .find_iter(formula)
        .map(|m| m.as_str().to_string())
        .collect()
}

// Python str.isdigit() for the tokens the tokenizer can produce.
fn is_digits(s: &str) -> bool {
    !s.is_empty() && s.chars().all(|c| c.is_ascii_digit())
}

fn add_count(counts: &mut Vec<(String, i64)>, el: &str, n: i64) {
    if let Some(slot) = counts.iter_mut().find(|(k, _)| k == el) {
        slot.1 += n;
    } else {
        counts.push((el.to_string(), n));
    }
}

/// Count of `el` in parsed counts (Python: `counts.get(el, 0)`).
pub fn count_of(counts: &[(String, i64)], el: &str) -> i64 {
    counts
        .iter()
        .find(|(k, _)| k == el)
        .map(|(_, v)| *v)
        .unwrap_or(0)
}

/// Recursive-descent parse of a chemical formula into element counts
/// (insertion-ordered like the Python dict), handling parenthesized
/// groups with multipliers (e.g. Ca(OH)2 -> [(Ca,1), (O,2), (H,2)]).
/// Err on anything unparsable or an unrecognized element symbol, rather
/// than silently returning a partial/wrong count.
pub fn parse_formula(formula: &str) -> Result<Vec<(String, i64)>, String> {
    let tokens = _tokenize_formula(formula);
    if tokens.concat() != formula.replace(' ', "") {
        return Err(format!("could not fully tokenize formula: '{}'", formula));
    }
    let mut pos = 0usize;
    let counts = parse_group(&tokens, &mut pos, formula)?;
    if pos != tokens.len() {
        return Err(format!("trailing unparsed content in '{}'", formula));
    }
    Ok(counts)
}

fn parse_group(
    tokens: &[String],
    pos: &mut usize,
    formula: &str,
) -> Result<Vec<(String, i64)>, String> {
    let mut counts: Vec<(String, i64)> = Vec::new();
    while *pos < tokens.len() && tokens[*pos] != ")" {
        let tok = tokens[*pos].clone();
        if tok == "(" {
            *pos += 1;
            let inner = parse_group(tokens, pos, formula)?;
            if *pos >= tokens.len() || tokens[*pos] != ")" {
                return Err(format!("unbalanced parentheses in '{}'", formula));
            }
            *pos += 1;
            let mut mult: i64 = 1;
            if *pos < tokens.len() && is_digits(&tokens[*pos]) {
                // int(tok); absurdly long digit runs saturate instead of panicking
                mult = tokens[*pos].parse::<i64>().unwrap_or(i64::MAX);
                *pos += 1;
            }
            for (el, c) in inner {
                add_count(&mut counts, &el, c * mult);
            }
        } else if element_fullmatch_re().is_match(&tok) {
            if !atomic_weights().contains_key(tok.as_str()) {
                return Err(format!("unrecognized element symbol: '{}'", tok));
            }
            *pos += 1;
            let mut mult: i64 = 1;
            if *pos < tokens.len() && is_digits(&tokens[*pos]) {
                mult = tokens[*pos].parse::<i64>().unwrap_or(i64::MAX);
                *pos += 1;
            }
            add_count(&mut counts, &tok, mult);
        } else {
            return Err(format!("unexpected token '{}' in '{}'", tok, formula));
        }
    }
    Ok(counts)
}

pub fn molar_mass(formula: &str) -> Result<f64, String> {
    let counts = parse_formula(formula)?;
    let mut total = 0.0;
    for (el, n) in &counts {
        // parse_formula already validated the symbol is in the table
        let w = atomic_weight(el).expect("element validated by parse_formula");
        total += w * *n as f64;
    }
    Ok(py_round(total, 2))
}

/// Result of a stoichiometry computation.
#[derive(Debug, Clone, PartialEq)]
pub struct StoichResult {
    pub moles_product: f64,
    pub mass_product: f64,
}

/// Given moles of a reactant and the balanced-equation coefficients for
/// that reactant and a product, compute moles and mass of product via the
/// mole-ratio method: moles_product = moles_reactant * (product_coeff /
/// reactant_coeff).
pub fn stoichiometry_moles(
    moles_reactant: f64,
    reactant_coeff: i64,
    product_coeff: i64,
    product_formula: &str,
) -> Result<StoichResult, String> {
    if reactant_coeff <= 0 || product_coeff <= 0 {
        return Err("coefficients must be positive integers".to_string());
    }
    let moles_product = moles_reactant * (product_coeff as f64 / reactant_coeff as f64);
    let mass_product = moles_product * molar_mass(product_formula)?;
    Ok(StoichResult {
        moles_product,
        mass_product: py_round(mass_product, 4),
    })
}
