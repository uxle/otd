//! Phase 043 — Unit-consistency verifier (design doc section 10)
//! (Rust port of python/verifier/units.py).
//!
//! A real gap the numeric/symbolic verifiers can't catch: "5 meters + 3
//! seconds" is numerically "5 + 3 = 8" but dimensionally nonsense. This
//! tracks units through basic arithmetic (a small dimensional-analysis
//! engine: multiplication/division combine units, addition/subtraction
//! require matching units) and flags mismatches the other verifiers are
//! blind to.
//!
//! Python's `Quantity` is renamed `UnitQuantity` to avoid clashing with
//! other quantity types. Python's `UnitMismatchError` / `ValueError` map
//! to `Result<_, String>` with the same messages.

use reasoning_common::py_float_str;
use std::collections::HashMap;
use std::fmt;

/// e.g. `[(m,1)]` for meters, `[(m,1),(s,-1)]` for m/s — sorted by unit
/// name with zero-exponent entries removed (Python: tuple of pairs).
pub type Dimension = Vec<(String, i32)>;

fn normalize(dim: &HashMap<String, i32>) -> Dimension {
    let mut out: Vec<(String, i32)> = dim
        .iter()
        .filter(|(_, v)| **v != 0)
        .map(|(k, v)| (k.clone(), *v))
        .collect();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out
}

#[derive(Debug, Clone, PartialEq)]
pub struct UnitQuantity {
    pub value: f64,
    pub unit_str: String,
    pub dimension: Dimension,
}

impl fmt::Display for UnitQuantity {
    /// Python `repr`: `f"{self.value} {self.unit_str}"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} {}", py_float_str(self.value), self.unit_str)
    }
}

/// `quantity(5, "m")` — only the base units m, s, kg are supported.
/// Unknown units raise (Python `ValueError` → `Err(String)`).
pub fn quantity(value: f64, unit: &str) -> Result<UnitQuantity, String> {
    let dim_map: HashMap<String, i32> = match unit {
        "m" => [("m".to_string(), 1)].into_iter().collect(),
        "s" => [("s".to_string(), 1)].into_iter().collect(),
        "kg" => [("kg".to_string(), 1)].into_iter().collect(),
        _ => {
            return Err(format!(
                "unknown base unit '{}' (supported: ['m', 's', 'kg'])",
                unit
            ))
        }
    };
    Ok(UnitQuantity {
        value,
        unit_str: unit.to_string(),
        dimension: normalize(&dim_map),
    })
}

/// `add(a, b)` — dimension mismatch raises (Python `UnitMismatchError` →
/// `Err(String)` with the same message).
pub fn add(a: UnitQuantity, b: UnitQuantity) -> Result<UnitQuantity, String> {
    if a.dimension != b.dimension {
        return Err(format!(
            "cannot add {} and {} -- dimension mismatch",
            a.unit_str, b.unit_str
        ));
    }
    Ok(UnitQuantity {
        value: a.value + b.value,
        unit_str: a.unit_str,
        dimension: a.dimension,
    })
}

pub fn multiply(a: UnitQuantity, b: UnitQuantity) -> Result<UnitQuantity, String> {
    let mut combined: HashMap<String, i32> = HashMap::new();
    for (k, v) in &a.dimension {
        *combined.entry(k.clone()).or_insert(0) += v;
    }
    for (k, v) in &b.dimension {
        *combined.entry(k.clone()).or_insert(0) += v;
    }
    let dim = normalize(&combined);
    // Python: " * ".join(sorted({a.unit_str, b.unit_str})) when the units
    // differ, else "{unit}^2"
    let unit_str = if a.unit_str != b.unit_str {
        let mut parts = vec![a.unit_str.clone(), b.unit_str.clone()];
        parts.sort();
        parts.dedup();
        parts.join(" * ")
    } else {
        format!("{}^2", a.unit_str)
    };
    Ok(UnitQuantity {
        value: a.value * b.value,
        unit_str,
        dimension: dim,
    })
}

pub fn divide(a: UnitQuantity, b: UnitQuantity) -> Result<UnitQuantity, String> {
    let mut combined: HashMap<String, i32> = HashMap::new();
    for (k, v) in &a.dimension {
        *combined.entry(k.clone()).or_insert(0) += v;
    }
    for (k, v) in &b.dimension {
        *combined.entry(k.clone()).or_insert(0) -= v;
    }
    let dim = normalize(&combined);
    Ok(UnitQuantity {
        value: a.value / b.value,
        unit_str: format!("{}/{}", a.unit_str, b.unit_str),
        dimension: dim,
    })
}

/// Returns true if the operation is dimensionally valid, false if it would
/// be a unit mismatch — doesn't raise, for use as a search-time filter
/// (an invalid combination is just a state to reject, not a crash).
pub fn check_expression_units(operation: &str, a: &UnitQuantity, b: &UnitQuantity) -> bool {
    if operation == "add" || operation == "subtract" {
        return a.dimension == b.dimension;
    }
    true // multiply/divide are always dimensionally valid
}
