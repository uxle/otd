//! Phase 116 — Chemistry: gas laws (Rust port of
//! python/science/chem_gas_domain.py)
//!
//! Textbook ground truth: ideal gas law P*V = n*R*T (R = 0.0821
//! L*atm/(mol*K)), Boyle, Charles, Gay-Lussac, combined gas law.
//! Independent cross-checks: whatever variable is solved for is
/// re-substituted into the full equation (both sides must agree), and
/// temperatures must be positive Kelvin.

/// L*atm/(mol*K) -- standard textbook value.
pub const R_ATM: f64 = 0.0821;

/// Result of a gas-law computation with its independent re-check value.
#[derive(Debug, Clone, PartialEq)]
pub struct GasResult {
    pub value: f64,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_value: f64,
}

/// Solve P*V = n*R*T for any one variable given the other three. `find`
/// accepts 'pressure'/'volume'/'moles'/'temperature' or 'P'/'V'/'n'/'T'.
pub fn ideal_gas(
    find: &str,
    pressure: Option<f64>,
    volume: Option<f64>,
    moles: Option<f64>,
    temperature: Option<f64>,
    r: Option<f64>,
) -> Result<GasResult, String> {
    let r = r.unwrap_or(R_ATM);
    // _aliases maps the long names onto the single letters
    let find_lower = find.to_lowercase();
    let find_alias = match find_lower.as_str() {
        "pressure" => "P",
        "volume" => "V",
        "moles" => "n",
        "temperature" => "T",
        _ => find,
    };
    if let Some(t) = temperature {
        if t <= 0.0 {
            return Err("temperature must be positive Kelvin".to_string());
        }
    }
    let known = [("P", pressure), ("V", volume), ("n", moles), ("T", temperature)];
    if !known.iter().any(|(k, _)| *k == find_alias) {
        return Err("find must be one of pressure, volume, moles, temperature".to_string());
    }
    if known.iter().any(|(k, v)| *k != find_alias && v.is_none()) {
        return Err(format!(
            "solving for {} needs the other three of P, V, n, T",
            find_alias
        ));
    }

    let val = match find_alias {
        "P" => moles.unwrap() * r * temperature.unwrap() / volume.unwrap(),
        "V" => moles.unwrap() * r * temperature.unwrap() / pressure.unwrap(),
        "n" => pressure.unwrap() * volume.unwrap() / (r * temperature.unwrap()),
        _ => pressure.unwrap() * volume.unwrap() / (moles.unwrap() * r),
    };
    // Independent re-check: substitute the ANSWER back into P*V == n*R*T
    // and require both sides to agree -- the answer must satisfy the full
    // identity, not just the rearrangement it came from.
    let (mut p, mut vv, mut nn, mut tt) = (pressure, volume, moles, temperature);
    match find_alias {
        "P" => p = Some(val),
        "V" => vv = Some(val),
        "n" => nn = Some(val),
        _ => tt = Some(val),
    }
    let lhs = p.unwrap() * vv.unwrap();
    let rhs = nn.unwrap() * r * tt.unwrap();
    if (lhs - rhs).abs() > 1e-6 * f64::max(1.0, lhs.abs()) {
        return Err("independent re-check failed: answer does not satisfy P*V = n*R*T".to_string());
    }
    Ok(GasResult {
        value: val,
        equation_used: "P*V = n*R*T".to_string(),
        verify_method: "substitution: P*V == n*R*T with the answer".to_string(),
        verify_value: rhs,
    })
}

// Python's _require_positive(**kwargs): checks each non-None value in the
// order the caller listed them, erroring with "<name> must be positive".
fn require_positive(name: &str, v: Option<f64>) -> Result<(), String> {
    if let Some(v) = v {
        if v <= 0.0 {
            return Err(format!("{} must be positive", name));
        }
    }
    Ok(())
}

/// P1*V1 = P2*V2 at constant temperature. `find` is 'p2' or 'v2'.
/// Re-check: substitute back -- P2*V2 must equal P1*V1 exactly.
///
/// (`require_same_temperature` appears in the Python signature, default
/// True, but the body never uses it; kept for API fidelity.)
#[allow(unused_variables)]
pub fn boyle_law(
    p1: f64,
    v1: f64,
    find: &str,
    p2: Option<f64>,
    v2: Option<f64>,
    require_same_temperature: bool,
) -> Result<GasResult, String> {
    require_positive("p1", Some(p1))?;
    require_positive("v1", Some(v1))?;
    require_positive("p2", p2)?;
    require_positive("v2", v2)?;
    let val = match find {
        "v2" => {
            let Some(p2) = p2 else {
                return Err("solving for v2 needs p2".to_string());
            };
            p1 * v1 / p2
        }
        "p2" => {
            let Some(v2) = v2 else {
                return Err("solving for p2 needs v2".to_string());
            };
            p1 * v1 / v2
        }
        _ => return Err("find must be 'p2' or 'v2'".to_string()),
    };
    let check = if find == "v2" {
        val * p2.unwrap()
    } else {
        val * v2.unwrap()
    };
    if (check - p1 * v1).abs() > 1e-9 * f64::max(1.0, p1 * v1) {
        return Err("substitution re-check failed for Boyle's law".to_string());
    }
    Ok(GasResult {
        value: val,
        equation_used: "P1*V1 = P2*V2".to_string(),
        verify_method: "substitution P2*V2 == P1*V1".to_string(),
        verify_value: check,
    })
}

/// V1/T1 = V2/T2 at constant pressure. Temperatures in Kelvin.
/// Re-check: substitution V2/T2 == V1/T1.
pub fn charles_law(
    v1: f64,
    t1: f64,
    find: &str,
    v2: Option<f64>,
    t2: Option<f64>,
) -> Result<GasResult, String> {
    require_positive("v1", Some(v1))?;
    require_positive("t1", Some(t1))?;
    require_positive("v2", v2)?;
    require_positive("t2", t2)?;
    if t1 <= 0.0 {
        return Err("t1 must be positive Kelvin".to_string());
    }
    let (val, check) = match find {
        "v2" => {
            let Some(t2) = t2 else {
                return Err("solving for v2 needs t2".to_string());
            };
            let val = v1 * t2 / t1;
            (val, val / t2) // val is V2
        }
        "t2" => {
            let Some(v2) = v2 else {
                return Err("solving for t2 needs v2".to_string());
            };
            let val = t1 * v2 / v1;
            (val, v2 / val) // val is T2
        }
        _ => return Err("find must be 'v2' or 't2'".to_string()),
    };
    if (check - v1 / t1).abs() > 1e-9 * f64::max(1e-9, v1 / t1) {
        return Err("substitution re-check failed for Charles's law".to_string());
    }
    Ok(GasResult {
        value: val,
        equation_used: "V1/T1 = V2/T2".to_string(),
        verify_method: "substitution V2/T2 == V1/T1".to_string(),
        verify_value: check,
    })
}

/// P1/T1 = P2/T2 at constant volume. Re-check: substitution P2/T2 == P1/T1.
pub fn gay_lussac_law(
    p1: f64,
    t1: f64,
    find: &str,
    p2: Option<f64>,
    t2: Option<f64>,
) -> Result<GasResult, String> {
    require_positive("p1", Some(p1))?;
    require_positive("t1", Some(t1))?;
    require_positive("p2", p2)?;
    require_positive("t2", t2)?;
    let (val, check) = match find {
        "p2" => {
            let Some(t2) = t2 else {
                return Err("solving for p2 needs t2".to_string());
            };
            let val = p1 * t2 / t1;
            (val, val / t2) // val is P2
        }
        "t2" => {
            let Some(p2) = p2 else {
                return Err("solving for t2 needs p2".to_string());
            };
            let val = t1 * p2 / p1;
            (val, p2 / val) // val is T2
        }
        _ => return Err("find must be 'p2' or 't2'".to_string()),
    };
    if (check - p1 / t1).abs() > 1e-9 * f64::max(1e-9, p1 / t1) {
        return Err("substitution re-check failed for Gay-Lussac's law".to_string());
    }
    Ok(GasResult {
        value: val,
        equation_used: "P1/T1 = P2/T2".to_string(),
        verify_method: "substitution P2/T2 == P1/T1".to_string(),
        verify_value: check,
    })
}

/// P1*V1/T1 = P2*V2/T2 for any of p2/v2/t2. Re-check: substitute the
/// answer back; both sides of the equality must agree exactly.
pub fn combined_gas_law(
    p1: f64,
    v1: f64,
    t1: f64,
    find: &str,
    p2: Option<f64>,
    v2: Option<f64>,
    t2: Option<f64>,
) -> Result<GasResult, String> {
    require_positive("p1", Some(p1))?;
    require_positive("v1", Some(v1))?;
    require_positive("t1", Some(t1))?;
    require_positive("p2", p2)?;
    require_positive("v2", v2)?;
    require_positive("t2", t2)?;
    if !matches!(find, "p2" | "v2" | "t2") {
        return Err("find must be 'p2', 'v2', or 't2'".to_string());
    }
    let mut p2v = p2.unwrap_or(1.0);
    let mut v2v = v2.unwrap_or(1.0);
    let mut t2v = t2.unwrap_or(1.0);
    let val;
    if find == "p2" {
        val = p1 * v1 * t2v / (t1 * v2v);
        p2v = val;
    } else if find == "v2" {
        val = p1 * v1 * t2v / (t1 * p2v);
        v2v = val;
    } else {
        val = t1 * p2v * v2v / (p1 * v1);
        t2v = val;
    }
    let lhs = p1 * v1 / t1;
    let rhs = p2v * v2v / t2v;
    if (lhs - rhs).abs() > 1e-6 * f64::max(1.0, lhs.abs()) {
        return Err("substitution re-check failed for combined gas law".to_string());
    }
    Ok(GasResult {
        value: val,
        equation_used: "P1*V1/T1 = P2*V2/T2".to_string(),
        verify_method: "substitution both sides equal".to_string(),
        verify_value: rhs,
    })
}
