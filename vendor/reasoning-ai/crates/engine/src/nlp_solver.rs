//! Phase 125 — The natural-language entry point: ask(text) (Rust port of
//! `python/nlp/solver.py`).
//!
//! Pipeline: parse(text) (deterministic router) -> if the parse produced a
//! typed payload, solve() routes it through the verified domain solvers
//! (Answer, abstention semantics intact) -> special NL kinds the typed API
//! doesn't have (arithmetic, and the limiting-reagent question that mixes
//! balancing + amounts) are handled here, still with independent
//! verification -> anything unparseable comes back as an honest, structured
//! "not parsed". ask() returns a JSON value so it can cross a process
//! boundary (the CLI and the API entry point both serialize it).

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::{json, Map, Value};

use reasoning_common::{py_float_str, py_round};
use reasoning_nlp::NLParse;
use reasoning_science::chem_balance_domain::limiting_reagent;
use reasoning_science::chemistry_domain::molar_mass;
use reasoning_symbolic::{eval_f64, parse_expr};
use reasoning_uncertainty::Answer;

use crate::solve::{solve, Problem};

/// Deterministic NL -> typed problem (re-export of the nlp crate's parse,
/// which is the Python `nlp.solver.parse`).
pub use reasoning_nlp::parse;

fn abstain_msg(msg: &str) -> Answer {
    Answer::new(false, None, 0.0, None, msg)
}

/// Python `_solve_arithmetic(payload)`: pure arithmetic computed via the
/// CAS, verified by re-parsing through a second, independent numeric
/// evaluation — two evaluations that must agree.
fn _solve_arithmetic(payload: &Map<String, Value>) -> Answer {
    let Some(expr) = payload.get("expr").and_then(Value::as_str) else {
        return abstain_msg("arithmetic: 'expr'");
    };
    let val_exact = match parse_expr(expr) {
        Ok(e) => e,
        Err(e) => return abstain_msg(&format!("arithmetic: could not evaluate ({})", e)),
    };
    let val_float = match eval_f64(&val_exact, &HashMap::new()) {
        Ok(v) => v,
        Err(e) => return abstain_msg(&format!("arithmetic: could not evaluate ({})", e)),
    };
    let val_exact_f = match val_exact.as_rat() {
        Some(r) => r.to_f64(),
        None => return abstain_msg("arithmetic: could not evaluate (non-numeric result)"),
    };
    if (val_exact_f - val_float).abs() > 1e-9 * f64::max(1.0, val_float.abs()) {
        return abstain_msg("arithmetic cross-evaluation disagreed; abstaining");
    }
    let is_integer = val_exact.as_rat().map(|r| r.den == 1).unwrap_or(false);
    let answer = if is_integer {
        val_exact.to_string()
    } else {
        py_float_str(py_round(val_float, 10))
    };
    Answer::new(
        true,
        Some(&answer),
        1.0,
        None,
        &format!(
            "Evaluated {} = {}; cross-checked exact vs numeric evaluation.",
            expr, answer
        ),
    )
}

// SI unit annotations for verified answers — the parser canonicalized
// every input to these units, so the annotated answer is honest.
const UNITS_BY_KIND: [((&str, &str), &str); 22] = [
    (("physics_speed", "speed"), "m/s"),
    (("physics_speed", "distance"), "m"),
    (("physics_speed", "time"), "s"),
    (("physics_kinematics", "v"), "m/s"),
    (("physics_kinematics", "u"), "m/s"),
    (("physics_kinematics", "a"), "m/s^2"),
    (("physics_kinematics", "t"), "s"),
    (("physics_kinematics", "s"), "m"),
    (("physics_force", "force"), "N"),
    (("physics_force", "mass"), "kg"),
    (("physics_force", "acceleration"), "m/s^2"),
    (("physics_energy", "kinetic"), "J"),
    (("physics_energy", "potential"), "J"),
    (("physics_energy", "work"), "J"),
    (("physics_energy", "power"), "W"),
    (("physics_energy", "impact_speed"), "m/s"),
    (("physics_energy", "height_for_speed"), "m"),
    (("physics_momentum", "momentum"), "kg*m/s"),
    (("physics_electricity", "voltage"), "V"),
    (("physics_electricity", "current"), "A"),
    (("physics_electricity", "resistance"), "ohm"),
    (("physics_density", "density"), "kg/m^3"),
];

/// Python truthiness for the slot lookup: `payload.get("find") or
/// payload.get("form") or payload.get("kind")`.
fn truthy_slot<'a>(payload: &'a Map<String, Value>) -> Option<&'a Value> {
    for key in ["find", "form", "kind"] {
        if let Some(v) = payload.get(key) {
            let truthy = match v {
                Value::Null => false,
                Value::Bool(b) => *b,
                Value::Number(n) => n.as_f64().map(|f| f != 0.0).unwrap_or(true),
                Value::String(s) => !s.is_empty(),
                Value::Array(a) => !a.is_empty(),
                Value::Object(o) => !o.is_empty(),
            };
            if truthy {
                return Some(v);
            }
        }
    }
    None
}

fn _annotate(kind: &str, payload: &Map<String, Value>, answer: &str) -> String {
    let slot = truthy_slot(payload).and_then(Value::as_str);
    let unit = slot.and_then(|slot| {
        UNITS_BY_KIND
            .iter()
            .find(|((k, s), _)| *k == kind && *s == slot)
            .map(|(_, u)| *u)
    });
    if let Some(unit) = unit {
        if !answer.is_empty() && !answer.chars().any(|c| c.is_alphabetic()) {
            return format!("{} {}", answer, unit);
        }
    }
    answer.to_string()
}

/// `([\d.]+)\s*(moles?|mols?|mol|grams?|g)\s+(?:of\s+)?([A-Z][A-Za-z0-9()]*)`
fn amount_pat() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"([\d.]+)\s*(moles?|mols?|mol|grams?|g)\s+(?:of\s+)?([A-Z][A-Za-z0-9()]*)")
            .unwrap()
    })
}

/// Python dict repr of an insertion-ordered float map:
/// `{'N2': 1.0, 'H2': 0.5}`.
fn py_dict_repr(entries: &[(String, f64)]) -> String {
    if entries.is_empty() {
        return "{}".to_string();
    }
    let items: Vec<String> = entries
        .iter()
        .map(|(k, v)| format!("'{}': {}", k, py_float_str(*v)))
        .collect();
    format!("{{{}}}", items.join(", "))
}

/// The NL front door. Returns a JSON object:
/// {question, parsed:{ok, kind, payload, explain, candidates},
///  verified, answer, confidence, explanation}.
///
/// (serde_json's Map is key-sorted without the `preserve_order` feature,
/// so the serialized key ORDER differs from the Python dict's insertion
/// order — same keys, same values, JSON-object semantics.)
pub fn ask(text: &str, budget: Option<usize>, seed: u64) -> Value {
    let p: NLParse = parse(text);
    let mut parsed = Map::new();
    parsed.insert("ok".to_string(), json!(p.ok));
    parsed.insert("kind".to_string(), json!(p.kind));
    parsed.insert("payload".to_string(), json!(p.payload));
    parsed.insert("explain".to_string(), json!(p.explain()));
    parsed.insert("candidates".to_string(), json!(p.candidates));

    let mut result = Map::new();
    result.insert("question".to_string(), json!(text));
    result.insert("parsed".to_string(), Value::Object(parsed));
    result.insert("verified".to_string(), json!(false));
    result.insert("answer".to_string(), Value::Null);
    result.insert("confidence".to_string(), json!(0.0));
    result.insert("explanation".to_string(), json!(""));

    if !p.ok || p.payload.is_none() {
        result.insert("explanation".to_string(), json!(p.explain()));
        return Value::Object(result);
    }

    let kind = p.kind.clone().unwrap_or_default();
    let payload = p.payload.clone().unwrap_or_default();
    let mut payload = payload;

    // NL-only kinds handled here (still verified, see helpers above)
    if kind == "arithmetic" {
        let ans = _solve_arithmetic(&payload);
        result.insert("verified".to_string(), json!(ans.verified));
        result.insert("answer".to_string(), json!(ans.final_answer()));
        result.insert(
            "confidence".to_string(),
            json!(if ans.verified { ans.confidence } else { 0.0 }),
        );
        result.insert("explanation".to_string(), json!(ans.explanation));
        return Value::Object(result);
    }

    // chem_limiting from NL: rebuild moles from the matched amounts
    if kind == "chem_limiting_nl" {
        let mut amounts: Vec<(String, f64)> = Vec::new();
        let want_mass = payload
            .get("_want_mass")
            .and_then(Value::as_bool)
            .unwrap_or(false);
        payload.remove("_want_mass");
        payload.remove("_grams");
        // amounts stated as "N moles of F" or "N grams of F"
        for m in amount_pat().captures_iter(text) {
            let val: Option<f64> = m.get(1).unwrap().as_str().parse().ok();
            let Some(val) = val else { continue };
            let unit = m.get(2).unwrap().as_str().to_lowercase();
            let f = m.get(3).unwrap().as_str().to_string();
            let existing = amounts.iter().find(|(k, _)| *k == f).map(|(_, v)| *v);
            if unit.starts_with("mol") {
                let v = existing.unwrap_or(val);
                match amounts.iter_mut().find(|(k, _)| *k == f) {
                    Some(slot) => slot.1 = v,
                    None => amounts.push((f, v)),
                }
            } else {
                // grams -> moles; an unrecognized formula is skipped
                match molar_mass(&f) {
                    Ok(mm) => {
                        let v = existing.unwrap_or(val / mm * 1.0);
                        match amounts.iter_mut().find(|(k, _)| *k == f) {
                            Some(slot) => slot.1 = v,
                            None => amounts.push((f, v)),
                        }
                    }
                    Err(_) => {}
                }
            }
        }
        let (Some(equation), Some(product)) = (
            payload.get("equation").and_then(Value::as_str).map(str::to_string),
            payload.get("product").and_then(Value::as_str).map(str::to_string),
        ) else {
            result.insert(
                "explanation".to_string(),
                json!("limiting-reagent solve failed verification: missing equation or product"),
            );
            return Value::Object(result);
        };
        let moles_ref: Vec<(&str, f64)> = amounts
            .iter()
            .map(|(k, v)| (k.as_str(), *v))
            .collect();
        match limiting_reagent(&equation, &moles_ref, &product) {
            Ok(r) => {
                if let Some(Value::Object(parsed)) = result.get_mut("parsed") {
                    parsed.insert("kind".to_string(), json!("chem_limiting"));
                    let mut moles_json = Map::new();
                    for (k, v) in &amounts {
                        moles_json.insert(k.clone(), json!(v));
                    }
                    let mut new_payload = Map::new();
                    new_payload.insert("equation".to_string(), json!(equation));
                    new_payload.insert("moles".to_string(), Value::Object(moles_json));
                    new_payload.insert("product".to_string(), json!(product));
                    parsed.insert("payload".to_string(), Value::Object(new_payload));
                }
                let product_str = if want_mass {
                    format!("{} g", py_float_str(r.mass_product))
                } else {
                    format!("{} mol", py_float_str(r.moles_product))
                };
                result.insert("verified".to_string(), json!(true));
                result.insert(
                    "answer".to_string(),
                    json!(format!(
                        "{} of {} (limiting reagent: {})",
                        product_str, r.product_formula, r.limiting_reactant
                    )),
                );
                result.insert("confidence".to_string(), json!(1.0));
                result.insert(
                    "explanation".to_string(),
                    json!(format!(
                        "Balanced {}; {} runs out first (leftovers: {}); verified by leftover identities.",
                        equation,
                        r.limiting_reactant,
                        py_dict_repr(&r.leftovers)
                    )),
                );
            }
            Err(e) => {
                result.insert(
                    "explanation".to_string(),
                    json!(format!("limiting-reagent solve failed verification: {}", e)),
                );
            }
        }
        return Value::Object(result);
    }

    let ans = solve(&Problem::new(kind.clone(), payload.clone()), budget, seed);
    let final_answer: Option<String> = ans.final_answer().map(|f| _annotate(&kind, &payload, f));
    result.insert("verified".to_string(), json!(ans.verified));
    result.insert("answer".to_string(), json!(final_answer));
    result.insert(
        "confidence".to_string(),
        json!(if ans.verified { ans.confidence } else { 0.0 }),
    );
    result.insert("explanation".to_string(), json!(ans.explanation));
    Value::Object(result)
}
