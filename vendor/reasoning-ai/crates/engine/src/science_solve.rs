//! Phases 110-123 — Unified typed solve() routes for the expanded science
//! and puzzle domains (Rust port of `python/apps/science_solve.py`).
//!
//! Every science and puzzle module is reachable through the same
//! Problem(kind, payload) API as the math domains. Each route follows the
//! house pattern: compute the answer -> the underlying module ALREADY
//! re-verifies through an independent equation/identity -> any failure
//! becomes an honest ABSTAIN (Answer.verified=false), never a guess.

use std::collections::HashMap;
use std::sync::OnceLock;

use regex::Regex;
use serde_json::{Map, Value};

use reasoning_common::{py_float_str, py_round, Rat};
use reasoning_science::chem_balance_domain::{balance_equation, limiting_reagent};
// OTD3 science expansion
use reasoning_science::magnetism_domain as magnetism;
use reasoning_science::relativity_domain as relativity;
use reasoning_science::particle_domain as particles;
use reasoning_science::thermo_domain as thermo;
use reasoning_science::waves_domain as waves;
use reasoning_science::chem_gas_domain::{
    boyle_law, charles_law, combined_gas_law, ideal_gas,
};
use reasoning_science::chem_ph_domain::{
    concentration_from_ph, neutralization, ph_from_concentration, ph_poh_pair,
};
use reasoning_science::chem_solutions_domain::{
    dilution, molarity, percent_composition, percent_yield,
};
use reasoning_science::chemistry_domain::{molar_mass, stoichiometry_moles};
use reasoning_science::density_domain::{
    density, float_test, hydrostatic_pressure, pressure_from_force,
};
use reasoning_science::electricity_domain::{
    electrical_power, ohms_law, parallel_resistance, series_parallel_current, series_resistance,
};
use reasoning_science::energy_domain::{
    chemical_energy, classify_energy_form, elastic_energy, electrical_energy,
    height_for_speed, impact_speed_from_height, kinetic_energy, nuclear_energy,
    photon_energy, potential_energy, power_from_force_velocity, power_from_work,
    sound_intensity, thermal_energy, work_done,
};
use reasoning_science::forces_domain::{solve_force, solve_friction, solve_net_acceleration, solve_weight};
use reasoning_science::momentum_domain::{
    elastic_collision, impulse, inelastic_collision, momentum,
};
use reasoning_science::physics_domain::{solve_kinematics, solve_speed_distance_time, KinematicsInputs};
use reasoning_science::bio_dihybrid_domain::dihybrid_cross;
use reasoning_science::bio_dogma_domain::{
    base_composition, reverse_complement, transcribe, translate, DogmaValue,
};
use reasoning_science::bio_ecology_domain::{
    doubling_time, energy_transfer_10_percent, exponential_growth, logistic_growth,
};
use reasoning_science::bio_popgen_domain::{
    hardy_weinberg_from_allele_count, hardy_weinberg_from_recessive,
};
use reasoning_science::biology_domain::{genotype_probabilities, phenotype_probability, punnett_square};
use reasoning_science::Val;
use reasoning_puzzles::calendar_domain::day_of_week;
use reasoning_puzzles::clock_domain::angle_between_hands;
use reasoning_puzzles::direction_domain::{walk, Move};
use reasoning_puzzles::family_tree_domain::FamilyTree;
use reasoning_puzzles::mirror_image_domain::{mirror_clock_time, mirror_image, water_image};
use reasoning_symbolic::{eval_f64, parse_expr};
use reasoning_uncertainty::Answer;

// ---------------- helpers ----------------

fn _abstain(reason: &str) -> Answer {
    Answer::new(false, None, 0.0, None, reason)
}

fn _ok(answer: String, equation: &str, verify: &str) -> Answer {
    Answer::new(
        true,
        Some(&answer),
        1.0,
        None,
        &format!("Solved via {}; independently re-verified ({}).", equation, verify),
    )
}

/// Python `_fmt(x)`: bool -> "yes"/"no", float -> `str(round(x, 6))`,
/// else `str(x)`.
fn _fmt(x: f64) -> String {
    // py_round(x, 6) destroys sub-micron magnitudes (a 7.27e-7 m de Broglie
    // wavelength would round to 1e-06) — keep full precision there,
    // Python-style scientific notation
    if x != 0.0 && x.abs() < 1e-6 {
        py_float_str(x)
    } else {
        py_float_str(py_round(x, 6))
    }
}

fn _fmt_bool(x: bool) -> String {
    if x { "yes".to_string() } else { "no".to_string() }
}

fn _fmt_val(v: &Val) -> String {
    match v {
        Val::Num(x) => _fmt(*x),
        Val::Bool(b) => _fmt_bool(*b),
        Val::Str(s) => s.clone(),
    }
}

fn _fmt_dogma(v: &DogmaValue) -> String {
    match v {
        DogmaValue::Str(s) => s.clone(),
        DogmaValue::Int(i) => i.to_string(),
        DogmaValue::Float(x) => _fmt(*x),
    }
}

// Payload accessors. A missing key in Python raises KeyError, caught by
// the route's `except (ValueError, KeyError)` and rendered
// "prefix: 'key'" — these mirror that message shape.

fn key_err(prefix: &str, key: &str) -> Answer {
    _abstain(&format!("{}: '{}'", prefix, key))
}

fn get_f64(p: &Map<String, Value>, key: &str) -> Option<f64> {
    p.get(key)?.as_f64()
}

fn get_i64(p: &Map<String, Value>, key: &str) -> Option<i64> {
    p.get(key)?.as_i64()
}

fn get_str<'a>(p: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    p.get(key)?.as_str()
}

fn get_bool(p: &Map<String, Value>, key: &str) -> Option<bool> {
    p.get(key)?.as_bool()
}

fn get_f64_vec(p: &Map<String, Value>, key: &str) -> Option<Vec<f64>> {
    let arr = p.get(key)?.as_array()?;
    arr.iter().map(|v| v.as_f64()).collect()
}

/// Python `str(x)` for a JSON scalar (used by family_tree fact coercion).
fn value_to_string(v: &Value) -> String {
    match v {
        Value::String(s) => s.clone(),
        Value::Number(n) => n.to_string(),
        Value::Bool(b) => b.to_string(),
        _ => String::new(),
    }
}

// ---------------- physics ----------------

/// Payload: `find` plus any of u/v/a/t/s, e.g. {'find':'v','u':0,'a':3,'t':5}.
fn solve_physics_kinematics(p: &Map<String, Value>) -> Answer {
    let Some(find) = get_str(p, "find").map(str::to_string) else {
        return key_err("kinematics", "find");
    };
    let inp = KinematicsInputs {
        u: get_f64(p, "u"),
        v: get_f64(p, "v"),
        a: get_f64(p, "a"),
        t: get_f64(p, "t"),
        s: get_f64(p, "s"),
    };
    let (val, eq) = match solve_kinematics(&find, &inp) {
        Ok(r) => (r.value, r.equation_used),
        Err(e) => return _abstain(&format!("kinematics: {}", e)),
    };
    // independent re-check (this layer's own, different algebra): every
    // quantity that appears twice must be reconstructible from the answer.
    let known = |k: &str| p.get(k).and_then(Value::as_f64);
    let mut recheck_failed = false;
    match find.as_str() {
        "v" => {
            if let (Some(u), Some(t)) = (known("u"), known("t")) {
                // Python `known.get("a", 0)` — the key always exists, so a
                // missing `a` yields None -> TypeError -> re-check failed.
                match known("a") {
                    Some(a) => {
                        if (u + a * t - val).abs() >= 1e-6 {
                            recheck_failed = true;
                        }
                    }
                    None => recheck_failed = true,
                }
            }
        }
        "s" => {
            if let (Some(u), Some(t)) = (known("u"), known("t")) {
                match known("a") {
                    Some(a) => {
                        if (u * t + 0.5 * a * t * t - val).abs() >= 1e-6 {
                            recheck_failed = true;
                        }
                    }
                    None => recheck_failed = true,
                }
            }
        }
        "a" => {
            // known["v"]/known["u"]/known["t"] — None or t==0 -> failure
            match (known("v"), known("u"), known("t")) {
                (Some(v), Some(u), Some(t)) if t != 0.0 => {
                    if ((v - u) / t - val).abs() >= 1e-6 {
                        recheck_failed = true;
                    }
                }
                _ => recheck_failed = true,
            }
        }
        "t" => {
            match (known("v"), known("u"), known("a")) {
                (Some(v), Some(u), Some(a)) if a != 0.0 => {
                    if ((v - u) / a - val).abs() >= 1e-6 {
                        recheck_failed = true;
                    }
                }
                _ => recheck_failed = true,
            }
        }
        _ => {}
    }
    if recheck_failed {
        return _abstain("kinematics: independent re-check failed");
    }
    _ok(_fmt(val), &eq, "independent rearrangement in this layer")
}

fn solve_physics_speed(p: &Map<String, Value>) -> Answer {
    let Some(find) = get_str(p, "find") else {
        return key_err("speed/distance/time", "find");
    };
    let v = match solve_speed_distance_time(
        find,
        get_f64(p, "speed"),
        get_f64(p, "distance"),
        get_f64(p, "time"),
    ) {
        Ok(v) => v,
        Err(e) => return _abstain(&format!("speed/distance/time: {}", e)),
    };
    // re-check: s * t == d with the found value substituted in
    let s_ = if find == "speed" { Some(v) } else { get_f64(p, "speed") };
    let d_ = if find == "distance" { Some(v) } else { get_f64(p, "distance") };
    let t_ = if find == "time" { Some(v) } else { get_f64(p, "time") };
    match (s_, t_, d_) {
        (Some(s), Some(t), Some(d)) => {
            if (s * t - d).abs() > 1e-6 {
                return _abstain("speed/distance/time: re-check failed (s*t != d)");
            }
        }
        // Python would raise TypeError on the None arithmetic; fail closed.
        _ => return _abstain("speed/distance/time: re-check failed (s*t != d)"),
    }
    _ok(_fmt(v), "speed = distance / time", "substitution s*t == d")
}

fn solve_physics_force(p: &Map<String, Value>) -> Answer {
    let r = match p.get("kind").and_then(Value::as_str) {
        Some("weight") => {
            let Some(mass) = get_f64(p, "mass") else {
                return key_err("forces", "mass");
            };
            solve_weight(mass, get_f64(p, "g"))
        }
        Some("friction") => {
            let Some(find) = get_str(p, "find") else {
                return key_err("forces", "find");
            };
            solve_friction(
                find,
                get_f64(p, "mass"),
                get_f64(p, "mu"),
                get_f64(p, "normal"),
                get_f64(p, "friction"),
                get_f64(p, "g"),
            )
        }
        Some("net_acceleration") => {
            let Some(mass) = get_f64(p, "mass") else {
                return key_err("forces", "mass");
            };
            let Some(applied_force) = get_f64(p, "applied_force") else {
                return key_err("forces", "applied_force");
            };
            solve_net_acceleration(mass, applied_force, get_f64(p, "mu"), get_f64(p, "g"))
        }
        _ => {
            let Some(find) = get_str(p, "find") else {
                return key_err("forces", "find");
            };
            solve_force(
                find,
                get_f64(p, "mass"),
                get_f64(p, "acceleration"),
                get_f64(p, "force"),
                get_f64(p, "g"),
            )
        }
    };
    match r {
        Ok(r) => _ok(_fmt(r.value), &r.equation_used, &r.verify_method),
        Err(e) => _abstain(&format!("forces: {}", e)),
    }
}

fn solve_physics_energy(p: &Map<String, Value>) -> Answer {
    let Some(form) = get_str(p, "form").map(str::to_string) else {
        return key_err("energy", "form");
    };
    let g = get_f64(p, "g");
    let r = match form.as_str() {
        "kinetic" => {
            let (Some(mass), Some(velocity)) = (get_f64(p, "mass"), get_f64(p, "velocity")) else {
                return key_err("energy", "mass");
            };
            kinetic_energy(mass, velocity)
        }
        "potential" => {
            let (Some(mass), Some(height)) = (get_f64(p, "mass"), get_f64(p, "height")) else {
                return key_err("energy", "mass");
            };
            potential_energy(mass, height, g)
        }
        "work" => {
            let (Some(force), Some(distance)) = (get_f64(p, "force"), get_f64(p, "distance")) else {
                return key_err("energy", "force");
            };
            work_done(force, distance)
        }
        "power" => {
            if get_f64(p, "velocity").is_some() && get_f64(p, "force").is_some() {
                power_from_force_velocity(get_f64(p, "force").unwrap(), get_f64(p, "velocity").unwrap())
            } else {
                let (Some(work), Some(time)) = (get_f64(p, "work"), get_f64(p, "time")) else {
                    return key_err("energy", "work");
                };
                power_from_work(work, time)
            }
        }
        "impact_speed" => {
            let Some(height) = get_f64(p, "height") else {
                return key_err("energy", "height");
            };
            impact_speed_from_height(height, g)
        }
        "height_for_speed" => {
            let Some(speed) = get_f64(p, "speed") else {
                return key_err("energy", "speed");
            };
            height_for_speed(speed, g)
        }
        // ---- OTD3: the full nine-form taxonomy ----
        "thermal" => {
            let (Some(m), Some(c), Some(dt)) = (get_f64(p, "mass"), get_f64(p, "c"), get_f64(p, "dt")) else {
                return key_err("energy", "mass");
            };
            thermal_energy(m, c, dt)
        }
        "radiant" | "photon" => {
            let Some(f) = get_f64(p, "f") else {
                return key_err("energy", "f");
            };
            photon_energy(f)
        }
        "electrical" => {
            let (Some(v), Some(i), Some(t)) = (get_f64(p, "voltage"), get_f64(p, "current"), get_f64(p, "time")) else {
                return key_err("energy", "voltage");
            };
            electrical_energy(v, i, t)
        }
        "sound" => {
            let (Some(pw), Some(d)) = (get_f64(p, "power"), get_f64(p, "distance")) else {
                return key_err("energy", "power");
            };
            sound_intensity(pw, d)
        }
        "chemical" => {
            let (Some(m), Some(ed)) = (get_f64(p, "mass"), get_f64(p, "energy_density")) else {
                return key_err("energy", "mass");
            };
            chemical_energy(m, ed)
        }
        "nuclear" => {
            let (Some(m), Some(fr)) = (get_f64(p, "mass"), get_f64(p, "fraction")) else {
                return key_err("energy", "mass");
            };
            nuclear_energy(m, fr)
        }
        "elastic" => {
            let (Some(k), Some(x)) = (get_f64(p, "k"), get_f64(p, "x")) else {
                return key_err("energy", "k");
            };
            elastic_energy(k, x)
        }
        "classify" => {
            match classify_energy_form(get_str(p, "target").unwrap_or(&form)) {
                Ok((cat, formula, example)) => {
                    return _ok(
                        format!("category: {} — formula: {} — example: {}", cat, formula, example),
                        "taxonomy: stored vs moving",
                        "the two categories, nine forms, one conservation law",
                    );
                }
                Err(e) => return _abstain(&format!("energy taxonomy: {}", e)),
            }
        }
        _ => return _abstain(&format!("unknown energy form '{}'", form)),
    };
    match r {
        Ok(r) => _ok(_fmt(r.value), &r.equation_used, &r.verify_method),
        Err(e) => _abstain(&format!("energy: {}", e)),
    }
}

fn solve_physics_momentum(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("momentum");
    let r = match kind {
        "momentum" => {
            let (Some(mass), Some(velocity)) = (get_f64(p, "mass"), get_f64(p, "velocity")) else {
                return key_err("momentum", "mass");
            };
            match momentum(mass, velocity) {
                Ok(r) => {
                    return _ok(_fmt(r.values[0]), &r.equation_used, &r.verify_method)
                }
                Err(e) => return _abstain(&format!("momentum: {}", e)),
            }
        }
        "impulse" => {
            let (Some(force), Some(dt), Some(mass)) =
                (get_f64(p, "force"), get_f64(p, "dt"), get_f64(p, "mass"))
            else {
                return key_err("momentum", "force");
            };
            impulse(force, dt, mass, get_f64(p, "v_initial"))
        }
        "inelastic" => {
            let (Some(m1), Some(v1), Some(m2), Some(v2)) = (
                get_f64(p, "m1"),
                get_f64(p, "v1"),
                get_f64(p, "m2"),
                get_f64(p, "v2"),
            ) else {
                return key_err("momentum", "m1");
            };
            inelastic_collision(m1, v1, m2, v2)
        }
        "elastic" => {
            let (Some(m1), Some(v1), Some(m2), Some(v2)) = (
                get_f64(p, "m1"),
                get_f64(p, "v1"),
                get_f64(p, "m2"),
                get_f64(p, "v2"),
            ) else {
                return key_err("momentum", "m1");
            };
            elastic_collision(m1, v1, m2, v2)
        }
        _ => return _abstain(&format!("unknown momentum kind '{}'", kind)),
    };
    match r {
        Ok(r) => match kind {
            "impulse" => _ok(
                format!("v_final = {} (impulse = {})", _fmt(r.values[0]), _fmt(r.values[1])),
                &r.equation_used,
                &r.verify_method,
            ),
            "inelastic" => _ok(
                format!(
                    "v_final = {}, energy lost = {}",
                    _fmt(r.values[0]),
                    _fmt(r.values[1])
                ),
                &r.equation_used,
                &r.verify_method,
            ),
            _ => _ok(
                format!("v1' = {}, v2' = {}", _fmt(r.values[0]), _fmt(r.values[1])),
                &r.equation_used,
                &r.verify_method,
            ),
        },
        Err(e) => _abstain(&format!("momentum: {}", e)),
    }
}

fn solve_physics_electricity(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("ohm");
    match kind {
        "ohm" => {
            let Some(find) = get_str(p, "find") else {
                return key_err("electricity", "find");
            };
            match ohms_law(find, get_f64(p, "voltage"), get_f64(p, "current"), get_f64(p, "resistance")) {
                Ok(r) => _ok(_fmt(r.values[0]), &r.equation_used, &r.verify_method),
                Err(e) => _abstain(&format!("electricity: {}", e)),
            }
        }
        "power" => match electrical_power(
            get_f64(p, "voltage"),
            get_f64(p, "current"),
            get_f64(p, "resistance"),
        ) {
            Ok(r) => _ok(_fmt(r.values[0]), &r.equation_used, &r.verify_method),
            Err(e) => _abstain(&format!("electricity: {}", e)),
        },
        "series" | "parallel" => {
            let Some(resistances) = get_f64_vec(p, "resistances") else {
                return key_err("electricity", "resistances");
            };
            let r = if kind == "series" {
                series_resistance(&resistances)
            } else {
                parallel_resistance(&resistances)
            };
            let r = match r {
                Ok(r) => r,
                Err(e) => return _abstain(&format!("electricity: {}", e)),
            };
            let r_total = r.values[0];
            let mut extra = String::new();
            if let Some(voltage) = get_f64(p, "voltage") {
                let cur = if kind == "series" {
                    series_parallel_current(voltage, &resistances)
                } else {
                    ohms_law("current", Some(voltage), None, Some(r_total))
                };
                let i = match cur {
                    Ok(c) => c.values[0],
                    Err(e) => return _abstain(&format!("electricity: {}", e)),
                };
                extra = format!(", total current = {} A", _fmt(i));
            }
            _ok(
                format!("R_total = {} ohm{}", _fmt(r_total), extra),
                &r.equation_used,
                &r.verify_method,
            )
        }
        _ => _abstain(&format!("unknown electricity kind '{}'", kind)),
    }
}

fn solve_physics_density(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("density");
    match kind {
        "density" => match density(get_f64(p, "mass"), get_f64(p, "volume"), get_f64(p, "rho")) {
            Ok(r) => {
                // Python: zip(("density", "floats?"), r.values)
                let vals = r
                    .values
                    .iter()
                    .zip(["density", "floats?"])
                    .map(|(v, k)| format!("{}={}", k, _fmt_val(v)))
                    .collect::<Vec<_>>()
                    .join(" , ");
                _ok(vals, &r.equation_used, &r.verify_method)
            }
            Err(e) => _abstain(&format!("density/pressure: {}", e)),
        },
        "pressure" => {
            let r = if get_f64(p, "rho").is_some() && get_f64(p, "height").is_some() {
                hydrostatic_pressure(
                    get_f64(p, "rho").unwrap(),
                    get_f64(p, "height").unwrap(),
                    get_f64(p, "area"),
                    get_f64(p, "g"),
                )
            } else {
                let (Some(force), Some(area)) = (get_f64(p, "force"), get_f64(p, "area")) else {
                    return key_err("density/pressure", "force");
                };
                pressure_from_force(force, area)
            };
            match r {
                Ok(r) => _ok(_fmt(r.values[0].num()), &r.equation_used, &r.verify_method),
                Err(e) => _abstain(&format!("density/pressure: {}", e)),
            }
        }
        "float" => {
            let Some(rho_object) = get_f64(p, "rho_object") else {
                return key_err("density/pressure", "rho_object");
            };
            match float_test(rho_object, get_f64(p, "rho_fluid")) {
                Ok(r) => _ok(
                    format!(
                        "floats = {}, submerged fraction = {}",
                        _fmt_val(&r.values[0]),
                        _fmt_val(&r.values[1])
                    ),
                    &r.equation_used,
                    &r.verify_method,
                ),
                Err(e) => _abstain(&format!("density/pressure: {}", e)),
            }
        }
        _ => _abstain(&format!("unknown density kind '{}'", kind)),
    }
}

// ---------------- chemistry ----------------

fn solve_chem_molar_mass(p: &Map<String, Value>) -> Answer {
    let Some(formula) = get_str(p, "formula") else {
        return key_err("molar mass", "formula");
    };
    let mm = match molar_mass(formula) {
        Ok(mm) => mm,
        Err(e) => return _abstain(&format!("molar mass: {}", e)),
    };
    // re-check: molar mass must be positive and within a plausible range
    if !(0.0 < mm && mm < 100000.0) {
        return _abstain("molar mass out of plausible range");
    }
    _ok(
        format!("{} g/mol", _fmt(mm)),
        "sum(atoms * atomic weight)",
        "recursive-descent formula parse (IUPAC weights)",
    )
}

fn solve_chem_stoich(p: &Map<String, Value>) -> Answer {
    let Some(moles_reactant) = get_f64(p, "moles_reactant") else {
        return key_err("stoichiometry", "moles_reactant");
    };
    let Some(reactant_coeff) = get_i64(p, "reactant_coeff") else {
        return key_err("stoichiometry", "reactant_coeff");
    };
    let Some(product_coeff) = get_i64(p, "product_coeff") else {
        return key_err("stoichiometry", "product_coeff");
    };
    let Some(product_formula) = get_str(p, "product_formula") else {
        return key_err("stoichiometry", "product_formula");
    };
    match stoichiometry_moles(moles_reactant, reactant_coeff, product_coeff, product_formula) {
        Ok(r) => _ok(
            format!("{} mol ({} g)", _fmt(r.moles_product), _fmt(r.mass_product)),
            "mole-ratio method",
            "coefficient ratio re-derivation",
        ),
        Err(e) => _abstain(&format!("stoichiometry: {}", e)),
    }
}

fn solve_chem_balance(p: &Map<String, Value>) -> Answer {
    let Some(equation) = get_str(p, "equation") else {
        return key_err("balancing", "equation");
    };
    match balance_equation(equation) {
        Ok(eq) => _ok(
            eq.formatted(),
            "nullspace of the element-conservation matrix",
            "element-by-element left/right equality + gcd == 1",
        ),
        Err(e) => _abstain(&format!("balancing: {}", e)),
    }
}

fn solve_chem_limiting(p: &Map<String, Value>) -> Answer {
    let Some(equation) = get_str(p, "equation") else {
        return key_err("limiting reagent", "equation");
    };
    let Some(product) = get_str(p, "product") else {
        return key_err("limiting reagent", "product");
    };
    // moles: dict formula -> f64 (insertion order irrelevant: only looked up)
    let Some(moles_obj) = p.get("moles").and_then(Value::as_object) else {
        return key_err("limiting reagent", "moles");
    };
    let moles: Vec<(String, f64)> = moles_obj
        .iter()
        .filter_map(|(k, v)| v.as_f64().map(|f| (k.clone(), f)))
        .collect();
    let moles_ref: Vec<(&str, f64)> = moles.iter().map(|(k, v)| (k.as_str(), *v)).collect();
    let r = match limiting_reagent(equation, &moles_ref, product) {
        Ok(r) => r,
        Err(e) => return _abstain(&format!("limiting reagent: {}", e)),
    };
    let leftovers = r
        .leftovers
        .iter()
        .map(|(k, v)| format!("{}: {} mol left", k, _fmt(*v)))
        .collect::<Vec<_>>()
        .join(", ");
    _ok(
        format!(
            "limiting = {}; {} = {} mol ({} g); {}",
            r.limiting_reactant,
            r.product_formula,
            _fmt(r.moles_product),
            _fmt(r.mass_product),
            leftovers
        ),
        "extent-of-reaction comparison",
        "leftover identities + product recompute",
    )
}

fn solve_chem_gas(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("ideal");
    let r = match kind {
        "ideal" => {
            let Some(find) = get_str(p, "find") else {
                return key_err("gas laws", "find");
            };
            ideal_gas(
                find,
                get_f64(p, "pressure"),
                get_f64(p, "volume"),
                get_f64(p, "moles"),
                get_f64(p, "temperature"),
                None,
            )
        }
        "boyle" => {
            let (Some(p1), Some(v1), Some(find)) = (
                get_f64(p, "p1"),
                get_f64(p, "v1"),
                get_str(p, "find"),
            ) else {
                return key_err("gas laws", "p1");
            };
            boyle_law(p1, v1, find, get_f64(p, "p2"), get_f64(p, "v2"), true)
        }
        "charles" => {
            let (Some(v1), Some(t1), Some(find)) = (
                get_f64(p, "v1"),
                get_f64(p, "t1"),
                get_str(p, "find"),
            ) else {
                return key_err("gas laws", "v1");
            };
            charles_law(v1, t1, find, get_f64(p, "v2"), get_f64(p, "t2"))
        }
        "combined" => {
            let (Some(p1), Some(v1), Some(t1), Some(find)) = (
                get_f64(p, "p1"),
                get_f64(p, "v1"),
                get_f64(p, "t1"),
                get_str(p, "find"),
            ) else {
                return key_err("gas laws", "p1");
            };
            combined_gas_law(
                p1,
                v1,
                t1,
                find,
                get_f64(p, "p2"),
                get_f64(p, "v2"),
                get_f64(p, "t2"),
            )
        }
        _ => return _abstain(&format!("unknown gas-law kind '{}'", kind)),
    };
    match r {
        Ok(r) => _ok(_fmt(r.value), &r.equation_used, &r.verify_method),
        Err(e) => _abstain(&format!("gas laws: {}", e)),
    }
}

fn solve_chem_solutions(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("molarity");
    let r = match kind {
        "molarity" => molarity(
            get_f64(p, "moles"),
            get_f64(p, "volume_l"),
            get_f64(p, "molarity"),
            get_f64(p, "mass_g"),
            get_str(p, "formula"),
        ),
        "dilution" => dilution(
            get_f64(p, "m1"),
            get_f64(p, "v1"),
            get_f64(p, "m2"),
            get_f64(p, "v2"),
        ),
        "percent_composition" => {
            let Some(formula) = get_str(p, "formula") else {
                return key_err("solutions", "formula");
            };
            percent_composition(formula)
        }
        "percent_yield" => {
            let (Some(actual), Some(theoretical)) = (get_f64(p, "actual"), get_f64(p, "theoretical"))
            else {
                return key_err("solutions", "actual");
            };
            percent_yield(actual, theoretical)
        }
        _ => return _abstain(&format!("unknown solutions kind '{}'", kind)),
    };
    match r {
        Ok(r) => {
            let vals = r
                .values
                .iter()
                .map(|(k, v)| format!("{} = {}", k, _fmt_val(v)))
                .collect::<Vec<_>>()
                .join(" , ");
            _ok(vals, &r.equation_used, &r.verify_method)
        }
        Err(e) => _abstain(&format!("solutions: {}", e)),
    }
}

fn solve_chem_ph(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("from_concentration");
    let r = match kind {
        "from_concentration" => {
            let Some(h) = get_f64(p, "h") else {
                return key_err("pH", "h");
            };
            ph_from_concentration(h)
        }
        "from_ph" => {
            let Some(ph) = get_f64(p, "pH") else {
                return key_err("pH", "pH");
            };
            concentration_from_ph(ph)
        }
        "pair" => ph_poh_pair(get_f64(p, "pH"), get_f64(p, "pOH")),
        "neutralization" => {
            let (Some(moles_acid), Some(moles_base)) = (get_f64(p, "moles_acid"), get_f64(p, "moles_base"))
            else {
                return key_err("pH", "moles_acid");
            };
            let (Some(acid_protons), Some(base_oh)) = (get_i64(p, "acid_protons"), get_i64(p, "base_oh"))
            else {
                return key_err("pH", "acid_protons");
            };
            neutralization(moles_acid, acid_protons, moles_base, base_oh)
        }
        _ => return _abstain(&format!("unknown pH kind '{}'", kind)),
    };
    match r {
        Ok(r) => {
            let vals = r
                .values
                .iter()
                .map(|(k, v)| format!("{} = {}", k, _fmt_val(v)))
                .collect::<Vec<_>>()
                .join(" , ");
            _ok(vals, &r.equation_used, &r.verify_method)
        }
        Err(e) => _abstain(&format!("pH: {}", e)),
    }
}

// ---------------- biology ----------------

/// Python's genotype-probabilities dict is insertion-ordered by first
/// occurrence in the Punnett square; the Rust science crate returns a
/// HashMap, so the order is re-derived here (call-site adaptation).
fn ordered_rat_probs(
    probs: &HashMap<String, Rat>,
    parent_a: &str,
    parent_b: &str,
) -> Vec<(String, Rat)> {
    let mut order: Vec<String> = Vec::new();
    if let Ok(offspring) = punnett_square(parent_a, parent_b) {
        for g in offspring {
            if probs.contains_key(&g) && !order.contains(&g) {
                order.push(g);
            }
        }
    }
    let mut out: Vec<(String, Rat)> = order
        .into_iter()
        .filter_map(|g| probs.get(&g).map(|r| (g, *r)))
        .collect();
    // any key not present in the square (unreachable) appended, sorted
    let mut extra: Vec<String> = probs
        .keys()
        .filter(|k| !out.iter().any(|(g, _)| g == *k))
        .cloned()
        .collect();
    extra.sort();
    for g in extra {
        if let Some(r) = probs.get(&g) {
            out.push((g, *r));
        }
    }
    out
}

fn solve_bio_monohybrid(p: &Map<String, Value>) -> Answer {
    let ask = p.get("ask").and_then(Value::as_str).unwrap_or("phenotype");
    let (Some(parent_a), Some(parent_b)) =
        (get_str(p, "parent_a"), get_str(p, "parent_b"))
    else {
        return key_err("monohybrid cross", "parent_a");
    };
    let r = if ask == "genotype" {
        genotype_probabilities(parent_a, parent_b)
    } else {
        phenotype_probability(parent_a, parent_b, true)
    };
    let probs = match r {
        Ok(probs) => probs,
        Err(e) => return _abstain(&format!("monohybrid cross: {}", e)),
    };
    let parts: Vec<String> = if ask == "genotype" {
        ordered_rat_probs(&probs, parent_a, parent_b)
            .into_iter()
            .map(|(k, v)| format!("{}: {}/{}", k, v.num, v.den))
            .collect()
    } else {
        // Python: {"dominant": ..., "recessive": ...} insertion order
        ["dominant", "recessive"]
            .iter()
            .filter_map(|k| probs.get(*k).map(|v| format!("{}: {}/{}", k, v.num, v.den)))
            .collect()
    };
    _ok(
        parts.join(", "),
        "4-cell Punnett square enumeration (exact Fractions)",
        "exhaustive gamete pairing, sum == 1",
    )
}

fn solve_bio_dihybrid(p: &Map<String, Value>) -> Answer {
    let (Some(parent_a), Some(parent_b)) =
        (get_str(p, "parent_a"), get_str(p, "parent_b"))
    else {
        return key_err("dihybrid cross", "parent_a");
    };
    let r = match dihybrid_cross(parent_a, parent_b) {
        Ok(r) => r,
        Err(e) => return _abstain(&format!("dihybrid cross: {}", e)),
    };
    // Python pheno dict order: dominant-both, dominant-A-only,
    // dominant-B-only, recessive-both
    let parts: Vec<String> = [
        "dominant-both",
        "dominant-A-only",
        "dominant-B-only",
        "recessive-both",
    ]
    .iter()
    .filter_map(|k| r.phenotype_probs.get(*k).map(|v| format!("{}: {}/{}", k, v.num, v.den)))
    .collect();
    _ok(
        parts.join(", "),
        "16-cell Punnett square enumeration",
        &format!("product rule over monohybrid crosses ({})", r.verify_method),
    )
}

fn solve_bio_hardy_weinberg(p: &Map<String, Value>) -> Answer {
    let r = if let Some(q_squared) = get_f64(p, "q_squared") {
        hardy_weinberg_from_recessive(q_squared)
    } else {
        let (Some(dominant_alleles), Some(recessive_alleles)) =
            (get_i64(p, "dominant_alleles"), get_i64(p, "recessive_alleles"))
        else {
            return key_err("Hardy-Weinberg", "dominant_alleles");
        };
        hardy_weinberg_from_allele_count(dominant_alleles, recessive_alleles)
    };
    let r = match r {
        Ok(r) => r,
        Err(e) => return _abstain(&format!("Hardy-Weinberg: {}", e)),
    };
    // Python freqs dict order: p^2, 2pq, q^2
    let freqs = ["p^2", "2pq", "q^2"]
        .iter()
        .filter_map(|k| r.freqs.get(*k).map(|v| format!("{} = {}", k, _fmt(*v))))
        .collect::<Vec<_>>()
        .join(", ");
    _ok(
        format!("p = {}, q = {}; {}", _fmt(r.p), _fmt(r.q), freqs),
        "p^2 + 2pq + q^2 = 1",
        &r.verify_method,
    )
}

fn solve_bio_dogma(p: &Map<String, Value>) -> Answer {
    let kind = match get_str(p, "kind") {
        Some(k) => k.to_string(),
        None => return key_err("central dogma", "kind"),
    };
    let r = match kind.as_str() {
        "transcribe" => {
            let Some(dna) = get_str(p, "dna") else {
                return key_err("central dogma", "dna");
            };
            transcribe(dna)
        }
        "reverse_complement" => {
            let Some(dna) = get_str(p, "dna") else {
                return key_err("central dogma", "dna");
            };
            reverse_complement(dna)
        }
        "translate" => {
            let Some(seq) = get_str(p, "seq") else {
                return key_err("central dogma", "seq");
            };
            translate(seq)
        }
        "composition" => {
            let Some(seq) = get_str(p, "seq") else {
                return key_err("central dogma", "seq");
            };
            base_composition(seq, get_bool(p, "double_stranded").unwrap_or(false))
        }
        _ => return _abstain(&format!("unknown dogma kind '{}'", kind)),
    };
    let r = match r {
        Ok(r) => r,
        Err(e) => return _abstain(&format!("central dogma: {}", e)),
    };
    // Python values dict insertion orders, re-derived at the call site
    let key_order: &[&str] = match kind.as_str() {
        "transcribe" => &["mRNA"],
        "reverse_complement" => &["reverse_complement"],
        "translate" => &["protein", "codons"],
        // counts dict order: "ATUGC" filtered by allowed bases, then GC_percent
        _ => &["A", "T", "U", "G", "C", "GC_percent"],
    };
    let mut parts: Vec<String> = key_order
        .iter()
        .filter_map(|k| r.values.get(*k).map(|v| format!("{} = {}", k, _fmt_dogma(v))))
        .collect();
    // any remaining keys (unreachable) appended in sorted order
    let mut extra: Vec<&String> = r
        .values
        .keys()
        .filter(|k| !key_order.contains(&k.as_str()))
        .collect();
    extra.sort();
    for k in extra {
        parts.push(format!("{} = {}", k, _fmt_dogma(&r.values[k])));
    }
    _ok(
        parts.join(" , "),
        "standard base-pairing / genetic code",
        &r.verify_method,
    )
}

fn solve_bio_ecology(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("exponential");
    let r = match kind {
        "exponential" => {
            let (Some(n0), Some(rate), Some(time)) =
                (get_f64(p, "n0"), get_f64(p, "rate"), get_f64(p, "time"))
            else {
                return key_err("ecology", "n0");
            };
            exponential_growth(n0, rate, time, get_bool(p, "continuous").unwrap_or(false))
        }
        "doubling_time" => {
            let Some(rate) = get_f64(p, "rate") else {
                return key_err("ecology", "rate");
            };
            doubling_time(rate, get_bool(p, "continuous").unwrap_or(true))
        }
        "logistic" => {
            let (Some(n0), Some(k), Some(rate), Some(time)) = (
                get_f64(p, "n0"),
                get_f64(p, "k"),
                get_f64(p, "rate"),
                get_f64(p, "time"),
            ) else {
                return key_err("ecology", "n0");
            };
            logistic_growth(n0, k, rate, time)
        }
        "energy_transfer" => {
            let (Some(producer_energy), Some(trophic_level)) =
                (get_f64(p, "producer_energy"), get_i64(p, "trophic_level"))
            else {
                return key_err("ecology", "producer_energy");
            };
            energy_transfer_10_percent(producer_energy, trophic_level as i32)
        }
        _ => return _abstain(&format!("unknown ecology kind '{}'", kind)),
    };
    let r = match r {
        Ok(r) => r,
        Err(e) => return _abstain(&format!("ecology: {}", e)),
    };
    // Python values dict insertion orders per function, re-derived here
    let key_order: &[&str] = match kind {
        "exponential" => &["population"],
        "doubling_time" => &["doubling_time"],
        "logistic" => &["population", "carrying_capacity"],
        _ => &["energy"],
    };
    let vals = key_order
        .iter()
        .filter_map(|k| r.values.get(*k).map(|v| format!("{} = {}", k, _fmt(*v))))
        .collect::<Vec<_>>()
        .join(" , ");
    _ok(vals, &r.equation_used, &r.verify_method)
}

// ---------------- puzzles ----------------

fn solve_family_tree(p: &Map<String, Value>) -> Answer {
    let facts = match p.get("facts").and_then(Value::as_array) {
        Some(f) => f.clone(),
        None => return key_err("family tree", "facts"),
    };
    let mut tree = FamilyTree::new();
    for fact in &facts {
        let cells = fact.as_array().cloned().unwrap_or_default();
        if cells.len() != 3 {
            return _abstain("family tree: facts must be [subject, relation, obj] triples");
        }
        if let Err(e) = tree.add_fact(
            &value_to_string(&cells[0]),
            &value_to_string(&cells[1]),
            &value_to_string(&cells[2]),
        ) {
            return _abstain(&format!("family tree: {}", e));
        }
    }
    let (Some(who), Some(whom)) = (get_str(p, "who"), get_str(p, "whom")) else {
        return key_err("family tree", "who");
    };
    let rel = match tree.relationship(who, whom) {
        Some(rel) => rel,
        None => {
            return _abstain(
                "relationship not derivable from the stated facts \
                 (abstaining rather than guessing)",
            )
        }
    };
    _ok(
        format!("{} is {}'s {}", who, whom, rel),
        "explicit relation graph + BFS to closest common ancestor",
        "graph-derived (no memorized answers)",
    )
}

fn solve_direction_walk(p: &Map<String, Value>) -> Answer {
    let moves_json = match p.get("moves").and_then(Value::as_array) {
        Some(m) => m.clone(),
        None => return key_err("direction walk", "moves"),
    };
    let mut moves: Vec<Move> = Vec::with_capacity(moves_json.len());
    for mv in &moves_json {
        let cells = mv.as_array().cloned().unwrap_or_default();
        if cells.len() != 2 {
            return _abstain("direction walk: moves must be [direction, distance] pairs");
        }
        let Some(d) = cells[0].as_str() else {
            return key_err("direction walk", "moves");
        };
        let Some(dist) = cells[1].as_f64() else {
            return key_err("direction walk", "moves");
        };
        moves.push(Move::new(d, dist));
    }
    let r = match walk(&moves) {
        Ok(r) => r,
        Err(e) => return _abstain(&format!("direction walk: {}", e)),
    };
    // re-check: rebuild displacement from components independently
    let dx: f64 = moves_json
        .iter()
        .map(|mv| {
            let cells = mv.as_array().cloned().unwrap_or_default();
            let d = cells.first().and_then(Value::as_str).unwrap_or("").to_uppercase();
            let dist = cells.get(1).and_then(Value::as_f64).unwrap_or(0.0);
            let sign = if d.starts_with('E') {
                1.0
            } else if d.starts_with('W') {
                -1.0
            } else {
                0.0
            };
            sign * dist
        })
        .sum();
    let dy: f64 = moves_json
        .iter()
        .map(|mv| {
            let cells = mv.as_array().cloned().unwrap_or_default();
            let d = cells.first().and_then(Value::as_str).unwrap_or("").to_uppercase();
            let dist = cells.get(1).and_then(Value::as_f64).unwrap_or(0.0);
            let sign = if d.starts_with('N') {
                1.0
            } else if d.starts_with('S') {
                -1.0
            } else {
                0.0
            };
            sign * dist
        })
        .sum();
    if (dx.hypot(dy) - r.straight_line_distance).abs() > 1e-9 {
        return _abstain("displacement re-check failed");
    }
    _ok(
        format!(
            "distance = {}, bearing = {} deg from North",
            _fmt(r.straight_line_distance),
            _fmt(r.bearing_degrees)
        ),
        "2D vector walk",
        "component-wise displacement recompute",
    )
}

fn solve_clock_angle(p: &Map<String, Value>) -> Answer {
    let (Some(hour), Some(minute)) = (get_i64(p, "hour"), get_i64(p, "minute")) else {
        return key_err("clock angle", "hour");
    };
    let a = match angle_between_hands(hour as i32, minute as i32) {
        Ok(a) => a,
        Err(e) => return _abstain(&format!("clock angle: {}", e)),
    };
    // independent re-check: angle swept difference must be < 180 smaller side
    let h_a = (hour % 12) as f64 * 30.0 + minute as f64 * 0.5;
    let m_a = minute as f64 * 6.0;
    let raw = (h_a - m_a).abs();
    let raw = if raw > 180.0 { 360.0 - raw } else { raw };
    if (raw - a).abs() > 1e-9 {
        return _abstain("clock-angle re-check failed");
    }
    _ok(
        format!("{} degrees", _fmt(a)),
        "hour-hand 30 deg/h + 0.5 deg/min creep",
        "independent swept-angle difference",
    )
}

fn solve_calendar_day(p: &Map<String, Value>) -> Answer {
    let (Some(year), Some(month), Some(day)) =
        (get_i64(p, "year"), get_i64(p, "month"), get_i64(p, "day"))
    else {
        return key_err("calendar", "year");
    };
    match day_of_week(year as i32, month as i32, day as i32) {
        Ok(d) => _ok(
            d,
            "Zeller's congruence",
            "cross-checked against datetime.weekday() on every call",
        ),
        Err(e) => _abstain(&format!("calendar: {}", e)),
    }
}

fn solve_mirror_image(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("mirror");
    match kind {
        "mirror" => {
            let Some(text) = get_str(p, "text") else {
                return key_err("mirror image", "text");
            };
            match mirror_image(text) {
                Ok(img) => _ok(img, "per-glyph mirror table", "hand-verified glyph table (stated as data)"),
                Err(e) => _abstain(&format!("mirror image: {}", e)),
            }
        }
        "water" => {
            let Some(text) = get_str(p, "text") else {
                return key_err("mirror image", "text");
            };
            match water_image(text) {
                Ok(img) => _ok(img, "per-glyph water-image table", "hand-verified glyph table (stated as data)"),
                Err(e) => _abstain(&format!("mirror image: {}", e)),
            }
        }
        "mirror_clock" => {
            let Some(time) = get_str(p, "time") else {
                return key_err("mirror image", "time");
            };
            match mirror_clock_time(time) {
                Ok(t) => _ok(t, "clock-face mirror geometry", "12h - t rule with roll-over handling"),
                Err(e) => _abstain(&format!("mirror image: {}", e)),
            }
        }
        _ => _abstain(&format!("unknown mirror kind '{}'", kind)),
    }
}

// ---------------- arithmetic (typed) ----------------

fn arithmetic_chars() -> &'static Regex {
    // Python: re.fullmatch(r"[-+*/(). \d**]+", expr) — a char class (the
    // duplicated `*` is redundant).
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"^[-+*/(). \d]+$").unwrap())
}

fn solve_arithmetic(p: &Map<String, Value>) -> Answer {
    // Pure arithmetic: compute via the CAS, cross-check via a second,
    // independent numeric evaluation before returning.
    let Some(expr) = get_str(p, "expr") else {
        return key_err("arithmetic", "expr");
    };
    if !arithmetic_chars().is_match(expr) {
        return _abstain("arithmetic: expression contains disallowed characters");
    }
    let val_exact = match parse_expr(expr) {
        Ok(e) => e,
        Err(e) => return _abstain(&format!("arithmetic: could not evaluate ({})", e)),
    };
    let val_float = match eval_f64(&val_exact, &HashMap::new()) {
        Ok(v) => v,
        Err(e) => return _abstain(&format!("arithmetic: could not evaluate ({})", e)),
    };
    // Python `float(val_exact)` — a non-numeric result raises TypeError
    let val_exact_f = match val_exact.as_rat() {
        Some(r) => r.to_f64(),
        None => {
            return _abstain("arithmetic: could not evaluate (non-numeric result)");
        }
    };
    if (val_exact_f - val_float).abs() > 1e-9 * f64::max(1.0, val_float.abs()) {
        return _abstain("arithmetic: cross-evaluation disagreed");
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

// ---------------- dispatch table ----------------

/// The 25 science/puzzle routes keyed by problem kind (Python
/// `SCIENCE_ROUTES`, built once).

// ---------------------------------------------------------------------------
// OTD3 — magnetism, waves, thermodynamics, relativity routes
// ---------------------------------------------------------------------------

fn solve_physics_magnetism(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("wire_force").to_string();
    let r = match kind.as_str() {
        "wire_force" => {
            let (Some(b), Some(i), Some(l)) = (get_f64(p, "b"), get_f64(p, "current"), get_f64(p, "length")) else {
                return key_err("magnetism", "b");
            };
            let sin = get_f64(p, "sin").unwrap_or(1.0);
            magnetism::force_on_wire(b, i, l, sin)
        }
        "lorentz" => {
            let (Some(q), Some(v), Some(b)) = (get_f64(p, "charge"), get_f64(p, "velocity"), get_f64(p, "b")) else {
                return key_err("magnetism", "charge");
            };
            let sin = get_f64(p, "sin").unwrap_or(1.0);
            magnetism::lorentz_force(q, v, b, sin)
        }
        "wire_field" => {
            let (Some(i), Some(r)) = (get_f64(p, "current"), get_f64(p, "distance")) else {
                return key_err("magnetism", "current");
            };
            magnetism::wire_field(i, r)
        }
        "solenoid" => {
            let (Some(n), Some(i)) = (get_f64(p, "turns_per_m"), get_f64(p, "current")) else {
                return key_err("magnetism", "turns_per_m");
            };
            magnetism::solenoid_field(n, i)
        }
        "faraday" => {
            let (Some(n), Some(dflux), Some(dt)) = (get_f64(p, "turns"), get_f64(p, "d_flux"), get_f64(p, "dt")) else {
                return key_err("magnetism", "turns");
            };
            magnetism::faraday_emf(n, dflux, dt)
        }
        _ => return _abstain(&format!("unknown magnetism kind '{}'", kind)),
    };
    match r {
        Ok(r) => _ok(_fmt(r.values[0]), &r.equation_used, &r.verify_method),
        Err(e) => _abstain(&format!("magnetism: {}", e)),
    }
}

fn solve_physics_waves(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("wave_equation").to_string();
    let r: Result<waves::WavesResult, String> = match kind.as_str() {
        "wave_equation" => {
            let v = get_f64(p, "v");
            let f = get_f64(p, "f");
            let lambda = get_f64(p, "lambda");
            waves::wave_equation(v, f, lambda)
        }
        "sound_speed" => {
            let Some(t) = get_f64(p, "temp_c") else {
                return key_err("waves", "temp_c");
            };
            waves::sound_speed_air(t)
        }
        "doppler" => {
            let (Some(f), Some(v), Some(vs)) = (get_f64(p, "f"), get_f64(p, "v_sound"), get_f64(p, "v_source")) else {
                return key_err("waves", "f");
            };
            waves::doppler(f, v, vs)
        }
        "echo_distance" => {
            let (Some(v), Some(t)) = (get_f64(p, "v_sound"), get_f64(p, "t")) else {
                return key_err("waves", "v_sound");
            };
            waves::echo_distance(v, t)
        }
        "snell" => {
            let (Some(n1), Some(t1), Some(n2)) = (get_f64(p, "n1"), get_f64(p, "theta1"), get_f64(p, "n2")) else {
                return key_err("waves", "n1");
            };
            waves::snell(n1, t1, n2)
        }
        "thin_lens" => {
            let f = get_f64(p, "f");
            let dobj = get_f64(p, "do");
            let dimg = get_f64(p, "di");
            waves::thin_lens(f, dobj, dimg)
        }
        "inverse_square" => {
            let (Some(i1), Some(d1), Some(d2)) = (get_f64(p, "i1"), get_f64(p, "d1"), get_f64(p, "d2")) else {
                return key_err("waves", "i1");
            };
            waves::inverse_square(i1, d1, d2)
        }
        _ => return _abstain(&format!("unknown waves kind '{}'", kind)),
    };
    match r {
        Ok(r) => _ok(_fmt(r.values[0]), &r.equation_used, &r.verify_method),
        Err(e) => _abstain(&format!("waves: {}", e)),
    }
}

fn solve_physics_thermo(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("convert").to_string();
    let r: Result<thermo::ThermoResult, String> = match kind.as_str() {
        "convert" => {
            let c = get_f64(p, "celsius");
            let k = get_f64(p, "kelvin");
            let f = get_f64(p, "fahrenheit");
            let target = p.get("target").and_then(Value::as_str).unwrap_or("");
            match thermo::convert_temp(c, k, f) {
                Ok(r) => {
                    // pick the asked-for scale: [0]=C [1]=K [2]=F
                    let idx = match target {
                        "to_kelvin" => 1,
                        "to_fahrenheit" => 2,
                        "identity" => 0,
                        _ => 1,
                    };
                    return _ok(_fmt(r.values[idx]), &r.equation_used, &r.verify_method);
                }
                Err(e) => return _abstain(&format!("thermo: {}", e)),
            }
        }
        "sensible_heat" => {
            let q = get_f64(p, "q");
            let m = get_f64(p, "mass");
            let c = get_f64(p, "c");
            let dt = get_f64(p, "dt");
            thermo::sensible_heat(q, m, c, dt)
        }
        "latent_heat" => {
            let (Some(m), Some(l)) = (get_f64(p, "mass"), get_f64(p, "l")) else {
                return key_err("thermo", "mass");
            };
            thermo::latent_heat(m, l)
        }
        "ideal_gas" => {
            let (pp, vv, nn, tt) = (get_f64(p, "p"), get_f64(p, "v"), get_f64(p, "n"), get_f64(p, "t"));
            thermo::ideal_gas(pp, vv, nn, tt)
        }
        "conduction" => {
            let (Some(k), Some(a), Some(dt), Some(d)) =
                (get_f64(p, "k"), get_f64(p, "area"), get_f64(p, "dt"), get_f64(p, "thickness")) else {
                return key_err("thermo", "k");
            };
            thermo::conduction(k, a, dt, d)
        }
        "stefan_boltzmann" => {
            let (Some(a), Some(t)) = (get_f64(p, "area"), get_f64(p, "t_kelvin")) else {
                return key_err("thermo", "area");
            };
            let e = get_f64(p, "emissivity").unwrap_or(1.0);
            thermo::stefan_boltzmann(a, t, e)
        }
        "expansion" => {
            let (Some(a), Some(l0), Some(dt)) = (get_f64(p, "alpha"), get_f64(p, "l0"), get_f64(p, "dt")) else {
                return key_err("thermo", "alpha");
            };
            thermo::thermal_expansion(a, l0, dt)
        }
        _ => return _abstain(&format!("unknown thermo kind '{}'", kind)),
    };
    match r {
        Ok(r) => _ok(_fmt(r.values[0]), &r.equation_used, &r.verify_method),
        Err(e) => _abstain(&format!("thermo: {}", e)),
    }
}

fn solve_physics_relativity(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("gamma").to_string();
    let r: Result<relativity::RelativityResult, String> = match kind.as_str() {
        "gamma" | "lorentz" => {
            let Some(v) = get_f64(p, "v") else {
                return key_err("relativity", "v");
            };
            relativity::lorentz_factor(v)
        }
        "time_dilation" => {
            let Some(v) = get_f64(p, "v") else {
                return key_err("relativity", "v");
            };
            let t0 = get_f64(p, "proper_time");
            let t = get_f64(p, "dilated_time");
            relativity::time_dilation(v, t0, t)
        }
        "length_contraction" => {
            let (Some(v), Some(l0)) = (get_f64(p, "v"), get_f64(p, "l0")) else {
                return key_err("relativity", "v");
            };
            relativity::length_contraction(v, l0)
        }
        "rest_energy" => {
            let Some(m) = get_f64(p, "mass") else {
                return key_err("relativity", "mass");
            };
            relativity::rest_energy(m)
        }
        "velocity_addition" => {
            let (Some(u), Some(v)) = (get_f64(p, "u"), get_f64(p, "v")) else {
                return key_err("relativity", "u");
            };
            relativity::velocity_addition(u, v)
        }
        _ => return _abstain(&format!("unknown relativity kind '{}'", kind)),
    };
    match r {
        Ok(r) => _ok(_fmt(r.values[0]), &r.equation_used, &r.verify_method),
        Err(e) => _abstain(&format!("relativity: {}", e)),
    }
}

fn solve_physics_particles(p: &Map<String, Value>) -> Answer {
    let kind = p.get("kind").and_then(Value::as_str).unwrap_or("quark_composition").to_string();
    // numeric answers (values/equation/verify) — mirror the other routes
    let numeric: Result<particles::ParticleResult, String> = match kind.as_str() {
        "particle_mass" => {
            let Some(name) = get_str(p, "name") else {
                return key_err("particles", "name");
            };
            particles::particle_mass(name)
        }
        "neutron_decay" => particles::neutron_decay_energy(),
        "annihilation" => particles::annihilation_energy(),
        "photon_energy" => {
            let Some(wl) = get_f64(p, "wavelength") else {
                return key_err("particles", "wavelength");
            };
            particles::photon_energy(wl)
        }
        "photon_wavelength" => {
            let Some(e) = get_f64(p, "energy") else {
                return key_err("particles", "energy");
            };
            particles::photon_wavelength(e)
        }
        "de_broglie" => {
            let (Some(m), Some(v)) = (get_f64(p, "mass"), get_f64(p, "velocity")) else {
                return key_err("particles", "mass");
            };
            particles::de_broglie(m, v)
        }
        "electron_de_broglie" => {
            let Some(v) = get_f64(p, "velocity") else {
                return key_err("particles", "velocity");
            };
            particles::electron_de_broglie(v)
        }
        "binding_energy" => {
            let (Some(a), Some(z)) = (get_f64(p, "a"), get_f64(p, "z")) else {
                return key_err("particles", "a");
            };
            if a.fract() != 0.0 || z.fract() != 0.0 {
                return _abstain("particles: A and Z must be integers");
            }
            particles::binding_energy(a as u32, z as u32)
        }
        "decay_age" => {
            let (Some(f), Some(hl)) = (get_f64(p, "fraction"), get_f64(p, "half_life")) else {
                return key_err("particles", "fraction");
            };
            particles::decay_age(f, hl)
        }
        "activity" => {
            let (Some(n), Some(hl)) = (get_f64(p, "n_atoms"), get_f64(p, "half_life")) else {
                return key_err("particles", "n_atoms");
            };
            particles::activity(n, hl)
        }
        "shell_capacity" => {
            let Some(n) = get_f64(p, "n") else {
                return key_err("particles", "n");
            };
            if n.fract() != 0.0 {
                return _abstain("particles: n must be an integer");
            }
            particles::shell_capacity(n as u32)
        }
        "rydberg" => {
            let (Some(lo), Some(hi)) = (get_f64(p, "n_lo"), get_f64(p, "n_hi")) else {
                return key_err("particles", "n_lo");
            };
            if lo.fract() != 0.0 || hi.fract() != 0.0 {
                return _abstain("particles: quantum numbers must be integers");
            }
            particles::rydberg_wavelength(lo as u32, hi as u32)
        }
        _ => {
            // text answers (compositions, configurations, half-lives)
            let textual: Result<particles::ParticleTextResult, String> = match kind.as_str() {
                "quark_composition" => {
                    let Some(hadron) = get_str(p, "hadron") else {
                        return key_err("particles", "hadron");
                    };
                    particles::quark_composition(hadron)
                }
                "half_life" => {
                    let Some(element) = get_str(p, "element") else {
                        return key_err("particles", "element");
                    };
                    let Some(a) = get_f64(p, "a") else {
                        return key_err("particles", "a");
                    };
                    if a.fract() != 0.0 {
                        return _abstain("particles: A must be an integer");
                    }
                    particles::half_life(element, a as u32)
                }
                "electron_configuration" => {
                    let Some(z) = get_f64(p, "z") else {
                        return key_err("particles", "z");
                    };
                    if z.fract() != 0.0 {
                        return _abstain("particles: Z must be an integer");
                    }
                    particles::electron_configuration(z as u32)
                }
                _ => return _abstain(&format!("unknown particles kind '{}'", kind)),
            };
            return match textual {
                Ok(r) => _ok(r.value, &r.equation_used, &r.verify_method),
                Err(e) => _abstain(&format!("particles: {}", e)),
            };
        }
    };
    match numeric {
        Ok(r) => _ok(_fmt(r.values[0]), &r.equation_used, &r.verify_method),
        Err(e) => _abstain(&format!("particles: {}", e)),
    }
}

pub fn science_routes() -> &'static HashMap<&'static str, fn(&Map<String, Value>) -> Answer> {
    static ROUTES: OnceLock<HashMap<&'static str, fn(&Map<String, Value>) -> Answer>> =
        OnceLock::new();
    ROUTES.get_or_init(|| {
        let mut m: HashMap<&'static str, fn(&Map<String, Value>) -> Answer> = HashMap::new();
        m.insert("arithmetic", solve_arithmetic);
        m.insert("physics_kinematics", solve_physics_kinematics);
        m.insert("physics_speed", solve_physics_speed);
        m.insert("physics_force", solve_physics_force);
        m.insert("physics_energy", solve_physics_energy);
        m.insert("physics_momentum", solve_physics_momentum);
        m.insert("physics_electricity", solve_physics_electricity);
        m.insert("physics_magnetism", solve_physics_magnetism);
        m.insert("physics_waves", solve_physics_waves);
        m.insert("physics_thermo", solve_physics_thermo);
        m.insert("physics_relativity", solve_physics_relativity);
        m.insert("physics_particles", solve_physics_particles);
        m.insert("physics_density", solve_physics_density);
        m.insert("chem_molar_mass", solve_chem_molar_mass);
        m.insert("chem_stoich", solve_chem_stoich);
        m.insert("chem_balance", solve_chem_balance);
        m.insert("chem_limiting", solve_chem_limiting);
        m.insert("chem_gas", solve_chem_gas);
        m.insert("chem_solutions", solve_chem_solutions);
        m.insert("chem_ph", solve_chem_ph);
        m.insert("bio_monohybrid", solve_bio_monohybrid);
        m.insert("bio_dihybrid", solve_bio_dihybrid);
        m.insert("bio_hardy_weinberg", solve_bio_hardy_weinberg);
        m.insert("bio_dogma", solve_bio_dogma);
        m.insert("bio_ecology", solve_bio_ecology);
        m.insert("family_tree", solve_family_tree);
        m.insert("direction_walk", solve_direction_walk);
        m.insert("clock_angle", solve_clock_angle);
        m.insert("calendar_day", solve_calendar_day);
        m.insert("mirror_image", solve_mirror_image);
        m
    })
}
