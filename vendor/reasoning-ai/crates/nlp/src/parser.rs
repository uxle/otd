//! Phase 124 — Deterministic natural-language router (science + puzzles).
//! (Rust port of python/nlp/parser.py)
//!
//! Takes plain-English questions and produces a TYPED problem payload for
//! the verified solve() API. Route decisions are made by explicit, ordered,
//! testable patterns — NOT by a model guess. Units are canonicalized
//! (nlp/quantities.rs) BEFORE any formula sees them. When the question
//! underdetermines the problem the parser ABSTAINS. `candidates` lists
//! other routes that also scored, for transparency.

use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use regex::Regex;
use serde_json::{json, Map, Value};

use crate::quantities::{rx, Quantity};

/// Python `NLParse` dataclass. `payload` uses a JSON object so it can cross
/// the process boundary exactly like the Python dict did.
#[derive(Debug, Clone, Default)]
pub struct NLParse {
    pub ok: bool,
    pub kind: Option<String>,
    pub payload: Option<Map<String, Value>>,
    pub rationale: String,
    /// slot -> source text
    pub matched: HashMap<String, String>,
    pub candidates: Vec<String>,
    pub confidence: f64,
}

impl NLParse {
    /// Python positional constructor `NLParse(True, kind, payload, rationale, matched, confidence)`.
    pub fn ok(
        kind: &str,
        payload: Map<String, Value>,
        rationale: &str,
        matched: HashMap<String, String>,
        confidence: f64,
    ) -> Self {
        NLParse {
            ok: true,
            kind: Some(kind.to_string()),
            payload: Some(payload),
            rationale: rationale.to_string(),
            matched,
            candidates: Vec::new(),
            confidence,
        }
    }

    pub fn ok_simple(kind: &str, payload: Map<String, Value>, rationale: &str, confidence: f64) -> Self {
        NLParse::ok(kind, payload, rationale, HashMap::new(), confidence)
    }

    /// `NLParse(False, rationale=...)`.
    pub fn fail(rationale: &str) -> Self {
        NLParse {
            ok: false,
            rationale: rationale.to_string(),
            ..Default::default()
        }
    }

    /// `NLParse(False, rationale=..., candidates=[...])`.
    pub fn fail_with_candidates(rationale: &str, candidates: Vec<String>) -> Self {
        NLParse {
            ok: false,
            rationale: rationale.to_string(),
            candidates,
            ..Default::default()
        }
    }

    pub fn explain(&self) -> String {
        if !self.ok {
            return format!(
                "Could not map this question onto a verified domain (deterministic parser). {}",
                self.rationale.trim()
            )
            .trim()
            .to_string();
        }
        let slots: Vec<String> = self
            .matched
            .iter()
            .map(|(k, v)| format!("{} <- {:?}", k, v))
            .collect();
        format!(
            "routed to {}. {} Mapped: {}",
            self.kind.as_deref().unwrap_or(""),
            self.rationale,
            slots.join("; ")
        )
    }
}

// ---------------- shared helpers ----------------

fn form_pat() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"\b(?:[A-Z][a-z]?\d*)+(?:\((?:[A-Z][a-z]?\d*)+\)\d*)*\b").unwrap()
    })
}

const MONTHS: [(&str, u32); 12] = [
    ("january", 1),
    ("february", 2),
    ("march", 3),
    ("april", 4),
    ("may", 5),
    ("june", 6),
    ("july", 7),
    ("august", 8),
    ("september", 9),
    ("october", 10),
    ("november", 11),
    ("december", 12),
];

fn month_number(name: &str) -> Option<u32> {
    let low = name.to_lowercase();
    MONTHS.iter().find(|(m, _)| *m == low).map(|(_, n)| *n)
}

fn direction_code(name: &str) -> Option<&'static str> {
    Some(match name.to_lowercase().as_str() {
        "north" | "n" => "N",
        "south" | "s" => "S",
        "east" | "e" => "E",
        "west" | "w" => "W",
        "northeast" => "NE",
        "northwest" => "NW",
        "southeast" => "SE",
        "southwest" => "SW",
        _ => return None,
    })
}

/// Python `_has(text, *patterns)`: any re.search(p, text, re.I).
pub fn has(text: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| re_search_ci(p, text))
}

/// case-insensitive search helper with a per-pattern compiled cache
/// (Python `re.search(p, text, re.I)`).
fn re_search_ci(pattern: &str, text: &str) -> bool {
    let ci = format!("(?i){}", pattern);
    regex_cache(&ci).is_match(text)
}

pub(crate) fn regex_cache(pattern: &str) -> &'static Regex {
    static CACHE: OnceLock<Mutex<HashMap<String, &'static Regex>>> = OnceLock::new();
    let cache = CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let mut guard = cache.lock().unwrap();
    if let Some(r) = guard.get(pattern) {
        return r;
    }
    // Box::leak gives a 'static Regex; patterns are a fixed small set from
    // the route code, so this is bounded (one leak per distinct pattern).
    let r: &'static Regex = Box::leak(Box::new(Regex::new(pattern).unwrap()));
    guard.insert(pattern.to_string(), r);
    r
}

fn q_value(q: &Quantity) -> f64 {
    q.value
}

fn first<'a>(qs: &'a [Quantity], unit: &str) -> Option<&'a Quantity> {
    qs.iter().find(|q| q.unit == unit)
}

fn all<'a>(qs: &'a [Quantity], unit: &str) -> Vec<&'a Quantity> {
    qs.iter().filter(|q| q.unit == unit).collect()
}

#[allow(dead_code)]
fn bare<'a>(qs: &'a [Quantity]) -> Vec<&'a Quantity> {
    qs.iter().filter(|q| q.unit.is_empty()).collect()
}

fn obj(pairs: Vec<(&str, Value)>) -> Map<String, Value> {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    m
}

// ---------------- science routes ----------------

pub type RouteFn = fn(&str, &[Quantity]) -> Option<NLParse>;
pub struct NamedRoute {
    pub name: &'static str,
    pub f: RouteFn,
}

fn route_kinematics(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !(has(
        text,
        &[
            r"accelerat",
            r"decelerat",
            r"free\s*fall",
            r"drops?",
            r"falls?",
        ],
    ) || first(qs, "m/s^2").is_some())
    {
        return None;
    }
    let mut p: Vec<(&str, Value)> = Vec::new();
    let mut matched: HashMap<String, String> = HashMap::new();
    if let Some(accel) = first(qs, "m/s^2") {
        p.push(("a", json!(accel.value)));
        matched.insert("a".into(), accel.text.clone());
    }
    // from rest / dropped -> u = 0
    if has(
        text,
        &[
            r"from\s+rest",
            r"at\s+rest",
            r"starts?\s+resting",
            r"dropped",
            r"falls?\s+from\s+rest",
            r"free\s*fall",
        ],
    ) {
        p.push(("u", json!(0.0)));
        matched.insert("u".into(), "'from rest' -> 0".into());
    }
    // comes to rest / stops -> v = 0
    if has(
        text,
        &[
            r"comes?\s+to\s+(?:a\s+)?rest",
            r"(?:comes?\s+to\s+)?a?\s*stop",
            r"halts?",
        ],
    ) {
        p.push(("v", json!(0.0)));
        matched.insert("v".into(), "'comes to rest' -> 0".into());
    }
    // explicit initial/final wording
    for q in all(qs, "m/s") {
        let before = text
            .find(&q.text)
            .map(|i| &text[..i])
            .unwrap_or("");
        let before_trim = before.trim_end();
        if regex_cache(r"(?i)(initial|starts?\s+at|from|begins?\s+at)\s*$")
            .is_match(before_trim)
        {
            p.push(("u", json!(q_value(q))));
            matched
                .entry("u".to_string())
                .or_insert_with(|| q.text.clone());
        } else if regex_cache(r"(?i)(final|reaches|to|tops?\s+at|ends?\s+at)\s*$").is_match(before_trim) {
            p.push(("v", json!(q_value(q))));
            matched
                .entry("v".to_string())
                .or_insert_with(|| q.text.clone());
        } else if !p.iter().any(|(k, _)| *k == "u") {
            p.push(("u", json!(q_value(q))));
            matched.insert("u".into(), format!("{} (taken as initial)", q.text));
        } else if !p.iter().any(|(k, _)| *k == "v") {
            p.push(("v", json!(q_value(q))));
            matched.insert("v".into(), format!("{} (taken as final)", q.text));
        }
    }
    if let Some(t) = first(qs, "s") {
        p.push(("t", json!(t.value)));
        matched.insert("t".into(), t.text.clone());
    }
    if let Some(dist) = first(qs, "m") {
        p.push(("s", json!(dist.value)));
        matched.insert("s".into(), dist.text.clone());
    }
    // free fall from a height: "dropped from 20 m" means s = 20 (not v)
    if has(
        text,
        &[
            r"dropped|falls?\s+from|fall(s)?\s+from\s+rest|free\s*fall",
        ],
    ) {
        if let Some(m) = regex_cache(
            r"(?i)(?:dropped|fall(?:s|ing)?|free\s*fall)\s+(?:from|a(?:t)?)\s+([-+]?\d+(?:\.\d+)?)\s*(m|meter|metre|km)\b",
        )
        .captures(text)
        {
            let val: f64 = m
                .get(1)
                .unwrap()
                .as_str()
                .parse()
                .unwrap();
            let unit = m.get(2).unwrap().as_str();
            let val = val * if unit.eq_ignore_ascii_case("km") { 1000.0 } else { 1.0 };
            p.retain(|(k, _)| *k != "s");
            p.push(("s", json!(val)));
            matched.insert("s".into(), m.get(0).unwrap().as_str().to_string());
            if !p.iter().any(|(k, _)| *k == "u") {
                p.push(("u", json!(0.0)));
            }
            if !p.iter().any(|(k, _)| *k == "a") {
                p.push(("a", json!(9.8)));
                matched
                    .entry("a".to_string())
                    .or_insert_with(|| "gravity default 9.8".to_string());
            }
        }
    }
    // what is asked
    let mut find: Option<&str> = None;
    if has(
        text,
        &[
            r"how\s+fast|final\s+(?:velocity|speed)|what\s+(?:is\s+the\s+)?(?:final\s+)?(?:velocity|speed)|impact\s+(?:speed|velocity)|tops?\s+speed",
        ],
    ) {
        find = Some("v");
    } else if has(text, &[r"how\s+long|how\s+much\s+time|what\s+time"]) {
        find = Some("t");
    } else if has(
        text,
        &[r"how\s+far|what\s+distance|distance\s+(?:does|did|it)\s+travel"],
    ) {
        find = Some("s");
    } else if has(
        text,
        &[r"what\s+acceleration|acceleration\s+of|deceleration"],
    ) {
        find = Some("a");
    } else if has(text, &[r"initial\s+(?:velocity|speed)"]) {
        find = Some("u");
    }
    if find.is_none() {
        let missing: Vec<&str> = ["u", "v", "a", "t", "s"]
            .iter()
            .filter(|k| !p.iter().any(|(pk, _)| *pk == **k))
            .copied()
            .collect();
        if missing.len() == 1 {
            find = Some(missing[0]);
        } else {
            return Some(NLParse::fail(&format!(
                "kinematics question but cannot tell which quantity is asked for (missing slots: {:?}); state 'what is the final velocity/speed', 'how long', 'how far', or 'what acceleration'",
                missing
            )));
        }
    }
    p.push(("find", json!(find.unwrap())));
    Some(NLParse::ok(
        "physics_kinematics",
        obj(p),
        "motion wording + SUVAT quantities",
        matched,
        0.9,
    ))
}

fn route_speed(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if has(text, &[r"accelerat|m/s\^?2|\u{00b2}"]) || first(qs, "m/s^2").is_some() {
        return None;
    }
    // direction-walk questions ("walks north 3 km") belong to the puzzle route
    if has(text, &[r"\b(?:north|south|east|west)\b"]) {
        return None;
    }
    if !(has(
        text,
        &[
            r"speed|how\s+fast|travel|drives?|rides?|runs?|walks?|km\s*/?\s*h|mph",
        ],
    ) || first(qs, "m/s").is_some())
    {
        return None;
    }
    let mut p: Vec<(&str, Value)> = Vec::new();
    let mut matched: HashMap<String, String> = HashMap::new();
    if let Some(v) = first(qs, "m/s") {
        p.push(("speed", json!(v.value)));
        matched.insert("speed".into(), v.text.clone());
    }
    if let Some(d) = first(qs, "m") {
        p.push(("distance", json!(d.value)));
        matched.insert("distance".into(), d.text.clone());
    }
    if let Some(t) = first(qs, "s") {
        p.push(("time", json!(t.value)));
        matched.insert("time".into(), t.text.clone());
    }
    let mut find: Option<&str> = None;
    if has(
        text,
        &[r"how\s+fast|what\s+(?:is\s+the\s+)?speed|average\s+speed"],
    ) {
        find = Some("speed");
    } else if has(text, &[r"how\s+far|what\s+distance"]) {
        find = Some("distance");
    } else if has(text, &[r"how\s+long|how\s+much\s+time|what\s+time"]) {
        find = Some("time");
    }
    if find.is_none() {
        let missing: Vec<&str> = ["speed", "distance", "time"]
            .iter()
            .filter(|k| !p.iter().any(|(pk, _)| *pk == **k))
            .copied()
            .collect();
        if missing.len() == 1 {
            find = Some(missing[0]);
        } else {
            return Some(NLParse::fail(&format!(
                "speed/distance/time question but the target is ambiguous (missing: {:?}); ask 'how fast', 'how far', or 'how long'",
                missing
            )));
        }
    }
    p.push(("find", json!(find.unwrap())));
    Some(NLParse::ok(
        "physics_speed",
        obj(p),
        "speed/distance/time wording",
        matched,
        0.85,
    ))
}

fn route_force(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"force|newton|\bN\b|friction|weight|push"]) {
        return None;
    }
    if has(text, &[r"kinetic|potential|momentum|collision"]) {
        return None;
    }
    let mut matched: HashMap<String, String> = HashMap::new();
    let mass = first(qs, "kg");
    let force = first(qs, "N");
    let mut mu: Option<f64> = None;
    if let Some(m) = regex_cache(
        r"(?i)(?:coefficient of (?:kinetic |static )?friction|mu|\u{03bc})\s*(?:is|=|of)?\s*(\d+(?:\.\d+)?)",
    )
    .captures(text)
    {
        mu = m.get(1).unwrap().as_str().parse().ok();
        matched.insert("mu".into(), m.get(0).unwrap().as_str().to_string());
    }
    if has(text, &[r"weight"]) {
        let Some(mass) = mass else { return None };
        let p = obj(vec![
            ("kind", json!("weight")),
            ("mass", json!(mass.value)),
        ]);
        matched.insert("mass".into(), mass.text.clone());
        return Some(NLParse::ok(
            "physics_force",
            p,
            "weight wording (W = m*g)",
            matched,
            0.9,
        ));
    }
    if has(text, &[r"friction"]) && mu.is_some() {
        let mut p: Vec<(&str, Value)> = vec![("kind", json!("friction")), ("mu", json!(mu.unwrap()))];
        if let Some(mass) = mass {
            p.push(("mass", json!(mass.value)));
            matched.insert("mass".into(), mass.text.clone());
            let find = if has(
                text,
                &[
                    r"friction\s+force|force of friction|how much friction|what friction",
                ],
            ) {
                "friction"
            } else if has(text, &[r"what (?:is the )?mass"]) {
                "mass"
            } else {
                "friction"
            };
            p.push(("find", json!(find)));
            if has(text, &[r"normal\s+force"]) {
                p.retain(|(k, _)| *k != "find");
                p.push(("find", json!("normal")));
            }
        } else if let Some(force) = force {
            p.push(("friction", json!(force.value)));
            matched.insert("friction".into(), force.text.clone());
            let find = if has(text, &[r"coefficient|mu|\u{03bc}"]) {
                Some("mu")
            } else {
                None
            };
            match find {
                Some(f) => p.push(("find", json!(f))),
                None => {
                    return Some(NLParse::fail(
                        "friction force given but unclear what is asked",
                    ))
                }
            }
        } else {
            return Some(NLParse::fail(
                "friction question needs a mass (or a friction force + coefficient)",
            ));
        }
        return Some(NLParse::ok(
            "physics_force",
            obj(p),
            "friction wording (f = mu*N, N = m*g on flat surface)",
            matched,
            0.85,
        ));
    }
    // plain F = m*a
    if let (Some(force), Some(mass)) = (force, mass) {
        let mut find = "acceleration";
        if has(
            text,
            &[
                r"what\s+(?:is\s+the\s+)?force|how\s+(?:much|many)\s+newtons?|force\s+(?:is\s+)?(?:needed|required|applied)",
            ],
        ) {
            find = "force";
        } else if has(text, &[r"what\s+(?:is\s+the\s+)?mass|how\s+(?:much|many)\s+(?:kg|kilograms?)"]) {
            find = "mass";
        }
        let p = obj(vec![
            ("find", json!(find)),
            ("force", json!(force.value)),
            ("mass", json!(mass.value)),
        ]);
        matched.insert("force".into(), force.text.clone());
        matched.insert("mass".into(), mass.text.clone());
        return Some(NLParse::ok(
            "physics_force",
            p,
            "force/mass/acceleration wording (F = m*a)",
            matched,
            0.85,
        ));
    }
    None
}

fn route_energy(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !has(
        text,
        &[
            r"kinetic|potential|energy|work|power|watt|joule|drops?|falls?\s+from|impact",
        ],
    ) {
        return None;
    }
    if has(text, &[r"momentum|collision"]) {
        return None;
    }
    let mut matched: HashMap<String, String> = HashMap::new();
    let mass = first(qs, "kg");
    let vel = first(qs, "m/s");
    let height = first(qs, "m");
    let j = first(qs, "J");
    let _w = first(qs, "W");
    let time = first(qs, "s");
    let force = first(qs, "N");
    if has(text, &[r"kinetic\s+energy|\bKE\b"]) {
        let (Some(mass), Some(vel)) = (mass, vel) else { return None };
        let p = obj(vec![
            ("form", json!("kinetic")),
            ("mass", json!(mass.value)),
            ("velocity", json!(vel.value)),
        ]);
        matched.insert("mass".into(), mass.text.clone());
        matched.insert("velocity".into(), vel.text.clone());
        return Some(NLParse::ok(
            "physics_energy",
            p,
            "kinetic energy (KE = 0.5*m*v^2)",
            matched,
            0.9,
        ));
    }
    if has(text, &[r"potential\s+energy|\bPE\b|gravitational"]) {
        let (Some(mass), Some(height)) = (mass, height) else { return None };
        let p = obj(vec![
            ("form", json!("potential")),
            ("mass", json!(mass.value)),
            ("height", json!(height.value)),
        ]);
        matched.insert("mass".into(), mass.text.clone());
        matched.insert("height".into(), height.text.clone());
        return Some(NLParse::ok(
            "physics_energy",
            p,
            "potential energy (PE = m*g*h)",
            matched,
            0.9,
        ));
    }
    if has(text, &[r"\bpower\b|watt"]) {
        if let (Some(j), Some(time)) = (j, time) {
            let p = obj(vec![
                ("form", json!("power")),
                ("work", json!(j.value)),
                ("time", json!(time.value)),
            ]);
            matched.insert("work".into(), j.text.clone());
            matched.insert("time".into(), time.text.clone());
            return Some(NLParse::ok(
                "physics_energy",
                p,
                "power (P = W/t or P = F*v)",
                matched,
                0.85,
            ));
        } else if let (Some(force), Some(vel)) = (force, vel) {
            let p = obj(vec![
                ("form", json!("power")),
                ("force", json!(force.value)),
                ("velocity", json!(vel.value)),
            ]);
            matched.insert("force".into(), force.text.clone());
            matched.insert("velocity".into(), vel.text.clone());
            return Some(NLParse::ok(
                "physics_energy",
                p,
                "power (P = W/t or P = F*v)",
                matched,
                0.85,
            ));
        } else {
            return Some(NLParse::fail(
                "power needs (work/energy + time) or (force + velocity)",
            ));
        }
    }
    if has(text, &[r"\bwork\b"]) {
        let (Some(force), Some(height)) = (force, height) else { return None };
        let p = obj(vec![
            ("form", json!("work")),
            ("force", json!(force.value)),
            ("distance", json!(height.value)),
        ]);
        matched.insert("force".into(), force.text.clone());
        matched.insert("distance".into(), height.text.clone());
        return Some(NLParse::ok(
            "physics_energy",
            p,
            "work (W = F*d)",
            matched,
            0.85,
        ));
    }
    if has(text, &[r"drops?|falls?\s+from|dropped|impact"]) {
        if let Some(m) = regex_cache(
            r"(?i)(?:dropped|drops?|falls?|falling)\s+from\s+(?:a\s+height\s+of\s+|a\s+)?([\d.]+)\s*(m|meter|metre|km|cm)\b",
        )
        .captures(text)
        {
            let val: f64 = m.get(1).unwrap().as_str().parse().unwrap();
            let factor = match m.get(2).unwrap().as_str().to_lowercase().as_str() {
                "km" => 1000.0,
                "cm" => 0.01,
                _ => 1.0,
            };
            let p = obj(vec![
                ("form", json!("impact_speed")),
                ("height", json!(val * factor)),
            ]);
            matched.insert("height".into(), m.get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "physics_energy",
                p,
                "free fall from a height (energy conservation, cross-checked vs kinematics)",
                matched,
                0.9,
            ));
        }
        if has(text, &[r"from\s+what\s+height|what\s+height"]) {
            let Some(vel) = vel else { return None };
            let p = obj(vec![("form", json!("height_for_speed")), ("speed", json!(vel.value))]);
            matched.insert("speed".into(), vel.text.clone());
            return Some(NLParse::ok(
                "physics_energy",
                p,
                "height needed for a given speed",
                matched,
                0.85,
            ));
        }
    }
    None
}

fn route_momentum(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"momentum|collide|collision|impulse|stick"]) {
        return None;
    }
    let masses = all(qs, "kg");
    let vels = all(qs, "m/s");
    let force = first(qs, "N");
    let time = first(qs, "s");
    let mut matched: HashMap<String, String> = HashMap::new();
    if has(text, &[r"impulse"]) && force.is_some() && time.is_some() && !masses.is_empty() {
        let mut p: Vec<(&str, Value)> = vec![
            ("kind", json!("impulse")),
            ("force", json!(force.unwrap().value)),
            ("dt", json!(time.unwrap().value)),
            ("mass", json!(masses[0].value)),
        ];
        matched.insert("force".into(), force.unwrap().text.clone());
        matched.insert("dt".into(), time.unwrap().text.clone());
        matched.insert("mass".into(), masses[0].text.clone());
        if has(text, &[r"from\s+rest|starts?\s+at\s+rest"]) {
            p.push(("v_initial", json!(0.0)));
        } else if let Some(m0) = regex_cache(r"(?i)moving\s+(?:at|with)\s+([-+]?\d+(?:\.\d+)?)")
            .captures(text)
        {
            let v: f64 = m0.get(1).unwrap().as_str().parse().unwrap();
            p.push(("v_initial", json!(v)));
        } else {
            p.push(("v_initial", json!(0.0)));
        }
        return Some(NLParse::ok(
            "physics_momentum",
            obj(p),
            "impulse-momentum (F*dt = m*dv)",
            matched,
            0.85,
        ));
    }
    if has(text, &[r"momentum"]) && !masses.is_empty() && !vels.is_empty() && masses.len() == 1 {
        let p = obj(vec![
            ("kind", json!("momentum")),
            ("mass", json!(masses[0].value)),
            ("velocity", json!(vels[0].value)),
        ]);
        matched.insert("mass".into(), masses[0].text.clone());
        matched.insert("velocity".into(), vels[0].text.clone());
        return Some(NLParse::ok(
            "physics_momentum",
            p,
            "momentum (p = m*v)",
            matched,
            0.9,
        ));
    }
    if masses.len() >= 2 && vels.len() >= 1 {
        let mut p: Vec<(&str, Value)> = vec![
            ("kind", json!(if has(text, &[r"elastic"]) { "elastic" } else { "inelastic" })),
            ("m1", json!(masses[0].value)),
            ("m2", json!(masses[1].value)),
        ];
        matched.insert("m1".into(), masses[0].text.clone());
        matched.insert("m2".into(), masses[1].text.clone());
        let v1 = vels[0].value;
        let mut v2: Option<f64> = if vels.len() > 1 { Some(vels[1].value) } else { None };
        if v2.is_none() {
            if has(text, &[r"at\s+rest|stationary"]) {
                v2 = Some(0.0);
            } else {
                return Some(NLParse::fail(
                    "collision needs a velocity for the second object",
                ));
            }
        }
        if has(
            text,
            &[r"toward each other|opposite direction|head-?on|approach"],
        ) {
            if v1 > 0.0 && v2.unwrap() > 0.0 {
                v2 = Some(-v2.unwrap());
            }
        }
        p.push(("v1", json!(v1)));
        p.push(("v2", json!(v2.unwrap())));
        matched.insert(
            "v1".into(),
            vels[0].text.clone(),
        );
        matched.insert(
            "v2".into(),
            if vels.len() > 1 { vels[1].text.clone() } else { "'at rest' -> 0".into() },
        );
        let kind_is_inelastic = !has(text, &[r"elastic"])
            && !has(text, &[r"stick|couple|together|inelastic|collision|collide"]);
        if kind_is_inelastic {
            return None;
        }
        let kind_str = if has(text, &[r"elastic"]) { "elastic" } else { "inelastic" };
        p.retain(|(k, _)| *k != "kind");
        p.insert(0, ("kind", json!(kind_str)));
        let rationale = if kind_str == "elastic" {
            "collision (elastic: both momentum and KE conserved)"
        } else {
            "collision (perfectly inelastic: objects stick, momentum conserved)"
        };
        return Some(NLParse::ok(
            "physics_momentum",
            obj(p),
            rationale,
            matched,
            0.85,
        ));
    }
    None
}

fn route_electricity(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !has(
        text,
        &[
            r"voltage|volt\b|current|amp|resistance|ohm|\u{03a9}|series|parallel|circuit",
        ],
    ) {
        return None;
    }
    let volts = all(qs, "V");
    let amps = all(qs, "A");
    let ohms = all(qs, "ohm");
    let mut matched: HashMap<String, String> = HashMap::new();
    if has(text, &[r"series|parallel"]) {
        if ohms.is_empty() {
            return Some(NLParse::fail(
                "series/parallel question needs resistor values in ohms",
            ));
        }
        let kind = if has(text, &[r"series"]) && !has(text, &[r"parallel"]) {
            "series"
        } else {
            "parallel"
        };
        let mut p: Vec<(&str, Value)> = vec![
            ("kind", json!(kind)),
            (
                "resistances",
                json!(ohms.iter().map(|o| o.value).collect::<Vec<f64>>()),
            ),
        ];
        matched.insert(
            "resistances".into(),
            ohms.iter().map(|o| o.text.clone()).collect::<Vec<_>>().join(", "),
        );
        if has(text, &[r"parallel"]) && has(text, &[r"series"]) {
            return Some(NLParse::fail(
                "both 'series' and 'parallel' mentioned; cannot disambiguate deterministically",
            ));
        }
        if !volts.is_empty() {
            p.push(("voltage", json!(volts[0].value)));
            matched.insert("voltage".into(), volts[0].text.clone());
        }
        return Some(NLParse::ok(
            "physics_electricity",
            obj(p),
            "resistor network wording",
            matched,
            0.9,
        ));
    }
    let mut find: Option<&str> = None;
    if has(
        text,
        &[
            r"what\s+(?:is\s+the\s+)?current|how\s+(?:much|many)\s+amps?",
        ],
    ) {
        find = Some("current");
    } else if has(text, &[r"what\s+(?:is\s+the\s+)?voltage|how\s+many\s+volts?"]) {
        find = Some("voltage");
    } else if has(
        text,
        &[r"what\s+(?:is\s+the\s+)?resistance|how\s+many\s+ohms?"],
    ) {
        find = Some("resistance");
    } else if has(text, &[r"power|watt"]) {
        let total = volts.len() + amps.len() + ohms.len();
        if total < 2 {
            return Some(NLParse::fail(
                "electrical power needs at least two of voltage, current, resistance",
            ));
        }
        let mut p: Vec<(&str, Value)> = vec![("kind", json!("power"))];
        if let Some(v) = volts.first() {
            p.push(("voltage", json!(v.value)));
            matched.insert("voltage".into(), v.text.clone());
        }
        if let Some(a) = amps.first() {
            p.push(("current", json!(a.value)));
            matched.insert("current".into(), a.text.clone());
        }
        if let Some(o) = ohms.first() {
            p.push(("resistance", json!(o.value)));
            matched.insert("resistance".into(), o.text.clone());
        }
        return Some(NLParse::ok(
            "physics_electricity",
            obj(p),
            "electrical power (three forms cross-checked)",
            matched,
            0.85,
        ));
    }
    if find.is_none() {
        let missing: Vec<&str> = [
            ("voltage", volts.is_empty()),
            ("current", amps.is_empty()),
            ("resistance", ohms.is_empty()),
        ]
        .iter()
        .filter(|(_, m)| *m)
        .map(|(n, _)| *n)
        .collect();
        if missing.len() == 1 {
            find = Some(missing[0]);
        } else {
            return Some(NLParse::fail(
                "Ohm's law question: state what is asked (current/voltage/resistance)",
            ));
        }
    }
    let mut p: Vec<(&str, Value)> = vec![
        ("kind", json!("ohm")),
        ("find", json!(find.unwrap())),
    ];
    if let Some(v) = volts.first() {
        p.push(("voltage", json!(v.value)));
        matched.insert("voltage".into(), v.text.clone());
    }
    if let Some(a) = amps.first() {
        p.push(("current", json!(a.value)));
        matched.insert("current".into(), a.text.clone());
    }
    if let Some(o) = ohms.first() {
        p.push(("resistance", json!(o.value)));
        matched.insert("resistance".into(), o.text.clone());
    }
    Some(NLParse::ok(
        "physics_electricity",
        obj(p),
        "Ohm's law (V = I*R)",
        matched,
        0.85,
    ))
}

fn route_density(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"density|pressure|floats?|sinks?|pascal|pa\b"]) {
        return None;
    }
    let masses = all(qs, "kg");
    let vols = all(qs, "m^3");
    let forces = all(qs, "N");
    let rhos = all(qs, "kg/m^3");
    let pascals = all(qs, "Pa");
    let heights = all(qs, "m");
    let mut matched: HashMap<String, String> = HashMap::new();
    if has(text, &[r"float|sink"]) && !rhos.is_empty() {
        let mut p: Vec<(&str, Value)> = vec![("kind", json!("float")), ("rho_object", json!(rhos[0].value))];
        matched.insert("rho_object".into(), rhos[0].text.clone());
        if rhos.len() > 1 {
            p.push(("rho_fluid", json!(rhos[1].value)));
            matched.insert("rho_fluid".into(), rhos[1].text.clone());
        }
        return Some(NLParse::ok(
            "physics_density",
            obj(p),
            "buoyancy float test (Archimedes)",
            matched,
            0.9,
        ));
    }
    if has(text, &[r"pressure"]) {
        if !rhos.is_empty() && !heights.is_empty() && forces.is_empty() {
            let p = obj(vec![
                ("kind", json!("pressure")),
                ("rho", json!(rhos[0].value)),
                ("height", json!(heights[0].value)),
            ]);
            matched.insert("rho".into(), rhos[0].text.clone());
            matched.insert("height".into(), heights[0].text.clone());
            if !vols.is_empty() {
                return None; // m^3 heights ambiguous
            }
            return Some(NLParse::ok(
                "physics_density",
                p,
                "hydrostatic pressure (P = rho*g*h, cross-checked vs F/A)",
                matched,
                0.9,
            ));
        }
        if !forces.is_empty() && forces.len() >= 2 {
            // force over area: "10 N over an area of 2 m^2"
            if let Some(m) = regex_cache(
                r"(?i)([\d.]+)\s*N\s*(?:over|on|across|distributed over)?\s*(?:an?|the)?\s*(?:area of\s*)?([\d.]+)\s*m\^?2",
            )
            .captures(text)
            {
                let f: f64 = m.get(1).unwrap().as_str().parse().unwrap();
                let a: f64 = m.get(2).unwrap().as_str().parse().unwrap();
                let p = obj(vec![
                    ("kind", json!("pressure")),
                    ("force", json!(f)),
                    ("area", json!(a)),
                ]);
                matched.insert("force/area".into(), m.get(0).unwrap().as_str().to_string());
                return Some(NLParse::ok(
                    "physics_density",
                    p,
                    "pressure (P = F/A)",
                    matched,
                    0.85,
                ));
            }
        }
        if !pascals.is_empty() && !forces.is_empty() {
            if regex_cache(r"(?i)([\d.]+)\s*m\^?2").is_match(text) {
                return None;
            }
        }
        return Some(NLParse::fail(
            "pressure question needs (rho + height) or (force + area)",
        ));
    }
    if has(text, &[r"density"]) && !masses.is_empty() && !vols.is_empty() {
        let p = obj(vec![
            ("kind", json!("density")),
            ("mass", json!(masses[0].value)),
            ("volume", json!(vols[0].value)),
        ]);
        matched.insert("mass".into(), masses[0].text.clone());
        matched.insert("volume".into(), vols[0].text.clone());
        return Some(NLParse::ok(
            "physics_density",
            p,
            "density (rho = m/V)",
            matched,
            0.9,
        ));
    }
    if has(text, &[r"density"]) && !masses.is_empty() && !rhos.is_empty() {
        let p = obj(vec![
            ("kind", json!("density")),
            ("mass", json!(masses[0].value)),
            ("rho", json!(rhos[0].value)),
        ]);
        matched.insert("mass".into(), masses[0].text.clone());
        matched.insert("rho".into(), rhos[0].text.clone());
        return Some(NLParse::ok(
            "physics_density",
            p,
            "volume from mass and density (V = m/rho)",
            matched,
            0.85,
        ));
    }
    if has(text, &[r"density"]) && !vols.is_empty() && !rhos.is_empty() {
        let p = obj(vec![
            ("kind", json!("density")),
            ("volume", json!(vols[0].value)),
            ("rho", json!(rhos[0].value)),
        ]);
        matched.insert("volume".into(), vols[0].text.clone());
        matched.insert("rho".into(), rhos[0].text.clone());
        return Some(NLParse::ok(
            "physics_density",
            p,
            "mass from volume and density (m = rho*V)",
            matched,
            0.85,
        ));
    }
    None
}

// ---------------- chemistry routes ----------------

/// Python `_looks_like_formula`.
fn looks_like_formula(token: &str) -> bool {
    if !regex_cache(r"^(?:[A-Z][a-z]?\d*)+(?:\((?:[A-Z][a-z]?\d*)+\)\d*)*$").is_match(token) {
        return false;
    }
    if !token.chars().any(|c| c.is_ascii_uppercase()) {
        return false;
    }
    let weights = reasoning_science::chemistry_domain::atomic_weights();
    for el in regex_cache(r"[A-Z][a-z]?").find_iter(token) {
        if !weights.contains_key(el.as_str()) {
            return false;
        }
    }
    true
}

/// Python `_split_chem_equation`.
fn split_chem_equation(text: &str) -> Option<(Vec<String>, Vec<String>)> {
    let arrows = ["->", "=>", "\u{2192}", "="];
    let mut sides: Option<(&str, &str)> = None;
    for arrow in arrows {
        if let Some(idx) = text.find(arrow) {
            sides = Some((&text[..idx], &text[idx + arrow.len()..]));
            break;
        }
    }
    let (left, right) = sides?;
    // the equation clause ends at the first sentence/prose boundary
    let right = regex_cache(r"(?i)[,.;?!]|\bif\b|\bhow\b|\bwhat\b|\bgiven\b|\bwith\b|\band\s+\d")
        .split(right)
        .next()
        .unwrap_or("");
    let species = |side: &str| -> Vec<String> {
        let mut out = Vec::new();
        for part in regex_cache(r"\s*\+\s*").split(side) {
            let part = regex_cache(r"^\s*\d+\s*").replace(part.trim(), "");
            let part = part.as_ref();
            if let Some(tok) = form_pat().find_iter(part).find(|m| looks_like_formula(m.as_str())) {
                out.push(tok.as_str().to_string());
            }
        }
        out
    };
    let (left_s, right_s) = (species(left), species(right));
    if !left_s.is_empty() && !right_s.is_empty() {
        Some((left_s, right_s))
    } else {
        None
    }
}

fn route_chem_molar_mass(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(
        text,
        &[
            r"molar\s+mass|molecular\s+(?:mass|weight)|formula\s+mass|mass\s+of\s+(?:one\s+)?mole",
        ],
    ) {
        return None;
    }
    let formulas: Vec<String> = form_pat()
        .find_iter(text)
        .map(|m| m.as_str().to_string())
        .filter(|f| looks_like_formula(f))
        .collect();
    if formulas.is_empty() {
        return Some(NLParse::fail(
            "molar mass question but no chemical formula recognized",
        ));
    }
    let f = formulas[0].clone();
    let mut matched = HashMap::new();
    matched.insert("formula".to_string(), f.clone());
    Some(NLParse::ok(
        "chem_molar_mass",
        obj(vec![("formula", json!(f.clone()))]),
        "molar mass wording + formula token",
        matched,
        0.95,
    ))
}

fn route_chem_balance(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"->|=>|\u{2192}|=|balance"]) {
        return None;
    }
    let sides = split_chem_equation(text)?;
    let equation = format!("{} -> {}", sides.0.join(" + "), sides.1.join(" + "));
    if has(text, &[r"limiting"]) {
        return None; // limiting route handles it
    }
    let mut matched = HashMap::new();
    matched.insert("equation".to_string(), equation.clone());
    Some(NLParse::ok(
        "chem_balance",
        obj(vec![("equation", json!(equation))]),
        "chemical equation with arrow detected",
        matched,
        0.95,
    ))
}

fn route_chem_limiting(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(
        text,
        &[r"limiting|how\s+(?:many|much)\s+(?:grams?|moles?)\s+of\s+[A-Z]"],
    ) {
        return None;
    }
    let sides = match split_chem_equation(text) {
        Some(s) => s,
        None => {
            return Some(NLParse::fail(
                "limiting-reagent question needs the balanced equation with an arrow",
            ))
        }
    };
    let equation = format!("{} -> {}", sides.0.join(" + "), sides.1.join(" + "));
    let mut moles: HashMap<String, f64> = HashMap::new();
    let mut matched: HashMap<String, String> = HashMap::new();
    for m in regex_cache(
        r"(?i)([\d.]+)\s*(?:moles?|mols?|mol)\s*(?:of\s*)?([A-Z][A-Za-z0-9()]*)",
    )
    .captures_iter(text)
    {
        let f = m.get(2).unwrap().as_str().to_string();
        if looks_like_formula(&f) {
            let v: f64 = m.get(1).unwrap().as_str().parse().unwrap();
            moles.entry(f.clone()).or_insert(v);
            matched.insert(format!("moles {}", f), m.get(0).unwrap().as_str().to_string());
        }
    }
    let mut grams: HashMap<String, f64> = HashMap::new();
    for m in regex_cache(r"(?i)([\d.]+)\s*g(?:rams?)?\s*(?:of\s*)?([A-Z][A-Za-z0-9()]*)")
        .captures_iter(text)
    {
        let f = m.get(2).unwrap().as_str().to_string();
        if looks_like_formula(&f) {
            let v: f64 = m.get(1).unwrap().as_str().parse().unwrap();
            grams.entry(f.clone()).or_insert(v);
            matched.insert(format!("grams {}", f), m.get(0).unwrap().as_str().to_string());
        }
    }
    if moles.is_empty() && grams.is_empty() {
        return Some(NLParse::fail(
            "limiting-reagent question needs amounts like '4 moles of H2'",
        ));
    }
    // product asked: "how many grams of X" pattern target
    let m = regex_cache(r"(?i)how\s+(?:many|much)\s+(?:grams?|moles?)\s+of\s+([A-Z][A-Za-z0-9()]*)")
        .captures(text);
    let want = match m {
        Some(m) if looks_like_formula(m.get(1).unwrap().as_str()) => m,
        _ => {
            return Some(NLParse::fail(
                "cannot tell which product is asked for",
            ))
        }
    };
    let product = want.get(1).unwrap().as_str().to_string();
    let want_mass = want.get(0).unwrap().as_str().to_lowercase().contains("gram");
    let mut payload = obj(vec![
        ("equation", json!(equation)),
        ("product", json!(product)),
        ("_want_mass", json!(want_mass)),
        ("_grams", json!(&grams)),
    ]);
    if let Some(p) = payload.get_mut("_grams") {
        let mut gm = Map::new();
        for (k, v) in &grams {
            gm.insert(k.clone(), json!(v));
        }
        *p = Value::Object(gm);
    }
    Some(NLParse::ok(
        "chem_limiting_nl",
        payload,
        "limiting-reagent wording with equation + amounts",
        matched,
        0.9,
    ))
}

fn route_chem_gas(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !(has(
        text,
        &[
            r"ideal\s+gas|gas\s+law|PV\s*=\s*nRT|boyle|charles|gay.?lussac|\bgas\b",
        ],
    ) || (first(qs, "atm").is_some() && first(qs, "L").is_some()))
    {
        return None;
    }
    let mut temps: Vec<&Quantity> = all(qs, "K");
    temps.extend(all(qs, "C"));
    let atms = all(qs, "atm");
    let pas = all(qs, "Pa");
    let vols = all(qs, "L");
    let mols = all(qs, "mol");
    let mut matched: HashMap<String, String> = HashMap::new();
    if has(text, &[r"boyle"]) {
        let mut p: Vec<(&str, Value)> = vec![("kind", json!("boyle"))];
        if atms.len() >= 2 {
            p.push(("p1", json!(atms[0].value)));
            p.push(("find", json!("v2")));
            p.push(("p2", json!(atms[1].value)));
            matched.insert("p1".into(), atms[0].text.clone());
            matched.insert("p2".into(), atms[1].text.clone());
            if let Some(v) = vols.first() {
                p.push(("v1", json!(v.value)));
                matched.insert("v1".into(), v.text.clone());
            }
        } else if vols.len() >= 2 {
            p.push(("v1", json!(vols[0].value)));
            p.push(("find", json!("p2")));
            p.push(("v2", json!(vols[1].value)));
            matched.insert("v1".into(), vols[0].text.clone());
            matched.insert("v2".into(), vols[1].text.clone());
            if let Some(a) = atms.first() {
                p.push(("p1", json!(a.value)));
                matched.insert("p1".into(), a.text.clone());
            }
        } else {
            return Some(NLParse::fail(
                "Boyle's law needs two of {P1,V1,P2,V2}",
            ));
        }
        return Some(NLParse::ok(
            "chem_gas",
            obj(p),
            "Boyle's law (P1*V1 = P2*V2)",
            matched,
            0.9,
        ));
    }
    let mut p: Vec<(&str, Value)> = vec![("kind", json!("ideal"))];
    let pressure = atms
        .first()
        .map(|a| a.value)
        .or_else(|| pas.first().map(|pa| pa.value / 101325.0));
    if let Some(pa) = pas.first() {
        matched.insert(
            "pressure".into(),
            format!("{} (converted to atm)", pa.text),
        );
    } else if let Some(a) = atms.first() {
        matched.insert("pressure".into(), a.text.clone());
    }
    if let Some(v) = vols.first() {
        p.push(("volume", json!(v.value)));
        matched.insert("volume".into(), v.text.clone());
    }
    if let Some(m) = mols.first() {
        p.push(("moles", json!(m.value)));
        matched.insert("moles".into(), m.text.clone());
    }
    if let Some(t) = temps.first() {
        let tval = crate::quantities::to_kelvin(t).unwrap_or(t.value);
        p.push(("temperature", json!(tval)));
        matched.insert(
            "temperature".into(),
            format!(
                "{}{}",
                t.text,
                if t.unit == "C" { " (+273.15 -> K)" } else { "" }
            ),
        );
    }
    if let Some(pressure) = pressure {
        p.push(("pressure", json!(pressure)));
    }
    let mut find: Option<&str> = None;
    if has(
        text,
        &[r"what\s+(?:is\s+the\s+)?volume|how\s+many\s+liters?"],
    ) {
        find = Some("volume");
    } else if has(text, &[r"what\s+(?:is\s+the\s+)?pressure"]) {
        find = Some("pressure");
    } else if has(text, &[r"how\s+many\s+(?:moles?|mols?)"]) {
        find = Some("moles");
    } else if has(text, &[r"what\s+(?:is\s+the\s+)?temperature"]) {
        find = Some("temperature");
    }
    if find.is_none() {
        let missing: Vec<&str> = ["pressure", "volume", "moles", "temperature"]
            .iter()
            .filter(|k| !p.iter().any(|(pk, _)| *pk == **k))
            .copied()
            .collect();
        if missing.len() == 1 {
            find = Some(missing[0]);
        } else {
            return Some(NLParse::fail(&format!(
                "gas law needs three of P/V/n/T (missing: {:?})",
                missing
            )));
        }
    }
    p.push(("find", json!(find.unwrap())));
    Some(NLParse::ok(
        "chem_gas",
        obj(p),
        "ideal gas law (P*V = n*R*T)",
        matched,
        0.85,
    ))
}

fn route_chem_solutions(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if has(text, &[r"\bpH\b|\bpOH\b"]) {
        return None;
    }
    if has(
        text,
        &[
            r"percent\s+composition|percentage\s+composition|percent\s+by\s+mass",
        ],
    ) {
        let formulas: Vec<String> = form_pat()
            .find_iter(text)
            .map(|m| m.as_str().to_string())
            .filter(|f| looks_like_formula(f))
            .collect();
        if formulas.is_empty() {
            return Some(NLParse::fail("percent composition needs a formula"));
        }
        let f = formulas[0].clone();
        let mut matched = HashMap::new();
        matched.insert("formula".to_string(), f.clone());
        return Some(NLParse::ok(
            "chem_solutions",
            obj(vec![
                ("kind", json!("percent_composition")),
                ("formula", json!(f)),
            ]),
            "percent composition wording",
            matched,
            0.95,
        ));
    }
    if has(text, &[r"percent\s+yield|percentage\s+yield|%?\s*yield"]) {
        let m = regex_cache(r"(?i)([\d.]+)\s*g(?:rams?)?\s+actual")
            .captures(text)
            .or_else(|| regex_cache(r"(?i)actual\s+(?:yield\s+)?(?:is\s+|of\s+)?([\d.]+)").captures(text));
        let t = regex_cache(r"(?i)([\d.]+)\s*g(?:rams?)?\s+theoretical")
            .captures(text)
            .or_else(|| regex_cache(r"(?i)theoretical\s+(?:yield\s+)?(?:is\s+|of\s+)?([\d.]+)").captures(text));
        let (Some(m), Some(t)) = (m, t) else {
            return Some(NLParse::fail(
                "percent yield needs an actual and a theoretical amount",
            ))
        };
        let actual: f64 = m.get(1).unwrap().as_str().parse().unwrap();
        let theoretical: f64 = t.get(1).unwrap().as_str().parse().unwrap();
        let mut matched = HashMap::new();
        matched.insert("actual".into(), m.get(0).unwrap().as_str().to_string());
        matched.insert("theoretical".into(), t.get(0).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "chem_solutions",
            obj(vec![
                ("kind", json!("percent_yield")),
                ("actual", json!(actual)),
                ("theoretical", json!(theoretical)),
            ]),
            "percent yield wording",
            matched,
            0.9,
        ));
    }
    if has(text, &[r"dilut"]) {
        // M1V1 -> M2V2: two molarities and two volumes
        let molars: Vec<f64> = regex_cache(r"(?i)([\d.]+)\s*(?:M|mol/L|moles per liter)")
            .captures_iter(text)
            .filter_map(|m| m.get(1).unwrap().as_str().parse().ok())
            .collect();
        let vols = all(qs, "L");
        if molars.len() >= 2 && vols.len() >= 2 {
            let mut matched = HashMap::new();
            matched.insert("m1".into(), molars[0].to_string());
            matched.insert("m2".into(), molars[1].to_string());
            matched.insert("v1".into(), vols[0].text.clone());
            matched.insert("v2".into(), vols[1].text.clone());
            return Some(NLParse::ok(
                "chem_solutions",
                obj(vec![
                    ("kind", json!("dilution")),
                    ("m1", json!(molars[0])),
                    ("v1", json!(vols[0].value)),
                    ("m2", json!(molars[1])),
                    ("v2", json!(vols[1].value)),
                ]),
                "dilution wording (M1*V1 = M2*V2)",
                matched,
                0.85,
            ));
        }
        return Some(NLParse::fail(
            "dilution needs two (molarity, volume) pairs",
        ));
    }
    if has(text, &[r"molar(?:ity)?|concentration|mol/L"]) {
        let vols = all(qs, "L");
        let grams = regex_cache(r"(?i)([\d.]+)\s*g(?:rams?)?\s*(?:of\s*)?([A-Z][A-Za-z0-9()]*)").captures(text);
        let mol_given = regex_cache(r"([\d.]+)\s*(?:moles?|mol)\s*(?:of\s*)?([A-Z][A-Za-z0-9()]*)").captures(text);
        let mut payload: Vec<(&str, Value)> = vec![("kind", json!("molarity"))];
        let mut matched: HashMap<String, String> = HashMap::new();
        if let Some(mg) = mol_given.as_ref().filter(|m| looks_like_formula(m.get(2).unwrap().as_str())) {
            let n: f64 = mg.get(1).unwrap().as_str().parse().unwrap();
            payload.push(("moles", json!(n)));
            matched.insert("moles".into(), mg.get(0).unwrap().as_str().to_string());
        } else if let Some(g) = grams.as_ref().filter(|m| looks_like_formula(m.get(2).unwrap().as_str())) {
            let mass: f64 = g.get(1).unwrap().as_str().parse().unwrap();
            payload.push(("mass_g", json!(mass)));
            payload.push(("formula", json!(g.get(2).unwrap().as_str())));
            matched.insert("mass".into(), g.get(0).unwrap().as_str().to_string());
        } else {
            return Some(NLParse::fail(
                "molarity needs an amount (moles or grams of a formula)",
            ));
        }
        if let Some(v) = vols.first() {
            payload.push(("volume_l", json!(v.value)));
            matched.insert("volume".into(), v.text.clone());
            return Some(NLParse::ok(
                "chem_solutions",
                obj(payload),
                "molarity (M = n/V)",
                matched,
                0.9,
            ));
        }
        let molars2: Vec<f64> = regex_cache(r"(?i)([\d.]+)\s*(?:M|mol/L)\b")
            .captures_iter(text)
            .filter_map(|m| m.get(1).unwrap().as_str().parse().ok())
            .collect();
        if let Some(mm) = molars2.first() {
            payload.push(("molarity", json!(mm)));
            matched.insert("molarity".into(), format!("{} M", mm));
            return Some(NLParse::ok(
                "chem_solutions",
                obj(payload),
                "volume from moles and molarity (V = n/M)",
                matched,
                0.85,
            ));
        }
        return Some(NLParse::fail(
            "molarity question needs a volume (or a molarity to solve for volume)",
        ));
    }
    None
}

fn route_chem_ph(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(
        text,
        &[
            r"\bpH\b|\bpOH\b|\[H\+?\]|\[OH-?\]|hydrogen\s+ion|acidity",
        ],
    ) {
        return None;
    }
    if has(text, &[r"neutrali[sz]|titrat"]) {
        let m_acid = regex_cache(r"(?i)([\d.]+)\s*(?:moles?|mol)\s*(?:of\s*)?(?:HCl|acid)").captures(text);
        let m_base = regex_cache(r"(?i)([\d.]+)\s*(?:moles?|mol)\s*(?:of\s*)?(?:NaOH|base)").captures(text);
        let (Some(m_acid), Some(m_base)) = (m_acid, m_base) else {
            return Some(NLParse::fail(
                "neutralization needs moles of acid and moles of base",
            ))
        };
        let protons = if has(text, &[r"H2SO4|diprotic|sulfuric"]) { 2 } else { 1 };
        let base_oh = if has(
            text,
            &[
                r"Ca\(OH\)2|Ba\(OH\)2|dihydrox|barium\s+hydroxide|calcium\s+hydroxide",
            ],
        ) {
            2
        } else {
            1
        };
        let moles_acid: f64 = m_acid.get(1).unwrap().as_str().parse().unwrap();
        let moles_base: f64 = m_base.get(1).unwrap().as_str().parse().unwrap();
        let mut matched = HashMap::new();
        matched.insert("acid".into(), m_acid.get(0).unwrap().as_str().to_string());
        matched.insert("base".into(), m_base.get(0).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "chem_ph",
            obj(vec![
                ("kind", json!("neutralization")),
                ("moles_acid", json!(moles_acid)),
                ("acid_protons", json!(protons)),
                ("moles_base", json!(moles_base)),
                ("base_oh", json!(base_oh)),
            ]),
            "neutralization wording",
            matched,
            0.85,
        ));
    }
    let m_h = regex_cache(r"\[H\+?\]\s*(?:is|=|of)?\s*([\d.eE+-]+)")
        .captures(text)
        .or_else(|| {
            regex_cache(r"hydrogen\s+ion\s+concentration\s*(?:is|=|of)?\s*([\d.eE+-]+)").captures(text)
        });
    let m_ph = regex_cache(r"(?i)\bpH\s*(?:is|=|of)?\s*([\d.]+)").captures(text);
    let m_poh = regex_cache(r"(?i)\bpOH\s*(?:is|=|of)?\s*([\d.]+)").captures(text);
    let h_val = m_h.as_ref().and_then(|m| m.get(1)).map(|g| g.as_str().parse::<f64>().unwrap());
    let h_text = m_h.as_ref().map(|m| m.get(0).unwrap().as_str().to_string());
    let ph_val = m_ph.as_ref().and_then(|m| m.get(1)).map(|g| g.as_str().parse::<f64>().unwrap());
    let ph_text = m_ph.as_ref().map(|m| m.get(0).unwrap().as_str().to_string());
    let poh_val = m_poh.as_ref().and_then(|m| m.get(1)).map(|g| g.as_str().parse::<f64>().unwrap());
    let poh_text = m_poh.as_ref().map(|m| m.get(0).unwrap().as_str().to_string());
    if has(text, &[r"what\s+(?:is\s+the\s+)?pH"]) && h_val.is_some() {
        let h: f64 = h_val.unwrap();
        let mut matched = HashMap::new();
        matched.insert("[H+]".into(), h_text.clone().unwrap());
        return Some(NLParse::ok(
            "chem_ph",
            obj(vec![("kind", json!("from_concentration")), ("h", json!(h))]),
            "pH from [H+]",
            matched,
            0.95,
        ));
    }
    if has(text, &[r"\[H\+?\]|hydrogen\s+ion"]) && ph_val.is_some() {
        let ph: f64 = ph_val.unwrap();
        let mut matched = HashMap::new();
        matched.insert("pH".into(), ph_text.clone().unwrap());
        return Some(NLParse::ok(
            "chem_ph",
            obj(vec![("kind", json!("from_ph")), ("pH", json!(ph))]),
            "[H+] from pH",
            matched,
            0.95,
        ));
    }
    if has(text, &[r"\bpOH\b"]) && ph_val.is_some() {
        let ph: f64 = ph_val.unwrap();
        let mut matched = HashMap::new();
        matched.insert("pH".into(), ph_text.clone().unwrap());
        return Some(NLParse::ok(
            "chem_ph",
            obj(vec![("kind", json!("pair")), ("pH", json!(ph))]),
            "pH/pOH pair",
            matched,
            0.95,
        ));
    }
    if has(text, &[r"\bpH\b"]) && poh_val.is_some() {
        let poh: f64 = poh_val.unwrap();
        let mut matched = HashMap::new();
        matched.insert("pOH".into(), poh_text.clone().unwrap());
        return Some(NLParse::ok(
            "chem_ph",
            obj(vec![("kind", json!("pair")), ("pOH", json!(poh))]),
            "pH/pOH pair",
            matched,
            0.95,
        ));
    }
    Some(NLParse::fail(
        "pH question but no usable (pH | pOH | [H+]) value found",
    ))
}

// ---------------- biology routes ----------------

fn geno2() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"\b([A-Z][a-z])\s*(?:x|\u{00d7}|crossed with|with|by)\s*([A-Z][a-z])\b").unwrap()
    })
}

fn geno4() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"\b([A-Z][a-z][A-Z][a-z])\s*(?:x|\u{00d7}|crossed with|with|by)\s*([A-Z][a-z][A-Z][a-z])\b")
            .unwrap()
    })
}

fn route_bio_genetics(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if has(text, &[r"hardy.?weinberg"]) {
        return None;
    }
    if !has(
        text,
        &[
            r"cross|offspring|punnett|genotype|phenotype|dominant|recessive|heterozyg|homozyg|testcross|gamet",
        ],
    ) {
        return None;
    }
    if let Some(m4) = geno4().captures(text) {
        if m4.get(1).unwrap().as_str().len() == 4 {
            let pa = m4.get(1).unwrap().as_str().to_string();
            let pb = m4.get(2).unwrap().as_str().to_string();
            let mut matched = HashMap::new();
            matched.insert("cross".into(), m4.get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "bio_dihybrid",
                obj(vec![("parent_a", json!(pa)), ("parent_b", json!(pb))]),
                "dihybrid genotype pair",
                matched,
                0.95,
            ));
        }
    }
    if let Some(m2) = geno2().captures(text) {
        let pa = m2.get(1).unwrap().as_str().to_string();
        let pb = m2.get(2).unwrap().as_str().to_string();
        // Python's condition: pa[0] != pb[0] or (pa[0].lower() != pb[1].lower() and pa[0].lower() != pb[0].lower())
        let pa0 = pa.chars().next().unwrap();
        let pb0 = pb.chars().next().unwrap();
        let pb1 = pb.chars().nth(1).unwrap();
        if pa0 != pb0 || (pa0.to_lowercase().eq(pb1.to_lowercase()) == false
            && pa0.to_lowercase().eq(pb0.to_lowercase()) == false)
        {
            return Some(NLParse::fail(&format!(
                "genotypes {} and {} are for different genes",
                pa, pb
            )));
        }
        let mut matched = HashMap::new();
        matched.insert("cross".into(), m2.get(0).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "bio_monohybrid",
            obj(vec![("parent_a", json!(pa)), ("parent_b", json!(pb))]),
            "monohybrid genotype pair",
            matched,
            0.95,
        ));
    }
    // wording-based: heterozygous/homozygous
    let letters: Vec<char> = regex_cache(r"\bgene\s+([A-Z])\b")
        .captures_iter(text)
        .filter_map(|m| m.get(1).unwrap().as_str().chars().next())
        .collect();
    let letter = letters.first().copied().unwrap_or('A');
    let mut seq: Vec<String> = Vec::new();
    for m in regex_cache(r"(?i)heterozygous|homozygous\s+dominant|homozygous\s+recessive")
        .captures_iter(text)
    {
        let w = m.get(0).unwrap().as_str().to_lowercase();
        let lower = letter.to_lowercase().next().unwrap();
        if w.starts_with("hetero") {
            seq.push(format!("{}{}", letter, lower));
        } else if w.contains("dominant") {
            seq.push(format!("{}{}", letter, letter));
        } else {
            seq.push(format!("{}{}", lower, lower));
        }
    }
    if seq.len() >= 2 {
        let mut matched = HashMap::new();
        matched.insert("cross".into(), format!("{} x {}", seq[0], seq[1]));
        return Some(NLParse::ok(
            "bio_monohybrid",
            obj(vec![("parent_a", json!(seq[0])), ("parent_b", json!(seq[1]))]),
            "genotype words -> allele pairs",
            matched,
            0.9,
        ));
    }
    None
}

fn route_bio_hardy_weinberg(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"hardy.?weinberg"]) {
        return None;
    }
    let m = regex_cache(
        r"(?i)([\d.]+)\s*%\s*(?:of\s+the\s+population\s+)?(?:shows?|have|has|exhibits?|displays?)\s*(?:the\s+)?recessive",
    )
    .captures(text)
    .or_else(|| {
        regex_cache(r"(?i)recessive\s+(?:trait|phenotype)\s*(?:is|=|in)?\s*([\d.]+)\s*%").captures(text)
    });
    if let Some(m) = m {
        let q2: f64 = m.get(1).unwrap().as_str().parse::<f64>().unwrap() / 100.0;
        let mut matched = HashMap::new();
        matched.insert("q^2".into(), m.get(0).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "bio_hardy_weinberg",
            obj(vec![("q_squared", json!(q2))]),
            "Hardy-Weinberg from recessive fraction",
            matched,
            0.95,
        ));
    }
    if let Some(m) = regex_cache(r"q\^?2?\s*(?:is|=)?\s*([\d.]+)").captures(text) {
        if has(text, &[r"q\^?2|q squared"]) {
            let q2: f64 = m.get(1).unwrap().as_str().parse().unwrap();
            let mut matched = HashMap::new();
            matched.insert("q^2".into(), m.get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "bio_hardy_weinberg",
                obj(vec![("q_squared", json!(q2))]),
                "Hardy-Weinberg from q^2",
                matched,
                0.95,
            ));
        }
    }
    let dom = regex_cache(r"(?i)([\d,]+)\s*(?:dominant\s+alleles?|copies of the dominant allele)").captures(text);
    let rec = regex_cache(r"(?i)([\d,]+)\s*(?:recessive\s+alleles?|copies of the recessive allele)").captures(text);
    if let (Some(dom), Some(rec)) = (dom, rec) {
        let d: i64 = dom.get(1).unwrap().as_str().replace(",", "").parse().unwrap();
        let r: i64 = rec.get(1).unwrap().as_str().replace(",", "").parse().unwrap();
        let mut matched = HashMap::new();
        matched.insert("dom".into(), dom.get(0).unwrap().as_str().to_string());
        matched.insert("rec".into(), rec.get(0).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "bio_hardy_weinberg",
            obj(vec![
                ("dominant_alleles", json!(d)),
                ("recessive_alleles", json!(r)),
            ]),
            "Hardy-Weinberg from allele census",
            matched,
            0.9,
        ));
    }
    Some(NLParse::fail(
        "Hardy-Weinberg question needs q^2/recessive-% or allele counts",
    ))
}

fn route_bio_dogma(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    let seq_m = regex_cache(r"\b([ACGTUacgtu]{6,})\b").captures(text);
    if has(text, &[r"reverse\s+complement"]) {
        let Some(seq_m) = seq_m else {
            return Some(NLParse::fail("reverse complement needs a DNA sequence"))
        };
        let dna = seq_m.get(1).unwrap().as_str().to_uppercase();
        let mut matched = HashMap::new();
        matched.insert("dna".into(), seq_m.get(1).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "bio_dogma",
            obj(vec![("kind", json!("reverse_complement")), ("dna", json!(dna))]),
            "reverse-complement wording",
            matched,
            0.95,
        ));
    }
    if has(text, &[r"transcrib|mRNA\s+(?:from|sequence)|DNA\s+to\s+(?:m)?RNA"]) {
        let Some(seq_m) = seq_m else {
            return Some(NLParse::fail("transcription needs a DNA sequence"))
        };
        let dna = seq_m.get(1).unwrap().as_str().to_uppercase();
        let mut matched = HashMap::new();
        matched.insert("dna".into(), seq_m.get(1).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "bio_dogma",
            obj(vec![("kind", json!("transcribe")), ("dna", json!(dna))]),
            "transcription wording",
            matched,
            0.95,
        ));
    }
    if has(text, &[r"translat|codon|amino\s+acid|protein\s+sequence"]) {
        let Some(seq_m) = seq_m else {
            return Some(NLParse::fail("translation needs a sequence"))
        };
        let seq = seq_m.get(1).unwrap().as_str().to_uppercase();
        let mut matched = HashMap::new();
        matched.insert("seq".into(), seq_m.get(1).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "bio_dogma",
            obj(vec![("kind", json!("translate")), ("seq", json!(seq))]),
            "translation wording",
            matched,
            0.95,
        ));
    }
    if has(
        text,
        &[
            r"base\s+composition|GC\s+content|gc\s+percent|nucleotide\s+(?:counts|composition)",
        ],
    ) {
        let Some(seq_m) = seq_m else {
            return Some(NLParse::fail("composition needs a sequence"))
        };
        let seq = seq_m.get(1).unwrap().as_str().to_uppercase();
        let ds = has(text, &[r"double[- ]stranded|duplex"]);
        let mut matched = HashMap::new();
        matched.insert("seq".into(), seq_m.get(1).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "bio_dogma",
            obj(vec![
                ("kind", json!("composition")),
                ("seq", json!(seq)),
                ("double_stranded", json!(ds)),
            ]),
            "base composition wording",
            matched,
            0.9,
        ));
    }
    None
}

fn route_bio_ecology(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(
        text,
        &[
            r"population|bacteria|trophic|carrying\s+capacity|exponential|doubl",
        ],
    ) {
        return None;
    }
    if has(text, &[r"trophic"]) {
        let m_e = regex_cache(
            r"(?i)([\d,\.]+)\s*(?:kJ|kJ?oules?|kcal|calories)\s*(?:of)?\s*(?:energy)?\s*(?:at|in|from)?\s*(?:the\s*)?(?:producer|first\s+trophic)",
        )
        .captures(text);
        let m_l = regex_cache(r"(?i)(?:trophic\s+level|level)\s*(\d)").captures(text);
        if let (Some(m_e), Some(m_l)) = (m_e, m_l) {
            let energy: f64 = m_e.get(1).unwrap().as_str().replace(",", "").parse().unwrap();
            let level: i64 = m_l.get(1).unwrap().as_str().parse().unwrap();
            let mut matched = HashMap::new();
            matched.insert("energy".into(), m_e.get(0).unwrap().as_str().to_string());
            matched.insert("level".into(), m_l.get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "bio_ecology",
                obj(vec![
                    ("kind", json!("energy_transfer")),
                    ("producer_energy", json!(energy)),
                    ("trophic_level", json!(level)),
                ]),
                "10% trophic transfer",
                matched,
                0.9,
            ));
        }
        return Some(NLParse::fail(
            "trophic question needs producer energy and a target level",
        ));
    }
    if has(text, &[r"carrying\s+capacity"]) {
        let n0 = regex_cache(r"(?i)(?:population\s+of|starts? with)\s*([\d,\.]+)").captures(text);
        let k = regex_cache(r"(?i)carrying\s+capacity\s+(?:of|is)?\s*([\d,\.]+)").captures(text);
        let r = regex_cache(r"(?i)(?:rate\s+of\s+)?([\d.]+)\s*(?:per\s+year|/year|per\s+capita)").captures(text);
        let t = regex_cache(r"(?i)(?:after|in)\s+([\d.]+)\s*(?:years?|yrs?)").captures(text);
        if n0.is_some() && k.is_some() && r.is_some() && t.is_some() {
            let mut matched = HashMap::new();
            for (key, m) in [("n0", &n0), ("k", &k), ("r", &r), ("t", &t)] {
                matched.insert(key.to_string(), m.as_ref().unwrap().get(0).unwrap().as_str().to_string());
            }
            let n0v: f64 = n0.as_ref().unwrap().get(1).unwrap().as_str().replace(",", "").parse().unwrap();
            let kv: f64 = k.as_ref().unwrap().get(1).unwrap().as_str().replace(",", "").parse().unwrap();
            let rv: f64 = r.as_ref().unwrap().get(1).unwrap().as_str().parse().unwrap();
            let tv: f64 = t.as_ref().unwrap().get(1).unwrap().as_str().parse().unwrap();
            return Some(NLParse::ok(
                "bio_ecology",
                obj(vec![
                    ("kind", json!("logistic")),
                    ("n0", json!(n0v)),
                    ("k", json!(kv)),
                    ("rate", json!(rv)),
                    ("time", json!(tv)),
                ]),
                "logistic growth wording",
                matched,
                0.9,
            ));
        }
        return Some(NLParse::fail(
            "logistic growth needs N0, K, rate, and time",
        ));
    }
    if has(text, &[r"population|bacteria|exponential|doubl"]) {
        let n0 = regex_cache(r"(?i)(?:population\s+of|starts? with|begins? with)\s*([\d,\.]+)").captures(text);
        let r_pct = regex_cache(
            r"(?i)(?:grows?|grow|rate(?:\s+of)?)\s*(?:at)?\s*([\d.]+)\s*%\s*(?:per|a)\s*(year|day|hour|month|decade)",
        )
        .captures(text);
        let r_dec = regex_cache(
            r"(?i)(?:rate\s+of\s+)?([\d.]+)\s*(?:per\s+capita\s+)?(?:per|a)\s*(year|day|hour|month)",
        )
        .captures(text);
        let t = regex_cache(r"(?i)(?:after|in|over|for)\s+([\d.]+)\s*(?:years?|days?|hours?|months?|decades?)")
            .captures(text);
        if n0.is_some() && r_pct.is_some() && t.is_some() {
            let rate: f64 = r_pct.as_ref().unwrap().get(1).unwrap().as_str().parse::<f64>().unwrap() / 100.0;
            let n0v: f64 = n0.as_ref().unwrap().get(1).unwrap().as_str().replace(",", "").parse().unwrap();
            let tv: f64 = t.as_ref().unwrap().get(1).unwrap().as_str().parse().unwrap();
            let mut matched = HashMap::new();
            matched.insert("n0".into(), n0.as_ref().unwrap().get(0).unwrap().as_str().to_string());
            matched.insert("rate".into(), r_pct.as_ref().unwrap().get(0).unwrap().as_str().to_string());
            matched.insert("t".into(), t.as_ref().unwrap().get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "bio_ecology",
                obj(vec![
                    ("kind", json!("exponential")),
                    ("n0", json!(n0v)),
                    ("rate", json!(rate)),
                    ("time", json!(tv)),
                    ("continuous", json!(false)),
                ]),
                "discrete exponential growth (N0*(1+r)^t)",
                matched,
                0.9,
            ));
        }
        if n0.is_some() && r_dec.is_some() && t.is_some() {
            let n0v: f64 = n0.as_ref().unwrap().get(1).unwrap().as_str().replace(",", "").parse().unwrap();
            let rv: f64 = r_dec.as_ref().unwrap().get(1).unwrap().as_str().parse().unwrap();
            let tv: f64 = t.as_ref().unwrap().get(1).unwrap().as_str().parse().unwrap();
            let continuous = has(text, &[r"continuou|exp model|e\^"]);
            let mut matched = HashMap::new();
            matched.insert("n0".into(), n0.as_ref().unwrap().get(0).unwrap().as_str().to_string());
            matched.insert("rate".into(), r_dec.as_ref().unwrap().get(0).unwrap().as_str().to_string());
            matched.insert("t".into(), t.as_ref().unwrap().get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "bio_ecology",
                obj(vec![
                    ("kind", json!("exponential")),
                    ("n0", json!(n0v)),
                    ("rate", json!(rv)),
                    ("time", json!(tv)),
                    ("continuous", json!(continuous)),
                ]),
                "exponential growth wording",
                matched,
                0.85,
            ));
        }
        if has(text, &[r"double"]) && r_dec.is_some() && n0.is_none() {
            let rv: f64 = r_dec.as_ref().unwrap().get(1).unwrap().as_str().parse().unwrap();
            let continuous = has(text, &[r"continuou"]);
            let mut matched = HashMap::new();
            matched.insert("rate".into(), r_dec.as_ref().unwrap().get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "bio_ecology",
                obj(vec![
                    ("kind", json!("doubling_time")),
                    ("rate", json!(rv)),
                    ("continuous", json!(continuous)),
                ]),
                "doubling-time wording",
                matched,
                0.85,
            ));
        }
    }
    None
}

// ---------------- puzzle routes ----------------

fn route_calendar(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"day of the week|what day"]) {
        return None;
    }
    if let Some(m) = regex_cache(r"([A-Z][a-z]+)\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})").captures(text) {
        if let Some(month) = month_number(m.get(1).unwrap().as_str()) {
            let year: i64 = m.get(3).unwrap().as_str().parse().unwrap();
            let day: u32 = m.get(2).unwrap().as_str().parse().unwrap();
            let mut matched = HashMap::new();
            matched.insert("date".into(), m.get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "calendar_day",
                obj(vec![
                    ("year", json!(year)),
                    ("month", json!(month)),
                    ("day", json!(day)),
                ]),
                "date pattern (Month D, YYYY)",
                matched,
                0.95,
            ));
        }
    }
    if let Some(m) = regex_cache(r"(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?([A-Z][a-z]+),?\s+(\d{4})").captures(text) {
        if let Some(month) = month_number(m.get(2).unwrap().as_str()) {
            let year: i64 = m.get(3).unwrap().as_str().parse().unwrap();
            let day: u32 = m.get(1).unwrap().as_str().parse().unwrap();
            let mut matched = HashMap::new();
            matched.insert("date".into(), m.get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "calendar_day",
                obj(vec![
                    ("year", json!(year)),
                    ("month", json!(month)),
                    ("day", json!(day)),
                ]),
                "date pattern (D Month YYYY)",
                matched,
                0.95,
            ));
        }
    }
    if let Some(m) = regex_cache(r"(\d{4})-(\d{2})-(\d{2})").captures(text) {
        let year: i64 = m.get(1).unwrap().as_str().parse().unwrap();
        let month: u32 = m.get(2).unwrap().as_str().parse().unwrap();
        let day: u32 = m.get(3).unwrap().as_str().parse().unwrap();
        let mut matched = HashMap::new();
        matched.insert("date".into(), m.get(0).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "calendar_day",
            obj(vec![
                ("year", json!(year)),
                ("month", json!(month)),
                ("day", json!(day)),
            ]),
            "ISO date",
            matched,
            0.95,
        ));
    }
    Some(NLParse::fail("calendar question but no parseable date"))
}

fn route_clock(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(
        text,
        &[
            r"angle.{0,20}hands|hands.{0,20}angle|clock.{0,30}angle|angle.{0,20}clock",
        ],
    ) {
        return None;
    }
    let m = regex_cache(r"\b(\d{1,2}):(\d{2})\b")
        .captures(text)
        .or_else(|| regex_cache(r"(?i)(\d{1,2})\s*o'?clock").captures(text));
    let Some(m) = m else {
        return Some(NLParse::fail("clock-angle question needs a time like 3:30"))
    };
    let hour: i64 = m.get(1).unwrap().as_str().parse().unwrap();
    let minute: i64 = if m.len() > 2 {
        m.get(2).unwrap().as_str().parse().unwrap()
    } else {
        0
    };
    let mut matched = HashMap::new();
    matched.insert("time".into(), m.get(0).unwrap().as_str().to_string());
    Some(NLParse::ok(
        "clock_angle",
        obj(vec![("hour", json!(hour)), ("minute", json!(minute))]),
        "clock-angle wording + time",
        matched,
        0.95,
    ))
}

fn route_mirror(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if has(text, &[r"clock"]) && has(text, &[r"mirror"]) && regex_cache(r"\b\d{1,2}:\d{2}\b").is_match(text)
    {
        if let Some(t) = regex_cache(r"\b(\d{1,2}):(\d{2})\b").captures(text) {
            let time = format!("{}:{}", t.get(1).unwrap().as_str(), t.get(2).unwrap().as_str());
            let mut matched = HashMap::new();
            matched.insert("time".into(), t.get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "mirror_image",
                obj(vec![("kind", json!("mirror_clock")), ("time", json!(time))]),
                "clock-in-a-mirror wording",
                matched,
                0.95,
            ));
        }
    }
    if let Some(m) = regex_cache(
        r#"(mirror|water)\s+image\s+of\s+(?:the\s+)?(?:word|text|string)?\s*["']?([A-Za-z0-9]+)["']?"#,
    )
    .captures(text)
    {
        let kind = if m.get(1).unwrap().as_str().eq_ignore_ascii_case("mirror") {
            "mirror"
        } else {
            "water"
        };
        let word = m.get(2).unwrap().as_str().to_string();
        let mut matched = HashMap::new();
        matched.insert("text".into(), word.clone());
        return Some(NLParse::ok(
            "mirror_image",
            obj(vec![("kind", json!(kind)), ("text", json!(word))]),
            "mirror/water image wording",
            matched,
            0.95,
        ));
    }
    None
}

fn route_family_tree(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    let fact_pat = regex_cache(
        r"(?i)\b([A-Z][a-z]+)\s+(?:is\s+)?the\s+(father|mother|son|daughter|husband|wife|brother|sister)\s+of\s+([A-Z][a-z]+)",
    );
    let facts: Vec<(String, String, String)> = fact_pat
        .captures_iter(text)
        .map(|m| {
            (
                m.get(1).unwrap().as_str().to_lowercase(),
                m.get(2).unwrap().as_str().to_lowercase(),
                m.get(3).unwrap().as_str().to_lowercase(),
            )
        })
        .collect();
    if facts.is_empty() {
        return None;
    }
    let m = regex_cache(
        r"(?i)how\s+is\s+([A-Z][a-z]+)\s+related\s+to\s+([A-Z][a-z]+)|what\s+is\s+([A-Z][a-z]+)\s+to\s+([A-Z][a-z]+)",
    )
    .captures(text);
    let Some(m) = m else {
        return Some(NLParse::fail(
            "family facts found but the question is missing; add 'how is X related to Y?'",
        ))
    };
    let who = m
        .get(1)
        .or_else(|| m.get(3))
        .unwrap()
        .as_str()
        .to_lowercase();
    let whom = m
        .get(2)
        .or_else(|| m.get(4))
        .unwrap()
        .as_str()
        .to_lowercase();
    let mut matched = HashMap::new();
    matched.insert(
        "facts".into(),
        format!("{:?}", facts),
    );
    matched.insert("question".into(), format!("{} -> {}", who, whom));
    let facts_json: Vec<Value> = facts
        .iter()
        .map(|(s, r, o)| json!([s, r, o]))
        .collect();
    Some(NLParse::ok(
        "family_tree",
        obj(vec![
            ("facts", json!(facts_json)),
            ("who", json!(who)),
            ("whom", json!(whom)),
        ]),
        "family-tree facts + relation question",
        matched,
        0.9,
    ))
}

fn route_directions(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(
        text,
        &[
            r"walk|walks|moves|heads|travels|runs|marches",
        ],
    ) {
        return None;
    }
    if !has(text, &[r"north|south|east|west"]) {
        return None;
    }
    let mut moves: Vec<Value> = Vec::new();
    for m in regex_cache(
        r"(?i)\b(north(?:east|west)?|south(?:east|west)?|east|west|N|S|E|W)\s+([\d.]+)\s*(m|km)\b",
    )
    .captures_iter(text)
    {
        let d = direction_code(m.get(1).unwrap().as_str()).unwrap();
        if matches!(d, "NE" | "NW" | "SE" | "SW") {
            return Some(NLParse::fail(
                "diagonal moves not supported by the verified direction domain",
            ));
        }
        let dist: f64 = m.get(2).unwrap().as_str().parse().unwrap();
        let dist = dist * if m.get(3).unwrap().as_str().eq_ignore_ascii_case("km") { 1000.0 } else { 1.0 };
        moves.push(json!([d, dist]));
    }
    if moves.len() < 2 {
        return None;
    }
    let mut matched = HashMap::new();
    matched.insert("moves".into(), format!("{:?}", moves));
    Some(NLParse::ok(
        "direction_walk",
        obj(vec![("moves", json!(moves))]),
        "direction-walk wording (N/S/E/W + distances)",
        matched,
        0.9,
    ))
}

/// The 22 science + puzzle routes, in the Python's exact order.

// ---------------------------------------------------------------------------
// OTD3 — natural-language routes: magnetism, waves, thermo, relativity
// ---------------------------------------------------------------------------

fn route_magnetism(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"magnet|tesla|\bT\b|solenoid|faraday|induct|lore?ntz|compass|field\s+of\s+a\s+wire"]) {
        return None;
    }
    let mut matched: HashMap<String, String> = HashMap::new();
    let current = first(qs, "A").or_else(|| {
        qs.iter().find(|q| q.raw_unit == "amp" || q.raw_unit == "amps" || q.raw_unit == "ampere")
    });
    let b = first(qs, "T").or_else(|| {
        qs.iter().find(|q| q.raw_unit == "tesla")
    });
    if has(text, &[r"solenoid"]) {
        let (Some(n), Some(i)) = (first(qs, "/m"), current) else { return None };
        matched.insert("turns_per_m".into(), n.text.clone());
        return Some(NLParse::ok(
            "physics_magnetism",
            obj(vec![("kind", json!("solenoid")), ("turns_per_m", json!(n.value)), ("current", json!(i.value))]),
            "solenoid field B = mu0*n*I",
            matched,
            0.85,
        ));
    }
    if has(text, &[r"faraday|emf|induc"]) {
        return None; // needs flux/dt phrasing the parser can't reliably extract yet
    }
    // asking for a FORCE wins over asking for the field
    if has(text, &[r"force"]) {
        let (Some(bv), Some(i), Some(l)) = (b, current, first(qs, "m")) else { return None };
        matched.insert("b".into(), "tesla".into());
        return Some(NLParse::ok(
            "physics_magnetism",
            obj(vec![
                ("kind", json!("wire_force")),
                ("b", json!(bv.value)),
                ("current", json!(i.value)),
                ("length", json!(l.value)),
            ]),
            "motor force F = B*I*L",
            matched,
            0.85,
        ));
    }
    if has(text, &[r"field"]) {
        let (Some(i), Some(r)) = (current, first(qs, "m")) else { return None };
        return Some(NLParse::ok(
            "physics_magnetism",
            obj(vec![("kind", json!("wire_field")), ("current", json!(i.value)), ("distance", json!(r.value))]),
            "wire field B = mu0*I/(2*pi*r)",
            HashMap::new(),
            0.8,
        ));
    }
    None
}

fn route_waves(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"sound|doppler|echo|sonar|wavelength|frequency|wave|snell|refract|lens|hz|thunder"]) {
        return None;
    }
    let mut matched: HashMap<String, String> = HashMap::new();
    let f = first(qs, "Hz").or_else(|| first(qs, "kHz")).or_else(|| first(qs, "MHz"));
    let lambda = first(qs, "m").filter(|_| has(text, &[r"wavelength|lambda"]));
    if has(text, &[r"doppler|mov(?:es?|ing)\s+(?:toward|away|towards)|approach(?:es|ing)|reced(?:es|ing)|passes\s+you|siren"]) {
        let (Some(fv), Some(vs)) = (f, first(qs, "m/s")) else { return None };
        return Some(NLParse::ok(
            "physics_waves",
            obj(vec![("kind", json!("doppler")), ("f", json!(fv.value)), ("v_sound", json!(343.0)), ("v_source", json!(vs.value))]),
            "Doppler f' = f*v/(v - vs)",
            HashMap::new(),
            0.8,
        ));
    }
    if has(text, &[r"speed\s+of\s+sound"]) {
        let t = qs.iter().find(|q| q.raw_unit == "C" || q.raw_unit == "celsius" || q.raw_unit == "°C").map(|q| q.value);
        let temp = t.unwrap_or(20.0);
        matched.insert("temp_c".into(), format!("{} °C", temp));
        return Some(NLParse::ok(
            "physics_waves",
            obj(vec![("kind", json!("sound_speed")), ("temp_c", json!(temp))]),
            "v = 331.3 + 0.606*T",
            matched,
            0.85,
        ));
    }
    if has(text, &[r"echo|sonar|thunder"]) {
        let (Some(t)) = (first(qs, "s")) else { return None };
        return Some(NLParse::ok(
            "physics_waves",
            obj(vec![("kind", json!("echo_distance")), ("v_sound", json!(343.0)), ("t", json!(t.value))]),
            "echo ranging d = v*t/2",
            HashMap::new(),
            0.8,
        ));
    }
    if has(text, &[r"snell|refract|angle\s+of\s+refraction"]) {
        return None; // needs two-index phrasing; typed API handles it
    }
    // wave equation: f + λ
    if let (Some(fv), Some(lv)) = (f, lambda) {
        matched.insert("f".into(), fv.text.clone());
        return Some(NLParse::ok(
            "physics_waves",
            obj(vec![("kind", json!("wave_equation")), ("f", json!(fv.value)), ("lambda", json!(lv.value))]),
            "wave equation v = f*lambda",
            matched,
            0.85,
        ));
    }
    None
}

fn route_thermo(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"temperature|celsius|kelvin|fahrenheit|heat|thermal|conduction|expand|ideal\s+gas|boil|melt|latent|specific\s+heat"]) {
        return None;
    }
    let mut matched: HashMap<String, String> = HashMap::new();
    // pure conversion: two of C/K/F mentioned with one number
    let c = qs.iter().find(|q| q.raw_unit == "C" || q.raw_unit == "celsius" || q.raw_unit == "°C");
    let k = qs.iter().find(|q| q.raw_unit == "K" || q.raw_unit == "kelvin");
    let f = qs.iter().find(|q| q.raw_unit == "F" || q.raw_unit == "fahrenheit");
    let _ = f;
    if let (Some(a), None, None) = (c, k, f) {
        // answer in the unit the question asks for
        let want_f = has(text, &[r"fahrenheit|\bin\s+degrees\s+F\b"]);
        let want_c = has(text, &[r"celsius|centigrade"]) && !has(text, &[r"kelvin"]);
        let kind = if want_f { "to_fahrenheit" } else if want_c { "identity" } else { "to_kelvin" };
        matched.insert("celsius".into(), a.text.clone());
        return Some(NLParse::ok(
            "physics_thermo",
            obj(vec![("kind", json!("convert")), ("celsius", json!(a.value)), ("kelvin", json!(a.value + 273.15)), ("target", json!(kind))]),
            "K = C + 273.15, F = 1.8C + 32",
            matched,
            0.9,
        ));
    }
    if has(text, &[r"specific\s+heat|heat\s+to\s+raise|heats?\s+up|thermal\s+energy"]) {
        let m = first(qs, "kg").or_else(|| first(qs, "g"));
        let dt = first(qs, "C").or_else(|| qs.iter().find(|q| q.raw_unit == "°C" || q.raw_unit == "celsius"));
        if let (Some(m), Some(dt)) = (m, dt) {
            let mass_kg = if m.raw_unit == "g" { m.value / 1000.0 } else { m.value };
            return Some(NLParse::ok(
                "physics_thermo",
                obj(vec![("kind", json!("sensible_heat")), ("mass", json!(mass_kg)), ("c", json!(4186.0)), ("dt", json!(dt.value))]),
                "water heating Q = m*c*dT (c = 4186 J/kgK)",
                HashMap::new(),
                0.75,
            ));
        }
    }
    if has(text, &[r"ideal\s+gas|pressure.*volume|pv\s*=\s*nrt"]) {
        return None; // multi-variable; typed API handles it
    }
    None
}

fn route_relativity(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"relativ|time\s+dilation|length\s+contraction|lorentz|e\s*=\s*mc|rest\s+energy|light\s+speed|speed\s+of\s+light"]) {
        return None;
    }
    let mut matched: HashMap<String, String> = HashMap::new();
    if has(text, &[r"rest\s+energy|e\s*=\s*mc"]) {
        let m = first(qs, "kg").or_else(|| first(qs, "g"));
        let mass = m?;
        let mass_kg = if m.map(|x| x.raw_unit == "g").unwrap_or(false) { mass.value / 1000.0 } else { mass.value };
        matched.insert("mass".into(), mass.text.clone());
        return Some(NLParse::ok(
            "physics_relativity",
            obj(vec![("kind", json!("rest_energy")), ("mass", json!(mass_kg))]),
            "E = mc^2",
            matched,
            0.9,
        ));
    }
    if has(text, &[r"time\s+dilation|clock.*slow|proper\s+time"]) {
        let v = first(qs, "m/s")?;
        let t0 = first(qs, "s").or_else(|| first(qs, "us")).or_else(|| first(qs, "ms"));
        if let Some(t0v) = t0 {
            matched.insert("v".into(), v.text.clone());
            return Some(NLParse::ok(
                "physics_relativity",
                obj(vec![("kind", json!("time_dilation")), ("v", json!(v.value)), ("proper_time", json!(t0v.value))]),
                "time dilation dt = gamma*dt0",
                matched,
                0.8,
            ));
        }
    }
    if has(text, &[r"lorentz|gamma.*factor"]) {
        let v = first(qs, "m/s")?;
        return Some(NLParse::ok(
            "physics_relativity",
            obj(vec![("kind", json!("gamma")), ("v", json!(v.value))]),
            "gamma = 1/sqrt(1 - v^2/c^2)",
            HashMap::new(),
            0.85,
        ));
    }
    None
}

// ---------------------------------------------------------------------------
// OTD3.3 — natural-language routes: particles & nuclei
// ---------------------------------------------------------------------------

/// (name, Z, common isotope A) — the periodic table the parser needs.
/// A is the most abundant / most relevant isotope (used when the question
/// doesn't pin one, e.g. "binding energy of iron").
const ELEMENTS_ZA: &[(&str, u32, u32)] = &[
    ("hydrogen", 1, 1), ("helium", 2, 4), ("lithium", 3, 7), ("beryllium", 4, 9),
    ("boron", 5, 11), ("carbon", 6, 12), ("nitrogen", 7, 14), ("oxygen", 8, 16),
    ("fluorine", 9, 19), ("neon", 10, 20), ("sodium", 11, 23), ("magnesium", 12, 24),
    ("aluminum", 13, 27), ("aluminium", 13, 27), ("silicon", 14, 28), ("phosphorus", 15, 31),
    ("sulfur", 16, 32), ("sulphur", 16, 32), ("chlorine", 17, 35), ("argon", 18, 40),
    ("potassium", 19, 39), ("calcium", 20, 40), ("scandium", 21, 45), ("titanium", 22, 48),
    ("vanadium", 23, 51), ("chromium", 24, 52), ("manganese", 25, 55), ("iron", 26, 56),
    ("cobalt", 27, 59), ("nickel", 28, 58), ("copper", 29, 63), ("zinc", 30, 64),
    ("gallium", 31, 69), ("germanium", 32, 74), ("arsenic", 33, 75), ("selenium", 34, 80),
    ("bromine", 35, 79), ("krypton", 36, 84), ("rubidium", 37, 85), ("strontium", 38, 88),
    ("yttrium", 39, 89), ("zirconium", 40, 90), ("niobium", 41, 93), ("molybdenum", 42, 98),
    ("technetium", 43, 98), ("ruthenium", 44, 102), ("rhodium", 45, 103), ("palladium", 46, 106),
    ("silver", 47, 107), ("cadmium", 48, 114), ("indium", 49, 115), ("tin", 50, 120),
    ("antimony", 51, 121), ("tellurium", 52, 130), ("iodine", 53, 127), ("xenon", 54, 132),
    ("cesium", 55, 133), ("barium", 56, 138), ("lanthanum", 57, 139), ("cerium", 58, 140),
    ("praseodymium", 59, 141), ("neodymium", 60, 142), ("promethium", 61, 145), ("samarium", 62, 152),
    ("europium", 63, 153), ("gadolinium", 64, 158), ("terbium", 65, 159), ("dysprosium", 66, 164),
    ("holmium", 67, 165), ("erbium", 68, 166), ("thulium", 69, 169), ("ytterbium", 70, 174),
    ("lutetium", 71, 175), ("hafnium", 72, 180), ("tantalum", 73, 181), ("tungsten", 74, 184),
    ("rhenium", 75, 187), ("osmium", 76, 192), ("iridium", 77, 193), ("platinum", 78, 195),
    ("gold", 79, 197), ("mercury", 80, 202), ("thallium", 81, 205), ("lead", 82, 208),
    ("bismuth", 83, 209), ("polonium", 84, 209), ("astatine", 85, 210), ("radon", 86, 222),
    ("francium", 87, 223), ("radium", 88, 226), ("actinium", 89, 227), ("thorium", 90, 232),
    ("protactinium", 91, 231), ("uranium", 92, 238), ("neptunium", 93, 237), ("plutonium", 94, 239),
    ("americium", 95, 243), ("curium", 96, 247), ("berkelium", 97, 247), ("californium", 98, 251),
    ("einsteinium", 99, 252), ("fermium", 100, 257), ("mendelevium", 101, 258), ("nobelium", 102, 259),
    ("lawrencium", 103, 266),
];

/// Find an element mentioned as a whole word in the text.
fn element_in_text(text: &str) -> Option<(&'static str, u32, u32)> {
    // longest names first so "technetium" beats "tellurium"-style prefixes
    let mut by_len: Vec<&(&str, u32, u32)> = ELEMENTS_ZA.iter().collect();
    by_len.sort_by_key(|(n, _, _)| std::cmp::Reverse(n.len()));
    for (name, z, a) in by_len {
        if has(text, &[&format!(r"\b{}\b", name)]) {
            return Some((name, *z, *a));
        }
    }
    None
}

/// First standalone integer in the text (a mass number, shell number…).
fn first_int_in_text(text: &str) -> Option<u32> {
    let re = rx(r"\b(\d{1,3})\b", false)?;
    re.captures(text).and_then(|c| c[1].parse().ok())
}

/// The mass number written next to an element: "uranium 238",
/// "carbon-14", "iron 56". Anchored to the element word so "1 gram of
/// uranium 238" picks 238 (not the 1).
fn mass_number_in_text(text: &str, element: &str) -> Option<u32> {
    let pat = format!(r"\b{}[-\s]+(\d{{1,3}})\b", element);
    let re = rx(&pat, true)?;
    re.captures(text).and_then(|c| c[1].parse().ok())
}

fn route_particles(text: &str, qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[
        r"quark|proton|neutron|electron|photon|neutrino|gluon|hadron|pion|muon",
        r"half.?life|radioactiv|decay|isotope|binding\s+energy|de\s?broglie",
        r"annihilat|rydberg|hydrogen\s+alpha|h\s*alpha|balmer|lyman",
        r"electron\s+configuration|configuration\s+of|shell|standard\s+model|becquerel|activity",
        r"sample|remaining|how\s+old|dating|archaeolog|carbon",
    ]) {
        return None;
    }
    let mut matched: HashMap<String, String> = HashMap::new();

    // ---- quark composition: "what are quarks in a proton" ----
    if has(text, &[r"quark"]) && has(text, &[r"proton|neutron|pion"]) {
        let hadron = if has(text, &[r"proton"]) { "proton" }
            else if has(text, &[r"neutron"]) { "neutron" }
            else { "pion" };
        matched.insert("hadron".into(), hadron.into());
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("quark_composition")), ("hadron", json!(hadron))]),
            "quark model: proton = uud, neutron = udd",
            matched,
            0.9,
        ));
    }

    // ---- electron configuration: "electron configuration of gold" ----
    if has(text, &[r"configuration"]) {
        let (name, z, _) = element_in_text(text)?;
        matched.insert("element".into(), name.into());
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("electron_configuration")), ("z", json!(z))]),
            "Madelung (n+l, n) filling with exceptions",
            matched,
            0.9,
        ));
    }

    // ---- half-life: "what is the half life of carbon 14" ----
    if has(text, &[r"half.?life"]) && !has(text, &[r"remaining|left|how\s+old|age\b"]) {
        let (name, _, _) = element_in_text(text)?;
        let a = mass_number_in_text(text, name).or_else(|| first_int_in_text(text)).unwrap_or(14);
        matched.insert("isotope".into(), format!("{}-{}", name, a));
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("half_life")), ("element", json!(name)), ("a", json!(a))]),
            "measured half-life from the nuclear data table",
            matched,
            0.9,
        ));
    }

    // ---- decay age: "25% carbon 14 remaining, how old" ----
    if has(text, &[r"remaining|left\s|left\b|how\s+old|age\b"]) {
        let f = first(qs, "pct")?;
        let (name, _, _) = element_in_text(text).unwrap_or(("carbon", 6, 12));
        let a = mass_number_in_text(text, name).unwrap_or(14);
        // the half-life from the same verified table the engine answers from
        let hl = reasoning_science::particle_domain::half_life_years(name, a)
            .unwrap_or(5730.0);
        matched.insert("fraction".into(), format!("{}", f.value));
        matched.insert("isotope".into(), format!("{}-{}", name, a));
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("decay_age")), ("fraction", json!(f.value)), ("half_life", json!(hl))]),
            "t = -T_half*log2(f)",
            matched,
            0.85,
        ));
    }

    // ---- binding energy: "binding energy of iron 56" ----
    if has(text, &[r"binding"]) {
        let (name, z, default_a) = element_in_text(text)?;
        let a = mass_number_in_text(text, name).unwrap_or(default_a);
        matched.insert("isotope".into(), format!("{}-{}", name, a));
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("binding_energy")), ("a", json!(a)), ("z", json!(z))]),
            "Weizsaecker liquid-drop binding energy",
            matched,
            0.85,
        ));
    }

    // ---- de Broglie: "de broglie wavelength of an electron at 1000 m/s" ----
    if has(text, &[r"de\s?broglie"]) {
        let v = first(qs, "m/s")?;
        matched.insert("v".into(), v.text.clone());
        if has(text, &[r"electron"]) {
            return Some(NLParse::ok(
                "physics_particles",
                obj(vec![("kind", json!("electron_de_broglie")), ("velocity", json!(v.value))]),
                "lambda = h/(m_e v)",
                matched,
                0.85,
            ));
        }
        let m = first(qs, "kg")?;
        matched.insert("m".into(), m.text.clone());
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("de_broglie")), ("mass", json!(m.value)), ("velocity", json!(v.value))]),
            "lambda = h/(mv)",
            matched,
            0.85,
        ));
    }

    // ---- spectral lines: "hydrogen alpha" / "lyman alpha" ----
    if has(text, &[r"hydrogen\s+alpha|h\s*alpha|halpha|balmer"]) && !has(text, &[r"lyman"]) {
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("rydberg")), ("n_lo", json!(2)), ("n_hi", json!(3))]),
            "Rydberg: 1/lambda = R(1/4 - 1/9) — the H-alpha line",
            HashMap::new(),
            0.85,
        ));
    }
    if has(text, &[r"lyman\s+alpha|lyman"]) {
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("rydberg")), ("n_lo", json!(1)), ("n_hi", json!(2))]),
            "Rydberg: 1/lambda = R(1 - 1/4) — the Lyman-alpha line",
            HashMap::new(),
            0.85,
        ));
    }

    // ---- photons: "energy of a 500 nanometer photon" or
    //      "wavelength of a 2 eV photon" ----
    if has(text, &[r"photon"]) {
        if has(text, &[r"wavelength"]) {
            let e = first(qs, "eV")?;
            matched.insert("E".into(), e.text.clone());
            return Some(NLParse::ok(
                "physics_particles",
                obj(vec![("kind", json!("photon_wavelength")), ("energy", json!(e.value))]),
                "lambda = hc/E",
                matched,
                0.85,
            ));
        }
        let wl = first(qs, "nm")?;
        matched.insert("lambda".into(), wl.text.clone());
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("photon_energy")), ("wavelength", json!(wl.value))]),
            "E = hc/lambda",
            matched,
            0.85,
        ));
    }

    // ---- neutron decay & annihilation ----
    if has(text, &[r"neutron\s+decay|neutron.*decay|decay.*neutron"]) {
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("neutron_decay"))]),
            "Q = (m_n - m_p - m_e)c^2",
            HashMap::new(),
            0.85,
        ));
    }
    if has(text, &[r"annihilat"]) {
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("annihilation"))]),
            "E = 2 m_e c^2",
            HashMap::new(),
            0.85,
        ));
    }

    // ---- shell capacity: "how many electrons in the third shell" ----
    if has(text, &[r"shell"]) {
        let n = if let Some(m) = rx(r"n\s*=\s*(\d)", true).and_then(|re| re.captures(text)) {
            m[1].parse().ok()
        } else if let Some(ordinal) = [("first", 1), ("second", 2), ("third", 3), ("fourth", 4), ("fifth", 5), ("sixth", 6), ("seventh", 7)]
            .iter()
            .find(|(w, _)| has(text, &[&format!(r"\b{}\b", w)]))
        {
            Some(ordinal.1)
        } else {
            first_int_in_text(text)
        }?;
        matched.insert("n".into(), format!("{}", n));
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("shell_capacity")), ("n", json!(n))]),
            "capacity = 2n^2",
            matched,
            0.8,
        ));
    }

    // ---- particle masses: "what is the mass of a proton" ----
    if has(text, &[r"mass\s+of|mass\s+of\s+a|weigh"]) {
        let name = if has(text, &[r"proton"]) { "proton" }
            else if has(text, &[r"neutron"]) { "neutron" }
            else if has(text, &[r"electron"]) { "electron" }
            else if has(text, &[r"muon"]) { "muon" }
            else if has(text, &[r"\btau\b"]) { "tau" }
            else { return None };
        matched.insert("particle".into(), name.into());
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("particle_mass")), ("name", json!(name))]),
            "PDG rest mass",
            matched,
            0.85,
        ));
    }

    // ---- activity: "activity of 1 gram of uranium 238" ----
    if has(text, &[r"activity|becquerel|decays?\s+per\s+second|bq\b"]) {
        let g = first(qs, "kg")?;
        let grams = g.value * 1000.0;
        let (name, _, _) = element_in_text(text)?;
        let a = mass_number_in_text(text, name)?;
        let hl = match reasoning_science::particle_domain::half_life_years(name, a) {
            Ok(h) => h,
            Err(_) => return None, // isotope not in the table — abstain
        };
        let n_atoms = grams / a as f64 * 6.02214076e23;
        matched.insert("sample".into(), format!("{:.3} g of {}-{}", grams, name, a));
        return Some(NLParse::ok(
            "physics_particles",
            obj(vec![("kind", json!("activity")), ("n_atoms", json!(n_atoms)), ("half_life", json!(hl))]),
            "A = lambda N = ln2*N/T_half",
            matched,
            0.8,
        ));
    }

    None
}

pub fn science_routes_nl() -> &'static [NamedRoute] {
    static ROUTES: OnceLock<Vec<NamedRoute>> = OnceLock::new();
    ROUTES.get_or_init(|| {
        vec![
            NamedRoute { name: "_route_chem_ph", f: route_chem_ph },
            NamedRoute { name: "_route_chem_molar_mass", f: route_chem_molar_mass },
            NamedRoute { name: "_route_chem_limiting", f: route_chem_limiting },
            NamedRoute { name: "_route_chem_balance", f: route_chem_balance },
            NamedRoute { name: "_route_chem_gas", f: route_chem_gas },
            NamedRoute { name: "_route_chem_solutions", f: route_chem_solutions },
            NamedRoute { name: "_route_bio_hardy_weinberg", f: route_bio_hardy_weinberg },
            NamedRoute { name: "_route_bio_dogma", f: route_bio_dogma },
            NamedRoute { name: "_route_bio_genetics", f: route_bio_genetics },
            NamedRoute { name: "_route_bio_ecology", f: route_bio_ecology },
            NamedRoute { name: "_route_magnetism", f: route_magnetism },
            NamedRoute { name: "_route_waves", f: route_waves },
            NamedRoute { name: "_route_thermo", f: route_thermo },
            NamedRoute { name: "_route_relativity", f: route_relativity },
            NamedRoute { name: "_route_particles", f: route_particles },
            NamedRoute { name: "_route_energy", f: route_energy },
            NamedRoute { name: "_route_kinematics", f: route_kinematics },
            NamedRoute { name: "_route_speed", f: route_speed },
            NamedRoute { name: "_route_momentum", f: route_momentum },
            NamedRoute { name: "_route_electricity", f: route_electricity },
            NamedRoute { name: "_route_force", f: route_force },
            NamedRoute { name: "_route_density", f: route_density },
            NamedRoute { name: "_route_calendar", f: route_calendar },
            NamedRoute { name: "_route_clock", f: route_clock },
            NamedRoute { name: "_route_mirror", f: route_mirror },
            NamedRoute { name: "_route_family_tree", f: route_family_tree },
            NamedRoute { name: "_route_directions", f: route_directions },
        ]
    })
}
