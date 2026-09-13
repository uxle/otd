//! Phase 123 — NL quantity extraction (deterministic, no external API).
//! (Rust port of python/nlp/quantities.py)
//!
//! Turns phrases like "3 m/s^2", "10 seconds", "1.5 atm" into structured
//! quantities. Units are matched from an explicit alias table and CONVERTED
//! to a canonical form (SI where a single canonical exists) -- "2 km"
//! becomes 2000 m, "3 min" becomes 180 s -- so the payload the solver
//! receives is always canonical; the formula layer never has to guess.
//! Numbers are matched as floats (incl. scientific notation); spelled-out
//! small numbers ("five", "half") are handled where unambiguous. Every
//! quantity keeps its ORIGINAL text span so the router can explain what it
//! matched ("t=180 s  [from 'for 3 minutes']").
//!
//! Composite units (m/s^2) are matched before their prefixes (m). The
//! Python lookarounds the regex crate lacks (`(?!\w)`, `(?<![\w.\-])`,
//! `(?![\w/]|\.\d)`, `(?![A-Za-z0-9/^])`) are reimplemented as manual
//! char-boundary checks after each match.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, OnceLock};

use regex::Regex;

use reasoning_common::py_float_str;

/// Python `@dataclass(frozen=True) Quantity`.
#[derive(Debug, Clone, PartialEq)]
pub struct Quantity {
    pub value: f64,
    pub unit: String,     // canonical unit ('', 'm', 's', 'm/s', 'm/s^2', 'kg', 'N', ...)
    pub raw_unit: String, // as written in the text
    pub text: String,     // the original span, e.g. "3 minutes"
}

impl fmt::Display for Quantity {
    /// Python dataclass repr (used verbatim in `to_kelvin`'s error message).
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Quantity(value={}, unit={}, raw_unit={}, text={})",
            py_float_str(self.value),
            py_repr(&self.unit),
            py_repr(&self.raw_unit),
            py_repr(&self.text)
        )
    }
}

// number: float incl. scientific notation and optional sign
const NUM: &str = r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?";

/// Spelled-out numbers (small, unambiguous ones only) — Python `WORDS`
/// (dict insertion order preserved).
pub fn words() -> &'static [(&'static str, f64)] {
    static WORDS: &[(&str, f64)] = &[
        ("zero", 0.0), ("one", 1.0), ("two", 2.0), ("three", 3.0), ("four", 4.0),
        ("five", 5.0), ("six", 6.0), ("seven", 7.0), ("eight", 8.0), ("nine", 9.0),
        ("ten", 10.0), ("eleven", 11.0), ("twelve", 12.0), ("fifteen", 15.0),
        ("twenty", 20.0), ("thirty", 30.0), ("forty", 40.0), ("fifty", 50.0),
        ("sixty", 60.0), ("hundred", 100.0), ("half", 0.5), ("quarter", 0.25),
    ];
    WORDS
}

/// alias -> (canonical unit, multiplicative factor) — Python `_SIMPLE_UNITS`
/// (insertion order preserved: it is the tie-break of the longest-first
/// alternation built below).
fn simple_units() -> &'static [(&'static str, &'static str, f64)] {
    static UNITS: &[(&str, &str, f64)] = &[
        ("s", "s", 1.0), ("sec", "s", 1.0), ("secs", "s", 1.0), ("second", "s", 1.0), ("seconds", "s", 1.0),
        ("min", "s", 60.0), ("mins", "s", 60.0), ("minute", "s", 60.0), ("minutes", "s", 60.0),
        ("h", "s", 3600.0), ("hr", "s", 3600.0), ("hrs", "s", 3600.0), ("hour", "s", 3600.0), ("hours", "s", 3600.0),
        ("day", "s", 86400.0), ("days", "s", 86400.0),
        ("m", "m", 1.0), ("meter", "m", 1.0), ("meters", "m", 1.0), ("metre", "m", 1.0), ("metres", "m", 1.0),
        ("km", "m", 1000.0), ("kilometer", "m", 1000.0), ("kilometers", "m", 1000.0),
        ("cm", "m", 0.01), ("centimeter", "m", 0.01), ("centimeters", "m", 0.01),
        ("centimetre", "m", 0.01), ("centimetres", "m", 0.01),
        ("mm", "m", 0.001), ("millimeter", "m", 0.001), ("millimeters", "m", 0.001),
        ("millimetre", "m", 0.001), ("millimetres", "m", 0.001),
        ("mile", "m", 1609.344), ("miles", "m", 1609.344),
        ("kg", "kg", 1.0), ("kilogram", "kg", 1.0), ("kilograms", "kg", 1.0),
        ("g", "kg", 0.001), ("gram", "kg", 0.001), ("grams", "kg", 0.001),
        ("N", "N", 1.0), ("newton", "N", 1.0), ("newtons", "N", 1.0),
        ("J", "J", 1.0), ("joule", "J", 1.0), ("joules", "J", 1.0),
        ("kJ", "J", 1000.0), ("kilojoule", "J", 1000.0), ("kilojoules", "J", 1000.0),
        ("W", "W", 1.0), ("watt", "W", 1.0), ("watts", "W", 1.0),
        ("kW", "W", 1000.0), ("kilowatt", "W", 1000.0), ("kilowatts", "W", 1000.0),
        ("V", "V", 1.0), ("volt", "V", 1.0), ("volts", "V", 1.0),
        ("A", "A", 1.0), ("amp", "A", 1.0), ("amps", "A", 1.0), ("ampere", "A", 1.0), ("amperes", "A", 1.0),
        ("ohm", "ohm", 1.0), ("ohms", "ohm", 1.0), ("Ω", "ohm", 1.0),
        ("Pa", "Pa", 1.0), ("pascal", "Pa", 1.0), ("pascals", "Pa", 1.0),
        ("kPa", "Pa", 1000.0), ("kilopascal", "Pa", 1000.0), ("kilopascals", "Pa", 1000.0),
        ("atm", "atm", 1.0), ("atmosphere", "atm", 1.0), ("atmospheres", "atm", 1.0),
        ("L", "L", 1.0), ("liter", "L", 1.0), ("liters", "L", 1.0), ("litre", "L", 1.0), ("litres", "L", 1.0),
        ("mL", "L", 0.001), ("ml", "L", 0.001), ("milliliter", "L", 0.001), ("milliliters", "L", 0.001),
        ("mol", "mol", 1.0), ("mole", "mol", 1.0), ("moles", "mol", 1.0), ("mols", "mol", 1.0),
        // OTD3 science expansion: magnetism, waves, thermo
        ("T", "T", 1.0), ("tesla", "T", 1.0), ("teslas", "T", 1.0), ("mT", "T", 0.001),
        ("gauss", "T", 1e-4), ("G", "T", 1e-4), ("mG", "T", 1e-7),
        ("Hz", "Hz", 1.0), ("hz", "Hz", 1.0), ("hertz", "Hz", 1.0),
        ("kHz", "Hz", 1000.0), ("kilohertz", "Hz", 1000.0),
        ("MHz", "Hz", 1e6), ("megahertz", "Hz", 1e6), ("GHz", "Hz", 1e9), ("gigahertz", "Hz", 1e9),
        ("K", "K", 1.0), ("kelvin", "K", 1.0), ("kelvins", "K", 1.0),
        ("C", "degC", 1.0), ("celsius", "degC", 1.0), ("°C", "degC", 1.0),
        ("F", "degF", 1.0), ("fahrenheit", "degF", 1.0), ("°F", "degF", 1.0),
        ("m/s", "m/s", 1.0), ("cm/s", "m/s", 0.01), ("km/h", "m/s", 1000.0/3600.0),
        ("km/hour", "m/s", 1000.0/3600.0), ("mph", "m/s", 1609.344/3600.0),
        ("meters per second", "m/s", 1.0), ("metres per second", "m/s", 1.0),
        ("meter per second", "m/s", 1.0), ("metre per second", "m/s", 1.0),
        ("meters/second", "m/s", 1.0), ("metres/second", "m/s", 1.0),
        // OTD3.3 — the subatomic layer: lengths of light, energies of
        // particles, radioactivity, and fractions ("25 percent remaining")
        ("nm", "nm", 1.0), ("nanometer", "nm", 1.0), ("nanometers", "nm", 1.0),
        ("nanometre", "nm", 1.0), ("nanometres", "nm", 1.0),
        ("picometer", "nm", 0.001), ("picometers", "nm", 0.001),
        ("micron", "nm", 1000.0), ("microns", "nm", 1000.0),
        ("eV", "eV", 1.0), ("ev", "eV", 1.0), ("electron volt", "eV", 1.0),
        ("electron volts", "eV", 1.0), ("electronvolt", "eV", 1.0), ("electronvolts", "eV", 1.0),
        ("keV", "eV", 1e3), ("kiloelectron volt", "eV", 1e3),
        ("MeV", "eV", 1e6), ("megaelectron volt", "eV", 1e6),
        ("GeV", "eV", 1e9), ("gigaelectron volt", "eV", 1e9),
        ("Bq", "Bq", 1.0), ("becquerel", "Bq", 1.0), ("becquerels", "Bq", 1.0),
        ("percent", "pct", 0.01), ("%", "pct", 0.01),
    ];
    UNITS
}

/// Units whose canonical name differs from the alias key (defined but never
/// used in the Python source either — kept for module parity).
#[allow(dead_code)]
const _CANONICAL_RENAME: [(&str, &str); 5] = [("N", "N"), ("J", "J"), ("W", "W"), ("V", "V"), ("A", "A")];

#[allow(dead_code)]
fn _rename(alias: &str) -> &str {
    if alias == "Ω" {
        "ohm"
    } else {
        alias
    }
}

/// `unit_alt` — all simple aliases joined longest-first (Python:
/// `sorted((re.escape(k) for k in _SIMPLE_UNITS), key=len, reverse=True)`;
/// `re.escape` is the identity for these keys and Python's stable sort keeps
/// equal-length keys in insertion order).
fn simple_unit_alt() -> &'static str {
    static ALT: OnceLock<String> = OnceLock::new();
    ALT.get_or_init(|| {
        let mut keys: Vec<&str> = simple_units().iter().map(|(k, _, _)| *k).collect();
        keys.sort_by(|a, b| b.chars().count().cmp(&a.chars().count()));
        keys.join("|")
    })
    .as_str()
}

/// Python `_canonical_unit`: raw written unit -> (canonical unit,
/// multiplicative factor).
fn canonical_unit(raw: &str) -> Option<(String, f64)> {
    let raw_n = raw.trim();
    let raw_low = raw_n.to_lowercase();

    // acceleration first (its unit contains m/s as a prefix)
    if fullmatch(r"m\s*(?:/|per\s+)\s*s\s*(?:\^2|²|\^\{2\}|\^\s*-\s*2)", raw_n, true) {
        return Some(("m/s^2".to_string(), 1.0));
    }
    // velocities (composite)
    if fullmatch(r"(?:km\s*/\s*h(?:our)?|kmh|kph)", &raw_low, false) {
        return Some(("m/s".to_string(), 1000.0 / 3600.0));
    }
    if fullmatch(r"m\s*(?:/|per\s+)\s*s", raw_n, true) {
        return Some(("m/s".to_string(), 1.0));
    }
    if fullmatch(r"mph|miles\s*/\s*h(?:our)?", &raw_low, false) {
        return Some(("m/s".to_string(), 1609.344 / 3600.0));
    }
    // momentum
    if fullmatch(r"kg\s*(?:·|•|\*|\s)?\s*m\s*/\s*s", raw_n, true) {
        return Some(("kg*m/s".to_string(), 1.0));
    }
    // density
    if fullmatch(r"kg\s*/\s*m\s*\^?3", raw_n, true) {
        return Some(("kg/m^3".to_string(), 1.0));
    }
    if fullmatch(r"g\s*/\s*(?:cm\s*\^?3|mL|ml)", raw_n, true) {
        return Some(("kg/m^3".to_string(), 1000.0));
    }
    if fullmatch(r"m\s*\^?3", raw_n, true) {
        return Some(("m^3".to_string(), 1.0));
    }
    if fullmatch(r"cm\s*\^?3", raw_n, true) {
        return Some(("m^3".to_string(), 1e-6));
    }
    // temperature
    if fullmatch(r"°\s*C|°C|celsius|C", raw_n, true) {
        return Some(("C".to_string(), 1.0));
    }
    if raw_n == "K" || raw_low == "kelvin" {
        return Some(("K".to_string(), 1.0));
    }
    // simple table (case-sensitive first, then case-insensitive)
    for (alias, unit, factor) in simple_units() {
        if *alias == raw_n {
            return Some((unit.to_string(), *factor));
        }
    }
    for (alias, unit, factor) in simple_units() {
        if *alias == raw_low {
            return Some((unit.to_string(), *factor));
        }
    }
    None
}

/// Manual reimplementation of the Python trailing lookarounds the regex
/// crate does not support. `end` is the byte offset just past the match.
#[derive(Clone, Copy, PartialEq)]
enum TailCheck {
    /// no trailing lookahead
    None,
    /// Python `(?!\w)` — next char must not be alphanumeric/underscore
    NotWord,
    /// Python `(?![A-Za-z])`
    NotAsciiAlpha,
    /// Python `(?![A-Za-z0-9/^])` (the simple-unit tail)
    NotUnitFollow,
}

impl TailCheck {
    fn ok(self, text: &str, end: usize) -> bool {
        match self {
            TailCheck::None => true,
            TailCheck::NotWord => text[end..]
                .chars()
                .next()
                .map_or(true, |c| !(c.is_alphanumeric() || c == '_')),
            TailCheck::NotAsciiAlpha => text[end..].chars().next().map_or(true, |c| !c.is_ascii_alphabetic()),
            TailCheck::NotUnitFollow => text[end..]
                .chars()
                .next()
                .map_or(true, |c| !(c.is_ascii_alphanumeric() || c == '/' || c == '^')),
        }
    }
}

/// One entry of Python's `composite` list: the unit alternation (group 2),
/// whether the Python pattern ended in `\b`, and the trailing lookahead
/// checked manually after the match.
struct CompositeSpec {
    unit_pat: &'static str,
    boundary: bool,
    tail: TailCheck,
}

const COMPOSITES: [CompositeSpec; 11] = [
    // (NUM)\s*(m\s*(?:/|per\s+)\s*s\s*(?:\^2|²|^\{2\}|^ -2))  — no \b
    CompositeSpec { unit_pat: r"m\s*(?:/|per\s+)\s*s\s*(?:\^2|²|\^\{2\}|\^\s*-\s*2)", boundary: false, tail: TailCheck::None },
    // (NUM)\s*(km/h|kmh|kph)\b
    CompositeSpec { unit_pat: r"km\s*/\s*h(?:our)?|kmh|kph", boundary: true, tail: TailCheck::None },
    // (NUM)\s*(mph|miles/hour)\b
    CompositeSpec { unit_pat: r"mph|miles\s*/\s*h(?:our)?", boundary: true, tail: TailCheck::None },
    // (NUM)\s*(m/s|m per s)\b
    CompositeSpec { unit_pat: r"m\s*(?:/|per\s+)\s*s", boundary: true, tail: TailCheck::None },
    // (NUM)\s*(kg·m/s etc.)\b
    CompositeSpec { unit_pat: r"kg\s*(?:·|•|\*|\s)?\s*m\s*/\s*s", boundary: true, tail: TailCheck::None },
    CompositeSpec { unit_pat: r"kg\s*/\s*m\s*\^?3", boundary: true, tail: TailCheck::None },
    CompositeSpec { unit_pat: r"g\s*/\s*(?:cm\s*\^?3|mL|ml)", boundary: true, tail: TailCheck::None },
    CompositeSpec { unit_pat: r"cm\s*\^?3", boundary: true, tail: TailCheck::None },
    CompositeSpec { unit_pat: r"m\s*\^?3", boundary: true, tail: TailCheck::None },
    // (NUM)\s*(°C|C|celsius)\b(?!\w)
    CompositeSpec { unit_pat: r"°\s*C|°C|celsius|C", boundary: true, tail: TailCheck::NotWord },
    // (NUM)\s*(kelvin|K)\b(?![A-Za-z])
    CompositeSpec { unit_pat: r"kelvin|K", boundary: true, tail: TailCheck::NotAsciiAlpha },
];

/// Python `overlaps` closure over the `taken` spans (half-open intervals).
fn overlaps(taken: &[(usize, usize)], a: usize, b: usize) -> bool {
    taken.iter().any(|&(s, e)| !(b <= s || a >= e))
}

/// All "number + unit" occurrences (canonicalized) followed by bare numbers
/// and spelled-out numbers. Composite units (m/s^2) are matched before their
/// prefixes (m).
pub fn extract_quantities(text: &str) -> Vec<Quantity> {
    let mut out: Vec<Quantity> = Vec::new();
    let mut taken: Vec<(usize, usize)> = Vec::new();

    // 1) composite-unit quantities, longest first
    for (i, spec) in COMPOSITES.iter().enumerate() {
        let pat = format!(r"({})\s*({}){}", NUM, spec.unit_pat, if spec.boundary { r"\b" } else { "" });
        let re = rx(&pat, true).expect("composite-unit regex");
        for caps in re.captures_iter(text) {
            let whole = caps.get(0).unwrap();
            if overlaps(&taken, whole.start(), whole.end()) {
                continue;
            }
            if !spec.tail.ok(text, whole.end()) {
                continue;
            }
            let num = py_float(&caps[1]);
            let raw_unit = caps[2].to_string();
            let whole_text = whole.as_str().to_string();
            let (unit, value) = match i {
                0 => ("m/s^2", num),
                1 => ("m/s", num * 1000.0 / 3600.0),
                2 => ("m/s", num * 1609.344 / 3600.0),
                3 => ("m/s", num),
                4 => ("kg*m/s", num),
                5 => ("kg/m^3", num),
                6 => ("kg/m^3", num * 1000.0),
                7 => ("m^3", num * 1e-6),
                8 => ("m^3", num),
                9 => ("C", num),
                _ => ("K", num),
            };
            out.push(Quantity { value, unit: unit.to_string(), raw_unit, text: whole_text });
            taken.push((whole.start(), whole.end()));
        }
    }

    // 2) simple-unit quantities (case-sensitive pattern; the Python tail
    //    lookahead (?![A-Za-z0-9/^]) is checked manually after the match)
    let simple_pat = format!(r"({})\s*({})", NUM, simple_unit_alt());
    let re = rx(&simple_pat, true).expect("simple-unit regex");
    for caps in re.captures_iter(text) {
        let whole = caps.get(0).unwrap();
        if overlaps(&taken, whole.start(), whole.end()) {
            continue;
        }
        if !TailCheck::NotUnitFollow.ok(text, whole.end()) {
            continue;
        }
        let (unit, factor) = match canonical_unit(&caps[2]) {
            Some(v) => v,
            None => continue,
        };
        out.push(Quantity {
            value: py_float(&caps[1]) * factor,
            unit,
            raw_unit: caps[2].to_string(),
            text: whole.as_str().to_string(),
        });
        taken.push((whole.start(), whole.end()));
    }

    // 3) bare numbers (no unit adjacent). Python:
    //    (?<![\w.\-])(NUM)(?![\w/]|\.\d)  — both lookarounds checked manually.
    //    Greedy-first + reject is equivalent to Python's backtracking here:
    //    any shorter _NUM sub-match at the same start fails the lookarounds
    //    too (the chars given up are digits or a `.`-digit pair).
    let re = rx(&format!(r"({})", NUM), false).expect("bare-number regex");
    for m in re.find_iter(text) {
        if overlaps(&taken, m.start(), m.end()) {
            continue;
        }
        // (?<![\w.\-])
        if let Some(c) = text[..m.start()].chars().next_back() {
            if c.is_alphanumeric() || c == '_' || c == '.' || c == '-' {
                continue;
            }
        }
        // (?![\w/]|\.\d)
        let mut reject = false;
        if let Some(c) = text[m.end()..].chars().next() {
            if c.is_alphanumeric() || c == '_' || c == '/' {
                reject = true;
            }
            if !reject && c == '.' {
                let after = m.end() + c.len_utf8();
                if let Some(d) = text[after..].chars().next() {
                    if d.is_numeric() {
                        reject = true;
                    }
                }
            }
        }
        if reject {
            continue;
        }
        out.push(Quantity {
            value: py_float(m.as_str()),
            unit: String::new(),
            raw_unit: String::new(),
            text: m.as_str().to_string(),
        });
        taken.push((m.start(), m.end()));
    }

    // 4) spelled-out numbers
    for (w, v) in words() {
        let re = rx(&format!(r"\b{}\b", w), true).expect("word regex");
        for m in re.find_iter(text) {
            if overlaps(&taken, m.start(), m.end()) {
                continue;
            }
            out.push(Quantity {
                value: *v,
                unit: String::new(),
                raw_unit: w.to_string(),
                text: m.as_str().to_string(),
            });
            taken.push((m.start(), m.end()));
        }
    }
    out
}

/// Temperature quantity -> Kelvin (applies the Celsius offset).
/// Python `to_kelvin` (ValueError message kept verbatim).
pub fn to_kelvin(q: &Quantity) -> Result<f64, String> {
    if q.unit == "K" {
        return Ok(q.value);
    }
    if q.unit == "C" {
        return Ok(q.value + 273.15);
    }
    Err(format!("not a temperature quantity: {}", q))
}

// ---------------- shared Python-builtin / `re`-module ports ----------------
// (pub(crate): every module of this crate ports Python code that leans on
// the `re` module's internal compile cache and on float()/int()/repr().)

static RE_CACHE: OnceLock<Mutex<HashMap<String, &'static Regex>>> = OnceLock::new();

/// Compile-once regex cache (the stand-in for Python `re`'s internal pattern
/// cache; every distinct pattern string is compiled once and leaked).
/// `case_insensitive` prepends `(?i)` exactly like Python's `re.I`.
/// Returns None for an invalid pattern (Python would raise `re.error`, which
/// `parse()` records as a crashed route; the pattern strings here are literal
/// ports that all compile).
pub(crate) fn rx(pattern: &str, case_insensitive: bool) -> Option<&'static Regex> {
    let cache = RE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    let key = if case_insensitive {
        format!("(?i){}", pattern)
    } else {
        pattern.to_string()
    };
    let mut guard = cache.lock().unwrap_or_else(|e| e.into_inner());
    if let Some(r) = guard.get(&key) {
        return Some(r);
    }
    match Regex::new(&key) {
        Ok(re) => {
            let leaked: &'static Regex = Box::leak(Box::new(re));
            guard.insert(key, leaked);
            Some(leaked)
        }
        Err(_) => None,
    }
}

/// `re.search(p, text, re.I) == None` check (convenience wrapper).
#[allow(dead_code)]
#[allow(dead_code)]
pub(crate) fn search_ci(pattern: &str, text: &str) -> bool {
    rx(pattern, true).map_or(false, |r| r.is_match(text))
}

/// `re.search(p, text)` (case-sensitive) check.
#[allow(dead_code)]
pub(crate) fn search(pattern: &str, text: &str) -> bool {
    rx(pattern, false).map_or(false, |r| r.is_match(text))
}

/// Python `re.fullmatch(p, text[, re.I])` — anchored both ends.
pub(crate) fn fullmatch(pattern: &str, text: &str, case_insensitive: bool) -> bool {
    let wrapped = format!(r"\A(?:{})\z", pattern);
    rx(&wrapped, case_insensitive).map_or(false, |r| r.is_match(text))
}

/// Python `float(s)` — panics with Python's message on a bad value (a route
/// crash that `parse()` catches per-route, like Python's ValueError).
pub(crate) fn py_float(s: &str) -> f64 {
    s.parse::<f64>()
        .unwrap_or_else(|_| panic!("could not convert string to float: '{}'", s))
}

/// Python `int(s)` — panics with Python's message on a bad value.
#[allow(dead_code)]
pub(crate) fn py_int(s: &str) -> i64 {
    s.parse::<i64>()
        .unwrap_or_else(|_| panic!("invalid literal for int() with base 10: '{}'", s))
}

/// Python `repr(s)` for the strings this crate produces (quote-char choice +
/// the escapes that can actually occur).
pub(crate) fn py_repr(s: &str) -> String {
    let quote = if s.contains('\'') && !s.contains('"') { '"' } else { '\'' };
    let mut out = String::with_capacity(s.len() + 2);
    out.push(quote);
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if c == quote => {
                out.push('\\');
                out.push(c);
            }
            c => out.push(c),
        }
    }
    out.push(quote);
    out
}

/// Python `repr` of a list of strings, e.g. `['u', 'v']` (used in the
/// rationales' "missing slots" messages).
#[allow(dead_code)]
pub(crate) fn py_list_str(items: &[&str]) -> String {
    format!("[{}]", items.iter().map(|s| py_repr(s)).collect::<Vec<_>>().join(", "))
}

/// Python `s[:n]` (character-indexed slice).
#[allow(dead_code)]
pub(crate) fn py_slice_chars(s: &str, n: usize) -> String {
    s.chars().take(n).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_unit_alt_is_longest_first() {
        let parts: Vec<&str> = simple_unit_alt().split('|').collect();
        assert_eq!(parts.len(), simple_units().len());
        for w in parts.windows(2) {
            assert!(w[0].chars().count() >= w[1].chars().count(), "{} before {}", w[0], w[1]);
        }
    }

    #[test]
    fn test_composite_before_prefix() {
        let qs = extract_quantities("3 m/s^2 for 5 s");
        assert_eq!(qs.len(), 2);
        assert_eq!((qs[0].value, qs[0].unit.as_str()), (3.0, "m/s^2"));
        assert_eq!((qs[1].value, qs[1].unit.as_str()), (5.0, "s"));
    }

    #[test]
    fn test_bare_number_lookarounds() {
        // (?<![\w.\-]) and (?![\w/]|\.\d) reject these
        assert!(extract_quantities("x-5").is_empty());
        assert!(extract_quantities("1.5.7").is_empty());
        let qs = extract_quantities("3.5 apples");
        assert_eq!(qs.len(), 1);
        assert_eq!(qs[0].value, 3.5);
        assert_eq!(qs[0].unit, "");
    }

    #[test]
    fn test_spelled_words_case_insensitive() {
        // no digit precedes "minutes", so only the spelled word matches
        let qs = extract_quantities("Five minutes");
        assert_eq!(qs.len(), 1);
        assert_eq!((qs[0].value, qs[0].raw_unit.as_str()), (5.0, "five"));
        assert_eq!(qs[0].text, "Five");
    }

    #[test]
    fn test_to_kelvin_error_message_is_python_repr() {
        let q = Quantity { value: 27.0, unit: "m".into(), raw_unit: "m".into(), text: "27 m".into() };
        assert_eq!(
            to_kelvin(&q).unwrap_err(),
            "not a temperature quantity: Quantity(value=27.0, unit='m', raw_unit='m', text='27 m')"
        );
    }
}
