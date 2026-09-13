//! Phase 124b — Natural-language routes for the MATH domains.
//! (Rust port of python/nlp/math_routes.py)
//!
//! Mirrors parser.rs's contract: each route takes (text, quantities) and
//! returns an NLParse or None. Math routes deliberately REUSE the existing
//! verified solvers — the NL layer only ever maps words to typed payloads.

use std::collections::HashMap;

use serde_json::json;

use crate::parser::{has, regex_cache, NLParse};
use crate::quantities::{words, Quantity};

use serde_json::{Map, Value};

fn obj(pairs: Vec<(&str, Value)>) -> Map<String, Value> {
    let mut m = Map::new();
    for (k, v) in pairs {
        m.insert(k.to_string(), v);
    }
    m
}

fn route_arithmetic(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    let stripped = regex_cache(
        r"(?i)^(?:what\s+is|whats|what's|calculate|compute|evaluate|how\s+much\s+is|solve)\s*",
    )
    .replace(text.trim(), "");
    let stripped = stripped.trim_matches(|c: char| c == ' ' || c == '?' || c == '.' || c == '!');
    if stripped.is_empty() {
        return None;
    }
    let expr = stripped
        .replace('\u{00d7}', "*")
        .replace('\u{00f7}', "/")
        .replace('^', "**");
    // Python: re.fullmatch(r"[-+*/(). \d**]+", expr)
    if !expr
        .chars()
        .all(|c| matches!(c, '-' | '+' | '*' | '/' | '(' | ')' | '.' | ' ' | '0'..='9'))
    {
        return None;
    }
    if !expr.chars().any(|c| c.is_ascii_digit()) || !expr.chars().any(|c| matches!(c, '-' | '+' | '*' | '/')) {
        return None;
    }
    let mut matched = HashMap::new();
    matched.insert("expr".to_string(), expr.clone());
    Some(NLParse::ok(
        "arithmetic",
        obj(vec![("expr", json!(expr))]),
        "pure numeric expression",
        matched,
        0.95,
    ))
}

fn route_linear(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"solve"]) {
        return None;
    }
    let m = regex_cache(r"([-+*/(). \d]*x[-+*/(). \dx]*)=([-+*/(). \dx]+)").captures(text)?;
    if !text.contains('=') {
        return None;
    }
    let eq = m.get(0).unwrap().as_str().trim().to_string();
    if !eq.contains('x') {
        return None;
    }
    // normalize implicit multiplication: 2x -> 2*x
    let eq = regex_cache(r"(\d)\s*x").replace_all(&eq, "${1}*x").to_string();
    let eq = eq.replace(' ', "");
    for part in eq.split('=') {
        if !part
            .chars()
            .all(|c| matches!(c, '-' | '+' | '*' | '/' | '(' | ')' | '.' | '0'..='9' | 'x'))
        {
            return None;
        }
    }
    let mut matched = HashMap::new();
    matched.insert("equation".to_string(), eq.clone());
    Some(NLParse::ok(
        "linear_equation",
        obj(vec![("equation", json!(eq))]),
        "solve + single-variable equation",
        matched,
        0.95,
    ))
}

fn route_quadratic(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"factor(?:ise|ize|)?"]) {
        return None;
    }
    let m = regex_cache(r"x\^?2\s*([+-]\s*\d+)\s*x\s*([+-]\s*\d+)").captures(text)?;
    let b: i64 = m.get(1).unwrap().as_str().replace(' ', "").parse().unwrap();
    let c: i64 = m.get(2).unwrap().as_str().replace(' ', "").parse().unwrap();
    let mut matched = HashMap::new();
    matched.insert("b".to_string(), b.to_string());
    matched.insert("c".to_string(), c.to_string());
    Some(NLParse::ok(
        "quadratic_factoring",
        obj(vec![("b", json!(b)), ("c", json!(c))]),
        "factor + monic quadratic pattern x^2 + bx + c",
        matched,
        0.9,
    ))
}

fn route_combinatorics(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    let mut m = regex_cache(r"\b(\d+)\s*(?:choose|C)\s*(\d+)\b")
        .captures(text)
        .or_else(|| regex_cache(r"C\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)").captures(text));
    if let Some(mm) = m.as_ref() {
        if has(text, &[r"choose|combination|C\s*(\()"]) {
            let n: i64 = mm.get(1).unwrap().as_str().parse().unwrap();
            let r: i64 = mm.get(2).unwrap().as_str().parse().unwrap();
            let mut matched = HashMap::new();
            matched.insert("nCr".to_string(), mm.get(0).unwrap().as_str().to_string());
            return Some(NLParse::ok(
                "combinatorics",
                obj(vec![
                    ("ctype", json!("combinations")),
                    ("n", json!(n)),
                    ("r", json!(r)),
                ]),
                "n-choose-r pattern",
                matched,
                0.95,
            ));
        }
    }
    m = regex_cache(r"P\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
        .captures(text)
        .or_else(|| {
            regex_cache(r"(\d+)\s*permutations?\s+taken\s+(\d+)")
                .captures(text)
                .or_else(|| {
                    regex_cache(
                        r"permutations?\s+of\s+(\d+)\s+(?:things|objects|items)\s+taken\s+(\d+)",
                    )
                    .captures(text)
                })
        });
    if let Some(mm) = m {
        let n: i64 = mm.get(1).unwrap().as_str().parse().unwrap();
        let r: i64 = mm.get(2).unwrap().as_str().parse().unwrap();
        let mut matched = HashMap::new();
        matched.insert("nPr".to_string(), mm.get(0).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "combinatorics",
            obj(vec![
                ("ctype", json!("permutations")),
                ("n", json!(n)),
                ("r", json!(r)),
            ]),
            "permutation pattern",
            matched,
            0.95,
        ));
    }
    m = regex_cache(
        r"combinations?\s+of\s+(\d+)\s+(?:things|objects|items)\s+taken\s+(\d+)",
    )
    .captures(text);
    if let Some(mm) = m {
        let n: i64 = mm.get(1).unwrap().as_str().parse().unwrap();
        let r: i64 = mm.get(2).unwrap().as_str().parse().unwrap();
        let mut matched = HashMap::new();
        matched.insert("nCr".to_string(), mm.get(0).unwrap().as_str().to_string());
        return Some(NLParse::ok(
            "combinatorics",
            obj(vec![
                ("ctype", json!("combinations")),
                ("n", json!(n)),
                ("r", json!(r)),
            ]),
            "combination pattern",
            matched,
            0.95,
        ));
    }
    None
}

fn route_gcd_bezout(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    let m = regex_cache(r"(?i)gcd\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)")
        .captures(text)
        .or_else(|| {
            regex_cache(r"(?i)greatest\s+common\s+(?:divisor|factor)\s+of\s+(\d+)\s+and\s+(\d+)")
                .captures(text)
        });
    let Some(m) = m else { return None };
    let a: i64 = m.get(1).unwrap().as_str().parse().unwrap();
    let b: i64 = m.get(2).unwrap().as_str().parse().unwrap();
    let want_bezout = has(text, &[r"bezout|coefficient|identity|express|write.*as"]);
    let mut matched = HashMap::new();
    matched.insert("a".to_string(), a.to_string());
    matched.insert("b".to_string(), b.to_string());
    let rationale = if want_bezout {
        "gcd pattern + Bezout coefficients requested"
    } else {
        "gcd pattern"
    };
    Some(NLParse::ok(
        "gcd_bezout",
        obj(vec![("a", json!(a)), ("b", json!(b))]),
        rationale,
        matched,
        0.95,
    ))
}

fn route_trig_evaluate(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"sin|cos|tan"]) {
        return None;
    }
    if has(text, &[r"simplify|identity|expand"]) {
        return None;
    }
    let expr = text.to_string();
    // Python: re.sub(..., flags=re.I) — the leading "What is" strip is
    // case-insensitive (a capital W otherwise blocks the whole route).
    let expr = regex_cache(
        r"(?i)^(?:what\s+is|evaluate|compute|find|calculate|the\s+value\s+of)\s*",
    )
    .replace(expr.trim(), "")
    .to_string();
    let expr = expr
        .trim_matches(|c: char| c == ' ' || c == '?' || c == '.' || c == '!')
        .to_string();
    // degrees -> radians conversion for bare angles: sin(30) or sin 30
    let deg_to_rad = |m: &regex::Captures| -> String {
        let inner = m.get(2).unwrap().as_str().trim();
        if regex_cache(r"[-+]?\d+(?:\.\d+)?").is_match(inner) {
            format!("{}({}*pi/180)", m.get(1).unwrap().as_str(), inner)
        } else {
            m.get(0).unwrap().as_str().to_string()
        }
    };
    let expr2 = regex_cache(
        r"(?i)(sin|cos|tan)\s*\(?\s*([-+]?\d+(?:\.\d+)?)\s*(?:\u{00b0}|degrees?)?\s*\)?",
    )
    .replace_all(&expr, deg_to_rad)
    .to_string();
    let expr2 = expr2.replace('\u{00b0}', "").replace("degrees", "");
    if !expr2
        .chars()
        .all(|c| matches!(c, '-' | '+' | '*' | '/' | '(' | ')' | '.' | ' ' | '0'..='9' | 'a'..='z' | 'A'..='Z'))
    {
        return None;
    }
    if !regex_cache(r"(?i)\b(?:sin|cos|tan)\b").is_match(&expr2) {
        return None;
    }
    // reject any alphabetic token other than sin/cos/tan/pi
    let probe = regex_cache(r"(?i)\b(?:sin|cos|tan|pi)\b")
        .replace_all(&expr2, "")
        .to_string();
    if probe.chars().any(|c| c.is_alphabetic()) {
        return None;
    }
    let expr2 = expr2.replace(' ', "");
    let mut matched = HashMap::new();
    matched.insert("expr".to_string(), expr2.clone());
    Some(NLParse::ok(
        "trig_evaluate",
        obj(vec![("expr", json!(expr2))]),
        "trig expression; bare angles interpreted as DEGREES (converted to radians)",
        matched,
        0.85,
    ))
}

fn route_determinant(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    if !has(text, &[r"determinant|det\s*\("]) {
        return None;
    }
    let m = regex_cache(r"(\[\s*(?:\[\s*[-\d.,\s]+\s*\]\s*,?\s*)+\])").captures(text)?;
    // Python: json.loads of the bracket text with "," normalization
    let raw = m.get(1).unwrap().as_str();
    let normalized = regex_cache(r"\]\s*,\s*\[").replace_all(raw, "],[").to_string();
    let json_text = normalized.replace(", ", ",").replace(' ', "");
    let rows: Vec<Vec<f64>> = match serde_json::from_str::<Vec<Vec<f64>>>(&json_text) {
        Ok(r) => r,
        Err(_) => return None,
    };
    let n = rows.len();
    if rows.iter().any(|r| r.len() != n) {
        return None;
    }
    let mut matched = HashMap::new();
    matched.insert("matrix".to_string(), raw.to_string());
    Some(NLParse::ok(
        "matrix_determinant",
        obj(vec![("matrix", json!(rows))]),
        "determinant + literal matrix",
        matched,
        0.95,
    ))
}

fn route_word_problem(text: &str, _qs: &[Quantity]) -> Option<NLParse> {
    // Delegate to the verified template word-problem parser (Phase 028).
    if !has(text, &[r"a number|times a number"]) {
        return None;
    }
    // spelled-out numbers -> digits first (longest word first)
    let mut digitized = text.to_string();
    let mut word_pairs: Vec<(&str, f64)> = words().to_vec();
    word_pairs.sort_by_key(|(w, _)| std::cmp::Reverse(w.len()));
    for (w, v) in word_pairs {
        let pat = format!(r"(?i)\b{}\b", w);
        let repl = if v == v.trunc() {
            format!("{}", v as i64)
        } else {
            reasoning_common::py_float_str(v)
        };
        digitized = regex_cache(&pat).replace_all(&digitized, repl.as_str()).to_string();
    }
    // the verified templates match "equals"; normalize "is"/"=" phrasing
    digitized = regex_cache(r"(?i)\bis\b").replace_all(&digitized, "equals").to_string();
    let parsed = reasoning_curriculum::word_problems::parse_word_problem(&digitized)?;
    let mut matched = HashMap::new();
    matched.insert("equation".to_string(), parsed.equation.clone());
    Some(NLParse::ok(
        "word_problem",
        obj(vec![("text", json!(digitized))]),
        &format!("matched word-problem template -> {}", parsed.equation),
        matched,
        0.9,
    ))
}

/// The 8 math routes, in the Python's exact order.
pub fn math_routes_nl() -> &'static [crate::parser::NamedRoute] {
    static ROUTES: std::sync::OnceLock<Vec<crate::parser::NamedRoute>> =
        std::sync::OnceLock::new();
    ROUTES.get_or_init(|| {
        use crate::parser::NamedRoute;
        vec![
            NamedRoute { name: "_route_word_problem", f: route_word_problem },
            NamedRoute { name: "_route_combinatorics", f: route_combinatorics },
            NamedRoute { name: "_route_gcd_bezout", f: route_gcd_bezout },
            NamedRoute { name: "_route_determinant", f: route_determinant },
            NamedRoute { name: "_route_quadratic", f: route_quadratic },
            NamedRoute { name: "_route_linear", f: route_linear },
            NamedRoute { name: "_route_trig_evaluate", f: route_trig_evaluate },
            NamedRoute { name: "_route_arithmetic", f: route_arithmetic },
        ]
    })
}
