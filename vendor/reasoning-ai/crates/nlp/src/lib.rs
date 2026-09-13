//! Reasoning AI — NLP crate (Rust port of python/nlp/):
//! quantity extraction + the deterministic science/puzzle/math routers.
//! The `ask()` entry point (python nlp/solver.py) lives in the engine
//! crate because it routes through apps.solve.

pub mod math_routes;
pub mod parser;
pub mod quantities;

pub use parser::{NLParse, NamedRoute};
pub use quantities::{extract_quantities, to_kelvin, Quantity};

use parser::regex_cache;

/// Deterministic NL -> typed problem (the Python `nlp.solver.parse`).
/// Returns NLParse(ok=False) with candidates when nothing matched.
pub fn parse(text: &str) -> NLParse {
    let text = text.trim();
    let qs = extract_quantities(text);

    let mut candidates: Vec<String> = Vec::new();
    let mut first_failure: Option<NLParse> = None;
    for route in parser::science_routes_nl()
        .iter()
        .chain(math_routes::math_routes_nl().iter())
    {
        // a route crashing must never become a wrong answer (Python's
        // try/except around each route) — routes in this port return
        // Option<NLParse> and never panic in practice; keep the same
        // defensive shape.
        let result = (route.f)(text, &qs);
        let Some(result) = result else { continue };
        if result.ok {
            let kind = result.kind.clone().unwrap_or_default();
            let mut out = result;
            out.candidates = candidates.into_iter().filter(|c| *c != kind).collect();
            return out;
        }
        candidates.push(result.rationale.chars().take(80).collect());
        if first_failure.is_none() {
            first_failure = Some(result);
        }
    }
    let rationale = match &first_failure {
        Some(f) => f.rationale.clone(),
        None => "no domain pattern matched this question".to_string(),
    };
    NLParse::fail_with_candidates(&rationale, candidates)
}

/// Warm the shared regex cache with the route patterns (so the first
/// `parse()` call doesn't pay compile costs) — optional helper.
pub fn warm_regex_cache() {
    for r in parser::science_routes_nl() {
        let _ = regex_cache(r.name);
    }
}
