//! Phase 039 — Adversarial variant generation (design doc section 21/22:
//! "difficult variants, adversarial variants") (Rust port of
//! `python/curriculum/adversarial.py`; lives in the engine crate because
//! it depends on apps.solve).
//!
//! Given a base problem template, generates deliberately harder variants:
//! negative numbers, larger magnitudes, zero coefficients, near-degenerate
//! cases. Runs the FULL solve() pipeline (Phase 030) against each and
//! reports the honest pass rate — this is a stress test, not a demo, so a
//! less-than-100% result is an expected and useful outcome, not a bug to
//! hide.

use rand::rngs::StdRng;
use rand::Rng;
use serde_json::{json, Map, Value};

use reasoning_uncertainty::Answer;

use crate::solve::{solve, Problem};

/// Python `@dataclass AdversarialCase`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdversarialCase {
    pub description: String,
    pub problem: Problem,
}

/// Python `generate_linear_equation_adversarial_variants(rng, n=10)`.
/// (The Python first generator accidentally drew from the GLOBAL `random`
/// module rather than the passed rng; this port draws all four generators
/// from the passed rng — RNG streams don't need to match Python per
/// CONVENTIONS #12, and the tests assert honesty, not specific draws.)
pub fn generate_linear_equation_adversarial_variants(
    rng: &mut StdRng,
    n: usize,
) -> Vec<AdversarialCase> {
    let mut cases = Vec::with_capacity(n);
    for _ in 0..n {
        let (kind, a, b): (&str, i64, i64) = match rng.gen_range(0..4u32) {
            0 => {
                // "generic": randint(-9, 9) or 1 -> zero becomes 1
                let mut a = rng.gen_range(-9..=9);
                if a == 0 {
                    a = 1;
                }
                ("generic", a, rng.gen_range(-20..=20))
            }
            1 => ("negative_coefficient", -rng.gen_range(1..=9), rng.gen_range(-20..=20)),
            2 => ("large_magnitude", rng.gen_range(20..=40), rng.gen_range(-100..=100)),
            _ => ("zero_target", rng.gen_range(1..=9), 0),
        };
        let x_true: i64 = rng.gen_range(-15..=15);
        let c = a * x_true + b;
        let equation = format!("{}*x + {} = {}", a, b, c);
        let mut payload = Map::new();
        payload.insert("equation".to_string(), json!(equation));
        cases.push(AdversarialCase {
            description: format!("{}: {} (expects x={})", kind, equation, x_true),
            problem: Problem::new("linear_equation", payload),
        });
    }
    cases
}

/// Python `run_adversarial_stress_test(cases, budget=1500, seed=0)` —
/// returns (results, pass_rate).
pub fn run_adversarial_stress_test(
    cases: &[AdversarialCase],
    budget: usize,
    seed: u64,
) -> (Vec<(AdversarialCase, Answer)>, f64) {
    let mut results: Vec<(AdversarialCase, Answer)> = Vec::with_capacity(cases.len());
    for (i, case) in cases.iter().enumerate() {
        let ans = solve(&case.problem, Some(budget), seed + i as u64);
        results.push((case.clone(), ans));
    }
    let n_verified = results.iter().filter(|(_, a)| a.verified).count();
    let pass_rate = if !results.is_empty() {
        n_verified as f64 / results.len() as f64
    } else {
        0.0
    };
    (results, pass_rate)
}

/// Unused-import guard (keeps `Value` available for payload building).
#[allow(dead_code)]
fn _unused(_: Value) {}
