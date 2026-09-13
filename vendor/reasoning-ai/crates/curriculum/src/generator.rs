//! Phase 005 — Curriculum problem generator (Rust port of
//! python/curriculum/generator.py).
//!
//! Every generated problem is checked for a valid ground-truth solution via
//! the Phase 002 verifier *before* being added to the curriculum (design doc
//! 21: "Every generated example should pass verification before becoming
//! training data"). Difficulty levels follow design doc 3.5's
//! easiest-to-verify-first ordering.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reasoning_verifier::symbolic_verifier::solve_equation;

/// Payload value: the Python dict held ints, int lists and strings.
#[derive(Debug, Clone, PartialEq)]
pub enum PayloadValue {
    Int(i64),
    Str(String),
    IntList(Vec<i64>),
}

/// Python `@dataclass Problem`.
#[derive(Debug, Clone, PartialEq)]
pub struct Problem {
    pub level: i64,
    pub kind: String,
    pub description: String,
    /// Domain-specific data (numbers/target, or equation string).
    pub payload: Vec<(String, PayloadValue)>,
    /// What a correct answer must match: an int (levels 1-2) or a string
    /// (level 3's expanded polynomial).
    pub ground_truth: PayloadValue,
}

impl Problem {
    pub fn payload_get(&self, key: &str) -> Option<&PayloadValue> {
        self.payload
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v)
    }
}

/// Python `gen_arithmetic(rng)`. Level 1: reach a target by combining small
/// integers (number-target domain).
pub fn gen_arithmetic(rng: &mut StdRng) -> Problem {
    let n = [2usize, 3usize][rng.gen_range(0..2)];
    let nums: Vec<i64> = (0..n).map(|_| rng.gen_range(1..=9)).collect();
    // pick a target reachable by at least +/- combos so the curriculum isn't
    // accidentally full of unsolvable problems
    let target = if rng.gen::<f64>() < 0.5 {
        nums.iter().sum::<i64>()
    } else {
        (if n == 2 { nums[0] - nums[1] } else { nums[0] }).abs()
    };
    // Python f"Combine {nums} to make {target}" — list repr like "[2, 3]"
    let nums_repr = nums
        .iter()
        .map(|v| v.to_string())
        .collect::<Vec<_>>()
        .join(", ");
    Problem {
        level: 1,
        kind: "number_target".to_string(),
        description: format!("Combine [{}] to make {}", nums_repr, target),
        payload: vec![
            ("numbers".to_string(), PayloadValue::IntList(nums)),
            ("target".to_string(), PayloadValue::Int(target)),
        ],
        ground_truth: PayloadValue::Int(target),
    }
}

/// Python `gen_linear_equation(rng)`. Level 2: solve a*x + b = c for x, with
/// an integer solution guaranteed by construction (a divides c - b).
///
/// Returns Err for the cases Python raised on (`VerificationError` from the
/// solver, or the sanity `assert`) so `generate_curriculum` can discard them
/// like Python's `except (VerificationError, AssertionError): continue`.
pub fn gen_linear_equation(rng: &mut StdRng) -> Result<Problem, String> {
    let a = rng.gen_range(1..=9);
    let x_true = rng.gen_range(-10..=10);
    let b = rng.gen_range(-10..=10);
    let c = a * x_true + b;
    let eq_str = format!("{}*x + {} = {}", a, b, c);
    let roots = solve_equation(&eq_str, "x").map_err(|e| e.0)?;
    let root_val = roots
        .first()
        .and_then(|r| r.as_rat())
        .map(|r| r.to_f64())
        .ok_or_else(|| format!("solver returned no numeric root for {}", eq_str))?;
    if !(roots.len() == 1 && root_val == x_true as f64) {
        // Python assert message (kept verbatim)
        return Err(format!(
            "curriculum generator bug: constructed {} expecting x={}, solver said {:?}",
            eq_str, x_true, roots
        ));
    }
    Ok(Problem {
        level: 2,
        kind: "linear_equation".to_string(),
        description: format!("Solve {}", eq_str),
        payload: vec![
            ("equation".to_string(), PayloadValue::Str(eq_str)),
            ("variable".to_string(), PayloadValue::Str("x".to_string())),
        ],
        ground_truth: PayloadValue::Int(x_true),
    })
}

/// Python `gen_polynomial_identity(rng)`. Level 3: verify/construct an
/// expansion identity, e.g. (x+a)(x+b) = x^2+(a+b)x+ab.
pub fn gen_polynomial_identity(rng: &mut StdRng) -> Problem {
    let a = rng.gen_range(-6..=6);
    let b = rng.gen_range(-6..=6);
    let lhs = format!("(x+{})*(x+{})", a, b);
    // ground truth expanded form, for the solver to compare against via
    // verify_algebraic_equivalence at evaluation time
    let rhs = format!("x**2 + {}*x + {}", a + b, a * b);
    Problem {
        level: 3,
        kind: "polynomial_identity".to_string(),
        description: format!("Expand {}", lhs),
        payload: vec![("lhs".to_string(), PayloadValue::Str(lhs))],
        ground_truth: PayloadValue::Str(rhs),
    }
}

fn run_generator(
    level: i64,
    rng: &mut StdRng,
) -> Result<Problem, String> {
    match level {
        1 => Ok(gen_arithmetic(rng)),
        2 => gen_linear_equation(rng),
        3 => Ok(gen_polynomial_identity(rng)),
        other => Err(format!("no generator for level {}", other)),
    }
}

/// Python `generate_curriculum(num_per_level, levels=None, seed=0)`.
/// Raises RuntimeError (Err here) if a level can't be filled.
pub fn generate_curriculum(
    num_per_level: usize,
    levels: Option<&[i64]>,
    seed: u64,
) -> Result<Vec<Problem>, String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let default_levels = [1i64, 2, 3];
    let levels: &[i64] = levels.unwrap_or(&default_levels);
    let mut problems: Vec<Problem> = Vec::new();
    for &level in levels {
        let mut count = 0usize;
        let mut attempts = 0usize;
        while count < num_per_level && attempts < num_per_level * 20 {
            attempts += 1;
            match run_generator(level, &mut rng) {
                // discard, never let a broken problem into the curriculum
                Err(_) => continue,
                Ok(p) => {
                    problems.push(p);
                    count += 1;
                }
            }
        }
        if count < num_per_level {
            return Err(format!(
                "Could only generate {}/{num_per_level} verified problems at level {}",
                count, level
            ));
        }
    }
    Ok(problems)
}
