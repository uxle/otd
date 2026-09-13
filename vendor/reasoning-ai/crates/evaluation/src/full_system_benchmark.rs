//! Phase 096 — Complete evaluation (design doc section 24: "Create a
//! rigorous benchmark system... Track actual benchmark performance")
//! (Rust port of `python/evaluation/full_system_benchmark.py`).
//!
//! Every prior benchmark in this project tested ONE domain or ONE training
//! approach at a time. Nothing has run solve() (Phase 030/089's unified API)
//! across every domain it now supports in one pass and reported a real,
//! honest per-domain scorecard. This is that.
//!
//! Two numbers matter, and they mean different things:
//!  - solve_rate: fraction of problems where the system produced a verified
//!    answer at all (allowed to be < 100%: correct abstention on a hard
//!    problem is success, not failure).
//!  - wrong_verified_rate: fraction of VERIFIED answers that were actually
//!    incorrect. This one should be 0.0 — if not, that's a correctness bug
//!    in the verifier chain somewhere; the single most important number
//!    this module produces.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use serde_json::{Map, Value};

use reasoning_common::{rat_matrix_from_rows, Rat};
use reasoning_engine::{solve, Problem};
use reasoning_verifier::symbolic_verifier::safe_parse_with_locals;

/// Python `@dataclass DomainResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainResult {
    pub kind: String,
    pub n: usize,
    pub solved: usize,
    /// verified=True but ground truth says it's actually wrong
    pub wrong_verified: usize,
}

impl DomainResult {
    /// Python `@property solve_rate`.
    pub fn solve_rate(&self) -> f64 {
        if self.n > 0 {
            self.solved as f64 / self.n as f64
        } else {
            0.0
        }
    }

    /// Python `@property wrong_verified_rate`.
    pub fn wrong_verified_rate(&self) -> f64 {
        if self.n > 0 {
            self.wrong_verified as f64 / self.n as f64
        } else {
            0.0
        }
    }
}

/// Python `@dataclass FullSystemReport`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FullSystemReport {
    pub domains: Vec<DomainResult>,
}

impl FullSystemReport {
    /// Python `@property overall_solve_rate`.
    pub fn overall_solve_rate(&self) -> f64 {
        let total_n: usize = self.domains.iter().map(|d| d.n).sum();
        let total_solved: usize = self.domains.iter().map(|d| d.solved).sum();
        if total_n > 0 {
            total_solved as f64 / total_n as f64
        } else {
            0.0
        }
    }

    /// Python `@property overall_wrong_verified_rate`.
    pub fn overall_wrong_verified_rate(&self) -> f64 {
        let total_n: usize = self.domains.iter().map(|d| d.n).sum();
        let total_wrong: usize = self.domains.iter().map(|d| d.wrong_verified).sum();
        if total_n > 0 {
            total_wrong as f64 / total_n as f64
        } else {
            0.0
        }
    }
}

/// The problem's own known ground truth (Python's oracle dict: `{}` /
/// `{"_a","_b","_c"}` / `{"matrix"}` / `{"expr"}`) — constructed by
/// generate_*, never read back from the verifier that already said yes.
#[derive(Debug, Clone, Default)]
pub enum GroundTruth {
    #[default]
    None,
    /// linear_equation: the constructed a, b, c
    Linear { a: i64, b: i64, c: i64 },
    /// matrix_determinant: the constructed matrix
    Matrix(Vec<Vec<i64>>),
    /// trig_evaluate: the original expression
    TrigExpr(String),
}

/// Python `_TRIG_LOCALS` ({"x", "sin", "cos", "tan", "pi", sqrt...} — the
/// search crate's trigonometry module keeps its own copy private, and the
/// symbolic parser knows sin/cos/tan/sqrt/pi natively; the only *local*
/// symbol it must allow is "x").
const TRIG_LOCALS: &[&str] = &["x"];

/// Python `_check_ground_truth(kind, payload, answer_str) -> bool`:
/// independent, from-scratch re-check of a verified answer against the
/// problem's own known ground truth — this is what makes
/// wrong_verified_rate a real second check, not the same check twice.
fn check_ground_truth(kind: &str, oracle: &GroundTruth, answer_str: &str) -> bool {
    match kind {
        // ground truth IS "equals target", already the verifier's job; no
        // independent oracle beyond it
        "number_target" => true,
        "linear_equation" => {
            let GroundTruth::Linear { a, b, c } = oracle else {
                return true;
            };
            // Python float(answer_str) raises on a bad parse; here a parse
            // failure fails closed (counted as wrong, surfacing the bug).
            match answer_str.parse::<f64>() {
                Ok(x) => ((*a as f64) * x + (*b as f64) - (*c as f64)).abs() < 1e-6,
                Err(_) => false,
            }
        }
        "matrix_determinant" => {
            let GroundTruth::Matrix(m) = oracle else {
                return true;
            };
            let det = rat_matrix_from_rows(m).det();
            // Python int(answer_str) == int(m.det()) — det of an integer
            // matrix is an exact integer, so a Rat comparison is exact.
            match answer_str.parse::<i64>() {
                Ok(v) => Rat::from_int(v) == det,
                Err(_) => false,
            }
        }
        "trig_evaluate" => {
            let GroundTruth::TrigExpr(expr) = oracle else {
                return true;
            };
            // Python: sympy.simplify(lhs - rhs) == 0 — the Rust symbolic
            // is_zero (norm + numeric fallback) is the equivalent exact
            // check for these standard-angle expressions.
            let lhs = match safe_parse_with_locals(expr, TRIG_LOCALS) {
                Ok(e) => e,
                Err(_) => return false,
            };
            let rhs = match safe_parse_with_locals(answer_str, TRIG_LOCALS) {
                Ok(e) => e,
                Err(_) => return false,
            };
            reasoning_symbolic::is_zero(&(lhs - rhs))
        }
        // domains without a cheap independent oracle here trust the
        // verifier chain (already covered elsewhere)
        _ => true,
    }
}

/// Python `hash(kind) % 10_000`. Python's str hash is salted per process,
/// so the original value was never reproducible across runs; this port
/// uses a deterministic FNV-1a stand-in (documented deviation — any fixed
/// hash gives the same "one stable rng per kind" semantics).
fn py_hash_mod(kind: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in kind.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h % 10_000
}

/// Python `_gen_problems(kind, n, rng)`.
fn gen_problems(kind: &str, n: usize, rng: &mut StdRng) -> Vec<(Problem, GroundTruth)> {
    let mut out = Vec::new();
    match kind {
        "number_target" => {
            for _ in 0..n {
                let nums: Vec<i64> = (0..3).map(|_| rng.gen_range(1..=9)).collect();
                // Python rng.sample(nums, rng.randint(1, 3)) — k distinct
                // elements in random order.
                let k = rng.gen_range(1..=3usize);
                let mut shuffled = nums.clone();
                shuffled.shuffle(rng);
                let target: i64 = shuffled[..k].iter().sum();
                let mut payload = Map::new();
                payload.insert(
                    "numbers".to_string(),
                    Value::Array(nums.iter().map(|&v| Value::from(v)).collect()),
                );
                payload.insert("target".to_string(), Value::from(target));
                out.push((Problem::new("number_target", payload), GroundTruth::None));
            }
        }
        "linear_equation" => {
            for _ in 0..n {
                let a = rng.gen_range(1..=9);
                let x_true = rng.gen_range(-10..=10);
                let b = rng.gen_range(-10..=10);
                let c = a * x_true + b;
                let eq = format!("{}*x + {} = {}", a, b, c);
                let mut payload = Map::new();
                payload.insert("equation".to_string(), Value::from(eq));
                out.push((
                    Problem::new("linear_equation", payload),
                    GroundTruth::Linear { a, b, c },
                ));
            }
        }
        "gcd_bezout" => {
            for _ in 0..n {
                let a = rng.gen_range(2..=40);
                let b = rng.gen_range(2..=40);
                let mut payload = Map::new();
                payload.insert("a".to_string(), Value::from(a));
                payload.insert("b".to_string(), Value::from(b));
                out.push((Problem::new("gcd_bezout", payload), GroundTruth::None));
            }
        }
        "combinatorics" => {
            for _ in 0..n {
                let ctype = ["permutations", "combinations"][rng.gen_range(0..2)];
                let nn = rng.gen_range(3..=8);
                let r = rng.gen_range(1..=nn);
                let mut payload = Map::new();
                payload.insert("ctype".to_string(), Value::from(ctype));
                payload.insert("n".to_string(), Value::from(nn));
                payload.insert("r".to_string(), Value::from(r));
                out.push((Problem::new("combinatorics", payload), GroundTruth::None));
            }
        }
        "trig_evaluate" => {
            let exprs = [
                "sin(pi/6)",
                "cos(pi/3)",
                "tan(pi/4)",
                "sin(pi/4)",
                "cos(pi/6)",
                "sin(pi/2)",
            ];
            for i in 0..n {
                let expr = exprs[i % exprs.len()];
                let mut payload = Map::new();
                payload.insert("expr".to_string(), Value::from(expr));
                out.push((
                    Problem::new("trig_evaluate", payload),
                    GroundTruth::TrigExpr(expr.to_string()),
                ));
            }
        }
        "trig_simplify" => {
            let exprs = ["sin(x)**2 + cos(x)**2", "2*sin(x)*cos(x)"];
            for i in 0..n {
                let expr = exprs[i % exprs.len()];
                let mut payload = Map::new();
                payload.insert("expr".to_string(), Value::from(expr));
                out.push((Problem::new("trig_simplify", payload), GroundTruth::None));
            }
        }
        "matrix_determinant" => {
            for _ in 0..n {
                let m: Vec<Vec<i64>> = (0..2)
                    .map(|_| (0..2).map(|_| rng.gen_range(-5..=5)).collect())
                    .collect();
                let mut payload = Map::new();
                payload.insert(
                    "matrix".to_string(),
                    Value::Array(
                        m.iter()
                            .map(|row| Value::Array(row.iter().map(|&v| Value::from(v)).collect()))
                            .collect(),
                    ),
                );
                out.push((
                    Problem::new("matrix_determinant", payload),
                    GroundTruth::Matrix(m),
                ));
            }
        }
        "matrix_multiply" => {
            for _ in 0..n {
                let a: Vec<Vec<i64>> = (0..2)
                    .map(|_| (0..2).map(|_| rng.gen_range(-5..=5)).collect())
                    .collect();
                let b: Vec<Vec<i64>> = (0..2)
                    .map(|_| (0..2).map(|_| rng.gen_range(-5..=5)).collect())
                    .collect();
                let mut payload = Map::new();
                payload.insert(
                    "a".to_string(),
                    Value::Array(
                        a.iter()
                            .map(|row| Value::Array(row.iter().map(|&v| Value::from(v)).collect()))
                            .collect(),
                    ),
                );
                payload.insert(
                    "b".to_string(),
                    Value::Array(
                        b.iter()
                            .map(|row| Value::Array(row.iter().map(|&v| Value::from(v)).collect()))
                            .collect(),
                    ),
                );
                out.push((Problem::new("matrix_multiply", payload), GroundTruth::None));
            }
        }
        "linear_system" => {
            for _ in 0..n {
                let mut a: Vec<Vec<i64>>;
                loop {
                    a = (0..2)
                        .map(|_| (0..2).map(|_| rng.gen_range(-5..=5)).collect())
                        .collect();
                    let det = a[0][0] * a[1][1] - a[0][1] * a[1][0];
                    if det != 0 {
                        break;
                    }
                }
                let x_true = [rng.gen_range(-5..=5), rng.gen_range(-5..=5)];
                let b = vec![
                    a[0][0] * x_true[0] + a[0][1] * x_true[1],
                    a[1][0] * x_true[0] + a[1][1] * x_true[1],
                ];
                let mut payload = Map::new();
                payload.insert(
                    "a".to_string(),
                    Value::Array(
                        a.iter()
                            .map(|row| Value::Array(row.iter().map(|&v| Value::from(v)).collect()))
                            .collect(),
                    ),
                );
                payload.insert(
                    "b".to_string(),
                    Value::Array(b.iter().map(|&v| Value::from(v)).collect()),
                );
                out.push((Problem::new("linear_system", payload), GroundTruth::None));
            }
        }
        _ => unreachable!("run_full_system_benchmark validates kinds up front"),
    }
    out
}

pub const DEFAULT_KINDS: [&str; 9] = [
    "number_target",
    "linear_equation",
    "gcd_bezout",
    "combinatorics",
    "trig_evaluate",
    "trig_simplify",
    "matrix_determinant",
    "matrix_multiply",
    "linear_system",
];

/// Python `run_full_system_benchmark(kinds=None, n_per_kind=15, seed=0,
/// budget=None)`.
pub fn run_full_system_benchmark(
    kinds: Option<&[&str]>,
    n_per_kind: usize,
    seed: u64,
    budget: Option<usize>,
) -> Result<FullSystemReport, String> {
    let kinds: Vec<&str> = match kinds {
        Some(k) => k.to_vec(),
        None => DEFAULT_KINDS.to_vec(),
    };
    let mut report = FullSystemReport::default();
    for kind in kinds {
        // Python raises ValueError(f"no generator for kind={kind!r}")
        if !DEFAULT_KINDS.contains(&kind) {
            return Err(format!("no generator for kind='{}'", kind));
        }
        let mut rng = StdRng::seed_from_u64(seed + py_hash_mod(kind));
        let problems = gen_problems(kind, n_per_kind, &mut rng);
        let mut solved = 0usize;
        let mut wrong_verified = 0usize;
        for (problem, oracle) in &problems {
            // Python solve(problem, budget=budget) — seed stays at its 0 default.
            let answer = solve(problem, budget, 0);
            if answer.verified {
                solved += 1;
                if let Some(final_answer) = answer.final_answer() {
                    if !check_ground_truth(kind, oracle, final_answer) {
                        wrong_verified += 1;
                    }
                }
            }
        }
        report.domains.push(DomainResult {
            kind: kind.to_string(),
            n: problems.len(),
            solved,
            wrong_verified,
        });
    }
    Ok(report)
}
