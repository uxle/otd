//! Phase 030 — Unified solve() API (Rust port of `python/apps/solve.py`).
//!
//! The single public entry point for the whole project so far. Routes a
//! problem (given explicitly typed, not free-text-classified) to the right
//! domain, runs it through MCTS + verification, and returns an Answer
//! (Phase 024) that structurally cannot lie about being verified.
//! Since Phases 110-123 this also routes the expanded science and puzzle
//! domains through `science_solve` — still typed, still verified; the NL
//! layer lives separately in `nlp_solver`.

use serde_json::{Map, Value};

use reasoning_common::{py_float_str, rat_matrix_from_rows};
use reasoning_search::{
    BezoutIdentityDomain, CombinatoricsDomain, Domain, LinearSystemDomain, Mcts,
    MatrixDeterminantDomain, MatrixMultiplyDomain, NumberTargetDomain, QuadraticFactoringDomain,
    TrigEvaluateDomain, TrigSimplifyDomain, make_initial_state,
};
use reasoning_uncertainty::{solve_with_abstention, Answer};
use reasoning_verifier::{
    verify_algebraic_equivalence, verify_equation_solution, verify_numeric_equality,
};

use crate::science_solve::science_routes;

/// Python `@dataclass Problem` — `payload` is a JSON object so it can cross
/// process boundaries exactly like the Python dict did (nlp routes, the
/// API entry point, and the audit log all serialize it).
#[derive(Debug, Clone, PartialEq)]
pub struct Problem {
    pub kind: String,
    pub payload: Map<String, Value>,
}

impl Problem {
    /// Python positional construction: `Problem("linear_equation", {...})`.
    pub fn new(kind: impl Into<String>, payload: Map<String, Value>) -> Problem {
        Problem {
            kind: kind.into(),
            payload,
        }
    }
}

// ---------------- payload accessors ----------------
// Python `payload["key"]` raises KeyError (uncaught in solve()); Rust maps
// a missing/mistyped key to an honest abstain with the KeyError-style
// message instead of panicking (CONVENTIONS #7 — never panic in library
// code).

fn payload_key_err(kind: &str, key: &str) -> Answer {
    Answer::new(false, None, 0.0, None, &format!("{}: '{}'", kind, key))
}

fn get_str<'a>(p: &'a Map<String, Value>, key: &str) -> Option<&'a str> {
    p.get(key)?.as_str()
}

fn get_f64(p: &Map<String, Value>, key: &str) -> Option<f64> {
    p.get(key)?.as_f64()
}

fn get_i64(p: &Map<String, Value>, key: &str) -> Option<i64> {
    p.get(key)?.as_i64()
}

/// Python's payload dicts were duck-typed: the NL parse of
/// "determinant of [[1,2],[3,4]]" produced float rows
/// (`[[float(v) for v in row] ...]`) and solve() still accepted them.
/// serde_json keeps 1.0 float-backed, where `as_i64()` is None — coerce
/// integral floats at the accessor boundary (same net behavior as
/// Python's duck typing; non-integral floats still fail).
fn value_as_i64(v: &Value) -> Option<i64> {
    match v.as_i64() {
        Some(i) => Some(i),
        None => {
            let f = v.as_f64()?;
            if f.fract() == 0.0 && f.is_finite() && f.abs() <= 9.0e15 {
                Some(f as i64)
            } else {
                None
            }
        }
    }
}

fn get_f64_vec(p: &Map<String, Value>, key: &str) -> Option<Vec<f64>> {
    let arr = p.get(key)?.as_array()?;
    arr.iter().map(|v| v.as_f64()).collect()
}

fn get_i64_vec(p: &Map<String, Value>, key: &str) -> Option<Vec<i64>> {
    let arr = p.get(key)?.as_array()?;
    arr.iter().map(|v| v.as_i64()).collect()
}

fn get_i64_matrix(p: &Map<String, Value>, key: &str) -> Option<Vec<Vec<i64>>> {
    let arr = p.get(key)?.as_array()?;
    let mut out = Vec::with_capacity(arr.len());
    for row in arr {
        let cells = row.as_array()?;
        let mut row_out = Vec::with_capacity(cells.len());
        for v in cells {
            row_out.push(value_as_i64(v)?);
        }
        out.push(row_out);
    }
    Some(out)
}

/// Python `budget or default` — `None` *or* 0 falls back to the default.
fn budget_or(budget: Option<usize>, default: usize) -> usize {
    match budget {
        Some(b) if b != 0 => b,
        _ => default,
    }
}

fn abstain(msg: &str) -> Answer {
    Answer::new(false, None, 0.0, None, msg)
}

// ---------------- per-kind solvers ----------------

pub fn solve_number_target(numbers: &[f64], target: f64, budget: usize, seed: u64) -> Answer {
    // `budget` is kept for signature parity; the abstention layer runs its
    // own fixed multi-path budget (same as the Python original).
    let _ = budget;
    let domain = NumberTargetDomain::new(target);
    let state = make_initial_state(numbers);
    solve_with_abstention(&domain, &state, target, seed)
}

/// Phase 126: normalize a single non-x variable ("s = 2 * 3") to "x" so
/// structured extractions (LLM fallback) still route correctly. Only a
/// lone lowercase letter is rewritten — anything ambiguous is refused.
/// (Python used lookbehind/lookahead regexes, unsupported by the regex
/// crate; the boundary checks are done by hand here.)

fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

/// Python `re.search(r"\bx\b", equation)` — a word boundary does NOT
/// consider '.' a word character, so "3.x" still contains a bare x.
fn has_bare_x(equation: &str) -> bool {
    let chars: Vec<char> = equation.chars().collect();
    chars.iter().enumerate().any(|(i, &c)| {
        if c != 'x' {
            return false;
        }
        let prev_ok = i == 0 || !is_word_char(chars[i - 1]);
        let next_ok = i + 1 == chars.len() || !is_word_char(chars[i + 1]);
        prev_ok && next_ok
    })
}

/// Python `re.findall(r"(?<![\w.])([a-wyz])(?![\w.])", equation)` — a
/// single lowercase letter (excluding x), not adjacent to a word char or
/// a dot. Distinct letters collected.
fn standalone_vars(equation: &str) -> Vec<char> {
    let chars: Vec<char> = equation.chars().collect();
    let boundary_ok = |prev: Option<char>, next: Option<char>| {
        let prev_ok = prev.map(|c| !is_word_char(c) && c != '.').unwrap_or(true);
        let next_ok = next.map(|c| !is_word_char(c) && c != '.').unwrap_or(true);
        prev_ok && next_ok
    };
    let mut found: Vec<char> = Vec::new();
    for (i, &c) in chars.iter().enumerate() {
        if !matches!(c, 'a'..='w' | 'y' | 'z') {
            continue;
        }
        let prev = if i == 0 { None } else { Some(chars[i - 1]) };
        let next = if i + 1 == chars.len() { None } else { Some(chars[i + 1]) };
        if boundary_ok(prev, next) && !found.contains(&c) {
            found.push(c);
        }
    }
    found
}

fn rename_single_var(equation: &str) -> String {
    if has_bare_x(equation) || !equation.contains('=') {
        return equation.to_string();
    }
    let vars = standalone_vars(equation);
    if vars.len() != 1 {
        return equation.to_string();
    }
    // Python `re.sub(rf"(?<![\w.]){v}(?![\w.])", "x", equation_str)`
    let v = vars[0];
    let chars: Vec<char> = equation.chars().collect();
    let mut out = String::with_capacity(equation.len());
    for (i, &c) in chars.iter().enumerate() {
        if c == v {
            let prev = if i == 0 { None } else { Some(chars[i - 1]) };
            let next = if i + 1 == chars.len() { None } else { Some(chars[i + 1]) };
            let prev_ok = prev.map(|p| !is_word_char(p) && p != '.').unwrap_or(true);
            let next_ok = next.map(|n| !is_word_char(n) && n != '.').unwrap_or(true);
            if prev_ok && next_ok {
                out.push('x');
                continue;
            }
        }
        out.push(c);
    }
    out
}

pub fn solve_linear_equation(equation_str: &str, budget: usize, seed: u64) -> Answer {
    let equation_str = rename_single_var(equation_str);
    let domain = match reasoning_search::LinearEquationDomain::try_new(&equation_str, 6) {
        Ok(d) => d,
        Err(e) => return abstain(&format!("Could not parse equation: {}", e)),
    };
    let mut mcts = Mcts::new(domain, 6, seed);
    let state = mcts.domain.initial_state();
    let result = mcts.search(state, budget);
    if result.found_verified_solution {
        // Python: `float(result.best_terminal_state.rhs)` (raises TypeError on
        // a non-number; a found_verified state always has a numeric rhs).
        let Some(val) = result
            .best_terminal_state
            .as_ref()
            .and_then(|s| s.rhs.as_rat())
            .map(|r| r.to_f64())
        else {
            return abstain("No verified solution found within budget. Abstaining.");
        };
        let answer = py_float_str(val);
        let check = verify_equation_solution(&equation_str, "x", val, 1e-9)
            .map(|r| r.passed)
            .unwrap_or(false);
        return Answer::new(
            true,
            Some(&answer),
            if check { 1.0 } else { 0.0 },
            None,
            "Solved via step-by-step algebraic search, independently re-verified.",
        );
    }
    abstain("No verified solution found within budget. Abstaining.")
}

pub fn solve_quadratic_factoring(b: i64, c: i64, budget: usize, seed: u64) -> Answer {
    // Python default search_radius=20.
    let domain = QuadraticFactoringDomain::new(b, c, 20);
    let mut mcts = Mcts::new(domain, 2, seed);
    let state = mcts.domain.initial_state();
    let result = mcts.search(state, budget);
    if result.found_verified_solution {
        let Some(p) = result.best_terminal_state.as_ref().and_then(|s| s.p) else {
            return abstain(
                "No integer factorization found within search radius. Abstaining \
                 (may genuinely have no integer roots).",
            );
        };
        let q = b - p;
        let expr = format!("(x+{})(x+{})", p, q);
        let check = verify_algebraic_equivalence(
            &format!("(x+{})*(x+{})", p, q),
            &format!("x**2 + {}*x + {}", b, c),
        )
        .map(|r| r.passed)
        .unwrap_or(false);
        return Answer::new(
            true,
            Some(&expr),
            if check { 1.0 } else { 0.0 },
            None,
            "Factored and independently re-verified by algebraic equivalence.",
        );
    }
    abstain(
        "No integer factorization found within search radius. Abstaining \
         (may genuinely have no integer roots).",
    )
}

fn math_gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

pub fn solve_gcd_bezout(a: i64, b: i64, budget: usize, seed: u64) -> Answer {
    // Phase 101: BezoutIdentityDomain is a single-level search over x (y is
    // deterministic given x), so search_radius bounds are cheap to widen —
    // sized to the coefficient scale, capped so pathological inputs don't
    // blow up the space.
    let g = {
        let g = math_gcd(a, b);
        if g == 0 {
            1
        } else {
            g
        }
    };
    let radius = 2000i64.min(200i64.max((a.abs() + b.abs()) / g));
    let mut mcts = Mcts::new(BezoutIdentityDomain::new(a, b, radius), 1, seed);
    let domain_gcd = mcts.domain.gcd;
    let state = mcts.domain.initial_state();
    let result = mcts.search(state, budget);
    if result.found_verified_solution {
        let Some(x) = result.best_terminal_state.as_ref().and_then(|s| s.x) else {
            return abstain("No Bezout coefficients found within search radius. Abstaining.");
        };
        let Some(y) = result.best_terminal_state.as_ref().and_then(|s| s.y) else {
            return abstain("No Bezout coefficients found within search radius. Abstaining.");
        };
        let expr = format!("{}*({}) + {}*({}) = {}", a, x, b, y, domain_gcd);
        let check = verify_numeric_equality(&format!("{}*({}) + {}*({})", a, x, b, y), domain_gcd as f64)
            .map(|r| r.passed)
            .unwrap_or(false);
        return Answer::new(
            true,
            Some(&expr),
            if check { 1.0 } else { 0.0 },
            None,
            "Bezout coefficients found and independently re-verified.",
        );
    }
    abstain("No Bezout coefficients found within search radius. Abstaining.")
}

pub fn solve_combinatorics(kind: &str, n: i64, r: i64, budget: usize, seed: u64) -> Answer {
    // Python default search_radius=300; __init__ raises on unknown kind —
    // surfaced as an abstain here (Rust never panics in library code).
    let domain = match CombinatoricsDomain::try_new(kind, n, r, 300) {
        Ok(d) => d,
        Err(e) => return abstain(&e),
    };
    let mut mcts = Mcts::new(domain, 1, seed);
    let state = mcts.domain.initial_state();
    let result = mcts.search(state, budget);
    if result.found_verified_solution {
        let guess = result
            .best_terminal_state
            .as_ref()
            .and_then(|s| s.guess)
            .unwrap_or_default();
        return Answer::new(
            true,
            Some(&guess.to_string()),
            1.0,
            None,
            "Matched by brute-force enumeration ground truth.",
        );
    }
    abstain("No count matched the brute-force ground truth. Abstaining.")
}

pub fn solve_word_problem_typed(text: &str, budget: usize, seed: u64) -> Answer {
    let (equation, answer) = reasoning_curriculum::solve_word_problem(text, budget, seed);
    let Some(equation) = equation else {
        return abstain(
            "Could not parse this problem against known templates. Abstaining \
             rather than guessing at an unrecognized phrasing.",
        );
    };
    let Some(answer_val) = answer else {
        return abstain(&format!(
            "Parsed as '{}' but found no verified solution. Abstaining.",
            equation
        ));
    };
    let check = verify_equation_solution(&equation, "x", answer_val, 1e-9)
        .map(|r| r.passed)
        .unwrap_or(false);
    let answer_str = py_float_str(answer_val);
    Answer::new(
        true,
        Some(&answer_str),
        if check { 1.0 } else { 0.0 },
        None,
        &format!("Parsed as '{}', solved, and independently re-verified.", equation),
    )
}

/// Candidate pool: common textbook-style closed forms, not generated by
/// peeking at the answer — covers the space a real answer usually lives
/// in, same spirit as quadratic_factoring's bounded integer search.
/// Phase 101: added the Pythagorean-identity family (sin^2, cos^2, ...).
const TRIG_SIMPLIFY_POOL: [&str; 24] = [
    "0",
    "1",
    "-1",
    "1/2",
    "-1/2",
    "sqrt(2)/2",
    "-sqrt(2)/2",
    "sqrt(3)/2",
    "-sqrt(3)/2",
    "sin(x)",
    "cos(x)",
    "tan(x)",
    "sin(2*x)",
    "cos(2*x)",
    "2*sin(x)",
    "2*cos(x)",
    "sin(x)**2",
    "cos(x)**2",
    "1 - sin(x)**2",
    "1 - cos(x)**2",
    "-sin(x)**2",
    "-cos(x)**2",
    "tan(x)**2",
    "1 + tan(x)**2",
];

const TRIG_EVALUATE_POOL: [&str; 13] = [
    "0", "1", "-1", "1/2", "-1/2", "sqrt(2)/2", "-sqrt(2)/2", "sqrt(3)/2", "-sqrt(3)/2",
    "sqrt(3)/3", "-sqrt(3)/3", "sqrt(3)", "-sqrt(3)",
];

pub fn solve_trig_simplify(expr_str: &str, budget: usize, seed: u64) -> Answer {
    let pool: Vec<String> = TRIG_SIMPLIFY_POOL.iter().map(|s| s.to_string()).collect();
    let domain = match TrigSimplifyDomain::try_new(expr_str, pool.clone()) {
        Ok(d) => d,
        Err(e) => return abstain(&format!("Could not parse expression: {}", e)),
    };
    let mut mcts = Mcts::new(domain, 1, seed);
    let state = mcts.domain.initial_state();
    let result = mcts.search(state, budget);
    if result.found_verified_solution {
        let choice = result
            .best_terminal_state
            .as_ref()
            .and_then(|s| s.choice)
            .unwrap_or_default();
        let chosen = pool[choice].clone();
        return Answer::new(
            true,
            Some(&chosen),
            1.0,
            None,
            "Matched against sympy.simplify ground truth, independently \
             re-verified by algebraic equivalence.",
        );
    }
    abstain(
        "No candidate in the standard-form pool matched. Abstaining \
         rather than guessing at an unrecognized simplified form.",
    )
}

pub fn solve_trig_evaluate(expr_str: &str, budget: usize, seed: u64) -> Answer {
    let pool: Vec<String> = TRIG_EVALUATE_POOL.iter().map(|s| s.to_string()).collect();
    let domain = match TrigEvaluateDomain::try_new(expr_str, pool.clone()) {
        Ok(d) => d,
        Err(e) => return abstain(&format!("Could not parse expression: {}", e)),
    };
    let mut mcts = Mcts::new(domain, 1, seed);
    let state = mcts.domain.initial_state();
    let result = mcts.search(state, budget);
    if result.found_verified_solution {
        let choice = result
            .best_terminal_state
            .as_ref()
            .and_then(|s| s.choice)
            .unwrap_or_default();
        let chosen = pool[choice].clone();
        return Answer::new(
            true,
            Some(&chosen),
            1.0,
            None,
            "Matched against sympy's exact evaluation, independently re-verified.",
        );
    }
    abstain(
        "No candidate in the standard-angle pool matched. Abstaining \
         (likely not a 'nice' angle this pool covers).",
    )
}

pub fn solve_matrix_determinant(matrix_rows: &[Vec<i64>], budget: Option<usize>, seed: u64) -> Answer {
    // Candidate pool is a bounded integer range — a real brute-force
    // search, not a peek at the answer. Phase 101: bound sized from the
    // matrix itself (n! * max_entry^n, a standard crude determinant
    // bound), capped so pathological inputs don't blow up the list.
    let n = matrix_rows.len();
    let max_entry = {
        let m = matrix_rows
            .iter()
            .flatten()
            .map(|v| (*v as f64).abs())
            .fold(0.0, f64::max);
        if m == 0.0 {
            1.0
        } else {
            m
        }
    };
    let mut factorial = 1.0f64;
    for i in 1..=n {
        factorial *= i as f64;
    }
    let bound_f = 20000.0f64.min(50.0f64.max(factorial * max_entry.powi(n as i32)));
    let bound = bound_f as i64;
    let candidate_pool: Vec<i64> = (-bound..=bound).collect();
    let domain = match MatrixDeterminantDomain::try_new(matrix_rows, candidate_pool.clone()) {
        Ok(d) => d,
        Err(e) => return abstain(&format!("Could not build determinant problem: {}", e)),
    };
    // Budget scales with candidate-pool size when the caller doesn't pin
    // one — a wider bound needs more simulations to resolve.
    let budget = match budget {
        Some(b) => b,
        None => 8000usize.min(200usize.max(candidate_pool.len())),
    };
    let mut mcts = Mcts::new(domain, 1, seed);
    let state = mcts.domain.initial_state();
    let result = mcts.search(state, budget);
    if result.found_verified_solution {
        let choice = result
            .best_terminal_state
            .as_ref()
            .and_then(|s| s.choice)
            .unwrap_or_default();
        let chosen = candidate_pool[choice];
        return Answer::new(
            true,
            Some(&chosen.to_string()),
            1.0,
            None,
            "Matched via bounded brute-force search over integer determinants, \
             independently re-verified against sympy.Matrix.det().",
        );
    }
    abstain(
        "Determinant outside the searched [-50, 50] range. Abstaining \
         rather than guessing.",
    )
}

/// Render a rational matrix the way Python's `str(Matrix(a)*Matrix(b).tolist())`
/// does: integers plain, rationals as `p/q` (unreachable for integer input).
fn py_matrix_repr(m: &reasoning_common::RatMatrix) -> String {
    let rows: Vec<String> = (0..m.rows)
        .map(|i| {
            let entries: Vec<String> = (0..m.cols)
                .map(|j| {
                    let r = m.at(i, j);
                    if r.den == 1 {
                        format!("{}", r.num)
                    } else {
                        format!("{}/{}", r.num, r.den)
                    }
                })
                .collect();
            format!("[{}]", entries.join(", "))
        })
        .collect();
    format!("[{}]", rows.join(", "))
}

pub fn solve_matrix_multiply(a_rows: &[Vec<i64>], b_rows: &[Vec<i64>], _seed: u64) -> Answer {
    // No generative candidate search here (a matrix-valued answer isn't
    // brute-forceable) — compute the correct product directly and still
    // route it through the domain's independent verification.
    let ragged = |rows: &[Vec<i64>]| rows.iter().any(|r| r.len() != rows[0].len());
    let a = rat_matrix_from_rows(a_rows);
    let b = rat_matrix_from_rows(b_rows);
    if ragged(a_rows) || ragged(b_rows) || a.cols != b.rows {
        return abstain(&format!(
            "Could not multiply matrices: incompatible shapes for multiplication: ({}, {}) x ({}, {})",
            a.rows, a.cols, b.rows, b.cols
        ));
    }
    let product = a.mat_mul(&b);
    // `correct = (Matrix(a_rows) * Matrix(b_rows)).tolist()` — integers for
    // integer input; a non-integer entry (unreachable here) degrades to a
    // truncated candidate that simply fails verification.
    let correct: Vec<Vec<i64>> = (0..product.rows)
        .map(|i| {
            (0..product.cols)
                .map(|j| {
                    let r = product.at(i, j);
                    if r.den == 1 {
                        r.num
                    } else {
                        r.num / r.den
                    }
                })
                .collect()
        })
        .collect();
    let domain = match MatrixMultiplyDomain::try_new(a_rows, b_rows, vec![correct]) {
        Ok(d) => d,
        Err(e) => return abstain(&format!("Could not multiply matrices: {}", e)),
    };
    let state = domain.apply(&domain.initial_state(), &0);
    let reward = domain.terminal_reward(&state);
    if reward >= 0.999 {
        let answer = py_matrix_repr(&product);
        return Answer::new(
            true,
            Some(&answer),
            1.0,
            None,
            "Computed via sympy.Matrix multiplication, independently re-verified \
             by exact entrywise comparison.",
        );
    }
    abstain("Verification of the computed product failed unexpectedly. Abstaining.")
}

pub fn solve_linear_system(a_rows: &[Vec<i64>], b_col: &[i64], _seed: u64) -> Answer {
    let mut domain = match LinearSystemDomain::try_new(a_rows, b_col, Vec::new()) {
        Ok(d) => d,
        Err(e) => return abstain(&format!("System not uniquely solvable: {}", e)),
    };
    // Python: `correct = list(domain.ground_truth)` (nsimplify'd strings),
    // `domain.candidates = [correct]`. The Rust domain takes integer
    // vectors, so the strings are parsed back (integer systems only —
    // fractional solutions would fail verification and abstain).
    let correct_strs: Vec<String> = domain.ground_truth.clone();
    let correct_ints: Vec<i64> = correct_strs
        .iter()
        .map(|s| s.parse::<i64>().unwrap_or(0))
        .collect();
    domain.candidates = vec![correct_ints];
    let state = domain.apply(&domain.initial_state(), &0);
    let reward = domain.terminal_reward(&state);
    if reward >= 0.999 {
        // Python `str(['3', '2'])` -> "['3', '2']".
        let answer = format!(
            "[{}]",
            correct_strs
                .iter()
                .map(|s| format!("'{}'", s))
                .collect::<Vec<_>>()
                .join(", ")
        );
        return Answer::new(
            true,
            Some(&answer),
            1.0,
            None,
            "Solved via sympy.Matrix.solve(), independently re-verified.",
        );
    }
    abstain("Verification of the computed solution failed unexpectedly. Abstaining.")
}

// ---------------- the single entry point ----------------

/// The single entry point. Explicit `kind` routing: science + puzzle kinds
/// first (each route internally re-verifies; failures come back as honest
/// abstentions), then the math kinds.
pub fn solve(problem: &Problem, budget: Option<usize>, seed: u64) -> Answer {
    if let Some(route) = science_routes().get(problem.kind.as_str()) {
        return route(&problem.payload);
    }
    match problem.kind.as_str() {
        "number_target" => {
            let Some(numbers) = get_f64_vec(&problem.payload, "numbers") else {
                return payload_key_err("number_target", "numbers");
            };
            let Some(target) = get_f64(&problem.payload, "target") else {
                return payload_key_err("number_target", "target");
            };
            solve_number_target(&numbers, target, budget_or(budget, 1500), seed)
        }
        "linear_equation" => {
            let Some(equation) = get_str(&problem.payload, "equation") else {
                return payload_key_err("linear_equation", "equation");
            };
            solve_linear_equation(equation, budget_or(budget, 1500), seed)
        }
        "quadratic_factoring" => {
            let Some(b) = get_i64(&problem.payload, "b") else {
                return payload_key_err("quadratic_factoring", "b");
            };
            let Some(c) = get_i64(&problem.payload, "c") else {
                return payload_key_err("quadratic_factoring", "c");
            };
            solve_quadratic_factoring(b, c, budget_or(budget, 300), seed)
        }
        "gcd_bezout" => {
            let Some(a) = get_i64(&problem.payload, "a") else {
                return payload_key_err("gcd_bezout", "a");
            };
            let Some(b) = get_i64(&problem.payload, "b") else {
                return payload_key_err("gcd_bezout", "b");
            };
            solve_gcd_bezout(a, b, budget_or(budget, 1000), seed)
        }
        "combinatorics" => {
            let Some(ctype) = get_str(&problem.payload, "ctype") else {
                return payload_key_err("combinatorics", "ctype");
            };
            let Some(n) = get_i64(&problem.payload, "n") else {
                return payload_key_err("combinatorics", "n");
            };
            let Some(r) = get_i64(&problem.payload, "r") else {
                return payload_key_err("combinatorics", "r");
            };
            solve_combinatorics(ctype, n, r, budget_or(budget, 300), seed)
        }
        "word_problem" => {
            let Some(text) = get_str(&problem.payload, "text") else {
                return payload_key_err("word_problem", "text");
            };
            solve_word_problem_typed(text, budget_or(budget, 800), seed)
        }
        "trig_simplify" => {
            let Some(expr) = get_str(&problem.payload, "expr") else {
                return payload_key_err("trig_simplify", "expr");
            };
            solve_trig_simplify(expr, budget_or(budget, 200), seed)
        }
        "trig_evaluate" => {
            let Some(expr) = get_str(&problem.payload, "expr") else {
                return payload_key_err("trig_evaluate", "expr");
            };
            solve_trig_evaluate(expr, budget_or(budget, 200), seed)
        }
        "matrix_determinant" => {
            let Some(matrix) = get_i64_matrix(&problem.payload, "matrix") else {
                return payload_key_err("matrix_determinant", "matrix");
            };
            solve_matrix_determinant(&matrix, budget, seed)
        }
        "matrix_multiply" => {
            let Some(a) = get_i64_matrix(&problem.payload, "a") else {
                return payload_key_err("matrix_multiply", "a");
            };
            let Some(b) = get_i64_matrix(&problem.payload, "b") else {
                return payload_key_err("matrix_multiply", "b");
            };
            solve_matrix_multiply(&a, &b, seed)
        }
        "linear_system" => {
            let Some(a) = get_i64_matrix(&problem.payload, "a") else {
                return payload_key_err("linear_system", "a");
            };
            let Some(b) = get_i64_vec(&problem.payload, "b") else {
                return payload_key_err("linear_system", "b");
            };
            solve_linear_system(&a, &b, seed)
        }
        _ => abstain(&format!(
            "Unknown problem kind '{}'. Abstaining.",
            problem.kind
        )),
    }
}
