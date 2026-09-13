//! Phase 010 — Evaluation harness (design doc section 24) (Rust port of
//! `python/evaluation/benchmark.py`).
//!
//! Produces a plain, measured report: accuracy, search cost, and PRM
//! calibration (does the PRM's predicted confidence match its actual success
//! rate?). No composite "IQ" score, no single number claiming overall
//! capability — multiple honest metrics instead, per the project's own rule
//! against fake intelligence scores.

use std::collections::HashMap;
use std::rc::Rc;

use reasoning_common::{py_round, Rat};
use reasoning_curriculum::generator::{generate_curriculum, Problem};
use reasoning_curriculum::word_problems::solve_word_problem;
use reasoning_prm::{extract_features, LinearPRM};
use reasoning_search::{
    make_initial_state, Domain, Mcts, BezoutIdentityDomain, CombinatoricsDomain,
    ExpectedValueDomain, LinearEquationDomain, LogicFormula, NumberTargetDomain,
    QuadraticFactoringDomain, SatisfiabilityDomain,
};
use reasoning_verifier::symbolic_verifier::verify_equation_solution;

/// One domain's plain numbers (Python's per-domain dict: n_problems,
/// accuracy, avg_nodes_expanded, budget_per_problem).
#[derive(Debug, Clone, PartialEq)]
pub struct DomainEval {
    pub n_problems: usize,
    pub accuracy: f64,
    pub avg_nodes_expanded: f64,
    pub budget_per_problem: usize,
}

impl DomainEval {
    /// Python's empty-dict default (`accuracy` reads as 0.0 when unset).
    fn empty() -> DomainEval {
        DomainEval {
            n_problems: 0,
            accuracy: 0.0,
            avg_nodes_expanded: 0.0,
            budget_per_problem: 0,
        }
    }

    fn new(n_problems: usize, accuracy: f64, avg_nodes_expanded: f64, budget: usize) -> Self {
        DomainEval {
            n_problems,
            accuracy,
            avg_nodes_expanded,
            budget_per_problem: budget,
        }
    }
}

/// One calibration bucket (Python dict keys kept as field names).
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationBucket {
    pub n: usize,
    pub avg_predicted: f64,
    pub actual_success_rate: f64,
    pub gap: f64,
}

/// Python dict {"buckets": [...], "mean_calibration_gap": float|None}.
#[derive(Debug, Clone, PartialEq)]
pub struct PrmCalibration {
    pub buckets: Vec<CalibrationBucket>,
    pub mean_calibration_gap: Option<f64>,
}

impl PrmCalibration {
    fn empty() -> PrmCalibration {
        PrmCalibration {
            buckets: Vec::new(),
            mean_calibration_gap: None,
        }
    }
}

/// Python `@dataclass BenchmarkReport`.
#[derive(Debug, Clone, PartialEq)]
pub struct BenchmarkReport {
    pub level1_number_target: DomainEval,
    pub level2_linear_equation: DomainEval,
    pub prm_calibration: PrmCalibration,
}

impl Default for BenchmarkReport {
    fn default() -> Self {
        BenchmarkReport {
            level1_number_target: DomainEval::empty(),
            level2_linear_equation: DomainEval::empty(),
            prm_calibration: PrmCalibration::empty(),
        }
    }
}

/// Python `@dataclass`-less dict returned by `run_full_benchmark` —
/// modeled as a struct with the same keys.
#[derive(Debug, Clone, PartialEq)]
pub struct FullReport {
    pub level1_number_target: DomainEval,
    pub level2_linear_equation: DomainEval,
    pub level3_quadratic_factoring: DomainEval,
    pub level5_number_theory_bezout: DomainEval,
    pub prm_calibration: PrmCalibration,
}

/// Phase 029: `run_phase_1_30_benchmark`'s dict (FullReport + two keys).
#[derive(Debug, Clone, PartialEq)]
pub struct Phase130Report {
    pub level1_number_target: DomainEval,
    pub level2_linear_equation: DomainEval,
    pub level3_quadratic_factoring: DomainEval,
    pub level5_number_theory_bezout: DomainEval,
    pub level3_combinatorics: DomainEval,
    pub level4_word_problems: DomainEval,
    pub prm_calibration: PrmCalibration,
}

/// Phase 040 capstone: Phase130Report + logic + statistics.
#[derive(Debug, Clone, PartialEq)]
pub struct Phase140Report {
    pub level1_number_target: DomainEval,
    pub level2_linear_equation: DomainEval,
    pub level3_quadratic_factoring: DomainEval,
    pub level5_number_theory_bezout: DomainEval,
    pub level3_combinatorics: DomainEval,
    pub level4_word_problems: DomainEval,
    pub level5_logic: DomainEval,
    pub level5_statistics: DomainEval,
    pub prm_calibration: PrmCalibration,
}

fn payload_f64(p: &Problem, key: &str) -> f64 {
    match p.payload_get(key) {
        Some(reasoning_curriculum::generator::PayloadValue::Int(v)) => *v as f64,
        _ => panic!("payload key '{}' missing or not an int", key),
    }
}

fn payload_int_list(p: &Problem, key: &str) -> Vec<f64> {
    match p.payload_get(key) {
        Some(reasoning_curriculum::generator::PayloadValue::IntList(v)) => {
            v.iter().map(|&x| x as f64).collect()
        }
        _ => panic!("payload key '{}' missing or not an int list", key),
    }
}

fn payload_str(p: &Problem, key: &str) -> String {
    match p.payload_get(key) {
        Some(reasoning_curriculum::generator::PayloadValue::Str(s)) => s.clone(),
        _ => panic!("payload key '{}' missing or not a string", key),
    }
}

/// Python `eval_number_target`.
pub fn eval_number_target(problems: &[Problem], budget: usize, seed: u64) -> DomainEval {
    let mut solved = 0usize;
    let mut total_nodes = 0usize;
    for (i, p) in problems.iter().enumerate() {
        let domain = NumberTargetDomain::new(payload_f64(p, "target"));
        let state = make_initial_state(&payload_int_list(p, "numbers"));
        let mut mcts = Mcts::new(domain, 6, seed + i as u64);
        let result = mcts.search(state, budget);
        solved += result.found_verified_solution as usize;
        total_nodes += result.nodes_expanded;
    }
    let n = problems.len();
    DomainEval::new(
        n,
        if n > 0 { solved as f64 / n as f64 } else { 0.0 },
        if n > 0 { total_nodes as f64 / n as f64 } else { 0.0 },
        budget,
    )
}

/// Python `eval_linear_equation`. `Err` propagates a broken equation the
/// way Python's uncaught `ValueError` would have crashed the run (every
/// curriculum problem is pre-verified, so this is unreachable in practice).
pub fn eval_linear_equation(
    problems: &[Problem],
    budget: usize,
    seed: u64,
) -> Result<DomainEval, String> {
    let mut solved = 0usize;
    let mut total_nodes = 0usize;
    for (i, p) in problems.iter().enumerate() {
        let domain = LinearEquationDomain::try_new(&payload_str(p, "equation"), 6)?;
        let mut mcts = Mcts::new(domain.clone(), 6, seed + i as u64);
        let result = mcts.search(domain.initial_state(), budget);
        solved += result.found_verified_solution as usize;
        total_nodes += result.nodes_expanded;
    }
    let n = problems.len();
    Ok(DomainEval::new(
        n,
        if n > 0 { solved as f64 / n as f64 } else { 0.0 },
        if n > 0 { total_nodes as f64 / n as f64 } else { 0.0 },
        budget,
    ))
}

/// Python `eval_prm_calibration`: for each problem, record the PRM's
/// predicted P(success) at the root alongside whether the problem was
/// actually solved. Bucket by predicted probability and compare each
/// bucket's average prediction to its observed success rate — a
/// well-calibrated PRM should have these close together, not just high
/// accuracy.
pub fn eval_prm_calibration(
    prm: &LinearPRM,
    problems: &[Problem],
    budget: usize,
    seed: u64,
    n_buckets: usize,
) -> PrmCalibration {
    let mut records: Vec<(f64, usize)> = Vec::new();
    for (i, p) in problems.iter().enumerate() {
        let domain = NumberTargetDomain::new(payload_f64(p, "target"));
        let state = make_initial_state(&payload_int_list(p, "numbers"));
        let feats = extract_features(&state, domain.target, 6, 0);
        let pred = prm.predict(&feats);
        let mut mcts = Mcts::new(domain, 6, seed + i as u64);
        let result = mcts.search(state, budget);
        records.push((pred, result.found_verified_solution as usize));
    }

    // Python `records.sort(key=lambda r: r[0])` — stable sort by prediction.
    records.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let bucket_size = (records.len() / n_buckets).max(1);
    let mut buckets: Vec<CalibrationBucket> = Vec::new();
    let mut i = 0usize;
    while i < records.len() {
        let chunk = &records[i..(i + bucket_size).min(records.len())];
        if chunk.is_empty() {
            i += bucket_size;
            continue;
        }
        let avg_pred = chunk.iter().map(|(p, _)| *p).sum::<f64>() / chunk.len() as f64;
        let actual_rate = chunk.iter().map(|(_, a)| *a).sum::<usize>() as f64 / chunk.len() as f64;
        buckets.push(CalibrationBucket {
            n: chunk.len(),
            avg_predicted: py_round(avg_pred, 3),
            actual_success_rate: py_round(actual_rate, 3),
            gap: py_round((avg_pred - actual_rate).abs(), 3),
        });
        i += bucket_size;
    }
    let mean_gap = if buckets.is_empty() {
        None
    } else {
        Some(buckets.iter().map(|b| b.gap).sum::<f64>() / buckets.len() as f64)
    };
    PrmCalibration {
        buckets,
        mean_calibration_gap: mean_gap,
    }
}

/// Python `run_benchmark(prm, seed=42)`.
pub fn run_benchmark(prm: &LinearPRM, seed: u64) -> Result<BenchmarkReport, String> {
    let problems = generate_curriculum(12, Some(&[1, 2]), seed)?;
    let level1: Vec<Problem> = problems.iter().filter(|p| p.level == 1).cloned().collect();
    let level2: Vec<Problem> = problems.iter().filter(|p| p.level == 2).cloned().collect();

    let mut report = BenchmarkReport::default();
    report.level1_number_target = eval_number_target(&level1, 300, seed);
    // slower domain, smaller n
    report.level2_linear_equation =
        eval_linear_equation(&level2[..level2.len().min(6)], 400, seed)?;
    report.prm_calibration = eval_prm_calibration(prm, &level1, 300, seed + 1, 5);
    Ok(report)
}

/// Python `format_report`.
pub fn format_report(report: &BenchmarkReport) -> String {
    let l1 = &report.level1_number_target;
    let l2 = &report.level2_linear_equation;
    let mut lines = vec![
        "=== Math Reasoning AI — Benchmark Report ===".to_string(),
        String::new(),
    ];
    lines.push(format!(
        "Level 1 (number-target arithmetic): {:.0}% ({} problems, avg {:.0} nodes/problem, budget={} sims)",
        l1.accuracy * 100.0,
        l1.n_problems,
        l1.avg_nodes_expanded,
        l1.budget_per_problem
    ));
    lines.push(format!(
        "Level 2 (linear equations, real algebraic steps): {:.0}% ({} problems, avg {:.0} nodes/problem, budget={} sims)",
        l2.accuracy * 100.0,
        l2.n_problems,
        l2.avg_nodes_expanded,
        l2.budget_per_problem
    ));
    lines.push(String::new());
    lines.push(
        "PRM calibration (predicted confidence vs actual outcome, by bucket):".to_string(),
    );
    for b in &report.prm_calibration.buckets {
        lines.push(format!(
            "  n={:>2}  predicted={:.2}  actual={:.2}  gap={:.2}",
            b.n, b.avg_predicted, b.actual_success_rate, b.gap
        ));
    }
    if let Some(gap) = report.prm_calibration.mean_calibration_gap {
        lines.push(format!("  mean calibration gap: {:.3}", gap));
    }
    lines.join("\n")
}

/// Python `eval_logic`. Problems are `(formula, variables)` pairs.
pub fn eval_logic(
    problems: &[(LogicFormula, Vec<String>)],
    budget: usize,
    seed: u64,
) -> DomainEval {
    let mut solved = 0usize;
    for (i, (formula, variables)) in problems.iter().enumerate() {
        let domain = SatisfiabilityDomain::new(formula.clone(), variables.clone());
        let mut mcts = Mcts::new(domain.clone(), variables.len(), seed + i as u64);
        let result = mcts.search(mcts.domain.initial_state(), budget);
        // correct behavior: satisfiable -> must find; unsatisfiable -> must NOT find
        let correct = result.found_verified_solution == domain.is_satisfiable;
        solved += correct as usize;
    }
    let n = problems.len();
    DomainEval::new(
        n,
        if n > 0 { solved as f64 / n as f64 } else { 0.0 },
        0.0,
        budget,
    )
}

/// Python `eval_statistics`. Problems are `(outcomes, metric)` pairs.
pub fn eval_statistics(
    problems: &[(Vec<(f64, Rat)>, String)],
    budget: usize,
    seed: u64,
) -> DomainEval {
    let mut solved = 0usize;
    for (i, (outcomes, metric)) in problems.iter().enumerate() {
        let domain = ExpectedValueDomain::new(outcomes.clone(), metric, 12);
        let mut mcts = Mcts::new(domain, 2, seed + i as u64);
        let result = mcts.search(mcts.domain.initial_state(), budget);
        solved += result.found_verified_solution as usize;
    }
    let n = problems.len();
    DomainEval::new(
        n,
        if n > 0 { solved as f64 / n as f64 } else { 0.0 },
        0.0,
        budget,
    )
}

/// Python `run_phase_1_40_benchmark`: Phase 040 capstone, every domain
/// built across the whole project.
pub fn run_phase_1_40_benchmark(prm: &LinearPRM, seed: u64) -> Result<Phase140Report, String> {
    let base = run_phase_1_30_benchmark(prm, seed)?;

    let logic_problems: Vec<(LogicFormula, Vec<String>)> = vec![
        // (a["p"] or a["q"]) and (not a["p"] or a["r"])
        (
            {
                let f: LogicFormula = Rc::new(|a: &HashMap<String, bool>| {
                    (a["p"] || a["q"]) && (!a["p"] || a["r"])
                });
                f
            },
            vec!["p".to_string(), "q".to_string(), "r".to_string()],
        ),
        // a["p"] and not a["p"] — UNSAT, correct answer is "not found"
        (
            {
                let f: LogicFormula =
                    Rc::new(|a: &HashMap<String, bool>| a["p"] && !a["p"]);
                f
            },
            vec!["p".to_string()],
        ),
        // a["p"] != a["q"]
        (
            {
                let f: LogicFormula = Rc::new(|a: &HashMap<String, bool>| a["p"] != a["q"]);
                f
            },
            vec!["p".to_string(), "q".to_string()],
        ),
    ];
    let stats_problems: Vec<(Vec<(f64, Rat)>, String)> = vec![
        (
            (1..=6).map(|v| (v as f64, Rat::new(1, 6))).collect(),
            "expectation".to_string(),
        ),
        (
            vec![(1.0, Rat::new(1, 2)), (0.0, Rat::new(1, 2))],
            "variance".to_string(),
        ),
    ];

    let level5_logic = eval_logic(&logic_problems, 200, seed);
    let level5_statistics = eval_statistics(&stats_problems, 1500, seed);

    Ok(Phase140Report {
        level1_number_target: base.level1_number_target,
        level2_linear_equation: base.level2_linear_equation,
        level3_quadratic_factoring: base.level3_quadratic_factoring,
        level5_number_theory_bezout: base.level5_number_theory_bezout,
        level3_combinatorics: base.level3_combinatorics,
        level4_word_problems: base.level4_word_problems,
        level5_logic,
        level5_statistics,
        prm_calibration: base.prm_calibration,
    })
}

/// Python `format_phase_1_40_report`.
pub fn format_phase_1_40_report(report: &Phase140Report) -> String {
    // Phase 040's text = Phase 030's text + logic/statistics lines (Python
    // calls format_phase_1_30_report on the same underlying report).
    let mut lines = vec![format_full_report_inner(
        &report.level1_number_target,
        &report.level2_linear_equation,
        &report.level3_quadratic_factoring,
        &report.level5_number_theory_bezout,
        &report.prm_calibration,
    )];
    lines.push(String::new());
    lines.extend(phase_130_extra_lines(
        &report.level3_combinatorics,
        &report.level4_word_problems,
    ));
    lines.push(String::new());
    for (r, label) in [
        (&report.level5_logic, "Level 5 — propositional logic (SAT/UNSAT)"),
        (
            &report.level5_statistics,
            "Level 5 — statistics (exact E[X]/Var(X))",
        ),
    ] {
        lines.push(format!(
            "{}: {:.0}% ({} problems, budget={} sims)",
            label,
            r.accuracy * 100.0,
            r.n_problems,
            r.budget_per_problem
        ));
    }
    lines.join("\n")
}

/// Python `eval_quadratic_factoring`. Problems are `(b, c)` pairs.
pub fn eval_quadratic_factoring(
    problems: &[(i64, i64)],
    budget: usize,
    seed: u64,
) -> DomainEval {
    let mut solved = 0usize;
    let mut total_nodes = 0usize;
    for (i, (b, c)) in problems.iter().enumerate() {
        let domain = QuadraticFactoringDomain::new(*b, *c, 15);
        let mut mcts = Mcts::new(domain, 2, seed + i as u64);
        let result = mcts.search(mcts.domain.initial_state(), budget);
        solved += result.found_verified_solution as usize;
        total_nodes += result.nodes_expanded;
    }
    let n = problems.len();
    DomainEval::new(
        n,
        if n > 0 { solved as f64 / n as f64 } else { 0.0 },
        if n > 0 { total_nodes as f64 / n as f64 } else { 0.0 },
        budget,
    )
}

/// Python `eval_bezout`. Problems are `(a, b)` pairs.
pub fn eval_bezout(problems: &[(i64, i64)], budget: usize, seed: u64) -> DomainEval {
    let mut solved = 0usize;
    let mut total_nodes = 0usize;
    for (i, (a, b)) in problems.iter().enumerate() {
        let domain = BezoutIdentityDomain::new(*a, *b, 15);
        let mut mcts = Mcts::new(domain, 3, seed + i as u64);
        let result = mcts.search(mcts.domain.initial_state(), budget);
        solved += result.found_verified_solution as usize;
        total_nodes += result.nodes_expanded;
    }
    let n = problems.len();
    DomainEval::new(
        n,
        if n > 0 { solved as f64 / n as f64 } else { 0.0 },
        if n > 0 { total_nodes as f64 / n as f64 } else { 0.0 },
        budget,
    )
}

/// Python `run_full_benchmark`: Phase 020, the complete evaluation across
/// every domain built so far (number-target arithmetic, real algebra,
/// quadratic factoring, number theory), in one honest report. This is the
/// apps/evaluate entry point the design doc's crate layout called for.
pub fn run_full_benchmark(prm: &LinearPRM, seed: u64) -> Result<FullReport, String> {
    let base = run_benchmark(prm, seed)?;

    let quad_problems = [(-5, 6), (7, 10), (-1, -6), (2, -8), (-7, 12)];
    let bezout_problems = [(35, 12), (24, 18), (17, 5), (100, 63)];

    Ok(FullReport {
        level1_number_target: base.level1_number_target,
        level2_linear_equation: base.level2_linear_equation,
        level3_quadratic_factoring: eval_quadratic_factoring(&quad_problems, 300, seed),
        level5_number_theory_bezout: eval_bezout(&bezout_problems, 800, seed),
        prm_calibration: base.prm_calibration,
    })
}

/// Python `eval_combinatorics`. Problems are `(kind, n, r)` triples.
pub fn eval_combinatorics(
    problems: &[(String, i64, i64)],
    budget: usize,
    seed: u64,
) -> DomainEval {
    let mut solved = 0usize;
    let mut total_nodes = 0usize;
    for (i, (kind, n, r)) in problems.iter().enumerate() {
        // Python: CombinatoricsDomain(kind=kind, n=n, r=r, search_radius=300)
        // — `CombinatoricsDomain::new` uses exactly that default radius.
        let domain = CombinatoricsDomain::new(kind, *n, *r);
        let mut mcts = Mcts::new(domain, 1, seed + i as u64);
        let result = mcts.search(mcts.domain.initial_state(), budget);
        solved += result.found_verified_solution as usize;
        total_nodes += result.nodes_expanded;
    }
    let n = problems.len();
    DomainEval::new(
        n,
        if n > 0 { solved as f64 / n as f64 } else { 0.0 },
        if n > 0 { total_nodes as f64 / n as f64 } else { 0.0 },
        budget,
    )
}

/// Python `eval_word_problems`.
pub fn eval_word_problems(problems: &[&str], budget: usize, seed: u64) -> DomainEval {
    let mut solved = 0usize;
    for (i, text) in problems.iter().enumerate() {
        let (equation, answer) = solve_word_problem(text, budget, seed + i as u64);
        if let (Some(equation), Some(answer)) = (equation, answer) {
            // Python's verify_equation_solution raises on a bad equation;
            // here a parse failure fails closed as not-solved.
            let check = verify_equation_solution(&equation, "x", answer, 1e-9);
            solved += check.map(|c| c.passed as usize).unwrap_or(0);
        }
    }
    let n = problems.len();
    DomainEval::new(
        n,
        if n > 0 { solved as f64 / n as f64 } else { 0.0 },
        0.0,
        budget,
    )
}

/// Python `run_phase_1_30_benchmark`: extends run_full_benchmark (Phase 020)
/// with the domains added in Phases 021-028 (combinatorics, word problems).
pub fn run_phase_1_30_benchmark(prm: &LinearPRM, seed: u64) -> Result<Phase130Report, String> {
    let base = run_full_benchmark(prm, seed)?;

    let combo_problems: Vec<(String, i64, i64)> = vec![
        ("combinations".to_string(), 5, 2),
        ("permutations".to_string(), 5, 2),
        ("combinations".to_string(), 6, 3),
        ("permutations".to_string(), 6, 2),
    ];
    let word_problems = [
        "4 more than 3 times a number is 19",
        "The sum of a number and 7 is 15",
        "A number minus 5 is 10",
        "6 times a number is 42",
    ];

    Ok(Phase130Report {
        level1_number_target: base.level1_number_target,
        level2_linear_equation: base.level2_linear_equation,
        level3_quadratic_factoring: base.level3_quadratic_factoring,
        level5_number_theory_bezout: base.level5_number_theory_bezout,
        level3_combinatorics: eval_combinatorics(&combo_problems, 200, seed),
        level4_word_problems: eval_word_problems(&word_problems, 600, seed),
        prm_calibration: base.prm_calibration,
    })
}

/// Python `format_full_report`.
pub fn format_full_report(report: &FullReport) -> String {
    format_full_report_inner(
        &report.level1_number_target,
        &report.level2_linear_equation,
        &report.level3_quadratic_factoring,
        &report.level5_number_theory_bezout,
        &report.prm_calibration,
    )
}

fn format_full_report_inner(
    level1: &DomainEval,
    level2: &DomainEval,
    level3_quad: &DomainEval,
    level5_bezout: &DomainEval,
    calibration: &PrmCalibration,
) -> String {
    let mut lines = vec![
        "=== Math Reasoning AI — Full Benchmark Report (Phases 1-20) ===".to_string(),
        String::new(),
    ];
    for (r, label) in [
        (level1, "Level 1 — number-target arithmetic"),
        (level2, "Level 2 — linear equations (real algebraic steps)"),
        (level3_quad, "Level 3 — quadratic factoring"),
        (level5_bezout, "Level 5 — number theory (Bezout identity)"),
    ] {
        lines.push(format!(
            "{}: {:.0}% ({} problems, avg {:.0} nodes/problem, budget={} sims)",
            label,
            r.accuracy * 100.0,
            r.n_problems,
            r.avg_nodes_expanded,
            r.budget_per_problem
        ));
    }
    lines.push(String::new());
    lines.push(
        "PRM calibration (predicted confidence vs actual outcome, by bucket):".to_string(),
    );
    for b in &calibration.buckets {
        lines.push(format!(
            "  n={:>2}  predicted={:.2}  actual={:.2}  gap={:.2}",
            b.n, b.avg_predicted, b.actual_success_rate, b.gap
        ));
    }
    if let Some(gap) = calibration.mean_calibration_gap {
        lines.push(format!("  mean calibration gap: {:.3}", gap));
    }
    lines.join("\n")
}

/// The two extra lines Phase 029 appends (combinatorics + word problems).
fn phase_130_extra_lines(
    level3_comb: &DomainEval,
    level4_word: &DomainEval,
) -> Vec<String> {
    [
        (
            level3_comb,
            "Level 3 — combinatorics (brute-force verified)",
        ),
        (
            level4_word,
            "Level 4 — word problems (parsed -> algebra)",
        ),
    ]
    .iter()
    .map(|(r, label)| {
        format!(
            "{}: {:.0}% ({} problems, budget={} sims)",
            label,
            r.accuracy * 100.0,
            r.n_problems,
            r.budget_per_problem
        )
    })
    .collect()
}

/// Python `format_phase_1_30_report`.
pub fn format_phase_1_30_report(report: &Phase130Report) -> String {
    let base_text = format_full_report_inner(
        &report.level1_number_target,
        &report.level2_linear_equation,
        &report.level3_quadratic_factoring,
        &report.level5_number_theory_bezout,
        &report.prm_calibration,
    );
    let mut lines = vec![base_text, String::new()];
    lines.extend(phase_130_extra_lines(
        &report.level3_combinatorics,
        &report.level4_word_problems,
    ));
    lines.join("\n")
}
