//! Phase 069 — Comparison benchmark: uniform-random baseline vs GRPO-trained
//! policy (Phase 061-067) vs the existing PRM-guided baseline
//! (`selfplay/self_evolution_v2.py`), all evaluated on the SAME held-out
//! problem set with the SAME MCTS search budget (design doc section 24:
//! accuracy and search efficiency, no composite score) (Rust port of
//! `python/evaluation/rl_benchmark.py`).
//!
//! This is the first real test of whether Phases 051-068's RL machinery
//! produces a policy that helps — whatever this finds is reported as-is.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reasoning_common::{py_float_str, py_round};
use reasoning_rl::run_grpo_training;
use reasoning_search::{make_initial_state, Mcts, NumberTargetDomain};
use reasoning_selfplay::{make_eval_set, make_round_problems, run_self_evolution_v2};

/// Python's `Problem` for selfplay (domain, numbers, max_depth).
type SelfplayProblem = reasoning_selfplay::Problem;
/// Python's `(domain, state, max_depth)` tuple for the RL training loops.
type RlProblem = reasoning_rl::Problem;

/// Python `@dataclass ThreeWayReport`.
#[derive(Debug, Clone, PartialEq)]
pub struct ThreeWayReport {
    pub uniform_solve_rate: f64,
    pub prm_guided_solve_rate: f64,
    pub grpo_solve_rate: f64,
    pub n_eval_problems: usize,
    pub eval_budget: usize,
    pub notes: Vec<String>,
}

/// Python `_eval_uniform(eval_problems, budget, seed)`.
fn eval_uniform(eval_problems: &[SelfplayProblem], budget: usize, seed: u64) -> f64 {
    let mut solved = 0usize;
    for (i, (domain, nums, max_depth)) in eval_problems.iter().enumerate() {
        let state = make_initial_state(nums);
        let mut mcts = Mcts::new(domain.clone(), *max_depth, seed + i as u64);
        let result = mcts.search(state, budget);
        solved += result.found_verified_solution as usize;
    }
    solved as f64 / eval_problems.len() as f64
}

/// Python `_make_guaranteed_solvable` from selfplay (kept private there, so
/// this is a local replica — same construction, same guarantee).
pub(crate) fn make_guaranteed_solvable(
    rng: &mut StdRng,
    count: usize,
    max_depth: usize,
) -> (NumberTargetDomain, Vec<f64>, usize) {
    const _OPS: [char; 4] = ['+', '-', '*', '/'];
    let nums: Vec<f64> = (0..count).map(|_| rng.gen_range(1..=9) as f64).collect();
    let mut values = nums.clone();
    while values.len() > 1 {
        let n = values.len();
        // Python rng.sample(range(n), 2): two distinct indices
        let i = rng.gen_range(0..n);
        let j = (i + rng.gen_range(1..n)) % n;
        let mut op = _OPS[rng.gen_range(0.._OPS.len())];
        let (a, b) = (values[i], values[j]);
        if op == '/' && b == 0.0 {
            op = '+';
        }
        let v = match op {
            '+' => a + b,
            '-' => a - b,
            '*' => a * b,
            _ => {
                if b != 0.0 {
                    a / b
                } else {
                    a + b
                }
            }
        };
        let mut remaining: Vec<f64> = values
            .iter()
            .enumerate()
            .filter(|(k, _)| *k != i && *k != j)
            .map(|(_, x)| *x)
            .collect();
        remaining.push(v);
        values = remaining;
    }
    let target = values[0];
    (NumberTargetDomain::new(target), nums, max_depth)
}

/// Render a `Vec<usize>` the way Python's f-string renders a list.
pub(crate) fn fmt_usize_list(v: &[usize]) -> String {
    format!("[{}]", v.iter().map(|x| x.to_string()).collect::<Vec<_>>().join(", "))
}

/// Render a `Vec<f64>` the way Python's f-string renders a list of floats.
pub(crate) fn fmt_f64_list(v: &[f64]) -> String {
    format!(
        "[{}]",
        v.iter().map(|x| py_float_str(*x)).collect::<Vec<_>>().join(", ")
    )
}

/// Render a `Vec<f64>` like Python `[round(x, 3) for x in ...]`.
pub(crate) fn fmt_f64_list_round3(v: &[f64]) -> String {
    fmt_f64_list(&v.iter().map(|x| py_round(*x, 3)).collect::<Vec<_>>())
}

/// Python `run_three_way_comparison(rounds=3, n_train_problems_per_round=8,
/// eval_n=20, round_budget=250, eval_budget=60, seed=0)`.
#[allow(clippy::too_many_arguments)]
pub fn run_three_way_comparison(
    rounds: usize,
    n_train_problems_per_round: usize,
    eval_n: usize,
    round_budget: usize,
    eval_budget: usize,
    seed: u64,
) -> Result<ThreeWayReport, String> {
    let eval_problems = make_eval_set(eval_n, 777);

    // 1) Uniform-random rollout baseline — zero learning, MCTS + verifier only
    let uniform_rate = eval_uniform(&eval_problems, eval_budget, 9500);

    // 2) Existing PRM-guided baseline (selfplay/self_evolution_v2)
    let prm_round_problem_fn =
        |round_idx: usize| -> Vec<SelfplayProblem> { make_round_problems(round_idx, 8, 200) };
    let (_prm, prm_history) = run_self_evolution_v2(
        rounds,
        &prm_round_problem_fn,
        &eval_problems,
        round_budget,
        eval_budget,
        8,   // Python default root_oversample
        60,  // Python default epochs
        0.3, // Python default lr
        seed,
    );
    let prm_rate = *prm_history.last().unwrap_or(&0) as f64 / eval_n as f64;

    // 3) New GRPO-trained policy (Phase 061-067)
    let grpo_round_problems = |round_idx: usize| -> Vec<RlProblem> {
        let mut rng = StdRng::seed_from_u64(200 + round_idx as u64);
        (0..n_train_problems_per_round)
            .map(|_| {
                let (domain, nums, max_depth) = make_guaranteed_solvable(&mut rng, 4, 6);
                (domain, make_initial_state(&nums), max_depth)
            })
            .collect()
    };

    let grpo_eval_problems: Vec<RlProblem> = eval_problems
        .iter()
        .map(|(domain, nums, max_depth)| (domain.clone(), make_initial_state(nums), *max_depth))
        .collect();

    let (_policy, grpo_history) = run_grpo_training(
        rounds,
        &grpo_round_problems,
        &grpo_eval_problems,
        6, // n_samples_per_problem
        0.2,
        0.2, // Python default epsilon
        eval_budget,
        seed,
    )?;
    let grpo_rate = grpo_history.eval_solve_rate_after;

    let notes = vec![
        format!(
            "PRM baseline eval history (solved/{} per round): {}",
            eval_n,
            fmt_usize_list(&prm_history)
        ),
        format!(
            "GRPO round solve rates (direct policy sampling, no search): {}",
            fmt_f64_list(&grpo_history.round_solve_rates)
        ),
        format!(
            "GRPO before/after MCTS-guided eval solve rate: {:.2} -> {:.2}",
            grpo_history.eval_solve_rate_before, grpo_history.eval_solve_rate_after
        ),
    ];

    Ok(ThreeWayReport {
        uniform_solve_rate: uniform_rate,
        prm_guided_solve_rate: prm_rate,
        grpo_solve_rate: grpo_rate,
        n_eval_problems: eval_n,
        eval_budget,
        notes,
    })
}
