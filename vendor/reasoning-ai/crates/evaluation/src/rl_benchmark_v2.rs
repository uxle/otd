//! Phase 075 — Re-run of Phase 69's comparison, fixing what Phase 69/70
//! flagged as unresolved: too few eval problems to say anything
//! statistically meaningful, and GRPO training starved of signal by
//! raw-sampling episodes (Rust port of `python/evaluation/rl_benchmark_v2.py`).
//!
//! Now compares FOUR conditions on a larger eval set with Wilson intervals
//! (Phase 074): uniform-random, PRM-guided (self_evolution_v2, unchanged),
//! GRPO trained on raw-sampled episodes (Phase 67, unchanged), and GRPO
//! trained on MCTS-sourced episodes (Phase 73, the fix).

use rand::rngs::StdRng;
use rand::SeedableRng;
use reasoning_common::py_round;
use reasoning_rl::{
    proportions_overlap, run_grpo_training, run_grpo_training_mcts, wilson_score_interval,
};
use reasoning_search::{make_initial_state, Mcts};
use reasoning_selfplay::{make_eval_set, make_round_problems, run_self_evolution_v2};

use crate::rl_benchmark::{fmt_f64_list, fmt_usize_list, make_guaranteed_solvable};

/// Python's `Problem` for selfplay (domain, numbers, max_depth).
type SelfplayProblem = reasoning_selfplay::Problem;
/// Python's `(domain, state, max_depth)` tuple for the RL training loops.
type RlProblem = reasoning_rl::Problem;

/// Wilson 95% interval (Python default z=1.96).
const Z: f64 = 1.96;

/// Python `@dataclass ConditionResult`.
#[derive(Debug, Clone, PartialEq)]
pub struct ConditionResult {
    pub name: String,
    pub successes: usize,
    pub n: usize,
    pub solve_rate: f64,
    pub ci_lo: f64,
    pub ci_hi: f64,
}

/// Python `@dataclass FourWayReport`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FourWayReport {
    pub conditions: Vec<ConditionResult>,
    pub notes: Vec<String>,
}

impl FourWayReport {
    /// Python `significantly_different_from_uniform`: names of conditions
    /// whose Wilson interval does NOT overlap the uniform baseline's — the
    /// only claim this report is willing to make plainly, everything else
    /// stays hedged.
    pub fn significantly_different_from_uniform(&self) -> Vec<String> {
        let Some(uniform) = self.conditions.iter().find(|c| c.name == "uniform") else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for c in &self.conditions {
            if c.name == "uniform" {
                continue;
            }
            let overlap = proportions_overlap(
                uniform.successes as i64,
                uniform.n as i64,
                c.successes as i64,
                c.n as i64,
                Z,
            )
            .expect("wilson inputs are valid counts by construction");
            if !overlap {
                out.push(c.name.clone());
            }
        }
        out
    }
}

/// Python `_eval_uniform(eval_problems, budget, seed) -> (solved, n)`.
fn eval_uniform(eval_problems: &[SelfplayProblem], budget: usize, seed: u64) -> (usize, usize) {
    let mut solved = 0usize;
    for (i, (domain, nums, max_depth)) in eval_problems.iter().enumerate() {
        let state = make_initial_state(nums);
        let mut mcts = Mcts::new(domain.clone(), *max_depth, seed + i as u64);
        let result = mcts.search(state, budget);
        solved += result.found_verified_solution as usize;
    }
    (solved, eval_problems.len())
}

/// Python `run_four_way_comparison(rounds=3, n_train_problems_per_round=8,
/// eval_n=40, round_budget=250, eval_budget=250,
/// train_num_simulations_mcts=150, seed=0)`.
#[allow(clippy::too_many_arguments)]
pub fn run_four_way_comparison(
    rounds: usize,
    n_train_problems_per_round: usize,
    eval_n: usize,
    round_budget: usize,
    eval_budget: usize,
    train_num_simulations_mcts: usize,
    seed: u64,
) -> Result<FourWayReport, String> {
    let eval_problems = make_eval_set(eval_n, 778);
    let eval_problems_states: Vec<RlProblem> = eval_problems
        .iter()
        .map(|(domain, nums, max_depth)| {
            (domain.clone(), make_initial_state(nums), *max_depth)
        })
        .collect();

    let mut report = FourWayReport::default();

    // 1) uniform
    let (succ, n) = eval_uniform(&eval_problems, eval_budget, 9500);
    let (lo, hi) =
        wilson_score_interval(succ as i64, n as i64, Z).expect("n > 0 by construction");
    report.conditions.push(ConditionResult {
        name: "uniform".to_string(),
        successes: succ,
        n,
        solve_rate: succ as f64 / n as f64,
        ci_lo: lo,
        ci_hi: hi,
    });

    // 2) PRM-guided (unchanged baseline)
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
    let succ = *prm_history.last().unwrap_or(&0);
    let (lo, hi) =
        wilson_score_interval(succ as i64, eval_n as i64, Z).expect("n > 0 by construction");
    report.conditions.push(ConditionResult {
        name: "prm_guided".to_string(),
        successes: succ,
        n: eval_n,
        solve_rate: succ as f64 / eval_n as f64,
        ci_lo: lo,
        ci_hi: hi,
    });

    // 3) GRPO on raw-sampled episodes (Phase 67, the version diagnosed as
    //    signal-starved)
    let raw_round_problems = |round_idx: usize| -> Vec<RlProblem> {
        let mut rng = StdRng::seed_from_u64(300 + round_idx as u64);
        (0..n_train_problems_per_round)
            .map(|_| {
                let (domain, nums, max_depth) = make_guaranteed_solvable(&mut rng, 4, 6);
                (domain, make_initial_state(&nums), max_depth)
            })
            .collect()
    };

    let (_policy_raw, history_raw) = run_grpo_training(
        rounds,
        &raw_round_problems,
        &eval_problems_states,
        6, // n_samples_per_problem
        0.2,
        0.2, // Python default epsilon
        eval_budget,
        seed,
    )?;
    let succ = py_round(history_raw.eval_solve_rate_after * eval_n as f64, 0) as i64 as usize;
    let (lo, hi) =
        wilson_score_interval(succ as i64, eval_n as i64, Z).expect("n > 0 by construction");
    report.conditions.push(ConditionResult {
        name: "grpo_raw_sampling".to_string(),
        successes: succ,
        n: eval_n,
        solve_rate: succ as f64 / eval_n as f64,
        ci_lo: lo,
        ci_hi: hi,
    });

    // 4) GRPO on MCTS-sourced episodes (Phase 73, the fix)
    let mcts_round_problems = |round_idx: usize| -> Vec<RlProblem> {
        let mut rng = StdRng::seed_from_u64(400 + round_idx as u64);
        (0..n_train_problems_per_round)
            .map(|_| {
                let (domain, nums, max_depth) = make_guaranteed_solvable(&mut rng, 4, 6);
                (domain, make_initial_state(&nums), max_depth)
            })
            .collect()
    };

    let (_policy_mcts, history_mcts, _round_diags) = run_grpo_training_mcts(
        rounds,
        &mcts_round_problems,
        &eval_problems_states,
        6, // n_samples_per_problem
        train_num_simulations_mcts,
        0.2,
        0.2, // Python default epsilon
        eval_budget,
        seed,
    )?;
    let succ = py_round(history_mcts.eval_solve_rate_after * eval_n as f64, 0) as i64 as usize;
    let (lo, hi) =
        wilson_score_interval(succ as i64, eval_n as i64, Z).expect("n > 0 by construction");
    report.conditions.push(ConditionResult {
        name: "grpo_mcts_sourced".to_string(),
        successes: succ,
        n: eval_n,
        solve_rate: succ as f64 / eval_n as f64,
        ci_lo: lo,
        ci_hi: hi,
    });

    report.notes.push(format!(
        "PRM baseline eval history (successes/{} per round): {}",
        eval_n,
        fmt_usize_list(&prm_history)
    ));
    report
        .notes
        .push(format!("GRPO-raw round solve rates during training: {}", fmt_f64_list(&history_raw.round_solve_rates)));
    report.notes.push(format!(
        "GRPO-mcts round solve rates during training: {}",
        fmt_f64_list(&history_mcts.round_solve_rates)
    ));
    report.notes.push(format!(
        "GRPO-mcts before/after MCTS-guided eval: {:.2} -> {:.2}",
        history_mcts.eval_solve_rate_before, history_mcts.eval_solve_rate_after
    ));
    Ok(report)
}
