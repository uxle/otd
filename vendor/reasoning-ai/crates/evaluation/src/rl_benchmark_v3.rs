//! Phase 095 — Extends Phase 75's harness with the Phase 91-94 distillation
//! approach, so it's compared on the exact same eval set and budget as
//! uniform / PRM-guided / GRPO-raw / GRPO-MCTS-sourced, answering the
//! question Phase 91's docstring posed directly: does distilling MCTS's own
//! visit-count distribution do any better than GRPO did? (Rust port of
//! `python/evaluation/rl_benchmark_v3.py`.)

use rand::rngs::StdRng;
use rand::SeedableRng;
use reasoning_common::py_round;
use reasoning_rl::{run_distill_training, wilson_score_interval};
use reasoning_search::make_initial_state;
use reasoning_selfplay::make_eval_set;

use crate::rl_benchmark::{fmt_f64_list, fmt_f64_list_round3, make_guaranteed_solvable};
use crate::rl_benchmark_v2::{run_four_way_comparison, ConditionResult};

/// Python's `Problem` for selfplay (domain, numbers, max_depth).
type SelfplayProblem = reasoning_selfplay::Problem;
/// Python's `(domain, state, max_depth)` tuple for the RL training loops.
type RlProblem = reasoning_rl::Problem;

/// Wilson 95% interval (Python default z=1.96).
const Z: f64 = 1.96;

/// Python `@dataclass FiveWayReport`.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct FiveWayReport {
    pub conditions: Vec<ConditionResult>,
    pub notes: Vec<String>,
}

/// Python `run_five_way_comparison(rounds=3, n_train_problems_per_round=8,
/// eval_n=40, round_budget=150, eval_budget=250, seed=0)`.
#[allow(clippy::too_many_arguments)]
pub fn run_five_way_comparison(
    rounds: usize,
    n_train_problems_per_round: usize,
    eval_n: usize,
    round_budget: usize,
    eval_budget: usize,
    seed: u64,
) -> Result<FiveWayReport, String> {
    let base = run_four_way_comparison(
        rounds,
        n_train_problems_per_round,
        eval_n,
        round_budget,
        eval_budget,
        round_budget, // Python passes train_num_simulations_mcts=round_budget
        seed,
    )?;

    let eval_problems: Vec<SelfplayProblem> = make_eval_set(eval_n, 778);
    let eval_problems_states: Vec<RlProblem> = eval_problems
        .iter()
        .map(|(domain, nums, max_depth)| {
            (domain.clone(), make_initial_state(nums), *max_depth)
        })
        .collect();

    let distill_round_problems = |round_idx: usize| -> Vec<RlProblem> {
        let mut rng = StdRng::seed_from_u64(500 + round_idx as u64);
        (0..n_train_problems_per_round)
            .map(|_| {
                let (domain, nums, max_depth) = make_guaranteed_solvable(&mut rng, 4, 6);
                (domain, make_initial_state(&nums), max_depth)
            })
            .collect()
    };

    let (_policy, history, _round_diags) = run_distill_training(
        rounds,
        &distill_round_problems,
        &eval_problems_states,
        round_budget, // num_simulations
        0.3,          // lr
        1.0,          // Python default temperature
        eval_budget,
        seed,
    )?;
    let succ = py_round(history.eval_solve_rate_after * eval_n as f64, 0) as i64 as usize;
    let (lo, hi) =
        wilson_score_interval(succ as i64, eval_n as i64, Z).expect("n > 0 by construction");

    let mut report = FiveWayReport {
        conditions: base.conditions.clone(),
        notes: base.notes.clone(),
    };
    report.conditions.push(ConditionResult {
        name: "distillation".to_string(),
        successes: succ,
        n: eval_n,
        solve_rate: succ as f64 / eval_n as f64,
        ci_lo: lo,
        ci_hi: hi,
    });
    report.notes.push(format!(
        "distillation round solve rates during training: {}",
        fmt_f64_list(&history.round_solve_rates)
    ));
    report.notes.push(format!(
        "distillation round mean loss (lower=better, not a reward): {}",
        fmt_f64_list_round3(&history.round_mean_rewards)
    ));
    report.notes.push(format!(
        "distillation before/after MCTS-guided eval: {:.2} -> {:.2}",
        history.eval_solve_rate_before, history.eval_solve_rate_after
    ));
    Ok(report)
}
