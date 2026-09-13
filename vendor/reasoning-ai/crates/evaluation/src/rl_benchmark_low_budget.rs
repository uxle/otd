//! Phase 083 — Low-search-budget re-test (design doc 24; direct follow-up to
//! the "what would make this a fair test" note at the end of RL_TRAINING.md
//! section 7) (Rust port of `python/evaluation/rl_benchmark_low_budget.py`).
//!
//! At a generous MCTS eval budget (250 simulations), search itself solves
//! most of the eval set regardless of rollout-policy quality, which makes
//! it hard for any trained policy to show an advantage. This module reruns
//! the same four-way comparison at a much smaller budget, where
//! rollout-policy quality should matter more (if it matters at all).
//!
//! Real result from the Python run (eval_budget=25, N=40): every condition
//! hit the floor (0/40) — the eval set is simply too hard for ANY policy to
//! solve with that little search, trained or not. That's a floor-effect
//! finding, not a "policy doesn't matter" finding, and it's reported as
//! such. At eval_budget=50, N=40: uniform 1/40, prm_guided 3/40,
//! grpo_raw_sampling 3/40, grpo_mcts_sourced 2/40 — all four intervals
//! overlap heavily, still no significant separation (near-floor counts too
//! small for N=40 to resolve).

use crate::rl_benchmark_v2::{run_four_way_comparison, FourWayReport};

/// Python `run_low_budget_comparison(eval_budget=50, rounds=3,
/// n_train_problems_per_round=8, eval_n=40, round_budget=100,
/// train_num_simulations_mcts=100, seed=1)`.
///
/// Thin, documented wrapper around Phase 75's harness with defaults tuned
/// for the low-budget regime — kept as its own entry point so the "which
/// budget was this?" question always has an unambiguous answer in code,
/// not just in a one-off script's arguments.
#[allow(clippy::too_many_arguments)]
pub fn run_low_budget_comparison(
    eval_budget: usize,
    rounds: usize,
    n_train_problems_per_round: usize,
    eval_n: usize,
    round_budget: usize,
    train_num_simulations_mcts: usize,
    seed: u64,
) -> Result<FourWayReport, String> {
    run_four_way_comparison(
        rounds,
        n_train_problems_per_round,
        eval_n,
        round_budget,
        eval_budget,
        train_num_simulations_mcts,
        seed,
    )
}
