//! Phase 094 — Multi-round distillation training loop. Same shape as
//! Phase 73's `run_grpo_training_mcts` (same eval procedure, same
//! TrainingHistory type) so Phase 095 can compare them on genuinely
//! equal footing — number of rounds, problems per round, and simulation
//! budget are the only things that should differ between a GRPO run and
//! a distillation run being compared.

use crate::distill_self_play::{run_distill_round, DistillRoundDiagnostics};
use crate::grpo_training_loop::{_depth_fn, evaluate_policy, TrainingHistory};
use crate::policy::SoftmaxPolicy;

use crate::Problem;

/// Python defaults: num_simulations=150, lr=0.3, temperature=1.0,
/// eval_num_simulations=60, seed=0.
#[allow(clippy::too_many_arguments)]
pub fn run_distill_training(
    rounds: usize,
    round_problem_fn: &dyn Fn(usize) -> Vec<Problem>,
    eval_problems: &[Problem],
    num_simulations: usize,
    lr: f64,
    temperature: f64,
    eval_num_simulations: usize,
    seed: u64,
) -> Result<(SoftmaxPolicy, TrainingHistory, Vec<DistillRoundDiagnostics>), String> {
    let mut policy = SoftmaxPolicy::new();
    let mut history = TrainingHistory::default();
    history.eval_solve_rate_before =
        evaluate_policy(&policy, eval_problems, eval_num_simulations, 9000);

    let mut round_diags: Vec<DistillRoundDiagnostics> = Vec::new();
    for round_idx in 0..rounds {
        let problems = round_problem_fn(round_idx);
        let (new_policy, diag) = run_distill_round(
            &policy,
            &problems,
            &_depth_fn,
            num_simulations,
            lr,
            temperature,
            seed + round_idx as u64 * 137,
        )?;
        policy = new_policy;
        history.round_solve_rates.push(diag.solve_rate);
        // NOTE: round_mean_rewards holds mean_loss_before_update for
        // distillation runs — a smaller number means something different
        // (better) than for GRPO. Kept for TrainingHistory shape parity
        // with Phase 067 (see the Python module's field note).
        history.round_mean_rewards.push(diag.mean_loss_before_update);
        history.round_grad_norms.push(diag.grad_norm);
        round_diags.push(diag);
    }

    history.eval_solve_rate_after =
        evaluate_policy(&policy, eval_problems, eval_num_simulations, 9000);
    Ok((policy, history, round_diags))
}
