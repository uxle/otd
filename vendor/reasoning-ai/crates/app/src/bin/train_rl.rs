//! Phase 079 — Training CLI (design doc 26: production entry points)
//! (Rust port of `python/apps/train_rl.py`).
//!
//! Wires Phases 71-78 into one runnable command: curriculum-scheduled,
//! MCTS-sourced GRPO training, with adaptive exploration and checkpointing
//! after every round.
//!
//! Usage:
//!     train_rl [--rounds 5] [--out /tmp/rl_run]
//!
//! Deliberately small a CLI (no config framework) — this project's
//! existing apps are similarly plain, and a training run here takes
//! seconds to minutes, not the hours that would justify a heavier config
//! system.
//!
//! Deviation note: `rl::make_problems_at_level` lives in the rl crate's
//! `curriculum_grpo` module, which is still a guarded stub there (the rl
//! crate doesn't depend on reasoning-curriculum yet — see
//! crates/rl/src/curriculum_grpo.rs). The 10-line helper is implemented
//! locally below, ported line-for-line from python/rl/curriculum_grpo.py,
//! rather than modifying the rl crate from this task.

use std::path::PathBuf;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use reasoning_curriculum::CurriculumController;
use reasoning_rl::{
    save_policy_checkpoint, save_training_history, AdaptiveExplorationController,
    MCTSRoundDiagnostics, SoftmaxPolicy, TrainingHistory, _depth_fn,
    evaluate_policy, policy_entropy_on_probe_states, run_grpo_round_mcts,
};
use reasoning_search::{make_initial_state, Domain, NumberTargetDomain, NtState};

/// rl::Problem = (domain, initial state, max_depth).
type Problem = (NumberTargetDomain, NtState, usize);

// python/rl/curriculum_grpo.py — difficulty = number count / max_depth.
fn level_to_num_count(level: i64) -> usize {
    match level {
        1 => 2,
        2 => 3,
        3 => 4,
        _ => 5, // unknown level falls back to the max-key value
    }
}

fn level_to_max_depth(level: i64) -> usize {
    match level {
        1 => 2,
        2 => 4,
        3 => 6,
        _ => 8,
    }
}

fn make_problems_at_level(level: i64, n_problems: usize, rng: &mut StdRng) -> Vec<Problem> {
    let n_count = level_to_num_count(level);
    let max_depth = level_to_max_depth(level);
    let mut problems = Vec::with_capacity(n_problems);
    for _ in 0..n_problems {
        let nums: Vec<f64> = (0..n_count).map(|_| rng.gen_range(1..=9) as f64).collect();
        // target reachable by summing a random non-empty subset —
        // guarantees solvability
        let k = rng.gen_range(1..=n_count);
        let mut idx: Vec<usize> = (0..n_count).collect();
        idx.shuffle(rng);
        let target: f64 = idx.iter().take(k).map(|&i| nums[i]).sum();
        let domain = NumberTargetDomain::new(target);
        problems.push((domain, make_initial_state(&nums), max_depth));
    }
    problems
}

struct Args {
    rounds: i64,
    problems_per_round: usize,
    samples_per_problem: usize,
    train_simulations: usize,
    eval_simulations: usize,
    eval_n: usize,
    lr: f64,
    seed: u64,
    out: String,
}

impl Default for Args {
    fn default() -> Self {
        Args {
            rounds: 5,
            problems_per_round: 8,
            samples_per_problem: 6,
            train_simulations: 150,
            eval_simulations: 60,
            eval_n: 10,
            lr: 0.2,
            seed: 0,
            out: "/tmp/rl_train_run".to_string(),
        }
    }
}

fn parse_args() -> Args {
    let mut args = Args::default();
    let argv: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < argv.len() {
        let a = argv[i].clone();
        if !a.starts_with("--") {
            i += 1;
            continue;
        }
        let (name, inline_val) = match a.split_once('=') {
            Some((n, v)) => (n.to_string(), Some(v.to_string())),
            None => (a.clone(), None),
        };
        let mut value = inline_val;
        if value.is_none() && i + 1 < argv.len() && !argv[i + 1].starts_with("--") {
            value = Some(argv[i + 1].clone());
            i += 1;
        }
        if let Some(v) = value {
            match name.as_str() {
                "--rounds" => args.rounds = v.parse().unwrap_or(args.rounds),
                "--problems-per-round" => {
                    args.problems_per_round = v.parse().unwrap_or(args.problems_per_round)
                }
                "--samples-per-problem" => {
                    args.samples_per_problem = v.parse().unwrap_or(args.samples_per_problem)
                }
                "--train-simulations" => {
                    args.train_simulations = v.parse().unwrap_or(args.train_simulations)
                }
                "--eval-simulations" => {
                    args.eval_simulations = v.parse().unwrap_or(args.eval_simulations)
                }
                "--eval-n" => args.eval_n = v.parse().unwrap_or(args.eval_n),
                "--lr" => args.lr = v.parse().unwrap_or(args.lr),
                "--seed" => args.seed = v.parse().unwrap_or(args.seed),
                "--out" => args.out = v.clone(),
                _ => {}
            }
        }
        i += 1;
    }
    args
}

fn main() {
    let args = parse_args();
    let out_dir = PathBuf::from(&args.out);
    if let Err(e) = std::fs::create_dir_all(&out_dir) {
        eprintln!("[train_rl] cannot create {}: {}", args.out, e);
        std::process::exit(1);
    }

    let mut controller = CurriculumController::new(1, 4, 0.7, 0.15, 2);
    let mut exploration = AdaptiveExplorationController::default();
    let mut policy = SoftmaxPolicy::new();

    let mut eval_rng = StdRng::seed_from_u64(9999);
    let eval_problems = make_problems_at_level(1, args.eval_n, &mut eval_rng);
    let mut history = TrainingHistory::default();

    println!(
        "[train_rl] evaluating initial (untrained) policy on {} problems...",
        args.eval_n
    );
    history.eval_solve_rate_before = evaluate_policy(&policy, &eval_problems, args.eval_simulations, 8000);
    println!("[train_rl] before: solve_rate={:.2}", history.eval_solve_rate_before);

    for round_idx in 0..args.rounds {
        let mut rng = StdRng::seed_from_u64(args.seed + round_idx as u64 * 97);
        let problems = make_problems_at_level(controller.level, args.problems_per_round, &mut rng);
        let (new_policy, diag): (SoftmaxPolicy, MCTSRoundDiagnostics) =
            match run_grpo_round_mcts(
                &policy,
                &problems,
                &|md: usize| _depth_fn(md),
                args.samples_per_problem,
                args.train_simulations,
                args.lr,
                0.2, // Python default epsilon
                args.seed + round_idx as u64 * 137,
            ) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[train_rl] round {} failed: {}", round_idx, e);
                    break;
                }
            };
        policy = new_policy;
        let curriculum_action = controller.record(diag.solve_rate);

        let probe_states: Vec<(NtState, Vec<reasoning_search::NtAction>)> = problems
            .iter()
            .take(3)
            .map(|(d, s, _)| (s.clone(), d.legal_actions(s)))
            .collect();
        let entropy = policy_entropy_on_probe_states(
            &policy,
            &probe_states,
            &problems[0].0,
            problems[0].0.target,
            problems[0].2,
            &|state: &NtState| _depth_fn(problems[0].2)(state),
        )
        .unwrap_or(0.0);
        let epsilon = exploration.record_and_get_epsilon(entropy, diag.grad_norm);

        history.round_solve_rates.push(diag.solve_rate);
        history.round_mean_rewards.push(diag.mean_reward);
        history.round_grad_norms.push(diag.grad_norm);

        let ckpt_path = out_dir.join(format!("policy_round_{:03}.json", round_idx));
        let metadata = serde_json::json!({
            "round": round_idx,
            "level": controller.level,
            "curriculum_action": curriculum_action,
            "solve_rate": diag.solve_rate,
            "entropy": entropy,
            "next_epsilon": epsilon,
        });
        if let Err(e) = save_policy_checkpoint(&policy, ckpt_path.to_str().unwrap(), Some(metadata)) {
            eprintln!("[train_rl] checkpoint save failed: {}", e);
        }
        println!(
            "[train_rl] round {}: level={} action={} solve_rate={:.2} entropy={:.2} \
             next_epsilon={:.2} -> {}",
            round_idx,
            controller.level,
            curriculum_action,
            diag.solve_rate,
            entropy,
            epsilon,
            ckpt_path.display()
        );
    }

    println!("[train_rl] evaluating final policy on {} problems...", args.eval_n);
    history.eval_solve_rate_after = evaluate_policy(&policy, &eval_problems, args.eval_simulations, 8000);
    println!("[train_rl] after: solve_rate={:.2}", history.eval_solve_rate_after);

    let final_path = out_dir.join("policy_final.json");
    if let Err(e) = save_policy_checkpoint(&policy, final_path.to_str().unwrap(), None) {
        eprintln!("[train_rl] final checkpoint save failed: {}", e);
    }
    let history_path = out_dir.join("history.json");
    if let Err(e) = save_training_history(&history, history_path.to_str().unwrap()) {
        eprintln!("[train_rl] history save failed: {}", e);
    }
    println!("[train_rl] wrote {} and {}", final_path.display(), history_path.display());
}
