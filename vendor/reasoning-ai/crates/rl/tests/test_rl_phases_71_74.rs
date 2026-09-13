//! Rust port of python/tests/test_rl_phases_71_74.py:
//!   071 — MCTS-sourced episode generation
//!   073 — GRPO training loop with MCTS episodes
//!   074 — statistical significance utilities
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use reasoning_search::{make_initial_state, Domain, NumberTargetDomain};

use reasoning_rl::grpo_training_loop_mcts::{run_grpo_round_mcts, run_grpo_training_mcts};
use reasoning_rl::mcts_episode_source::sample_episode_via_mcts;
use reasoning_rl::policy::SoftmaxPolicy;
use reasoning_rl::stats::{
    proportions_overlap, required_n_for_margin, two_proportion_z_test,
    wilson_score_interval,
};
use reasoning_rl::trainable_rollout::sample_episode;
use reasoning_rl::{Problem, _depth_fn};

fn _mk_problem(numbers: &[f64], target: f64) -> Problem {
    let max_depth = 6;
    (
        NumberTargetDomain::new(target),
        make_initial_state(numbers),
        max_depth,
    )
}

// ── Phase 071 ───────────────────────────────────────────────────────────
#[test]
fn test_mcts_episode_source_returns_well_formed_episode() {
    let policy = SoftmaxPolicy::new();
    let domain = NumberTargetDomain::new(10.0);
    let state = make_initial_state(&[2.0, 3.0, 5.0]);
    let depth_fn = _depth_fn(4);
    let mut rng = StdRng::seed_from_u64(1);
    let (ep, reward, passed) = sample_episode_via_mcts(
        &policy,
        &domain,
        &state,
        10.0,
        4,
        depth_fn,
        &mut rng,
        100,
        0.1,
    );
    assert_eq!(ep.states.len(), ep.actions.len());
    assert_eq!(ep.actions.len(), ep.log_probs.len());
    assert!(ep.log_probs.iter().all(|lp| lp.is_finite()));
    // passed is a bool by construction in Rust; keep the type assertion spirit
    let _is_bool: bool = passed;
    assert!(-3.0 <= reward && reward <= 3.0);
}

#[test]
fn test_mcts_episode_source_solve_rate_beats_raw_sampling_on_easy_problem() {
    // The whole point of Phase 71: on a problem raw policy sampling rarely
    // solves, MCTS-assisted sampling should solve noticeably more often.
    let policy = SoftmaxPolicy::new(); // untrained/uniform, worst case for raw sampling
    let domain = NumberTargetDomain::new(24.0);
    let state = make_initial_state(&[4.0, 6.0, 8.0, 3.0]);
    let depth_fn = _depth_fn(6);

    let mut raw_solved = 0usize;
    let mut mcts_solved = 0usize;
    let trials = 15usize;
    for i in 0..trials {
        let mut rng = StdRng::seed_from_u64(i as u64);
        let ep = sample_episode(
            &policy,
            &domain,
            &state,
            24.0,
            6,
            depth_fn.as_ref(),
            &mut rng,
        );
        let mut final_state = state.clone();
        for a in &ep.actions {
            final_state = domain.apply(&final_state, a);
        }
        raw_solved += (domain.is_terminal(&final_state)
            && domain.terminal_reward(&final_state) >= 0.999) as usize;

        let mut rng = StdRng::seed_from_u64(i as u64);
        let (_ep, _reward, passed) = sample_episode_via_mcts(
            &policy,
            &domain,
            &state,
            24.0,
            6,
            depth_fn.clone(),
            &mut rng,
            120,
            0.1,
        );
        mcts_solved += passed as usize;
    }

    assert!(mcts_solved >= raw_solved);
}

#[test]
fn test_mcts_episode_source_empty_children_returns_empty_episode_not_crash() {
    // A trivially-terminal single-number "problem" should not crash the
    // extractor even though the tree may have no expandable children.
    let policy = SoftmaxPolicy::new();
    let domain = NumberTargetDomain::new(5.0);
    let state = make_initial_state(&[5.0]);
    let depth_fn = _depth_fn(1);
    let mut rng = StdRng::seed_from_u64(0);
    let (ep, _reward, passed) = sample_episode_via_mcts(
        &policy,
        &domain,
        &state,
        5.0,
        1,
        depth_fn,
        &mut rng,
        20,
        0.1,
    );
    assert_eq!(ep.states.len(), 0);
    assert!(passed); // [5] already equals target 5 -> terminal+solved trivially
}

// ── Phase 073 ───────────────────────────────────────────────────────────
#[test]
fn test_grpo_training_loop_mcts_one_mcts_round_runs_and_produces_diagnostics() {
    let policy = SoftmaxPolicy::new();
    let problems = vec![
        _mk_problem(&[2.0, 3.0, 5.0], 10.0),
        _mk_problem(&[1.0, 4.0, 6.0], 9.0),
    ];
    let (_new_policy, diag) = run_grpo_round_mcts(
        &policy,
        &problems,
        &_depth_fn,
        4,
        100,
        0.2,
        0.2,
        1,
    )
    .unwrap();
    assert!(0.0 <= diag.solve_rate && diag.solve_rate <= 1.0);
    assert!(diag.grad_norm.is_finite());
    assert!(
        0.0 <= diag.fraction_groups_with_nonzero_variance
            && diag.fraction_groups_with_nonzero_variance <= 1.0
    );
}

#[test]
fn test_grpo_training_loop_mcts_short_training_run_produces_history() {
    let round_problems = |round_idx: usize| -> Vec<Problem> {
        let mut rng = StdRng::seed_from_u64(700 + round_idx as u64);
        let mut out = Vec::new();
        for _ in 0..4 {
            let nums: Vec<f64> = (0..3).map(|_| rng.gen_range(1..=9) as f64).collect();
            let target = nums[0] + nums[1] + nums[2];
            out.push(_mk_problem(&nums, target));
        }
        out
    };

    let eval_problems = vec![
        _mk_problem(&[2.0, 3.0, 5.0], 10.0),
        _mk_problem(&[1.0, 1.0, 8.0], 10.0),
    ];
    let (_policy, history, round_diags) = run_grpo_training_mcts(
        2,
        &round_problems,
        &eval_problems,
        4,
        80,
        0.2,
        0.2,
        30,
        11,
    )
    .unwrap();
    assert_eq!(history.round_solve_rates.len(), 2);
    assert_eq!(round_diags.len(), 2);
    assert!(0.0 <= history.eval_solve_rate_after && history.eval_solve_rate_after <= 1.0);
}

// ── Phase 074 ───────────────────────────────────────────────────────────
#[test]
fn test_stats_wilson_interval_contains_point_estimate() {
    let (lo, hi) = wilson_score_interval(5, 20, 1.96).unwrap();
    assert!(lo <= 5.0 / 20.0);
    assert!(hi >= 5.0 / 20.0);
}

#[test]
fn test_stats_wilson_interval_narrows_with_more_data() {
    let (lo1, hi1) = wilson_score_interval(50, 100, 1.96).unwrap();
    let (lo2, hi2) = wilson_score_interval(500, 1000, 1.96).unwrap();
    assert!(hi2 - lo2 < hi1 - lo1);
}

#[test]
fn test_stats_wilson_interval_bounds_valid() {
    for (successes, n) in [(0, 20), (20, 20), (10, 20)] {
        let (lo, hi) = wilson_score_interval(successes, n, 1.96).unwrap();
        assert!(lo >= 0.0);
        assert!(hi <= 1.0);
        assert!(lo <= hi);
    }
}

#[test]
fn test_stats_invalid_n_raises() {
    assert!(wilson_score_interval(1, 0, 1.96).is_err());
    assert!(wilson_score_interval(5, 3, 1.96).is_err());
}

#[test]
fn test_stats_identical_proportions_overlap() {
    assert!(proportions_overlap(5, 20, 5, 20, 1.96).unwrap());
}

#[test]
fn test_stats_far_apart_proportions_do_not_overlap() {
    assert!(!proportions_overlap(0, 100, 90, 100, 1.96).unwrap());
}

#[test]
fn test_stats_phase_69_style_numbers_do_overlap() {
    // uniform=7/20, prm=5/20, grpo=9/20 (0.35/0.25/0.45 from the real run)
    assert!(proportions_overlap(7, 20, 9, 20, 1.96).unwrap());
    assert!(proportions_overlap(5, 20, 9, 20, 1.96).unwrap());
}

#[test]
fn test_stats_z_test_identical_proportions_gives_p_near_one() {
    let (z, p) = two_proportion_z_test(10, 20, 10, 20).unwrap();
    assert!(z.abs() < 1e-8);
    assert!((p - 1.0).abs() < 1e-6);
}

#[test]
fn test_stats_z_test_extreme_difference_gives_small_p() {
    let (_z, p) = two_proportion_z_test(2, 100, 90, 100).unwrap();
    assert!(p < 0.001);
}

#[test]
fn test_stats_required_n_for_margin_decreases_with_larger_margin() {
    let n_tight = required_n_for_margin(0.3, 0.02, 1.96).unwrap();
    let n_loose = required_n_for_margin(0.3, 0.10, 1.96).unwrap();
    assert!(n_tight > n_loose);
}
