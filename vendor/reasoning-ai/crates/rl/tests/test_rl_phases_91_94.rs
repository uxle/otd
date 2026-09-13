//! Rust port of python/tests/test_rl_phases_91_94.py:
//!   091 — distillation math
//!   092 — distillation weight update
//!   093 — distillation self-play round
//!   094 — distillation training loop
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use reasoning_search::{make_initial_state, Domain, NumberTargetDomain};

use reasoning_rl::distill_self_play::{extract_distill_samples, run_distill_round};
use reasoning_rl::distill_training_loop::run_distill_training;
use reasoning_rl::distill_update::{distill_weight_update, DistillSample};
use reasoning_rl::distillation::{
    distillation_grad_log_probs, soft_target_cross_entropy, softmax,
    visit_counts_to_teacher_distribution,
};
use reasoning_rl::entropy_reg::entropy_from_logprobs;
use reasoning_rl::policy::SoftmaxPolicy;
use reasoning_rl::{Problem, _depth_fn};

fn _mk_problem(numbers: &[f64], target: f64) -> Problem {
    let max_depth = 6;
    (
        NumberTargetDomain::new(target),
        make_initial_state(numbers),
        max_depth,
    )
}

// ── Phase 091 ───────────────────────────────────────────────────────────
#[test]
fn test_distillation_softmax_sums_to_one() {
    let probs = softmax(&[1.0, 2.0, 3.0], 1.0).unwrap();
    assert!((probs.iter().sum::<f64>() - 1.0).abs() < 1e-10);
}

#[test]
fn test_distillation_softmax_temperature_one_matches_standard() {
    let probs_t1 = softmax(&[1.0, 2.0], 1.0).unwrap();
    let expected = 2.0f64.exp() / (1.0f64.exp() + 2.0f64.exp());
    assert!((probs_t1[1] - expected).abs() < 1e-8);
}

#[test]
fn test_distillation_high_temperature_flattens_distribution() {
    let probs_low_t = softmax(&[1.0, 5.0], 0.1).unwrap();
    let probs_high_t = softmax(&[1.0, 5.0], 100.0).unwrap();
    // higher temperature -> closer to uniform
    assert!((probs_high_t[0] - 0.5).abs() < (probs_low_t[0] - 0.5).abs());
}

#[test]
fn test_distillation_visit_counts_proportional_at_temperature_one() {
    let dist = visit_counts_to_teacher_distribution(&[10, 30], 1.0).unwrap();
    assert!((dist[0] - 0.25).abs() < 1e-8);
    assert!((dist[1] - 0.75).abs() < 1e-8);
}

#[test]
fn test_distillation_visit_counts_all_zero_falls_back_uniform() {
    let dist = visit_counts_to_teacher_distribution(&[0, 0, 0], 1.0).unwrap();
    for p in &dist {
        assert!((p - 1.0 / 3.0).abs() < 1e-8);
    }
}

#[test]
fn test_distillation_visit_counts_negative_raises() {
    assert!(visit_counts_to_teacher_distribution(&[-1, 5], 1.0).is_err());
}

#[test]
fn test_distillation_cross_entropy_zero_when_distributions_match() {
    // H(p,p) is the minimum possible cross entropy for that teacher, i.e.
    // equal to the teacher's own entropy.
    let p = [0.3, 0.7];
    let ce = soft_target_cross_entropy(&p, &p).unwrap();
    let log_p: Vec<f64> = p.iter().map(|x| x.ln()).collect();
    let ent = entropy_from_logprobs(&log_p).unwrap();
    assert!((ce - ent).abs() < 1e-8);
}

#[test]
fn test_distillation_cross_entropy_higher_for_mismatched_distributions() {
    let teacher = [0.9, 0.1];
    let matched = soft_target_cross_entropy(&teacher, &teacher).unwrap();
    let mismatched = soft_target_cross_entropy(&teacher, &[0.1, 0.9]).unwrap();
    assert!(mismatched > matched);
}

#[test]
fn test_distillation_gradient_matches_finite_difference() {
    // Verify d(L)/d(logit) = p_student - p_teacher via finite differences
    // on the actual loss, per house convention.
    let teacher = [0.2, 0.5, 0.3];
    let logits = [0.5, -0.3, 1.1];
    let h = 1e-5;
    let p_student = softmax(&logits, 1.0).unwrap();
    let analytic = distillation_grad_log_probs(&teacher, &p_student).unwrap();

    for i in 0..logits.len() {
        let mut plus = logits.to_vec();
        plus[i] += h;
        let mut minus = logits.to_vec();
        minus[i] -= h;
        let loss_plus = soft_target_cross_entropy(&teacher, &softmax(&plus, 1.0).unwrap()).unwrap();
        let loss_minus = soft_target_cross_entropy(&teacher, &softmax(&minus, 1.0).unwrap()).unwrap();
        let numeric = (loss_plus - loss_minus) / (2.0 * h);
        assert!((analytic[i] - numeric).abs() < 1e-4);
    }
}

// ── Phase 092 ───────────────────────────────────────────────────────────
struct DistillSetup {
    domain: NumberTargetDomain,
    state: reasoning_search::NtState,
    legal: Vec<reasoning_search::NtAction>,
    max_depth: usize,
    depth_fn: std::rc::Rc<dyn Fn(&reasoning_search::NtState) -> usize>,
}

impl DistillSetup {
    fn new() -> Self {
        let domain = NumberTargetDomain::new(10.0);
        let state = make_initial_state(&[2.0, 3.0, 5.0]);
        let legal = domain.legal_actions(&state);
        let max_depth = 4;
        DistillSetup {
            domain,
            state,
            legal,
            max_depth,
            depth_fn: _depth_fn(max_depth),
        }
    }
}

#[test]
fn test_distill_weight_update_update_reduces_loss_on_same_batch() {
    let s = DistillSetup::new();
    let policy = SoftmaxPolicy::new();
    let n = s.legal.len();
    // avoid exact 0/1
    let teacher_dist: Vec<f64> = (0..n)
        .map(|i| if i == 0 { 0.9 } else { 0.1 / (n - 1) as f64 })
        .collect();
    let sample = DistillSample {
        state: s.state.clone(),
        legal: s.legal.clone(),
        teacher_distribution: teacher_dist,
    };

    let (_before_policy, diag_before) = distill_weight_update(
        &policy,
        &[sample.clone()],
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
        0.5,
    )
    .unwrap();
    let loss_before = diag_before.mean_loss_before_update;

    let (updated_policy, _) = distill_weight_update(
        &policy,
        &[sample.clone()],
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
        0.5,
    )
    .unwrap();
    let (_, diag_after) = distill_weight_update(
        &updated_policy,
        &[sample],
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
        0.0,
    )
    .unwrap();
    let loss_after = diag_after.mean_loss_before_update;
    assert!(loss_after < loss_before);
}

#[test]
fn test_distill_weight_update_mismatched_teacher_length_raises() {
    let s = DistillSetup::new();
    let policy = SoftmaxPolicy::new();
    let sample = DistillSample {
        state: s.state.clone(),
        legal: s.legal.clone(),
        teacher_distribution: vec![1.0], // wrong length
    };
    assert!(distill_weight_update(
        &policy,
        &[sample],
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
        0.3
    )
    .is_err());
}

#[test]
fn test_distill_weight_update_empty_samples_raises() {
    let s = DistillSetup::new();
    let policy = SoftmaxPolicy::new();
    assert!(distill_weight_update(
        &policy,
        &[],
        &s.domain,
        10.0,
        s.max_depth,
        s.depth_fn.as_ref(),
        0.3
    )
    .is_err());
}

// ── Phase 093 ───────────────────────────────────────────────────────────
#[test]
fn test_distill_self_play_extract_distill_samples_well_formed() {
    let policy = SoftmaxPolicy::new();
    let domain = NumberTargetDomain::new(10.0);
    let state = make_initial_state(&[2.0, 3.0, 5.0]);
    let depth_fn = _depth_fn(4);
    let mut rng = StdRng::seed_from_u64(0);
    let (samples, _passed) = extract_distill_samples(
        &policy,
        &domain,
        &state,
        10.0,
        4,
        depth_fn,
        &mut rng,
        100,
        0.1,
        1.0,
    )
    .unwrap();
    for s in &samples {
        assert!((s.teacher_distribution.iter().sum::<f64>() - 1.0).abs() < 1e-6);
        assert_eq!(s.teacher_distribution.len(), s.legal.len());
    }
}

#[test]
fn test_distill_self_play_run_distill_round_produces_diagnostics() {
    let policy = SoftmaxPolicy::new();
    let problems = vec![
        _mk_problem(&[2.0, 3.0, 5.0], 10.0),
        _mk_problem(&[1.0, 4.0, 6.0], 9.0),
    ];
    let (_new_policy, diag) =
        run_distill_round(&policy, &problems, &_depth_fn, 100, 0.3, 1.0, 1).unwrap();
    assert!(0.0 <= diag.solve_rate && diag.solve_rate <= 1.0);
    assert!(diag.mean_loss_before_update.is_finite());
    assert!(diag.n_samples > 0);
}

#[test]
fn test_distill_self_play_weights_change_after_a_round() {
    let policy = SoftmaxPolicy::new();
    let problems = vec![
        _mk_problem(&[2.0, 3.0, 5.0], 10.0),
        _mk_problem(&[1.0, 4.0, 6.0], 9.0),
        _mk_problem(&[2.0, 2.0, 8.0], 12.0),
    ];
    let (new_policy, _diag) =
        run_distill_round(&policy, &problems, &_depth_fn, 100, 0.3, 1.0, 2).unwrap();
    assert_ne!(new_policy.weights, policy.weights);
}

// ── Phase 094 ───────────────────────────────────────────────────────────
#[test]
fn test_distill_training_loop_short_training_run_produces_history() {
    let round_problems = |round_idx: usize| -> Vec<Problem> {
        let mut rng = StdRng::seed_from_u64(900 + round_idx as u64);
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
    let (_policy, history, round_diags) = run_distill_training(
        2,
        &round_problems,
        &eval_problems,
        80,
        0.3,
        1.0,
        30,
        21,
    )
    .unwrap();
    assert_eq!(history.round_solve_rates.len(), 2);
    assert_eq!(round_diags.len(), 2);
    assert!(0.0 <= history.eval_solve_rate_after && history.eval_solve_rate_after <= 1.0);
}
