//! Port of python/tests/test_calibration_fix.py (TestCalibrationFix).
//!
//! The Python test imports `generate_curriculum` (curriculum crate) and
//! `eval_prm_calibration` (evaluation crate) — both are scheduled for a
//! later conversion wave, so faithful local stand-ins for exactly those two
//! functions are inlined below (level-1 arithmetic generator + calibration
//! evaluator, same logic and constants). They will be swapped for the real
//! crate calls when those crates land.

mod common;

use common::train_a_prm;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reasoning_common::py_round;
use reasoning_prm::{build_calibrated_training_set, extract_features, LinearPRM};
use reasoning_search::{make_initial_state, Mcts, NumberTargetDomain};

/// (numbers, target) — the payload of a curriculum level-1 Problem.
fn generate_curriculum_level1(num_per_level: usize, seed: u64) -> Vec<(Vec<f64>, f64)> {
    // Python gen_arithmetic: n in {2, 3}, nums 1..=9, target biased toward
    // reachable (sum, or abs-diff/first-number).
    let mut rng = StdRng::seed_from_u64(seed);
    let mut problems = Vec::new();
    while problems.len() < num_per_level {
        let n: usize = if rng.gen::<bool>() { 2 } else { 3 };
        let nums: Vec<i64> = (0..n).map(|_| rng.gen_range(1..=9)).collect();
        let target = if rng.gen::<f64>() < 0.5 {
            nums.iter().sum::<i64>()
        } else if n == 2 {
            (nums[0] - nums[1]).abs()
        } else {
            nums[0].abs()
        };
        problems.push((
            nums.iter().map(|&v| v as f64).collect(),
            target as f64,
        ));
    }
    problems
}

/// Python eval_prm_calibration -> mean_calibration_gap: bucket the
/// (predicted, solved) records by predicted probability and compare each
/// bucket's average prediction to its observed success rate.
fn eval_prm_calibration_mean_gap(
    prm: &LinearPRM,
    problems: &[(Vec<f64>, f64)],
    budget: usize,
    seed: u64,
) -> f64 {
    let mut records: Vec<(f64, i32)> = Vec::new();
    for (i, (nums, target)) in problems.iter().enumerate() {
        let domain = NumberTargetDomain::new(*target);
        let state = make_initial_state(nums);
        let feats = extract_features(&state, domain.target, 6, 0);
        let pred = prm.predict(&feats);
        let mut mcts = Mcts::new(domain, 6, seed + i as u64);
        let result = mcts.search(state, budget);
        records.push((pred, result.found_verified_solution as i32));
    }

    records.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
    let n_buckets = 5usize;
    let bucket_size = (records.len() / n_buckets).max(1);
    let mut gaps: Vec<f64> = Vec::new();
    let mut i = 0usize;
    while i < records.len() {
        let chunk = &records[i..(i + bucket_size).min(records.len())];
        i += bucket_size;
        if chunk.is_empty() {
            continue;
        }
        let avg_pred: f64 = chunk.iter().map(|(p, _)| *p).sum::<f64>() / chunk.len() as f64;
        let actual_rate: f64 = chunk.iter().map(|(_, a)| *a as f64).sum::<f64>() / chunk.len() as f64;
        // Python: gap = round(abs(avg_pred - actual_rate), 3)
        gaps.push(py_round((avg_pred - actual_rate).abs(), 3));
    }
    if gaps.is_empty() {
        0.0
    } else {
        gaps.iter().sum::<f64>() / gaps.len() as f64
    }
}

fn train_problems() -> Vec<(NumberTargetDomain, Vec<f64>, usize)> {
    vec![
        (NumberTargetDomain::new(4.0), vec![2.0, 2.0], 4usize),
        (NumberTargetDomain::new(24.0), vec![4.0, 7.0, 8.0, 8.0], 6),
        (NumberTargetDomain::new(10.0), vec![2.0, 3.0, 5.0], 4),
        (NumberTargetDomain::new(13.0), vec![1.0, 2.0, 6.0], 4),
        (NumberTargetDomain::new(1.0), vec![3.0, 3.0], 4),
        (NumberTargetDomain::new(100.0), vec![1.0, 1.0, 1.0, 1.0], 6),
    ]
}

#[test]
fn test_root_oversampling_improves_calibration_gap() {
    // baseline: original (Phase 004-style) training, uses tree-walk data only
    let uncalibrated = train_a_prm(0);

    // fixed: tree-walk data + oversampled root examples
    let problems = train_problems();
    let calibrated_examples = build_calibrated_training_set(&problems, 15, 800, 0);
    let mut calibrated = LinearPRM::new();
    let mut rng = StdRng::seed_from_u64(0);
    calibrated.train(&calibrated_examples, 60, 0.3, &mut rng);

    let eval_problems = generate_curriculum_level1(12, 42);

    let before = eval_prm_calibration_mean_gap(&uncalibrated, &eval_problems, 300, 100);
    let after = eval_prm_calibration_mean_gap(&calibrated, &eval_problems, 300, 100);

    println!("\n[Phase 011] calibration gap before fix: {:.3}", before);
    println!("[Phase 011] calibration gap after fix:  {:.3}", after);

    assert!(
        after < before,
        "root-oversampled training should reduce the calibration gap, not just accuracy"
    );
}
