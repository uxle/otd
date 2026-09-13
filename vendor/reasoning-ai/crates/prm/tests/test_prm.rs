//! Port of python/tests/test_prm.py (TestPRMLearnsFromVerifiedData).

use rand::rngs::StdRng;
use rand::SeedableRng;
use reasoning_prm::{extract_features, extract_training_examples, FeatureMap, LinearPRM};
use reasoning_search::{make_initial_state, Mcts, NumberTargetDomain};

fn mean_abs_error(prm: &LinearPRM, examples: &[(FeatureMap, f64)]) -> f64 {
    let total: f64 = examples
        .iter()
        .map(|(feats, target)| (prm.predict(feats) - target).abs())
        .sum();
    total / examples.len().max(1) as f64
}

fn generate_dataset(seed_offset: u64) -> Vec<(FeatureMap, f64)> {
    // Run MCTS on a mix of easy/hard/impossible instances and pool the
    // resulting (features, Q-value) pairs. Mirrors the self-evolution data
    // pipeline in design doc 2.9, minus the retrain-the-policy step.
    let problems = vec![
        (NumberTargetDomain::new(4.0), vec![2.0, 2.0], 4usize),
        (NumberTargetDomain::new(24.0), vec![4.0, 7.0, 8.0, 8.0], 6),
        (NumberTargetDomain::new(10.0), vec![2.0, 3.0, 5.0], 4),
        (NumberTargetDomain::new(1.0), vec![3.0, 3.0], 4),
        (NumberTargetDomain::new(100.0), vec![1.0, 1.0, 1.0, 1.0], 6), // unsolvable
        (NumberTargetDomain::new(13.0), vec![1.0, 2.0, 6.0], 4),
    ];
    let mut all_examples = Vec::new();
    for (i, (domain, nums, max_depth)) in problems.iter().enumerate() {
        let state = make_initial_state(nums);
        let mut mcts = Mcts::new(domain.clone(), *max_depth, 100 + i as u64 + seed_offset);
        let result = mcts.search(state, 800);
        let examples =
            extract_training_examples(&result, domain.target, *max_depth, 2);
        all_examples.extend(examples);
    }
    all_examples
}

#[test]
fn test_prm_beats_constant_baseline() {
    let train_examples = generate_dataset(0);
    let test_examples = generate_dataset(1000); // different rng -> different tree
    assert!(
        train_examples.len() > 20,
        "need enough data to train on"
    );
    assert!(
        test_examples.len() > 20,
        "need enough data to evaluate on"
    );

    let mut prm = LinearPRM::new();
    let mut rng = StdRng::seed_from_u64(0);
    let history = prm.train(&train_examples, 60, 0.3, &mut rng);

    // loss should have gone down over training, not be flat/diverging
    assert!(history[history.len() - 1] < history[0]);

    let prm_mae = mean_abs_error(&prm, &test_examples);

    // trivial baseline: always predict the mean training Q-value
    let mean_target: f64 =
        train_examples.iter().map(|(_, t)| *t).sum::<f64>() / train_examples.len() as f64;
    let baseline_mae: f64 = test_examples
        .iter()
        .map(|(_, t)| (mean_target - t).abs())
        .sum::<f64>()
        / test_examples.len() as f64;

    assert!(
        prm_mae < baseline_mae,
        "PRM (MAE={:.4}) failed to beat constant baseline (MAE={:.4})",
        prm_mae,
        baseline_mae
    );
}

#[test]
fn test_prm_ranks_exact_match_state_higher() {
    // sanity: a state that already exactly equals the target should score
    // higher than one far from it, even without training, once trained.
    let train_examples = generate_dataset(0);
    let mut prm = LinearPRM::new();
    let mut rng = StdRng::seed_from_u64(0);
    prm.train(&train_examples, 60, 0.3, &mut rng);

    let exact_feats = extract_features(&vec![(4.0, "4".to_string())], 4.0, 4, 3);
    let far_feats = extract_features(&vec![(999.0, "999".to_string())], 4.0, 4, 3);

    assert!(prm.predict(&exact_feats) > prm.predict(&far_feats));
}
