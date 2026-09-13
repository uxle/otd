//! Port of python/tests/test_equation_prm.py (TestEquationPRM).

use rand::rngs::StdRng;
use rand::SeedableRng;
use reasoning_prm::{
    extract_equation_features, extract_equation_training_examples, FeatureMap, LinearPRM,
};
use reasoning_search::{Domain, LinearEquationDomain, Mcts};

fn build_dataset(equations: &[&str], budget: usize, seed: u64) -> Vec<(FeatureMap, f64)> {
    let mut examples = Vec::new();
    for (i, eq) in equations.iter().enumerate() {
        let domain = LinearEquationDomain::try_new(eq, 5).unwrap();
        let mut mcts = Mcts::new(domain.clone(), 5, seed + i as u64);
        let result = mcts.search(domain.initial_state(), budget);
        examples.extend(extract_equation_training_examples(&result, 5, 2));
    }
    examples
}

#[test]
fn test_prm_learns_on_algebra_domain() {
    let train_eqs = ["2*x + 4 = 10", "3*x - 9 = -18", "5*x + 1 = 16"];
    let test_eqs = ["4*x - 2 = 10", "2*x + 7 = 1"];

    let train_examples = build_dataset(&train_eqs, 600, 0);
    let test_examples = build_dataset(&test_eqs, 600, 1000);

    assert!(train_examples.len() > 5);
    assert!(test_examples.len() > 5);

    let mut prm = LinearPRM::new();
    let mut rng = StdRng::seed_from_u64(0);
    let history = prm.train(&train_examples, 60, 0.3, &mut rng);
    assert!(history[history.len() - 1] < history[0]);

    let mae = |examples: &[(FeatureMap, f64)]| -> f64 {
        examples
            .iter()
            .map(|(f, t)| (prm.predict(f) - t).abs())
            .sum::<f64>()
            / examples.len() as f64
    };

    let mean_target: f64 =
        train_examples.iter().map(|(_, t)| *t).sum::<f64>() / train_examples.len() as f64;
    let baseline_mae: f64 = test_examples
        .iter()
        .map(|(_, t)| (mean_target - t).abs())
        .sum::<f64>()
        / test_examples.len() as f64;
    let prm_mae = mae(&test_examples);

    println!(
        "\n[Phase 012] equation-domain PRM MAE: {:.4} vs constant-baseline MAE: {:.4}",
        prm_mae, baseline_mae
    );

    assert!(prm_mae < baseline_mae);
}

#[test]
fn test_solved_state_scores_higher_than_unsolved() {
    let train_examples = build_dataset(&["2*x + 4 = 10"], 600, 0);
    let mut prm = LinearPRM::new();
    let mut rng = StdRng::seed_from_u64(0);
    prm.train(&train_examples, 60, 0.3, &mut rng);

    let domain = LinearEquationDomain::try_new("2*x + 4 = 10", 5).unwrap();
    let solved_state = domain.apply(
        &domain.apply(&domain.initial_state(), &("sub".to_string(), 4.0)),
        &("div".to_string(), 2.0),
    );
    assert!(domain.is_terminal(&solved_state));

    let unsolved_feats = extract_equation_features(&domain.initial_state(), 5);
    let solved_feats = extract_equation_features(&solved_state, 5);

    assert!(prm.predict(&solved_feats) > prm.predict(&unsolved_feats));
}
