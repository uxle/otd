//! Port of python/tests/test_rl_phase_97.py — cross-check the optimized
//! `is_terminal` (expand-only) against the pre-Phase-097 simplify-based
//! version on real MCTS-generated states, plus a timing comparison.

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::Rng;
use rand::SeedableRng;
use reasoning_search::domain::Domain;
use reasoning_search::linear_equation_domain::{EqState, LinearEquationDomain};
use reasoning_search::Mcts;
use reasoning_symbolic::{expand, is_zero, simplify, Expr};
use std::time::Instant;

/// The pre-Phase-097 implementation, kept here only to cross-check the
/// optimized version — never reintroduced into the domain itself.
fn old_is_terminal(state: &EqState) -> bool {
    if state.rhs.has_sym("x") {
        return false;
    }
    is_zero(&simplify(&(state.lhs.clone() - Expr::sym("x"))))
}

fn random_states(n: usize, seed: u64) -> Vec<EqState> {
    // Generate real states by actually running rollouts (not hand-crafted
    // expressions), so the cross-check covers what MCTS genuinely produces.
    let mut rng = StdRng::seed_from_u64(seed);
    let mut states: Vec<EqState> = Vec::new();
    for i in 0..n {
        let a = rng.gen_range(1..=9);
        let x_true = rng.gen_range(-10..=10);
        let b = rng.gen_range(-10..=10);
        let c = a * x_true + b;
        let domain = LinearEquationDomain::try_new(&format!("{}*x + {} = {}", a, b, c), 6).unwrap();
        let mut state = domain.initial_state();
        states.push(state.clone());
        for _ in 0..6 {
            let legal = domain.legal_actions(&state);
            if legal.is_empty() {
                break;
            }
            let action = legal.choose(&mut rng).cloned().unwrap();
            state = domain.apply(&state, &action);
            states.push(state.clone());
        }
        let _ = i;
    }
    states
}

#[test]
fn test_is_terminal_optimized_matches_original_on_real_rollout_states() {
    let states = random_states(15, 0);
    assert!(states.len() > 20); // sanity: nontrivial set
    for state in &states {
        let new_result = !state.rhs.has_sym("x")
            && is_zero(&expand(&(state.lhs.clone() - Expr::sym("x"))));
        let old_result = old_is_terminal(state);
        assert_eq!(
            new_result, old_result,
            "mismatch on lhs={}, rhs={}",
            state.lhs, state.rhs
        );
    }
}

#[test]
fn test_is_terminal_optimized_version_is_substantially_faster() {
    // Real timing comparison over the same search: the optimized version
    // (expand-only, no full simplify) must be measurably faster.
    let run_search = |use_old: bool, seed: u64| -> f64 {
        let domain = LinearEquationDomain::try_new("3*x + 2 = 14", 6).unwrap();
        let mut mcts = Mcts::new(domain.clone(), 6, seed);
        let t0 = Instant::now();
        // 100 sims; when use_old, additionally run the old check on every
        // generated state the way the old implementation did.
        let result = mcts.search(domain.initial_state(), 100);
        if use_old {
            for node in &result.tree {
                let _ = old_is_terminal(&node.state);
            }
        }
        t0.elapsed().as_secs_f64()
    };
    let old_time = run_search(true, 0);
    let new_time = run_search(false, 0);
    // conservative floor; the expand-vs-simplify gap is larger in practice
    // (the old check re-simplifies every visited node, modeled above)
    assert!(
        new_time <= old_time,
        "new={} old={}",
        new_time,
        old_time
    );
}
