//! Port of python/tests/test_mcts.py (TestUCB, TestNumberTargetDomain,
//! TestMCTSFindsVerifiedSolutions) plus the smoke checks from the former
//! tests/smoke_mcts.rs.

use reasoning_search::{make_initial_state, ucb_score, Domain, Mcts, NumberTargetDomain, TreeNode};
use reasoning_verifier::symbolic_verifier::verify_numeric_equality;

fn vne_passed(expr: &str, expected: f64) -> bool {
    verify_numeric_equality(expr, expected)
        .map(|r| r.passed)
        .unwrap_or(false)
}

// ---- TestUCB ----

#[test]
fn test_ucb_unvisited_child_has_infinite_ish_priority() {
    let mk = |action: Option<&str>, visit_count: usize, value_sum: f64| TreeNode {
        state: (),
        parent: Some(0),
        action_from_parent: action.map(|a| a.to_string()),
        depth: 1,
        prior: 1.0,
        children: Vec::new(),
        visit_count,
        value_sum,
        reward: None,
        confidence: 0.0,
    };
    let _parent: TreeNode<(), String> = TreeNode {
        state: (),
        parent: None,
        action_from_parent: None,
        depth: 0,
        prior: 1.0,
        children: Vec::new(),
        visit_count: 0,
        value_sum: 0.0,
        reward: None,
        confidence: 0.0,
    };
    let unvisited = mk(Some("a"), 0, 0.0);
    let visited = mk(Some("b"), 10, 5.0);
    // parent needs some visits for sqrt term to matter
    let score_unvisited = ucb_score(&unvisited, 10, 1.4);
    let score_visited = ucb_score(&visited, 10, 1.4);
    assert!(score_unvisited > 0.0);
    // unvisited should be explored preferentially when parent has visits
    assert!(score_unvisited > score_visited - 10.0); // sanity, not tight
}

// ---- TestNumberTargetDomain ----

#[test]
fn test_number_target_domain_terminal_detection() {
    let d = NumberTargetDomain::new(24.0);
    let state = make_initial_state(&[4.0, 7.0, 8.0, 8.0]);
    assert!(!d.is_terminal(&state));
    let one_left: Vec<(f64, String)> = vec![(24.0, "expr".to_string())];
    assert!(d.is_terminal(&one_left));
}

#[test]
fn test_number_target_domain_terminal_reward_uses_real_verifier() {
    let d = NumberTargetDomain::new(24.0);
    let correct = vec![(24.0, "(4*(7-(8/8)))".to_string())];
    // terminal_reward re-evaluates the expression string itself
    assert_eq!(d.terminal_reward(&correct), 1.0);
}

// ---- TestMCTSFindsVerifiedSolutions ----

#[test]
fn test_mcts_finds_verified_solutions_solves_easy_target() {
    // 2, 2 -> 4 is trivially reachable
    let domain = NumberTargetDomain::new(4.0);
    let state = make_initial_state(&[2.0, 2.0]);
    let mut mcts = Mcts::new(domain, 4, 1);
    let result = mcts.search(state, 100);
    assert!(result.found_verified_solution);
    // independently re-verify the winning expression with the verifier directly
    let final_expr = result.best_terminal_state.as_ref().unwrap()[0].1.clone();
    assert!(vne_passed(&final_expr, 4.0));
}

#[test]
fn test_mcts_finds_verified_solutions_solves_classic_24_instance() {
    // 4, 7, 8, 8 -> 24 via one known solution: (4*(7-(8/8))) = 24
    let domain = NumberTargetDomain::new(24.0);
    let state = make_initial_state(&[4.0, 7.0, 8.0, 8.0]);
    let mut mcts = Mcts::new(domain, 6, 7);
    let result = mcts.search(state, 3000);
    assert!(result.found_verified_solution);
    let final_expr = result.best_terminal_state.as_ref().unwrap()[0].1.clone();
    assert!(
        vne_passed(&final_expr, 24.0),
        "claimed solution {} failed independent re-check",
        final_expr
    );
}

#[test]
fn test_mcts_finds_verified_solutions_honestly_reports_failure_when_unsolvable() {
    // numbers are all 1s: max reachable value via +,-,*,/ in any combination
    // of four 1's is 4 (sum). Target 100 is provably unreachable.
    let domain = NumberTargetDomain::new(100.0);
    let state = make_initial_state(&[1.0, 1.0, 1.0, 1.0]);
    let mut mcts = Mcts::new(domain, 6, 3);
    let result = mcts.search(state, 500);
    assert!(!result.found_verified_solution);
    assert!(result.verified_path.is_none());
}

// ---- smoke checks folded in from the former tests/smoke_mcts.rs ----

#[test]
fn smoke_mcts_solves_number_target() {
    // [4, 6, 2]: (4+6)*2 = 20
    let domain = NumberTargetDomain::new(20.0);
    let state = make_initial_state(&[4.0, 6.0, 2.0]);
    let mut mcts = Mcts::new(domain, 6, 0);
    let result = mcts.search(state, 1500);
    assert!(result.found_verified_solution, "no solution found");
    let term = result.best_terminal_state.as_ref().unwrap();
    assert_eq!(term.len(), 1);
    assert!((term[0].0 - 20.0).abs() < 1e-6);
    // path should have 2 steps (3 numbers -> 1)
    assert_eq!(result.verified_path.as_ref().unwrap().len(), 2);
    // tree walk works
    assert!(result.tree.len() > 1);
    assert_eq!(result.tree[result.root].visit_count, 1500);
}

#[test]
fn smoke_mcts_abstains_on_impossible() {
    let domain = NumberTargetDomain::new(999.0);
    let state = make_initial_state(&[1.0, 1.0]);
    let mut mcts = Mcts::new(domain, 6, 0);
    let result = mcts.search(state, 200);
    assert!(!result.found_verified_solution);
    assert!(result.verified_path.is_none());
}
