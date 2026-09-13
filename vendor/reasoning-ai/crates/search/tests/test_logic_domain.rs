//! Port of python/tests/test_logic_domain.py (TestLogicDomain).

use reasoning_search::{
    brute_force_satisfying_assignments, Domain, LogicFormula, Mcts,
    SatisfiabilityDomain,
};
use std::collections::HashMap;
use std::rc::Rc;

fn formula_abc(a: &HashMap<String, bool>) -> bool {
    // (a OR b) AND (NOT a OR c)
    (a["a"] || a["b"]) && (!a["a"] || a["c"])
}

#[test]
fn test_logic_finds_satisfying_assignment_for_satisfiable_formula() {
    let formula: LogicFormula = Rc::new(|a: &HashMap<String, bool>| {
        (a["a"] || a["b"]) && (!a["a"] || a["c"])
    });
    let domain = SatisfiabilityDomain::new(
        formula.clone(),
        vec!["a".to_string(), "b".to_string(), "c".to_string()],
    );
    assert!(domain.is_satisfiable);

    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain.clone(), 3, 1);
    let result = mcts.search(initial, 200);
    assert!(result.found_verified_solution);

    // independently re-verify against the brute-force ground truth
    let assignment: HashMap<String, bool> = domain
        .variables
        .iter()
        .zip(result.best_terminal_state.as_ref().unwrap().assignment.iter())
        .map(|(v, b)| (v.clone(), b.expect("terminal state is fully assigned")))
        .collect();
    assert!(formula_abc(&assignment));
}

#[test]
fn test_logic_honestly_reports_unsatisfiable_formula() {
    // a AND NOT a -- never satisfiable
    let formula: LogicFormula = Rc::new(|a: &HashMap<String, bool>| a["a"] && !a["a"]);
    let domain = SatisfiabilityDomain::new(formula, vec!["a".to_string()]);
    assert!(!domain.is_satisfiable);

    let initial = domain.initial_state();
    let mut mcts = Mcts::new(domain, 2, 2);
    let result = mcts.search(initial, 50);
    assert!(!result.found_verified_solution);
}

#[test]
fn test_logic_brute_force_matches_hand_computed_truth_table() {
    // a XOR b: satisfying assignments are (T,F) and (F,T)
    let formula: LogicFormula = Rc::new(|a: &HashMap<String, bool>| a["a"] != a["b"]);
    let results = brute_force_satisfying_assignments(&formula, &["a".to_string(), "b".to_string()]);
    assert_eq!(results.len(), 2);
    assert!(results.contains(&{
        let mut m = HashMap::new();
        m.insert("a".to_string(), true);
        m.insert("b".to_string(), false);
        m
    }));
    assert!(results.contains(&{
        let mut m = HashMap::new();
        m.insert("a".to_string(), false);
        m.insert("b".to_string(), true);
        m
    }));
}
