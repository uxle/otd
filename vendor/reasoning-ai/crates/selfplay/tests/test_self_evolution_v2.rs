//! Port of python/tests/test_self_evolution_v2.py (TestSelfEvolutionV2).

use reasoning_selfplay::self_evolution_v2::{make_eval_set, make_round_problems, run_self_evolution_v2};

#[test]
fn test_self_evolution_v2_does_not_regress() {
    let eval_set = make_eval_set(25, 777);
    let (_prm, history) = run_self_evolution_v2(
        4,
        &|r| make_round_problems(r, 10, 200),
        &eval_set,
        300, // round_budget
        300, // eval_budget
        8,   // root_oversample
        60,  // epochs (Python default)
        0.3, // lr (Python default)
        0,   // seed
    );
    println!(
        "\n[Phase 013] self-evolution v2 history: {:?} / {}",
        history,
        eval_set.len()
    );
    // Honest conclusion from this phase (see docstring in
    // self_evolution_v2.rs and the phase writeup): flat across three
    // very different configurations (toy 3-number domain, deeper
    // 4-number domain, multiple budgets from underpowered to generous).
    // A 5-feature linear PRM has essentially no spare capacity, so it
    // converges by round 1 and additional self-play rounds teach it
    // nothing new -- this is a capacity ceiling, not a bug in the
    // pipeline. The pipeline itself is verified correct (weights change
    // every round, no crashes, PRM demonstrably learns per Phase 004/012).
    // Compounding self-evolution gains require a model with real
    // capacity (the neural PRM), not this linear stand-in.

    assert!(
        history[history.len() - 1] >= history[0],
        "self-evolution v2 should not end up worse than the untrained starting point"
    );
}
