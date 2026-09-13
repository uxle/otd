//! Phase 047 — apps/train: training orchestration script (Rust port of
//! `python/apps/train.py`).
//!
//! Ties together pieces already built and individually tested (Phase 005
//! curriculum, Phase 009/013 self-evolution, Phase 011 calibration fix,
//! Phase 010/020 evaluation) into one runnable pipeline. Doesn't claim
//! this produces compounding self-improvement — Phase 013 already
//! established that requires more model capacity than the linear PRM has.
//! What this DOES verify: the full pipeline runs start-to-finish without
//! manual wiring, and produces a real before/after evaluation number
//! either way.
//!
//! Run with: reasoning-train (rounds/seed default to the Python's 3/0).

use reasoning_selfplay::{make_eval_set, make_round_problems, run_self_evolution_v2};

fn main() {
    // Python: `if __name__ == "__main__": main()` with main(rounds=3, seed=0).
    let rounds = 3usize;
    let seed = 0u64;

    println!("=== apps/train: full pipeline run ===");
    println!("Rounds: {}", rounds);

    let eval_set = make_eval_set(15, 777);
    println!(
        "Held-out eval set: {} problems (not used in training)",
        eval_set.len()
    );

    println!(
        "\nRunning self-evolution loop (curriculum -> guided search -> verify -> \
         retrain PRM -> repeat)..."
    );
    // Python defaults for the unpinned kwargs: epochs=60, lr=0.3.
    let (_prm, history) = run_self_evolution_v2(
        rounds,
        &|r| make_round_problems(r, 8, 200),
        &eval_set,
        300, // round_budget
        300, // eval_budget
        8,   // root_oversample
        60,  // epochs
        0.3, // lr
        seed,
    );

    println!(
        "\nEval history (round 0 = untrained baseline): {:?} / {}",
        history,
        eval_set.len()
    );
    let (first, last) = (history[0], history[history.len() - 1]);
    if last > first {
        println!("Result: measurable improvement over training.");
    } else if last == first {
        println!(
            "Result: no measurable improvement (consistent with Phase 013's \
             finding that the linear PRM has no spare capacity -- expected, \
             not a pipeline failure)."
        );
    } else {
        println!(
            "Result: WORSE than baseline -- this would be a genuine regression, \
             worth investigating, not smoothing over."
        );
    }
}
