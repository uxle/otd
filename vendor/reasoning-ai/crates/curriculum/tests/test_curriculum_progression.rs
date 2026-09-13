//! Rust port of python/tests/test_curriculum_progression.py.

use reasoning_curriculum::progression::{run_curriculum, CurriculumController};

// ── TestCurriculumController ────────────────────────────────────────────
#[test]
fn test_advances_after_consecutive_high_accuracy() {
    let mut c = CurriculumController::new(1, 3, 0.8, 0.2, 2);
    assert_eq!(c.level, 1);
    assert_eq!(c.record(0.9), "hold"); // 1st good, not enough yet
    assert_eq!(c.record(0.95), "advance"); // 2nd consecutive good -> advance
    assert_eq!(c.level, 2);
}

#[test]
fn test_non_consecutive_good_scores_dont_advance() {
    let mut c = CurriculumController::new(1, 3, 0.8, 0.2, 2);
    c.record(0.9); // good (1/2)
    c.record(0.5); // mediocre, resets streak
    c.record(0.9); // good (1/2 again)
    assert_eq!(c.level, 1);
}

#[test]
fn test_regresses_on_bad_accuracy() {
    let mut c = CurriculumController::new(1, 3, 0.8, 0.2, 1);
    c.record(0.9); // advance to level 2
    assert_eq!(c.level, 2);
    c.record(0.1); // crater -> regress back to level 1
    assert_eq!(c.level, 1);
}

#[test]
fn test_never_exceeds_bounds() {
    let mut c = CurriculumController::new(1, 2, 0.8, 0.2, 1);
    c.record(0.9); // -> level 2
    c.record(0.9); // already at max, should hold not error
    assert_eq!(c.level, 2);

    let mut c2 = CurriculumController::new(1, 3, 0.8, 0.2, 1);
    c2.record(0.1); // already at min, should not go below 1
    assert_eq!(c2.level, 1);
}

#[test]
fn test_run_curriculum_with_a_simulated_learner() {
    // simulated learner: gets 95% on level 1 always, 90% on level 2, then a
    // wall at level 3 (30%) -- should climb to 2, maybe touch 3 and bounce
    // back, but never get stuck oscillating forever below 1.
    let accuracy_fn = |level: i64| match level {
        1 => 0.95,
        2 => 0.9,
        3 => 0.3,
        _ => unreachable!(),
    };

    let mut c = CurriculumController::new(1, 3, 0.8, 0.4, 2);
    run_curriculum(&mut c, accuracy_fn, 12);
    assert!(c.level >= 2); // made real progress past level 1
    let levels_seen: Vec<i64> = c.history.iter().map(|h| h.level_after).collect();
    assert!(levels_seen.contains(&1));
}
