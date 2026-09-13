//! Rust port of python/tests/test_rl_phases_76_78.py:
//!   077 — adaptive exploration
//!   078 — checkpointing
//!
//! Phase 076 (TestCurriculumGRPO) is NOT ported here: it exercises
//! `curriculum::progression::CurriculumController`, which lives in the
//! `reasoning-curriculum` crate that was still a placeholder when this
//! crate was ported (see src/curriculum_grpo.rs). Those tests belong to
//! that crate's port of test_curriculum_progression plus the eventual
//! wiring of `run_curriculum_grpo_training`.
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};

use serde_json::json;

use reasoning_rl::adaptive_exploration::AdaptiveExplorationController;
use reasoning_rl::checkpoint::{
    load_policy_checkpoint, load_training_history, save_policy_checkpoint,
    save_training_history,
};
use reasoning_rl::grpo_training_loop::TrainingHistory;
use reasoning_rl::policy::{SoftmaxPolicy, FEATURE_KEYS};

fn temp_path(name: &str) -> PathBuf {
    static COUNTER: AtomicUsize = AtomicUsize::new(0);
    let d = std::env::temp_dir().join(format!(
        "rl_ckpt_{}_{}/",
        std::process::id(),
        COUNTER.fetch_add(1, Ordering::SeqCst)
    ));
    std::fs::create_dir_all(&d).expect("temp dir");
    d.join(name)
}

// ── Phase 077 ───────────────────────────────────────────────────────────
#[test]
fn test_adaptive_exploration_healthy_entropy_and_gradient_gives_base_epsilon() {
    let mut ctrl = AdaptiveExplorationController::default();
    let eps = ctrl.record_and_get_epsilon(0.8, 0.5);
    assert_eq!(eps, ctrl.base_epsilon);
}

#[test]
fn test_adaptive_exploration_collapsed_entropy_triggers_boost() {
    let mut ctrl = AdaptiveExplorationController::default();
    ctrl.record_and_get_epsilon(0.8, 0.5);
    let eps = ctrl.record_and_get_epsilon(0.01, 0.5);
    assert_eq!(eps, ctrl.boosted_epsilon);
}

#[test]
fn test_adaptive_exploration_stagnant_gradient_triggers_boost() {
    let mut ctrl = AdaptiveExplorationController::default();
    let mut eps = 0.0;
    for _ in 0..3 {
        eps = ctrl.record_and_get_epsilon(0.8, 1e-6);
    }
    assert_eq!(eps, ctrl.boosted_epsilon);
}

#[test]
fn test_adaptive_exploration_summary_reports_boosted_fraction() {
    let mut ctrl = AdaptiveExplorationController::default();
    ctrl.record_and_get_epsilon(0.01, 0.5); // boosted (collapsed)
    ctrl.record_and_get_epsilon(0.8, 0.5); // not boosted
    let summary = ctrl.summary();
    assert_eq!(summary.rounds_recorded, 2);
    assert!((summary.fraction_boosted - 0.5).abs() < 1e-7);
}

#[test]
fn test_adaptive_exploration_empty_summary_does_not_crash() {
    let ctrl = AdaptiveExplorationController::default();
    let summary = ctrl.summary();
    assert_eq!(summary.rounds_recorded, 0);
    assert_eq!(summary.fraction_boosted, 0.0);
}

// ── Phase 078 ───────────────────────────────────────────────────────────
#[test]
fn test_checkpoint_save_and_load_policy_roundtrip() {
    let policy = SoftmaxPolicy {
        weights: [
            ("bias", 0.3),
            ("num_remaining", -0.1),
            ("min_diff_to_target_norm", 0.5),
            ("depth_norm", -0.2),
            ("has_exact_match", 1.7),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect(),
    };
    let path = temp_path("policy.json");
    let p_str = path.to_str().unwrap().to_string();
    save_policy_checkpoint(
        &policy,
        &p_str,
        Some(json!({"round": 5})),
    )
    .unwrap();
    let loaded = load_policy_checkpoint(&p_str).unwrap();
    for k in FEATURE_KEYS {
        assert!(
            (loaded.weights.get(k).copied().unwrap_or(0.0)
                - policy.weights.get(k).copied().unwrap_or(0.0))
                .abs()
                < 1e-10
        );
    }
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(path.parent().unwrap());
}

#[test]
fn test_checkpoint_file_is_valid_json_not_pickle() {
    let policy = SoftmaxPolicy::new();
    let path = temp_path("policy.json");
    let p_str = path.to_str().unwrap().to_string();
    save_policy_checkpoint(&policy, &p_str, None).unwrap();
    // would fail if it weren't plain JSON
    let text = std::fs::read_to_string(&p_str).unwrap();
    let data: serde_json::Value = serde_json::from_str(&text).unwrap();
    let obj = data.as_object().unwrap();
    assert!(obj.contains_key("weights"));
    assert!(obj.contains_key("saved_at"));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(path.parent().unwrap());
}

#[test]
fn test_checkpoint_load_rejects_unknown_feature_keys() {
    let path = temp_path("bad.json");
    let p_str = path.to_str().unwrap().to_string();
    std::fs::write(
        &p_str,
        serde_json::to_string(&json!({"weights": {"totally_made_up_feature": 1.0}})).unwrap(),
    )
    .unwrap();
    let err = load_policy_checkpoint(&p_str).unwrap_err();
    assert!(err.contains("unknown feature keys"));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(path.parent().unwrap());
}

#[test]
fn test_checkpoint_save_and_load_training_history_roundtrip() {
    let history = TrainingHistory {
        round_solve_rates: vec![0.1, 0.2],
        round_mean_rewards: vec![0.5, 0.6],
        round_grad_norms: vec![0.3, 0.1],
        eval_solve_rate_before: 0.1,
        eval_solve_rate_after: 0.3,
    };
    let path = temp_path("history.json");
    let p_str = path.to_str().unwrap().to_string();
    save_training_history(&history, &p_str).unwrap();
    let loaded = load_training_history(&p_str).unwrap();
    assert_eq!(loaded["round_solve_rates"], json!([0.1, 0.2]));
    assert_eq!(loaded["eval_solve_rate_after"], json!(0.3));
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(path.parent().unwrap());
}

#[test]
fn test_checkpoint_write_is_atomic_no_tmp_left_behind() {
    let policy = SoftmaxPolicy::new();
    let path = temp_path("policy.json");
    let p_str = path.to_str().unwrap().to_string();
    save_policy_checkpoint(&policy, &p_str, None).unwrap();
    assert!(!std::path::Path::new(&format!("{}.tmp", p_str)).exists());
    assert!(std::path::Path::new(&p_str).exists());
    let _ = std::fs::remove_file(&path);
    let _ = std::fs::remove_dir(path.parent().unwrap());
}
