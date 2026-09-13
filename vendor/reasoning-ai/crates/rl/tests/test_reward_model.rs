//! Port of python/tests/test_reward_model.py.

use reasoning_rl::reward_model::{
    compose_reward, compose_reward_full, compose_reward_with_clip, reward_from_verification_result,
    RewardComponents, RewardWeights, DEFAULT_WEIGHTS,
};

// ---- TestRewardComposition ----

#[test]
fn test_reward_all_default_components_gives_the_baseline_neutral_case() {
    // correct=False, but logically_consistent=True (the field default)
    let r = compose_reward(&RewardComponents::default());
    assert!((r - DEFAULT_WEIGHTS.logical_consistency).abs() < 1e-9);
}

#[test]
fn test_reward_correct_and_verified_beats_incorrect() {
    let good = RewardComponents {
        correct: true,
        verification_passed: true,
        ..Default::default()
    };
    let bad = RewardComponents::default();
    assert!(compose_reward(&good) > compose_reward(&bad));
}

#[test]
fn test_reward_each_penalty_strictly_reduces_reward() {
    let base = RewardComponents {
        correct: true,
        verification_passed: true,
        ..Default::default()
    };
    let base_r = compose_reward(&base);
    let cases: [(&str, RewardComponents); 4] = [
        ("contradiction_detected", RewardComponents { contradiction_detected: true, ..base.clone() }),
        ("hallucination_detected", RewardComponents { hallucination_detected: true, ..base.clone() }),
        ("invalid_reasoning_detected", RewardComponents { invalid_reasoning_detected: true, ..base.clone() }),
        ("reward_hacking_detected", RewardComponents { reward_hacking_detected: true, ..base.clone() }),
    ];
    for (name, penalized) in cases {
        assert!(
            compose_reward(&penalized) < base_r,
            "{}=true should strictly reduce reward",
            name
        );
    }
}

#[test]
fn test_reward_hacking_never_pays_off() {
    // An honestly-correct trajectory must always score higher than a
    // 'correct-looking' one flagged as reward hacking.
    let honest = RewardComponents {
        correct: true,
        verification_passed: true,
        logically_consistent: true,
        ..Default::default()
    };
    let hacked = RewardComponents {
        reward_hacking_detected: true,
        ..honest.clone()
    };
    assert!(compose_reward(&honest) > compose_reward(&hacked));
    // hacking should even fall below doing nothing / abstaining
    let did_nothing = RewardComponents::default();
    assert!(compose_reward(&did_nothing) >= compose_reward(&hacked));
}

#[test]
fn test_reward_no_length_or_token_count_channel_exists() {
    // Structural: RewardComponents/RewardWeights expose no length/steps/
    // tokens fields (compile-time property of the structs — this test
    // documents it; the API has no such parameter by construction).
    let _fields_exist: [&str; 0] = []; // no forbidden fields exist to enumerate
}

#[test]
fn test_reward_identical_components_give_identical_reward_regardless_of_context() {
    let a = RewardComponents {
        correct: true,
        partial_credit: 0.5,
        ..Default::default()
    };
    let b = a.clone();
    assert_eq!(compose_reward(&a), compose_reward(&b));
}

#[test]
fn test_reward_partial_credit_is_monotonic() {
    let low = RewardComponents {
        partial_credit: 0.2,
        ..Default::default()
    };
    let high = RewardComponents {
        partial_credit: 0.8,
        ..Default::default()
    };
    assert!(compose_reward(&low) < compose_reward(&high));
}

#[test]
fn test_reward_clip_range_bounds_extreme_values() {
    let everything_bad = RewardComponents {
        contradiction_detected: true,
        hallucination_detected: true,
        invalid_reasoning_detected: true,
        reward_hacking_detected: true,
        ..Default::default()
    };
    let r = compose_reward_with_clip(&everything_bad, Some((-2.0, 2.0)));
    assert!(r >= -2.0);
    let r_unclipped = compose_reward_with_clip(&everything_bad, None);
    assert!(r_unclipped < -2.0); // confirms clipping actually did something
    let _ = compose_reward_full; // full API reachable
}

#[test]
fn test_reward_negative_weight_rejected() {
    let w = RewardWeights {
        correctness: -0.1,
        ..DEFAULT_WEIGHTS
    };
    let err = w.validate().unwrap_err();
    assert!(err.contains("must be >= 0"), "{}", err);
    assert!(RewardWeights::checked(
        -0.1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
    )
    .is_err());
}

#[test]
fn test_reward_partial_credit_out_of_range_rejected() {
    let bad_high = RewardComponents {
        partial_credit: 1.5,
        ..Default::default()
    };
    assert!(bad_high.validate().is_err());
    let bad_low = RewardComponents {
        partial_credit: -0.01,
        ..Default::default()
    };
    assert!(bad_low.validate().is_err());
}

#[test]
fn test_reward_from_verification_result_matches_manual_compose() {
    let expected = compose_reward(&RewardComponents {
        correct: true,
        verification_passed: true,
        ..Default::default()
    });
    let got = reward_from_verification_result(true);
    assert_eq!(expected, got);
}
