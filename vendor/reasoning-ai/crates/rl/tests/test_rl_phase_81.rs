//! Rust port of python/tests/test_rl_phase_81.py:
//!   081 — effect size and power analysis for proportion comparisons
use reasoning_rl::effect_size::{
    cohens_h, effect_size_label, inverse_standard_normal_cdf, required_n_for_power,
};

// ── Inverse normal CDF ──────────────────────────────────────────────────
#[test]
fn test_inverse_normal_cdf_known_quantiles() {
    assert!((inverse_standard_normal_cdf(0.975).unwrap() - 1.959964).abs() < 1e-4);
    assert!(inverse_standard_normal_cdf(0.5).unwrap().abs() < 1e-6);
    assert!((inverse_standard_normal_cdf(0.8).unwrap() - 0.841621).abs() < 1e-4);
    assert!((inverse_standard_normal_cdf(0.9).unwrap() - 1.281552).abs() < 1e-4);
}

#[test]
fn test_inverse_normal_cdf_symmetry() {
    for p in [0.1, 0.3, 0.6, 0.9] {
        assert!(
            (inverse_standard_normal_cdf(p).unwrap()
                - -inverse_standard_normal_cdf(1.0 - p).unwrap())
            .abs()
                < 1e-4
        );
    }
}

#[test]
fn test_inverse_normal_cdf_out_of_range_raises() {
    assert!(inverse_standard_normal_cdf(0.0).is_err());
    assert!(inverse_standard_normal_cdf(1.0).is_err());
}

// ── Cohen's h ───────────────────────────────────────────────────────────
#[test]
fn test_cohens_h_identical_proportions_zero_effect() {
    assert!(cohens_h(0.5, 0.5).unwrap().abs() < 1e-10);
}

#[test]
fn test_cohens_h_sign_flips_with_argument_order() {
    let h1 = cohens_h(0.7, 0.3).unwrap();
    let h2 = cohens_h(0.3, 0.7).unwrap();
    assert!((h1 - -h2).abs() < 1e-10);
}

#[test]
fn test_cohens_h_known_reference_value() {
    // Cohen's textbook example: p1=0.5, p2=0.7 -> h ~ -0.4115
    let h = cohens_h(0.5, 0.7).unwrap();
    assert!((h - -0.4115168).abs() < 1e-5);
}

#[test]
fn test_cohens_h_phase_75_numbers_are_small_or_negligible() {
    // uniform=0.75 vs grpo_mcts=0.75 -> exactly zero (identical in that run)
    assert!(cohens_h(0.75, 0.75).unwrap().abs() < 1e-10);
    // uniform=0.75 vs prm_guided=0.575
    let h = cohens_h(0.75, 0.575).unwrap();
    let label = effect_size_label(h);
    assert!(label == "small" || label == "medium");
}

#[test]
fn test_cohens_h_out_of_range_raises() {
    assert!(cohens_h(1.5, 0.5).is_err());
}

// ── Effect size labels ──────────────────────────────────────────────────
#[test]
fn test_effect_size_labels() {
    assert_eq!(effect_size_label(0.05), "negligible");
    assert_eq!(effect_size_label(0.3), "small");
    assert_eq!(effect_size_label(0.6), "medium");
    assert_eq!(effect_size_label(0.9), "large");
}

#[test]
fn test_effect_size_labels_are_sign_insensitive() {
    assert_eq!(effect_size_label(-0.9), "large");
}

// ── Required N for power ────────────────────────────────────────────────
#[test]
fn test_required_n_for_power_larger_effect_needs_fewer_samples() {
    let n_small_effect = required_n_for_power(0.5, 0.55, 0.8, 0.05).unwrap();
    let n_large_effect = required_n_for_power(0.3, 0.8, 0.8, 0.05).unwrap();
    assert!(n_small_effect > n_large_effect);
}

#[test]
fn test_required_n_for_power_higher_power_needs_more_samples() {
    let n_80 = required_n_for_power(0.3, 0.5, 0.8, 0.05).unwrap();
    let n_95 = required_n_for_power(0.3, 0.5, 0.95, 0.05).unwrap();
    assert!(n_95 > n_80);
}

#[test]
fn test_required_n_for_power_equal_proportions_raises() {
    assert!(required_n_for_power(0.5, 0.5, 0.8, 0.05).is_err());
}

#[test]
fn test_required_n_for_power_matches_phase_75_scale() {
    // To reliably detect uniform=0.75 vs prm=0.575 (a real gap Phase 75
    // saw but couldn't confirm with N=40), how many WOULD be needed?
    let n = required_n_for_power(0.75, 0.575, 0.8, 0.05).unwrap();
    assert!(n > 40); // confirms N=40 genuinely was too few
}
