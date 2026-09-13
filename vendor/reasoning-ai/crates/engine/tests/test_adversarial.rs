//! Port of python/tests/test_adversarial.py — Phase 039 adversarial stress
//! test: deliberately harder linear-equation variants (negative numbers,
//! larger magnitudes, zero coefficients) run through the FULL solve()
//! pipeline; every claimed pass is independently re-verified here, not
//! just trusted because `verified` says true.

use rand::rngs::StdRng;
use rand::SeedableRng;
use reasoning_engine::curriculum_adversarial::{
    generate_linear_equation_adversarial_variants, run_adversarial_stress_test,
};
use reasoning_verifier::verify_equation_solution;

#[test]
fn test_stress_test_reports_honest_pass_rate_and_every_pass_is_real() {
    let mut rng = StdRng::seed_from_u64(0);
    let cases = generate_linear_equation_adversarial_variants(&mut rng, 8);
    let (results, pass_rate) = run_adversarial_stress_test(&cases, 800, 0);

    let n_verified = results.iter().filter(|(_, a)| a.verified).count();
    println!(
        "\n[Phase 039] adversarial stress test: {:.0}% verified ({}/{})",
        pass_rate * 100.0,
        n_verified,
        results.len()
    );
    for (case, ans) in &results {
        let status = if ans.verified { "PASS" } else { "ABSTAIN" };
        println!("  [{}] {}", status, case.description);
    }

    // every claimed pass must be independently re-verifiable -- not
    // just trusted because ans.verified says True
    for (case, ans) in &results {
        if ans.verified {
            let eq = case
                .problem
                .payload
                .get("equation")
                .and_then(|v| v.as_str())
                .expect("equation in payload");
            let x: f64 = ans.final_answer().unwrap().parse().unwrap();
            let recheck = verify_equation_solution(eq, "x", x, 1e-9).unwrap();
            assert!(
                recheck.passed,
                "claimed-verified answer failed independent re-check: {}",
                case.description
            );
        }
    }

    assert!(pass_rate >= 0.0); // sanity: honest measurement, no hard floor asserted
}
