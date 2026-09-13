//! Rust port of python/tests/test_curriculum.py.

use reasoning_curriculum::generator::{generate_curriculum, PayloadValue};
use reasoning_verifier::symbolic_verifier::{verify_algebraic_equivalence, verify_equation_solution};

// ── TestCurriculum ──────────────────────────────────────────────────────
#[test]
fn test_generates_requested_counts() {
    let problems = generate_curriculum(10, None, 1).unwrap();
    let mut by_level: std::collections::HashMap<i64, usize> = std::collections::HashMap::new();
    for p in &problems {
        *by_level.entry(p.level).or_insert(0) += 1;
    }
    assert_eq!(by_level[&1], 10);
    assert_eq!(by_level[&2], 10);
    assert_eq!(by_level[&3], 10);
}

#[test]
fn test_linear_equations_are_actually_correct() {
    let problems = generate_curriculum(15, Some(&[2]), 2).unwrap();
    for p in &problems {
        let equation = match p.payload_get("equation") {
            Some(PayloadValue::Str(s)) => s.clone(),
            other => panic!("missing equation payload: {:?}", other),
        };
        let variable = match p.payload_get("variable") {
            Some(PayloadValue::Str(s)) => s.clone(),
            other => panic!("missing variable payload: {:?}", other),
        };
        let gt = match &p.ground_truth {
            PayloadValue::Int(i) => *i as f64,
            other => panic!("expected int ground truth: {:?}", other),
        };
        let r = verify_equation_solution(&equation, &variable, gt, 1e-9).unwrap();
        assert!(r.passed, "generated bad problem: {}", p.description);
    }
}

#[test]
fn test_polynomial_identities_are_actually_correct() {
    let problems = generate_curriculum(15, Some(&[3]), 3).unwrap();
    for p in &problems {
        let lhs = match p.payload_get("lhs") {
            Some(PayloadValue::Str(s)) => s.clone(),
            other => panic!("missing lhs payload: {:?}", other),
        };
        let gt = match &p.ground_truth {
            PayloadValue::Str(s) => s.clone(),
            other => panic!("expected str ground truth: {:?}", other),
        };
        let r = verify_algebraic_equivalence(&lhs, &gt).unwrap();
        assert!(r.passed, "generated bad identity: {}", p.description);
    }
}

#[test]
fn test_deterministic_with_seed() {
    let a = generate_curriculum(5, None, 42).unwrap();
    let b = generate_curriculum(5, None, 42).unwrap();
    let pa: Vec<&Vec<(String, PayloadValue)>> = a.iter().map(|p| &p.payload).collect();
    let pb: Vec<&Vec<(String, PayloadValue)>> = b.iter().map(|p| &p.payload).collect();
    assert_eq!(pa, pb);
}
