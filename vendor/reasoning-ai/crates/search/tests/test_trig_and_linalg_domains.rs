//! Port of python/tests/test_trig_and_linalg_domains.py
//! (TestTrigSimplifyDomain, TestTrigEvaluateDomain,
//! TestMatrixDeterminantDomain, TestMatrixMultiplyDomain,
//! TestLinearSystemDomain).

use reasoning_search::{
    Domain, LinAlgState, LinearSystemDomain, MatrixDeterminantDomain, MatrixMultiplyDomain,
    TrigEvaluateDomain, TrigSimplifyDomain,
};

// ---- TestTrigSimplifyDomain ----

#[test]
fn test_trig_simplify_pythagorean_identity() {
    // Hand-known: sin(x)^2 + cos(x)^2 = 1 for all x (Pythagorean identity)
    let domain = TrigSimplifyDomain::try_new(
        "sin(x)**2 + cos(x)**2",
        vec![
            "1".to_string(),
            "0".to_string(),
            "2".to_string(),
            "sin(x)".to_string(),
        ],
    )
    .unwrap();
    assert_eq!(domain.ground_truth, "1");
    for (i, cand) in domain.candidates.iter().enumerate() {
        let state = domain.apply(&domain.initial_state(), &i);
        let reward = domain.terminal_reward(&state);
        let expected = if cand == "1" { 1.0 } else { 0.0 };
        assert_eq!(reward, expected, "candidate {:?} reward mismatch", cand);
    }
}

#[test]
fn test_trig_simplify_double_angle_like_simplification() {
    // Hand-known: 2*sin(x)*cos(x) = sin(2x)
    let domain = TrigSimplifyDomain::try_new(
        "2*sin(x)*cos(x)",
        vec![
            "sin(2*x)".to_string(),
            "cos(2*x)".to_string(),
            "2*sin(x)".to_string(),
        ],
    )
    .unwrap();
    let state = domain.apply(&domain.initial_state(), &0); // "sin(2*x)"
    assert_eq!(domain.terminal_reward(&state), 1.0);
    let state_wrong = domain.apply(&domain.initial_state(), &1); // "cos(2*x)"
    assert_eq!(domain.terminal_reward(&state_wrong), 0.0);
}

#[test]
fn test_trig_simplify_legal_actions_empty_after_choice() {
    let domain =
        TrigSimplifyDomain::try_new("sin(x)**2 + cos(x)**2", vec!["1".to_string(), "0".to_string()])
            .unwrap();
    let state = domain.initial_state();
    assert_eq!(domain.legal_actions(&state), vec![0, 1]);
    let chosen = domain.apply(&state, &0);
    assert_eq!(domain.legal_actions(&chosen), Vec::<usize>::new());
    assert!(domain.is_terminal(&chosen));
}

// ---- TestTrigEvaluateDomain ----

#[test]
fn test_trig_evaluate_sin_pi_over_6() {
    // Hand-known exact value: sin(pi/6) = 1/2
    let domain = TrigEvaluateDomain::try_new(
        "sin(pi/6)",
        vec![
            "1/2".to_string(),
            "sqrt(3)/2".to_string(),
            "1".to_string(),
            "0".to_string(),
        ],
    )
    .unwrap();
    let state = domain.apply(&domain.initial_state(), &0);
    assert_eq!(domain.terminal_reward(&state), 1.0);
}

#[test]
fn test_trig_evaluate_cos_pi_over_3() {
    // Hand-known exact value: cos(pi/3) = 1/2
    let domain = TrigEvaluateDomain::try_new(
        "cos(pi/3)",
        vec![
            "1/2".to_string(),
            "sqrt(3)/2".to_string(),
            "0".to_string(),
        ],
    )
    .unwrap();
    let state = domain.apply(&domain.initial_state(), &0);
    assert_eq!(domain.terminal_reward(&state), 1.0);
}

#[test]
fn test_trig_evaluate_tan_pi_over_4() {
    // Hand-known exact value: tan(pi/4) = 1
    let domain = TrigEvaluateDomain::try_new(
        "tan(pi/4)",
        vec!["1".to_string(), "0".to_string(), "sqrt(2)".to_string()],
    )
    .unwrap();
    let state = domain.apply(&domain.initial_state(), &0);
    assert_eq!(domain.terminal_reward(&state), 1.0);
}

#[test]
fn test_trig_evaluate_sin_pi_over_4_is_sqrt2_over_2() {
    // Hand-known exact value: sin(pi/4) = sqrt(2)/2
    let domain = TrigEvaluateDomain::try_new(
        "sin(pi/4)",
        vec![
            "sqrt(2)/2".to_string(),
            "1/2".to_string(),
            "1".to_string(),
        ],
    )
    .unwrap();
    let state = domain.apply(&domain.initial_state(), &0);
    assert_eq!(domain.terminal_reward(&state), 1.0);
    let wrong = domain.apply(&domain.initial_state(), &1);
    assert_eq!(domain.terminal_reward(&wrong), 0.0);
}

// ---- TestMatrixDeterminantDomain ----

#[test]
fn test_matrix_determinant_2x2_hand_computed() {
    // Hand-known: det([[3,8],[4,6]]) = 3*6 - 8*4 = 18-32 = -14
    let domain = MatrixDeterminantDomain::try_new(
        &[vec![3, 8], vec![4, 6]],
        vec![-14, 14, 18, 0],
    )
    .unwrap();
    assert_eq!(domain.ground_truth, -14);
    let state = domain.apply(&domain.initial_state(), &0);
    assert_eq!(domain.terminal_reward(&state), 1.0);
    let wrong = domain.apply(&domain.initial_state(), &1);
    assert_eq!(domain.terminal_reward(&wrong), 0.0);
}

#[test]
fn test_matrix_determinant_3x3_hand_computed() {
    // Hand-known: det(identity-like) det([[1,0,0],[0,1,0],[0,0,1]]) = 1
    let domain = MatrixDeterminantDomain::try_new(
        &[vec![1, 0, 0], vec![0, 1, 0], vec![0, 0, 1]],
        vec![1, 0, -1],
    )
    .unwrap();
    assert_eq!(domain.ground_truth, 1);
}

#[test]
fn test_matrix_determinant_non_square_raises() {
    let err = MatrixDeterminantDomain::try_new(&[vec![1, 2, 3], vec![4, 5, 6]], vec![0]);
    assert!(err.is_err()); // Python: assertRaises(ValueError)
}

// ---- TestMatrixMultiplyDomain ----

#[test]
fn test_matrix_multiply_2x2_hand_computed() {
    // Hand-known: [[1,2],[3,4]] @ [[5,6],[7,8]] = [[19,22],[43,50]]
    let a = vec![vec![1, 2], vec![3, 4]];
    let b = vec![vec![5, 6], vec![7, 8]];
    let correct = vec![vec![19, 22], vec![43, 50]];
    let wrong = vec![vec![1, 1], vec![1, 1]];
    let domain =
        MatrixMultiplyDomain::try_new(&a, &b, vec![correct, wrong]).unwrap();
    let state_correct = domain.apply(&domain.initial_state(), &0);
    let state_wrong = domain.apply(&domain.initial_state(), &1);
    assert_eq!(domain.terminal_reward(&state_correct), 1.0);
    assert_eq!(domain.terminal_reward(&state_wrong), 0.0);
}

#[test]
fn test_matrix_multiply_incompatible_shapes_raises() {
    let err = MatrixMultiplyDomain::try_new(&[vec![1, 2, 3]], &[vec![1, 2]], vec![vec![vec![0]]]);
    assert!(err.is_err()); // Python: assertRaises(ValueError)
}

#[test]
fn test_matrix_multiply_malformed_candidate_scores_zero_not_crash() {
    let a = vec![vec![1, 0], vec![0, 1]];
    let b = vec![vec![2, 0], vec![0, 2]];
    let domain = MatrixMultiplyDomain::try_new(&a, &b, vec![vec![vec![1, 2, 3]]]).unwrap();
    let state = domain.apply(&domain.initial_state(), &0); // wrong shape
    assert_eq!(domain.terminal_reward(&state), 0.0);
}

// ---- TestLinearSystemDomain ----

#[test]
fn test_linear_system_2x2_system_hand_computed() {
    // Hand-known: x + y = 5, x - y = 1 -> x=3, y=2
    let a = vec![vec![1, 1], vec![1, -1]];
    let b = vec![5, 1];
    let domain = LinearSystemDomain::try_new(
        &a,
        &b,
        vec![vec![3, 2], vec![2, 3], vec![0, 5]],
    )
    .unwrap();
    let state_correct = domain.apply(&domain.initial_state(), &0);
    let state_wrong = domain.apply(&domain.initial_state(), &1);
    assert_eq!(domain.terminal_reward(&state_correct), 1.0);
    assert_eq!(domain.terminal_reward(&state_wrong), 0.0);
}

#[test]
fn test_linear_system_ground_truth_matches_hand_solution() {
    let a = vec![vec![2, 0], vec![0, 3]];
    let b = vec![6, 9];
    let domain = LinearSystemDomain::try_new(&a, &b, vec![vec![3, 3]]).unwrap();
    // 2x=6->x=3, 3y=9->y=3
    assert_eq!(domain.ground_truth, vec!["3".to_string(), "3".to_string()]);
}

#[test]
fn test_linear_system_singular_system_raises() {
    // x + y = 1, 2x + 2y = 2 — infinitely many solutions, not uniquely
    // solvable (Python: assertRaises(Exception))
    let err = LinearSystemDomain::try_new(&[vec![1, 1], vec![2, 2]], &[1, 2], vec![vec![1, 0]]);
    assert!(err.is_err());
}

// LinAlgState equality helper sanity (state structs are comparable).
#[test]
fn test_linalg_state_structs() {
    let s1 = LinAlgState { choice: None };
    let s2 = LinAlgState { choice: Some(1) };
    assert!(s1 != s2);
    assert_eq!(s1.clone(), s1);
}
