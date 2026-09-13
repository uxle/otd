//! Phase 032 — Code-generation domain (Rust port of
//! `search/codegen_domain.py`, design doc section 17: "test generation,
//! test execution, error analysis").
//!
//! Search proposes candidate function BODIES (small code templates), each
//! is run in the Phase 031 sandbox against a set of input/output test
//! cases, and only a candidate that passes every test case counts as
//! verified. No candidate is trusted on a single test.

use crate::domain::Domain;
use reasoning_verifier::code_sandbox::{run_sandboxed, SandboxValue};

#[derive(Debug, Clone, PartialEq)]
pub struct CodeGenState {
    pub template_choice: Option<usize>,
}

pub struct CodeGenDomain {
    /// complete small programs (templates) implementing `f(n)`
    pub templates: Vec<String>,
    /// (input, expected_output) pairs — expected values as sandbox values
    pub test_cases: Vec<(i64, SandboxValue)>,
}

impl CodeGenDomain {
    /// Python `__init__` never fails.
    pub fn new(templates: Vec<String>, test_cases: Vec<(i64, SandboxValue)>) -> Self {
        CodeGenDomain {
            templates,
            test_cases,
        }
    }
}

impl Domain for CodeGenDomain {
    type State = CodeGenState;
    type Action = usize;

    fn initial_state(&self) -> CodeGenState {
        CodeGenState {
            template_choice: None,
        }
    }

    fn legal_actions(&self, state: &CodeGenState) -> Vec<usize> {
        if state.template_choice.is_some() {
            return Vec::new();
        }
        (0..self.templates.len()).collect()
    }

    fn apply(&self, _state: &CodeGenState, action: &usize) -> CodeGenState {
        CodeGenState {
            template_choice: Some(*action),
        }
    }

    fn is_terminal(&self, state: &CodeGenState) -> bool {
        state.template_choice.is_some()
    }

    fn terminal_reward(&self, state: &CodeGenState) -> f64 {
        assert!(self.is_terminal(state));
        let code = &self.templates[state.template_choice.expect("is_terminal checked")];
        let mut passed = 0usize;
        for (arg, expected) in &self.test_cases {
            // a failing/unsafe candidate just scores 0 on this case
            if let Ok(result) = run_sandboxed(code, &format!("f({})", arg), 50_000) {
                if result == *expected {
                    passed += 1;
                }
            }
        }
        if self.test_cases.is_empty() {
            0.0
        } else {
            passed as f64 / self.test_cases.len() as f64
        }
    }
}
