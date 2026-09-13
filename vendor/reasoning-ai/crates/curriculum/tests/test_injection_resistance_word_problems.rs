//! Rust port of the word-problem part of python/tests/test_injection_resistance.py:
//! `TestWordProblemParserAlreadySafeByConstruction` (Phase 046).
//!
//! The other classes of that file (core parser / domain injection) exercise
//! the verifier and search crates and were ported with those crates; this
//! file covers the word-problem parser path, which belongs to curriculum.

use reasoning_curriculum::word_problems::solve_word_problem;

// The Phase 046 payload list (only the first is used by this test class).
const INJECTION_PAYLOADS: [&str; 5] = [
    "__import__(\"os\").system(\"echo INJECTION_SUCCEEDED\")",
    "eval(\"1+1\")",
    "exec(\"import os\")",
    "open(\"/etc/passwd\").read()",
    "().__class__.__bases__[0].__subclasses__()",
];

#[test]
fn test_injection_embedded_in_word_problem_text_never_reaches_a_parser() {
    // The word-problem parser only extracts digit groups via regex
    // into a fixed equation template -- it never interpolates
    // arbitrary user text into anything that gets parsed as code or
    // math. Confirm that holds even with a payload embedded in
    // otherwise-valid-looking problem text.
    let text = format!(
        "4 more than 3 times a number is 19; {}",
        INJECTION_PAYLOADS[0]
    );
    let (equation, answer) = solve_word_problem(&text, 800, 1);
    // it should still correctly extract just the equation, ignoring
    // the injected payload entirely
    assert_eq!(equation.as_deref(), Some("3*x + 4 = 19"));
    assert!((answer.unwrap() - 5.0).abs() < 1e-9);
}
