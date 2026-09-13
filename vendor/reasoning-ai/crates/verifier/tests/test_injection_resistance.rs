//! Port of python/tests/test_injection_resistance.py (the word-problem
//! portion lives in reasoning-curriculum's tests where the parser is).

use reasoning_search::calculus_domain::DerivativeDomain;
use reasoning_search::linear_equation_domain::LinearEquationDomain;
use reasoning_verifier::symbolic_verifier::{safe_parse, verify_numeric_equality};

const INJECTION_PAYLOADS: [&str; 5] = [
    "__import__(\"os\").system(\"echo INJECTION_SUCCEEDED\")",
    "eval(\"1+1\")",
    "exec(\"import os\")",
    "open(\"/etc/passwd\").read()",
    "().__class__.__bases__[0].__subclasses__()",
];

// ---- TestCoreParserBlocksInjection ----

#[test]
fn test_injection_safe_parse_blocks_every_known_payload() {
    for payload in INJECTION_PAYLOADS {
        let err = safe_parse(payload)
            .map_err(|_| ())
            .expect_err(&format!("payload not blocked: {}", payload));
        let _ = err;
    }
}

#[test]
fn test_injection_verify_numeric_equality_blocks_injection_in_expression() {
    assert!(verify_numeric_equality("__import__(\"os\").system(\"echo x\")", 1.0).is_err());
}

// ---- TestDomainsBlockInjection ----

#[test]
fn test_injection_linear_equation_domain_blocks_injection() {
    for payload in INJECTION_PAYLOADS {
        let eq = format!("{} + x = 0", payload);
        assert!(
            LinearEquationDomain::try_new(&eq, 6).is_err(),
            "payload not blocked: {}",
            payload
        );
    }
}

#[test]
fn test_injection_calculus_domain_blocks_injection() {
    for payload in INJECTION_PAYLOADS {
        assert!(
            DerivativeDomain::try_new(payload, vec!["1".to_string()]).is_err(),
            "payload not blocked: {}",
            payload
        );
    }
}
