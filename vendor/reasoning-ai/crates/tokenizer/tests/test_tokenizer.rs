//! Rust port of python/tests/test_tokenizer.py.

use reasoning_tokenizer::MathTokenizer;

// ── TestMathTokenizer ───────────────────────────────────────────────────
#[test]
fn test_digit_level_number_tokenization() {
    let tok = MathTokenizer::new();
    let tokens = tok.tokenize("123");
    assert_eq!(tokens, vec!["1", "2", "3"]);
}

#[test]
fn test_operators_are_atomic() {
    let tok = MathTokenizer::new();
    let tokens = tok.tokenize("2+3*4");
    assert_eq!(tokens, vec!["2", "+", "3", "*", "4"]);
}

#[test]
fn test_power_operator_atomic() {
    let tok = MathTokenizer::new();
    let tokens = tok.tokenize("x**2");
    assert_eq!(tokens, vec!["x", "**", "2"]);
}

#[test]
fn test_equation_tokenization() {
    let tok = MathTokenizer::new();
    let tokens = tok.tokenize("2*x+4=10");
    assert_eq!(
        tokens,
        vec!["2", "*", "x", "+", "4", "=", "1", "0"]
    );
}

#[test]
fn test_encode_decode_roundtrip() {
    let tok = MathTokenizer::new();
    let original = "(x+1)*(x-3)=0";
    let ids = tok.encode(original, true);
    let decoded = tok.decode(&ids, true);
    assert_eq!(decoded, original);
}

#[test]
fn test_encode_adds_bos_eos() {
    let tok = MathTokenizer::new();
    let ids = tok.encode("2+2", true);
    assert_eq!(ids[0], tok.bos_id);
    assert_eq!(ids[ids.len() - 1], tok.eos_id);
}

#[test]
fn test_negative_numbers_and_decimals() {
    let tok = MathTokenizer::new();
    let original = "3.14-2.71";
    let ids = tok.encode(original, true);
    assert_eq!(tok.decode(&ids, true), original);
}

#[test]
fn test_unknown_character_does_not_crash_or_silently_drop() {
    // '@' isn't in the vocab -- must become <UNK>, not vanish
    let tok = MathTokenizer::new();
    let ids = tok.encode("2@3", false);
    let decoded_with_special = tok.decode(&ids, false);
    assert!(decoded_with_special.contains("<UNK>"));
}

#[test]
fn test_padding() {
    let tok = MathTokenizer::new();
    let ids = tok.encode("1+1", false);
    let padded = tok.pad(ids.clone(), 10);
    assert_eq!(padded.len(), 10);
    let tail: Vec<usize> = padded[ids.len()..].to_vec();
    assert_eq!(tail, vec![tok.pad_id; 10 - ids.len()]);
}

#[test]
fn test_padding_truncates_if_too_long() {
    let tok = MathTokenizer::new();
    let ids = tok.encode("123456789", false);
    let padded = tok.pad(ids, 3);
    assert_eq!(padded.len(), 3);
}

#[test]
fn test_vocab_size_reasonable() {
    // 4 special + 10 digits + 10 operators + 3 variables = 27
    let tok = MathTokenizer::new();
    assert_eq!(tok.vocab_size(), 27);
}

// ---- extras pinning the exact special-token ids and vocab order ----

#[test]
fn test_special_token_ids_are_0_through_3() {
    let tok = MathTokenizer::new();
    assert_eq!(tok.pad_id, 0);
    assert_eq!(tok.bos_id, 1);
    assert_eq!(tok.eos_id, 2);
    assert_eq!(tok.unk_id, 3);
    assert_eq!(tok.token_to_id["**"], 18); // 4 special + 10 digits + index 4 of OPERATORS
    assert_eq!(tok.token_to_id["x"], 24); // 4 + 10 + 10
}
