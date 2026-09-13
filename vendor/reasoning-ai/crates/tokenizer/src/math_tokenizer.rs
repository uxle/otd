//! Phase 016 — Math tokenizer (Rust port of python/tokenizer/math_tokenizer.py)
//!
//! Digit-level tokenization for numbers (each digit its own token) rather
//! than whole-number tokens — standard practice in math-reasoning
//! literature because it lets a small model generalize arithmetic across
//! magnitudes it hasn't literally memorized. Everything else (operators,
//! parens, variables) is atomic.
//!
//! The Python scanned with the regex `\*\*|[+\-*/(),=]|\d|[a-z]|\.` via
//! `findall` with an <UNK> fallback; this port hand-rolls the identical
//! scan (match `**` first, then the single-character classes) so no
//! unrecognized input is ever silently dropped.

use std::collections::HashMap;

/// Python `SPECIAL_TOKENS`.
pub const SPECIAL_TOKENS: [&str; 4] = ["<PAD>", "<BOS>", "<EOS>", "<UNK>"];
/// Python `OPERATORS` (order matters — it fixes the vocab ids).
pub const OPERATORS: [&str; 10] = ["+", "-", "*", "/", "**", "(", ")", "=", ",", "."];
/// Python `VARIABLES`.
pub const VARIABLES: [&str; 3] = ["x", "y", "z"];
/// Python `DIGITS`.
pub const DIGITS: [&str; 10] = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"];

/// Python `VOCAB_TOKENS = SPECIAL_TOKENS + DIGITS + OPERATORS + VARIABLES`
/// (4 + 10 + 10 + 3 = 27 tokens).
pub fn vocab_tokens() -> Vec<&'static str> {
    let mut v: Vec<&'static str> = Vec::with_capacity(27);
    v.extend_from_slice(&SPECIAL_TOKENS);
    v.extend_from_slice(&DIGITS);
    v.extend_from_slice(&OPERATORS);
    v.extend_from_slice(&VARIABLES);
    v
}

/// Is `c` one of the single-character alternatives of
/// `[+\-*/(),=]|\d|[a-z]|\.`?
fn is_single_token_char(c: char) -> bool {
    matches!(c, '+' | '-' | '*' | '/' | '(' | ')' | ',' | '=' | '.')
        || c.is_ascii_digit()
        || c.is_ascii_lowercase()
}

/// Does a token match at `chars[i]`? Returns the matched token and how many
/// chars it consumed (mirrors `_TOKEN_PATTERN.match(text, i)`).
fn match_at(chars: &[char], i: usize) -> Option<(String, usize)> {
    // alternation order: '**' first, then single chars
    if chars[i] == '*' && i + 1 < chars.len() && chars[i + 1] == '*' {
        return Some(("**".to_string(), 2));
    }
    if is_single_token_char(chars[i]) {
        return Some((chars[i].to_string(), 1));
    }
    None
}

pub struct MathTokenizer {
    pub token_to_id: HashMap<String, usize>,
    pub id_to_token: HashMap<usize, String>,
    pub pad_id: usize,
    pub bos_id: usize,
    pub eos_id: usize,
    pub unk_id: usize,
}

impl MathTokenizer {
    pub fn new() -> Self {
        let tokens = vocab_tokens();
        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut id_to_token = HashMap::with_capacity(tokens.len());
        for (i, tok) in tokens.into_iter().enumerate() {
            token_to_id.insert(tok.to_string(), i);
            id_to_token.insert(i, tok.to_string());
        }
        let pad_id = token_to_id["<PAD>"];
        let bos_id = token_to_id["<BOS>"];
        let eos_id = token_to_id["<EOS>"];
        let unk_id = token_to_id["<UNK>"];
        MathTokenizer {
            token_to_id,
            id_to_token,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.token_to_id.len()
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        let text: String = text.replace(' ', "");
        let chars: Vec<char> = text.chars().collect();
        // fast path: Python findall — collect every match, skipping anything
        // unrecognized (the sanity check below catches that case)
        let mut tokens = Vec::new();
        let mut i = 0usize;
        let mut consumed = String::new();
        while i < chars.len() {
            if let Some((tok, adv)) = match_at(&chars, i) {
                tokens.push(tok.clone());
                consumed.push_str(&tok);
                i += adv;
            } else {
                i += 1;
            }
        }
        // sanity: reconstructed length should match input length (the module
        // doesn't silently drop unrecognized characters into nothing)
        if consumed != text {
            // fall back: whatever wasn't matched becomes explicit <UNK> markers
            // rather than being silently dropped
            tokens = self.tokenize_with_unk(&chars);
        }
        tokens
    }

    /// Python `_tokenize_with_unk`: walk char-by-char, matching at the exact
    /// position; a non-match emits `<UNK>` and advances one char.
    fn tokenize_with_unk(&self, chars: &[char]) -> Vec<String> {
        let mut tokens = Vec::new();
        let mut i = 0usize;
        while i < chars.len() {
            if let Some((tok, adv)) = match_at(chars, i) {
                tokens.push(tok);
                i += adv;
            } else {
                tokens.push("<UNK>".to_string());
                i += 1;
            }
        }
        tokens
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Vec<usize> {
        let tokens = self.tokenize(text);
        let mut ids: Vec<usize> = tokens
            .iter()
            .map(|t| self.token_to_id.get(t).copied().unwrap_or(self.unk_id))
            .collect();
        if add_special {
            let mut with_special = Vec::with_capacity(ids.len() + 2);
            with_special.push(self.bos_id);
            with_special.append(&mut ids);
            with_special.push(self.eos_id);
            return with_special;
        }
        ids
    }

    pub fn decode(&self, ids: &[usize], strip_special: bool) -> String {
        let mut tokens = Vec::new();
        for &i in ids {
            let tok = self.id_to_token.get(&i).map(|s| s.as_str()).unwrap_or("<UNK>");
            if strip_special && SPECIAL_TOKENS.contains(&tok) {
                continue;
            }
            tokens.push(tok);
        }
        tokens.join("")
    }

    pub fn pad(&self, ids: Vec<usize>, length: usize) -> Vec<usize> {
        if ids.len() >= length {
            return ids[..length].to_vec();
        }
        let mut out = ids;
        out.resize(length, self.pad_id);
        out
    }
}

impl Default for MathTokenizer {
    fn default() -> Self {
        MathTokenizer::new()
    }
}
