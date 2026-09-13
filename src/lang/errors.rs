//! P0140 — friendly errors with "did you mean?" suggestions (Levenshtein).

#[derive(Clone, Debug)]
pub struct Error {
    pub line: usize,
    pub msg: String,
    pub hint: Option<String>,
}

impl Error {
    pub fn new(line: usize, msg: impl Into<String>) -> Error {
        Error { line, msg: msg.into(), hint: None }
    }
    pub fn with_hint(mut self, hint: impl Into<String>) -> Error {
        self.hint = Some(hint.into());
        self
    }
    /// Render for the console / browser.
    pub fn render(&self) -> String {
        match &self.hint {
            Some(h) => format!("line {}: {} — {}", self.line, self.msg, h),
            None => format!("line {}: {}", self.line, self.msg),
        }
    }
}

/// Classic Levenshtein edit distance (case-insensitive inputs compared lowercased).
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.to_lowercase().chars().collect();
    let b: Vec<char> = b.to_lowercase().chars().collect();
    if a.is_empty() { return b.len(); }
    if b.is_empty() { return a.len(); }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Best suggestion from a candidate list, or None if nothing is close.
pub fn suggest(word: &str, candidates: &[&str]) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for c in candidates {
        let d = levenshtein(word, c);
        let max = (word.len() / 3).max(1);
        if d <= max {
            match best {
                Some((bd, _)) if bd <= d => {}
                _ => best = Some((d, c)),
            }
        }
    }
    best.map(|(_, c)| c.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("cermic", "ceramic"), 1);
        assert_eq!(levenshtein("sphre", "sphere"), 1);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }
    #[test]
    fn suggestions() {
        assert_eq!(suggest("cermic", &["ceramic", "iron", "oak"]).unwrap(), "ceramic");
        assert!(suggest("zzzzzz", &["ceramic", "iron"]).is_none());
    }
    #[test]
    fn friendly_negative_radius() {
        let e = Error::new(3, "radius cannot be negative (-5mm)").with_hint("did you mean 5mm?");
        assert!(e.render().contains("did you mean 5mm?"));
    }
}
