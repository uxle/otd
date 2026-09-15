//! P0100 — the lexer. Line-based statements (no semicolons required), `#` comments
//! (+ `#[ … ]#` block comments since 2.1), units glued to numbers (`40mm`,
//! `3.5cm`, `90deg`, `90°`, `1.5e3`), scientific notation, and the full
//! 2.1 operator set: `+ - & * / % ^ !` and comparisons `< > <= >= == !=`, 
//! logic `&& ||`, ranges `..`, compound assignment `+= -= *= /=`.
//! A line continues onto the next when it ends with `,` or an unclosed bracket;
//! `;` also separates statements (2.1).

use crate::units::unit_factor;

#[derive(Clone, Debug, PartialEq)]
pub enum Tok {
    Nl,                          // statement separator (newline or ;)
    Ident(String),
    Num(f64, Option<String>),    // value + attached unit (validated later)
    Str(String),
    P(&'static str),             // = + - & * / ( ) [ ] , : ° and the 2.1 operators:
                                 // ^ % ! < > <= >= == != && || .. += -= *= /=
    Eof,
}

#[derive(Clone, Debug)]
pub struct Token {
    pub tok: Tok,
    pub line: usize,
}

pub struct LexOut {
    pub toks: Vec<Token>,
    pub errors: Vec<crate::lang::errors::Error>,
}

/// Paste-proofing: characters that look like OTD punctuation but come from
/// word processors / web pages are normalized to their ASCII twins.
fn normalize(c: char) -> char {
    match c {
        '\u{2010}' | '\u{2011}' | '\u{2012}' | '\u{2013}' | '\u{2014}' | '\u{2212}' => '-', // dashes / minus
        '\u{201C}' | '\u{201D}' => '"', // curly double quotes
        '\u{FEFF}' => ' ',             // zero-width BOM mid-file
        _ => c,
    }
}

pub fn lex(src: &str) -> LexOut {
    let mut toks: Vec<Token> = Vec::new();
    let mut errors: Vec<crate::lang::errors::Error> = Vec::new();
    let chars: Vec<char> = src.chars().collect();
    let n = chars.len();
    let mut i = 0usize;
    // strip a leading UTF-8 BOM (files saved on Windows carry one)
    if chars.first() == Some(&'\u{FEFF}') {
        i = 1;
    }
    let mut line = 1usize;
    let mut depth = 0i32; // ( [ nesting
    let mut open_at: Option<(usize, char)> = None; // first unclosed bracket

    while i < n {
        let c = normalize(chars[i]);
        // whitespace
        if c == ' ' || c == '\t' || c == '\r' {
            i += 1;
            continue;
        }
        // newline → statement separator (unless continuing)
        if c == '\n' {
            line += 1;
            i += 1;
            if depth > 0 {
                continue; // inside brackets: pure whitespace
            }
            if let Some(last) = toks.last() {
                if matches!(last.tok, Tok::P(",")) {
                    continue; // trailing comma: continue statement
                }
            }
            if let Some(last) = toks.last() {
                if matches!(last.tok, Tok::Nl) {
                    continue; // collapse blank lines
                }
            }
            if toks.is_empty() {
                continue; // leading blank lines
            }
            toks.push(Token { tok: Tok::Nl, line: line - 1 });
            continue;
        }
        // `#[ … ]#` block comment, or `#rrggbb` color, or a line comment
        if c == '#' {
            if i + 1 < n && chars[i + 1] == '[' {
                // block comment — runs until ]#
                let start_line = line;
                let mut j = i + 2;
                let mut closed = false;
                while j < n {
                    if chars[j] == '\n' { line += 1; }
                    if chars[j] == ']' && j + 1 < n && chars[j + 1] == '#' {
                        closed = true;
                        j += 2;
                        break;
                    }
                    j += 1;
                }
                if !closed {
                    errors.push(crate::lang::errors::Error::new(
                        start_line,
                        "a block comment #[ … ]# never closed",
                    ).with_hint("add the closing ]#"));
                }
                i = j;
                continue;
            }
            // exactly 6 hex digits followed by a non-identifier char → color word
            let is_color = i + 7 <= n && {
                let six: Vec<char> = chars[i + 1..i + 7].to_vec();
                let after = if i + 7 < n { normalize(chars[i + 7]) } else { '\n' };
                six.iter().all(|h| h.is_ascii_hexdigit())
                    && !(after.is_ascii_alphanumeric() || after == '_')
            };
            if is_color {
                let word: String = chars[i..i + 7].iter().collect();
                toks.push(Token { tok: Tok::Ident(word), line });
                i += 7;
                continue;
            }
            while i < n && chars[i] != '\n' {
                i += 1;
            }
            continue;
        }
        // string
        if c == '"' {
            let start_line = line;
            i += 1;
            let mut s = String::new();
            let mut closed = false;
            while i < n {
                let ch = normalize(chars[i]);
                if ch == '"' {
                    closed = true;
                    i += 1;
                    break;
                }
                if ch == '\\' && i + 1 < n {
                    i += 1;
                    match normalize(chars[i]) {
                        'n' => s.push('\n'),
                        't' => s.push('\t'),
                        '"' => s.push('"'),
                        '\\' => s.push('\\'),
                        other => {
                            s.push('\\');
                            s.push(other);
                        }
                    }
                    i += 1;
                    continue;
                }
                if ch == '\n' {
                    break; // unterminated string ends at newline
                }
                s.push(chars[i]);
                i += 1;
            }
            if !closed {
                errors.push(crate::lang::errors::Error::new(
                    start_line,
                    "a string started but never closed",
                ).with_hint("add the closing \" — strings live on one line"));
            }
            toks.push(Token { tok: Tok::Str(s), line: start_line });
            continue;
        }
        // number
        if c.is_ascii_digit() || (c == '.' && i + 1 < n && chars[i + 1].is_ascii_digit()) {
            let start_line = line;
            let mut s = String::new();
            while i < n && chars[i].is_ascii_digit() {
                s.push(chars[i]);
                i += 1;
            }
            if i < n && chars[i] == '.' && i + 1 < n && chars[i + 1].is_ascii_digit() {
                s.push('.');
                i += 1;
                while i < n && chars[i].is_ascii_digit() {
                    s.push(chars[i]);
                    i += 1;
                }
            }
            let mut v: f64 = s.parse().unwrap_or(0.0);
            // scientific notation (2.1): 1.5e3, 2E-4, 3e2
            if i < n && (chars[i] == 'e' || chars[i] == 'E') {
                let mut j = i + 1;
                let mut exp = String::new();
                exp.push(chars[i]); // e or E
                if j < n && (chars[j] == '+' || chars[j] == '-') {
                    exp.push(chars[j]);
                    j += 1;
                }
                if j < n && chars[j].is_ascii_digit() {
                    while j < n && chars[j].is_ascii_digit() {
                        exp.push(chars[j]);
                        j += 1;
                    }
                    let full = format!("{}{}", s, exp);
                    v = full.parse().unwrap_or(v);
                    i = j;
                }
            }
            // attached unit?
            let mut unit: Option<String> = None;
            if i < n && chars[i] == '°' {
                unit = Some("deg".into());
                i += 1;
            } else if i < n && (chars[i].is_ascii_alphabetic()) {
                let start = i;
                while i < n && chars[i].is_ascii_alphabetic() {
                    i += 1;
                }
                let u: String = chars[start..i].iter().collect();
                // 'in' / 'ft' are units; longer words are NOT (avoid eating identifiers)
                if crate::lang::keywords::is_unit(&u) {
                    unit = Some(u);
                } else {
                    // not a known unit → push number and let the word lex as ident
                    i = start;
                }
            }
            // unit separated by a space on the SAME line (`5 cm`, `45 deg`):
            // glue it too — beginners write it this way constantly.
            if unit.is_none() {
                let mut j = i;
                while j < n && (chars[j] == ' ' || chars[j] == '\t') {
                    j += 1;
                }
                if j > i && j < n && chars[j].is_ascii_alphabetic() {
                    let start = j;
                    while j < n && chars[j].is_ascii_alphabetic() {
                        j += 1;
                    }
                    let u: String = chars[start..j].iter().collect();
                    let after = if j < n { normalize(chars[j]) } else { '\n' };
                    // `3 in [1, 2, 3]` is a membership test, not 3 inches (2.1):
                    // a bare `in` before a list, parens, or a number is the
                    // operator. Attached `3in` is always inches.
                    let mut k = j;
                    while k < n && (chars[k] == ' ' || chars[k] == '\t') {
                        k += 1;
                    }
                    let next_non_space = if k < n { normalize(chars[k]) } else { '\n' };
                    // `5 in xs` / `3 in [1, 2]` / `5 in 1..10` are membership
                    // tests, not inches: any value-like follower means operator.
                    let looks_like_in_operator = u == "in"
                        && (next_non_space == '[' || next_non_space == '('
                            || next_non_space.is_ascii_alphanumeric() || next_non_space == '_');
                    if crate::lang::keywords::is_unit(&u)
                        && !after.is_ascii_alphanumeric()
                        && after != '_'
                        && after != '('
                        && !looks_like_in_operator
                    {
                        unit = Some(u);
                        i = j;
                    }
                }
            }
            toks.push(Token { tok: Tok::Num(v, unit), line: start_line });
            continue;
        }
        // identifier
        if c.is_ascii_alphabetic() || c == '_' {
            let start_line = line;
            let start = i;
            while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
                i += 1;
            }
            let w: String = chars[start..i].iter().collect();
            toks.push(Token { tok: Tok::Ident(w), line: start_line });
            continue;
        }
        // standalone ° — attach degrees to the number just before it
        if c == '°' {
            let mut attached = false;
            if let Some(Token { tok: Tok::Num(_, ref mut unit), .. }) = toks.last_mut() {
                if unit.is_none() {
                    *unit = Some("deg".into());
                    attached = true;
                }
            }
            if !attached {
                errors.push(crate::lang::errors::Error::new(
                    line,
                    "° turns a number into degrees",
                ).with_hint("put it right after the number, like 90° or 90 deg"));
            }
            i += 1;
            continue;
        }
        // punctuation
        // two-character operators first (2.1)
        if i + 1 < n {
            let two: Option<&'static str> = match (c, chars[i + 1]) {
                ('=', '=') => Some("=="),
                ('!', '=') => Some("!="),
                ('<', '=') => Some("<="),
                ('>', '=') => Some(">="),
                ('&', '&') => Some("&&"),
                ('|', '|') => Some("||"),
                ('.', '.') => Some(".."),
                ('+', '=') => Some("+="),
                ('-', '=') => Some("-="),
                ('*', '=') => Some("*="),
                ('/', '=') => Some("/="),
                _ => None,
            };
            if let Some(op) = two {
                toks.push(Token { tok: Tok::P(op), line });
                i += 2;
                continue;
            }
        }
        // `;` separates statements like a newline (2.1)
        if c == ';' {
            if depth > 0 {
                errors.push(crate::lang::errors::Error::new(
                    line,
                    "; separates statements — it can't live inside ( ) or [ ]",
                ).with_hint("remove it, or close the bracket first"));
            } else {
                toks.push(Token { tok: Tok::Nl, line });
            }
            i += 1;
            continue;
        }
        let p: Option<&'static str> = match c {
            '=' => Some("="),
            '+' => Some("+"),
            '-' => Some("-"),
            '&' => Some("&"),
            '*' => Some("*"),
            '/' => Some("/"),
            '^' => Some("^"),
            '%' => Some("%"),
            '!' => Some("!"),
            '<' => Some("<"),
            '>' => Some(">"),
            '(' => {
                if depth == 0 { open_at = Some((line, '(')); }
                depth += 1;
                Some("(")
            }
            ')' => {
                depth -= 1;
                if depth < 0 {
                    errors.push(crate::lang::errors::Error::new(
                        line,
                        "this ) doesn't match any (",
                    ).with_hint("remove it, or open a ( earlier"));
                    depth = 0;
                }
                if depth == 0 { open_at = None; }
                    Some(")")
            }
            '[' => {
                if depth == 0 { open_at = Some((line, '[')); }
                depth += 1;
                Some("[")
            }
            ']' => {
                depth -= 1;
                if depth < 0 {
                    errors.push(crate::lang::errors::Error::new(
                        line,
                        "this ] doesn't match any [",
                    ).with_hint("remove it, or open a [ earlier"));
                    depth = 0;
                }
                if depth == 0 { open_at = None; }
                Some("]")
            }
            ',' => Some(","),
            ':' => Some(":"),
            _ => None,
        };
        if let Some(p) = p {
            toks.push(Token { tok: Tok::P(p), line });
            i += 1;
            continue;
        }
        errors.push(crate::lang::errors::Error::new(
            line,
            format!("I don't understand the character '{}'", chars[i]),
        ).with_hint("OTD uses letters, numbers, = + - & * / % ^ ! < > and ( ) [ ] , : — plus #rrggbb colors"));
        i += 1;
    }
    // an unclosed bracket swallows every newline after it — say so plainly
    if depth > 0 {
        if let Some((l, ch)) = open_at {
            errors.push(crate::lang::errors::Error::new(
                l,
                format!("a {} opened here never closed", ch),
            ).with_hint("add the missing bracket — a line ending in , or an open bracket continues to the next line"));
        }
    }
    toks.push(Token { tok: Tok::Eof, line });
    LexOut { toks, errors }
}

/// Convert a raw (value, unit) pair into a Qty in canonical units (mm / deg).
/// `default_unit` applies to bare numbers (scene `unit:` statement, default cm).
pub fn qty_of(v: f64, unit: &Option<String>, line: usize, _default_unit: &str) -> Result<crate::units::Qty, crate::lang::errors::Error> {
    use crate::units::Qty;
    match unit {
        None => {
            // bare numbers stay RAW (Plain): counts like n: 4 keep 4, and
            // length contexts apply the scene default unit themselves.
            Ok(Qty::plain(v))
        }
        Some(u) => {
            match unit_factor(u) {
                Some((f, d)) => Ok(Qty { v: v * f, dim: d }),
                None => {
                    let s = crate::lang::errors::suggest(u, crate::lang::keywords::UNITS)
                        .map(|s| format!("did you mean {}?", s))
                        .unwrap_or_else(|| "units are mm, cm, m, in, ft, deg".into());
                    Err(crate::lang::errors::Error::new(
                        line,
                        format!("'{}' is not a unit I know", u),
                    ).with_hint(s))
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Tok> {
        lex(src).toks.into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn units_glue() {
        let t = kinds("40mm 3.5cm 2m 45deg 90° 1.5");
        assert_eq!(t[0], Tok::Num(40.0, Some("mm".into())));
        assert_eq!(t[1], Tok::Num(3.5, Some("cm".into())));
        assert_eq!(t[2], Tok::Num(2.0, Some("m".into())));
        assert_eq!(t[3], Tok::Num(45.0, Some("deg".into())));
        assert_eq!(t[4], Tok::Num(90.0, Some("deg".into())));
        assert_eq!(t[5], Tok::Num(1.5, None));
    }

    #[test]
    fn statements_newline() {
        let t = kinds("cube 2cm\nsphere 3cm\n");
        let nl = t.iter().filter(|t| **t == Tok::Nl).count();
        assert_eq!(nl, 2);
    }

    #[test]
    fn continuation_rules() {
        // trailing comma continues
        let t = kinds("add cup,\n handle\n");
        assert_eq!(t.iter().filter(|t| **t == Tok::Nl).count(), 1);
        // open bracket continues
        let t = kinds("revolve [(0,0),\n (5cm, 2cm)]\n");
        assert_eq!(t.iter().filter(|t| **t == Tok::Nl).count(), 1);
    }

    #[test]
    fn comments_skipped() {
        let t = kinds("# hello\nsphere 1cm\n");
        assert_eq!(t[0], Tok::Ident("sphere".into()));
        assert_eq!(t[1], Tok::Num(1.0, Some("cm".into())));
    }

    #[test]
    fn unknown_char_error() {
        let out = lex("sphere @ 1cm");
        assert!(!out.errors.is_empty());
    }

    #[test]
    fn operators_lex_21() {
        let t = kinds("a < b > c <= d >= e == f != g");
        assert_eq!(t[1], Tok::P("<"));
        assert_eq!(t[3], Tok::P(">"));
        assert_eq!(t[5], Tok::P("<="));
        assert_eq!(t[7], Tok::P(">="));
        assert_eq!(t[9], Tok::P("=="));
        assert_eq!(t[11], Tok::P("!="));
        let t = kinds("x && y || ! z");
        assert_eq!(t[1], Tok::P("&&"));
        assert_eq!(t[3], Tok::P("||"));
        assert_eq!(t[4], Tok::P("!"));
        let t = kinds("2 ^ 10 % 3");
        assert_eq!(t[1], Tok::P("^"));
        assert_eq!(t[3], Tok::P("%"));
    }

    #[test]
    fn two_char_ops_beat_single() {
        // = = is two assignments, == is comparison
        assert_eq!(kinds("a == b")[1], Tok::P("=="));
        assert_eq!(kinds("a = b")[1], Tok::P("="));
        // .. range vs . (never a lone dot in OTD anyway)
        assert_eq!(kinds("1..5")[1], Tok::P(".."));
        assert_eq!(kinds("1..5")[0], Tok::Num(1.0, None));
        assert_eq!(kinds("1..5")[2], Tok::Num(5.0, None));
    }

    #[test]
    fn compound_assign_lexes() {
        let t = kinds("x += 1");
        assert_eq!(t[1], Tok::P("+="));
        let t = kinds("x -= 2");
        assert_eq!(t[1], Tok::P("-="));
        let t = kinds("x *= 3");
        assert_eq!(t[1], Tok::P("*="));
        let t = kinds("x /= 4");
        assert_eq!(t[1], Tok::P("/="));
    }

    #[test]
    fn semicolon_separates() {
        let t = kinds("a = 1; b = 2");
        let nl = t.iter().filter(|t| **t == Tok::Nl).count();
        assert_eq!(nl, 1, "one ; → one statement break");
    }

    #[test]
    fn semicolon_inside_brackets_errors() {
        let out = lex("at (1, 2; 3)");
        assert!(out.errors.iter().any(|e| e.msg.contains(";")));
    }

    #[test]
    fn scientific_notation() {
        assert_eq!(kinds("1.5e3")[0], Tok::Num(1500.0, None));
        assert_eq!(kinds("2E-4")[0], Tok::Num(0.0002, None));
        assert_eq!(kinds("3e2mm")[0], Tok::Num(300.0, Some("mm".into())));
        // no exponent digits → stays a plain number + word (compat)
        let t = kinds("2e");
        assert_eq!(t[0], Tok::Num(2.0, None));
        assert_eq!(t[1], Tok::Ident("e".into()));
    }

    #[test]
    fn block_comments() {
        let t = kinds("#[ a whole\n multiline note ]# sphere 1cm");
        assert_eq!(t[0], Tok::Ident("sphere".into()));
        assert_eq!(t[1], Tok::Num(1.0, Some("cm".into())));
        // unterminated → friendly error, rest still lexes
        let out = lex("#[ never closed");
        assert!(out.errors.iter().any(|e| e.msg.contains("block comment")));
    }

    #[test]
    fn new_units_glue() {
        assert_eq!(kinds("2km 3yd 4um 1rad")[0], Tok::Num(2.0, Some("km".into())));
        assert_eq!(kinds("2km 3yd 4um 1rad")[1], Tok::Num(3.0, Some("yd".into())));
        assert_eq!(kinds("2km 3yd 4um 1rad")[2], Tok::Num(4.0, Some("um".into())));
        assert_eq!(kinds("2km 3yd 4um 1rad")[3], Tok::Num(1.0, Some("rad".into())));
    }
}
