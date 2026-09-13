//! Expression parser mirroring the project's sympy `parse_expr` usage:
//! `standard_transformations + implicit_multiplication_application +
//! convert_xor` with a restricted symbol table. Supports implicit
//! multiplication ("2x" -> 2*x, "2(x+1)" -> 2*(x+1)), `**` and `^` as
//! power (right-associative), function calls, and `pi`/`E` constants.
//!
//! Security parity with the Python verifier: attribute access
//! ("x.__class__") is rejected at parse time — the tokenizer only accepts
//! dots inside numeric literals, so "x.y" is a hard error with the same
//! message shape the Python layer produces.

use crate::expr::{Expr, FuncKind};
use reasoning_common::Rat;

#[derive(Debug, Clone, PartialEq)]
pub enum ParseError {
    /// Attribute access attempted — mirrors the Python
    /// `_reject_attribute_access` defense.
    AttributeAccess(String),
    /// Generic syntax/parse failure.
    Syntax(String),
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ParseError::AttributeAccess(msg) => write!(f, "{}", msg),
            ParseError::Syntax(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for ParseError {}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Num(Rat),
    Ident(String),
    Plus,
    Minus,
    Star,
    Slash,
    Pow,     // ** or ^
    LParen,
    RParen,
    Comma,
}

fn tokenize(input: &str) -> Result<Vec<Tok>, ParseError> {
    let chars: Vec<char> = input.chars().collect();
    let mut toks = Vec::new();
    let mut i = 0;
    let n = chars.len();
    while i < n {
        let c = chars[i];
        match c {
            ' ' | '\t' | '\n' | '\r' => {
                i += 1;
            }
            '+' => {
                toks.push(Tok::Plus);
                i += 1;
            }
            '-' => {
                toks.push(Tok::Minus);
                i += 1;
            }
            '*' => {
                if i + 1 < n && chars[i + 1] == '*' {
                    toks.push(Tok::Pow);
                    i += 2;
                } else {
                    toks.push(Tok::Star);
                    i += 1;
                }
            }
            '/' => {
                toks.push(Tok::Slash);
                i += 1;
            }
            '^' => {
                toks.push(Tok::Pow);
                i += 1;
            }
            '(' => {
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                toks.push(Tok::RParen);
                i += 1;
            }
            ',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            '0'..='9' => {
                // number: int / decimal / scientific
                let start = i;
                let mut saw_dot = false;
                while i < n && (chars[i].is_ascii_digit() || (chars[i] == '.' && !saw_dot)) {
                    if chars[i] == '.' {
                        // A dot must be followed by a digit to be part of a
                        // number; "x.5" or "1.x" is attribute-ish garbage.
                        if i + 1 < n && chars[i + 1].is_ascii_digit() {
                            saw_dot = true;
                            i += 1;
                        } else {
                            return Err(ParseError::AttributeAccess(format!(
                                "attribute access (.{}) is not permitted in math expressions",
                                trailing_ident(&chars, i + 1)
                            )));
                        }
                    } else {
                        i += 1;
                    }
                }
                // exponent part
                if i < n && (chars[i] == 'e' || chars[i] == 'E') {
                    let mut j = i + 1;
                    if j < n && (chars[j] == '+' || chars[j] == '-') {
                        j += 1;
                    }
                    if j < n && chars[j].is_ascii_digit() {
                        while j < n && chars[j].is_ascii_digit() {
                            j += 1;
                        }
                        i = j;
                    }
                    // else: 'e' is the constant E as a factor (implicit mult)
                }
                let text: String = chars[start..i].iter().collect();
                let rat = decimal_to_rat(&text)
                    .ok_or_else(|| ParseError::Syntax(format!("bad number literal '{}'", text)))?;
                toks.push(Tok::Num(rat));
            }
            _ if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < n && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let name: String = chars[start..i].iter().collect();
                if i < n && chars[i] == '.' {
                    return Err(ParseError::AttributeAccess(format!(
                        "attribute access (.{}) is not permitted in math expressions",
                        trailing_ident(&chars, i + 1)
                    )));
                }
                toks.push(Tok::Ident(name));
            }
            _ => {
                return Err(ParseError::Syntax(format!(
                    "unexpected character '{}'",
                    c
                )))
            }
        }
    }
    Ok(toks)
}

fn trailing_ident(chars: &[char], mut i: usize) -> String {
    let start = i;
    while i < chars.len() && chars[i].is_alphanumeric() {
        i += 1;
    }
    chars[start..i].iter().collect()
}

/// "1.5" -> 3/2, "2" -> 2, "1e-3" -> 1/1000 (exact for finite decimals).
fn decimal_to_rat(text: &str) -> Option<Rat> {
    let t = text.trim();
    if t.is_empty() {
        return None;
    }
    let (mantissa, exp): (&str, i32) = match t.find(['e', 'E']) {
        Some(pos) => {
            let e: i32 = t[pos + 1..].parse().ok()?;
            (&t[..pos], e)
        }
        None => (t, 0),
    };
    let (int_part, frac_part) = match mantissa.find('.') {
        Some(pos) => (&mantissa[..pos], &mantissa[pos + 1..]),
        None => (mantissa, ""),
    };
    if int_part.is_empty() && frac_part.is_empty() {
        return None;
    }
    let ip: i64 = if int_part.is_empty() {
        0
    } else {
        int_part.parse().ok()?
    };
    // num = all mantissa digits as one integer; value = num * 10^shift
    let mut num: i64 = ip;
    for c in frac_part.chars() {
        if !c.is_ascii_digit() {
            return None;
        }
        let d = (c as u8 - b'0') as i64;
        // guard overflow for absurdly long literals
        if num > i64::MAX / 10 {
            let v: f64 = text.parse().ok()?;
            return Some(Rat::from_f64(v));
        }
        num = num * 10 + d;
    }
    let shift = exp - frac_part.len() as i32;
    if shift >= 0 {
        let mut m = num;
        for _ in 0..shift {
            if m > i64::MAX / 10 {
                let v: f64 = text.parse().ok()?;
                return Some(Rat::from_f64(v));
            }
            m *= 10;
        }
        Some(Rat::from_int(m))
    } else {
        let den = 10i64.checked_pow((-shift) as u32).unwrap_or(i64::MAX);
        if den == i64::MAX {
            let v: f64 = text.parse().ok()?;
            return Some(Rat::from_f64(v));
        }
        Some(Rat::new(num, den))
    }
}

/// Parse a math expression. `local_symbols` lists variable names that
/// should be treated as single symbols rather than split into implicit
/// products (the Python code passes e.g. {"x": X}).
pub fn parse_expr_with_locals(input: &str, local_symbols: &[&str]) -> Result<Expr, ParseError> {
    let toks = tokenize(input)?;
    if toks.is_empty() {
        return Err(ParseError::Syntax("empty expression".to_string()));
    }
    let mut p = P {
        toks,
        pos: 0,
        locals: local_symbols,
    };
    let e = p.parse_sum()?;
    if p.pos != p.toks.len() {
        return Err(ParseError::Syntax(format!(
            "unexpected trailing tokens at position {}",
            p.pos
        )));
    }
    Ok(crate::norm::norm(&e))
}

/// Parse a math expression with the default symbol table (no extra locals).
pub fn parse_expr(input: &str) -> Result<Expr, ParseError> {
    parse_expr_with_locals(input, &[])
}

struct P<'a> {
    toks: Vec<Tok>,
    pos: usize,
    locals: &'a [&'a str],
}

impl<'a> P<'a> {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    /// sum := term (('+'|'-') term)*
    fn parse_sum(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_term()?;
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    lhs = Expr::Add(vec![lhs, rhs]);
                }
                Some(Tok::Minus) => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    lhs = Expr::Add(vec![lhs, Expr::Mul(vec![Expr::int(-1), rhs])]);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    /// term := unary (('*'|'/') unary | implicit-mult unary)*
    fn parse_term(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_unary()?;
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.pos += 1;
                    let rhs = self.parse_unary()?;
                    lhs = Expr::Mul(vec![lhs, rhs]);
                }
                Some(Tok::Slash) => {
                    self.pos += 1;
                    let rhs = self.parse_unary()?;
                    lhs = Expr::Mul(vec![lhs, Expr::Pow(Box::new(rhs), Box::new(Expr::int(-1)))]);
                }
                // implicit multiplication: next token starts a value
                Some(t) if starts_value(t) => {
                    let rhs = self.parse_unary()?;
                    lhs = Expr::Mul(vec![lhs, rhs]);
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    /// unary := ('-'|'+')* power
    fn parse_unary(&mut self) -> Result<Expr, ParseError> {
        if let Some(Tok::Minus) = self.peek() {
            self.pos += 1;
            let e = self.parse_unary()?;
            return Ok(Expr::Mul(vec![Expr::int(-1), e]));
        }
        if let Some(Tok::Plus) = self.peek() {
            self.pos += 1;
            return self.parse_unary();
        }
        self.parse_power()
    }

    /// power := atom ('**' unary)?   (right-assoc, exponent may be signed)
    fn parse_power(&mut self) -> Result<Expr, ParseError> {
        let base = self.parse_atom()?;
        if let Some(Tok::Pow) = self.peek() {
            self.pos += 1;
            let exp = self.parse_unary()?;
            return Ok(Expr::Pow(Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn parse_atom(&mut self) -> Result<Expr, ParseError> {
        match self.next() {
            Some(Tok::Num(r)) => Ok(Expr::Num(r)),
            Some(Tok::LParen) => {
                let e = self.parse_sum()?;
                match self.next() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err(ParseError::Syntax("expected ')'".to_string())),
                }
            }
            Some(Tok::Ident(name)) => {
                if let Some(func) = FuncKind::from_name(&name) {
                    // function call: sin(x), sin x (implicit), possibly
                    // multiple args for log(x, b) — we accept and take 1 arg.
                    if matches!(self.peek(), Some(Tok::LParen)) {
                        self.pos += 1;
                        let mut args = vec![self.parse_sum()?];
                        while matches!(self.peek(), Some(Tok::Comma)) {
                            self.pos += 1;
                            args.push(self.parse_sum()?);
                        }
                        match self.next() {
                            Some(Tok::RParen) => {}
                            _ => {
                                return Err(ParseError::Syntax(
                                    "expected ')' after function arguments".to_string(),
                                ))
                            }
                        }
                        if args.len() != 1 {
                            return Err(ParseError::Syntax(format!(
                                "{}() takes exactly 1 argument, got {}",
                                name,
                                args.len()
                            )));
                        }
                        return Ok(Expr::Func(func, Box::new(args.pop().unwrap())));
                    }
                    // implicit application: "sin x" or "sin 2*x" — take the
                    // next unary as the argument (sympy's
                    // implicit_multiplication_application does this)
                    if matches!(self.peek(), Some(t) if starts_value(t)) {
                        let arg = self.parse_unary()?;
                        return Ok(Expr::Func(func, Box::new(arg)));
                    }
                    // bare "sin" without argument is an error
                    return Err(ParseError::Syntax(format!(
                        "function '{}' used without an argument",
                        name
                    )));
                }
                if name == "pi" {
                    return Ok(Expr::Pi);
                }
                if name == "E" || name == "e" {
                    // 'e' as Euler's number only when it is not a local
                    // symbol — single-letter lowercase 'e' is more likely a
                    // variable in this project's domains; match sympy: 'E'
                    // is the constant, bare 'e' is a symbol. Treat 'E' as
                    // the constant.
                    if name == "E" {
                        return Ok(Expr::E);
                    }
                }
                if self.locals.contains(&name.as_str()) {
                    return Ok(Expr::Sym(name));
                }
                if name.len() > 1 {
                    // sympy's split_symbols: "xy" -> x*y (unknown
                    // multi-letter identifiers split into single symbols)
                    let chars: Vec<char> = name.chars().collect();
                    if chars.iter().all(|c| c.is_alphabetic()) {
                        let mut parts: Vec<Expr> = Vec::new();
                        for c in &chars {
                            parts.push(Expr::Sym(c.to_string()));
                        }
                        return Ok(Expr::Mul(parts));
                    }
                }
                Ok(Expr::Sym(name))
            }
            other => Err(ParseError::Syntax(format!(
                "unexpected token {:?}",
                other
            ))),
        }
    }
}

/// Does this token start a value (for implicit multiplication)?
fn starts_value(t: &Tok) -> bool {
    matches!(t, Tok::Num(_) | Tok::Ident(_) | Tok::LParen)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(s: &str) -> Expr {
        parse_expr(s).unwrap()
    }

    #[test]
    fn test_basic_parse() {
        assert_eq!(p("1+2"), Expr::int(3));
        assert_eq!(p("2*3+1"), Expr::int(7));
        assert_eq!(p("10/4"), Expr::rat(5, 2));
        assert_eq!(p("2**3"), Expr::int(8));
        assert_eq!(p("2^3"), Expr::int(8));
        assert_eq!(p("0.5"), Expr::rat(1, 2));
        assert_eq!(p("1e-3"), Expr::rat(1, 1000));
        assert_eq!(p("-3"), Expr::int(-3));
        assert_eq!(p("2**-2"), Expr::rat(1, 4));
    }

    #[test]
    fn test_implicit_mult() {
        assert_eq!(p("2x"), Expr::Mul(vec![Expr::int(2), Expr::sym("x")]));
        assert_eq!(p("2(x+1)"), p("2*(x+1)"));
        assert_eq!(p("(x+1)(x-1)"), p("(x+1)*(x-1)"));
        assert_eq!(p("xy"), p("x*y"));
        assert_eq!(p("3xy"), p("3*x*y"));
    }

    #[test]
    fn test_precedence() {
        assert_eq!(p("-x**2"), p("-(x**2)"));
        assert_eq!(p("2*x**2"), p("2*(x**2)"));
        // left-assoc division: x/y/z == x/(y*z) mathematically
        assert!(crate::evalf::equiv(&p("x/y/z"), &p("x/(y*z)")));
        assert_eq!(p("1/2*x"), p("(1/2)*x"));
    }

    #[test]
    fn test_functions() {
        assert_eq!(p("sin(x)"), Expr::Func(FuncKind::Sin, Box::new(Expr::sym("x"))));
        // sin pi evaluates exactly to 0
        assert_eq!(p("sin pi"), Expr::int(0));
        assert_eq!(p("sqrt(3)"), Expr::sqrt(Expr::int(3)));
        assert_eq!(p("pi/6"), p("pi*(1/6)"));
    }

    #[test]
    fn test_attribute_access_rejected() {
        assert!(matches!(
            parse_expr("x.__class__"),
            Err(ParseError::AttributeAccess(_))
        ));
        assert!(matches!(
            parse_expr("f.__globals__"),
            Err(ParseError::AttributeAccess(_))
        ));
    }

    #[test]
    fn test_display_roundtrip() {
        for s in [
            "2*x + 3",
            "x**2 - 5*x + 6",
            "sqrt(3)/2",
            "sin(x)**2 + cos(x)**2",
            "3/4",
            "pi/6",
            "-1/2*x + 7",
            "(x+1)*(x-1)",
            "2*pi*r",
            "1e-3",
        ] {
            let e = p(s);
            let text = e.to_string();
            let e2 = parse_expr(&text).unwrap();
            assert_eq!(e, e2, "roundtrip failed for {} -> {}", s, text);
        }
    }
}
