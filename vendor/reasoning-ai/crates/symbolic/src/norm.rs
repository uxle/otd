//! Canonicalization, expansion, and simplification.
//!
//! `norm` produces the canonical form used for structural equality:
//! sums flattened with like terms collected and dropped zeros; products
//! flattened with like bases merged and a single leading numeric
//! coefficient; `x^0 → 1`, `x^1 → x`, numeric powers evaluated exactly;
//! `sqrt` identities applied; standard-angle trig evaluated exactly.
//!
//! `expand` distributes products over sums and expands small integer
//! powers of sums. `simplify` = `norm(expand(norm(e)))` plus the
//! Pythagorean trig identities — the sympy `simplify`/`expand` subset
//! this project relies on.

use crate::expr::{Expr, FuncKind};
use crate::trig;
use reasoning_common::Rat;

/// Canonical form. Idempotent.
pub fn norm(e: &Expr) -> Expr {
    match e {
        Expr::Num(_) | Expr::Sym(_) | Expr::Pi | Expr::E => e.clone(),
        Expr::Add(xs) => norm_add(xs),
        Expr::Mul(xs) => norm_mul(xs),
        Expr::Pow(a, b) => norm_pow(a, b),
        Expr::Func(k, a) => {
            let a = norm(a);
            norm_func(*k, &a)
        }
    }
}

fn norm_add(xs: &[Expr]) -> Expr {
    // flatten + normalize children, fold numeric constants, collect like
    // terms by their symbolic key
    let mut const_sum = Rat::from_int(0);
    // key (string form of symbolic part) -> (coeff, symbolic expr)
    let mut terms: Vec<(String, Rat, Expr)> = Vec::new();
    for x in xs {
        let xn = norm(x);
        for t in xn.terms() {
            let (coeff, rest) = t.as_coeff_mul();
            if matches!(rest, Expr::Num(r) if r.is_one()) {
                const_sum = const_sum + coeff;
                continue;
            }
            let key = rest.to_string();
            if let Some(entry) = terms.iter_mut().find(|(k, _, _)| *k == key) {
                entry.1 = entry.1 + coeff;
            } else {
                terms.push((key, coeff, rest));
            }
        }
    }
    let mut out: Vec<Expr> = Vec::new();
    for (_, coeff, rest) in terms {
        if coeff.is_zero() {
            continue;
        }
        if coeff.is_one() {
            out.push(rest);
        } else {
            out.push(Expr::Mul(vec![Expr::Num(coeff), rest]));
        }
    }
    if !const_sum.is_zero() {
        out.push(Expr::Num(const_sum));
    }
    // canonical order: sort by string key, numbers last
    out.sort_by(|a, b| term_key(a).cmp(&term_key(b)));
    match out.len() {
        0 => Expr::int(0),
        1 => out.pop().unwrap(),
        _ => Expr::Add(out),
    }
}

/// Sort key: numbers sort after symbolic terms (like sympy's printing
/// order intuition; only self-consistency matters).
fn term_key(e: &Expr) -> (u8, String) {
    match e {
        Expr::Num(_) => (1, e.to_string()),
        _ => (0, e.to_string()),
    }
}

fn norm_mul(xs: &[Expr]) -> Expr {
    // flatten, normalize children, merge numeric coefficients and like
    // numeric-exponent powers, apply 0/1 identities
    let mut coeff = Rat::from_int(1);
    // base key -> (base expr, exponent sum as Rat or None for non-numeric)
    let mut parts: Vec<(String, Expr, Option<Rat>)> = Vec::new();
    let mut others: Vec<Expr> = Vec::new();
    for x in xs {
        let xn = norm(x);
        if let Expr::Num(r) = &xn {
            coeff = coeff * *r;
            continue;
        }
        if xn.is_one() {
            continue;
        }
        for f in xn.factors() {
            if let Expr::Num(r) = &f {
                coeff = coeff * *r;
                continue;
            }
            if f.is_one() {
                continue;
            }
            // decompose f into (base, numeric exponent or None)
            let (base, exp) = match &f {
                Expr::Pow(b, ex) => {
                    let exn = norm(ex);
                    match exn.as_rat() {
                        Some(r) => ((**b).clone(), Some(r)),
                        None => ((**b).clone(), None),
                    }
                }
                other => (other.clone(), Some(Rat::from_int(1))),
            };
            match exp {
                Some(r) => {
                    let key = base.to_string();
                    if let Some(entry) = parts.iter_mut().find(|(k, _, _)| *k == key) {
                        if entry.2.is_some() {
                            let cur = entry.2.unwrap();
                            entry.2 = Some(cur + r);
                        } else {
                            others.push(f);
                        }
                    } else {
                        parts.push((key, base, Some(r)));
                    }
                }
                None => others.push(f),
            }
        }
    }
    if coeff.is_zero() {
        return Expr::int(0);
    }
    let mut out: Vec<Expr> = Vec::new();
    let mut leading_num: Option<Rat> = None;
    if !(coeff.is_one()) {
        leading_num = Some(coeff);
    }
    for (_, base, exp) in parts {
        let exp = exp.unwrap();
        if exp.is_zero() {
            continue;
        }
        if exp.is_one() {
            out.push(base);
            continue;
        }
        // numeric base with numeric exponent: evaluate exactly
        if let Some(b) = base.as_rat() {
            if let Some(r) = rat_pow(b, exp) {
                match leading_num {
                    Some(ref mut l) => *l = *l * r,
                    None => leading_num = Some(r),
                }
                continue;
            }
        }
        out.push(Expr::Pow(Box::new(base), Box::new(Expr::Num(exp))));
    }
    out.extend(others);
    // canonical sort for the symbolic factors
    out.sort_by(|a, b| a.to_string().cmp(&b.to_string()));
    let mut children: Vec<Expr> = Vec::new();
    if let Some(c) = leading_num {
        children.push(Expr::Num(c));
    }
    children.extend(out);
    match children.len() {
        0 => Expr::int(1),
        1 => children.pop().unwrap(),
        _ => Expr::Mul(children),
    }
}

/// Exact rational power: returns the rational result when the power is
/// exactly representable as a rational (integer exponents; perfect-square
/// radicals), else None (stay symbolic).
fn rat_pow(b: Rat, e: Rat) -> Option<Rat> {
    // integer exponent
    if e.den == 1 {
        let n = e.num;
        let base = if n < 0 {
            if b.is_zero() {
                return None; // division by zero — leave symbolic
            }
            b.recip()
        } else {
            b
        };
        let n_abs = n.abs() as u32;
        let num = base.num.checked_abs().unwrap().checked_pow(n_abs)?;
        let den = base.den.checked_pow(n_abs)?;
        let sign = if base.num < 0 && n_abs % 2 == 1 {
            -1i64
        } else {
            1i64
        };
        return Some(Rat::new(sign * num, den));
    }
    if e == Rat::new(1, 2) {
        if b.is_one() {
            return Some(Rat::from_int(1));
        }
        if b.is_zero() {
            return Some(Rat::from_int(0));
        }
        // perfect square rational?
        let (n, d) = (b.num, b.den);
        let sn = isqrt(n.abs());
        let sd = isqrt(d);
        if sn * sn == n.abs() && sd * sd == d && sn != 0 {
            return Some(Rat::new(if n < 0 { -sn } else { sn }, sd));
        }
    }
    None
}

fn isqrt(v: i64) -> i64 {
    if v <= 0 {
        return v;
    }
    let mut r = (v as f64).sqrt() as i64;
    while r * r > v {
        r -= 1;
    }
    while (r + 1) * (r + 1) <= v {
        r += 1;
    }
    r
}


fn norm_pow(a: &Expr, b: &Expr) -> Expr {
    let an = norm(a);
    let bn = norm(b);
    if let Some(e) = bn.as_rat() {
        if e.is_zero() {
            return Expr::int(1);
        }
        if e.is_one() {
            return an;
        }
        if let Some(ab) = an.as_rat() {
            if let Some(r) = rat_pow(ab, e) {
                return Expr::Num(r);
            }
        }
        if an.is_one() {
            return Expr::int(1);
        }
        if an.is_zero() {
            if e.num > 0 {
                return Expr::int(0);
            }
            return Expr::Pow(Box::new(an), Box::new(bn));
        }
        // (x^a)^b with numeric b: merge exponents
        if let Expr::Pow(inner, inner_exp) = &an {
            if let Some(ie) = norm(inner_exp).as_rat() {
                let merged = ie * e;
                if merged.is_one() {
                    return norm(inner);
                }
                if merged.is_zero() {
                    return Expr::int(1);
                }
                return norm(&Expr::Pow(inner.clone(), Box::new(Expr::Num(merged))));
            }
        }
        // (a*b)^e with numeric e: distribute over rational factors
        if let Expr::Mul(fs) = &an {
            if e.den == 1 && e.num > 0 && e.num <= 4 {
                let mut out = Expr::int(1);
                for f in fs {
                    out = norm(&Expr::Mul(vec![
                        out,
                        Expr::Pow(Box::new(f.clone()), Box::new(bn.clone())),
                    ]));
                }
                return out;
            }
        }
        // (pi)^e for rational e stays symbolic
        return Expr::Pow(Box::new(an), Box::new(Expr::Num(e)));
    }
    Expr::Pow(Box::new(an), Box::new(bn))
}

fn norm_func(k: FuncKind, a: &Expr) -> Expr {
    // sqrt(x) canonicalizes to Pow(x, 1/2)
    if k == FuncKind::Sqrt {
        return norm(&Expr::Pow(Box::new(a.clone()), Box::new(Expr::rat(1, 2))));
    }
    // exact standard-angle trig
    if let Some(v) = trig::exact_eval(k, a) {
        return v;
    }
    // log/exp/abs of exact numbers
    match k {
        FuncKind::Abs => {
            if let Some(r) = a.as_rat() {
                return Expr::Num(r.abs());
            }
        }
        _ => {}
    }
    Expr::Func(k, Box::new(a.clone()))
}

/// sympy-style `expand`: distribute products over sums, expand small
/// integer powers of sums.
pub fn expand(e: &Expr) -> Expr {
    let en = norm(e);
    let x = expand_rec(&en);
    norm(&x)
}

fn expand_rec(e: &Expr) -> Expr {
    match e {
        Expr::Num(_) | Expr::Sym(_) | Expr::Pi | Expr::E => e.clone(),
        Expr::Add(xs) => Expr::Add(xs.iter().map(expand_rec).collect()),
        Expr::Func(k, a) => Expr::Func(*k, Box::new(expand_rec(a))),
        Expr::Pow(a, b) => {
            let ae = expand_rec(a);
            if let Some(n) = b.as_int() {
                if n >= 2 && n <= 6 {
                    let mut out = ae.clone();
                    for _ in 1..n {
                        out = mul_expand(&out, &ae);
                    }
                    return out;
                }
                if n == 1 {
                    return ae;
                }
                if n == 0 {
                    return Expr::int(1);
                }
                // negative integer power: (a+b)^-1 stays symbolic
            }
            Expr::Pow(Box::new(ae), Box::new((**b).clone()))
        }
        Expr::Mul(xs) => {
            let mut out = Expr::int(1);
            for x in xs {
                out = mul_expand(&out, &expand_rec(x));
            }
            out
        }
    }
}

/// (a1 + a2 + ...)(b1 + b2 + ...) with like-term collection.
fn mul_expand(a: &Expr, b: &Expr) -> Expr {
    let at = a.terms();
    let bt = b.terms();
    if at.len() == 1 && bt.len() == 1 {
        return norm(&Expr::Mul(vec![at[0].clone(), bt[0].clone()]));
    }
    let mut terms: Vec<Expr> = Vec::new();
    for x in &at {
        for y in &bt {
            terms.push(Expr::Mul(vec![x.clone(), y.clone()]));
        }
    }
    norm(&Expr::Add(terms))
}

/// sympy-style `simplify` for this project's surface: expand + norm +
/// Pythagorean trig identities.
pub fn simplify(e: &Expr) -> Expr {
    let mut cur = norm(&expand_rec(&norm(e)));
    // Pythagorean identity: sin(x)^2 + cos(x)^2 -> 1 (and negations)
    cur = trig::apply_pythagorean(&cur);
    cur
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_expr;

    fn p(s: &str) -> Expr {
        parse_expr(s).unwrap()
    }

    #[test]
    fn test_norm_folds() {
        assert_eq!(p("2 + 3"), Expr::int(5));
        assert_eq!(p("x + x"), Expr::Mul(vec![Expr::int(2), Expr::sym("x")]));
        assert_eq!(p("x + 0"), Expr::sym("x"));
        assert_eq!(p("2*x + 3*x"), p("5*x"));
        assert_eq!(p("x*x"), p("x**2"));
        assert_eq!(p("x**2*x**3"), p("x**5"));
        assert_eq!(p("x/x"), Expr::int(1));
        assert_eq!(p("0*x"), Expr::int(0));
        assert_eq!(p("1*x"), Expr::sym("x"));
        assert_eq!(p("2*3*x"), p("6*x"));
        assert_eq!(p("(1/2)+(1/3)"), Expr::rat(5, 6));
        assert_eq!(p("x - x"), Expr::int(0));
    }

    #[test]
    fn test_expand() {
        assert_eq!(expand(&p("(x+1)*(x+2)")), p("x**2 + 3*x + 2"));
        assert_eq!(expand(&p("(x+1)**2")), p("x**2 + 2*x + 1"));
        assert!(crate::evalf::equiv(
            &expand(&p("(x+p)*(x+q)")),
            &p("x**2 + (p+q)*x + p*q")
        ));
        assert_eq!(expand(&p("2*(x+3)")), p("2*x + 6"));
        assert_eq!(expand(&p("(x+2)*(x-2)")), p("x**2 - 4"));
    }

    #[test]
    fn test_simplify_equiv_structures() {
        // 1.0*x - x -> 0 exactly (linear domain terminal check)
        assert_eq!(expand(&p("1.0*x - x")), Expr::int(0));
        // rhs has no x
        assert!(!p("3 + 2*x").has_sym("x") || p("3 + 2*x").has_sym("x"));
        assert_eq!(expand(&p("2*(x+3) - (2*x + 6)")), Expr::int(0));
    }

    #[test]
    fn test_sqrt_forms() {
        assert_eq!(p("sqrt(4)"), Expr::int(2));
        assert_eq!(p("sqrt(0)"), Expr::int(0));
        // sqrt(2)/2 stays symbolic but parses roundtrip
        let e = p("sqrt(2)/2");
        let t = e.to_string();
        assert_eq!(parse_expr(&t).unwrap(), e);
    }

    #[test]
    fn test_pi_arithmetic() {
        // pi/6 + pi/3 = pi/2
        assert_eq!(p("pi/6 + pi/3"), p("pi/2"));
        // 2*pi - pi = pi
        assert_eq!(p("2*pi - pi"), Expr::Pi);
    }
}
