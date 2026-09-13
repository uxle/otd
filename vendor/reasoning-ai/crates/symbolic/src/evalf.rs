//! Numeric evaluation + zero-equivalence + algebraic equivalence.
//!
//! `is_zero` mirrors `simplify(a - b) == 0`: structural zero first, then a
//! numeric cross-check — constants are evaluated directly; expressions
//! with free symbols are sampled at several well-separated points and must
//! vanish at every one. This reproduces sympy's verdicts for the
//! well-separated candidate sets this project tests (off-by-constant,
//! missing chain-rule term, etc.).

use crate::expr::{Expr, FuncKind};
use std::collections::HashMap;

/// Numeric evaluation with f64. Free symbols come from `env`; a missing
/// symbol is an error (like sympy failing to `N()` an expression).
pub fn eval_f64(e: &Expr, env: &HashMap<String, f64>) -> Result<f64, String> {
    match e {
        Expr::Num(r) => Ok(r.to_f64()),
        Expr::Sym(s) => env
            .get(s)
            .copied()
            .ok_or_else(|| format!("free symbol '{}' has no value", s)),
        Expr::Pi => Ok(std::f64::consts::PI),
        Expr::E => Ok(std::f64::consts::E),
        Expr::Add(xs) => {
            let mut acc = 0.0;
            for x in xs {
                acc += eval_f64(x, env)?;
            }
            Ok(acc)
        }
        Expr::Mul(xs) => {
            let mut acc = 1.0;
            for x in xs {
                acc *= eval_f64(x, env)?;
            }
            Ok(acc)
        }
        Expr::Pow(a, b) => {
            let base = eval_f64(a, env)?;
            let exp = eval_f64(b, env)?;
            Ok(base.powf(exp))
        }
        Expr::Func(k, a) => {
            let v = eval_f64(a, env)?;
            Ok(match k {
                FuncKind::Sin => v.sin(),
                FuncKind::Cos => v.cos(),
                FuncKind::Tan => v.tan(),
                FuncKind::Log => {
                    if v <= 0.0 {
                        return Err(format!("log of non-positive {}", v));
                    }
                    v.ln()
                }
                FuncKind::Exp => v.exp(),
                FuncKind::Abs => v.abs(),
                FuncKind::Sqrt => {
                    if v < 0.0 {
                        return Err(format!("sqrt of negative {}", v));
                    }
                    v.sqrt()
                }
            })
        }
    }
}

/// Is this expression exactly zero? Structural check first, then numeric
/// fallback (constants evaluated; symbolic expressions sampled).
pub fn is_zero(e: &Expr) -> bool {
    let en = crate::norm::norm(e);
    if en.is_zero() {
        return true;
    }
    // numeric fallback
    let syms = en.free_symbols();
    if syms.is_empty() {
        match eval_f64(&en, &HashMap::new()) {
            Ok(v) => v.abs() <= 1e-9,
            Err(_) => false,
        }
    } else {
        // sample several points; must vanish at ALL of them
        for pt in SAMPLE_POINTS {
            let env: HashMap<String, f64> = syms
                .iter()
                .enumerate()
                .map(|(i, s)| (s.clone(), pt + i as f64 * 0.37))
                .collect();
            match eval_f64(&en, &env) {
                Ok(v) => {
                    if v.abs() > 1e-7 {
                        return false;
                    }
                }
                Err(_) => return false,
            }
        }
        true
    }
}

const SAMPLE_POINTS: [f64; 5] = [0.7311, 1.2937, 2.1179, 3.4043, 5.2123];

/// Are two expressions equivalent (simplify(a - b) == 0)?
pub fn equiv(a: &Expr, b: &Expr) -> bool {
    let an = crate::norm::norm(a);
    let bn = crate::norm::norm(b);
    if an == bn {
        return true;
    }
    let diff = an - bn;
    is_zero(&diff)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_expr;

    fn p(s: &str) -> Expr {
        parse_expr(s).unwrap()
    }

    #[test]
    fn test_eval() {
        let env: HashMap<String, f64> = [("x".to_string(), 3.0)].into();
        assert_eq!(eval_f64(&p("2*x + 1"), &env).unwrap(), 7.0);
        assert_eq!(eval_f64(&p("x**2"), &env).unwrap(), 9.0);
        assert_eq!(eval_f64(&p("1/2*x"), &env).unwrap(), 1.5);
        assert_eq!(eval_f64(&p("2*pi"), &HashMap::new()).unwrap(), 2.0 * std::f64::consts::PI);
        assert_eq!(eval_f64(&p("sqrt(9)"), &HashMap::new()).unwrap(), 3.0);
    }

    #[test]
    fn test_equiv_structural() {
        assert!(equiv(&p("(x+1)*(x+2)"), &p("x**2 + 3*x + 2")));
        assert!(equiv(&p("x**2 + 2*x + 1"), &p("(x+1)**2")));
        assert!(!equiv(&p("x**2 + 2*x + 1"), &p("x**2 + 2*x + 2")));
    }

    #[test]
    fn test_equiv_trig_numeric_fallback() {
        // sin^2 + cos^2 == 1 (needs the numeric fallback)
        assert!(equiv(&p("sin(x)**2 + cos(x)**2"), &p("1")));
        // 1 - cos^2 == sin^2
        assert!(equiv(&p("1 - cos(x)**2"), &p("sin(x)**2")));
        // sqrt(3)/3 == 1/sqrt(3)
        assert!(equiv(&p("sqrt(3)/3"), &p("1/sqrt(3)")));
        // tan = sin/cos
        assert!(equiv(&p("tan(x)"), &p("sin(x)/cos(x)")));
        // wrong chain rule: d/dx x*cos(x) is cos(x) - x*sin(x), NOT cos(x)
        assert!(!equiv(&p("cos(x)"), &p("cos(x) - x*sin(x)")));
        // constant equality
        assert!(equiv(&p("sin(pi/6)"), &p("1/2")));
    }

    #[test]
    fn test_equiv_two_variables() {
        assert!(equiv(&p("x + y"), &p("y + x")));
        assert!(!equiv(&p("x + y"), &p("x + 2*y")));
    }
}
