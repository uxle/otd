//! Symbolic differentiation (the sympy `diff` subset used by the calculus
//! domain): sum/product/quotient/power rules, chain rule for
//! sin/cos/tan/log/exp/abs, exact rational coefficients.

use crate::expr::{Expr, FuncKind};
use reasoning_common::Rat;

/// d/d(var of `e`.
pub fn diff(e: &Expr, var: &str) -> Expr {
    match e {
        Expr::Num(_) | Expr::Pi | Expr::E => Expr::int(0),
        Expr::Sym(s) => {
            if s == var {
                Expr::int(1)
            } else {
                Expr::int(0)
            }
        }
        Expr::Add(xs) => {
            let parts: Vec<Expr> = xs.iter().map(|x| diff(x, var)).collect();
            crate::norm::norm(&Expr::Add(parts))
        }
        Expr::Mul(xs) => {
            // product rule over all factors
            let mut terms: Vec<Expr> = Vec::new();
            for i in 0..xs.len() {
                let mut factors: Vec<Expr> = Vec::with_capacity(xs.len());
                for (j, x) in xs.iter().enumerate() {
                    factors.push(if i == j {
                        diff(x, var)
                    } else {
                        x.clone()
                    });
                }
                terms.push(Expr::Mul(factors));
            }
            crate::norm::norm(&Expr::Add(terms))
        }
        Expr::Pow(base, exp) => {
            // x^n (numeric n): n*x^(n-1) * dx
            if let Some(n) = exp.as_rat() {
                let d_base = diff(base, var);
                let new_exp = Expr::Num(n - Rat::from_int(1));
                let pow_part = Expr::Pow(Box::new((**base).clone()), Box::new(new_exp));
                return crate::norm::norm(&Expr::Mul(vec![
                    Expr::Num(n),
                    pow_part,
                    d_base,
                ]));
            }
            // a^f(x) (constant base): a^f * ln(a) * f'
            if let Some(_a) = base.as_rat() {
                let d_exp = diff(exp, var);
                return crate::norm::norm(&Expr::Mul(vec![
                    Expr::Pow(Box::new((**base).clone()), Box::new((**exp).clone())),
                    Expr::Func(FuncKind::Log, Box::new((**base).clone())),
                    d_exp,
                ]));
            }
            // f(x)^g(x): f^g * (g' ln f + g f'/f)
            let f = (**base).clone();
            let g = (**exp).clone();
            let dg = diff(&g, var);
            let df = diff(&f, var);
            let term1 = crate::norm::norm(&Expr::Mul(vec![
                dg,
                Expr::Func(FuncKind::Log, Box::new(f.clone())),
            ]));
            let term2 = crate::norm::norm(&Expr::Mul(vec![g.clone(), df, inv(&f)]));
            let inner = crate::norm::norm(&Expr::Add(vec![term1, term2]));
            crate::norm::norm(&Expr::Mul(vec![
                Expr::Pow(Box::new(f), Box::new(g)),
                inner,
            ]))
        }
        Expr::Func(k, a) => {
            // chain rule: f'(a) * a'
            let a = (**a).clone();
            let da = diff(&a, var);
            let outer = match k {
                FuncKind::Sin => Expr::Func(FuncKind::Cos, Box::new(a.clone())),
                FuncKind::Cos => Expr::Mul(vec![
                    Expr::int(-1),
                    Expr::Func(FuncKind::Sin, Box::new(a.clone())),
                ]),
                FuncKind::Tan => {
                    // sec^2 = 1/cos^2
                    let cos_a = Expr::Func(FuncKind::Cos, Box::new(a.clone()));
                    let cos2 = Expr::Pow(Box::new(cos_a), Box::new(Expr::int(2)));
                    inv(&cos2)
                }
                FuncKind::Log => inv(&a),
                FuncKind::Exp => Expr::Func(FuncKind::Exp, Box::new(a.clone())),
                FuncKind::Abs => {
                    // d|x| = x/|x| (undefined at 0 — keep symbolic)
                    crate::norm::norm(&Expr::Mul(vec![
                        a.clone(),
                        inv(&Expr::Func(FuncKind::Abs, Box::new(a.clone()))),
                    ]))
                }
                // sqrt never survives norm (canonicalized to Pow(x,1/2));
                // treat defensively via the power rule
                FuncKind::Sqrt => diff(
                    &crate::norm::norm(&Expr::Pow(Box::new(a.clone()), Box::new(Expr::rat(1, 2)))),
                    var,
                ),
            };
            crate::norm::norm(&Expr::Mul(vec![outer, da]))
        }
    }
}

fn inv(e: &Expr) -> Expr {
    Expr::Pow(Box::new(e.clone()), Box::new(Expr::int(-1)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evalf::equiv;
    use crate::parser::parse_expr;

    fn p(s: &str) -> Expr {
        parse_expr(s).unwrap()
    }

    #[test]
    fn test_basic_derivatives() {
        assert_eq!(diff(&p("x**3 + 2*x"), "x"), p("3*x**2 + 2"));
        assert_eq!(diff(&p("5"), "x"), Expr::int(0));
        assert!(equiv(&diff(&p("(x+1)**3"), "x"), &p("3*(x+1)**2")));
        // product rule: d(x*sin(x)) = sin(x) + x*cos(x)
        assert!(equiv(
            &diff(&p("x*sin(x)"), "x"),
            &p("sin(x) + x*cos(x)")
        ));
        // quotient rule check via equiv: d(sin(x)/x)
        let d = diff(&p("sin(x)/x"), "x");
        assert!(equiv(
            &d,
            &p("(x*cos(x) - sin(x))/x**2")
        ));
        // chain rule: d(sin(x**2)) = 2*x*cos(x**2)
        assert!(equiv(
            &diff(&p("sin(x**2)"), "x"),
            &p("2*x*cos(x**2)")
        ));
        // exp
        assert!(equiv(&diff(&p("exp(x)"), "x"), &p("exp(x)")));
        // e^x via Pow with base e? parsed as Sym('e')... skip
        // log
        assert!(equiv(&diff(&p("log(x)"), "x"), &p("1/x")));
        // constant power: d(2**x) = 2**x*log(2)
        assert!(equiv(&diff(&p("2**x"), "x"), &p("2**x*log(2)")));
        // tan
        assert!(equiv(&diff(&p("tan(x)"), "x"), &p("1/cos(x)**2")));
    }

    #[test]
    fn test_multivariable() {
        // d/dx of x + y is 1
        assert_eq!(diff(&p("x + y"), "x"), Expr::int(1));
        assert_eq!(diff(&p("x*y"), "y"), p("x"));
    }
}
