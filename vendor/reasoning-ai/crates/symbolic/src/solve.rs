//! Equation solving (the sympy `solve` subset used here): single-variable
//! linear and quadratic equations over exact rationals.

use crate::expr::Expr;
use crate::norm::expand;
use reasoning_common::Rat;

/// Extract the polynomial coefficients of `e` in variable `var` when `e`
/// is (after expansion) a univariate polynomial in `var`: returns a map
/// degree -> coefficient, or None when any term mixes `var` with other
/// symbols/functions.
pub fn poly_coeffs(e: &Expr, var: &str) -> Option<Vec<Rat>> {
    let ex = expand(e);
    let mut coeffs: Vec<Rat> = vec![];
    for t in ex.terms() {
        let (coeff, rest) = t.as_coeff_mul();
        // rest must be var^k with nothing else
        let deg = symbolic_degree(&rest, var)?;
        while coeffs.len() <= deg {
            coeffs.push(Rat::from_int(0));
        }
        coeffs[deg] = coeffs[deg] + coeff;
    }
    Some(coeffs)
}

/// Degree of `rest` as a pure power of `var`; None if it involves anything
/// else (other symbols, functions, non-integer powers).
fn symbolic_degree(rest: &Expr, var: &str) -> Option<usize> {
    match rest {
        Expr::Num(r) if r.is_one() => Some(0),
        Expr::Sym(s) if s == var => Some(1),
        Expr::Pow(base, exp) => {
            let b_deg = symbolic_degree(base, var)?;
            let n = exp.as_int()?;
            if b_deg == 0 {
                Some(0)
            } else {
                Some(b_deg * n.max(0) as usize)
            }
        }
        _ => None,
    }
}

/// Solve `equation_str` (either "lhs = rhs" or "expr" meaning expr = 0)
/// for `variable`, exactly like the Python project's
/// `verifier.symbolic_verifier.solve_equation`. Returns the list of roots
/// as exact expressions (empty when there is no solution).
pub fn solve_equation(equation_str: &str, variable: &str) -> Result<Vec<Expr>, String> {
    let lhs = if let Some((l, r)) = equation_str.split_once('=') {
        let l = crate::parser::parse_expr(l)
            .map_err(|e| format!("Could not parse {:?}: {}", l, e))?;
        let r = crate::parser::parse_expr(r)
            .map_err(|e| format!("Could not parse {:?}: {}", r, e))?;
        l - r
    } else {
        crate::parser::parse_expr(equation_str)
            .map_err(|e| format!("Could not parse {:?}: {}", equation_str, e))?
    };
    let coeffs = poly_coeffs(&lhs, variable)
        .ok_or_else(|| format!("equation is not a polynomial in {}", variable))?;
    Ok(solve_poly(&coeffs))
}

/// Solve a polynomial given by ascending-degree coefficients.
pub fn solve_poly(coeffs: &[Rat]) -> Vec<Expr> {
    // trim trailing zeros
    let mut n = coeffs.len();
    while n > 0 && coeffs[n - 1].is_zero() {
        n -= 1;
    }
    let c = &coeffs[..n];
    match c.len() {
        0 | 1 => vec![], // 0 = 0 (any x) or const = 0 (no solution)
        2 => {
            // linear: a*x + b = 0 -> x = -b/a
            let a = c[1];
            let b = c[0];
            if a.is_zero() {
                vec![]
            } else {
                vec![Expr::Num(-b / a)]
            }
        }
        3 => {
            // quadratic: a*x^2 + b*x + c = 0
            let (a, b, cc) = (c[2], c[1], c[0]);
            if a.is_zero() {
                return solve_poly(&[b, cc]);
            }
            let disc = b * b - Rat::from_int(4) * a * cc;
            let two_a = Rat::from_int(2) * a;
            if disc.is_zero() {
                return vec![Expr::Num(-b / two_a)];
            }
            if disc.num > 0 {
                // perfect square -> exact rational roots
                let (dn, dd) = (disc.num, disc.den);
                let sn = isqrt_i64(dn);
                let sd = isqrt_i64(dd);
                if sn * sn == dn && sd * sd == dd {
                    let sqrt_disc = Rat::new(sn, sd);
                    let r1 = (-b + sqrt_disc) / two_a;
                    let r2 = (-b - sqrt_disc) / two_a;
                    let mut out = vec![Expr::Num(r1), Expr::Num(r2)];
                    out.sort_by_key(|e| {
                        let r = e.as_rat().unwrap_or(Rat::from_int(0));
                        r.to_f64() as i64
                    });
                    return out;
                }
                // keep exact symbolic sqrt form
                let sqrt_disc = Expr::sqrt(Expr::Num(disc));
                let r1 = (-Expr::Num(b) + sqrt_disc.clone()) / Expr::Num(two_a);
                let r2 = (-Expr::Num(b) - sqrt_disc) / Expr::Num(two_a);
                return vec![r1, r2];
            }
            // complex roots — sympy returns them; this project never needs
            // them for its linear/word-problem paths, so return empty to
            // signal "no real root" callers can detect via .is_empty().
            vec![]
        }
        _ => vec![], // higher degree: not needed by this project
    }
}

fn isqrt_i64(v: i64) -> i64 {
    if v <= 0 {
        return 0;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_linear() {
        assert_eq!(solve_equation("2*x + 3 = 11", "x").unwrap(), vec![Expr::int(4)]);
        assert_eq!(solve_equation("2x + 3 = 11", "x").unwrap(), vec![Expr::int(4)]);
        assert_eq!(solve_equation("x - 5 = 0", "x").unwrap(), vec![Expr::int(5)]);
        assert_eq!(
            solve_equation("3*x = 7", "x").unwrap(),
            vec![Expr::rat(7, 3)]
        );
        assert_eq!(solve_equation("2*x = 4*x", "x").unwrap(), vec![Expr::int(0)]);
        // no solution
        assert!(solve_equation("x = x + 1", "x").unwrap().is_empty());
    }

    #[test]
    fn test_quadratic() {
        // x^2 - 5x + 6 = 0 -> 2, 3
        let roots = solve_equation("x**2 - 5*x + 6 = 0", "x").unwrap();
        assert_eq!(roots.len(), 2);
        assert_eq!(roots[0], Expr::int(2));
        assert_eq!(roots[1], Expr::int(3));
        // x^2 = 2 -> +-sqrt(2) symbolic
        let roots = solve_equation("x**2 - 2 = 0", "x").unwrap();
        assert_eq!(roots.len(), 2);
        let v1 = crate::evalf::eval_f64(&roots[0], &Default::default()).unwrap();
        let v2 = crate::evalf::eval_f64(&roots[1], &Default::default()).unwrap();
        assert!((v1.abs() - 2f64.sqrt()).abs() < 1e-12);
        assert!((v2.abs() - 2f64.sqrt()).abs() < 1e-12);
    }

    #[test]
    fn test_non_poly() {
        // sin(x) = 0 is not polynomial — error like sympy would refuse
        assert!(solve_equation("sin(x) = 0", "x").is_err());
    }
}
