//! Exact trigonometry at standard angles (rational multiples of pi with
//! denominators 1, 2, 3, 4, 6) plus the Pythagorean identity rewrite used
//! by `simplify`. Values are built as exact symbolic expressions
//! (1/2, sqrt(2)/2, sqrt(3)/2, ...), mirroring sympy's exact evaluation
//! that the trig domains rely on.

use crate::expr::Expr;
use crate::norm::norm;
use reasoning_common::Rat;

/// Evaluate sin/cos/tan of an exact rational multiple of pi, if `a` is one.
/// Returns None when the argument is not a supported standard angle.
pub fn exact_eval(k: crate::expr::FuncKind, a: &Expr) -> Option<Expr> {
    use crate::expr::FuncKind;
    match k {
        FuncKind::Sin | FuncKind::Cos | FuncKind::Tan => {}
        _ => return None,
    }
    // rational-constant argument (0 is the standard case)
    if let Some(r) = a.as_rat() {
        if r.is_zero() {
            return Some(match k {
                FuncKind::Sin => Expr::int(0),
                FuncKind::Tan => Expr::int(0),
                _ => Expr::int(1), // cos(0) = 1
            });
        }
        return None; // nonzero rational radians: not a standard angle
    }
    // a must be (rational) * pi
    let r = pi_multiple(a)?;
    let (num, den) = (r.num, r.den);
    if den != 1 && den != 2 && den != 3 && den != 4 && den != 6 {
        return None;
    }
    // reduce angle modulo 2*pi: k/den -> k' / den with k' in [0, 2*den)
    let period = 2 * den;
    let k2 = num.rem_euclid(period);
    let sign_flip = if k2 >= period / 2 {
        // angle in [pi, 2pi): use reference angle in [0, pi) and flip sign
        true
    } else {
        false
    };
    let k3 = if sign_flip { k2 - period / 2 } else { k2 };
    let val: Expr = match (k, den, k3) {
        // sin(0) = 0
        (FuncKind::Sin, _, 0) => Expr::int(0),
        (FuncKind::Cos, _, 0) => Expr::int(1),
        (FuncKind::Tan, _, 0) => Expr::int(0),
        // pi/6 family
        (FuncKind::Sin, 6, 1) => Expr::rat(1, 2),
        (FuncKind::Sin, 3, 1) => Expr::sqrt(Expr::int(3)) / Expr::int(2),
        (FuncKind::Sin, 4, 1) => Expr::sqrt(Expr::int(2)) / Expr::int(2),
        (FuncKind::Sin, 2, 1) => Expr::int(1),
        (FuncKind::Sin, 1, _) => Expr::int(0), // sin(k*pi) = 0
        (FuncKind::Sin, 6, 5) => Expr::rat(1, 2),
        (FuncKind::Sin, 3, 2) => Expr::sqrt(Expr::int(3)) / Expr::int(2),
        (FuncKind::Sin, 4, 3) => Expr::sqrt(Expr::int(2)) / Expr::int(2),
        // cos family: cos(t) = sin(pi/2 - t)
        (FuncKind::Cos, 6, 1) => Expr::sqrt(Expr::int(3)) / Expr::int(2),
        (FuncKind::Cos, 3, 1) => Expr::rat(1, 2),
        (FuncKind::Cos, 4, 1) => Expr::sqrt(Expr::int(2)) / Expr::int(2),
        (FuncKind::Cos, 2, 1) => Expr::int(0),
        (FuncKind::Cos, 1, _) => if k3 == 1 { Expr::int(-1) } else { Expr::int(1) },
        (FuncKind::Cos, 6, 5) => -Expr::sqrt(Expr::int(3)) / Expr::int(2),
        (FuncKind::Cos, 3, 2) => -Expr::rat(1, 2),
        (FuncKind::Cos, 4, 3) => -Expr::sqrt(Expr::int(2)) / Expr::int(2),
        // tan family
        (FuncKind::Tan, 6, 1) => Expr::sqrt(Expr::int(3)) / Expr::int(3),
        (FuncKind::Tan, 3, 1) => Expr::sqrt(Expr::int(3)),
        (FuncKind::Tan, 4, 1) => Expr::int(1),
        (FuncKind::Tan, 2, 1) => {
            return None; // tan(pi/2) undefined
        }
        (FuncKind::Tan, 1, _) => Expr::int(0),
        (FuncKind::Tan, 6, 5) => -Expr::sqrt(Expr::int(3)) / Expr::int(3),
        (FuncKind::Tan, 3, 2) => -Expr::sqrt(Expr::int(3)),
        (FuncKind::Tan, 4, 3) => Expr::int(-1),
        // sin/cos at 5*pi/6, 2*pi/3, 3*pi/4 etc. via (k3, den) leftovers:
        (FuncKind::Sin, 6, 3) => Expr::int(1), // 6/6*... handled below
        _ => {
            // generic fallback: k3/den with k3 >= den means pi - ref
            let ref_k = if k3 > den / 2 && k3 < den { den - k3 } else { k3 };
            let _ = ref_k;
            return None;
        }
    };
    Some(norm(&if sign_flip { -val } else { val }))
}

/// If `a` is a rational multiple of pi (r*pi or pi*r or r*pi/d forms),
/// return the rational multiplier.
fn pi_multiple(a: &Expr) -> Option<Rat> {
    // pi itself
    if matches!(a, Expr::Pi) {
        return Some(Rat::from_int(1));
    }
    // r * pi (canonical Mul with numeric coefficient)
    if let Expr::Mul(xs) = a {
        let mut coeff: Option<Rat> = None;
        let mut has_pi = false;
        for x in xs {
            match x {
                Expr::Pi => has_pi = true,
                Expr::Num(r) => coeff = Some(coeff.map(|c: Rat| c * *r).unwrap_or(*r)),
                _ => return None,
            }
        }
        if has_pi {
            return Some(coeff.unwrap_or(Rat::from_int(1)));
        }
        return None;
    }
    None
}

/// Rewrite at the top level of a sum:
///   sin(x)^2 + cos(x)^2 -> 1
///   -(sin(x)^2 + cos(x)^2) -> -1
///   1 - sin(x)^2 -> cos(x)^2 ; 1 - cos(x)^2 -> sin(x)^2
///   sin(x)^2 - 1 -> -cos(x)^2 ; cos(x)^2 - 1 -> -sin(x)^2
pub fn apply_pythagorean(e: &Expr) -> Expr {
    let Expr::Add(terms) = e else {
        return e.clone();
    };
    // (sign, arg) of sin^2 / cos^2 terms with |coeff| == 1
    let mut sin2: Option<(i64, Expr)> = None;
    let mut cos2: Option<(i64, Expr)> = None;
    let mut const_sum = Rat::from_int(0);
    let mut rest: Vec<Expr> = Vec::new();
    let mut consumed: Vec<usize> = Vec::new();
    for (i, t) in terms.iter().enumerate() {
        let (coeff, r) = t.as_coeff_mul();
        if let Expr::Num(nr) = &r {
            if nr.is_one() {
                const_sum = const_sum + coeff;
                consumed.push(i);
                continue;
            }
        }
        if coeff.den == 1 && coeff.num.abs() == 1 {
            if let Some(arg) = squared_func(&r, "sin") {
                sin2 = Some((coeff.num, arg));
                consumed.push(i);
                continue;
            }
            if let Some(arg) = squared_func(&r, "cos") {
                cos2 = Some((coeff.num, arg));
                consumed.push(i);
                continue;
            }
        }
        rest.push(t.clone());
    }
    let keep: Vec<Expr> = terms
        .iter()
        .enumerate()
        .filter(|(i, _)| !consumed.contains(i))
        .map(|(_, t)| t.clone())
        .collect();
    let mut out: Vec<Expr> = keep;
    let mut new_const = const_sum;
    match (sin2, cos2) {
        (Some((s, a)), Some((c, b))) if a == b && s == c => {
            // +- (sin^2 + cos^2) -> +-1
            new_const = new_const + Rat::from_int(s);
        }
        (Some((-1, a)), None) if new_const.is_one() => {
            // 1 - sin^2 = cos^2
            out.push(Expr::Pow(
                Box::new(Expr::Func(crate::expr::FuncKind::Cos, Box::new(a))),
                Box::new(Expr::int(2)),
            ));
            new_const = Rat::from_int(0);
        }
        (None, Some((-1, a))) if new_const.is_one() => {
            // 1 - cos^2 = sin^2
            out.push(Expr::Pow(
                Box::new(Expr::Func(crate::expr::FuncKind::Sin, Box::new(a))),
                Box::new(Expr::int(2)),
            ));
            new_const = Rat::from_int(0);
        }
        (Some((1, a)), None) if new_const == Rat::from_int(-1) => {
            // sin^2 - 1 = -cos^2
            out.push(Expr::Mul(vec![
                Expr::int(-1),
                Expr::Pow(
                    Box::new(Expr::Func(crate::expr::FuncKind::Cos, Box::new(a))),
                    Box::new(Expr::int(2)),
                ),
            ]));
            new_const = Rat::from_int(0);
        }
        (None, Some((1, a))) if new_const == Rat::from_int(-1) => {
            // cos^2 - 1 = -sin^2
            out.push(Expr::Mul(vec![
                Expr::int(-1),
                Expr::Pow(
                    Box::new(Expr::Func(crate::expr::FuncKind::Sin, Box::new(a))),
                    Box::new(Expr::int(2)),
                ),
            ]));
            new_const = Rat::from_int(0);
        }
        _ => {
            // no rule applied — restore everything
            return e.clone();
        }
    }
    if !new_const.is_zero() {
        out.push(Expr::Num(new_const));
    }
    crate::norm::norm(&Expr::Add(out))
}

/// term body (coefficient already stripped by the caller) is `f(a)**2`?
fn squared_func(t: &Expr, fname: &str) -> Option<Expr> {
    if let Expr::Pow(base, exp) = t {
        if exp.as_int() == Some(2) {
            if let Expr::Func(k, arg) = base.as_ref() {
                if k.name() == fname {
                    return Some((**arg).clone());
                }
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::parse_expr;
    use crate::norm::simplify;

    fn p(s: &str) -> Expr {
        parse_expr(s).unwrap()
    }

    #[test]
    fn test_standard_angles() {
        assert_eq!(p("sin(pi/6)"), Expr::rat(1, 2));
        assert_eq!(p("cos(pi/3)"), Expr::rat(1, 2));
        assert_eq!(p("sin(pi/2)"), Expr::int(1));
        assert_eq!(p("cos(0)"), Expr::int(1));
        assert_eq!(p("sin(0)"), Expr::int(0));
        assert_eq!(p("tan(pi/4)"), Expr::int(1));
        assert_eq!(p("sin(pi)"), Expr::int(0));
        assert_eq!(p("sin(2*pi)"), Expr::int(0));
        assert_eq!(p("cos(pi)"), Expr::int(-1));
        // 30 degrees in radians
        assert_eq!(p("sin(30*pi/180)"), Expr::rat(1, 2));
        // sin(5*pi/6) = 1/2
        assert_eq!(p("sin(5*pi/6)"), Expr::rat(1, 2));
        // sin(3*pi/4) = sqrt(2)/2
        assert!((crate::evalf::eval_f64(&p("sin(3*pi/4)"), &Default::default()).unwrap()
            - std::f64::consts::FRAC_1_SQRT_2)
            .abs()
            < 1e-12);
    }

    #[test]
    fn test_tan_exact() {
        // tan(pi/6) = sqrt(3)/3 numerically
        let t = p("tan(pi/6)");
        let v = crate::evalf::eval_f64(&t, &Default::default()).unwrap();
        assert!((v - 3f64.sqrt() / 3.0).abs() < 1e-12);
    }

    #[test]
    fn test_pythagorean_simplify() {
        assert_eq!(simplify(&p("sin(x)**2 + cos(x)**2")), Expr::int(1));
        // 1 - cos(x)^2 = sin(x)^2 (sympy behavior — the square must be kept)
        assert_eq!(simplify(&p("1 - cos(x)**2")), p("sin(x)**2"));
        assert_eq!(simplify(&p("1 - sin(x)**2")), p("cos(x)**2"));
        assert_eq!(simplify(&p("sin(x)**2 - 1")), p("-cos(x)**2"));
        assert_eq!(simplify(&p("cos(x)**2 - 1")), p("-sin(x)**2"));
    }
}
