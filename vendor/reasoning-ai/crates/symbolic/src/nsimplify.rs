//! `nsimplify` — convert approximate values / expressions into nice exact
//! forms: small rationals, rational multiples of pi, and sqrt of small
//! rationals. Mirrors the sympy `nsimplify(value, [pi])` calls in the
//! trig and linear-algebra domains (their string ground truths).

use crate::expr::Expr;
use reasoning_common::Rat;

/// Convert an f64 to its nicest exact form:
/// - a rational with small terms (denominator <= 1000) → Rat
/// - a rational multiple of pi (v = k*pi/d with small d) → (k/d)*pi
/// - sqrt of a small rational → sqrt form
/// Falls back to the closest rational.
pub fn nsimplify_f64(v: f64) -> Expr {
    if !v.is_finite() {
        return Expr::Num(Rat::from_int(0));
    }
    if v == 0.0 {
        return Expr::int(0);
    }
    // 1) rational multiple of pi?
    if let Some(r) = pi_rational(v) {
        return crate::norm::norm(&Expr::Mul(vec![Expr::Num(r), Expr::Pi]));
    }
    // 2) plain small rational?
    let r = Rat::from_f64(v);
    if r.den <= 10_000 {
        return Expr::Num(r);
    }
    // 3) sqrt of a small rational? (v^2 rational with small terms)
    let v2 = v * v;
    let r2 = Rat::from_f64(v2);
    if r2.den <= 100 && r2.num.abs() <= 100_000 {
        let cand = Expr::sqrt(Expr::Num(r2));
        if let Ok(cv) = crate::evalf::eval_f64(&cand, &Default::default()) {
            if (cv - v).abs() <= 1e-10 * v.abs().max(1.0) {
                return cand;
            }
        }
    }
    // 4) sqrt multiple of pi? (v/pi)^2 rational — rare; skip
    Expr::Num(Rat::from_f64(v))
}

/// Is v (approximately) a rational multiple of pi with small terms?
fn pi_rational(v: f64) -> Option<Rat> {
    let q = v / std::f64::consts::PI;
    let r = Rat::from_f64(q);
    // tolerate the conversion error
    let rq = r.to_f64();
    if (rq - q).abs() > 1e-9 * q.abs().max(1.0) {
        // try harder: continued fraction with looser tolerance
        return None;
    }
    if r.den <= 24 && r.num.abs() <= 24 {
        // exclude the trivial "everything is 0*pi" case
        if r.is_zero() {
            return None;
        }
        return Some(r);
    }
    None
}

/// nsimplify an expression: evaluate constants to exact standard-angle
/// trig / rational values (the expression version of the sympy call).
pub fn nsimplify(e: &Expr) -> Expr {
    // exact trig at standard angles is already folded by norm; evaluate
    // anything that reduces to a constant exactly
    let en = crate::norm::norm(e);
    let syms = en.free_symbols();
    if syms.is_empty() {
        // constant expression: try to keep an exact symbolic form rather
        // than a float (e.g. sqrt(2)/2, pi/6, 1/2)
        if let Ok(v) = crate::evalf::eval_f64(&en, &Default::default()) {
            // keep exact structure when it evaluates exactly (rational)
            if let Some(r) = en.as_rat() {
                return Expr::Num(r);
            }
            // if the whole expression is (k*sqrt(n))/d form, keep it
            let txt = en.to_string();
            if txt.contains("sqrt") || txt.contains("pi") {
                return en;
            }
            // otherwise pick the nicest exact form of the float
            return nsimplify_f64(v);
        }
    }
    en
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nsimplify_rationals() {
        assert_eq!(nsimplify_f64(0.5), Expr::rat(1, 2));
        assert_eq!(nsimplify_f64(2.0), Expr::int(2));
        assert_eq!(nsimplify_f64(-1.5), Expr::rat(-3, 2));
    }

    #[test]
    fn test_nsimplify_pi() {
        // 1.5708... = pi/2
        assert_eq!(
            nsimplify_f64(std::f64::consts::FRAC_PI_2),
            crate::norm::norm(&Expr::Mul(vec![Expr::rat(1, 2), Expr::Pi]))
        );
        // 3.14159 = pi
        assert_eq!(nsimplify_f64(std::f64::consts::PI), Expr::Pi);
        // pi/6 = 0.5236
        assert_eq!(
            nsimplify_f64(std::f64::consts::PI / 6.0),
            crate::norm::norm(&Expr::Mul(vec![Expr::rat(1, 6), Expr::Pi]))
        );
    }

    #[test]
    fn test_nsimplify_sqrt() {
        // 0.7071... = sqrt(2)/2 stays as a nice radical
        let e = nsimplify_f64(std::f64::consts::FRAC_1_SQRT_2);
        let v = crate::evalf::eval_f64(&e, &Default::default()).unwrap();
        assert!((v - std::f64::consts::FRAC_1_SQRT_2).abs() < 1e-10);
    }
}
