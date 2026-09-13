//! Exact rational arithmetic on i64 (mirrors Python's `fractions.Fraction`
//! for the ranges this project uses). Always kept in lowest terms with a
//! positive denominator.

use std::cmp::Ordering;
use std::ops::{Add, Div, Mul, Neg, Sub};

fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Rat {
    pub num: i64,
    pub den: i64, // always > 0
}

impl Rat {
    pub fn new(num: i64, den: i64) -> Rat {
        assert!(den != 0, "Rat denominator must be nonzero");
        let sign = if den < 0 { -1 } else { 1 };
        let g0 = gcd(num, den);
        let g = if g0 == 0 { 1 } else { g0 };
        Rat {
            num: sign * num / g,
            den: (den * sign).abs() / g,
        }
    }

    pub fn from_int(v: i64) -> Rat {
        Rat { num: v, den: 1 }
    }

    pub fn from_f64(v: f64) -> Rat {
        // Best rational approximation with bounded terms (like sympy's
        // nsimplify(rational=True) for typical float inputs).
        if v == 0.0 {
            return Rat::from_int(0);
        }
        // handle sign
        let neg = v < 0.0;
        let x = v.abs();
        // continued fraction, stop when close enough or terms get big
        let eps = 1e-12 * x.max(1.0);
        let mut h0: i64 = 0;
        let mut h1: i64 = 1;
        let mut k0: i64 = 1;
        let mut k1: i64 = 0;
        let mut b = x;
        for _ in 0..64 {
            let a = b.floor() as i64;
            let h2 = a
                .checked_mul(h1)
                .and_then(|t| t.checked_add(h0));
            let k2 = a
                .checked_mul(k1)
                .and_then(|t| t.checked_add(k0));
            match (h2, k2) {
                (Some(h2), Some(k2)) if k2 != 0 => {
                    h0 = h1;
                    h1 = h2;
                    k0 = k1;
                    k1 = k2;
                }
                _ => break,
            }
            let approx = h1 as f64 / k1 as f64;
            if (approx - x).abs() <= eps || k1 > 10_000_000 {
                break;
            }
            let frac = b - a as f64;
            if frac < 1e-15 {
                break;
            }
            b = 1.0 / frac;
        }
        if k1 == 0 {
            return Rat::from_int(0);
        }
        let r = Rat::new(if neg { -h1 } else { h1 }, k1);
        if r.to_f64().is_finite() {
            r
        } else {
            Rat::from_int(if neg { -(x.round() as i64) } else { x.round() as i64 })
        }
    }

    pub fn to_f64(self) -> f64 {
        self.num as f64 / self.den as f64
    }

    pub fn is_integer(self) -> bool {
        self.den == 1
    }

    pub fn is_zero(self) -> bool {
        self.num == 0
    }

    pub fn is_one(self) -> bool {
        self.num == 1 && self.den == 1
    }

    pub fn recip(self) -> Rat {
        Rat::new(self.den, self.num)
    }

    pub fn abs(self) -> Rat {
        Rat {
            num: self.num.abs(),
            den: self.den,
        }
    }
}

impl PartialOrd for Rat {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Rat {
    fn cmp(&self, other: &Self) -> Ordering {
        // a/b vs c/d  ->  a*d vs c*b  (dens are positive)
        let l = self
            .num
            .checked_mul(other.den)
            .unwrap_or(i64::MAX / 2);
        let r = other
            .num
            .checked_mul(self.den)
            .unwrap_or(i64::MAX / 2);
        l.cmp(&r)
    }
}

impl Add for Rat {
    type Output = Rat;
    fn add(self, rhs: Rat) -> Rat {
        Rat::new(
            self.num * rhs.den + rhs.num * self.den,
            self.den * rhs.den,
        )
    }
}

impl Sub for Rat {
    type Output = Rat;
    fn sub(self, rhs: Rat) -> Rat {
        Rat::new(
            self.num * rhs.den - rhs.num * self.den,
            self.den * rhs.den,
        )
    }
}

impl Mul for Rat {
    type Output = Rat;
    fn mul(self, rhs: Rat) -> Rat {
        Rat::new(self.num * rhs.num, self.den * rhs.den)
    }
}

impl Div for Rat {
    type Output = Rat;
    fn div(self, rhs: Rat) -> Rat {
        assert!(rhs.num != 0, "Rat division by zero");
        Rat::new(self.num * rhs.den, self.den * rhs.num)
    }
}

impl Neg for Rat {
    type Output = Rat;
    fn neg(self) -> Rat {
        Rat {
            num: -self.num,
            den: self.den,
        }
    }
}

impl std::fmt::Display for Rat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.den == 1 {
            write!(f, "{}", self.num)
        } else {
            write!(f, "{}/{}", self.num, self.den)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basics() {
        assert_eq!(Rat::new(2, 4), Rat::new(1, 2));
        assert_eq!(Rat::new(-2, 4), Rat::new(-1, 2));
        assert_eq!(Rat::new(2, -4), Rat::new(-1, 2));
        assert_eq!(Rat::new(1, 2) + Rat::new(1, 3), Rat::new(5, 6));
        assert_eq!(Rat::new(1, 2) * Rat::new(3, 4), Rat::new(3, 8));
        assert_eq!(Rat::new(3, 4) / Rat::new(3, 2), Rat::new(1, 2));
        assert!(Rat::new(1, 3) < Rat::new(1, 2));
        assert_eq!(Rat::new(1, 3).to_f64(), 1.0 / 3.0);
    }

    #[test]
    fn test_from_f64() {
        assert_eq!(Rat::from_f64(0.5), Rat::new(1, 2));
        assert_eq!(Rat::from_f64(0.25), Rat::new(1, 4));
        assert_eq!(Rat::from_f64(2.0), Rat::from_int(2));
        assert_eq!(Rat::from_f64(-1.5), Rat::new(-3, 2));
        assert_eq!(Rat::from_f64(1.0 / 3.0), Rat::new(1, 3));
    }
}
