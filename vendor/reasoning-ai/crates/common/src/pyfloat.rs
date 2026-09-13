//! Python float semantics: `round(x, n)` (banker's rounding) and
//! `str(float)` (shortest repr, scientific notation for |exp| extremes).

/// Python `round(x, n)` — round-half-to-even applied to the EXACT binary
/// value's decimal expansion (Python uses correctly-rounded decimal
/// conversion, not naive multiply-and-round). Returns a float.
pub fn py_round(x: f64, ndigits: i32) -> f64 {
    if !x.is_finite() {
        return x;
    }
    if x == 0.0 {
        return 0.0;
    }
    if ndigits < 0 {
        // round to a multiple of 10^-ndigits, half-to-even
        let factor = 10f64.powi(-ndigits);
        let scaled = x / factor;
        if !scaled.is_finite() {
            return x;
        }
        let r = round_half_even_i64(scaled);
        let out = r as f64 * factor;
        if out == 0.0 {
            0.0
        } else {
            out
        }
    } else {
        // Rust's `{:.n}` formats the exact decimal expansion of the binary
        // value with round-half-to-even — same semantics as Python's round.
        let s = format!("{:.*}", ndigits as usize, x);
        let out: f64 = s.parse().unwrap_or(x);
        if out == 0.0 {
            0.0
        } else {
            out
        }
    }
}

/// Python `round(x)` with no digits — half-to-even integer rounding.
fn round_half_even_i64(x: f64) -> i64 {
    let floor = x.floor();
    let diff = x - floor;
    let fl = floor as i64;
    if diff > 0.5 {
        fl + 1
    } else if diff < 0.5 {
        fl
    } else {
        // exactly .5 — tie to even
        if fl % 2 == 0 {
            fl
        } else {
            fl + 1
        }
    }
}

/// Python `str(float)`: shortest-roundtrip repr with Python's
/// plain/scientific switching rules (scientific when decimal exponent
/// is < -4 or >= 16).
pub fn py_float_str(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    let neg = x < 0.0;
    let ax = x.abs();
    // Shortest-representation digits + decimal exponent via {:e}
    let e_str = format!("{:e}", ax); // "d.ddde<exp>" or "de<exp>"
    let (mant, exp) = e_str.split_once('e').expect("{:e} always has 'e'");
    let exp: i32 = exp.parse().unwrap();
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let digits = digits.trim_end_matches('0');
    let digits = if digits.is_empty() { "0" } else { digits };
    let n = digits.len() as i32;

    let mut out = String::new();
    if neg {
        out.push('-');
    }
    // Python: plain notation when -4 <= exp < 16
    if exp >= -4 && exp < 16 {
        if exp >= 0 {
            if exp >= n - 1 {
                out.push_str(digits);
                for _ in 0..(exp - (n - 1)) {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..(exp + 1) as usize]);
                out.push('.');
                out.push_str(&digits[(exp + 1) as usize..]);
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(digits);
        }
    } else {
        // scientific: "d.ddd" mantissa (no trailing zeros), e, sign, >=2 exp digits
        out.push(digits.chars().next().unwrap());
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        if exp < 0 {
            out.push('-');
        } else {
            out.push('+');
        }
        let a = exp.abs();
        if a < 10 {
            out.push('0');
        }
        out.push_str(&a.to_string());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_py_float_str_basic() {
        assert_eq!(py_float_str(5.0), "5.0");
        assert_eq!(py_float_str(-5.0), "-5.0");
        assert_eq!(py_float_str(0.5), "0.5");
        assert_eq!(py_float_str(1.0 / 3.0), "0.3333333333333333");
        assert_eq!(py_float_str(1e16), "1e+16");
        assert_eq!(py_float_str(1.5e16), "1.5e+16");
        assert_eq!(py_float_str(1e15), "1000000000000000.0");
        assert_eq!(py_float_str(1e-4), "0.0001");
        assert_eq!(py_float_str(1e-5), "1e-05");
        assert_eq!(py_float_str(1.23e-5), "1.23e-05");
        assert_eq!(py_float_str(-1e-7), "-1e-07");
        assert_eq!(py_float_str(0.0), "0.0");
        assert_eq!(py_float_str(100.0), "100.0");
        assert_eq!(py_float_str(0.1), "0.1");
        assert_eq!(py_float_str(2.5), "2.5");
    }

    #[test]
    fn test_py_round() {
        assert_eq!(py_round(2.5, 0), 2.0);
        assert_eq!(py_round(3.5, 0), 4.0);
        assert_eq!(py_round(-2.5, 0), -2.0);
        assert_eq!(py_round(0.25, 1), 0.2);
        assert_eq!(py_round(0.35, 1), 0.3); // 0.35 is 0.34999... in binary
        assert_eq!(py_round(2.675, 2), 2.67);
        assert_eq!(py_round(123.456, 2), 123.46);
        assert_eq!(py_round(123.456, 4), 123.456);
    }
}
