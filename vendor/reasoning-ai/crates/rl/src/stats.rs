//! Phase 074 — Statistical significance utilities (design doc 24:
//! evaluation must be rigorous, not vibes). Wilson score intervals and
//! two-proportion tests for solve-rate comparisons.

/// 95% (default, z=1.96) Wilson score confidence interval for a binomial
/// proportion. Preferred over the naive normal-approximation interval
/// because it stays inside [0,1] and is much better calibrated for small
/// n or p near 0/1 — exactly the regime solve-rate evaluations (n=20,
/// p often < 0.2) are in. Returns `(lo, hi)`.
pub fn wilson_score_interval(successes: i64, n: i64, z: f64) -> Result<(f64, f64), String> {
    if n <= 0 {
        return Err(format!("n must be > 0, got {}", n));
    }
    if !(0 <= successes && successes <= n) {
        return Err(format!("successes must be in [0, n], got {}/{}", successes, n));
    }
    let p = successes as f64 / n as f64;
    let n_f = n as f64;
    let denom = 1.0 + z * z / n_f;
    let center = (p + z * z / (2.0 * n_f)) / denom;
    let half_width =
        (z * (p * (1.0 - p) / n_f + z * z / (4.0 * n_f * n_f)).sqrt()) / denom;
    Ok((
        py_max(0.0, center - half_width),
        py_min(1.0, center + half_width),
    ))
}

/// True if the two Wilson intervals overlap — a conservative,
/// easy-to-explain check for "not distinguishable at this sample size".
/// Overlapping intervals should be read as "no significant difference
/// detected", not "proven equal".
pub fn proportions_overlap(
    successes_a: i64,
    n_a: i64,
    successes_b: i64,
    n_b: i64,
    z: f64,
) -> Result<bool, String> {
    let (lo_a, hi_a) = wilson_score_interval(successes_a, n_a, z)?;
    let (lo_b, hi_b) = wilson_score_interval(successes_b, n_b, z)?;
    Ok(!(hi_a < lo_b || hi_b < lo_a))
}

/// Two-sided z-test for a difference in two independent proportions.
/// Returns `(z_statistic, p_value)`; uses the pooled proportion for the
/// standard error under the null hypothesis p_a == p_b.
pub fn two_proportion_z_test(
    successes_a: i64,
    n_a: i64,
    successes_b: i64,
    n_b: i64,
) -> Result<(f64, f64), String> {
    if n_a <= 0 || n_b <= 0 {
        return Err("n_a and n_b must both be > 0".to_string());
    }
    let (p_a, p_b) = (
        successes_a as f64 / n_a as f64,
        successes_b as f64 / n_b as f64,
    );
    let p_pool = (successes_a + successes_b) as f64 / (n_a + n_b) as f64;
    let se = (p_pool * (1.0 - p_pool) * (1.0 / n_a as f64 + 1.0 / n_b as f64)).sqrt();
    if se == 0.0 {
        return Ok((0.0, 1.0));
    }
    let z = (p_a - p_b) / se;
    let p_value = 2.0 * (1.0 - standard_normal_cdf(z.abs()));
    Ok((z, p_value))
}

/// Phi(x) via the error function — no external dependency needed.
fn standard_normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / std::f64::consts::SQRT_2))
}

/// Rust's std doesn't expose `erf`; use the classic Abramowitz & Stegun
/// 7.1.26 rational approximation (|error| < 1.5e-7) — more than enough
/// for p-values reported next to n=20 solve rates.
fn erf(x: f64) -> f64 {
    // Portable implementation of the error function (SGPP / Cephes-style).
    let sign = if x < 0.0 { -1.0 } else { 1.0 };
    let x = x.abs();
    // constants
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;
    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();
    sign * y
}

/// Rough sample size needed for a Wilson-ish interval half-width <=
/// `margin` around a proportion near `p_estimate` (normal-approximation
/// formula, used only as a planning heuristic — Wilson intervals should
/// still be used for the actual reported interval).
pub fn required_n_for_margin(p_estimate: f64, margin: f64, z: f64) -> Result<i64, String> {
    if !(0.0 < p_estimate && p_estimate < 1.0) {
        return Err("p_estimate must be in (0, 1)".to_string());
    }
    if margin <= 0.0 {
        return Err(format!(
            "margin must be > 0, got {}",
            reasoning_common::py_float_str(margin)
        ));
    }
    let n = (z * z * p_estimate * (1.0 - p_estimate)) / (margin * margin);
    Ok(n.ceil() as i64)
}

#[inline]
fn py_min(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

#[inline]
fn py_max(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}
