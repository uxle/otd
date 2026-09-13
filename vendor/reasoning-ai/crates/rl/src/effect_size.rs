//! Phase 081 — Effect size and power analysis for proportion comparisons
//! (design doc 24): Cohen's h and required-N-for-power, so "how big would
//! the effect need to be to detect" is computable before the experiment
//! runs, not after.

/// Cohen's h: effect size for a difference between two proportions via
/// the arcsine-square-root transform (variance-stabilizing, so h is
/// comparable across the whole [0,1] range unlike a raw difference).
///
/// Conventional benchmarks (Cohen 1988): 0.2 small, 0.5 medium, 0.8 large.
pub fn cohens_h(p1: f64, p2: f64) -> Result<f64, String> {
    if !((0.0..=1.0).contains(&p1) && (0.0..=1.0).contains(&p2)) {
        return Err(format!(
            "proportions must be in [0,1], got {}, {}",
            reasoning_common::py_float_str(p1),
            reasoning_common::py_float_str(p2)
        ));
    }
    Ok(2.0 * p1.sqrt().asin() - 2.0 * p2.sqrt().asin())
}

/// Threshold label for Cohen's h (sign-insensitive).
pub fn effect_size_label(h: f64) -> &'static str {
    let ah = h.abs();
    if ah < 0.2 {
        "negligible"
    } else if ah < 0.5 {
        "small"
    } else if ah < 0.8 {
        "medium"
    } else {
        "large"
    }
}

/// Approximate required sample size PER GROUP to detect a difference
/// between p1 and p2 at the given significance level and power, using the
/// standard two-proportion normal-approximation formula. A planning
/// estimate, not a substitute for the exact Wilson-based interval that
/// should be reported once data exists (Phase 074).
pub fn required_n_for_power(
    p1: f64,
    p2: f64,
    power: f64,
    alpha: f64,
) -> Result<i64, String> {
    if p1 == p2 {
        return Err("p1 and p2 must differ to compute a required sample size".to_string());
    }
    if !(0.0 < power && power < 1.0) {
        return Err(format!(
            "power must be in (0,1), got {}",
            reasoning_common::py_float_str(power)
        ));
    }
    if !(0.0 < alpha && alpha < 1.0) {
        return Err(format!(
            "alpha must be in (0,1), got {}",
            reasoning_common::py_float_str(alpha)
        ));
    }

    let z_alpha = inverse_standard_normal_cdf(1.0 - alpha / 2.0)?;
    let z_beta = inverse_standard_normal_cdf(power)?;
    let p_bar = (p1 + p2) / 2.0;

    let term1 = z_alpha * (2.0 * p_bar * (1.0 - p_bar)).sqrt();
    let term2 = z_beta * (p1 * (1.0 - p1) + p2 * (1.0 - p2)).sqrt();
    let n = ((term1 + term2) * (term1 + term2)) / ((p1 - p2) * (p1 - p2));
    Ok(n.ceil() as i64)
}

/// Acklam's rational approximation to the inverse standard normal CDF
/// (probit function), accurate to ~1e-9. Port of the Python module's
/// `_inverse_standard_normal_cdf` (made public because the Phase 081
/// tests exercise it directly, as they did in Python).
pub fn inverse_standard_normal_cdf(p: f64) -> Result<f64, String> {
    if !(0.0 < p && p < 1.0) {
        return Err(format!(
            "p must be in (0,1), got {}",
            reasoning_common::py_float_str(p)
        ));
    }
    let a: [f64; 6] = [
        -3.969683028665376e+01,
        2.209460984245205e+02,
        -2.759285104469687e+02,
        1.383577518672690e+02,
        -3.066479806614716e+01,
        2.506628277459239e+00,
    ];
    let b: [f64; 5] = [
        -5.447609879822406e+01,
        1.615858368580409e+02,
        -1.556989798598866e+02,
        6.680131188771972e+01,
        -1.328068155288572e+01,
    ];
    let c: [f64; 6] = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
        4.374664141464968e+00,
        2.938163982698783e+00,
    ];
    let d: [f64; 4] = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e+00,
        3.754408661907416e+00,
    ];

    let p_low = 0.02425;
    let p_high = 1.0 - p_low;

    if p < p_low {
        let q = (-2.0 * p.ln()).sqrt();
        Ok((((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    } else if p <= p_high {
        let q = p - 0.5;
        let r = q * q;
        Ok((((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0))
    } else {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        Ok(-(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0))
    }
}
