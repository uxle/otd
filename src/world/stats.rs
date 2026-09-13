//! P2220 — OTD3 STATISTICS & PROBABILITY — the science of uncertainty.
//!
//! A self-made statistics library: descriptive measures, four distributions
//! with exact PMFs/PDFs, hypothesis tests, and least-squares regression.
//! `simulate: stats` turns the scene itself into a dataset — masses,
//! volumes, densities — and reads it back the way a scientist would:
//! centre, spread, shape, correlation, and the honest line of best fit.
//!
//! Laws carried here:
//!   Law of Large Numbers — more samples ⇒ the mean settles toward truth.
//!   Central Limit Theorem — averaged noise goes normal, always, whatever
//!   the shape of the noise itself.

use super::eval::{ConsoleLine, LineKind, World};
use crate::rng::Rng;

// ─────────────────────────────────────────────────────────────────────────────
// Descriptive statistics
// ─────────────────────────────────────────────────────────────────────────────

pub fn mean(xs: &[f64]) -> f64 {
    if xs.is_empty() { 0.0 } else { xs.iter().sum::<f64>() / xs.len() as f64 }
}

/// The middle value (median of the sorted copy).
pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    let mut s = xs.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = s.len();
    if n % 2 == 1 {
        s[n / 2]
    } else {
        (s[n / 2 - 1] + s[n / 2]) / 2.0
    }
}

/// Population variance (divide by n).
pub fn variance(xs: &[f64]) -> f64 {
    if xs.len() < 2 {
        return 0.0;
    }
    let m = mean(xs);
    xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / xs.len() as f64
}

/// Sample standard deviation (the usual reported spread).
pub fn stddev(xs: &[f64]) -> f64 {
    variance(xs).sqrt()
}

/// Skewness — the third standardised moment. Negative: left tail.
pub fn skewness(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 3 {
        return 0.0;
    }
    let m = mean(xs);
    let s = stddev(xs);
    if s == 0.0 {
        return 0.0;
    }
    xs.iter().map(|x| ((x - m) / s).powi(3)).sum::<f64>() / n as f64
}

/// Excess kurtosis — fourth standardised moment minus 3 (0 = normal).
pub fn kurtosis(xs: &[f64]) -> f64 {
    let n = xs.len();
    if n < 4 {
        return 0.0;
    }
    let m = mean(xs);
    let s = stddev(xs);
    if s == 0.0 {
        return 0.0;
    }
    xs.iter().map(|x| ((x - m) / s).powi(4)).sum::<f64>() / n as f64 - 3.0
}

/// Pearson correlation coefficient, in [−1, 1].
pub fn pearson(xs: &[f64], ys: &[f64]) -> f64 {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return 0.0;
    }
    let (mx, my) = (mean(&xs[..n]), mean(&ys[..n]));
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    let mut syy = 0.0;
    for i in 0..n {
        let (dx, dy) = (xs[i] - mx, ys[i] - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx == 0.0 || syy == 0.0 {
        return 0.0;
    }
    sxy / (sxx * syy).sqrt()
}

/// Least-squares line y = intercept + slope·x, with r².
pub fn linreg(xs: &[f64], ys: &[f64]) -> (f64, f64, f64) {
    let n = xs.len().min(ys.len());
    if n < 2 {
        return (0.0, 0.0, 0.0);
    }
    let (mx, my) = (mean(&xs[..n]), mean(&ys[..n]));
    let mut sxy = 0.0;
    let mut sxx = 0.0;
    for i in 0..n {
        sxy += (xs[i] - mx) * (ys[i] - my);
        sxx += (xs[i] - mx) * (xs[i] - mx);
    }
    if sxx == 0.0 {
        return (my, 0.0, 0.0);
    }
    let slope = sxy / sxx;
    let intercept = my - slope * mx;
    let r = pearson(xs, ys);
    (intercept, slope, r * r)
}

// ─────────────────────────────────────────────────────────────────────────────
// Special functions
// ─────────────────────────────────────────────────────────────────────────────

/// Lanczos log-gamma (g=7, n=9) — exact enough for PMFs at OTD scale.
pub fn ln_gamma(x: f64) -> f64 {
    const G: [f64; 9] = [
        0.999_999_999_999_809_93,
        676.520_368_121_885_1,
        -1_259.139_216_722_402_8,
        771.323_428_777_653_13,
        -176.615_029_162_140_59,
        12.507_343_278_686_905,
        -0.138_571_095_265_720_12,
        9.984_369_578_019_571_6e-6,
        1.505_632_735_149_311_6e-7,
    ];
    if x < 0.5 {
        // reflection: Γ(x)Γ(1−x) = π/sin(πx)
        (std::f64::consts::PI / (std::f64::consts::PI * x).sin()).ln() - ln_gamma(1.0 - x)
    } else {
        let x = x - 1.0;
        let mut a = G[0];
        let t = x + 7.5;
        for (i, g) in G.iter().enumerate().skip(1) {
            a += g / (x + i as f64);
        }
        0.5 * (2.0 * std::f64::consts::PI).ln() + (x + 0.5) * t.ln() - t + a.ln()
    }
}

pub fn ln_fact(n: u64) -> f64 {
    ln_gamma(n as f64 + 1.0)
}

pub fn ln_choose(n: u64, k: u64) -> f64 {
    if k > n {
        return f64::NEG_INFINITY;
    }
    ln_fact(n) - ln_fact(k) - ln_fact(n - k)
}

// ─────────────────────────────────────────────────────────────────────────────
// Distributions
// ─────────────────────────────────────────────────────────────────────────────

/// Standard normal PDF φ(z).
pub fn normal_pdf(z: f64) -> f64 {
    (-0.5 * z * z).exp() / (2.0 * std::f64::consts::PI).sqrt()
}

/// Standard normal CDF Φ(z) — Zelen & Severo's rational approximation,
/// |error| < 7.5e-8 (the classic Abramowitz & Stegun 26.2.17).
pub fn normal_cdf(z: f64) -> f64 {
    let (sign, z) = if z < 0.0 { (-1.0, -z) } else { (1.0, z) };
    let t = 1.0 / (1.0 + 0.231_641_9 * z);
    let poly = 0.319_381_530
        * t
        + (-0.356_563_782) * t.powi(2)
        + 1.781_477_937 * t.powi(3)
        + (-1.821_255_978) * t.powi(4)
        + 1.330_274_429 * t.powi(5);
    let p = 1.0 - normal_pdf(z) * poly;
    0.5 * (1.0 + sign * (2.0 * p - 1.0).clamp(0.0, 1.0))
}

/// Inverse standard normal CDF — Acklam's algorithm (|relative error|
/// < 1.15e-9 before refinement), with one Halley polishing step.
pub fn normal_inv_cdf(p: f64) -> f64 {
    let p = p.clamp(1e-300, 1.0 - 1e-16);
    let plow = 0.024_25;
    let a: [f64; 6] = [
        -3.969_683_028_665_376e1, 2.209_460_984_245_205e2, -2.759_285_104_469_687e2,
        1.383_577_518_672_69e2, -3.066_479_806_614_716e1, 2.506_628_277_459_239,
    ];
    let b: [f64; 5] = [
        -5.447_609_879_822_406e1, 1.615_858_368_580_409e2, -1.556_989_798_598_866e2,
        6.680_131_188_771_972e1, -1.328_068_155_288_572e1,
    ];
    let c: [f64; 6] = [
        -7.784_894_002_430_293e-3, -3.223_964_580_411_365e-1, -2.400_758_277_161_838,
        -2.549_732_539_343_734, 4.374_664_141_464_968, 2.938_163_982_698_783,
    ];
    let d: [f64; 4] = [
        7.784_695_709_041_462e-3, 3.224_671_290_700_398e-1, 2.445_134_137_142_996,
        3.754_408_661_907_416,
    ];
    let x = if p < plow {
        let q = (-2.0 * p.ln()).sqrt();
        (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else if p > 1.0 - plow {
        let q = (-2.0 * (1.0 - p).ln()).sqrt();
        -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    } else {
        let q = p - 0.5;
        let r = q * q;
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
            / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)
    };
    // Halley refinement drives the round-trip to ~1e-12
    let e = normal_cdf(x) - p;
    let u = e * (2.0 * std::f64::consts::PI).sqrt() * (x * x / 2.0).exp();
    x - u / (1.0 + x * u / 2.0)
}

/// Binomial PMF: P(X = k) for n trials at probability p.
pub fn binomial_pmf(n: u64, k: u64, p: f64) -> f64 {
    if k > n || p < 0.0 || p > 1.0 {
        return 0.0;
    }
    (ln_choose(n, k) + (k as f64) * p.ln() + ((n - k) as f64) * (1.0 - p).ln()).exp()
}

/// Poisson PMF: P(X = k) at rate λ.
pub fn poisson_pmf(lambda: f64, k: u64) -> f64 {
    if lambda <= 0.0 {
        return if k == 0 { 1.0 } else { 0.0 };
    }
    (-lambda + k as f64 * lambda.ln() - ln_fact(k)).exp()
}

/// Exponential PDF at x, rate λ.
pub fn exponential_pdf(x: f64, lambda: f64) -> f64 {
    if x < 0.0 || lambda <= 0.0 {
        0.0
    } else {
        lambda * (-lambda * x).exp()
    }
}

/// Draw a standard normal sample (Box–Muller on the seeded PRNG).
pub fn sample_normal(rng: &mut Rng) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Draw a binomial sample by direct Bernoulli counting (n is small here).
pub fn sample_binomial(n: u64, p: f64, rng: &mut Rng) -> u64 {
    (0..n).filter(|_| rng.next_f64() < p).count() as u64
}

// ─────────────────────────────────────────────────────────────────────────────
// Hypothesis tests
// ─────────────────────────────────────────────────────────────────────────────

/// Two-sided z-test of the mean against μ₀: returns (z, p-value).
/// Uses the sample standard deviation (large-sample approximation).
pub fn z_test(xs: &[f64], mu0: f64) -> (f64, f64) {
    let n = xs.len() as f64;
    if n < 2.0 {
        return (0.0, 1.0);
    }
    let m = mean(xs);
    let s = stddev(xs);
    if s == 0.0 {
        return (0.0, 1.0);
    }
    let z = (m - mu0) / (s / n.sqrt());
    let p = 2.0 * (1.0 - normal_cdf(z.abs()));
    (z, p)
}

// ─────────────────────────────────────────────────────────────────────────────
// simulate: stats — read the scene like a dataset
// ─────────────────────────────────────────────────────────────────────────────

pub fn stats_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    let parts: Vec<_> = world.parts.iter().filter(|p| !p.hidden).collect();
    if parts.len() < 2 {
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: "STATISTICS needs at least two parts to have something to say — add more shapes and ask again".into(),
        });
        return out;
    }
    let masses: Vec<f64> = parts.iter().map(|p| p.mass_g).collect();
    let vols: Vec<f64> = parts.iter().map(|p| p.volume_mm3 / 1000.0).collect(); // cm³
    let densities: Vec<f64> = parts
        .iter()
        .map(|p| {
            if p.volume_mm3 > 0.0 {
                (p.mass_g / 1000.0) / (p.volume_mm3 / 1e9)
            } else {
                0.0
            }
        })
        .collect();

    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!("STATISTICAL PORTRAIT — {} parts as a dataset", parts.len()),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  mass: mean {:.1} g, median {:.1} g, σ {:.1} g, range {:.1}…{:.1} g",
            mean(&masses),
            median(&masses),
            stddev(&masses),
            masses.iter().cloned().fold(f64::INFINITY, f64::min),
            masses.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        ),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  volume: mean {:.2} cm³, σ {:.2} cm³, skewness {:+.2} ({} tail), kurtosis {:+.2} ({} normal)",
            mean(&vols),
            stddev(&vols),
            skewness(&vols),
            if skewness(&vols) >= 0.0 { "right" } else { "left" },
            kurtosis(&vols),
            if kurtosis(&vols).abs() < 0.5 { "≈" } else { "not" },
        ),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  density: mean {:.0} kg/m³, median {:.0} kg/m³ — {} float on water (ρ < 1000)",
            mean(&densities),
            median(&densities),
            densities.iter().filter(|&&d| d < crate::units::WATER_DENSITY).count(),
        ),
    });

    // mass ↔ volume: the materials' fingerprint
    let r = pearson(&vols, &masses);
    let (b0, b1, r2) = linreg(&vols, &masses);
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  correlation mass ↔ volume: r = {:.3} — the line of best fit is mass ≈ {:.2} g + {:.3} g/cm³ × volume (r² = {:.3})",
            r, b0, b1, r2
        ),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: "  (that slope is the scene's effective density — identical materials make r = 1.000, mixtures pull it down)".into(),
    });

    // the Central Limit Theorem, live: means of random part samples go normal
    let mut rng = Rng::new(11);
    let k = 5.min(parts.len());
    let means: Vec<f64> = (0..2000)
        .map(|_| {
            let s: Vec<f64> = (0..k)
                .map(|_| masses[(rng.next_f64() * masses.len() as f64) as usize % masses.len()])
                .collect();
            mean(&s)
        })
        .collect();
    let clt_sigma = stddev(&masses) / (k as f64).sqrt();
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  Central Limit Theorem, live: mean-of-{} masses over 2000 resamples — σ measured {:.1} g, σ predicted σ/√{} = {:.1} g — noise averages into the bell curve whatever its shape",
            k, stddev(&means), k, clt_sigma
        ),
    });

    // z-test: is this scene's mean density distinguishable from water?
    let (z, p) = z_test(&densities, crate::units::WATER_DENSITY);
    let verdict = if p < 0.05 {
        "significantly NOT water-like (p < 0.05)"
    } else {
        "indistinguishable from water at the 5% level"
    };
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!("  z-test vs ρ = 1000 kg/m³: z = {:+.2}, p = {:.3} — {}", z, p, verdict),
    });
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "the numbers above are the scene reading itself — statistics is just physics with uncertainty admitted".into(),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn descriptive_basics() {
        let xs = [2.0, 4.0, 4.0, 4.0, 5.0, 5.0, 7.0, 9.0];
        assert_eq!(mean(&xs), 5.0);
        assert_eq!(median(&xs), 4.5);
        assert!((variance(&xs) - 4.0).abs() < 1e-12);
        assert!((stddev(&xs) - 2.0).abs() < 1e-12);
    }

    #[test]
    fn normal_cdf_known_values() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 1e-9);
        assert!((normal_cdf(1.959_964) - 0.975).abs() < 1e-6);
        assert!((normal_cdf(-1.0) - 0.158_655).abs() < 1e-6);
        assert!(normal_cdf(8.0) > 0.999_999);
    }

    #[test]
    fn inverse_cdf_roundtrips() {
        for &p in &[0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999] {
            let z = normal_inv_cdf(p);
            let back = normal_cdf(z);
            assert!((back - p).abs() < 1e-8, "p={} → z={} → {}", p, z, back);
        }
    }

    #[test]
    fn binomial_and_poisson_known_values() {
        assert!((binomial_pmf(10, 5, 0.5) - 0.246_093_75).abs() < 1e-9);
        assert!((binomial_pmf(10, 0, 0.5) - (0.5f64).powi(10)).abs() < 1e-12);
        assert!((poisson_pmf(3.0, 3) - 0.224_041_807_655_555).abs() < 1e-9);
        assert!((poisson_pmf(1.0, 0) - 1.0 / std::f64::consts::E).abs() < 1e-12);
    }

    #[test]
    fn gamma_matches_factorials() {
        assert!((ln_gamma(1.0)).abs() < 1e-10);
        assert!((ln_fact(10) - (3_628_800.0f64).ln()).abs() < 1e-10);
        assert!((ln_choose(10, 5) - 252.0f64.ln()).abs() < 1e-10);
    }

    #[test]
    fn regression_recovers_a_line() {
        let mut rng = Rng::new(3);
        let xs: Vec<f64> = (0..50).map(|i| i as f64).collect();
        let ys: Vec<f64> = xs.iter().map(|&x| 2.0 * x + 1.0 + 0.05 * sample_normal(&mut rng)).collect();
        let (b0, b1, r2) = linreg(&xs, &ys);
        assert!((b0 - 1.0).abs() < 0.1, "intercept {}", b0);
        assert!((b1 - 2.0).abs() < 0.01, "slope {}", b1);
        assert!(r2 > 0.999, "r² {}", r2);
    }

    #[test]
    fn pearson_detects_perfect_correlation() {
        let xs = vec![1.0, 2.0, 3.0, 4.0];
        let ys = vec![10.0, 20.0, 30.0, 40.0];
        assert!((pearson(&xs, &ys) - 1.0).abs() < 1e-12);
        let zs = vec![10.0, 20.0, 30.0, 5.0];
        assert!(pearson(&xs, &zs) < 0.9);
    }

    #[test]
    fn clt_sigma_shrinks_like_sqrt_n() {
        let mut rng = Rng::new(7);
        let base: Vec<f64> = (0..2000).map(|_| rng.next_f64()).collect(); // uniform
        let sigma1 = stddev(&base);
        let means: Vec<f64> = (0..2000)
            .map(|_| {
                let s: Vec<f64> = (0..16).map(|_| base[(rng.next_f64() * 2000.0) as usize]).collect();
                mean(&s)
            })
            .collect();
        let sigma16 = stddev(&means);
        let ratio = sigma1 / sigma16;
        assert!((ratio - 4.0).abs() < 0.6, "σ₁/σ₁₆ = {} (want ≈ √16 = 4)", ratio);
    }

    #[test]
    fn z_test_rejects_a_moved_mean() {
        let mut rng = Rng::new(9);
        let xs: Vec<f64> = (0..500).map(|_| 10.0 + sample_normal(&mut rng)).collect();
        let (z, p) = z_test(&xs, 12.0);
        assert!(z < -10.0);
        assert!(p < 1e-10);
        let (_, p_same) = z_test(&xs, 10.0);
        assert!(p_same > 0.01);
    }

    #[test]
    fn exponential_pdf_integrates_to_one() {
        // ∫λe^(−λx)dx = 1 − e^(−λ·hi)
        let (lambda, hi) = (0.7, 30.0);
        let mut acc = 0.0;
        let steps = 60_000;
        let dx = hi / steps as f64;
        for i in 0..steps {
            acc += exponential_pdf(i as f64 * dx, lambda) * dx;
        }
        assert!((acc - (1.0 - (-lambda * hi).exp())).abs() < 1e-3);
    }
}
