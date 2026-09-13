//! Phase 120 — Biology: Hardy-Weinberg population genetics
//! (Rust port of python/science/bio_popgen_domain.py).
//!
//! The law (ground truth): in a population meeting the Hardy-Weinberg
//! conditions, allele frequencies p and q (p + q = 1) produce genotype
//! frequencies p^2 + 2pq + q^2 = 1, stable across generations.
//!
//! Independent cross-checks implemented here:
//!   - whatever route computes p and q, the genotype frequencies are ALSO
//!     computed by explicit allele-counting simulation (round-trip through
//!     a different representation),
//!   - the three genotype frequencies must sum to exactly 1 (tolerance
//!     stated explicitly),
//!   - a multi-generation stability check: applying the frequencies as
//!     mating outcomes reproduces the same genotype frequencies -- the
//!     actual content of the law.

use std::collections::HashMap;

use reasoning_common::Rat;

#[derive(Debug, Clone, PartialEq)]
pub struct HWResult {
    /// frequency of dominant allele
    pub p: f64,
    /// frequency of recessive allele
    pub q: f64,
    /// p2, 2pq, q2
    pub freqs: HashMap<String, f64>,
    pub verify_method: String,
    pub verify_values: HashMap<String, f64>,
}

fn _check_freqs(p: f64, q: f64) -> Result<HashMap<String, f64>, String> {
    if !(0.0 <= p && p <= 1.0 && 0.0 <= q && q <= 1.0) {
        return Err("allele frequencies must be within [0, 1]".to_string());
    }
    if (p + q - 1.0).abs() > 1e-9 {
        return Err("p + q must equal 1".to_string());
    }
    let (p2, pq, q2) = (p * p, 2.0 * p * q, q * q);
    let total = p2 + pq + q2;
    if (total - 1.0).abs() > 1e-9 {
        return Err("genotype frequencies must sum to 1".to_string());
    }
    let mut out = HashMap::new();
    out.insert("p^2".to_string(), p2);
    out.insert("2pq".to_string(), pq);
    out.insert("q^2".to_string(), q2);
    Ok(out)
}

/// Python `Fraction(x).limit_denominator(max_den)` (exact port of the
/// CPython 3.12 algorithm) for the value ranges this module sees
/// (probabilities in [0, 1]): exact continued-fraction convergents of the
/// binary double, best approximation with denominator <= max_den.
///
/// i128 intermediates: the exact binary ratio of a normal double >= 2^-73
/// has a power-of-two denominator <= 2^126, and at the loop break the
/// remainder numerator d <= 2^53 with (q0 + k*q1) <= max_den, so
/// `2*d*(q0 + k*q1)` always fits.
fn py_fraction_limit_denominator(x: f64, max_den: i64) -> Rat {
    if x == 0.0 {
        return Rat::from_int(0);
    }
    let neg = x < 0.0;
    let ax = x.abs();

    // exact decomposition ax = m * 2^e (m odd after stripping 2s, matching
    // Python's float.as_integer_ratio())
    let bits = ax.to_bits();
    let exp_field = ((bits >> 52) & 0x7ff) as i32;
    let frac = (bits & 0x000f_ffff_ffff_ffff) as i128;
    let (mut m, mut e): (i128, i32) = if exp_field == 0 {
        (frac, -1074) // subnormal
    } else {
        (frac | (1i128 << 52), exp_field - 1075)
    };
    if e < -126 {
        // |x| < 2^-73: far below 1/(2*max_den) for any max_den <= 10^9, so
        // the closest fraction with denominator <= max_den is 0
        return Rat::from_int(0);
    }
    if e > 60 {
        // |x| > 2^76: not reachable in this module; fall back to rounding
        return Rat::from_int(if neg {
            -(ax.round() as i64)
        } else {
            ax.round() as i64
        });
    }
    // strip common factors of two (as_integer_ratio normalizes)
    while e < 0 && m % 2 == 0 {
        m /= 2;
        e += 1;
    }

    let (num, den): (i128, i128) = if e >= 0 {
        (m << e, 1)
    } else {
        (m, 1i128 << (-e))
    };

    let sign = |r: Rat| if neg { -r } else { r };

    // Python fast path: already within the denominator bound
    if den <= max_den as i128 {
        return sign(Rat::new(num as i64, den as i64));
    }

    // continued fraction convergents (CPython fractions.limit_denominator)
    let mut p0: i128 = 0;
    let mut q0: i128 = 1;
    let mut p1: i128 = 1;
    let mut q1: i128 = 0;
    let mut n = num;
    let mut d = den;
    loop {
        let a = n / d;
        let q2 = q0 + a * q1;
        if q2 > max_den as i128 {
            break;
        }
        let (np0, nq0, np1, nq1) = (p1, q1, p0 + a * p1, q2);
        p0 = np0;
        q0 = nq0;
        p1 = np1;
        q1 = nq1;
        let (nn, nd) = (d, n - a * d);
        n = nn;
        d = nd;
        if d == 0 {
            // exact convergent reached (unreachable after the odd-m
            // normalization above, kept safe)
            return sign(Rat::new(p1 as i64, q1 as i64));
        }
    }
    let k = (max_den as i128 - q0) / q1;
    // Python: if 2*d*(q0+k*q1) <= den -> p1/q1 else (p0+k*p1)/(q0+k*q1)
    let (rp, rq) = if 2 * d * (q0 + k * q1) <= den {
        (p1, q1)
    } else {
        (p0 + k * p1, q0 + k * q1)
    };
    sign(Rat::new(rp as i64, rq as i64))
}

/// Independent verification path: count alleles from the genotype
/// frequencies (as you would for a real population census) and recover
/// p and q. Uses exact Fractions so no float error creeps in.
fn _allele_roundtrip(freqs: &HashMap<String, f64>) -> HashMap<String, f64> {
    let p2 = py_fraction_limit_denominator(freqs["p^2"], 10_i64.pow(9));
    let pq = py_fraction_limit_denominator(freqs["2pq"], 10_i64.pow(9));
    let q2 = py_fraction_limit_denominator(freqs["q^2"], 10_i64.pow(9));
    let total_alleles = (p2 + pq + q2) * Rat::from_int(2);
    let p_recovered = (p2 * Rat::from_int(2) + pq) / total_alleles;
    let q_recovered = (q2 * Rat::from_int(2) + pq) / total_alleles;
    let mut out = HashMap::new();
    out.insert("p".to_string(), p_recovered.to_f64());
    out.insert("q".to_string(), q_recovered.to_f64());
    out
}

/// Given the fraction of the population showing the recessive phenotype
/// (q^2), recover q = sqrt(q^2), p = 1 - q, and all genotype frequencies.
/// Verified by the allele-counting round-trip.
pub fn hardy_weinberg_from_recessive(q_squared: f64) -> Result<HWResult, String> {
    if !(0.0 < q_squared && q_squared <= 1.0) {
        return Err("q^2 (recessive phenotype fraction) must be in (0, 1]".to_string());
    }
    let q = q_squared.powf(0.5);
    let p = 1.0 - q;
    let freqs = _check_freqs(p, q)?;
    let rt = _allele_roundtrip(&freqs);
    if (rt["q"] - q).abs() > 1e-6 || (rt["p"] - p).abs() > 1e-6 {
        return Err("allele round-trip verification failed".to_string());
    }
    Ok(HWResult {
        p,
        q,
        freqs,
        verify_method: "allele-counting round-trip".to_string(),
        verify_values: rt,
    })
}

/// Given a raw allele census (counts of dominant vs recessive alleles),
/// p = dominant/(total), q = recessive/(total), then the usual law.
/// Verified by re-deriving the allele counts from the genotype freqs.
pub fn hardy_weinberg_from_allele_count(
    dominant_alleles: i64,
    recessive_alleles: i64,
) -> Result<HWResult, String> {
    if dominant_alleles < 0 || recessive_alleles < 0 {
        return Err("allele counts cannot be negative".to_string());
    }
    let total = dominant_alleles + recessive_alleles;
    if total == 0 {
        return Err("no alleles given".to_string());
    }
    let p = dominant_alleles as f64 / total as f64;
    let q = recessive_alleles as f64 / total as f64;
    let freqs = _check_freqs(p, q)?;
    let p_back = freqs["p^2"] + freqs["2pq"] / 2.0;
    let q_back = freqs["q^2"] + freqs["2pq"] / 2.0;
    if (p_back * total as f64 - dominant_alleles as f64).abs() > 1e-6
        || (q_back * total as f64 - recessive_alleles as f64).abs() > 1e-6
    {
        return Err("allele count reconstruction failed".to_string());
    }
    let mut verify_values = HashMap::new();
    verify_values.insert("p".to_string(), p_back);
    verify_values.insert("q".to_string(), q_back);
    Ok(HWResult {
        p,
        q,
        freqs,
        verify_method: "allele count reconstruction".to_string(),
        verify_values,
    })
}

/// The actual predictive content of the law: after random mating, the
/// genotype frequencies are p^2 : 2pq : q^2 EVERY generation. This walks
/// `generations` rounds of allele reconstruction -> genotype frequencies
/// and requires the frequencies to be stable -- if they drift, the
/// population is NOT in Hardy-Weinberg equilibrium and this refuses.
/// (Python default `generations: int = 2`.)
pub fn hardy_weinberg_stability(p: f64, q: f64, generations: i32) -> Result<HWResult, String> {
    let mut freqs = _check_freqs(p, q)?;
    let mut history: Vec<HashMap<String, f64>> = vec![freqs.clone()];
    let mut cur_p = p;
    let mut cur_q = q;
    for _ in 0..(generations.max(1) - 1) {
        let p_next = freqs["p^2"] + freqs["2pq"] / 2.0;
        let q_next = freqs["q^2"] + freqs["2pq"] / 2.0;
        freqs = _check_freqs(p_next, q_next)?;
        history.push(freqs.clone());
        cur_p = p_next;
        cur_q = q_next;
    }
    for (i, f) in history.iter().enumerate() {
        for (k, v) in f {
            if (v - history[0][k]).abs() > 1e-9 {
                return Err(format!(
                    "instability detected at generation {}: population is not \
                     in Hardy-Weinberg equilibrium (frequencies drifted)",
                    i
                ));
            }
        }
    }
    let last = history.last().expect("history never empty").clone();
    Ok(HWResult {
        p: cur_p,
        q: cur_q,
        freqs: last.clone(),
        verify_method: "multi-generation stability walk".to_string(),
        verify_values: last,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn limit_denominator_matches_python() {
        // pinned against Python:
        // Fraction(0.36).limit_denominator(10**9) -> 9/25
        // Fraction(0.48).limit_denominator(10**9) -> 12/25
        // Fraction(0.16000000000000003).limit_denominator(10**9) -> 4/25
        // Fraction(0.48000000000000004).limit_denominator(10**9) -> 12/25
        // Fraction(1.0).limit_denominator(10**9) -> 1
        // Fraction(0.5).limit_denominator(10**9) -> 1/2
        // Fraction(0.3333333333333333).limit_denominator(10**9) -> 1/3
        assert_eq!(
            py_fraction_limit_denominator(0.36, 10_i64.pow(9)),
            Rat::new(9, 25)
        );
        assert_eq!(
            py_fraction_limit_denominator(0.48, 10_i64.pow(9)),
            Rat::new(12, 25)
        );
        assert_eq!(
            py_fraction_limit_denominator(0.16000000000000003, 10_i64.pow(9)),
            Rat::new(4, 25)
        );
        assert_eq!(
            py_fraction_limit_denominator(0.48000000000000004, 10_i64.pow(9)),
            Rat::new(12, 25)
        );
        assert_eq!(
            py_fraction_limit_denominator(1.0, 10_i64.pow(9)),
            Rat::from_int(1)
        );
        assert_eq!(
            py_fraction_limit_denominator(0.5, 10_i64.pow(9)),
            Rat::new(1, 2)
        );
        assert_eq!(
            py_fraction_limit_denominator(0.3333333333333333, 10_i64.pow(9)),
            Rat::new(1, 3)
        );
    }
}
