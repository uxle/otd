//! Fundamental-frequency (F0) estimation — the WHO layer's voice fingerprint.
//!
//! Upgrade over the TypeScript autocorrelation: the **YIN algorithm**
//! (de Cheveigné & Kawahara, 2002):
//!   1. difference function d(tau) = Σ (x[j] − x[j+tau])²
//!   2. cumulative-mean-normalized difference d'(tau)
//!   3. absolute-threshold dip search with local-minimum descent
//!   4. parabolic interpolation for sub-sample period accuracy
//!
//! YIN resists octave errors far better than peak-picking on the raw
//! autocorrelation, which makes the speaker fingerprints more stable.

/// Estimate the pitch of one analysis frame. Returns Hz or `None` when the
/// frame is too quiet or has no confident periodicity.
pub fn estimate_pitch_yin(frame: &[f32], sample_rate: f32) -> Option<f32> {
    let n = frame.len();
    let w = n / 2; // integration window (1024 @ 2048-frame)
    if w < 64 {
        return None;
    }

    let min_tau = ((sample_rate / 400.0).floor() as usize).max(2); // 400 Hz ceiling
    let max_tau = ((sample_rate / 70.0).floor() as usize).min(w - 1); // 70 Hz floor
    if min_tau >= max_tau {
        return None;
    }

    // one extra lag (max_tau + 1) is computed so the parabolic interpolation
    // can look at d[tau+1] without branching
    let used = w + max_tau + 1;
    if used > n {
        return None;
    }

    // remove DC over the region the algorithm touches
    let mut mean = 0f64;
    for &v in &frame[..used] {
        mean += v as f64;
    }
    mean /= used as f64;

    let mut x = vec![0f32; used];
    let mut energy = 0f64;
    for i in 0..used {
        let v = frame[i] as f64 - mean;
        x[i] = v as f32;
        if i < w {
            energy += v * v;
        }
    }
    // voicing energy gate — mirrors the TS engine's 1e-4 mean-square floor
    if energy / (w as f64) < 1e-4 {
        return None;
    }

    // difference function for all lags 1..=max_tau (needed by the CMND)
    let mut d = vec![0f64; max_tau + 2];
    for tau in 1..=max_tau + 1 {
        let mut sum = 0f64;
        for j in 0..w {
            let diff = (x[j] - x[j + tau]) as f64;
            sum += diff * diff;
        }
        d[tau] = sum;
    }

    // cumulative-mean-normalized difference
    let mut cmnd = vec![1f64; max_tau + 2];
    let mut running = 0f64;
    for tau in 1..=max_tau + 1 {
        running += d[tau];
        let avg = running / tau as f64;
        cmnd[tau] = if avg > f64::EPSILON { d[tau] / avg } else { 1.0 };
    }

    // absolute-threshold search: first dip below 0.15, then descend to the
    // local minimum (classic YIN). Fallback: global minimum if it is decent.
    const THRESHOLD: f64 = 0.15;
    const MAX_ACCEPT: f64 = 0.65;
    let mut tau_est: Option<usize> = None;
    let mut tau = min_tau;
    while tau <= max_tau {
        if cmnd[tau] < THRESHOLD {
            while tau + 1 <= max_tau && cmnd[tau + 1] < cmnd[tau] {
                tau += 1;
            }
            tau_est = Some(tau);
            break;
        }
        tau += 1;
    }
    let tau = match tau_est {
        Some(t) => t,
        None => {
            // no dip under threshold — take the global minimum in range only
            // if it is still reasonably confident
            let mut best = min_tau;
            for t in min_tau..=max_tau {
                if cmnd[t] < cmnd[best] {
                    best = t;
                }
            }
            if cmnd[best] <= MAX_ACCEPT {
                best
            } else {
                return None;
            }
        }
    };

    // parabolic interpolation on the difference function for sub-lag precision
    let t0 = d[tau.saturating_sub(1)];
    let t1 = d[tau];
    let t2 = d[(tau + 1).min(max_tau + 1)];
    let denom = t0 + t2 - 2.0 * t1;
    let shifted = if denom.abs() > f64::EPSILON {
        let offset = (t0 - t2) / (2.0 * denom);
        tau as f64 + offset.clamp(-1.0, 1.0)
    } else {
        tau as f64
    };

    let f0 = sample_rate / shifted as f32;
    if !(60.0..=420.0).contains(&f0) {
        return None;
    }
    Some(f0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_sine_pitch() {
        // 150 Hz sine at 16 kHz
        let frame: Vec<f32> = (0..2048)
            .map(|i| (2.0 * std::f32::consts::PI * 150.0 * (i as f32) / 16000.0).sin())
            .collect();
        let f0 = estimate_pitch_yin(&frame, 16000.0).expect("voiced sine");
        assert!((f0 - 150.0).abs() < 2.0, "got {f0}");
    }

    #[test]
    fn rejects_noise_floor() {
        let frame = vec![1e-6f32; 2048];
        assert!(estimate_pitch_yin(&frame, 16000.0).is_none());
    }
}
