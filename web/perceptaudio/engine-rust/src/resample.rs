//! Sample-rate conversion.
//!
//! Upgrades the TypeScript linear interpolator to Catmull-Rom cubic
//! interpolation — noticeably cleaner highs when decimating 44.1/48 kHz mic
//! audio down to the 16 kHz analysis rate (less aliasing, less dulling).

/// Resample a mono signal with a Catmull-Rom cubic interpolator.
/// Falls back to the identity when rates already match.
pub fn resample(input: &[f32], from_rate: f32, to_rate: f32) -> Vec<f32> {
    if (from_rate - to_rate).abs() < 1e-6 || input.is_empty() {
        return input.to_vec();
    }

    let ratio = to_rate / from_rate;
    let out_len = ((input.len() as f32) * ratio).floor().max(1.0) as usize;
    let mut out = vec![0f32; out_len];
    let last = input.len() - 1;

    for (i, o) in out.iter_mut().enumerate() {
        let pos = (i as f32) / ratio;
        let i1 = pos.floor() as usize;
        let t = pos - i1 as f32;

        let p0 = input[i1.saturating_sub(1)];
        let p1 = input[i1.min(last)];
        let p2 = input[(i1 + 1).min(last)];
        let p3 = input[(i1 + 2).min(last)];

        // Catmull-Rom spline (uniform, tension 0.5)
        *o = p1
            + 0.5 * t * (p2 - p0 + t * (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3
                + t * (3.0 * (p1 - p2) + p3 - p0)));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity() {
        let x = vec![0.1, 0.2, 0.3];
        assert_eq!(resample(&x, 16000.0, 16000.0), x);
    }

    #[test]
    fn length_matches_ratio() {
        let x = vec![0.0f32; 1600];
        let y = resample(&x, 44100.0, 16000.0);
        assert!((y.len() as f32 - 1600.0 * 16000.0 / 44100.0).abs() <= 1.0);
    }
}
