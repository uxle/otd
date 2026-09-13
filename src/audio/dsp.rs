//! Segment-level acoustics — the WHERE layer (and shared DSP utilities).
//!
//! Distance is a heuristic: loudness (dBFS) + high-frequency energy ratio →
//! distance bands. Mic gain varies across devices, so this is an estimate,
//! not a measurement — exactly like the TS engine it replaces.

use crate::audio::fft::{Fft, FRAME_SIZE, TARGET_RATE};

pub const DISTANCE_BANDS: [&str; 5] = ["very-close", "close", "normal", "far", "very-far"];

#[derive(Debug, Clone)]
pub struct AcousticProfile {
    pub dbfs: f32,
    pub peak_dbfs: f32,
    pub hf_ratio: f32,
    pub pitch_hz: Option<f32>,
    pub distance_band: String,
    pub duration_ms: usize,
}

pub fn frame_rms(frame: &[f32]) -> f32 {
    if frame.is_empty() {
        return 0.0;
    }
    let sum: f32 = frame.iter().map(|v| v * v).sum();
    (sum / frame.len() as f32).sqrt()
}

pub fn frame_peak(frame: &[f32]) -> f32 {
    frame.iter().fold(0.0f32, |m, &v| m.max(v.abs()))
}

/// Spectral energy above 3.5 kHz relative to total energy, one 2048 frame.
/// Returns (hf_ratio, spectral_flatness, spectral_centroid).
pub fn frame_spectrum_features(
    fft: &Fft,
    frame: &[f32],
    sample_rate: f32,
) -> (f32, f32, f32) {
    if frame.len() != FRAME_SIZE {
        return (0.0, 1.0, 0.0);
    }
    let n = FRAME_SIZE;
    let mut re = vec![0f32; n];
    let mut im = vec![0f32; n];
    fft.hann_multiply(frame, &mut re);
    fft.forward(&mut re, &mut im);

    let half = n / 2;
    let bin_hz = sample_rate / n as f32;
    let cutoff_bin = (3500.0 / bin_hz).ceil().max(1.0) as usize;

    let mut hi = 0f64;
    let mut tot = 0f64;
    let mut log_sum = 0f64;
    let mut power_sum = 0f64;
    let mut weighted = 0f64;
    let mut count = 0usize;
    for b in 1..=half {
        let e = (re[b] * re[b] + im[b] * im[b]) as f64;
        if e <= 0.0 {
            continue;
        }
        tot += e;
        if b >= cutoff_bin {
            hi += e;
        }
        log_sum += e.max(1e-12).ln();
        power_sum += e;
        weighted += e * (b as f64 * bin_hz as f64);
        count += 1;
    }
    let hf = if tot > 0.0 { (hi / tot) as f32 } else { 0.0 };
    // flatness = geometric mean / arithmetic mean of the power spectrum
    let flatness = if count > 0 && power_sum > 0.0 {
        let geo = (log_sum / count as f64).exp();
        (geo / (power_sum / count as f64)).clamp(0.0, 1.0) as f32
    } else {
        1.0
    };
    let centroid = if power_sum > 0.0 {
        (weighted / power_sum) as f32
    } else {
        0.0
    };
    (hf, flatness, centroid)
}

/// Heuristic distance classification from loudness + treble presence.
pub fn classify_distance(dbfs: f32, hf_ratio: f32) -> &'static str {
    if dbfs >= -14.0 {
        if hf_ratio >= 0.03 {
            "very-close"
        } else {
            "close"
        }
    } else if dbfs >= -22.0 {
        if hf_ratio >= 0.12 {
            "close"
        } else {
            "normal"
        }
    } else if dbfs >= -30.0 {
        if hf_ratio >= 0.05 {
            "normal"
        } else {
            "far"
        }
    } else if dbfs >= -38.0 {
        "far"
    } else {
        "very-far"
    }
}

fn median_f32(values: &mut [f32]) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    values.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = values.len() / 2;
    if values.len() % 2 == 0 {
        (values[mid - 1] + values[mid]) / 2.0
    } else {
        values[mid]
    }
}

/// Full acoustic profile of one segment (16 kHz mono).
/// Pitch is the median across the strongest voiced frames — far more stable
/// for speaker clustering than a single-frame estimate.
pub fn analyze_segment(fft: &Fft, pcm: &[f32]) -> AcousticProfile {
    let sample_rate = TARGET_RATE;
    let mut rms_sum = 0f32;
    let mut rms_count = 0usize;
    let mut peak = 0f32;
    let mut hf_sum = 0f32;
    let min_frame_rms = 0.004f32;

    let mut by_loudness: Vec<(f32, usize)> = Vec::new(); // (rms, frame index)
    let mut frame_rms_cache: Vec<f32> = Vec::new();

    let mut off = 0usize;
    let mut frame_idx = 0usize;
    while off + FRAME_SIZE <= pcm.len() {
        let f = &pcm[off..off + FRAME_SIZE];
        let r = frame_rms(f);
        let p = frame_peak(f);
        if p > peak {
            peak = p;
        }
        if r > min_frame_rms {
            rms_sum += r;
            rms_count += 1;
            let (hf, _, _) = frame_spectrum_features(fft, f, sample_rate);
            hf_sum += hf;
        }
        if r > 0.01 {
            by_loudness.push((r, frame_idx));
        }
        frame_rms_cache.push(r);
        off += FRAME_SIZE;
        frame_idx += 1;
    }

    let avg_rms = if rms_count > 0 { rms_sum / rms_count as f32 } else { 0.0 };
    let dbfs = if avg_rms > 0.0 { 20.0 * avg_rms.log10() } else { -100.0 };
    let peak_dbfs = if peak > 0.0 { 20.0 * peak.log10() } else { -100.0 };
    let hf_ratio = if rms_count > 0 { hf_sum / rms_count as f32 } else { 0.0 };

    // pitch: median across the five loudest frames (YIN)
    by_loudness.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut pitches: Vec<f32> = Vec::new();
    for (_, idx) in by_loudness.iter().take(5) {
        let off = idx * FRAME_SIZE;
        let frame = &pcm[off..off + FRAME_SIZE];
        if let Some(f0) = crate::audio::pitch::estimate_pitch_yin(frame, sample_rate) {
            pitches.push(f0);
        }
    }
    let pitch_hz = if pitches.is_empty() {
        None
    } else {
        Some(median_f32(&mut pitches))
    };

    let duration_ms = ((pcm.len() as f32 * 1000.0) / sample_rate).round() as usize;

    AcousticProfile {
        dbfs: (dbfs * 10.0).round() / 10.0,
        peak_dbfs: (peak_dbfs * 10.0).round() / 10.0,
        hf_ratio: (hf_ratio * 1000.0).round() / 1000.0,
        pitch_hz: pitch_hz.map(|p| p.round()),
        distance_band: classify_distance(dbfs, hf_ratio).to_string(),
        duration_ms,
    }
}

/// Scale a cleaned segment to a healthy listening level for the audio model.
/// Quiet/distant speech gets boosted so the ASR hears every word; already-loud
/// audio passes through untouched (never make things worse).
pub fn normalize_loudness(pcm: &[f32], target_dbfs: f32, peak_ceiling: f32) -> Vec<f32> {
    let rms = frame_rms(pcm);
    if rms <= 1e-5 {
        return pcm.to_vec();
    }
    let mut gain = 10f32.powf(target_dbfs / 20.0) / rms;
    let peak = frame_peak(pcm);
    if peak > 0.0 && peak * gain > peak_ceiling {
        gain = peak_ceiling / peak;
    }
    if gain <= 1.0001 {
        return pcm.to_vec();
    }
    pcm.iter().map(|&v| v * gain).collect()
}
