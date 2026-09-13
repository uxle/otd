//! Log-band spectrogram for the UI — a new layer the TS engine never had.
//!
//! Each cleaned utterance is rendered to a small time × frequency heat map
//! (48 log-spaced bands, 80 Hz – 7.5 kHz) normalized to 0..1. The frontend
//! paints it under every perception card, so you can literally *see* the
//! voice the engine just heard.

use crate::fft::{Fft, FRAME_SIZE};

pub const SPEC_ROWS: usize = 48;
pub const SPEC_MAX_COLS: usize = 96;
const SPEC_MIN_HZ: f32 = 80.0;
const SPEC_MAX_HZ: f32 = 7500.0;
const SPEC_FLOOR_DB: f32 = -70.0;

/// Returns (cols, row-major-per-column data: `data[col * ROWS + row]`).
pub fn spectrogram(fft: &Fft, pcm: &[f32], sample_rate: f32) -> (usize, Vec<f32>) {
    let usable = pcm.len().max(FRAME_SIZE);
    let cols = ((usable / 1024) as usize).clamp(1, SPEC_MAX_COLS);
    let hop = (usable - FRAME_SIZE) / cols.max(1) + 1;

    let half = FRAME_SIZE / 2;
    let bin_hz = sample_rate / FRAME_SIZE as f32;
    let min_bin = (SPEC_MIN_HZ / bin_hz).floor().max(1.0) as usize;
    let max_bin = ((SPEC_MAX_HZ / bin_hz).ceil() as usize).min(half);

    // log-spaced band edges between min_bin and max_bin
    let mut edges = vec![0f32; SPEC_ROWS + 1];
    for r in 0..=SPEC_ROWS {
        let t = r as f32 / SPEC_ROWS as f32;
        let b = min_bin as f32 * (max_bin as f32 / min_bin as f32).powf(t);
        edges[r] = b;
    }

    let mut raw = vec![0f32; cols * SPEC_ROWS];
    let mut re = vec![0f32; FRAME_SIZE];
    let mut im = vec![0f32; FRAME_SIZE];

    for c in 0..cols {
        let pos = (c * hop).min(pcm.len().saturating_sub(FRAME_SIZE));
        re.iter_mut().for_each(|v| *v = 0.0);
        im.iter_mut().for_each(|v| *v = 0.0);
        for i in 0..FRAME_SIZE {
            if pos + i < pcm.len() {
                re[i] = pcm[pos + i];
            }
        }
        fft.hann_apply(&mut re);
        fft.forward(&mut re, &mut im);

        // aggregate power into log bands
        for r in 0..SPEC_ROWS {
            let b0 = edges[r].ceil() as usize;
            let b1 = (edges[r + 1].floor() as usize).max(b0 + 1);
            let mut sum = 0f64;
            for b in b0.min(max_bin)..b1.min(max_bin + 1) {
                sum += (re[b] * re[b] + im[b] * im[b]) as f64;
            }
            raw[c * SPEC_ROWS + r] = sum as f32;
        }
    }

    // dB + normalize to 0..1
    let mut max_db = f32::MIN;
    for v in raw.iter_mut() {
        *v = 10.0 * (*v).max(1e-12).log10();
        max_db = max_db.max(*v);
    }
    for v in raw.iter_mut() {
        *v = (((*v - max_db) - SPEC_FLOOR_DB) / (-SPEC_FLOOR_DB)).clamp(0.0, 1.0);
    }

    (cols, raw)
}
