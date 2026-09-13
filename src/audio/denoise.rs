//! Spectral noise suppression — the attention filter.
//!
//! While nobody talks, the engine keeps an exponential average of the room's
//! power spectrum (fan hum, street rumble, keyboard clatter). When a speech
//! segment is finalized, every STFT frame gets a per-frequency gain that
//! attenuates whatever looks like the learned noise.
//!
//! Upgrades over the TypeScript engine:
//! * **Wiener-style gain** `mag² / (mag² + (α·noise)²)` instead of plain
//!   magnitude subtraction — far fewer musical-noise artifacts.
//! * **SNR-adaptive over-subtraction** α: gentle (1.2) on high-SNR frames so
//!   speech is not chewed, aggressive (2.5) on low-SNR frames.
//! * Per-bin minimum-statistics fallback so uploads that start talking
//!   immediately are still cleaned.
//!
//! STFT: Hann analysis window, 50 % overlap, rectangular synthesis — Hann at
//! hop N/2 overlap-adds to exactly 1, so with unity gains the round trip is
//! transparent.

use crate::audio::fft::{Fft, FRAME_SIZE};

pub struct DenoiseResult {
    pub pcm: Vec<f32>,
    /// mean gain applied to attenuated bins, in dB (negative = removal)
    pub attenuation_db: f32,
}

pub struct SpectralDenoiser {
    fft: std::sync::Arc<Fft>,
    /// per-bin noise power, bins 0..=N/2
    noise_power: Option<Vec<f32>>,
    learned_frames: usize,
    last_attenuation_db: f32,
    /// temporal gain smoothing state
    gains: Vec<f32>,
}

impl SpectralDenoiser {
    pub fn new(fft: std::sync::Arc<Fft>) -> Self {
        let half = FRAME_SIZE / 2;
        Self {
            fft,
            noise_power: None,
            learned_frames: 0,
            last_attenuation_db: 0.0,
            gains: vec![1.0; half + 1],
        }
    }

    pub fn ready(&self) -> bool {
        self.noise_power.is_some() && self.learned_frames >= 4
    }

    /// Ambient noise floor in dBFS (time-domain equivalent via Parseval).
    pub fn noise_floor_db(&self) -> f32 {
        match &self.noise_power {
            None => -100.0,
            Some(p) => {
                let n = FRAME_SIZE as f32;
                let s: f32 = p.iter().sum();
                let mean_time_power = (16.0 / 3.0) * (s / (n * n));
                if mean_time_power > 0.0 {
                    10.0 * mean_time_power.log10()
                } else {
                    -100.0
                }
            }
        }
    }

    pub fn attenuation_db(&self) -> f32 {
        self.last_attenuation_db
    }

    pub fn reset(&mut self) {
        self.noise_power = None;
        self.learned_frames = 0;
        self.last_attenuation_db = 0.0;
        self.gains.fill(1.0);
    }

    /// Learn the ambient spectrum from one silent frame (raw PCM 16 kHz).
    pub fn learn(&mut self, frame: &[f32]) {
        if frame.len() != FRAME_SIZE {
            return;
        }
        let half = FRAME_SIZE / 2;
        let mut re = vec![0f32; FRAME_SIZE];
        let mut im = vec![0f32; FRAME_SIZE];
        self.fft.hann_multiply(frame, &mut re);
        self.fft.forward(&mut re, &mut im);

        let p = self.noise_power.get_or_insert_with(|| vec![0f32; half + 1]);
        for b in 0..=half {
            let power = re[b] * re[b] + im[b] * im[b];
            p[b] = p[b] * 0.88 + power * 0.12;
        }
        self.learned_frames += 1;
    }

    /// Clean a finalized speech segment with STFT spectral subtraction.
    pub fn denoise_segment(&mut self, pcm: &[f32]) -> DenoiseResult {
        let n = FRAME_SIZE;
        let hop = n / 2;
        let half = n / 2;
        if pcm.len() < n {
            return DenoiseResult {
                pcm: pcm.to_vec(),
                attenuation_db: 0.0,
            };
        }

        let total_frames = pcm.len() / hop + 1;

        // ---- noise estimate: streaming EMA, else per-bin minimum statistics
        let mut noise: Vec<f32> = match (&self.noise_power, self.ready()) {
            (Some(p), true) => p.clone(),
            _ => {
                let mut mins = vec![f32::INFINITY; half + 1];
                let mut re = vec![0f32; n];
                let mut im = vec![0f32; n];
                for f in 0..total_frames {
                    let pos = f * hop;
                    re.iter_mut().for_each(|v| *v = 0.0);
                    im.iter_mut().for_each(|v| *v = 0.0);
                    for i in 0..n {
                        if pos + i < pcm.len() {
                            re[i] = pcm[pos + i];
                        }
                    }
                    self.fft.hann_apply(&mut re);
                    self.fft.forward(&mut re, &mut im);
                    for b in 0..=half {
                        let p = re[b] * re[b] + im[b] * im[b];
                        if p < mins[b] {
                            mins[b] = p;
                        }
                    }
                }
                mins.iter_mut().for_each(|v| {
                    if !v.is_finite() {
                        *v = 0.0;
                    }
                });
                mins
            }
        };

        // frequency-smooth the estimate (3-tap) to avoid isolated notches
        let mut smooth = vec![0f32; half + 1];
        for b in 0..=half {
            let a = noise[b.saturating_sub(1)];
            let c = noise[(b + 1).min(half)];
            smooth[b] = 0.25 * a + 0.5 * noise[b] + 0.25 * c;
        }
        noise = smooth;

        // ---- spectral subtraction, frame by frame
        let out_len = (total_frames - 1) * hop + n;
        let mut out = vec![0f32; out_len];
        self.gains.fill(1.0);
        let mut gain_sum_db = 0f64;
        let mut gain_count = 0usize;

        let mut re = vec![0f32; n];
        let mut im = vec![0f32; n];
        for f in 0..total_frames {
            let pos = f * hop;
            re.iter_mut().for_each(|v| *v = 0.0);
            im.iter_mut().for_each(|v| *v = 0.0);
            for i in 0..n {
                if pos + i < pcm.len() {
                    re[i] = pcm[pos + i];
                }
            }
            self.fft.hann_apply(&mut re);
            self.fft.forward(&mut re, &mut im);

            // frame SNR drives the over-subtraction strength
            let mut sig_power = 0f64;
            let mut noise_power_f = 0f64;
            for b in 1..=half {
                sig_power += (re[b] * re[b] + im[b] * im[b]) as f64;
                noise_power_f += noise[b] as f64;
            }
            let snr_db = if noise_power_f > 1e-12 {
                10.0 * (sig_power / noise_power_f).max(1e-9).log10()
            } else {
                40.0
            };
            let alpha: f32 = if snr_db >= 20.0 {
                1.2
            } else if snr_db <= 6.0 {
                2.5
            } else {
                1.2 + (20.0 - snr_db as f32) * (1.3 / 14.0)
            };
            const BETA: f32 = 0.1; // spectral floor — dodges musical noise

            for b in 0..=half {
                let mag = (re[b] * re[b] + im[b] * im[b]).sqrt();
                let noise_mag = noise[b].sqrt() * alpha;
                // Wiener-style smooth gain + hard floor
                let g_w = if mag > 1e-9 {
                    mag * mag / (mag * mag + noise_mag * noise_mag)
                } else {
                    1.0
                };
                let g_sub = if mag > 1e-9 { (mag - noise_mag) / mag } else { 1.0 };
                // blend: Wiener keeps it artifact-free, subtraction does the work
                let mut g = 0.6 * g_w + 0.4 * g_sub.max(BETA);
                if g < BETA {
                    g = BETA;
                }
                if g > 1.0 {
                    g = 1.0;
                }
                self.gains[b] = self.gains[b] * 0.55 + g * 0.45;
                let gg = self.gains[b];
                if gg < 0.999 {
                    gain_sum_db += (20.0 * gg.max(1e-4).log10()) as f64;
                    gain_count += 1;
                }
                // keep the spectrum conjugate-symmetric: bin b and mirror n−b
                re[b] *= gg;
                im[b] *= gg;
                if b > 0 && b < half {
                    re[n - b] *= gg;
                    im[n - b] *= gg;
                }
            }

            self.fft.inverse(&mut re, &mut im);
            // overlap-add: Hann at 50% overlap sums to exactly 1
            for i in 0..n {
                if pos + i < out_len {
                    out[pos + i] += re[i];
                }
            }
        }

        self.last_attenuation_db = if gain_count > 0 {
            (((gain_sum_db / gain_count as f64) * 10.0).round() / 10.0) as f32
        } else {
            0.0
        };
        out.truncate(pcm.len());
        DenoiseResult {
            pcm: out,
            attenuation_db: self.last_attenuation_db,
        }
    }
}
