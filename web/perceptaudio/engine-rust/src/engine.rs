//! High-level perception engine — the state machine the browser talks to.
//!
//! Owns the segmenter (turn-taking), the denoiser (attention filter), the
//! speaker registry (WHO) and the live-metric smoothers. The TypeScript hook
//! feeds it 2048-sample frames and pulls finished utterances out.

use std::sync::Arc;

use serde::Serialize;

use crate::denoise::SpectralDenoiser;
use crate::dsp::{self, AcousticProfile};
use crate::fft::{Fft, FRAME_SIZE, TARGET_RATE};
use crate::pitch;
use crate::speaker::{SpeakerInfo, SpeakerRegistry};
use crate::vad::VoiceSegmenter;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct LiveMetrics {
    pub dbfs: f32,
    pub pitch_hz: Option<f32>,
    pub hf_ratio: f32,
    pub distance_band: String,
    pub vad: String,
    pub noise_floor_db: f32,
    pub segment_ready: bool,
    pub elapsed_us: f64,
}

pub struct PerceptEngine {
    fft: Arc<Fft>,
    segmenter: VoiceSegmenter,
    registry: SpeakerRegistry,
    denoiser: SpectralDenoiser,
    pending_segment: Option<Vec<f32>>,

    // live-metric smoothers (same constants as the TS hook)
    smooth_dbfs: f32,
    smooth_hf: f32,
    smooth_pitch: Option<f32>,
    last_flatness: f32,
}

impl PerceptEngine {
    pub fn new() -> Self {
        let fft = Arc::new(Fft::new(FRAME_SIZE));
        Self {
            denoiser: SpectralDenoiser::new(Arc::clone(&fft)),
            fft,
            segmenter: VoiceSegmenter::new(),
            registry: SpeakerRegistry::new(),
            pending_segment: None,
            smooth_dbfs: -100.0,
            smooth_hf: 0.0,
            smooth_pitch: None,
            last_flatness: 1.0,
        }
    }

    /// Feed one 2048-sample frame at 16 kHz.
    ///
    /// While silent and quiet, the frame also feeds the noise learner —
    /// the "attention" loop lives inside the engine now.
    pub fn feed_frame(&mut self, frame: &[f32]) -> LiveMetrics {
        let t0 = now_ms();

        let rms = dsp::frame_rms(frame);
        let (hf_raw, flatness, _) = dsp::frame_spectrum_features(&self.fft, frame, TARGET_RATE);
        self.last_flatness = flatness;

        // attention filter: while nobody talks, keep learning the room
        if self.segmenter.vad_state == "silence" && rms < 0.04 {
            self.denoiser.learn(frame);
        }

        // live pitch for the meter (only when actually voiced)
        let mut pitch_now: Option<f32> = None;
        if rms > 0.008 {
            if let Some(p) = pitch::estimate_pitch_yin(frame, TARGET_RATE) {
                pitch_now = Some(match self.smooth_pitch {
                    Some(prev) => (prev * 0.6 + p * 0.4).round(),
                    None => p.round(),
                });
            }
        }
        if pitch_now.is_some() {
            self.smooth_pitch = pitch_now;
        }

        self.smooth_dbfs = self.smooth_dbfs * 0.75 + (if rms > 0.0 { 20.0 * rms.log10() } else { -100.0 }) * 0.25;
        self.smooth_hf = self.smooth_hf * 0.8 + hf_raw * 0.2;

        let seg = self.segmenter.feed(frame, flatness);
        if let Some(s) = seg {
            self.pending_segment = Some(s.pcm);
        }

        let metrics = LiveMetrics {
            dbfs: (self.smooth_dbfs * 10.0).round() / 10.0,
            pitch_hz: self.smooth_pitch,
            hf_ratio: (self.smooth_hf * 1000.0).round() / 1000.0,
            distance_band: dsp::classify_distance(self.smooth_dbfs, self.smooth_hf).to_string(),
            vad: self.segmenter.vad_state.to_string(),
            noise_floor_db: (self.denoiser.noise_floor_db() * 10.0).round() / 10.0,
            segment_ready: self.pending_segment.is_some(),
            elapsed_us: (now_ms() - t0) * 1000.0,
        };
        metrics
    }

    /// Pull the pending raw segment PCM (16 kHz mono), if a turn just closed.
    pub fn take_segment(&mut self) -> Option<Vec<f32>> {
        self.pending_segment.take()
    }

    /// Finalize any in-progress turn (used on stop / end of file).
    /// Returns `true` when a segment became available via `take_segment`.
    pub fn flush(&mut self) -> bool {
        if let Some(seg) = self.segmenter.flush() {
            self.pending_segment = Some(seg.pcm);
            return true;
        }
        false
    }

    /// Full analysis of a finished utterance:
    /// acoustics (WHERE) + speaker (WHO) + denoise (attention) + normalized
    /// WAV for the ASR + spectrogram for the UI, with a Rust timing stamp.
    pub fn analyze_segment(&mut self, pcm: &[f32], target_dbfs: f32) -> Analyzed {
        let t0 = now_ms();

        let acoustics = dsp::analyze_segment(&self.fft, pcm);

        // timbre feature for the WHO layer
        let mut centroids: Vec<f32> = Vec::new();
        let mut by_loudness: Vec<(f32, usize)> = Vec::new();
        let mut off = 0usize;
        let mut idx = 0usize;
        while off + FRAME_SIZE <= pcm.len() {
            let r = dsp::frame_rms(&pcm[off..off + FRAME_SIZE]);
            if r > 0.01 {
                by_loudness.push((r, idx));
            }
            off += FRAME_SIZE;
            idx += 1;
        }
        by_loudness.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        for (_, i) in by_loudness.iter().take(5) {
            let off = i * FRAME_SIZE;
            let (_, _, c) = dsp::frame_spectrum_features(&self.fft, &pcm[off..off + FRAME_SIZE], TARGET_RATE);
            centroids.push(c);
        }
        let centroid_hz = if centroids.is_empty() { None } else { Some(median(&mut centroids)) };

        let speaker = self.registry.assign(acoustics.pitch_hz, centroid_hz);

        let denoised = self.denoiser.denoise_segment(pcm);
        let attenuation_db = denoised.attenuation_db;
        let normalized = dsp::normalize_loudness(&denoised.pcm, target_dbfs, 0.97);
        let wav = crate::wav::encode_wav(&normalized, TARGET_RATE as u32);
        let (cols, spec_data) = crate::spectrogram::spectrogram(&self.fft, &denoised.pcm, TARGET_RATE);

        Analyzed {
            acoustics,
            speaker,
            denoise_db: attenuation_db,
            clean_pcm: denoised.pcm,
            wav,
            spec_cols: cols,
            spec_rows: crate::spectrogram::SPEC_ROWS,
            spec_data,
            elapsed_us: (now_ms() - t0) * 1000.0,
        }
    }

    pub fn rename_speaker(&mut self, id: u32, name: &str) {
        self.registry.rename(id, name);
    }

    /// Current speaker roster (for the UI's room snapshot).
    pub fn list_speakers(&self) -> Vec<SpeakerInfo> {
        self.registry.list()
    }

    pub fn reset(&mut self) {
        self.segmenter.reset();
        self.registry.reset();
        self.denoiser.reset();
        self.pending_segment = None;
        self.smooth_dbfs = -100.0;
        self.smooth_hf = 0.0;
        self.smooth_pitch = None;
    }
}

fn median(values: &mut [f32]) -> f32 {
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

/// wall clock in ms (JS `Date.now` — always available on wasm32-unknown-unknown)
fn now_ms() -> f64 {
    js_sys::Date::now()
}

/// Result bundle of `analyze_segment`, marshalled to JS as one object.
pub struct Analyzed {
    pub acoustics: AcousticProfile,
    pub speaker: SpeakerInfo,
    pub denoise_db: f32,
    /// denoised (unnormalized) PCM — kept for the louder second listen
    pub clean_pcm: Vec<f32>,
    /// normalized 16-bit WAV ready for the ASR
    pub wav: Vec<u8>,
    pub spec_cols: usize,
    pub spec_rows: usize,
    pub spec_data: Vec<f32>,
    pub elapsed_us: f64,
}

impl Default for PerceptEngine {
    fn default() -> Self {
        Self::new()
    }
}
