//! P2160 — OTD3 PERCEPTAUDIO ENGINE — the native port of PerceptAudio 2.0.
//!
//! PerceptAudio 2.0 (the Rust/WASM voice-perception engine: VAD, YIN pitch,
//! speaker clustering, spectral denoise, spectrograms) now lives inside OTD
//! as a pure native module — no browser, no wasm-bindgen, no dependencies.
//! Its five perception layers are intact and two new ones ride along:
//!
//!   WHO    — speaker clustering (pitch + timbre)
//!   WHERE  — distance, now METRIC (inverse-square ranging, echo timing,
//!            band anchors — see acoustics.rs)
//!   WHAT   — clean, normalized audio out (WAV) for any ASR
//!   HOW    — the live acoustic profile (level, pitch, bands)
//!   WHY    — the full-spectrum report: every frequency band, named and
//!            placed on the physics ladder with its wavelength
//!
//! The engine eats 2048-sample frames at 16 kHz (exactly like the browser
//! build) or whole WAV files at any rate (Catmull-Rom resampled down).

use crate::audio::denoise::SpectralDenoiser;
use crate::audio::dsp::{self, AcousticProfile};
use crate::audio::fft::{Fft, FRAME_SIZE, TARGET_RATE};
use crate::audio::{acoustics, pitch, speaker, spectrogram, wav};
use crate::audio::vad::VoiceSegmenter;
use std::sync::Arc;

/// Live per-frame metrics (the browser meter, native again).
#[derive(Debug, Clone)]
pub struct LiveMetrics {
    pub dbfs: f32,
    pub pitch_hz: Option<f32>,
    pub hf_ratio: f32,
    pub distance_band: String,
    pub vad: String,
    pub noise_floor_db: f32,
    pub segment_ready: bool,
}

pub struct PerceptEngine {
    fft: Arc<Fft>,
    segmenter: VoiceSegmenter,
    registry: speaker::SpeakerRegistry,
    denoiser: SpectralDenoiser,
    pending_segment: Option<Vec<f32>>,
    // live-metric smoothers (same constants as the browser build)
    smooth_dbfs: f32,
    smooth_hf: f32,
    smooth_pitch: Option<f32>,
}

impl PerceptEngine {
    pub fn new() -> Self {
        let fft = Arc::new(Fft::new(FRAME_SIZE));
        Self {
            denoiser: SpectralDenoiser::new(Arc::clone(&fft)),
            fft,
            segmenter: VoiceSegmenter::new(),
            registry: speaker::SpeakerRegistry::new(),
            pending_segment: None,
            smooth_dbfs: -100.0,
            smooth_hf: 0.0,
            smooth_pitch: None,
        }
    }

    /// Feed one 2048-sample frame at 16 kHz.
    pub fn feed_frame(&mut self, frame: &[f32]) -> LiveMetrics {
        let rms = dsp::frame_rms(frame);
        let (hf_raw, flatness, _) = dsp::frame_spectrum_features(&self.fft, frame, TARGET_RATE);

        // attention filter: while nobody talks, keep learning the room
        if self.segmenter.vad_state == "silence" && rms < 0.04 {
            self.denoiser.learn(frame);
        }

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

        LiveMetrics {
            dbfs: (self.smooth_dbfs * 10.0).round() / 10.0,
            pitch_hz: self.smooth_pitch,
            hf_ratio: (self.smooth_hf * 1000.0).round() / 1000.0,
            distance_band: dsp::classify_distance(self.smooth_dbfs, self.smooth_hf).to_string(),
            vad: self.segmenter.vad_state.to_string(),
            noise_floor_db: (self.denoiser.noise_floor_db() * 10.0).round() / 10.0,
            segment_ready: self.pending_segment.is_some(),
        }
    }

    /// Pull the pending raw segment PCM (16 kHz mono), if a turn closed.
    pub fn take_segment(&mut self) -> Option<Vec<f32>> {
        self.pending_segment.take()
    }

    /// Finalize any in-progress turn. Returns true when a segment landed.
    pub fn flush(&mut self) -> bool {
        if let Some(seg) = self.segmenter.flush() {
            self.pending_segment = Some(seg.pcm);
            return true;
        }
        false
    }

    /// Full analysis of a finished utterance (the five-layer bundle).
    pub fn analyze_segment(&mut self, pcm: &[f32], target_dbfs: f32) -> Analyzed {
        let acoustics = dsp::analyze_segment(&self.fft, pcm);
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
        let wav_bytes = wav::encode_wav(&normalized, TARGET_RATE as u32);
        let (cols, spec_data) = spectrogram::spectrogram(&self.fft, &denoised.pcm, TARGET_RATE);
        // OTD3: metric distance + full-spectrum bands ride on the bundle
        let distance = acoustics::distance_from_band(&acoustics.distance_band, acoustics.hf_ratio);
        let bands = acoustics::analyze_bands(&self.fft, &denoised.pcm, TARGET_RATE);
        Analyzed {
            acoustics,
            speaker,
            denoise_db: attenuation_db,
            clean_pcm: denoised.pcm,
            wav: wav_bytes,
            spec_cols: cols,
            spec_rows: spectrogram::SPEC_ROWS,
            spec_data,
            distance,
            bands,
        }
    }

    pub fn rename_speaker(&mut self, id: u32, name: &str) {
        self.registry.rename(id, name);
    }

    pub fn list_speakers(&self) -> Vec<speaker::SpeakerInfo> {
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

impl Default for PerceptEngine {
    fn default() -> Self {
        Self::new()
    }
}

/// Result bundle of `analyze_segment`.
pub struct Analyzed {
    pub acoustics: AcousticProfile,
    pub speaker: speaker::SpeakerInfo,
    pub denoise_db: f32,
    pub clean_pcm: Vec<f32>,
    pub wav: Vec<u8>,
    pub spec_cols: usize,
    pub spec_rows: usize,
    pub spec_data: Vec<f32>,
    /// OTD3: the METRIC distance estimate (inverse-square + treble anchor)
    pub distance: acoustics::DistanceEstimate,
    /// OTD3: full-spectrum band readings (named bands, wavelengths)
    pub bands: Vec<acoustics::BandReading>,
}

/// Analyse a whole PCM buffer (any sample rate): resample to 16 kHz,
/// stream frames through the live engine, then deep-analyse each detected
/// utterance. Returns one Analyzed per voice segment.
pub fn analyze_pcm(engine: &mut PerceptEngine, pcm: &[f32], sample_rate: f32) -> Vec<Analyzed> {
    let mono16 = if (sample_rate - TARGET_RATE).abs() < 1.0 {
        pcm.to_vec()
    } else {
        crate::audio::resample::resample(pcm, sample_rate, TARGET_RATE)
    };
    let mut results = Vec::new();
    let mut off = 0usize;
    while off + FRAME_SIZE <= mono16.len() {
        let f = &mono16[off..off + FRAME_SIZE];
        let _ = engine.feed_frame(f);
        if let Some(seg) = engine.take_segment() {
            results.push(engine.analyze_segment(&seg, -18.0));
        }
        off += FRAME_SIZE;
    }
    if engine.flush() {
        if let Some(seg) = engine.take_segment() {
            results.push(engine.analyze_segment(&seg, -18.0));
        }
    }
    results
}

/// Analyse a WAV file from disk.
pub fn analyze_wav_file(path: &str) -> Result<Vec<Analyzed>, String> {
    let bytes = std::fs::read(path).map_err(|e| format!("can't read {}: {}", path, e))?;
    let (pcm, rate) = wav::decode_wav(&bytes)?;
    let mut engine = PerceptEngine::new();
    Ok(analyze_pcm(&mut engine, &pcm, rate as f32))
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_streams_frames() {
        let mut e = PerceptEngine::new();
        // 1 second of silence then a voiced burst
        let mut pcm: Vec<f32> = vec![0.0; TARGET_RATE as usize];
        for i in 0..(TARGET_RATE as usize) {
            let t = i as f32 / TARGET_RATE;
            pcm[i] = 0.4 * (2.0 * std::f32::consts::PI * 220.0 * t).sin();
        }
        let mut saw_speech = false;
        let mut off = 0usize;
        while off + FRAME_SIZE <= pcm.len() {
            let m = e.feed_frame(&pcm[off..off + FRAME_SIZE]);
            if m.vad == "speech" {
                saw_speech = true;
            }
            off += FRAME_SIZE;
        }
        assert!(saw_speech, "the segmenter must hear the 220 Hz burst");
        e.flush();
        let seg = e.take_segment();
        assert!(seg.is_some(), "a closed turn must be collectable");
    }

    #[test]
    fn analyze_finds_pitch_and_bands() {
        let mut e = PerceptEngine::new();
        // two seconds of a 220 Hz sawtooth — harmonic-rich like a voice
        // (a pure sine is periodic at 2× its period too, which is exactly
        // the octave ambiguity YIN's harmonics-dependence resolves)
        let n = (2.0 * TARGET_RATE) as usize;
        let mut pcm = vec![0f32; n];
        for i in 0..n {
            let t = i as f32 / TARGET_RATE;
            let phase = (t * 220.0) % 1.0;
            let mut v = 0f32;
            for h in 1..=8 {
                v += (2.0 * std::f32::consts::PI * h as f32 * phase).sin() / h as f32;
            }
            pcm[i] = 0.4 * v * 0.5;
        }
        let outs = analyze_pcm(&mut e, &pcm, TARGET_RATE);
        assert!(!outs.is_empty(), "tone must be segmented as a turn");
        let a = &outs[0];
        // YIN should land the fundamental near 220 Hz
        if let Some(p) = a.acoustics.pitch_hz {
            assert!((p - 220.0).abs() < 12.0, "YIN pitch ≈ 220, got {}", p);
        }
        // the low-mid band must dominate (220 Hz fundamental + harmonics)
        let strongest = a.bands.iter().max_by(|x, y| x.share.partial_cmp(&y.share).unwrap()).unwrap();
        assert!(matches!(strongest.name, "low-mids" | "midrange"),
            "220 Hz saw lives in low-mids/midrange, got {}", strongest.name);
        // distance estimate exists and is metric
        assert!(a.distance.metres.is_finite() || a.distance.band == "unknown");
    }

    #[test]
    fn wav_round_trip_through_engine() {
        let pcm: Vec<f32> = (0..16000).map(|i| {
            0.3 * (2.0 * std::f32::consts::PI * 300.0 * (i as f32 / 16000.0)).sin()
        }).collect();
        let bytes = wav::encode_wav(&pcm, 16000);
        let path = std::env::temp_dir().join("otd3_test_engine.wav");
        std::fs::write(&path, &bytes).unwrap();
        let outs = analyze_wav_file(path.to_str().unwrap()).unwrap();
        assert!(!outs.is_empty());
        let _ = std::fs::remove_file(&path);
    }
}
