//! P2160 — OTD3 AUDIO: PerceptAudio 2.0, native and upgraded.
//!
//! The complete Rust voice-perception DSP engine (VAD, YIN pitch, speaker
//! clustering, Wiener-style spectral denoise, spectrograms, Catmull-Rom
//! resampling, WAV I/O) merged into OTD as pure native modules — plus the
//! OTD3 upgrades: full-spectrum band analysis and metric distance.

pub mod fft;
pub mod dsp;
pub mod pitch;
pub mod vad;
pub mod speaker;
pub mod denoise;
pub mod spectrogram;
pub mod resample;
pub mod wav;
// OTD3 upgrades
pub mod acoustics;
pub mod engine;

pub use engine::{analyze_pcm, analyze_wav_file, Analyzed, LiveMetrics, PerceptEngine};

/// Engine version (carried from PerceptAudio 2.0, now OTD3-native).
pub const PERCEPTAUDIO_VERSION: &str = "2.0+otd3";
