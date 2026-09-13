//! PerceptAudio 2.0 — Rust/WASM voice-perception DSP engine.
//!
//! Everything the browser used to do in hand-rolled TypeScript now runs here,
//! compiled to WebAssembly: turn-taking VAD, YIN pitch, distance bands,
//! speaker clustering, spectral noise suppression, loudness normalization,
//! WAV encoding and spectrogram rendering.
//!
//! # JS API (wasm-bindgen)
//! * [`PerceptEngine`] — stateful live/upload engine
//! * [`resample`] — Catmull-Rom cubic SRC (any rate → 16 kHz)
//! * [`render_wav`] — normalize + WAV-encode (the "second, louder listen")
//! * [`version`] — engine version string

#![warn(clippy::all)]

mod denoise;
mod dsp;
mod engine;
mod fft;
mod pitch;
mod resample;
mod speaker;
mod spectrogram;
mod vad;
mod wav;

use std::sync::Arc;

use wasm_bindgen::prelude::*;

pub use engine::{Analyzed, LiveMetrics, PerceptEngine as CoreEngine};

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}

/// Engine version — surfaced in the UI's "Rust WASM" badge.
#[wasm_bindgen]
pub fn version() -> String {
    "2.0.0".to_string()
}

/// Catmull-Rom cubic resample. `resample(chunk, 48000, 16000)`.
#[wasm_bindgen]
pub fn resample(input: &[f32], from_rate: f32, to_rate: f32) -> Vec<f32> {
    resample::resample(input, from_rate, to_rate)
}

/// Normalize a denoised segment to `target_dbfs` and encode 16-bit WAV.
/// Used for the second, louder listen (−12 dBFS retry).
#[wasm_bindgen]
pub fn render_wav(pcm: &[f32], target_dbfs: f32, peak_ceiling: f32) -> Vec<u8> {
    let norm = dsp::normalize_loudness(pcm, target_dbfs, peak_ceiling);
    wav::encode_wav(&norm, fft::TARGET_RATE as u32)
}

/// The stateful perception engine.
#[wasm_bindgen]
pub struct PerceptEngine {
    core: CoreEngine,
}

#[wasm_bindgen]
impl PerceptEngine {
    #[wasm_bindgen(constructor)]
    pub fn new() -> Self {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        Self { core: CoreEngine::new() }
    }

    /// Feed one 2048-sample frame at 16 kHz. Returns live metrics JSON:
    /// `{ dbfs, pitchHz, hfRatio, distanceBand, vad, noiseFloorDb,
    ///    segmentReady, elapsedUs }`
    pub fn feed_frame(&mut self, frame: &[f32]) -> JsValue {
        let m = self.core.feed_frame(frame);
        serde_wasm_bindgen::to_value(&m).unwrap_or(JsValue::NULL)
    }

    /// Pull the pending raw segment PCM, if a turn just closed.
    /// Returns `undefined` when nothing is pending.
    pub fn take_segment(&mut self) -> Option<Vec<f32>> {
        self.core.take_segment()
    }

    /// Finalize any in-progress turn (on stop / end of file).
    /// Returns `true` when a segment became available via `take_segment`.
    pub fn flush(&mut self) -> bool {
        self.core.flush()
    }

    /// Analyze a finished utterance. Returns an object:
    /// `{ acoustics, speaker, denoiseDb, cleanPcm, wav, spectrogram, elapsedUs }`
    /// where `wav` is a Uint8Array, `cleanPcm` a Float32Array and
    /// `spectrogram` = `{ cols, rows, data: Float32Array }`.
    pub fn analyze_segment(&mut self, pcm: &[f32], target_dbfs: f32) -> JsValue {
        let a = self.core.analyze_segment(pcm, target_dbfs);
        analyzed_to_js(&a)
    }

    /// Associate a self-introduced name with a speaker's voice.
    pub fn rename_speaker(&mut self, id: u32, name: &str) {
        self.core.rename_speaker(id, name);
    }

    /// Current speaker roster for the UI.
    pub fn list_speakers(&self) -> JsValue {
        serde_wasm_bindgen::to_value(&self.core.list_speakers()).unwrap_or(JsValue::NULL)
    }

    /// Clear all session state (segmenter, speakers, noise profile).
    pub fn reset(&mut self) {
        self.core.reset();
    }
}

fn analyzed_to_js(a: &Analyzed) -> JsValue {
    use js_sys::{Object, Reflect, Uint8Array};
    use wasm_bindgen::JsValue;

    let obj = Object::new();

    let acoustics = serde_wasm_bindgen::to_value(&a.acoustics).unwrap_or(JsValue::NULL);
    let speaker = serde_wasm_bindgen::to_value(&a.speaker).unwrap_or(JsValue::NULL);

    let _ = Reflect::set(&obj, &JsValue::from_str("acoustics"), &acoustics);
    let _ = Reflect::set(&obj, &JsValue::from_str("speaker"), &speaker);
    let _ = Reflect::set(
        &obj,
        &JsValue::from_str("denoiseDb"),
        &JsValue::from_f64(a.denoise_db as f64),
    );
    let _ = Reflect::set(
        &obj,
        &JsValue::from_str("cleanPcm"),
        &js_sys::Float32Array::from(a.clean_pcm.as_slice()).into(),
    );
    let _ = Reflect::set(
        &obj,
        &JsValue::from_str("wav"),
        &Uint8Array::from(a.wav.as_slice()).into(),
    );

    let spec = Object::new();
    let _ = Reflect::set(&spec, &JsValue::from_str("cols"), &JsValue::from_f64(a.spec_cols as f64));
    let _ = Reflect::set(&spec, &JsValue::from_str("rows"), &JsValue::from_f64(a.spec_rows as f64));
    let _ = Reflect::set(
        &spec,
        &JsValue::from_str("data"),
        &js_sys::Float32Array::from(a.spec_data.as_slice()).into(),
    );
    let _ = Reflect::set(&obj, &JsValue::from_str("spectrogram"), &spec.into());

    let _ = Reflect::set(
        &obj,
        &JsValue::from_str("elapsedUs"),
        &JsValue::from_f64(a.elapsed_us),
    );

    obj.into()
}
