//! Energy-based voice activity segmentation — the turn-taking layer.
//!
//! Ported from the TypeScript engine with one enhancement: a **spectral
//! flatness start gate**. Pure noise (hiss, fan) is spectrally flat; speech
//! onsets are peaky. Frames that are loud *and* flat no longer open a turn,
//! which kills most false triggers from broadband clatter while leaving real
//! speech untouched.

pub const TARGET_RATE: f32 = 16000.0;
pub const FRAME_SIZE: usize = 2048;
pub const MAX_SEGMENT_MS: usize = 15000;
const MIN_SEGMENT_MS: usize = 400;
const PREROLL_FRAMES: usize = 3;
const START_CONFIRM_FRAMES: usize = 2;
const SILENCE_END_MS: usize = 1100;

/// Spectral flatness of the current frame (0 = peaky/tonal, 1 = white noise).
/// Provided by the caller so the segmenter stays a pure state machine.
pub type Flatness = f32;

#[derive(Debug, Clone)]
pub struct RawSegment {
    pub pcm: Vec<f32>,
    pub duration_ms: usize,
}

pub struct VoiceSegmenter {
    preroll: Vec<Vec<f32>>,
    frames: Vec<Vec<f32>>,
    speaking: bool,
    loud_run: usize,
    silence_run: usize,
    noise_floor: f32,
    frame_count: usize,
    /// public read-only VAD state, mirrors the TS engine
    pub vad_state: &'static str,
}

impl VoiceSegmenter {
    pub fn new() -> Self {
        Self {
            preroll: Vec::with_capacity(PREROLL_FRAMES),
            frames: Vec::new(),
            speaking: false,
            loud_run: 0,
            silence_run: 0,
            noise_floor: 0.003,
            frame_count: 0,
            vad_state: "silence",
        }
    }

    /// Feed one 2048-sample frame at 16 kHz. Returns a finalized segment when
    /// the current turn closes (silence timeout or max length).
    pub fn feed(&mut self, frame: &[f32], flatness: Flatness) -> Option<RawSegment> {
        let rms = crate::dsp::frame_rms(frame);
        if !self.speaking {
            // track ambient level only while nobody is talking
            self.noise_floor = self.noise_floor * 0.96 + rms.min(0.2) * 0.04;
        }
        let start_thresh = (self.noise_floor * 3.5).max(0.012);
        let end_thresh = (self.noise_floor * 2.0).max(0.005);

        if !self.speaking {
            self.preroll.push(frame.to_vec());
            if self.preroll.len() > PREROLL_FRAMES {
                self.preroll.remove(0);
            }
            // energy gate + spectral-peakiness gate: flat loud noise stays out
            let speech_like = rms > start_thresh && flatness < 0.92;
            if speech_like {
                self.loud_run += 1;
            } else {
                self.loud_run = 0;
            }
            if self.loud_run >= START_CONFIRM_FRAMES {
                self.speaking = true;
                self.vad_state = "speech";
                self.frames = self.preroll.clone();
                self.silence_run = 0;
                self.frame_count = self.frames.len();
                self.preroll.clear();
            }
            return None;
        }

        self.frames.push(frame.to_vec());
        self.frame_count += 1;
        if rms > end_thresh {
            self.silence_run = 0;
        } else {
            self.silence_run += 1;
        }

        let silence_ms = (self.silence_run * FRAME_SIZE * 1000) as f32 / TARGET_RATE;
        let seg_ms = (self.frame_count * FRAME_SIZE * 1000) as f32 / TARGET_RATE;
        if silence_ms >= SILENCE_END_MS as f32 || seg_ms >= MAX_SEGMENT_MS as f32 {
            return self.finalize();
        }
        None
    }

    /// Finalize the in-progress segment (used on stop / end of file).
    pub fn flush(&mut self) -> Option<RawSegment> {
        if !self.speaking {
            return None;
        }
        self.finalize()
    }

    pub fn reset(&mut self) {
        self.preroll.clear();
        self.frames.clear();
        self.speaking = false;
        self.loud_run = 0;
        self.silence_run = 0;
        self.frame_count = 0;
        self.vad_state = "silence";
        self.noise_floor = 0.003;
    }

    fn finalize(&mut self) -> Option<RawSegment> {
        // trim most of the trailing silence, keep a natural tail
        let trailing_silent = self.silence_run.saturating_sub(2);
        let keep = if trailing_silent > 0 {
            (self.frames.len() - trailing_silent).max(PREROLL_FRAMES + 1)
        } else {
            self.frames.len()
        };
        let kept: Vec<Vec<f32>> = self.frames.drain(..keep).collect();
        self.frames.clear();

        let total: usize = kept.iter().map(|f| f.len()).sum();
        let mut pcm = Vec::with_capacity(total);
        for f in &kept {
            pcm.extend_from_slice(f);
        }
        let duration_ms = (kept.len() * FRAME_SIZE * 1000) / TARGET_RATE as usize;

        self.preroll.clear();
        self.speaking = false;
        self.loud_run = 0;
        self.silence_run = 0;
        self.frame_count = 0;
        self.vad_state = "silence";

        if duration_ms < MIN_SEGMENT_MS {
            return None;
        }
        Some(RawSegment { pcm, duration_ms })
    }
}

impl Default for VoiceSegmenter {
    fn default() -> Self {
        Self::new()
    }
}
