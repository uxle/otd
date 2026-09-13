//! Speaker clustering — the WHO layer.
//!
//! Upgrade over the TypeScript engine: each voice keeps a **two-feature
//! centroid** (pitch F0 + spectral centroid/timbre brightness). Pitch alone
//! merges similar voices; adding the timbre axis splits same-pitch speakers
//! while tolerating prosody swings of one voice.

use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct SpeakerInfo {
    pub id: u32,
    pub label: String,
    pub pitch_hz: Option<f32>,
    pub utterances: u32,
}

#[derive(Debug, Clone)]
struct Centroid {
    pitch_hz: f32,
    centroid_hz: f32,
}

#[derive(Debug, Clone)]
struct Speaker {
    id: u32,
    label: String,
    centroid: Option<Centroid>,
    utterances: u32,
}

pub struct SpeakerRegistry {
    speakers: Vec<Speaker>,
}

/// pitch tolerance in Hz (mirrors the TS engine)
const PITCH_TOL: f32 = 30.0;
/// relative pitch tolerance for higher voices (adult F0 overlap)
const PITCH_REL_TOL: f32 = 0.09;
/// timbre tolerance: same voice rarely shifts its spectral centroid more than
/// this between turns (distance + emotion do move it, hence the generous band)
const CENTROID_TOL: f32 = 900.0;

impl SpeakerRegistry {
    pub fn new() -> Self {
        Self { speakers: Vec::new() }
    }

    /// Attribute one utterance to a speaker.
    /// `pitch_hz` — median F0 of the loudest voiced frames (None when unvoiced).
    /// `centroid_hz` — median spectral centroid of the loudest frames.
    pub fn assign(&mut self, pitch_hz: Option<f32>, centroid_hz: Option<f32>) -> SpeakerInfo {
        if pitch_hz.is_none() {
            // unvoiced burst: attribute to the most recent speaker if any
            if let Some(last) = self.speakers.last_mut() {
                last.utterances += 1;
                return SpeakerInfo {
                    id: last.id,
                    label: last.label.clone(),
                    pitch_hz: last.centroid.as_ref().map(|c| c.pitch_hz.round() as u32 as f32),
                    utterances: last.utterances,
                };
            }
            let s = Speaker {
                id: 1,
                label: "Speaker 1".into(),
                centroid: None,
                utterances: 1,
            };
            let info = SpeakerInfo {
                id: 1,
                label: s.label.clone(),
                pitch_hz: None,
                utterances: 1,
            };
            self.speakers.push(s);
            return info;
        }

        let pitch = pitch_hz.unwrap();
        let timbre = centroid_hz.unwrap_or(0.0);

        let mut best: Option<(usize, f32)> = None;
        for (idx, s) in self.speakers.iter().enumerate() {
            let Some(c) = &s.centroid else { continue };
            let d_pitch = (c.pitch_hz - pitch).abs();
            let rel = d_pitch / c.pitch_hz.max(1.0);
            let pitch_ok = d_pitch <= PITCH_TOL || (rel <= PITCH_REL_TOL && d_pitch <= PITCH_TOL * 1.6);

            // timbre gate only when both sides have a usable centroid
            let timbre_ok = match (centroid_hz, c.centroid_hz > 0.0, timbre > 0.0) {
                (Some(_), true, true) => (c.centroid_hz - timbre).abs() <= CENTROID_TOL,
                _ => true,
            };

            if pitch_ok && timbre_ok {
                let score = d_pitch + 0.05 * (c.centroid_hz - timbre).abs();
                if best.map(|(_, bs)| score < bs).unwrap_or(true) {
                    best = Some((idx, score));
                }
            }
        }

        if let Some((idx, _)) = best {
            let s = &mut self.speakers[idx];
            if let Some(c) = &mut s.centroid {
                // EMA centroids — returning voices are re-recognized, slowly drifting
                c.pitch_hz = c.pitch_hz * 0.7 + pitch * 0.3;
                if timbre > 0.0 {
                    c.centroid_hz = c.centroid_hz * 0.8 + timbre * 0.2;
                }
            }
            s.utterances += 1;
            return SpeakerInfo {
                id: s.id,
                label: s.label.clone(),
                pitch_hz: Some(s.centroid.as_ref().unwrap().pitch_hz.round()),
                utterances: s.utterances,
            };
        }

        // new voice
        let id = self.speakers.len() as u32 + 1;
        let s = Speaker {
            id,
            label: format!("Speaker {id}"),
            centroid: Some(Centroid {
                pitch_hz: pitch,
                centroid_hz: timbre,
            }),
            utterances: 1,
        };
        let info = SpeakerInfo {
            id,
            label: s.label.clone(),
            pitch_hz: Some(pitch.round()),
            utterances: 1,
        };
        self.speakers.push(s);
        info
    }

    pub fn list(&self) -> Vec<SpeakerInfo> {
        self.speakers
            .iter()
            .map(|s| SpeakerInfo {
                id: s.id,
                label: s.label.clone(),
                pitch_hz: s.centroid.as_ref().map(|c| c.pitch_hz.round()),
                utterances: s.utterances,
            })
            .collect()
    }

    /// Associate a self-introduced name with a speaker's voice.
    pub fn rename(&mut self, id: u32, name: &str) {
        if let Some(s) = self.speakers.iter_mut().find(|s| s.id == id) {
            s.label = format!("{name} (Speaker {})", s.id);
        }
    }

    pub fn label_of(&self, id: u32) -> String {
        self.speakers
            .iter()
            .find(|s| s.id == id)
            .map(|s| s.label.clone())
            .unwrap_or_else(|| format!("Speaker {id}"))
    }

    pub fn reset(&mut self) {
        self.speakers.clear();
    }
}

impl Default for SpeakerRegistry {
    fn default() -> Self {
        Self::new()
    }
}
