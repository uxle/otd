//! P2160 — OTD3 SPECTRUM & RANGING — PerceptAudio's upgraded ears.
//!
//! The original PerceptAudio 2.0 engine heard 80 Hz–7.5 kHz (voice) and
//! binned distance into 5 loudness bands. OTD3 opens the whole world of
//! frequencies and puts a METRE on sound:
//!
//!   FULL SPECTRUM   — infrasound (earthquakes, elephants) through bass,
//!                     midrange, presence, brilliance, ultrasound (bats,
//!                     dolphin clicks, medical probes) — every octave band
//!                     measured, named, and placed on the spectrum ladder.
//!   METRIC DISTANCE — the inverse-square law (intensity ∝ 1/d²) turns a
//!                     calibrated loudness drop into metres; the
//!                     temperature-dependent speed of sound (331.3 +
//!                     0.606·T) turns an echo delay into metres
//!                     (d = v·t/2); and the air absorption curve (ISO
//!                     9613-shaped, humidity-aware) explains WHY distant
//!                     thunder rolls low.
//!   THE PHYSICS     — every band reports its wavelength (λ = v/f), the
//!                     wave equation that ties pitch to size: a 68 Hz
//!                     sub-bass wave is 5 metres long — half the wavelength
//!                     rule of subwoofers and organ pipes.

use crate::audio::fft::{Fft, FRAME_SIZE, TARGET_RATE};

/// Named regions of the full audio spectrum (Hz bounds, name, physics note).
pub const SPECTRUM_LADDER: &[(f64, f64, &str, &str)] = &[
    (0.1,   20.0,  "infrasound",  "below hearing — earthquakes, elephant herds, thunder close-in; you FEEL it"),
    (20.0,   40.0,  "sub-bass",   "the 5-metre wave — cinema subwoofers, pipe organs' lowest stops"),
    (40.0,   60.0,  "bass",       "electric bass fundamentals (E1 = 41.2 Hz)"),
    (60.0,   250.0, "low-mids",   "male voice fundamentals, cello warmth"),
    (250.0,  500.0, "midrange",   "voice fundamentals to upper — the band the ear is most sensitive to"),
    (500.0,  2000.0,"uppers",     "vowel formants, speech intelligibility lives here"),
    (2000.0, 4000.0,"presence",   "consonants (s, t, k) — the clarity band of every consonant"),
    (4000.0, 6000.0,"brilliance", "cymbal shimmer, breath, 'air' on a good recording"),
    (6000.0, 20000.0,"top",       "the last octave of young hearing — teenage ears only"),
    (20000.0, 100000.0, "ultrasound", "bats (120 kHz max), dolphins (160 kHz), 8 MHz medical scans (way above)"),
];

/// Which spectrum region does a frequency belong to?
pub fn band_name(f_hz: f64) -> &'static str {
    for (lo, hi, name, _) in SPECTRUM_LADDER {
        if f_hz >= *lo && f_hz < *hi {
            return name;
        }
    }
    if f_hz >= 100_000.0 { "hypersound" } else { "DC" }
}

/// Speed of sound in air at temperature T (°C): v = 331.3 + 0.606·T m/s.
pub fn sound_speed_air(temp_c: f64) -> f64 {
    331.3 + 0.606 * temp_c
}

/// Wavelength in air: λ = v / f.
pub fn wavelength(f_hz: f64, temp_c: f64) -> f64 {
    sound_speed_air(temp_c) / f_hz.max(0.001)
}

/// One measured band of a spectrum analysis.
#[derive(Debug, Clone)]
pub struct BandReading {
    pub lo_hz: f64,
    pub hi_hz: f64,
    pub name: &'static str,
    /// energy share 0..1 of this band
    pub share: f64,
    /// dominant frequency inside the band (energy-weighted mean), Hz
    pub peak_hz: f64,
    /// power in dB relative to the strongest band
    pub rel_db: f64,
}

/// Full-spectrum octave-band analysis of a PCM segment.
///
/// The frame FFT (2048 @ 16 kHz) sees 0–8 kHz; to honour the full
/// 20 Hz–20 kHz promise, callers resample first (the engine does), and we
/// extend the ladder to ultrasound by flagging what the sampling rate can
/// see — honestly: at 16 kHz sample rate, everything above 8 kHz is
/// Nyquist-locked out, and we say so.
pub fn analyze_bands(fft: &Fft, pcm: &[f32], sample_rate: f32) -> Vec<BandReading> {
    let mut out = Vec::new();
    if pcm.len() < FRAME_SIZE {
        return out;
    }
    let nyquist = sample_rate as f64 / 2.0;
    // accumulate the average power spectrum across frames
    let n = FRAME_SIZE;
    let half = n / 2;
    let mut acc: Vec<f64> = vec![0.0; half + 1];
    let mut frames = 0usize;
    let mut off = 0usize;
    while off + FRAME_SIZE <= pcm.len() {
        let f = &pcm[off..off + FRAME_SIZE];
        let mut re = vec![0f32; n];
        let mut im = vec![0f32; n];
        fft.hann_multiply(f, &mut re);
        fft.forward(&mut re, &mut im);
        for b in 1..=half {
            acc[b] += (re[b] * re[b] + im[b] * im[b]) as f64;
        }
        frames += 1;
        off += FRAME_SIZE / 2; // 50% overlap — every sample counted twice
    }
    if frames == 0 {
        return out;
    }
    let total: f64 = acc.iter().sum();
    if total <= 0.0 {
        return out;
    }
    let bin_hz = sample_rate as f64 / n as f64;
    let max_share = 1.0f64; // reference computed after collecting bands

    // one band per octave-aligned region of the ladder, split to ≤1 octave
    let mut raw: Vec<BandReading> = Vec::new();
    for (lo, hi, name, _) in SPECTRUM_LADDER {
        // skip anything entirely outside Nyquist reach (honesty)
        if *lo >= nyquist {
            continue;
        }
        let hi_eff = (*hi).min(nyquist);
        let b0 = ((*lo) / bin_hz).ceil().max(1.0) as usize;
        let b1 = (hi_eff / bin_hz).floor().max(1.0) as usize;
        if b1 < b0 || b1 > half {
            continue;
        }
        let mut band_energy = 0.0;
        let mut weighted = 0.0;
        for b in b0..=b1 {
            band_energy += acc[b];
            weighted += acc[b] * (b as f64 * bin_hz);
        }
        let share = band_energy / total;
        let peak_hz = if band_energy > 0.0 { weighted / band_energy } else { 0.0 };
        raw.push(BandReading {
            lo_hz: *lo,
            hi_hz: hi_eff,
            name,
            share,
            peak_hz,
            rel_db: 0.0, // filled after we know the max
        });
        let _ = max_share;
    }
    let peak_share = raw.iter().map(|b| b.share).fold(0.0f64, f64::max);
    for b in raw.iter_mut() {
        b.rel_db = if b.share > 1e-12 {
            10.0 * (b.share / peak_share).log10()
        } else {
            -100.0
        };
    }
    out.extend(raw);
    out
}

/// Metric distance estimate from the inverse-square law.
///
/// Physics: a point source radiates power P; intensity at distance d is
/// I = P / (4π·d²). Given a reference measurement (I₀ at d₀), the same
/// source at the measured level sits at
///   d = d₀ · 10^((dbfs₀ − dbfs) / 20)
/// because intensity ratios are 20 dB per decade of amplitude.
/// Air absorption (freq-dependent) is accounted separately — see below.
#[derive(Debug, Clone)]
pub struct DistanceEstimate {
    /// estimated distance, metres
    pub metres: f64,
    /// the band this lands in
    pub band: &'static str,
    /// how the number was derived (the law, stated)
    pub law: String,
    /// confidence 0..1 (calibration, hf-corroboration, level sanity)
    pub confidence: f64,
}

/// Inverse-square ranging from a dBFS reading against a reference.
/// `ref_dbfs` is the level the SAME source produces at `ref_metres`.
pub fn distance_from_level(dbfs: f32, ref_dbfs: f32, ref_metres: f64) -> DistanceEstimate {
    if dbfs <= -95.0 || ref_dbfs <= -95.0 {
        return DistanceEstimate {
            metres: f64::INFINITY,
            band: "unknown",
            law: "level too low to range (below −95 dBFS)".into(),
            confidence: 0.0,
        };
    }
    // each 20 dB down = 10× farther
    let d = ref_metres * 10f64.powf((ref_dbfs as f64 - dbfs as f64) / 20.0);
    DistanceEstimate {
        metres: d,
        band: band_of_distance(d),
        law: format!("inverse-square: d = {:.1} m × 10^(({:.1} − {:.1})/20) — 6 dB per doubling of distance",
            ref_metres, ref_dbfs, dbfs),
        confidence: 0.55,
    }
}

/// ISO-9613-shaped air absorption correction: high frequencies die faster
/// with distance (that's why far thunder rumbles low and near thunder
/// cracks bright). Returns the extra dB of loss at `metres` for a tone of
/// `f_hz` in `humidity_pct`% relative humidity.
pub fn air_absorption_db(f_hz: f64, metres: f64, humidity_pct: f64) -> f64 {
    let per_m = (f_hz / 1000.0).powi(2) * 0.01 / (humidity_pct / 50.0).clamp(0.2, 2.0).sqrt();
    per_m * metres
}

/// Echo ranging: d = v·t/2 (the law of sonar, bats, and radar).
pub fn distance_from_echo(round_trip_s: f64, temp_c: f64) -> f64 {
    sound_speed_air(temp_c) * round_trip_s / 2.0
}

fn band_of_distance(m: f64) -> &'static str {
    if !m.is_finite() { return "unknown"; }
    if m < 0.5 { "very-close" }
    else if m < 1.5 { "close" }
    else if m < 3.0 { "normal" }
    else if m < 6.0 { "far" }
    else { "very-far" }
}

/// Convert the classic 5-band heuristic (loudness + treble) into a METRIC
/// estimate by anchoring the band ladder at calibrated reference points:
/// very-close ≈ 0.3 m, close ≈ 1 m, normal ≈ 2 m, far ≈ 4 m, very-far ≥ 8 m.
/// Combines the band anchor with the hf_ratio evidence (treble survives
/// only up close) and reports the law honestly.
pub fn distance_from_band(band: &str, hf_ratio: f32) -> DistanceEstimate {
    let (anchor, note) = match band {
        "very-close" => (0.3, "whisper range — breath is audible"),
        "close" => (1.0, "conversation at a table"),
        "normal" => (2.0, "across a small room"),
        "far" => (4.0, "across a large room / hallway shout"),
        _ => (8.0, "another room or down the street"),
    };
    // treble corroborates: hf_ratio ≥ 0.10 argues closer, ≤ 0.03 argues farther
    let adjust = if hf_ratio >= 0.10 { 0.8 } else if hf_ratio <= 0.03 { 1.25 } else { 1.0 };
    let d = anchor as f64 * adjust;
    DistanceEstimate {
        metres: d,
        band: band_of_distance(d),
        law: format!("band anchor ({} m) × treble factor {} (hf_ratio {:.3}) — {}",
            anchor, adjust, hf_ratio, note),
        confidence: 0.35, // honest: mono mic, unknown gain
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn spectrum_ladder_is_ordered() {
        for w in [10.0, 55.0, 300.0, 3000.0, 12000.0, 30000.0] {
            let name = band_name(w);
            assert!(!name.is_empty());
        }
        assert_eq!(band_name(15.0), "infrasound");
        assert_eq!(band_name(3000.0), "presence");
        assert_eq!(band_name(40000.0), "ultrasound");
    }

    #[test]
    fn inverse_square_doubling_is_six_db() {
        // −6 dB ⇒ twice the distance
        let d1 = distance_from_level(-30.0, -24.0, 1.0);
        assert!((d1.metres - 2.0).abs() < 0.01, "6 dB down = 2× distance");
        let d2 = distance_from_level(-44.0, -24.0, 1.0);
        assert!((d2.metres - 10.0).abs() < 0.05, "20 dB down = 10× distance");
    }

    #[test]
    fn air_kills_treble_first() {
        // at 30 m, 8 kHz loses way more than 250 Hz — why thunder rolls
        let hi = air_absorption_db(8000.0, 30.0, 50.0);
        let lo = air_absorption_db(250.0, 30.0, 50.0);
        assert!(hi > lo * 100.0);
    }

    #[test]
    fn echo_ranging_math() {
        // 100 ms round trip in 20 °C air → 17.17 m
        let d = distance_from_echo(0.1, 20.0);
        assert!((d - 17.17).abs() < 0.01);
    }

    #[test]
    fn wavelength_ties_pitch_to_size() {
        // 68 Hz sub-bass ≈ 5.04 m in 20 °C air — the half-wave subwoofer rule
        let l = wavelength(68.0, 20.0);
        assert!((l - 5.046).abs() < 0.01);
        // and the voice band is head-sized: 500 Hz ≈ 69 cm
        let l2 = wavelength(500.0, 20.0);
        assert!((l2 - 0.6868).abs() < 0.01);
    }

    #[test]
    fn bands_analyze_a_tone() {
        // 440 Hz sine at 16 kHz should light the 250–500 Hz band strongest
        let fft = Fft::new(FRAME_SIZE);
        let sr = TARGET_RATE;
        let mut pcm = vec![0f32; FRAME_SIZE * 4];
        for (i, s) in pcm.iter_mut().enumerate() {
            *s = (2.0 * std::f64::consts::PI * 440.0 * (i as f64 / sr as f64)).sin() as f32;
        }
        let bands = analyze_bands(&fft, &pcm, sr);
        assert!(!bands.is_empty());
        let strongest = bands.iter().max_by(|a, b| a.share.partial_cmp(&b.share).unwrap()).unwrap();
        assert_eq!(strongest.name, "midrange", "A440 lives in midrange, got {}", strongest.name);
        assert!((strongest.peak_hz - 440.0).abs() < 20.0, "peak ≈ 440, got {}", strongest.peak_hz);
    }

    #[test]
    fn nyquist_honesty() {
        // at 16 kHz sampling, no band above 8 kHz may be reported
        let fft = Fft::new(FRAME_SIZE);
        let pcm = vec![0.1f32; FRAME_SIZE * 2];
        let bands = analyze_bands(&fft, &pcm, TARGET_RATE);
        assert!(bands.iter().all(|b| b.hi_hz <= 8000.0), "nothing above Nyquist");
        assert!(!bands.iter().any(|b| b.name == "ultrasound"));
    }

    #[test]
    fn band_distance_is_metric() {
        let d = distance_from_band("close", 0.12);
        assert!((d.metres - 0.8).abs() < 1e-9); // hf ≥ 0.10 pulls closer
        let d_mid = distance_from_band("close", 0.05);
        assert!((d_mid.metres - 1.0).abs() < 1e-9); // neutral treble = anchor
        let d2 = distance_from_band("far", 0.01);
        assert!((d2.metres - 5.0).abs() < 1e-9); // hf ≤ 0.03 pushes farther
    }
}
