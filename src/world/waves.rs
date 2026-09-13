//! P2130 — OTD3 WAVES — sound and light, the two waves every human meets.
//!
//! SOUND is a pressure wave travelling THROUGH matter (air, water, steel) —
//! it cannot cross a vacuum, which is why space is silent. Its speed depends
//! on the medium and its temperature; its pitch on the frequency; its
//! loudness on the amplitude.
//!
//! LIGHT is an electromagnetic wave that needs no medium at all — it crosses
//! vacuum at exactly 299,792,458 m/s (by definition: the metre was built
//! from it). Radio, microwaves, infrared, visible, UV, X-rays and gamma are
//! all the same wave at different frequencies — the spectrum is one thing.
//!
//! Laws carried here:
//!   Wave equation     — v = f·λ (speed = pitch × wavelength, always)
//!   Sound speed/air   — v ≈ 331.3 + 0.606·T °C m/s (hotter = faster)
//!   Sound speed/media — air 343, water 1481, steel 5960 m/s
//!   Doppler           — approaching pitch rises, receding falls
//!   Inverse square    — intensity ÷ distance² — the reason far is quiet
//!   Snell             — light bends crossing media (n = c/v)
//!   Wien              — hot things glow; peak wavelength says how hot
//!   Echo ranging      — d = v·t/2 (sonar, bats, and OTD's own audio engine)

use super::eval::{ConsoleLine, LineKind, World};

/// Speed of light in vacuum, m/s.
pub const C_LIGHT: f64 = 299_792_458.0;
/// Speed of sound in air at 20 °C, m/s.
pub const SOUND_AIR_20C: f64 = 343.0;

/// Speed of sound in air at temperature T (°C): v = 331.3 + 0.606·T.
pub fn sound_speed_air(temp_c: f64) -> f64 {
    331.3 + 0.606 * temp_c
}

/// Speed of sound in a medium, m/s (textbook).
pub fn sound_speed_in(mat: &str) -> f64 {
    match mat {
        "air" => 343.0,
        "hydrogen" => 1310.0,   // light atoms jiggle fast — the squeakiest gas
        "helium" => 1007.0,    // party-trick voice physics
        "carbon_dioxide" => 267.0,
        "water" => 1481.0,     // 4.3× air — whales talk across oceans
        "sea water" => 1531.0,
        "ice" => 3200.0,
        "ethanol" => 1144.0,
        "mercury" => 1451.0,
        "oil" => 1440.0,
        "rubber" => 60.0,      // the silence of engine mounts
        "wood" | "oak" | "teak" => 3800.0, // along the grain
        "pine" => 3500.0,
        "glass" => 4540.0,
        "concrete" => 3200.0,
        "marble" => 3810.0,
        "iron" | "steel" | "stainless" => 5960.0, // rail-tapping telegraph
        "aluminum" => 6320.0,  // the fastest common metal
        "copper" => 4760.0,
        "brass" | "bronze" => 4700.0,
        "titanium" => 6070.0,
        "gold" => 3240.0,
        "silver" => 3650.0,
        "lead" => 2160.0,      // dead metal — X-ray rooms line with it
        "nickel" => 6040.0,
        "tungsten" => 5180.0,
        "granite" => 5950.0,
        "diamond" => 12000.0, // the champion of them all
        _ => 1500.0,
    }
}

/// Refractive index n = c/v (light in the material).
pub fn refractive_index(mat: &str) -> f64 {
    match mat {
        "vacuum" => 1.0,
        "air" => 1.0003,
        "ice" => 1.31,
        "water" => 1.333,
        "ethanol" => 1.36,
        "acetone" => 1.36,
        "glycerin" => 1.473,
        "glass" => 1.52,      // crown glass
        "plastic" => 1.49,   // acrylic
        "quartz" => 1.46,
        "diamond" => 2.417,  // the sparkle champion — cut angles are pure Snell
        "sapphire" => 1.77,
        "ruby" => 1.77,
        _ => 1.5,
    }
}

/// Doppler shift for a moving source: f' = f·v/(v − v_s·cosθ).
pub fn doppler(f_hz: f64, v_sound: f64, v_source_mps: f64) -> f64 {
    let denom = v_sound - v_source_mps;
    if denom.abs() < 1e-6 { return f64::INFINITY; }
    f_hz * v_sound / denom
}

/// Mach number of a speed in a given medium.
pub fn mach(speed_mps: f64, medium: &str, temp_c: f64) -> f64 {
    let v = if medium == "air" { sound_speed_air(temp_c) } else { sound_speed_in(medium) };
    speed_mps / v
}

/// Human hearing: 20 Hz .. 20 kHz. Returns a description of where a
/// frequency sits in the full spectrum (infrasound → ultrasound).
pub fn hearing_band(f_hz: f64) -> &'static str {
    if f_hz < 0.1 { "DC / geological" }
    else if f_hz < 20.0 { "infrasound — below hearing: earthquakes, elephants, thunder-close" }
    else if f_hz < 250.0 { "bass — felt as much as heard (sub-bass + bass)" }
    else if f_hz < 2000.0 { "midrange — the human voice lives here (250 Hz–2 kHz)" }
    else if f_hz < 6000.0 { "presence — speech consonants, clarity" }
    else if f_hz < 20_000.0 { "brilliance — air, cymbals, the top of hearing" }
    else if f_hz < 100_000.0 { "ultrasound — bats (up to 200 kHz), dolphins, medical scans" }
    else if f_hz < 1e9 { "megasonic / NDT — industrial and lab imaging" }
    else { "hypersound / thermal phonons" }
}

/// Wavelength of sound in air: λ = v/f.
pub fn sound_wavelength(f_hz: f64, temp_c: f64) -> f64 {
    sound_speed_air(temp_c) / f_hz
}

/// The full electromagnetic spectrum ladder (frequency → name + λ).
pub fn em_band(f_hz: f64) -> (&'static str, f64) {
    let lambda = C_LIGHT / f_hz.max(1e-9);
    let name = if f_hz < 3e3 { "radio (ELF/VLF/LF/MF/HF/VHF)" }
        else if f_hz < 3e9 { "radio (UHF) / radar / microwave boundary" }
        else if f_hz < 3e11 { "microwave — WiFi, radar, ovens" }
        else if f_hz < 4e14 { "infrared — heat you feel from across the room" }
        else if f_hz < 7.9e14 { "VISIBLE LIGHT — the 300 THz-wide window your eyes answer to" }
        else if f_hz < 3e16 { "ultraviolet — sunburn, vitamin D, blacklights" }
        else if f_hz < 3e19 { "X-rays — see through flesh, stopped by bone" }
        else { "gamma rays — nuclear transitions, the highest energies" };
    (name, lambda)
}

/// Wien's displacement law: peak emission wavelength of a hot body (m).
/// λ_max = b/T with b = 2.8978e-3 m·K. The Sun (5800 K) peaks at 500 nm —
/// the exact middle of visible light. Not a coincidence: eyes evolved there.
pub fn wien_peak(t_kelvin: f64) -> f64 {
    2.897_771_955e-3 / t_kelvin.max(1.0)
}

/// Echo ranging: distance from a round trip time. d = v·t/2.
pub fn echo_distance(v_mps: f64, round_trip_s: f64) -> f64 {
    v_mps * round_trip_s / 2.0
}

/// Attenuation of sound in air (dB loss per metre, rough, by humidity+freq):
/// the classic teaching value ~1 dB/100 m at speech frequencies, rising fast
/// with frequency. Returns dB/m.
pub fn air_absorption_db_per_m(f_hz: f64, humidity_pct: f64) -> f64 {
    // simplified ISO 9613-shaped curve: ∝ f², eased by humidity
    let f = f_hz.max(1.0);
    let hum = (humidity_pct / 50.0).clamp(0.2, 2.0);
    (f / 1000.0).powi(2) * 0.01 / hum.sqrt()
}

/// `simulate: sound` — the acoustics report of the scene: what the parts
/// are made of tells you how sound crosses them, what they'd sound like
/// when struck, and how far away you could hear them.
pub fn sound_sim(world: &World, temp_c: f64) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    let v_air = sound_speed_air(temp_c);
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!("SOUND REPORT — air at {:.0} °C carries it at {:.1} m/s ({:.0} km/h); in water it would do {:.0} m/s, in iron {:.0} m/s",
            temp_c, v_air, v_air * 3.6, sound_speed_in("water"), sound_speed_in("iron")),
    });
    let visible: Vec<&super::eval::Part> = world.parts.iter().filter(|p| !p.hidden).collect();
    if visible.is_empty() {
        out.push(ConsoleLine { kind: LineKind::Warn, text: "nothing to ring — the scene is empty".into() });
        return out;
    }
    for part in &visible {
        let mat = part.material.map(|m| m.name).unwrap_or("plastic");
        let v_mat = sound_speed_in(mat);
        // longitudinal wave → the material's "ring": metals ring because
        // low internal loss; rubber is silent because huge loss
        let ring = match mat {
            "iron" | "steel" | "stainless" | "brass" | "bronze" | "copper" | "gold" | "silver" | "aluminum" | "titanium" | "tungsten" | "chrome" | "glass" => "rings — a bell, if you shape it so",
            "rubber" | "foam" | "fabric" => "silences — the molecular chains eat vibration (why engine mounts are rubber)",
            "wood" | "oak" | "pine" | "teak" => "resonates warmly — the violin family's whole career",
            "water" | "oil" | "mercury" | "ethanol" => "splashes — liquids carry sound but never ring",
            "air" | "hydrogen" | "helium" | "methane" | "nitrogen" | "oxygen" => "is the medium itself",
            _ => "thuds",
        };
        // wavelength of the part's own "note": a bar's fundamental ≈ v/2L
        let bb = part.mesh.bbox();
        let sz = bb.size();
        let l_m = sz.x().max(sz.y()).max(sz.z()) / 1000.0;
        let f_note = if l_m > 0.001 { v_mat / (2.0 * l_m) } else { 0.0 };
        let band = if f_note > 0.0 { hearing_band(f_note) } else { "too small to speak" };
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("  {} [{}] — sound inside: {:.0} m/s; struck, a {:.0} Hz tone ({}); {}",
                part.name, mat, v_mat, f_note, band, ring),
        });
    }
    // echo ranging demo tied to the scene size
    let bb = world.stats.bbox;
    let sz = bb.size();
    let extent_m = sz.x().max(sz.y()).max(sz.z()) / 1000.0;
    if extent_m > 0.01 {
        let t_echo = 2.0 * extent_m / v_air;
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!("  sonar across this scene ({:.2} m): shout, wait {:.1} ms, hear the wall — d = v·t/2, the law OTD's PerceptAudio engine uses backwards",
                extent_m, t_echo * 1000.0),
        });
    }
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!("  reference tones: A4 = 440 Hz (λ = {:.2} m in this air), middle-C = 262 Hz, bat chirps at 50 kHz (λ = {:.1} mm) — ultrasound is just very high music",
            sound_wavelength(440.0, temp_c), sound_wavelength(50_000.0, temp_c) * 1000.0),
    });
    out
}

/// `simulate: light` — the optics report: transparency, refraction, and
/// Wien glow of the parts.
pub fn light_sim(world: &World, temp_c: f64) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!("LIGHT REPORT — {} nm is the middle of what you can see; the scene sits at {:.1} °C and glows brightest at {:.0} nm",
            550, temp_c, wien_peak(temp_c + 273.15) * 1e9),
    });
    let visible: Vec<&super::eval::Part> = world.parts.iter().filter(|p| !p.hidden).collect();
    if visible.is_empty() {
        out.push(ConsoleLine { kind: LineKind::Warn, text: "nothing to see — the scene is empty".into() });
        return out;
    }
    for part in &visible {
        let mat = part.material.map(|m| m.name).unwrap_or("plastic");
        let n = refractive_index(mat);
        let opaque = part.material.map(|m| m.opacity).unwrap_or(1.0) >= 0.999;
        let optics = if opaque {
            let metal = part.material.map(|m| m.metal).unwrap_or(false);
            if metal { "mirror — metals reflect ~95% by free-electron screen" } else { "opaque — absorbs or scatters everything" }
        } else {
            &format!("n = {:.3} — light enters at {:+.1}° per 45° incidence (Snell); a lens waiting to happen", n,
                // Snell: sin θ₂ = sin 45° / n
                ((45.0f64.to_radians().sin() / n).asin().to_degrees() - 45.0))
        };
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("  {} [{}] — {}", part.name, mat, optics),
        });
    }
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!("  the Sun (5778 K) peaks at {:.0} nm — green, mid-visible; a candle flame (1400 K) at {:.0} nm — infrared, which is why candlelight feels warm and looks orange",
            wien_peak(5778.0) * 1e9, wien_peak(1400.0) * 1e9),
    });
    // total internal reflection teaser — the fibre-optic law
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!("  glass at n=1.52: light hitting the inside face beyond 41.1° cannot leave — total internal reflection, the entire internet rides on it in fibre",
        ),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wave_equation_holds() {
        // A4 in 20 °C air: λ = v/f with v = 331.3 + 0.606·20 = 343.42
        let l = sound_wavelength(440.0, 20.0);
        assert!((l - sound_speed_air(20.0) / 440.0).abs() < 1e-9);
        // f·λ = v for any pair
        let v = sound_speed_air(25.0);
        assert!((1000.0 * (v / 1000.0) - v).abs() < 1e-9);
    }

    #[test]
    fn hot_air_carries_faster() {
        assert!(sound_speed_air(30.0) > sound_speed_air(10.0));
        assert!((sound_speed_air(20.0) - 343.4).abs() < 0.5);
    }

    #[test]
    fn diamond_outruns_iron_outruns_rubber() {
        assert!(sound_speed_in("diamond") > sound_speed_in("iron"));
        assert!(sound_speed_in("iron") > sound_speed_in("rubber"));
        assert!(sound_speed_in("water") > sound_speed_in("air"));
    }

    #[test]
    fn doppler_ambulance() {
        // source at 30 m/s toward you, sounding 440 Hz
        let f = doppler(440.0, 343.0, 30.0);
        assert!(f > 440.0);
        // and away:
        let f2 = doppler(440.0, 343.0, -30.0);
        assert!(f2 < 440.0);
    }

    #[test]
    fn mach_1_is_the_sound_barrier() {
        assert!((mach(343.0, "air", 20.0) - 1.0).abs() < 0.01);
        assert!(mach(6000.0, "iron", 20.0) > 0.9); // trains rumble near Mach 1 in rail
    }

    #[test]
    fn hearing_bands_are_ordered() {
        assert!(hearing_band(10.0).contains("infra"));
        assert!(hearing_band(500.0).contains("midrange"));
        assert!(hearing_band(25_000.0).contains("ultra"));
    }

    #[test]
    fn em_spectrum_ladder() {
        let (name, l) = em_band(5.0e14);
        assert!(name.contains("VISIBLE"));
        assert!((l - 6e-7).abs() < 1e-7); // ~600 nm
        assert!(em_band(2.45e9).0.contains("microwave")); // the oven
        assert!(em_band(3e18).0.contains("X-ray"));
    }

    #[test]
    fn wien_says_the_sun_is_green() {
        assert!((wien_peak(5778.0) * 1e9 - 501.0).abs() < 5.0);
        assert!(wien_peak(300.0) * 1e9 > 9000.0); // room temp glows deep IR
    }

    #[test]
    fn snell_bends_toward_normal() {
        let n = refractive_index("water");
        let r = 45.0f64.to_radians().sin() / n;
        assert!(r.asin().to_degrees() < 45.0);
    }

    #[test]
    fn echo_range_round_trip() {
        // 343 m/s, 0.1 s round trip → 17.15 m (thunder-counting physics)
        assert!((echo_distance(343.0, 0.1) - 17.15).abs() < 0.01);
    }

    #[test]
    fn sims_run() {
        let w = crate::world::eval::compile("scene \"t\"\na = cube 3cm at (0, 2cm, 0) material: glass\nb = sphere 2cm at (6cm, 2cm, 0) material: iron");
        let s = sound_sim(&w, 20.0);
        assert!(s.iter().any(|l| l.text.contains("SOUND REPORT")));
        let l = light_sim(&w, 20.0);
        assert!(l.iter().any(|l| l.text.contains("LIGHT REPORT")));
        assert!(l.iter().any(|l| l.text.contains("Snell")));
    }
}
