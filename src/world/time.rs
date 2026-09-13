//! P2140 — OTD3 TIME & MOTION — time, speed, distance, and relativity.
//!
//! Time is what a clock reads; speed is distance per time; and the two
//! braid together in Einstein's relativity: the faster you move through
//! space, the slower you move through time. Even GPS satellites — moving
//! at 3.9 km/s — run slow by 7 μs a day and must be corrected or maps
//! drift by kilometres.
//!
//! Laws carried here:
//!   Galileo          — all bodies fall at the same rate (no mass in g!)
//!   SUVAT            — v = u + at;  s = ut + ½at²;  v² = u² + 2as
//!   Pendulum (Huygens) — T = 2π√(L/g), mass-free: the metronome of physics
//!   Free-fall time   — t = √(2h/g)
//!   Time dilation    — Δt = γ·Δt₀, γ = 1/√(1 − v²/c²)
//!   Length contraction — L = L₀/γ
//!   Velocity add (rel.) — (u+v)/(1 + uv/c²) — the reason c is a speed limit
//!   Einstein         — E = mc²; energy has mass, mass has energy

use super::eval::{ConsoleLine, LineKind, World};

/// Speed of light, m/s.
pub const C_LIGHT: f64 = 299_792_458.0;

/// The famous speed table (m/s) — a ladder from snail to light.
pub fn speed_table() -> Vec<(&'static str, f64)> {
    vec![
        ("snail", 0.001),
        ("walking human", 1.4),
        ("Olympic sprinter", 10.4),
        ("city traffic", 11.0),
        ("highway car", 33.0),
        ("peregrine falcon dive", 108.0),   // fastest animal
        ("Formula 1 car", 103.0),
        ("speed of sound (air, 20 °C)", 343.0),
        ("commercial jet", 250.0),
        (".357 Magnum bullet", 440.0),
        ("speed of sound in water", 1481.0),
        ("SR-71 Blackbird", 980.0),
        ("rifle bullet (AK-47)", 715.0),
        ("Earth orbit speed", 7_800.0),      // 28,000 km/h
        ("escape velocity (Earth)", 11_186.0),
        ("Apollo re-entry", 11_100.0),
        ("New Horizons probe", 16_260.0),
        ("Sun's orbit around the galaxy", 220_000.0),
        ("speed of sound in iron", 5_960.0),
        ("speed of light in vacuum", C_LIGHT),
    ]
}

/// Free-fall time from height h: t = √(2h/g).
pub fn freefall_time(h_m: f64, g: f64) -> f64 {
    (2.0 * h_m / g.max(1e-9)).sqrt()
}

/// Impact speed: v = √(2gh).
pub fn impact_speed(h_m: f64, g: f64) -> f64 {
    (2.0 * g * h_m).sqrt()
}

/// Pendulum period: T = 2π√(L/g) — no mass anywhere in the formula.
pub fn pendulum_period(l_m: f64, g: f64) -> f64 {
    2.0 * std::f64::consts::PI * (l_m / g.max(1e-9)).sqrt()
}

/// Lorentz factor γ.
pub fn lorentz(v_mps: f64) -> f64 {
    let b = (v_mps / C_LIGHT).clamp(0.0, 0.999999999);
    1.0 / (1.0 - b * b).sqrt()
}

/// Time dilation: how much proper-time Δt₀ stretches to lab time.
pub fn dilated(dt0_s: f64, v_mps: f64) -> f64 {
    lorentz(v_mps) * dt0_s
}

/// Relativistic velocity addition: (u + v)/(1 + uv/c²).
pub fn vel_add(u: f64, v: f64) -> f64 {
    (u + v) / (1.0 + u * v / (C_LIGHT * C_LIGHT))
}

/// Rest-mass energy: E = mc².
pub fn rest_energy(mass_kg: f64) -> f64 {
    mass_kg * C_LIGHT * C_LIGHT
}

/// `simulate: time` — the motion & time report for the scene: how long a
/// drop takes, pendulum scales, and how relativistic the scene's motions are.
pub fn time_sim(world: &World, g: f64) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!("TIME & MOTION REPORT — g = {} m/s²; Galileo's law: every mass falls at the same rate (mass cancels, always)", g),
    });
    let bb = world.stats.bbox;
    let h_m = bb.max.y().max(0.0) / 1000.0; // scene height
    if h_m > 0.01 {
        let t = freefall_time(h_m, g);
        let v = impact_speed(h_m, g);
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!("  drop this scene's own height ({:.2} m): t = √(2h/g) = {:.2} s, landing at v = √(2gh) = {:.1} m/s — mass never enters, only height and g",
                h_m, t, v),
        });
        // pendulum matched to scene height
        let tt = pendulum_period(h_m, g);
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("  a pendulum as tall as this scene swings with T = 2π√(L/g) = {:.2} s per beat — Huygens' clock: 1 m beats ~1 s on Earth, ~2.4 s on the Moon",
                tt),
        });
    }
    // relativity corner: if the scene moved at various speeds
    let mass_kg = world.stats.total_mass_g / 1000.0;
    if mass_kg > 0.0 {
        let e = rest_energy(mass_kg);
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!("  E = mc² for this scene's {:.2} kg = {:.3} GJ — annihilate it and you run a city for days; that is what 'mass is frozen energy' means",
                mass_kg, e / 1e9),
        });
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("  at jet speed (250 m/s) time dilates ×{:.12} — unmeasurable; at 99% of c it would stretch ×{:.2}: a 1-year trip lands 7.09 years later home",
                lorentz(250.0), lorentz(0.99 * C_LIGHT)),
        });
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: "  the speed limit is structural: u+v never reaches c — two 99% c trains pass at 0.99995 c, not 1.98 c (relativistic addition)".into(),
        });
    }
    // where common speeds sit
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: "  the ladder: snail 0.001 · walk 1.4 · sprint 10.4 · sound 343 · jet 250 · bullet 715 · orbit 7.8 km/s · escape 11.2 km/s · light 299,792.458 km/s".into(),
    });
    // one-second distance yardsticks
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!("  in ONE second: sound crosses {:.0} m, light crosses {:.0} km (7.5 laps of Earth), and Earth moves {} km around the Sun",
            343.0, C_LIGHT / 1000.0, 29.8),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn galileo_same_rate() {
        // a feather and a cannonball: same impact time from 5 m
        assert!((freefall_time(5.0, 9.81) - 1.0096).abs() < 0.001);
    }

    #[test]
    fn suvat_cross_checks() {
        // v = gt and t = √(2h/g) must agree with v² = 2gh
        let h = 20.0;
        let g = 9.81;
        let v = impact_speed(h, g);
        let t = freefall_time(h, g);
        assert!((v - g * t).abs() < 1e-9);
    }

    #[test]
    fn pendulum_is_mass_free() {
        let t1 = pendulum_period(1.0, 9.81);
        assert!((t1 - 2.006).abs() < 0.01); // the famous 1 m ≈ 2 s
        let t2 = pendulum_period(1.0, 1.62);
        assert!((t2 / t1 - (9.81f64 / 1.62).sqrt()).abs() < 1e-9); // Moon slows by √(g ratio)
    }

    #[test]
    fn lorentz_grows_hyperbolically() {
        assert!((lorentz(0.0) - 1.0).abs() < 1e-12);
        assert!((lorentz(0.5 * C_LIGHT) - 1.1547).abs() < 1e-3);
        assert!(lorentz(0.99 * C_LIGHT) > 7.0);
        assert!(lorentz(0.999 * C_LIGHT) > 22.0);
    }

    #[test]
    fn velocity_addition_respects_the_limit() {
        assert!((vel_add(0.9 * C_LIGHT, 0.9 * C_LIGHT) - (1.8 / 1.81) * C_LIGHT).abs() < 1.0);
        assert!(vel_add(0.99 * C_LIGHT, 0.99 * C_LIGHT) < C_LIGHT);
        // slow speeds: Newton right again
        assert!((vel_add(10.0, 20.0) - 30.0).abs() < 1e-12);
    }

    #[test]
    fn rest_energy_of_a_paperclip() {
        // 1 g → 9e13 J ≈ 25 GWh: the Hiroshima bomb released ~63 g of mass-energy
        let e = rest_energy(0.001);
        assert!((e - 9.0e13).abs() < 1e12);
    }

    #[test]
    fn time_sim_runs() {
        let w = crate::world::eval::compile("scene \"t\"\na = cube 3cm at (0, 50cm, 0) material: iron");
        let lines = time_sim(&w, 9.81);
        assert!(lines.iter().any(|l| l.text.contains("TIME & MOTION")));
        assert!(lines.iter().any(|l| l.text.contains("E = mc²")));
    }
}
