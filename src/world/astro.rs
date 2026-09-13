//! P2230 — OTD3 ASTRONOMY — gravity's big stage, beyond the drop table.
//!
//! The solar system as data, Kepler's three laws as code, and the
//! relativistic ceiling objects (escape velocity, Schwarzschild radius).
//! `simulate: orbit` reads this scene's parts as satellites: their height
//! above the floor becomes altitude above a planet, and the report shows
//! the orbital speed, period, and escape velocity their position implies.
//!
//! Laws carried here:
//!   Kepler I   — orbits are ellipses with the primary at one focus.
//!   Kepler II  — equal areas in equal times (fast at perihelion).
//!   Kepler III — T² = 4π²a³ / GM: period grows as a^1.5, no exceptions.
//!   Newton     — vis-viva v² = GM(2/r − 1/a) along the whole ellipse.
//!   Wien       — a star's colour is its temperature, λ_peak = b/T.

use super::eval::{ConsoleLine, LineKind, World};

/// Gravitational constant, m³/(kg·s²) (CODATA 2018).
pub const G: f64 = 6.674_30e-11;
/// Speed of light, m/s.
pub const C_LIGHT: f64 = 299_792_458.0;
/// Stefan–Boltzmann constant, W/(m²·K⁴).
pub const STEFAN_BOLTZMANN: f64 = 5.670_374_419e-8;
/// Wien's displacement constant, m·K.
pub const WIEN_B: f64 = 2.897_771_955e-3;
/// Hubble constant, km/s/Mpc (Planck 2018).
pub const H0: f64 = 67.4;
/// Parsec in metres.
pub const PARSEC: f64 = 3.085_677_581_491_367e16;
/// Astronomical unit, m.
pub const AU: f64 = 1.495_978_707e11;

/// A body worth knowing: mass kg, mean radius km, plus orbit where relevant.
pub struct Body {
    pub name: &'static str,
    pub mass_kg: f64,
    pub radius_km: f64,
    /// semi-major axis in AU (0 for satellites of the Sun's planets)
    pub a_au: f64,
    /// orbital period in days
    pub period_d: f64,
    pub ecc: f64,
}

pub const SUN: Body = Body { name: "Sun", mass_kg: 1.989e30, radius_km: 696_340.0, a_au: 0.0, period_d: 0.0, ecc: 0.0 };
pub const EARTH: Body = Body { name: "Earth", mass_kg: 5.972e24, radius_km: 6371.0, a_au: 1.0, period_d: 365.256, ecc: 0.0167 };
pub const MOON: Body = Body { name: "Moon", mass_kg: 7.346e22, radius_km: 1737.4, a_au: 0.0, period_d: 27.3217, ecc: 0.0549 };

/// The eight planets + Pluto, inner to outer.
pub const PLANETS: &[Body] = &[
    Body { name: "Mercury", mass_kg: 3.301e23, radius_km: 2439.7, a_au: 0.3871, period_d: 87.969, ecc: 0.2056 },
    Body { name: "Venus", mass_kg: 4.867e24, radius_km: 6051.8, a_au: 0.7233, period_d: 224.701, ecc: 0.0068 },
    EARTH,
    Body { name: "Mars", mass_kg: 6.417e23, radius_km: 3389.5, a_au: 1.5237, period_d: 686.980, ecc: 0.0934 },
    Body { name: "Jupiter", mass_kg: 1.898e27, radius_km: 69911.0, a_au: 5.2026, period_d: 4332.59, ecc: 0.0489 },
    Body { name: "Saturn", mass_kg: 5.683e26, radius_km: 58232.0, a_au: 9.5549, period_d: 10759.22, ecc: 0.0565 },
    Body { name: "Uranus", mass_kg: 8.681e25, radius_km: 25362.0, a_au: 19.2184, period_d: 30688.5, ecc: 0.0463 },
    Body { name: "Neptune", mass_kg: 1.024e26, radius_km: 24622.0, a_au: 30.1104, period_d: 60195.0, ecc: 0.0086 },
    Body { name: "Pluto", mass_kg: 1.303e22, radius_km: 1188.3, a_au: 39.4821, period_d: 90560.0, ecc: 0.2488 },
];

/// Kepler III: orbital period in seconds, a in metres, M the primary mass.
pub fn kepler_period(a_m: f64, m_kg: f64) -> f64 {
    2.0 * std::f64::consts::PI * (a_m * a_m * a_m / (G * m_kg)).sqrt()
}

/// Surface gravity g = GM/r², r in metres.
pub fn surface_g(m_kg: f64, r_m: f64) -> f64 {
    G * m_kg / (r_m * r_m)
}

/// Vis-viva: orbital speed at radius r on an orbit of semi-major axis a.
pub fn vis_viva(r_m: f64, a_m: f64, m_kg: f64) -> f64 {
    (G * m_kg * (2.0 / r_m - 1.0 / a_m)).sqrt()
}

/// Escape velocity at radius r: √(2GM/r).
pub fn escape_velocity(m_kg: f64, r_m: f64) -> f64 {
    (2.0 * G * m_kg / r_m).sqrt()
}

/// Schwarzschild radius: 2GM/c² — the point of no return.
pub fn schwarzschild_radius(m_kg: f64) -> f64 {
    2.0 * G * m_kg / (C_LIGHT * C_LIGHT)
}

/// Solve Kepler's equation M = E − e·sin E for the eccentric anomaly E,
/// Newton's method from E = M. |residual| < 1e-12 in a handful of steps.
pub fn solve_kepler(m_rad: f64, e: f64) -> f64 {
    let mut e_anom = m_rad;
    for _ in 0..50 {
        let f = e_anom - e * e_anom.sin() - m_rad;
        let fp = 1.0 - e * e_anom.cos();
        if fp.abs() < 1e-14 {
            break;
        }
        let step = f / fp;
        e_anom -= step;
        if step.abs() < 1e-12 {
            break;
        }
    }
    e_anom
}

/// True anomaly ν and radius r from eccentric anomaly E (a in metres).
pub fn orbit_state(a_m: f64, e: f64, e_anom: f64) -> (f64, f64) {
    let nu = 2.0 * (((1.0 + e) / 2.0).sqrt() * (e_anom / 2.0).sin())
        .atan2(((1.0 - e) / 2.0).sqrt() * (e_anom / 2.0).cos());
    let r = a_m * (1.0 - e * e_anom.cos());
    (nu, r)
}

/// Hohmann transfer between circular orbits r1 → r2 around mass M:
/// total Δv in m/s (two burns).
pub fn hohmann_dv(r1_m: f64, r2_m: f64, m_kg: f64) -> (f64, f64) {
    let r_transfer = r1_m + r2_m;
    let v1 = (G * m_kg / r1_m).sqrt();
    let v2 = (G * m_kg / r2_m).sqrt();
    let a_t = r_transfer / 2.0;
    let vp = vis_viva(r1_m, a_t, m_kg);
    let va = vis_viva(r2_m, a_t, m_kg);
    ((vp - v1).abs(), (v2 - va).abs())
}

/// Star luminosity from radius and surface temperature: L = 4πR²σT⁴.
pub fn luminosity(r_m: f64, t_k: f64) -> f64 {
    4.0 * std::f64::consts::PI * r_m * r_m * STEFAN_BOLTZMANN * t_k.powi(4)
}

/// Wien's law: peak wavelength in metres.
pub fn wien_peak(t_k: f64) -> f64 {
    WIEN_B / t_k
}

/// Distance modulus: apparent − absolute magnitude ↔ parsecs.
pub fn distance_modulus(pc: f64) -> f64 {
    5.0 * (pc / 10.0).log10()
}

/// Hubble's law: recession velocity km/s at distance Mpc.
pub fn hubble_velocity(mpc: f64) -> f64 {
    H0 * mpc
}

/// Tidal acceleration difference across a body of extent Δr at distance d.
pub fn tidal_accel(m_kg: f64, d_m: f64, dr_m: f64) -> f64 {
    2.0 * G * m_kg * dr_m / (d_m * d_m * d_m)
}

// ─────────────────────────────────────────────────────────────────────────────
// simulate: orbit — the scene as satellites
// ─────────────────────────────────────────────────────────────────────────────

pub fn orbit_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "ORBITAL MECHANICS — heights above the floor become altitudes above Earth".into(),
    });

    // the highest part is the satellite
    let mut best: Option<(&super::eval::Part, f64)> = None;
    for p in &world.parts {
        if p.hidden {
            continue;
        }
        if let Some(c) = p.centroid {
            let h_m = (c.y() / 1000.0).max(0.0);
            if best.map_or(true, |(_, bh)| h_m > bh) {
                best = Some((p, h_m));
            }
        }
    }
    let r_earth = EARTH.radius_km * 1000.0;
    match best {
        Some((p, h_m)) if h_m > 0.0 => {
            let r_orbit = r_earth + h_m;
            let v = vis_viva(r_orbit, r_orbit, EARTH.mass_kg); // circular
            let period_s = kepler_period(r_orbit, EARTH.mass_kg);
            let v_esc = escape_velocity(EARTH.mass_kg, r_orbit);
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!(
                    "  '{}' at {:.2} m up → orbital radius {:.1} km (R⊕ {:.0} km + {:.2} m):",
                    p.name, h_m, r_orbit / 1000.0, EARTH.radius_km, h_m
                ),
            });
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!(
                    "    circular speed {:.5} m/s ({:.2} km/h) — period {:.2} s — escape velocity {:.3} m/s",
                    v, v * 3.6, period_s, v_esc
                ),
            });
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!(
                    "    (LEO is 7.8 km/s for ~400 km up; the ISS laps Earth every 92 min — same two formulas, taller scene)"
                ),
            });
        }
        _ => {
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: "  no part sits above the floor — lift something with `at (…, 30cm, …)` and it becomes the satellite".into(),
            });
        }
    }

    // Kepler III across the solar system: theory vs measured periods
    let mut worst = 0.0f64;
    for b in PLANETS.iter().skip(1) {
        let a_m = b.a_au * AU;
        let theory_d = kepler_period(a_m, SUN.mass_kg) / 86400.0;
        worst = worst.max((theory_d - b.period_d).abs() / b.period_d);
    }
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  Kepler III check across {} planets: T² = 4π²a³/GM⊙ predicts every measured period to {:.2}% — Titius–Bode's clockwork",
            PLANETS.len() - 1,
            worst * 100.0
        ),
    });

    // the geometry of one real ellipse: Mercury
    let merc = &PLANETS[0];
    let a_m = merc.a_au * AU;
    let peri = a_m * (1.0 - merc.ecc);
    let aph = a_m * (1.0 + merc.ecc);
    let e_anom = solve_kepler(std::f64::consts::PI / 2.0, merc.ecc);
    let (nu, r) = orbit_state(a_m, merc.ecc, e_anom);
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  Mercury's ellipse: e = {:.3}, perihelion {:.2} AU, aphelion {:.2} AU — at E = 90°, ν = {:.1}°, r = {:.3} AU (Kepler II: it flies fastest down deep in the Sun's well)",
            merc.ecc, peri / AU, aph / AU, nu.to_degrees(), r / AU
        ),
    });

    // Einstein versus Newton, one number each
    let rs_earth = schwarzschild_radius(EARTH.mass_kg);
    let rs_sun = schwarzschild_radius(SUN.mass_kg);
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  the relativistic floor: Schwarzschild radius of Earth {:.2} mm, of the Sun {:.2} km — compress mass inside 2GM/c² and no trajectory comes back out",
            rs_earth * 1000.0, rs_sun / 1000.0
        ),
    });

    // stars: the Sun's ledger
    let r_sun = SUN.radius_km * 1000.0;
    let t_sun = 5772.0;
    let l_sun = luminosity(r_sun, t_sun);
    let peak = wien_peak(t_sun) * 1e9;
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  the Sun as a lightbulb: 4πR²σT⁴ = {:.3e} W at {} K — Wien says it peaks at {:.0} nm, which is why daytime is yellow-white",
            l_sun, t_sun as u32, peak
        ),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  the expanding frame: Hubble v = H₀·d — a galaxy {} Mpc out recedes at {:.0} km/s; distance modulus m−M = 5 puts {} pc at exactly 5 magnitudes",
            100.0,
            hubble_velocity(100.0),
            100.0
        ),
    });
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "orbits are just freefall that keeps missing — everything above is one inverse-square law wearing different hats".into(),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kepler_third_law_for_earth() {
        let t = kepler_period(AU, SUN.mass_kg) / 86400.0;
        assert!((t - 365.256).abs() < 0.5, "Earth period {} d", t);
    }

    #[test]
    fn all_planets_match_kepler_three() {
        for b in PLANETS.iter().skip(1) {
            let t = kepler_period(b.a_au * AU, SUN.mass_kg) / 86400.0;
            let rel = (t - b.period_d).abs() / b.period_d;
            assert!(rel < 0.02, "{}: {} d vs {} d", b.name, t, b.period_d);
        }
    }

    #[test]
    fn earth_orbital_speed_at_1_au() {
        let v = vis_viva(AU, AU, SUN.mass_kg);
        assert!((v - 29_780.0).abs() < 60.0, "v = {} m/s", v);
    }

    #[test]
    fn escape_velocities_match_textbook() {
        let r_e = EARTH.radius_km * 1000.0;
        assert!((escape_velocity(EARTH.mass_kg, r_e) - 11_186.0).abs() < 15.0);
        let r_m = MOON.radius_km * 1000.0;
        assert!((escape_velocity(MOON.mass_kg, r_m) - 2_380.0).abs() < 10.0);
    }

    #[test]
    fn surface_gravity_earth_and_moon() {
        let g_e = surface_g(EARTH.mass_kg, EARTH.radius_km * 1000.0);
        assert!((g_e - 9.82).abs() < 0.03, "g = {}", g_e);
        let g_m = surface_g(MOON.mass_kg, MOON.radius_km * 1000.0);
        assert!((g_m - 1.62).abs() < 0.02, "g = {}", g_m);
    }

    #[test]
    fn kepler_equation_solves_to_machine_precision() {
        for &(e, m) in &[(0.2056, 1.2), (0.7, 0.3), (0.9, 2.8), (0.01, 5.0)] {
            let ea = solve_kepler(m, e);
            let resid = (ea - e * ea.sin() - m).abs();
            assert!(resid < 1e-11, "e={} M={} residual {}", e, m, resid);
        }
    }

    #[test]
    fn orbit_state_perihelion_and_aphelion() {
        // at E = 0, r = a(1−e); at E = π, r = a(1+e)
        let (a, e) = (1.0e11, 0.5);
        let (_, r0) = orbit_state(a, e, 0.0);
        let (_, r1) = orbit_state(a, e, std::f64::consts::PI);
        assert!((r0 - a * (1.0 - e)).abs() < 1e-3);
        assert!((r1 - a * (1.0 + e)).abs() < 1e-3);
    }

    #[test]
    fn schwarzschild_radii() {
        assert!((schwarzschild_radius(EARTH.mass_kg) - 0.00887).abs() < 2e-4);
        assert!((schwarzschild_radius(SUN.mass_kg) - 2953.0).abs() < 2.0);
    }

    #[test]
    fn sun_luminosity_and_wien() {
        let l = luminosity(SUN.radius_km * 1000.0, 5772.0);
        assert!((l - 3.828e26).abs() / 3.828e26 < 0.01, "L = {} W", l);
        let peak = wien_peak(5772.0) * 1e9;
        assert!((peak - 502.0).abs() < 4.0, "λ_peak = {} nm", peak);
        // a 10,000 K star peaks in the ultraviolet-blue
        assert!(wien_peak(10_000.0) * 1e9 < 300.0);
    }

    #[test]
    fn hohmann_leo_to_geo_total_dv() {
        let r1 = EARTH.radius_km * 1000.0 + 400e3;
        let r2 = EARTH.radius_km * 1000.0 + 35_786e3;
        let (dv1, dv2) = hohmann_dv(r1, r2, EARTH.mass_kg);
        // textbook: ≈2.43 km/s + ≈1.47 km/s
        assert!((dv1 - 2430.0).abs() < 40.0, "dv1 = {}", dv1);
        assert!((dv2 - 1470.0).abs() < 40.0, "dv2 = {}", dv2);
    }

    #[test]
    fn distance_modulus_basics() {
        assert!((distance_modulus(10.0) - 0.0).abs() < 1e-12);
        assert!((distance_modulus(100.0) - 5.0).abs() < 1e-9);
        assert!((distance_modulus(1_000_000.0) - 25.0).abs() < 1e-6);
    }

    #[test]
    fn tide_pulls_two_ways() {
        // the Moon's tidal acceleration across 1 m at 384,400 km
        let d = 384_400e3;
        let a = tidal_accel(MOON.mass_kg, d, 1.0);
        assert!(a > 1.5e-13 && a < 2.0e-13, "a = {} m/s²", a);
        // and it scales as 1/d³
        let a2 = tidal_accel(MOON.mass_kg, d / 2.0, 1.0);
        assert!((a2 / a - 8.0).abs() < 1e-9);
    }

    #[test]
    fn hubble_linear_law() {
        assert!((hubble_velocity(50.0) - H0 * 50.0).abs() < 1e-9);
    }
}
