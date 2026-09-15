//! P2330 — OTD4 STELLAR DYNAMICS — many-body gravity, the virial theorem, the Jeans scale.
//!
//! Where astro.rs studies Kepler (two bodies), stellar dynamics studies many
//! bodies — globular clusters, galaxies, star-forming clouds. The math is
//! harder than a single orbit: N bodies pull each other all at once. But the
//! big-picture laws still hold, and they are beautiful.
//!
//! Laws carried here:
//!   N-body             — F_i = Σ_j G·m_i·m_j·(r_j − r_i)/|r|³
//!   Virial theorem     — 2K + U = 0 (a bound system's kinetic / potential ratio)
//!   Jeans length       — λ_J = c_s·√(π/(Gρ)) (the size a cloud must exceed to collapse)
//!   Jeans mass         — M_J = (π^(5/2)/6)·c_s³/(G^(3/2)·ρ^(1/2))
//!   Escape velocity    — v_e = √(2GM/R) (the speed that leaves forever)
//!   Dynamical time     — t_dyn = √(R³/(GM)) (a free-fall crossing time)
//!   Tully–Fisher       — L ∝ v_max⁴ (brighter spirals spin faster)
//!   Fundamental plane  — mass = constant × σ² × R (galaxies live on a plane)

use super::eval::{ConsoleLine, LineKind, World};

/// Newton's gravitational constant, m³/(kg·s²).
pub const G_NEWTON: f64 = 6.6743e-11;
/// Solar mass, kg.
pub const M_SUN: f64 = 1.989e30;
/// Solar luminosity, W.
pub const L_SUN: f64 = 3.828e26;
/// Parsec in metres.
pub const PARSEC_M: f64 = 3.0857e16;
/// Speed of light (m/s).
pub const C_LIGHT: f64 = 299_792_458.0;

/// Force between two masses (N). F = G·m₁·m₂/r²
pub fn two_body_force(m1: f64, m2: f64, r: f64) -> f64 {
    if r <= 0.0 {
        return f64::INFINITY;
    }
    G_NEWTON * m1 * m2 / (r * r)
}

/// Orbital velocity (m/s): v = √(G·M/r) for a circular orbit at radius r.
pub fn orbital_velocity(m_central_kg: f64, r_m: f64) -> f64 {
    if r_m <= 0.0 {
        return 0.0;
    }
    (G_NEWTON * m_central_kg / r_m).sqrt()
}

/// Escape velocity (m/s): v_e = √(2GM/R)
pub fn escape_velocity(m_kg: f64, r_m: f64) -> f64 {
    if r_m <= 0.0 {
        return 0.0;
    }
    (2.0 * G_NEWTON * m_kg / r_m).sqrt()
}

/// Dynamical (free-fall) time: t_dyn = √(R³ / (G·M)) — for the Sun around
/// the Milky Way, ~2.3 × 10⁸ years.
pub fn dynamical_time(m_kg: f64, r_m: f64) -> f64 {
    if m_kg <= 0.0 {
        return f64::INFINITY;
    }
    (r_m * r_m * r_m / (G_NEWTON * m_kg)).sqrt()
}

/// Virial theorem: 2K + U = 0 ⇒ for a self-gravitating system, K = -U/2.
/// Returns the total energy E = K + U = -K = U/2 (bound, negative).
pub fn virial_energy(kinetic_j: f64, potential_j: f64) -> f64 {
    2.0 * kinetic_j + potential_j
}

/// Virial mass estimate: M = 5·σ²·R / G  (for a cluster with line-of-sight
/// velocity dispersion σ and half-mass radius R).
pub fn virial_mass(sigma_v: f64, r_m: f64) -> f64 {
    if G_NEWTON <= 0.0 {
        return f64::INFINITY;
    }
    5.0 * sigma_v * sigma_v * r_m / G_NEWTON
}

/// Sound speed in an ideal gas (m/s): c_s = √(γ·k_B·T / (μ·m_H)).
/// γ=5/3 mono, 7/5 diatomic. μ = mean molecular weight (1.27 for solar).
pub fn sound_speed(t_k: f64, gamma: f64, mu: f64) -> f64 {
    const K_B: f64 = 1.380649e-23;
    const M_H: f64 = 1.6735575e-27;
    (gamma * K_B * t_k / (mu * M_H)).sqrt()
}

/// Jeans length (m): λ_J = c_s · √(π / (G·ρ))
/// A cloud bigger than this collapses under its own gravity; smaller, it puffs apart.
pub fn jeans_length(c_s: f64, rho: f64) -> f64 {
    if rho <= 0.0 || G_NEWTON <= 0.0 {
        return f64::INFINITY;
    }
    c_s * (std::f64::consts::PI / (G_NEWTON * rho)).sqrt()
}

/// Jeans mass (kg): the mass inside a sphere of radius λ_J/2.
pub fn jeans_mass(c_s: f64, rho: f64) -> f64 {
    let lj = jeans_length(c_s, rho);
    if !lj.is_finite() {
        return f64::INFINITY;
    }
    let r = lj / 2.0;
    4.0 / 3.0 * std::f64::consts::PI * r * r * r * rho
}

/// Acceleration at the surface of a uniform sphere (m/s²).
pub fn surface_g(m_kg: f64, r_m: f64) -> f64 {
    if r_m <= 0.0 {
        return 0.0;
    }
    G_NEWTON * m_kg / (r_m * r_m)
}

/// `simulate: stellar` — the many-body gravity survey of the scene.
pub fn stellar_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "STELLAR DYNAMICS SURVEY — many-body gravity, the virial theorem, the Jeans scale (where stars are born)".into(),
    });

    let visible: Vec<usize> = world
        .parts
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.hidden && !p.mesh.is_empty())
        .map(|(i, _)| i)
        .collect();
    if visible.is_empty() {
        out.push(ConsoleLine {
            kind: LineKind::Warn,
            text: "  nothing to survey — make something first".into(),
        });
        return out;
    }

    // Read each part as a "star" with mass = its real mass; positions in metres.
    let stars: Vec<(String, f64, crate::math3::V3)> = visible
        .iter()
        .map(|&i| {
            let p = &world.parts[i];
            let m_kg = p.mass_g / 1000.0;
            let c = p.centroid.unwrap_or(p.mesh.bbox().center());
            // mm → m
            (p.name.clone(), m_kg, crate::math3::V3::new(c.x() / 1000.0, c.y() / 1000.0, c.z() / 1000.0))
        })
        .collect();

    // 1) total mass + scale
    let total_kg: f64 = stars.iter().map(|(_, m, _)| *m).sum();
    let bb = world.stats.bbox;
    let r_max = ((bb.max.0[0] - bb.min.0[0]).max(bb.max.0[1] - bb.min.0[1]).max(bb.max.0[2] - bb.min.0[2]))
        * 1e-3;
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  scene mass: {:.3e} kg ({:.2e} M_sun) — spread over {:.2} m. The Sun is {} × heavier.",
            total_kg,
            total_kg / M_SUN,
            r_max,
            (M_SUN / total_kg.max(1e-9)) as i64
        ),
    });

    // 2) N-body pairwise forces: report the strongest one
    let mut strongest = ("".into(), "".into(), 0.0f64);
    let mut total_pe = 0.0;
    for i in 0..stars.len() {
        for j in i + 1..stars.len() {
            let (n1, m1, p1) = &stars[i];
            let (n2, m2, p2) = &stars[j];
            let d = p1.sub(p2).len();
            if d <= 1e-9 {
                continue;
            }
            let f = two_body_force(*m1, *m2, d);
            if f > strongest.2 {
                strongest = (n1.clone(), n2.clone(), f);
            }
            total_pe += -G_NEWTON * m1 * m2 / d;
        }
    }
    if strongest.2 > 0.0 {
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!(
                "  strongest pair: {} ↔ {} ⇒ F = G·m₁·m₂/r² = {:.3e} N — gravity is the weakest force, but it has infinite range and never cancels",
                strongest.0, strongest.1, strongest.2
            ),
        });
    }

    // 3) virial theorem: 2K + U = 0
    // approximate K from the velocity needed to stay in orbit at half the cluster's radius
    let r_half = (r_max / 2.0).max(1e-3);
    let v_orb = orbital_velocity(total_kg, r_half);
    let k_half = 0.5 * total_kg * v_orb * v_orb;
    let virial = virial_energy(k_half, total_pe);
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!(
            "  virial: 2K + U = {:.3e} J (K={:.3e}, U={:.3e}) — {}",
            virial,
            k_half,
            total_pe,
            if virial.abs() < k_half.abs() * 0.5 {
                "near equilibrium (a stable cluster)"
            } else if virial < 0.0 {
                "bound — the scene would hold together under its own gravity"
            } else {
                "unbound — the parts would fly apart in a real gravitational sense"
            }
        ),
    });

    // 4) Jeans length: how big a cloud of this density needs to be to collapse
    let rho_avg = if r_max > 0.0 {
        total_kg / (4.0 / 3.0 * std::f64::consts::PI * (r_max / 2.0).powi(3))
    } else {
        0.0
    };
    if rho_avg > 0.0 {
        let t = world.temp_c + 273.15;
        let c_s = sound_speed(t.max(2.7), 5.0 / 3.0, 1.27);
        let lj = jeans_length(c_s, rho_avg);
        let mj = jeans_mass(c_s, rho_avg);
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!(
                "  Jeans: c_s = {:.1} m/s at {:.0} K, ρ = {:.2e} kg/m³ ⇒ λ_J = {:.2e} m, M_J = {:.2e} kg — a cloud bigger than this collapses to form stars",
                c_s, t, rho_avg, lj, mj
            ),
        });
    }

    // 5) dynamical time: how long the system takes to cross itself
    let t_dyn = dynamical_time(total_kg, r_half);
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  dynamical time t_dyn = √(R³/GM) = {:.2e} s = {:.1e} yr — the natural clock of the system (galaxies: ~10⁸ yr; star clusters: ~10⁷ yr)",
            t_dyn, t_dyn / (365.25 * 86400.0)
        ),
    });

    // 6) escape velocity from the centre of mass
    let v_esc = escape_velocity(total_kg, r_half);
    out.push(ConsoleLine {
        kind: LineKind::Answer,
        text: format!(
            "  escape velocity v_e = √(2GM/R) = {:.1} m/s — {}",
            v_esc,
            if v_esc < 11_200.0 {
                format!("less than Earth's 11.2 km/s — this scene barely holds itself")
            } else if v_esc < C_LIGHT {
                format!("less than c — light escapes freely")
            } else {
                "greater than c — would be a black hole at this radius!".to_string()
            }
        ),
    });

    // closing: the larger context
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "---- THE LARGER PICTURE ----".into(),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  the Milky Way: M ≈ 10¹² M_sun, R ≈ 50 kpc, v_circ ≈ 220 km/s, t_dyn ≈ 2×10⁸ yr — we orbit it once per 240 Myr. The Andromeda galaxy approaches at 110 km/s; in 4.5 Gyr the two merge."
        ),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn earth_orbital_speed_29_8_kmps() {
        // Earth around the Sun: 1 AU = 1.496e11 m, M_sun = 1.989e30 kg
        let v = orbital_velocity(M_SUN, 1.496e11);
        assert!((v - 29_780.0).abs() / 29_780.0 < 0.01, "v = {}", v);
    }

    #[test]
    fn earth_escape_velocity_11_2_kmps() {
        let v = escape_velocity(5.972e24, 6.371e6);
        assert!((v - 11_180.0).abs() / 11_180.0 < 0.01, "v_e = {}", v);
    }

    #[test]
    fn virial_mass_of_a_globular_cluster() {
        // M3-like: σ = 5 km/s, R = 10 pc
        let r = 10.0 * PARSEC_M;
        let m = virial_mass(5_000.0, r);
        assert!(m > 1e35 && m < 1e38, "M3-like mass: {} kg", m);
    }

    #[test]
    fn jeans_length_for_molecular_cloud() {
        // T = 10 K, n = 100 cm^-3, μ = 2.33 (molecular)
        // ⇒ c_s ≈ 220 m/s, ρ ≈ 3.9e-19 kg/m³
        // ⇒ λ_J ≈ 7.7e16 m ≈ 2.5 pc — the textbook molecular cloud Jeans length
        let c_s = sound_speed(10.0, 7.0 / 5.0, 2.33);
        let rho = 100.0e6 * 2.33 * 1.6735575e-27;
        let lj = jeans_length(c_s, rho);
        // expected ~ 10^16–10^17 m (a few parsecs)
        assert!(lj > 1e16 && lj < 1e17, "molecular cloud λ_J = {} m", lj);
    }

    #[test]
    fn dynamical_time_galactic_scale() {
        // The Milky Way's dynamical time at the Sun's orbit (R = 8 kpc, M_enc ~ 1e11 M_sun)
        // is ~10^8 yr — the "galactic year" the Sun orbits once.
        let r = 8_000.0 * 3.0857e16; // 8 kpc in m
        let m = 1.0e11 * M_SUN;
        let t = dynamical_time(m, r);
        let yr = t / (365.25 * 86400.0);
        // expected ~ 10^8 yr (the standard "galactic dynamical time")
        assert!(yr > 1e7 && yr < 1e9, "Milky Way t_dyn at Sun's orbit = {} yr", yr);
    }

    #[test]
    fn stellar_sim_runs() {
        let w = crate::world::eval::compile("scene \"t\"\na = sphere 4cm at (0, 4cm, 0) material: lead\nb = sphere 4cm at (10cm, 4cm, 0) material: lead");
        let lines = stellar_sim(&w);
        assert!(lines.iter().any(|l| l.text.contains("STELLAR DYNAMICS SURVEY")));
        assert!(lines.iter().any(|l| l.text.contains("virial")));
    }
}
