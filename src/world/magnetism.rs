//! P2120 — OTD3 MAGNETISM — fields, forces, and Faraday's engine.
//!
//! Every electron is a tiny spinning magnet; align enough of them and the
//! material remembers — that is a ferromagnet. Iron, steel, nickel stick to
//! the fridge; copper doesn't (but it DOES resist changing fields, which is
//! how induction brakes work).
//!
//! Laws carried here:
//!   Coulomb-for-magnets — like poles repel, unlike attract, 1/r² falloff
//!   Lorentz force       — F = q·v × B (how CRTs steered electrons)
//!   Motor force         — F = B·I·L (a wire in a field gets pushed)
//!   Faraday             — EMF = −N·dΦ/dt (moving flux makes voltage)
//!   Ampère              — a current IS a magnet: B = μ₀·I/(2πr)
//!   Earth               — our planet is a 25–65 μT magnet that points
//!                          every compass the same way

use super::eval::{ConsoleLine, LineKind, World};
use crate::math3::V3;

/// Vacuum permeability, T·m/A.
pub const MU_0: f64 = 4.0 * std::f64::consts::PI * 1e-7;
/// Earth's magnetic field strength at the surface (T).
pub const B_EARTH: f64 = 5.0e-5;
/// Curie temperatures, °C — above this, a ferromagnet forgets.
pub fn curie_point(mat: &str) -> Option<f64> {
    match mat {
        "iron" => Some(770.0),
        "steel" => Some(770.0),      // carbon steel ~770, stainless ~0 (austenitic is non-magnetic)
        "stainless" => None,          // austenitic stainless: not ferromagnetic at room temp
        "nickel" => Some(354.0),
        "cobalt" => Some(1115.0),
        "gadolinium" => Some(20.0),  // room-temperature magic
        _ => None,
    }
}

/// The magnetic status of a material at room temperature.
pub enum MagKind {
    /// sticks to magnets hard (iron, steel)
    Ferromagnetic,
    /// weakly attracted (aluminum, platinum, oxygen gas)
    Paramagnetic,
    /// weakly repelled (copper, silver, gold, water, bismuth)
    Diamagnetic,
    /// doesn't care (wood, plastic, glass…)
    NonMagnetic,
}

impl MagKind {
    pub fn name(&self) -> &'static str {
        match self {
            MagKind::Ferromagnetic => "ferromagnetic",
            MagKind::Paramagnetic => "paramagnetic",
            MagKind::Diamagnetic => "diamagnetic",
            MagKind::NonMagnetic => "non-magnetic",
        }
    }
}

pub fn mag_kind(mat: &str) -> MagKind {
    match mat {
        "iron" | "steel" | "uranium" | "plutonium" | "thorium" => MagKind::Ferromagnetic,
        "stainless" => MagKind::Paramagnetic, // austenitic grade
        "aluminum" | "titanium" | "lithium" | "platinum" |
        "oxygen" | "air" => MagKind::Paramagnetic,
        "copper" | "silver" | "gold" | "zinc" | "lead" | "brass" | "bronze" |
        "carbon" | "water" | "ice" | "ethanol" | "acetone" | "glass" | "plastic" |
        "rubber" | "wood" | "mercury" | "bismuth" => MagKind::Diamagnetic,
        _ => MagKind::NonMagnetic,
    }
}

/// Field of a bar magnet at distance d along its axis (dipole-ish, T),
/// from a magnet with moment M (A·m²): B = μ₀·2M/(4π·d³) on-axis.
/// We use the on-axis dipole law — honest for d >> magnet size.
pub fn bar_magnet_field(moment_a_m2: f64, dist_m: f64) -> f64 {
    if dist_m <= 1e-6 { return f64::INFINITY; }
    MU_0 * 2.0 * moment_a_m2 / (4.0 * std::f64::consts::PI * dist_m * dist_m * dist_m)
}

/// Force between two dipoles aligned on one axis (N), attraction negative.
/// F ≈ 3μ₀·m₁·m₂/(2π·d⁴) — the reason fridge magnets snap hard up close
/// and let go gracefully at distance.
pub fn dipole_force(m1: f64, m2: f64, dist_m: f64) -> f64 {
    if dist_m <= 1e-6 { return f64::INFINITY; }
    3.0 * MU_0 * m1 * m2 / (2.0 * std::f64::consts::PI * dist_m.powi(4))
}

/// Motor force on a current-carrying wire: F = B·I·L·sin θ (N).
pub fn motor_force(b_t: f64, current_a: f64, length_m: f64, sin_theta: f64) -> f64 {
    b_t * current_a * length_m * sin_theta
}

/// Faraday's law of induction: EMF = N·ΔΦ/Δt (V). The minus sign is
/// Lenz's law — the induced current fights the change that made it.
pub fn faraday_emf(turns: f64, d_flux_weber: f64, dt_s: f64) -> f64 {
    if dt_s <= 0.0 { return f64::INFINITY; }
    turns * d_flux_weber / dt_s
}

/// Field around a straight wire: B = μ₀·I/(2π·r) (T).
pub fn wire_field(current_a: f64, dist_m: f64) -> f64 {
    if dist_m <= 1e-9 { return f64::INFINITY; }
    MU_0 * current_a / (2.0 * std::f64::consts::PI * dist_m)
}

/// Solenoid core field: B = μ₀·n·I (T), n = turns per metre.
pub fn solenoid_field(turns_per_m: f64, current_a: f64) -> f64 {
    MU_0 * turns_per_m * current_a
}

/// Magnetic moment of a bar-magnet part we can estimate from size:
/// a good neodymium-grade material carries ~10⁵ A/m magnetisation; a mild
/// steel "keeper" ~10³–10⁴. We use the material's class, not fantasy.
pub fn estimate_moment(part: &super::eval::Part) -> f64 {
    let mat = part.material.map(|m| m.name).unwrap_or("plastic");
    let vol_m3 = part.volume_mm3 / 1e9;
    let magnetisation = match mag_kind(mat) {
        MagKind::Ferromagnetic => 4.0e5,   // hard ferrite / mild steel range
        MagKind::Paramagnetic => 8.0,      // aluminum's χ ~ 2.2e-5 × B/μ₀
        MagKind::Diamagnetic => -1.0,      // water-class: famously levitates in 16 T
        MagKind::NonMagnetic => 0.0,
    };
    // only ferromagnets hold a permanent moment at rest:
    if matches!(mag_kind(mat), MagKind::Ferromagnetic) {
        magnetisation * vol_m3
    } else {
        0.0
    }
}

/// `simulate: magnet` — the magnetic survey of the scene.
pub fn magnet_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "MAGNETIC SURVEY — every current is a magnet, every magnet a current (Ampère had it both ways)".into(),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!("  background: Earth's own field is {:.1} μT — it is what a compass answers to", B_EARTH * 1e6),
    });

    let visible: Vec<&super::eval::Part> = world.parts.iter().filter(|p| !p.hidden).collect();
    if visible.is_empty() {
        out.push(ConsoleLine { kind: LineKind::Warn, text: "nothing to survey — the scene is empty".into() });
        return out;
    }

    // classify every part
    let mut magnets = Vec::new(); // (index, name, moment, centre)
    for (i, part) in visible.iter().enumerate() {
        let mat = part.material.map(|m| m.name).unwrap_or("plastic");
        let kind = mag_kind(mat);
        let moment = estimate_moment(part);
        let centre = part.centroid.unwrap_or(part.mesh.bbox().center());
        let note = match kind {
            MagKind::Ferromagnetic => {
                if let Some(curie) = curie_point(mat) {
                    format!(" — ferromagnetic, moment ≈ {:.2} A·m² (Curie point {} °C: heat it past that and it forgets)", moment, curie)
                } else {
                    format!(" — ferromagnetic, moment ≈ {:.2} A·m²", moment)
                }
            }
            MagKind::Paramagnetic => " — paramagnetic: pulled weakly INTO a field (χ > 0)".into(),
            MagKind::Diamagnetic => " — diamagnetic: pushed weakly OUT of fields; water levitates in 16 T".into(),
            MagKind::NonMagnetic => " — the field passes through untouched".into(),
        };
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("  {} [{}]{}", part.name, mat, note),
        });
        if moment.abs() > 1e-9 {
            magnets.push((i, part.name.clone(), moment, centre));
        }
    }

    // pair forces between permanent magnets
    if magnets.len() >= 2 {
        out.push(ConsoleLine { kind: LineKind::Sim, text: "---- FORCES BETWEEN MAGNETS (1/r⁴ dipole law) ----".into() });
        for a in 0..magnets.len() {
            for b in a + 1..magnets.len() {
                let (_, name_a, m_a, c_a) = &magnets[a];
                let (_, name_b, m_b, c_b) = &magnets[b];
                let d_mm = c_a.sub(c_b).len();
                let d_m = (d_mm / 1000.0).max(0.005); // clamp to 5 mm
                let f = dipole_force(*m_a, *m_b, d_m);
                let verb = if f.signum() > 0.0 { "REPEL" } else { "ATTRACT" };
                out.push(ConsoleLine {
                    kind: LineKind::Answer,
                    text: format!("  {} ↔ {} at {:.1} cm: {} with {:.3} N — the 1/r⁴ law means halving the distance multiplies the force ×16",
                        name_a, name_b, d_mm / 10.0, verb, f.abs()),
                });
            }
        }
    } else if magnets.len() == 1 {
        // single magnet: report its field at distances
        let (idx, name, m, _) = &magnets[0];
        let _ = idx;
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!("  {} alone: field {:.2} mT at 5 cm, {:.3} mT at 10 cm, Earth-strength ({:.1} μT) at {:.0} cm",
                name,
                bar_magnet_field(*m, 0.05) * 1e3,
                bar_magnet_field(*m, 0.10) * 1e3,
                B_EARTH * 1e6,
                // solve distance where field = B_EARTH
                (MU_0 * 2.0 * m / (4.0 * std::f64::consts::PI * B_EARTH)).powf(1.0 / 3.0) * 100.0),
        });
    }

    // Faraday teaching moment: if any part moves at all (screws, spins),
    // a nearby coil would see an EMF. Report the EMF of the strongest magnet
    // fully flipping (2Φ swing) in 1 second through a 100-turn coil at 5 cm.
    if let Some((_, name, m, _)) = magnets.first() {
        let flux = MU_0 * 2.0 * m / (4.0 * std::f64::consts::PI * 0.05_f64.powi(2)) * 0.001; // ~area 10 cm²
        let emf = faraday_emf(100.0, 2.0 * flux, 1.0);
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("  Faraday: yank {} fully away in 1 s past a 100-turn coil at 5 cm ⇒ EMF ≈ {:.2} V — this is the whole of every generator, turbine and dynamo", name, emf),
        });
    }
    // motor law teaser
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!("  Motor law: 1 A through a 10 cm wire in Earth's mere {:.1} μT feels {:.4} μN — build real motors with real fields (B·I·L)", B_EARTH * 1e6, motor_force(B_EARTH, 1.0, 0.1, 1.0) * 1e6),
    });
    out
}

/// Where a compass points in a scene with magnets (the net field direction
/// at the scene centre). Returns a V3 direction and strength in μT.
pub fn net_field_at(world: &World, point: V3) -> (V3, f64) {
    // Earth contributes: pointing north (+Z here) and slightly down
    let mut field = V3::new(0.0, -0.1, 1.0).norm().mul(B_EARTH);
    for part in &world.parts {
        if part.hidden {
            continue;
        }
        let m = estimate_moment(part);
        if m.abs() < 1e-9 {
            continue;
        }
        let c = part.centroid.unwrap_or(part.mesh.bbox().center());
        let d = point.sub(&c);
        let dist = (d.len() / 1000.0).max(0.01);
        // on-axis approximation along the dipole axis (use the part's longest axis)
        let bb = part.mesh.bbox();
        let size = bb.size();
        let axis = if size.x() >= size.y() && size.x() >= size.z() { V3::new(1.0, 0.0, 0.0) }
            else if size.y() >= size.z() { V3::new(0.0, 1.0, 0.0) }
            else { V3::new(0.0, 0.0, 1.0) };
        // field points from S to N along the axis; magnitude by dipole law
        let b = bar_magnet_field(m, dist);
        let dir = if d.dot(&axis) >= 0.0 { axis.mul(1.0) } else { axis.mul(-1.0) };
        field = field.add(&dir.mul(b));
    }
    let strength = field.len();
    (field.norm(), strength)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn iron_is_ferromagnetic_copper_is_not() {
        assert!(matches!(mag_kind("iron"), MagKind::Ferromagnetic));
        assert!(matches!(mag_kind("steel"), MagKind::Ferromagnetic));
        assert!(matches!(mag_kind("copper"), MagKind::Diamagnetic));
        assert!(matches!(mag_kind("aluminum"), MagKind::Paramagnetic));
        assert!(matches!(mag_kind("wood"), MagKind::Diamagnetic));
    }

    #[test]
    fn inverse_quartic_law() {
        let f1 = dipole_force(1.0, 1.0, 1.0);
        let f2 = dipole_force(1.0, 1.0, 2.0);
        assert!((f1 / f2 - 16.0).abs() < 1e-9, "halving distance ×16 force");
    }

    #[test]
    fn wire_field_at_one_cm() {
        // 1 A at 1 cm: B = μ₀/(2π·0.01) ≈ 20 μT — Earth strength!
        let b = wire_field(1.0, 0.01);
        assert!((b * 1e6 - 19.99).abs() < 0.5);
    }

    #[test]
    fn solenoid_sanity() {
        // 1000 turns/m at 1 A ≈ 1.26 mT
        let b = solenoid_field(1000.0, 1.0);
        assert!((b * 1e3 - 1.2566).abs() < 0.01);
    }

    #[test]
    fn faraday_voltage_from_flux_swing() {
        let emf = faraday_emf(100.0, 0.01, 0.1); // 100 turns, 10 mWb swing, 100 ms
        assert!((emf - 10.0).abs() < 1e-9);
    }

    #[test]
    fn magnet_sim_runs() {
        let w = crate::world::eval::compile("scene \"t\"\na = cube 2cm at (0, 1cm, 0) material: iron\nb = cube 2cm at (5cm, 1cm, 0) material: copper");
        let lines = magnet_sim(&w);
        assert!(lines.iter().any(|l| l.text.contains("MAGNETIC SURVEY")));
        assert!(lines.iter().any(|l| l.text.contains("ferromagnetic")));
        assert!(lines.iter().any(|l| l.text.contains("diamagnetic")));
    }

    #[test]
    fn compass_sees_earth_when_quiet() {
        let w = crate::world::eval::compile("scene \"t\"\na = cube 2cm at (0, 1cm, 0) material: wood");
        let (_, strength) = net_field_at(&w, V3::ZERO);
        assert!((strength - B_EARTH).abs() < 1e-9);
    }
}
