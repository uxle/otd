//! P0540 — physics simulations: drop, float, collapse. Declarative science:
//! users never configure physics — materials carry the values, laws do the rest.

use super::eval::{ConsoleLine, LineKind, World};
use crate::units::{G_EARTH, WATER_DENSITY};

/// Material toughness classes for drop verdicts (impact speed thresholds, m/s).
fn shatter_speed(mat: &str) -> Option<f64> {
    match mat {
        "glass" => Some(4.0),
        "ceramic" => Some(4.0),
        "ice" => Some(3.0),
        "concrete" => Some(6.0),
        "marble" => Some(4.0),
        _ => None,
    }
}
fn dent_speed(mat: &str) -> f64 {
    match mat {
        "lead" => 4.0,
        "aluminum" => 8.0,
        "brass" => 8.0,
        "bronze" => 8.0,
        "copper" => 10.0,
        "steel" => 15.0,
        "stainless" => 16.0,
        "chrome" => 16.0,
        "titanium" => 20.0,
        "tungsten" => 22.0,
        "iron" => 12.0,
        "zinc" => 7.0,
        "gold" => 6.0,
        "silver" => 7.0,
        "wood" | "oak" | "pine" | "teak" => 25.0,
        "plastic" => 30.0,
        "rubber" => 60.0,
        "fabric" => 50.0,
        "foam" => 50.0,
        "carbon" => 40.0,
        _ => 20.0,
    }
}

pub fn simulate(kind: &str, world: &World, line: usize, g: f64) -> Vec<ConsoleLine> {
    match kind {
        "drop" => drop_sim(world, g),
        "float" => float_sim(world),
        "collapse" => collapse_sim(world),
        "splash" => splash_sim(world, g),
        // ---- OTD3 science expansion ----
        "energy" => super::energy::energy_sim(world, g, world.temp_c),
        "heat" | "thermal" | "thermo" => super::thermo::heat_sim(world, world.temp_c),
        "magnet" | "magnetism" => super::magnetism::magnet_sim(world),
        "sound" | "acoustics" => super::waves::sound_sim(world, world.temp_c),
        "light" | "optics" => super::waves::light_sim(world, world.temp_c),
        "time" | "motion" => super::time::time_sim(world, g),
        // ---- OTD3.1 self-make expansion (P2200 series) ----
        // P2200/P2210 — the neural nets: nn.rs trains, OTD-Burn explains
        "learn" | "nn" | "brain" | "neural" => crate::nn::nn_sim(world, g),
        // P2220 — the scene reads itself as a dataset
        "stats" | "statistics" => super::stats::stats_sim(world),
        // P2230 — orbital mechanics: Kepler, vis-viva, the relativistic floor
        "orbit" | "astro" | "space" => super::astro::orbit_sim(world),
        // ---- OTD3.3: the subatomic layer (P2250 series) ----
        // P2260 — the atomic census: protons, neutrons, electrons, shells
        "atom" | "atoms" | "nucleus" | "nuclear" => super::atom::atom_sim(world),
        // P2260 — half-lives and live activity from the actual atom count
        "decay" | "radioactive" | "radioactivity" | "halflife" => super::atom::decay_sim(world),
        // P2250 — the Standard Model briefing: quarks, gluons, photons, neutrinos
        "particles" | "particle" | "standardmodel" | "quark" | "quarks" => super::particles::particles_sim(world),
        // ---- OTD4 dynamics expansion (P2300 series) ----
        // P2300 — aerodynamics: drag, lift, terminal velocity, Reynolds, Mach
        "aero" | "aerodynamics" | "drag" | "lift" | "flight" => super::aerodynamics::aero_sim(world),
        // P2310 — fluid dynamics: continuity, Bernoulli, Poiseuille, Stokes
        "fluid" | "fluiddynamics" | "fluid_dynamics" | "bernoulli" | "poiseuille" => super::fluiddynamics::fluid_sim(world),
        // P2320 — electrodynamics: Ohm, Kirchhoff, RC/RL/LC, Maxwell
        "electro" | "electrodynamics" | "ohm" | "current" => super::electrodynamics::electro_sim(world),
        // P2330 — stellar dynamics: N-body, virial, Jeans
        "stellar" | "stellardynamics" | "stellar_dynamics" | "nbody" | "virial" => super::stellardynamics::stellar_sim(world),
        // P2340 — rigid body dynamics: inertia, angular momentum, gyroscopes
        "rigid" | "rigidbody" | "rigid_body" | "inertia" | "gyroscope" | "spin" => super::rigidbody::rigid_sim(world),
        // ---- OTD6: motor + circuit ----
        "motor" | "electric_motor" | "electricmotor" => super::motor::motor_sim(world),
        "circuit" => super::motor::circuit_sim(world),
        other => vec![ConsoleLine {
            kind: LineKind::Error,
            text: format!("line {}: '{}' is not a simulation I know — try drop, float, collapse, splash, settle, solidity, gas, mix, energy, heat, magnet, sound, light, time, learn, stats, orbit, atom, decay, particles, aero, fluid, electro, stellar, rigid, motor, or circuit", line, other),
        }],
    }
}

/// P1420b/P1430 — simulations that MOVE or MERGE geometry need `&mut World`:
/// settle (gravity + real solidity), solidity (static audit), gas (mixing
/// chamber), mix (liquid beaker). Read-only sims delegate to `simulate`.
pub fn simulate_mut(kind: &str, world: &mut World, line: usize, g: f64) -> Vec<ConsoleLine> {
    match kind {
        "settle" => super::settle::settle_world(world, g),
        "solidity" => super::settle::solidity_check(world),
        "gas" => super::chem::simulate_gas(world),
        "mix" => super::chem::simulate_mix(world),
        other => simulate(other, world, line, g),
    }
}

/// v = √(2gh) — impact speed from 1 m, plus energy and material verdict.
fn drop_sim(world: &World, g: f64) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    if world.parts.iter().all(|p| p.hidden) || world.stats.total_mass_g <= 0.0 {
        out.push(ConsoleLine { kind: LineKind::Sim, text: "nothing to drop — make something first!".into() });
        return out;
    }
    let h = 1.0; // m
    let v = (2.0 * g * h).sqrt();
    let mass_kg = world.stats.total_mass_g / 1000.0;
    let ke = 0.5 * mass_kg * v * v;
    let g_name = if (g - G_EARTH).abs() < 0.01 { "earth".into() } else if g > 0.0 { format!("{:.2} m/s²", g) } else { "zero gravity".into() };
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!(
            "drop from {} m: v = √(2·{:.2}·1) = {:.2} m/s — kinetic energy {:.1} J ({} kg, {})",
            h, g, v, ke, mass_kg, g_name
        ),
    });
    // material verdicts
    for p in world.parts.iter().filter(|p| !p.hidden) {
        let mat = p.material.map(|m| m.name).unwrap_or("plastic");
        if let Some(limit) = shatter_speed(mat) {
            if v > limit {
                out.push(ConsoleLine {
                    kind: LineKind::Warn,
                    text: format!("{} shatters on impact — {} breaks above {:.0} m/s", p.name, mat, limit),
                });
            } else {
                let margin = (limit / v).max(0.1);
                out.push(ConsoleLine {
                    kind: LineKind::Sim,
                    text: format!("{} survives the drop with {:.1}× margin ({} breaks above {:.0} m/s)", p.name, margin, mat, limit),
                });
            }
        } else if v > dent_speed(mat) {
            out.push(ConsoleLine { kind: LineKind::Warn, text: format!("{} dents on impact — {} yields above {:.0} m/s", p.name, mat, dent_speed(mat)) });
        } else {
            let bounce = p.material.map(|m| m.bounce).unwrap_or(0.3);
            let back = 1.0 * bounce * bounce;
            out.push(ConsoleLine {
                kind: LineKind::Sim,
                text: format!("{} bounces back to {:.0} cm ({} restitution {:.2})", p.name, back * 100.0, mat, bounce),
            });
        }
    }
    out
}

/// Archimedes: floats ⇔ ρ_avg < ρ_water. Reports submerged fraction.
fn float_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    let mass_kg = world.stats.total_mass_g / 1000.0;
    if mass_kg <= 0.0 {
        out.push(ConsoleLine { kind: LineKind::Sim, text: "nothing to float — make something first!".into() });
        return out;
    }
    let rho = world.stats.avg_density_kg_m3;
    if rho <= 0.0 {
        out.push(ConsoleLine { kind: LineKind::Sim, text: "nothing to float — make something first!".into() });
        return out;
    }
    if rho < WATER_DENSITY {
        let submerged = rho / WATER_DENSITY * 100.0;
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!(
                "floats — average density {:.0} kg/m³ < water 1000 kg/m³, so {:.0}% of it sits under the waterline (Archimedes: the displaced water weighs exactly the object)",
                rho, submerged
            ),
        });
    } else {
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!(
                "sinks — average density {:.0} kg/m³ > water 1000 kg/m³ (it needs to weigh less than the water it pushes away)",
                rho
            ),
        });
    }
    // per-material breakdown
    for p in world.parts.iter().filter(|p| !p.hidden) {
        let d = p.material.map(|m| m.density).unwrap_or(1050.0);
        let verdict = if d < WATER_DENSITY { "floats" } else { "sinks" };
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!("  {}: {} ({:.0} kg/m³, {:.0} g)", p.name, verdict, d, p.mass_g),
        });
    }
    out
}

/// Static load check: 75 kg on top of the object, stress vs compressive strength
/// at the narrowest slice. σ = F/A.
fn collapse_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    let load_kg = 75.0;
    let mut weakest: Option<(&super::eval::Part, f64, f64)> = None; // (part, slice area, strength)
    for p in world.parts.iter().filter(|p| !p.hidden) {
        let bb = p.mesh.bbox();
        let mid = (bb.min.y() + bb.max.y()) * 0.5;
        let area_mm2 = crate::geo::measure::slice_area_at(&p.mesh, mid);
        if area_mm2 < 1.0 {
            continue;
        }
        let strength = p.material.map(|m| m.compressive_mpa).unwrap_or(45.0);
        if weakest.is_none() || area_mm2 < weakest.unwrap().1 {
            weakest = Some((p, area_mm2, strength));
        }
    }
    let (p, area_mm2, strength) = match weakest {
        Some(x) => x,
        None => {
            out.push(ConsoleLine { kind: LineKind::Sim, text: "nothing stands to check — make something with walls or legs first".into() });
            return out;
        }
    };
    // total force: load + self weight
    let force_n = (load_kg + world.stats.total_mass_g / 1000.0) * G_EARTH;
    let stress_mpa = force_n / area_mm2; // N/mm² = MPa
    let factor = strength / stress_mpa;
    let verdict = if factor >= 1.0 {
        format!("HOLDS — {:.2}× safety factor", factor)
    } else {
        format!("COLLAPSES — stress exceeds strength by {:.1}×", 1.0 / factor)
    };
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!(
            "load test: {} kg on top of {} → F = {:.0} N over {:.1} cm² at its waist → σ = {:.2} MPa vs {} strength {:.0} MPa → {}",
            load_kg, p.name, force_n, area_mm2 / 100.0, stress_mpa,
            p.material.map(|m| m.name).unwrap_or("plastic"), strength, verdict
        ),
    });
    if world.stats.overlaps {
        out.push(ConsoleLine { kind: LineKind::Warn, text: "objects overlap — fuse them with add for an honest load test".into() });
    }
    out
}

/// P1410 — `simulate: splash`: drop the scene's parts into a water tank and
/// show the full fluid response: impact speed, buoyancy verdict, the bobbing
/// oscillation (damped mass-on-spring), and the waves that make the water
/// "shake". Depth of the tank: 25 cm (a standard lab tank).
///
/// Physics used, all real:
///   impact speed      v = √(2gh)
///   buoyancy          F_b = ρ_water · g · V_sub   (Archimedes)
///   bobbing ω         ω = √(ρ_water · g · A / m)  (hydrostatic spring)
///   first overshoot   d = v / ω                   (energy→spring compression)
///   damping           envelope e^(−ζωt), ζ ≈ 0.18 for water
///   wave speed        c = √(g · H)   (shallow-water waves, H = 0.25 m)
fn splash_sim(world: &World, g: f64) -> Vec<ConsoleLine> {
    const WATER: f64 = WATER_DENSITY;
    const TANK_DEPTH: f64 = 0.25; // m
    let mut out = Vec::new();
    let mut any = false;
    for p in world.parts.iter().filter(|p| !p.hidden) {
        // liquids ARE the medium — a water pool doesn't splash into itself
        let mat_name = p.material.map(|m| m.name).unwrap_or("plastic");
        if mat_name == "water" || mat_name == "oil" || mat_name == "mercury" {
            continue;
        }
        any = true;
        let m_kg = p.mass_g / 1000.0;
        if m_kg <= 0.0 {
            continue;
        }
        let h = 1.0; // drop height, m
        let v = (2.0 * g * h).sqrt();
        let rho = p.material.map(|m| m.density).unwrap_or(1050.0);
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!(
                "splash: {} ({} {:.0} kg/m³, {:.0} g) hits the water at v = √(2·{:.2}·1) = {:.2} m/s",
                p.name, mat_name, rho, p.mass_g, g, v
            ),
        });
        let bb = p.mesh.bbox();
        let mid = (bb.min.y() + bb.max.y()) * 0.5;
        let a_m2 = (crate::geo::measure::slice_area_at(&p.mesh, mid) / 1.0e6).max(1e-4); // mm² → m²
        if rho < WATER {
            // floats — hydrostatic spring: ω = √(ρ_w g A / m)
            let omega = (WATER * g * a_m2 / m_kg).sqrt();
            let period = 2.0 * std::f64::consts::PI / omega;
            let overshoot = (v / omega * 100.0).min(99.0); // cm
            let bounces = (1.0 / (2.0 * 0.18 * omega)).ln() / 1.0f64.max(0.0).min(f64::MAX);
            let settle = (1.0 / (0.18 * omega)).max(0.2);
            let _ = bounces;
            out.push(ConsoleLine {
                kind: LineKind::Sim,
                text: format!(
                    "  {} floats — bobbing ω = √(ρw·g·A/m) = {:.2} rad/s, period {:.2} s; the first plunge pushes it {:.1} cm under, then it sways e^(−0.18ωt) and settles in ~{:.1} s",
                    p.name, omega, period, overshoot, settle
                ),
            });
        } else {
            // sinks — gravity beats buoyancy; drag caps the fall speed
            let net = (1.0 - WATER / rho) * g;
            out.push(ConsoleLine {
                kind: LineKind::Sim,
                text: format!(
                    "  {} sinks — density {:.0} > water 1000 kg/m³, so it keeps falling at {:.1} m/s² net (buoyancy cancels the rest) and the tank floor catches it",
                    p.name, rho, net
                ),
            });
        }
        // the water itself reacts — this is the "shaking" the user sees
        let c = (g * TANK_DEPTH).sqrt(); // shallow-water wave speed
        let splash_h = (v * 0.045 * (m_kg / (WATER * a_m2 * 0.05)).min(2.2)).clamp(0.004, 0.30);
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!(
                "  the water shakes — a {:.1} cm splash dome, then ripples race out at c = √(g·H) = {:.2} m/s (H = {} cm tank), wobbling the surface ±{:.1} mm as they pass",
                splash_h * 100.0, c, (TANK_DEPTH * 100.0) as i64, splash_h * 1000.0 * 0.35
            ),
        });
    }
    if !any {
        out.push(ConsoleLine { kind: LineKind::Sim, text: "nothing to splash — make something first!".into() });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::eval::compile;

    #[test]
    fn splash_reports_impact_and_waves() {
        let w = compile("scene \"t\"\ncube 5cm material: oak at (0, 30cm, 0)\nsimulate: splash");
        let lines = simulate("splash", &w, 1, G_EARTH);
        let text: String = lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join("\n");
        assert!(text.contains("splash"), "should mention splash: {}", text);
        assert!(text.contains("floats"), "oak must float: {}", text);
        assert!(text.contains("shakes"), "water must shake: {}", text);
    }

    #[test]
    fn splash_steel_sinks() {
        let w = compile("scene \"t\"\nsphere 4cm material: steel at (0, 30cm, 0)\nsimulate: splash");
        let lines = simulate("splash", &w, 1, G_EARTH);
        let text: String = lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join("\n");
        assert!(text.contains("sinks"), "steel must sink: {}", text);
    }

    #[test]
    fn splash_empty_world_is_polite() {
        let w = compile("scene \"empty\"");
        let lines = simulate("splash", &w, 1, G_EARTH);
        assert!(lines[0].text.contains("nothing to splash"));
    }
}
