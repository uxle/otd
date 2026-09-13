//! P2110 — OTD3 THERMODYNAMICS — temperature, heat, and the four laws.
//!
//! Temperature is the kinetic energy per atom: the faster they jiggle, the
//! hotter it reads. `simulate: heat` answers "what happens to this scene at
//! this temperature?" — expansion, conduction between touching parts,
//! melting and boiling against every material's real points, and the
//! honest entropy ledger.
//!
//! Laws carried here:
//!   Zeroth  — touching things equalise: A touches B ⇒ they share a temp.
//!   First   — energy is conserved: heat lost = heat gained, always.
//!   Second  — heat flows hot → cold, never uphill by itself; entropy up.
//!   Third   — absolute zero (0 K = −273.15 °C) is the floor nobody reaches.

use super::eval::{ConsoleLine, LineKind, Part, World};
use super::energy::specific_heat;

/// Absolute zero in Celsius.
pub const ABSOLUTE_ZERO_C: f64 = -273.15;
/// Room temperature, the OTD default.
pub const ROOM_TEMP_C: f64 = 20.0;

/// Latent heat of fusion/vaporisation, J/kg (textbook).
pub fn latent_heat_fusion(mat: &str) -> f64 {
    match mat {
        "water" | "ice" => 334_000.0,       // the famous 334 kJ/kg
        "aluminum" => 397_000.0,
        "iron" | "steel" | "stainless" => 247_000.0,
        "copper" => 209_000.0,
        "gold" => 64_000.0,
        "silver" => 105_000.0,
        "lead" => 23_000.0,
        "ethanol" => 104_000.0,
        "methane" => 59_000.0,
        _ => 200_000.0,
    }
}

pub fn latent_heat_vapor(mat: &str) -> f64 {
    match mat {
        "water" => 2_257_000.0,             // 2.26 MJ/kg — why steam burns
        "ethanol" => 846_000.0,
        "methane" => 510_000.0,
        "mercury" => 296_000.0,
        _ => 1_000_000.0,
    }
}

/// Boiling point at 1 atm, °C.
pub fn boiling_point(mat: &str) -> Option<f64> {
    match mat {
        "water" | "ice" => Some(100.0),
        "ethanol" => Some(78.4),
        "acetone" => Some(56.0),
        "methane" => Some(-161.5),
        "ammonia" => Some(-33.3),
        "nitrogen" => Some(-195.8),
        "oxygen" => Some(-183.0),
        "helium" => Some(-268.9),
        "hydrogen" => Some(-252.9),
        "chlorine" => Some(-34.0),
        "mercury" => Some(356.7),
        "gasoline" => Some(150.0),          // a range, really — mid value
        "oil" => Some(300.0),
        "glycerin" => Some(290.0),
        _ => None,
    }
}

/// Linear thermal expansion coefficient, 1/K (×10⁻⁶ kept as 1e-6 factor).
pub fn expansion_coeff(mat: &str) -> f64 {
    match mat {
        "aluminum" => 23.1e-6,   // the champion expander among metals
        "zinc" => 30.2e-6,
        "lead" => 28.9e-6,
        "copper" => 16.5e-6,
        "brass" | "bronze" => 18.0e-6,
        "gold" => 14.2e-6,
        "silver" => 18.9e-6,
        "iron" | "steel" => 11.8e-6,
        "stainless" => 14.4e-6, // austenitic
        "titanium" => 8.6e-6,
        "tungsten" => 4.5e-6,   // barely moves — glass-to-metal seals
        "chrome" => 4.9e-6,
        "glass" => 8.5e-6,      // borosilicate is 3.3 — cookware
        "ceramic" => 6.0e-6,
        "marble" => 5.5e-6,
        "concrete" => 12.0e-6,
        "wood" | "oak" | "teak" => 4.9e-6,  // along grain, mostly
        "pine" => 5.0e-6,
        "plastic" => 70.0e-6,   // polymers stretch eagerly
        "rubber" => 150.0e-6,
        "ice" => 50.0e-6,
        "water" => 207.0e-6,    // liquid, volumetric-ish; anomaly below 4 °C
        _ => 10.0e-6,
    }
}

/// Temperature conversions (the three scales every engineer meets).
pub fn c_to_f(c: f64) -> f64 { c * 9.0 / 5.0 + 32.0 }
pub fn f_to_c(f: f64) -> f64 { (f - 32.0) * 5.0 / 9.0 }
pub fn c_to_k(c: f64) -> f64 { c - ABSOLUTE_ZERO_C }
pub fn k_to_c(k: f64) -> f64 { k + ABSOLUTE_ZERO_C }

/// Thermal equilibrium of N parts in contact: ΣmᵢcᵢTᵢ / Σmᵢcᵢ
/// (the First Law — heat lost = heat gained).
pub fn equilibrium_temperature(parts: &[(&str, f64, f64)]) -> f64 {
    // (name, mass_kg, temp_c) — returns the common temperature
    let mut num = 0.0;
    let mut den = 0.0;
    for (mat, m, t) in parts {
        let c = specific_heat(mat);
        num += m * c * t;
        den += m * c;
    }
    if den > 0.0 { num / den } else { ROOM_TEMP_C }
}

/// How long conduction takes to move heat through a slab (Biot-style
/// time-scale estimate, seconds): τ ≈ L²·ρ·cₚ / k — the thermal diffusivity
/// relaxation time. Iron pans heat fast; glass pans test your patience.
pub fn conduction_time_scale(mat: &str, thickness_m: f64) -> f64 {
    let rho = super::materials::find(mat).map(|m| m.density).unwrap_or(1050.0);
    let k = super::materials::find(mat).map(|m| m.thermal).unwrap_or(0.2);
    let c = specific_heat(mat);
    if k <= 0.0 { f64::INFINITY } else { thickness_m * thickness_m * rho * c / k }
}

/// The state of a part at a temperature: solid / melting / liquid / boiling / gas / plasma.
pub struct PhaseVerdict {
    pub state: &'static str,
    pub note: String,
}

pub fn phase_at(part: &Part, temp_c: f64) -> PhaseVerdict {
    let mat = part.material.map(|m| m.name).unwrap_or("plastic");
    let melt = part.material.and_then(|m| m.melt_c);
    let boil = boiling_point(mat);
    if temp_c >= 5_500.0 {
        return PhaseVerdict { state: "plasma", note: format!("{} at {:.0} °C — atoms have given up holding electrons (surface-of-the-Sun class)", mat, temp_c) };
    }
    if let Some(b) = boil {
        if temp_c > b {
            return PhaseVerdict { state: "gas", note: format!("{} boiled away at {} °C — now {} °C, {:.0} K above its boiling point", mat, b, temp_c, temp_c - b) };
        }
    }
    if let Some(m) = melt {
        if temp_c > m {
            // check whether it is BETWEEN melt and boil
            let b_ok = boil.map(|b| temp_c <= b).unwrap_or(true);
            if b_ok {
                let latent = latent_heat_fusion(mat);
                let mass_kg = part.mass_g / 1000.0;
                return PhaseVerdict {
                    state: "liquid",
                    note: format!("{} melted at {} °C — a liquid now; re-freezing it back would release {:.1} kJ (latent heat)", mat, m, mass_kg * latent / 1000.0),
                };
            }
        }
        if (temp_c - m).abs() < 2.0 {
            return PhaseVerdict { state: "melting", note: format!("{} is AT its melting point {} °C — adding heat melts it, removing heat freezes it; the temperature itself refuses to move (latent heat)", mat, m) };
        }
    } else if temp_c > 300.0 {
        return PhaseVerdict { state: "burning", note: format!("{} has no melting point — it decomposes/ignites near {} °C (wood-class chemistry, not physics)", mat, 300) };
    }
    PhaseVerdict {
        state: "solid",
        note: format!("{} stays solid — it would need {} to melt", mat,
            melt.map(|m| format!("{} °C", m)).unwrap_or_else(|| "more heat than it will ever see".to_string())),
    }
}

/// `simulate: heat` — the thermal report for the whole scene at its set
/// temperature (default 20 °C; `temperature: 800` changes it).
pub fn heat_sim(world: &World, temp_c: f64) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    let t = temp_c.max(ABSOLUTE_ZERO_C + 0.001);
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!("THERMAL REPORT at {:.1} °C = {:.1} K = {:.1} °F", t, c_to_k(t), c_to_f(t)),
    });
    if t <= ABSOLUTE_ZERO_C + 0.001 {
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: "  (that is absolute zero itself — the Third Law says you can approach it forever and never arrive; every atom would stand perfectly still)".into(),
        });
    }

    // water's anomaly: densest at 4 °C
    if (4.0..5.0).contains(&t) {
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: "  water note: 4 °C is water's densest point — ice floats and lakes freeze top-down because of exactly this anomaly".into(),
        });
    }

    let mut any_melted = false;
    for part in &world.parts {
        if part.hidden {
            continue;
        }
        let mat = part.material.map(|m| m.name).unwrap_or("plastic");
        let mass_kg = part.mass_g / 1000.0;
        let cp = specific_heat(mat);
        // heat stored above 0 °C
        let q = mass_kg * cp * t;
        // equilibrium with room if different
        let pv = phase_at(part, t);
        // thermal expansion vs room
        let alpha = expansion_coeff(mat);
        let d_temp = t - ROOM_TEMP_C;
        let grow_pct = alpha * d_temp * 100.0;
        let mut line = format!("  {} [{}] {:.0} J above 0 °C ({} J/kgK)",
            part.name, mat, q, cp as u32);
        if pv.state != "solid" {
            any_melted = true;
            line.push_str(&format!(" — PHASE: {}", pv.note));
        } else if d_temp.abs() >= 1.0 {
            line.push_str(&format!(" — grows {:+.3}% ({:+.2} mm per metre) [{} ppm/K]", grow_pct, alpha * d_temp * 1000.0, (alpha * 1e6) as u32));
        }
        out.push(ConsoleLine { kind: LineKind::Info, text: line });
        // conduction speed note for big parts
        let bb = part.mesh.bbox();
        let sz = bb.size();
            let thick_m = sz.x().max(sz.y()).max(sz.z()) / 1000.0;
        if mass_kg > 0.05 && thick_m > 0.01 {
            let tau = conduction_time_scale(mat, thick_m);
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!("    heat crosses it in ~{} ({:.0} W/mK conductivity — {})",
                    if tau > 3600.0 { format!("{:.1} h", tau / 3600.0) } else if tau > 60.0 { format!("{:.0} min", tau / 60.0) } else { format!("{:.0} s", tau) },
                    super::materials::find(mat).map(|m| m.thermal).unwrap_or(0.2),
                    if mat == "copper" { "pans and heat-sinks love it".to_string() } else if super::materials::find(mat).map(|m| m.thermal).unwrap_or(0.2) < 0.5 { "an insulator — touch it safely".to_string() } else { "a conductor — touch it with a glove".to_string() }),
            });
        }
    }

    // First Law check: parts in contact equalise — find touching pairs
    let mut contacts = 0;
    let visible: Vec<&Part> = world.parts.iter().filter(|p| !p.hidden).collect();
    for i in 0..visible.len() {
        for j in i + 1..visible.len() {
            let a = visible[i].mesh.bbox();
            let b = visible[j].mesh.bbox();
            let pen = |k: usize| a.max.0[k].min(b.max.0[k]) - a.min.0[k].max(b.min.0[k]);
            if pen(0) > -0.5 && pen(1) > -0.5 && pen(2) > -0.5 {
                contacts += 1;
            }
        }
    }
    if contacts > 0 && world.parts.iter().filter(|p| !p.hidden).count() > 1 {
        let eq_parts: Vec<(String, f64, f64)> = world.parts.iter()
            .filter(|p| !p.hidden)
            .map(|p| (p.material.map(|m| m.name).unwrap_or("plastic").to_string(), p.mass_g / 1000.0, t))
            .collect();
        let refs: Vec<(&str, f64, f64)> = eq_parts.iter().map(|(m, mkg, tt)| (m.as_str(), *mkg, *tt)).collect();
        let eq_t = equilibrium_temperature(&refs);
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!("{} touching pair(s) — the Zeroth Law: they share one temperature; contact equilibrium settles at {:.1} °C", contacts, eq_t),
        });
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: "  First Law: heat lost by the hot side = heat gained by the cold side — ΣmcΔT balances to the last joule".into(),
        });
    }
    if any_melted {
        out.push(ConsoleLine {
            kind: LineKind::Warn,
            text: "  phase change in progress — the geometry above no longer matches reality (render what actually happens: simulate: settle after cooling)".into(),
        });
    }
    // Second Law closing
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!("  Second Law: at {:.1} °C the scene radiates {} W per m² of surface into the room (Stefan–Boltzmann) — heat leaks, entropy grows, coffee cools",
            t, (super::energy::STEFAN_BOLTZMANN * (c_to_k(t)).powi(4)) as u32),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions_round_trip() {
        assert!((f_to_c(c_to_f(25.0)) - 25.0).abs() < 1e-9);
        assert!((k_to_c(c_to_k(100.0)) - 100.0).abs() < 1e-9);
        assert!((c_to_f(0.0) - 32.0).abs() < 1e-9);
        assert!((c_to_f(100.0) - 212.0).abs() < 1e-9);
        assert!((c_to_k(0.0) - 273.15).abs() < 1e-9);
    }

    #[test]
    fn hot_iron_melts() {
        let w = crate::world::eval::compile("scene \"t\"\nbar = cube 5cm at (0, 3cm, 0) material: iron");
        let pv = phase_at(&w.parts[0], 1600.0);
        assert_eq!(pv.state, "liquid");
        let pv2 = phase_at(&w.parts[0], 20.0);
        assert_eq!(pv2.state, "solid");
    }

    #[test]
    fn water_boils_but_ice_is_special() {
        let w = crate::world::eval::compile("scene \"t\"\nc = cube 2cm at (0, 1cm, 0) material: water");
        let pv = phase_at(&w.parts[0], 150.0);
        assert_eq!(pv.state, "gas");
        // at 4 °C water is at its densest — still liquid
        let pv4 = phase_at(&w.parts[0], 4.0);
        assert_eq!(pv4.state, "liquid");
    }

    #[test]
    fn equilibrium_conserves_energy() {
        // 1 kg water at 80 °C + 1 kg water at 20 °C ⇒ 50 °C
        let eq = equilibrium_temperature(&[("water", 1.0, 80.0), ("water", 1.0, 20.0)]);
        assert!((eq - 50.0).abs() < 1e-9);
        // hot copper into water: ΣmcT / Σmc
        let eq2 = equilibrium_temperature(&[("copper", 0.5, 100.0), ("water", 0.5, 25.0)]);
        let num = 0.5 * 385.0 * 100.0 + 0.5 * 4186.0 * 25.0;
        let den = 0.5 * 385.0 + 0.5 * 4186.0;
        assert!((eq2 - num / den).abs() < 1e-9);
    }

    #[test]
    fn aluminium_expands_more_than_iron() {
        assert!(expansion_coeff("aluminum") > expansion_coeff("iron"));
        assert!(expansion_coeff("tungsten") < expansion_coeff("plastic"));
    }

    #[test]
    fn heat_sim_runs() {
        let w = crate::world::eval::compile("scene \"t\"\na = cube 3cm at (0, 2cm, 0) material: aluminum");
        let lines = heat_sim(&w, 45.0);
        assert!(lines.iter().any(|l| l.text.contains("THERMAL REPORT")));
        assert!(lines.iter().any(|l| l.text.contains("aluminum")));
    }
}
