//! P1430 — real chemistry: which liquids mix, which refuse, which REACT,
//! and how gases mix by diffusion. Every rule below is real chemistry with
//! the reason spelled out — polar solvents mix (like dissolves like),
//! non-polar oils refuse water, liquid metal refuses everything, and two
//! flammable gases + oxygen do not "mix", they REACT.
//!
//! Balanced equations were independently verified with the vendored
//! reasoning-AI engine (vendor/reasoning-ai — verification-gated chemistry):
//!   2 H2 + O2 -> 2 H2O        (VERIFIED)
//!   CH4 + 2 O2 -> CO2 + 2 H2O (VERIFIED)
//!   H2 + Cl2 -> 2 HCl         (VERIFIED)

use super::eval::{ConsoleLine, LineKind, Part, World};
use super::materials::{self, State};
use crate::math3::M4;

// ---------------------------------------------------------------- miscibility

/// Verdict when two LIQUIDS meet at room temperature.
#[derive(Debug, PartialEq)]
pub enum LiquidMix {
    /// one phase — volume-weighted density, blended color
    Miscible { reason: &'static str },
    /// two phases stay separate, lighter floats on heavier
    Immiscible { reason: &'static str },
}

/// Verdict when two GASES meet: diffusion mixes everything except pairs
/// that react on contact.
#[derive(Debug, PartialEq)]
pub enum GasMix {
    /// uniform mixture (that's what diffusion does — no effort needed)
    Mixes,
    /// reaction — balanced equation + product name + molar masses (g/mol)
    Reacts { equation: &'static str, product: &'static str, notes: &'static str },
}

/// Miscibility table (like dissolves like): polar+poler mixes, non-polar
/// refuses polar, mercury refuses everything.
pub fn liquid_mix(a: &str, b: &str) -> LiquidMix {
    let (x, y) = if a <= b { (a, b) } else { (b, a) };
    match (x, y) {
        // polar + polar: hydrogen bonding swaps partners freely
        ("ethanol", "water") => LiquidMix::Miscible { reason: "both polar — ethanol's –OH hydrogen-bonds into water, one phase at any ratio" },
        ("acetone", "water") => LiquidMix::Miscible { reason: "acetone's polar C=O accepts hydrogen bonds from water — one phase" },
        ("glycerin", "water") => LiquidMix::Miscible { reason: "glycerin's three –OH groups hydrogen-bond into water — one phase" },
        ("acetone", "ethanol") => LiquidMix::Miscible { reason: "two polar organic solvents — one phase" },
        ("ethanol", "glycerin") => LiquidMix::Miscible { reason: "–OH meets –OH — both alcohols, one phase" },
        // non-polar refuses polar (and vice versa)
        ("oil", "water") => LiquidMix::Immiscible { reason: "oil is non-polar, water is polar — water would have to give up its hydrogen-bond network, so oil floats on top" },
        ("oil", "ethanol") => LiquidMix::Immiscible { reason: "ethanol is only PARTLY soluble in oils — mostly two layers" },
        ("oil", "glycerin") => LiquidMix::Immiscible { reason: "non-polar oil against a dense poly-ol — they refuse each other" },
        // mercury refuses everything (it's a liquid METAL — metallic bonds)
        ("mercury", "water") => LiquidMix::Immiscible { reason: "mercury is a liquid metal — metallic bonds will not mix with polar water (this is why it beads)" },
        ("mercury", x) if x != "mercury" => LiquidMix::Immiscible { reason: "mercury mixes with essentially nothing at room temperature — it beads up" },
        _ => LiquidMix::Immiscible { reason: "no known mixing at room temperature — kept as separate phases" },
    }
}

/// Gas chemistry: every pair diffuses into one uniform mixture EXCEPT the
/// reactive pairs below (verified equations — see module docs).
pub fn gas_mix(a: &str, b: &str) -> GasMix {
    let (x, y) = if a <= b { (a, b) } else { (b, a) };
    match (x, y) {
        ("hydrogen", "oxygen") => GasMix::Reacts {
            equation: "2 H2 + O2 -> 2 H2O",
            product: "steam",
            notes: "needs a spark — then it goes with a bang (286 kJ/mol of H2)",
        },
        ("methane", "oxygen") => GasMix::Reacts {
            equation: "CH4 + 2 O2 -> CO2 + 2 H2O",
            product: "carbon dioxide + steam",
            notes: "needs ignition — 890 kJ/mol of methane, the cleanest fossil flame",
        },
        ("chlorine", "hydrogen") => GasMix::Reacts {
            equation: "H2 + Cl2 -> 2 HCl",
            product: "hydrogen chloride",
            notes: "explodes in sunlight — no spark needed",
        },
        // everything else just mixes: N2 + O2 -> air, He + anything, CO2 + N2 …
        _ => GasMix::Mixes,
    }
}

/// Molar masses (g/mol) for stoichiometry of the reactive gas pairs.
fn molar_mass(name: &str) -> Option<f64> {
    match name {
        "hydrogen" => Some(2.016),
        "oxygen" => Some(32.00),
        "methane" => Some(16.04),
        "chlorine" => Some(70.90),
        _ => None,
    }
}

/// A balanced reaction: how many grams of B consume 1 g of A, and what
/// product mass comes out (mass conserved).
fn stoichiometry(a: &str, b: &str) -> Option<(f64, &'static str)> {
    // (reactant A grams per 1 g of B? — we return g of B per g of A, product name)
    match (a, b) {
        ("hydrogen", "oxygen") | ("oxygen", "hydrogen") => Some((7.936, "steam")), // 32/2.016·(2/1)… = 7.94 g O2 per g H2
        ("methane", "oxygen") | ("oxygen", "methane") => Some((3.99, "flue gas")), // 64/16.04
        ("chlorine", "hydrogen") | ("hydrogen", "chlorine") => Some((35.17, "hydrogen chloride")), // 70.9/2.016
        _ => None,
    }
}

// ---------------------------------------------------------------- mixing sims

fn blend_color(a: [u8; 3], b: [u8; 3], wa: f64, wb: f64) -> [u8; 3] {
    let t = wa / (wa + wb).max(1e-9);
    let mix = |x: u8, y: u8| -> u8 { (x as f64 * t + y as f64 * (1.0 - t)).round().clamp(0.0, 255.0) as u8 };
    [mix(a[0], b[0]), mix(a[1], b[1]), mix(a[2], b[2])]
}

fn translate_part(p: &mut Part, dx: f64, dy: f64, dz: f64) {
    if dx == 0.0 && dy == 0.0 && dz == 0.0 {
        return;
    }
    p.mesh.transform(&M4::translate(dx, dy, dz));
    p.centroid = p.mesh.centroid();
}

/// `mix: A + B` — the chemistry verdict for a material pair (any states).
pub fn mix_pair(a: &str, b: &str, line: usize) -> Vec<ConsoleLine> {
    let (ma, mb) = (materials::find(a), materials::find(b));
    let mut out = vec![ConsoleLine {
        kind: LineKind::Error,
        text: format!("line {}: unknown material in mix (try: {})", line, materials::NAMES.join(", ")),
    }];
    let (ma, mb) = match (ma, mb) {
        (Some(a), Some(b)) => (a, b),
        _ => return out,
    };
    out = Vec::new();
    let (sa, sb) = (materials::state(ma), materials::state(mb));
    if sa == State::Gas || sb == State::Gas {
        match gas_mix(ma.name, mb.name) {
            GasMix::Mixes => out.push(ConsoleLine {
                kind: LineKind::Answer,
                text: format!(
                    "{} + {} MIX into one uniform gas — diffusion does it for free (gases have no surface tension to fight), a heavier-one-sinks-then-it-all-stirs story",
                    ma.name, mb.name
                ),
            }),
            GasMix::Reacts { equation, product, notes } => out.push(ConsoleLine {
                kind: LineKind::Warn,
                text: format!("{} + {} REACT: {} — {} ({})", ma.name, mb.name, equation, product, notes),
            }),
        }
    } else if sa == State::Liquid && sb == State::Liquid {
        match liquid_mix(ma.name, mb.name) {
            LiquidMix::Miscible { reason } => {
                let rho = (ma.density + mb.density) / 2.0;
                out.push(ConsoleLine {
                    kind: LineKind::Answer,
                    text: format!("{} + {} MIX into ONE phase — {} → a solution of ρ ≈ {:.0} kg/m³", ma.name, mb.name, reason, rho),
                });
            }
            LiquidMix::Immiscible { reason } => {
                let (top, bot) = if ma.density <= mb.density { (ma, mb) } else { (mb, ma) };
                out.push(ConsoleLine {
                    kind: LineKind::Answer,
                    text: format!("{} + {} do NOT mix — {}. {} ({:.0} kg/m³) floats on {} ({:.0} kg/m³)", ma.name, mb.name, reason, top.name, top.density, bot.name, bot.density),
                });
            }
        }
    } else {
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!(
                "{} ({}) + {} ({}) — mixing verdicts are for two liquids or two gases; solids just sit there being solid (that's real solidity)",
                ma.name, sa.name(), mb.name, sb.name()
            ),
        });
    }
    out
}

/// `simulate: mix` — pour every liquid in the scene into one beaker:
/// merge miscible neighbors into solutions, keep immiscible ones as clean
/// density-stratified layers, and move the parts so the stack is VISIBLE.
pub fn simulate_mix(world: &mut World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    // gather visible liquid parts, heaviest first
    let mut idxs: Vec<usize> = world
        .parts
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.hidden && p.material.map(|m| materials::state(m)) == Some(State::Liquid))
        .map(|(i, _)| i)
        .collect();
    if idxs.len() < 2 {
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: "nothing to mix — put at least two liquids in the scene (water, oil, ethanol, acetone, glycerin, mercury)".into(),
        });
        return out;
    }
    idxs.sort_by(|&a, &b| {
        let da = world.parts[a].material.map(|m| m.density).unwrap_or(1000.0);
        let db = world.parts[b].material.map(|m| m.density).unwrap_or(1000.0);
        db.partial_cmp(&da).unwrap_or(std::cmp::Ordering::Equal) // heaviest first (bottom)
    });
    // report each chemistry adjacency, merging miscible runs into solutions
    let mut layers: Vec<(usize, String, f64)> = Vec::new(); // (part idx, display name, density)
    for (k, &i) in idxs.iter().enumerate() {
        let name_i = world.parts[i].name.clone();
        let rho_i = world.parts[i].material.map(|m| m.density).unwrap_or(1000.0);
        if let Some(last) = layers.last_mut() {
            let other = world.parts[last.0].material.map(|m| m.name).unwrap_or("");
            let me = world.parts[i].material.map(|m| m.name).unwrap_or("");
            match liquid_mix(other, me) {
                LiquidMix::Miscible { reason } => {
                    out.push(ConsoleLine {
                        kind: LineKind::Sim,
                        text: format!("{} + {} MIX into one phase — {}", other, me, reason),
                    });
                    // merge b into a: combined mesh, mass-weighted color, solution density
                    let (vol_a, vol_b) = (world.parts[last.0].volume_mm3, world.parts[i].volume_mm3);
                    let (m_a, m_b) = (world.parts[last.0].mass_g, world.parts[i].mass_g);
                    let col_a = world.parts[last.0].color.map(|c| [c.r, c.g, c.b]).unwrap_or([200, 200, 205]);
                    let col_b = world.parts[i].color.map(|c| [c.r, c.g, c.b]).unwrap_or([200, 200, 205]);
                    let blended = blend_color(col_a, col_b, m_a, m_b);
                    let rho_sol = (m_a + m_b) / ((vol_a + vol_b) * 1e-9) / 1000.0; // kg/m³
                    let merged = {
                        let mesh_b = world.parts[i].mesh.clone();
                        let mut m = std::mem::take(&mut world.parts[last.0].mesh);
                        m.merge(&mesh_b);
                        m
                    };
                    world.parts[last.0].mesh = merged;
                    world.parts[last.0].volume_mm3 = vol_a + vol_b;
                    world.parts[last.0].mass_g = m_a + m_b;
                    world.parts[last.0].color = Some(super::colors::Color::new(blended[0], blended[1], blended[2]));
                    world.parts[last.0].material = None; // a solution is its own thing
                    world.parts[last.0].centroid = world.parts[last.0].mesh.centroid();
                    world.parts[i].hidden = true;
                    last.1 = format!("{}+{}", last.1, name_i);
                    last.2 = rho_sol;
                    let _ = k;
                    continue;
                }
                LiquidMix::Immiscible { reason } => {
                    let top_rho = rho_i;
                    let bottom_rho = world.parts[last.0].material.map(|m| m.density).unwrap_or(1000.0);
                    let sitting = if top_rho <= bottom_rho { "floats on" } else { "sinks below" };
                    out.push(ConsoleLine {
                        kind: LineKind::Sim,
                        text: format!("{} {} {} — {}", name_i, sitting, last.1, reason),
                    });
                }
            }
        }
        layers.push((i, name_i, rho_i));
    }
    // stack the layers visibly: common X/Z center, from the ground up.
    // If the scene has a container (any solid part — a beaker, a tank),
    // pour into IT: layers center on the solid parts, not on the poured
    // puddles' original scatter.
    let solid_bb = {
        let mut bb = crate::math3::Aabb::empty();
        for p in world.parts.iter().filter(|p| {
            !p.hidden && !p.mesh.is_empty() && p.material.map(|m| materials::state(m)) != Some(State::Liquid)
        }) {
            bb.grow_box(&p.mesh.bbox());
        }
        if bb.is_empty() {
            world.stats.bbox
        } else {
            bb
        }
    };
    let (cx, cz) = (solid_bb.center().0[0], solid_bb.center().0[2]);
    let mut y = 0.0f64;
    out.push(ConsoleLine { kind: LineKind::Sim, text: "layer stack, bottom → top:".into() });
    for (idx, name, rho) in &layers {
        if world.parts[*idx].hidden {
            continue; // merged into a solution
        }
        let pb = world.parts[*idx].mesh.bbox();
        let h = (pb.max.y() - pb.min.y()).max(1.0);
        let dx = cx - (pb.min.x() + pb.max.x()) * 0.5;
        let dz = cz - (pb.min.z() + pb.max.z()) * 0.5;
        translate_part(&mut world.parts[*idx], dx, y - pb.min.y(), dz);
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!("  {:.0}–{:.0} mm: {} ({} kg/m³)", y, y + h, name, rho),
        });
        y += h;
    }
    if layers.len() > 1 {
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: "immiscible layers sort by density — that's chemistry, not luck: each liquid stays where its weight balances".into(),
        });
    }
    out
}

/// `simulate: gas` — release every gas in the scene inside a chamber:
/// reactive pairs react (stoichiometry, mass conserved), the rest diffuse
/// into a uniform mixture, light species rise / heavy species sink, and
/// gases EXPAND to fill the chamber (they have no fixed volume).
pub fn simulate_gas(world: &mut World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    let gas_idxs: Vec<usize> = world
        .parts
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.hidden && p.material.map(|m| materials::state(m)) == Some(State::Gas))
        .map(|(i, _)| i)
        .collect();
    if gas_idxs.is_empty() {
        out.push(ConsoleLine { kind: LineKind::Sim, text: "no gases in the scene — try material: hydrogen, oxygen, nitrogen, methane, chlorine…".into() });
        return out;
    }
    // 1) reactions between co-present reactive pairs (mass-conserving)
    for (a_i, b_i) in reactive_pairs(world, &gas_idxs) {
        let (na, nb) = (world.parts[a_i].material.map(|m| m.name).unwrap_or("").to_string(), world.parts[b_i].material.map(|m| m.name).unwrap_or("").to_string());
        let GasMix::Reacts { equation, product, notes } = gas_mix(&na, &nb) else { continue };
        // convert the limiting reagent with real stoichiometry
        let (m_a_g, m_b_g) = (world.parts[a_i].mass_g, world.parts[b_i].mass_g);
        let (g_b_per_g_a, prod_name) = stoichiometry(&na, &nb).unwrap_or((1.0, "products"));
        let consumed_a = m_a_g.min(m_b_g / g_b_per_g_a);
        let consumed_b = consumed_a * g_b_per_g_a;
        let product_mass = consumed_a + consumed_b;
        out.push(ConsoleLine {
            kind: LineKind::Warn,
            text: format!("reaction: {} + {} — {} ({}); {:.1} g + {:.1} g → {:.1} g of {} (mass conserved — chemistry never loses atoms)", na, nb, equation, notes, consumed_a, consumed_b, product_mass, prod_name),
        });
        // shrink both reactants by consumed volume; hide a part when fully consumed
        for (idx, consumed) in [(a_i, consumed_a), (b_i, consumed_b)] {
            let m0 = world.parts[idx].mass_g;
            if consumed >= m0 - 1e-9 {
                world.parts[idx].hidden = true;
            } else {
                let frac = 1.0 - consumed / m0;
                world.parts[idx].volume_mm3 *= frac;
                world.parts[idx].mass_g = m0 - consumed;
                world.parts[idx].centroid = world.parts[idx].mesh.centroid();
            }
        }
        // the product: a steam part with the reacted mass, at the reaction site
        let bb = world.parts[a_i.min(world.parts.len() - 1)].mesh.bbox();
        let c = bb.center();
        let prod_part = Part {
            name: product.split('+').next().unwrap_or("product").trim().to_string(),
            mesh: crate::geo::mesh::Mesh::new(),
            material: materials::find("steam"),
            color: Some(super::colors::Color::new(0xe8, 0xf0, 0xf2)),
            hidden: false,
            volume_mm3: product_mass * 1000.0 / 0.598 / 1000.0, // g→kg / (kg/m³) → m³ → mm³
            mass_g: product_mass,
            centroid: Some(c),
            area_mm2: 0.0,
        };
        world.parts.push(prod_part);
    }
    // 2) who is left — rise/sink in the current medium, then mix by diffusion
    let medium = super::settle::medium_density(&world.environment);
    let mut left: Vec<usize> = world
        .parts
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.hidden && p.material.map(|m| materials::state(m)) == Some(State::Gas))
        .map(|(i, _)| i)
        .collect();
    if left.is_empty() {
        out.push(ConsoleLine { kind: LineKind::Sim, text: "every gas reacted away — the chamber holds only products now".into() });
        return out;
    }
    left.sort_by(|&a, &b| {
        let da = world.parts[a].material.map(|m| m.density).unwrap_or(1.0);
        let db = world.parts[b].material.map(|m| m.density).unwrap_or(1.0);
        da.partial_cmp(&db).unwrap_or(std::cmp::Ordering::Equal) // lightest first → top
    });
    // chamber bounds: prefer the SOLID enclosure (a glass tank, a pipe…)
    // so gases collect INSIDE it; fall back to the whole scene bbox
    let bb = {
        let mut sb = crate::math3::Aabb::empty();
        for p in world.parts.iter().filter(|p| {
            !p.hidden && !p.mesh.is_empty() && p.material.map(|m| materials::state(m)) != Some(State::Gas)
        }) {
            sb.grow_box(&p.mesh.bbox());
        }
        if sb.is_empty() { world.stats.bbox } else { sb }
    };
    let top_y = bb.max.y().max(10.0);
    let bot_y = bb.min.y().min(0.0).max(0.0);
    let (cx, cz) = (bb.center().0[0], bb.center().0[2]);
    out.push(ConsoleLine { kind: LineKind::Sim, text: format!("chamber medium: {} kg/m³ — gases expand to fill it (no fixed volume, that's the gas definition)", medium) });
    let mut mixture: Vec<String> = Vec::new();
    let mut mixture_mass = 0.0;
    let mut mixture_moles = 0.0;
    for &i in &left {
        let (rho, name, mass_g) = {
            let p = &world.parts[i];
            (p.material.map(|m| m.density).unwrap_or(1.0), p.material.map(|m| m.name).unwrap_or("?"), p.mass_g)
        };
        let target_y = if rho < medium { top_y } else { bot_y }; // buoyancy in the medium decides
        let (dx, dy, dz) = {
            let pb = world.parts[i].mesh.bbox();
            (cx - (pb.min.x() + pb.max.x()) * 0.5, target_y - (pb.min.y() + pb.max.y()) * 0.5, cz - (pb.min.z() + pb.max.z()) * 0.5)
        };
        translate_part(&mut world.parts[i], dx, dy, dz);
        let dir = if rho < medium { "rises" } else { "sinks" };
        let moles = mass_g / molar_mass(name).unwrap_or(28.97);
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!("  {} ({} kg/m³) {} to the {} — {:.1} g = {:.2} mol", name, rho, dir, if rho < medium { "top" } else { "bottom" }, mass_g, moles),
        });
        mixture.push(format!("{} {:.0}%", name, 100.0 * mass_g / left_masses(world, &left)));
        mixture_mass += mass_g;
        mixture_moles += moles;
    }
    // diffusion verdict
    if left.len() > 1 {
        let rho_mix = mixture_mass * 1.0e6 / left_volume(world, &left).max(1e-9); // (g/1000)/(mm³·1e-9) = g·1e6/mm³
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!(
                "diffusion mixes them into ONE phase in ~seconds: mixture ≈ {} — ρ_mix ≈ {:.2} kg/m³, molar mass {:.1} g/mol (each gas keeps its own molecules; only the arrangement changes)",
                mixture.join(" + "), rho_mix, mixture_mass / mixture_moles.max(1e-9)
            ),
        });
    }
    out
}

// ---- helpers over world state ----

fn left_masses(world: &World, idxs: &[usize]) -> f64 {
    idxs.iter().map(|&i| world.parts[i].mass_g).sum()
}
fn left_volume(world: &World, idxs: &[usize]) -> f64 {
    idxs.iter().map(|&i| world.parts[i].volume_mm3).sum()
}

/// find co-present reactive gas pairs (indices into world.parts)
fn reactive_pairs(world: &World, gas_idxs: &[usize]) -> Vec<(usize, usize)> {
    let mut pairs = Vec::new();
    for (x, &i) in gas_idxs.iter().enumerate() {
        for &j in gas_idxs.iter().skip(x + 1) {
            let (ni, nj) = (world.parts[i].material.map(|m| m.name).unwrap_or(""), world.parts[j].material.map(|m| m.name).unwrap_or(""));
            if matches!(gas_mix(ni, nj), GasMix::Reacts { .. }) {
                pairs.push((i, j));
            }
        }
    }
    pairs
}



#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn polar_liquids_mix() {
        assert!(matches!(liquid_mix("water", "ethanol"), LiquidMix::Miscible { .. }));
        assert!(matches!(liquid_mix("water", "glycerin"), LiquidMix::Miscible { .. }));
        assert!(matches!(liquid_mix("acetone", "water"), LiquidMix::Miscible { .. }));
    }

    #[test]
    fn oil_refuses_water_and_floats() {
        match liquid_mix("water", "oil") {
            LiquidMix::Immiscible { .. } => {}
            other => panic!("oil must not mix with water: {:?}", other),
        }
        // layering: oil (920) above water (997)
        let w = super::super::eval::compile(
            "scene \"t\"\nunit: mm\nw1 = cube 60mm material: water at (0, 30mm, 0)\no1 = cube 60mm material: oil at (200mm, 30mm, 0)\nsimulate: mix",
        );
        let text: String = w.console.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join("\n");
        assert!(text.contains("floats on w1"), "oil must refuse water and layer on top: {}", text);
    }

    #[test]
    fn mercury_refuses_everything() {
        assert!(matches!(liquid_mix("mercury", "water"), LiquidMix::Immiscible { .. }));
        assert!(matches!(liquid_mix("mercury", "oil"), LiquidMix::Immiscible { .. }));
    }

    #[test]
    fn flammable_pairs_react_others_mix() {
        assert!(matches!(gas_mix("hydrogen", "oxygen"), GasMix::Reacts { .. }));
        assert!(matches!(gas_mix("methane", "oxygen"), GasMix::Reacts { .. }));
        assert!(matches!(gas_mix("nitrogen", "oxygen"), GasMix::Mixes));
        assert!(matches!(gas_mix("helium", "nitrogen"), GasMix::Mixes));
    }

    #[test]
    fn mix_pair_reports_layers() {
        let lines = mix_pair("water", "oil", 1);
        assert!(lines[0].text.contains("floats on") || lines[0].text.contains("do NOT mix"), "{:?}", lines);
        let lines = mix_pair("hydrogen", "oxygen", 1);
        assert!(lines[0].text.contains("2 H2 + O2 -> 2 H2O"), "{:?}", lines);
    }

    #[test]
    fn scene_gas_mixing_reports_mixture() {
        let w = super::super::eval::compile(
            "scene \"chamber\"\nunit: cm\na = sphere 8cm material: nitrogen at (0, 10cm, 0)\nb = sphere 8cm material: oxygen at (20cm, 10cm, 0)\nsimulate: gas",
        );
        let text: String = w.console.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join("\n");
        assert!(text.contains("diffusion mixes them"), "{}", text);
    }
}
