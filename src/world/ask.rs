//! P0550 — the ask engine: natural questions about the model, answered with
//! the math shown (so answers teach instead of hiding).

use super::eval::{ConsoleLine, LineKind, World};
use crate::units::WATER_DENSITY;

/// Answer a question about the world. Returns None when nothing sensible matches
/// (the caller then offers the catalog).
pub fn answer(q: &str, world: &World, _line: usize) -> Option<ConsoleLine> {
    let ql = q.to_lowercase();
    let text = answer_text(&ql, q, world)?;
    Some(ConsoleLine { kind: LineKind::Answer, text })
}

fn answer_text(ql: &str, _orig: &str, world: &World) -> Option<String> {
    // "mass of table?" — named object
    if let Some(name) = object_named(ql, "mass") {
        if let Some(p) = world.parts.iter().find(|p| p.name == name && !p.hidden) {
            let d = p.material.map(|m| m.density).unwrap_or(1050.0);
            return Some(format!(
                "{}: {:.1} g — {:.1} cm³ × {:.2} g/cm³ {}",
                p.name, p.mass_g, p.volume_mm3 / 1000.0, d / 1000.0,
                p.material.map(|m| m.name).unwrap_or("plastic")
            ));
        }
        return Some(format!("I can't find an object called \"{}\"", name));
    }
    // FIX(P0550-bug): "weight" contains "weigh" — without the exclusion the
    // mass branch swallowed `ask "weight?"` and the weight branch below was
    // unreachable (dead code). Route weight queries to the weight answer.
    if contains_any(&ql, &["mass", "weigh", "heavy"]) && !ql.contains("how many") && !ql.contains("weight") {
        let m = world.stats.total_mass_g;
        let v = world.stats.total_volume_mm3 / 1000.0;
        let rho = world.stats.avg_density_kg_m3;
        return Some(format!(
            "mass = {} = {:.2} kg — {:.1} cm³ of stuff at an average {:.0} kg/m³",
            mass_str(m), m / 1000.0, v, rho
        ));
    }
    if ql.contains("weight") {
        let n = world.stats.total_mass_g / 1000.0 * world.gravity;
        return Some(format!(
            "weight = m·g = {:.2} kg × {:.2} m/s² = {:.1} N (on {})",
            world.stats.total_mass_g / 1000.0,
            world.gravity,
            n,
            gravity_name(world.gravity)
        ));
    }
    if ql.contains("volume") || ql.contains("capacity") {
        return Some(format!("volume = {} of solid material", vol_str(world.stats.total_volume_mm3)));
    }
    if ql.contains("surface") || ql.contains("area") {
        return Some(format!("surface area = {}", area_str(world.stats.total_area_mm2)));
    }
    if ql.contains("density") {
        return Some(format!(
            "average density = {:.0} kg/m³ ({} / {})",
            world.stats.avg_density_kg_m3,
            mass_str(world.stats.total_mass_g),
            vol_str(world.stats.total_volume_mm3)
        ));
    }
    if ql.contains("center of mass") || ql.contains("centre of mass") || ql.contains("balance point") {
        return match world.stats.com {
            Some(c) => Some(format!(
                "center of mass = ({:.1}, {:.1}, {:.1}) cm — {}",
                c.x() / 10.0, c.y() / 10.0, c.z() / 10.0,
                if c.y() < world.stats.bbox.min.y() + world.stats.bbox.size().y() * 0.33 {
                    "it stands low and stable"
                } else {
                    "it stands tall — watch the balance"
                }
            )),
            None => Some("center of mass: nothing here yet".into()),
        };
    }
    if ql.contains("watertight") || ql.contains("manifold") || ql.contains("water tight") {
        let mut out = String::new();
        for p in world.parts.iter().filter(|p| !p.hidden) {
            let he = crate::geo::halfedge::HalfEdges::build(&p.mesh);
            let rep = he.manifold_report();
            if !out.is_empty() {
                out.push('\n');
            }
            if rep.closed {
                out.push_str(&format!("{}: watertight — closed 2-manifold, {} triangles", p.name, p.mesh.tris.len()));
            } else {
                out.push_str(&format!("{}: open mesh ({} boundary edges, {} open rings)", p.name, rep.boundary_edges, rep.open_vertex_rings));
            }
        }
        if out.is_empty() {
            return Some("nothing to check yet".into());
        }
        return Some(out);
    }
    if ql.contains("float") || ql.contains("buoy") || ql.contains("water") {
        let rho = world.stats.avg_density_kg_m3;
        if world.stats.total_mass_g <= 0.0 {
            return Some("nothing to float yet".into());
        }
        return Some(if rho < WATER_DENSITY {
            format!(
                "floats — {:.0} kg/m³ < water 1000 kg/m³ ({:.0}% submerged)",
                rho, rho / WATER_DENSITY * 100.0
            )
        } else {
            format!("sinks — {:.0} kg/m³ > water 1000 kg/m³", rho)
        });
    }
    if ql.contains("height") || ql.contains("tall") {
        let h = world.stats.bbox.size().y();
        return Some(format!("height = {}", len_str(h)));
    }
    // ---- 2.0 deep-tier questions ----
    if ql.contains("inertia") || ql.contains("spin") || ql.contains("moment of") {
        // per-part inertia tensors about their centers of mass
        let mut out = String::new();
        for p in world.parts.iter().filter(|p| !p.hidden) {
            let rho = p.material.map(|m| m.density).unwrap_or(1050.0);
            if let Some(i) = crate::geo::measure::inertia(&p.mesh, rho) {
                if !out.is_empty() {
                    out.push('\n');
                }
                out.push_str(&format!(
                    "{}: Ixx = {:.0} g·mm², Iyy = {:.0} g·mm², Izz = {:.0} g·mm² — {}",
                    p.name,
                    i.diag[0], i.diag[1], i.diag[2],
                    if i.diag[1] <= i.diag[0].min(i.diag[2]) {
                        "it spins most easily about Y (the vertical axis)"
                    } else {
                        "the vertical axis is not the easy spin axis"
                    }
                ));
            }
        }
        if !out.is_empty() {
            return Some(out);
        }
        return Some("make something first, then ask about inertia".into());
    }
    if ql.contains("cable sag") || ql.contains("rope sag") || (ql.contains("sag") && world.rope_sag.is_some()) {
        return match world.rope_sag {
            Some(s) => Some(format!(
                "cable sag = {:.1} mm — the XPBD solve (measured from the solved polyline, not guessed)"
                , s
            )),
            None => Some("no rope in this scene — make one: rope(from: (0, 30cm, 0), to: (40cm, 30cm, 0))".into()),
        };
    }
    if ql.contains("triangle") || ql.contains("tri ") || ql.contains("mesh") || ql.contains("polycount") {
        let t: usize = world.parts.iter().filter(|p| !p.hidden).map(|p| p.mesh.tris.len()).sum();
        let v: usize = world.parts.iter().filter(|p| !p.hidden).map(|p| p.mesh.verts.len()).sum();
        return Some(format!("{} triangles, {} vertices across the scene", t, v));
    }
    if ql.contains("how big") || ql.contains("size") || ql.contains("dimensions") || ql.contains("bounding") {
        let s = world.stats.bbox.size();
        return Some(format!(
            "size = {} × {} × {} (x × y × z)",
            len_str(s.x()), len_str(s.y()), len_str(s.z())
        ));
    }
    if ql.contains("how many") && (ql.contains("object") || ql.contains("part") || ql.contains("thing")) {
        let total = world.parts.iter().filter(|p| !p.hidden).count();
        return Some(format!(
            "{} object{} in the scene",
            total,
            if total == 1 { "" } else { "s" }
        ));
    }
    if ql.contains("melt") || ql.contains("melting") || ql.contains("temperature") || ql.contains("hot") {
        let mats: Vec<&str> = world
            .parts
            .iter()
            .filter(|p| !p.hidden)
            .filter_map(|p| p.material.map(|m| m.name))
            .collect();
        if mats.is_empty() {
            return Some("make something first, then ask about melting".into());
        }
        let mut lines = String::new();
        for mn in mats {
            match world.parts.iter().find(|p| p.material.map(|m| m.name) == Some(mn)) {
                Some(p) => {
                    if let Some(m) = p.material {
                        match m.melt_c {
                            Some(t) => lines.push_str(&format!("{} melts at {}°C — {} ({} g)", mn, t, if t > 2000.0 { "extreme" } else if t > 660.0 { "furnace territory" } else { "kitchen-oven hot" }, p.mass_g)),
                            None => lines.push_str(&format!("{} doesn't melt — it burns or decomposes first", mn)),
                        }
                        lines.push_str("\n");
                    }
                }
                None => {}
            }
        }
        return Some(format!("melting points:\n{}", lines.trim_end()));
    }
    if ql.contains("magnetic") || ql.contains("magnet") {
        let mags: Vec<&str> = world
            .parts
            .iter()
            .filter(|p| !p.hidden)
            .filter(|p| p.material.map(|m| m.magnetic).unwrap_or(false))
            .filter_map(|p| p.material.map(|m| m.name))
            .collect();
        return Some(if mags.is_empty() {
            "nothing here sticks to a magnet".into()
        } else {
            format!("magnetic: {}", mags.join(", "))
        });
    }
    // catalog
    Some(
        "I can answer: mass?, weight?, volume?, surface area?, density?, center of mass?, \
         will it float?, height?, size?, how many objects?, melting point?, magnetic?"
            .into(),
    )
}

fn object_named(ql: &str, word: &str) -> Option<String> {
    // "mass of table" / "mass of the table"
    let idx = ql.find(&format!("{} of", word))?;
    let rest = &ql[idx + word.len() + 3..];
    let rest = rest.strip_prefix("the ").unwrap_or(rest);
    let rest = rest.trim().trim_end_matches('?').trim();
    if rest.is_empty() || rest.contains(' ') {
        return None;
    }
    Some(rest.to_string())
}

fn contains_any(s: &str, words: &[&str]) -> bool {
    words.iter().any(|w| s.contains(w))
}

fn gravity_name(g: f64) -> String {
    if (g - crate::units::G_EARTH).abs() < 0.01 { "earth".into() }
    else if (g - crate::units::G_MOON).abs() < 0.01 { "the moon".into() }
    else if (g - crate::units::G_MARS).abs() < 0.01 { "mars".into() }
    else if g == 0.0 { "zero gravity".into() }
    else { "this planet".into() }
}

// ---------- friendly unit formatting ----------

pub fn len_str(mm: f64) -> String {
    if mm.abs() >= 1000.0 {
        format!("{:.2} m", mm / 1000.0)
    } else if mm.abs() >= 10.0 {
        format!("{:.1} cm", mm / 10.0)
    } else {
        format!("{:.1} mm", mm)
    }
}

pub fn mass_str(g: f64) -> String {
    if g >= 1_000_000.0 {
        format!("{:.2} t", g / 1e6)
    } else if g >= 1000.0 {
        format!("{:.1} kg", g / 1000.0)
    } else if g >= 1.0 {
        format!("{:.1} g", g)
    } else {
        format!("{:.0} mg", g * 1000.0)
    }
}

pub fn vol_str(mm3: f64) -> String {
    if mm3 >= 1e9 {
        format!("{:.2} m³", mm3 / 1e9)
    } else if mm3 >= 1e6 {
        format!("{:.2} L", mm3 / 1e6)
    } else if mm3 >= 1000.0 {
        format!("{:.1} cm³", mm3 / 1000.0)
    } else {
        format!("{:.0} mm³", mm3)
    }
}

pub fn area_str(mm2: f64) -> String {
    if mm2 >= 1e6 {
        format!("{:.2} m²", mm2 / 1e6)
    } else if mm2 >= 100.0 {
        format!("{:.1} cm²", mm2 / 100.0)
    } else {
        format!("{:.0} mm²", mm2)
    }
}
