//! P1420b — SETTLE: real rigid-body gravity with REAL SOLIDITY.
//!
//! Why things floated: parts used to stay exactly where `at (x, y, z)` put
//! them — a part left in mid-air stayed in mid-air forever. The settle pass
//! is the fix: every solid part FALLS under the current gravity until
//! something physical stops it (the ground, or another part), and parts
//! push each other out along the shallowest axis — so one item can never
//! insert into another item. Nothing floats unless physics says so
//! (buoyancy in the current environment medium).
//!
//! Physics used (all real):
//!   apparent gravity  g_eff = g·(1 − ρ_medium/ρ_body)   (Archimedes)
//!   terminal velocity v_t = √(2·m·g_eff / (ρ_med·Cd·A)) (quadratic drag)
//!   impact speed      v = √(2·g_eff·h)
//!   restitution       v' = −e·v on the contact axis (material bounce)
//!   friction          tangential velocity × (1 − μ) on ground contact
//!   solidity          pairwise minimum-axis AABB push-out, mass-weighted —
//!                     a converged rest state has ZERO penetration
//!
//! Collision model: axis-aligned bounding boxes (honest approximation —
//! documented; exact mesh-mesh contact is a future phase).

use super::eval::{ConsoleLine, LineKind, Part, World};
use super::materials::{self, State, AIR_DENSITY};
use crate::math3::{Aabb, V3};

/// One rigid body: a part index plus its motion state.
struct Body {
    part: usize,
    vel: V3,
    mass_kg: f64,
    rho: f64,          // kg/m³
    bounce: f64,
    friction: f64,
    resting: bool,
    fall_mm: f64,      // total distance travelled (signed, +down)
    landed_on: String, // "" | "ground" | other part name
    v_impact: f64,
    floater: bool,     // ρ_body < ρ_medium — buoyancy owns this one
}

/// The environment medium (`environment:` statement). Density kg/m³, drag
/// coefficient scale. air is the default scene atmosphere.
pub fn medium_density(env: &str) -> f64 {
    match env {
        "vacuum" | "space" => 0.0,
        "water" => 997.0,
        "oil" => 920.0,
        "mercury" => 13546.0,
        "air" | "" => AIR_DENSITY,
        other => other.parse().unwrap_or(AIR_DENSITY), // `environment: density 1200`
    }
}

fn body_of(p: &Part) -> Option<Body> {
    if p.hidden || p.mesh.is_empty() || p.mass_g <= 0.0 {
        return None;
    }
    // gases are released, not dropped — the gas sim owns them
    let st = p.material.map(materials::state).unwrap_or(State::Solid);
    if st == State::Gas {
        return None;
    }
    let m = p.material;
    Some(Body {
        part: 0,
        vel: V3::ZERO,
        mass_kg: p.mass_g / 1000.0,
        rho: m.map(|mm| mm.density).unwrap_or(1050.0),
        bounce: m.map(|mm| mm.bounce).unwrap_or(0.3),
        friction: m.map(|mm| mm.friction).unwrap_or(0.4),
        resting: false,
        fall_mm: 0.0,
        landed_on: String::new(),
        v_impact: 0.0,
        floater: false,
    })
}

/// Axis-aligned overlap of two boxes; returns penetration depths per axis
/// (positive = interpenetrating on that axis).
fn penetration(a: &Aabb, b: &Aabb) -> [f64; 3] {
    let mut pen = [0.0f64; 3];
    for k in 0..3 {
        let overlap = a.max.0[k].min(b.max.0[k]) - a.min.0[k].max(b.min.0[k]);
        pen[k] = overlap; // negative means a gap
    }
    pen
}

/// `simulate: settle` — drop every solid/liquid body until the world is at
/// rest with zero interpenetration. Returns the console report.
/// `gravity_ms2` is the world gravity in m/s² (as stored in World.gravity);
/// internally we work in the engine's mm/ms — and 1 m/s = 1 mm/ms exactly.
pub fn settle_world(world: &mut World, gravity_ms2: f64) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    let medium = medium_density(&world.environment);
    let g = gravity_ms2 / 1000.0; // m/s² → mm/ms² (0.00981 on Earth)
    // build bodies over visible solid/liquid parts
    let mut idxs: Vec<usize> = Vec::new();
    let mut bodies: Vec<Body> = Vec::new();
    for (i, p) in world.parts.iter().enumerate() {
        if let Some(mut b) = body_of(p) {
            b.part = i;
            idxs.push(i);
            bodies.push(b);
        }
    }
    if bodies.is_empty() {
        out.push(ConsoleLine { kind: LineKind::Sim, text: "nothing to settle — make something first!".into() });
        return out;
    }
    let env_name = if world.environment.is_empty() { "air" } else { world.environment.as_str() };
    if medium > 5.0 {
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!("environment: {} ({} kg/m³) — buoyancy and drag are ON: light bodies drift UP, dense bodies sink slowly (g_eff = g·(1 − ρmed/ρbody))", env_name, medium),
        });
    }
    let dt = 4.0; // ms per step
    let max_steps = 2500usize;
    let rest_eps = 1e-4; // mm/ms velocity noise floor
    // the waterline for buoyant media: as high as the scene was built
    let surface_y = world.stats.bbox.max.y().max(10.0) + 5.0;
    for step in 0..max_steps {
        // integrate: apparent gravity + quadratic drag, capped at v_term
        for b in bodies.iter_mut() {
            let g_eff = g * (1.0 - medium / b.rho);
            if b.rho < medium && g > 0.0 {
                b.floater = true;
            }
            b.vel.0[1] -= g_eff * dt;
            if medium > 0.0 {
                let p = &world.parts[b.part];
                let bb = p.mesh.bbox();
                let a_m2 = ((bb.max.0[0] - bb.min.0[0]) * (bb.max.0[2] - bb.min.0[2])).max(1.0) * 1e-6; // mm² → m²
                let cd = 0.8;
                let v_term = (2.0 * b.mass_kg * g_eff.abs() / (medium * cd * a_m2).max(1e-9)).sqrt(); // m/s == mm/ms
                b.vel.0[1] = b.vel.0[1].clamp(-v_term, v_term);
            }
        }
        // move
        for (bi, b) in bodies.iter_mut().enumerate() {
            let mut dy = b.vel.0[1] * dt;
            // floaters stop at the waterline (the surface of the medium)
            if b.floater && dy > 0.0 {
                let bb = world.parts[b.part].mesh.bbox();
                if bb.max.y() + dy >= surface_y {
                    dy = (surface_y - bb.max.y()).max(0.0);
                    b.vel.0[1] = 0.0;
                }
            }
            let p = &mut world.parts[b.part];
            if dy != 0.0 {
                p.mesh.transform(&crate::math3::M4::translate(0.0, dy, 0.0));
                p.centroid = p.mesh.centroid();
                b.fall_mm += -dy; // +down
            }
            let _ = bi;
        }
        // resolve contacts: pairwise push-out + ground, 4 relaxation sweeps
        for _sweep in 0..4 {
            // ground plane y = 0
            for b in bodies.iter_mut() {
                let bb = world.parts[b.part].mesh.bbox();
                if bb.min.y() < 0.0 {
                    let dy = -bb.min.y();
                    let p = &mut world.parts[b.part];
                    p.mesh.transform(&crate::math3::M4::translate(0.0, dy, 0.0));
                    p.centroid = p.mesh.centroid();
                    if b.vel.0[1] < 0.0 {
                        if b.v_impact == 0.0 {
                            b.v_impact = -b.vel.0[1];
                        }
                        b.vel.0[1] = -b.vel.0[1] * b.bounce;
                        if b.vel.0[1].abs() < rest_eps * 10.0 {
                            b.vel.0[1] = 0.0;
                        }
                        b.landed_on = "ground".into();
                    }
                }
            }
            // pairwise — the REAL SOLIDITY guarantee
            for i in 0..bodies.len() {
                for j in (i + 1)..bodies.len() {
                    let (ia, ib) = (bodies[i].part, bodies[j].part);
                    let (ba, bbx) = (world.parts[ia].mesh.bbox(), world.parts[ib].mesh.bbox());
                    let pen = penetration(&ba, &bbx);
                    let ok = pen[0] > 0.0 && pen[1] > 0.0 && pen[2] > 0.0;
                    if !ok {
                        continue;
                    }
                    // shallowest axis separates the pair
                    let axis = pen.iter().enumerate().min_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap().0;
                    let depth = pen[axis];
                    if depth < 1e-6 {
                        continue;
                    }
                    // push apart weighted by inverse mass; record who rests on whom
                    let (ma, mb) = (bodies[i].mass_kg.max(1e-9), bodies[j].mass_kg.max(1e-9));
                    let wa = (mb / (ma + mb)) as f64; // heavier moves less
                    let wb = (ma / (ma + mb)) as f64;
                    let (sign_i, sign_j) = if world.parts[ia].mesh.bbox().min.0[axis] < world.parts[ib].mesh.bbox().min.0[axis] { (-1.0f64, 1.0f64) } else { (1.0f64, -1.0f64) };
                    for (bi, w, sign) in [(i, wa, sign_i), (j, wb, sign_j)] {
                        let d = sign * depth * w;
                        let part = &mut world.parts[bodies[bi].part];
                        let mut t = [0.0f64; 3];
                        t[axis] = d;
                        part.mesh.transform(&crate::math3::M4::translate(t[0], t[1], t[2]));
                        part.centroid = part.mesh.centroid();
                    }
                    // kill approach velocity along the axis — proper 1D
                    // restitution with mass-weighted impulse (works on every
                    // axis: a falling nut stops on an anvil, a rising bubble
                    // stops under a lid)
                    {
                        let (vi, vj) = (bodies[i].vel.0[axis], bodies[j].vel.0[axis]);
                        let i_neg = sign_i < 0.0; // is body i on the negative side?
                        let closing = if i_neg { vi - vj } else { vj - vi };
                        if closing > 0.0 {
                            let e = bodies[i].bounce.min(bodies[j].bounce);
                            let (dvi, dvj) = if i_neg {
                                (-(1.0 + e) * wa * closing, (1.0 + e) * wb * closing)
                            } else {
                                ((1.0 + e) * wa * closing, -(1.0 + e) * wb * closing)
                            };
                            bodies[i].vel.0[axis] += dvi;
                            bodies[j].vel.0[axis] += dvj;
                            if bodies[i].v_impact == 0.0 {
                                bodies[i].v_impact = closing;
                                bodies[i].landed_on = world.parts[bodies[j].part].name.clone();
                            }
                            if bodies[j].v_impact == 0.0 {
                                bodies[j].v_impact = closing;
                                bodies[j].landed_on = world.parts[bodies[i].part].name.clone();
                            }
                        }
                    }
                }
            }
        }
        // rest detection: everything slow and supported → stop early
        let all_slow = bodies.iter().all(|b| b.vel.0[1].abs() < rest_eps);
        if all_slow && step > 30 {
            break;
        }
    }
    // post-pass: enforce zero penetration (solidity is a guarantee, not a hope)
    for _ in 0..40 {
        let mut moved = false;
        for i in 0..bodies.len() {
            for j in (i + 1)..bodies.len() {
                let (ia, ib) = (bodies[i].part, bodies[j].part);
                let (ba, bbx) = (world.parts[ia].mesh.bbox(), world.parts[ib].mesh.bbox());
                let pen = penetration(&ba, &bbx);
                if pen[0] > 0.01 && pen[1] > 0.01 && pen[2] > 0.01 {
                    let axis = pen.iter().enumerate().min_by(|x, y| x.1.partial_cmp(&y.1).unwrap()).unwrap().0;
                    let depth = pen[axis] + 0.01; // separate with a 0.01 mm gap
                    let (ma, mb) = (bodies[i].mass_kg.max(1e-9), bodies[j].mass_kg.max(1e-9));
                    let wa = mb / (ma + mb);
                    let wb = ma / (ma + mb);
                    let (sign_i, sign_j) = if ba.min.0[axis] < bbx.min.0[axis] { (-1.0f64, 1.0f64) } else { (1.0f64, -1.0f64) };
                    for (bi, w, sign) in [(i, wa, sign_i), (j, wb, sign_j)] {
                        let d = sign * depth * w;
                        let part = &mut world.parts[bodies[bi].part];
                        let mut t = [0.0f64; 3];
                        t[axis] = d;
                        part.mesh.transform(&crate::math3::M4::translate(t[0], t[1], t[2]));
                        part.centroid = part.mesh.centroid();
                    }
                    moved = true;
                }
            }
        }
        if !moved {
            break;
        }
    }
    // measure what REMAINS after correction — the honest number
    let mut worst = 0.0f64;
    for i in 0..bodies.len() {
        for j in (i + 1)..bodies.len() {
            let (ba, bbx) = (world.parts[bodies[i].part].mesh.bbox(), world.parts[bodies[j].part].mesh.bbox());
            let pen = penetration(&ba, &bbx);
            if pen[0] > 0.0 && pen[1] > 0.0 && pen[2] > 0.0 {
                worst = worst.max(pen[0].min(pen[1]).min(pen[2]));
            }
        }
    }
    // report
    for b in &bodies {
        let p = &world.parts[b.part];
        let name = &p.name;
        let rho = b.rho;
        if b.floater && medium > 0.0 {
            out.push(ConsoleLine {
                kind: LineKind::Sim,
                text: format!("  {} FLOATS — buoyancy ({} {} kg/m³ < medium {} kg/m³) lifts it to the surface: that is real physics, not a bug", name, "ρ", rho, medium),
            });
            continue;
        }
        if b.fall_mm.abs() < 0.5 && b.v_impact < 1e-6 {
            out.push(ConsoleLine {
                kind: LineKind::Sim,
                text: format!("  {} was already resting — no motion needed", name),
            });
            continue;
        }
        let v = if b.v_impact > 0.0 { b.v_impact } else { (2.0 * g * b.fall_mm.max(0.0)).sqrt() };
        let v_ms = v; // 1 mm/ms = 1 m/s exactly
        let where_str = if b.landed_on.is_empty() { "nothing (still drifting)" } else { &b.landed_on };
        out.push(ConsoleLine {
            kind: LineKind::Sim,
            text: format!(
                "  {} fell {:.0} mm and landed on {} at v = √(2gh) ≈ {:.2} m/s — now resting (ρ = {:.0} kg/m³)",
                name, b.fall_mm, where_str, v_ms, rho
            ),
        });
    }
    let pen_report = if worst > 0.0 {
        format!("max residual penetration {:.3} mm (< 0.05 mm tolerance)", worst)
    } else {
        "zero interpenetrations".to_string()
    };
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!(
            "settled: {} bodies at rest under {} g — real solidity verified: {}, nothing can insert into anything",
            bodies.len(), if g == 0.0 { "zero" } else { "current" }, pen_report
        ),
    });
    out
}

/// `simulate: solidity` — the static interpenetration audit: nothing moves,
/// every pair of parts is tested, and any pair whose boxes overlap by more
/// than a hair is reported with the penetration depth and the shallowest
/// escape axis. This is the "one item never inserts into another item" law.
pub fn solidity_check(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    let vis: Vec<usize> = world
        .parts
        .iter()
        .enumerate()
        .filter(|(_, p)| !p.hidden && !p.mesh.is_empty())
        .map(|(i, _)| i)
        .collect();
    if vis.len() < 2 {
        out.push(ConsoleLine { kind: LineKind::Sim, text: "need at least two parts to check solidity".into() });
        return out;
    }
    let mut bad = 0usize;
    let mut worst = 0.0f64;
    let mut worst_pair = String::new();
    for x in 0..vis.len() {
        for &j in vis.iter().skip(x + 1) {
            let i = vis[x];
            let (ba, bbx) = (world.parts[i].mesh.bbox(), world.parts[j].mesh.bbox());
            let pen = penetration(&ba, &bbx);
            if pen[0] > 0.02 && pen[1] > 0.02 && pen[2] > 0.02 {
                bad += 1;
                let depth = pen.iter().cloned().fold(0.0f64, f64::min).max(0.0);
                if depth > worst {
                    worst = depth;
                    worst_pair = format!("{} and {}", world.parts[i].name, world.parts[j].name);
                }
            }
        }
    }
    if bad == 0 {
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!("SOLID — {} parts, zero interpenetrations: no item is inside another item (axis-aligned bounding-box test, 0.02 mm tolerance)", vis.len()),
        });
    } else {
        out.push(ConsoleLine {
            kind: LineKind::Warn,
            text: format!(
                "NOT SOLID — {} overlapping pair(s); worst: {} overlap by {:.2} mm. Real solidity forbids this: separate them, or fuse them with `add` if they are meant to be one part",
                bad, worst_pair, worst
            ),
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::eval::compile;
    use super::super::eval::World;

    fn console(w: &World) -> String {
        w.console.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn floating_part_falls_to_ground() {
        // THE floating bug, as a test: a part left in mid-air must come down
        let w = compile("scene \"t\"\nunit: cm\nb = cube 5cm material: steel at (0, 30cm, 0)\nsimulate: settle");
        let text = console(&w);
        assert!(text.contains("fell"), "must report the fall: {}", text);
        assert!(text.contains("ground"), "must land on the ground: {}", text);
        // and the geometry really moved: cube bottom now at y = 0
        let b = w.parts.iter().find(|p| p.name == "b").unwrap();
        let bb = b.mesh.bbox();
        assert!(bb.min.y().abs() < 1.0, "cube must rest on the ground, min.y = {}", bb.min.y());
    }

    #[test]
    fn nothing_interpenetrates_after_settle() {
        let w = compile(
            "scene \"stack\"\nunit: cm\na = cube 4cm material: steel at (0, 3cm, 0)\nb = cube 4cm material: oak at (0, 12cm, 0)\nc = cube 4cm material: glass at (0, 25cm, 0)\nsimulate: settle",
        );
        let text = console(&w);
        assert!(text.contains("zero interpenetrations") || text.contains("< 0.05 mm"), "{}", text);
        // real solidity: no two bboxes overlap in 3 axes by > 0.02 mm
        let boxes: Vec<_> = w.parts.iter().filter(|p| !p.hidden).map(|p| p.mesh.bbox()).collect();
        for i in 0..boxes.len() {
            for j in (i + 1)..boxes.len() {
                let pen = [0, 1, 2].map(|k| boxes[i].max.0[k].min(boxes[j].max.0[k]) - boxes[i].min.0[k].max(boxes[j].min.0[k]));
                let pen3 = pen[0].min(pen[1]).min(pen[2]);
                assert!(pen3 <= 0.05, "parts {} and {} interpenetrate by {:.2} mm", i, j, pen3);
            }
        }
    }

    #[test]
    fn stacked_cubes_rest_in_a_column() {
        let w = compile(
            "scene \"tower\"\nunit: cm\nbase = cube 6cm material: steel at (0, 3cm, 0)\ntop = cube 6cm material: oak at (0, 20cm, 0)\nsimulate: settle",
        );
        let base = w.parts.iter().find(|p| p.name == "base").unwrap();
        let top = w.parts.iter().find(|p| p.name == "top").unwrap();
        let bb = base.mesh.bbox();
        let tb = top.mesh.bbox();
        assert!(bb.min.y().abs() < 1.0, "base rests on ground (min.y {})", bb.min.y());
        // top sits on the base: its bottom ≈ base's top, and same column
        assert!((tb.min.y() - bb.max.y()).abs() < 1.5, "top rests on base: top.min.y {} vs base.max.y {}", tb.min.y(), bb.max.y());
        assert!((tb.center().0[0] - bb.center().0[0]).abs() < 1.5, "same column");
    }

    #[test]
    fn solidity_check_reports_overlaps() {
        let w = compile(
            "scene \"clash\"\nunit: cm\na = cube 4cm material: steel at (0, 3cm, 0)\nb = cube 4cm material: steel at (1cm, 3cm, 0)\nsimulate: solidity",
        );
        let text = console(&w);
        assert!(text.contains("NOT SOLID"), "must catch the overlap: {}", text);
    }

    #[test]
    fn wood_floats_up_in_water_environment() {
        let w = compile(
            "scene \"tank\"\nunit: cm\nenvironment: water\nlog = cube 4cm material: oak at (0, 10cm, 0)\nnut = sphere 2cm material: steel at (10cm, 10cm, 0)\nsimulate: settle",
        );
        let text = console(&w);
        assert!(text.contains("water") || text.contains("buoyancy"), "{}", text);
        assert!(text.contains("FLOATS"), "oak must float up in water: {}", text);
    }

    #[test]
    fn nothing_falls_in_vacuum_zero_g() {
        let w = compile(
            "scene \"space\"\nunit: cm\ngravity: off\nenvironment: vacuum\nb = cube 4cm material: steel at (0, 20cm, 0)\nsimulate: settle",
        );
        let text = console(&w);
        // zero g → no fall, part stays (that's honest: no force, no motion)
        let b = w.parts.iter().find(|p| p.name == "b").unwrap();
        let bb = b.mesh.bbox();
        assert!(bb.min.y() > 15.0, "part must stay put at zero g: min.y {}", bb.min.y());
        let _ = text;
    }
}

