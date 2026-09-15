//! P2300 — OTD4 AERODYNAMICS — the science of air moving past things.
//!
//! Where fluid dynamics studies the fluid in its own frame, aerodynamics
//! studies the body that pushes through it. The wing, the windshield, the
//! parachute, the propeller — every shape that meets moving air answers to
//! the same four forces: lift, drag, thrust, weight.
//!
//! Laws carried here:
//!   Continuity      — A·v = const (narrow a pipe, the flow speeds up)
//!   Bernoulli       — P + ½ρv² + ρgh = const (faster air = lower pressure)
//!   Lift            — L = ½·ρ·v²·A·C_L (the wing's reason to exist)
//!   Drag            — D = ½·ρ·v²·A·C_D (every shape pays this toll)
//!   Reynolds        — Re = ρvL/μ (laminar vs turbulent switch)
//!   Mach            — M = v/c (the speed-of-sound fraction that breaks wings)
//!   Terminal vel.   — v_t = √(2mg / ρ·A·C_D) (the speed a fall stops accelerating)

use super::eval::{ConsoleLine, LineKind, World};
use crate::units::G_EARTH;

/// Sea-level air density at 15 °C, kg/m³ (the ISA standard).
pub const RHO_AIR_SL: f64 = 1.225;
/// Speed of sound at sea level, 15 °C, m/s.
pub const C_SOUND_SL: f64 = 340.3;
/// Dynamic viscosity of air at 15 °C, Pa·s.
pub const MU_AIR: f64 = 1.81e-5;
/// Kinematic viscosity of air at 15 °C, m²/s.
pub const NU_AIR: f64 = 1.48e-5;

/// Air density at altitude h (m), ISO standard atmosphere (0–11 km troposphere).
pub fn air_density_at(h_m: f64) -> f64 {
    if h_m < 0.0 {
        return RHO_AIR_SL;
    }
    if h_m > 11000.0 {
        return 0.364; // stratosphere isothermal
    }
    let t0 = 288.15; // K
    let lapse = 0.0065; // K/m
    let t = t0 - lapse * h_m;
    RHO_AIR_SL * (t / t0).powf(4.256)
}

/// Speed of sound at altitude h (m), m/s. c = √(γRT), γ = 1.4.
pub fn sound_speed_at(h_m: f64) -> f64 {
    let t0 = 288.15;
    let lapse = 0.0065;
    let t = (t0 - lapse * h_m).max(220.0);
    (1.4 * 287.05 * t).sqrt()
}

/// Reynolds number: Re = ρ·v·L / μ. Below ~2300 in a pipe = laminar, above = turbulent.
pub fn reynolds(rho: f64, v: f64, length: f64, mu: f64) -> f64 {
    if mu <= 0.0 {
        return f64::INFINITY;
    }
    rho * v * length / mu
}

/// Mach number: v / c. M<1 subsonic, M≈1 transonic (drag rises 10×), M>1 supersonic.
pub fn mach(v: f64, c: f64) -> f64 {
    if c <= 0.0 {
        return f64::INFINITY;
    }
    v / c
}

/// Lift force (N): L = ½·ρ·v²·A·C_L. A typical airfoil C_L ≈ 1.0 at takeoff.
pub fn lift(rho: f64, v: f64, area_m2: f64, c_l: f64) -> f64 {
    0.5 * rho * v * v * area_m2 * c_l
}

/// Drag force (N): D = ½·ρ·v²·A·C_D.
pub fn drag(rho: f64, v: f64, area_m2: f64, c_d: f64) -> f64 {
    0.5 * rho * v * v * area_m2 * c_d
}

/// Terminal velocity (m/s): the speed where drag balances weight.
/// v_t = √(2·m·g / (ρ·A·C_D))
pub fn terminal_velocity(mass_kg: f64, g: f64, rho: f64, area_m2: f64, c_d: f64) -> f64 {
    if rho <= 0.0 || area_m2 <= 0.0 || c_d <= 0.0 {
        return f64::INFINITY;
    }
    (2.0 * mass_kg * g / (rho * area_m2 * c_d)).sqrt()
}

/// Drag coefficient by shape class (typical values, honest ranges).
pub fn drag_coefficient(shape_hint: &str) -> f64 {
    match shape_hint.to_lowercase().as_str() {
        "sphere" => 0.47,
        "cube" | "box" => 1.05,
        "cylinder" | "rod" => 0.82, // long axis across flow
        "cone" => 0.50,
        "streamlined" | "teardrop" | "airfoil" => 0.04,
        "flat_plate" | "plate" => 1.28,
        "hemisphere" => 0.42,
        "bicycle" | "cyclist" => 1.0,
        "car" | "sedan" => 0.30,
        "truck" => 0.8,
        "wing" | "plane" => 0.05,
        "parachute" => 1.50,
        _ => 1.0,
    }
}

/// Lift coefficient by class. Airfoils: 0.0 (cruise) to 1.4 (flaps down).
pub fn lift_coefficient(hint: &str) -> f64 {
    match hint.to_lowercase().as_str() {
        "wing" | "airfoil" | "plane" => 1.0,
        "wing_flaps" | "takeoff" => 1.4,
        "cylinder" | "sphere" => 0.0, // bluff bodies don't lift by default
        _ => 0.0,
    }
}

/// Estimate frontal area (m²) from the part's bounding box.
pub fn frontal_area_m2(world: &World, idx: usize) -> f64 {
    if idx >= world.parts.len() {
        return 0.0;
    }
    let bb = world.parts[idx].mesh.bbox();
    let w = (bb.max.0[0] - bb.min.0[0]).max(0.1) * 1e-3;
    let h = (bb.max.0[1] - bb.min.0[1]).max(0.1) * 1e-3;
    w * h
}

/// `simulate: aero` — the aerodynamic survey of the scene at sea level.
pub fn aero_sim(world: &World) -> Vec<ConsoleLine> {
    aero_sim_at(world, 0.0, 30.0)
}

/// `simulate: aero` with altitude (m) and airspeed (m/s) from the caller.
pub fn aero_sim_at(world: &World, altitude_m: f64, airspeed_mps: f64) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    let rho = air_density_at(altitude_m);
    let c = sound_speed_at(altitude_m);
    let v = airspeed_mps;
    let m = mach(v, c);
    let regime = if m < 0.8 {
        "subsonic"
    } else if m < 1.2 {
        "transonic (drag rises 10× here — the sound barrier)"
    } else if m < 5.0 {
        "supersonic (shockwaves form, leading edges heat up)"
    } else {
        "hypersonic (plasma forms, surfaces ablate)"
    };

    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!(
            "AERODYNAMIC SURVEY at {:.0} m altitude, airspeed {:.1} m/s",
            altitude_m, v
        ),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  air: ρ = {:.3} kg/m³, c = {:.1} m/s — Mach {:.2} ⇒ {}",
            rho, c, m, regime
        ),
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

    for &i in &visible {
        let p = &world.parts[i];
        let mat_name = p.material.map(|m| m.name).unwrap_or("plastic");
        let mass_kg = p.mass_g / 1000.0;
        let bb = p.mesh.bbox();
        let size_x = (bb.max.0[0] - bb.min.0[0]).max(1.0) * 1e-3;
        let size_y = (bb.max.0[1] - bb.min.0[1]).max(1.0) * 1e-3;
        let size_z = (bb.max.0[2] - bb.min.0[2]).max(1.0) * 1e-3;
        let length = size_x.max(size_y).max(size_z);
        let a_front = size_y * size_z; // assume facing forward along x
        let shape_hint = infer_shape(p);
        let c_d = drag_coefficient(&shape_hint);
        let c_l = lift_coefficient(&shape_hint);
        let re = reynolds(rho, v, length, MU_AIR);
        let d = drag(rho, v, a_front, c_d);
        let l = lift(rho, v, a_front, c_l);
        let v_t = terminal_velocity(mass_kg, G_EARTH, rho, a_front, c_d);
        let re_regime = if re < 2300.0 {
            "laminar"
        } else if re < 1.0e5 {
            "transitional"
        } else {
            "turbulent"
        };

        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!(
                "  {} [{}] {} kg, {} m² frontal, length {:.2} m — Re {:.2e} ({})",
                p.name, mat_name, mass_kg, shape_hint, a_front, re, re_regime
            ),
        });
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!(
                "    drag D = ½ρv²AC_D = {:.2} N (C_D {:.2}); terminal velocity {:.1} m/s; falls like this for {:.0} m before reaching steady speed",
                d, c_d, v_t, v_t * v_t / (2.0 * G_EARTH)
            ),
        });
        if c_l > 0.0 {
            let wing_loading = mass_kg * G_EARTH / a_front.max(1e-6);
            let lift_to_weight = l / (mass_kg * G_EARTH).max(1e-9);
            out.push(ConsoleLine {
                kind: LineKind::Answer,
                text: format!(
                    "    lift L = ½ρv²AC_L = {:.2} N (C_L {:.2}); L/W = {:.2} ⇒ {}, wing loading {:.0} N/m²",
                    l, c_l, lift_to_weight,
                    if lift_to_weight > 1.0 { "FLIES at this speed" } else { "too heavy to fly at this speed" },
                    wing_loading
                ),
            });
            let stall_speed = (2.0 * mass_kg * G_EARTH / (rho * a_front * c_l.max(1e-9))).sqrt();
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!(
                    "    stall speed v_min = √(2mg/ρAC_L) = {:.1} m/s — fly slower and the wing stops lifting",
                    stall_speed
                ),
            });
        }
    }

    // The scene-wide picture: total drag and the power to overcome it
    let mut total_drag = 0.0;
    for &i in &visible {
        let p = &world.parts[i];
        let bb = p.mesh.bbox();
        let a_front = ((bb.max.0[1] - bb.min.0[1]).max(1.0) * 1e-3)
            * ((bb.max.0[2] - bb.min.0[2]).max(1.0) * 1e-3);
        let c_d = drag_coefficient(&infer_shape(p));
        total_drag += drag(rho, v, a_front, c_d);
    }
    let power_w = total_drag * v;
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!(
            "scene total drag = {:.1} N — to sustain {:.1} m/s requires {:.0} W of thrust power",
            total_drag, v, power_w
        ),
    });
    // Earth's magnetic field does NOT affect airflow, but the Sun's wind does.
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  (the Sun's solar wind — a 400 km/s stream of protons — also obeys these laws; Earth's magnetic field deflects it into the auroras)"
        ),
    });
    out
}

/// Infer a shape hint from a part's bounding-box aspect ratio.
/// (Mesh itself doesn't carry its primitive Kind, so we go by dimensions.)
fn infer_shape(p: &super::eval::Part) -> String {
    let bb = p.mesh.bbox();
    let sx = (bb.max.0[0] - bb.min.0[0]).max(1e-3);
    let sy = (bb.max.0[1] - bb.min.0[1]).max(1e-3);
    let sz = (bb.max.0[2] - bb.min.0[2]).max(1e-3);
    let aspect = (sx / sy.min(sz)).max(sy / sx.min(sz)).max(sz / sx.min(sy));
    // approximate the shape class from the material's name and aspect ratio
    let mat_name = p.material.map(|m| m.name).unwrap_or("plastic");
    let _ = mat_name;
    if aspect > 5.0 {
        "rod".into()
    } else if aspect > 3.0 {
        "box".into()
    } else {
        "box".into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reynolds_laminar_vs_turbulent() {
        // honey at 1 m/s in a 1 cm pipe: Re ~ 0.1, very laminar
        let re_honey = reynolds(1420.0, 1.0, 0.01, 10000.0);
        assert!(re_honey < 1.0, "honey flow is laminar: Re = {}", re_honey);
        // air at 10 m/s over a 1 m wing: Re ~ 6.9e5, turbulent
        let re_air = reynolds(RHO_AIR_SL, 10.0, 1.0, MU_AIR);
        assert!(re_air > 1e5, "wing flow is turbulent: Re = {}", re_air);
    }

    #[test]
    fn terminal_velocity_parachute_vs_bowling_ball() {
        // parachute: 80 kg, 30 m² area, C_D 1.5 — v_t = 5.9 m/s
        let vp = terminal_velocity(80.0, G_EARTH, RHO_AIR_SL, 30.0, 1.5);
        assert!(vp < 8.0 && vp > 4.0, "parachute v_t = {}", vp);
        // bowling ball: 7 kg, 0.04 m², C_D 0.47 — v_t = ~77 m/s
        let vb = terminal_velocity(7.0, G_EARTH, RHO_AIR_SL, 0.04, 0.47);
        assert!(vb > 60.0, "ball v_t = {}", vb);
    }

    #[test]
    fn air_density_drops_with_altitude() {
        let r0 = air_density_at(0.0);
        let r8k = air_density_at(8000.0); // Everest-ish
        assert!(r8k < r0 * 0.55, "Everest air is thin: {}", r8k);
    }

    #[test]
    fn mach_breakdown() {
        assert!(mach(100.0, 340.0) < 1.0, "100 m/s is subsonic");
        assert!(mach(400.0, 340.0) > 1.0, "400 m/s is supersonic");
    }

    #[test]
    fn aero_sim_runs() {
        let w = crate::world::eval::compile("scene \"t\"\nw = cube 4cm at (0, 2cm, 0) material: wood");
        let lines = aero_sim(&w);
        assert!(lines.iter().any(|l| l.text.contains("AERODYNAMIC SURVEY")));
        assert!(lines.iter().any(|l| l.text.contains("drag")));
    }
}
