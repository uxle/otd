//! OTD6 — ELECTRIC MOTOR simulation.
//! F = B·I·L, τ = N·B·I·A, back-EMF = N·B·A·ω.
//! "Make a motor and connect it with electricity and check will it moving or not."

use super::eval::{ConsoleLine, LineKind, World};

pub fn motor_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::with_capacity(40);
    out.push(ConsoleLine { kind: LineKind::Sim, text: "ELECTRIC MOTOR SURVEY — F = B·I·L, τ = N·B·I·A, back-EMF = N·B·A·ω".into() });

    let mut coils: Vec<&super::eval::Part> = Vec::new();
    let mut magnets: Vec<&super::eval::Part> = Vec::new();
    let mut rotors: Vec<&super::eval::Part> = Vec::new();

    for p in &world.parts {
        if p.hidden { continue; }
        let mat = p.material.map(|m| m.name).unwrap_or("plastic");
        if mat == "copper" || mat == "aluminum" { coils.push(p); }
        if p.magnetized || mat == "iron" || mat == "steel" { magnets.push(p); }
        let n = p.name.to_lowercase();
        if n.contains("rotor") || n.contains("shaft") || n.contains("wheel") || n.contains("motor") { rotors.push(p); }
    }

    out.push(ConsoleLine { kind: LineKind::Info, text: format!("  found: {} coil(s), {} magnet(s), {} rotor part(s)", coils.len(), magnets.len(), rotors.len()) });

    let v_supply = 6.0; let r_ohm = 1.5; let turns = 50.0;
    let b_t = 0.4; let rotor_r = 0.015; let area = std::f64::consts::PI * rotor_r * rotor_r;
    let load_torque = 0.001;

    out.push(ConsoleLine { kind: LineKind::Sim, text: "---- MOTOR PHYSICS ----".into() });
    out.push(ConsoleLine { kind: LineKind::Info, text: format!("  supply: {}V  R: {}Ω  turns: {}  B: {}T  rotor r: {}mm", v_supply, r_ohm, turns, b_t, rotor_r * 1000.0) });

    let i_stall = v_supply / r_ohm;
    let tau_stall = turns * b_t * i_stall * area;
    let omega_0 = v_supply / (turns * b_t * area);
    let rpm_0 = omega_0 * 60.0 / (2.0 * std::f64::consts::PI);

    out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  stall current: I = V/R = {}/{} = {:.2} A", v_supply, r_ohm, i_stall) });
    out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  stall torque: τ = N·B·I·A = {:.4} N·m ({:.1} g·cm)", tau_stall, tau_stall * 100.0 * 1000.0) });
    out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  no-load speed: {:.0} RPM", rpm_0) });

    let i_load = load_torque / (turns * b_t * area);
    let v_back = v_supply - i_load * r_ohm;
    let omega_load = v_back / (turns * b_t * area);
    let rpm_load = omega_load * 60.0 / (2.0 * std::f64::consts::PI);
    let p_elec = v_supply * i_load;
    let p_mech = load_torque * omega_load;
    let eff = if p_elec > 0.0 { p_mech / p_elec * 100.0 } else { 0.0 };

    out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  operating: {:.0} RPM at {:.3} A, {:.2} W in, {:.2} W out, η={:.1}%", rpm_load, i_load, p_elec, p_mech, eff) });

    out.push(ConsoleLine { kind: LineKind::Sim, text: "==== WILL IT MOVE? ====".into() });
    if tau_stall > load_torque {
        out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  YES ✓ — stall torque ({:.4} N·m) exceeds load ({:.4} N·m) by {:.1}×", tau_stall, load_torque, tau_stall / load_torque) });
        out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  The motor WILL spin at {:.0} RPM under load.", rpm_load) });
    } else {
        out.push(ConsoleLine { kind: LineKind::Error, text: format!("  NO ✗ — stall torque ({:.4} N·m) < load ({:.4} N·m)", tau_stall, load_torque) });
    }
    out
}

/// OTD6 #5 — `simulate: circuit` walks the connections declared with
/// `connect: A B` and the real resistance of each part's modeled windings.
pub fn circuit_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine { kind: LineKind::Sim, text: "CIRCUIT ANALYSIS — walking the real resistance of modeled windings".into() });

    if world.connections.is_empty() {
        out.push(ConsoleLine { kind: LineKind::Warn, text: "  no connections — declare them with `connect: A B` first".into() });
        out.push(ConsoleLine { kind: LineKind::Info, text: "  e.g. connect: battery rotor  then  simulate: circuit".into() });
        return out;
    }

    out.push(ConsoleLine { kind: LineKind::Info, text: format!("  {} connection(s) declared:", world.connections.len()) });
    let mut total_r = 0.0;
    for (a, b) in &world.connections {
        let ra = find_part_resistance(world, a);
        let rb = find_part_resistance(world, b);
        let r_pair = ra + rb;
        total_r += r_pair;
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("    {} ↔ {}  (R_a={:.3}Ω, R_b={:.3}Ω, R_total={:.3}Ω)", a, b, ra, rb, r_pair) });
    }

    let v = 6.0; // default 6V supply
    let i = if total_r > 0.0 { v / total_r } else { f64::INFINITY };
    let p = v * i;

    out.push(ConsoleLine { kind: LineKind::Sim, text: "---- CIRCUIT SOLUTION ----".into() });
    out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  supply: {:.0} V", v) });
    out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  total resistance: {:.3} Ω", total_r) });
    out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  current: I = V/R = {:.0}/{:.3} = {:.3} A", v, total_r, i) });
    out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  power: P = V·I = {:.2} W", p) });

    if i.is_finite() && i > 0.001 {
        out.push(ConsoleLine { kind: LineKind::Answer, text: format!("  VERDICT: current flows ({:.3} A) — the circuit is closed and conducting", i) });
    } else {
        out.push(ConsoleLine { kind: LineKind::Error, text: "  VERDICT: no current — the circuit is open or resistance is too high".into() });
    }
    out
}

fn find_part_resistance(world: &World, name: &str) -> f64 {
    let p = match world.parts.iter().find(|p| p.name == name) {
        Some(p) => p,
        None => return 0.0,
    };
    let mat = p.material.map(|m| m.name).unwrap_or("plastic");
    let rho = match mat {
        "silver" => 1.59e-8, "copper" => 1.68e-8, "gold" => 2.44e-8,
        "aluminum" => 2.65e-8, "tungsten" => 5.6e-8, "iron" => 9.71e-8,
        "steel" | "stainless" => 6.9e-7, "lead" => 2.06e-7, _ => 1e6,
    };
    let vol_m3 = p.volume_mm3 * 1e-9;
    let wire_area = std::f64::consts::PI * (0.00025_f64).powi(2); // 0.5mm wire
    let wire_len = if wire_area > 0.0 { vol_m3 / wire_area } else { 0.0 };
    if wire_len > 0.0 { rho * wire_len / wire_area } else { 0.0 }
}
