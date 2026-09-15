//! P2320 — OTD4 ELECTRODYNAMICS — currents, fields, and circuits in time.
//!
//! Magnetism is the steady state — a permanent bar magnet is unchanging.
//! Electrodynamics is what happens when currents and fields *change*:
//! capacitors charge, inductors build magnetic fields, RC circuits decay
//! with their own characteristic time, and the whole of modern electronics
//! follows.
//!
//! Laws carried here:
//!   Ohm's law          — V = I·R (the simplest, the most used)
//!   Power              — P = V·I = I²R = V²/R (heat, light, work)
//!   Kirchhoff (KVL)   — ΣV around a loop = 0 (energy is conserved)
//!   Kirchhoff (KCL)   — ΣI at a node = 0 (charge is conserved)
//!   Capacitance        — Q = C·V; energy U = ½·C·V²
//!   Inductance         — V = L·dI/dt; energy U = ½·L·I²
//!   RC time constant   — τ = R·C (63% charge in one τ, 95% in three)
//!   RL time constant   — τ = L/R
//!   LC oscillation     — ω = 1/√(LC) (the tuner in every radio)
//!   Maxwell (light)    — c = 1/√(μ₀ε₀) (light IS electromagnetism)

use super::eval::{ConsoleLine, LineKind, World};
use super::magnetism::MU_0;

/// Vacuum permittivity, F/m. ε₀ = 1/(μ₀c²).
pub const EPSILON_0: f64 = 8.854_187_817e-12;
/// Speed of light in vacuum (m/s) — derived from Maxwell: c = 1/√(μ₀ε₀).
pub const C_LIGHT: f64 = 299_792_458.0;
/// Elementary charge, C.
pub const E_CHARGE: f64 = 1.602_176_634e-19;
/// Boltzmann constant, J/K (thermal noise).
pub const K_B: f64 = 1.380_649e-23;

/// Ohm's law: V = I·R
pub fn ohm_v(i: f64, r: f64) -> f64 {
    i * r
}
pub fn ohm_i(v: f64, r: f64) -> f64 {
    if r.abs() < 1e-12 {
        return f64::INFINITY;
    }
    v / r
}
pub fn ohm_r(v: f64, i: f64) -> f64 {
    if i.abs() < 1e-12 {
        return f64::INFINITY;
    }
    v / i
}

/// Electric power (W): P = V·I = I²R = V²/R
pub fn power_vi(v: f64, i: f64) -> f64 {
    v * i
}
pub fn power_ir(i: f64, r: f64) -> f64 {
    i * i * r
}

/// Capacitor energy (J): U = ½·C·V²
pub fn capacitor_energy(c_f: f64, v: f64) -> f64 {
    0.5 * c_f * v * v
}

/// Inductor energy (J): U = ½·L·I²
pub fn inductor_energy(l_h: f64, i: f64) -> f64 {
    0.5 * l_h * i * i
}

/// RC time constant (s): τ = R·C
pub fn rc_tau(r: f64, c: f64) -> f64 {
    r * c
}

/// RL time constant (s): τ = L/R
pub fn rl_tau(r: f64, l: f64) -> f64 {
    if r.abs() < 1e-12 {
        return f64::INFINITY;
    }
    l / r
}

/// LC resonance angular frequency (rad/s): ω = 1/√(LC)
pub fn lc_omega(l: f64, c: f64) -> f64 {
    if l <= 0.0 || c <= 0.0 {
        return 0.0;
    }
    1.0 / (l * c).sqrt()
}

/// Maxwell's prediction: c = 1/√(μ₀ε₀). Returns m/s.
pub fn speed_of_light_from_maxwell() -> f64 {
    1.0 / (MU_0 * EPSILON_0).sqrt()
}

/// Resistivity of common conductors at 20 °C, Ω·m.
pub fn resistivity(name: &str) -> f64 {
    match name {
        "silver" => 1.59e-8,
        "copper" => 1.68e-8,
        "gold" => 2.44e-8,
        "aluminum" => 2.65e-8,
        "tungsten" => 5.6e-8,
        "zinc" => 5.90e-8,
        "brass" => 6.4e-8,
        "iron" => 9.71e-8,
        "platinum" => 1.06e-7,
        "lead" => 2.06e-7,
        "titanium" => 4.2e-7,
        "stainless" => 6.9e-7,
        "mercury" => 9.6e-7,
        "carbon" => 3.5e-5,
        _ => 1.0e-6,
    }
}

/// Resistance of a wire: R = ρ·L/A (Ω)
pub fn wire_resistance(rho: f64, length_m: f64, area_m2: f64) -> f64 {
    if area_m2 <= 0.0 {
        return f64::INFINITY;
    }
    rho * length_m / area_m2
}

/// `simulate: electro` — the circuit & material electrodynamics survey.
pub fn electro_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "ELECTRODYNAMICS SURVEY — currents, fields, and circuits in time (Ohm, Kirchhoff, Maxwell)".into(),
    });

    // the Maxwell identity — light IS electromagnetism
    let c_derived = speed_of_light_from_maxwell();
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  Maxwell's identity: c = 1/√(μ₀ε₀) = {:.0} m/s — light is electromagnetism (μ₀ = {:.3e}, ε₀ = {:.3e})",
            c_derived, MU_0, EPSILON_0
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
        let mat = p.material.map(|m| m.name).unwrap_or("plastic");
        let cond = p.material.and_then(|m| m.conductive);
        let mass_kg = p.mass_g / 1000.0;
        let bb = p.mesh.bbox();
        let length_m = (bb.max.0[0] - bb.min.0[0]).max(1.0) * 1e-3;
        let w_m = (bb.max.0[1] - bb.min.0[1]).max(1.0) * 1e-3;
        let d_m = (bb.max.0[2] - bb.min.0[2]).max(1.0) * 1e-3;
        let cross_section_m2 = w_m * d_m;

        if let Some(c_sm) = cond {
            // conductivity in MS/m → resistivity in Ω·m: ρ = 1/(σ × 1e6)
            // (copper: 59.6 MS/m → 1.68e-8 Ω·m ✓)
            let rho = 1.0e-6 / c_sm;
            let r = wire_resistance(rho, length_m, cross_section_m2);
            // a household 12 V across this length
            let i_12v = ohm_i(12.0, r);
            let p_w = power_vi(12.0, i_12v);
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!(
                    "  {} [{}] — conductor ({:.1} MS/m), as a {:.1} cm wire cross-section {:.0} mm² ⇒ R = {:.3e} Ω",
                    p.name, mat, c_sm, length_m * 100.0, cross_section_m2 * 1e6, r
                ),
            });
            out.push(ConsoleLine {
                kind: LineKind::Answer,
                text: format!(
                    "    Ohm's law: 12 V across it ⇒ I = {:.2} A, P = {:.1} W — {}",
                    i_12v,
                    p_w,
                    if p_w > 100.0 {
                        "that melts it (use a thicker wire or lower voltage)"
                    } else if p_w > 5.0 {
                        "warm to the touch — a heater wire"
                    } else {
                        "barely noticeable — a signal wire"
                    }
                ),
            });
            // time constant with 1 µH inductance and 1 µF capacitance
            let tau_rc = rc_tau(r, 1e-6);
            let omega_lc = lc_omega(1e-6, 1e-6);
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!(
                    "    with 1 µF across it: τ_RC = R·C = {:.2e} s; with 1 µH in series: ω_LC = 1/√(LC) = {:.0} rad/s ({} Hz) — the tuner's law",
                    tau_rc,
                    omega_lc,
                    omega_lc / (2.0 * std::f64::consts::PI)
                ),
            });
        } else {
            // insulator — dielectric. Estimate capacitance as parallel-plate.
            let eps_r = dielectric_constant(mat);
            let c_pp = EPSILON_0 * eps_r * cross_section_m2 / length_m.max(1e-6);
            let e_max = capacitor_energy(c_pp, 12.0);
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!(
                    "  {} [{}] — insulator (ε_r = {:.1}); as a {:.1} cm parallel-plate capacitor (area {:.0} mm², gap {:.1} cm) ⇒ C = {:.2e} F",
                    p.name, mat, eps_r, length_m * 100.0, cross_section_m2 * 1e6, length_m * 100.0, c_pp
                ),
            });
            out.push(ConsoleLine {
                kind: LineKind::Answer,
                text: format!(
                    "    at 12 V: U = ½CV² = {:.2e} J; charge Q = CV = {:.2e} C (= {:.1e} electrons)",
                    e_max, c_pp * 12.0, c_pp * 12.0 / E_CHARGE
                ),
            });
            // RC discharge through 1 MΩ — the bleeder resistor
            let tau = rc_tau(1.0e6, c_pp);
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!(
                    "    discharging through 1 MΩ: τ = R·C = {:.2e} s — voltage falls to 37% in one τ, 5% in three",
                    tau
                ),
            });
        }
        // thermal noise (Johnson–Nyquist) at the scene temperature
        let t = world.temp_c + 273.15;
        let bw = 1.0e6; // 1 MHz bandwidth
        let v_noise = (4.0 * K_B * t * 1.0e3 * bw).sqrt(); // 1 kΩ source
        let _ = mass_kg;
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!(
                "    Johnson noise at {:.0} K, 1 kΩ, 1 MHz bw: {:.1} µV — the floor every amplifier hears",
                t, v_noise * 1e6
            ),
        });
    }

    // The KVL/KCL teaching line
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "---- KIRCHHOFF (the two conservation laws of every circuit) ----".into(),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  KVL: ΣV around any loop = 0 (energy is conserved — the battery's push equals the resistors' drop). KCL: ΣI at any node = 0 (charge is conserved — what flows in, flows out)."
        ),
    });
    out
}

/// Dielectric constant (relative permittivity) for OTD insulators.
pub fn dielectric_constant(name: &str) -> f64 {
    match name {
        "vacuum" | "air" | "hydrogen" | "helium" | "nitrogen" | "oxygen" => 1.0,
        "water" => 80.1,  // the famous one — why microwaves heat it
        "ice" => 3.5,
        "glass" => 5.5,
        "ceramic" => 100.0, // barium titanate class
        "plastic" => 3.0,
        "rubber" => 2.5,
        "oil" => 2.2,
        "ethanol" => 24.5,
        "acetone" => 20.7,
        "glycerin" => 42.5,
        "marble" => 8.0,
        "concrete" => 6.0,
        "wood" | "oak" | "pine" | "teak" => 4.0,
        "fabric" => 2.0,
        "foam" => 1.05,
        "carbon" => 12.0, // graphite along basal plane
        _ => 3.0,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ohms_law_round_trips() {
        assert!((ohm_v(2.0, 5.0) - 10.0).abs() < 1e-9);
        assert!((ohm_i(10.0, 5.0) - 2.0).abs() < 1e-9);
        assert!((ohm_r(10.0, 2.0) - 5.0).abs() < 1e-9);
    }

    #[test]
    fn power_formulas() {
        assert!((power_vi(10.0, 2.0) - 20.0).abs() < 1e-9);
        assert!((power_ir(2.0, 5.0) - 20.0).abs() < 1e-9);
    }

    #[test]
    fn rc_tau_one_kohm_one_uf() {
        let tau = rc_tau(1.0e3, 1.0e-6);
        assert!((tau - 1.0e-3).abs() < 1e-12, "1kΩ × 1µF = 1 ms");
    }

    #[test]
    fn lc_omega_1h_1f() {
        let omega = lc_omega(1.0, 1.0);
        assert!((omega - 1.0).abs() < 1e-9);
    }

    #[test]
    fn maxwell_speed_of_light() {
        let c = speed_of_light_from_maxwell();
        assert!((c - C_LIGHT).abs() / C_LIGHT < 1e-6, "c = {}", c);
    }

    #[test]
    fn copper_resistance_for_1m_1mm2_wire() {
        let r = wire_resistance(resistivity("copper"), 1.0, 1e-6);
        assert!((r - 0.0168).abs() / 0.0168 < 0.01, "1 m of 1 mm² copper = 16.8 mΩ: {}", r);
    }

    #[test]
    fn capacitor_energy_1f_1v() {
        let u = capacitor_energy(1.0, 1.0);
        assert!((u - 0.5).abs() < 1e-9);
    }

    #[test]
    fn electro_sim_runs() {
        let w = crate::world::eval::compile("scene \"t\"\nw = cylinder 2cm, height: 10cm at (0, 5cm, 0) material: copper");
        let lines = electro_sim(&w);
        assert!(lines.iter().any(|l| l.text.contains("ELECTRODYNAMICS SURVEY")));
        assert!(lines.iter().any(|l| l.text.contains("Ohm")));
    }
}
