//! P2310 — OTD4 FLUID DYNAMICS — how liquids and gases actually move.
//!
//! Aerodynamics is the body's story; fluid dynamics is the fluid's. Where
//! Bernoulli flies wings, Poiseuille pushes blood through capillaries, and
//! Navier–Stokes is the law the whole weather obeys.
//!
//! Laws carried here:
//!   Continuity         — A₁v₁ = A₂v₂ (incompressible; mass in = mass out)
//!   Bernoulli          — P + ½ρv² + ρgh = const along a streamline
//!   Hydrostatic        — P = P₀ + ρgh (the weight of the column above)
//!   Poiseuille         — Q = π·r⁴·ΔP / (8·μ·L) (laminar pipe flow, r⁴ law)
//!   Stokes drag        — F = 6π·μ·r·v (slow sphere through honey)
//!   Reynolds           — Re = ρvL/μ (the laminar/turbulent switch)
//!   Viscosity          — τ = μ·dv/dy (the friction inside the fluid)
//!   Surface tension    — γ (water = 72 mN/m; the raindrop is round because of it)

use super::eval::{ConsoleLine, LineKind, World};
use crate::units::G_EARTH;

/// Dynamic viscosity, Pa·s (kg/(m·s)).
pub fn viscosity(name: &str) -> f64 {
    match name {
        "water" => 1.0e-3,
        "oil" => 8.0e-2,
        "mercury" => 1.526e-3,
        "ethanol" => 1.074e-3,
        "acetone" => 3.06e-4,
        "glycerin" => 1.412, // famous — 1400× water
        "gasoline" => 2.9e-4,
        "air" => 1.81e-5,
        "hydrogen" => 8.8e-6,
        "helium" => 1.96e-5,
        "steam" => 1.34e-5,
        _ => 1.0e-3,
    }
}

/// Density in kg/m³ (defers to the materials table, with sensible fallback).
pub fn density(name: &str) -> f64 {
    super::materials::find(name).map(|m| m.density).unwrap_or(match name {
        "water" => 997.0,
        "air" => 1.225,
        _ => 1000.0,
    })
}

/// Surface tension, N/m (J/m²).
pub fn surface_tension(name: &str) -> f64 {
    match name {
        "water" => 0.0728,
        "mercury" => 0.485, // the heavy non-wetter
        "ethanol" => 0.0223,
        "oil" => 0.032,
        "acetone" => 0.0237,
        "glycerin" => 0.064,
        _ => 0.030,
    }
}

/// Hydrostatic pressure at depth h (Pa): P = P₀ + ρgh.
pub fn hydrostatic(rho: f64, g: f64, h_m: f64) -> f64 {
    rho * g * h_m
}

/// Bernoulli's total head: P + ½ρv² + ρgh.
pub fn bernoulli_total(p_pa: f64, rho: f64, v: f64, g: f64, h_m: f64) -> f64 {
    p_pa + 0.5 * rho * v * v + rho * g * h_m
}

/// Pressure from velocity (Bernoulli, ignoring gravity): ΔP = ½ρ(v₁² − v₂²).
pub fn dynamic_pressure(rho: f64, v1: f64, v2: f64) -> f64 {
    0.5 * rho * (v1 * v1 - v2 * v2)
}

/// Poiseuille flow: Q = π·r⁴·ΔP / (8·μ·L)  (m³/s, laminar pipe).
pub fn poiseuille(r_m: f64, dp_pa: f64, mu: f64, l_m: f64) -> f64 {
    if mu <= 0.0 || l_m <= 0.0 {
        return f64::INFINITY;
    }
    std::f64::consts::PI * r_m.powi(4) * dp_pa / (8.0 * mu * l_m)
}

/// Stokes drag on a slow sphere: F = 6π·μ·r·v (N). Valid for Re < 0.1.
pub fn stokes_drag(mu: f64, r_m: f64, v: f64) -> f64 {
    6.0 * std::f64::consts::PI * mu * r_m * v
}

/// Reynolds number.
pub fn reynolds(rho: f64, v: f64, length: f64, mu: f64) -> f64 {
    if mu <= 0.0 {
        return f64::INFINITY;
    }
    rho * v * length / mu
}

/// Capillary rise h = 2γ·cos(θ) / (ρ·g·r) — water climbs 14 mm in a 1 mm tube.
pub fn capillary_rise(gamma: f64, theta_cos: f64, rho: f64, g: f64, r_m: f64) -> f64 {
    if rho <= 0.0 || g <= 0.0 || r_m <= 0.0 {
        return 0.0;
    }
    2.0 * gamma * theta_cos / (rho * g * r_m)
}

/// `simulate: fluid` — the fluid-dynamics survey of the scene's liquids.
pub fn fluid_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "FLUID DYNAMICS SURVEY — how liquids and gases move (continuity, Bernoulli, Poiseuille, Stokes)".into(),
    });

    let mut any_liquid = false;
    for p in &world.parts {
        if p.hidden {
            continue;
        }
        let mat = p.material.map(|m| m.name).unwrap_or("plastic");
        // Determine state — gas/liquid/solid — to know whether to apply fluid dynamics
        let st = match p.material {
            Some(m) => super::materials::state(m),
            None => super::materials::State::Solid,
        };
        if st != super::materials::State::Liquid && st != super::materials::State::Gas {
            continue;
        }
        any_liquid = true;
        let rho = density(mat);
        let mu = viscosity(mat);
        let bb = p.mesh.bbox();
        let h_m = (bb.max.0[1] - bb.min.0[1]).max(1.0) * 1e-3;
        let size_x = (bb.max.0[0] - bb.min.0[0]).max(1.0) * 1e-3;
        let size_z = (bb.max.0[2] - bb.min.0[2]).max(1.0) * 1e-3;
        // hydrostatic pressure at the bottom of this column
        let p_bot = hydrostatic(rho, G_EARTH, h_m);
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!(
                "  {} [{}] — ρ {:.0} kg/m³, μ {:.2e} Pa·s, depth {:.2} m, bottom pressure +{:.0} Pa ({:.3} atm) above the top",
                p.name, mat, rho, mu, h_m, p_bot, p_bot / 101325.0
            ),
        });
        // capillary rise for this liquid in a 1 mm tube
        let gamma = surface_tension(mat);
        let h_cap = capillary_rise(gamma, 1.0, rho, G_EARTH, 0.5e-3);
        if h_cap.abs() > 1e-6 {
            out.push(ConsoleLine {
                kind: LineKind::Answer,
                text: format!(
                    "    surface tension γ = {:.3} N/m ⇒ climbs {:.1} mm in a 1 mm tube (the smaller the tube, the higher the climb)",
                    gamma, h_cap * 1000.0
                ),
            });
        }
        // a notional flow speed through a 1 cm pipe at 1 m/s
        let v = 1.0;
        let re = reynolds(rho, v, size_x.min(size_z).max(0.001), mu);
        let regime = if re < 2300.0 {
            "laminar (smooth layers, Poiseuille's law)"
        } else if re < 1.0e5 {
            "transitional"
        } else {
            "turbulent (eddies, mixing, weather)"
        };
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!(
                "    flow at 1 m/s through {:.1} cm ⇒ Re = {:.2e} ⇒ {}",
                size_x.min(size_z) * 100.0,
                re,
                regime
            ),
        });
        // Poiseuille through the part's smallest dimension as a notional pipe
        let r_pipe = size_x.min(size_z).max(0.001) / 2.0;
        let dp = 1000.0; // 1 kPa pressure drop
        let q = poiseuille(r_pipe, dp, mu, h_m);
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!(
                "    Poiseuille through a {:.1} mm radius, {:.2} m long pipe at 1 kPa: Q = {:.2e} m³/s (the r⁴ law — halve the radius, flow drops ×16)",
                r_pipe * 1000.0,
                h_m,
                q
            ),
        });
    }

    if !any_liquid {
        out.push(ConsoleLine {
            kind: LineKind::Warn,
            text: "  no liquids or gases in the scene — fluid dynamics needs a body of water/oil/mercury/etc. (try `material: water`)".into(),
        });
    }

    // Bernoulli teaching: the hose-with-thumb effect
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "---- BERNOULLI (the hose-with-thumb effect) ----".into(),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  P + ½ρv² + ρgh = const — narrow the outlet (continuity: A₁v₁ = A₂v₂), the water speeds up, the pressure drops: this is how wings lift, perfume atomisers spray, and a ping-pong ball balances in a hair-drier blast"
        ),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poiseuille_r4_law() {
        let q1 = poiseuille(0.01, 1000.0, 1e-3, 1.0);
        let q2 = poiseuille(0.005, 1000.0, 1e-3, 1.0);
        assert!((q1 / q2 - 16.0).abs() < 1e-6, "halve radius, flow ×16: q1={}", q1 / q2);
    }

    #[test]
    fn stokes_for_honey_and_water() {
        let f_w = stokes_drag(1e-3, 0.001, 0.01); // 1 mm sphere at 1 cm/s in water
        let f_h = stokes_drag(1.412, 0.001, 0.01); // glycerin
        assert!(f_h > f_w * 1000.0, "honey drags 1400× water: {}", f_h / f_w);
    }

    #[test]
    fn hydrostatic_10m_water_is_1atm() {
        let p = hydrostatic(997.0, G_EARTH, 10.0);
        assert!((p - 97800.0).abs() / 97800.0 < 0.01, "10 m of water ~ 1 atm: {}", p);
    }

    #[test]
    fn capillary_water_in_1mm_tube() {
        // water in a 1 mm DIAMETER tube (r = 0.5 mm): h = 2γ/(ρgr) ≈ 29.8 mm
        // (the famous "10 m = 1 atm" hydrostatic and "water climbs in narrow tubes")
        let h = capillary_rise(0.0728, 1.0, 997.0, G_EARTH, 0.5e-3);
        assert!(h > 0.025 && h < 0.035, "water rises ~30 mm in a 1 mm tube: {}", h);
    }

    #[test]
    fn fluid_sim_runs() {
        let w = crate::world::eval::compile("scene \"t\"\np = cube 4cm at (0, 2cm, 0) material: water");
        let lines = fluid_sim(&w);
        assert!(lines.iter().any(|l| l.text.contains("FLUID DYNAMICS SURVEY")));
        assert!(lines.iter().any(|l| l.text.contains("Poiseuille")));
    }
}
