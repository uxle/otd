//! P2340 — OTD4 RIGID BODY DYNAMICS — rotation, inertia, gyroscopes.
//!
//! A rigid body keeps its shape — its particles don't move relative to each
//! other. It can still translate and rotate, and rotation is where the
//! interesting math lives. The inertia tensor, the angular momentum, the
//! gyroscopic effect, Euler's equations for a tumbling body — every satellite
//! and every spinning top answers to them.
//!
//! Laws carried here:
//!   Moment of inertia   — I = Σ m·r² (resistance to angular acceleration)
//!   Angular momentum    — L = I·ω (conserved without external torque)
//!   Torque              — τ = r × F (the lever's best friend)
//!   Rotational energy   — E = ½·I·ω²
//!   Parallel axis       — I = I_cm + M·d² (steiner's shift)
//!   Precession          — Ω_p = τ/(I·ω) (the gyroscope's slow drift)
//!   Euler's equation    — I₁ω̇₁ + (I₃−I₂)ω₂ω₃ = τ₁ (a tumbling body)
//!   Tennis racket       — rotation about the middle-inertia axis is unstable

use super::eval::{ConsoleLine, LineKind, Part, World};
use crate::math3::V3;
use crate::units::G_EARTH;

/// Moment of inertia (kg·m²) for common uniform shapes about the natural axis.
#[allow(dead_code)]
pub enum Shape {
    SolidSphere,
    HollowSphere,
    SolidCylinder, // about long axis
    HollowCylinder,
    Rod,       // about center, perpendicular to length
    Slab,      // about center, perpendicular to face
    Block,     // about an axis through the center, perpendicular to a face
}

/// I for a uniform shape. Mass m (kg), characteristic size r1 (m), and r2 (m)
/// (for rectangular shapes: r1 = full extent along axis 1, r2 = full extent
/// along axis 2).
pub fn inertia(s: Shape, m: f64, r1: f64, r2: f64) -> f64 {
    match s {
        Shape::SolidSphere => 0.4 * m * r1 * r1,                  // 2/5 m r²
        Shape::HollowSphere => (2.0 / 3.0) * m * r1 * r1,         // 2/3 m r²
        Shape::SolidCylinder => 0.5 * m * r1 * r1,                // 1/2 m r²
        Shape::HollowCylinder => m * r1 * r1,                     // m r²
        Shape::Rod => (1.0 / 12.0) * m * r1 * r1,                 // 1/12 m L²
        Shape::Slab => (1.0 / 12.0) * m * (r1 * r1 + r2 * r2),    // 1/12 m (a² + b²)
        Shape::Block => (1.0 / 12.0) * m * (r1 * r1 + r2 * r2),
    }
}

/// Parallel axis theorem: I = I_cm + M·d²
pub fn parallel_axis(i_cm: f64, m: f64, d_m: f64) -> f64 {
    i_cm + m * d_m * d_m
}

/// Angular momentum L = I·ω (kg·m²/s)
pub fn angular_momentum(i: f64, omega_rad_s: f64) -> f64 {
    i * omega_rad_s
}

/// Rotational kinetic energy E = ½·I·ω² (J)
pub fn rotational_energy(i: f64, omega: f64) -> f64 {
    0.5 * i * omega * omega
}

/// Torque τ = r × F (N·m) — magnitude when r and F are perpendicular
pub fn torque(r_m: f64, f_n: f64, sin_theta: f64) -> f64 {
    r_m * f_n * sin_theta
}

/// Gyroscopic precession rate (rad/s): Ω_p = τ / (I·ω)
/// A spinning top precesses slowly because I·ω is huge.
pub fn precession(tau: f64, i: f64, omega: f64) -> f64 {
    if i * omega == 0.0 {
        return f64::INFINITY;
    }
    tau / (i * omega)
}

/// Angular momentum vector from an inertia tensor and angular velocity.
pub fn angular_momentum_vec(i_tensor: [[f64; 3]; 3], omega: V3) -> V3 {
    let mut out = [0.0f64; 3];
    for r in 0..3 {
        for c in 0..3 {
            out[r] += i_tensor[r][c] * omega.0[c];
        }
    }
    V3::new(out[0], out[1], out[2])
}

/// Estimate the inertia tensor (kg·m²) of a part from its bounding box and mass.
/// Treats it as a uniform rectangular block.
pub fn inertia_tensor_of_part(part: &Part) -> [[f64; 3]; 3] {
    let bb = part.mesh.bbox();
    let sx = (bb.max.0[0] - bb.min.0[0]).max(1e-6) * 1e-3; // m
    let sy = (bb.max.0[1] - bb.min.0[1]).max(1e-6) * 1e-3;
    let sz = (bb.max.0[2] - bb.min.0[2]).max(1e-6) * 1e-3;
    let m = part.mass_g / 1000.0;
    // I_xx = (1/12) m (sy² + sz²), etc.
    let ixx = m * (sy * sy + sz * sz) / 12.0;
    let iyy = m * (sx * sx + sz * sz) / 12.0;
    let izz = m * (sx * sx + sy * sy) / 12.0;
    [
        [ixx, 0.0, 0.0],
        [0.0, iyy, 0.0],
        [0.0, 0.0, izz],
    ]
}

/// `simulate: rigid` — the rigid-body survey of the scene.
pub fn rigid_sim(world: &World) -> Vec<ConsoleLine> {
    rigid_sim_spin(world, 1.0)
}

/// `simulate: rigid` with a notional angular speed (rad/s) applied to every part.
pub fn rigid_sim_spin(world: &World, omega: f64) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!(
            "RIGID BODY DYNAMICS SURVEY — moment of inertia, angular momentum, gyroscopes (ω = {:.2} rad/s = {:.0} rpm)",
            omega, omega * 60.0 / (2.0 * std::f64::consts::PI)
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

    let mut total_i = 0.0;
    let mut total_l = 0.0;
    let mut total_ke = 0.0;
    for &i in &visible {
        let p = &world.parts[i];
        let mat = p.material.map(|m| m.name).unwrap_or("plastic");
        let bb = p.mesh.bbox();
        let sx = (bb.max.0[0] - bb.min.0[0]).max(1e-6) * 1e-3;
        let sy = (bb.max.0[1] - bb.min.0[1]).max(1e-6) * 1e-3;
        let sz = (bb.max.0[2] - bb.min.0[2]).max(1e-6) * 1e-3;
        let m = p.mass_g / 1000.0;
        // dominant axis: the longest dimension
        let (i_principal, axis_name, _r1, _r2) = if sx >= sy && sx >= sz {
            (m * (sy * sy + sz * sz) / 12.0, 'x', sy, sz)
        } else if sy >= sx && sy >= sz {
            (m * (sx * sx + sz * sz) / 12.0, 'y', sx, sz)
        } else {
            (m * (sx * sx + sy * sy) / 12.0, 'z', sx, sy)
        };
        let l = angular_momentum(i_principal, omega);
        let ke = rotational_energy(i_principal, omega);
        total_i += i_principal;
        total_l += l;
        total_ke += ke;
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!(
                "  {} [{}] {:.3} kg, {:.1}×{:.1}×{:.1} cm ⇒ principal I_{} = {:.4e} kg·m² (about the long axis)",
                p.name, mat, m, sx * 100.0, sy * 100.0, sz * 100.0, axis_name, i_principal
            ),
        });
        out.push(ConsoleLine {
            kind: LineKind::Answer,
            text: format!(
                "    L = I·ω = {:.4e} kg·m²/s, K_rot = ½Iω² = {:.3} J — {}",
                l, ke,
                if i_principal > 0.01 {
                    "a heavy flywheel (smooths out the spin)"
                } else if i_principal < 1e-6 {
                    "tiny — barely holds angular momentum"
                } else {
                    "ordinary spinning part"
                }
            ),
        });
        // gravity torque on this part if it's off-centre
        let com = world.stats.com.unwrap_or(V3::ZERO);
        let c = p.centroid.unwrap_or(p.mesh.bbox().center());
        let arm = V3::new(c.x() / 1000.0, 0.0, c.z() / 1000.0).sub(&V3::new(com.x() / 1000.0, 0.0, com.z() / 1000.0));
        let arm_m = arm.len();
        let weight = m * G_EARTH;
        let tau = torque(arm_m, weight, 1.0);
        let omega_p = precession(tau, i_principal, omega);
        if omega > 0.01 && tau > 1e-6 {
            out.push(ConsoleLine {
                kind: LineKind::Info,
                text: format!(
                    "    gyroscope: gravity-arm {:.2} m × weight {:.2} N = τ {:.3e} N·m ⇒ precesses at Ω_p = τ/(Iω) = {:.3e} rad/s (a top traces a slow circle while it spins)",
                    arm_m, weight, tau, omega_p
                ),
            });
        }
    }

    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: format!(
            "scene totals: I = {:.4e} kg·m², L = {:.4e} kg·m²/s, K_rot = {:.3} J",
            total_i, total_l, total_ke
        ),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  conservation: with no external torque, L stays constant — a figure skater pulling arms in spins faster (I↓ ⇒ ω↑ to keep Iω constant). The Earth–Moon system does this on a 10⁹-year timescale (the day lengthens by 2 ms/century)."
        ),
    });

    // Tennis racket theorem teaching
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "---- THE TENNIS RACKET THEOREM ----".into(),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: format!(
            "  rotation about the axis with the MIDDLE moment of inertia is unstable — the body tumbles. Spinning about the largest or smallest is stable. This is why a tossed racket flips unexpectedly."
        ),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn solid_sphere_2_5_mr2() {
        let i = inertia(Shape::SolidSphere, 1.0, 0.1, 0.0);
        assert!((i - 0.4 * 0.01).abs() < 1e-9);
    }

    #[test]
    fn parallel_axis_for_rod_at_end() {
        let i_cm = inertia(Shape::Rod, 1.0, 1.0, 0.0); // 1/12
        let i_end = parallel_axis(i_cm, 1.0, 0.5); // 1/12 + 1/4 = 1/3
        assert!((i_end - 1.0 / 3.0).abs() < 1e-9, "I_end = {}", i_end);
    }

    #[test]
    fn angular_momentum_conservation() {
        // figure skater: I = 4 kg·m², ω = 6 rad/s ⇒ L = 24
        // arms in: I = 1, ω' = 24 (L conserved)
        let l1 = angular_momentum(4.0, 6.0);
        let omega2 = l1 / 1.0;
        assert!((omega2 - 24.0).abs() < 1e-9, "ω' = {}", omega2);
    }

    #[test]
    fn gyroscope_precession_earth() {
        // Earth: I ~ 8e37 kg m², ω = 2π/86400 rad/s, τ from Sun ~ 5.5e22 N·m
        let i = 8.0e37;
        let omega = 2.0 * std::f64::consts::PI / 86400.0;
        let tau = 5.5e22;
        let omega_p = precession(tau, i, omega);
        let period_yr = 2.0 * std::f64::consts::PI / omega_p / (365.25 * 86400.0);
        // expected ~ 26,000 yr
        assert!(period_yr > 20_000.0 && period_yr < 30_000.0, "precession period: {} yr", period_yr);
    }

    #[test]
    fn rigid_sim_runs() {
        let w = crate::world::eval::compile("scene \"t\"\nw = cylinder 2cm, height: 10cm at (0, 5cm, 0) material: steel");
        let lines = rigid_sim(&w);
        assert!(lines.iter().any(|l| l.text.contains("RIGID BODY DYNAMICS SURVEY")));
        assert!(lines.iter().any(|l| l.text.contains("principal")));
    }
}
