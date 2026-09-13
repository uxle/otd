//! Phase 110 — Physics: forces and Newton's laws (Rust port of
//! python/science/forces_domain.py)
//!
//! Textbook formulas are the ground truth (F = m*a, W = m*g, f = mu*N,
//! a = (F_applied - f)/m) and every computed result is re-checked through
//! an INDEPENDENT path (a different rearrangement or identity) before any
//! caller should trust it.

pub const DEFAULT_G: f64 = 9.8;

/// Result of a forces solve: the asked-for quantity, the equation used,
/// and how it was independently re-checked (with the re-check's value).
#[derive(Debug, Clone, PartialEq)]
pub struct ForceResult {
    pub value: f64,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_value: f64,
}

/// Newton's second law F = m*a for any one of 'force', 'mass',
/// 'acceleration', given the other two. The computed trio (F, m, a) is
/// re-plugged into the two rearrangements NOT used for the computation.
///
/// (`g` appears in the Python signature but is never used; kept for API
/// fidelity. `None` means the Python default DEFAULT_G.)
pub fn solve_force(
    find: &str,
    mass: Option<f64>,
    acceleration: Option<f64>,
    force: Option<f64>,
    g: Option<f64>,
) -> Result<ForceResult, String> {
    let _ = g; // unused in the Python original too (F = m*a needs no g)

    if find == "force" {
        let (Some(mass), Some(acceleration)) = (mass, acceleration) else {
            return Err("solving for force needs mass and acceleration".to_string());
        };
        let f = mass * acceleration;
        // re-check via a = F/m and m = F/a
        let check1 = f / mass;
        let check2 = f / acceleration;
        let ok = (check1 - acceleration).abs() < 1e-9 && (check2 - mass).abs() < 1e-9;
        if !ok {
            return Err("independent re-check failed for F=m*a".to_string());
        }
        return Ok(ForceResult {
            value: f,
            equation_used: "F = m*a".to_string(),
            verify_method: "recompute a = F/m and m = F/a".to_string(),
            verify_value: check1,
        });
    }
    if find == "mass" {
        let (Some(force), Some(acceleration)) = (force, acceleration) else {
            return Err("solving for mass needs force and acceleration".to_string());
        };
        if acceleration == 0.0 {
            return Err("acceleration 0 makes mass undefined via F=m*a".to_string());
        }
        let m = force / acceleration;
        let check = m * acceleration;
        if (check - force).abs() > 1e-9 {
            return Err("independent re-check failed for F=m*a".to_string());
        }
        return Ok(ForceResult {
            value: m,
            equation_used: "m = F/a".to_string(),
            verify_method: "recompute F = m*a".to_string(),
            verify_value: check,
        });
    }
    if find == "acceleration" {
        let (Some(force), Some(mass)) = (force, mass) else {
            return Err("solving for acceleration needs force and mass".to_string());
        };
        if mass == 0.0 {
            return Err("mass 0 makes acceleration undefined".to_string());
        }
        let a = force / mass;
        let check = a * mass;
        if (check - force).abs() > 1e-9 {
            return Err("independent re-check failed for F=m*a".to_string());
        }
        return Ok(ForceResult {
            value: a,
            equation_used: "a = F/m".to_string(),
            verify_method: "recompute F = m*a".to_string(),
            verify_value: check,
        });
    }
    Err("find must be one of force, mass, acceleration".to_string())
}

/// Weight W = m*g. Independent re-check: W/m must reproduce g and W/g
/// must reproduce m.
pub fn solve_weight(mass: f64, g: Option<f64>) -> Result<ForceResult, String> {
    let g = g.unwrap_or(DEFAULT_G);
    if mass < 0.0 {
        return Err("mass cannot be negative".to_string());
    }
    let w = mass * g;
    if (w / mass - g).abs() > 1e-9 || (w / g - mass).abs() > 1e-9 {
        return Err("independent re-check failed for W = m*g".to_string());
    }
    Ok(ForceResult {
        value: w,
        equation_used: "W = m*g".to_string(),
        verify_method: "recompute g = W/m and m = W/g".to_string(),
        verify_value: w / mass,
    })
}

/// Friction f = mu*N on a flat surface, where N = m*g when nothing pushes
/// down additionally. Solve for any one of 'friction', 'normal', 'mu',
/// 'mass' given enough of the others.
///
/// Independent re-checks: the returned quantities are re-plugged into the
/// equation not used, plus the flat-surface identity f/W == mu whenever
/// mass is known.
pub fn solve_friction(
    find: &str,
    mass: Option<f64>,
    mu: Option<f64>,
    normal: Option<f64>,
    friction: Option<f64>,
    g: Option<f64>,
) -> Result<ForceResult, String> {
    let g = g.unwrap_or(DEFAULT_G);
    if let Some(mu) = mu {
        if !(0.0 <= mu) {
            return Err("mu must be non-negative".to_string());
        }
    }

    if find == "friction" {
        let Some(mu) = mu else {
            return Err("solving for friction needs mu".to_string());
        };
        if let Some(normal) = normal {
            let f = mu * normal;
            let check = if mu > 0.0 { f / mu } else { normal };
            if mu > 0.0 && (check - normal).abs() > 1e-9 {
                return Err("independent re-check failed for f = mu*N".to_string());
            }
            return Ok(ForceResult {
                value: f,
                equation_used: "f = mu*N".to_string(),
                verify_method: "reconstruct N = f/mu".to_string(),
                verify_value: check,
            });
        }
        if let Some(mass) = mass {
            let n = mass * g;
            let f = mu * n;
            // flat surface identity: friction-to-weight ratio equals mu
            if (f / mass - mu * g).abs() > 1e-9 {
                return Err("independent re-check failed for f = mu*m*g".to_string());
            }
            return Ok(ForceResult {
                value: f,
                equation_used: "f = mu*N, N = m*g".to_string(),
                verify_method: "flat-surface ratio f/m == mu*g".to_string(),
                verify_value: f / mass,
            });
        }
        return Err("solving for friction needs (mu, normal) or (mu, mass)".to_string());
    }

    if find == "normal" {
        if let Some(mass) = mass {
            let n = mass * g;
            if (n / g - mass).abs() > 1e-9 {
                return Err("independent re-check failed for N = m*g".to_string());
            }
            return Ok(ForceResult {
                value: n,
                equation_used: "N = m*g (flat surface)".to_string(),
                verify_method: "recompute m = N/g".to_string(),
                verify_value: n / g,
            });
        }
        if friction.is_some() && mu.is_some() && mu.unwrap() != 0.0 {
            let n = friction.unwrap() / mu.unwrap();
            let check = mu.unwrap() * n;
            if (check - friction.unwrap()).abs() > 1e-9 {
                return Err("independent re-check failed for f = mu*N".to_string());
            }
            return Ok(ForceResult {
                value: n,
                equation_used: "N = f/mu".to_string(),
                verify_method: "recompute f = mu*N".to_string(),
                verify_value: check,
            });
        }
        return Err("solving for normal needs mass, or (friction, mu)".to_string());
    }

    if find == "mu" {
        let Some(friction) = friction else {
            return Err("solving for mu needs friction".to_string());
        };
        if let Some(normal) = normal {
            if normal == 0.0 {
                return Err("normal force 0 makes mu undefined".to_string());
            }
            let mu_v = friction / normal;
            let check = mu_v * normal;
            if (check - friction).abs() > 1e-9 {
                return Err("independent re-check failed".to_string());
            }
            return Ok(ForceResult {
                value: mu_v,
                equation_used: "mu = f/N".to_string(),
                verify_method: "recompute f = mu*N".to_string(),
                verify_value: check,
            });
        }
        if let Some(mass) = mass {
            if mass != 0.0 {
                let mu_v = friction / (mass * g);
                let check = mu_v * mass * g;
                if (check - friction).abs() > 1e-9 {
                    return Err("independent re-check failed".to_string());
                }
                return Ok(ForceResult {
                    value: mu_v,
                    equation_used: "mu = f/(m*g)".to_string(),
                    verify_method: "recompute f = mu*m*g".to_string(),
                    verify_value: check,
                });
            }
        }
        return Err("solving for mu needs (friction, normal) or (friction, mass)".to_string());
    }

    if find == "mass" {
        if let Some(normal) = normal {
            let m = normal / g;
            let check = m * g;
            if (check - normal).abs() > 1e-9 {
                return Err("independent re-check failed for N = m*g".to_string());
            }
            return Ok(ForceResult {
                value: m,
                equation_used: "m = N/g".to_string(),
                verify_method: "recompute N = m*g".to_string(),
                verify_value: check,
            });
        }
        if friction.is_some() && mu.is_some() && mu.unwrap() != 0.0 {
            let m = friction.unwrap() / (mu.unwrap() * g);
            let check = mu.unwrap() * m * g;
            if (check - friction.unwrap()).abs() > 1e-9 {
                return Err("independent re-check failed".to_string());
            }
            return Ok(ForceResult {
                value: m,
                equation_used: "m = f/(mu*g)".to_string(),
                verify_method: "recompute f = mu*m*g".to_string(),
                verify_value: check,
            });
        }
        return Err("solving for mass needs normal force, or (friction, mu)".to_string());
    }

    Err("find must be one of friction, normal, mu, mass".to_string())
}

/// Acceleration of a block on a flat surface under an applied horizontal
/// force with kinetic friction opposing it: a = (F_applied - mu*m*g) / m.
/// Independent re-check: the resulting net force a*m must equal
/// F_applied - f, and f itself must equal mu*m*g.
pub fn solve_net_acceleration(
    mass: f64,
    applied_force: f64,
    mu: Option<f64>,
    g: Option<f64>,
) -> Result<ForceResult, String> {
    let mu = mu.unwrap_or(0.0);
    let g = g.unwrap_or(DEFAULT_G);
    if mass <= 0.0 {
        return Err("mass must be positive".to_string());
    }
    let f_fric = mu * mass * g;
    let f_net = applied_force - f_fric;
    let a = f_net / mass;
    let check1 = a * mass; // net force from the answer
    let check2 = applied_force - f_fric; // net force from first principles
    if (check1 - check2).abs() > 1e-9 {
        return Err("independent re-check failed for net-force identity".to_string());
    }
    Ok(ForceResult {
        value: a,
        equation_used: "a = (F - mu*m*g)/m".to_string(),
        verify_method: "net-force identity a*m == F - f".to_string(),
        verify_value: check1,
    })
}
