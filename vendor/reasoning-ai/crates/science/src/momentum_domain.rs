//! Phase 112 — Physics: momentum and collisions (Rust port of
//! python/science/momentum_domain.py)
//!
//! Textbook ground truth: p = m*v, J = F*dt = delta_p, perfectly
//! inelastic v' = (m1*v1 + m2*v2)/(m1+m2), elastic 1D collision equations.
//! Independent cross-checks: momentum conservation (always), kinetic
//! energy non-increase for inelastic / exact conservation for elastic.

/// Result of a momentum-domain computation (tuples of values) with its
/// independent re-check values.
#[derive(Debug, Clone, PartialEq)]
pub struct MomentumResult {
    pub values: Vec<f64>,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: Vec<f64>,
}

/// p = m*v. Re-check: p/m == v and p/v == m.
pub fn momentum(mass: f64, velocity: f64) -> Result<MomentumResult, String> {
    if mass <= 0.0 {
        return Err("mass must be positive".to_string());
    }
    let p = mass * velocity;
    if (p / mass - velocity).abs() > 1e-9
        || (velocity != 0.0 && (p / velocity - mass).abs() > 1e-9)
    {
        return Err("independent re-check failed for p = m*v".to_string());
    }
    Ok(MomentumResult {
        values: vec![p],
        equation_used: "p = m*v".to_string(),
        verify_method: "reconstruct v = p/m".to_string(),
        verify_values: vec![p / mass],
    })
}

/// Impulse-momentum theorem: F*dt = m*(v_f - v_i); returns v_final (and
/// the impulse dp). Re-check: dp/dt reproduces the force.
pub fn impulse(
    force: f64,
    dt: f64,
    mass: f64,
    v_initial: Option<f64>,
) -> Result<MomentumResult, String> {
    let v_initial = v_initial.unwrap_or(0.0);
    if dt <= 0.0 {
        return Err("dt must be positive".to_string());
    }
    if mass <= 0.0 {
        return Err("mass must be positive".to_string());
    }
    let dp = force * dt;
    let v_final = v_initial + dp / mass;
    let f_back = (mass * (v_final - v_initial)) / dt;
    if (f_back - force).abs() > 1e-9 {
        return Err("independent re-check failed for J = F*dt = dp".to_string());
    }
    Ok(MomentumResult {
        values: vec![v_final, dp],
        equation_used: "v_f = v_i + F*dt/m".to_string(),
        verify_method: "reconstruct F = dp/dt".to_string(),
        verify_values: vec![f_back],
    })
}

/// Perfectly inelastic 1D collision (objects stick together).
/// Re-checks (two independent laws):
///   1. momentum conservation: (m1+m2)*v' == m1*v1 + m2*v2
///   2. energy non-increase: KE_after <= KE_before + tiny epsilon
pub fn inelastic_collision(m1: f64, v1: f64, m2: f64, v2: f64) -> Result<MomentumResult, String> {
    if m1 <= 0.0 || m2 <= 0.0 {
        return Err("masses must be positive".to_string());
    }
    let p_before = m1 * v1 + m2 * v2;
    let v_final = p_before / (m1 + m2);
    let p_after = (m1 + m2) * v_final;
    if (p_after - p_before).abs() > 1e-9 {
        return Err("momentum conservation re-check failed".to_string());
    }
    let ke_before = 0.5 * m1 * v1.powi(2) + 0.5 * m2 * v2.powi(2);
    let ke_after = 0.5 * (m1 + m2) * v_final.powi(2);
    if ke_after > ke_before + 1e-9 {
        return Err("energy constraint violated: inelastic collision gained KE".to_string());
    }
    let lost = ke_before - ke_after;
    Ok(MomentumResult {
        values: vec![v_final, lost],
        equation_used: "v' = (m1*v1 + m2*v2)/(m1+m2)".to_string(),
        verify_method: "momentum conservation + KE non-increase".to_string(),
        verify_values: vec![p_after, lost],
    })
}

/// 1D elastic collision. Re-checks (two independent laws, both exact):
///   1. momentum conservation: m1*v1' + m2*v2' == m1*v1 + m2*v2
///   2. kinetic energy conservation: KE_after == KE_before
/// Special case handled exactly: m1==m2 swaps the velocities.
pub fn elastic_collision(m1: f64, v1: f64, m2: f64, v2: f64) -> Result<MomentumResult, String> {
    if m1 <= 0.0 || m2 <= 0.0 {
        return Err("masses must be positive".to_string());
    }
    let (v1p, v2p) = if m1 == m2 {
        (v2, v1)
    } else {
        (
            ((m1 - m2) * v1 + 2.0 * m2 * v2) / (m1 + m2),
            ((m2 - m1) * v2 + 2.0 * m1 * v1) / (m1 + m2),
        )
    };
    let p_before = m1 * v1 + m2 * v2;
    let p_after = m1 * v1p + m2 * v2p;
    if (p_after - p_before).abs() > 1e-6 {
        return Err("momentum conservation re-check failed".to_string());
    }
    let ke_before = 0.5 * m1 * v1.powi(2) + 0.5 * m2 * v2.powi(2);
    let ke_after = 0.5 * m1 * v1p.powi(2) + 0.5 * m2 * v2p.powi(2);
    if (ke_after - ke_before).abs() > 1e-6 {
        return Err("energy conservation re-check failed".to_string());
    }
    Ok(MomentumResult {
        values: vec![v1p, v2p],
        equation_used: "elastic 1D collision equations".to_string(),
        verify_method: "momentum + kinetic-energy conservation".to_string(),
        verify_values: vec![p_after, ke_after],
    })
}

/// Sum of m*v over objects -- the quantity conservation laws compare
/// against. Re-check: recompute in reverse order (float addition is not
/// associative; identical results both ways is a cheap symmetry check).
pub fn total_momentum(objects: &[(f64, f64)]) -> Result<MomentumResult, String> {
    let p_forward: f64 = objects.iter().map(|(m, v)| m * v).sum();
    let p_reverse: f64 = objects.iter().rev().map(|(m, v)| m * v).sum();
    if (p_forward - p_reverse).abs() > 1e-9 {
        return Err("order-symmetry re-check failed".to_string());
    }
    Ok(MomentumResult {
        values: vec![p_forward],
        equation_used: "p_total = sum(m*v)".to_string(),
        verify_method: "reverse-order recompute".to_string(),
        verify_values: vec![p_reverse],
    })
}
