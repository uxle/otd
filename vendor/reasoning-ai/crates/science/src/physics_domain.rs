//! Phase 102 — Physics: kinematics and speed/distance/time (Rust port of
//! python/science/physics_domain.py)
//!
//! Ground truth comes from the standard SUVAT equations of constant-
//! acceleration motion (u, v, a, t, s). Given any 3 of the 5 quantities the
//! others are determined; this module solves for exactly one missing
//! quantity given the 4 *specific* knowns each SUVAT equation needs
//! (deliberately not a general 5-variable solver).

/// Result of a kinematics solve: the value plus the SUVAT equation used.
#[derive(Debug, Clone, PartialEq)]
pub struct KinematicsResult {
    pub value: f64,
    pub equation_used: String,
}

/// Optional SUVAT inputs (the Python keyword arguments u/v/a/t/s, all
/// defaulting to None).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct KinematicsInputs {
    pub u: Option<f64>,
    pub v: Option<f64>,
    pub a: Option<f64>,
    pub t: Option<f64>,
    pub s: Option<f64>,
}

/// `find`: which quantity to solve for -- one of 'u','v','a','t','s'.
/// Provide the OTHER quantities needed; Err if what's provided isn't
/// enough to pin the answer down with a single SUVAT equation (rather than
/// silently picking an arbitrary one).
pub fn solve_kinematics(find: &str, inp: &KinematicsInputs) -> Result<KinematicsResult, String> {
    let (u, v, a, t, s) = (inp.u, inp.v, inp.a, inp.t, inp.s);

    if find == "v" {
        if let (Some(u), Some(a), Some(t)) = (u, a, t) {
            return Ok(KinematicsResult {
                value: u + a * t,
                equation_used: "v = u + a*t".to_string(),
            });
        }
        if let (Some(u), Some(a), Some(s)) = (u, a, s) {
            let val_sq = u.powi(2) + 2.0 * a * s;
            if val_sq < 0.0 {
                return Err("v^2 would be negative -- inconsistent inputs".to_string());
            }
            return Ok(KinematicsResult {
                value: val_sq.powf(0.5),
                equation_used: "v^2 = u^2 + 2*a*s".to_string(),
            });
        }
        return Err("solving for v needs (u,a,t) or (u,a,s)".to_string());
    }

    if find == "s" {
        if let (Some(u), Some(t), Some(a)) = (u, t, a) {
            return Ok(KinematicsResult {
                value: u * t + 0.5 * a * t.powi(2),
                equation_used: "s = u*t + 0.5*a*t^2".to_string(),
            });
        }
        if let (Some(u), Some(v), Some(t)) = (u, v, t) {
            return Ok(KinematicsResult {
                value: (u + v) / 2.0 * t,
                equation_used: "s = (u+v)/2 * t".to_string(),
            });
        }
        return Err("solving for s needs (u,a,t) or (u,v,t)".to_string());
    }

    if find == "u" {
        if let (Some(v), Some(a), Some(t)) = (v, a, t) {
            return Ok(KinematicsResult {
                value: v - a * t,
                equation_used: "u = v - a*t".to_string(),
            });
        }
        return Err("solving for u needs (v,a,t)".to_string());
    }

    if find == "a" {
        if let (Some(v), Some(u), Some(t)) = (v, u, t) {
            if t == 0.0 {
                return Err("t=0 makes acceleration undefined here".to_string());
            }
            return Ok(KinematicsResult {
                value: (v - u) / t,
                equation_used: "a = (v-u)/t".to_string(),
            });
        }
        return Err("solving for a needs (v,u,t)".to_string());
    }

    if find == "t" {
        if let (Some(v), Some(u), Some(a)) = (v, u, a) {
            if a == 0.0 {
                return Err("a=0 makes t undefined via this equation".to_string());
            }
            return Ok(KinematicsResult {
                value: (v - u) / a,
                equation_used: "t = (v-u)/a".to_string(),
            });
        }
        return Err("solving for t needs (v,u,a)".to_string());
    }

    Err(format!("find must be one of u,v,a,t,s -- got '{}'", find))
}

/// speed = distance / time. Given the other two, solve for the third.
pub fn solve_speed_distance_time(
    find: &str,
    speed: Option<f64>,
    distance: Option<f64>,
    time: Option<f64>,
) -> Result<f64, String> {
    if find == "speed" {
        let (Some(distance), Some(time)) = (distance, time) else {
            return Err("need distance and time".to_string());
        };
        if time == 0.0 {
            return Err("time cannot be 0".to_string());
        }
        return Ok(distance / time);
    }
    if find == "distance" {
        let (Some(speed), Some(time)) = (speed, time) else {
            return Err("need speed and time".to_string());
        };
        return Ok(speed * time);
    }
    if find == "time" {
        let (Some(distance), Some(speed)) = (distance, speed) else {
            return Err("need distance and speed".to_string());
        };
        if speed == 0.0 {
            return Err("speed cannot be 0".to_string());
        }
        return Ok(distance / speed);
    }
    Err(format!(
        "find must be one of speed,distance,time -- got '{}'",
        find
    ))
}

/// Classic relative-speed reasoning: same direction -> difference (faster
/// overtakes slower at this closing rate); opposite direction -> sum (they
/// approach/separate at this combined rate).
pub fn relative_speed(speed_a: f64, speed_b: f64, same_direction: bool) -> f64 {
    if same_direction {
        (speed_a - speed_b).abs()
    } else {
        speed_a + speed_b
    }
}
