//! Phase 107 — Clock angle reasoning (Promot: "time")
//! (Rust port of python/puzzles/clock_domain.py).
//!
//! Ground truth: standard analog-clock angle formulas.
//!   - Minute hand moves 360/60 = 6 deg/min.
//!   - Hour hand moves 30 deg/hr = 0.5 deg/min (it creeps forward
//!     continuously with the minutes -- the detail that trips up naive
//!     solutions).
//!   - Angle between them: absolute difference, then the smaller of that
//!     and 360-that (the non-reflex angle, <=180).

/// Default tolerance used by `is_straight_line` / `is_right_angle`
/// (Python default argument `tolerance: float = 1e-9`).
pub const DEFAULT_TOLERANCE: f64 = 1e-9;

pub fn hour_hand_angle(hour: i32, minute: i32) -> Result<f64, String> {
    if !(0..=23).contains(&hour) {
        return Err(format!("hour must be 0-23, got {}", hour));
    }
    if !(0..60).contains(&minute) {
        return Err(format!("minute must be 0-59, got {}", minute));
    }
    let h12 = hour % 12;
    Ok((h12 as f64 * 30.0 + minute as f64 * 0.5) % 360.0)
}

pub fn minute_hand_angle(minute: i32) -> Result<f64, String> {
    if !(0..60).contains(&minute) {
        return Err(format!("minute must be 0-59, got {}", minute));
    }
    Ok((minute as f64 * 6.0) % 360.0)
}

pub fn angle_between_hands(hour: i32, minute: i32) -> Result<f64, String> {
    let h = hour_hand_angle(hour, minute)?;
    let m = minute_hand_angle(minute)?;
    let diff = (h - m).abs();
    Ok(f64::min(diff, 360.0 - diff))
}

/// Hands form a straight line (0 or 180 degrees apart).
pub fn is_straight_line(hour: i32, minute: i32, tolerance: f64) -> Result<bool, String> {
    let a = angle_between_hands(hour, minute)?;
    Ok(a.abs() < tolerance || (a - 180.0).abs() < tolerance)
}

pub fn is_right_angle(hour: i32, minute: i32, tolerance: f64) -> Result<bool, String> {
    Ok((angle_between_hands(hour, minute)? - 90.0).abs() < tolerance)
}
