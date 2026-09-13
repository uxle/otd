//! Phase 106 — Direction and distance (Promot: "direction and distance")
//! (Rust port of python/puzzles/direction_domain.py).
//!
//! Ground truth: standard 2D coordinate geometry. North=+y, East=+x. Walk
//! a sequence of (direction, distance) moves, track exact position,
//! report straight-line (Euclidean) distance from start and compass
//! bearing of the final position from start. Everything is computed from
//! exact input arithmetic before the one unavoidable sqrt, matching how
//! these problems are graded (numeric distance, exact final coordinates).

use std::collections::HashMap;
use std::sync::OnceLock;

/// `DIRECTION_VECTORS` in the Python module (short and full names).
pub fn direction_vectors() -> &'static HashMap<&'static str, (f64, f64)> {
    static M: OnceLock<HashMap<&'static str, (f64, f64)>> = OnceLock::new();
    M.get_or_init(|| {
        let mut m = HashMap::new();
        m.insert("N", (0.0, 1.0));
        m.insert("S", (0.0, -1.0));
        m.insert("E", (1.0, 0.0));
        m.insert("W", (-1.0, 0.0));
        m.insert("NORTH", (0.0, 1.0));
        m.insert("SOUTH", (0.0, -1.0));
        m.insert("EAST", (1.0, 0.0));
        m.insert("WEST", (-1.0, 0.0));
        m
    })
}

/// `_TURN_LEFT` table (dict in Python; facing is pre-validated by `turn`).
fn _turn_left(facing: &str) -> &'static str {
    match facing {
        "N" => "W",
        "W" => "S",
        "S" => "E",
        "E" => "N",
        // mirrors Python's KeyError on an impossible lookup
        _ => panic!("key error: '{}' not in _TURN_LEFT", facing),
    }
}

/// `_TURN_RIGHT` table.
fn _turn_right(facing: &str) -> &'static str {
    match facing {
        "N" => "E",
        "E" => "S",
        "S" => "W",
        "W" => "N",
        _ => panic!("key error: '{}' not in _TURN_RIGHT", facing),
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Move {
    /// 'N','S','E','W' (or full word)
    pub direction: String,
    pub distance: f64,
}

impl Move {
    /// Positional constructor like the Python dataclass `Move('N', 10)`.
    pub fn new(direction: &str, distance: f64) -> Move {
        Move {
            direction: direction.to_string(),
            distance,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct WalkResult {
    pub final_x: f64,
    pub final_y: f64,
    pub straight_line_distance: f64,
    /// compass bearing (0=N, 90=E, 180=S, 270=W) of final position FROM start
    pub bearing_degrees: f64,
}

pub fn walk(moves: &[Move]) -> Result<WalkResult, String> {
    let mut x = 0.0;
    let mut y = 0.0;
    for m in moves {
        let key = m.direction.trim().to_uppercase();
        let (dx, dy) = match direction_vectors().get(key.as_str()) {
            Some(&v) => v,
            None => return Err(format!("unsupported direction: '{}'", m.direction)),
        };
        x += dx * m.distance;
        y += dy * m.distance;
    }

    let distance = (x * x + y * y).sqrt();
    // compass bearing: atan2(east_component, north_component), 0=N clockwise
    // to 360. Python's float `%` is floored (non-negative for positive
    // divisor) -> rem_euclid.
    let bearing = x.atan2(y).to_degrees().rem_euclid(360.0);
    Ok(WalkResult {
        final_x: x,
        final_y: y,
        straight_line_distance: distance,
        bearing_degrees: bearing,
    })
}

/// Facing 'N'/'S'/'E'/'W', turn left or right by a multiple of 90
/// degrees (the standard puzzle granularity). Returns the new facing.
/// (`degrees` defaults to 90 in Python.)
pub fn turn(facing: &str, turn_direction: &str, degrees: i32) -> Result<String, String> {
    let mut facing = facing.trim().to_uppercase();
    if !matches!(facing.as_str(), "N" | "S" | "E" | "W") {
        return Err(format!("facing must be one of N,S,E,W -- got '{}'", facing));
    }
    if degrees % 90 != 0 {
        return Err("this domain only supports 90-degree-multiple turns".to_string());
    }
    // Python `//` and `%` are floored: div_euclid/rem_euclid match for
    // negative degrees too.
    let steps = degrees.div_euclid(90).rem_euclid(4);
    let left = turn_direction.trim().to_lowercase() == "left";
    for _ in 0..steps {
        facing = if left {
            _turn_left(&facing).to_string()
        } else {
            _turn_right(&facing).to_string()
        };
    }
    Ok(facing)
}
