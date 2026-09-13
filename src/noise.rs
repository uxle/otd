//! P0440 — 2D value noise + fBm for terrain (seeded, deterministic).

pub use crate::rng::Rng;

fn hash2(x: i64, y: i64, seed: u64) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9E3779B97F4A7C15)
        ^ (y as u64).wrapping_mul(0xC2B2AE3D27D4EB4F)
        ^ seed.wrapping_mul(0x165667B19E3779F9);
    h ^= h >> 15; h = h.wrapping_mul(0x85EBCA6B); h ^= h >> 13;
    (h >> 11) as f64 / (1u64 << 53) as f64
}

fn smooth(t: f64) -> f64 { t * t * (3.0 - 2.0 * t) }

pub fn value_noise(x: f64, z: f64, seed: u64) -> f64 {
    let xi = x.floor() as i64;
    let zi = z.floor() as i64;
    let tx = smooth(x - xi as f64);
    let tz = smooth(z - zi as f64);
    let a = hash2(xi, zi, seed);
    let b = hash2(xi + 1, zi, seed);
    let c = hash2(xi, zi + 1, seed);
    let d = hash2(xi + 1, zi + 1, seed);
    let ab = a + (b - a) * tx;
    let cd = c + (d - c) * tx;
    ab + (cd - ab) * tz
}

/// fBm, 4 octaves, roughly [-1, 1].
pub fn fbm(x: f64, z: f64, seed: u64, octaves: u32) -> f64 {
    let mut amp = 1.0;
    let mut freq = 1.0;
    let mut sum = 0.0;
    let mut norm = 0.0;
    for _ in 0..octaves {
        sum += amp * (value_noise(x * freq, z * freq, seed) * 2.0 - 1.0);
        norm += amp;
        amp *= 0.5;
        freq *= 2.0;
    }
    sum / norm
}
