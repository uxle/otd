//! P0440 support — seeded PRNG (splitmix64/xorshift) and value-noise fBm.
//! Deterministic: `seed: 7` always gives the same scatter / terrain.

pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Rng {
        // splitmix64 warm-up so small seeds (0, 1, 7) don't correlate.
        let mut z = seed.wrapping_add(0x9E3779B97F4A7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        Rng { state: z ^ (z >> 31) }
    }
    /// xorshift64* — [0,1)
    pub fn next_f64(&mut self) -> f64 {
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        let v = x.wrapping_mul(0x2545F4914F6CDD1D);
        (v >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn next_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next_f64()
    }
}

/// Smooth value noise on a 2D integer lattice (seeded).
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

/// Fractal Brownian motion, 4 octaves, returns roughly [-1, 1].
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

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rng_deterministic() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..100 { assert_eq!(a.next_f64(), b.next_f64()); }
    }
    #[test]
    fn noise_in_range() {
        for i in 0..100 {
            let v = fbm(i as f64 * 0.13, i as f64 * 0.29, 42, 4);
            assert!(v >= -1.05 && v <= 1.05);
        }
    }
}
