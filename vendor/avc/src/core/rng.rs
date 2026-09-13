//! Deterministic pseudo-random number generation (no external crates).
//!
//! xorshift64* core + Box-Muller for gaussians. Deterministic seeding keeps
//! the synthetic validation runs reproducible byte-for-byte.

pub struct GaussRng {
    state: u64,
    spare: Option<f64>,
}

impl GaussRng {
    pub fn new(seed: u64) -> Self {
        // avoid the all-zero state and low-quality low bits
        let s = if seed == 0 { 0x9E3779B97F4A7C15 } else { seed };
        GaussRng { state: s ^ 0xD1B54A32D192ED03, spare: None }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        // xorshift64*
        let mut x = self.state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.state = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }

    #[inline]
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }

    #[inline]
    pub fn uniform_range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    #[inline]
    pub fn uniform_usize(&mut self, n: usize) -> usize {
        if n == 0 {
            0
        } else {
            (self.next_u64() % n as u64) as usize
        }
    }

    /// Standard normal via Box-Muller (with cached spare).
    pub fn gauss(&mut self, mu: f64, sigma: f64) -> f64 {
        if let Some(s) = self.spare.take() {
            return mu + sigma * s;
        }
        let mut u1 = self.uniform();
        if u1 < 1e-300 {
            u1 = 1e-300;
        }
        let u2 = self.uniform();
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = 2.0 * std::f64::consts::PI * u2;
        self.spare = Some(r * theta.sin());
        mu + sigma * (r * theta.cos())
    }

    /// Fill a vector with gaussians.
    pub fn gauss_vec3(&mut self, mu: [f64; 3], sigma: f64) -> [f64; 3] {
        [
            self.gauss(mu[0], sigma),
            self.gauss(mu[1], sigma),
            self.gauss(mu[2], sigma),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_mean_sigma() {
        let mut rng = GaussRng::new(123);
        let n = 20000;
        let mut s = 0.0;
        let mut s2 = 0.0;
        for _ in 0..n {
            let x = rng.gauss(0.0, 2.0);
            s += x;
            s2 += x * x;
        }
        let mean = s / n as f64;
        let var = s2 / n as f64 - mean * mean;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 4.0).abs() < 0.3, "var {var}");
    }

    #[test]
    fn deterministic_given_seed() {
        let mut a = GaussRng::new(99);
        let mut b = GaussRng::new(99);
        for _ in 0..100 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_usize_in_range() {
        let mut rng = GaussRng::new(5);
        for _ in 0..1000 {
            let i = rng.uniform_usize(10);
            assert!(i < 10);
        }
    }
}
