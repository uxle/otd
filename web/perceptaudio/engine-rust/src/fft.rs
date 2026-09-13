//! Radix-2 FFT with precomputed twiddle factors and bit-reversal tables.
//!
//! Upgrades over the TypeScript engine: the twiddle factors and the
//! bit-reversal permutation are computed once per engine lifetime instead of
//! per butterfly, and all hot loops work on contiguous `&mut [f32]` slices the
//! optimizer can vectorize.

pub const FRAME_SIZE: usize = 2048;
pub const TARGET_RATE: f32 = 16000.0;

/// Precomputed radix-2 complex FFT plan.
pub struct Fft {
    n: usize,
    /// bit-reversal permutation table
    bitrev: Vec<usize>,
    /// twiddle factors exp(-2*pi*i*k/n) for k in 0..n/2
    tw_re: Vec<f32>,
    tw_im: Vec<f32>,
    /// Hann window of length n
    hann: Vec<f32>,
}

impl Fft {
    pub fn new(n: usize) -> Self {
        assert!(n.is_power_of_two(), "FFT size must be a power of two");

        // bit-reversal permutation
        let bits = n.trailing_zeros();
        let mut bitrev = vec![0usize; n];
        for i in 0..n {
            let mut r = 0usize;
            let mut v = i;
            for _ in 0..bits {
                r = (r << 1) | (v & 1);
                v >>= 1;
            }
            bitrev[i] = r;
        }

        // twiddle table for the largest stage; smaller stages stride it
        let mut tw_re = vec![0f32; n / 2];
        let mut tw_im = vec![0f32; n / 2];
        for k in 0..n / 2 {
            let ang = -2.0 * std::f32::consts::PI * (k as f32) / (n as f32);
            tw_re[k] = ang.cos();
            tw_im[k] = ang.sin();
        }

        // Hann window (periodic form matches overlap-add at 50% exactly)
        let mut hann = vec![0f32; n];
        for i in 0..n {
            hann[i] = 0.5 * (1.0 - (2.0 * std::f32::consts::PI * (i as f32) / (n as f32)).cos());
        }

        Self { n, bitrev, tw_re, tw_im, hann }
    }

    /// Forward, in-place, complex FFT.
    pub fn forward(&self, re: &mut [f32], im: &mut [f32]) {
        self.run(re, im, 1.0);
    }

    /// Inverse, in-place, complex FFT (scaled by 1/n).
    pub fn inverse(&self, re: &mut [f32], im: &mut [f32]) {
        self.run(re, im, -1.0);
        let inv = 1.0 / self.n as f32;
        for v in re.iter_mut() {
            *v *= inv;
        }
        for v in im.iter_mut() {
            *v *= inv;
        }
    }

    fn run(&self, re: &mut [f32], im: &mut [f32], sign: f32) {
        let n = self.n;

        // permute to bit-reversed order
        for i in 0..n {
            let j = self.bitrev[i];
            if i < j {
                re.swap(i, j);
                im.swap(i, j);
            }
        }

        // butterflies — twiddles are strided reads from the precomputed table
        let half_n = n / 2;
        let mut len = 2;
        while len <= n {
            let half = len / 2;
            let stride = half_n / half;
            let mut base = 0;
            while base < n {
                let mut k = 0;
                for j in 0..half {
                    let tw = k * stride;
                    let w_re = self.tw_re[tw];
                    let w_im = sign * self.tw_im[tw];
                    let a = base + j;
                    let b = a + half;
                    let ar = re[a];
                    let ai = im[a];
                    let br = re[b] * w_re - im[b] * w_im;
                    let bi = re[b] * w_im + im[b] * w_re;
                    re[a] = ar + br;
                    im[a] = ai + bi;
                    re[b] = ar - br;
                    im[b] = ai - bi;
                    k += 1;
                }
                base += len;
            }
            len <<= 1;
        }
    }

    /// Apply the precomputed Hann window in place.
    pub fn hann_apply(&self, out: &mut [f32]) {
        for (o, &w) in out.iter_mut().zip(self.hann.iter()) {
            *o *= w;
        }
    }

    /// Multiply input by the Hann window into `out`.
    pub fn hann_multiply(&self, input: &[f32], out: &mut [f32]) {
        for i in 0..self.n {
            out[i] = input[i] * self.hann[i];
        }
    }

    pub fn hann(&self) -> &[f32] {
        &self.hann
    }

    pub fn size(&self) -> usize {
        self.n
    }
}

/// Shared 2048-point plan used across the engine (FFT frames are always 2048).
pub const fn frame_size() -> usize {
    FRAME_SIZE
}
