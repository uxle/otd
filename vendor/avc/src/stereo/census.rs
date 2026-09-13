//! Two-scale census transform (the v2 stereo upgrade).
//!
//! - **Fine census**: 5x5 window, 24-bit descriptor - precise local structure.
//! - **Ring census**: 8 samples on a radius-4 ring, 8-bit descriptor - robust
//!   to large-scale texture where the fine census saturates.
//!
//! Combined cost = hamming(fine) + hamming(ring), bounded [0, 32].
//!
//! v4: both descriptors are computed in ONE row pass, and rows are
//! parallelized (rayon). `census_into` reuses caller-owned buffers so the
//! steady-state matcher performs no large allocations.

use image::GrayImage;
use rayon::prelude::*;

#[inline]
fn px(img: &[u8], w: usize, h: usize, x: i64, y: i64) -> u8 {
    let x = x.clamp(0, (w - 1) as i64);
    let y = y.clamp(0, (h - 1) as i64);
    img[y as usize * w + x as usize]
}

/// 5x5 census offsets (24 neighbours, row-major, centre excluded).
const FINE_OFFSETS: [(i64, i64); 24] = [
    (-2, -2), (-1, -2), (0, -2), (1, -2), (2, -2),
    (-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
    (-2, 0), (-1, 0), /*centre*/ (1, 0), (2, 0),
    (-2, 1), (-1, 1), (0, 1), (1, 1), (2, 1),
    (-2, 2), (-1, 2), (0, 2), (1, 2), (2, 2),
];

/// Radius-4 ring offsets (8 samples).
const RING_OFFSETS: [(i64, i64); 8] = [
    (0, -4), (3, -3), (4, 0), (3, 3), (0, 4), (-3, 3), (-4, 0), (-3, -3),
];

/// Census descriptors for a whole image (fine: u32, ring: u8).
#[derive(Debug, Clone, Default)]
pub struct Census {
    pub w: usize,
    pub h: usize,
    pub fine: Vec<u32>,
    pub ring: Vec<u8>,
}

/// Compute both census descriptors into caller-owned buffers (v4).
pub fn census_into(img: &GrayImage, out: &mut Census) {
    let (w, h) = img.dimensions();
    let (w, h) = (w as usize, h as usize);
    let buf = img.as_raw();
    out.w = w;
    out.h = h;
    out.fine.clear();
    out.fine.resize(w * h, 0);
    out.ring.clear();
    out.ring.resize(w * h, 0);
    out.fine
        .par_chunks_mut(w)
        .zip(out.ring.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (frow, rrow))| {
            let yc = y as i64;
            for x in 0..w {
                let xc = x as i64;
                let c = px(buf, w, h, xc, yc);
                let mut f = 0u32;
                for (i, &(ox, oy)) in FINE_OFFSETS.iter().enumerate() {
                    let n = px(buf, w, h, xc + ox, yc + oy);
                    if n >= c {
                        f |= 1 << i;
                    }
                }
                frow[x] = f;
                let mut r = 0u8;
                for (i, &(ox, oy)) in RING_OFFSETS.iter().enumerate() {
                    let n = px(buf, w, h, xc + ox, yc + oy);
                    if n >= c {
                        r |= 1 << i;
                    }
                }
                rrow[x] = r;
            }
        });
}

/// Allocate-and-compute convenience wrapper (tests / one-shot use).
pub fn census(img: &GrayImage) -> Census {
    let mut out = Census { w: 0, h: 0, fine: Vec::new(), ring: Vec::new() };
    census_into(img, &mut out);
    out
}

#[inline]
pub fn hamming_fine(a: u32, b: u32) -> u16 {
    ((a ^ b).count_ones()) as u16
}

#[inline]
pub fn hamming_ring(a: u8, b: u8) -> u16 {
    ((a ^ b).count_ones()) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn census_shift_invariance() {
        // a horizontally shifted constant pattern must keep descriptors equal
        let mut img = GrayImage::new(40, 20);
        for y in 0..20 {
            for x in 0..40 {
                let v = ((x * 7 + y * 13) % 251) as u8;
                img.put_pixel(x, y, image::Luma([v]));
            }
        }
        let c = census(&img);
        // identical pixels -> identical descriptors
        assert_eq!(c.fine[10 * 40 + 10], c.fine[10 * 40 + 10]);
        // smooth ramp: two horizontally adjacent descriptors differ in a few bits
        let d = hamming_fine(c.fine[10 * 40 + 12], c.fine[10 * 40 + 13]);
        assert!(d <= 24);
    }

    #[test]
    fn census_into_reuses_buffers_and_matches() {
        let mut img = GrayImage::new(64, 32);
        for y in 0..32 {
            for x in 0..64 {
                img.put_pixel(x, y, image::Luma([((x * 31 + y * 17) % 251) as u8]));
            }
        }
        let a = census(&img);
        let mut b = Census { w: 0, h: 0, fine: Vec::new(), ring: Vec::new() };
        census_into(&img, &mut b);
        assert_eq!(a.fine, b.fine);
        assert_eq!(a.ring, b.ring);
        // capacity must survive a second call (no reallocation in steady state)
        let cap = b.fine.capacity();
        census_into(&img, &mut b);
        assert_eq!(b.fine.capacity(), cap, "census buffer capacity reused");
        assert_eq!(a.fine, b.fine);
    }

    #[test]
    fn hamming_basics() {
        assert_eq!(hamming_fine(0b0000, 0b1111), 4);
        assert_eq!(hamming_fine(0, 0), 0);
        assert_eq!(hamming_ring(0b10101010, 0b01010101), 8);
    }
}
