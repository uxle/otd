//! P0610 — SIMD + assembly kernels with runtime dispatch and pure-Rust
//! fallbacks. SSE2 is the x86-64 baseline (guaranteed); AVX2 is detected at
//! runtime. Every kernel has a parity test against scalar.

pub mod assembly;

#[cfg(target_arch = "x86_64")]
pub mod x86 {
    use core::arch::x86_64::*;

    /// Blend 4 pixels at once: out = dst·(1−a) + src·a. (SSE2)
    #[target_feature(enable = "sse2")]
    pub unsafe fn blend4(px: &mut [u32], src: [u32; 3], alpha: f32, n: usize) {
        let a = _mm_set1_ps(alpha);
        let ia = _mm_set1_ps(1.0 - alpha);
        let sr = _mm_set1_ps(src[0] as f32);
        let sg = _mm_set1_ps(src[1] as f32);
        let sb = _mm_set1_ps(src[2] as f32);
        let zero = _mm_setzero_ps();
        let cap = _mm_set1_ps(255.0);
        let mut i = 0usize;
        while i + 4 <= n {
            let p: [u32; 4] = [px[i], px[i + 1], px[i + 2], px[i + 3]];
            let dr = _mm_setr_ps(
                ((p[0] >> 16) & 0xFF) as f32,
                ((p[1] >> 16) & 0xFF) as f32,
                ((p[2] >> 16) & 0xFF) as f32,
                ((p[3] >> 16) & 0xFF) as f32,
            );
            let dg = _mm_setr_ps(
                ((p[0] >> 8) & 0xFF) as f32,
                ((p[1] >> 8) & 0xFF) as f32,
                ((p[2] >> 8) & 0xFF) as f32,
                ((p[3] >> 8) & 0xFF) as f32,
            );
            let db = _mm_setr_ps(
                (p[0] & 0xFF) as f32,
                (p[1] & 0xFF) as f32,
                (p[2] & 0xFF) as f32,
                (p[3] & 0xFF) as f32,
            );
            let outr = _mm_add_ps(_mm_mul_ps(dr, ia), _mm_mul_ps(sr, a));
            let outg = _mm_add_ps(_mm_mul_ps(dg, ia), _mm_mul_ps(sg, a));
            let outb = _mm_add_ps(_mm_mul_ps(db, ia), _mm_mul_ps(sb, a));
            // cvttps truncates, matching the scalar `as u32` path bit-for-bit
            let ir = _mm_cvttps_epi32(_mm_min_ps(_mm_max_ps(outr, zero), cap));
            let ig = _mm_cvttps_epi32(_mm_min_ps(_mm_max_ps(outg, zero), cap));
            let ib = _mm_cvttps_epi32(_mm_min_ps(_mm_max_ps(outb, zero), cap));
            let rarr = std::mem::transmute::<__m128i, [i32; 4]>(ir);
            let garr = std::mem::transmute::<__m128i, [i32; 4]>(ig);
            let barr = std::mem::transmute::<__m128i, [i32; 4]>(ib);
            for k in 0..4 {
                px[i + k] = ((rarr[k] as u32 & 0xFF) << 16)
                    | ((garr[k] as u32 & 0xFF) << 8)
                    | (barr[k] as u32 & 0xFF);
            }
            i += 4;
        }
        while i < n {
            blend1(px, src, alpha, i);
            i += 1;
        }
    }

    #[inline(always)]
    unsafe fn blend1(px: &mut [u32], src: [u32; 3], alpha: f32, i: usize) {
        let d = px[i];
        let dr = ((d >> 16) & 0xFF) as f32;
        let dg = ((d >> 8) & 0xFF) as f32;
        let db = (d & 0xFF) as f32;
        let r = dr * (1.0 - alpha) + src[0] as f32 * alpha;
        let g = dg * (1.0 - alpha) + src[1] as f32 * alpha;
        let b = db * (1.0 - alpha) + src[2] as f32 * alpha;
        px[i] = ((r.clamp(0.0, 255.0) as u32) << 16)
            | ((g.clamp(0.0, 255.0) as u32) << 8)
            | (b.clamp(0.0, 255.0) as u32);
    }

    /// 4-wide dot product via intrinsics (parity-tested against asm + scalar).
    #[target_feature(enable = "sse2")]
    pub unsafe fn dot4(a: [f32; 4], b: [f32; 4]) -> f32 {
        let va = _mm_loadu_ps(a.as_ptr());
        let vb = _mm_loadu_ps(b.as_ptr());
        let m = _mm_mul_ps(va, vb);
        let arr = std::mem::transmute::<__m128, [f32; 4]>(m);
        arr[0] + arr[1] + arr[2] + arr[3]
    }
}

/// Alpha-blend one pixel (scalar — also the correctness oracle).
pub fn blend_pixel(px: &mut u32, r: u32, g: u32, b: u32, alpha: f32) {
    let d = *px;
    let dr = ((d >> 16) & 0xFF) as f32;
    let dg = ((d >> 8) & 0xFF) as f32;
    let db = (d & 0xFF) as f32;
    let nr = dr * (1.0 - alpha) + (r & 0xFF) as f32 * alpha;
    let ng = dg * (1.0 - alpha) + (g & 0xFF) as f32 * alpha;
    let nb = db * (1.0 - alpha) + (b & 0xFF) as f32 * alpha;
    *px = ((nr.clamp(0.0, 255.0) as u32) << 16)
        | ((ng.clamp(0.0, 255.0) as u32) << 8)
        | (nb.clamp(0.0, 255.0) as u32);
}

/// Blend a run of pixels (SIMD on x86-64, scalar elsewhere).
pub fn blend_run(px: &mut [u32], r: u32, g: u32, b: u32, alpha: f32) {
    #[cfg(target_arch = "x86_64")]
    {
        // SSE2 is part of the x86-64 baseline — always available
        unsafe {
            x86::blend4(px, [r & 0xFF, g & 0xFF, b & 0xFF], alpha, px.len());
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        for p in px.iter_mut() {
            blend_pixel(p, r, g, b, alpha);
        }
    }
}

/// Fill a u32 buffer — uses the raw-assembly `rep stosd` kernel on x86-64.
pub fn memset32(dst: &mut [u32], val: u32) {
    #[cfg(target_arch = "x86_64")]
    {
        if !dst.is_empty() {
            unsafe {
                assembly::x86_64_kernels::memset32(dst.as_mut_ptr(), val, dst.len());
            }
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        assembly::memset32_rust(dst, val);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blend_run_matches_scalar() {
        let n = 1000;
        let mut a = vec![0x4080C0u32; n];
        let mut b = a.clone();
        blend_run(&mut a, 200, 100, 50, 0.37);
        for i in 0..n {
            blend_pixel(&mut b[i], 200, 100, 50, 0.37);
        }
        assert_eq!(a, b, "SIMD blend must be bit-identical to scalar");
    }

    #[test]
    fn memset32_matches_rust() {
        let mut a = vec![0u32; 5000];
        let mut b = vec![0u32; 5000];
        memset32(&mut a, 0x1A2B3C4D);
        assembly::memset32_rust(&mut b, 0x1A2B3C4D);
        assert_eq!(a, b);
        assert!(a.iter().all(|&p| p == 0x1A2B3C4D));
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn asm_hsum4_matches_rust() {
        unsafe {
            let cases = [
                (1.0f32, 2.0, 3.0, 4.0),
                (0.5, -0.5, 10.0, -10.0),
                (100.25, 0.125, 7.5, 2.25),
            ];
            for (x, y, z, w) in cases {
                let asm_v = assembly::x86_64_kernels::hsum4(x, y, z, w);
                let rust_v = assembly::hsum4_rust(x, y, z, w);
                assert!((asm_v - rust_v).abs() < 1e-5, "{} vs {}", asm_v, rust_v);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn asm_dot4_matches_rust() {
        unsafe {
            let cases: Vec<([f32; 4], [f32; 4])> = vec![
                ([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]),
                ([0.5, -1.5, 2.5, -0.5], [2.0, 2.0, 2.0, 2.0]),
                ([10.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0]),
            ];
            for (a, b) in cases {
                let asm_v = assembly::x86_64_kernels::dot4_asm(a, b);
                let rust_v = assembly::dot4_rust(a, b);
                assert!((asm_v - rust_v).abs() < 1e-4, "dot4 {} vs {}", asm_v, rust_v);
                let intrin = x86::dot4(a, b);
                assert!((intrin - rust_v).abs() < 1e-4, "intrinsics {} vs {}", intrin, rust_v);
            }
        }
    }
}
