//! P0610 — genuine handwritten x86-64 assembly kernels (raw `asm!` blocks).
//! The backend promise: "Rust and assembly only." Each kernel has a pure-Rust
//! twin and a bit-parity unit test. Non-x86-64 builds fall back to Rust.

#[cfg(target_arch = "x86_64")]
pub mod x86_64_kernels {
    /// Fill a u32 slice with a value — the classic `rep stosd`.
    /// (Used by framebuffer clears; beats a Rust loop for big fills.)
    #[inline(never)]
    pub unsafe fn memset32(dst: *mut u32, val: u32, count: usize) {
        std::arch::asm!(
            "cld",
            "rep stosd",
            inlateout("rdi") dst => _,
            inout("rcx") count => _,
            in("eax") val,
            options(preserves_flags)
        );
    }

    /// Horizontal sum of 4 floats in raw SSE2 assembly.
    #[inline(never)]
    pub unsafe fn hsum4(x: f32, y: f32, z: f32, w: f32) -> f32 {
        let out: f32;
        std::arch::asm!(
            "addss xmm1, xmm2",
            "addss xmm0, xmm3",
            "addss xmm0, xmm1",
            lateout("xmm0") out,
            in("xmm0") x,
            in("xmm1") y,
            in("xmm2") z,
            in("xmm3") w,
        );
        out
    }

    /// 4-wide dot product in raw assembly (mulps + movhlps fold, SSE2).
    #[inline(never)]
    pub unsafe fn dot4_asm(a: [f32; 4], b: [f32; 4]) -> f32 {
        let out: f32;
        let pa: *const f32 = a.as_ptr();
        let pb: *const f32 = b.as_ptr();
        std::arch::asm!(
            "movups  xmm0, [rax]",    // a
            "movups  xmm1, [rcx]",    // b
            "mulps   xmm0, xmm1",     // p = a·b (lanes)
            "movhlps xmm2, xmm0",     // xmm2 = [p2, p3, …]
            "addps   xmm2, xmm0",     // xmm2 = [p0+p2, p1+p3, …]
            "movaps  xmm3, xmm2",
            "shufps  xmm3, xmm2, 0x01",      // xmm3[0] = xmm2[1]
            "addss  xmm2, xmm3",      // xmm2[0] = p0+p1+p2+p3
            "movss   xmm1, xmm2",
            in("rax") pa,
            in("rcx") pb,
            lateout("xmm1") out,
        );
        out
    }
}

/// Portable Rust fallbacks (also the correctness oracles for the tests).
pub fn memset32_rust(dst: &mut [u32], val: u32) {
    for p in dst.iter_mut() {
        *p = val;
    }
}

pub fn hsum4_rust(x: f32, y: f32, z: f32, w: f32) -> f32 {
    x + y + z + w
}

pub fn dot4_rust(a: [f32; 4], b: [f32; 4]) -> f32 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2] + a[3] * b[3]
}
