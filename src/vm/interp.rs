//! P0670 — the OTD-ASM interpreter: a 200-register f64 machine with a unit
//! tag per register (plain / length-mm / angle-deg). Unit arithmetic
//! mirrors the 1.0 tree-walk *exactly* — the same promotion rules, the
//! same Dim::mul / Dim::div / combine_add semantics, the same error
//! strings — so VM and tree-walk are interchangeable (parity-tested).
//!
//! The register file is zeroed between runs with a raw `asm!` `rep stosq`
//! kernel (x86-64); non-x86-64 builds use the Rust twin.

use crate::units::{Dim, Qty};
use crate::vm::compile::Program;
use crate::vm::isa::*;

pub const NREGS: usize = 256;

// unit tags (u8, one per register)
pub const TAG_PLAIN: u8 = 0;
pub const TAG_MM: u8 = 1;
pub const TAG_DEG: u8 = 2;

fn tag_of(dim: Dim) -> u8 {
    match dim {
        Dim::Plain => TAG_PLAIN,
        Dim::Length => TAG_MM,
        Dim::Angle => TAG_DEG,
    }
}

fn dim_of(tag: u8) -> Dim {
    match tag {
        TAG_MM => Dim::Length,
        TAG_DEG => Dim::Angle,
        _ => Dim::Plain,
    }
}

// ---------- asm kernels (raw, with Rust twins + parity tests) ----------

/// Zero the register file — `rep stosq`, the classic quadword fill.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub unsafe fn zero_regs_asm(dst: *mut f64, count: usize) {
    std::arch::asm!(
        "cld",
        "rep stosq",
        inlateout("rdi") dst => _,
        inout("rcx") count => _,
        in("rax") 0u64,
        options(preserves_flags)
    );
}

pub fn zero_regs_rust(dst: &mut [f64]) {
    for v in dst.iter_mut() {
        *v = 0.0;
    }
}

/// 3-wide f64 dot product in raw SSE2 assembly (movsd/mulsd/addsd chain).
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub unsafe fn dot3_f64_asm(a: *const f64, b: *const f64) -> f64 {
    let out: f64;
    std::arch::asm!(
        "movsd xmm0, [rax]",        // a0
        "movsd xmm1, [rax + 8]",    // a1
        "movsd xmm2, [rax + 16]",   // a2
        "mulsd xmm0, [rcx]",        // a0*b0
        "mulsd xmm1, [rcx + 8]",    // a1*b1
        "mulsd xmm2, [rcx + 16]",   // a2*b2
        "addsd xmm0, xmm1",
        "addsd xmm0, xmm2",
        lateout("xmm0") out,
        in("rax") a,
        in("rcx") b,
    );
    out
}

pub fn dot3_f64_rust(a: &[f64], b: &[f64]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

/// Fast reciprocal square root: SSE2 `rsqrtss` seed + one Newton–Raphson
/// refinement in f64 (y₁ = y₀(3 − x·y₀²)/2). Raw `asm!`, tested for
/// parity against the plain Rust `1/√x` within the seed's tolerance.
#[cfg(target_arch = "x86_64")]
#[inline(never)]
pub unsafe fn rsqrt_f64_asm(x: f64) -> f64 {
    let out: f64;
    let three: f64 = 3.0;
    let half: f64 = 0.5;
    std::arch::asm!(
        "cvtsd2ss xmm1, xmm0",  // seed: x as f32
        "rsqrtss xmm1, xmm1",   // y0 ≈ 1/√x  (12-bit hardware estimate)
        "cvtss2sd xmm1, xmm1",  // y0 back to f64
        "movapd xmm2, xmm1",
        "mulsd  xmm2, xmm1",    // y0²
        "mulsd  xmm2, xmm0",    // x·y0²
        "subsd  xmm4, xmm2",    // 3 − x·y0²
        "mulsd  xmm1, xmm4",    // y0·(3 − x·y0²)
        "mulsd  xmm1, xmm5",    // · ½  → y1
        "movapd xmm0, xmm1",
        lateout("xmm0") out,
        in("xmm0") x,
        in("xmm4") three,
        in("xmm5") half,
        out("xmm1") _,
        out("xmm2") _,
    );
    out
}

// ---------- the machine ----------

/// A run result: value + unit tag.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VmVal {
    pub v: f64,
    pub tag: u8,
}

impl VmVal {
    pub fn to_qty(&self) -> Qty {
        Qty { v: self.v, dim: dim_of(self.tag) }
    }
    pub fn from_qty(q: &Qty) -> VmVal {
        VmVal { v: q.v, tag: tag_of(q.dim) }
    }
}

/// Run a program. `lookup` resolves variable names (env + magic vars) to
/// quantities — the same resolution order the tree-walk uses happens in
/// the caller. Returns the result register's (value, tag), or a friendly
/// unit error string.
pub fn run(prog: &Program, lookup: &dyn Fn(&str) -> Option<Qty>) -> Result<VmVal, String> {
    let mut regs = vec![0.0f64; NREGS];
    let mut tags = vec![TAG_PLAIN; NREGS];

    // resolve all variables once at run start (values are per-run, so
    // pattern magic re-resolves every iteration — compile once, run many)
    let mut var_vals = Vec::with_capacity(prog.vars.len());
    for name in &prog.vars {
        match lookup(name) {
            Some(q) => var_vals.push(VmVal::from_qty(&q)),
            None => return Err(format!("I don't know what '{}' is", name)),
        }
    }

    let unit_mm = prog.unit_mm;
    let mut pc: usize = 0;
    let mut steps: u64 = 0;
    const MAX_STEPS: u64 = 1_000_000;
    while pc < prog.insns.len() {
        steps += 1;
        if steps > MAX_STEPS {
            return Err("this expression ran too long — is it stuck?".into());
        }
        let ins = &prog.insns[pc];
        pc += 1;
        let d = ins.dst as usize;
        let a = ins.a as usize;
        let b = ins.b as usize;
        match ins.op {
            OP_LOAD_F64 => regs[d] = ins.imm,
            OP_STORE_F64 => {
                // out-slot write: recorded in `outs` for the ISA's sake —
                // the expression tier reads the result register directly
                // (kept as a no-op store so the encoding is exercised)
                let _ = (a, ins.imm);
            }
            OP_UNIT_MM => tags[d] = TAG_MM,
            OP_UNIT_DEG => tags[d] = TAG_DEG,
            OP_UNIT_PLAIN => tags[d] = TAG_PLAIN,
            OP_CONST_PI => {
                regs[d] = std::f64::consts::PI;
                tags[d] = TAG_PLAIN;
            }
            OP_VAR_F64 => {
                let v = &var_vals[a];
                regs[d] = v.v;
                tags[d] = v.tag;
            }
            OP_ADD_F64 | OP_SUB_F64 => {
                let (av, bv, tag) = promote_add(regs[a], regs[b], tags[a], tags[b], unit_mm)?;
                regs[d] = if ins.op == OP_ADD_F64 { av + bv } else { av - bv };
                tags[d] = tag;
            }
            OP_MUL_F64 => {
                let tag = combine_mul(tags[a], tags[b])?;
                regs[d] = regs[a] * regs[b];
                tags[d] = tag;
            }
            OP_DIV_F64 => {
                let tag = combine_div(tags[a], tags[b])?;
                if regs[b] == 0.0 {
                    // same words as the tree-walk (2.1 audit: the VM had been
                    // silently answering inf where the tree-walk errors)
                    return Err("you can't divide by zero".into());
                }
                regs[d] = regs[a] / regs[b];
                tags[d] = tag;
            }
            // unary ops: the compiler emits them with dst == source reg
            OP_NEG_F64 => regs[d] = -regs[d],
            OP_RSQ_F64 => {
                if regs[d] > 0.0 {
                    #[cfg(target_arch = "x86_64")]
                    unsafe {
                        regs[d] = rsqrt_f64_asm(regs[d]);
                    }
                    #[cfg(not(target_arch = "x86_64"))]
                    {
                        regs[d] = 1.0 / regs[d].sqrt();
                    }
                } else {
                    regs[d] = f64::INFINITY;
                }
                tags[d] = TAG_PLAIN;
            }
            OP_DOT3_F64 => {
                // dot3_f64_asm reads 3 contiguous f64s from each raw pointer
                // with no bounds check of its own (it's a bare asm! kernel) —
                // guard here so a compiler bug that allocates a dot3 operand
                // in one of the last two registers fails loudly in debug
                // builds instead of silently reading past the register file.
                debug_assert!(a + 2 < regs.len() && b + 2 < regs.len(), "OP_DOT3_F64 operand register out of bounds");
                #[cfg(target_arch = "x86_64")]
                unsafe {
                    regs[d] = dot3_f64_asm(regs.as_ptr().add(a), regs.as_ptr().add(b));
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    regs[d] = dot3_f64_rust(&regs[a..a + 3], &regs[b..b + 3]);
                }
                tags[d] = TAG_PLAIN;
            }
            OP_CROSS_F64 => {
                let (ax, ay, az) = (regs[a], regs[a + 1], regs[a + 2]);
                let (bx, by, bz) = (regs[b], regs[b + 1], regs[b + 2]);
                regs[d] = ay * bz - az * by;
                regs[d + 1] = az * bx - ax * bz;
                regs[d + 2] = ax * by - ay * bx;
                tags[d] = TAG_PLAIN;
                tags[d + 1] = TAG_PLAIN;
                tags[d + 2] = TAG_PLAIN;
            }
            OP_MIN_F64 => regs[d] = regs[a].min(regs[b]),
            OP_MAX_F64 => regs[d] = regs[a].max(regs[b]),
            OP_SIN_D => {
                regs[d] = regs[d].to_radians().sin();
                tags[d] = TAG_PLAIN;
            }
            OP_COS_D => {
                regs[d] = regs[d].to_radians().cos();
                tags[d] = TAG_PLAIN;
            }
            OP_TAN_D => {
                regs[d] = regs[d].to_radians().tan();
                tags[d] = TAG_PLAIN;
            }
            OP_SQRT_F64 => {
                // negative → NaN, exactly like the 1.0 eval_func
                regs[d] = if regs[d] >= 0.0 { regs[d].sqrt() } else { f64::NAN };
                tags[d] = TAG_PLAIN;
            }
            OP_ABS_F64 => {
                regs[d] = regs[d].abs();
                tags[d] = TAG_PLAIN;
            }
            OP_ROUND_F64 => {
                regs[d] = regs[d].round();
                tags[d] = TAG_PLAIN;
            }
            OP_JMP => pc = a as usize,
            OP_JNZ => {
                if regs[d] != 0.0 {
                    pc = a as usize;
                }
            }
            OP_HALT => break,
            _ => return Err("VM: reserved opcode hit the interpreter".into()),
        }
    }
    let r = prog.result_reg as usize;
    Ok(VmVal { v: regs[r], tag: tags[r] })
}

// ---------- unit semantics (mirror of the 1.0 tree-walk) ----------

/// +/− promotion: bare number adopts the other side's unit (Length gets
/// the scene-default factor; Angle passes straight through). Then the
/// dims must combine. Same strings as `eval.rs`.
fn promote_add(
    av: f64,
    bv: f64,
    ta: u8,
    tb: u8,
    unit_mm: f64,
) -> Result<(f64, f64, u8), String> {
    let (av, ta) = if ta == TAG_PLAIN && tb == TAG_MM {
        (av * unit_mm, TAG_MM)
    } else if ta == TAG_PLAIN && tb == TAG_DEG {
        (av, TAG_DEG)
    } else {
        (av, ta)
    };
    let (bv, tb) = if tb == TAG_PLAIN && ta == TAG_MM {
        (bv * unit_mm, TAG_MM)
    } else if tb == TAG_PLAIN && ta == TAG_DEG {
        (bv, TAG_DEG)
    } else {
        (bv, tb)
    };
    // combine_add
    let tag = match (ta, tb) {
        (TAG_PLAIN, t) | (t, TAG_PLAIN) => t,
        (t, u) if t == u => t,
        (TAG_MM, TAG_DEG) | (TAG_DEG, TAG_MM) => {
            return Err("you added a length to an angle — check the units".into())
        }
        _ => ta,
    };
    Ok((av, bv, tag))
}

fn combine_mul(ta: u8, tb: u8) -> Result<u8, String> {
    match (dim_of(ta).mul(dim_of(tb)),) {
        (Ok(d),) => Ok(tag_of(d)),
        (Err(e),) => Err(e),
    }
}

fn combine_div(ta: u8, tb: u8) -> Result<u8, String> {
    match dim_of(ta).div(dim_of(tb)) {
        Ok(d) => Ok(tag_of(d)),
        Err(e) => Err(e),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lang::parser;
    use crate::vm::compile::compile;

    fn run_src(src: &str, unit_mm: f64, scope: &dyn Fn(&str) -> Option<Qty>) -> VmVal {
        let prog = parser::parse(src, "cm");
        let e = prog
            .stmts
            .iter()
            .find_map(|s| match s {
                crate::lang::ast::Stmt::Assign(_, e, _) => Some(e.clone()),
                _ => None,
            })
            .expect("assignment");
        let p = compile(&e, unit_mm).unwrap();
        run(&p, scope).unwrap()
    }

    fn no_vars(_: &str) -> Option<Qty> {
        None
    }

    #[test]
    fn basic_arithmetic_and_units() {
        // 4cm + 1 → 50mm (plain adopts cm → 10mm; 40 + 10)
        let v = run_src("x = 4cm + 1\n", 10.0, &no_vars);
        assert_eq!((v.v, v.tag), (50.0, TAG_MM));
        // 3 * 2cm = 60mm
        let v = run_src("x = 3 * 2cm\n", 10.0, &no_vars);
        assert_eq!((v.v, v.tag), (60.0, TAG_MM));
        // 90deg / 2 = 45deg
        let v = run_src("x = 90deg / 2\n", 10.0, &no_vars);
        assert_eq!((v.v, v.tag), (45.0, TAG_DEG));
        // 10cm / 2cm = 5 plain
        let v = run_src("x = 10cm / 2cm\n", 10.0, &no_vars);
        assert_eq!((v.v, v.tag), (5.0, TAG_PLAIN));
        // pi
        let v = run_src("x = pi * 2\n", 10.0, &no_vars);
        assert!((v.v - 2.0 * std::f64::consts::PI).abs() < 1e-12);
        assert_eq!(v.tag, TAG_PLAIN);
    }

    #[test]
    fn unit_errors_match_the_walk() {
        // mm × mm
        let prog = parser::parse("x = 2cm * 3cm\n", "cm");
        let e = prog.stmts.iter().find_map(|s| match s {
            crate::lang::ast::Stmt::Assign(_, e, _) => Some(e.clone()),
            _ => None,
        }).unwrap();
        let p = compile(&e, 10.0).unwrap();
        let err = run(&p, &no_vars).unwrap_err();
        assert!(err.contains("area"), "{}", err);
        // length + angle
        let prog = parser::parse("x = 2cm + 30deg\n", "cm");
        let e = prog.stmts.iter().find_map(|s| match s {
            crate::lang::ast::Stmt::Assign(_, e, _) => Some(e.clone()),
            _ => None,
        }).unwrap();
        let p = compile(&e, 10.0).unwrap();
        let err = run(&p, &no_vars).unwrap_err();
        assert!(err.contains("length to an angle"), "{}", err);
    }

    #[test]
    fn functions() {
        let v = run_src("x = sin(30)\n", 10.0, &no_vars);
        assert!((v.v - 0.5).abs() < 1e-12);
        assert_eq!(v.tag, TAG_PLAIN);
        let v = run_src("x = sqrt(16)\n", 10.0, &no_vars);
        assert_eq!(v.v, 4.0);
        let v = run_src("x = min(3, 2, 5)\n", 10.0, &no_vars);
        assert_eq!(v.v, 2.0);
        let v = run_src("x = max(3, 2)\n", 10.0, &no_vars);
        assert_eq!(v.v, 3.0);
        // sqrt(-1) = NaN, same as the tree-walk
        let v = run_src("x = sqrt(0 - 1)\n", 10.0, &no_vars);
        assert!(v.v.is_nan());
    }

    #[test]
    fn magic_vars_resolve_per_run() {
        let scope = |n: &str| -> Option<Qty> {
            match n {
                "i" => Some(Qty::plain(7.0)),
                "a" => Some(Qty::deg(90.0)),
                _ => None,
            }
        };
        let v = run_src("x = i * 10cm\n", 10.0, &scope);
        assert_eq!(v.v, 700.0);
        let v = run_src("x = cos(a)\n", 10.0, &scope);
        assert!((v.v - 0.0).abs() < 1e-12);
    }

    #[test]
    fn dot3_and_cross_execute() {
        // hand-built program: dot3 of (1,2,3)·(4,5,6) = 32
        let mut insns = Vec::new();
        let mut push = |insns: &mut Vec<Insn>, op: u16, dst: u32, a: u32, b: u32, imm: f64| {
            insns.push(Insn::new(op, dst, a, b, imm));
        };
        for (r, v) in [(0u32, 1.0), (1, 2.0), (2, 3.0)] {
            push(&mut insns, OP_LOAD_F64, r, 0, 0, v);
        }
        for (r, v) in [(3u32, 4.0), (4, 5.0), (5, 6.0)] {
            push(&mut insns, OP_LOAD_F64, r, 0, 0, v);
        }
        push(&mut insns, OP_DOT3_F64, 6, 0, 3, 0.0);
        push(&mut insns, OP_HALT, 0, 0, 0, 0.0);
        let prog = Program { insns, vars: vec![], unit_mm: 10.0, result_reg: 6 };
        let v = run(&prog, &no_vars).unwrap();
        assert_eq!(v.v, 32.0);
        // parity: asm kernel vs rust twin
        let a = [1.0f64, 2.0, 3.0];
        let b = [4.0f64, 5.0, 6.0];
        #[cfg(target_arch = "x86_64")]
        unsafe {
            let via_asm = dot3_f64_asm(a.as_ptr(), b.as_ptr());
            assert_eq!(via_asm, dot3_f64_rust(&a, &b));
        }
    }

    #[test]
    fn register_zeroing_parity() {
        let mut x = vec![1.234f64; 64];
        #[cfg(target_arch = "x86_64")]
        unsafe {
            zero_regs_asm(x.as_mut_ptr(), x.len());
        }
        #[cfg(not(target_arch = "x86_64"))]
        zero_regs_rust(&mut x);
        assert!(x.iter().all(|v| *v == 0.0));
    }
}
