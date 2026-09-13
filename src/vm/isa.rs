//! P0650 — the OTD-ASM instruction set. Every instruction is exactly
//! **32 bytes** (the v5 §12 layout), SIMD/cache-aligned:
//!
//! ```text
//!  0                   1                   2                   3
//!  0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//! +-------------------+-------------------+-------------------+
//! |  Opcode (16 bits) | Exec Mask (8 bits)| Vector Stride (8b)|
//! +-------------------+-------------------+-------------------+
//! |               Destination Register Index (32 bits)          |
//! +-----------------------------------------------------------+
//! |               Source A Register Index (32 bits)             |
//! +-----------------------------------------------------------+
//! |               Source B Register Index (32 bits)             |
//! +-----------------------------------------------------------+
//! |               Source C Register Index (32 bits)             |
//! +-----------------------------------------------------------+
//! |               Immediate Value Payload (64 bits)             |
//! +-----------------------------------------------------------+
//! |               Predicate / Swizzle Control (32 bits)         |
//! +-----------------------------------------------------------+
//! ```
//!
//! The mask/stride/pred lanes are reserved for the future warp path
//! (P1300+); the numeric tier executes scalar f64 ops plus three vector
//! opcodes (dot3, cross, rsq) that the expression compiler does not emit
//! but the ISA fully defines and the tests exercise.

// ---------- opcodes ----------

pub const OP_LOAD_F64: u16 = 0x0001; // rDst = imm
pub const OP_STORE_F64: u16 = 0x0002; // out[a] = rDst
pub const OP_ADD_F64: u16 = 0x0010; // rDst = rA + rB (unit-checked)
pub const OP_SUB_F64: u16 = 0x0011; // rDst = rA − rB (unit-checked)
pub const OP_MUL_F64: u16 = 0x0012; // rDst = rA × rB (unit-checked)
pub const OP_DIV_F64: u16 = 0x0013; // rDst = rA ÷ rB (unit-checked)
pub const OP_NEG_F64: u16 = 0x0014; // rDst = −rA
pub const OP_RSQ_F64: u16 = 0x0015; // rDst = 1/√rA (asm! fast path)
pub const OP_DOT3_F64: u16 = 0x0016; // rDst = (rA,rA+1,rA+2) · (rB,…)
pub const OP_CROSS_F64: u16 = 0x0017; // rDst..+2 = a × b
pub const OP_MIN_F64: u16 = 0x0018; // rDst = min(rA, rB)
pub const OP_MAX_F64: u16 = 0x0019; // rDst = max(rA, rB)
pub const OP_SIN_D: u16 = 0x0020; // rDst = sin(rA°)
pub const OP_COS_D: u16 = 0x0021; // rDst = cos(rA°)
pub const OP_TAN_D: u16 = 0x0022; // rDst = tan(rA°)
pub const OP_SQRT_F64: u16 = 0x0023; // rDst = √rA (neg → NaN, like 1.0)
pub const OP_ABS_F64: u16 = 0x0024; // rDst = |rA|
pub const OP_ROUND_F64: u16 = 0x0025; // rDst = round(rA)
pub const OP_CONST_PI: u16 = 0x0030; // rDst = π
pub const OP_UNIT_MM: u16 = 0x0040; // tag rDst as length
pub const OP_UNIT_DEG: u16 = 0x0041; // tag rDst as angle
pub const OP_UNIT_PLAIN: u16 = 0x0042; // tag rDst as plain
pub const OP_VAR_F64: u16 = 0x0050; // rDst = scope var a (name in var table)
pub const OP_JMP: u16 = 0x0100; // pc = a
pub const OP_JNZ: u16 = 0x0101; // if rDst ≠ 0: pc = a
pub const OP_HALT: u16 = 0x0102; // end

// reserved for the Deep tiers (v5 encodings, honored as future slots):
// 0x0031 TOPO_SPLIT_EDGE · 0x0032 TOPO_COLLAPSE · 0x0040 BVH8_TEST_RAYS
// 0x0050 XPBD_PROJ_DIST · 0x0060 DEC_COTAN_LAP

pub fn op_name(op: u16) -> &'static str {
    match op {
        OP_LOAD_F64 => "V_LOAD_F64",
        OP_STORE_F64 => "V_STORE_F64",
        OP_ADD_F64 => "V_ADD_F64",
        OP_SUB_F64 => "V_SUB_F64",
        OP_MUL_F64 => "V_MUL_F64",
        OP_DIV_F64 => "V_DIV_F64",
        OP_NEG_F64 => "V_NEG_F64",
        OP_RSQ_F64 => "V_RSQ_F64",
        OP_DOT3_F64 => "V_DOT3_F64",
        OP_CROSS_F64 => "V_CROSS_F64",
        OP_MIN_F64 => "V_MIN_F64",
        OP_MAX_F64 => "V_MAX_F64",
        OP_SIN_D => "V_SIN_D",
        OP_COS_D => "V_COS_D",
        OP_TAN_D => "V_TAN_D",
        OP_SQRT_F64 => "V_SQRT_F64",
        OP_ABS_F64 => "V_ABS_F64",
        OP_ROUND_F64 => "V_ROUND_F64",
        OP_CONST_PI => "V_CONST_PI",
        OP_UNIT_MM => "V_UNIT_MM",
        OP_UNIT_DEG => "V_UNIT_DEG",
        OP_UNIT_PLAIN => "V_UNIT_PLAIN",
        OP_VAR_F64 => "V_VAR_F64",
        OP_JMP => "JMP",
        OP_JNZ => "JNZ",
        OP_HALT => "HALT",
        _ => "V_RESERVED",
    }
}

/// The 32-byte instruction word: all nine v5 ISA fields in exactly 32
/// bytes, 32-byte aligned. (Field *order* is adapted to Rust's alignment
/// rules — `pred` sits at bytes 4–7 instead of 28–31 — because `f64` must
/// land on an 8-byte boundary; every field and the total size/stride match
/// the v5 spec.)
#[repr(C, align(32))]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Insn {
    pub op: u16,
    pub mask: u8,
    pub stride: u8,
    pub pred: u32,
    pub dst: u32,
    pub a: u32,
    pub b: u32,
    pub c: u32,
    pub imm: f64,
}

impl Insn {
    pub fn new(op: u16, dst: u32, a: u32, b: u32, imm: f64) -> Insn {
        Insn { op, mask: 0xFF, stride: 1, dst, a, b, c: 0, imm, pred: 0 }
    }
}

// compile-time size guarantee
const _: () = assert!(std::mem::size_of::<Insn>() == 32, "OTD-ASM instructions are 32 bytes");
const _: () = assert!(std::mem::align_of::<Insn>() == 32, "OTD-ASM instructions align to 32");

/// One line of disassembly (no trailing newline).
pub fn disasm_at(insn: &Insn, var_names: &[String]) -> String {
    let name = op_name(insn.op);
    match insn.op {
        OP_LOAD_F64 => format!("{:<12} r{}, {:.6}", name, insn.dst, insn.imm),
        OP_STORE_F64 => format!("{:<12} out[{}], r{}", name, insn.a, insn.dst),
        OP_UNIT_MM | OP_UNIT_DEG | OP_UNIT_PLAIN | OP_NEG_F64 | OP_SIN_D | OP_COS_D | OP_TAN_D
        | OP_SQRT_F64 | OP_ABS_F64 | OP_ROUND_F64 | OP_RSQ_F64 => {
            format!("{:<12} r{}", name, insn.dst)
        }
        OP_DOT3_F64 | OP_CROSS_F64 | OP_MIN_F64 | OP_MAX_F64 | OP_ADD_F64 | OP_SUB_F64
        | OP_MUL_F64 | OP_DIV_F64 => format!("{:<12} r{}, r{}, r{}", name, insn.dst, insn.a, insn.b),
        OP_VAR_F64 => {
            let vn = var_names.get(insn.a as usize).map(|s| s.as_str()).unwrap_or("?");
            format!("{:<12} r{}, {}", name, insn.dst, vn)
        }
        OP_JMP | OP_JNZ => format!("{:<12} {:04}", name, insn.a),
        OP_HALT => "HALT".to_string(),
        _ => format!("{:<12} (reserved)", name),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn instruction_is_exactly_32_bytes() {
        assert_eq!(std::mem::size_of::<Insn>(), 32);
        assert_eq!(std::mem::align_of::<Insn>(), 32);
    }

    #[test]
    fn disassembly_is_readable() {
        let i = Insn::new(OP_LOAD_F64, 0, 0, 0, 40.0);
        let s = disasm_at(&i, &[]);
        assert!(s.contains("V_LOAD_F64") && s.contains("40"));
        let j = Insn::new(OP_JMP, 0, 7, 0, 0.0);
        assert!(disasm_at(&j, &[]).contains("JMP"));
        let v = Insn::new(OP_VAR_F64, 2, 0, 0, 0.0);
        assert!(disasm_at(&v, &["i".to_string()]).contains("i"));
    }
}
