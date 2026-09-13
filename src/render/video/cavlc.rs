//! P2210 — CAVLC: Context-Adaptive Variable-Length Coding (H.264 §9.3).
//!
//! STATUS: syntax scaffolding + empirically verified entries. The entries
//! marked VERIFIED below were extracted from reference-encoder bitstreams
//! (scripts/extract3.py methodology — crafted single-macroblock residuals,
//! exhaustive hypothesis search over the VLC space, cross-checked against
//! decoded pixel ground truth):
//!
//!   * coeff_token (0,0) → "1"                     [every table]
//!   * coeff_token (0,1) → "000101"                [nC < 2]
//!   * total_zeros (tc=1, zeros=0) → "1"
//!   * level VLC mechanics: prefix = N zeros + 1;  suffixLength starts 0
//!     (T1s < 3); 4-bit suffix at prefix 14/sL 0; 12-bit escape at 15;
//!     parity carries the sign; first-level adjustment when T1s < 3
//!   * run_before reads zeros_left-capped VLCs, last run implicit
//!
//! The full table set (ITU-T H.264 Tables 9-5…9-11, all five nC classes,
//! plus the 2×2 chroma-DC variant of total_zeros) is the documented next
//! step for the intra-CAVLC residual path; until then the encoder ships
//! the I_PCM macroblock mode — bit-exact, lossless, universally decodable
//! — and this module carries the scaffolding so the upgrade lands here.

/// Verified coeff_token entries, table 9-5 (0 ≤ nC < 2, incl. nC = −1 for
/// 4:2:0 chroma DC). (trailing_ones, total_coeff) → (code, bit length).
pub const TOKEN_NC_LOW_VERIFIED: &[((u8, u8), (u32, u8))] = &[
    ((0, 0), (0b1, 1)),       // empty block — 1 bit
    ((0, 1), (0b000101, 6)),  // one non-trailing coefficient
];

/// Verified total_zeros entries (Table 9-7, 4×4 blocks):
/// (total_coeff, total_zeros) → (code, len).
pub const TOTAL_ZEROS_VERIFIED: &[((u8, u8), (u32, u8))] = &[
    ((1, 0), (0b1, 1)),
];

/// Verified level-coding facts (see module doc). The level VLC:
/// prefix N = N zeros followed by a 1; suffix bits only when suffixLength
/// > 0 or prefix ≥ 14 (then 4 bits at prefix 14/sL 0, 12 bits at 15).
pub const LEVEL_ESCAPE_SUFFIX_14: u8 = 4;
pub const LEVEL_ESCAPE_SUFFIX_15: u8 = 12;

/// The five nC classes that select coeff_token tables (spec §9.3.1.1):
/// 4×4 blocks use nC = nA + nB (neighbour nonzero counts); chroma DC 2×2
/// in 4:2:0 uses nC = −1 which shares the 0 ≤ nC < 2 table.
pub fn nc_class(nc: i32) -> u8 {
    match nc {
        i32::MIN..=1 => 0, // nC = −1, 0, 1 → Table 9-5
        2..=3 => 1,        // Table 9-6
        4..=7 => 2,        // Table 9-7
        8..=15 => 3,       // Table 9-8
        _ => 4,            // nC ≥ 16 → Table 9-9
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn verified_entries_are_prefix_consistent() {
        // the two verified 6-bit-ish codes must not prefix-collide with '1'
        let ((_, (c0, l0)), (_, (c1, l1))) = (TOKEN_NC_LOW_VERIFIED[0], TOKEN_NC_LOW_VERIFIED[1]);
        assert_eq!(l0, 1);
        assert_eq!(c0, 1);
        assert!(c1 >> (l1 - 1) == 0, "code must start with 0 to coexist with the 1-bit empty token");
    }

    #[test]
    fn nc_classes_match_spec() {
        assert_eq!(nc_class(-1), 0, "chroma DC shares the nC<2 table");
        assert_eq!(nc_class(0), 0);
        assert_eq!(nc_class(1), 0);
        assert_eq!(nc_class(2), 1);
        assert_eq!(nc_class(3), 1);
        assert_eq!(nc_class(4), 2);
        assert_eq!(nc_class(7), 2);
        assert_eq!(nc_class(8), 3);
        assert_eq!(nc_class(15), 3);
        assert_eq!(nc_class(16), 4);
        assert_eq!(nc_class(32), 4);
    }
}
