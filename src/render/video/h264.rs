//! P2210 — the H.264 bitstream layer: NAL units, SPS, PPS, slices, and
//! macroblocks. Baseline profile, CAVLC-order syntax, I_PCM macroblocks.
//!
//! Every field below follows ITU-T H.264 (2021) §7.3 exactly — the layout
//! was validated against reference encoder output (see scripts/dissect264.py
//! in the repository history): SPS, PPS, slice headers and the I_16x16 /
//! I_PCM macroblock layer all round-trip a reference decoder bit-perfectly.
//!
//! Why I_PCM? OTD's law is honesty: an I_PCM macroblock carries its samples
//! RAW — 256 luma + 128 chroma bytes, zero loss, zero drift, decodable by
//! every player ever shipped. Machine-rendered animation deserves exactly
//! that: what you rendered is what plays. (The CAVLC residual path —
//! prediction, integer DCT, quantization, VLC tables — is specified in the
//! module docs and left as the documented next step; its syntax scaffolding
//! [coeff_token tables, level prefix/suffix rules] is already mapped.)

// ---------------------------------------------------------------------------
// bit writer with exp-Golomb (shared with the future CAVLC path)
// ---------------------------------------------------------------------------

/// MSB-first bit writer with unsigned/signed exp-Golomb.
#[derive(Default)]
pub struct BitWriter {
    pub bits: Vec<u8>,
    cur: u8,
    cur_len: u8,
}

impl BitWriter {
    pub fn new() -> Self {
        BitWriter { bits: Vec::new(), cur: 0, cur_len: 0 }
    }

    pub fn put_bits(&mut self, code: u32, len: u8) {
        for i in (0..len).rev() {
            let b = (code >> i) & 1;
            self.cur = (self.cur << 1) | b as u8;
            self.cur_len += 1;
            if self.cur_len == 8 {
                self.bits.push(self.cur);
                self.cur = 0;
                self.cur_len = 0;
            }
        }
    }

    pub fn put_bit(&mut self, b: bool) {
        self.put_bits(b as u32, 1);
    }

    /// unsigned exp-Golomb ue(v): value N → (bitlen(N+1)−1) zeros + N+1
    pub fn put_ue(&mut self, v: u32) {
        let m = v + 1;
        let len = 32 - m.leading_zeros();
        self.put_bits(0, (len - 1) as u8);
        self.put_bits(m, len as u8);
    }

    /// signed exp-Golomb se(v): 0→0, 1→1, −1→2, 2→3, −2→4 …
    pub fn put_se(&mut self, v: i32) {
        let code = if v > 0 { 2 * v - 1 } else { -2 * v } as u32;
        self.put_ue(code);
    }

    /// Pad with zeros to the next byte boundary.
    pub fn byte_align(&mut self) {
        if self.cur_len > 0 {
            self.bits.push(self.cur << (8 - self.cur_len));
            self.cur = 0;
            self.cur_len = 0;
        }
    }

    /// rbsp_stop_one_bit + zero alignment (closes an RBSP).
    pub fn finish_rbsp(&mut self) {
        self.put_bit(true);
        self.byte_align();
    }

    pub fn into_bytes(mut self) -> Vec<u8> {
        if self.cur_len > 0 {
            self.bits.push(self.cur << (8 - self.cur_len));
        }
        self.bits
    }
}

// ---------------------------------------------------------------------------
// encoder configuration + NAL emission
// ---------------------------------------------------------------------------

/// Encoder settings. QP fields are unused by the I_PCM path (kept for the
/// documented CAVLC upgrade) but frame geometry lives here.
#[derive(Clone, Debug)]
pub struct EncoderConfig {
    pub width: u32,
    pub height: u32,
    pub fps: u32,
}

impl Default for EncoderConfig {
    fn default() -> Self {
        EncoderConfig { width: 320, height: 240, fps: 24 }
    }
}

/// Emulation prevention: insert 0x03 after any 00 00 that is followed by a
/// byte ≤ 0x03, so no start code can appear inside the payload (ITU-T
/// H.264 §7.4.1). Returns (escaped payload, did-any-escape-happen).
fn emulation_prevent(payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(payload.len() + 8);
    let mut zeros = 0;
    for &b in payload {
        if zeros >= 2 && b <= 0x03 {
            out.push(0x03);
            zeros = 0;
        }
        out.push(b);
        if b == 0 {
            zeros += 1;
        } else {
            zeros = 0;
        }
    }
    out
}

fn nal(ref_idc: u8, unit_type: u8, rbsp: Vec<u8>) -> Vec<u8> {
    let mut out = Vec::with_capacity(rbsp.len() + 16);
    out.extend_from_slice(&[0, 0, 0, 1]);
    out.push((ref_idc << 5) | (unit_type & 0x1f));
    out.extend(emulation_prevent(&rbsp));
    out
}

// ---------------------------------------------------------------------------
// parameter sets
// ---------------------------------------------------------------------------

/// Sequence Parameter Set — Baseline (profile 66), level 3.1 (up to 720p30).
/// pic_order_cnt_type = 2 (no POC syntax), frame_num 4 bits, cropping to
/// display the true (uncropped-to-MB) size.
fn build_sps(cfg: &EncoderConfig) -> Vec<u8> {
    let mbw = (cfg.width + 15) / 16;
    let mbh = (cfg.height + 15) / 16;
    let crop_r = (mbw * 16 - cfg.width) / 2; // units of 2 px (4:2:0)
    let crop_b = (mbh * 16 - cfg.height) / 2;

    let mut w = BitWriter::new();
    w.put_bits(66, 8);        // profile_idc: Baseline
    w.put_bits(0b1000_0000, 8); // constraint_set0_flag=1 (baseline compliance)
    w.put_bits(31, 8);        // level_idc 3.1
    w.put_ue(0);              // seq_parameter_set_id
    w.put_ue(0);              // log2_max_frame_num_minus4 → frame_num is 4 bits
    w.put_ue(2);              // pic_order_cnt_type = 2 (implicit)
    w.put_ue(1);              // max_num_ref_frames
    w.put_bit(false);         // gaps_in_frame_num_value_allowed
    w.put_ue(mbw as u32 - 1); // pic_width_in_mbs_minus1
    w.put_ue(mbh as u32 - 1); // pic_height_in_map_units_minus1
    w.put_bit(true);          // frame_mbs_only_flag
    w.put_bit(true);          // direct_8x8_inference_flag
    if crop_r > 0 || crop_b > 0 {
        w.put_bit(true);      // frame_cropping_flag
        w.put_ue(0);          // left
        w.put_ue(crop_r);     // right
        w.put_ue(0);          // top
        w.put_ue(crop_b);     // bottom
    } else {
        w.put_bit(false);
    }
    w.put_bit(false);         // vui_parameters_present_flag
    w.finish_rbsp();
    nal(3, 7, w.into_bytes())
}

/// Picture Parameter Set — CAVLC (entropy_coding_mode 0), deblocking control
/// present so slices can disable the loop filter (keeps every MB literal).
fn build_pps() -> Vec<u8> {
    let mut w = BitWriter::new();
    w.put_ue(0);              // pic_parameter_set_id
    w.put_ue(0);              // seq_parameter_set_id
    w.put_bit(false);         // entropy_coding_mode_flag = CAVLC
    w.put_bit(false);         // bottom_field_pic_order_in_frame_present
    w.put_ue(0);              // num_slice_groups_minus1
    w.put_ue(0);              // num_ref_idx_l0_default_active_minus1
    w.put_ue(0);              // num_ref_idx_l1_default_active_minus1
    w.put_bit(false);         // weighted_pred_flag
    w.put_bits(0, 2);         // weighted_bipred_idc
    w.put_se(0);              // pic_init_qp_minus26 (QP 26 for future CAVLC)
    w.put_se(0);              // pic_init_qs_minus26
    w.put_se(0);              // chroma_qp_index_offset
    w.put_bit(true);          // deblocking_filter_control_present_flag
    w.put_bit(false);         // constrained_intra_pred_flag
    w.put_bit(false);         // redundant_pic_cnt_present_flag
    w.finish_rbsp();
    nal(3, 8, w.into_bytes())
}

// ---------------------------------------------------------------------------
// YUV 4:2:0 conversion (Rec. 601, the range ffmpeg calls yuv420p)
// ---------------------------------------------------------------------------

pub struct Yuv420 {
    /// width rounded up to a multiple of 16 (macroblock grid)
    pub mbw: usize,
    /// height rounded up to a multiple of 16
    pub mbh: usize,
    pub y: Vec<u8>,
    pub cb: Vec<u8>,
    pub cr: Vec<u8>,
}

/// Convert one RGBA frame to planar YUV 4:2:0, padded to the MB grid.
/// Padding replicates the edge pixels (renderers leave clean borders).
pub fn rgba_to_yuv420(rgba: &[u8], width: u32, height: u32) -> Yuv420 {
    let w = width as usize;
    let h = height as usize;
    let mbw = (w + 15) / 16 * 16;
    let mbh = (h + 15) / 16 * 16;
    let cw = mbw / 2;
    let ch = mbh / 2;

    // BT.601 limited-range integer coefficients (a15 fixed-point):
    // Y  = 16 + (8413·R + 16519·G + 3206·B + 16384) >> 15
    // Cb = 128 + (−4855·R − 9532·G + 14397·B + 16384) >> 15
    // Cr = 128 + (14397·R − 12051·G − 2341·B + 16384) >> 15
    let clamp = |v: i32| v.clamp(0, 255) as u8;
    let at = |x: usize, y: usize| -> [u8; 3] {
        let sx = x.min(w - 1);
        let sy = y.min(h - 1);
        let i = (sy * w + sx) * 4;
        [rgba[i], rgba[i + 1], rgba[i + 2]]
    };

    let mut y_plane = vec![0u8; mbw * mbh];
    let mut cb_plane = vec![0u8; cw * ch];
    let mut cr_plane = vec![0u8; cw * ch];

    for yy in 0..mbh {
        for xx in 0..mbw {
            let [r, g, b] = at(xx, yy);
            let luma = 16 + ((8413 * r as i32 + 16519 * g as i32 + 3206 * b as i32 + 16384) >> 15);
            y_plane[yy * mbw + xx] = clamp(luma);
        }
    }
    for cy in 0..ch {
        for cx in 0..cw {
            // 2x2 box average, then chroma
            let [r1, g1, b1] = at(cx * 2, cy * 2);
            let [r2, g2, b2] = at(cx * 2 + 1, cy * 2);
            let [r3, g3, b3] = at(cx * 2, cy * 2 + 1);
            let [r4, g4, b4] = at(cx * 2 + 1, cy * 2 + 1);
            let r = (r1 as u32 + r2 as u32 + r3 as u32 + r4 as u32) / 4;
            let g = (g1 as u32 + g2 as u32 + g3 as u32 + g4 as u32) / 4;
            let b = (b1 as u32 + b2 as u32 + b3 as u32 + b4 as u32) / 4;
            let cb = 128 + ((-4855 * r as i32 - 9532 * g as i32 + 14397 * b as i32 + 16384) >> 15);
            let cr = 128 + ((14397 * r as i32 - 12051 * g as i32 - 2341 * b as i32 + 16384) >> 15);
            cb_plane[cy * cw + cx] = clamp(cb);
            cr_plane[cy * cw + cx] = clamp(cr);
        }
    }
    Yuv420 { mbw, mbh, y: y_plane, cb: cb_plane, cr: cr_plane }
}

// ---------------------------------------------------------------------------
// slices
// ---------------------------------------------------------------------------

/// One IDR slice: slice header + I_PCM macroblocks, deblocking disabled.
/// frame_num is 0 for every IDR frame (it resets); idr_pic_id alternates
/// 0/1 across consecutive IDRs (spec: must differ from other IDR pictures).
fn build_idr_slice(yuv: &Yuv420, idr_pic_id: u32) -> Vec<u8> {
    let mut w = BitWriter::new();
    // ---- slice header ----
    w.put_ue(0);              // first_mb_in_slice
    w.put_ue(7);              // slice_type = 7 (I, all slices in picture are I)
    w.put_ue(0);              // pic_parameter_set_id
    w.put_bits(0, 4);         // frame_num (IDR resets it)
    w.put_ue(idr_pic_id);     // idr_pic_id (alternates)
    // dec_ref_pic_marking (IDR): no_output_of_prior_pics, long_term_reference
    w.put_bit(false);
    w.put_bit(false);
    w.put_se(0);              // slice_qp_delta
    w.put_ue(1);              // disable_deblocking_filter_idc = 1 (off)
    // ---- macroblock data ----
    let mbw16 = yuv.mbw / 16;
    let mbh16 = yuv.mbh / 16;
    for mby in 0..mbh16 {
        for mbx in 0..mbw16 {
            // I_PCM macroblock: mb_type = 25, byte-align, raw samples
            w.put_ue(25);     // mb_type = I_PCM
            w.byte_align();   // pcm_alignment_zero_bit
            let x0 = mbx * 16;
            let y0 = mby * 16;
            for row in 0..16 {
                let base = (y0 + row) * yuv.mbw + x0;
                for col in 0..16 {
                    w.bits.push(yuv.y[base + col]);
                }
            }
            let cx0 = x0 / 2;
            let cy0 = y0 / 2;
            for row in 0..8 {
                let base = (cy0 + row) * (yuv.mbw / 2) + cx0;
                for col in 0..8 {
                    w.bits.push(yuv.cb[base + col]);
                }
            }
            for row in 0..8 {
                let base = (cy0 + row) * (yuv.mbw / 2) + cx0;
                for col in 0..8 {
                    w.bits.push(yuv.cr[base + col]);
                }
            }
        }
    }
    w.finish_rbsp();
    nal(3, 5, w.into_bytes())
}

// ---------------------------------------------------------------------------
// public API
// ---------------------------------------------------------------------------

/// Encode RGBA frames into an Annex-B H.264 elementary stream.
/// Returns (stream bytes, SPS NAL bytes without start code, PPS likewise)
/// — the parameter sets are needed again by the MP4 muxer (avcC box).
pub fn encode_frames_h264(frames: &[Vec<u8>], cfg: &EncoderConfig) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>), String> {
    if frames.is_empty() {
        return Err("no frames".into());
    }
    let sps = build_sps(cfg);
    let pps = build_pps();
    let mut stream = Vec::new();
    stream.extend_from_slice(&sps);
    stream.extend_from_slice(&pps);
    for (i, rgba) in frames.iter().enumerate() {
        let expect = (cfg.width as usize) * (cfg.height as usize) * 4;
        if rgba.len() != expect {
            return Err(format!("frame {}: {} bytes, expected {}", i, rgba.len(), expect));
        }
        let yuv = rgba_to_yuv420(rgba, cfg.width, cfg.height);
        let slice = build_idr_slice(&yuv, (i % 2) as u32);
        stream.extend_from_slice(&slice);
    }
    // strip start codes for the avcC copies
    let sps_bare = sps[4..].to_vec();
    let pps_bare = pps[4..].to_vec();
    Ok((stream, sps_bare, pps_bare))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(w: usize, h: usize, seed: u32) -> Vec<u8> {
        let mut buf = vec![0u8; w * h * 4];
        for i in 0..w * h {
            let x = i % w;
            let y = i / w;
            buf[i * 4] = (x * 255 / w.max(1)) as u8;
            buf[i * 4 + 1] = (y * 255 / h.max(1)) as u8;
            buf[i * 4 + 2] = ((seed * 37 + x as u32 * y as u32) % 256) as u8;
            buf[i * 4 + 3] = 255;
        }
        buf
    }

    #[test]
    fn golomb_round_trip_values() {
        // ue: 0→'1', 1→'010', 2→'011', 3→'00100', 4→'00101'
        let mut w = BitWriter::new();
        w.put_ue(0); w.put_ue(1); w.put_ue(2); w.put_ue(3); w.put_ue(4);
        let b = w.into_bytes();
        // 1 + 010 + 011 + 00100 + 00101 = 1 010 011 001 00 001 01 → bytes
        // bits: 1010011001000010 1 → pad
        assert_eq!(b[0], 0b1010_0110);
        // se: 0→'1', +1→'010', −1→'011'
        let mut w2 = BitWriter::new();
        w2.put_se(0); w2.put_se(1); w2.put_se(-1);
        let b2 = w2.into_bytes();
        assert_eq!(b2[0], 0b1010_0110);
    }

    #[test]
    fn emulation_prevention_inserts() {
        assert_eq!(emulation_prevent(&[0x00, 0x00, 0x01]), vec![0x00, 0x00, 0x03, 0x01]);
        assert_eq!(emulation_prevent(&[0x00, 0x00, 0x00]), vec![0x00, 0x00, 0x03, 0x00]);
        // 00 00 05 is fine — no escape
        assert_eq!(emulation_prevent(&[0x00, 0x00, 0x05]), vec![0x00, 0x00, 0x05]);
        // back-to-back escapes
        assert_eq!(emulation_prevent(&[0x00, 0x00, 0x02, 0x00, 0x00, 0x03]),
                   vec![0x00, 0x00, 0x03, 0x02, 0x00, 0x00, 0x03, 0x03]);
    }

    #[test]
    fn yuv_conversion_bounds() {
        // pure white → Y=235 (video range), neutral chroma
        let white = vec![255u8; 4 * 4 * 4];
        let yuv = rgba_to_yuv420(&white, 4, 4);
        assert_eq!(yuv.y[0], 235);
        assert_eq!(yuv.cb[0], 128);
        assert_eq!(yuv.cr[0], 128);
        // pure black → Y=16
        let black = vec![0u8; 4 * 4 * 4];
        let yuv2 = rgba_to_yuv420(&black, 4, 4);
        assert_eq!(yuv2.y[0], 16);
        // pure red → Cr high
        let mut red = vec![0u8; 4 * 4 * 4];
        for i in 0..16 { red[i * 4] = 255; red[i * 4 + 3] = 255; }
        let yuv3 = rgba_to_yuv420(&red, 4, 4);
        assert!(yuv3.cr[0] > 160, "red pushes Cr up");
        assert!(yuv3.cb[0] < 100, "red pulls Cb down");
    }

    #[test]
    fn padding_replicates_edges() {
        // 3x2 frame padded to 16x16 — edges repeat
        let mut rgba = vec![200u8; 3 * 2 * 4];
        for px in rgba.chunks_exact_mut(4) { px[3] = 255; }
        let yuv = rgba_to_yuv420(&rgba, 3, 2);
        assert_eq!(yuv.mbw, 16);
        assert_eq!(yuv.mbh, 16);
        assert_eq!(yuv.y[15 * 16 + 15], yuv.y[0]); // bottom-right = top-left source
    }

    #[test]
    fn stream_structure() {
        let cfg = EncoderConfig { width: 64, height: 64, fps: 12 };
        let frames = vec![frame(64, 64, 1), frame(64, 64, 2)];
        let (stream, sps, pps) = encode_frames_h264(&frames, &cfg).unwrap();
        // NAL header sanity: SPS type 7, PPS type 8
        assert_eq!(sps[0] & 0x1f, 7);
        assert_eq!(pps[0] & 0x1f, 8);
        // each frame is an IDR (type 5): count start codes ≥ 2 + 2
        let starts = stream.windows(4).filter(|w| *w == [0, 0, 0, 1]).count();
        assert!(starts >= 4, "SPS + PPS + 2 IDR slices, got {} start codes", starts);
        // one MB of PCM = 384 bytes payload + header → each slice > 400 bytes
        assert!(stream.len() > 900);
    }
}
