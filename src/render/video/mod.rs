//! P2200 — OTD3 VIDEO: a real H.264/AVC encoder, written from nothing.
//!
//! The zero-dependency rule holds: no external codec, no libx264, no ffmpeg
//! at runtime. What ships here is a genuine Baseline-profile H.264
//! intra-picture encoder — the same arithmetic every AVC file on Earth is
//! made of:
//!
//!   * YUV 4:2:0 conversion (Rec. 601) from the renderer's RGBA
//!   * Intra prediction: I_16x16 (vertical / horizontal / DC / plane) luma,
//!     DC chroma — predicted from the encoder's own reconstruction, exactly
//!     the way the decoder will
//!   * The integer 4x4 transform + the 4x4 Hadamard for Intra16 DC
//!   * Quantization with the standard MF/V tables (QP 12–51)
//!   * CAVLC entropy coding: coeff_token, trailing-one signs, levels
//!     (prefix/suffix with 6-bit growth), total_zeros, run_before
//!   * NAL units: SPS + PPS + IDR slices, byte-aligned, emulation
//!     prevention inserted — a real Annex-B elementary stream
//!
//! All frames are intra (IDR) — every frame stands alone, which is exactly
//! right for machine-rendered turntable and screw animations: seekable to
//! any frame, immune to drift.

pub mod cavlc;
pub mod h264;
pub mod mp4;

pub use h264::{encode_frames_h264, EncoderConfig};

/// Convenience: RGBA frames in → MP4 (H.264 + yuv420 avc1) file bytes out.
pub fn encode_mp4(frames: &[Vec<u8>], width: u32, height: u32, fps: u32) -> Result<Vec<u8>, String> {
    if frames.is_empty() {
        return Err("no frames to encode".into());
    }
    if width == 0 || height == 0 || width % 2 != 0 || height % 2 != 0 {
        return Err(format!("width/height must be positive even numbers (got {}x{})", width, height));
    }
    let expect = (width as usize) * (height as usize) * 4;
    for (i, f) in frames.iter().enumerate() {
        if f.len() != expect {
            return Err(format!("frame {} has {} bytes, expected {} (RGBA {}x{})", i, f.len(), expect, width, height));
        }
    }
    let cfg = EncoderConfig { width, height, fps: fps.max(1), ..Default::default() };
    let (nal_stream, sps, pps) = h264::encode_frames_h264(frames, &cfg)?;
    let mp4 = mp4::mux_h264_to_mp4(&nal_stream, width, height, fps.max(1), &sps, &pps)?;
    Ok(mp4)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gradient(w: usize, h: usize, t: u32) -> Vec<u8> {
        let mut buf = vec![0u8; w * h * 4];
        for y in 0..h {
            for x in 0..w {
                let i = (y * w + x) * 4;
                buf[i] = ((x * 255) / w.max(1)) as u8;
                buf[i + 1] = ((y * 255) / h.max(1)) as u8;
                buf[i + 2] = ((t * 37 % 256)) as u8;
                buf[i + 3] = 255;
            }
        }
        buf
    }

    #[test]
    fn mp4_container_structure() {
        let frames: Vec<Vec<u8>> = (0..8).map(|t| gradient(160, 128, t)).collect();
        let bytes = encode_mp4(&frames, 160, 128, 24).unwrap();
        assert_eq!(&bytes[4..8], b"ftyp");
        assert!(bytes.windows(4).any(|w| w == b"moov"), "moov box missing");
        assert!(bytes.windows(4).any(|w| w == b"mdat"), "mdat box missing");
        assert!(bytes.len() > 1000, "suspiciously small mp4: {} bytes", bytes.len());
    }

    #[test]
    fn h264_stream_has_avc_nals() {
        let frames: Vec<Vec<u8>> = (0..2).map(|t| gradient(64, 64, t)).collect();
        let cfg = EncoderConfig { width: 64, height: 64, fps: 12, ..Default::default() };
        let (nals, _, _) = h264::encode_frames_h264(&frames, &cfg).unwrap();
        assert!(nals.windows(4).any(|w| w == &[0, 0, 0, 1]), "4-byte start codes present");
        assert!(nals.windows(3).any(|w| w == &[0, 0, 1]), "3-byte start codes present");
    }
}
