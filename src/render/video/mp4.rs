//! P2220 — MP4 (ISO Base Media File Format) muxer for the H.264 stream.
//!
//! A playable video needs its container as much as its codec. This muxer
//! writes the minimal complete set of boxes every player reads:
//!
//!   ftyp  — "isom/iso2/avc1/mp41" brand ladder (what players check first)
//!   mdat  — the Annex-B H.264 stream, byte-for-byte
//!   moov  — movie header: one video track, avc1 sample entry carrying the
//!           avcC record (SPS+PPS, so the decoder is configured from the
//!           container), stbl sample tables (stts/stsc/stsz/stco), one
//!           sample per frame, all sync samples (every frame is IDR)
//!
//! Zero dependencies, big-endian field order per ISO/IEC 14496-12/-14.

/// 4-byte box header helpers.
fn be32(v: u32) -> [u8; 4] { v.to_be_bytes() }
fn be16(v: u16) -> [u8; 2] { v.to_be_bytes() }

/// Build one box: size(4) + type(4) + payload.
fn box_bytes(kind: &[u8; 4], payload: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(8 + payload.len());
    out.extend_from_slice(&be32(8 + payload.len() as u32));
    out.extend_from_slice(kind);
    out.extend_from_slice(payload);
    out
}

/// Full box: size + type + version(1) + flags(3) + payload.
fn full_box(kind: &[u8; 4], version: u8, flags: u32, payload: &[u8]) -> Vec<u8> {
    let mut body = Vec::with_capacity(4 + payload.len());
    body.push(version);
    body.extend_from_slice(&flags.to_be_bytes()[1..4]);
    body.extend_from_slice(payload);
    box_bytes(kind, &body)
}

/// Split an Annex-B stream into NAL units (returns payloads without start
/// codes). Used to build per-frame samples (each IDR slice = one sample).
fn split_annexb(stream: &[u8]) -> Vec<Vec<u8>> {
    let mut nals = Vec::new();
    let mut i = 0usize;
    let mut starts: Vec<(usize, usize)> = Vec::new(); // (pos, start-code len)
    while i + 3 < stream.len() {
        if stream[i] == 0 && stream[i + 1] == 0 && stream[i + 2] == 1 {
            starts.push((i, 3));
            i += 3;
        } else if i + 4 <= stream.len() && stream[i] == 0 && stream[i + 1] == 0
            && stream[i + 2] == 0 && stream[i + 3] == 1 {
            starts.push((i, 4));
            i += 4;
        } else {
            i += 1;
        }
    }
    for (idx, (pos, sc)) in starts.iter().enumerate() {
        let end = if idx + 1 < starts.len() { starts[idx + 1].0 } else { stream.len() };
        nals.push(stream[pos + sc..end].to_vec());
    }
    nals
}

/// The avcC record (AVCDecoderConfigurationRecord, ISO/IEC 14496-15):
/// SPS/PPS + NAL length size (= 4 bytes) so mdat samples use length
/// prefixes instead of start codes.
fn build_avcc(sps: &[u8], pps: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.push(1);                     // configurationVersion
    out.push(sps[1]);                // AVCProfileIndication (from SPS payload)
    out.push(sps[2]);                // profile_compatibility
    out.push(sps[3]);                // AVCLevelIndication
    out.push(0xff);                  // lengthSizeMinusOne = 3 (4-byte lengths)
    out.push(0xe1);                  // numOfSequenceParameterSets = 1
    out.extend_from_slice(&be16(sps.len() as u16));
    out.extend_from_slice(sps);
    out.push(1);                     // numOfPictureParameterSets
    out.extend_from_slice(&be16(pps.len() as u16));
    out.extend_from_slice(pps);
    out
}

/// Mux an Annex-B H.264 stream (SPS/PPS/IDR slices) into a playable MP4.
///
/// Each IDR slice becomes one sample; every sample is a sync sample.
/// `fps` sets the fixed tick (one frame per delta).
pub fn mux_h264_to_mp4(
    annexb: &[u8],
    width: u32,
    height: u32,
    fps: u32,
    sps: &[u8],
    pps: &[u8],
) -> Result<Vec<u8>, String> {
    if annexb.is_empty() {
        return Err("empty H.264 stream".into());
    }
    let nals = split_annexb(annexb);
    // sample NALs = the IDR slices (type 5); SPS/PPS go in avcC
    let frame_nals: Vec<&Vec<u8>> = nals.iter()
        .filter(|n| !n.is_empty() && (n[0] & 0x1f) == 5)
        .collect();
    if frame_nals.is_empty() {
        return Err("no IDR frames in stream".into());
    }

    // ---- mdat: length-prefixed NAL samples ----
    let mut mdat: Vec<u8> = Vec::new();
    let mut sizes: Vec<u32> = Vec::with_capacity(frame_nals.len());
    for nal in &frame_nals {
        sizes.push((4 + nal.len()) as u32);
        mdat.extend_from_slice(&be32(nal.len() as u32));
        mdat.extend_from_slice(nal);
    }
    let mdat_box = box_bytes(b"mdat", &mdat);
    let mdat_offset = 0u32; // patched after ftyp is sized — computed below

    // ---- moov ----
    let n = sizes.len() as u32;
    let dur = n * 1000 / fps.max(1); // timescale 1000 → duration in ms
    let mdia_dur = n * 1000 / fps.max(1);

    // avc1 sample entry (visual sample entry)
    let avcc = build_avcc(sps, pps);
    let mut avc1_body = Vec::new();
    // SampleEntry: reserved[6] + data_reference_index[2]
    avc1_body.extend_from_slice(&[0u8; 6]);
    avc1_body.extend_from_slice(&be16(1)); // data_reference_index
    // VisualSampleEntry: pre_defined[2] + reserved[2] + pre_defined[12]
    avc1_body.extend_from_slice(&[0u8; 16]);
    avc1_body.extend_from_slice(&be16(width as u16));
    avc1_body.extend_from_slice(&be16(height as u16));
    avc1_body.extend_from_slice(&be32(0x00480000)); // horiz res 72 dpi (16.16)
    avc1_body.extend_from_slice(&be32(0x00480000)); // vert res
    avc1_body.extend_from_slice(&be32(0));          // reserved
    avc1_body.extend_from_slice(&be16(1));          // frame count
    avc1_body.extend_from_slice(&[0u8; 32]);        // compressor name
    avc1_body.extend_from_slice(&be16(0x0018));     // depth
    avc1_body.extend_from_slice(&be16(0xffff));     // pre_defined −1
    // avcC rides inside avc1 as its own box: size + 'avcC' + record
    let avcc_box = box_bytes(b"avcC", &avcc);
    let avc1 = box_bytes(b"avc1", &{
        let mut b = avc1_body.clone();
        b.extend_from_slice(&avcc_box);
        b
    });

    // stsd (one entry)
    let stsd_payload = {
        let mut p = Vec::new();
        p.extend_from_slice(&be32(1));
        p.extend_from_slice(&avc1);
        p
    };
    let stsd = full_box(b"stsd", 0, 0, &stsd_payload);

    // stts: (sample_count=1 entry, delta = 1000/fps in timescale 1000)
    let delta = (1000 / fps.max(1)).max(1);
    let stts_payload = {
        let mut p = Vec::new();
        p.extend_from_slice(&be32(1)); // entry count
        p.extend_from_slice(&be32(n));
        p.extend_from_slice(&be32(delta));
        p
    };
    let stts = full_box(b"stts", 0, 0, &stts_payload);

    // stss: all frames are sync (IDR) — list every sample
    let stss_payload = {
        let mut p = Vec::new();
        p.extend_from_slice(&be32(n));
        for i in 1..=n {
            p.extend_from_slice(&be32(i));
        }
        p
    };
    let stss = full_box(b"stss", 0, 0, &stss_payload);

    // stsc: one chunk, all samples contiguous
    let stsc_payload = {
        let mut p = Vec::new();
        p.extend_from_slice(&be32(1));
        p.extend_from_slice(&be32(1)); // first chunk
        p.extend_from_slice(&be32(n)); // samples per chunk
        p.extend_from_slice(&be32(1)); // sample description index
        p
    };
    let stsc = full_box(b"stsc", 0, 0, &stsc_payload);

    // stsz
    let stsz_payload = {
        let mut p = Vec::new();
        p.extend_from_slice(&be32(0)); // sample_size (variable)
        p.extend_from_slice(&be32(n));
        for s in &sizes {
            p.extend_from_slice(&be32(*s));
        }
        p
    };
    let stsz = full_box(b"stsz", 0, 0, &stsz_payload);

    // stco: chunk offset — one chunk starting right after ftyp+moov… we
    // place mdat AFTER moov, so the offset = ftyp_size + moov_size + 8.
    // moov size depends on stco itself only through fixed-width fields, so
    // we can compute it: build moov with a placeholder, measure, rebuild.
    let build_moov = |chunk_offset: u32| -> Vec<u8> {
        let stco_payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&be32(1));
            p.extend_from_slice(&be32(chunk_offset));
            p
        };
        let stco = full_box(b"stco", 0, 0, &stco_payload);
        let mut stbl_payload = Vec::new();
        stbl_payload.extend_from_slice(&stsd);
        stbl_payload.extend_from_slice(&stts);
        stbl_payload.extend_from_slice(&stss);
        stbl_payload.extend_from_slice(&stsc);
        stbl_payload.extend_from_slice(&stsz);
        stbl_payload.extend_from_slice(&stco);
        let stbl = box_bytes(b"stbl", &stbl_payload);

        // mdhd (timescale 1000)
        let mdhd_payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&be32(0)); // creation
            p.extend_from_slice(&be32(0)); // modification
            p.extend_from_slice(&be32(1000)); // timescale
            p.extend_from_slice(&be32(mdia_dur));
            p.extend_from_slice(&be16(0x55c4)); // language 'und'
            p.extend_from_slice(&be16(0));
            p
        };
        let mdhd = full_box(b"mdhd", 0, 0, &mdhd_payload);
        // hdlr — vide
        let hdlr_payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&be32(0)); // pre_defined
            p.extend_from_slice(b"vide");
            p.extend_from_slice(&be32(0)); // reserved ×3
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(b"OTD3\x00"); // name (null-terminated)
            p
        };
        let hdlr = full_box(b"hdlr", 0, 0, &hdlr_payload);
        // minf: vmhd (video media header) + dinf (data reference) + stbl
        let vmhd_payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&be16(0));     // graphicsmode: copy
            p.extend_from_slice(&be16(0));     // opcolor[0]
            p.extend_from_slice(&be16(0));     // opcolor[1]
            p.extend_from_slice(&be16(0));     // opcolor[2]
            p
        };
        let vmhd = full_box(b"vmhd", 0, 1, &vmhd_payload);
        // url box (self-contained data reference: flags=1, empty payload)
        let url_box = full_box(b"url ", 0, 1, &[]);
        let dref_payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&be32(1)); // entry_count
            p.extend_from_slice(&url_box);
            p
        };
        let dref = full_box(b"dref", 0, 0, &dref_payload);
        let dinf = box_bytes(b"dinf", &dref);
        let mut minf_payload = Vec::new();
        minf_payload.extend_from_slice(&vmhd);
        minf_payload.extend_from_slice(&dinf);
        minf_payload.extend_from_slice(&stbl);
        let minf = box_bytes(b"minf", &minf_payload);
        let mut mdia_payload = Vec::new();
        mdia_payload.extend_from_slice(&mdhd);
        mdia_payload.extend_from_slice(&hdlr);
        mdia_payload.extend_from_slice(&minf);
        let mdia = box_bytes(b"mdia", &mdia_payload);

        // tkhd
        let tkhd_payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&be32(0)); // creation
            p.extend_from_slice(&be32(0)); // modification
            p.extend_from_slice(&be32(1)); // track_ID
            p.extend_from_slice(&be32(0)); // reserved
            p.extend_from_slice(&be32(dur));
            p.extend_from_slice(&be32(0)); // reserved ×2
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be16(0)); // layer
            p.extend_from_slice(&be16(0)); // alternate group
            p.extend_from_slice(&be16(0)); // volume (video: 0)
            p.extend_from_slice(&be16(0)); // reserved
            // matrix: identity (16.16 fixed)
            p.extend_from_slice(&be32(0x00010000));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0x00010000));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0x40000000));
            p.extend_from_slice(&be32(width << 16));  // width  (16.16)
            p.extend_from_slice(&be32(height << 16)); // height
            p
        };
        let tkhd = full_box(b"tkhd", 0, 7, &tkhd_payload); // flags 7: enabled+in movie+in preview

        let mut trak_payload = Vec::new();
        trak_payload.extend_from_slice(&tkhd);
        trak_payload.extend_from_slice(&mdia);
        let trak = box_bytes(b"trak", &trak_payload);

        // mvhd
        let mvhd_payload = {
            let mut p = Vec::new();
            p.extend_from_slice(&be32(0)); // creation
            p.extend_from_slice(&be32(0)); // modification
            p.extend_from_slice(&be32(1000)); // timescale
            p.extend_from_slice(&be32(dur));
            p.extend_from_slice(&be32(0x00010000)); // rate 1.0
            p.extend_from_slice(&be16(0x0100)); // volume 1.0
            p.extend_from_slice(&be16(0)); // reserved
            p.extend_from_slice(&be32(0)); // reserved ×2
            p.extend_from_slice(&be32(0));
            // matrix identity
            p.extend_from_slice(&be32(0x00010000));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0x00010000));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0));
            p.extend_from_slice(&be32(0x40000000));
            p.extend_from_slice(&[0u8; 24]); // pre_defined[6]
            p.extend_from_slice(&be32(2)); // next_track_ID
            p
        };
        let mvhd = full_box(b"mvhd", 0, 0, &mvhd_payload);

        let mut moov_payload = Vec::new();
        moov_payload.extend_from_slice(&mvhd);
        moov_payload.extend_from_slice(&trak);
        box_bytes(b"moov", &moov_payload)
    };

    // ftyp
    let ftyp_payload = {
        let mut p = Vec::new();
        p.extend_from_slice(b"isom");
        p.extend_from_slice(&be32(512));
        p.extend_from_slice(b"isomiso2avc1mp41");
        p
    };
    let ftyp = box_bytes(b"ftyp", &ftyp_payload);
    let _ = mdat_offset;

    // two-pass: measure moov with placeholder offset, then rebuild
    let probe = build_moov(0);
    let chunk_offset = (ftyp.len() + probe.len() + 8) as u32;
    let moov = build_moov(chunk_offset);

    let mut out = Vec::with_capacity(ftyp.len() + moov.len() + mdat_box.len());
    out.extend_from_slice(&ftyp);
    out.extend_from_slice(&moov);
    out.extend_from_slice(&mdat_box);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render::video::h264::{encode_frames_h264, EncoderConfig};

    #[test]
    fn annexb_split_finds_nals() {
        let stream = [0, 0, 0, 1, 0x67, 0x42, 0, 0, 1, 0x68, 0xce, 0, 0, 0, 1, 0x65, 0x88];
        let nals = split_annexb(&stream);
        assert_eq!(nals.len(), 3);
        assert_eq!(nals[0][0] & 0x1f, 7);
        assert_eq!(nals[1][0] & 0x1f, 8);
        assert_eq!(nals[2][0] & 0x1f, 5);
    }

    #[test]
    fn avcc_record_layout() {
        let sps = [0x67u8, 66, 0x80, 31, 0xaa, 0xbb];
        let pps = [0x68u8, 0xce, 0x32];
        let avcc = build_avcc(&sps, &pps);
        assert_eq!(avcc[0], 1);
        assert_eq!(avcc[1], 66);
        assert_eq!(avcc[4], 0xff);
        assert_eq!(avcc[5], 0xe1);
        assert_eq!(u16::from_be_bytes([avcc[6], avcc[7]]), 6);
        assert_eq!(avcc[8], 0x67);
    }

    #[test]
    fn moov_box_size_is_stable() {
        // the two-pass offset patch must not change moov's size
        let frames: Vec<Vec<u8>> = (0..3).map(|t| {
            let mut v = vec![0u8; 64 * 64 * 4];
            for i in 0..64 * 64 {
                v[i * 4] = t as u8 * 40;
                v[i * 4 + 3] = 255;
            }
            v
        }).collect();
        let cfg = EncoderConfig { width: 64, height: 64, fps: 10 };
        let (stream, sps, pps) = encode_frames_h264(&frames, &cfg).unwrap();
        let mp4 = mux_h264_to_mp4(&stream, 64, 64, 10, &sps, &pps).unwrap();
        // ftyp at 0, moov follows immediately
        assert_eq!(&mp4[4..8], b"ftyp");
        // moov starts right after the 32-byte ftyp box
        let moov_size = u32::from_be_bytes(mp4[32..36].try_into().unwrap()) as usize;
        assert_eq!(&mp4[32 + 4..32 + 8], b"moov");
        // ftyp: size + 'ftyp' + brand(4) + version(4) + compat(16) = 32
        let ftyp_size = u32::from_be_bytes(mp4[0..4].try_into().unwrap()) as usize;
        assert_eq!(ftyp_size, 32);
        let mdat_pos = ftyp_size + moov_size as usize;
        assert_eq!(&mp4[mdat_pos + 4..mdat_pos + 8], b"mdat");
    }
}
