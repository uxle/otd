//! 16-bit mono PCM WAV encoder + decoder (OTD3: native decode so the CLI
//! can analyse real recordings, not only live microphone frames).

/// Decode a RIFF/WAVE file into (mono f32 PCM, sample_rate).
/// Accepts 16-bit PCM mono or stereo (stereo is averaged); 8/24/32-bit
/// float WAVs are rejected with a clear message.
pub fn decode_wav(bytes: &[u8]) -> Result<(Vec<f32>, u32), String> {
    if bytes.len() < 44 || &bytes[0..4] != b"RIFF" || &bytes[8..12] != b"WAVE" {
        return Err("not a RIFF/WAVE file".into());
    }
    let mut pos = 12usize;
    let mut format: Option<u16> = None;
    let mut channels: Option<u16> = None;
    let mut rate: Option<u32> = None;
    let mut bits: Option<u16> = None;
    let mut data: Option<&[u8]> = None;
    while pos + 8 <= bytes.len() {
        let id = &bytes[pos..pos + 4];
        let size = u32::from_le_bytes(bytes[pos + 4..pos + 8].try_into().unwrap()) as usize;
        let body = pos + 8;
        if body + size > bytes.len() {
            break;
        }
        match id {
            b"fmt " => {
                if size >= 16 {
                    format = Some(u16::from_le_bytes(bytes[body..body + 2].try_into().unwrap()));
                    channels = Some(u16::from_le_bytes(bytes[body + 2..body + 4].try_into().unwrap()));
                    rate = Some(u32::from_le_bytes(bytes[body + 4..body + 8].try_into().unwrap()));
                    bits = Some(u16::from_le_bytes(bytes[body + 14..body + 16].try_into().unwrap()));
                }
            }
            b"data" => {
                data = Some(&bytes[body..body + size]);
            }
            _ => {}
        }
        pos = body + size + (size & 1); // chunks are word-aligned
    }
    let fmt = format.ok_or("missing fmt chunk")?;
    if fmt != 1 {
        return Err(format!("unsupported WAV format {} (want plain PCM)", fmt));
    }
    let ch = channels.ok_or("missing channel count")? as usize;
    let rate = rate.ok_or("missing sample rate")?;
    let bits = bits.ok_or("missing bit depth")?;
    let data = data.ok_or("missing data chunk")?;
    match bits {
        16 => {
            let samples: Vec<i16> = data.chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]))
                .collect();
            let mut pcm: Vec<f32> = Vec::with_capacity(samples.len() / ch);
            for frame in samples.chunks(ch) {
                let avg: f32 = frame.iter().map(|&s| s as f32).sum::<f32>() / ch as f32;
                pcm.push(avg / 32768.0);
            }
            Ok((pcm, rate))
        }
        8 => {
            let mut pcm: Vec<f32> = Vec::with_capacity(data.len() / ch);
            for frame in data.chunks(ch) {
                let avg: f32 = frame.iter().map(|&b| b as f32 - 128.0).sum::<f32>() / ch as f32;
                pcm.push(avg / 128.0);
            }
            Ok((pcm, rate))
        }
        _ => Err(format!("{}-bit WAV not supported (16-bit is the standard)", bits)),
    }
}

/// Encode mono f32 PCM (−1..1) as a 44-byte-header RIFF/WAVE file, 16-bit.
pub fn encode_wav(pcm: &[f32], sample_rate: u32) -> Vec<u8> {
    let data_bytes = (pcm.len() * 2) as u32;
    let mut out = Vec::with_capacity(44 + data_bytes as usize);

    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data_bytes).to_le_bytes());
    out.extend_from_slice(b"WAVE");
    out.extend_from_slice(b"fmt ");
    out.extend_from_slice(&16u32.to_le_bytes()); // PCM chunk size
    out.extend_from_slice(&1u16.to_le_bytes()); // PCM format
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&sample_rate.to_le_bytes());
    out.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
    out.extend_from_slice(&2u16.to_le_bytes()); // block align
    out.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data_bytes.to_le_bytes());

    for &s in pcm {
        let clamped = s.clamp(-1.0, 1.0);
        let v = if clamped < 0.0 {
            (clamped * 0x8000 as f32) as i16
        } else {
            (clamped * 0x7FFF as f32) as i16
        };
        out.extend_from_slice(&v.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let pcm = vec![0.5f32, -0.25, 0.0, 0.9];
        let wav = encode_wav(&pcm, 8000);
        let (back, rate) = decode_wav(&wav).unwrap();
        assert_eq!(rate, 8000);
        assert_eq!(back.len(), 4);
        assert!((back[0] - 0.5).abs() < 0.001);
        assert!((back[1] + 0.25).abs() < 0.001);
    }

    #[test]
    fn header_and_length() {
        let pcm = vec![0f32; 160];
        let wav = encode_wav(&pcm, 16000);
        assert_eq!(wav.len(), 44 + 320);
        assert_eq!(&wav[..4], b"RIFF");
        assert_eq!(&wav[8..12], b"WAVE");
        assert_eq!(u32::from_le_bytes(wav[24..28].try_into().unwrap()), 16000);
    }
}
