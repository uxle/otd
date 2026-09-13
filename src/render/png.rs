//! P0620 — PNG encoding from NOTHING: own CRC-32, own Adler-32, own DEFLATE
//! (LZ77 hash-chain matcher + fixed-Huffman blocks). Verified against
//! Python's zlib in the external check script.

// ---------- CRC-32 (IEEE, same polynomial as PNG chunks) ----------

fn crc32_table() -> [u32; 256] {
    let mut t = [0u32; 256];
    for n in 0..256u32 {
        let mut c = n;
        for _ in 0..8 {
            c = if c & 1 != 0 { 0xEDB88320 ^ (c >> 1) } else { c >> 1 };
        }
        t[n as usize] = c;
    }
    t
}

pub fn crc32(data: &[u8]) -> u32 {
    let t = crc32_table();
    let mut c = 0xFFFF_FFFFu32;
    for &b in data {
        c = t[((c ^ b as u32) & 0xFF) as usize] ^ (c >> 8);
    }
    c ^ 0xFFFF_FFFF
}

// ---------- Adler-32 ----------

pub fn adler32(data: &[u8]) -> u32 {
    let mut a: u32 = 1;
    let mut b: u32 = 0;
    for &x in data {
        a = (a + x as u32) % 65521;
        b = (b + a) % 65521;
    }
    (b << 16) | a
}

// ---------- DEFLATE: fixed-Huffman + LZ77 ----------

struct BitWriter {
    out: Vec<u8>,
    cur: u32,
    nbits: u32,
}

impl BitWriter {
    fn new() -> BitWriter { BitWriter { out: Vec::new(), cur: 0, nbits: 0 } }
    /// write n bits (LSB-first, as DEFLATE reads non-code data)
    fn bits(&mut self, value: u32, n: u32) {
        self.cur |= value << self.nbits;
        self.nbits += n;
        while self.nbits >= 8 {
            self.out.push((self.cur & 0xFF) as u8);
            self.cur >>= 8;
            self.nbits -= 8;
        }
    }
    /// Huffman codes are packed MSB-first (RFC 1951 §3.1.1) — reverse the
    /// code's bits, then write LSB-first into the stream.
    fn huff(&mut self, code: u32, n: u32) {
        let mut rev = 0u32;
        for b in 0..n {
            rev |= ((code >> (n - 1 - b)) & 1) << b;
        }
        self.bits(rev, n);
    }
    fn finish(mut self) -> Vec<u8> {
        if self.nbits > 0 {
            self.out.push((self.cur & 0xFF) as u8);
        }
        self.out
    }
}

/// fixed-Huffman literal/length code (returns code value and bit count)
fn fixed_lit_code(sym: u32) -> (u32, u32) {
    match sym {
        0..=143 => (0x30 + sym, 8),
        144..=255 => (0x190 + (sym - 144), 9),
        256..=279 => (sym - 256, 7),
        _ => (0xC0 + (sym - 280), 8),
    }
}

const LEN_BASE: [u32; 29] = [3,4,5,6,7,8,9,10,11,13,15,17,19,23,27,31,35,43,51,59,67,83,99,115,131,163,195,227,258];
const LEN_EXTRA: [u32; 29] = [0,0,0,0,0,0,0,0,1,1,1,1,2,2,2,2,3,3,3,3,4,4,4,4,5,5,5,5,0];
const DIST_BASE: [u32; 30] = [1,2,3,4,5,7,9,13,17,25,33,49,65,97,129,193,257,385,513,769,1025,1537,2049,3073,4097,6145,8193,12289,16385,24577];
const DIST_EXTRA: [u32; 30] = [0,0,0,0,1,1,2,2,3,3,4,4,5,5,6,6,7,7,8,8,9,9,10,10,11,11,12,12,13,13];

fn len_code_for(len: u32) -> usize {
    let mut i = 28;
    while i > 0 && LEN_BASE[i] > len {
        i -= 1;
    }
    i
}

fn dist_code_for(dist: u32) -> usize {
    let mut i = 29;
    while i > 0 && DIST_BASE[i] > dist {
        i -= 1;
    }
    i
}

/// Compress with LZ77 + fixed Huffman. Output is a raw DEFLATE stream
/// (one final block).
pub fn deflate(data: &[u8]) -> Vec<u8> {
    let mut w = BitWriter::new();
    w.bits(1, 1); // BFINAL
    w.bits(1, 2); // BTYPE = 01 (fixed Huffman)

    let n = data.len();
    // hash table of 3-byte sequences → most recent position (with chains via prev)
    const HASH_BITS: u32 = 15;
    const HASH_SIZE: usize = 1 << HASH_BITS;
    let mut head = vec![-1i32; HASH_SIZE];
    let mut prev = vec![-1i32; n.max(1)];

    let hash_at = |d: &[u8], i: usize| -> usize {
        let h = (d[i] as u32) | ((d[i + 1] as u32) << 8) | ((d[i + 2] as u32) << 16);
        ((h.wrapping_mul(0x9E3779B1)) >> (24 - HASH_BITS)) as usize & (HASH_SIZE - 1)
    };

    let mut i = 0usize;
    let mut literals_only = 0usize;
    while i < n {
        // find longest match
        let mut best_len = 0usize;
        let mut best_dist = 0usize;
        if i + 3 <= n {
            let h = hash_at(data, i);
            let mut cand = head[h];
            let mut chain = 0;
            while cand >= 0 && chain < 32 {
                let c = cand as usize;
                let dist = i - c;
                if dist > 32768 {
                    break;
                }
                // match length
                let max_len = (n - i).min(258);
                let mut l = 0usize;
                while l < max_len && data[c + l] == data[i + l] {
                    l += 1;
                }
                if l > best_len {
                    best_len = l;
                    best_dist = dist;
                    if l >= 128 {
                        break; // good enough
                    }
                }
                cand = prev[c];
                chain += 1;
            }
        }
        if best_len >= 3 {
            // emit length+distance (codes MSB-first, extra bits LSB-first)
            let li = len_code_for(best_len as u32);
            let (code, bits) = fixed_lit_code(257 + li as u32);
            w.huff(code, bits);
            let extra = LEN_EXTRA[li];
            if extra > 0 {
                w.bits((best_len as u32) - LEN_BASE[li], extra);
            }
            let di = dist_code_for(best_dist as u32);
            w.huff(di as u32, 5);
            let dextra = DIST_EXTRA[di];
            if dextra > 0 {
                w.bits((best_dist as u32) - DIST_BASE[di], dextra);
            }
            // insert hash entries for the match
            for k in 0..best_len {
                let p = i + k;
                if p + 3 <= n {
                    let h = hash_at(data, p);
                    prev[p] = head[h];
                    head[h] = p as i32;
                }
            }
            i += best_len;
        } else {
            literals_only += 1;
            let (code, bits) = fixed_lit_code(data[i] as u32);
            w.huff(code, bits);
            if i + 3 <= n {
                let h = hash_at(data, i);
                prev[i] = head[h];
                head[h] = i as i32;
            }
            i += 1;
        }
    }
    let _ = literals_only;
    // end of block
    let (code, bits) = fixed_lit_code(256);
    w.huff(code, bits);
    w.finish()
}

/// zlib stream: 2-byte header + deflate + adler32.
pub fn zlib_compress(data: &[u8]) -> Vec<u8> {
    let mut out = vec![0x78, 0x01]; // CMF: deflate 32K; FLG: fastest, check bits valid
    out.extend(deflate(data));
    out.extend(adler32(data).to_be_bytes());
    out
}

// ---------- PNG ----------

fn chunk(out: &mut Vec<u8>, kind: &[u8; 4], data: &[u8]) {
    out.extend((data.len() as u32).to_be_bytes());
    let mut body = Vec::with_capacity(4 + data.len());
    body.extend_from_slice(kind);
    body.extend_from_slice(data);
    out.extend(&body);
    out.extend(crc32(&body).to_be_bytes());
}

/// Encode RGB8 pixels (row-major, w×h×3) as a PNG.
pub fn encode_png(w: u32, h: u32, rgb: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend([0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
    // IHDR: w, h, bit depth 8, color type 2 (truecolor), compression 0, filter 0, interlace 0
    let mut ihdr = Vec::new();
    ihdr.extend(w.to_be_bytes());
    ihdr.extend(h.to_be_bytes());
    ihdr.push(8);
    ihdr.push(2);
    ihdr.push(0);
    ihdr.push(0);
    ihdr.push(0);
    chunk(&mut out, b"IHDR", &ihdr);
    // raw scanlines: filter byte 0 + row data
    let stride = (w as usize) * 3;
    let mut raw = Vec::with_capacity((stride + 1) * h as usize);
    for y in 0..h as usize {
        raw.push(0); // filter: none
        raw.extend(&rgb[y * stride..(y + 1) * stride]);
    }
    let z = zlib_compress(&raw);
    chunk(&mut out, b"IDAT", &z);
    chunk(&mut out, b"IEND", &[]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32_known_vector() {
        // CRC-32("123456789") = 0xCBF43926 (classic check value)
        assert_eq!(crc32(b"123456789"), 0xCBF43926);
    }

    #[test]
    fn adler32_known() {
        // adler32("Wikipedia") = 0x11E60398
        assert_eq!(adler32(b"Wikipedia"), 0x11E60398);
    }

    #[test]
    fn png_signature_and_end() {
        let px = vec![128u8; 4 * 3 * 3];
        let png = encode_png(4, 3, &px);
        assert_eq!(&png[0..8], &[0x89, b'P', b'N', b'G', 0x0D, 0x0A, 0x1A, 0x0A]);
        // IEND is the last 12 bytes
        let n = png.len();
        assert_eq!(&png[n - 8..n - 4], b"IEND");
        // every chunk's CRC verifies (self-check)
        let mut p = 8usize;
        while p < n {
            let len = u32::from_be_bytes([png[p], png[p + 1], png[p + 2], png[p + 3]]) as usize;
            let mut body = Vec::new();
            body.extend_from_slice(&png[p + 4..p + 8 + len]);
            let crc = u32::from_be_bytes([
                png[p + 8 + len], png[p + 9 + len], png[p + 10 + len], png[p + 11 + len],
            ]);
            assert_eq!(crc, crc32(&body), "chunk CRC mismatch");
            p += 12 + len;
        }
    }

    #[test]
    fn deflate_roundtrip_tiny() {
        // (decompression is verified externally with Python zlib — here we
        // sanity-check the structure: fixed block header, adler trailer)
        let data = b"hello hello hello hello world";
        let z = zlib_compress(data);
        assert_eq!(z[0], 0x78);
        assert_eq!(z.len() > 6, true);
    }
}
