#!/usr/bin/env python3
"""External truth check: decode OTD's own PNG/DEFLATE output with Python's zlib.

Usage: python3 scripts/verify_png.py image.png [image2.png ...]
Exits non-zero on any failure. This is the independent verification of our
from-scratch CRC-32, Adler-32 and DEFLATE (fixed-Huffman + LZ77) encoder.
"""
import sys, zlib, struct

def check(path):
    data = open(path, 'rb').read()
    assert data[:8] == b'\x89PNG\r\n\x1a\n', f"{path}: bad signature"
    p, idat, kinds = 8, b'', []
    while p < len(data):
        (ln,) = struct.unpack('>I', data[p:p+4])
        kind = data[p+4:p+8].decode('latin1')
        body = data[p+4:p+8+ln]
        (crc,) = struct.unpack('>I', data[p+8+ln:p+12+ln])
        assert crc == (zlib.crc32(body) & 0xffffffff), f"{path}: CRC mismatch in {kind}"
        kinds.append(kind)
        if kind == 'IDAT':
            idat += data[p+8:p+8+ln]
        p += 12 + ln
    assert kinds[-1] == 'IEND', f"{path}: missing IEND"
    w, h, depth, ctype = struct.unpack('>IIBB', data[16:26])
    raw = zlib.decompress(idat)
    assert len(raw) == h * (w*3 + 1), f"{path}: wrong raw size"
    for y in range(h):
        assert raw[y*(w*3+1)] == 0, f"{path}: unexpected filter (row {y})"
    print(f"OK {path}: {w}x{h}, {len(idat)} compressed bytes -> {len(raw)} raw, chunks {kinds}")

if __name__ == '__main__':
    for f in sys.argv[1:]:
        check(f)
