//! FAST-9 corner detector with non-maximum suppression.

use image::GrayImage;

/// The 16-pixel Bresenham circle (clockwise from top).
const CIRCLE: [(i64, i64); 16] = [
    (0, -3), (1, -3), (2, -2), (3, -1), (3, 0), (3, 1), (2, 2), (1, 3),
    (0, 3), (-1, 3), (-2, 2), (-3, 1), (-3, 0), (-3, -1), (-2, -2), (-1, -3),
];

/// Detect FAST-9 corners. Returns (x, y, score) sorted by score descending.
pub fn fast9(img: &GrayImage, threshold: u8) -> Vec<(u32, u32, i32)> {
    let (w, h) = img.dimensions();
    let (w, h) = (w as i64, h as i64);
    let buf = img.as_raw();
    let mut out = Vec::new();
    for y in 3..h - 3 {
        for x in 3..w - 3 {
            let c = buf[(y * w + x) as usize] as i32;
            let thr = threshold as i32;
            // quick reject: pixels 1, 5, 9, 13 (indices 0, 4, 8, 12)
            let (i0, i4, i8, i12) = (
                buf[((y + CIRCLE[0].1) * w + (x + CIRCLE[0].0)) as usize] as i32,
                buf[((y + CIRCLE[4].1) * w + (x + CIRCLE[4].0)) as usize] as i32,
                buf[((y + CIRCLE[8].1) * w + (x + CIRCLE[8].0)) as usize] as i32,
                buf[((y + CIRCLE[12].1) * w + (x + CIRCLE[12].0)) as usize] as i32,
            );
            let b_count = [i0, i4, i8, i12].iter().filter(|&&v| v > c + thr).count();
            let d_count = [i0, i4, i8, i12].iter().filter(|&&v| v < c - thr).count();
            if b_count < 3 && d_count < 3 {
                continue;
            }
            // full circle test: 9 contiguous brighter or darker
            let vals: Vec<i32> = CIRCLE
                .iter()
                .map(|&(ox, oy)| buf[((y + oy) * w + (x + ox)) as usize] as i32)
                .collect();
            let bright: Vec<bool> = vals.iter().map(|&v| v > c + thr).collect();
            let dark: Vec<bool> = vals.iter().map(|&v| v < c - thr).collect();
            if !contiguous_run(&bright, 9) && !contiguous_run(&dark, 9) {
                continue;
            }
            // score: total absolute difference over the circle
            let score: i32 = vals.iter().map(|&v| (v - c).abs()).sum();
            out.push((x as u32, y as u32, score));
        }
    }
    out.sort_by(|a, b| b.2.cmp(&a.2));
    out
}

/// Longest contiguous circular run >= n.
fn contiguous_run(flags: &[bool], n: usize) -> bool {
    let len = flags.len();
    let mut best = 0;
    let mut cur = 0;
    // go around twice to handle wrap-around
    for i in 0..len * 2 {
        if flags[i % len] {
            cur += 1;
            best = best.max(cur);
        } else {
            cur = 0;
        }
        if best >= n {
            return true;
        }
    }
    best >= n
}

/// Greedy non-maximum suppression with a minimum distance.
pub fn nms(points: &[(u32, u32, i32)], min_dist: u32, max_points: usize) -> Vec<(u32, u32, i32)> {
    let mut kept: Vec<(u32, u32, i32)> = Vec::new();
    'outer: for &p in points {
        for &k in &kept {
            let dx = (p.0 as i64 - k.0 as i64).abs();
            let dy = (p.1 as i64 - k.1 as i64).abs();
            if dx < min_dist as i64 && dy < min_dist as i64 {
                continue 'outer;
            }
        }
        kept.push(p);
        if kept.len() >= max_points {
            break;
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_corners_in_textured_image() {
        // random noise has corners everywhere
        let mut rng = crate::core::rng::GaussRng::new(21);
        let mut img = GrayImage::new(120, 120);
        for y in 0..120 {
            for x in 0..120 {
                img.put_pixel(x, y, image::Luma([(rng.uniform() * 255.0) as u8]));
            }
        }
        let pts = fast9(&img, 25);
        assert!(pts.len() > 60, "corners detected: {}", pts.len());
        // NMS keeps well-separated points
        let kept = nms(&pts, 5, 100);
        assert!(kept.len() > 10 && kept.len() <= 100);
        for i in 0..kept.len() {
            for j in i + 1..kept.len() {
                let dx = (kept[i].0 as i64 - kept[j].0 as i64).abs();
                let dy = (kept[i].1 as i64 - kept[j].1 as i64).abs();
                assert!(dx >= 5 || dy >= 5, "NMS spacing violated");
            }
        }
    }

    #[test]
    fn pure_edge_is_not_a_corner() {
        // vertical edge only: arcs are at most 8 long -> no detections
        let mut img = GrayImage::new(100, 100);
        for y in 0..100 {
            for x in 0..100 {
                let v = if x < 50 { 220u8 } else { 30 };
                img.put_pixel(x, y, image::Luma([v]));
            }
        }
        assert!(fast9(&img, 25).is_empty(), "edges must not trigger FAST-9");
    }

    #[test]
    fn flat_image_no_corners() {
        let img = GrayImage::from_pixel(100, 100, image::Luma([128u8]));
        assert!(fast9(&img, 25).is_empty());
    }
}
