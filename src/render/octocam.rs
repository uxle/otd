//! P2230 — OCTOCAM: the eight-camera preview panel.
//!
//! Eight cameras stand at the eight compass directions around the scene,
//! every 45° of yaw, all at the same orbit height and distance — a full
//! ring of eyes with nothing hidden:
//!
//! ```text
//!        NW   N   NE            N = 0°   front
//!      W  ·   ·   ·  E          every 45° of yaw
//!        SW   S   SE            pitch shared with the iso view
//! ```
//!
//! The panel composes the eight renders into one 4×2 grid image (labelled,
//! with the active view bright and the rest slightly dimmed so the eye
//! finds the hero first). `--cams 8` on the CLI renders it; with
//! `--video out.mp4` the same ring becomes a real H.264/AVC film.
//!
//! Eight eyes is the classic machine-vision number: enough to resolve
//! occlusion from any side without the cost of a dense ring.

use crate::math3::Aabb;
use crate::render::camera::Camera;
use crate::render::{prepare_lit_scene, render_view_rgb, RenderOpts};
use crate::world::eval::World;

/// The eight compass directions, in panel reading order.
pub const DIRECTIONS: [&str; 8] = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"];

/// Yaw angles (degrees) for the eight directions. N = 0° looks from +Z
/// toward the target (the renderer's "front"), E = 90°, S = 180°, W = 270°,
/// with the diagonals between.
pub const YAWS: [f64; 8] = [0.0, 45.0, 90.0, 135.0, 180.0, 225.0, 270.0, 315.0];

/// Build the camera ring for a bounding box: same fit distance and pitch,
/// eight yaws.
pub fn camera_ring(bb: &Aabb, pitch: f64) -> Vec<Camera> {
    let base = Camera::fit(bb);
    YAWS.iter()
        .map(|&yaw| Camera { yaw, pitch, dist: base.dist * 1.35, target: base.target, fov: base.fov })
        .collect()
}

/// Label renderer for the panel corners: 5×7 dot font, drawn as solid
/// pixels (the renderer has a font module, but a tiny local copy keeps
/// this self-contained and compositing-friendly).
fn draw_label(img: &mut [u8], w: u32, h: u32, x0: u32, y0: u32, text: &str, rgb: [u8; 3]) {
    const FONT: [(&str, [u8; 7]); 8] = [
        ("N", [0b00100, 0b01110, 0b10101, 0b10001, 0b10001, 0b10001, 0b10001]),
        ("E", [0b11111, 0b10000, 0b10000, 0b11110, 0b10000, 0b10000, 0b11111]),
        ("S", [0b11111, 0b10000, 0b10000, 0b11110, 0b00001, 0b00001, 0b11110]),
        ("W", [0b10001, 0b10001, 0b10010, 0b10101, 0b10101, 0b11001, 0b10001]),
        ("C", [0b01110, 0b10001, 0b10000, 0b10000, 0b10000, 0b10001, 0b01110]),
        ("A", [0b00100, 0b01010, 0b10001, 0b11111, 0b10001, 0b10001, 0b10001]),
        ("M", [0b10001, 0b11011, 0b10101, 0b10101, 0b10001, 0b10001, 0b10001]),
        (" ", [0, 0, 0, 0, 0, 0, 0]),
    ];
    let mut cx = x0;
    for ch in text.chars() {
        let key = ch.to_string();
        if let Some((_, rows)) = FONT.iter().find(|(k, _)| *k == key) {
            for (ry, bits) in rows.iter().enumerate() {
                for rx in 0..5 {
                    if (bits >> (4 - rx)) & 1 == 1 {
                        let px = cx + rx as u32;
                        let py = y0 + ry as u32;
                        if px < w && py < h {
                            let i = ((py * w + px) * 4) as usize;
                            img[i] = rgb[0];
                            img[i + 1] = rgb[1];
                            img[i + 2] = rgb[2];
                            img[i + 3] = 255;
                        }
                    }
                }
            }
        }
        cx += 6;
        if cx >= w {
            break;
        }
    }
}

/// Render the eight-view panel for one world state.
///
/// Layout: 4 columns × 2 rows of views (N NE E SE / S SW W NW), each view
/// `vw×vh`, separated by 2-px gutters, plus a 12-px label strip at the
/// bottom of each cell. Returns RGBA pixels of the composite.
pub fn render_panel(world: &World, vw: u32, vh: u32) -> Result<Vec<u8>, String> {
    if vw < 32 || vh < 32 {
        return Err("panel view size too small (need ≥ 32×32)".into());
    }
    let gut = 2u32;
    let panel_w = vw * 4 + gut * 5;
    let panel_h = vh * 2 + gut * 3;
    let mut panel = vec![0u8; (panel_w * panel_h * 4) as usize];
    // background: dark slate
    for px in panel.chunks_exact_mut(4) {
        px[0] = 0x1a;
        px[1] = 0x1d;
        px[2] = 0x24;
        px[3] = 0xff;
    }

    let opts = RenderOpts { width: vw, height: vh, ssaa: 1, show_grid: true };
    let ring = camera_ring(&world.stats.bbox, 22.0);
    // PERF: the eight views share one geometry — only the camera differs —
    // so the expensive ray-traced lighting (BVH build + shadow/AO) is done
    // ONCE here instead of eight times (it used to be recomputed from
    // scratch inside render_world_rgb on every iteration of this loop).
    // Projecting/rasterizing each of the eight cheap views is then
    // embarrassingly parallel (each writes its own buffer), so they run
    // across threads too.
    let lit = prepare_lit_scene(world);
    let rgbs: Vec<Vec<u8>> = {
        let threads = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1).min(ring.len());
        if threads <= 1 {
            ring.iter().map(|cam| render_view_rgb(world, &lit, cam, &opts)).collect()
        } else {
            let mut out: Vec<Vec<u8>> = Vec::with_capacity(ring.len());
            std::thread::scope(|s| {
                let lit_ref = &lit;
                let opts_ref = &opts;
                let handles: Vec<_> = ring
                    .chunks((ring.len() + threads - 1) / threads)
                    .map(|cams| s.spawn(move || cams.iter().map(|cam| render_view_rgb(world, lit_ref, cam, opts_ref)).collect::<Vec<Vec<u8>>>()))
                    .collect();
                for h in handles {
                    out.extend(h.join().unwrap());
                }
            });
            out
        }
    };
    for (i, cam) in ring.iter().enumerate() {
        let rgb = &rgbs[i];
        let col = i as u32 % 4;
        let row = i as u32 / 4;
        let x0 = gut + col * (vw + gut);
        let y0 = gut + row * (vh + gut);
        // composite with a slight dim on non-hero views (N is the hero)
        let dim = if i == 0 { 1.0 } else { 0.82 };
        for y in 0..vh {
            for x in 0..vw {
                let src = ((y * vw + x) * 3) as usize;
                let dx = x0 + x;
                let dy = y0 + y;
                let dst = ((dy * panel_w + dx) * 4) as usize;
                panel[dst] = (rgb[src] as f32 * dim) as u8;
                panel[dst + 1] = (rgb[src + 1] as f32 * dim) as u8;
                panel[dst + 2] = (rgb[src + 2] as f32 * dim) as u8;
                panel[dst + 3] = 255;
            }
        }
        // label in the view's corner
        let bright = if i == 0 { [0xffu8, 0xd8, 0x66] } else { [0x9cu8, 0xb0, 0xc8] };
        draw_label(&mut panel, panel_w, panel_h, x0 + 4, y0 + 4, DIRECTIONS[i], bright);
    }
    Ok(panel)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::world::eval::compile;

    #[test]
    fn ring_covers_all_directions() {
        let w = compile("scene \"t\"\nc = cube 4cm at (0, 2cm, 0)");
        let ring = camera_ring(&w.stats.bbox, 25.0);
        assert_eq!(ring.len(), 8);
        for (i, cam) in ring.iter().enumerate() {
            assert!((cam.yaw - YAWS[i]).abs() < 1e-9);
            assert_eq!(cam.pitch, 25.0);
        }
        // neighbours are 45° apart
        for i in 0..7 {
            let d = (ring[i + 1].yaw - ring[i].yaw).abs();
            assert!((d - 45.0).abs() < 1e-9);
        }
    }

    #[test]
    fn opposite_cameras_see_opposite_sides() {
        let w = compile("scene \"t\"\nc = cube 4cm at (0, 2cm, 0)");
        let ring = camera_ring(&w.stats.bbox, 5.0);
        // N camera sits at +Z, S camera at −Z (same distance)
        let n = ring[0].eye();
        let s = ring[4].eye();
        assert!(n.z() > 0.0 && s.z() < 0.0);
        assert!((n.x() - s.x()).abs() < 1e-9);
        assert!((n.z() + s.z()).abs() < 1e-6);
        // E camera at +X, W at −X
        let e = ring[2].eye();
        let ww = ring[6].eye();
        assert!(e.x() > 0.0 && ww.x() < 0.0);
    }

    #[test]
    fn panel_renders_all_eight() {
        let w = compile("scene \"t\"\nc = cube 4cm at (0, 2cm, 0) material: steel");
        let panel = render_panel(&w, 96, 72).unwrap();
        // panel geometry: 96*4 + 2*5 = 394 wide, 72*2 + 2*3 = 150 tall
        let pw = 96 * 4 + 2 * 5;
        let ph = 72 * 2 + 2 * 3;
        assert_eq!(panel.len(), (pw * ph * 4) as usize);
        // background corners are slate
        assert_eq!(panel[0], 0x1a);
        // not all background: the views drew something
        let non_bg = panel.chunks_exact(4).filter(|p| p[0] != 0x1a || p[1] != 0x1d).count();
        assert!(non_bg > 1000, "views must cover most of the panel");
        // labels: the N label's first dot (yellow) exists
        let yellow = panel.chunks_exact(4).any(|p| p[0] == 0xff && p[1] == 0xd8 && p[2] == 0x66);
        assert!(yellow, "the N label must be drawn");
    }
}
