//! P0600 — the software rasterizer: z-buffered triangles with top-left fill,
//! near-plane polygon clipping, Blinn-Phong Gouraud shading, light-projected
//! ground shadows, adaptive metric grid, 2×2 supersampling.

use crate::math3::{Aabb, V3};
use crate::render::camera::Camera;

pub struct Framebuffer {
    pub w: usize,
    pub h: usize,
    /// 0x00RRGGBB
    pub px: Vec<u32>,
    pub depth: Vec<f32>,
}

impl Framebuffer {
    pub fn new(w: usize, h: usize) -> Framebuffer {
        Framebuffer {
            w,
            h,
            px: vec![0u32; w * h],
            depth: vec![f32::INFINITY; w * h],
        }
    }
    #[inline]
    pub fn blend(&mut self, x: usize, y: usize, r: u8, g: u8, b: u8) {
        // helpers live in simd for the hot path; scalar fallback here
        let i = y * self.w + x;
        let dst = self.px[i];
        let (dr, dg, db) = (((dst >> 16) & 0xFF), ((dst >> 8) & 0xFF), (dst & 0xFF));
        // caller passes premultiplied alpha via r,g,b already blended? no —
        // plain blend with fixed alpha is done by callers via blend_alpha
        let _ = (dr, dg, db);
        self.px[i] = ((r as u32) << 16) | ((g as u32) << 8) | (b as u32);
    }
    #[inline]
    pub fn blend_alpha(&mut self, x: usize, y: usize, r: u32, g: u32, b: u32, a: f32) {
        crate::simd::blend_pixel(&mut self.px[y * self.w + x], r, g, b, a);
    }
    pub fn to_rgb_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(self.w * self.h * 3);
        for p in &self.px {
            out.push(((*p >> 16) & 0xFF) as u8);
            out.push(((*p >> 8) & 0xFF) as u8);
            out.push((*p & 0xFF) as u8);
        }
        out
    }
}

// ---------- shading ----------

pub struct Lights {
    /// direction TO the key light (normalized, world space)
    pub key: V3,
    pub key_i: f32,
    pub fill: V3,
    pub fill_i: f32,
    pub ambient: f32,
}

impl Default for Lights {
    fn default() -> Lights {
        Lights {
            key: V3::new(-0.45, 0.85, 0.35).norm(),
            key_i: 0.95,
            fill: V3::new(0.7, 0.25, -0.55).norm(),
            fill_i: 0.28,
            ambient: 0.22,
        }
    }
}

/// Cook-Torrance vertex color → (r, g, b) 0-255 (gamma-encoded).
/// P1250: GGX + Smith + Schlick (see `ggx.rs`); `shadow` (0..1) is the
/// ray-traced visibility of the key light (soft shadows), `ao` (0..1)
/// darkens the ambient where nearby geometry blocks the sky.
#[inline]
pub fn shade(
    p: V3,
    n: V3,
    eye: V3,
    base: [f32; 3],
    rough: f32,
    metal: bool,
    shadow: f32,
    ao: f32,
    lights: &Lights,
) -> (f32, f32, f32) {
    let n = if n.len() < 1e-9 { V3::new(0.0, 1.0, 0.0) } else { n.norm() };
    let v = eye.sub(&p).norm();
    // ambient with sky/ground bounce, scaled by baked occlusion
    let amb = (lights.ambient + 0.14 * n.y().max(0.0) as f32 + 0.05 * (-n.y()).max(0.0) as f32)
        * ao.clamp(0.0, 1.0);
    // key light (shadowed) and fill light
    let d1 = n.dot(&lights.key).max(0.0) as f32 * lights.key_i * shadow.clamp(0.0, 1.0);
    let d2 = n.dot(&lights.fill).max(0.0) as f32 * lights.fill_i;
    let alpha = (rough * rough).clamp(0.002, 1.0);
    // GGX specular for the key light (fills are diffuse-only, like 1.0)
    let h = lights.key.add(&v).norm();
    let ndoth = n.dot(&h).max(0.0) as f32;
    let vdoth = v.dot(&h).max(0.0) as f32;
    let ndotv = n.dot(&v).max(1e-4) as f32;
    let ndotl = n.dot(&lights.key).max(0.0) as f32;
    let (kd, spec) = crate::render::ggx::brdf_terms(ndotv, ndotl, ndoth, vdoth, alpha, metal);
    // the specular highlight only where the key light is actually visible
    let spec_sum = spec * d1;
    let (sr, sg, sb) = if metal {
        (base[0], base[1], base[2])
    } else {
        (1.0, 1.0, 1.0)
    };
    let light_sum = amb + d1 + d2;
    // FIX(P1430 / B16): physically kd = 0 for metals, but this rig has no
    // environment map — ambient and fill only ever reached the diffuse (kd)
    // term, so every metal rendered as a black silhouette with one highlight
    // (found while shooting the steam engine). A rough metal DOES reflect
    // its surroundings broadly: give metals a sky-tinted environment term
    // built from the same ambient/fill energies.
    let metal_env = if metal { amb * 2.2 + d2 * 2.5 + 0.07 } else { 0.0 };
    let r = (base[0] * light_sum * kd + base[0] * metal_env + sr * spec_sum).clamp(0.0, 1.0);
    let g = (base[1] * light_sum * kd + base[1] * metal_env + sg * spec_sum).clamp(0.0, 1.0);
    let b = (base[2] * light_sum * kd + base[2] * metal_env + sb * spec_sum).clamp(0.0, 1.0);
    // gamma encode
    let gr = r.powf(1.0 / 2.2);
    let gg = g.powf(1.0 / 2.2);
    let gb = b.powf(1.0 / 2.2);
    (gr, gg, gb)
}

// ---------- drawing ----------

/// Draw the vertical sky gradient.
pub fn draw_sky(fb: &mut Framebuffer) {
    let top = (0.93f32, 0.96, 0.99);
    let bot = (0.80, 0.85, 0.90);
    for y in 0..fb.h {
        let t = y as f32 / (fb.h - 1).max(1) as f32;
        let r = ((top.0 + (bot.0 - top.0) * t) * 255.0) as u32;
        let g = ((top.1 + (bot.1 - top.1) * t) * 255.0) as u32;
        let b = ((top.2 + (bot.2 - top.2) * t) * 255.0) as u32;
        let row = y * fb.w;
        for x in 0..fb.w {
            fb.px[row + x] = (r << 16) | (g << 8) | b;
        }
    }
}

fn nice_step(raw: f64) -> f64 {
    let mag = 10.0f64.powf(raw.abs().log10().floor());
    let norm = raw / mag;
    if norm < 1.5 { mag } else if norm < 3.5 { 2.0 * mag } else if norm < 7.5 { 5.0 * mag } else { 10.0 * mag }
}

pub struct GridParams {
    pub step: f64,
    pub extent: f64,
}

pub fn grid_params(bb: &Aabb) -> GridParams {
    let r = bb.radius();
    if r < 1e-6 {
        return GridParams { step: 10.0, extent: 100.0 };
    }
    let step = nice_step(r / 7.0);
    let extent = (r * 2.4).max(step * 8.0);
    GridParams { step, extent }
}

/// Draw the adaptive metric ground grid (projected through the camera).
pub fn draw_grid(fb: &mut Framebuffer, cam: &Camera, gp: &GridParams) {
    let step = gp.step;
    let ext = gp.extent;
    let view = cam.view();
    let (fw, fh) = (fb.w as f64, fb.h as f64);
    // project a world point; skip if behind camera
    let proj = move |p: V3| -> Option<(f32, f32)> {
        let v = view.apply(&p);
        cam.project(v, fw, fh).map(|t| (t.0, t.1))
    };
    // vertical lines (along z) and horizontal (along x), at y=0
    let mut lines: Vec<(V3, V3, bool)> = Vec::new();
    let n = (ext / step).round() as i64;
    for i in -n..=n {
        let x = i as f64 * step;
        lines.push((V3::new(x, 0.0, -ext), V3::new(x, 0.0, ext), i % 5 == 0));
        let z = i as f64 * step;
        lines.push((V3::new(-ext, 0.0, z), V3::new(ext, 0.0, z), i % 5 == 0));
    }
    let minor: u32 = (160.0 * 0.55 + 80.0 * 0.45) as u32;
    let major: u32 = (120.0 * 0.55 + 60.0 * 0.45) as u32;
    for (a, b, is_major) in lines {
        let pa = match proj(a) { Some(p) => p, None => continue };
        let pb = match proj(b) { Some(p) => p, None => continue };
        // skip lines fully off-screen
        if (pa.0.max(pb.0) < -50.0 || pa.0.min(pb.0) > fb.w as f32 + 50.0)
            || (pa.1.max(pb.1) < -50.0 || pa.1.min(pb.1) > fb.h as f32 + 50.0)
        {
            continue;
        }
        draw_line(fb, pa, pb, if is_major { major } else { minor });
    }
}

/// Bresenham-ish line with 2px thickness (looks right under 2× SSAA).
fn draw_line(fb: &mut Framebuffer, a: (f32, f32), b: (f32, f32), col: u32) {
    let (r, g, bl) = (((col >> 16) & 0xFF), ((col >> 8) & 0xFF), (col & 0xFF));
    let dx = b.0 - a.0;
    let dy = b.1 - a.1;
    let steps = (dx.abs().max(dy.abs()).ceil()).max(1.0) as usize;
    for s in 0..=steps {
        let t = s as f32 / steps as f32;
        let x = a.0 + dx * t;
        let y = a.1 + dy * t;
        let (xi, yi) = (x.round() as i64, y.round() as i64);
        for (ox, oy) in [(0i64, 0), (1, 0), (0, 1)] {
            let px = xi + ox;
            let py = yi + oy;
            if px >= 0 && py >= 0 && (px as usize) < fb.w && (py as usize) < fb.h {
                let i = py as usize * fb.w + px as usize;
                fb.px[i] = ((r & 0xFF) << 16) | ((g & 0xFF) << 8) | (bl & 0xFF);
            }
        }
    }
}

/// Project a triangle along the key light onto the ground and darken it.
pub fn draw_shadow_tri(fb: &mut Framebuffer, cam: &Camera, tri: (V3, V3, V3), light_dir: V3, gp: &GridParams) {
    if light_dir.y() <= 0.05 {
        return; // light at/below the horizon: no ground shadow
    }
    let project = |p: V3| -> V3 {
        let t = p.y() / light_dir.y();
        p.sub(&light_dir.mul(t))
    };
    let (a, b, c) = (project(tri.0), project(tri.1), project(tri.2));
    // conservative: skip shadows far outside the grid
    let far = gp.extent * 2.0;
    for p in [&a, &b, &c] {
        if p.x().abs() > far || p.z().abs() > far {
            return;
        }
    }
    let view = cam.view();
    let pa = cam.project(view.apply(&a), fb.w as f64, fb.h as f64);
    let pb = cam.project(view.apply(&b), fb.w as f64, fb.h as f64);
    let pc = cam.project(view.apply(&c), fb.w as f64, fb.h as f64);
    if let (Some(pa), Some(pb), Some(pc)) = (pa, pb, pc) {
        raster_flat(fb, &(pa.0, pa.1), &(pb.0, pb.1), &(pc.0, pc.1), 45, 52, 66, 0.24);
    }
}

// ---------- triangle raster ----------

/// Raster a screen-space triangle with a flat color and alpha (no depth write).
/// Used for shadows.
pub fn raster_flat(fb: &mut Framebuffer, a: &(f32, f32), b: &(f32, f32), c: &(f32, f32), r: u32, g: u32, b2: u32, alpha: f32) {
    let min_x = a.0.min(b.0).min(c.0).floor().max(0.0) as usize;
    let max_x = a.0.max(b.0).max(c.0).ceil().min(fb.w as f32 - 1.0) as usize;
    let min_y = a.1.min(b.1).min(c.1).floor().max(0.0) as usize;
    let max_y = a.1.max(b.1).max(c.1).ceil().min(fb.h as f32 - 1.0) as usize;
    let area = (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0);
    if area.abs() < 1e-9 {
        return;
    }
    let sign = if area < 0.0 { -1.0 } else { 1.0 };
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let px = x as f32 + 0.5;
            let py = y as f32 + 0.5;
            let w0 = ((b.0 - a.0) * (py - a.1) - (b.1 - a.1) * (px - a.0)) * sign;
            let w1 = ((c.0 - b.0) * (py - b.1) - (c.1 - b.1) * (px - b.0)) * sign;
            let w2 = ((a.0 - c.0) * (py - c.1) - (a.1 - c.1) * (px - c.0)) * sign;
            if w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0 {
                crate::simd::blend_pixel(&mut fb.px[y * fb.w + x], r, g, b2, alpha);
            }
        }
    }
}

/// A fully-shaded, clipped, projected triangle ready for raster.
pub struct ScreenTri {
    pub x: [f32; 3],
    pub y: [f32; 3],
    pub z: [f32; 3],
    pub r: [f32; 3],
    pub g: [f32; 3],
    pub b: [f32; 3],
}

/// Z-buffered Gouraud triangle with top-left fill rule.
pub fn raster_shaded(fb: &mut Framebuffer, t: &ScreenTri) {
    let (x0, x1, x2) = (t.x[0], t.x[1], t.x[2]);
    let (y0, y1, y2) = (t.y[0], t.y[1], t.y[2]);
    let area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if area.abs() < 1e-12 {
        return;
    }
    let min_x = x0.min(x1).min(x2).floor().max(0.0) as i64;
    let max_x = x0.max(x1).max(x2).ceil().min(fb.w as f32 - 1.0) as i64;
    let min_y = y0.min(y1).min(y2).floor().max(0.0) as i64;
    let max_y = y0.max(y1).max(y2).ceil().min(fb.h as f32 - 1.0) as i64;
    if max_x < min_x || max_y < min_y {
        return;
    }
    // per-pixel edge functions (clarity over micro-speed; SSAA keeps edges clean)
    let inv_area = 1.0 / area;
    for yy in min_y.max(0) as usize..=(max_y as usize).min(fb.h - 1) {
        for xx in min_x.max(0) as usize..=(max_x as usize).min(fb.w - 1) {
            let px = xx as f32 + 0.5;
            let py = yy as f32 + 0.5;
            // barycentric (unnormalized)
            let w0 = (x1 - px) * (y2 - py) - (y1 - py) * (x2 - px);
            let w1 = (x2 - px) * (y0 - py) - (y2 - py) * (x0 - px);
            let w2 = (x0 - px) * (y1 - py) - (y0 - py) * (x1 - px);
            // top-left fill: sample if all edge weights have the same sign as area
            let inside = if area > 0.0 {
                w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
            } else {
                w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
            };
            if !inside {
                continue;
            }
            let u = w0 * inv_area;
            let v = w1 * inv_area;
            let w = w2 * inv_area;
            let z = t.z[0] * u + t.z[1] * v + t.z[2] * w;
            let idx = yy * fb.w + xx;
            if z < fb.depth[idx] {
                fb.depth[idx] = z;
                let r = t.r[0] * u + t.r[1] * v + t.r[2] * w;
                let g = t.g[0] * u + t.g[1] * v + t.g[2] * w;
                let b = t.b[0] * u + t.b[1] * v + t.b[2] * w;
                let ri = (r * 255.0 + 0.5).clamp(0.0, 255.0) as u32;
                let gi = (g * 255.0 + 0.5).clamp(0.0, 255.0) as u32;
                let bi = (b * 255.0 + 0.5).clamp(0.0, 255.0) as u32;
                fb.px[idx] = (ri << 16) | (gi << 8) | bi;
            }
        }
    }
}

/// P1400 — Z-tested alpha-blended Gouraud triangle for transparent parts
/// (glass, water, ice). Z-buffer is READ (a hidden face never shows) but
/// never WRITTEN, so see-through surfaces layer correctly: a steel bolt
/// inside a glass cube stays fully visible behind the blended faces.
pub fn raster_shaded_alpha(fb: &mut Framebuffer, t: &ScreenTri, alpha: f32) {
    let (x0, x1, x2) = (t.x[0], t.x[1], t.x[2]);
    let (y0, y1, y2) = (t.y[0], t.y[1], t.y[2]);
    let area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0);
    if area.abs() < 1e-12 {
        return;
    }
    let min_x = x0.min(x1).min(x2).floor().max(0.0) as i64;
    let max_x = x0.max(x1).max(x2).ceil().min(fb.w as f32 - 1.0) as i64;
    let min_y = y0.min(y1).min(y2).floor().max(0.0) as i64;
    let max_y = y0.max(y1).max(y2).ceil().min(fb.h as f32 - 1.0) as i64;
    if max_x < min_x || max_y < min_y {
        return;
    }
    let inv_area = 1.0 / area;
    let a = alpha.clamp(0.0, 1.0);
    for yy in min_y.max(0) as usize..=(max_y as usize).min(fb.h - 1) {
        for xx in min_x.max(0) as usize..=(max_x as usize).min(fb.w - 1) {
            let px = xx as f32 + 0.5;
            let py = yy as f32 + 0.5;
            let w0 = (x1 - px) * (y2 - py) - (y1 - py) * (x2 - px);
            let w1 = (x2 - px) * (y0 - py) - (y2 - py) * (x0 - px);
            let w2 = (x0 - px) * (y1 - py) - (y0 - py) * (x1 - px);
            let inside = if area > 0.0 {
                w0 >= 0.0 && w1 >= 0.0 && w2 >= 0.0
            } else {
                w0 <= 0.0 && w1 <= 0.0 && w2 <= 0.0
            };
            if !inside {
                continue;
            }
            let u = w0 * inv_area;
            let v = w1 * inv_area;
            let w = w2 * inv_area;
            let z = t.z[0] * u + t.z[1] * v + t.z[2] * w;
            let idx = yy * fb.w + xx;
            // z-TEST only — never occlude what's behind the glass
            if z < fb.depth[idx] {
                let r = t.r[0] * u + t.r[1] * v + t.r[2] * w;
                let g = t.g[0] * u + t.g[1] * v + t.g[2] * w;
                let b = t.b[0] * u + t.b[1] * v + t.b[2] * w;
                let ri = (r * 255.0 + 0.5).clamp(0.0, 255.0) as u32;
                let gi = (g * 255.0 + 0.5).clamp(0.0, 255.0) as u32;
                let bi = (b * 255.0 + 0.5).clamp(0.0, 255.0) as u32;
                fb.blend_alpha(xx, yy, ri, gi, bi, a);
            }
        }
    }
}

/// Near-plane clip in view space: returns clipped polygon (2-4 verts).
/// Input: 3 view-space points + their attributes (rgb).
pub fn clip_near(pts: &[(V3, f32, f32, f32); 3]) -> Vec<(V3, f32, f32, f32)> {
    const NEAR: f64 = 0.6; // mm
    let classify = |p: &V3| -> i32 {
        if p.z() < -NEAR { 1 } else if p.z() > -NEAR { -1 } else { 0 }
    };
    let mut out: Vec<(V3, f32, f32, f32)> = Vec::with_capacity(4);
    let mut has_out = false;
    let mut has_in = false;
    for (p, ..) in pts {
        let c = classify(p);
        if c >= 0 { has_in = true; }
        if c <= 0 { has_out = true; }
    }
    if !has_out {
        return pts.to_vec();
    }
    if has_in && has_out {
        // spanning: clip edges
        for i in 0..3 {
            let j = (i + 1) % 3;
            let (pi, ci) = (&pts[i].0, classify(&pts[i].0));
            let (pj, cj) = (&pts[j].0, classify(&pts[j].0));
            if ci >= 0 {
                out.push(pts[i]);
            }
            if (ci > 0 && cj < 0) || (ci < 0 && cj > 0) {
                // interpolate to the near plane
                let d = pi.z() - pj.z();
                let t = if d.abs() > 1e-12 { (pi.z() + NEAR) / d } else { 0.5 };
                let t = t.clamp(0.0, 1.0);
                let p = pi.lerp(pj, t);
                let r = pts[i].1 + (pts[j].1 - pts[i].1) * t as f32;
                let g = pts[i].2 + (pts[j].2 - pts[i].2) * t as f32;
                let b = pts[i].3 + (pts[j].3 - pts[i].3) * t as f32;
                out.push((p, r, g, b));
            }
        }
    } else {
        // fully behind the near plane: clipped away entirely
    }
    out
}
