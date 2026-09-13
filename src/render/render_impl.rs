//! P0600 — render orchestration: world → framebuffer → PNG.
//!
//! PERF: rendering has two halves with very different costs.
//!   1. LIGHTING (expensive): merge visible parts into one BVH, then
//!      ray-trace soft shadows + ambient occlusion per surface cell.
//!      This depends ONLY on geometry + the (fixed) light rig — never on
//!      the camera.
//!   2. PROJECTION (cheap): transform, clip, project, rasterize into a
//!      framebuffer. This is the only part that depends on the camera.
//! `prepare_lit_scene` does (1) once; `render_view`/`render_view_rgb` do
//! (2) as many times as needed (camera orbit, the eight-view OCTOCAM
//! panel, stereo/video frames from a fixed scene). `render_world`/
//! `render_world_rgb` remain as single-shot convenience wrappers for
//! callers that only ever need one frame — same cost as before, no call
//! site elsewhere in the crate has to change.

use crate::math3::V3;
use crate::render::camera::Camera;
use crate::render::raster::*;
use crate::render::png::encode_png;
use crate::world::eval::World;

pub struct RenderOpts {
    pub width: u32,
    pub height: u32,
    /// supersampling factor (1 or 2)
    pub ssaa: u32,
    pub show_grid: bool,
}

impl Default for RenderOpts {
    fn default() -> RenderOpts {
        RenderOpts { width: 900, height: 600, ssaa: 2, show_grid: true }
    }
}

/// Everything about a world's appearance that does NOT depend on the
/// camera: the merged scene BVH and, per visible part, creased normals +
/// ray-traced shadow/AO. Building this is the expensive step (ray tracing
/// against a BVH); reuse it across every camera angle for a given world
/// instead of rebuilding it once per frame.
pub struct LitScene {
    lights: Lights,
    per_part: Vec<PartLight>,
}

struct PartLight {
    normals: Vec<V3>,
    shadow: Vec<f32>,
    ao: Vec<f32>,
}

fn visible_parts(world: &World) -> Vec<&crate::world::eval::Part> {
    world.parts.iter().filter(|p| !p.hidden && !p.mesh.is_empty()).collect()
}

/// Build the camera-independent lighting data for a world. Dominated by
/// ray tracing (O(cells × rays × log tris)); run this once per distinct
/// geometry — not once per frame — and pass the result to `render_view`.
pub fn prepare_lit_scene(world: &World) -> LitScene {
    let visible = visible_parts(world);
    let lights = Lights::default();
    let mut scene = crate::geo::mesh::Mesh::new();
    for part in &visible {
        scene.merge(&part.mesh);
    }
    let bvh = crate::geo::bvh::Bvh::build(&scene);
    let ground_y = if visible.is_empty() { 0.0 } else { world.stats.bbox.min.y() };
    let per_part = build_part_lighting(&visible, &bvh, &scene, ground_y, &lights);
    LitScene { lights, per_part }
}

/// Compute per-part creased normals + shadow/AO. Parts are independent of
/// one another (each only reads the shared, read-only `bvh`/`scene`), so
/// this splits the part list across `available_parallelism()` threads via
/// `std::thread::scope` — no unsafe, no locks, borrow-checked as usual.
fn build_part_lighting(
    visible: &[&crate::world::eval::Part],
    bvh: &crate::geo::bvh::Bvh,
    scene: &crate::geo::mesh::Mesh,
    ground_y: f64,
    lights: &Lights,
) -> Vec<PartLight> {
    let n = visible.len();
    if n == 0 {
        return Vec::new();
    }
    let threads = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1).min(n);
    if threads <= 1 {
        return visible
            .iter()
            .map(|part| light_one_part(part, bvh, scene, ground_y, lights))
            .collect();
    }
    let chunk = (n + threads - 1) / threads;
    let mut out: Vec<PartLight> = Vec::with_capacity(n);
    std::thread::scope(|s| {
        let handles: Vec<_> = visible
            .chunks(chunk)
            .map(|part_chunk| {
                s.spawn(move || {
                    part_chunk
                        .iter()
                        .map(|part| light_one_part(part, bvh, scene, ground_y, lights))
                        .collect::<Vec<PartLight>>()
                })
            })
            .collect();
        for h in handles {
            out.extend(h.join().unwrap());
        }
    });
    out
}

fn light_one_part(
    part: &crate::world::eval::Part,
    bvh: &crate::geo::bvh::Bvh,
    scene: &crate::geo::mesh::Mesh,
    ground_y: f64,
    lights: &Lights,
) -> PartLight {
    let normals = creased_normals(&part.mesh, 55.0);
    let (shadow, ao) = vertex_shadow_ao(&part.mesh, &normals, bvh, scene, ground_y, lights);
    PartLight { normals, shadow, ao }
}

/// Render the world → PNG file bytes (single-shot: builds its own lit
/// scene, so this costs the same as the original one-call render).
pub fn render_world(world: &World, cam: &Camera, opts: &RenderOpts) -> Vec<u8> {
    let rgb = render_world_rgb(world, cam, opts);
    encode_png(opts.width, opts.height, &rgb)
}

/// OTD3 — render the world → raw RGB8 pixels (width×height×3), single-shot.
pub fn render_world_rgb(world: &World, cam: &Camera, opts: &RenderOpts) -> Vec<u8> {
    let lit = prepare_lit_scene(world);
    render_view_rgb(world, &lit, cam, opts)
}

/// Render one camera view of an already-lit world → PNG bytes. Use this
/// (with a `LitScene` built once via `prepare_lit_scene`) whenever the
/// same geometry is viewed from more than one camera.
pub fn render_view(world: &World, lit: &LitScene, cam: &Camera, opts: &RenderOpts) -> Vec<u8> {
    let rgb = render_view_rgb(world, lit, cam, opts);
    encode_png(opts.width, opts.height, &rgb)
}

/// Render one camera view of an already-lit world → raw RGB8 pixels.
pub fn render_view_rgb(world: &World, lit: &LitScene, cam: &Camera, opts: &RenderOpts) -> Vec<u8> {
    let w = (opts.width * opts.ssaa) as usize;
    let h = (opts.height * opts.ssaa) as usize;
    let mut fb = Framebuffer::new(w, h);
    draw_sky(&mut fb);

    let visible = visible_parts(world);
    let bb = if visible.is_empty() {
        crate::math3::Aabb::from_center_extent(&V3::ZERO, &V3::new(50.0, 50.0, 50.0))
    } else {
        world.stats.bbox
    };
    let gp = grid_params(&bb);
    if opts.show_grid {
        draw_grid(&mut fb, cam, &gp);
    }

    // ground contact shadows (the projected silhouettes — a 1.0 signature)
    for part in &visible {
        for t in 0..part.mesh.tris.len() {
            let tri = part.mesh.tri(t);
            draw_shadow_tri(&mut fb, cam, tri, lit.lights.key, &gp);
        }
    }

    debug_assert_eq!(
        visible.len(),
        lit.per_part.len(),
        "LitScene was built from a different set of visible parts than this render call"
    );

    // model — P1400 two-pass: opaque parts first (z-buffer solid), then
    // transparent parts (glass/water/ice) far-to-near with alpha blending
    let view = cam.view();
    let eye = cam.eye();
    let mut transparent: Vec<usize> = Vec::new();
    for (i, part) in visible.iter().enumerate() {
        let alpha = part.material.map(|m| m.opacity).unwrap_or(1.0);
        if alpha < 0.999 {
            transparent.push(i);
        } else {
            draw_part(&mut fb, part, &lit.per_part[i], cam, &view, &eye, &lit.lights, alpha as f32);
        }
    }
    if !transparent.is_empty() {
        let c = eye;
        transparent.sort_by(|&i, &j| {
            let dp = visible[i].mesh.bbox().center().sub(&c).len();
            let dq = visible[j].mesh.bbox().center().sub(&c).len();
            dq.partial_cmp(&dp).unwrap_or(std::cmp::Ordering::Equal) // far → near
        });
        for i in transparent {
            let part = visible[i];
            let alpha = part.material.map(|m| m.opacity).unwrap_or(1.0);
            draw_part(&mut fb, part, &lit.per_part[i], cam, &view, &eye, &lit.lights, alpha as f32);
        }
    }

    // downsample
    downsample(&fb, opts.ssaa)
}

fn draw_part(
    fb: &mut Framebuffer,
    part: &crate::world::eval::Part,
    lit: &PartLight,
    cam: &Camera,
    view: &crate::math3::M4,
    eye: &V3,
    lights: &Lights,
    alpha: f32,
) {
    let color = part
        .color
        .map(|c| [c.r as f32 / 255.0, c.g as f32 / 255.0, c.b as f32 / 255.0])
        .unwrap_or([0.8, 0.8, 0.85]);
    let (rough, metal, base) = match part.material {
        Some(m) => (
            m.roughness as f32,
            m.metal,
            if part.color.is_some() { color } else { [m.color[0] as f32 / 255.0, m.color[1] as f32 / 255.0, m.color[2] as f32 / 255.0] },
        ),
        None => (0.5, false, color),
    };
    // creased vertex normals + ray-traced shadow/AO: precomputed once in
    // `prepare_lit_scene` (camera-independent), reused here every frame.
    let normals = &lit.normals;
    let mesh = &part.mesh;
    let (shadow, ao) = (&lit.shadow, &lit.ao);
    for t in 0..mesh.tris.len() {
        let (a, b, c) = mesh.tri(t);
        let idx = mesh.tris[t];
        // shade in world space
        let mut tri_data: [(V3, f32, f32, f32); 3] = [(a, 0.0, 0.0, 0.0), (b, 0.0, 0.0, 0.0), (c, 0.0, 0.0, 0.0)];
        for k in 0..3 {
            let p = tri_data[k].0;
            let vidx = idx[k] as usize;
            let (r, g, bl) = shade(p, normals[vidx], *eye, base, rough, metal, shadow[vidx], ao[vidx], lights);
            tri_data[k].1 = r;
            tri_data[k].2 = g;
            tri_data[k].3 = bl;
        }
        // FIX(P0600-bug): clip_near expects VIEW-space points (camera looks
        // down −Z, near plane at z = −0.6). We used to pass world-space
        // points, which silently discarded every triangle resting fully in
        // the world half-space z > −0.6 mm and sliced the rest at that
        // plane — near faces vanished in iso/front views. Transform to
        // camera space FIRST, then clip, then project.
        let mut view_tri: [(V3, f32, f32, f32); 3] = [(V3::ZERO, 0.0, 0.0, 0.0); 3];
        for k in 0..3 {
            view_tri[k] = (view.apply(&tri_data[k].0), tri_data[k].1, tri_data[k].2, tri_data[k].3);
        }
        let clipped = clip_near(&view_tri);
        if clipped.len() < 3 {
            continue;
        }
        let mut proj: Vec<(f32, f32, f32, f32, f32, f32)> = Vec::with_capacity(clipped.len());
        for (p, r, g, bl) in &clipped {
            if let Some((sx, sy, z)) = cam.project(*p, fb.w as f64, fb.h as f64) {
                proj.push((sx, sy, z, *r, *g, *bl));
            }
        }
        if proj.len() < 3 {
            continue;
        }
        // fan-triangulate the clipped polygon
        for k in 1..proj.len() - 1 {
            let st = ScreenTri {
                x: [proj[0].0, proj[k].0, proj[k + 1].0],
                y: [proj[0].1, proj[k].1, proj[k + 1].1],
                z: [proj[0].2, proj[k].2, proj[k + 1].2],
                r: [proj[0].3, proj[k].3, proj[k + 1].3],
                g: [proj[0].4, proj[k].4, proj[k + 1].4],
                b: [proj[0].5, proj[k].5, proj[k + 1].5],
            };
            if alpha < 0.999 {
                raster_shaded_alpha(fb, &st, alpha);
            } else {
                raster_shaded(fb, &st);
            }
        }
    }
}

/// Vertex normals with a crease threshold: if a vertex's adjacent faces
/// disagree by more than `angle_deg`, that vertex renders flat (hard edge).
pub fn creased_normals(mesh: &crate::geo::mesh::Mesh, angle_deg: f64) -> Vec<V3> {
    let smooth = mesh.vertex_normals(); // area-weighted averages
    let cos_limit = angle_deg.to_radians().cos();
    let mut hard = vec![false; mesh.verts.len()];
    // per vertex: find the max deviation of any adjacent face normal from the average
    let mut faces_per_vert: Vec<Vec<usize>> = vec![Vec::new(); mesh.verts.len()];
    for (t, tri) in mesh.tris.iter().enumerate() {
        for &v in tri {
            faces_per_vert[v as usize].push(t);
        }
    }
    for (v, faces) in faces_per_vert.iter().enumerate() {
        if faces.len() < 2 {
            continue;
        }
        let avg = smooth[v].norm();
        for &f in faces {
            let fn_ = mesh.face_normal(f).norm();
            if avg.dot(&fn_) < cos_limit {
                hard[v] = true;
                break;
            }
        }
    }
    // hard vertices get per-face normals — approximate with the first adjacent face
    let out: Vec<V3> = (0..mesh.verts.len())
        .map(|v| {
            if hard[v] {
                match faces_per_vert[v].first() {
                    Some(&f) => mesh.face_normal(f).norm(),
                    None => smooth[v],
                }
            } else {
                smooth[v]
            }
        })
        .collect();
    out
}

/// Box-filter downsample by factor `s`.
fn downsample(fb: &Framebuffer, s: u32) -> Vec<u8> {
    if s <= 1 {
        return fb.to_rgb_bytes();
    }
    let s = s as usize;
    let ow = fb.w / s;
    let oh = fb.h / s;
    let mut out = Vec::with_capacity(ow * oh * 3);
    for y in 0..oh {
        for x in 0..ow {
            let mut r = 0u32;
            let mut g = 0u32;
            let mut b = 0u32;
            for dy in 0..s {
                for dx in 0..s {
                    let p = fb.px[(y * s + dy) * fb.w + (x * s + dx)];
                    r += (p >> 16) & 0xFF;
                    g += (p >> 8) & 0xFF;
                    b += p & 0xFF;
                }
            }
            let n = (s * s) as u32;
            out.push((r / n) as u8);
            out.push((g / n) as u8);
            out.push((b / n) as u8);
        }
    }
    out
}

/// Per-vertex key-light visibility (soft shadows) and sky occlusion (AO),
/// computed per SPATIAL CELL (CSG meshes duplicate vertices heavily — the
/// cup is 92k verts but only ~4k distinct surface cells) then mapped back.
/// 3 jittered shadow rays toward the key + 6 hemisphere AO rays per cell;
/// the ground plane is analytic.
///
/// PERF: this is the dominant cost of a render. Pass 1 (bucket vertices
/// into cells) is O(V) and cheap — pure hashing, no ray tracing. Pass 2
/// (the actual ray tracing) touches only the handful of *unique* cells and
/// is embarrassingly parallel — each cell reads the shared, read-only BVH
/// and nothing else — so it is split across threads. Pass 3 (scatter cell
/// results back to all V vertices) is O(V) and cheap again.
fn vertex_shadow_ao(
    mesh: &crate::geo::mesh::Mesh,
    normals: &[V3],
    bvh: &crate::geo::bvh::Bvh,
    scene: &crate::geo::mesh::Mesh,
    ground_y: f64,
    lights: &Lights,
) -> (Vec<f32>, Vec<f32>) {
    let n = mesh.verts.len();
    let diag = scene.bbox().size().len().max(1.0);
    let shadow_len = 2.0 * diag;
    let ao_len = 0.06 * diag; // occlusion is LOCAL
    // cell size: ~150 cells across the longest side
    let cell = diag / 150.0;
    let ga = std::f64::consts::PI * (3.0 - 5.0f64.sqrt()); // golden-angle spiral jitter (deterministic)
    let light = V3::new(lights.key.x() as f64, lights.key.y() as f64, lights.key.z() as f64);

    // ---- pass 1: bucket vertices into unique spatial cells (cheap) ----
    use std::collections::HashMap;
    let mut cell_of = vec![0u32; n];
    let mut reps: Vec<(V3, V3)> = Vec::new(); // (position, normal) per unique cell
    let mut seen: HashMap<(i64, i64, i64), u32> = HashMap::with_capacity(n / 4);
    for v in 0..n {
        let p = mesh.verts[v];
        let key = ((p.x() / cell).round() as i64, (p.y() / cell).round() as i64, (p.z() / cell).round() as i64);
        let idx = *seen.entry(key).or_insert_with(|| {
            reps.push((p, normals[v]));
            (reps.len() - 1) as u32
        });
        cell_of[v] = idx;
    }
    drop(seen);

    // ---- pass 2: ray-trace each unique cell once, in parallel (expensive) ----
    let ncells = reps.len();
    let mut results = vec![(1.0f32, 1.0f32); ncells];
    let threads = std::thread::available_parallelism().map(|c| c.get()).unwrap_or(1).min(ncells.max(1));
    if ncells > 0 {
        if threads <= 1 {
            for (i, &(p, nrm)) in reps.iter().enumerate() {
                results[i] = shade_cell(p, nrm, bvh, scene, ground_y, shadow_len, ao_len, ga, light);
            }
        } else {
            let chunk = (ncells + threads - 1) / threads;
            let reps_ref = &reps;
            let mut chunks: Vec<(usize, Vec<(f32, f32)>)> = Vec::with_capacity(threads);
            std::thread::scope(|s| {
                let handles: Vec<_> = reps_ref
                    .chunks(chunk)
                    .enumerate()
                    .map(|(ti, part)| {
                        let start = ti * chunk;
                        s.spawn(move || {
                            (
                                start,
                                part.iter()
                                    .map(|&(p, nrm)| shade_cell(p, nrm, bvh, scene, ground_y, shadow_len, ao_len, ga, light))
                                    .collect::<Vec<(f32, f32)>>(),
                            )
                        })
                    })
                    .collect();
                for h in handles {
                    chunks.push(h.join().unwrap());
                }
            });
            for (start, vals) in chunks {
                results[start..start + vals.len()].copy_from_slice(&vals);
            }
        }
    }

    // ---- pass 3: scatter cell results back to every vertex (cheap) ----
    let mut shadow = vec![1.0f32; n];
    let mut ao = vec![1.0f32; n];
    for v in 0..n {
        let (s, a) = results[cell_of[v] as usize];
        shadow[v] = s;
        ao[v] = a;
    }
    (shadow, ao)
}

/// The exact per-cell shadow/AO computation, factored out of `vertex_shadow_ao`
/// so the single-threaded and multi-threaded paths above share one
/// implementation (never two copies of the ray-tracing math to drift apart).
#[allow(clippy::too_many_arguments)]
fn shade_cell(
    p: V3,
    nrm: V3,
    bvh: &crate::geo::bvh::Bvh,
    scene: &crate::geo::mesh::Mesh,
    ground_y: f64,
    shadow_len: f64,
    ao_len: f64,
    ga: f64,
    light: V3,
) -> (f32, f32) {
    // ---- soft shadow: 3 jittered rays toward the key light ----
    let sh = if nrm.dot(&light) > 0.0 {
        let mut hits = 0.0f32;
        for k in 0..3u32 {
            let ang = ga * k as f64;
            let jitter = 0.10;
            let perp = V3::new(-light.y(), light.x(), 0.0).norm();
            let perp2 = light.cross(&perp).norm();
            let dir = light
                .add(&perp.mul(jitter * ang.cos()))
                .add(&perp2.mul(jitter * ang.sin()))
                .norm();
            let o = p.add(&nrm.mul(0.05));
            if bvh.ray_hit(scene, &o, &dir.mul(shadow_len)).is_some() {
                hits += 1.0;
            }
        }
        1.0 - 0.85 * hits / 3.0
    } else {
        0.0
    };
    // ---- AO: 6 hemisphere rays, short (local), analytic ground ----
    let mut blocked = 0.0f32;
    for k in 0..6u32 {
        let (a1, a2) = (ga * k as f64, (k as f64 + 0.5) / 6.0);
        let t = V3::new(-nrm.y(), nrm.x(), 0.0).norm();
        let b = nrm.cross(&t).norm();
        let r = (1.0 - a2 * a2).sqrt();
        let dir = nrm
            .add(&t.mul(r * a1.cos() * 0.7))
            .add(&b.mul(r * a1.sin() * 0.7))
            .norm();
        let o = p.add(&nrm.mul(0.05));
        let mut hit = bvh.ray_hit(scene, &o, &dir.mul(ao_len)).is_some();
        // analytic ground: down-going rays from above the ground
        if !hit && dir.y() < -1e-6 && o.y() > ground_y {
            let t_g = (ground_y - o.y()) / dir.y();
            if t_g > 0.0 && t_g < ao_len {
                hit = true;
            }
        }
        if hit {
            blocked += 1.0;
        }
    }
    (sh, (1.0 - 0.6 * blocked / 6.0).clamp(0.0, 1.0))
}
