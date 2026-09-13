//! Software renderer with per-pixel ground truth.
//!
//! Rasterises triangle meshes with a z-buffer, perspective-correct world
//! interpolation, Lambert shading, a checker-textured ground and a
//! deterministic world-anchored value noise (identical at the same surface
//! point in both cameras -> perfectly correlated stereo texture).
//!
//! Per-pixel outputs: gray image, camera-space depth, object id and world
//! position. This is what makes every downstream claim measurable.

use image::GrayImage;
use nalgebra::{Unit, Vector3};

use crate::camera::model::Camera;
use crate::core::se3::Se3;

pub const GROUND_ID: i32 = 0;

#[derive(Debug, Clone)]
pub struct TriFace {
    pub a: usize,
    pub b: usize,
    pub c: usize,
    pub object_id: i32,
    pub base: u8,
}

#[derive(Debug, Clone, Default)]
pub struct TriMesh {
    pub verts: Vec<[f64; 3]>,
    pub faces: Vec<TriFace>,
}

impl TriMesh {
    pub fn add_box(&mut self, center: [f64; 3], half: [f64; 3], yaw: f64, object_id: i32, base: u8) {
        let (cy, sy) = (yaw.cos(), yaw.sin());
        // rotation about world Y (up): x' = cy*x + sy*z? (yaw convention: 0 = axis aligned)
        let rot = |x: f64, z: f64| (cy * x + sy * z, -sy * x + cy * z);
        let base_idx = self.verts.len();
        for dz in [-1.0, 1.0] {
            for dy in [-1.0, 1.0] {
                for dx in [-1.0, 1.0] {
                    let (rx, rz) = rot(dx * half[0], dz * half[2]);
                    self.verts.push([
                        center[0] + rx,
                        center[1] + dy * half[1],
                        center[2] + rz,
                    ]);
                }
            }
        }
        // 8 verts ordered: dz, dy, dx (each -1..1): idx = dz*4 + dy*2 + dx (as 0/1 bits)
        let v = |dz: usize, dy: usize, dx: usize| base_idx + dz * 4 + dy * 2 + dx;
        let quads: [[usize; 4]; 6] = [
            [v(0, 0, 0), v(0, 1, 0), v(0, 1, 1), v(0, 0, 1)], // z- face
            [v(1, 0, 0), v(1, 0, 1), v(1, 1, 1), v(1, 1, 0)], // z+ face
            [v(0, 0, 0), v(1, 0, 0), v(1, 0, 1), v(0, 0, 1)], // y- face
            [v(0, 1, 0), v(0, 1, 1), v(1, 1, 1), v(1, 1, 0)], // y+ face
            [v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0)], // x- face
            [v(0, 0, 1), v(0, 1, 1), v(1, 1, 1), v(1, 0, 1)], // x+ face
        ];
        for q in quads {
            self.faces.push(TriFace { a: q[0], b: q[1], c: q[2], object_id, base });
            self.faces.push(TriFace { a: q[0], b: q[2], c: q[3], object_id, base });
        }
    }

    pub fn add_cylinder(&mut self, center: [f64; 3], radius: f64, height: f64, n_seg: usize, object_id: i32, base: u8) {
        let base_idx = self.verts.len();
        for i in 0..n_seg {
            let a = i as f64 * 2.0 * std::f64::consts::PI / n_seg as f64;
            self.verts.push([
                center[0] + radius * a.cos(),
                center[1],
                center[2] + radius * a.sin(),
            ]);
        }
        for i in 0..n_seg {
            let a = i as f64 * 2.0 * std::f64::consts::PI / n_seg as f64;
            self.verts.push([
                center[0] + radius * a.cos(),
                center[1] + height,
                center[2] + radius * a.sin(),
            ]);
        }
        self.verts.push([center[0], center[1] + height, center[2]]); // top centre
        let top_center = self.verts.len() - 1;
        for i in 0..n_seg {
            let i2 = (i + 1) % n_seg;
            // side quad (two tris, outward normals)
            self.faces.push(TriFace { a: base_idx + i, b: base_idx + n_seg + i2, c: base_idx + i2, object_id, base });
            self.faces.push(TriFace { a: base_idx + i, b: base_idx + n_seg + i, c: base_idx + n_seg + i2, object_id, base });
            // top cap
            self.faces.push(TriFace { a: base_idx + n_seg + i, b: top_center, c: base_idx + n_seg + i2, object_id, base });
        }
    }

    pub fn add_ground(&mut self, half: f64) {
        let b = self.verts.len();
        self.verts.push([-half, 0.0, -half]);
        self.verts.push([half, 0.0, -half]);
        self.verts.push([half, 0.0, half]);
        self.verts.push([-half, 0.0, half]);
        // upward-facing normals (visible from above)
        for t in [[b, b + 2, b + 1], [b, b + 3, b + 2]] {
            self.faces.push(TriFace { a: t[0], b: t[1], c: t[2], object_id: GROUND_ID, base: 0 });
        }
    }
}

/// Per-pixel render outputs (all same size).
pub struct RenderResult {
    pub gray: GrayImage,
    /// Camera-space z (m); 0 = no return.
    pub depth: Vec<f32>,
    /// Object id per pixel; -1 = no return.
    pub obj_id: Vec<i32>,
    /// World position per pixel.
    pub world: Vec<[f32; 3]>,
    pub width: usize,
    pub height: usize,
}

#[inline]
fn hash3(x: i64, y: i64, z: i64) -> f64 {
    let mut h = (x as u64).wrapping_mul(0x9E3779B185EBCA87u64)
        ^ (y as u64).wrapping_mul(0xC2B2AE3D27D4EB4Fu64)
        ^ (z as u64).wrapping_mul(0x165667B19E3779F9u64);
    h ^= h >> 15;
    h = h.wrapping_mul(0x85EBCA6Bu64);
    h ^= h >> 13;
    ((h >> 11) as f64 / 9007199254740992.0) - 0.5 // [-0.5, 0.5)
}

/// World-anchored trilinear value noise (4 cm feature scale, +/- 9 gray
/// levels). Smooth (band-limited) so corresponding left/right pixels see
/// consistent values; identical at the same surface point in both cameras.
/// The 4 cm scale keeps usable census texture at QVGA..VGA resolutions.
#[inline]
fn world_noise(p: [f32; 3]) -> f64 {
    const CELL: f64 = 0.04;
    let fx = p[0] as f64 / CELL;
    let fy = p[1] as f64 / CELL;
    let fz = p[2] as f64 / CELL;
    let x0 = fx.floor() as i64;
    let y0 = fy.floor() as i64;
    let z0 = fz.floor() as i64;
    let tx = fx - x0 as f64;
    let ty = fy - y0 as f64;
    let tz = fz - z0 as f64;
    let c000 = hash3(x0, y0, z0);
    let c100 = hash3(x0 + 1, y0, z0);
    let c010 = hash3(x0, y0 + 1, z0);
    let c110 = hash3(x0 + 1, y0 + 1, z0);
    let c001 = hash3(x0, y0, z0 + 1);
    let c101 = hash3(x0 + 1, y0, z0 + 1);
    let c011 = hash3(x0, y0 + 1, z0 + 1);
    let c111 = hash3(x0 + 1, y0 + 1, z0 + 1);
    let cx00 = c000 + (c100 - c000) * tx;
    let cx10 = c010 + (c110 - c010) * tx;
    let cx01 = c001 + (c101 - c001) * tx;
    let cx11 = c011 + (c111 - c011) * tx;
    let cxy0 = cx00 + (cx10 - cx00) * ty;
    let cxy1 = cx01 + (cx11 - cx01) * ty;
    let c = cxy0 + (cxy1 - cxy0) * tz;
    // single smooth octave (4 cm, +/- 9): continuous spectrum avoids the
    // disparity-search aliasing that a periodic/blocky texture would cause
    c * 18.0
}

const LIGHT: [f64; 3] = [0.4, 0.75, 0.5];

/// Render the mesh from one camera (no sensor noise yet).
pub fn render(mesh: &TriMesh, cam: &Camera) -> RenderResult {
    let k = cam.k;
    let (w, h) = (k.width as usize, k.height as usize);
    let n_px = w * h;
    let mut gray = vec![0u8; n_px];
    let mut depth = vec![0f32; n_px];
    let mut obj_id = vec![-1i32; n_px];
    let mut world = vec![[0f32; 3]; n_px];
    let cam_inv = cam.pose.inverse();

    let near = 0.08f64;

    for face in &mesh.faces {
        let pa = mesh.verts[face.a];
        let pb = mesh.verts[face.b];
        let pc = mesh.verts[face.c];
        // world-space face normal
        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let nx = ab[1] * ac[2] - ab[2] * ac[1];
        let ny = ab[2] * ac[0] - ab[0] * ac[2];
        let nz = ab[0] * ac[1] - ab[1] * ac[0];
        let nl = (nx * nx + ny * ny + nz * nz).sqrt();
        if nl < 1e-12 {
            continue;
        }
        let (nx, ny, nz) = (nx / nl, ny / nl, nz / nl);
        let lambert = 0.55 + 0.45 * (nx * LIGHT[0] + ny * LIGHT[1] + nz * LIGHT[2]).max(0.0);

        // to cam space
        let to_cam = |p: [f64; 3]| -> [f64; 3] {
            let v = cam_inv.transform_point(&Vector3::new(p[0], p[1], p[2]));
            [v[0], v[1], v[2]]
        };
        let ca = to_cam(pa);
        let cb = to_cam(pb);
        let cc = to_cam(pc);
        // backface cull (normal toward camera)
        let centroid = [
            (ca[0] + cb[0] + cc[0]) / 3.0,
            (ca[1] + cb[1] + cc[1]) / 3.0,
            (ca[2] + cb[2] + cc[2]) / 3.0,
        ];
        let n_cam = cam.pose.inverse().transform_dir(&Vector3::new(nx, ny, nz));
        if n_cam[0] * centroid[0] + n_cam[1] * centroid[1] + n_cam[2] * centroid[2] > 0.0 {
            continue; // facing away
        }

        // near clipping (Sutherland-Hodgman against z >= near)
        let mut poly_cam = vec![ca, cb, cc];
        let mut poly_world = vec![pa, pb, pc];
        let mut out_cam = Vec::with_capacity(4);
        let mut out_world = Vec::with_capacity(4);
        for i in 0..poly_cam.len() {
            let j = (i + 1) % poly_cam.len();
            let zi = poly_cam[i][2];
            let zj = poly_cam[j][2];
            let pi = poly_cam[i];
            let pj = poly_cam[j];
            let wi = poly_world[i];
            let wj = poly_world[j];
            if zi >= near {
                out_cam.push(pi);
                out_world.push(wi);
                if zj < near {
                    let t = (near - zi) / (zj - zi);
                    out_cam.push([
                        pi[0] + t * (pj[0] - pi[0]),
                        pi[1] + t * (pj[1] - pi[1]),
                        near,
                    ]);
                    out_world.push([
                        wi[0] + t * (wj[0] - wi[0]),
                        wi[1] + t * (wj[1] - wi[1]),
                        wi[2] + t * (wj[2] - wi[2]),
                    ]);
                }
            } else if zj >= near {
                let t = (near - zi) / (zj - zi);
                out_cam.push([
                    pi[0] + t * (pj[0] - pi[0]),
                    pi[1] + t * (pj[1] - pi[1]),
                    near,
                ]);
                out_world.push([
                    wi[0] + t * (wj[0] - wi[0]),
                    wi[1] + t * (wj[1] - wi[1]),
                    wi[2] + t * (wj[2] - wi[2]),
                ]);
            }
        }
        if out_cam.len() < 3 {
            continue;
        }
        poly_cam = out_cam;
        poly_world = out_world;

        // rasterise every triangle of the clipped polygon
        for t in 0..poly_cam.len() {
            let (ia, ib, ic) = (t, (t + 1) % poly_cam.len(), (t + 2) % poly_cam.len());
            let va = poly_cam[ia];
            let vb = poly_cam[ib];
            let vc = poly_cam[ic];
            let wa = poly_world[ia];
            let wb = poly_world[ib];
            let wc = poly_world[ic];
            // project
            let proj = |v: [f64; 3]| -> (f64, f64) {
                (k.fx * v[0] / v[2] + k.cx, k.cy - k.fy * v[1] / v[2])
            };
            let (ax, ay) = proj(va);
            let (bx, by) = proj(vb);
            let (cx, cy) = proj(vc);
            let area = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
            if area.abs() < 1e-9 {
                continue;
            }
            let inv_area = 1.0 / area;
            let x0 = (ax.min(bx).min(cx).ceil().max(0.0) as i64).min(w as i64 - 1);
            let x1 = (ax.max(bx).max(cx).floor().min((w - 1) as f64) as i64).max(0);
            let y0 = (ay.min(by).min(cy).ceil().max(0.0) as i64).min(h as i64 - 1);
            let y1 = (ay.max(by).max(cy).floor().min((h - 1) as f64) as i64).max(0);
            // 1/z at vertices
            let iz = [1.0 / va[2], 1.0 / vb[2], 1.0 / vc[2]];
            for py in y0..=y1 {
                for px in x0..=x1 {
                    let pxf = px as f64 + 0.5;
                    let pyf = py as f64 + 0.5;
                    let e0 = (bx - ax) * (pyf - ay) - (by - ay) * (pxf - ax);
                    let e1 = (cx - bx) * (pyf - by) - (cy - by) * (pxf - bx);
                    let e2 = (ax - cx) * (pyf - cy) - (ay - cy) * (pxf - cx);
                    // e0/e1/e2 are the barycentric weights of vertices C/A/B
                    // (sub-triangle areas opposite to those vertices).
                    let (lc, la, lb) = (e0 * inv_area, e1 * inv_area, e2 * inv_area);
                    // inside test: same sign as area (nonzero)
                    let inside = if area > 0.0 {
                        la >= 0.0 && lb >= 0.0 && lc >= 0.0
                    } else {
                        la <= 0.0 && lb <= 0.0 && lc <= 0.0
                    };
                    if !inside {
                        continue;
                    }
                    // perspective-correct depth
                    let invz = la * iz[0] + lb * iz[1] + lc * iz[2];
                    if invz <= 1e-9 {
                        continue;
                    }
                    let z = 1.0 / invz;
                    let idx = py as usize * w + px as usize;
                    if depth[idx] > 0.0 && z >= depth[idx] as f64 {
                        continue;
                    }
                    // perspective-correct world position
                    let s = la * iz[0] + lb * iz[1] + lc * iz[2];
                    let ua = la * iz[0] / s;
                    let ub = lb * iz[1] / s;
                    let uc = lc * iz[2] / s;
                    let wpos = [
                        (ua * wa[0] + ub * wb[0] + uc * wc[0]) as f32,
                        (ua * wa[1] + ub * wb[1] + uc * wc[1]) as f32,
                        (ua * wa[2] + ub * wb[2] + uc * wc[2]) as f32,
                    ];
                    // shade
                    let shade: f64 = if face.object_id == GROUND_ID {
                        // soft random mosaic: 0.25 m cells with random per-cell
                        // gray (+/- 30). Non-periodic -> no disparity-search
                        // aliasing; every cell boundary is matchable texture.
                        let cell = |i: i64, k: i64| -> f64 {
                            118.0 + 60.0 * hash3(i, 0, k)
                        };
                        let smoothstep = |t: f64| -> f64 {
                            let t = t.clamp(0.0, 1.0);
                            t * t * (3.0 - 2.0 * t)
                        };
                        let fx = wpos[0] as f64 / 0.25;
                        let fz = wpos[2] as f64 / 0.25;
                        let ix = fx.floor() as i64;
                        let iz = fz.floor() as i64;
                        let tx = smoothstep((fx - ix as f64 - 0.45) / 0.1);
                        let tz = smoothstep((fz - iz as f64 - 0.45) / 0.1);
                        let c00 = cell(ix, iz);
                        let c10 = cell(ix + 1, iz);
                        let c01 = cell(ix, iz + 1);
                        let c11 = cell(ix + 1, iz + 1);
                        (1.0 - tz) * ((1.0 - tx) * c00 + tx * c10)
                            + tz * ((1.0 - tx) * c01 + tx * c11)
                    } else {
                        face.base as f64 * lambert
                    };
                    let g = (shade + world_noise(wpos)).round().clamp(0.0, 255.0);
                    gray[idx] = g as u8;
                    depth[idx] = z as f32;
                    obj_id[idx] = face.object_id;
                    world[idx] = wpos;
                }
            }
        }
    }

    RenderResult {
        gray: GrayImage::from_raw(w as u32, h as u32, gray).unwrap(),
        depth,
        obj_id,
        world,
        width: w,
        height: h,
    }
}

/// Add independent per-camera sensor noise (in-place).
pub fn add_sensor_noise(res: &mut RenderResult, sigma: f64, rng: &mut crate::core::rng::GaussRng) {
    if sigma <= 0.0 {
        return;
    }
    for p in res.gray.pixels_mut() {
        let v = p[0] as f64 + rng.gauss(0.0, sigma);
        p[0] = v.round().clamp(0.0, 255.0) as u8;
    }
}

#[allow(dead_code)]
fn _unused(_: Unit<Vector3<f64>>) {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::camera::model::{Intrinsics, look_at};
    use crate::camera::model::StereoRig;

    #[test]
    fn box_renders_at_expected_depth() {
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let cam = Camera::new(
            k,
            Se3::from_parts(
                look_at(Vector3::new(0.0, 1.6, 0.0), Vector3::new(0.0, 1.6, 4.0), Vector3::y()),
                Vector3::new(0.0, 1.6, 0.0),
            ),
        );
        let mut mesh = TriMesh::default();
        mesh.add_ground(12.0);
        mesh.add_box([0.0, 0.5, 4.0], [0.5, 0.5, 0.5], 0.0, 7, 180);
        let res = render(&mesh, &cam);
        // pixel (320, 420): ray hits the box front face (z = 3.5), y ~ 0.70
        let c = 420 * 640 + 320;
        assert_eq!(res.obj_id[c], 7, "centre pixel on box");
        assert!((res.depth[c] - 3.5).abs() < 0.02, "depth {}", res.depth[c]);
        // world position of that pixel should be on the front face plane
        assert!((res.world[c][2] - 3.5).abs() < 0.02, "world z {}", res.world[c][2]);
        assert!((res.world[c][1] - 0.70).abs() < 0.05, "world y {} ", res.world[c][1]);
    }

    /// THE critical test: the rendered stereo pair is geometrically consistent
    /// with the pinhole model - right(x - d_gt) == left(x) up to noise.
    #[test]
    fn rendered_stereo_pair_is_consistent() {
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let pose = Se3::from_parts(
            look_at(Vector3::new(0.2, 1.6, 0.0), Vector3::new(0.0, 1.4, 4.0), Vector3::y()),
            Vector3::new(0.2, 1.6, 0.0),
        );
        let rig = StereoRig::canonical(k, pose, 0.16);
        let mut mesh = TriMesh::default();
        mesh.add_ground(12.0);
        mesh.add_box([1.0, 0.4, 3.5], [0.4, 0.4, 0.4], 0.0, 3, 170);
        mesh.add_cylinder([-0.8, 0.0, 3.0], 0.18, 1.7, 12, 4, 100);
        let left = render(&mesh, &rig.left);
        let right = render(&mesh, &rig.right);
        let f = k.fx;
        let b = 0.16;
        let mut checked = 0;
        let mut bad = 0;
        for y in (40..440).step_by(4) {
            for x in (40..600).step_by(4) {
                let i = y * 640 + x;
                if left.obj_id[i] < 0 {
                    continue;
                }
                let z = left.depth[i] as f64;
                if z < 0.5 {
                    continue;
                }
                // skip pixels near an object-id boundary (silhouette mixing)
                let idc = left.obj_id[i];
                let mut edge = false;
                for (dx, dy) in [(1i64, 0i64), (-1, 0), (0, 1), (0, -1)] {
                    let j = (((y as i64 + dy).clamp(0, 479)) * 640
                        + (x as i64 + dx).clamp(0, 639)) as usize;
                    if left.obj_id[j] != idc {
                        edge = true;
                        break;
                    }
                }
                if edge {
                    continue;
                }
                let d = f * b / z;
                let xr = x as f64 - d;
                if xr < 1.0 || xr > 638.0 {
                    continue;
                }
                // sample right image at the corresponding location
                let x0 = xr.floor() as usize;
                let fx = xr - x0 as f64;
                let r0 = right.gray.get_pixel(x0 as u32, y as u32)[0] as f64;
                let r1 = right.gray.get_pixel(x0 as u32 + 1, y as u32)[0] as f64;
                let r = r0 * (1.0 - fx) + r1 * fx;
                let l = left.gray.get_pixel(x as u32, y as u32)[0] as f64;
                // skip occlusion boundaries: the right view's corresponding
                // pixel may see a different surface (id mismatch)
                let rid = right.obj_id[y * 640 + x0];
                if rid != idc {
                    continue;
                }
                // skip mosaic-edge samples: subpixel footprint can straddle
                // the soft cell transition (high local contrast)
                if idc == 0 {
                    let mut lmin = 255.0f64;
                    let mut lmax = 0.0f64;
                    for dy in -1i64..=1 {
                        for dx in -1i64..=1 {
                            let v = left.gray.get_pixel((x as i64 + dx).clamp(0, 639) as u32, (y as i64 + dy).clamp(0, 479) as u32)[0] as f64;
                            lmin = lmin.min(v);
                            lmax = lmax.max(v);
                        }
                    }
                    if lmax - lmin > 22.0 {
                        continue;
                    }
                }
                if (l - r).abs() > 14.0 {
                    bad += 1;
                }
                checked += 1;
            }
        }
        assert!(checked > 1200, "checked {checked} pixels");
        assert!(bad as f64 / (checked as f64) < 0.02, "inconsistent: {bad}/{checked}");
    }

    #[test]
    fn world_noise_is_deterministic() {
        let a = world_noise([0.13, 0.7, -0.44]);
        let b = world_noise([0.13, 0.7, -0.44]);
        assert_eq!(a, b);
        let c = world_noise([0.16, 0.7, -0.44]);
        assert!(a != c, "adjacent cells differ");
    }
}

