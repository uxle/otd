//! P0500–P0510 — measurement: volume/centroid/area live in Mesh (divergence
//! theorem). Here: cross-section slice area at a height (for `simulate
//! collapse` stress checks) via segment rasterization + flood fill.

use crate::geo::mesh::Mesh;
use crate::math3::V3;

/// Area of the mesh's horizontal cross-section at height y (mm²).
/// Triangles crossing the plane produce line segments; we rasterize the
/// segments into a grid, flood-fill from outside, and count interior cells.
/// Grid resolution ~0.5% of the bbox — accurate to ~1% for chunky shapes.
pub fn slice_area_at(m: &Mesh, y: f64) -> f64 {
    let bb = m.bbox();
    if m.is_empty() || y < bb.min.y() - 1e-9 || y > bb.max.y() + 1e-9 {
        return 0.0;
    }
    // collect segments in the XZ plane
    let mut segs: Vec<(f64, f64, f64, f64)> = Vec::new(); // (x0,z0,x1,z1)
    for t in 0..m.tris.len() {
        let (a, b, c) = m.tri(t);
        let pts = [a, b, c];
        let mut below: Vec<V3> = Vec::new();
        let mut above: Vec<V3> = Vec::new();
        for p in pts {
            if p.y() < y { below.push(p) } else { above.push(p) }
        }
        if below.is_empty() || above.is_empty() {
            continue;
        }
        // crossing points
        let mut cross: Vec<(f64, f64)> = Vec::new();
        for i in 0..3 {
            let p = pts[i];
            let q = pts[(i + 1) % 3];
            let (pi, qi) = (p.y() < y, q.y() < y);
            if pi != qi {
                let t_frac = if (q.y() - p.y()).abs() > 1e-12 {
                    (y - p.y()) / (q.y() - p.y())
                } else { 0.0 };
                let x = p.x() + (q.x() - p.x()) * t_frac;
                let z = p.z() + (q.z() - p.z()) * t_frac;
                cross.push((x, z));
            }
        }
        if cross.len() >= 2 {
            segs.push((cross[0].0, cross[0].1, cross[1].0, cross[1].1));
        }
    }
    if segs.is_empty() {
        return 0.0;
    }
    // rasterize: grid over bbox XZ (pad slightly)
    let pad = (bb.size().x() + bb.size().z()) * 0.01 + 1.0;
    let (x0, x1) = (bb.min.x() - pad, bb.max.x() + pad);
    let (z0, z1) = (bb.min.z() - pad, bb.max.z() + pad);
    let n = 160usize;
    let dx = (x1 - x0) / n as f64;
    let dz = (z1 - z0) / n as f64;
    let cell = dx * dz;
    let mut solid = vec![false; n * n];
    // mark boundary cells: supercover line rasterization
    for (ax, az, bx, bz) in segs {
        let (ia, ja) = (((ax - x0) / dx) as i64, ((az - z0) / dz) as i64);
        let (ib, jb) = (((bx - x0) / dx) as i64, ((bz - z0) / dz) as i64);
        // Bresenham-ish supercover
        let (mut i, mut j) = (ia, ja);
        let (di, dj) = (ib - ia, jb - ja);
        let (si, sj) = (di.signum(), dj.signum());
        let (adx, adz) = (di.abs(), dj.abs());
        let (mut errx, mut errz) = (adx, adz);
        let steps = (adx + adz).max(1);
        for _ in 0..=steps {
            if i >= 0 && i < n as i64 && j >= 0 && j < n as i64 {
                solid[j as usize * n + i as usize] = true;
            }
            if i == ib && j == jb { break; }
            if errx >= errz {
                i += si;
                errx -= adz.max(1);
            } else {
                j += sj;
                errz -= adx.max(1);
            }
        }
    }
    // flood fill from borders: everything reachable = outside
    let mut outside = vec![false; n * n];
    let mut stack: Vec<usize> = Vec::new();
    for i in 0..n {
        for j in [0usize, n - 1] {
            let idx = j * n + i;
            if !solid[idx] && !outside[idx] { outside[idx] = true; stack.push(idx); }
        }
    }
    for j in 0..n {
        for i in [0usize, n - 1] {
            let idx = j * n + i;
            if !solid[idx] && !outside[idx] { outside[idx] = true; stack.push(idx); }
        }
    }
    while let Some(idx) = stack.pop() {
        let (i, j) = (idx % n, idx / n);
        let neighbors = [
            (i > 0).then(|| idx - 1),
            (i + 1 < n).then(|| idx + 1),
            (j > 0).then(|| idx - n),
            (j + 1 < n).then(|| idx + n),
        ];
        for nb in neighbors.into_iter().flatten() {
            if !solid[nb] && !outside[nb] {
                outside[nb] = true;
                stack.push(nb);
            }
        }
    }
    let mut area = 0.0f64;
    for idx in 0..n * n {
        if !outside[idx] {
            area += cell;
        }
    }
    area
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geo::prims::{cube, frustum, sphere};

    #[test]
    fn slice_of_cube() {
        let m = cube(20.0, 20.0, 20.0).0;
        let a = slice_area_at(&m, 10.0);
        assert!((a - 400.0).abs() / 400.0 < 0.05, "area {}", a);
    }

    #[test]
    fn slice_of_cylinder() {
        // r=10 → π·100 = 314.16
        let m = frustum(10.0, 10.0, 20.0, 48).0;
        let a = slice_area_at(&m, 10.0);
        assert!((a - 314.16).abs() / 314.16 < 0.03, "area {}", a);
    }

    #[test]
    fn slice_outside_is_zero() {
        let m = cube(20.0, 20.0, 20.0).0;
        assert_eq!(slice_area_at(&m, 100.0), 0.0);
    }

    #[test]
    fn slice_of_sphere_varies() {
        let m = sphere(10.0, 32).0;
        // slice at center height (y = r) = full circle πr²
        let a = slice_area_at(&m, 10.0);
        assert!((a - 314.16).abs() / 314.16 < 0.08, "area {}", a);
        // slice near the bottom is small
        let b = slice_area_at(&m, 1.0);
        assert!(b < 100.0, "near-bottom slice {}", b);
    }
}

// ---------- P1180: the mass moment of inertia tensor ----------

/// Second moments of a closed mesh about the ORIGIN (uniform density 1):
/// Sxx = ∫x²dV etc., plus first moments and volume. Exact closed forms per
/// signed tetrahedron conv(origin, a, b, c) — the same chain identity that
/// powers `volume_signed`, validated against the analytic cube.
#[derive(Clone, Copy, Debug, Default)]
pub struct SecondMoments {
    pub xx: f64,
    pub yy: f64,
    pub zz: f64,
    pub xy: f64,
    pub xz: f64,
    pub yz: f64,
    pub vol: f64,
    pub mx: f64,
    pub my: f64,
    pub mz: f64,
}

pub fn second_moments(m: &Mesh) -> SecondMoments {
    let mut s = SecondMoments::default();
    for t in 0..m.tris.len() {
        let (a, b, c) = m.tri(t);
        let det = a.dot(&b.cross(&c));
        s.vol += det / 6.0;
        s.xx += det * quad(a.x(), b.x(), c.x()) / 60.0;
        s.yy += det * quad(a.y(), b.y(), c.y()) / 60.0;
        s.zz += det * quad(a.z(), b.z(), c.z()) / 60.0;
        s.xy += det * mixed(a.x(), a.y(), b.x(), b.y(), c.x(), c.y()) / 120.0;
        s.xz += det * mixed(a.x(), a.z(), b.x(), b.z(), c.x(), c.z()) / 120.0;
        s.yz += det * mixed(a.y(), a.z(), b.y(), b.z(), c.y(), c.z()) / 120.0;
        s.mx += det * (a.x() + b.x() + c.x()) / 24.0;
        s.my += det * (a.y() + b.y() + c.y()) / 24.0;
        s.mz += det * (a.z() + b.z() + c.z()) / 24.0;
    }
    s
}

/// ∫(αu + βv + γw)² over the canonical simplex = (α² + β² + γ² + αβ + αγ + βγ)/60.
fn quad(p: f64, q: f64, r: f64) -> f64 {
    p * p + q * q + r * r + p * q + p * r + q * r
}

/// ∫(αu + βv + γw)(α′u + β′v + γ′w) over the canonical simplex
/// = [2(αα′+ββ′+γγ′) + (αβ′+βα′) + (αγ′+γα′) + (βγ′+γβ′)]/120.
fn mixed(a1: f64, a2: f64, b1: f64, b2: f64, c1: f64, c2: f64) -> f64 {
    2.0 * (a1 * a2 + b1 * b2 + c1 * c2)
        + a1 * b2 + b1 * a2
        + a1 * c2 + c1 * a2
        + b1 * c2 + c1 * b2
}

/// The mass moment of inertia tensor of a closed mesh with uniform density
/// ρ (g/cm³ — the material database unit), about its CENTER OF MASS, in
/// g·mm². Ixx = ∫(y²+z²)ρdV etc.; off-diagonals are the −∫xy products.
pub struct Inertia {
    /// [Ixx, Iyy, Izz] (g·mm²)
    pub diag: [f64; 3],
    /// [Ixy, Ixz, Iyz] (g·mm²)
    pub off: [f64; 3],
    pub mass_g: f64,
    /// the center of mass the tensor was shifted to
    pub com: V3,
}

pub fn inertia(m: &Mesh, density_g_cm3: f64) -> Option<Inertia> {
    let s = second_moments(m);
    if s.vol.abs() < 1e-12 {
        return None;
    }
    let rho = density_g_cm3 / 1000.0; // g/cm³ → g/mm³
    let mass = s.vol * rho; // grams
    let com = V3::new(s.mx / s.vol, s.my / s.vol, s.mz / s.vol);
    // about the origin …
    let ixx_o = (s.yy + s.zz) * rho;
    let iyy_o = (s.xx + s.zz) * rho;
    let izz_o = (s.xx + s.yy) * rho;
    let ixy_o = -s.xy * rho;
    let ixz_o = -s.xz * rho;
    let iyz_o = -s.yz * rho;
    // … then the parallel-axis theorem moves us to the COM
    let (dx, dy, dz) = (com.x(), com.y(), com.z());
    let m = mass;
    Some(Inertia {
        diag: [
            ixx_o - m * (dy * dy + dz * dz),
            iyy_o - m * (dx * dx + dz * dz),
            izz_o - m * (dx * dx + dy * dy),
        ],
        off: [ixy_o + m * dx * dy, ixz_o + m * dx * dz, iyz_o + m * dy * dz],
        mass_g: m,
        com,
    })
}

#[cfg(test)]
mod inertia_tests {
    use super::*;
    use crate::geo::prims::{cube, sphere};

    #[test]
    fn cube_exact() {
        // 20mm cube, ρ = 1: m = 8 g; I_xx = m(s²+s²)/12 = 8·400/6… s=20mm:
        // volume 8000 mm³ = 8 cm³ → 8 g; I = 8·(20²+20²)/12 = 533.33 g·mm²·… wait
        // units: I in g·mm²: m(g)·mm²/12·(s²+s²) with s in mm → 8·(400+400)/12 = 533.3
        let m = cube(20.0, 20.0, 20.0).0;
        let i = inertia(&m, 1.0).unwrap();
        let want = i.mass_g * (20.0f64 * 20.0 + 20.0 * 20.0) / 12.0;
        for d in i.diag {
            assert!((d - want).abs() / want < 1e-9, "diag {} vs {}", d, want);
        }
        for o in i.off {
            assert!(o.abs() < 1e-6, "cube off-diagonals vanish, got {}", o);
        }
        assert!((i.mass_g - 8.0).abs() < 1e-9);
    }

    #[test]
    fn sphere_close_to_analytic() {
        // 2/5 m r² — the faceted sphere converges as smooth grows
        let (m, _) = sphere(10.0, 32);
        let i = inertia(&m, 2.7).unwrap();
        let r = 10.0;
        let m_analytic = 4.0 / 3.0 * std::f64::consts::PI * r * r * r / 1000.0 * 2.7;
        let want = 0.4 * m_analytic * r * r;
        let avg = (i.diag[0] + i.diag[1] + i.diag[2]) / 3.0;
        // the faceted inscribed sphere runs ~2–3 % light (chord deficit,
        // r²-weighted) — honest tolerance for smooth 32
        assert!((avg - want).abs() / want < 0.04, "I {} vs analytic {}", avg, want);
        // isotropic
        let spread = (i.diag[0] - i.diag[1]).abs() / avg;
        assert!(spread < 0.02, "sphere must be isotropic, spread {}", spread);
        assert!(i.off.iter().all(|o| o.abs() / avg < 0.01));
    }

    #[test]
    fn moments_first_and_second_on_cube() {
        // prims::cube is x/z-centered, y 0..h: x∈[−10,10], y∈[0,20], z∈[−10,10]
        let m = cube(20.0, 20.0, 20.0).0;
        let s = second_moments(&m);
        assert!((s.vol - 8000.0).abs() < 1e-9);
        // ∫x dV = 0 (x centered); ∫y dV = 10·8000
        assert!(s.mx.abs() < 1e-6, "mx {}", s.mx);
        assert!((s.my - 80000.0).abs() < 1e-6, "my {}", s.my);
        // ∫x² dV = ∫₋₁₀¹⁰ x²dx·20·20 = (2000/3)·400 = 266,666.67
        assert!((s.xx - 2000.0 / 3.0 * 400.0).abs() < 1e-6, "xx {}", s.xx);
        // ∫xy = (∫x)(∫y)(∫z) = 0 (x centered)
        assert!(s.xy.abs() < 1e-6, "xy {}", s.xy);
        // ∫y² = ∫₀²⁰ y²dy·20·20 = (8000/3)·400 = 1,066,666.67
        assert!((s.yy - 8000.0 / 3.0 * 400.0).abs() < 1e-6, "yy {}", s.yy);
    }
}
