//! P0210–P0240 — the 12 primitives. Every shape is a closed manifold resting
//! on the ground (y = 0), in millimeters, f64.

use crate::geo::mesh::{Kind, Mesh};
use crate::math3::V3;

pub const DEFAULT_SMOOTH: u32 = 48;

/// cube: s (all sides) or w/d/h. Centered in XZ, resting on ground.
pub fn cube(w: f64, d: f64, h: f64) -> (Mesh, Kind) {
    let mut m = Mesh::new();
    let (x0, x1) = (-w / 2.0, w / 2.0);
    let (y0, y1) = (0.0, h);
    let (z0, z1) = (-d / 2.0, d / 2.0);
    m.add_quad(V3::new(x0, y1, z0), V3::new(x0, y1, z1), V3::new(x1, y1, z1), V3::new(x1, y1, z0)); // +y
    m.add_quad(V3::new(x0, y0, z0), V3::new(x1, y0, z0), V3::new(x1, y0, z1), V3::new(x0, y0, z1)); // -y
    m.add_quad(V3::new(x0, y0, z1), V3::new(x1, y0, z1), V3::new(x1, y1, z1), V3::new(x0, y1, z1)); // +z
    m.add_quad(V3::new(x0, y0, z0), V3::new(x0, y1, z0), V3::new(x1, y1, z0), V3::new(x1, y0, z0)); // -z
    m.add_quad(V3::new(x1, y0, z0), V3::new(x1, y1, z0), V3::new(x1, y1, z1), V3::new(x1, y0, z1)); // +x
    m.add_quad(V3::new(x0, y0, z0), V3::new(x0, y0, z1), V3::new(x0, y1, z1), V3::new(x0, y1, z0)); // -x
    m.ensure_outward();
    (m, Kind::Box { w, d, h })
}

/// sphere resting on the ground (center at y = r).
pub fn sphere(r: f64, smooth: u32) -> (Mesh, Kind) {
    let meridians = smooth.max(8);
    let rings = (meridians / 2).max(4);
    let c = V3::new(0.0, r, 0.0);
    let mut m = Mesh::new();
    // vertex grid: rings 0..=rings (0 = north pole), meridians 0..meridians
    let mut grid: Vec<Vec<u32>> = Vec::with_capacity(rings as usize + 1);
    for i in 0..=rings {
        let phi = std::f64::consts::PI * (i as f64) / (rings as f64); // 0..π
        let sy = phi.cos();
        let sr = phi.sin();
        let mut row = Vec::with_capacity(meridians as usize + 1);
        for j in 0..=meridians {
            let th = 2.0 * std::f64::consts::PI * (j as f64) / (meridians as f64);
            let p = c.add(&V3::new(r * sr * th.cos(), r * sy, r * sr * th.sin()));
            row.push(m.add_vert(p));
        }
        grid.push(row);
    }
    for i in 0..rings as usize {
        for j in 0..meridians as usize {
            let (a, b, c_, d) = (grid[i][j], grid[i][j + 1], grid[i + 1][j + 1], grid[i + 1][j]);
            if i == 0 {
                // band between the pole row (all identical) and the first real ring
                m.add_tri(a, c_, d); // (pole, ring j+1, ring j)
            } else if i == rings as usize - 1 {
                m.add_tri(a, b, d); // (ring j, ring j+1, south pole)
            } else {
                m.add_tri(a, b, c_);
                m.add_tri(a, c_, d);
            }
        }
    }
    m.ensure_outward();
    (m, Kind::Sphere { r })
}

/// cylinder / frustum: bottom radius at y=0, top radius at y=h (taper for cups).
pub fn frustum(top: f64, bottom: f64, h: f64, smooth: u32) -> (Mesh, Kind) {
    let n = smooth.max(3);
    let mut m = Mesh::new();
    let mut bot: Vec<u32> = Vec::with_capacity(n as usize);
    let mut topv: Vec<u32> = Vec::with_capacity(n as usize);
    for j in 0..n {
        let th = 2.0 * std::f64::consts::PI * (j as f64) / (n as f64);
        let (s, c) = th.sin_cos();
        bot.push(m.add_vert(V3::new(bottom * c, 0.0, bottom * s)));
        topv.push(m.add_vert(V3::new(top * c, h, top * s)));
    }
    // side quads
    for j in 0..n as usize {
        let jn = (j + 1) % n as usize;
        m.add_quad(
            m.verts[bot[j] as usize], m.verts[bot[jn] as usize],
            m.verts[topv[jn] as usize], m.verts[topv[j] as usize],
        );
    }
    // caps (fan from first vertex)
    if bottom > 1e-9 {
        for j in 1..n as usize - 1 {
            m.add_tri(bot[0], bot[j + 1], bot[j]);
        }
    }
    if top > 1e-9 {
        for j in 1..n as usize - 1 {
            m.add_tri(topv[0], topv[j], topv[j + 1]);
        }
    }
    m.ensure_outward();
    (m, Kind::Frustum { top, bottom, h })
}

/// cone = frustum with top = 0
pub fn cone(r: f64, h: f64, smooth: u32) -> (Mesh, Kind) {
    frustum(0.0, r, h, smooth)
}

/// torus lying flat like a donut (main axis = y), resting on the ground.
pub fn torus(r_main: f64, tube: f64, smooth: u32) -> (Mesh, Kind) {
    let around = smooth.max(8); // around the main circle
    let rings = smooth.max(12); // around the tube (full res: chord deficit < 0.6%)
    let mut m = Mesh::new();
    let center_y = tube; // resting: bottom at y = 0
    let mut grid: Vec<Vec<u32>> = Vec::with_capacity(rings as usize);
    for i in 0..rings as usize {
        let phi = 2.0 * std::f64::consts::PI * (i as f64) / (rings as f64);
        let (sp, cp) = phi.sin_cos();
        let mut row = Vec::with_capacity(around as usize);
        for j in 0..around as usize {
            let th = 2.0 * std::f64::consts::PI * (j as f64) / (around as f64);
            let (st, ct) = th.sin_cos();
            let rr = r_main + tube * cp;
            let p = V3::new(rr * ct, center_y + tube * sp, rr * st);
            row.push(m.add_vert(p));
        }
        grid.push(row);
    }
    for i in 0..rings as usize {
        let in_ = (i + 1) % rings as usize;
        for j in 0..around as usize {
            let jn = (j + 1) % around as usize;
            let (a, b, c_, d) = (grid[i][j], grid[i][jn], grid[in_][jn], grid[in_][j]);
            m.add_quad(
                m.verts[a as usize], m.verts[b as usize],
                m.verts[c_ as usize], m.verts[d as usize],
            );
        }
    }
    m.ensure_outward();
    (m, Kind::Torus { r_main, tube })
}

/// pyramid: square base on the ground, apex up.
pub fn pyramid(base: f64, h: f64, sides: u32) -> (Mesh, Kind) {
    let n = sides.max(3);
    let mut m = Mesh::new();
    let mut ring: Vec<u32> = Vec::with_capacity(n as usize);
    for j in 0..n {
        let th = 2.0 * std::f64::consts::PI * (j as f64) / (n as f64);
        // start at angle offset so a flat face faces +z for square pyramids
        let t = th + std::f64::consts::PI / (n as f64);
        let (s, c) = t.sin_cos();
        // circumradius so the *edge* length = base
        let r = base / (2.0 * (std::f64::consts::PI / (n as f64)).sin());
        ring.push(m.add_vert(V3::new(r * c, 0.0, r * s)));
    }
    let apex = m.add_vert(V3::new(0.0, h, 0.0));
    for j in 0..n as usize {
        let jn = (j + 1) % n as usize;
        m.add_tri(ring[j], ring[jn], apex);
    }
    // base fan
    for j in 1..n as usize - 1 {
        m.add_tri(ring[0], ring[j + 1], ring[j]);
    }
    m.ensure_outward();
    (m, Kind::Prism { sides: n, r: base, h }) // approx kind; hollow uses Generic path
}

/// regular n-gon prism on the ground.
pub fn prism(sides: u32, r: f64, h: f64) -> (Mesh, Kind) {
    let n = sides.max(3);
    let (mut m, _) = frustum(r, r, h, n);
    m.ensure_outward();
    (m, Kind::Prism { sides: n, r, h })
}

/// capsule: cylinder + hemispherical caps, resting on ground. h = cylinder part.
pub fn capsule(r: f64, h: f64, smooth: u32) -> (Mesh, Kind) {
    let meridians = smooth.max(8);
    let half_rings = (smooth / 4).max(3); // per hemisphere
    let mut m = Mesh::new();
    // ring vertex helper at height y with radius rr
    let ring = |m: &mut Mesh, y: f64, rr: f64| -> Vec<u32> {
        (0..meridians)
            .map(|j| {
                let th = 2.0 * std::f64::consts::PI * (j as f64) / (meridians as f64);
                m.add_vert(V3::new(rr * th.cos(), y, rr * th.sin()))
            })
            .collect()
    };
    // bottom hemisphere rings (from pole y=0 up to equator y=r)
    let mut rows: Vec<Vec<u32>> = Vec::new();
    for i in 0..=half_rings as usize {
        let t = (i as f64) / (half_rings as f64); // 0..1
        let phi = t * std::f64::consts::PI / 2.0;
        rows.push(ring(&mut m, r - r * phi.cos(), r * phi.sin()));
    }
    // cylinder rows
    rows.push(ring(&mut m, r + h, r));
    // top hemisphere
    for i in 1..=half_rings as usize {
        let t = (i as f64) / (half_rings as f64);
        let phi = t * std::f64::consts::PI / 2.0;
        rows.push(ring(&mut m, r + h + r * phi.sin(), r * phi.cos()));
    }
    let bottom_pole = m.add_vert(V3::new(0.0, 0.0, 0.0));
    let top_pole = m.add_vert(V3::new(0.0, 2.0 * r + h, 0.0));
    // connect consecutive rows
    for ri in 0..rows.len() - 1 {
        for j in 0..meridians as usize {
            let jn = (j + 1) % meridians as usize;
            let (a, b, c_, d) = (rows[ri][j], rows[ri][jn], rows[ri + 1][jn], rows[ri + 1][j]);
            m.add_quad(
                m.verts[a as usize], m.verts[b as usize],
                m.verts[c_ as usize], m.verts[d as usize],
            );
        }
    }
    // caps
    let first = &rows[0];
    let last = rows.last().unwrap();
    for j in 0..meridians as usize {
        let jn = (j + 1) % meridians as usize;
        m.add_tri(bottom_pole, first[jn], first[j]);
        m.add_tri(top_pole, last[j], last[jn]);
    }
    m.ensure_outward();
    (m, Kind::Capsule { r, h })
}

/// wedge (ramp): rises from y=0 at x=0 to h at x=w, depth d along z.
pub fn wedge(w: f64, d: f64, h: f64) -> (Mesh, Kind) {
    let mut m = Mesh::new();
    let (x0, x1) = (-w / 2.0, w / 2.0);
    let (z0, z1) = (-d / 2.0, d / 2.0);
    let a = V3::new(x0, 0.0, z0);
    let b = V3::new(x1, 0.0, z0);
    let c = V3::new(x1, h, z0);
    let d2 = V3::new(x0, 0.0, z1);
    let e = V3::new(x1, 0.0, z1);
    let f = V3::new(x1, h, z1);
    // bottom
    m.add_quad(a, b, e, d2);
    // back (z0) triangle
    m.add_tri_pts(a, b, c);
    // front (z1) triangle
    m.add_tri_pts(d2, f, e);
    // slope (from a-d edge up to c-f edge)
    m.add_quad(a, c, f, d2);
    // right end triangle (x1)
    m.add_tri_pts(b, e, f);
    m.ensure_outward();
    (m, Kind::Generic)
}

/// plane: thin slab (watertight), w×d, thickness h.
pub fn plane(w: f64, d: f64, h: f64) -> (Mesh, Kind) {
    cube(w, d, h.max(0.5))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn vol(m: &Mesh) -> f64 { m.volume_signed().abs() }

    #[test]
    fn volumes_match_analytic() {
        // cube 2×3×4 cm → 24 cm³ = 24000 mm³
        let (m, _) = cube(20.0, 30.0, 40.0);
        assert!((vol(&m) - 24000.0).abs() / 24000.0 < 1e-9);

        // sphere r=10mm → 4/3π·1000 = 4188.79
        let (m, _) = sphere(10.0, 48);
        assert!((vol(&m) - 4188.790205).abs() / 4188.79 < 0.01, "{}", vol(&m));

        // cylinder r=10 h=20 → 6283.19 (48-gon cross-section: ~0.3% chord deficit)
        let (m, _) = frustum(10.0, 10.0, 20.0, 48);
        assert!((vol(&m) - 6283.185307).abs() / 6283.19 < 0.005, "{}", vol(&m));

        // frustum top 40 bottom 30 h 100 → π·h/3·(R²+Rr+r²) = 387,464 mm³
        let (m, _) = frustum(40.0, 30.0, 100.0, 48);
        assert!((vol(&m) - 387464.0969).abs() / 387464.1 < 0.005, "{}", vol(&m));

        // torus R=10 tube=3 → 2π²·R·r² = 1776.53
        let (m, _) = torus(10.0, 3.0, 48);
        assert!((vol(&m) - 1776.5288).abs() / 1776.53 < 0.01, "{}", vol(&m));

        // pyramid base 20 h 10 → (1/3)·400·10 = 1333.33
        let (m, _) = pyramid(20.0, 10.0, 4);
        assert!((vol(&m) - 1333.3333).abs() / 1333.33 < 0.02, "{}", vol(&m));

        // capsule r=5 h=10 → πr²h + 4/3πr³ = 785.4 + 523.6 = 1309.0
        let (m, _) = capsule(5.0, 10.0, 32);
        assert!((vol(&m) - 1308.997).abs() / 1309.0 < 0.02, "{}", vol(&m));

        // prism hex r=10 h=20 → (3√3/2)r²·h = 5196.15
        let (m, _) = prism(6, 10.0, 20.0);
        assert!((vol(&m) - 5196.1524).abs() / 5196.15 < 1e-3, "{}", vol(&m));
    }

    #[test]
    fn everything_rests_on_ground() {
        let (m, _) = sphere(10.0, 24);
        assert!(m.bbox().min.y() > -1e-9 && m.bbox().min.y() < 1e-9);
        let (m, _) = torus(10.0, 3.0, 24);
        assert!(m.bbox().min.y() > -1e-9 && m.bbox().min.y() < 1e-9);
        let (m, _) = capsule(5.0, 10.0, 24);
        assert!(m.bbox().min.y() > -1e-9 && m.bbox().min.y() < 1e-9);
        let (m, _) = frustum(4.0, 3.0, 10.0, 12);
        assert!(m.bbox().min.y() > -1e-9 && m.bbox().min.y() < 1e-9);
    }

    #[test]
    fn negative_size_is_rejected_upstream() {
        // (validation of friendly errors happens in eval tests; here: sizes > 0)
        let (m, _) = cube(1.0, 1.0, 1.0);
        assert!(m.volume_signed() > 0.0);
    }
}
