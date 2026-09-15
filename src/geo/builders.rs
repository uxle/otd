//! P0400–P0470 — advanced builders: extrude (ear-clip), revolve, swept tubes
//! (tube/helix via parallel-transport frames), 3D text (dot font), terrain
//! (fBm heightfield), metaballs (marching tetrahedra), loft, STL/OBJ import.

use crate::font::glyph;
use crate::geo::mesh::{Mesh};
use crate::math3::V3;
use crate::noise::fbm;

// ---------- extrude (P0400) ----------

/// Ear-clipping triangulation of a simple polygon (CCW in the XY plane).
fn ear_clip(pts: &[(f64, f64)]) -> Vec<(usize, usize, usize)> {
    let n = pts.len();
    let mut tris = Vec::new();
    if n < 3 {
        return tris;
    }
    // ensure CCW orientation
    let mut area = 0.0;
    for i in 0..n {
        let j = (i + 1) % n;
        area += pts[i].0 * pts[j].1 - pts[j].0 * pts[i].1;
    }
    let idx: Vec<usize> = if area < 0.0 {
        (0..n).rev().collect()
    } else {
        (0..n).collect()
    };
    let mut idx = idx;
    let cross2 = |o: usize, a: usize, b: usize| -> f64 {
        (pts[a].0 - pts[o].0) * (pts[b].1 - pts[o].1) - (pts[a].1 - pts[o].1) * (pts[b].0 - pts[o].0)
    };
    let contains = |a: usize, b: usize, c: usize, p: usize| -> bool {
        let d = cross2(a, b, c);
        if d.abs() < 1e-12 {
            return false;
        }
        let d1 = cross2(a, b, p);
        let d2 = cross2(b, c, p);
        let d3 = cross2(c, a, p);
        let neg = (d < 0.0) && (d1 <= 0.0 || d2 <= 0.0 || d3 <= 0.0);
        let pos = (d > 0.0) && (d1 >= 0.0 || d2 >= 0.0 || d3 >= 0.0);
        neg || pos
    };
    let mut guard = 0;
    while idx.len() > 3 && guard < n * n {
        guard += 1;
        let m = idx.len();
        let mut clipped = false;
        for i in 0..m {
            let a = idx[(i + m - 1) % m];
            let b = idx[i];
            let c = idx[(i + 1) % m];
            if cross2(a, b, c) > 1e-12 {
                let mut any_inside = false;
                for &p in idx.iter() {
                    if p != a && p != b && p != c && contains(a, b, c, p) {
                        any_inside = true;
                        break;
                    }
                }
                if !any_inside {
                    tris.push((a, b, c));
                    idx.remove(i);
                    clipped = true;
                    break;
                }
            }
        }
        if !clipped {
            // degenerate polygon: fan it and stop
            for i in 1..idx.len() - 1 {
                tris.push((idx[0], idx[i], idx[i + 1]));
            }
            break;
        }
    }
    if idx.len() == 3 {
        tris.push((idx[0], idx[1], idx[2]));
    }
    tris
}

/// Extrude a closed 2D polygon (points in "paper" XY plane facing +Z…
/// OTD convention: the profile is drawn in the XY ground plane and the solid
/// rises from y=0 to y=depth — so it reads naturally as a footprint).
pub fn extrude(profile: &[(f64, f64)], depth: f64) -> Mesh {
    let mut m = Mesh::new();
    if profile.len() < 3 || depth <= 0.0 {
        return m;
    }
    let tris = ear_clip(profile);
    // footprint in XZ plane: (x, z) from (x, y-paper) — map profile y → world z
    let map = |p: &(f64, f64)| V3::new(p.0, 0.0, p.1);
    // bottom (facing down) + top (facing up) with wall quads
    let mut bottom: Vec<u32> = Vec::new();
    let mut top: Vec<u32> = Vec::new();
    for p in profile {
        bottom.push(m.add_vert(map(p)));
    }
    for p in profile {
        top.push(m.add_vert(map(p).add(&V3::new(0.0, depth, 0.0))));
    }
    for (a, b, c) in &tris {
        // bottom: reversed winding (faces −y)
        m.add_tri(bottom[*a], bottom[*c], bottom[*b]);
        // top: faces +y
        m.add_tri(top[*a], top[*b], top[*c]);
    }
    let n = profile.len();
    for i in 0..n {
        let j = (i + 1) % n;
        m.add_quad(
            m.verts[bottom[i] as usize], m.verts[bottom[j] as usize],
            m.verts[top[j] as usize], m.verts[top[i] as usize],
        );
    }
    m.ensure_outward();
    m
}

// ---------- revolve (P0410) ----------

/// Revolve a (radius, height) profile around the Y axis. Auto-closed to the
/// axis so the result is watertight. `angle` < 360 gives cut-away views
/// (open wedge — volume still measured over the swept angle).
pub fn revolve(profile: &[(f64, f64)], angle: f64, smooth: u32) -> (Mesh, crate::geo::mesh::Kind) {
    let mut m = Mesh::new();
    let n = smooth.max(4);
    let full = (360.0 - angle.abs()).abs() < 1e-9;
    let steps = if full { n } else { (n as f64 * angle.abs() / 360.0).max(2.0) as u32 };
    // profile rings
    let mut rings: Vec<Vec<u32>> = Vec::new();
    for (r, y) in profile {
        let r = r.max(0.0);
        let mut ring = Vec::new();
        if full {
            for j in 0..n {
                let th = 2.0 * std::f64::consts::PI * (j as f64) / (n as f64);
                ring.push(m.add_vert(V3::new(r * th.cos(), *y, r * th.sin())));
            }
        } else {
            for j in 0..=steps {
                let th = angle.to_radians() * (j as f64) / (steps as f64);
                ring.push(m.add_vert(V3::new(r * th.cos(), *y, r * th.sin())));
            }
        }
        rings.push(ring);
    }
    let ring_len = rings[0].len();
    let wrap = if full { ring_len } else { ring_len - 1 }; // full rings close on themselves
    for ri in 0..rings.len() - 1 {
        for j in 0..wrap {
            let jn = (j + 1) % ring_len;
            let (a, b, c, d) = (rings[ri][j], rings[ri][jn], rings[ri + 1][jn], rings[ri + 1][j]);
            m.add_quad(
                m.verts[a as usize], m.verts[b as usize],
                m.verts[c as usize], m.verts[d as usize],
            );
        }
    }
    // caps: close profile start/end to the axis
    let close_cap = |m: &mut Mesh, ring: &[u32]| {
        if ring.is_empty() {
            return;
        }
        let p0 = m.verts[ring[0] as usize];
        let p1 = m.verts[ring[ring_len - 1] as usize];
        let start_on_axis = p0.x().abs() < 1e-9 && p0.z().abs() < 1e-9;
        let end_on_axis = p1.x().abs() < 1e-9 && p1.z().abs() < 1e-9;
        if start_on_axis && end_on_axis {
            return; // profile closes itself through the axis
        }
        let y = p0.y();
        let axis_pt = m.add_vert(V3::new(0.0, y, 0.0));
        let n = ring.len();
        for j in 0..n {
            let jn = (j + 1) % n;
            // winding: fan triangles must face outward (−y for a bottom disc,
            // +y for a top disc) — ensure_outward fixes the global sign, and
            // since the mesh is otherwise consistent, per-cap consistency with
            // the sides comes from the shared ring order.
            m.add_tri(axis_pt, ring[j], ring[jn]);
        }
    };
    if full {
        close_cap(&mut m, &rings[0]);
        close_cap(&mut m, rings.last().unwrap());
    }
    m.ensure_outward();
    (m, crate::geo::mesh::Kind::Revolve { profile: profile.to_vec(), angle })
}

// ---------- swept tubes: tube & helix (P0420) ----------

/// Sweep a circular profile along a polyline path using parallel-transport
/// frames (no twist seams). Watertight, capped with fans.
pub fn sweep_path(path: &[V3], r: f64, smooth: u32) -> Mesh {
    let mut m = Mesh::new();
    if path.len() < 2 || r <= 0.0 {
        return m;
    }
    let n = smooth.max(6).min(48);
    // tangents
    let mut tans: Vec<V3> = Vec::with_capacity(path.len());
    for i in 0..path.len() {
        if i == 0 {
            tans.push(path[1].sub(&path[0]).norm());
        } else if i == path.len() - 1 {
            tans.push(path[i].sub(&path[i - 1]).norm());
        } else {
            tans.push(path[i + 1].sub(&path[i - 1]).norm());
        }
    }
    // parallel transport: initial normal from an arbitrary perpendicular
    let t0 = tans[0];
    let mut nrm = if t0.y().abs() < 0.9 {
        V3::new(0.0, 1.0, 0.0).cross(&t0).norm()
    } else {
        V3::new(1.0, 0.0, 0.0).cross(&t0).norm()
    };
    let mut rings: Vec<Vec<u32>> = Vec::with_capacity(path.len());
    for (i, p) in path.iter().enumerate() {
        if i > 0 {
            // transport: project previous normal onto plane ⊥ tangent
            let t = tans[i];
            let d = nrm.dot(&t);
            let proj = nrm.sub(&t.mul(d)).norm();
            nrm = if proj.len() > 1e-9 { proj } else { nrm };
        }
        let bin = tans[i].cross(&nrm).norm();
        let mut ring = Vec::with_capacity(n as usize);
        for j in 0..n {
            let th = 2.0 * std::f64::consts::PI * (j as f64) / (n as f64);
            let dir = nrm.mul(th.cos()).add(&bin.mul(th.sin()));
            ring.push(m.add_vert(p.add(&dir.mul(r))));
        }
        rings.push(ring);
    }
    for ri in 0..rings.len() - 1 {
        for j in 0..n as usize {
            let jn = (j + 1) % n as usize;
            let (a, b, c, d) = (rings[ri][j], rings[ri][jn], rings[ri + 1][jn], rings[ri + 1][j]);
            m.add_quad(
                m.verts[a as usize], m.verts[b as usize],
                m.verts[c as usize], m.verts[d as usize],
            );
        }
    }
    // end caps (fans)
    let cap = |m: &mut Mesh, ring: &[u32], center: V3, flip: bool| {
        let ci = m.add_vert(center);
        for j in 0..n as usize {
            let jn = (j + 1) % n as usize;
            if flip {
                m.add_tri(ci, ring[j], ring[jn]);
            } else {
                m.add_tri(ci, ring[jn], ring[j]);
            }
        }
    };
    cap(&mut m, &rings[0], path[0], true);
    cap(&mut m, rings.last().unwrap(), *path.last().unwrap(), true);
    m.ensure_outward();
    m
}

/// tube: pipe of radius r through given 3D points.
pub fn tube(path: &[V3], r: f64, smooth: u32) -> Mesh {
    sweep_path(path, r, smooth)
}

/// helix: spring. radius, pitch (height per turn), turns, tube radius.
pub fn helix(radius: f64, pitch: f64, turns: f64, tube_r: f64, smooth: u32) -> Mesh {
    let steps = (turns * 24.0).max(24.0) as usize;
    let mut path: Vec<V3> = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let t = i as f64 / steps as f64;
        let th = turns * 2.0 * std::f64::consts::PI * t;
        path.push(V3::new(radius * th.cos(), pitch * turns * t, radius * th.sin()));
    }
    sweep_path(&path, tube_r, smooth)
}

/// OTD3 P2150 — MACHINE THREAD: the spiral ridge that makes a bolt a bolt.
///
/// ISO-metric-style V-thread as a helical fin: an outer crest ring at
/// `r_crest = radius + depth` and an inner root ring at `radius`, both
/// advancing `pitch` mm per turn, stitched by triangles (double-sided, so
/// the ridge reads from every angle). The geometry is the real law:
/// one turn ⇔ one pitch of travel — exactly the coupling the screw
/// animation (`--screw`) replays.
///
/// * `radius` — root (major-body) radius, mm
/// * `pitch`  — axial advance per full turn, mm (M6 standard: 1.0)
/// * `turns`  — how many thread turns to wind
/// * `depth`  — radial ridge height (ISO V-thread ≈ 0.61 × pitch)
pub fn thread(radius: f64, pitch: f64, turns: f64, depth: f64) -> Mesh {
    let mut m = Mesh::new();
    if radius <= 0.0 || pitch <= 0.0 || turns <= 0.0 || depth <= 0.0 {
        return m;
    }
    // enough angular steps so the ridge is smooth: ~48 per turn
    let steps = (turns * 48.0).max(48.0) as usize;
    let total = turns * 2.0 * std::f64::consts::PI;
    let r_crest = radius + depth;
    // two vertices per step: root and crest (same angle, same height —
    // the "fin" thread; wrapped at pitch it reads as a spiral ridge)
    let mut roots: Vec<u32> = Vec::with_capacity(steps + 1);
    let mut crests: Vec<u32> = Vec::with_capacity(steps + 1);
    for i in 0..=steps {
        let th = total * (i as f64 / steps as f64);
        let y = pitch * turns * (i as f64 / steps as f64);
        let (s, c) = th.sin_cos();
        let root = m.add_vert(V3::new(radius * c, y, radius * s));
        let crest = m.add_vert(V3::new(r_crest * c, y, r_crest * s));
        roots.push(root);
        crests.push(crest);
    }
    for i in 0..steps {
        let (r0, r1) = (roots[i], roots[i + 1]);
        let (c0, c1) = (crests[i], crests[i + 1]);
        // ridge wall, both sides (double-sided so it never vanishes)
        m.add_tri(c0, c1, r1);
        m.add_tri(c0, r1, r0);
        m.add_tri(r0, r1, c1); // mirrored winding — the other face
        m.add_tri(r0, c1, c0);
    }
    m
}

// ---------- text (P0430) ----------

/// 3D text in the 5×7 dot font. Each lit dot = a box of `dot` size, `depth`
/// tall (along y), reading along +x, resting on the ground.
pub fn text3d(s: &str, size: f64, depth: f64) -> Mesh {
    let dot = (size / 7.0).max(0.1);
    let cell = dot; // one dot per cell
    let mut m = Mesh::new();
    let chars: Vec<char> = s.chars().collect();
    let width = chars.len() as f64 * 6.0 * cell;
    let mut x = -width / 2.0;
    for c in &chars {
        if let Some(rows) = glyph(*c) {
            for (row, bits) in rows.iter().enumerate() {
                for (col, ch) in bits.chars().enumerate() {
                    if ch == '1' {
                        let cx = x + col as f64 * cell;
                        let cz = row as f64 * cell;
                        let (mut boxm, _) = crate::geo::prims::cube(cell, cell, depth.max(dot));
                        boxm.transform(&crate::math3::M4::translate(cx, 0.0, cz - 3.5 * cell));
                        m.merge(&boxm);
                    }
                }
            }
        }
        x += 6.0 * cell;
    }
    m
}

// ---------- terrain (P0440) ----------

/// Rolling landscape: solid block with a fBm heightfield top, from −size/2 to
/// +size/2 in XZ, base at y=0, peak up to `height`.
pub fn terrain(size: f64, height: f64, seed: u64, res: u32) -> Mesh {
    let res = (res as usize).clamp(8, 96);
    let mut m = Mesh::new();
    let h_at = |i: usize, j: usize| -> f64 {
        let x = i as f64 / res as f64 * 4.0; // noise-space
        let z = j as f64 / res as f64 * 4.0;
        let n = fbm(x, z, seed, 4);
        height * (0.5 + 0.5 * n)
    };
    let pos = |i: usize, j: usize| -> V3 {
        let x = -size / 2.0 + size * (i as f64 / res as f64);
        let z = -size / 2.0 + size * (j as f64 / res as f64);
        V3::new(x, h_at(i, j), z)
    };
    let mut grid: Vec<Vec<u32>> = Vec::with_capacity(res + 1);
    for j in 0..=res {
        let mut row = Vec::with_capacity(res + 1);
        for i in 0..=res {
            row.push(m.add_vert(pos(i, j)));
        }
        grid.push(row);
    }
    // top surface
    for j in 0..res {
        for i in 0..res {
            let (a, b, c, d) = (grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]);
            m.add_quad(
                m.verts[a as usize], m.verts[b as usize],
                m.verts[c as usize], m.verts[d as usize],
            );
        }
    }
    // base (flat at y=0) + walls
    let (_x0, _x1) = (-size / 2.0, size / 2.0);
    let (_z0, _z1) = (-size / 2.0, size / 2.0);
    let yb = 0.0;
    // build boundary walk explicitly (top surface boundary, CCW seen from above)
    let mut boundary: Vec<u32> = Vec::new();
    for i in 0..=res { boundary.push(grid[0][i]); }                       // z = z0 edge
    for j in 1..=res { boundary.push(grid[j][res]); }                     // x = x1 edge
    for i in (0..res).rev() { boundary.push(grid[res][i]); }              // z = z1 edge
    for j in (1..res).rev() { boundary.push(grid[j][0]); }                // x = x0 edge
    // base ring must have the same order/count; rebuild base ring from boundary
    let mut base2: Vec<u32> = Vec::with_capacity(boundary.len());
    for v in &boundary {
        let p = m.verts[*v as usize];
        base2.push(m.add_vert(V3::new(p.x(), yb, p.z())));
    }
    // base polygon fan (winding facing down)
    for i in 1..base2.len() - 1 {
        m.add_tri(base2[0], base2[i], base2[i + 1]);
    }
    // walls
    let bl = boundary.len();
    for i in 0..bl {
        let j = (i + 1) % bl;
        let (ta, tb) = (boundary[i], boundary[j]);
        let (ba, bb) = (base2[i], base2[j]);
        m.add_quad(
            m.verts[ba as usize], m.verts[bb as usize],
            m.verts[tb as usize], m.verts[ta as usize],
        );
    }
    m.ensure_outward();
    m
}

// ---------- metaballs (P0450) ----------

/// Marching tetrahedra isosurface over a sum of metaball fields.
pub fn metaballs(balls: &[V3], radii: &[f64], grid: u32) -> Mesh {
    if balls.is_empty() {
        return Mesh::new();
    }
    // bounds with margin — grown until the iso-surface provably closes
    // inside the box (every corner strictly outside the f = 1 blob)
    let mut bb = crate::math3::Aabb::empty();
    for (b, r) in balls.iter().zip(radii.iter()) {
        bb.grow_box(&crate::math3::Aabb::from_center_extent(b, &V3::new(*r, *r, *r)));
    }
    let f_raw = |p: V3| -> f64 {
        let mut f = 0.0;
        for (b, r) in balls.iter().zip(radii.iter()) {
            let d2 = p.sub(b).len();
            f += (r * r) / (d2 * d2 + 1e-9);
        }
        f
    };
    let mut margin = bb.size().mul(0.15);
    for _ in 0..12 {
        let mut probe = bb.clone();
        probe.min = probe.min.sub(&margin);
        probe.max = probe.max.add(&margin);
        let corners_out = [probe.min, probe.max,
            V3::new(probe.min.x(), probe.min.y(), probe.max.z()),
            V3::new(probe.min.x(), probe.max.y(), probe.min.z()),
            V3::new(probe.max.x(), probe.min.y(), probe.min.z()),
            V3::new(probe.min.x(), probe.max.y(), probe.max.z()),
            V3::new(probe.max.x(), probe.min.y(), probe.max.z()),
            V3::new(probe.max.x(), probe.max.y(), probe.min.z())];
        let open = corners_out.iter().all(|c| f_raw(*c) < 0.95);
        if open {
            bb = probe;
            break;
        }
        margin = margin.mul(1.6);
    }
    // metaball field: iso-value 1.0 — march_tets wants iso 0, so shift:
    // field > 0 = inside. (f - 1.0) flips the sign convention exactly.
    let field = |p: V3| -> f64 { f_raw(p) - 1.0 };
    // P1050: shared, manifold-correct marching tetrahedra (sdf::march_tets)
    crate::geo::sdf::march_tets(&bb, grid, &field)
}

pub fn loft(sections: &[Vec<(f64, f64)>], spacing: f64) -> Mesh {
    let mut m = Mesh::new();
    if sections.len() < 2 {
        return m;
    }
    let count = sections[0].len();
    if count < 3 || sections.iter().any(|s| s.len() != count) {
        return m;
    }
    let mut rings: Vec<Vec<u32>> = Vec::new();
    for (si, sec) in sections.iter().enumerate() {
        let y = si as f64 * spacing;
        let ring: Vec<u32> = sec.iter().map(|p| m.add_vert(V3::new(p.0, y, p.1))).collect();
        rings.push(ring);
    }
    for ri in 0..rings.len() - 1 {
        for j in 0..count {
            let jn = (j + 1) % count;
            let (a, b, c, d) = (rings[ri][j], rings[ri][jn], rings[ri + 1][jn], rings[ri + 1][j]);
            m.add_quad(
                m.verts[a as usize], m.verts[b as usize],
                m.verts[c as usize], m.verts[d as usize],
            );
        }
    }
    // caps
    let cap = |m: &mut Mesh, ring: &[u32], up: bool| {
        // project centroid
        let mut c = V3::ZERO;
        for v in ring {
            c = c.add(&m.verts[*v as usize]);
        }
        c = c.mul(1.0 / ring.len() as f64);
        let ci = m.add_vert(c);
        for j in 0..count {
            let jn = (j + 1) % count;
            if up {
                m.add_tri(ci, ring[j], ring[jn]);
            } else {
                m.add_tri(ci, ring[jn], ring[j]);
            }
        }
    };
    cap(&mut m, &rings[0], false);
    cap(&mut m, rings.last().unwrap(), true);
    m.ensure_outward();
    m
}

// ---------- sweep along a path with a polygon profile (P0460) ----------

/// Sweep a 2D profile along a 3D path (profile points are (offset-x, offset-y)
/// in the ring plane).
pub fn sweep(profile: &[(f64, f64)], path: &[V3], twist_deg: f64) -> Mesh {
    let mut m = Mesh::new();
    if path.len() < 2 || profile.len() < 3 {
        return m;
    }
    let tans: Vec<V3> = path
        .iter()
        .enumerate()
        .map(|(i, _)| {
            if i == 0 {
                path[1].sub(&path[0]).norm()
            } else if i == path.len() - 1 {
                path[i].sub(&path[i - 1]).norm()
            } else {
                path[i + 1].sub(&path[i - 1]).norm()
            }
        })
        .collect();
    let t0 = tans[0];
    let mut nrm = if t0.y().abs() < 0.9 {
        V3::new(0.0, 1.0, 0.0).cross(&t0).norm()
    } else {
        V3::new(1.0, 0.0, 0.0).cross(&t0).norm()
    };
    let mut rings: Vec<Vec<u32>> = Vec::with_capacity(path.len());
    for (i, p) in path.iter().enumerate() {
        if i > 0 {
            let t = tans[i];
            let d = nrm.dot(&t);
            let proj = nrm.sub(&t.mul(d)).norm();
            nrm = if proj.len() > 1e-9 { proj } else { nrm };
        }
        let tw = twist_deg.to_radians() * (i as f64 / (path.len() - 1) as f64);
        let bin0 = tans[i].cross(&nrm).norm();
        let nrm2 = nrm.mul(tw.cos()).add(&bin0.mul(tw.sin()));
        let bin = tans[i].cross(&nrm2).norm();
        let ring: Vec<u32> = profile
            .iter()
            .map(|q| m.add_vert(p.add(&nrm2.mul(q.0)).add(&bin.mul(q.1))))
            .collect();
        rings.push(ring);
    }
    let count = profile.len();
    for ri in 0..rings.len() - 1 {
        for j in 0..count {
            let jn = (j + 1) % count;
            let (a, b, c, d) = (rings[ri][j], rings[ri][jn], rings[ri + 1][jn], rings[ri + 1][j]);
            m.add_quad(
                m.verts[a as usize], m.verts[b as usize],
                m.verts[c as usize], m.verts[d as usize],
            );
        }
    }
    // caps (fan around centroid)
    let cap = |m: &mut Mesh, ring: &[u32]| {
        let mut c = V3::ZERO;
        for v in ring {
            c = c.add(&m.verts[*v as usize]);
        }
        c = c.mul(1.0 / ring.len() as f64);
        let ci = m.add_vert(c);
        for j in 0..count {
            let jn = (j + 1) % count;
            m.add_tri(ci, ring[jn], ring[j]);
        }
    };
    cap(&mut m, &rings[0]);
    cap(&mut m, rings.last().unwrap());
    m.ensure_outward();
    m
}

// ---------- import (P0470) ----------

/// Import a binary or ASCII STL (millimeters assumed).
pub fn import_stl(bytes: &[u8]) -> Result<Mesh, String> {
    // binary STL: 80-byte header + count + 50-byte triangles
    if bytes.len() > 84 {
        let count = u32::from_le_bytes([bytes[80], bytes[81], bytes[82], bytes[83]]) as usize;
        if bytes.len() == 84 + count * 50 {
            let mut m = Mesh::new();
            for t in 0..count {
                let off = 84 + t * 50 + 12; // skip normal
                let mut vs = [V3::ZERO; 3];
                for (vi, v) in vs.iter_mut().enumerate() {
                    let b = off + vi * 12;
                    let fx = f32::from_le_bytes([bytes[b], bytes[b + 1], bytes[b + 2], bytes[b + 3]]);
                    let fy = f32::from_le_bytes([bytes[b + 4], bytes[b + 5], bytes[b + 6], bytes[b + 7]]);
                    let fz = f32::from_le_bytes([bytes[b + 8], bytes[b + 9], bytes[b + 10], bytes[b + 11]]);
                    *v = V3::new(fx as f64, fy as f64, fz as f64);
                }
                m.add_tri_pts(vs[0], vs[1], vs[2]);
            }
            return Ok(m);
        }
    }
    // ASCII STL
    let s = String::from_utf8(bytes.to_vec()).map_err(|_| "STL file is not valid UTF-8 text".to_string())?;
    let mut m = Mesh::new();
    let mut cur: Vec<V3> = Vec::new();
    for line in s.lines() {
        let line = line.trim();
        if line.starts_with("vertex") {
            let parts: Vec<f64> = line
                .split_whitespace()
                .skip(1)
                .filter_map(|w| w.parse().ok())
                .collect();
            if parts.len() == 3 {
                cur.push(V3::new(parts[0], parts[1], parts[2]));
            }
        }
        if line.starts_with("endloop") {
            if cur.len() == 3 {
                m.add_tri_pts(cur[0], cur[1], cur[2]);
            }
            cur.clear();
        }
    }
    if m.is_empty() {
        return Err("no triangles found in the STL".into());
    }
    Ok(m)
}

/// Import an OBJ (v / f lines, triangulated).
pub fn import_obj(text: &str) -> Result<Mesh, String> {
    let mut m = Mesh::new();
    let mut verts: Vec<V3> = Vec::new();
    for line in text.lines() {
        let line = line.trim();
        if line.starts_with("v ") {
            let p: Vec<f64> = line.split_whitespace().skip(1).filter_map(|w| w.parse().ok()).collect();
            if p.len() >= 3 {
                verts.push(V3::new(p[0], p[1], p[2]));
            }
        } else if line.starts_with("f ") {
            let idx: Vec<Vec<i64>> = line
                .split_whitespace()
                .skip(1)
                .map(|w| w.split('/').filter_map(|x| x.parse::<i64>().ok()).collect())
                .collect();
            let resolve = |i: i64| -> Option<usize> {
                let n = verts.len() as i64;
                let r = if i < 0 { n + i } else { i - 1 };
                if r >= 0 && r < n { Some(r as usize) } else { None }
            };
            let flat: Vec<usize> = idx.iter().filter_map(|f| resolve(*f.first()?)).collect();
            for t in 1..flat.len().saturating_sub(1) {
                let (a, b, c) = (flat[0], flat[t], flat[t + 1]);
                m.add_tri_pts(verts[a], verts[b], verts[c]);
            }
        }
    }
    if m.is_empty() {
        return Err("no faces found in the OBJ".into());
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extrude_square_volume() {
        // 10×10mm square, depth 5 → 500 mm³
        let m = extrude(&[(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)], 5.0);
        let v = m.volume_signed().abs();
        assert!((v - 500.0).abs() / 500.0 < 0.01, "volume {}", v);
    }

    #[test]
    fn extrude_star_volume() {
        // simple 5-point-ish star (non-convex) — ear clip must not crash
        let star = [
            (0.0, 30.0), (10.0, 10.0), (30.0, 10.0), (12.0, -10.0),
            (20.0, -30.0), (0.0, -15.0), (-20.0, -30.0), (-12.0, -10.0),
            (-30.0, 10.0), (-10.0, 10.0),
        ];
        let m = extrude(&star, 4.0);
        assert!(!m.is_empty());
        assert!(m.volume_signed() > 0.0);
    }

    #[test]
    fn revolve_cylinder_volume() {
        // profile (10, 0) → (10, 20): a cylinder r=10 h=20 → 6283 mm³
        let (m, _) = revolve(&[(10.0, 0.0), (10.0, 20.0)], 360.0, 48);
        let v = m.volume_signed().abs();
        assert!((v - 6283.185).abs() / 6283.19 < 0.01, "volume {}", v);
    }

    #[test]
    fn revolve_cone_volume() {
        // (0,0) → (10, 0) → (0, 20): cone r10 h20 → 1/3·π·100·20 = 2094.4
        let (m, _) = revolve(&[(0.0, 0.0), (10.0, 0.0), (0.0, 20.0)], 360.0, 48);
        let v = m.volume_signed().abs();
        assert!((v - 2094.4).abs() / 2094.4 < 0.02, "volume {}", v);
    }

    #[test]
    fn tube_volume() {
        // straight tube along y: r=5, length 20 → π·25·20 = 1570.8
        let m = tube(&[V3::new(0.0, 0.0, 0.0), V3::new(0.0, 20.0, 0.0)], 5.0, 24);
        let v = m.volume_signed().abs();
        assert!((v - 1570.8).abs() / 1570.8 < 0.03, "volume {}", v);
    }

    #[test]
    fn helix_is_watertight() {
        let m = helix(30.0, 10.0, 5.0, 5.0, 24);
        let v = m.volume_signed();
        // spring volume ≈ tube volume along the path: path length ≈ turns·2π·R
        let path_len = 5.0 * 2.0 * std::f64::consts::PI * 30.0;
        let expect = std::f64::consts::PI * 25.0 * path_len;
        assert!((v.abs() - expect).abs() / expect < 0.1, "helix {} vs {}", v.abs(), expect);
    }

    #[test]
    fn text_boxes() {
        let m = text3d("A", 14.0, 5.0);
        assert!(!m.is_empty());
        assert!(m.volume_signed() > 0.0);
    }

    #[test]
    fn terrain_solid() {
        let m = terrain(100.0, 20.0, 7, 24);
        let v = m.volume_signed().abs();
        // volume between 0 and height·area — fBm mean ≈ half
        assert!(v > 100.0 * 100.0 * 5.0 && v < 100.0 * 100.0 * 20.0, "terrain volume {}", v);
    }

    #[test]
    fn metaballs_merge() {
        // two unit balls close together → blobby solid with positive volume
        let m = metaballs(
            &[V3::new(-8.0, 10.0, 0.0), V3::new(8.0, 10.0, 0.0)],
            &[10.0, 10.0],
            24,
        );
        let v = m.volume_signed().abs();
        // one ball at iso 1 is exactly a sphere of radius r; two merged blobs
        // (16 apart, r=10 each) ≈ 2×4188 + merge bulge → 8k..16k mm³
        assert!(v > 8000.0 && v < 16000.0, "metaball volume {}", v);
    }

    #[test]
    fn loft_box() {
        // two 10×10 sections 10 apart → box-ish 1000 mm³
        let sec = vec![(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)];
        let m = loft(&[sec.clone(), sec], 10.0);
        let v = m.volume_signed().abs();
        assert!((v - 1000.0).abs() / 1000.0 < 0.01, "volume {}", v);
    }

    #[test]
    fn stl_roundtrip() {
        // build a cube, export to STL bytes, import back, volume must match
        let (c, _) = crate::geo::prims::cube(20.0, 20.0, 20.0);
        let bytes = crate::export::stl::to_binary_stl(&c, "test");
        let back = import_stl(&bytes).unwrap();
        let v = back.volume_signed().abs();
        assert!((v - 8000.0).abs() / 8000.0 < 0.001, "roundtrip volume {}", v);
    }
}
