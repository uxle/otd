//! Census/SGM stereo matching on a half-pixel cost grid (the AVC v2 core,
//! re-engineered as the v4 engine).
//!
//! v2's benchmark wins over OpenCV SGBM came from four ideas, all kept:
//! 1. **Two-scale census** (fine 5x5 + radius-4 ring) - texture-scale robustness.
//! 2. **Half-pixel cost grid**: 2N slices; odd slices hold the average of the
//!    neighbouring integer costs. WTA directly on the grid gives 0.5 px
//!    quantisation without biased parabola fits.
//! 3. **Uniqueness gate with SGBM semantics**: ambiguous matches are rejected
//!    (later filled + flagged), not silently kept.
//! 4. **Median pre/post filters** for sensor-noise robustness.
//!
//! v4 re-engineering (low specs, high output):
//! 5. **Pyramid mode** (`SgmParams::pyramid`): the dense search runs at half
//!    resolution; full resolution only refines +-3 px around the coarse
//!    disparity with the full-res two-scale census. ~4x cheaper search,
//!    ~6x less peak memory.
//! 6. **Exact two-smallest SGM penalties**: the `min_{d'!=d} L + P2` term is
//!    evaluated with the min1/min2 trick - faster (no second scan per pixel)
//!    AND closer to exact SGM than the v3 `min + P2` approximation.
//! 7. **Voting LR check**: the right-view map is derived from the left
//!    matches by disparity voting instead of a full second cost+SGM pass
//!    (~45% of v3 stereo runtime removed).
//! 8. **`StereoBuffers`**: all scratch is caller-owned; steady-state stereo
//!    performs no large allocations.

use image::GrayImage;
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::stereo::census::{census_into, hamming_fine, hamming_ring, Census};
use crate::stereo::map::{DisparityMap, Quality, SgmParams};
use crate::stereo::post;

/// Border (px) excluded from matching (ring census radius).
pub const MATCH_BORDER: usize = 4;
/// Cost assigned to out-of-range / border samples.
const BORDER_COST: u16 = 200;
/// Refinement search radius (px) around the coarse pyramid disparity.
const REFINE_RADIUS: i64 = 3;

#[derive(Default)]
pub struct CostVolume {
    pub w: usize,
    pub h: usize,
    pub slices: usize,
    pub disp_min: usize,
    pub data: Vec<u8>,
}

/// Caller-owned matcher scratch (v4). Keep one per engine / benchmark loop;
/// every buffer is resized-in-place so the steady state allocates ~nothing
/// except the output map.
#[derive(Default)]
pub struct StereoBuffers {
    /// median-filtered rectified pair (full res)
    pub li: GrayImage,
    pub ri: GrayImage,
    /// half-res pair (pyramid mode)
    pub half_l: GrayImage,
    pub half_r: GrayImage,
    /// full-res census (refinement + full-res mode)
    pub c_l: Census,
    pub c_r: Census,
    /// half-res census (pyramid dense search)
    pub c_lh: Census,
    pub c_rh: Census,
    /// cost volume for whichever level is active
    pub vol: CostVolume,
    /// aggregation sums for the active level
    pub sums: Vec<u16>,
    /// half-res WTA output
    pub disp_h: Vec<f32>,
    pub qual_h: Vec<u8>,
    /// full-res output being assembled
    pub disp: Vec<f32>,
    pub qual: Vec<u8>,
    /// v4: coarse-fallback mask for the full-res output
    pub coarse: Vec<bool>,
    /// LR voting scratch
    pub vsum: Vec<f32>,
    pub vcnt: Vec<u32>,
}

impl StereoBuffers {
    pub fn new() -> Self {
        Self::default()
    }
}

/// Copy `src` into `dst` without reallocation when the dimensions match.
fn clone_gray_into(src: &GrayImage, dst: &mut GrayImage) {
    let (w, h) = src.dimensions();
    if dst.dimensions() != (w, h) {
        *dst = GrayImage::new(w, h);
    }
    let mut buf = std::mem::replace(dst, GrayImage::new(1, 1)).into_raw();
    buf.clear();
    buf.extend_from_slice(src.as_raw());
    *dst = GrayImage::from_raw(w, h, buf).unwrap();
}

/// 2x2 box downsample (edge-replicated) into a caller-owned buffer.
fn downsample2x(src: &GrayImage, out: &mut GrayImage) {
    let (w, h) = src.dimensions();
    let (w, h) = (w as usize, h as usize);
    let hw = (w + 1) / 2;
    let hh = (h + 1) / 2;
    if out.dimensions() != (hw as u32, hh as u32) {
        *out = GrayImage::new(hw as u32, hh as u32);
    }
    let s = src.as_raw();
    let mut d = std::mem::replace(out, GrayImage::new(1, 1)).into_raw();
    d.clear();
    d.resize(hw * hh, 0);
    for y in 0..hh {
        let y0 = 2 * y;
        let y1 = (y0 + 1).min(h - 1);
        for x in 0..hw {
            let x0 = 2 * x;
            let x1 = (x0 + 1).min(w - 1);
            let v = (s[y0 * w + x0] as u32
                + s[y0 * w + x1] as u32
                + s[y1 * w + x0] as u32
                + s[y1 * w + x1] as u32
                + 2)
                / 4;
            d[y * hw + x] = v as u8;
        }
    }
    *out = GrayImage::from_raw(hw as u32, hh as u32, d).unwrap();
}

/// Build the cost volume (left reference; right sampled at `x - d`).
fn build_cost_into(
    c_ref: &Census,
    c_match: &Census,
    disp_min: usize,
    num_disp: usize,
    vol: &mut CostVolume,
) {
    let w = c_ref.w;
    let h = c_ref.h;
    let slices = 2 * num_disp;
    let n_base = num_disp + 1; // integer costs disp_min .. disp_min+num_disp
    vol.w = w;
    vol.h = h;
    vol.slices = slices;
    vol.disp_min = disp_min;
    let need = w * h * slices;
    if vol.data.len() != need {
        vol.data.clear();
        vol.data.resize(need, 0);
    }

    vol.data
        .par_chunks_mut(w * slices)
        .enumerate()
        .for_each(|(y, row)| {
            for x in 0..w {
                // base integer costs for this pixel
                let mut base = [0u16; 129];
                let nb = n_base.min(129);
                for k in 0..nb {
                    let d = disp_min + k;
                    let xs = x as i64 - d as i64;
                    if xs < MATCH_BORDER as i64 || xs > (w - 1 - MATCH_BORDER) as i64 {
                        base[k] = BORDER_COST;
                    } else {
                        let i_ref = y * w + x;
                        let i_mat = y * w + xs as usize;
                        base[k] = hamming_fine(c_ref.fine[i_ref], c_match.fine[i_mat])
                            + hamming_ring(c_ref.ring[i_ref], c_match.ring[i_mat]);
                    }
                }
                let row_off = x * slices;
                for s in 0..slices {
                    let v = if s % 2 == 0 {
                        base[s / 2]
                    } else {
                        let lo = base[s / 2];
                        let hi = base[s / 2 + 1];
                        (lo + hi + 1) / 2
                    };
                    row[row_off + s] = v.min(255) as u8;
                }
            }
        });
}

fn dirs_4() -> [(i32, i32); 4] {
    [(0, 1), (0, -1), (1, 0), (-1, 0)]
}

fn dirs_8() -> [(i32, i32); 8] {
    [
        (0, 1), (0, -1), (1, 0), (-1, 0),
        (1, 1), (1, -1), (-1, 1), (-1, -1),
    ]
}

/// SGM aggregation into `sums` (u16 per pixel & slice). `ref_img` is the
/// reference (left) image for the adaptive P2. v4: the far-jump penalty uses
/// the exact two-smallest (min1/min2) of the previous line, tracked while
/// that line is written - no separate min scan, and closer to exact SGM than
/// v3's `min + P2` approximation.
fn aggregate(cost: &CostVolume, ref_img: &[u8], params: &SgmParams, sums: &mut [u16]) {
    let w = cost.w;
    let h = cost.h;
    let s_n = cost.slices;
    let p1 = params.p1;
    let p2_base = params.p2_base;
    let p2_min = params.p2_min;
    let pen1 = (p1 / 2).max(1); // 0.5 px step
    let pen2 = p1.max(pen1); // 1.0 px step

    let dirs: Vec<(i32, i32)> = if params.paths >= 8 {
        dirs_8().to_vec()
    } else {
        dirs_4().to_vec()
    };

    for (dy, dx) in dirs {
        if dy == 0 {
            // horizontal: rows are independent -> rayon over rows
            sums.par_chunks_mut(w * s_n).enumerate().for_each(|(y, row_sums)| {
                let forward = dx > 0;
                let mut prev_l = vec![0u16; s_n];
                let mut cur = vec![0u16; s_n];
                // two smallest of `prev_l` + argmin index, tracked while writing
                let mut m1: u16 = 0;
                let mut m2: u16 = 0;
                let mut m1i: usize = 0;
                for xi in 0..w {
                    let x = if forward { xi } else { w - 1 - xi };
                    let px = y * w + x;
                    let c_off = px * s_n;
                    let prev_x = if forward { x.wrapping_sub(1) } else { x + 1 };
                    let has_prev = if forward { x > 0 } else { x + 1 < w };
                    // adaptive P2 from intensity difference
                    let di = if has_prev {
                        let a = ref_img[px] as u16;
                        let b = ref_img[y * w + prev_x] as u16;
                        (a.max(b) - a.min(b)) as u16
                    } else {
                        0
                    };
                    let p2 = ((p2_base as u32 * 8) / (8 + di as u32)).max(p2_min as u32) as u16;
                    // track the two smallest of the line being written
                    let mut n1: u16 = u16::MAX;
                    let mut n2: u16 = u16::MAX;
                    let mut n1i: usize = 0;
                    for s in 0..s_n {
                        let c = cost.data[c_off + s] as u16;
                        let mut best = prev_l[s];
                        if s >= 1 {
                            best = best.min(prev_l[s - 1] + pen1);
                        }
                        if s + 1 < s_n {
                            best = best.min(prev_l[s + 1] + pen1);
                        }
                        if s >= 2 {
                            best = best.min(prev_l[s - 2] + pen2);
                        }
                        if s + 2 < s_n {
                            best = best.min(prev_l[s + 2] + pen2);
                        }
                        // exact far-jump term: min over d' != s of prev + P2
                        let far = (if s == m1i { m2 } else { m1 }).saturating_add(p2);
                        best = best.min(far);
                        let v = c + best - m1;
                        cur[s] = v;
                        if v < n1 {
                            n2 = n1;
                            n1 = v;
                            n1i = s;
                        } else if v < n2 {
                            n2 = v;
                        }
                    }
                    let sums_off = x * s_n;
                    for s in 0..s_n {
                        row_sums[sums_off + s] += cur[s];
                    }
                    std::mem::swap(&mut prev_l, &mut cur);
                    m1 = n1;
                    m2 = n2;
                    m1i = n1i;
                }
            });
        } else {
            // vertical / diagonal: rows sequential, previous row at x - dx.
            // Within a row the dependency is only on the PREVIOUS row, so
            // x-parallelism is safe. v4: per-pixel (min1, min2, argmin) are
            // computed while the row is written and reused by the next row.
            let mut prev_row = vec![0u16; w * s_n];
            let mut cur_row = vec![0u16; w * s_n];
            let mut prev_mins = vec![(0u16, 0u16, 0usize); w];
            let mut cur_mins = vec![(0u16, 0u16, 0usize); w];
            let y_order: Vec<usize> = if dy > 0 { (0..h).collect() } else { (0..h).rev().collect() };
            let chunk = 64usize.max(1);
            for y in y_order {
                {
                    let prev: &[u16] = &prev_row;
                    let pmins: &[(u16, u16, usize)] = &prev_mins;
                    cur_row
                        .par_chunks_mut(chunk * s_n)
                        .zip(cur_mins.par_chunks_mut(chunk))
                        .enumerate()
                        .for_each(|(ci, (row_cur, mins_cur))| {
                            let x_base = ci * chunk;
                            let x_end = (x_base + chunk).min(w);
                            for (k, x) in (x_base..x_end).enumerate() {
                                let px = y * w + x;
                                let c_off = px * s_n;
                                let xs = (x as i64 - dx as i64).clamp(0, (w - 1) as i64) as usize;
                                let prev_off = xs * s_n;
                                // adaptive P2
                                let yp = (y as i64 - dy as i64).clamp(0, (h - 1) as i64) as usize;
                                let a = ref_img[px] as u16;
                                let b = ref_img[yp * w + xs] as u16;
                                let di = a.max(b) - a.min(b);
                                let p2 = ((p2_base as u32 * 8) / (8 + di as u32))
                                    .max(p2_min as u32)
                                    as u16;
                                let (min1, min2, i1) = pmins[xs];
                                let mut n1: u16 = u16::MAX;
                                let mut n2: u16 = u16::MAX;
                                let mut n1i: usize = 0;
                                let cur_off = k * s_n;
                                for s in 0..s_n {
                                    let c = cost.data[c_off + s] as u16;
                                    let mut best = prev[prev_off + s];
                                    if s >= 1 {
                                        best = best.min(prev[prev_off + s - 1] + pen1);
                                    }
                                    if s + 1 < s_n {
                                        best = best.min(prev[prev_off + s + 1] + pen1);
                                    }
                                    if s >= 2 {
                                        best = best.min(prev[prev_off + s - 2] + pen2);
                                    }
                                    if s + 2 < s_n {
                                        best = best.min(prev[prev_off + s + 2] + pen2);
                                    }
                                    let far =
                                        (if s == i1 { min2 } else { min1 }).saturating_add(p2);
                                    best = best.min(far);
                                    let v = c + best - min1;
                                    row_cur[cur_off + s] = v;
                                    if v < n1 {
                                        n2 = n1;
                                        n1 = v;
                                        n1i = s;
                                    } else if v < n2 {
                                        n2 = v;
                                    }
                                }
                                mins_cur[k] = (n1, n2, n1i);
                            }
                        });
                }
                // accumulate row into sums (sequential, cache-friendly)
                for x in 0..w {
                    let px = y * w + x;
                    let sums_off = px * s_n;
                    let cur_off = x * s_n;
                    for s in 0..s_n {
                        sums[sums_off + s] += cur_row[cur_off + s];
                    }
                }
                std::mem::swap(&mut prev_row, &mut cur_row);
                std::mem::swap(&mut prev_mins, &mut cur_mins);
            }
        }
    }
}

/// Winner-take-all + uniqueness gate over the aggregated costs, writing into
/// caller-owned `disp` / `qual` (both w*h, overwritten entirely).
/// Returns the number of uniqueness rejections.
fn wta_into(
    cost: &CostVolume,
    sums: &[u16],
    params: &SgmParams,
    n_dirs: usize,
    disp: &mut [f32],
    qual: &mut [u8],
) -> usize {
    let w = cost.w;
    let h = cost.h;
    let s_n = cost.slices;
    let disp_min = cost.disp_min as f32;
    let sat = (n_dirs as u32 * (BORDER_COST - 10) as u32) as u16;
    let n_uniq = AtomicUsize::new(0);

    disp.par_chunks_mut(w)
        .zip(qual.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, (drow, qrow))| {
            drow.fill(f32::NAN);
            qrow.fill(Quality::Invalid as u8);
            for x in 0..w {
                if x < MATCH_BORDER
                    || x + MATCH_BORDER >= w
                    || y < MATCH_BORDER
                    || y + MATCH_BORDER >= h
                {
                    continue;
                }
                let off = (y * w + x) * s_n;
                // pass 1: best slice
                let mut best_s = 0usize;
                let mut best_v = u16::MAX;
                for s in 0..s_n {
                    let v = sums[off + s];
                    if v < best_v {
                        best_v = v;
                        best_s = s;
                    }
                }
                if best_v >= sat {
                    continue; // no usable match in range
                }
                // pass 2: second best at >= 1 px distance (2 slices)
                let mut second_v = u16::MAX;
                for s in 0..s_n {
                    if (s as i64 - best_s as i64).abs() >= 2 {
                        let v = sums[off + s];
                        if v < second_v {
                            second_v = v;
                        }
                    }
                }
                // uniqueness gate (SGBM semantics): reject ambiguous matches
                let ambig =
                    second_v != u16::MAX && (best_v as f32 * params.uniqueness) > second_v as f32;
                if ambig {
                    qrow[x] = Quality::Invalid as u8;
                    n_uniq.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let d = disp_min + best_s as f32 * 0.5;
                // subpixel: parabola on the aggregated cost at +-1 px (2 slices)
                let mut d_sub = d;
                if best_s >= 2 && best_s + 2 < s_n {
                    let c0 = sums[off + best_s - 2] as f32;
                    let c1 = best_v as f32;
                    let c2 = sums[off + best_s + 2] as f32;
                    let denom = c0 - 2.0 * c1 + c2;
                    if denom.abs() > 1e-6 {
                        let delta_slices = ((c0 - c2) / (2.0 * denom)).clamp(-2.0, 2.0);
                        d_sub = disp_min + (best_s as f32 + delta_slices) * 0.5;
                    }
                }
                drow[x] = d_sub;
                let strong = second_v == u16::MAX || second_v as f32 >= best_v as f32 * 1.6;
                qrow[x] = if strong { Quality::Strong as u8 } else { Quality::Weak as u8 };
            }
        });
    n_uniq.load(Ordering::Relaxed)
}

/// v4 pyramid refinement: for every full-res pixel, re-cost the integer
/// disparities in `[round(2*d_half) - 3, +3]` with the FULL-res two-scale
/// census, WTA, then a raw-cost parabola at +-1 px. Writes `disp` / `qual`.
/// Returns the number of uniqueness rejections.
fn refine_fullres(
    c_l: &Census,
    c_r: &Census,
    disp_h: &[f32],
    dm: usize,
    nd: usize,
    params: &SgmParams,
    w: usize,
    h: usize,
    hw: usize,
    disp: &mut [f32],
    qual: &mut [u8],
    coarse: &mut [bool],
) -> usize {
    let dm_i = dm as i64;
    let nd_i = nd as i64;
    let dmax_i = dm_i + nd_i - 1;
    let hh = (h + 1) / 2;
    let uniqueness = params.uniqueness;
    let n_uniq = AtomicUsize::new(0);

    disp.par_chunks_mut(w)
        .zip(qual.par_chunks_mut(w))
        .zip(coarse.par_chunks_mut(w))
        .enumerate()
        .for_each(|(y, ((drow, qrow), crow))| {
            drow.fill(f32::NAN);
            qrow.fill(Quality::Invalid as u8);
            crow.fill(false);
            if y < MATCH_BORDER || y + MATCH_BORDER >= h {
                return;
            }
            let row_h = (y / 2) * hw;
            let i_row = y * w;
            for x in MATCH_BORDER..(w - MATCH_BORDER) {
                let dh = disp_h[row_h + x / 2];
                if !dh.is_finite() {
                    continue; // unmatchable at half res; fill handles it
                }
                let d0 = dh as f64 * 2.0;
                // census cost at an arbitrary integer disparity (full-res)
                let cost_at = |k: i64| -> u16 {
                    let xs = x as i64 - k;
                    if k < dm_i
                        || k > dmax_i
                        || xs < MATCH_BORDER as i64
                        || xs > (w - 1 - MATCH_BORDER) as i64
                    {
                        BORDER_COST
                    } else {
                        hamming_fine(c_l.fine[i_row + x], c_r.fine[i_row + xs as usize])
                            + hamming_ring(c_l.ring[i_row + x], c_r.ring[i_row + xs as usize])
                    }
                };
                let dc = d0.round() as i64;
                let lo = (dc - REFINE_RADIUS).max(dm_i);
                let hi = (dc + REFINE_RADIUS)
                    .min(dmax_i)
                    .min(x as i64 - MATCH_BORDER as i64);
                // candidate integer costs (full-res census)
                let mut best_k = i64::MIN;
                let mut best_c = u16::MAX;
                let mut second_c = u16::MAX;
                if hi >= lo {
                    for k in lo..=hi {
                        let xs = x as i64 - k;
                        if xs < MATCH_BORDER as i64 || xs > (w - 1 - MATCH_BORDER) as i64 {
                            continue;
                        }
                        let c = hamming_fine(c_l.fine[i_row + x], c_r.fine[i_row + xs as usize])
                            + hamming_ring(c_l.ring[i_row + x], c_r.ring[i_row + xs as usize]);
                        if c < best_c {
                            second_c = best_c;
                            best_c = c;
                            best_k = k;
                        } else if c < second_c {
                            second_c = c;
                        }
                    }
                }
                // v4.1 GLOBAL RESCUE: the +-3 window can never recover when the
                // coarse prior is off by more than 3 px (small objects,
                // half-res boundary ramps) - those pixels would emit
                // confidently-WRONG disparities that poison clusters. When
                // the window holds no confident match (cost > 12/32), run a
                // full-range census WTA and adopt it when clearly better.
                if best_c > 12 || best_k == i64::MIN {
                    let k_lo = dm_i.max(x as i64 - (w as i64 - 1 - MATCH_BORDER as i64));
                    let k_hi = dmax_i.min(x as i64 - MATCH_BORDER as i64);
                    let mut gk = i64::MIN;
                    let mut gc = u16::MAX;
                    let mut gs = u16::MAX;
                    if k_hi >= k_lo {
                        for k in k_lo..=k_hi {
                            let c = cost_at(k);
                            if c < gc {
                                gs = gc;
                                gc = c;
                                gk = k;
                            } else if c < gs {
                                gs = c;
                            }
                        }
                    }
                    if gk != i64::MIN && gc + 2 < best_c {
                        best_k = gk;
                        best_c = gc;
                        second_c = gs;
                    }
                }
                // Confidence policy (the v4 architectural lesson): SGM's
                // smoothness-interpolated coarse estimate BEATS a blind local
                // census WTA on weak texture - refine only when the local
                // minimum is unique, otherwise keep the coarse value as Weak.
                // The fallback is only allowed on LOCALLY SMOOTH half-res
                // disparities: at depth discontinuities the half-res map
                // smears across the edge and would bridge objects.
                let ambiguous =
                    best_k == i64::MIN || (second_c != u16::MAX && (best_c as f32 * uniqueness) > second_c as f32);
                if ambiguous {
                    n_uniq.fetch_add(1, Ordering::Relaxed);
                    // Fallback trust test (v4.1): the coarse SGM estimate is
                    // kept only where the FULL-RES census cost at the coarse
                    // disparity is consistent with the local best candidate
                    // (+4 hamming). SGM smears depth discontinuities into
                    // smooth RAMPS - the neighbour smoothness gate cannot see
                    // those, but at the smeared pixels the coarse value sits
                    // BETWEEN two surfaces and the census cost exposes it.
                    let cd0 = cost_at(d0.round() as i64);
                    if d0 >= dm_i as f64
                        && d0 <= dmax_i as f64
                        && best_c != u16::MAX
                        && cd0 <= best_c.saturating_add(4)
                        && half_res_locally_smooth(disp_h, x / 2, y / 2, hw, hh)
                    {
                        drow[x] = d0 as f32;
                        qrow[x] = Quality::Weak as u8; // coarse estimate
                        crow[x] = true;
                    }
                    continue;
                }
                // subpixel: parabola on the RAW integer costs at best_k +- 1
                let c0 = cost_at(best_k - 1) as f32;
                let c2 = cost_at(best_k + 1) as f32;
                let c1 = best_c as f32;
                let mut d = best_k as f32;
                let denom = c0 - 2.0 * c1 + c2;
                if denom.abs() > 1e-6 {
                    let delta = ((c0 - c2) / (2.0 * denom)).clamp(-1.0, 1.0);
                    d = best_k as f32 + delta;
                }
                drow[x] = d;
                let strong = second_c == u16::MAX || second_c as f32 >= best_c as f32 * 1.6;
                qrow[x] = if strong { Quality::Strong as u8 } else { Quality::Weak as u8 };
            }
        });
    n_uniq.load(Ordering::Relaxed)
}

/// v4: is the half-res disparity locally smooth at (xh, yh)? Used to gate
/// the coarse fallback: SGM interpolation is trustworthy on smooth surfaces,
/// but smears across depth discontinuities (all 4 half-res neighbours must
/// be valid and agree within 1.5 half-px).
#[inline]
fn half_res_locally_smooth(disp_h: &[f32], xh: usize, yh: usize, hw: usize, hh: usize) -> bool {
    let d = disp_h[yh * hw + xh];
    for (dx, dy) in [(0i64, 1i64), (0, -1), (1, 0), (-1, 0)] {
        let nx = xh as i64 + dx;
        let ny = yh as i64 + dy;
        if nx < 0 || ny < 0 || nx >= hw as i64 || ny >= hh as i64 {
            return false;
        }
        let nb = disp_h[ny as usize * hw + nx as usize];
        if !nb.is_finite() || (nb - d).abs() > 1.5 {
            return false;
        }
    }
    true
}

/// Full-resolution path (v3 behaviour, v4 optimizations).
fn fullres_pass(
    params: &SgmParams,
    bufs: &mut StereoBuffers,
    w: usize,
    h: usize,
    n_dirs: usize,
) -> (usize, usize) {
    census_into(&bufs.li, &mut bufs.c_l);
    census_into(&bufs.ri, &mut bufs.c_r);
    build_cost_into(&bufs.c_l, &bufs.c_r, params.disp_min, params.num_disparities, &mut bufs.vol);
    let need = w * h * bufs.vol.slices;
    bufs.sums.clear();
    bufs.sums.resize(need, 0);
    aggregate(&bufs.vol, bufs.li.as_raw(), params, &mut bufs.sums);
    bufs.disp.clear();
    bufs.disp.resize(w * h, f32::NAN);
    bufs.qual.clear();
    bufs.qual.resize(w * h, 0);
    bufs.coarse.clear();
    bufs.coarse.resize(w * h, false);
    let n_uniq = wta_into(&bufs.vol, &bufs.sums, params, n_dirs, &mut bufs.disp, &mut bufs.qual);
    let n_lr = if params.do_lr {
        resize_voting(bufs, w * h);
        post::lr_check_voting(
            &mut bufs.disp,
            &mut bufs.qual,
            w,
            h,
            params.lr_thresh,
            &mut bufs.vsum,
            &mut bufs.vcnt,
        )
    } else {
        0
    };
    (n_uniq, n_lr)
}

fn resize_voting(bufs: &mut StereoBuffers, n: usize) {
    if bufs.vsum.len() != n {
        bufs.vsum.clear();
        bufs.vsum.resize(n, 0.0);
        bufs.vcnt.clear();
        bufs.vcnt.resize(n, 0);
    }
}

/// Pyramid path (v4): dense search at half res + full-res local refinement.
fn pyramid_pass(
    params: &SgmParams,
    bufs: &mut StereoBuffers,
    w: usize,
    h: usize,
    n_dirs: usize,
) -> (usize, usize) {
    let hw = (w + 1) / 2;
    let hh = (h + 1) / 2;
    downsample2x(&bufs.li, &mut bufs.half_l);
    downsample2x(&bufs.ri, &mut bufs.half_r);
    census_into(&bufs.half_l, &mut bufs.c_lh);
    census_into(&bufs.half_r, &mut bufs.c_rh);
    census_into(&bufs.li, &mut bufs.c_l);
    census_into(&bufs.ri, &mut bufs.c_r);

    let dm = params.disp_min;
    let nd = params.num_disparities;
    // disparity range scales by 0.5 at half resolution
    let dm_h = dm / 2;
    let dmax_h = (((dm + nd) as f64) / 2.0).ceil() as usize;
    let nd_h = dmax_h.saturating_sub(dm_h).max(2);

    build_cost_into(&bufs.c_lh, &bufs.c_rh, dm_h, nd_h, &mut bufs.vol);
    let need = hw * hh * bufs.vol.slices;
    bufs.sums.clear();
    bufs.sums.resize(need, 0);
    aggregate(&bufs.vol, bufs.half_l.as_raw(), params, &mut bufs.sums);
    bufs.disp_h.clear();
    bufs.disp_h.resize(hw * hh, f32::NAN);
    bufs.qual_h.clear();
    bufs.qual_h.resize(hw * hh, 0);
    let _n_uniq_h = wta_into(
        &bufs.vol,
        &bufs.sums,
        params,
        n_dirs,
        &mut bufs.disp_h,
        &mut bufs.qual_h,
    );

    // full-res refinement around the upsampled coarse disparity
    bufs.disp.clear();
    bufs.disp.resize(w * h, f32::NAN);
    bufs.qual.clear();
    bufs.qual.resize(w * h, 0);
    bufs.coarse.clear();
    bufs.coarse.resize(w * h, false);
    let n_uniq = refine_fullres(
        &bufs.c_l,
        &bufs.c_r,
        &bufs.disp_h,
        dm,
        nd,
        params,
        w,
        h,
        hw,
        &mut bufs.disp,
        &mut bufs.qual,
        &mut bufs.coarse,
    );
    let n_lr = if params.do_lr {
        resize_voting(bufs, w * h);
        post::lr_check_voting(
            &mut bufs.disp,
            &mut bufs.qual,
            w,
            h,
            params.lr_thresh,
            &mut bufs.vsum,
            &mut bufs.vcnt,
        )
    } else {
        0
    };
    (n_uniq, n_lr)
}

/// Full pipeline: returns the left-image disparity map. Uses (and reuses)
/// caller-owned scratch buffers; the steady state allocates only the output
/// map itself.
pub fn match_pair_with_buffers(
    left: &GrayImage,
    right: &GrayImage,
    params: &SgmParams,
    bufs: &mut StereoBuffers,
) -> DisparityMap {
    let (lw, lh) = left.dimensions();
    let (rw, rh) = right.dimensions();
    assert_eq!((lw, lh), (rw, rh), "stereo pair dimensions must match");
    let (w, h) = (lw as usize, lh as usize);
    let dm = params.disp_min;

    if params.median_prefilter {
        post::median3_gray_into(left, &mut bufs.li);
        post::median3_gray_into(right, &mut bufs.ri);
    } else {
        clone_gray_into(left, &mut bufs.li);
        clone_gray_into(right, &mut bufs.ri);
    }

    let n_dirs = if params.paths >= 8 { 8 } else { 4 };
    let (n_uniq, n_lr) = if params.pyramid {
        pyramid_pass(params, bufs, w, h, n_dirs)
    } else {
        fullres_pass(params, bufs, w, h, n_dirs)
    };

    let mut n_speckle = 0usize;
    if params.do_speckle {
        let valid_before = bufs.disp.iter().filter(|d| d.is_finite()).count();
        post::speckle_filter(
            &mut bufs.disp,
            &mut bufs.qual,
            w,
            h,
            params.speckle_size,
            params.speckle_diff,
        );
        let valid_after = bufs.disp.iter().filter(|d| d.is_finite()).count();
        n_speckle = valid_before - valid_after;
    }
    if params.do_fill {
        post::fill_rowwise(&mut bufs.disp, &mut bufs.qual, w, h);
    }
    if params.median_postfilter {
        post::median3_disparity(&mut bufs.disp, w, h);
    }

    let mut stats = post::collect_stats(&bufs.disp, &bufs.qual, w, h);
    stats.n_rejected_uniqueness = n_uniq;
    stats.n_rejected_lr = n_lr;
    stats.n_rejected_speckle = n_speckle;
    let mut sigma = post::sigma_from_quality(&bufs.qual);
    // v4: coarse half-res estimates carry a wider honest sigma (half-res
    // quantisation + SGM smearing tail; calibrated on the validation scene)
    for i in 0..w * h {
        if bufs.coarse[i] {
            sigma[i] = sigma[i].max(0.80);
        }
    }
    let disp = std::mem::take(&mut bufs.disp);
    let qual = std::mem::take(&mut bufs.qual);
    let coarse = std::mem::take(&mut bufs.coarse);
    DisparityMap {
        width: w,
        height: h,
        disp,
        sigma,
        quality: qual,
        coarse,
        disp_min: dm as f32,
        stats,
    }
}

/// Allocating wrapper (tests / one-shot use); see `match_pair_with_buffers`.
pub fn match_pair(left: &GrayImage, right: &GrayImage, params: &SgmParams) -> DisparityMap {
    let mut bufs = StereoBuffers::new();
    match_pair_with_buffers(left, right, params, &mut bufs)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Random-dot stereogram: right(x) = left(x + d(x)). The matcher must
    /// recover piecewise-constant and smooth disparities.
    fn rds(w: usize, h: usize, d_fn: &dyn Fn(usize) -> f64, seed: u64) -> (GrayImage, GrayImage) {
        let mut rng = crate::core::rng::GaussRng::new(seed);
        let mut left = GrayImage::new(w as u32, h as u32);
        let mut right = GrayImage::new(w as u32, h as u32);
        let vals: Vec<u8> = (0..w * h).map(|_| (rng.uniform() * 255.0) as u8).collect();
        for y in 0..h {
            for x in 0..w {
                left.put_pixel(x as u32, y as u32, image::Luma([vals[y * w + x]]));
                let src = (x as f64 + d_fn(x)).round() as i64;
                let src = src.clamp(0, (w - 1) as i64) as usize;
                right.put_pixel(x as u32, y as u32, image::Luma([vals[y * w + src]]));
            }
        }
        (left, right)
    }

    #[test]
    fn rds_bands_recovered() {
        let w = 200usize;
        let h = 96usize;
        let d_fn = |x: usize| if x < w / 3 { 10.0 } else if x < 2 * w / 3 { 20.0 } else { 30.0 };
        let (left, right) = rds(w, h, &d_fn, 7);
        let params = SgmParams {
            num_disparities: 40,
            disp_min: 4,
            p1: 8,
            p2_base: 64,
            paths: 4,
            ..Default::default()
        };
        let map = match_pair(&left, &right, &params);
        // Explicit evaluation windows away from occlusion zones. At a band
        // edge E where disparity jumps to d_next, left pixels [E, E+d_next)
        // have no valid right-image correspondent. Edges: 66 (d->20), 133 (d->30).
        let zones: [(usize, usize); 3] = [
            (MATCH_BORDER + 12, 64),
            (88, 131),
            (165, w - MATCH_BORDER - 12),
        ];
        let mut n_ok = 0;
        let mut n_tot = 0;
        let mut sum_err = 0.0;
        let mut n_miss = 0usize;
        for y in (MATCH_BORDER + 6)..h - MATCH_BORDER - 6 {
            for &(x0, x1) in &zones {
                for x in x0..=x1.min(w - 1) {
                    let gt = d_fn(x) as f32;
                    let d = map.get(x, y);
                    n_tot += 1;
                    if d.is_finite() && (d - gt).abs() <= 0.6 {
                        n_ok += 1;
                    } else if d.is_finite() {
                        sum_err += (d - gt).abs();
                        n_miss += 1;
                    }
                }
            }
        }
        let frac = n_ok as f64 / n_tot.max(1) as f64;
        let mean_miss = if n_miss > 0 { sum_err as f64 / n_miss as f64 } else { 0.0 };
        assert!(frac > 0.95, "RDS band recovery {frac:.3} ({n_ok}/{n_tot}), mean abs err on misses {mean_miss:.2}");
    }

    #[test]
    fn rds_bands_recovered_pyramid() {
        // v4: the pyramid path must keep the same band accuracy
        let w = 200usize;
        let h = 96usize;
        let d_fn = |x: usize| if x < w / 3 { 10.0 } else if x < 2 * w / 3 { 20.0 } else { 30.0 };
        let (left, right) = rds(w, h, &d_fn, 7);
        let params = SgmParams {
            num_disparities: 40,
            disp_min: 4,
            p1: 8,
            p2_base: 64,
            paths: 4,
            pyramid: true,
            ..Default::default()
        };
        let map = match_pair(&left, &right, &params);
        let zones: [(usize, usize); 3] = [
            (MATCH_BORDER + 12, 64),
            (88, 131),
            (165, w - MATCH_BORDER - 12),
        ];
        let mut n_ok = 0;
        let mut n_tot = 0;
        for y in (MATCH_BORDER + 6)..h - MATCH_BORDER - 6 {
            for &(x0, x1) in &zones {
                for x in x0..=x1.min(w - 1) {
                    let gt = d_fn(x) as f32;
                    let d = map.get(x, y);
                    n_tot += 1;
                    if d.is_finite() && (d - gt).abs() <= 0.75 {
                        n_ok += 1;
                    }
                }
            }
        }
        let frac = n_ok as f64 / n_tot.max(1) as f64;
        assert!(frac > 0.93, "pyramid RDS band recovery {frac:.3} ({n_ok}/{n_tot})");
    }

    #[test]
    fn rds_half_pixel_disparity() {
        // constant 12.5 px shift must be recovered at 0.5 px accuracy
        let w = 160usize;
        let h = 96usize;
        let (left, right) = rds(w, h, &|_| 12.5, 11);
        let params = SgmParams {
            num_disparities: 32,
            disp_min: 4,
            paths: 4,
            ..Default::default()
        };
        let map = match_pair(&left, &right, &params);
        let mut errs = Vec::new();
        for y in 10..h - 10 {
            for x in 24..w - 24 {
                let d = map.get(x, y);
                if d.is_finite() {
                    errs.push((d - 12.5).abs());
                }
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = errs[errs.len() / 2];
        assert!(med <= 0.5, "half-pixel median error {med:.3}");
    }

    #[test]
    fn rds_half_pixel_disparity_pyramid() {
        // v4: pyramid path, same 0.5 px sub-pixel expectation
        let w = 160usize;
        let h = 96usize;
        let (left, right) = rds(w, h, &|_| 12.5, 11);
        let params = SgmParams {
            num_disparities: 32,
            disp_min: 4,
            paths: 4,
            pyramid: true,
            ..Default::default()
        };
        let map = match_pair(&left, &right, &params);
        let mut errs = Vec::new();
        for y in 10..h - 10 {
            for x in 24..w - 24 {
                let d = map.get(x, y);
                if d.is_finite() {
                    errs.push((d - 12.5).abs());
                }
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = errs[errs.len() / 2];
        assert!(med <= 0.6, "pyramid half-pixel median error {med:.3}");
    }

    #[test]
    fn buffers_reused_give_identical_results() {
        // v4 contract: steady-state reuse must be bit-identical to fresh buffers
        let w = 160usize;
        let h = 96usize;
        let (left, right) = rds(w, h, &|x: usize| if x < 80 { 12.0 } else { 24.0 }, 21);
        for pyramid in [false, true] {
            let params = SgmParams {
                num_disparities: 32,
                disp_min: 4,
                pyramid,
                ..Default::default()
            };
            let fresh = match_pair(&left, &right, &params);
            let mut bufs = StereoBuffers::new();
            let mut first: Option<DisparityMap> = None;
            // NaN == NaN for this comparison (invalid pixels must agree too)
            let disp_eq = |a: &[f32], b: &[f32]| {
                a.len() == b.len()
                    && a.iter().zip(b.iter()).all(|(x, y)| {
                        (x.is_nan() && y.is_nan()) || x == y
                    })
            };
            for _ in 0..3 {
                let m = match_pair_with_buffers(&left, &right, &params, &mut bufs);
                if let Some(f) = &first {
                    assert!(disp_eq(&f.disp, &m.disp), "pyramid={pyramid}: reused buffers must be deterministic");
                    assert_eq!(f.quality, m.quality);
                } else {
                    first = Some(m);
                }
            }
            let reused = first.unwrap();
            assert!(disp_eq(&fresh.disp, &reused.disp), "pyramid={pyramid}: reused == fresh");
        }
    }

    #[test]
    fn textureless_region_flagged() {
        // flat gray: no usable texture; the honest matcher must NOT emit
        // strong matches there
        let w = 120usize;
        let h = 80usize;
        let mut left = GrayImage::new(w as u32, h as u32);
        let mut right = GrayImage::new(w as u32, h as u32);
        for y in 0..h {
            for x in 0..w {
                left.put_pixel(x as u32, y as u32, image::Luma([128u8]));
                right.put_pixel(x as u32, y as u32, image::Luma([128u8]));
            }
        }
        // sprinkle a textured border band so the map is not 100% empty
        let mut rng = crate::core::rng::GaussRng::new(3);
        for y in 0..h {
            for x in 0..12 {
                let v = (rng.uniform() * 255.0) as u8;
                left.put_pixel(x as u32, y as u32, image::Luma([v]));
                right.put_pixel(x as u32, y as u32, image::Luma([v]));
            }
        }
        let params = SgmParams { num_disparities: 24, paths: 4, ..Default::default() };
        let map = match_pair(&left, &right, &params);
        let mut n_strong = 0;
        let mut n_tot = 0;
        for y in 10..h - 10 {
            for x in 40..w - 10 {
                n_tot += 1;
                if map.quality(x, y) == Quality::Strong {
                    n_strong += 1;
                }
            }
        }
        assert!(n_strong * 100 < n_tot, "textureless region must not emit strong matches: {n_strong}/{n_tot}");
    }
}

#[cfg(test)]
mod debug_vga {
    use super::*;
    use crate::camera::model::Intrinsics;
    use crate::sim::scene::SceneSpec;

    #[test]
    fn debug_pyramid_stages() {
        // where does the pyramid lose coverage: half-res WTA or refinement?
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let scene = SceneSpec::demo();
        let pair = scene.render_frame(0, k, 77);
        let params = SgmParams { pyramid: true, ..Default::default() };
        let mut bufs = StereoBuffers::new();
        post::median3_gray_into(&pair.left, &mut bufs.li);
        post::median3_gray_into(&pair.right, &mut bufs.ri);
        pyramid_pass(&params, &mut bufs, 640, 480, 4);
        let hw = 320usize;
        let hh = 240usize;
        let valid_h = bufs.disp_h.iter().filter(|d| d.is_finite()).count();
        let strong_h = bufs.qual_h.iter().filter(|&&q| q == 3).count();
        let weak_h = bufs.qual_h.iter().filter(|&&q| q == 2).count();
        eprintln!(
            "half-res WTA: valid {valid_h}/{} ({:.1}%), strong {strong_h}, weak {weak_h}",
            hw * hh,
            100.0 * valid_h as f64 / (hw * hh) as f64
        );
        let valid = bufs.disp.iter().filter(|d| d.is_finite()).count();
        eprintln!("full-res refined valid before post: {valid} ({:.1}%)", 100.0 * valid as f64 / 307200.0);
        let mut errs = Vec::new();
        for i in 0..640 * 480 {
            if bufs.qual[i] >= 2 && pair.gt.depth[i] > 0.0 {
                let z = k.fx * 0.16 / bufs.disp[i] as f64;
                errs.push((z - pair.gt.depth[i] as f64).abs());
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !errs.is_empty() {
            eprintln!("refined depth err: median {:.3} p90 {:.3} (n={})", errs[errs.len() / 2], errs[errs.len() * 9 / 10], errs.len());
        }
        // where is disp_h invalid? sample rows
        let mut rows = vec![0usize; 12];
        for i in 0..hw * hh {
            if !bufs.disp_h[i].is_finite() {
                rows[i / hw / 20] += 1;
            }
        }
        eprintln!("disp_h invalid per 20-row band (half-res): {rows:?}");
        // depth error of the HALF map itself (x2 scale)
        let mut errs_h = Vec::new();
        for i in 0..hw * hh {
            if bufs.qual_h[i] >= 2 {
                // GT depth at the corresponding full-res pixel
                let fx = (i % hw) * 2;
                let fy = (i / hw) * 2;
                let gt = pair.gt.depth[fy * 640 + fx];
                if gt > 0.0 {
                    let z = k.fx * 0.16 / (bufs.disp_h[i] as f64 * 2.0);
                    errs_h.push((z - gt as f64).abs());
                }
            }
        }
        errs_h.sort_by(|a, b| a.partial_cmp(b).unwrap());
        if !errs_h.is_empty() {
            eprintln!(
                "half-res depth err (2x scaled): median {:.3} p90 {:.3} (n={})",
                errs_h[errs_h.len() / 2],
                errs_h[errs_h.len() * 9 / 10],
                errs_h.len()
            );
        }
    }

    #[test]
    fn debug_vga_stats() {
        let k = Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480);
        let scene = SceneSpec::demo();
        let pair = scene.render_frame(0, k, 77);
        for (label, params) in [
            ("default", SgmParams::default()),
            ("no-lr", SgmParams { do_lr: false, ..Default::default() }),
            ("loose-uniq", SgmParams { uniqueness: 1.1, ..Default::default() }),
            ("no-uniq-no-lr", SgmParams { uniqueness: 0.9, do_lr: false, ..Default::default() }),
            ("pyramid", SgmParams { pyramid: true, ..Default::default() }),
        ] {
            let d = match_pair(&pair.left, &pair.right, &params);
            eprintln!("{label}: strong {} weak {} filled {} invalid {} | uniq_rej {} lr_rej {} speckle_rej {}",
                d.stats.n_strong, d.stats.n_weak, d.stats.n_filled, d.stats.n_invalid,
                d.stats.n_rejected_uniqueness, d.stats.n_rejected_lr, d.stats.n_rejected_speckle);
            // depth error on measured pixels
            let mut errs = Vec::new();
            for i in 0..640 * 480 {
                if d.quality[i] >= 2 && pair.gt.depth[i] > 0.0 {
                    let z = k.fx * 0.16 / d.disp[i] as f64;
                    errs.push((z - pair.gt.depth[i] as f64).abs());
                }
            }
            errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            if !errs.is_empty() {
                eprintln!("   depth err: median {:.3} p90 {:.3} (n={})", errs[errs.len()/2], errs[errs.len()*9/10], errs.len());
            }
        }
    }
}
