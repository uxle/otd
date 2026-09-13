# 11 — Performance Architecture: never lag, on any device

Goal: OTD3 should feel instant on a five-year-old laptop and a flagship
desktop alike. This document works the way the brief asked — philosophy
first, then the math, then the numbers that test the math, then (in the
source tree) the implementation. Everything below was checked against the
actual `src/` code, not designed in the abstract; call sites and line
numbers are named so this stays falsifiable.

## 1. Design philosophy

Four principles, in priority order:

1. **Never redo work a camera move didn't require.** A render has an
   expensive geometry-dependent half (lighting) and a cheap
   camera-dependent half (projection). If a frame only moved the camera,
   only the cheap half should run. This is the single biggest lever,
   because it turns *N frames of interaction* from *N × expensive* into
   *1 × expensive + N × cheap* (§3).
2. **Degrade quality before you degrade responsiveness.** A slightly
   blurrier frame during a drag is invisible; a frozen frame is not.
   Perceived lag is a latency problem first and a quality problem second
   — solve it by shrinking the frame, not by trying to make ray tracing
   fast enough to hide the fact that it's still O(cells × rays) per
   mouse-move.
3. **Parallelize the part of the cost that's embarrassingly parallel, and
   nothing else.** Not every loop needs threads; the ones worth
   threading are the ones where Amdahl's law actually pays off (§2).
   Threading a loop that's 20% of total cost wastes engineering effort
   for a ≤1.25× ceiling no matter the core count.
4. **A stale result in flight is worse than no result.** If the user has
   already moved past the frame the server is computing, finishing that
   computation is pure waste — cancel it and spend the cycles on the
   frame that matters now.

None of these require a faster ray tracer. They require not calling the
ray tracer when it isn't needed, and shrinking its input when its output
won't be looked at for more than 30ms anyway.

## 2. The math

### 2.1 Where the cost actually is

`render_world_rgb` (`src/render/render_impl.rs`) does, per frame:

| Step | Depends on camera? | Cost |
|---|---|---|
| merge visible parts into one mesh | no | O(V) |
| `Bvh::build` | no | O(T log T), T = triangles |
| `creased_normals` per part | no | O(T) |
| `vertex_shadow_ao` (soft shadows + AO) | no | O(cells × 9 rays × log T) |
| clip / project / rasterize | **yes** | O(T) |
| PNG encode | **yes** (resolution) | O(pixels) |

`vertex_shadow_ao` deduplicates by spatial cell before ray tracing (a
92k-vertex mesh can be ~4k distinct cells — see the function's own
comment), but every one of those cells still fires 9 rays through the
BVH, and that dominates. None of it reads the camera: the key light
direction is fixed in world space (`Lights::default()`,
`src/render/raster.rs`), so shadow/AO visibility is identical from every
angle.

**Theory:** split the render into `prepare_lit_scene` (rows 1–4 above)
and `render_view` (rows 5–6). Cache the former per distinct geometry;
call the latter once per camera.

### 2.2 Amdahl's law: is threading the ray-tracing pass worth it?

Let `f` be the fraction of `vertex_shadow_ao`'s cost that's the
embarrassingly-parallel ray-tracing pass (pass 2) vs. the sequential
bucket/scatter passes (1 and 3, pure O(V) hashing — not worth
threading, their absolute cost is small and threading them would spend
more on synchronization than it saves). Speedup on P cores:

```
speedup(P) = 1 / ((1 − f) + f / P)
```

Calibrating against the cup example already documented in the code (92k
verts, ~4k cells, 9 rays/cell, a BVH ray query costing roughly 50× a
bucket-hash op) gives **f ≈ 0.91** — 91% of the cost is the parallel
part. That predicts:

| cores | Amdahl speedup |
|---|---|
| 2 | 1.83× |
| 4 | 3.13× |
| 8 | 4.85× |
| 16 | 6.69× |

This says two things: threading pass 2 is worth doing (large `f`), and
it scales with whatever hardware the visitor happens to have — a 2-core
phone still gets ~1.8×, a 16-core workstation gets ~6.7×, with zero
device-specific tuning. That's what "any device" needs to mean
architecturally: code that automatically uses however much parallelism
is present (`std::thread::available_parallelism()`), never a number
baked in for one machine.

### 2.3 Two-tier adaptive quality: how much does dropping quality save?

Rasterization + downsampling + PNG cost scales with pixel count, which
scales with `(resolution scale)² × (ssaa)²`. Relative cost:

```
cost(scale, ssaa) = scale² × ssaa²
```

Full quality (`scale=1.0, ssaa=2`) = 4.0 units. A drag-time preview at
`scale=0.6, ssaa=1` = 0.36 units — **11.1× less raster/encode work and
proportionally less base64 payload over the network**, on top of
skipping the lighting pass entirely via the cache in §2.1. Confirmed by
direct calculation (§4), not assumed.

### 2.4 Total interactive-session cost, old vs. new

For an N-frame orbit drag of the same script, with `T_lit` = one lighting
pass and `T_raster` = one raster/encode pass:

```
old:  N × (T_lit + T_raster_full)
new:  T_lit + N × T_raster_interactive
```

Because `T_lit` is normally the larger term (it's the ray-traced part),
the new architecture's cost approaches a **constant** (the one-time
`T_lit`) as N grows, while the old cost grows linearly with every mouse
event. The crossover is immediate — even N=5 already shows a 5× total
reduction in §4's numbers, and it keeps growing with drag length.

## 3. What actually got built (see source for details)

- `src/render/render_impl.rs` — introduces `LitScene` /
  `prepare_lit_scene` / `render_view` / `render_view_rgb`, splitting the
  render exactly along the table in §2.1. `render_world`/
  `render_world_rgb` remain as single-shot wrappers — every existing
  caller (`main.rs`, `tests/`, the export routes) is unchanged and pays
  the same cost as before. `vertex_shadow_ao` now ray-traces its unique
  cells across `available_parallelism()` threads via `std::thread::scope`
  (no unsafe, no locks — each thread only reads the shared, immutable
  BVH/mesh).
- `src/render/octocam.rs` — the eight-view OCTOCAM panel used to call the
  old all-in-one render eight times per panel, rebuilding the identical
  BVH and re-ray-tracing the identical shadows/AO eight times over (this
  was a real, provable 8× waste — same geometry, only the camera
  differs). It now calls `prepare_lit_scene` once and renders the eight
  views — themselves independent — across threads.
- `src/net/routes.rs` — the HTTP render route now caches the `LitScene`
  per script (same content hash already used for the compile cache), so
  a browser orbit/zoom drag pays the ray-tracing cost once per script,
  not once per mouse-move. (This sits alongside an earlier fix in the
  same file: the cache mutex used to be held across the actual
  compile/render call, serializing every connection on the server's
  thread-per-connection model; it's now held only for the map
  lookup/insert.)
- `assets/app.js` — the client now requests a smaller, ssaa=1 preview
  while a drag/zoom/touch is in progress, and one full-quality frame the
  moment it stops (mirroring how Blender/Fusion360/SketchUp-style
  viewers behave). A new in-flight render is aborted (`AbortController`)
  rather than queued behind, so responsiveness during a drag is never
  bounded by a frame the user has already moved past.

## 4. Test results (the numbers above, actually computed)

```
parallel fraction f = 0.9073  (pass2=1800000, pass1+3=184000)
  P= 2 cores -> Amdahl speedup x1.83
  P= 4 cores -> Amdahl speedup x3.13
  P= 8 cores -> Amdahl speedup x4.85
  P=16 cores -> Amdahl speedup x6.69

full-quality relative raster cost units = 4.000
interactive relative raster cost units  = 0.360
raster/PNG/network-payload reduction during drag = 11.1x

N= 1 frames: old=  1989000u  new=  1984450u  ->   1.0x less server work
N= 5 frames: old=  9945000u  new=  1986250u  ->   5.0x less server work
N=10 frames: old= 19890000u  new=  1988500u  ->  10.0x less server work
N=30 frames: old= 59670000u  new=  1997500u  ->  29.9x less server work
```

These were computed from a small calibrated cost model (§2), not
measured on hardware — **there is no Rust toolchain in the environment
this change was made in**, so `cargo build`/`cargo test` could not be
run to confirm the code compiles or to measure real wall-clock numbers.
The refactor was written to be mechanically behavior-preserving (same
formulas, same iteration, just relocated and — for the ray-tracing
pass — reordered across threads with no shared mutable state, so results
should be bit-identical to before, just faster) and the existing
`tests/perf.rs` / `tests/render_cup.rs` should still pass unchanged.

**Before shipping:** run `cargo build && cargo test`, then re-run
`tests/perf.rs` and note the new `render cup … ms` figure against the
number already printed by that test today — that's the real-hardware
confirmation this document's math predicts but couldn't execute here.

## 5. What this deliberately does *not* do (next candidates, not done)

- **Tile-parallel rasterization within one view.** Skipped for now: it
  needs either per-tile framebuffers merged afterward or triangle
  binning, more invasive than the changes above, and rasterization is no
  longer the dominant cost once lighting is cached — lower expected
  payoff for the risk.
- **A continuous frame-budget controller** (server measures its own
  render time and picks `scale`/`ssaa` to hit a target, rather than the
  binary "interacting vs. settled" switch shipped here). The binary
  switch is simpler, needs no telemetry plumbing, and is easier to
  reason about without a compiler to test against; a continuous
  controller is the natural next step once real timing data exists.
- **Geometry-aware caching for the stereo-pair renderer**
  (`main.rs`'s `left`/`right` PNGs) — those two `World`s are genuinely
  different geometry (translated meshes), so the content-hash cache
  doesn't apply without a shape-aware key. Not attempted here.
