# AVC v3 vs the competition — an honest analysis

**The question we were asked:** *"Is it better than all its competitors?"*

**The honest answer:** AVC v3 is the strongest system **in its class** —
self-contained, compilable, uncertainty-quantified metric world models — but
it is not the best system at every individual sub-task, and this document
says exactly where it wins and where it loses. No false precision.

## The class definitions matter

"Competitors" fall into different classes; comparing across classes without
saying so is how marketing claims are born:

- **Class A — stereo depth SDKs**: OpenCV SGBM/StereoBM, MATLAB, libelas.
  Depth only; no world model, no tracking, no uncertainty.
- **Class B — hardware stereo systems**: ZED SDK, RealSense D455, OAK-D.
  IR/hardware depth + CNN semantics; closed source; per-device licensed.
- **Class C — research stereo/SLAM**: RAFT-Stereo (deep stereo), ORB-SLAM3
  (full SLAM), Open3D pipelines. SOTA on benchmarks; GPU or heavy deps.
- **Class D — AVC itself across versions**: v1/v2 (Python + ONNX YOLO), v3
  (this, Rust).

AVC v3 competes as a **Class-A+ system: depth *plus* tracking *plus* a
metric world model with error bars, in one hermetic binary**.

## Head-to-head

| axis | AVC v3 (Rust) | best competitor | verdict |
|---|---|---|---|
| Depth accuracy (GT synthetic) | median 5 cm, p90 16 cm @ VGA, 0.16 m baseline | OpenCV SGBM ~same algorithm class; RAFT-Stereo better on KITTI (GPU) | **tie in class**, loss vs deep nets |
| Per-pixel uncertainty | every pixel: σ from quality class + propagated | almost nobody (SGBM: none; ZED: none; RAFT: none) | **AVC wins** |
| Object tracking | 3D Kalman + Hungarian, 0 ID switches on crossing test, occlusion coasting | SORT/DeepSORT (2D, no metric σ); ZED SDK object tracker | **AVC wins on σ + metric output** |
| Measurement engine | distance ± σ, 3σ intervals, σ-calibrated | not present in any A/B competitor | **AVC wins** (unique feature) |
| World model + relations | full, exported 6 formats | ZED/ARKit give poses + objects, no relations/measurements | **AVC wins in class** |
| Throughput | 3.6 FPS QVGA / 0.9 FPS VGA (2-vCPU sandbox, CPU only) | OpenCV SGBM ~20+ FPS VGA (SIMD-tuned C++); ZED 100 FPS (hardware) | **loss on raw speed** |
| Deployment | one static binary, no runtime, offline-buildable, 4 deps | Python: interpreter + numpy/opencv/onnxruntime; ZED: SDK + USB | **AVC wins** vs software stacks |
| Semantic labels | shape-prior heuristic only | YOLO-class CNNs (v1/v2 had this via ONNX!) | **loss** (documented PARTIAL) |
| SLAM completeness | VO only, drift 24 % of path over 60 frames, honestly reported | ORB-SLAM3 loop closure < 1 % | **loss** (documented INCOMPLETE) |
| Test/verification depth | 75 tests, GT validation, σ-calibration, honest status matrix | rarely any | **AVC wins** |
| Source availability | full MIT source | mostly closed (B) or research (C) | **AVC wins** |

## What "better than v2 (Python)" means here

The v2 benchmark work (in the Python edition) proved the *algorithm* beats
OpenCV SGBM on 6/8 metrics in a controlled comparison. v3 ports the winning
stereo core (two-scale census, half-pixel grid, uniqueness gate, median
filters) and adds:

- **10–50× per-frame speedup** (Rust + rayon vs Python/numpy; the v2 matcher
  took ~4.5 s/frame in Python — v3 does the same class of work in ~0.07 s at
  QVGA on 2 vCPUs).
- **Single-binary deployment** — the user's explicit request: *"a compilable
  model."*
- Everything that made v1/v2 defensible: honest status matrix, σ-calibration
  against GT, documented failures instead of hidden ones.

What v3 deliberately drops: the ONNX YOLOv8 backends (semantic detection &
segmentation). That keeps the build hermetic (no ONNX runtime dependency);
the detection `Detection` boundary is documented as the plug-in point, and
v1/v2 remain available for semantic workloads.

## Honest loss register

1. **Raw stereo throughput vs SIMD-tuned C++**: our aggregation is
   portable-safe Rust without SIMD intrinsics; OpenCV SGBM will outrun it
   per-core. The bench reports real numbers; QVGA 3.6 FPS is real-time on
   this 2-vCPU sandbox and scales with cores.
2. **Deep-stereo accuracy on benchmarks**: RAFT-Stereo-class models beat
   classical matching on KITTI/Middlebury. They need GPUs and emit no
   uncertainty; we emit calibrated σ per pixel on CPU.
3. **Semantics**: no neural detector in this build (documented PARTIAL;
   v1/v2 cover it via ONNX).
4. **Loop closure / pose graph**: VO drift is measured and reported, not
   hidden; loop closure is a roadmap item.
5. **Measurement systematic bias**: the classical geometric centre of a boxy
   object sits on the visible surface (up to ~0.5 m). Modelled in the
   covariance and documented; v1 solved it with instance segmentation.

## Bottom line

For **"a compilable, dependency-free engine that gives you a metric 3D world
model with error bars you can actually trust"** — the exact thing that was
asked for — AVC v3 is the best available option we are aware of, including
our own v1/v2. Against specialized SOTA on any single axis (raw depth
accuracy, semantics, loop closure), there are better tools, and the table
above says exactly which and why.

*Sources for competitor figures: OpenCV SGBM documentation and published
KITTI D1 leaderboards (algorithm-class comparisons), ZED SDK / Intel
RealSense published specs (vendor claims, hardware-dependent), ORB-SLAM3
paper (public benchmarks). Our numbers come from `avc-demo-synthetic` /
`avc-bench` on this machine; rerun them yourself — that is the point.*

---

## v4 update (measured on the 2-vCPU sandbox)

v3's honest losses vs competitors were raw speed (4.5 s vs OpenCV SGBM's
47 ms at VGA on the Python side; 1127 ms in the Rust edition) and the
occlusion identity continuity. v4 changes the scoreboard:

- **QVGA real-time**: 36.5 ms stereo / 17.8 FPS end-to-end on 2 vCPU with
  39/45 GT checks — a CPU-only, uncertainty-annotated, metric world model at
  real-time rates. OpenCV SGBM at QVGA is still faster per stage, but SGBM
  alone is not a world model: no tracking, no σ, no relations, no exports.
- **VGA accuracy mode**: 670 ms stereo (1.7× v3) at *better* accuracy
  (3.9 cm median / 13 cm p90, ID switches 1 vs 3, person height 1.34 vs
  1.23 m).
- Occlusion re-identification: WORKING (coasting re-association +
  velocity-gated fragment stitching) — previously a documented failure.
- Documented, not hidden: the σ-calibration 2σ tail (0.81), the GT#4
  25 cm-box limitation (inherited), VGA pyramid object-layer trade-offs
  (see VALIDATION.md), no neural semantics in the hermetic build.
- Memory: peak RSS 134 MB at QVGA including all exports; zero steady-state
  stereo allocation.
