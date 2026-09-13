# OTD — The 10,000-Phase Plan (2.0: P0000–P1299)

OTD is not built in one shot. It is built as **P0000 … P9999**: ten mega-stages,
each split into 1000 numbered phases. A *phase* is the smallest capability that
can be independently built, tested, and that keeps every older `.otd` file
working (forward-compatibility policy, doc 08).

**This build delivers P0000–P0649 — the complete Phase-1 vertical slice.**

## Stage map

| Stage | Phases | Theme | Status |
|---|---|---|---|
| S0 | P0000–P0999 | **Foundation** — language, geometry, truth | ✅ P0000–P0649 done |
| S1 | P1000–P1999 | **Robustness** — CSG hardening, fillets, offsets | designed |
| S2 | P2000–P2999 | **Motion** — animation, joints, gears that turn | designed |
| S3 | P3000–P3999 | **Parametrics** — variables, constraints, math everywhere | designed |
| S4 | P4000–P4999 | **Ecosystem** — package registry, import network models | designed |
| S5 | P5000–P5999 | **Physics Pro** — rigid bodies, contacts, fluid fill | designed |
| S6 | P6000–P6999 | **GPU path** — optional compute renderer (still no libs) | designed |
| S7 | P7000–P7999 | **Collaboration** — multi-user scenes, history | designed |
| S8 | P8000–P8999 | **Fabrication** — slicer hints, print checks, AR export | designed |
| S9 | P9000–P9999 | **Intelligence** — "make me a castle" model synthesis | designed |

## Implemented phases (P0000–P0649)

| Phase | Capability |
|---|---|
| P0000 | cargo skeleton, zero-dependency guarantee |
| P0010 | units: mm/cm/m/in/ft + deg, dimension checking |
| P0020 | Vec3/Mat4/AABB math core (f64) |
| P0100 | lexer: unit-glued numbers, strings, `#` comments |
| P0110 | AST |
| P0120 | recursive-descent parser (line-based statements) |
| P0130 | the 54-keyword registry (count < 60, tested) |
| P0140 | friendly errors + Levenshtein suggestions |
| P0200 | mesh container, transforms, normals |
| P0210 | primitives: cube/sphere |
| P0220 | cylinder (tapered) / cone |
| P0230 | torus / pyramid / prism |
| P0240 | capsule / wedge / plane |
| P0300 | BSP CSG: union |
| P0310 | BSP CSG: subtract, intersect |
| P0320 | analytic hollow (profile shapes, exact) |
| P0330 | generic shell fallback (reported honestly) |
| P0400 | builders: extrude (ear-clip) |
| P0410 | revolve |
| P0420 | tube / helix |
| P0430 | text (5×7 dot font) |
| P0440 | terrain (own fBm) |
| P0450 | metaballs (marching tetrahedra) |
| P0460 | sweep / loft |
| P0470 | STL/OBJ import |
| P0500 | divergence-theorem volume/centroid/area |
| P0510 | slice area (collapse stress) |
| P0520 | 27 materials database |
| P0530 | 147 named colors + hex |
| P0540 | physics: mass/float/drop/collapse + gravity presets |
| P0550 | ask engine (natural questions, answers with math) |
| P0560 | evaluator: patterns i/j/a, define/use, add-fuse, hide/show |
| P0600 | software rasterizer + Blinn-Phong + shadows + SSAA |
| P0610 | SSE2/AVX2 kernels + `asm!` assembly kernels |
| P0620 | own DEFLATE/PNG/CRC32/Adler32 |
| P0630 | own HTTP/1.1 server + JSON + Base64 + caches |
| P0640 | exporters STL/OBJ/GLB/SCAD/PNG + vanilla viewer |

## Rules that hold for every future phase

1. **A phase never removes a keyword, never changes a default, never re-parses
   an old file differently.** New capabilities are new words or new named
   arguments only (doc 08).
2. Every phase lands with tests (`cargo test`) and a doc note.
3. Zero-dependency and Rust+asm-only are permanent constraints.
4. If a phase would need an external library, the phase is re-designed — not
   the constraint.


---

## Stage S1 — OTD 2.0 "DEEP" (P0650–P1299) — COMPLETE

| Phases | Delivered |
|---|---|
| P0650–P0680 | OTD-ASM: 32-byte vector ISA, expression→bytecode compiler, register VM with asm! kernels, `--vm-dump` |
| P0700 | VM wired into the evaluator (compile-once cache, magic vars per iteration) + `pi` |
| P0800–P0810 | half-edge topology + mesh welding |
| P0900 | DEC: cotangent Laplacian, Taubin `smooth` |
| P1000 | Loop `subdiv` |
| P1050–P1080 | BVH, SDF fields, `blend`, Freudenthal–Kuhn marching tets, metaball manifold fix |
| P1150–P1190 | XPBD solver, catenary math, `rope` with measured sag |
| P1180 | exact inertia tensor (tetrahedron quadrature) |
| P1250–P1270 | GGX/Smith/Schlick shading, ray-traced soft shadows, baked AO |
| P1290–P1295 | deep-tier `ask` queries, 2.0 docs/examples/tests/validation |

Stage S2 (P1300+): parametrics, animation timeline, cloth/soft-body XPBD
bodies, crease-tagged subdivision, dual contouring with sharp features,
collaborative editing, the GPU path. All designed-for, none break old
files (doc 08).

