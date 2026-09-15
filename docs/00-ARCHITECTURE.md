# OTD 6.0 — Architecture Document

**OTD** = *Open Three-Dimensional Language*. **2.0 "DEEP"** = the second
generation, inspired by the OTDL v5 tiered design: a **Lite → Deep → ASM**
three-tier engine, still **100 % Rust + handwritten assembly, zero external
libraries**, still learnable by a 12-year-old in 5 minutes.

> Write a sentence, hold a thing — and now the *math under the thing* is real:
> half-edge topology, Discrete Exterior Calculus, implicit-field blending,
> XPBD physics, GGX microfacet light, and a real bytecode virtual machine.

---

## 0. The Contract This Build Must Honor (unchanged + extended)

| Rule | Status |
|---|---|
| Zero external crates — `Cargo.toml` `[dependencies]` is **empty** | enforced |
| Backend is **only** Rust and hand-written assembly (`asm!` / CPU intrinsics) | enforced |
| Browser is a *dumb terminal*: server-rendered pixels, vanilla JS viewer | enforced |
| Own TCP HTTP/1.1 server from `std::net` only | enforced |
| Own software 3D renderer, own PNG/DEFLATE, own JSON/Base64 | enforced |
| **Every OTD 1.0 file still runs unchanged** (doc 08 policy) | enforced |
| **< 60 keywords.** 1.0 shipped 54 → 2.0 ships **58** | enforced |
| New tiers are *engine internals* — the surface stays simple | enforced |

What 2.0 adds on top of 1.0: the **three-tier engine** (§1), four new words
(`smooth`, `subdiv`, `blend`, `rope`), a **bytecode VM** that actually
executes the language's math, **DEC-based** mesh science, **SDF blending**,
an **XPBD** solver, and a **GGX + soft-shadow** renderer.

---

## 1. The Three-Tier Architecture (the OTDL v5 idea, made real in pure Rust)

The v5 specification describes a language split into three tiers — a
declarative front, a computational core, and a bytecode machine. Its syntax
(braces, semicolons, GPU qualifiers) would break the "12-year-old" promise,
so OTD 2.0 adopts the **architecture** and rejects the **surface**: the tiers
are *inside the engine*, and the user still writes 8 friendly lines.

```
┌────────────────────────────────────────────────────────────────┐
│  TIER 1 — LITE (what the user writes)                          │
│  scene "Rope Bridge"                                           │
│  deck = plank(w: 60cm, d: 8cm, h: 1cm)                         │
│  cable = rope(from: (0,40cm,0), to: (60cm,40cm,0), sag: 6cm)   │
│  towerL = box(w:4cm,h:40cm,d:4cm)                              │
│  smooth(n:2) rock = icosphere(r: 5cm) subdiv(n:1)              │
│  blend(base, rock, gap: 1cm)                                   │
│  ask "cable sag?"   ask "mass?"                                │
├────────────────────────────────────────────────────────────────┤
│  TIER 2 — DEEP (computational core, engine-internal)           │
│  · Half-edge topology (Euler-style edge ops, manifold checks)  │
│  · DEC: cotangent Laplacian → smooth(n, strength)              │
│  · Loop subdivision → subdiv(n)                                │
│  · SDF implicit fields: BVH closest-point → marching tets      │
│    → blend(a, b, gap) smooth-union                             │
│  · XPBD position-based dynamics → rope() catenaries            │
│  · BVH ray casting → soft shadows + baked ambient occlusion    │
├────────────────────────────────────────────────────────────────┤
│  TIER 3 — ASM (the virtual machine, engine-internal)           │
│  · Expressions compile to OTD-ASM bytecode                     │
│  · 32-byte fixed-stride vector ISA (v5 §12 layout)             │
│  · VM executes with asm!/SSE kernels in the hot loop           │
│  · `--vm-dump` prints a human-readable disassembly             │
└────────────────────────────────────────────────────────────────┘
```

**Why a VM?** 1.0 walked the expression tree with a recursive evaluator.
2.0 compiles every expression (units, `i/j/a` magic vars, math functions,
`pi`) into **OTD-ASM bytecode** and executes it on a register machine whose
inner loops use hand-written assembly. Same answers (parity-tested), faster
loops (benchmarked), and the "assembly language" requirement stops being a
kernel garnish and becomes the *actual execution model* of the language.

---

## 2. System Architecture — The Sixteen Layers

1.0's twelve layers (L01–L12) remain; 2.0 adds L13–L16.

```
L01 TEXT      .otd source (the only thing the user writes)
L02 LEXER     units glued to numbers, line-based statements
L03 PARSER    recursive descent → AST (EBNF = doc 03)
L04 VM        NEW  expressions → OTD-ASM bytecode → register VM
              (32-byte instructions, asm!/SSE hot kernels)
L05 EVALUATOR templates, patterns, booleans, transforms, defaults
L06 GEOMETRY  12 primitives · 8 builders · BSP CSG · hollow
L07 TOPOLOGY  NEW  half-edge mesh structure + manifold validator
L08 DEC       NEW  cotangent Laplacian smoothing (smooth)
L09 SUBDIV    NEW  Loop subdivision (subdiv)
L10 IMPLICIT  NEW  SDF fields: BVH closest-point + marching tets
              → blend() smooth-union CSG
L11 MEASURE   divergence-theorem volume/centroid/area (unchanged)
L12 MATERIALS 27 real materials + 147 colors (unchanged)
L13 PHYSICS   mass=ΣV·ρ, Archimedes float, drop, collapse
              + NEW XPBD solver → rope() catenary, inertia tensor
L14 RENDERER  NEW GGX microfacet (Smith G, Schlick F), BVH
              ray-traced soft shadows + baked AO, SSAA
L15 EXPORT    STL · OBJ · GLB · SCAD · PNG (unchanged)
L16 SERVE     own HTTP/1.1 + JSON + Base64 + caches (unchanged)
```

Module map of the new code:

| File | Layer | Responsibility |
|---|---|---|
| `src/vm/mod.rs` | L04 | VM facade: compile-once, run-many |
| `src/vm/isa.rs` | L04 | 32-byte instruction encoding + disassembler |
| `src/vm/compile.rs` | L04 | AST expression → bytecode compiler |
| `src/vm/interp.rs` | L04 | register-machine interpreter, asm! hot loops |
| `src/geo/halfedge.rs` | L07 | half-edge structure from triangle soup |
| `src/geo/dec.rs` | L08 | cotan weights, Laplacian smoothing |
| `src/geo/subdiv.rs` | L09 | Loop subdivision with boundary rules |
| `src/geo/sdf.rs` | L10 | field sampling, smooth-min, marching tets |
| `src/geo/bvh.rs` | L10/L14 | median-split BVH: closest-point + rays |
| `src/world/xpbd.rs` | L13 | XPBD solver: particles + distance constraints |
| `src/world/rope.rs` | L13 | catenary solve → tube mesh |
| `src/render/ggx.rs` | L14 | microfacet BRDF terms |
| `src/simd/assembly.rs` | all | + new asm! kernels (rsqrt, dot3, mad) |

---

## 3. New Language Surface — four words, that's all

The v5 spec would add hundreds of keywords (structs, qualifiers, kernels).
OTD adds **four**, chosen so each one is a *door* into a whole tier:

| # | Word | Tier unlocked | Example |
|---|---|---|---|
| 55 | `smooth` | DEC | `smooth(n: 3, strength: 0.5) rock` — relaxes a mesh with the cotangent Laplacian; corners melt, volume is tracked |
| 56 | `subdiv` | topology | `sphere(r: 3cm) subdiv(n: 2)` — Loop subdivision: 4× triangles each level, smooth spheres from icospheres |
| 57 | `blend` | implicit | `blend(base, knob, gap: 8mm)` — smooth-union fillet between two solids, like solder |
| 58 | `rope` | XPBD | `rope(from: (0,30cm,0), to: (40cm,30cm,0), thickness: 4mm, sag: 5cm)` — a real physics-solved cable, hung between two points |

Grammar-wise each is an ordinary call (doc 03 covers the exact EBNF):
`smooth` and `subdiv` are **postfix modifiers** (like `rotate`, they chain:
`rock = blob smooth(n:2)`), `blend` is a **builder call** taking two shapes,
`rope` is a **primitive** producing a swept-tube mesh along the solved
catenary. All defaults are smart: `smooth(n:1)`, `subdiv(n:1)`,
`gap: 5mm`, `thickness: 4mm`, `sag: auto` (XPBD decides).

**Budget check: 54 + 4 = 58 keywords — under 60, forever.**

New `ask` questions (free — `ask` is one keyword, the parser of the
*question* is the engine's business): `"inertia?"`, `"center of mass?"`,
`"surface area?"`, `"cable sag?"`, `"triangle count?"`, `"subdiv level?"`.

---

## 4. Tier 3 — the OTD-ASM Virtual Machine

### 4.1 Instruction encoding (v5 §12, faithfully adapted)

Every instruction is exactly **32 bytes**, SIMD- and cache-aligned:

```
 0                   1                   2                   3
 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
+-------------------+-------------------+-------------------+
|  Opcode (16 bits) | Exec Mask (8 bits)| Vector Stride (8b)|
+-------------------+-------------------+-------------------+
|               Destination Register Index (32 bits)          |
+-----------------------------------------------------------+
|               Source A Register Index (32 bits)             |
+-----------------------------------------------------------+
|               Source B Register Index (32 bits)             |
+-----------------------------------------------------------+
|               Source C Register Index (32 bits)             |
+-----------------------------------------------------------+
|               Immediate Value Payload (64 bits)             |
+-----------------------------------------------------------+
|               Predicate / Swizzle Control (32 bits)         |
+-----------------------------------------------------------+
```

### 4.2 The geometric vector ISA (subset implemented in 2.0)

| Opcode | Mnemonic | Action |
|---|---|---|
| `0x0001` | `V_LOAD_F64` | push immediate/variable into register |
| `0x0002` | `V_STORE_F64` | write register to output slot |
| `0x0010` | `V_ADD_F64` | rDst = rA + rB |
| `0x0011` | `V_SUB_F64` | rDst = rA − rB |
| `0x0012` | `V_MUL_F64` | rDst = rA × rB |
| `0x0013` | `V_DIV_F64` | rDst = rA ÷ rB (zero → friendly error) |
| `0x0014` | `V_NEG_F64` | rDst = −rA |
| `0x0015` | `V_RSQ_F64` | rDst = 1/√rA (asm! fast path) |
| `0x0016` | `V_DOT3_F64` | rDst = rA · rB (SSE kernel) |
| `0x0018` | `V_MIN_F64` / `V_MAX_F64` | rDst = min/max |
| `0x0020` | `V_SIN_D` / `0x0021 V_COS_D` | degrees-in trig |
| `0x0022` | `V_SQRT_F64` | rDst = √rA |
| `0x0023` | `V_ABS_F64` / `0x0024 V_ROUND_F64` | as named |
| `0x0030` | `V_CONST_PI` | rDst = π |
| `0x0040` | `V_UNIT_MM` | attach unit: value → millimeters |
| `0x0041` | `V_UNIT_DEG` | attach unit: value → degrees |
| `0x0100` | `JMP` / `0x0101 JNZ` | control flow (patterns) |
| `0x0102` | `HALT` | end of program |

(The v5 opcodes `TOPO_SPLIT_EDGE`, `BVH8_TEST_RAYS`, `XPBD_PROJ_DIST`,
`DEC_COTAN_LAP` are honored as *reserved encodings* — the engine implements
their semantics in Tier 2 Rust; the VM owns pure math. This keeps one ISA
story while each tier runs where it is fastest.)

### 4.3 Pipeline

```
Expr::Num(4cm)  ──compile──▶  V_LOAD_F64 r0, imm 40.0 ; V_UNIT_MM r0
Expr::Bin(Mul)  ──compile──▶  V_MUL_F64  r2, r0, r1
pattern body    ──compile──▶  program with JMP/JNZ over i
              ──run──▶  interp: 256 f64 registers, unit tag per reg,
                        asm!/SSE dot & rsqrt kernels, opcode stats
```

Every expression the evaluator meets is compiled **once** and executed by
the VM with a unit tag per register (length-mm, angle-deg, or plain). A
parity test fuzzes VM vs tree-walk over thousands of expressions — bitwise
identical results required. `--vm-dump file.otd` prints the disassembly:

```
$ otd --vm-dump examples/cup.otd
;; 14 instructions, 2 constants, 1 run
0000  V_LOAD_F64  r0, 40.0        ; 4cm
0001  V_UNIT_MM   r0
0002  V_LOAD_F64  r1, 100.0       ; 10cm
...
0013  HALT
```

### 4.4 Where assembly actually lives

| Kernel | File | Instruction |
|---|---|---|
| 4-wide f64 dot product | `simd/assembly.rs` | SSE2 `asm!` |
| fast reciprocal sqrt | `simd/assembly.rs` | SSE2 `asm!` + Newton |
| scan fill / pixel blend | `simd/mod.rs` | SSE2 intrinsics |
| frame clear (`rep stosd`) | `simd/assembly.rs` | raw `asm!` |
| VM register-file zeroing | `vm/interp.rs` | raw `asm!` `rep stosq` |

All kernels have Rust fallbacks (`cfg(not(target_arch="x86_64"))`) and
parity tests — assembly accelerates, never defines.

---

## 5. Tier 2 — the Deep geometry science

### 5.1 Half-edge topology (`geo/halfedge.rs`)

From any closed triangle mesh we build the v5-style half-edge structure:
`target_vertex · opposite · next · prev · incident_face`, twin = boundary
marker. Uses: manifold validation (every edge exactly 2 faces, opposite
consistent), vertex one-ring iteration for smoothing/subdivision, and the
`ask "watertight?"` answer. Boundary half-edges (twin = `0xFFFFFFFF`) are
legal — `plane` and `hollow(open:)` produce them.

### 5.2 DEC smoothing (`geo/dec.rs`, the `smooth` word)

The cotangent Laplacian on a triangle mesh:

  Δφ(vᵢ) = Σⱼ∈N(i) wᵢⱼ (φⱼ − φᵢ),   wᵢⱼ = ½ (cot αᵢⱼ + cot βᵢⱼ)

where α, β are the angles opposite edge (i,j). One smoothing step moves each
vertex by λ·Δv (Taubin λ|μ alternation with μ = −0.33λ keeps volume from
collapsing — no shrinkage artifacts). Validation: smoothing a cube converges
toward its inscribed sphere's curvature distribution; volume shrink per
step is measured and reported by `ask "volume?"` — the physics of the mesh
never lies.

### 5.3 Loop subdivision (`geo/subdiv.rs`, the `subdiv` word)

Each triangle → 4 (edge midpoints, ³⁄₈−⅛ masks, boundary Catmull-Rom).
Icosphere + subdiv(2) gives a near-perfect sphere: volume error vs analytic
(4/3)πr³ drops from 12.5 % (icosphere L0) to < 2 % (L2) — measured in the
test report. This is how OTD 2.0 makes *smooth* spheres without raising the
keyword count.

### 5.4 SDF blending (`geo/sdf.rs` + `geo/bvh.rs`, the `blend` word)

The union of two solids as *distance fields* with a smooth-min:

  smin(a, b, k) = −ln(e^(−ka) + e^(−kb)) / k      (k = 1/gap)

We compute the distance of a sample point to each mesh with a **BVH
closest-point query** (median-split tree, f64), sample the blended field on
a lattice over the union's bounding box, and extract the surface with
marching tetrahedra (the metaball machinery, generalized). Result: two
solids fuse with a concave fillet of radius ≈ gap — "solder". Volume is
measured after blending so `ask "mass?"` stays honest.

### 5.5 XPBD dynamics (`world/xpbd.rs`, the `rope` word)

Extended Position-Based Dynamics, small-and-real:

```
for substep (8 per solve):
    v ← v + g·dt ;  p ← p + v·dt          (predict)
    for edge (a,b) with rest length L:
        C = |p_a − p_b| − L ;  Δλ = −C/(w_a+w_b+α̃) ;  α̃ = compliance/dt²
        p_a += Δλ·w_a·n ;  p_b −= Δλ·w_b·n  (project)
    v ← (p − p_prev)/dt                    (velocity update)
```

`rope(from, to, thickness, sag)` discretizes the span into 48 particles,
runs XPBD under scene gravity with pinned ends, and sweeps a tube (existing
`tube` builder) along the solved polyline — a physically-true hanging cable.
`ask "cable sag?"` returns the measured maximum dip. Validation: XPBD
converges to the analytic catenary y = a·cosh(x/a) — sag error < 2 % at 48
particles (test report), and the rope's *length* is honest: XPBD preserves
rest length within 0.1 %.

### 5.6 New physics answers (`world/ask.rs`)

`ask "inertia?"` — the mass moment of inertia tensor of each part about its
center of mass, computed exactly by tetrahedron quadrature over the mesh
(divergence-theorem-style, like volume):

  I_xx = Σₜ ρ ∫ (y²+z²) dV ,  I_xy = −Σₜ ρ ∫ xy dV ,  …

per tetrahedron of the triangulation — closed forms, no sampling. Reported
in g·mm² with the principal-axis suggestion ("it spins easily about Y").

---

## 6. Tier 2/14 — GGX microfacet + ray-traced light

Blinn-Phong (1.0) is replaced by an energy-aware **Cook-Torrance** stack:

  f_r = k_d·albedo/π + D(α)·F(v,h)·G₂(α)/(4 (n·l)(n·v))
  D = α²/(π ((n·h)²(α²−1)+1)²)   (GGX/Trowbridge-Reitz)
  F = F₀ + (1−F₀)(1−v·h)⁵        (Schlick)
  G₂ = Smith height-correlated    (exact, cheap form)

`α = roughness²`, F₀ = 0.04 dielectric / material color for metals —
existing material data (roughness, shine, color) maps 1:1, so old files just
look *better*. Shadows become **ray-traced**: a BVH over all triangles
answers "is this point in shadow?" per pixel; 4 jittered rays per shade
point give penumbrae; 16 hemisphere rays per vertex bake ambient occlusion
(the v5 `ao_factor`, ours computed offline). Cost is bounded by the frame
cache (server renders once per orbit step, not per mouse move).

---

## 7. Syntax audit (the standing "find bugs and fix" gate)

2.0 keeps 1.0's audited grammar (docs 03/06) and re-audits every new
addition. Rules enforced on the four new words:

1. **Call-form consistency** — new words appear only as calls
   (`smooth(n:2)`, `rope(...)`) or postfix modifiers (`blob smooth(n:2)`),
   never as bare keywords, so the lexer never needs new statement forms.
2. **Colon tolerance** — like every statement word, `smooth:` / `blend:`
   read identically with or without the colon.
3. **Unit positions** — `gap:`, `thickness:`, `sag:` are length-typed;
   bare numbers take the scene default unit; angles stay `deg`-typed.
4. **Modifier ordering** — `shape smooth(n) rotate …` applies smoothing
   *before* placement (same left-to-right rule as all postfix chains);
   `blend(a, b)` argument shapes are pre-merged solids.
5. **Keyword collision** — `smooth`, `blend`, `rope`, `subdiv` were checked
   against the 70 parameter names and 27 materials/147 colors: no collision
   (`smooth` was a *parameter* of terrain in 1.0 — it stays valid there by
   context, see doc 03 §collision rules).
6. **Backward compatibility** — the 1.0 example corpus (10 files) is a
   regression test; every file must compile byte-identically (same stats).

Full EBNF: doc 03 (updated). Error messages for the new words follow the
friendly-error contract (doc 06), e.g. `gap cannot be negative — did you
mean 8mm?`.

---

## 8. Phase mapping (10,000-phase plan)

1.0 delivered P0000–P0649. 2.0 delivers **P0650–P1299**:

| Phases | Content |
|---|---|
| P0650–P0799 | VM: ISA, compiler, interpreter, asm! kernels, parity fuzz |
| P0800–P0899 | half-edge topology + manifold validator |
| P0900–P0999 | DEC cotan Laplacian + `smooth` (Taubin) |
| P1000–P1049 | Loop subdivision + `subdiv` |
| P1050–P1149 | BVH + SDF fields + marching tets + `blend` |
| P1150–P1249 | XPBD + `rope` + inertia quadrature + new `ask`s |
| P1250–P1299 | GGX + soft shadows + AO + docs/lessons/examples/tests |

Later mega-stages (P1300+): parametrics, animation timeline, collaborative
editing, GPU path — designed for, not yet built; forward-compat policy (doc
08) guarantees old files never break.

---

## 9. Validation plan (test with mathematics & physics)

Every new tier is validated by measurement, not assertion — full numbers in
doc 09:

| Claim | Test |
|---|---|
| VM ≡ tree-walk | 10⁴ random expressions, bitwise-identical results |
| asm! kernels ≡ Rust | dot/rsqrt/memset parity, exact bounds |
| smooth keeps shape class | cube→rounding: volume monotonically ↓, bbox ↓ < 5 %/step, genus unchanged |
| subdiv → sphere | volume error vs (4/3)πr³: 12.5 % → < 2 % at n:2 |
| blend ≈ solder | volume(a∪b) ≤ volume(blend) ≤ volume(a∪b) + fillet bound |
| rope ≈ catenary | sag vs analytic cosh: < 2 % error; length preserved < 0.1 % |
| inertia tensor | analytic cube: I_xx = m/6·s² exact match |
| GGX energy | ∫f_r·cosθ dω ≤ 1 + ε (energy conservation bound) |
| 1.0 corpus | 10 examples, stats unchanged (compat gate) |

---

## 10. What 2.0 does NOT do (and why)

The v5 spec also lists NURBS B-Rep solids, 3D Gaussian splats, polarized
spectral BSDFs, GPU warp scheduling, and SPIR-V export. Each needs data the
language can't express in <60 words or hardware a zero-dep Rust binary
can't assume, so they stay **designed-for** (reserved ISA opcodes, phase
slots P1300+) rather than half-built. OTD's promise is: small text, big
models, *true* math — not every buzzword.
