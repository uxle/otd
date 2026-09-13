# OTD3.3 "QUARK" — Open Three-Dimensional Language

> **The Subatomic Edition.** Everything you build is made of exactly
> three particles — proton, neutron, electron — and those are made of
> two quarks stitched by gluons. This edition adds the layer underneath
> the grams: `simulate: atom` converts every part's measured mass into a
> real particle census (a 984 g iron cube = 1.06×10²⁵ atoms, K2 L8 M14 N2
> shells, and a charge ledger that balances to the last electron);
> `simulate: decay` reports the **live activity in becquerels** of the
> radioactive materials from their actual atom count (955 g of uranium =
> 11.88 MBq — twelve million α-decays per second, right now, and nothing
> you do can slow or speed them); and `simulate: particles` is the
> Standard Model briefing — all 17 fundamental particles, the quark model
> of the proton, and the famous accounting: **the quarks in a proton
> weigh 9.1 MeV; the proton weighs 938.3 MeV — 99% of your mass is gluon
> field energy.** The new `particle: <name>` statement prints any card
> from the deck (`particle: gluon`, `particle: neutrino`…). Underneath:
> the full 118-element periodic table, Madelung electron configurations
> with the 20 known exceptions (gold really answers 4f¹⁴ 5d¹⁰ 6s¹),
> Weizsäcker's binding-energy formula (Fe-56 to 0.4% of the measured
> value), the Rydberg spectrum, and the decay law — with the vendored
> reasoning-AI answering it all in plain language, VERIFIED or ABSTAINED
> as always (`docs/12-PARTICLES.md`).

| Area | The upgrade |
|---|---|
| **`simulate: atom`** | The census: per-material element, Z, shells (K2 L8 M14 N2), full electron configuration, binding energy vs the iron-56 peak, nucleus-vs-atom size ratio ("if the nucleus were a 1 cm marble, the electrons would be 220 m away"), scene-wide proton/neutron/electron totals and the charge ledger |
| **`simulate: decay`** | Half-lives, decay modes and chains (U-238 → Pb-206 in 14 steps), live activity A = λN in Bq from the actual atom count, after-100-years fractions, and the trace C-14 in every organic part (the archaeologist's clock) |
| **`simulate: particles`** | The Standard Model briefing: all 17 cards (6 quarks, 6 leptons, 4 gauge bosons, Higgs), the proton's gluon-energy accounting, confinement, the scene's own thermal photon glow, and the neutrino flux passing through it |
| **`particle: <name>`** | Keyword #78 — one particle's card straight from the deck: mass, charge, spin, statistics, and its one-paragraph story |
| **The tables** | 118 elements with Z, dominant isotope and molar mass; 15 famous radioactive isotopes with half-lives and modes; material → chemical composition for every OTD material (steel = Fe + C, oak = cellulose, water = H₂O…) |
| **The laws** | Weizsäcker liquid drop (Fe-56 8.76 vs 8.79 measured), Rydberg (Hα 656.5 nm), decay law (25% → exactly 2 half-lives), E = hc/λ, λ = h/p, β⁻ Q = 0.782 MeV, e⁺e⁻ → 2γ = 1.022 MeV |
| **Reasoning-AI** | The `physics_particles` domain with NL routes — quark compositions, half-lives, decay ages ("25% carbon-14 remaining" → 11,460 years), binding energies, photon energies, de Broglie wavelengths, configurations — every answer re-checked through an independent path |

## What's new in OTD3.2 — the performance + website edition

> **The Velocity Edition.** The renderer learned the oldest performance
> law there is: *never redo work a camera move didn't require.* Rendering
> is now split into `prepare_lit_scene` (the expensive geometry half —
> BVH build, ray-traced shadows, ambient occlusion) and `render_view`
> (the cheap camera half — project, rasterize, encode). The HTTP server
> caches the lit scene per script hash, so an orbit drag ray-traces
> **once** and then flies: measured **9.8× faster** on the chess-set
> scene (994 ms cold → 101 ms per frame). The eight-camera OCTOCAM panel
> shares one lit scene across all eight views and rasterizes them across
> threads. The interactive viewer degrades quality mid-drag and settles
> to full quality when you stop, aborting stale renders in flight. And
> OTD3 grew an official **website** (`website/` — open `index.html`), with
> every number on it measured from the real engine.

| Area | The upgrade |
|---|---|
| **Lit-scene cache** | `prepare_lit_scene` + `render_view` split — the server keys ray-traced lighting by code hash; orbit/zoom pays once, then only the cheap camera-dependent half runs (`docs/11-PERFORMANCE-ARCHITECTURE.md`) |
| **Threaded OCTOCAM** | The 8-camera panel builds the lighting **once** and rasterizes the eight views across threads — the panel now renders faster than a single full-quality view did before |
| **Server caches** | Compile cache and frame cache reworked: lookups under a short-lived lock, compilation and rendering outside it — one slow request can no longer stall the pool |
| **Interactive viewer** | `assets/app.js` rewritten: in-flight renders abort when the camera moves on, quality degrades during a drag and settles back when it stops |
| **Website** | `website/` — the official landing page (hero, physics, language, self-make, video, docs sections); every figure on it verified against engine output |
| **VM guard** | `OP_DOT3_F64` asm kernel gains a debug_assert bounds guard — a compiler bug in register allocation now fails loudly instead of reading past the register file |

> **The base, in one paragraph.** OTD3 "UNIVERSE" took OTD 2.3's Solidity
> base to a full physics playground: the complete energy taxonomy (2
> categories, 9 forms), thermodynamics (`temperature: 800` melts the
> iron), magnetism (dipoles, Faraday, Earth's field), sound and light
> (Doppler, Snell, Wien), relativity (E = mc², time dilation), native
> PerceptAudio hearing with metric ranging, and the eight-camera
> H.264/AVC video studio (`--video out.mp4 --cams 8 --screw nut=6@y:1.75`)
> — the encoder written from nothing, validated bit-exact against
> reference decoders. The nut & bolt demo renders as a real MP4: one
> turn, one pitch — Archimedes' 2300-year-old law, animated by a
> 2020s zero-dependency engine.

## What's new in OTD3.1 — the self-make expansion

> **The Forge Edition: libraries that self make anything.** The real nn.rs
> crate was unreachable and the real Burn framework would drag ~50 dependency
> crates behind it — so OTD built its own, from nothing, keeping the P0000
> zero-dependency guarantee. The result: a neural network library
> (`otd::nn`), a complete deep-learning framework in Burn's image
> (`otd::burn`: Backend → Tensor → Autodiff → Module → Optimizer → Learner),
> a statistics & probability library (`simulate: stats`), an astronomy
> library (`simulate: orbit`), and the **program synthesizer**
> (`otd --make "a titanium ball dropping into water from 1m"`) that writes
> complete .otd programs from plain-language goals and compiles its own
> output until the parser reports zero errors. `simulate: learn` trains a
> net on the scene's own freefall law — AI rediscovering t = √(2h/g) from
> samples alone, verified to <1% by the closed form. `otd --train` runs the
> full OTD-Burn learner end to end.

| Area | The upgrade |
|---|---|
| **`otd::nn` — nn.rs, self-made** | A single-file zero-dependency neural network library in the nn.rs spirit: `Tensor` (shape-checked, blocked 4-wide matmul), `Linear` with fused forward/backward, ReLU/GELU/**CELU**/sigmoid/tanh/softmax, `Adam`/`AdamW`/`Sgd`, mini-batch trainer with seeded shuffling — every op parity-tested |
| **`otd::burn` — OTD-Burn** | The Burn architecture, rebuilt: `Backend` trait + `NdArray` CPU backend + `Autodiff<B>` wrapper backend, generic `Tensor<B>`, reverse-mode tape autodiff (`Var` — **gradcheck'd against finite differences on every op**), `Module` trait with save/load checkpoints, `Sequential`/`Linear`/`Embedding`/`Dropout`, SGD/Adam/AdamW + LR schedules, datasets with deterministic splits, and the `Learner` loop with metrics and early stopping |
| **`simulate: learn`** | The language-level hook: trains a GELU MLP (AdamW) on t(h) = √(2h/g) data generated by OTD's own kinematics, reports the loss curve, probes three heights against the closed form, and ties the result to the highest part in the scene |
| **`simulate: stats`** | The scene as a dataset: mean/median/σ/skewness/kurtosis of mass/volume/density, Pearson correlation mass↔volume, least-squares line of best fit, the Central Limit Theorem demonstrated live on the scene's own parts, and a z-test of mean density vs water |
| **`simulate: orbit`** | Kepler's three laws as code: the planetary table checked against T² = 4π²a³/GM⊙, Kepler's equation solved by Newton iteration, vis-viva, escape velocity, Hohmann transfers, Schwarzschild radii, Stefan–Boltzmann + Wien for stars, the distance modulus and Hubble's law — plus this scene's highest part read as a satellite |
| **`otd --make "goal"`** | **The synthesizer** — scans the goal (materials with synonyms, shapes, sizes with units, counts, heights, action verbs), plans a template, emits idiomatic .otd with physics comments, then compiles its own output and repairs it (safe sizes, fallback materials) until zero errors. Nut/bolt goals get the thread machine; video goals get the exact `--cams 8 --screw` command line |
| **`otd --train`** | The OTD-Burn showcase: experiment 1 rediscovers freefall from 96 noisy samples (worst probe <1% off the law); experiment 2 is the XOR sanity check (loss ~1e-13) |
| **New examples** | `selfmake-tour.otd` (six materials → drop, stats, orbit, learn in one scene) and `ai-lab.otd` (five densities, one law) — both in `--dump-examples` |

## What's new in OTD3

| Area | The upgrade |
|---|---|
| **Energy taxonomy** | `simulate: energy` — the full ledger: KINETIC (mechanical, thermal, radiant, electrical, sound) × POTENTIAL (chemical, gravitational, nuclear, elastic), each with its law: PE = mgh, E = mc², U = ½kx², Stefan–Boltzmann… |
| **Thermodynamics** | `temperature: 350K` / `72F` + `simulate: heat` — phase changes against real melting/boiling points, thermal expansion (bridge grows 48 mm on a summer day), the four laws narrated |
| **Magnetism** | `simulate: magnet` — ferromagnetic/paramagnetic/diamagnetic classification, dipole 1/r⁴ forces, Faraday EMF, Curie points, compass field at the scene centre |
| **Sound & light** | `simulate: sound` + `simulate: light` — speed of sound per material (iron 5960 m/s), Doppler, echo ranging, Snell refraction, Wien's law (the Sun peaks at 501 nm — green!) |
| **Time & motion** | `simulate: time` — free-fall, pendulum (1 m ≈ 1 s), and relativity: time dilation, E = mc² for the whole scene, the velocity-addition speed limit |
| **PerceptAudio native** | The Rust/WASM voice engine merged in as `otd::audio` — VAD, YIN pitch, speaker clustering, Wiener denoise, spectrograms, WAV I/O + full-spectrum band analysis (infrasound → ultrasound) and metric distance (inverse-square, echo timing, band anchors) |
| **AVC video studio** | `--video out.mp4` — a real Baseline H.264 encoder (I_PCM macroblocks: bit-exact, lossless, plays everywhere) + a hand-written MP4 (ISO-BMFF) muxer. `--cams 8` renders the eight-camera preview panel |
| **The screw** | `thread(...)` geometry primitive (ISO V-thread), `--screw nut=6@y:1.75` animation (rotation and translation coupled — one pitch per turn), M-series thread table, mechanical advantage law |
| **New materials** | uranium, plutonium, thorium (nuclear energy!), lithium (battery metal), coal, gasoline |
| **Reasoning-AI** | Four new verified domains — magnetism (F = BIL, Faraday, solenoids), waves (v = fλ, Doppler, Snell, thin lens), thermo (Q = mcΔT, PV = nRT, Stefan–Boltzmann), relativity (γ, time dilation, E = mc²) — plus the nine-form energy taxonomy, all NL-routed |
| **Bug fixes** | The animation phase bug (frames now advance cumulatively: frame f shows f/(n−1) of the motion — the 2.3 `--frames` rendered identical phases!) |

# OTD 2.3 "SOLIDITY" — the base OTD3 builds on

> **The Solidity Edition.** Everything 2.2 shipped (mechanical parts,
> steam engine, 118 × 50 periodic table) plus the three laws of the new ask:
> **nothing floats unless physics says so** (things fall and LAND now),
> **nothing can ever insert into anything else** (real solidity, verified to
> < 0.05 mm), and **chemistry is real** — gases mix by diffusion into one
> phase, liquids mix or refuse each other by polarity and stack into
> density-sorted layers, and reactive pairs REACT with balanced,
> independently verified equations. The project now ships with two
> pre-added intelligence toolchains: a verification-gated **reasoning
> engine** (`otd --ai "…"` — every answer checked, or it abstains) and the
> **AVC stereo perception engine** (`otd --perceive scene.otd` — the engine
> literally SEES its own scene and reports objects, distances and relations
> with honest sigmas).

> Write a sentence, hold a thing. **8 lines of text → a real, watertight,
> physics-checked 3D model.** A 12-year-old can learn it in 5 minutes —
> and now the language can *decide*, *repeat*, and *check itself*.

**OTD 2.2** builds on the 2.1 syntax-expansion generation of the pure-Rust +
handwritten-assembly, **zero external libraries** engine — a three-tier
machine inspired by the OTDL v5 architecture: the friendly **Lite**
language you type (now with `if / for / while`, compound assignment,
comparisons and logic, ranges, indexing, block templates and `assert`), a
**Deep** computational core (half-edge topology, Discrete Exterior
Calculus, SDF blending, XPBD physics), and an **ASM** tier where your
arithmetic compiles to real 32-byte vector bytecode. Still one binary, no
runtime, no JavaScript 3D engine — the browser stays a dumb terminal
showing server-rendered pixels. **Every 1.0 and 2.0 file still compiles.**

```otd
scene "Coffee Cup"
cup = cylinder(top: 40mm, bottom: 30mm, height: 100mm) - hollow(wall: 3mm)
handle = torus(radius: 20mm, tube: 6mm) rotate (90deg, 0, 0) at (35mm, 25mm, 0)
add cup, handle
material cup: ceramic
color cup: ivory
ask "mass?"
```
→ renders, weighs **220.7 g** (91.9 cm³ × 2.40 g/cm³, measured from the mesh),
and tells you it sinks — with the math shown.

## Quickstart

```bash
cargo run --release            # serve the viewer at http://127.0.0.1:6830
cargo run --release -- --vm-dump examples/gear-system.otd   # see the bytecode
cargo test                     # 160+ math/physics/byte-level tests
cargo run --release -- --check examples/cup.otd     # stats + errors in terminal
cargo run --release -- --png examples/cup.otd cup.png --view iso
cargo run --release -- --export examples/cup.otd cup.stl --fmt stl
```

### Self-make quickstart — AI that writes and learns OTD

```bash
# the synthesizer: a sentence becomes a verified .otd program
otd --make "a titanium ball dropping into water from 1m"
otd --make "two balls racing to the ground, lead and pine" --out race.otd
otd --make "nut and bolt screwing animation in brass and steel"

# the trainer: OTD-Burn learns OTD's own gravity law, then XOR
otd --train

# or from inside the language itself — a scene that learns its own physics:
otd --check examples/selfmake-tour.otd      # drop, stats, orbit, learn
```

### Mechanics quickstart — parts, physics, animation, elements

```bash
# the steam engine + 5 views (flywheel really rotates between shots)
otd --png examples/steam-engine.otd shot1.png
otd --png examples/steam-engine.otd shot4.png --spin flywheel=90@x
otd --png examples/steam-engine.otd shot5.png --spin flywheel=210@x --turn 35

# motor animation: 8 frames of a full rotation (motor "connected to electricity")
otd --png library/parts/motor_assembly.otd frame.png --spin rotor --frames 8

# transparency + splash physics
otd --png library/materials/glass.otd glass.png
otd --check library/parts/nut_m8.otd          # real steel mass + drop physics
otd --png library/elements/026-iron/49-density-compare.otd fe-vs-water.png


No dependencies to install — the only build requirement is the Rust
toolchain. `[dependencies]` in `Cargo.toml` is **empty, forever**.

## New in 2.3 — solidity, environments, chemistry, perception

```otd
scene "Tower"
base = cube 6cm material: steel at (0, 18cm, 0)
top  = cube 6cm material: oak  at (0, 32cm, 0)
simulate: settle     # they FALL and LAND on each other — zero penetration
simulate: solidity   # audit: "SOLID — 2 parts, zero interpenetrations"
```

- **Settle + real solidity** — `simulate: settle` drops every body under the
  current gravity until it rests on the ground or another part; contacts
  separate along the shallowest axis with a mass-weighted impulse, so one
  item can never insert into another. `--settle` does it before any render.
- **Environments** — `environment: water` (or vacuum / air / oil / density N):
  buoyancy + drag — wood and ice rise to the surface, steel sinks slowly.
- **Gases mix** — `simulate: gas` in a chamber: light species rise, heavy
  sink, diffusion merges the rest into one phase with a reported composition
  and density. N2 + O2 → something that is basically air (ρ 1.29 kg/m³).
- **Liquids mix by chemistry** — `mix: water + oil` (refuse — polar vs
  non-polar), `mix: water + ethanol` (one phase, both polar),
  `simulate: mix` stacks immiscible liquids into clean layers, mercury at
  the bottom. Reactive pairs don't mix — they burn: `2 H2 + O2 -> 2 H2O`.
- **The engine understands real things** —
  `otd --ai "why does a ship float?"` asks the vendored reasoning engine
  (VERIFIED or ABSTAINED, never a guess), and
  `otd --perceive bench.otd` renders a stereo pair and lets the vendored AVC
  engine see the scene: "box #1 centre (0.5, 0.10, 1.83) m, left_of #2" —
  metric depths with honest sigmas. Both ship pre-added under `vendor/`,
  built once by `run.sh`.

## What's inside

| Layer | Files | Written from scratch |
|---|---|---|
| Language | `src/lang/` | lexer (unit-glued numbers), recursive-descent parser, 58-keyword registry, friendly errors with "did you mean?" |
| **VM (2.0)** | `src/vm/` | OTD-ASM: 32-byte vector ISA, expression→bytecode compiler, 256-register interpreter with asm! kernels, `--vm-dump` disassembler |
| **Deep tier (2.0)** | `src/geo/` | half-edge topology, DEC cotan-Laplacian `smooth`, Loop `subdiv`, BVH + SDF `blend`, XPBD `rope`, exact inertia tensor |
| Geometry | `src/geo/` | 12 primitives, BSP CSG (union/subtract/intersect), analytic hollow, extrude/revolve/sweep/loft/tube/helix, 3D text, fBm terrain, marching-tetrahedra metaballs, STL/OBJ import |
| Measurement | `src/geo/measure.rs` | divergence-theorem volume/centroid/area, rasterized cross-section slices |
| Materials | `src/world/materials.rs` | 27 real materials (density, melting point, conductivity, friction, bounce, magnetism, strength) + 147 named colors |
| Physics | `src/world/physics.rs`, `ask.rs` | mass = ΣV·ρ, Archimedes float, drop energy, collapse stress, the `ask` engine |
| Renderer | `src/render/` | SIMD software rasterizer: z-buffer, **GGX microfacet light, ray-traced soft shadows, baked AO**, adaptive grid, 2× SSAA, own DEFLATE/PNG/CRC-32 |
| Assembly | `src/simd/` | raw `asm!` kernels (`rep stosd`, SSE hsum, SSE dot) + SSE2 intrinsics, all parity-tested |
| Server | `src/net/` | HTTP/1.1 (threads, keep-alive, 100-continue), own JSON, own Base64, compile + **lit-scene** + frame caches — orbit/zoom ray-traces once, then flies |
| Viewer | `assets/` | vanilla HTML/JS/CSS — editor with syntax highlight, orbit/zoom with degrade-then-settle quality, in-flight render abort, ask console, lessons, exports |
| Website | `website/` | the official landing page — hero, physics, language, self-make (FORGE), video sections; static HTML/CSS/JS, every figure verified against engine output |

## The language in 30 seconds

- **Shapes:** `sphere cube cylinder cone torus pyramid prism capsule wedge plane tube helix` + `extrude revolve sweep loft text terrain metaball import`
- **Booleans:** `a + b` fuse, `a - b` cut, `a & b` overlap, `hollow(wall: 3mm)` shell
- **Place:** `at (x, y, z)` (rest point), `rotate`, `scale`, `mirror x`
- **Patterns:** `repeat(n: 4, step: (2cm, 0, 0))`, `grid(nx: 3, nz: 3)`, `ring(n: 8, radius: 5cm)` — magic `i`, `j`, `a`
- **Parts:** `define wheel(r) = …` then `use wheel(r: 3cm)` — or a multi-step `define … end` block (2.1)
- **Science:** `material: gold`, `ask "mass?"`, `simulate: drop / float / collapse`, `gravity: moon`
- **2.0 deep words:** `smooth(n: 4)` melts corners (real DEC math), `subdiv(n: 1)` makes fairer surfaces, `blend(a, b, gap: 8mm)` solders solids, `rope(from:, to:, sag:)` hangs a real catenary — `ask "cable sag?"` / `"inertia?"` / `"watertight?"` answer with measured numbers
- **2.1 syntax:** `if / else / end` (one-liner `if x > 5cm: cube 1cm`), `for i = 1 to 10 by 2` and `for r in [1cm, 2cm]`, `while`, `break` / `continue`, `x += 1cm`, `^` `%` `mod`, comparisons `< > <= >= == !=` + `is / is not / in`, `&&`/`and` `||`/`or` `!`/`not` (short-circuit), `true / false`, ranges `1..12`, indexing `xs[0]` `xs[-1]`, slicing `xs[1..3]`, `assert x > 0, "message"`, `print "r = {r}"` — see `docs/03-GRAMMAR.md`
- **Functions (25):** `cos sin tan sqrt abs min max round floor ceil pow log ln exp sign hypot atan atan2 asin acos lerp clamp count sum avg`
- **Units:** `mm cm m km in ft yd um deg rad`; bare numbers are centimeters
- 75 keywords + 25 functions — the 2.1 expansion raised the 60-word budget to 100 (the cheat sheet grew a second side).

## Docs

- `library/00-INDEX.md` — **master index of the 1,695-program content library** (12 categories, every file validated)
- `docs/00-ARCHITECTURE.md` — the full system design (12 layers)
- `docs/01-ROADMAP-10000-PHASES.md` — the 10,000-phase plan (this build: P0000–P0649)
- `docs/02-KEYWORDS.md` … `08-COMPATIBILITY.md` — the complete language spec
- `docs/09-TEST-REPORT.md` — math & physics validation with measured numbers
- `docs/07-CHEATSHEET.md` — one-page beginner card
- `docs/10-LESSONS.md` — 23 lessons from "hello cube" to decide-and-repeat
- `examples/` — 15 curated models (cup, table, gears, robot arm, house, DNA, chess, lamp, bridge, solar system + 2.0's deep-blob, rope-bridge, smooth-vase + 2.1's staircase, orbit-tower)
- `library/` — **1,695 ready-to-run programs**: materials (brick, wood, metal, glass, water…), primitives, patterns, architecture, nature, furniture, machines, art, games, physics demos, 46+ lessons, showcase scenes

## The content library — 1,695 programs, all validated

`library/` is a full content corpus organized in 12 categories, each with its own
README, and a master index at `library/00-INDEX.md`. Flagships:

```bash
otd --png library/materials/brick.otd brick.png        # the classic brick
otd --png library/materials/water.otd water.png        # float simulation with water plane
otd --check library/physics/float-basic.otd            # wood floats, iron sinks
otd --png library/showcase/scene-harbor.otd harbor.png # a full scene
```

| Category | Files | | Category | Files |
|---|---|---|---|---|
| materials | 263 | | machines | 137 |
| primitives | 293 | | art | 115 |
| patterns | 149 | | games | 83 |
| architecture | 160 | | physics | 87 |
| nature | 141 | | lessons | 92 |
| furniture | 118 | | showcase | 57 |

Quality gates every file passes:
1. `otd --check` — full parse + evaluate + world build (run over all 1,695 files, 0 errors)
2. `cargo test library_parses` — the suite walks `library/**/*.otd` on every test run
3. Sample renders spot-checked as PNGs

## Constraints we keep, forever

1. **Zero external crates.** If a phase needs a library, the phase is redesigned — not the constraint.
2. **Rust + assembly only** on the backend; the browser stays a dumb terminal.
3. **< 60 keywords.** New capabilities are new words only if the budget allows; otherwise they are parameters.
4. **Old `.otd` files never break** (forward-compatibility policy, doc 08).

MIT license. 1.0 delivered P0000–P0649; 2.0 "DEEP" delivers P0650–P1299 — see you at P1300.
