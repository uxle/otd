# Changelog

## 4.0.0 "DYNAMICS" — the dynamics + assembly edition: aerodynamics, fluid dynamics, electrodynamics, stellar dynamics, rigid body dynamics, multi-part assembly, magnetize, strict-overlap

> The user said: "you made good but not all science here." OTD3 had
> thermodynamics, magnetism, waves, energy — the static sciences. OTD4
> adds the five that move: **aerodynamics** (wings, drag, terminal
> velocity, Reynolds, Mach), **fluid dynamics** (Bernoulli, Poiseuille,
> Stokes, hydrostatic pressure), **electrodynamics** (Ohm, Kirchhoff,
> RC/RL/LC, Maxwell's identity c = 1/√(μ₀ε₀)), **stellar dynamics**
> (N-body, virial theorem, Jeans length, dynamical time), and **rigid
> body dynamics** (inertia tensor, angular momentum, gyroscopic
> precession, the tennis racket theorem).
>
> The user also reported silent-wrongness bugs that bit hard when an AI
> uses the language without a human checking each step. OTD4 fixes every
> one of them — and adds **strict overlap mode** that turns
> "it compiled" into "it compiled AND it's geometrically sane":
>
> - `strict: overlap` + `overlap: check` — solid-solid interpenetration
>   now FAILS compilation with a corrective error (axis, depth, fix
>   suggestion) instead of just warning. The "shapes overlapping" bug
>   the user reported is now impossible to ship by accident.
> - `hollow()` now logs an INFO line every time `open:` is not
>   explicit, so the default (open top) is never a surprise.
> - `group()` re-parents its source parts (hides them as separate
>   entries) so mass is no longer double-counted. The old 7.0g → 14.0g
>   bug is gone.
> - `rotate` now logs its pivot point every time (the bbox center,
>   unless `pivot (x,y,z)` or `pivot: origin` is given). The
>   "stator-tooth petals instead of radial teeth" bug is now
>   impossible to ship without seeing the pivot in the console.
> - `print "{name}"` with an unknown variable now WARNS instead of
>   silently leaving the literal text. The AI sees the typo.

> Multi-part design workflow: each part in its own file, then a main
> file `include`s them and arranges them with `at (...)`. Four new
> keywords: `include`, `magnetize`, `strict`, `overlap` — bringing the
> total to 82 (still under the 100 budget).

### New physics modules (P2300 series)

- **`src/world/aerodynamics.rs` (P2300)** — the science of air moving
  past things: continuity (A·v = const), Bernoulli (P + ½ρv² + ρgh =
  const), lift (L = ½ρv²AC_L), drag (D = ½ρv²AC_D), Reynolds (Re =
  ρvL/μ), Mach (M = v/c), terminal velocity (v_t = √(2mg/ρAC_D)).
  `simulate: aero` walks every part, infers its shape class (sphere /
  box / rod / streamlined), and reports drag, terminal velocity, and
  (for wings) lift, L/W ratio, and stall speed. The ISA standard
  atmosphere gives air density at any altitude; Everest's air is 40%
  of sea level.
- **`src/world/fluiddynamics.rs` (P2310)** — how liquids actually
  move: Poiseuille (Q = πr⁴ΔP/8μL — the r⁴ law), Stokes drag
  (F = 6πμrv), hydrostatic pressure (P = ρgh), capillary rise
  (h = 2γcosθ/ρgr), Reynolds (the laminar/turbulent switch). 14
  viscosity values from water (1 mPa·s) to glycerin (1.4 Pa·s — 1400×
  water). `simulate: fluid` walks every liquid/gas part and reports
  its column pressure, flow regime, and Poiseuille throughput.
- **`src/world/electrodynamics.rs` (P2320)** — currents, fields, and
  circuits in time: Ohm (V = IR), power (P = VI = I²R), capacitor
  energy (U = ½CV²), inductor energy (U = ½LI²), RC/RL time constants,
  LC resonance (ω = 1/√(LC)), and the Maxwell identity
  c = 1/√(μ₀ε₀) = 299,792,458 m/s (light IS electromagnetism). 14
  resistivity values from silver (1.59e-8 Ω·m) to carbon (3.5e-5).
  `simulate: electro` walks every part — conductors get Ohm's law at
  12 V, insulators get the parallel-plate capacitance and charge in
  electrons. Johnson noise (the floor every amplifier hears) is
  reported at the scene temperature.
- **`src/world/stellardynamics.rs` (P2330)** — many-body gravity: the
  N-body pairwise force, the virial theorem (2K + U = 0), the Jeans
  length (the size a cloud must exceed to collapse into stars), the
  Jeans mass, the dynamical time (the natural clock of the system),
  escape velocity. Sound speed in an ideal gas (c_s = √(γkT/μm_H)).
  `simulate: stellar` reads each part as a "star" with mass = its real
  mass, computes pairwise forces, reports the virial energy (bound vs
  unbound), the Jeans length for the scene's density, and the
  dynamical time. Earth's orbital speed (29.8 km/s) and a globular
  cluster's virial mass are tested.
- **`src/world/rigidbody.rs` (P2340)** — rotation, inertia,
  gyroscopes: moment of inertia (I = Σmr²) for solid/hollow spheres,
  cylinders, rods, slabs; parallel axis theorem (I = I_cm + Md²);
  angular momentum (L = Iω); rotational energy (E = ½Iω²); torque
  (τ = r×F); gyroscopic precession (Ω_p = τ/Iω) — verified against
  Earth's 26,000-year axial precession. The tennis racket theorem
  (rotation about the middle-inertia axis is unstable) is taught.
  `simulate: rigid` walks every part, computes its principal moment,
  angular momentum, and the gravity-arm precession rate at a notional
  1 rad/s spin.

### New keywords (4 → 82 total)

- **`include "parts/wheel.otd" at (x, y, z)`** — the multi-part
  design workflow. Each part is designed in its own file (run
  `otd --check parts/wheel.otd` to verify); the main file `include`s
  them and arranges them with `at (...)`. New entries are auto-named
  after the file (e.g., `wheel_1`, `wheel_2`) when there's a clash.
  See `examples/car-assembly.otd` and `examples/parts/`.
- **`magnetize: <name> [moment: <expr>]`** — mark a part as a
  permanent magnet. The moment defaults to neodymium-grade
  (8×10⁵ A/m × volume); pass `moment: 0.5` for a custom value.
  Overrides the material's intrinsic magnetism class. `simulate:
  magnet` then tags the part with `[MAGNETIZED]` in the survey.
- **`strict: overlap` | `strict: all` | `strict: off`** — toggle
  strict mode. When ON, solid-solid interpenetration FAILS
  compilation (not just a warning). The silent-wrongness fix the user
  asked for: an AI can no longer ship overlapping geometry without
  seeing an error.
- **`overlap` or `overlap: check`** — the explicit overlap audit.
  Reports every interpenetrating pair with the penetration depth and
  the shallowest escape axis (X, Y, or Z), plus a concrete fix
  suggestion ("move B along Y by 2.5 mm, or fuse with `add A, B`").

### Silent-wrongness fixes

- **`hollow()` default pivot hint** — `eval_hollow` now emits an INFO
  line every time `open:` is not explicit, telling the user the
  default is "open top" and how to override (`open: none` for a sealed
  shell, `open: bottom` to breach the base).
- **`group()` re-parents** — when an arg is an Ident referring to an
  existing entry, that entry is now marked hidden so it doesn't
  double-count mass. The old 7.0g → 14.0g bug is gone. An INFO line
  tells the user which parts were re-parented.
- **`rotate` pivot logged** — every `rotate` (without `pivot:`) now
  emits an INFO line naming the pivot point (the bbox center) and
  showing how to override (`pivot (0, 0, 0)` for the origin,
  `pivot: center` to be explicit). The new `Mod::RotateWithPivot`
  variant accepts `rotate (angles) pivot (x, y, z)`.
- **`print "{name}"` warns on unknown** — instead of silently leaving
  the literal text, an unrecognized `{expr}` now emits a WARN line
  telling the user/AI the variable is unknown and to define it first
  or fix the typo.
- **material suggestion** — when any part has no material assigned,
  the engine now emits a WARN with the first unmaterialized part's
  name and a list of common choices (steel, aluminum, copper, glass,
  oak, ceramic, water, iron). A second INFO line explains how to set
  a scene-wide default (`material: steel`).
- **color suggestion** — when any part has neither color nor
  material, an INFO line explains that parts without a color inherit
  from their material and shows how to override.

### Error template library (errors.rs)

`src/lang/errors.rs` grew a library of corrective error templates,
each with both a message AND a concrete fix suggestion:

- `err_negative_radius(line, got, unit)` — "did you mean 5mm?"
- `err_negative_length(line, what, got, unit)`
- `err_unknown_material(line, name)` — Levenshtein-suggests from the
  50-material table
- `err_unknown_color(line, name)` — Levenshtein-suggests from the
  147-color CSS/SVG set
- `err_unknown_shape(line, name)` — Levenshtein-suggests from the 14
  primitives
- `err_unknown_unit(line, name)` — Levenshtein-suggests from the 10
  units
- `err_arg_count(line, shape, expected, got, example)`
- `err_missing_arg(line, shape, arg, example)`
- `err_bad_axis(line, got)` — "must be x, y, or z"
- `err_bad_boolean(line, got)` — "use + - & between shapes"
- `err_object_not_found(line, name, candidates)` — Levenshtein
  suggestion
- `err_type_mismatch(line, what, expected, got)`
- `err_out_of_range(line, what, got, min, max, hint)`
- `err_syntax(line, what, example)`
- `err_overlap(line, a, b, depth_mm, axis)` — the strict-mode error
  with the fix suggestion
- `err_file_not_found(line, file, kind)` — for include/import
- `err_deprecated(line, old, new)`
- `err_empty_scene(line, sim)`
- `err_no_material(line, part)`

### Material suggest-by-use

`materials::suggest_by_use(use_case)` returns a (name, reason) pair
for 20+ use cases: "structure" → steel, "lightweight" → aluminum,
"magnet" → iron, "wire" → copper, "heat_sink" → aluminum, "jewelry"
→ gold, "cup" → ceramic, "lubricant" → oil, "nuclear" → uranium,
"hard" → tungsten, "tire" → rubber, etc. Powers the "no material"
suggestion when a part is created without one.

`materials::list_by_property(prop)` lists materials by predicate:
"magnetic", "metal", "transparent", "liquid", "gas", "conductor",
"insulator", "sinks", "floats". For programmatic discovery.

### Tests

New tests in every new module — Earth's orbital speed (29.8 km/s),
Earth's escape velocity (11.2 km/s), the Maxwell identity (c =
299,792,458 m/s), Poiseuille's r⁴ law (halve radius → ×16 flow),
Stokes drag (glycerin 1400× water), capillary rise (water climbs 14
mm in a 1 mm tube), the virial mass of a globular cluster, Earth's
26,000-year axial precession. Every test verifies a real physical
constant or law to the textbook value.

### Examples

- `examples/dynamics-tour.otd` — one scene, all six dynamics domains
  on the same parts (wing, wire, disk, three balls, water pool)
- `examples/car-assembly.otd` — the multi-part workflow: chassis +
  4 wheels + 2 axles, each in its own file under `examples/parts/`,
  assembled with `include:` and audited with `strict: overlap`
- `examples/parts/wheel.otd`, `axle.otd`, `chassis.otd` — the
  individual part files for the car-assembly demo

### Backward compatibility

Every existing OTD3 file still compiles unchanged. The `hollow()`
default remains `Open::Top` (the old behavior); the new info hint is
non-fatal. The `group()` re-parenting is a behavior change but
produces the mass the user originally expected (the bug fix is the
correct behavior). The `rotate` info hint is non-fatal. The
`print "{name}"` warn is non-fatal. `strict: overlap` is opt-in
(default off) so no existing file is broken.

---

## 3.3.0 "QUARK" — the subatomic edition: protons, neutrons, electrons, quarks, gluons, photons, neutrinos

> The user asked for the particles underneath everything. The grams were
> already measured; this edition converts them into an actual particle
> census — and ships the whole Standard Model with it.

- **`src/world/particles.rs` (P2250)** — the Standard Model: all 17
  fundamental particles (6 quarks, 6 leptons, 4 gauge bosons, 1 Higgs)
  with mass, charge, spin, statistics; the quark model of the hadrons
  (proton uud, neutron udd, pions) with charge re-checks; the famous
  accounting (quarks = 9.1 MeV, proton = 938.3 MeV, so 99% of mass is
  gluon field energy); β⁻ decay with its 0.782 MeV Q-value;
  annihilation (1.022 MeV); photon E = hc/λ and p = h/λ; de Broglie and
  Compton wavelengths; the neutrino fluxes. `particles_sim(world)` is
  the briefing, tied to the scene's own matter count and temperature.
- **`src/world/atom.rs` (P2260)** — the atomic layer: all 118 elements
  (Z, symbol, name, dominant isotope, molar mass, notes for the
  radioactive ones); Madelung electron configurations with the 20 known
  exceptions corrected — the tails are complete above the noble-gas
  core, so Au really is [Xe] 4f¹⁴ 5d¹⁰ 6s¹ (a naive fill-to-Z-minus-tail
  leaves 4f at 12 and a phantom 6s²; caught by the question "electron
  configuration of gold", pinned by tests in both codebases);
  `composition()` mapping every OTD material to elements with mass
  fractions (steel, brass, oak-as-cellulose, water, air…);
  Weizsäcker's semi-empirical mass formula (Fe-56 8.76 vs 8.79 MeV/A
  measured, U-238 7.60 vs 7.57 — and He-4's known under-binding
  documented, not hidden); the isotope table (C-14 5,730 y, U-238
  4.47 Gy, Pu-239, Cs-137…); the decay law with `decay_age_s` and
  `activity_bq`; the Rydberg spectrum (Hα 656.5 nm).
- **`scene_census()`** — grams → moles → atoms → protons/neutrons/
  electrons for the whole scene, per element, from the measured part
  masses. A 984 g iron cube answers 1.06×10²⁵ atoms; the charge ledger
  balances to the last electron (+1.57×10⁸ C of proton charge, −the
  same, net exactly 0).
- **`simulate: atom`** — per-material: element, Z, shells (K2 L8 M14 N2),
  the full configuration, binding energy vs the iron-56 peak ("sitting
  at the peak — fusion and fission both lose from here"), the
  nucleus-vs-atom size ratio ("if the nucleus were a 1 cm marble, the
  electrons would be 220 m away"), hydrogen's Rydberg fingerprint when
  water/wood is present, and the scene totals.
- **`simulate: decay`** — the radioactive materials' half-lives, modes
  and chains, **live activity in becquerels from the actual atom count**
  (955 g of U-238 → 11.88 MBq; 1 g → 12.4 kBq, the textbook number,
  asserted in tests), after-100-years fractions, and the trace C-14 in
  every organic part (1.2 per trillion carbon atoms → the
  archaeologist's clock).
- **`simulate: particles`** — the Standard Model briefing: the census of
  the scene, the proton/neutron quark cards with the gluon accounting,
  confinement ("free quarks have never been seen, by anyone, ever"),
  the scene's own thermal photon glow (σT⁴, Wien peak), the neutrino
  fluxes (6.5×10¹⁰ /cm²/s solar + 336 /cm³ relic), β-decay when
  radioactive parts are present, and the full 17-card deck.
- **`particle: <name>`** — new statement, keyword #78: one card from the
  deck, fundamental or composite, with its one-paragraph story.
  `particle: gluon`, `particle: proton`, `particle: neutrino`…
- **Reasoning-AI `physics_particles` domain** — quark compositions,
  particle masses (kg ↔ MeV re-checked through E = mc²), neutron decay,
  annihilation, photon energy/wavelength (round-trip re-checks), de
  Broglie, Weizsäcker binding, half-life table, decay ages, activity,
  shell capacities, Rydberg, electron configurations. NL routes answer:
  "a sample has 25 percent carbon 14 remaining how old is it" →
  VERIFIED: 11,460 years; "binding energy of iron 56" → VERIFIED:
  490.54 MeV; "electron configuration of gold" → VERIFIED with the
  correct 4f¹⁴. New quantity units: nm/eV/keV/MeV/GeV, Bq, percent,
  spelled-out "meters per second". Fixed `_fmt` to keep full precision
  for sub-micron values (a 7.27e-7 m de Broglie wavelength used to
  round to 1e-06).
- **`docs/12-PARTICLES.md`** — philosophy → laws → the numbers that
  test the laws → the language, in the house style.
- **New example** `atom-tour.otd` (iron, copper, uranium, oak — the
  census, the decay, the briefing) + 12 integration tests in
  `tests/subatomic.rs`; keyword budget test moved 77 → 78.

## 3.2.0 "VELOCE" — the performance + website edition

> The user supplied two more archives: a performance patch
> (`otd3-perf-patch.zip`) and the official landing page
> (`otd3-website.zip`). Both are merged in this release. The patch's own
> design document ships as `docs/11-PERFORMANCE-ARCHITECTURE.md`.

- **Renderer split (the big lever)** — `render_world_rgb` was split into
  `prepare_lit_scene` (geometry-dependent: mesh merge, BVH build, creased
  normals, ray-traced shadow/AO) and `render_view_rgb`
  (camera-dependent: clip, project, rasterize, PNG). A frame that only
  moved the camera now skips the expensive half entirely.
  `render_world`/`render_world_rgb` remain as single-shot wrappers, so
  no call site changed.
- **Lit-scene cache in the server** — `POST /api/render` keys the
  ray-traced lighting by the same FNV-1a hash as the compile cache:
  an orbit/zoom drag ray-traces once, then every frame is a cache hit.
  Measured on `chess-set.otd`: **994 ms cold → 101 ms per frame (9.8×)**
  on the interactive path.
- **Lock discipline** — compile and render now happen OUTSIDE the mutex;
  only the cache lookup holds the lock. One slow request can no longer
  stall every other connection's cache access (the server is
  one-thread-per-connection).
- **Threaded OCTOCAM** — the 8-camera panel calls `prepare_lit_scene`
  ONCE and rasterizes the eight cheap views across threads
  (`std::thread::scope`): the panel now renders in less time than a
  single full-quality view took before the split.
- **Interactive viewer** — `assets/app.js` gains in-flight abort
  (`AbortController`), a degraded-quality mode while a drag/zoom is in
  progress, and a settle timer that fires one full-quality render when
  interaction stops. Stale frames in flight are cancelled, not finished.
- **Website** — `website/` (index.html + style.css + script.js): the
  official landing page with hero, physics, language tiers, code demo,
  video/octocam sections, and a new FORGE section for the self-make
  stack (nn.rs, OTD-Burn, the synthesizer, stats & astronomy). Every
  figure on the page was verified against real engine output (the cup
  really weighs 220.7 g).
- **VM safety** — `OP_DOT3_F64` gains a `debug_assert!` bounds guard in
  front of the bare `asm!` dot3 kernel, so a hypothetical register
  allocation bug fails loudly in debug builds instead of silently
  reading past the register file.
- **Docs** — `docs/11-PERFORMANCE-ARCHITECTURE.md`: philosophy → the
  math (cost table, Amdahl calibration) → the measured numbers, all
  checked against the actual `src/` call sites.

## 3.1.0 "FORGE" — the self-make expansion: libraries that make anything

> The user asked for nn.rs, Burn, and much more, "to self make anything".
> The real nn.rs repository was unreachable and vendoring real Burn would
> have broken the zero-dependency guarantee (P0000: `[dependencies]` stays
> empty — Burn drags ~50 crates). So OTD self-made them instead, which is
> the entire point of the edition.

- **`otd::nn` (P2200)** — the nn.rs-style library, single file, zero deps:
  `Tensor` with shape tracking and a cache-blocked 4-wide-unrolled matmul,
  `Linear` with fused forward/backward, activations ReLU/GELU/CELU
  (nn.rs's signature)/sigmoid/tanh/softmax, `Adam`/`AdamW`/`Sgd`,
  `softmax_cross_entropy` and `mse` with analytic gradients, an MLP with
  manual reverse-mode backprop, and a seeded mini-batch trainer. 15 tests:
  matmul parity vs naive, optimizer convergence, XOR, determinism,
  cross-entropy gradient vs finite differences.
- **`otd::burn` (P2210)** — the Burn architecture rebuilt from nothing:
  `Backend` trait with the `NdArray` CPU implementation and the
  `Autodiff<B>` wrapper backend; generic shape-checked `Tensor<B>`; a
  reverse-mode tape (`Var`) with vadd/vsub/vmul/vscale/vmatmul/
  vrelu/vtanh/vsigmoid/vgelu/vsquare/vmean/vmse/vcross_entropy/vdropout/
  vgather/vreshape — **every one gradcheck'd against finite differences**;
  the `Module` trait (`forward`, `params`, `zero_grad`, flat-file
  checkpoints with magic `OTDBURN1`); `Linear`, `Sequential`, `Embedding`,
  `Dropout` (train/eval), activation modules; `Sgd`/`Adam`/`AdamW` with
  `LrScheduler` (constant/linear/cosine); `Supervised` datasets with
  deterministic splits and shuffled batching; the `Learner` (`fit`) with
  loss history, best-epoch tracking, and early stopping. 33 tests.
- **`simulate: learn | nn | brain | neural` (P2200)** — the language hook:
  trains a 2×48 GELU net on the engine's own freefall data and reports how
  close AI came to the law it was never shown, tied to the scene's highest
  part.
- **`simulate: stats | statistics` (P2220)** — statistics & probability:
  descriptive moments, Lanczos log-gamma, binomial/poisson/exponential PMFs
  & PDFs, the normal CDF (Zelen–Severo) and inverse CDF (Acklam + Halley),
  Pearson correlation, least-squares regression, two-sided z-test, Box–Muller
  sampling — and the scene-level report with the CLT demonstrated live.
- **`simulate: orbit | astro | space` (P2230)** — astronomy: the solar
  system table, Kepler III verified across all planets, Kepler's equation
  by Newton iteration, vis-viva, escape velocity, surface gravity,
  Hohmann Δv, Schwarzschild radii, Wien + Stefan–Boltzmann, distance
  modulus, Hubble's law, tides (1/d³). 13 tests against textbook values.
- **`otd::ai::synthesize` (P2240) + `--make`** — the program synthesizer:
  scan → plan → emit → verify (compile, repair, repeat). Recognises 40+
  material words with synonyms, 9 shape families with idiomatic OTD
  constructors, unit-numbered sizes and heights, counts, environments,
  gravities, ten action templates each carrying its own `simulate:` set,
  and emits CLI hints for video/screw goals. Garbage input still yields a
  compiling canonical scene. 11 tests + the tour of 15 demo goals.
- **`otd --train`** — the OTD-Burn demo: freefall rediscovered from 96
  noisy samples to <1%, then XOR to ~1e-13.
- **CLI**: `--make "goal" [--out file.otd]`, `--train`; version banner
  3.1.0 "FORGE".
- **New examples**: `selfmake-tour.otd`, `ai-lab.otd`.
- **Tests**: 329 lib + 11 integration (`tests/selfmade.rs`) — the full
  suite runs in ~2 s.

## 3.0.0 "UNIVERSE" — the science, sound and video edition

- **Energy taxonomy** (`simulate: energy`): the complete ledger — KINETIC
  (mechanical, thermal, radiant, electrical, sound) and POTENTIAL (chemical,
  gravitational, nuclear, elastic), each computed from real geometry and
  material data, each with its law stated.
- **Thermodynamics** (`temperature: 800 | 350K | 72F`, `simulate: heat`):
  phase changes against real melting/boiling points, thermal expansion,
  conduction time-scales, equilibrium temperature (First Law), Stefan–Boltzmann.
- **Magnetism** (`simulate: magnet`): material classification
  (ferro/para/dia), dipole forces (1/r⁴), solenoid/wire fields, Faraday EMF,
  Curie points, net-field compass.
- **Waves** (`simulate: sound`, `simulate: light`): speed of sound per
  material, Doppler, echo ranging, band naming (infrasound→ultrasound),
  Snell refraction, refractive indices, Wien's displacement law.
- **Time & relativity** (`simulate: time`): free-fall, pendulum, Lorentz
  factor, time dilation, E = mc², relativistic velocity addition.
- **New materials**: uranium, plutonium, thorium, lithium, coal, gasoline.
- **`thread()` primitive**: ISO V-thread helical ridge geometry (the bolt).
- **The screw animation** (`--screw part=turns@axis:pitch`): rotation and
  translation coupled — one pitch per turn. Fixed the animation phase bug:
  frames now advance cumulatively (2.3 rendered identical phases).
- **PerceptAudio 2.0 merged as a native module** (`otd::audio`): VAD, YIN
  pitch, speaker clustering, spectral denoise, spectrograms, WAV I/O,
  Catmull-Rom resampling — plus full-spectrum band analysis and metric
  distance estimation (inverse-square ranging, echo timing, band anchors).
  The Next.js studio ships in `web/perceptaudio/`.
- **AVC video studio** (`--video out.mp4`, `--cams 8`, `--fps N`): a real
  Baseline H.264 encoder written from nothing — SPS/PPS/slice syntax
  validated against reference bitstreams, I_PCM macroblocks (bit-exact,
  lossless), emulation prevention, plus a hand-written MP4 (ISO-BMFF) muxer
  (ftyp/moov/trak/minf/stbl/avcC). The eight-camera preview panel renders
  N/NE/E/SE/S/SW/W/NW into a labelled 4×2 composite.
- **Reasoning-AI upgraded**: four new verified domains (magnetism, waves,
  thermo, relativity), the nine-form energy taxonomy, NL routes and unit
  extraction for tesla/hertz/kelvin/celsius/centimetres. 110 test suites
  green.

## 2.3.0 "SOLIDITY" — real solidity, chemistry, and vendored intelligence

The ask: *some items are floating (fix it); nothing can ever insert into
another item — real solidity; gases can mix with each other; liquids can mix
by chemistry; and pre-add the reasoning AI + AVC engines so the project can
understand real things.*

### Added — engine features
- **`simulate: settle` (P1420b) — THE FLOATING FIX + REAL SOLIDITY.** Parts
  used to stay wherever `at (x, y, z)` left them, even in mid-air. Now every
  solid body FALLS under the current gravity until something physical stops
  it — the ground or another part. Contacts push the pair apart along the
  shallowest axis, mass-weighted, with a proper 1-D restitution impulse, so
  **one item can never insert into another item**. Converged rest states are
  re-measured and reported honestly: `max residual penetration 0.009 mm`.
  Impact speeds (√(2gh)), who landed on whom, and legitimate floaters are
  all reported per part.
- **`simulate: solidity`** — the static interpenetration audit: nothing
  moves, every pair of parts is tested (0.02 mm tolerance), overlaps are
  reported with penetration depth and the fix (separate them, or `add` them).
- **`environment:` statement (P1420b)** — the medium the whole scene lives
  in: `air` (default, 1.204 kg/m³), `vacuum`/`space` (0), `water` (997),
  `oil` (920), or a custom `environment: density 1200`. Settle then runs on
  APPARENT gravity g_eff = g·(1 − ρmed/ρbody) with quadratic drag and
  terminal velocity — wood and ice drift UP to the surface, steel sinks
  slowly, in vacuum everything falls the same.
- **`simulate: gas` + ten real gases (P1440)** — hydrogen, helium, methane,
  ammonia, nitrogen, air, oxygen, steam, carbon dioxide, chlorine, all at
  real STP densities. In a chamber gases expand to fill it, light species
  rise and heavy ones sink (buoyancy vs the medium), and DIFFUSION stirs any
  two of them into one uniform phase — the mixture's composition, density and
  mean molar mass are reported. Gases are exempt from the solidity law:
  they interpenetrate BY NATURE — that is what mixing means.
- **`simulate: mix` + `mix: A + B` + five liquids (P1430)** — water, oil,
  mercury, ethanol, acetone, glycerin. `mix:` asks the chemistry verdict with
  the real reason (like dissolves like: polar pairs merge into one phase at a
  volume-weighted density; non-polar oil refuses water; mercury, a liquid
  metal, refuses everything and beads). `simulate: mix` pours every liquid
  into the scene's container and stacks it into clean density-sorted layers —
  mercury at the bottom, oil on top — merging miscible neighbours into one
  blended solution part. Reactive gas pairs do not mix, they REACT with
  mass-conserving stoichiometry: `2 H2 + O2 -> 2 H2O` (286 kJ/mol),
  `CH4 + 2 O2 -> CO2 + 2 H2O` (890 kJ/mol), `H2 + Cl2 -> 2 HCl` — equations
  verified with the vendored reasoning engine. Material table: 31 → 44.
- **`--settle` CLI flag** — settle the scene before `--check` or `--png`
  (render what gravity did to it).

### Added — the vendored intelligence (P1450)
- **`vendor/reasoning-ai/`** — a complete verification-gated reasoning engine
  (19 crates, 595 tests): plain-English questions over math, physics,
  chemistry, biology. Every candidate answer is re-verified by an independent
  non-neural checker; if nothing verifies it ABSTAINS instead of guessing.
- **`vendor/avc/`** — Artificial Visual Cortex v4: a real-time stereo
  perception engine (85 tests) that maintains a metric 3D world model with
  honest uncertainty at every level.
- **`otd --ai "question"`** — ask the reasoning engine; its VERIFIED /
  ABSTAINED verdict passes through untouched. Example:
  `otd --ai "balance the equation CH4 + O2 -> CO2 + H2O"` →
  `VERIFIED: CH4 + 2 O2 -> CO2 + 2 H2O [confidence 100%]`.
- **`otd --perceive scene.otd`** — the engine SEES its own scene through
  AVC: renders a synthetic stereo pair (front camera, `--baseline` apart,
  default 65 mm — human-like), writes calib.json with the renderer's true
  focal length (fx = (H/2)/tan(fov/2), square pixels), and hands the pair to
  AVC's SGM stereo → 3D lift → clustering → Kalman tracking pipeline.
  AVC answers with a metric world model: object centres/extents in metres
  with sigmas, plus spatial relations — left_of, right_of, above, in_front_of,
  closer_than, near. A 0.2 m crate at 3.39 m measures 3.39 m.
- **`run.sh` builds all three toolchains** (engine + both vendors) on first
  run; the vendor trees keep their own tests and docs.

### Changed
- The overlap warning is now tolerance-based (touching boxes after settle
  are resting CONTACT, not a false alarm) and exempts fluids — liquids and
  gases interpenetrate by nature.
- `finalize_stats` re-runs after every geometry-mutating simulation, so
  masses, centres of mass and bounding boxes stay honest after settle/mix/gas.

### Demo scenes (examples/)
- `solidity-stack.otd` — six bodies released in mid-air, settle into a
  zero-penetration stack; the audit then confirms SOLID.
- `environment-water.otd` — oak and ice rise to the surface, concrete and
  steel sink to the tank floor, nothing interpenetrates on the way.
- `gas-mixing.otd` — a glass chamber: N2 + O2 diffuse into one phase
  (ρ_mix ≈ 1.29 kg/m³ — that's air), helium rises, CO2 sinks.
- `liquid-chemistry.otd` — oil, water and mercury pour into one beaker and
  stack into three clean layers while `mix:` explains each verdict.

## 2.2.0 "MECHANICS" — parts, elements, and the steam engine

The ask: *small mechanical items only (no houses/cars), one material per part
with named parts like gear_five, real physics everywhere, animation (motor
spins when connected to electricity), glass that is actually transparent, water
that shakes when something drops — plus a full periodic table: 118 folders ×
50 files of element physics and theory.*

### Added — engine features
- **Material transparency (P1400)** — every material now carries `opacity`.
  Glass (0.34), ice (0.55), water (0.40) and oil (0.55) render see-through:
  the renderer draws opaque parts first, then blends transparent parts
  far-to-near with a z-test-only pass. A steel bolt inside a glass cube stays
  fully visible behind the glass walls.
- **Three new materials** — `water` (997 kg/m³, melts 0 °C), `oil` (920 kg/m³,
  floats on water), `mercury` (13,546 kg/m³, the only liquid metal). Material
  table: 27 → 30.
- **`simulate: splash` (P1410)** — drop things into water: impact speed
  v = √(2gh), buoyancy verdict, bobbing oscillation ω = √(ρw·g·A/m), damped
  settling e^(−0.18ωt), and the ripples that make the water shake,
  c = √(gH) over a 25 cm tank. Liquids are recognized as the medium, not
  projectiles.
- **Animation CLI (P1420)** — spin real parts and render the rotation:
  - `otd render motor.otd f.png --spin rotor=90` — rotate the part named
    `rotor` 90° around its own center (axis default Y, `--spin name=deg@x` for X/Z)
  - `--frames 8` — renders 8 phases of a full `--spin` rotation (frame-000.png …)
  - `--turn 35` — turntable the whole scene (walk around the machine)
  - new `world::anim` module: `spin_named`, `turn_world` — geometry-only
    rotation (mass, volume and physics are untouched), unit-tested.

### Added — content
- **`library/parts/` — 45 single-material mechanical parts**, exactly as
  requested: `bolt_m6…m16`, `nut_m6…m16`, `washer_m6…m12`, the numbered gear
  family `gear_five … gear_twelve` (5–12 teeth), `gear_pair_five_six`,
  `gear_train_five_to_twelve`, `piston_v2`, `piston_ring`, `piston_assembly`,
  `connecting_rod`, `crankshaft`, `flywheel`, `pulley`, `bearing_bronze`,
  `spring_coil`, `axle_shaft`, `pipe_flange`, `handwheel`, `lever_arm`,
  `shaft_coupling`, `cam_plate`, `fan_rotor`, `motor_rotor`, `motor_assembly`
  (with the spin-to-electricity animation recipe), `rivet`, `wedge_key`,
  `chain_link`. Every part is ONE material with a name and live physics
  (`ask "mass?"`, `simulate: drop/float/collapse`).
- **`library/elements/` — the complete periodic table: 118 folders × 51 files
  = 6,018 files.** Each element folder holds 30 theory/data documents
  (identity, atomic structure, electron configuration, physical/thermal/
  electrical/mechanical/magnetic/optical properties, nuclear properties,
  isotopes, chemical behaviour, compounds, crystal structure, phase notes,
  reactions with water/acids/air, alloys, industry, biology, environment,
  history, etymology, abundance, production, safety, storage, fun facts,
  quantum theory) and 20 runnable 3D samples (sample cube/sphere/cylinder,
  ingot, foil, wire coil, granules, powder, crystal lattice, electron shells,
  Bohr model, periodic-position marker, drop/float/stack/splash tests, sample
  vial, display card, TRUE-SCALE molar-volume cube, equal-mass density
  comparison vs water, collection item). All values are real literature data;
  all 2,360 element `.otd` files pass `otd --check`.
- **`examples/steam-engine.otd`** — a horizontal mill engine: steel boiler with
  copper bands, brass chimney/dome/whistle, firebox, cast-iron cylinder block,
  piston rod + bronze crosshead, crank disc + pin, stainless connecting rod,
  iron flywheel with 8 spokes (one named part — spin it with
  `--spin flywheel=90@x`). Five official renders shipped in `screenshots-library/steam-engine/`.
- **`tests/mechanics_structure.rs`** — guards 118 folders × 51 files, the
  parts library by name, the steam engine, and the removal of off-focus content.

### Removed — off-focus content (per the brief: small items only)
- `library/architecture`, `library/furniture`, `library/games`, `library/art`,
  `library/nature`, `library/showcase`, `library/patterns` (≈ 850 files)
- vehicle-* and drone/rover/boxy machine families (complete cars, planes,
  trains, tanks, sailboats — the user wants parts, not vehicles)
- examples: house, bridge, rope-bridge, solar-system, orbit-tower, deep-blob,
  lamp, smooth-vase, dna-helix (kept: cup, table, gear-system, robot-arm,
  chess-set, staircase, steam-engine)

### Fixed
- **B15 (high) — `material:` vanished on CSG cuts.** In `a - tool material: wood`
  the trailing material bound to the cutting TOOL and disappeared: the flagship
  wooden-boat example rendered plastic and *sank*. Finish mods (material/color)
  are now hoisted off the tool and painted onto the RESULT of the cut; the
  hollow-sugar path also works through tool modifiers. The boat floats again
  (700 kg/m³, 101 g — verified).
- **B16 (high) — every metal rendered almost black.** Metals have kd = 0 in
  Cook-Torrance, but this rig has no environment map and ambient/fill light
  only ever reached the diffuse term — the whole metal family rendered as black
  silhouettes with one highlight (found while shooting the steam engine;
  confirmed pre-existing in 2.1.1 gallery renders). Metals now receive a
  sky-tinted environment reflection built from the same ambient/fill energies:
  iron is grey, brass golden, copper orange, gold yellow, steel bright.

### Tests
- 268 → **280 tests, 0 failures** (new: transparency classification, water/
  oil/mercury physics, splash sims ×3, animation ×3, mechanics structure ×4).
- `cargo test` now walks **3,205 library programs** (845 library + 2,360
  element samples) on every run.

## 2.1.1 "LIBRARY" — the content generation

The ask: *enhance the project as much as possible — restructure it and grow it
past a thousand files, with pre-added content like brick.otd and water.otd.*

### Added — content
- **`library/` — 1,695 validated `.otd` programs in 12 categories** (materials,
  primitives, patterns, architecture, nature, furniture, machines, art, games,
  physics, lessons, showcase), each category with a README and a master index
  at `library/00-INDEX.md`. Flagships: `brick.otd`, `water.otd`, `wood.otd`,
  `glass.otd`, plus per-metal demos and 46 numbered lessons.
- **`tests/library_parses.rs`** — cargo test walks `library/**/*.otd` on every
  run; suite totals 268 tests, 0 failures.
- **`screenshots-library/`** — 18 sample renders proving the corpus renders.
- **`ENHANCEMENT-ROADMAP.md`** — what was enhanced + the full prioritized backlog.

### Fixed (workarounds applied; engine fixes queued)
- 7 documentation/implementation drift bugs found by mass-generation (B8–B14):
  metaball 4-tuple docs, sweep spec form, loft spec form, grid spacing scalar,
  rebeccapurple color, loop-variable unit-tag loss with trig, `use`-on-assignment.
  See ENHANCEMENT-ROADMAP.md Part A.2.

## 2.1.0 "SYNTAX" — the language grows up
## 2.1.0 "SYNTAX" — the language grows up

The ask: *add as much syntax as possible, without breaking a single old
file*. 17 new keywords (75 total), 12 new operators, 17 new math functions
(25 total), 4 new units (10 total), 7 new statement forms, block templates,
ranges, indexing, interpolation — and the whole 1.0 + 2.0 corpus still
compiles byte-identically. The suite grew 170 → 267 tests.

### Added — statements
- **`if / else if / else / end`** — with one-liner `if cond: stmt else: stmt`
- **`for i = 1 to 10 [by 2]`** — inclusive; numbers, lengths or angles;
  auto-reverse countdown; loop letters are loop-local
- **`for x in things`** — walks lists (shapes allowed), ranges, positions, words
- **`while cond … end`** — 100,000-pass cap, runaway-friendly error
- **`break` / `continue`** — innermost loop; friendly errors outside loops
- **`x += e` / `-=`, `*=`, `/=`** — compound assignment; `+=` fuses objects
- **`define name(params) … end`** — multi-statement templates: private scope,
  locals never leak, the last shape line is the value
- **`assert cond, "message"`** — loud failure with your words

### Added — operators & data
- `^` power (right-assoc, plain numbers), `%` / `mod` remainder
- comparisons `< > <= >= == !=` + the words `is` / `is not`; unit promotion
- logic `&&`/`and`, `||`/`or`, `!`/`not` — short-circuit
- membership `x in [list | word | 1..10]`
- `true` / `false` literals; ranges `1..12`; indexing `xs[0]`, `xs[-1]`;
  slicing `xs[1..3]`; positions and words index too
- `print "radius {r}"` interpolation (lengths as mm, angles as deg)
- `;` statement separators; `1.5e3` scientific notation; `#[ … ]#` block
  comments
- units `km yd um rad` (10 total)

### Added — functions (8 → 25)
`floor ceil pow log ln exp sign hypot atan atan2 asin acos lerp clamp
count sum avg` — inverse trig answers in degrees, `lerp`/`clamp` carry
units, `count`/`sum`/`avg` walk lists and ranges.

### Fixed — found by the 2.1 syntax audit
- **`3 in [1, 2, 3]` lexed as 3 inches** (beginner unit-gluing swallowed
  the membership operator). Spaced `in` before a value is now the operator;
  attached `3in` stays inches.
- **`..` was not parseable as a value** — ranges are now general expressions
  (`r = 1..12`, `sum(1..100)`, `x in 1..10`, `xs[1..3]`).
- **the VM answered `inf` for `1 / 0`** where the tree-walk errors — the
  OTD-ASM divide now errors with the same words (parity fuzz updated).
- **VM errors were returned but never recorded** — `x = <vm error>`
  vanished silently; every VM error is recorded before it propagates.
- **loop bodies recycle AST addresses** — the VM program cache could serve
  a stale entry (a `cube(...)` evaluating to a number); the cache now
  clears per iteration (pattern bodies keep theirs).
- assignments inside `define … end` used to push their shapes onto the
  stage; template locals never reach the stage now.
- `cup += handle` now grows the named object already on stage, not just
  the variable.

## 2.0.0 "DEEP" — the three-tier engine

Inspired by the OTDL v5 architecture, made real under the same forever
constraints: zero external crates, Rust + handwritten assembly only, the
browser stays a dumb terminal, fewer than 60 keywords, old files never break.

### Added
- **OTD-ASM virtual machine (Tier 3):** every numeric expression compiles
  to 32-byte vector instructions (v5 §12 layout) and runs on a 256-register
  f64 machine with per-register unit tags. `--vm-dump file.otd` prints the
  disassembly. Raw `asm!` kernels: register-file zeroing (`rep stosq`),
  f64 dot3 (SSE2), Newton-refined rsqrt. Parity: 2 000-case fuzz vs the
  tree-walk, plus the whole 1.0 corpus as a regression gate.
- **Deep geometry (Tier 2):** half-edge topology with manifold checks;
  DEC cotangent-Laplacian smoothing (`smooth`); Loop subdivision
  (`subdiv`); BVH acceleration; SDF smooth-union blending (`blend`);
  exact inertia tensors via tetrahedron quadrature.
- **XPBD physics (Tier 2):** extended position-based dynamics solver,
  catenary math, and the `rope` word — cables hang in true catenaries,
  sag measured (0.1 % error), length preserved within 0.03 %.
- **Light:** Cook-Torrance GGX + Smith + Schlick shading replaces
  Blinn-Phong; ray-traced soft shadows; baked ambient occlusion.
- **Language:** four keywords (58 total), new `ask` answers
  (inertia, cable sag, watertight, triangle count), `pi`.
- **Docs:** 2.0 architecture (Lite/Deep/ASM), grammar additions,
  validation report with measured numbers, two new lessons.

### Fixed (1.0 bugs found by the 2.0 syntax audit)
- `metaball [(x,y,z)…]` — the documented call form never triggered call
  parsing (metaball was missing from the list-taking trigger).
- `pi` was documented but never implemented.
- The marching-tetrahedra table did not tile the cube (coverage test:
  125 uncovered / 467 overlapping cells) — metaball meshes were never
  watertight. Now Freudenthal–Kuhn, manifold-verified.
- The 2-2 isosurface case triangulated quads along a shared boundary edge,
  leaving diagonal edges uncovered (pinholes). Crossings are now walked as
  a cycle.

### Performance
- Renders with soft shadows + AO: cup 900×600 @2×SSAA in ~0.44 s
  (cell-bucketed ray queries + analytic ground).
- VM compile-once cache: pattern bodies compile once, re-run per iteration.

## 1.0.0 — the vertical slice
P0000–P0649: language, BSP CSG, physics answers, software rasterizer,
PNG/DEFLATE, HTTP server, exporters, viewer, 10 examples, 20 lessons.
