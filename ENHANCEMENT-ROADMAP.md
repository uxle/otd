# OTD 6.0 — Enhancement Roadmap & Realization Report

> The question this document answers: **"How much can this project be enhanced,
> and what's been done vs what's left?"**

## Current state: OTD 6.0 "INTELLIGENCE"

| Category | Done | Count |
|---|---|---|
| **Keywords** | scene, unit, version, gravity, camera, hide, show, temperature, particle, include, magnetize, strict, overlap, connect, 14 primitives, 11 builders, smooth, subdiv, add, subtract, intersect, at, rotate, scale, mirror, repeat, grid, ring, scatter, define, use, material, color, simulate, ask, export, print, mm, cm, m, in, ft, deg, and, or, not, true, false, is, mod, if, else, end, for, to, by, while, break, continue, assert, tolerance, validate, explain, quality, array, layer, macro, measure, profile, benchmark, snapshot, print_json | 83 |
| **Functions** | cos, sin, tan, sqrt, abs, min, max, round, floor, ceil, pow, log, ln, exp, sign, hypot, atan, atan2, asin, acos, lerp, clamp, count, sum, avg, help, functions, materials, keywords, shapes, sims, ohm_v, ohm_i, ohm_r, power_vi, power_ir, cap_energy, ind_energy, rc_tau, lc_omega | 40 |
| **Shapes** | sphere, cube, cylinder, cone, torus, pyramid, prism, capsule, wedge, plane, tube, helix, rope, thread, gear, spring, bolt, nut, extrude, revolve, sweep, loft, text, terrain, metaball, import, blend, hollow, group | 29 |
| **Materials** | iron, steel, stainless, aluminum, copper, brass, bronze, gold, silver, titanium, zinc, lead, chrome, tungsten, wood, oak, pine, teak, glass, plastic, rubber, ceramic, concrete, marble, fabric, carbon, ice, foam, water, oil, mercury, ethanol, acetone, glycerin, hydrogen, helium, methane, ammonia, nitrogen, air, oxygen, steam, carbon_dioxide, chlorine, uranium, plutonium, thorium, lithium, coal, gasoline | 50 |
| **Simulate domains** | drop, float, collapse, splash, settle, solidity, gas, mix, energy, heat, magnet, sound, light, time, learn, stats, orbit, atom, decay, particles, aero, fluid, electro, stellar, rigid, motor, circuit | 27 |
| **Colors** | 147 named CSS/SVG colors + #rrggbb hex | 147 |
| **Units** | mm, cm, m, km, in, ft, yd, um, deg, rad | 10 |

## Enhancement history

### OTD 6.0 "INTELLIGENCE" — self-describing, single-source version, callable electrodynamics, electrical connectivity, expression interpolation

- **Self-describing:** `--list-*` CLI flags + in-script `help()/functions()/materials()/keywords()/shapes()/sims()`
- **Version reporting:** `env!("CARGO_PKG_VERSION")` everywhere — single source of truth
- **Callable electrodynamics:** 9 functions wired up (ohm_v, ohm_i, ohm_r, power_vi, power_ir, cap_energy, ind_energy, rc_tau, lc_omega)
- **Electrical connectivity:** `connect: A B` + `simulate: circuit` (real winding resistance, I=V/R)
- **Motor simulation:** `simulate: motor` (F=B·I·L, τ=N·B·I·A, "WILL IT MOVE?")
- **Expression interpolation:** `print "{a+b}"` evaluates expressions
- **Min Rust version:** 1.85+ documented

### OTD 4.0 "DYNAMICS" — aerodynamics, fluid dynamics, electrodynamics, stellar dynamics, rigid body dynamics, multi-part assembly, magnetize, strict-overlap

- 5 new physics domains, 4 new keywords (include, magnetize, strict, overlap)
- 5 silent-wrongness fixes (hollow, group, rotate pivot, print interpolation, overlap detection)
- 18 corrective error templates
- Blender-like editor (menu bar, toolbar, outliner, properties, timeline, gizmo, console, status bar)
- Command palette (Ctrl+P), find/replace (Ctrl+F), 3 themes
- Parallel downsample, FxHash, mesh-level overlap SAT
- 4 new shapes (gear, spring, bolt, nut), 2 new physics (vibration, optics)

### OTD 3.x — energy, thermodynamics, magnetism, waves, relativity, PerceptAudio, AVC video, OTD-Burn, Standard Model

- Complete energy taxonomy (2 categories, 9 forms)
- Thermodynamics (temperature, heat, phase, the four laws)
- Magnetism (fields, forces, Faraday, Ampère, Earth)
- Sound & light (Doppler, Snell, Wien)
- Time & motion (free-fall, pendulum, relativity)
- PerceptAudio native (VAD, YIN pitch, speaker clustering, Wiener denoise)
- AVC video studio (H.264 encoder + MP4 muxer, 8-camera panel)
- Self-made nn.rs + OTD-Burn deep learning
- Statistics, astronomy, program synthesizer
- Standard Model (17 particles, quark model, Weizsäcker binding energy)

### OTD 2.x — Solidity, syntax expansion, deep tier

- Real solidity (settle + interpenetration audit)
- Environments (buoyancy + drag)
- Chemistry (liquid miscibility, gas mixing, reactions)
- if/for/while/break/continue, compound assignment, comparisons, logic
- DEC cotan-Laplacian smooth, Loop subdivision, SDF blend, XPBD rope
- VM (OTD-ASM: 32-byte vector ISA)

### OTD 1.0 — the base

- 14 primitives, BSP CSG, analytic hollow
- 27 materials, 147 colors, 75 keywords
- SIMD renderer (z-buffer, GGX, ray-traced shadows, AO)
- HTTP server, PNG encoder, STL/OBJ export
- Zero external crates — the constraint that started it all

## What's left (future enhancements)

- **BLDC commutation:** `simulate: motor` currently models a brushed DC motor; a BLDC needs commutation timing
- **Multi-select in the outliner:** currently single-select only
- **Node-based material editor:** the "Nodes" tab is a placeholder
- **Sculpt mode:** the "Sclpt" toolbar button is a placeholder
- **Edit mode:** vertex-level editing is a placeholder
- **Restore snapshots:** `snapshot:` saves but `restore:` is not yet implemented
- **Macro expansion:** `macro` defines but `use` expansion is not yet wired
- **More materials:** composites, semiconductors, superconductors
- **More shapes:** bevel, chamfer, fillet, boolean patterns
- **GPU rendering:** currently CPU-only (the constraint is zero external deps)
