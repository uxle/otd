# OTD 6.0 "INTELLIGENCE" — start here

## Build & run (one command builds the engine + both intelligence tools)

```bash
./run.sh                 # build + serve at http://127.0.0.1:6830
```

**Minimum Rust version: 1.85+** (vendor/avc needs edition2024). Run `rustup update` if you have an older toolchain.

## What is OTD?

OTD is a **physics-first 3D modeling language**. You write plain text; the engine compiles it into a watertight, mass-checked, physics-verified 3D model. Pure Rust + assembly, zero external libraries.

```otd
scene "Coffee Cup"
cup = cylinder(top: 40mm, bottom: 30mm, height: 100mm) - hollow(wall: 3mm, open: top)
handle = torus(radius: 20mm, tube: 6mm) rotate (90deg, 0, 0) at (35mm, 25mm, 0)
add cup, handle
material cup: ceramic
color cup: ivory
ask "mass?"
```

This renders, weighs **220.7 g** (91.9 cm³ × 2.40 g/cm³), and tells you it sinks — with the math shown.

## What's new in OTD 6.0

| Feature | How to use |
|---|---|
| **Self-describing** | `otd --list-all` (CLI) or `functions()` / `materials()` / `keywords()` (in-script) |
| **Single-source version** | `otd --version` → `OTD 6.0.0` (from Cargo.toml, nowhere else) |
| **Callable electrodynamics** | `v = ohm_i(12, 5)` — 9 functions: ohm_v/ohm_i/ohm_r/power_vi/power_ir/cap_energy/ind_energy/rc_tau/lc_omega |
| **Electrical connectivity** | `connect: battery rotor` + `simulate: circuit` |
| **Motor simulation** | `simulate: motor` — "WILL IT MOVE? YES ✓" |
| **Expression interpolation** | `print "{a+b}"` evaluates expressions |
| **Blender-like editor** | menu bar, toolbar, outliner, properties, timeline, gizmo, console, command palette (Ctrl+P), find/replace (Ctrl+F), 3 themes |

## Quick CLI reference

```bash
otd                           # serve the studio (Blender-like editor)
otd --check examples/cup.otd  # stats + physics in terminal
otd --png examples/cup.otd cup.png --view iso   # render to PNG
otd --export examples/cup.otd cup.stl --fmt stl # export to STL
otd --list-all                # self-describing: dump everything
otd --version                 # OTD 6.0.0
otd --make "a titanium ball dropping into water from 1m"  # AI synthesizer
```

## The language in 60 seconds

- **Shapes:** `sphere cube cylinder cone torus pyramid prism capsule wedge plane tube helix rope thread`
- **Booleans:** `a + b` fuse, `a - b` cut, `a & b` overlap
- **Place:** `at (x, y, z)`, `rotate (0, 45deg, 0)`, `scale 2`, `mirror x`
- **Science:** `material: gold`, `ask "mass?"`, `simulate: drop / float / motor / circuit / magnet / aero / fluid / electro / stellar / rigid / ...`
- **Electrical:** `connect: A B`, `simulate: circuit`, `simulate: motor`
- **Functions:** `sqrt(16)`, `ohm_i(12, 5)`, `functions()`, `materials()`
- **Units:** `mm cm m km in ft yd um deg rad`; bare numbers are centimeters
- 83 keywords + 40 functions, all under the 100 budget.

## Examples to try

```bash
otd --check examples/cup.otd              # the hello-world: a coffee cup
otd --check examples/electric-motor.otd   # motor + circuit + connect
otd --check examples/gear-system.otd      # mechanical gears
otd --check examples/steam-engine.otd     # animated steam engine
otd --png examples/chess-set.otd chess.png # a full chess set
```

## Docs

- `README.md` — the full feature tour
- `docs/07-CHEATSHEET.md` — one-page beginner card
- `docs/10-LESSONS.md` — 23 lessons from "hello cube" to decide-and-repeat
- `CHANGELOG.md` — every version, every fix
- `BUGFIXES.md` — the bugs that were found and how they were fixed

MIT license. Zero external crates. Forever.
