# OTD6.0 "INTELLIGENCE" — Open Three-Dimensional Language

> **The Intelligence Edition.** OTD6 makes the tool self-describing,
> fixes version reporting, wires up advertised functions, adds an
> electrical connectivity layer, and brings expression interpolation —
> all based on concrete feedback from an AI that actually used the tool.
>
> **Self-describing:** `--list-functions`, `--list-materials`,
> `--list-keywords`, `--list-shapes`, `--list-simulate`, `--list-colors`,
> `--list-all` from the CLI; `help()`, `functions()`, `materials()`,
> `keywords()`, `shapes()`, `sims()` in-script. All dumped from source,
> never hand-maintained.
>
> **Single-source version:** `Cargo.toml` is the one source of truth;
> `--version`, the banner, and the phase banner all read
> `env!("CARGO_PKG_VERSION")`.
>
> **Callable electrodynamics:** `ohm_v(i,r)`, `ohm_i(v,r)`,
> `ohm_r(v,i)`, `power_vi(v,i)`, `power_ir(i,r)`, `cap_energy(c,v)`,
> `ind_energy(l,i)`, `rc_tau(r,c)`, `lc_omega(l,c)` — all usable
> from scripts.
>
> **Electrical connectivity:** `connect: A B` + `simulate: circuit`
> walks the real resistance of modeled windings and reports
> current/voltage/power.
>
> **Motor simulation:** `simulate: motor` — F=B·I·L, τ=N·B·I·A,
> back-EMF, stall torque, RPM, efficiency, and the verdict:
> "WILL IT MOVE? YES ✓ / NO ✗".
>
> **Expression interpolation:** `print "{a+b}"` now evaluates
> expressions, not just bare names.
>
> **Minimum Rust version:** 1.85+ (documented in `run.sh`).

| Area | The upgrade |
|---|---|
| **Self-describing** | `--list-functions/materials/keywords/shapes/simulate/colors/all` + in-script `help()/functions()/materials()/keywords()/shapes()/sims()` |
| **Version reporting** | Single source: `Cargo.toml` → `env!("CARGO_PKG_VERSION")` everywhere. No more four different version strings. |
| **Electrodynamics functions** | 9 callable functions: `ohm_v`, `ohm_i`, `ohm_r`, `power_vi`, `power_ir`, `cap_energy`, `ind_energy`, `rc_tau`, `lc_omega` |
| **`connect: A B`** | Declare electrical connectivity between named parts. Stored on `World::connections`. |
| **`simulate: circuit`** | Walks connections, computes real resistance from material resistivity × wire geometry, reports I=V/R, P=VI, verdict. |
| **`simulate: motor`** | Full motor analysis: stall torque, no-load RPM, back-EMF, operating point, efficiency, "WILL IT MOVE?" |
| **Expression interpolation** | `print "{a+b}"` evaluates expressions. Falls back to bare name, then warns. |
| **Min Rust version** | 1.85+ documented in `run.sh` (vendor/avc needs edition2024). |
| **CLI flags** | `--list-all`, `--list-functions`, `--list-materials`, `--list-keywords`, `--list-shapes`, `--list-simulate`, `--list-colors` |

## Quickstart

```bash
cargo run --release            # serve the studio at http://127.0.0.1:6830
cargo test                     # 516 tests pass, 0 failures
otd --version                  # OTD 6.0.0 — Open Three-Dimensional Language
otd --list-all                 # dump everything the tool can do
otd --check examples/electric-motor.otd   # motor + circuit demo
```

### The motor + circuit demo

```otd
scene "Electric Motor + Circuit"
unit: cm

# build the motor parts
stator_n = cube(w: 2cm, d: 4cm, h: 4cm) at (-3cm, 2cm, 0) material: iron
stator_s = cube(w: 2cm, d: 4cm, h: 4cm) at (3cm, 2cm, 0)  material: iron
magnetize: stator_n
magnetize: stator_s
rotor = cylinder(r: 1.5cm, h: 4cm) at (0, 2cm, 0) material: copper
shaft = cylinder(r: 3mm, h: 8cm) rotate (0, 0, 90deg) at (0, 2cm, 0) material: steel
battery = cylinder(r: 1.5cm, h: 5cm) at (0, 2cm, 8cm) material: lithium

# connect the electricity
connect: battery rotor

# check the circuit and motor
simulate: circuit
simulate: motor

# use the callable electrodynamics functions
v = 6
r = ohm_r(v, 2)
i = ohm_i(v, r)
p = power_vi(v, i)
print "V = {v}V, R = {r}ohm, I = {i}A, P = {p}W"
print "Also: v+r = {v+r} (expression interpolation)"

# self-describing
functions()
materials()
```

Output:
```
connect: battery ↔ rotor — electrical path established
CIRCUIT ANALYSIS — walking the real resistance of modeled windings
  1 connection(s) declared
  supply: 6 V
  current: I = V/R = 0.003 A
  VERDICT: current flows — the circuit is closed and conducting
ELECTRIC MOTOR SURVEY — F = B·I·L, τ = N·B·I·A
  stall torque: 0.0565 N·m
  no-load speed: 6366 RPM
==== WILL IT MOVE? ====
  YES ✓ — stall torque exceeds load by 56.5×
  The motor WILL spin at 4899 RPM under load.
V = 6V, R = 3ohm, I = 2A, P = 12W
Also: v+r = 9 (expression interpolation)
callable functions (40): cos sin tan sqrt abs ...
materials (50): iron steel stainless aluminum copper ...
```

### CLI self-describing flags

```bash
otd --list-functions    # 40 callable functions (from keywords.rs FUNCS)
otd --list-materials    # 50 materials (from materials.rs MATERIALS)
otd --list-keywords     # 83 keywords (from keywords.rs KEYWORDS)
otd --list-shapes       # 14 primitives + 11 builders
otd --list-simulate     # all simulate domains by category
otd --list-colors       # 147 named colors + hex
otd --list-all          # everything at once
```

### In-script self-describing

```otd
help()          # prints CLI flag hints
functions()     # lists all 40 callable functions
materials()     # lists all 50 materials
keywords()      # lists all 83 keywords
shapes()        # lists all primitives + builders
sims()          # lists all simulate domains
```

---

## What's inside

| Layer | Files | Written from scratch |
|---|---|---|
| Language | `src/lang/` | lexer, recursive-descent parser, 83-keyword registry, friendly errors with "did you mean?" + 18 corrective templates |
| VM (2.0) | `src/vm/` | OTD-ASM: 32-byte vector ISA, expression→bytecode compiler, 256-register interpreter |
| Deep tier (2.0) | `src/geo/` | half-edge topology, DEC cotan-Laplacian `smooth`, Loop `subdiv`, BVH + SDF `blend`, XPBD `rope`, exact inertia tensor, mesh-level overlap SAT |
| Geometry | `src/geo/` | 14 primitives + 11 builders, BSP CSG, analytic hollow, extrude/revolve/sweep/loft/tube/helix, 3D text, fBm terrain, metaballs, STL/OBJ import |
| Materials | `src/world/materials.rs` | 50 real materials (density, melting point, conductivity, friction, bounce, magnetism, strength) + 147 named colors |
| Physics | `src/world/physics.rs` | mass = ΣV·ρ, Archimedes float, drop energy, collapse stress, motor force, circuit analysis |
| Dynamics | `src/world/aerodynamics.rs`, `fluiddynamics.rs`, `electrodynamics.rs`, `stellardynamics.rs`, `rigidbody.rs` | 5 dynamics domains: drag/lift/Reynolds/Mach, Bernoulli/Poiseuille/Stokes, Ohm/Kirchhoff/Maxwell, N-body/virial/Jeans, inertia/gyroscope |
| Motor + Circuit | `src/world/motor.rs` | `simulate: motor` (F=B·I·L, τ=N·B·I·A, back-EMF, RPM, "WILL IT MOVE?") + `simulate: circuit` (real winding resistance, I=V/R) |
| Renderer | `src/render/` | SIMD software rasterizer: z-buffer, GGX microfacet light, ray-traced soft shadows, baked AO, parallel downsample, 2× SSAA |
| Server | `src/net/` | HTTP/1.1 (threads, keep-alive), own JSON, own Base64, compile + lit-scene + frame caches |
| Viewer | `assets/` | Blender-like studio: menu bar, toolbar, outliner, properties, timeline, gizmo, console, status bar, command palette (Ctrl+P), find/replace (Ctrl+F), 3 themes |
| Website | `website/` | landing page (hero, physics, language, self-make, video sections) |
| Self-describing | `src/main.rs` | `--list-functions/materials/keywords/shapes/simulate/colors/all` |

## The language in 30 seconds

- **Shapes:** `sphere cube cylinder cone torus pyramid prism capsule wedge plane tube helix rope thread` + `extrude revolve sweep loft text import terrain metaball hollow group blend`
- **Booleans:** `a + b` fuse, `a - b` cut, `a & b` overlap, `hollow(wall: 3mm)` shell
- **Place:** `at (x, y, z)`, `rotate`, `scale`, `mirror x`, `rotate (...) pivot (x, y, z)`
- **Patterns:** `repeat(n: 4, step: (2cm, 0, 0))`, `grid(nx: 3, nz: 3)`, `ring(n: 8, radius: 5cm)`
- **Parts:** `define wheel(r) = …` then `use wheel(r: 3cm)` — multi-part `include:` workflow
- **Science:** `material: gold`, `ask "mass?"`, `simulate: drop / float / collapse / splash / settle / solidity / gas / mix / energy / heat / magnet / sound / light / time / learn / stats / orbit / atom / decay / particles / aero / fluid / electro / stellar / rigid / motor / circuit`
- **Electrical:** `connect: A B`, `simulate: circuit`, `simulate: motor`
- **Functions (40):** `cos sin tan sqrt abs min max round floor ceil pow log ln exp sign hypot atan atan2 asin acos lerp clamp count sum avg help functions materials keywords shapes sims ohm_v ohm_i ohm_r power_vi power_ir cap_energy ind_energy rc_tau lc_omega`
- **Units:** `mm cm m km in ft yd um deg rad`; bare numbers are centimeters
- **OTD6 keywords:** `connect` (electrical), self-describing functions, expression interpolation
- 83 keywords + 40 functions — the cheat sheet grew but stayed under the 100 budget.

## Docs

- `docs/00-ARCHITECTURE.md` — the full system design
- `docs/02-KEYWORDS.md` — the complete keyword list (83)
- `docs/03-GRAMMAR.md` — the EBNF grammar
- `docs/04-SPECIFICATION.md` — the full language spec
- `docs/05-MATERIALS.md` — the 50 materials
- `docs/07-CHEATSHEET.md` — one-page beginner card
- `docs/10-LESSONS.md` — 23 lessons from "hello cube" to decide-and-repeat
- `examples/electric-motor.otd` — motor + circuit + connect demo

## Constraints we keep, forever

1. **Zero external crates.** If a phase needs a library, the phase is redesigned — not the constraint.
2. **Rust + assembly only** on the backend; the browser stays a dumb terminal.
3. **< 100 keywords.** New capabilities are new words only if the budget allows; otherwise they are parameters.
4. **Old `.otd` files never break** (forward-compatibility policy).

MIT license.
