# OTD Mechanics Library — Master Index

Focused for mechanical work: parts, materials, machines, physics demos,
primitives, lessons — plus the complete periodic table in `elements/`.

| category | files | what's inside |
|----------|-------|---------------|
| `parts/` | 45 | see `parts/00-README.md` |
| `materials/` | 263 | see `materials/README.md` |
| `machines/` | 65 | see `machines/README.md` |
| `physics/` | 87 | see `physics/README.md` |
| `primitives/` | 293 | see `primitives/README.md` |
| `lessons/` | 92 | see `lessons/README.md` |
| `elements/` | 6019 | 118 element folders × 51 files — periodic table |

Library totals: **3205 .otd programs** + 3659 element data files.

## Start here

- build a machine: `library/parts/` → bolt, nut, gear_five…twelve, piston, crankshaft
- watch physics: `library/physics/` and `simulate: drop / float / collapse / splash`
- learn materials: `library/materials/` (brick.otd, water.otd, glass.otd — now transparent!)
- element science: `library/elements/026-iron/01-identity.md` … all 118
- the flagship: `examples/steam-engine.otd` + 5 official renders

Every file validated: `cargo test` walks the whole library on each run (280 tests).
