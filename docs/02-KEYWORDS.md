# OTD 2.1 — Complete Keyword List

## Counting Methodology

OTD follows the SVG model: **structural words are keywords** (like SVG's ~50
elements); **parameter names are attributes** (like SVG's hundreds of
attributes); and **materials, colors, units are value libraries** (like SVG's
147 named colors, which nobody counts as keywords).

> **OTD 2.1 ships 75 structural keywords + 25 math functions.** The 2.1
> syntax expansion added 17 control-flow and logic words and raised the hard
> budget from 60 to 100 — the cheat sheet grew its second side.

## The 75 Keywords

### Scene & Output (7)

| # | Keyword | One-line meaning |
|---|---------|------------------|
| 1 | `scene` | title of the model: `scene "Coffee Cup"` |
| 2 | `unit` | default unit for bare numbers: `unit: mm` (default cm) |
| 3 | `version` | compatibility header: `version 1` (optional) |
| 4 | `gravity` | `gravity: earth` / `moon` / `mars` / `off` / `9.81` |
| 5 | `camera` | start view: `camera: iso` (front/top/side/iso) |
| 6 | `hide` | hide a named object |
| 7 | `show` | show it again |

### Primitives (13 — `rope` joins in 2.0)

| # | Keyword | Default (all sizes smart) |
|---|---------|--------------------------|
| 8 | `sphere` | radius 1cm |
| 9 | `cube` | size 2cm (alias: `box` with w/h/d) |
| 10 | `cylinder` | r 1cm, h 2cm; `top:`/`bottom:` → tapered cup shapes |
| 11 | `cone` | r 1cm, h 2cm |
| 12 | `torus` | R 2cm, tube 5mm; lies flat like a donut |
| 13 | `pyramid` | base 2cm, h 2cm |
| 14 | `prism` | 6 sides, r 1cm, h 2cm |
| 15 | `capsule` | r 1cm, h 3cm |
| 16 | `wedge` | ramp: 2×2×1cm |
| 17 | `plane` | 10×10cm thin slab |
| 18 | `tube` | pipe r 5mm through given points |
| 19 | `helix` | spring: r 3cm, pitch 1cm, 5 turns, tube 5mm |
| 20 | `rope` | **2.0** XPBD-solved cable: `rope(from: (0,30cm,0), to: (40cm,30cm,0), thickness: 4mm, sag: 5cm)` — hangs in a true catenary |


### Advanced Builders (10 — `blend` joins in 2.0)

| # | Keyword | Meaning |
|---|---------|---------|
| 21 | `extrude` | lift a flat polygon into 3D |
| 22 | `revolve` | spin a 2D profile around Y (vases, chess pieces) |
| 23 | `sweep` | drag a profile along a 3D path |
| 24 | `loft` | skin cross-sections into one solid |
| 25 | `text` | real 3D letters: `text "HELLO"` |
| 26 | `import` | load an STL/OBJ from a URL |
| 27 | `terrain` | rolling landscape from seeded noise |
| 28 | `metaball` | mercury-like blobby union |
| 29 | `hollow` | shell out a solid, leaving walls (`hollow(a, wall: 3mm)`) |
| 30 | `blend` | **2.0** smooth-union fillet: `blend(base, knob, gap: 8mm)` — solder-like fusion via SDF fields |


### Boolean (3)

| # | Keyword | Operator form |
|---|---------|---------------|
| 29 | `add` | `a + b` — fuse into one solid |
| 30 | `subtract` | `a - b` — cut |
| 31 | `intersect` | `a & b` — keep the overlap |

### Transforms & Deep Modifiers (6 — `smooth` + `subdiv` join in 2.0)

| # | Keyword | Example |
|---|---------|---------|
| 35 | `at` | `at (4cm, 5cm, 0)` — where it stands |
| 36 | `rotate` | `rotate (0, 45deg, 0)` or `rotate 90deg` |
| 37 | `scale` | `scale 2` or `scale (2, 1, 1)` |
| 38 | `mirror` | `mirror x` |
| 39 | `smooth` | **2.0** DEC cotan-Laplacian relaxation: `blob smooth(n: 3, strength: 0.5)` — corners melt, volume tracked |
| 40 | `subdiv` | **2.0** Loop subdivision: `sphere(r: 2cm) subdiv(n: 2)` — 4× triangles per level, fairer surface |


### Patterns (4)

| # | Keyword | Magic vars |
|---|---------|-----------|
| 36 | `repeat` | `repeat(n: 4, step: (5cm, 0, 0)) …` uses `i` |
| 37 | `grid` | `grid(nx: 3, nz: 3, spacing: 5cm) …` uses `i`, `j` |
| 38 | `ring` | `ring(n: 8, radius: 5cm) …` uses `i`, `a` |
| 39 | `scatter` | `scatter(n: 20, radius: 12cm, seed: 7) …` uses `i` |

### Parts & Appearance (5)

| # | Keyword | Meaning |
|---|---------|--------|
| 40 | `group` | parts stay parts (no fusing) |
| 41 | `define` | template: `define wheel(r) = …` |
| 42 | `use` | place a copy: `use wheel at (2cm, 0, 4cm)` |
| 43 | `material` | `material cup: ceramic` or scene default `material: steel` |
| 44 | `color` | `color cup: ivory` (147 names + hex) |

### Physics, Query, Export (5)

| # | Keyword | Meaning |
|---|---------|--------|
| 45 | `simulate` | `simulate: drop` / `float` / `collapse` |
| 46 | `ask` | `ask "mass?"` — the model answers |
| 47 | `export` | `export stl "cup.stl"` |

### Math Functions (8 → 25 in 2.1)

| # | Keyword | Note |
|---|---------|------|
| 48 | `cos` | angles in degrees |
| 49 | `sin` | angles in degrees |
| 50 | `sqrt` | |
| 51 | `abs` | |
| 52 | `min` | |
| 53 | `max` | |
| 54 | `round` | |

Plus one predefined constant: `pi` (wired for real in 2.0 — it was
documented in 1.0 but never implemented; the syntax audit caught it).

### The 2.1 Syntax Expansion (17 new words)

Everything above is OTD 2.0. The 2.1 expansion adds the words that let a
model *decide* and *repeat* — full grammar in `03-GRAMMAR.md`.

| # | Keyword | Kind | One-line meaning |
|---|---------|------|------------------|
| 55 | `if` | control flow | `if x > 5cm … else if … else … end` |
| 56 | `else` | control flow | the other branch (also `else if`) |
| 57 | `end` | control flow | closes an if / for / while / define block |
| 58 | `for` | loop | `for i = 1 to 10 by 2` or `for x in [a, b, c]` |
| 59 | `to` | loop | the far end of a `for` count (inclusive) |
| 60 | `by` | loop | the step: `for i = 0deg to 90deg by 15deg` |
| 61 | `while` | loop | `while x < 5cm … end` (100,000-pass cap) |
| 62 | `break` | loop | leave the innermost loop now |
| 63 | `continue` | loop | skip to the next pass |
| 64 | `and` | logic | `x > 1 and x < 5` (same as `&&`) |
| 65 | `or` | logic | `x < 1 or x > 5` (same as `||`) |
| 66 | `not` | logic | `not broken` (same as `!`) |
| 67 | `true` | logic | the boolean literal |
| 68 | `false` | logic | the other boolean literal |
| 69 | `is` | compare | `x is 5` (same as `==`); `is not` = `!=` |
| 70 | `mod` | arithmetic | remainder: `7 mod 2` (same as `%`) |
| 71 | `assert` | checking | `assert x > 0, "message"` — fails loud |

(`in` — the fourth membership word — lives in the unit library, doing
double duty since 1.0: attached `3in` is inches, `x in [1, 2, 3]` is
membership. The lexer tells them apart by what follows.)

### New math functions (2.1, 17 more → 25 total)

`floor` `ceil` `pow` `log` `ln` `exp` `sign` `hypot` `atan` `atan2` `asin`
`acos` `lerp` `clamp` `count` `sum` `avg` — inverse trig answers in degrees;
`lerp`/`clamp` carry units through; `count`/`sum`/`avg` walk lists and
ranges (`sum(1..100)` is 5050).

## Not Counted (by design)

- **Operators:** `=` `+` `-` `&` `*` `/` `(` `)` `[` `]` `,` `:` `#` and the
  2.1 set `^` `%` `!` `<` `>` `<=` `>=` `==` `!=` `&&` `||` `..` `+=` `-=`
  `*=` `/=` `;`
- **Parameter names (~70):** `radius, r, size, height, h, width, w, depth, d,
  top, bottom, wall, open, tube, turns, pitch, path, points, sections, sides,
  base, teeth, n, step, spacing, seed, string, smooth, from, to, axis, …`
  (full table in `04-SPECIFICATION.md`)
- **Value libraries:** 27 materials · 147 SVG colors · units `mm cm m km in ft yd um deg rad` ·
  gravity presets `earth moon mars off` · sim modes `drop float collapse` ·
  camera views `front top side iso` · export formats `stl obj gltf usdz png scad`
- **Magic pattern variables:** `i`, `j`, `a` (copy index / angle)
- **Booleans:** `true`/`false` (keywords since 2.1), plus the value words `on`, `off`

## Value Library Sizes

| Library | Count | Examples |
|---|---|---|
| Materials | 27 | iron, steel, stainless, aluminum, copper, brass, bronze, gold, silver, titanium, zinc, lead, chrome, tungsten, wood, oak, pine, teak, glass, plastic, rubber, ceramic, concrete, marble, fabric, carbon, water* |
| Colors | 147 + hex | red, steelblue, gold, ivory, burlywood… (full CSS/SVG set) |
| Units | 8 length + 2 angle | mm, cm, m, km, in, ft, yd, um, deg, rad |

*`water` is a physics reference (1000 kg/m³) used by buoyancy answers.
