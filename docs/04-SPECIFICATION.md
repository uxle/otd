# OTD 2.1 — Full Language Specification

This is the normative reference for every OTD construct: parameters, defaults,
semantics, and examples. Defaults are chosen so that **every command works with
zero parameters**.

Conventions: lengths in any unit (`mm cm m km in ft yd um`), angles in
`deg rad` (degrees in the language, radians internally where trig needs
them). Internal canonical unit is mm; default unit for bare numbers is
**cm**.

**2.1 adds a whole specification part** — see the final section
("9. OTD 2.1 — decide, repeat, check") for `if`/`for`/`while`, compound
assignment, operators, ranges, indexing, block templates, `assert`, and the
new functions.

---

## 1. Statements

### `scene "Title"`
Names the model (shown in the app title bar and exports). Optional.
```
scene "Coffee Cup"
```

### `version 1`
Optional compatibility header, always the first line. See `08-COMPATIBILITY.md`.

### `unit: mm|cm|m|km|in|ft|yd|um`
Default unit for bare numbers. Default: `cm`.
```
unit: mm
bolt = cylinder(r: 3, h: 12)        # 3 mm × 12 mm
```

### `gravity: earth|moon|mars|off` or `gravity 9.81`
Gravity for simulations. Presets: earth 9.81, moon 1.62, mars 3.71 m/s².
`off` freezes physics (used by the solar-system example).

### `camera: front|top|side|iso`
Starting camera view. Default `iso`.

### `name = expression`
Creates a named object and **shows it immediately**. Re-assignment replaces it.
```
cup = cylinder(top: 40mm, bottom: 30mm, height: 100mm)
```

### `define name(params) = expression`
Reusable template (hidden until used). Parameters are passed by name at call
time; simple values may be positional.
```
define wheel(r) = torus(r, 2cm) + cylinder(r: r, h: 1cm, material: rubber)
cart_wheel = wheel(r: 4cm)
```

### `use name at (…)` (plus rotate/scale/color postfixes)
Places a visible copy of a template. Copies are independent objects.
```
use wheel at (-20cm, 0, 10cm) color: black
```

### `add target, part1, part2, …`
Fuses parts into target (one mesh, one material) and removes the part entries.
```
add cup, handle
```

### `material name: steel` / `material: steel`
Sets one object's material, or the **scene default** (applies to every object
that never gets an explicit material).

### `color name: ivory` / `color: ivory`
Same for colors. Material names and color names never collide (resolved by
context); when a word exists in both (e.g. `gold`), the parameter name decides.

### `hide name` / `show name`
Toggles visibility — handy for construction geometry.

### `simulate: drop|float|collapse`
Runs a physics simulation (see §6). Executed on manual Run, not on every
keystroke.

### `ask "…"`
Asks the model a question (see §7). Answers appear in the console.

### `export stl|obj|gltf|usdz|png|scad "file"`
Exports the current model. Executed on manual Run. In the browser each export
downloads a file.

---

## 2. Primitives (12)

All sizes accept any length unit. Every shape is built **resting on the ground**.

### sphere
| Parameter | Default | Notes |
|---|---|---|
| `r` / `radius` | 1cm | main parameter (bare form: `sphere 3cm`) |
| `smooth` | 48 | meridian segments (24 rings) |

Ball rests at y=0 (center at y=r). Volume 4/3·πr³.

### cube (alias box)
| Parameter | Default | Notes |
|---|---|---|
| `s` / `size` | 2cm | main parameter — cube all sides |
| `w`, `d`, `h` | from `s` | width (x), depth (z), height (y) |
```
plate = cube(w: 90cm, d: 60cm, h: 4cm)
```

### cylinder
| Parameter | Default | Notes |
|---|---|---|
| `r` / `radius` | 1cm | main parameter |
| `h` / `height` | 2cm | |
| `top`, `bottom` | = `r` | set different radii for a **taper** (cups, flower pots) |
| `smooth` | 48 | radial segments |
```
pot = cylinder(bottom: 25mm, top: 35mm, height: 90mm)
```
Sits on its base. Frustum volume πh/3·(R²+Rr+r²).

### cone
| Parameter | Default | Notes |
|---|---|---|
| `r` / `radius` | 1cm | base radius (main parameter) |
| `h` / `height` | 2cm | |
| `smooth` | 48 | radial segments |
| `sides` | 32 | n-sided pyramid-style cone |

### torus
| Parameter | Default | Notes |
|---|---|---|
| `R` / `radius` | 2cm | ring radius (main parameter) |
| `r` / `tube` | 5mm | thickness of the ring |
| `smooth` | 48 | |
Lies flat like a donut. Volume 2π²Rr².

### pyramid
| Parameter | Default | Notes |
|---|---|---|
| `base` | 2cm | square base edge (main parameter) |
| `h` / `height` | 2cm | |
| `sides` | 4 | 3 = triangular pyramid |

### prism
| Parameter | Default | Notes |
|---|---|---|
| `sides` | 6 | 3 = triangular, 8 = octagonal… |
| `r` / `radius` | 1cm | distance to corners |
| `h` / `height` | 2cm | |

### capsule
| Parameter | Default | Notes |
|---|---|---|
| `r` / `radius` | 1cm | (main parameter) |
| `h` / `height` | 3cm | length of the straight section |

Pill shape (cylinder + hemisphere caps). Volume πr²h + 4/3πr³.

### wedge
| Parameter | Default | Notes |
|---|---|---|
| `w` | 2cm | width (ramp run) |
| `d` | 2cm | depth |
| `h` | 1cm | height (ramp rise) |

Right-triangle ramp. Volume w·d·h/2.

### plane
| Parameter | Default | Notes |
|---|---|---|
| `w` | 10cm | width |
| `d` | 10cm | depth |
| `t` | 1mm | thickness |

Thin slab — tabletops, floors, water surfaces.

### tube
| Parameter | Default | Notes |
|---|---|---|
| `r` | 5mm | pipe radius |
| `path` | `[(0,0,0),(0,5cm,0)]` | 3D points; smoothed through a spline |
```
pipe = tube(r: 6mm, path: [(0,0,0), (0,20cm,0), (8cm,30cm,0)])
```
Ends are capped (watertight).

### helix
| Parameter | Default | Notes |
|---|---|---|
| `r` / `radius` | 3cm | helix radius |
| `pitch` | 1cm | height gained per full turn |
| `turns` | 5 | total height = turns × pitch |
| `r` / `tube` | 5mm | thickness of the spring wire |
```
spring = helix(radius: 2cm, pitch: 8mm, turns: 6, tube: 2mm) material: steel
```

---

## 3. Advanced Builders (8 + hollow)

### extrude — lift a flat shape
Draws a closed polygon in the XY "paper" plane (facing you) and pulls it toward
the viewer (+Z). Use `rotate`/`at` to place it.
```
plate = extrude [(0,0), (10cm,0), (10cm,10cm), (0,10cm)] depth: 5mm
star  = extrude [(0cm,3cm), (1cm,1cm), (3cm,1cm), (1.2cm,-1cm), (2cm,-3cm),
                 (0cm,-1.5cm), (-2cm,-3cm), (-1.2cm,-1cm), (-3cm,1cm), (-1cm,1cm)]
        depth: 4mm material: gold
```
| Parameter | Default | Notes |
|---|---|---|
| `depth` | 1cm | pull distance |

### revolve — spin a profile
Profile points are `(radius, height)` pairs, revolved around the Y axis.
The profile is auto-closed to the axis, so the solid is watertight.
```
pawn = revolve [(0,0), (12mm,0), (12mm,5mm), (8mm,10mm), (8mm,35mm),
                (13mm,40mm), (5mm,45mm)]
```
| Parameter | Default | Notes |
|---|---|---|
| `angle` | 360deg | partial revolve for cut-away views |

### sweep — drag a profile along a path
| Parameter | Default |
|---|---|
| `path` | `[(0,0,0),(0,5cm,0)]` |
| `r` | 5mm — circular profile |
| `profile` | `circle` (or `square(s)`) |
```
handrail = sweep(path: [(0,90cm,0), (40cm,95cm,0), (80cm,90cm,0)], r: 2cm)
```

### loft — skin cross-sections
Cross-sections are `(radius, height)` pairs; OTD skins them into one solid
ring-by-ring and caps the ends. Sections must be ordered bottom → top.
```
vase_body = loft [(4cm, 0), (2cm, 8cm), (3cm, 12cm)]
```

### text — real 3D letters
| Parameter | Default | Notes |
|---|---|---|
| first positional | `"OTD"` | the words (string) |
| `size` | 2cm | letter height |
| `depth` | 5mm | thickness |
| `spacing` | 0.3 | letter gap (× letter width) |

Letters stand on the ground facing +Z (toward the default camera).
```
label = text "HELLO" size: 3cm depth: 6mm color: steelblue
```
Built from a 5×7 dot font (37 glyphs: A–Z, 0–9, `!`, `?`, `.`).

### import — load a mesh
```
part = import "https://example.com/bracket.stl"
```
STL (binary + ASCII) and OBJ. URLs only in the browser (no local-file access
from a single HTML file — Phase 2 adds drag & drop).

### terrain — landscape
| Parameter | Default | Notes |
|---|---|---|
| `size` | 20cm | square footprint |
| `peaks` | 3 | maximum height |
| `roughness` | 0.5 | 0 = gentle hills, 1 = jagged |
| `seed` | 1 | same seed = same landscape |
```
island = terrain(size: 30cm, peaks: 4cm, roughness: 0.7, seed: 12)
```

### metaball — mercury blobs
Takes a list of `(x, y, z, radius)` balls and builds their blended iso-surface
with marching tetrahedra.
```
blob = metaball [(0,0,0, 1.5cm), (3cm,0,0, 1.5cm), (1.5cm,2cm,0, 1.2cm)] material: chrome
```

### hollow — shell out a solid
| Parameter | Default | Notes |
|---|---|---|
| `wall` | 2mm | wall thickness |
| `open` | `top` | which side gets the opening: `top`, `bottom`, `none` |
```
cup  = cylinder(top: 40mm, bottom: 30mm, height: 100mm) - hollow(wall: 3mm)
box  = hollow(cube 10cm, wall: 5mm)
```
For cylinders, cones and revolves the inner profile is offset **exactly**
(analytic path). For arbitrary meshes a scaled-shell approximation is used and
reported as such. Sugar: `a - hollow(wall: 3mm)`.

---

## 4. Booleans, Transforms, Patterns, Parts

### Booleans
```
fused   = cup + handle              # add (union)
cut     = cheese - sphere 6cm       # subtract
core    = apple & cube 4cm          # intersect
```
Operands are placed first (their `at`/`rotate` apply before the operation).
CSG results have flat shading — that is normal for cut geometry.
Union takes the **left** operand's material and color.

### Transforms (postfix, left-to-right)
```
sphere 3cm at (10cm, 0, 0)
cube 4cm rotate (0, 45deg, 0)
sphere 2cm scale 2
text "OTD" mirror x
lamp = shade rotate (0, 0, -30deg) at (12cm, 34cm, 0)
```
`at` uses the rest point (§ placement rules in `03-GRAMMAR.md`).

### Patterns
```
row    = repeat(n: 5, step: (4cm, 0, 0)) cube(w: 1cm, h: 3cm, d: 1cm)
field  = grid(nx: 4, nz: 4, spacing: 5cm) sphere 1cm
petals = ring(n: 8, radius: 5cm) sphere(1cm) color: pink
confetti = scatter(n: 40, radius: 12cm, seed: 7) sphere(3mm) color: orange
```
Magic variables inside the copied shape: `i` (copy number from 0), `j` (grid
column), `a` (ring angle in degrees).
```
stairs = repeat(n: 6) cube(4cm, 2cm, 2cm) at (i * 4cm, i * 2cm, 0)
teeth  = ring(n: 20, radius: 34mm) cube(5mm, 10mm, 8mm) rotate (0, a, 0)
```
`ring` extras: `from:`, `to:` (partial arcs — bridges), `axis: x|y|z` (plane of
the ring; default `y` = flat on the ground).

### Parts
```
define wheel(r) = group(torus(r, 2cm), cylinder(r: r + 5mm, h: 2cm, material: rubber))
car = group(
  box(w: 24cm, d: 12cm, h: 4cm) at (0, 4cm, 0),
  use wheel(r: 3cm) at (-8cm, 0, -6cm),
  use wheel(r: 3cm) at (8cm, 0, -6cm),
  use wheel(r: 3cm) at (-8cm, 0, 6cm),
  use wheel(r: 3cm) at (8cm, 0, 6cm)
) material: red
```
Variables hold values: `h = 10cm` then `cube(w: 2cm, d: 2cm, h: h)`.
Math works with units: `at (i * 4cm, h / 2, 0)`, `sphere(r: 1cm + 2 * 3mm)`.

---

## 5. Materials & Colors

```
material: steel                 # scene default
material cup: ceramic           # one object
sphere 3cm material: gold       # inline parameter
sphere 3cm color: steelblue     # color parameter
color cup: ivory                # statement form
```
- If no color is given, the material's real color is used (steel looks like
  steel).
- If no material is given, the scene default is used; if that is unset,
  `plastic` (matte white).
- 27 materials with real physics: `05-MATERIALS.md` has the full table.
- 147 SVG colors + `#rrggbb`: see Appendix A.

---

## 6. Physics

Mass is always computed, never typed: `mass = mesh volume × material density`.
The HUD shows live totals while you type.

- `gravity: earth | moon | mars | off | <number>` (m/s²)
- `simulate: drop` — objects fall, bounce (material restitution), settle.
  Reports fall time and impact speed (v = √(2gh)).
- `simulate: float` — a water plane appears; objects that are lighter than
  water (average density < 1000 kg/m³) rise until exactly
  submerged-fraction = ρ_object/ρ_water (Archimedes), others sink.
  The waterline is solved on the **real mesh** by clipping, not by boxes.
- `simulate: collapse` — static stability check: center of mass vs support
  base per object and per stack; unstable stacks tip over.

Approximations in Phase 1 (documented, honest): no object-vs-object collision
in `drop`; `collapse` is a static test, not full dynamics.

---

## 7. `ask` — questions the model answers

| You ask | It answers (example) |
|---|---|
| `ask "mass?"` | `mass = 209 g (0.209 kg) — 87.1 cm³ × 2.40 g/cm³ ceramic` |
| `ask "mass of table?"` | names an object: `table: 26.9 kg` |
| `ask "volume?"` | `volume = 87.1 cm³` |
| `ask "surface area?"` | `surface = 310.2 cm²` |
| `ask "density?"` | average density incl. per-object breakdown |
| `ask "center of mass?"` | position in cm + visual marker |
| `ask "will it float?"` | `floats — average density 755 kg/m³ < water 1000 kg/m³ (76% submerged)` |
| `ask "how many objects?"` | counts entries |

Answers show the math so they teach physics instead of hiding it.

---

## 8. Export

| Format | Command | Notes |
|---|---|---|
| STL | `export stl "cup.stl"` | ASCII, mm, watertight |
| OBJ | `export obj "cup.obj"` | + `.mtl` material colors |
| GLTF | `export gltf "cup.gltf"` | GLTF 2.0, embedded buffers |
| USDZ | `export usdz "cup.usdz"` | AR-ready zip of USD ASCII (experimental) |
| PNG | `export png "view.png"` | 2× resolution render |
| OpenSCAD | `export scad "cup.scad"` | compiles to editable CSG code |

---

## 9. OTD 2.1 — decide, repeat, check

The syntax-expansion specification. Full grammar in `03-GRAMMAR.md`; this
section fixes the semantics.

### `if` / `else if` / `else` / `end`

```
if cond              cond: true/false, a comparison, a number (nonzero = true)
  …                  block form: statements until end / else
else if cond2
  …
else
  …
end
if cond: stmt        one-liner form (else: allowed) — no end needed
```

Truthiness: `true`/`false` literals, comparison results, numbers (`0` is
false), the words `on`/`yes` (true) and `off`/`no`/`none` (false). Anything
else is a friendly error ("if wants something true or false").

### `for i = from to to [by step]`

Inclusive on both ends. The counter may be plain (`1`), a length (`0cm`), or
an angle (`0deg`); the step matches. Default step is +1 (1 mm for lengths,
1 deg for angles); counting down (`for i = 5 to 1`) auto-reverses. The loop
letter is **local**: an outer variable of the same name is untouched.
100,000-pass cap with a friendly error.

### `for x in things`

Walks lists (`[1cm, 2cm]` — shapes allowed), ranges (`1..12`), positions
(component-wise) and words (letter by letter).

### `while cond … end`

Same truthiness as `if`. 100,000-pass cap: runaways report
"this while never ends — does something inside change the condition?"

### `break` / `continue`

Affect the innermost `for`/`while`. Outside a loop they are friendly errors.

### `x += e` / `x -= e` / `x *= e` / `x /= e`

Read, combine, write back. `x` must already exist (friendly error
otherwise). On objects, `+=` fuses (`blob += cube 1cm at (10cm, 0, 0)`
grows the named shape on stage) and the other operators are shape errors.

### Operators

| Op | Words | Semantics |
|---|---|---|
| `^` | — | plain numbers only (a length squared would be an area); right-assoc; `-2^2 = -4` |
| `%` | `mod` | remainder, plain numbers, zero divisor errors |
| `< > <= >=` | — | lengths/angles/numbers promote bare numbers with the scene unit; mixed dimensions error |
| `== !=` | `is`, `is not` | numbers, words, booleans, lists (element-wise), positions; mismatched kinds are simply unequal |
| membership | `in` | lists, substring in a word, ranges |
| `&&` | `and` | short-circuit |
| `\|\|` | `or` | short-circuit |
| `!` | `not` | flips truthiness |

### Ranges, indexing, slicing

`a..b` is inclusive, plain numbers, usable as a value (`r = 1..12`). `xs[0]`
counts from zero; negative counts from the end (`xs[-1]` is last); `xs[1..3]`
is an inclusive slice (clamped). Out of range is a friendly error that says
how long the list is. Positions and words index too: `p[0]`, `"cup"[0]` is `"c"`.

### `define name(params) … end` — block templates

Statements run in a private scope: parameters bind, locals never leak to the
program and never reach the stage. **The last line that makes or names a
shape is the template's value.** Both forms coexist:
`define wheel(r) = expr` (one line) and the block form above.

### `assert cond, "message"`

Passes silently; fails as a loud error with your message (or a default one).
The classic use is checking your own model math before exporting.

### `print "… {name} …"` — interpolation

Known names substitute their value: lengths as millimeters (`5cm` prints
`50mm`), angles as degrees, plain numbers trimmed. Unknown `{names}` stay
as written. Magic pattern letters (`i`, `j`, `a`) interpolate too.

### New functions (25 total)

| Function | Answer | Notes |
|---|---|---|
| `floor(x)` `ceil(x)` | plain | |
| `pow(a, b)` | plain | same rules as `^` |
| `log(x)` `ln(x)` `exp(x)` | plain | log is base 10 |
| `sign(x)` | -1, 0, 1 | |
| `hypot(a, b)` | plain | |
| `atan(x)` `asin(x)` `acos(x)` | **deg** | inverse trig |
| `atan2(y, x)` | **deg** | |
| `lerp(a, b, t)` | like a | units carry; t 0..1 |
| `clamp(x, lo, hi)` | like x | units promote |
| `count(xs)` | plain | lists, ranges, positions, words |
| `sum(xs)` | like elements | lists, ranges (`sum(1..100)` = 5050) |
| `avg(xs)` | like elements | |

### New units

`km` = 1,000,000mm · `yd` = 914.4mm · `um` = 0.001mm · `rad` = 57.2958deg.

### Lexer niceties

`1.5e3` scientific notation (units glue: `3e2mm` is 300mm) · `x = 1; y = 2`
semicolons separate statements (never inside brackets) · `#[ … ]#` block
comments span lines · `#[` never collides with `#rrggbb` colors.

---

## Appendix A — The 147 Named Colors

All classic SVG/CSS color names work. Format: `name (hex)`, grouped by hue.

**Reds/pinks:** indianred #CD5C5C, lightcoral #F08080, salmon #FA8072, darksalmon #E9967A,
lightsalmon #FFA07A, crimson #DC143C, red #FF0000, firebrick #B22222, darkred #8B0000,
pink #FFC0CB, lightpink #FFB6C1, hotpink #FF69B4, deeppink #FF1493, mediumvioletred #C71585,
palevioletred #DB7093, coral #FF7F50, tomato #FF6347, orangered #FF4500, plum #DDA0DD,
violet #EE82EE, orchid #DA70D6, fuchsia/magenta #FF00FF, mediumorchid #BA55D3,
mediumpurple #9370DB, rebeccapurple #663399, blueviolet #8A2BE2, darkviolet #9400D3,
darkorchid #9932CC, darkmagenta #8B008B, purple #800080, thistle #D8BFD8,
lavender #E6E6FA, lavenderblush #FFF0F5, mistyrose #FFE4E1

**Oranges/yellows:** orange #FFA500, gold #FFD700, darkorange #FF8C00, goldenrod #DAA520,
darkgoldenrod #B8860B, palegoldenrod #EEE8AA, yellow #FFFF00, lightyellow #FFFFE0,
lemonchiffon #FFFACD, lightgoldenrodyellow #FAFAD2, papayawhip #FFEFD5, moccasin #FFE4B5,
peachpuff #FFDAB9, navajowhite #FFDEAD, wheat #F5DEB3, burlywood #DEB887, tan #D2B48C,
rosybrown #BC8F8F, sandybrown #F4A460, chocolate #D2691E, saddlebrown #8B4513, sienna #A0522D,
brown #A52A2A, maroon #800000, darksalmon #E9967A

**Greens:** greenyellow #ADFF2F, chartreuse #7FFF00, lawngreen #7CFC00, lime #00FF00,
limegreen #32CD32, forestgreen #228B22, green #008000, darkgreen #006400, olive #808000,
olivedrab #6B8E23, darkolivegreen #556B2F, mediumaquamarine #66CDAA, darkseagreen #8FBC8F,
lightgreen #90EE90, springgreen #00FF7F, mediumspringgreen #00FA9A, seagreen #2E8B57,
mediumseagreen #3CB371, lightseagreen #20B2AA, palegreen #98FB98, mintcream #F5FFFA,
honeydew #F0FFF0, khaki #F0E68C, darkkhaki #BDB76B

**Cyans/blues:** aqua/cyan #00FFFF, lightcyan #E0FFFF, turquoise #40E0D0, mediumturquoise #48D1CC,
paleturquoise #AFEEEE, aquamarine #7FFFD4, powderblue #B0E0E6, steelblue #4682B4,
cornflowerblue #6495ED, deepskyblue #00BFFF, dodgerblue #1E90FF, lightblue #ADD8E6,
skyblue #87CEEB, lightskyblue #87CEFA, midnightblue #191970, navy #000080, darkblue #00008B,
mediumblue #0000CD, royalblue #4169E1, blue #0000FF, azure #F0FFFF, aliceblue #F0F8FF,
lightsteelblue #B0C4DE, slateblue #6A5ACD, mediumslateblue #7B68EE, darkslateblue #483D8B,
cadetblue #5F9EA0, cornsilk #FFF8DC, lightcyan #E0FFFF

**Grays/whites:** gray/grey #808080, lightgray/lightgrey #D3D3D3, darkgray/darkgrey #A9A9A9,
dimgray/dimgrey #696969, slategray/slategrey #708090, lightslategray/lightslategrey #778899,
darkslategray/darkslategrey #2F4F4F, silver #C0C0C0, gainsboro #DCDCDC, whitesmoke #F5F5F5,
white #FFFFFF, ghostwhite #F8F8FF, floralwhite #FFFAF0, seashell #FFF5EE, ivory #FFFFF0,
beige #F5F5DC, antiquewhite #FAEBD7, linen #FAF0E6, oldlace #FDF5E6, snow #FFFAFA,
black #000000, blanchedalmond #FFEBCD, bisque #FFE4C4

(*British `-grey` spellings are accepted as aliases of the `-gray` forms.*)

Also supported: `#rrggbb` and `#rgb` hex, e.g. `color: #1e90ff`.

---

# OTD 2.0 "DEEP" — the four new words (P0650–P1299)

## smooth — melt the corners (DEC tier)

```
blob smooth                      # 1 pass at default strength
rock = sphere(r: 3cm) smooth(n: 4, strength: 0.6)
```

One pass = one Taubin λ|μ pair (λ = 0.35·strength, μ = −λ — unconditionally
stable: the per-iteration gain 1 − λ²(α−1)² never exceeds 1). Weights are
the cotangent Laplacian over the welded half-edge topology — real discrete
differential geometry. Corners melt; low frequencies (the overall shape)
survive. `n:` clamps to 50, `strength:` to [0,1].

## subdiv — 4× the triangles, fairer surface (topology tier)

```
fine = sphere(r: 2cm) subdiv(n: 1)
```

Loop subdivision: every triangle splits into four; new edge points use the
⅜–⅛ mask, updated vertices use Loop's β(n) weight (β(3)=3/16, β(n) =
(1/n)(⅝−(⅜+¼cos(2π/n))²)); boundary curves follow Catmull–Rom. Honest
notes: sharp box corners round aggressively (no crease tags in 2.0), and a
lat-long sphere's poles are extraordinary vertices — for exact spheres use
`sphere` itself. `n:` clamps to 4 (that's 256× the triangles).

## blend — solder two solids together (implicit-field tier)

```
fused = blend(base, knob, gap: 8mm)
```

Both shapes become signed-distance fields (BVH closest-point + ray-parity
signs), fuse with the exponential smooth-min smin(a,b,k) = −ln(e^−ka +
e^−kb)/k (k = 1/gap), and re-surface by conforming marching tetrahedra.
The fillet radius ≈ `gap`. Volume is measured after, so `ask "mass?"`
stays honest. `res:` (default 34) is the lattice resolution (14–72).

## rope — a cable that hangs for real (XPBD tier)

```
cable = rope(from: (0, 30cm, 0), to: (40cm, 30cm, 0), thickness: 4mm, sag: 5cm)
```

The rest length comes from the inverted catenary equation; the initial
shape is the analytic catenary; XPBD dynamics then settle it over two
natural periods (the equilibrium is proven stable, not assumed). The tube
is swept along the *solved* polyline. `ask "cable sag?"` returns the
measured dip. Physics fact: an ideal cable's shape follows its **length**
(not gravity — gravity sets the tension), so `gravity: moon` ropes look
the same and pull less. Default sag: 6 % of the span.

## New questions for ask

- `ask "inertia?"` — per-part mass moment of inertia tensor about the
  center of mass (exact tetrahedron quadrature; cube test matches
  m(s²+s²)/12 to 1e−9), with the easy-spin-axis hint
- `ask "cable sag?"` — the measured XPBD dip in mm
- `ask "watertight?"` — per-part manifold report (closed 2-manifold?
  boundary edges?)
- `ask "triangle count?"` / `ask "mesh?"` — scene polycount

## The VM under the hood

Every numeric expression now compiles to **OTD-ASM** — 32-byte vector
instructions — and runs on a 256-register f64 machine with unit tags
(plain / mm / deg). Semantics are identical to the 1.0 tree-walk (same
promotion rules, same error strings; 2 000-case fuzz + the 1.0 corpus are
the regression gates). See the disassembly of any file with:

```
otd --vm-dump examples/gear-system.otd
```
