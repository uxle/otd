# OTD 6.0 Cheat Sheet — one page, that's all

> **Type on the left. Watch the right. Everything has a default — start anywhere.**

## Start here (60 seconds)

```
sphere                      # a ball
sphere 5cm                  # a bigger ball
sphere 3cm color: red material: gold
cube(w: 2m, d: 1m, h: 20cm)
torus 3cm                   # a donut
ask "mass?"                 # the model answers!
```

## The 12 shapes

`sphere r` · `cube s (w,d,h)` · `cylinder r,h (top: bottom: = cup shapes)` ·
`cone r,h` · `torus R, tube` · `pyramid base,h` · `prism sides,r,h` ·
`capsule r,h` · `wedge w,d,h` · `plane w,d` · `tube r, path:[points]` ·
`helix radius,pitch,turns,tube`

**Fancy:** `extrude [2D points] depth:` · `revolve [(radius,height)…]` ·
`sweep(path:[3D points], r:)` · `loft [section1, section2…]` · `text "WORD"` ·
`terrain(peaks:, seed:)` · `metaball [(x,y,z,r)…]` · `import "url.stl"`

## Put it somewhere

```
sphere 2cm at (10cm, 0, 0)          # stands here (y is up)
cube 4cm rotate (0, 45deg, 0)       # spins in place
sphere 2cm scale 2                  # twice as big
text "HI" mirror x
```

## Combine & cut

```
snowman = sphere(3cm) + sphere(2cm) at (0, 4cm, 0)      # fuse
cheese  = cube(10cm) - sphere(6cm) at (5cm, 5cm, 0)     # bite
core    = apple & cube 4cm                              # overlap only
cup     = cylinder(top: 4cm, bottom: 3cm, h: 10cm) - hollow(wall: 3mm)
add cup, handle                                         # fuse + tidy up
```

## Copy things

```
row    = repeat(n: 5, step: (4cm, 0, 0)) cube(1cm, 3cm, 1cm)
board  = grid(nx: 4, nz: 4, spacing: 5cm) sphere 1cm
petals = ring(n: 8, radius: 5cm) sphere 1cm color: pink
stars  = scatter(n: 30, radius: 10cm, seed: 7) sphere 3mm
stairs = repeat(n: 6) cube(4cm,2cm,2cm) at (i * 4cm, i * 2cm, 0)   # i = 0,1,2…
```

## Parts & materials

```
define wheel(r) = torus(r, 2cm) + cylinder(r: r, h: 1cm)
use wheel at (0, 0, 10cm) color: black

material: steel        # everything after becomes steel (until told otherwise)
material cup: ceramic  # just this one
color cup: ivory       # 147 color names: red, steelblue, burlywood… or #ff8844
```

**Materials:** iron steel stainless aluminum copper brass bronze gold silver
titanium zinc lead chrome tungsten wood oak pine teak glass plastic rubber
ceramic concrete marble fabric carbon — each with real weight & physics.

## Physics & questions

```
gravity: moon
environment: water      # medium: air (default) | vacuum | water | oil | density N
simulate: drop          # it falls & bounces
simulate: float         # water plane — Archimedes decides
simulate: collapse      # will it tip over?
simulate: splash        # drop into a tank — bobbing, ripples, the water shakes
simulate: settle        # REAL SOLIDITY: everything falls & lands, nothing floats
                        # unless buoyancy says so, nothing inserts into anything
simulate: solidity      # static audit: "SOLID — zero interpenetrations"
simulate: gas           # gases rise/sink, expand, DIFFUSE into one mixture
simulate: mix           # pour every liquid in: layers by density, miscible
                        # pairs become one phase (chemistry, with reasons)

# chemistry verdicts (no scene needed):
mix: water + oil        # refuse — polar vs non-polar, oil floats on top
mix: water + ethanol    # MIX — both polar, one phase, ρ ≈ 893 kg/m³
mix: hydrogen + oxygen  # REACT: 2 H2 + O2 -> 2 H2O (equation verified)

# the engine understands real things (pre-added intelligence, build once):
#   otd --ai "balance the equation CH4 + O2 -> CO2 + H2O"
#   otd --perceive bench.otd     # the engine SEES its scene via stereo vision
ask "mass?"             # volume × density, automatic
ask "will it float?"
```

## Export

`export stl "cup.stl"` · `export obj "cup.obj"` · `export gltf "cup.gltf"` ·
`export usdz "cup.usdz"` · `export png "view.png"` · `export scad "cup.scad"`

## Remember these 5 rules

1. **Y is up**, the ground is y = 0, shapes stand on the ground.
2. `at (x, y, z)` = *where it stands* (its bottom point).
3. `+` fuses, `-` cuts, `&` overlaps. `hollow` shells.
4. Units: `mm cm m km in ft yd um`, angles `deg rad`. No unit = **cm**.
5. `# starts a comment`. Semicolons optional — newlines end statements.


---
## OTD 2.0 "DEEP" — the four new words (58 keywords total)

```otd
rock = sphere(r: 3cm) smooth(n: 4)            # melt corners (real DEC math)
fine = sphere(r: 2cm) subdiv(n: 1)            # 4× triangles, fairer surface
fused = blend(base, knob, gap: 8mm)           # solder two solids together
cable = rope(from: (0, 30cm, 0),              # a cable that hangs for real
             to: (40cm, 30cm, 0), thickness: 4mm, sag: 5cm)
ask "cable sag?"   ask "inertia?"   ask "watertight?"
```

- `smooth(n:, strength:)` — cotan-Laplacian relaxation; corners melt,
  shape survives. Defaults n:1, strength:0.5
- `subdiv(n:)` — Loop subdivision; each level = 4× triangles. Max n:4
- `blend(a, b, gap:, res:)` — smooth-union fillet ≈ gap
- `rope(from:, to:, thickness:, sag:)` — XPBD physics; the sag is measured,
  not guessed. Cables keep their length within 0.03 %
- `pi` finally works (it was in the 1.0 docs but never wired — audit fix)


---
## OTD 2.1 "SYNTAX" — decide, repeat, check (83 keywords + 40 functions)

```otd
# decide
if r > 5cm
  cube 2cm
else if r > 2cm
  sphere 2cm
else
  cone 2cm
end
if x > 5cm: cube 1cm else: sphere 1cm     # one-liners

# repeat
for i = 0 to 15
  a = i * 22.5deg
  cube(3cm, 1cm, 9cm) at (12cm * cos(a), i * 3cm, -12cm * sin(a))
end
for r in [1cm, 2cm, 3cm]: sphere r          # walk a list, one line
for a = 0deg to 90deg by 15deg               # angles, lengths, numbers
  print "angle {a}"
end
while x < 30cm
  x += 10cm
end

# skip and leave
for i in 1..100
  if i mod 15 is 0
    continue
  end
  if i > 20
    break
  end
  cube 1cm
end

# check your own math
assert wall > 1mm, "walls too thin"
assert sum([1, 2, 3]) is 6

# multi-step templates
define wheel(r)
  rim = torus(radius: r, tube: 1cm)
  hub = cylinder(r: r / 2, h: 2cm)
  rim + hub
end
use wheel(r: 3cm)
```

**New operators** — `^` power · `%` or `mod` remainder · `< > <= >= == !=`
comparisons · `&&`/`and` · `||`/`or` · `!`/`not` · `is` / `is not` ·
`x in [list, word, 1..10]` membership · `x += 1cm` `-=`, `*=`, `/=` ·
divide-by-zero is a friendly error.

**New data** — `true` / `false` · ranges `1..12` (inclusive) · indexing
`xs[0]` (from zero), `xs[-1]` (last), slicing `xs[1..3]` · lists hold shapes.

**New functions (25 total)** — `floor ceil pow log ln exp sign hypot atan
atan2 asin acos lerp clamp` + `count sum avg` walk lists and ranges:
`sum(1..100)` is 5050.

**New units** — `km yd um rad`.

**New conveniences** — `print "r = {r}"` interpolation · `x = 1; y = 2`
semicolons · `1.5e3` scientific notation · `#[ block comments ]#`.


---
## OTD 2.2 "MECHANICS" — transparency, splash, animation (80 keywords)

```otd
# see-through materials: glass, water, ice, oil render transparent
box = cube 40mm - hollow(wall: 2mm) material: glass at (0, 20mm, 0)
bolt = cylinder(r: 4mm, h: 50mm) material: steel at (0, 25mm, 0)

# new liquids: water, oil (floats on water), mercury (liquid metal)
pool = cube(w: 30cm, d: 20cm, h: 8cm) material: water

# drop something in the water — it shakes: bobbing, ripples, damping
simulate: splash          # impact v=√(2gh), ω=√(ρw·g·A/m), waves c=√(gH)
```

**Animation from the command line** — spin named parts, render the rotation:

```bash
otd --png motor.otd f.png --spin rotor=90        # one phase (deg, @x/@y/@z)
otd --png motor.otd f.png --spin rotor --frames 8  # 8 frames of a full turn
otd --png engine.otd side.png --turn 35           # turntable the whole scene
```

**One-material parts library** (`library/parts/`): bolt_m6…m16, nut_m6…m16,
washer_m6…m12, gear_five … gear_twelve, piston_v2, connecting_rod, crankshaft,
flywheel, bearing_bronze, spring_coil, motor_assembly, pipe_flange, …

**Periodic table** (`library/elements/001-hydrogen … 118-oganesson`):
50 files per element — 30 theory/data documents + 20 runnable 3D samples with
real densities, true-scale molar volumes and equal-mass water comparisons.
