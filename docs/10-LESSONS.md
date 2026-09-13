# OTD — 20 Lessons: from "hello cube" to a full mechanical assembly

Every lesson's code runs in the OTD app (open `app/index.html`). Type it, watch
the right side, then try the challenge. Each lesson builds on the last — by
lesson 20 you assemble a robot arm from named parts.

---

## Lesson 1 — Hello Cube
**Goal:** your first 3D object in one word. OTD already knows a sensible size,
a color, and a place — the middle of the ground.

```
cube
```

**You see:** a gray box standing on the grid. Drag to orbit, scroll to zoom.
**Try this:** change `cube` to `sphere`, then `torus`, then `pyramid`.

## Lesson 2 — Sizes and Units
**Goal:** control how big things are. Units glue to the number: `5cm`, `30mm`,
`2m`. No unit means centimeters.

```
sphere 5cm
cube 30mm at (15cm, 0, 0)
sphere 3cm at (30cm, 0, 0)
```

**You see:** three objects in a row — a 5 cm ball, a 3 cm box, a 3 cm ball.
**Try this:** add `unit: mm` as your first line, then write `sphere 50` — the
same ball as `sphere 5cm`.

## Lesson 3 — Colors
**Goal:** paint your model. All 147 SVG color names work (`red`, `steelblue`,
`gold`…), plus hex codes like `#1e90ff`.

```
sphere 3cm color: red at (0, 0, 0)
sphere 3cm color: #1e90ff at (8cm, 0, 0)
sphere 3cm color: limegreen at (16cm, 0, 0)
```

**Try this:** find your favorite color name in the Help panel, and try a hex
code from a color picker.

## Lesson 4 — Materials (real weight included)
**Goal:** make things out of real stuff. Every material carries its true
density — the mass is already computed.

```
sphere 3cm material: gold
cube 5cm material: oak at (10cm, 0, 0)
sphere 3cm material: wood at (22cm, 0, 0)
ask "mass?"
```

**You see:** a shining gold ball and two wooden pieces.
**Try this:** swap `gold` for `lead`, `rubber`, `glass`. Watch the mass answer
change — real physics, automatic.

## Lesson 5 — Standing Somewhere
**Goal:** place things in 3D. Y is up. `at (x, y, z)` means *where the object
stands* — its bottom point. Stacking is natural.

```
cube 4cm
cube 3cm at (0, 4cm, 0)
sphere 2cm at (0, 7cm, 0)
```

**You see:** a snowman-tower: box, smaller box, ball resting on top.
**Try this:** build stairs: three cubes, each one step higher and further along.

## Lesson 6 — Spin and Grow
**Goal:** rotate and scale. `rotate` spins a shape around its own middle;
`scale` grows it.

```
cube 4cm rotate (0, 45deg, 0) color: tomato
sphere 2cm scale 2 at (12cm, 0, 0) color: turquoise
text "HI" size: 2cm at (0, 0, 8cm) color: steelblue
```

**You see:** a turned box, a doubled ball, real 3D letters.
**Try this:** rotate the letters too — chain `rotate (0, 30deg, 0)` onto the
text line.

## Lesson 7 — Names are Power
**Goal:** give things names and do math with them. Variables hold sizes;
named objects are shown in the scene.

```
ball_r = 4cm
gap = 10cm
ball1 = sphere ball_r at (0, 0, 0)
ball2 = sphere ball_r at (gap, 0, 0)
ball3 = sphere ball_r at (gap * 2, 0, 0)
```

**Try this:** change `ball_r` to `2cm` once — all three balls update.

## Lesson 8 — Fuse It (+)
**Goal:** join solids into one. `+` fuses, and the result keeps the left
object's material.

```
snowman = sphere(5cm) + sphere(3.5cm) at (0, 5cm, 0) material: plastic color: white
head = sphere(2.5cm) at (0, 7.5cm, 0) color: white
hat = cone(r: 1.5cm, h: 3cm) at (0, 9.5cm, 0) color: black
ask "mass?"
```

**You see:** a two-ball snowman body (fused into one solid), a head and a cone
hat. Fusing high-resolution shapes takes a moment — that's real solid math.
**Try this:** fuse the hat on too: `add snowman, hat`. Watch the mass change.

## Lesson 9 — Bite Into It (−)
**Goal:** subtract shapes. Anything on the right of `−` is removed from the
left. This is how holes and doors are made.

```
cheese = cube(10cm, 4cm, 10cm) material: plastic color: gold
cheese = cheese - sphere(r: 3cm) at (10cm, 2cm, 0)
cheese = cheese - sphere(r: 2cm) at (5cm, 4cm, 10cm)
ask "mass?"
```

**You see:** a golden block with two spherical bites — Swiss-cheese style.
**Try this:** cut a cylindrical hole straight through with
`- cylinder(r: 1.5cm, h: 12cm) at (5cm, 0, 5cm)`.

## Lesson 10 — Hollow Things
**Goal:** shells. `hollow` removes the inside and leaves walls. Cups, vases,
pots, houses.

```
cup = cylinder(top: 40mm, bottom: 30mm, height: 100mm) - hollow(wall: 3mm)
handle = torus(radius: 20mm, tube: 6mm) rotate (90deg, 0, 0) at (35mm, 25mm, 0)
add cup, handle
material cup: ceramic
color cup: ivory
ask "mass?"
```

**You see:** the famous OTD coffee cup — 8 statements, real ceramic weight
(about 206 g).
**Try this:** thin the wall to `2mm` — the cup gets lighter. Exactly as real
physics says.

## Lesson 11 — Donuts and Rings
**Goal:** the torus — a ring with thickness. Flat by default, like a donut on
a table.

```
donut = torus(R: 4cm, tube: 1.5cm) material: plastic color: chocolate
poolRing = torus(R: 6cm, tube: 1cm) at (16cm, 0, 0) color: deepskyblue
ask "volume?"
```

**You see:** a chocolate donut and a pool ring.
**Try this:** rotate a torus `rotate (90deg, 0, 0)` — it becomes a hoop facing
you (that's how cup handles are made).

## Lesson 12 — Repeat After Me
**Goal:** copies in a row. `repeat` places n copies, stepping by a direction.
The magic letter `i` counts from 0.

```
fence = repeat(n: 5, step: (4cm, 0, 0)) cube(1cm, 4cm, 1cm) material: wood
stairs = repeat(n: 6) cube(4cm, 2cm, 2cm) at (i * 4cm, i * 2cm, 12cm) material: concrete
```

**You see:** a fence of posts and a staircase (each step uses `i` to climb).
**Try this:** make the fence posts taller as they go: `cube(1cm, 2cm + i * 1cm, 1cm)`.

## Lesson 13 — Grids and Rings
**Goal:** 2D grids and circles of things. `grid` uses `i` and `j`; `ring`
uses `i` and the angle `a`.

```
orchard = grid(nx: 4, nz: 4, spacing: 8cm) group(sphere(2cm, material: wood), cone(r: 3cm, h: 4cm) at (0, 4cm, 0))
flower = ring(n: 8, radius: 5cm) sphere 1cm color: pink
center = sphere 1.5cm color: gold
```

**You see:** a grid of little trees and an 8-petal flower. The trees use
`group` — parts stay parts, so grids of groups stay fast.
**Try this:** make the petals rotate outward: add `rotate (0, a, 0)` after
`sphere 1cm`.

## Lesson 14 — Build a Wheel, Use It Four Times
**Goal:** reusable parts with parameters. `define` once, `use` anywhere.

```
define wheel(r) = group(torus(r, 1cm, material: rubber), cylinder(r: r + 4mm, h: 2cm, material: steel))

chassis = cube(w: 24cm, d: 12cm, h: 3cm) at (0, 4cm, 0) material: steel color: crimson
use wheel(r: 3cm) at (-8cm, 0, -6.2cm)
use wheel(r: 3cm) at (8cm, 0, -6.2cm)
use wheel(r: 3cm) at (-8cm, 0, 6.2cm)
use wheel(r: 3cm) at (8cm, 0, 6.2cm)
ask "mass?"
```

**You see:** a little car chassis on four rubber tires.
**Try this:** change every wheel at once by editing one number — the `define`.

## Lesson 15 — Spin a Profile (vases and chess pieces)
**Goal:** revolve a 2D outline around an axis. This is how wood lathes,
vases, pawns and rooks are made. Points are `(radius, height)` pairs.

```
vase = revolve [(0, 0), (5cm, 0), (5cm, 1cm), (3cm, 3cm), (4cm, 12cm), (2cm, 14cm)] material: glass
pawn = revolve [(0, 0), (13mm, 0), (13mm, 4mm), (9mm, 8mm), (9mm, 28mm), (14mm, 33mm), (6mm, 36mm)] material: wood at (12cm, 0, 0)
ask "mass?"
```

**You see:** a glass vase and a wooden chess pawn.
**Try this:** hollow the vase: wrap it as `vase - hollow(wall: 2mm)`.

## Lesson 16 — Flat Shapes, Raised
**Goal:** two more builders. `extrude` lifts a flat outline into 3D; `tube`
runs a pipe through 3D points.

```
plate = extrude [(0, 0), (12cm, 0), (12cm, 12cm), (0, 12cm)] depth: 5mm material: steel
pipe = tube(r: 8mm, path: [(0, 2cm, 0), (0, 20cm, 0), (10cm, 28cm, 0)]) material: copper at (20cm, 0, 0)
```

**You see:** a steel plate (drawn as a square, raised 5 mm) and a copper pipe
bending through space.
**Try this:** extrude a triangle: `[(0,0), (10cm,0), (5cm,8cm)]`.

## Lesson 17 — Springs and Helixes
**Goal:** the helix — DNA, springs, drill bits. `pitch` is the height of one
full turn; total height = `turns × pitch`.

```
spring = helix(radius: 2cm, pitch: 8mm, turns: 6, tube: 2mm) material: steel
drill = helix(radius: 1cm, pitch: 2cm, turns: 5, tube: 4mm) material: steel color: silver at (10cm, 0, 0)
ask "height?"
```

**You see:** a tight steel spring and a wide drill-bit spiral.
**Try this:** make a slinky: `radius: 4cm, pitch: 5mm, turns: 10`.

## Lesson 18 — Ask the Model Anything
**Goal:** the science layer. The model answers questions about itself.

```
ball = sphere 5cm material: gold
plank = cube(w: 90cm, d: 60cm, h: 4cm) material: oak at (0, 5cm, 0)
ask "mass?"
ask "volume?"
ask "density?"
ask "center of mass?"
ask "will it float?"
ask "melting point?"
```

**You see:** answers in the console, each showing the math — e.g.
"523.6 cm³ × 19.32 g/cm³".
**Try this:** ask `"mass of plank?"` — naming one object narrows the answer.

## Lesson 19 — Drop It, Float It, Topple It
**Goal:** real simulations. Place objects in the air; press Run (▶) or the
physics buttons.

```
ball = sphere 3cm material: rubber at (0, 40cm, 0)
ball2 = sphere 2cm material: steel at (6cm, 60cm, 0)
log = cube 10cm material: oak at (14cm, 30cm, 6cm)
simulate: drop
```

**You see (press Run ▶):** the rubber ball bounces high, the steel ball barely,
the oak log thuds — all from real material physics.
**Try this:** replace `simulate: drop` with `simulate: float` — Archimedes
decides who floats. Then `gravity: moon` and drop again.

## Lesson 20 — Graduation: the Robot Arm
**Goal:** combine everything — parts, booleans, placement, materials — into a
machine. This is a full mechanical assembly.

```
scene "Robot Arm"
material: steel

base = cylinder(r: 9cm, h: 3cm) color: dimgray
column = cylinder(r: 3.5cm, h: 30cm) at (0, 3cm, 0)
shoulder = sphere(r: 4.5cm) at (0, 33cm, 0) color: gold

define arm(len, r) = capsule(r: r, h: len)

lowerArm = arm(28cm, 3cm) rotate (0, 0, -60deg) at (0, 36cm, 0)
elbow = sphere(r: 4cm) at (14cm, 48cm, 0) color: gold
upperArm = arm(24cm, 2.5cm) rotate (0, 0, -25deg) at (12cm, 52cm, 0)
wrist = sphere(r: 3cm) at (24cm, 62cm, 0) color: gold

gripL = cube(1.5cm, 9cm, 3cm) at (26cm, 62cm, -2.6cm)
gripR = cube(1.5cm, 9cm, 3cm) at (26cm, 62cm, 2.6cm)

ask "mass?"
ask "how many objects?"
```

**You see:** a golden-jointed robot arm with a two-finger gripper — 33 kg of
steel, built from 13 lines.
**You graduated.** Where to next: the 10 example models, the cheat sheet, and
exporting your creations to STL for 3D printing.


---

## Lesson 21 — Melt it smooth (2.0)

Sharp corners are 1.0's default. Real objects are softer. `smooth` runs
the cotangent Laplacian — the same math mesh-sculptors use:

```otd
scene "Pebble"
rock = sphere(r: 3cm) scale (1.4, 0.8, 1) smooth(n: 5)
material: concrete
color: dimgray
ask "mass?"
```

Try `n: 0`, `n: 2`, `n: 8` — watch the corners melt while the volume stays
in class. `strength:` (0 to 1) controls how far each pass pulls.

## Lesson 22 — Physics ropes and solder blends (2.0)

Two words that make scenes *behave*:

```otd
scene "Swing Bridge"
towerL = cube(w: 3cm, h: 20cm, d: 3cm) at (0, 0, 0)
towerR = cube(w: 3cm, h: 20cm, d: 3cm) at (40cm, 0, 0)
cable = rope(from: (0, 19cm, 0), to: (40cm, 19cm, 0), thickness: 2mm, sag: 4cm)
blob = blend(towerL, sphere(r: 4cm) at (2cm, 18cm, 0), gap: 1cm)
material cable: steel
ask "cable sag?"
ask "watertight?"
```

The cable hangs in a true catenary (0.1 % of the physics answer), and the
blend solders the sphere onto the tower with a smooth fillet. Change
`sag:` and re-ask — the answer is re-measured from the solved rope, never
guessed.


## Lesson 23 — Decide and repeat (2.1)

The 2.1 syntax lets a model make choices and do work on its own:

```otd
for i = 0 to 7
  if i is 0
    cube(20cm, 3cm, 8cm) color: crimson at (10cm, i * 4cm, 0)
  else
    cube(20cm, 3cm, 8cm) color: tan at (10cm, i * 4cm, 0)
  end
end
column = cylinder(r: 2cm, h: 36cm) color: dimgray
```

Four lines, eight steps, one column. `if` picks, `for` repeats, and the loop
letter `i` belongs to the loop. Try the tour in the
`staircase` and `orbit-tower` examples — angles step with
`for a = 0deg to 90deg by 15deg`, shapes grow with `spire += cone(...)`,
and `print "count {count(xs)}"` tells you what you built.

`while` keeps going as long as something is true; `break` leaves early,
`continue` skips a pass; `assert x > 0, "message"` checks your own math.
