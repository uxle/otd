# BUGFIXES — OTD 2.2.0 "MECHANICS"

## B17–B21 — silent-wrongness bugs reported by an AI user (fixed in OTD4.0)

These are the five "it compiled but it was wrong" bugs an AI user hit when
using OTD3 without a human checking each step. Each fix preserves backward
compatibility — every existing OTD3 file still compiles unchanged.

### B17 — `hollow()` defaults to an open top (silent)

**Symptom:** `sphere 4cm - hollow(wall: 2mm)` quietly opens the top of the
sphere, surprising the user who expected a sealed shell.

**Fix (P2400, `src/world/eval.rs::eval_hollow`):** every `hollow()` call
without an explicit `open:` argument now emits an INFO line:
`hollow() default: open top (write open: none for a sealed shell, or open: bottom to breach the base)`.
The default itself is unchanged (backward compat).

### B18 — `group()` silently double-counts mass (7.0g → 14.0g)

**Symptom:** `g = group(a, b)` produces a group whose mass equals `a.mass + b.mass`,
but `a` and `b` are still in the scene as separate parts — so `total_mass_g`
double-counts them.

**Fix (P2410, `src/world/eval.rs::eval_call`):** when `group()` is called with
an Ident argument referring to an existing entry, that entry is now marked
hidden. An INFO line reports the re-parenting: `group() re-parented a, b —
they are now hidden as separate parts (the group is the single combined part)`.

### B19 — `rotate()` pivots on the bbox center, undocumented

**Symptom:** `tooth = cube(2cm, 0.5cm, 5cm) rotate (0, 60deg, 0) at (5cm, 0, 0)`
in a ring of 8 produces "petals" instead of radial teeth — because the pivot
is the tooth's bbox center, not the world origin where the ring center is.

**Fix (P2420, `src/lang/parser.rs::parse_mods`, `src/world/eval.rs::apply_mod`):**
the parser now accepts `rotate (angles) pivot (x, y, z) | pivot: origin | pivot: center`.
A new `Mod::RotateWithPivot` variant + `PivotSpec` enum carry the explicit
pivot. When `pivot:` is NOT specified, an INFO line reports the default:
`rotate: pivot = bounding-box center (X, Y, Z) mm — write pivot (0,0,0) to
pivot around the origin, or pivot: center to be explicit`.

### B20 — string interpolation silently leaves `{expr}` as literal text

**Symptom:** `print "the radius is {r}"` with `r` undefined prints the literal
string `the radius is {r}` — no error, no warning.

**Fix (P2430, `src/world/eval.rs::interpolate`):** an unrecognized `{name}`
now emits a WARN line: `print interpolation 'r' is not a known variable —
left as literal text. Define it first (r = 5cm) or fix the typo.` The text
is still emitted (backward compat) so existing programs that depend on the
literal-output behavior still work.

### B21 — overlapping shapes only emit a vague warning (no fix suggestion)

**Symptom:** two cubes placed at overlapping positions produce the warning
`objects overlap each other — mass counts the overlap twice until you fuse
them with add`. No specific pair, no depth, no axis, no fix.

**Fix (P2440, `src/world/eval.rs::finalize_stats` + new `overlap_audit`):**
the warning now names the pair and the depth:
`objects a and b overlap by 25.00 mm — mass counts the overlap twice until
you fuse them with add (run strict: overlap to make this an error, or
simulate: settle / simulate: solidity to fix/check)`. The new `strict:
overlap` mode turns this into a hard Error. The new `overlap: check`
statement runs an explicit audit that reports every interpenetrating pair
with the penetration depth, the shallowest escape axis (X/Y/Z), and a
concrete fix: `move b along X by 25.00 mm, or fuse with add a, b if they
are meant to be one part`.

---

## B15 — CSG cut loses `material:` / `color:` (fixed in 2.2.0)

**Symptom:** `hull = cube(...) - hollow(wall: 5mm, open: top) at (0, 8cm, 0) material: wood`
reported **plastic** and the flagship wooden boat *sank* (1050 kg/m³ default density
instead of 700).

**Root cause:** the postfix-modifier parser binds trailing `material:`/`color:`
to the nearest shape term — the cutting TOOL on the right of `-`. The tool is
consumed by the subtraction and its material never reaches the result.

**Fix (P1415, `src/world/eval.rs`):** for `-` and `&`, finish mods (Material/Color)
found on a postfix tool are hoisted and applied to the RESULT; the tool itself is
evaluated without them. The `a - hollow(...)` sugar also matches through tool
modifiers now (e.g. `a - hollow(wall: 2mm) at (0, 2cm, 0) material: glass`).

**Verify:** the wooden boat now reports 101.1 g (wood) and floats with 70% submerged.

## B16 — metals render almost black (fixed in 2.2.0)

**Symptom:** every metal material (steel, aluminum, gold, …) rendered as a
near-black shape with a single highlight. Visible in the 2.1.1 gallery renders
too (`screenshots-library/gear-pair.png`) — pre-existing, not caused by 2.2 work.

**Root cause:** Cook-Torrance gives metals kd = 0 (no diffuse). The renderer has
no environment map, and its ambient + fill light terms only multiplied the
diffuse (kd) path — so metals received no ambient/fill energy at all.

**Fix (P1430, `src/render/raster.rs`):** metals now receive a sky-tinted
environment term `metal_env = amb·2.2 + d2·2.5 + 0.07` multiplied by their base
color, alongside the GGX highlight. Iron is grey, brass golden, copper orange,
gold yellow, steel bright. Material chart verified with an 8-metal side-by-side.

## Legacy bugs B1–B14

B1–B7 (near-plane clipping, cube argument order, weight/weigh matching, CSG open
edges, deferred simulate, float formatting, `box` alias) and B8–B14
(documentation drift found by mass generation) are documented in the 2.1.0 /
2.1.1 sections of `CHANGELOG.md` and in `BUGFIXES.md` of the release archives.
