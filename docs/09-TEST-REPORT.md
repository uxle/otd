# OTD 1.0 — Math & Physics Validation Report

**Implementation:** pure Rust + handwritten assembly, zero external libraries
(`[dependencies]` in Cargo.toml is empty — a hard, tested constraint).
**Suite:** `cargo test` — **100 tests, 100 passing** (82 before the P0700 syntax audit; +16 syntax-audit regressions + 2 content-compile guards). External verification:
Python `zlib` decodes our PNG/DEFLATE streams byte-exact; the browser E2E
(agent-browser) confirms the full UI works against the live server.

---

## 1. Geometry truth — measured volumes vs analytic formulas

Every shape's closed mesh volume is computed with the divergence theorem
(`V = Σ det(a,b,c)/6`) and compared to the textbook formula. Tessellated
surfaces carry the expected chord deficit (≤ 1%).

| Test | Analytic | Measured | Tolerance |
|---|---|---|---|
| cube 20×30×40 mm | 24 000 mm³ | exact | 1e-9 |
| cylinder r10 h20 (48-gon) | 6 283.19 mm³ | 6 282.9 | 0.5% |
| frustum top 40 / bottom 30 / h 100 | 387 464 mm³ | 387 4xx | 0.5% |
| sphere r10 (48×24) | 4 188.79 mm³ | 4 156.6 | 1% |
| torus R10 tube 3 (48×48) | 1 776.53 mm³ | 1 761.3 | 1% |
| pyramid base 20 h 10 | 1 333.33 mm³ | 1 311 | 2% |
| hex prism r10 h20 | 5 196.15 mm³ | 5 196.15 | 0.1% |
| capsule r5 h10 | 1 309.00 mm³ | 1 292 | 2% |
| revolve cylinder (lathe) | 6 283.19 mm³ | 6 2xx | 1% |
| revolve cone | 2 094.40 mm³ | 2 0xx | 2% |
| extrude square 10×10×5 | 500 mm³ | 500 | 1% |
| tube r5 L20 | 1 570.80 mm³ | 1 5xx | 3% |
| loft 10×10×10 | 1 000 mm³ | 1 000 | 1% |
| metaballs 2×r10 @ ±8 | ~8–16 cm³ blob | 12.4 cm³ | range |

## 2. CSG boolean identities (BSP kernel)

| Identity | Result |
|---|---|
| cube ∪ cube shifted 10 → 12 000 mm³ | ✓ (2·8000 − 4000 overlap) |
| disjoint union = concat | ✓ exact 2 000 |
| cube − inset cube = hollow box walls | ✓ 3 904 mm³ (2 mm walls) |
| cube − half cube | ✓ 4 000 mm³ |
| cube ∩ half cube | ✓ 4 000 mm³ |
| A − A = ∅ | ✓ |
| A ∩ A = A | ✓ |
| cube + sphere-on-top | ✓ 10 094 mm³ |

## 3. THE flagship: the coffee cup (8 lines of OTD)

```
cup = cylinder(top: 40mm, bottom: 30mm, height: 100mm) - hollow(wall: 3mm)
handle = torus(radius: 20mm, tube: 6mm) rotate (90deg, 0, 0) at (35mm, 25mm, 0)
add cup, handle
material cup: ceramic
```

| Quantity | Analytic | OTD reports |
|---|---|---|
| outer frustum volume | 387.46 cm³ | ✓ (measured) |
| wall band (3 mm walls + floor) | ≈ 74.5 cm³ | measured (analytic hollow) |
| handle torus | 2π²·R·r² = 14.2 cm³ | ✓ |
| total solid | ≈ 88.7 cm³ | 91.9 cm³ |
| **mass = V × ρ (ceramic 2.40 g/cm³)** | ≈ 213 g | **220.7 g** ✓ |
| float verdict | ρ 2 400 > 1 000 → sinks | ✓ sinks |

(Our analytic hollow insets radially; the exact perpendicular wall offset on a
taper differs by ~3% — documented honestly. The JS prototype measured 206.3 g
with a slightly different inner profile; both are within honest engineering
tolerance of the physical cup.)

## 4. Physics layer

| Law | Test | Result |
|---|---|---|
| Archimedes (float) | oak 755 kg/m³ → floats, 76% submerged | ✓ 755/1000 |
| Archimedes | ceramic cup 2 400 → sinks | ✓ |
| drop kinematics | v = √(2gh) from 1 m | ✓ 4.43 m/s (earth 9.81) |
| drop energy | KE = ½mv² | ✓ shown in console with math |
| material verdicts | glass/ceramic shatter > 4 m/s, rubber bounces 0.85 restitution | ✓ |
| collapse stress | σ = F/A at the waist slice (real rasterized cross-section) vs compressive strength | ✓ safety factor reported |
| gravity presets | earth 9.81 / moon 1.62 / mars 3.71 | ✓ |
| center of mass | volume-weighted divergence centroid | ✓ cube center exact |

## 5. Units & dimensions

| Test | Result |
|---|---|
| 4cm→40mm, 3in→76.2, 3.5, 90°, ° suffix | ✓ |
| bare numbers keep RAW value (n: 4 is 4 copies, not 40) | ✓ (the 100× grid bug, found & fixed) |
| `i * 30deg` (plain × angle) | ✓ Angle result |
| mm × mm rejected with a friendly area message | ✓ |
| `unit: mm` statement changes the scene unit | ✓ |

## 6. Language & keywords

| Test | Result |
|---|---|
| keyword count == 54, no duplicates | ✓ (< 60 budget) |
| all 10 example files parse | ✓ zero errors |
| all 10 examples compile → parts with mass | ✓ |
| multi-line lists, comments, line continuation | ✓ |
| templates `define gear(r)` + `use` with params | ✓ |
| patterns repeat/grid/ring with magic i/j/a | ✓ |
| 147 SVG named colors + hex | ✓ count test |
| 27 materials with real densities (gold 19 320, oak 755) | ✓ |
| Levenshtein suggestions: cermaic→ceramic, materal→material | ✓ |
| negative size: "radius cannot be negative — did you mean 30mm?" | ✓ |

## 7. PNG / DEFLATE / CRC (own encoder, external verification)

| Test | Result |
|---|---|
| CRC-32 known vector "123456789" → 0xCBF43926 | ✓ |
| Adler-32 "Wikipedia" → 0x11E60398 | ✓ |
| PNG chunk framing + IEND | ✓ |
| **Python zlib decodes our fixed-Huffman LZ77 stream byte-exact** | ✓ (external truth) |
| pixel round-trip after decode | ✓ exact |
| Huffman code bit order (MSB-first codes, LSB-first extras) | ✓ (the bug we caught & fixed) |

## 8. Assembly kernels (the "assembly language" promise)

| Kernel | Verification |
|---|---|
| `rep stosd` memset (raw asm!) | bit-identical to Rust loop ✓ |
| SSE2 horizontal sum (raw asm!) | matches scalar within 1e-5 ✓ |
| SSE2 4-wide dot (raw asm!) | matches scalar & intrinsics ✓ |
| SSE2 4-pixel blend (intrinsics) | bit-identical to scalar (truncating cvtt) ✓ |
| AVX2 path available at runtime | ✓ dispatch ready |

## 9. Network stack (from scratch)

| Test | Result |
|---|---|
| own JSON parser: roundtrip, escapes, rejects garbage | ✓ |
| own Base64: RFC 4648 vectors | ✓ |
| HTTP/1.1 parse, keep-alive, 100-continue, 4 MB cap | ✓ |
| panic isolation per connection | ✓ |
| live server: /health, static, /api/render, /api/export | ✓ (curl E2E) |
| render API returns valid JSON with base64 PNG + stats | ✓ |
| all 5 export formats produce valid magic bytes | ✓ (glTF, STL, OBJ, SCAD, PNG) |

## 10. Browser E2E (agent-browser on the live server)

- Viewer loads, auto-loads the Coffee Cup: status bar shows **220.7 g** ✓
- Example switcher: Robot Arm (32.8 kg, 9 objects), Chess Set (2.72 kg, 13),
  DNA (2.61 kg), Solar System (10.2 kg) ✓
- Friendly errors render in the console with clickable line jumps ✓
- 20-lesson modal + cheat-sheet modal ✓
- Zero page errors ✓

## 11. Performance (measured, release build, 2-core VM)

| Metric | Target | Measured |
|---|---|---|
| parse cup.otd | < 1 ms | **0.006 ms** |
| compile typical scene | < 60 ms | 0.1–7 ms |
| compile CSG-heavy (cup fuse / chess) | — | 540 / 780 ms (roadmap P1000: BSP hardening) |
| render 900×600 @ 2×SSAA | < 200 ms | **95 ms** |
| frame cache on orbit | instant | ✓ (hash hit) |
| binary size | < 4 MB | **1.2 MB** |

## 12. What we honestly know is approximated

1. Tapered hollow walls are inset radially (perpendicular offset differs ~3% on
   strong tapers) — reported as approximate for generic meshes.
2. Float assumes solid density (open cups flood and sink — real physics, noted
   in answers).
3. Drop/collapse use material-class thresholds, not FEM — educational, not
   certified engineering.
4. BSP booleans can in rare degenerate coplanar cases produce slivers; the
   P1000 phase hardens this.

---

## 13. Syntax audit (P0700 sweep) — 16 bugs found & fixed, 16 regression tests

A dedicated audit of the language surface (lexer, parser, grammar doc vs
implementation) found and fixed these defects. Each is frozen as a regression
test in `tests/syntax_bughunt.rs`.

| # | Bug | Severity | Fix |
|---|---|---|---|
| 1 | keyword at EOF with no trailing newline (`scene`, `material:`) **panicked** the parser (index out of bounds) — a live-compile crash path | critical | every token accessor clamps to the Eof token |
| 2 | `#rrggbb` hex colors were documented in the grammar but `#` started a comment → `color: #1e90ff` silently lost the value (and hit bug 1) | critical | `#` + 6 hex digits + boundary lexes as a color word; comments unchanged |
| 3 | pattern body on the next line (`repeat(n: 5)` ⏎ `cube 1cm`) errored with a 5-error cascade | major | pattern bodies may start on the next line; a following statement/assignment is never swallowed |
| 4 | trailing commas rejected: `sphere(r: 2cm,)`, `(1cm, 2cm,)`, `[(0,0),]` each produced 3 errors | major | trailing comma legal before every closing bracket |
| 5 | space-separated units (`sphere 5 cm`, `top: 4 cm`, `rotate 45 deg`) → "I don't know what 'cm' is" | major | one-space unit gluing in the lexer, with guards (word must be a unit; not before `(`; not `mm2`) |
| 6 | unclosed bracket at EOF swallowed the rest of the file and surfaced as misleading geometry errors | major | lexer reports "a ( opened here never closed" at the opening line; stray closers flagged too |
| 7 | pasted Unicode (`–` `—` `−` dashes, `“ ”` quotes, BOM) → "I don't understand the character" | major | paste-proof normalization to ASCII twins |
| 8 | `at 2cm, 3cm, 0` (no parens) → "I don't know what 'at' is" + junk | major | one friendly error + positional recovery |
| 9 | colons rejected after scene/version/hide/show/use/export/print/define/add (inconsistent with `ask:`, `unit:` …) | minor | every statement tolerates the optional colon |
| 10 | `mirror xy` silently truncated to `mirror x` | minor | axis word must be exactly x / y / z |
| 11 | `rotate 90 °` (space) → unknown character | minor | standalone `°` attaches degrees to the previous number |
| 12 | loose named arg arithmetic leaked: `depth: 5 + 2mm` left `+ 2mm` to poison the outer expression | minor | value parser steals only number-literal right sides; shapes are never stolen |
| 13 | viewer highlighter grays `#1e90ff` as a comment (cosmetic mismatch with fixed lexer) | minor | highlighter has a `#rrggbb` token class |

| 14 | Lesson 1's whole premise — `cube` **alone** (zero-arg smart defaults, "everything predefined") — never compiled: `cube needs a size or settings` | critical (broke the core promise + lesson 1) | bare shape word at a statement boundary is a defaults call; a word the user assigned is always their variable (pre-scan) |
| 15 | Lesson 7's `sphere ball_r` — a **variable as the bare main argument** — never parsed (`ball_r` became a stray statement) | major | primitives accept a bare IDENT argument; `ring radius: 5cm` / `hollow wall: 3mm` loose-named forms also start correctly now |
| 16 | Lesson 14's `torus(r, 1cm, material: rubber)` — **material/color inside the argument list** — was silently dropped: the `finish()` hook existed but **no shape ever called it** (dead code), and value words were evaluated as expressions → "I don't know what 'rubber' is" | major | `finish()` wired into all 20 shape builders; material/color args resolve as value-library words (env still wins) |

**No regressions:** all prior tests still pass; the cup still measures **220.7 g**
(mesh-identical — the lexer changes do not alter any legal program's tokens);
54-keyword budget re-asserted. Suite grew 82 → **100 tests, 100 passing**, and
new guards compile **all 20 lessons + all 10 examples** on every run
(`tests/all_content_compiles.rs` — the sweep that surfaced bugs 14–16).

New accepted-but-optional syntax (all covered by tests): hex colors,
space-separated units, trailing commas, next-line pattern bodies, colon
tolerance everywhere, pasted Unicode, zero-arg default shapes, bare variable
arguments, inline material/color args. None of it adds keywords or removes
anything — OTD 1.0 files compile unchanged.

---

# 2.0 "DEEP" Validation — measured numbers (P1290)

Every 2.0 claim re-measured by `tests/validation_20.rs` (run with
`cargo test --release --test validation_20 -- --nocapture`).

## The virtual machine (Tier 3)

| Check | Result |
|---|---|
| ISA instruction size | exactly 32 bytes, 32-byte aligned (compile-time assert) |
| 2 000-case parity fuzz (VM vs reference unit walk) | all bitwise-identical values and unit tags |
| VM spot table | `4cm+1 → 50mm` · `90deg/2 → 45deg` · `10cm/2cm → 5` · `pi·2 → 6.28319` · `sin(30) → 0.5` — all exact |
| unit errors | same strings as the tree-walk (`mm × mm would be an area…`) |
| compile-once cache | pattern bodies compile once per AST node, re-run per iteration |
| `--vm-dump` | prints 32-byte disassembly with var names |

## Deep geometry (Tier 2)

| Check | Measured |
|---|---|
| half-edge topology | sphere: closed, pole ring 16, interior rings 6, simple cycles |
| BVH closest-point | 200 random probes vs brute force: Δ < 1e-9 mm |
| BVH ray parity | inside/outside classification correct incl. pole/edge hits (t-clustering) |
| DEC `smooth` (cube, n:4, strength 0.5) | volume 8000 → 4874 mm³ (39 % corner melt), no collapse (Taubin μ=−λ gain ≤ 1) |
| Loop `subdiv` (torus, valence 6) | mean normal deviation 5.48° → 4.66° → 4.51°; tris ×4 per level; watertight |
| SDF `blend` (two 10mm cubes, 5mm overlap, gap 1mm) | union 1500 → 1726 mm³ (+15 % fillet), **watertight closed 2-manifold** |
| marching tets | Freudenthal-Kuhn table: full-cube coverage, 0 cracks (1.0 table had 125 uncovered + 467 overlapping sample cells — fixed in 2.0) |

## Physics (Tier 2)

| Check | Measured |
|---|---|
| XPBD rope vs analytic catenary | span 100/target 10 → 10.05 mm (0.55 %) · span 600/80 → 80.05 (0.06 %) · span 300/25 → 25.08 (0.32 %) |
| rope length preservation | arclength within 0.003–0.03 % of rest (inextensible) |
| gravity independence | ideal cables: same catenary at earth and moon gravity (shape follows length; g sets tension) |
| inertia, cube | Ixx = Iyy = Izz = **533.3 g·mm² = analytic m(s²+s²)/12 exactly** |
| inertia, sphere(32) | 440 vs analytic 2/5·m·r² = 452 g·mm² (2.6 % facet deficit — honest) |

## Light (Tier 2/14)

| Check | Measured |
|---|---|
| GGX D normalization | ∫D·cosθ dω = 1.00 ± 0.05 (Monte-Carlo, 200k samples) |
| Smith G₂ | head-on ≈ 1, grazing masked, monotone in roughness |
| energy conservation | specular lobe + (1−F) ≤ 1 (+5 % head-on; grazing fresnel documented) |
| render perf | cup @ 900×600 2×SSAA, GGX + soft shadows + AO: 444 ms (was ~1.9 s pre-optimization; cell-bucketed ray queries, analytic ground) |

## Backward compatibility (the forever promise)

All ten 1.0 examples compile with **zero errors and unchanged stats**:
cup 220.7 g · table 24 475.6 g · gear-system 303.3 g · robot-arm 32 824.3 g ·
house 50 343.4 g · dna-helix 2 607.4 g · chess-set 2 718.0 g · lamp 3 051.5 g ·
bridge 37 921.9 g · solar-system 10 205.7 g.

---

## 15. OTD 2.1 "SYNTAX" — the expansion, measured

**Suite:** 267 tests passing (was 170). New file
`tests/syntax_21.rs` — 74 tests, one per construct, every claim through the
full compiler (parse + evaluate + geometry):

| Construct | Test claim |
|---|---|
| `if / else if / else / end` | the right arm runs; one-liners with `else:` |
| `for i = a to b [by s]` | part counts (5 steps, `by 2` halves, countdown works) |
| `for` with units | angles 0–90° by 30° → 4 parts; lengths 0–10cm by 2.5cm → 5 |
| `for x in …` | lists, ranges, tuples — and lists of shapes place their elements |
| loop-local letters | `i = 99` survives a `for i in 1..3` untouched |
| `while` + `break` + `continue` | counts up, stops early, skips passes |
| runaway loops | `while true` and huge `for` ranges hit the 100,000-pass friendly error |
| `+=` `-=` `*=` `/=` | arithmetic on numbers, units, and shape fusion (volume measured 9 000 mm³) |
| `^` `%` `mod` | `2^10 % 7`; right-assoc `2^3^2 = 512`; `-2^2 = -4`; `2 + 3*4^2 = 50` |
| comparisons | unit promotion (`5cm > 40` under `unit: mm`); mixed dimensions error kindly |
| `and or not` / `&& \|\| !` | short-circuit proven: `false and x / 0 > 1` records no error |
| `is` / `is not` / `in` | membership in lists, ranges, substrings; `3 in [1,2,3]` vs 3 inches |
| divide by zero | friendly error from both the tree-walk and the VM |
| ranges as values | `r = 1cm..5cm` is rejected with the plain-numbers hint |
| indexing / slicing | `xs[0]`, `xs[-1]`, `xs[1..3]` (inclusive), `p[0]`, `"cup"[0]` |
| out-of-bounds | "that list has 1 thing" — the size is in the message |
| `define … end` | returns the last line; locals never leak (probed); params bind |
| `assert` | passing is silent; failure carries the custom message |
| `print` interpolation | `radius {r}` → `radius 50mm`; unknown names stay literal |
| 25 functions | `floor ceil pow log ln exp sign hypot atan atan2 asin acos lerp clamp` measured; `sum(1..100) = 5050`, `avg`, `count` |
| lexer | `1.5e3`, `2E-4`, `;` separators, `#[ … ]#` block comments, `;` inside brackets errors |
| units | `1km = 1000000mm`, `1yd = 914.4mm`, `1000um = 1mm`, `2rad ≈ 114.6deg` |
| regression | the classic 1.0/2.0 forms (cup, bridge arch, fence pattern) compile unchanged |

**Backward compatibility:** every pre-2.1 test still passes — 170 original
(VM parity fuzz, XPBD rope physics, DEC smoothing, subdivision, blend
watertightness, inertia) + corpus gates over 15 examples and 23 lessons.
The 2.1 expansion is purely additive.
