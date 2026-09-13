# OTD 2.1 — Enhancement Roadmap & Realization Report

> The question this document answers: **"How much can this project be enhanced,
> and how?"** — Part A reports what was enhanced *now* (all delivered in this
> build). Part B is the prioritized backlog for the next rounds.

---

## PART A — Enhanced in this build (all shipped & verified)

### A1. Project structure: from 15 examples → a real content library

Before: `examples/` (15 files) + `tests/` + `docs/`. After:

```
otd2/
├── library/                      ← NEW: 1,695 validated programs
│   ├── 00-INDEX.md               ← master index (every file, one-line description)
│   ├── materials/  (263)         ← brick.otd, water.otd, wood.otd, glass.otd, gold.otd …
│   │   ├── 42 metal demos (one per metal × ingot/rod/sphere + plates, springs, gears)
│   │   ├── 30+ brick demos (walls, piles, towers, arches, huts)
│   │   ├── 30+ water demos (float tests per material, boats, buoys, islands, ice)
│   │   └── wood/stone/glass/ceramic/soft families
│   ├── primitives/ (293)         ← every shape & builder, parameter sweeps
│   ├── patterns/   (149)         ← repeat/grid/ring/scatter studies + combos
│   ├── architecture/ (160)       ← houses, towers, bridges, castles, stairs, columns
│   ├── nature/     (141)         ← trees, terrains, crystals, space
│   ├── furniture/  (118)         ← tables, chairs, shelves, lamps
│   ├── machines/   (137)         ← gears, robots, vehicles, tools
│   ├── art/        (115)         ← sculptures, spirals, mosaics, jewelry
│   ├── games/      (83)          ← chess, dice, dominoes, toys, sports
│   ├── physics/    (87)          ← drop / float / collapse experiments
│   ├── lessons/    (92)          ← 46 numbered lessons + labs & exercises
│   └── showcase/   (57)          ← harbors, cities, wind farms, orbit studies
├── tests/library_parses.rs       ← NEW: cargo test walks all 1,695 files every run
└── README.md                     ← updated with library section
```

**Every one of the 1,695 files passes `otd --check`** (full parse + evaluate +
world build — not just syntax). Test totals: **268 tests, 0 failures**
(157 unit + 74 syntax + 23 bughunt + 6 validation + content/integration/perf/render + library).

### A2. Seven new findings from mass-generation testing (B8–B14)

Generating 1,695 programs exercised the engine far beyond the original corpus
and surfaced **7 documentation/implementation drift bugs** (all worked around in
the library; fixes queued in Part B):

| ID | Severity | Finding |
|---|---|---|
| B8 | HIGH | Docs/spec show `metaball [(x,y,z,r)…]` 4-tuples — engine only accepts 3-tuple positions + separate `r:` parameter (deep-blob.otd is correct, spec is wrong) |
| B9 | HIGH | Spec's `sweep(path: […], r: 2cm)` form does not parse; the parser's own error hint `sweep […] path: […]` also does not parse. Working form: `sweep(profile: […3+ corners…], path: […])` |
| B10 | HIGH | Spec's `loft [(r,h)…]` form does not parse. Working form: `loft(sections: [[profile1], [profile2]])` with equal corner counts |
| B11 | MEDIUM | Docs show `grid(…, spacing: 5cm)` — engine requires a tuple: `spacing: (5cm, 5cm)` |
| B12 | LOW | `rebeccapurple` is listed in spec Appendix A (147 colors) but rejected by the engine (146 actually work) |
| B13 | MEDIUM | **Engine bug:** a length stored in a variable inside a `for` loop loses its unit tag when multiplied by a trig result — `r = 2cm + i*0.5cm` then `at (r * cos(a), …)` errors "at wants lengths, not angles". The identical inline expression works. Staircase-style literal form is unaffected |
| B14 | MEDIUM | `use name` only works on `define`d templates, never on assigned shapes — friendly error exists but docs never state the rule |

### A3. Screenshots

18 sample renders shipped in `screenshots-library/` (also visible in the repo):
brick, water, glass, 3D text, flowers, house, lighthouse, pine, desk lamp,
gears, car, sculpture, chess pawn, float demo, first cup, harbor, skyline,
wind farm.

---

## PART B — The full enhancement backlog (how much more is possible)

Ranked by impact ÷ effort. The engine's phase plan (P0000–P1299 done of 10,000)
means there is **deliberate headroom for ~100× more growth** — this list is the
practical slice.

### Tier 1 — fix the drift (hours, do first)

1. **B8–B12 doc fixes**: correct the spec/cheatsheet for metaball, sweep, loft,
   grid-spacing, and the color table (or implement the documented forms —
   both are small parser changes).
2. **B13 unit-tag bug**: in the VM/interpreter, propagate the length unit tag
   through loop-scoped variables used in `*` with trig results. Add a
   regression test: `for` + variable + `cos(a)` in `at(...)`.
3. **B14 doc note**: one paragraph in 04-SPECIFICATION.md §Parts — "`use`
   works on `define`d templates; to copy an assigned shape use a template."

### Tier 2 — content & docs (days)

4. **Renderer packs**: render every library file to PNG (≈35 min at ~1.2 s
   each with -P8) and ship a `docs/img/library/` contact sheet — turns the
   index into a visual catalog.
5. **Library → web viewer**: load `library/` categories into the viewer's
   lesson picker (assets/index.html already has a lessons list — extend it).
6. **Per-category quizzes**: each lesson gains a `#[ try changing … ]#`
   exercise footer; auto-checkable via `assert`.
7. **Localization pass**: the friendly error strings are the product — ship
   translated error tables as data files.

### Tier 3 — engine (weeks, matches the phase plan)

8. **T-junction stitching pass** (the one HIGH bug left from the first audit,
   B4): after BSP booleans, weld T-vertices to make hollow/subtract watertight
   and honest against `ask "watertight?"`.
9. **Named-arg validation**: unknown named parameters (e.g. `sphere(bogus: 3)`)
   are silently ignored today — emit a warning listing near-miss names.
10. **`import` local files + drag-drop** (Phase 2 promise in the spec).
11. **Animation keywords** (`simulate` already runs XPBD — add `for t in
    0..60` frames export) → GIF/APNG via the existing PNG writer.
12. **Sub-category tags**: a `tags: architecture, gothic` header the viewer
    can filter on (machine-readable, no new keywords in programs).

### Growth ceiling

The 10,000-phase roadmap targets ~100 keyword budget, a GPU-tier renderer,
full CSG watertightness, and a plugin-free AR export path. With 1,695 programs
+ 268 tests + 7 documented drift findings, the project now has the *content
mass* and the *safety net* to absorb all of Tier 3 without regressions.

---

## Verification commands (everything in Part A)

```bash
cargo test --release            # 268 tests incl. library_parses (1,695 files)
find library -name "*.otd" | wc -l                    # 1695
find library -name "*.otd" -exec otd --check {} \;   # 0 errors (use xargs -P8)
```
