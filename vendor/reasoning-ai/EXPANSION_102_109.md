# Phase 101–109: fixes from live-testing, plus physics/chemistry/biology/puzzle reasoning

> **Rust port:** this project was ported from Python to Rust. File paths use
> `crates/<name>/src/`; commands use `cargo`. The original Python layout
> (`python/<pkg>/<mod>.py`) maps to `crates/<pkg>/src/<mod>.rs`. See
> `README.md` for the current workspace layout and commands.

## Phase 101 — three real bugs found by actually using the system

The previous session's live test found the system correctly abstaining
on three problems it should have been able to solve. Each was
investigated to a root cause, not just patched until the symptom went away:

1. **gcd(240,46) Bézout coefficients** — `BezoutIdentityDomain` searched
   `(x, y)` as two *independent* free choices (a `search_radius²` space),
   when `y` is always uniquely determined by `x` via
   `y = (gcd - a·x) / b`. A reward-shaping patch alone didn't fix this
   (the shaped landscape turned out nearly flat — verified by directly
   inspecting reward values, not assumed). **Real fix:** restructured to
   a single-level search over `x` only, with `y` computed analytically.
   `240·(-193) + 46·(1007) = 2` now verifies in 0.12s.
2. **1 − cos²(x) → sin²(x)** — the `solve_trig_simplify` candidate pool
   had no squared trig forms at all. Added the Pythagorean-identity
   family (`sin(x)**2`, `cos(x)**2`, and complements).
3. **3×3 determinant = −306** — `solve_matrix_determinant`'s candidate
   range was a fixed ±50. Now sized from the matrix itself (a standard
   `n!·max_entry^n` determinant bound), with distance-shaped partial
   credit added to `MatrixDeterminantDomain` so a several-thousand-wide
   candidate search actually has a gradient — verified this landscape
   directly this time, unlike the first attempt at Bézout, and confirmed
   it converges (0/3000 sims failed, 1/4000 succeeded, tuned from there).

All three are locked in as regression tests
(`crates/engine/tests/test_rl_phase_101.rs`)
so a future change can't silently reintroduce any of them. Zero
regressions in the existing Bézout/trig/matrix/solve() test suites.

## Phases 102–104 — physics, chemistry, biology

New science modules (now the `reasoning-science` crate):

- **`crates/science/src/physics_domain.rs`** — SUVAT kinematics (`v=u+at`, `s=ut+½at²`,
  `v²=u²+2as`) and speed/distance/time, solving for any one unknown
  given the others needed. Cross-checked internally: solving for `v` via
  the time-based and distance-based equations for the same scenario
  agree to 6 decimal places.
- **`crates/science/src/chemistry_domain.rs`** — a real recursive-descent formula parser
  (handles nesting, e.g. `Ca(OH)2`), an IUPAC atomic-weight table, and
  mole-ratio stoichiometry. Tested against hand-computed molar masses
  (water = 18.02, glucose = 180.18) and a classic uneven-ratio reaction
  (N₂ + 3H₂ → 2NH₃).
- **`crates/science/src/biology_domain.rs`** — Mendelian genetics via *exhaustive Punnett
  square enumeration* (not a memorized ratio formula), returning exact
  rational probabilities. Verified against the textbook Aa×Aa → 1:2:1
  genotype / 3:1 phenotype result, and a testcross's 1:1 ratio.

25 tests, all passing, all hand-cross-checked rather than only checked
against each other.

## Phases 105–109 — general/aptitude reasoning

New puzzle modules (now the `reasoning-puzzles` crate):

- **`crates/puzzles/src/family_tree_domain.rs`** — builds an explicit relation graph from
  stated facts (father/mother/son/daughter/husband/wife/brother/sister),
  derives relationships (grandfather, uncle, cousin, etc.) by graph
  distance to the closest common ancestor. **Caught a real directional
  bug during testing**: uncle/aunt vs. nephew/niece was backwards in the
  first version (fixed, now has a passing regression test for it). When
  a person's gender was never stated, it correctly reports "child" or
  "sibling" rather than guessing "son" or "daughter" — the same
  abstain-rather-than-guess principle as the math domains, applied to a
  different kind of uncertainty.
- **`crates/puzzles/src/direction_domain.rs`** — 2D coordinate walk (N/S/E/W moves),
  Euclidean distance and compass bearing back to start. Cross-checked
  against a 3-4-5 right triangle by hand.
- **`crates/puzzles/src/clock_domain.rs`** — hour/minute hand angles, including the detail
  that trips up naive solutions (the hour hand creeps continuously with
  the minutes, not just at the hour mark: at 3:30 it's at 105°, not 90°).
- **`crates/puzzles/src/calendar_domain.rs`** — day-of-week via Zeller's congruence,
  *implemented explicitly and then independently cross-checked against
  a from-first-principles proleptic-Gregorian calendar on every call* —
  if the two ever disagreed, it panics (the Python original raised)
  rather than picking one. Verified against
  the real historical fact that July 4, 1776 was a Thursday. (The
  Python version cross-checked against `datetime.date.weekday()`; the
  Rust port's epoch-day calendar is the stand-in for that, since Rust's
  std has no date arithmetic here.)
- **`crates/puzzles/src/mirror_image_domain.rs`** — mirror/water image of text (explicit,
  hand-verified per-glyph lookup tables — stated as data, not derived,
  since glyph-mirror shape isn't a formula), plus the classic "what does
  a digital clock look like in a mirror" puzzle (4:40 → 07:20, using the
  correct clock-face-geometry rule rather than a naive character flip).

37 tests, all passing — including the caught-and-fixed uncle/nephew bug.

## What's honestly out of scope here

Open-ended physics/chemistry/biology *concept* questions ("why does X
happen") aren't attempted — this system verifies against a computable
ground truth, and a concept explanation has no such ground truth to check
against. What's built is the *computable, formula-driven* slice of each
subject (motion equations, stoichiometry, Mendelian ratios), which is
exactly the slice this project's verification-first architecture is
suited for, and exactly where a wrong "explanation-style" answer would be
hardest to catch. This is the same scope discipline as the rest of the
project, applied to three new subjects rather than relaxed for them.

## Not yet wired into `solve()`

These 8 new modules are usable directly (see the tests for calling
convention) but not yet routed through the unified `solve(Problem(...))`
API the math domains use — that would mean designing a consistent typed
payload schema for 8 different problem shapes (kinematics needs 5
optional numeric fields, family tree needs a fact list, mirror image
needs a string), which is real, scoped work worth doing deliberately
rather than bolted on at the end of an already-long batch. Flagged
explicitly rather than left to look finished.

(This gap was closed in Phases 110–126 — see `EXPANSION_110_126.md`; the
science and puzzle routes live in `crates/engine/src/science_solve.rs`.)
