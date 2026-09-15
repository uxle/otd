# BUGFIXES — OTD 6.0 "INTELLIGENCE"

## B22–B27 — six fixes from an AI user's real experience (fixed in OTD 6.0)

### B22 — tool was not self-describing (fixed)

**Symptom:** No way to ask OTD what it can do; required `grep`ping Rust source.

**Fix:** Added `--list-functions`, `--list-materials`, `--list-keywords`,
`--list-shapes`, `--list-simulate`, `--list-colors`, `--list-all` CLI flags,
plus in-script `help()`, `functions()`, `materials()`, `keywords()`,
`shapes()`, `sims()`. All dump from source tables, never hand-maintained.

### B23 — four different version strings disagreed (fixed)

**Symptom:** `Cargo.toml` said 4.0.0, `--version` said 3.3.0 "QUARK",
the banner said 3.1, the phase banner said "63/10000 phases".

**Fix:** `Cargo.toml` is now the single source of truth via
`env!("CARGO_PKG_VERSION")`. `--version`, the ASCII art banner, and the
phase banner all read from it. No more hardcoded version strings.

### B24 — electrodynamics functions advertised but not callable (fixed)

**Symptom:** `ohm_i`/`ohm_v`/`ohm_r`/etc. existed in
`src/world/electrodynamics.rs` and were listed in the changelog, but
calling `ohm_i(12, 5)` from a script failed with "unknown function".

**Fix:** Wired up 9 functions in `eval_func`: `ohm_v`, `ohm_i`, `ohm_r`,
`power_vi`, `power_ir`, `cap_energy`, `ind_energy`, `rc_tau`, `lc_omega`.
Added them to the `FUNCS` table in `keywords.rs`. A changelog entry now
means "usable from a script."

### B25 — minimum Rust version not documented (fixed)

**Symptom:** A plain build on an older toolchain failed with a cryptic
manifest error from `vendor/avc` (needs edition2024 via transitive dep
`pxfm`).

**Fix:** `run.sh` now states: "Minimum Rust version is 1.85+ (edition2024
via vendor/avc transitive dep pxfm). If you have an older toolchain, run
`rustup update` first."

### B26 — no electrical connectivity layer (fixed)

**Symptom:** The motor simulation asked "will it move?" but had no way
to model the actual electrical circuit — the user had to calculate
resistance/current outside the tool.

**Fix:** New keyword `connect: A B` declares electrical paths. New
`simulate: circuit` walks the connections, computes real resistance from
each part's material resistivity × wire geometry (R = ρL/A), and reports
total R, current I=V/R, power P=VI, and the verdict. New `simulate: motor`
does the full motor analysis (F=B·I·L, τ=N·B·I·A, back-EMF, stall torque,
RPM, efficiency, "WILL IT MOVE?").

### B27 — expression interpolation only did bare names (fixed)

**Symptom:** `print "{a+b}"` left the literal text `{a+b}` — only bare
`{name}` substitutions worked.

**Fix:** The interpolate function now tries bare name first, then falls
back to parsing+evaluating as an expression. If both fail, it warns (not
errors). Error rollback ensures failed expression eval doesn't pollute
the error list. Verified: `print "v+r = {v+r}"` → `v+r = 9`.

---

## B17–B21 — silent-wrongness bugs reported by an AI user (fixed in OTD4.0)

These are the five "it compiled but it was wrong" bugs an AI user hit when
using OTD3 without a human checking each step. Each fix preserves backward
compatibility — every existing OTD3 file still compiles unchanged.

### B17 — `hollow()` defaults to an open top (silent)

**Symptom:** `sphere 4cm - hollow(wall: 2mm)` quietly opens the top of the
sphere, surprising the user who expected a sealed shell.

**Fix:** every `hollow()` call without an explicit `open:` argument now
emits an INFO line telling the user the default is "open top" and how to
override. The default itself is unchanged (backward compat).

### B18 — `group()` silently double-counts mass (7.0g → 14.0g)

**Fix:** `group()` now re-parents: Ident args referring to existing entries
are marked hidden. An INFO line reports the re-parenting.

### B19 — `rotate()` pivots on the bbox center, undocumented

**Fix:** Parser now accepts `rotate (...) pivot (x, y, z)`. An INFO line
reports the default pivot when `pivot:` is not specified.

### B20 — string interpolation silently leaves `{expr}` as literal text

**Fix:** Unrecognized `{name}` now emits a WARN. (Further fixed in B27
above — expression interpolation now actually evaluates expressions.)

### B21 — overlapping shapes only emit a vague warning (no fix suggestion)

**Fix:** Warning now names the pair and depth. `strict: overlap` turns it
into a hard Error. `overlap: check` runs an explicit audit with axis +
depth + fix suggestion.

---

## B15 — CSG cut loses `material:` / `color:` (fixed in 2.2.0)

**Fix:** for `-` and `&`, finish mods (Material/Color) found on a postfix
tool are hoisted and applied to the RESULT.

## B16 — metals render almost black (fixed in 2.2.0)

**Fix:** metals now receive a sky-tinted environment term alongside the GGX
highlight.

## Legacy bugs B1–B14

B1–B7 (near-plane clipping, cube argument order, weight/weigh matching, CSG
open edges, deferred simulate, float formatting, `box` alias) and B8–B14
(documentation drift found by mass generation) are documented in the 2.1.0 /
2.1.1 sections of `CHANGELOG.md`.
