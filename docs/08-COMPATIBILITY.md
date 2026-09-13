# OTD Compatibility Policy — "old files keep working"

## The Promise

Any `.otd` file that runs on version N keeps running on every future version,
without a single edit. This is a language-level guarantee, enforced by four rules.

## Rule 1 — Additive-Only Evolution

New releases may **add** keywords, parameters, materials, colors and formats.
They may never **re-mean** an existing word. `sphere` will always mean a sphere
with `r:` and `smooth:`; `cylinder` will always sit on its base. Parameter
names are frozen once shipped.

## Rule 2 — The Version Header

```
version 1
scene "Coffee Cup"
…
```

- Optional; if absent the file is treated as `version 1`.
- Future interpreters read it and apply the old semantics where anything ever
  differed. Writing it is the "insurance" form — recommended for shared files.
- `version 2` files opened by a v1 interpreter produce a warning, not a crash.

## Rule 3 — Unknown Words Warn, Never Crash

If a future file uses a keyword this interpreter doesn't know (say `snow`),
OTD **ignores the statement and prints a warning**:

```
[warning] line 5: 'snow' is not in OTD 1.0 — skipped (the rest of your file still runs)
```

The model still renders. This single rule eliminates the worst failure mode of
small languages (a hard error on version skew) while telling the user exactly
what was dropped.

## Rule 4 — Deprecation Takes Two Versions

If a feature must ever be retired: version N prints a migration hint, version
N+1 still runs it, only N+2 may remove it. The hint includes the exact rewrite.

## Non-Goals (honest limits)

- Phase 1 interpreters are the only arbiters: there is no "strict mode".
- File *format* stability (binary caches) is not promised — only source-text
  compatibility.
- Simulation results may improve between versions (better physics), which
  changes numbers, not validity: `ask "mass?"` answers stay identical because
  they depend only on volume × density, which is spec-frozen.

## Tested

The test suite includes a forward-compatibility case: a file using an unknown
future keyword compiles with a warning and all other statements still execute.


---

## 2.0 compatibility notes

- OTD 2.0 adds four keywords (58 total) — **no 1.0 statement, parameter,
  or file changes meaning**. The ten 1.0 examples are a regression test
  that must compile with zero errors and unchanged stats forever.
- The expression evaluator now runs the OTD-ASM VM first and falls back to
  the 1.0 tree-walk for non-numeric expressions — same values, same unit
  errors, same hints (2 000-case fuzz + corpus gates).
- `pi` now works (it was documented in 1.0 but unimplemented — a 2.0 audit
  fix, not a break).
- 2.0 fixes three 1.0 bugs (metaball call form, marching-tet tables, quad
  diagonal) — outputs that were quietly wrong are now right; outputs that
  worked are byte-identical.

