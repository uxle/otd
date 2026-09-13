# Contributing to OTD

OTD is an exercise in discipline as much as engineering. Four rules guard
every pull request:

1. **Zero external crates.** `Cargo.toml`'s `[dependencies]` stays empty,
   forever. If a feature seems to need a library, the feature is
   redesigned — not the constraint.
2. **Rust + handwritten assembly only** (backend); the browser is a dumb
   terminal. No JavaScript 3D library, no WebGL engine, no client-side
   scene graph.
3. **Fewer than 60 keywords.** New capability = new parameters or engine
   internals, not new words, unless the budget allows.
4. **Old `.otd` files never break.** The 1.0 example corpus is a
   regression test; every PR must keep it compiling with zero errors and
   unchanged stats.

## Workflow

- `cargo test --release` — the full gate (unit + integration + validation).
- `cargo test --release --test validation_20 -- --nocapture` — the
  measured-numbers report (docs/09).
- Every phase lands with its own tests (see docs/01 for the phase plan).
- Docs are part of the code: a feature without its doc section and
  validation numbers is not done.
