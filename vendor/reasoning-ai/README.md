# Reasoning AI (Rust) — a verified reasoning engine

A small, verification-gated reasoning system for **math, physics, chemistry,
biology, and classic reasoning puzzles**. You ask a question in plain
English; a deterministic parser maps it onto a typed problem; search
(MCTS / beam / best-first) proposes candidate solutions; and every candidate
is **independently re-verified by a non-neural checker** — a from-scratch
computer-algebra engine, numeric evaluation, or exhaustive enumeration —
before anything is shown to you.

The honest-abstention design is the point: if nothing verifies, the engine
**says so instead of guessing**. The abstention gate
(`crates/uncertainty/src/abstention.rs`, `Answer::final_answer()`) has no
code path that can return an unverified answer — it is structurally
enforced, not a promise.

This workspace is a faithful Rust port of the original Python project
(126 build phases; see the [documentation index](#project-documentation)
for the full design narrative and phase history). Port stats: **19 crates,
219 `.rs` files (~29,400 lines of source + ~8,300 lines of tests), 595
tests, zero compiler warnings in the crate build, no external ML runtime** —
the only dependencies are `regex`, `rand`, `serde`, and `serde_json`.

---

## Quick start

### Prerequisites

- **Rust 1.75+** (edition 2021) — install via [rustup](https://rustup.rs):
  `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`
- That's it. No Python, no PyTorch, no sympy, no BLAS — everything is pure
  Rust. (`rustc --version` to confirm.)

### Build & test

```bash
cd reasoning-ai-rust
cargo build --release        # ~1 min cold; binary at target/release/reasoning-ai
cargo test                   # 595 tests, all passing
```

### Ask a question (one-shot)

```bash
cargo run -q -p reasoning-ai -- "What is 12 * 7 + 5?"
# or, after cargo build --release:
./target/release/reasoning-ai "What is 12 * 7 + 5?"
```

### REPL

```bash
./target/release/reasoning-ai
```

Real session (colors — green VERIFIED / yellow ABSTAINED — are enabled on
a terminal; `help` prints ~40 more example questions):

```
$ ./target/release/reasoning-ai

====================================================================
  Verified Reasoning AI  |  math - physics - chemistry - biology
  ------------------------------------------------------------------
  Type a question in plain English. Every answer is independently
  verified before it is shown; if nothing verifies, the engine says
  so instead of guessing.
  Type 'help' for examples, 'explain' to toggle routing details.
====================================================================

you>   VERIFIED: 10   [confidence 100%]

you>   VERIFIED: 4.0   [confidence 100%]

you>   routing details ON

you>   VERIFIED: 2 H2 + O2 -> 2 H2O   [confidence 100%]
  [route] routed to chem_balance. chemical equation with arrow detected Mapped: equation <- "H2 + O2 -> H2O"

you> bye.
```

(The prompts above were `What is 5 choose 2?`, `Solve 2x + 3 = 11`,
`explain`, `Balance the equation H2 + O2 -> H2O`, `quit`.)

### `--json` (machine-readable output)

```bash
./target/release/reasoning-ai --json "What is gcd(240, 46)?"
```

```json
{
  "answer": "240*(-193) + 46*(1007) = 2",
  "confidence": 1.0,
  "explanation": "Bezout coefficients found and independently re-verified.",
  "parsed": {
    "candidates": [],
    "explain": "routed to gcd_bezout. gcd pattern Mapped: a <- \"240\"; b <- \"46\"",
    "kind": "gcd_bezout",
    "ok": true,
    "payload": {
      "a": 240,
      "b": 46
    }
  },
  "question": "What is gcd(240, 46)?",
  "verified": true
}
```

### `--explain` (show how the question was routed)

```bash
./target/release/reasoning-ai --explain "Balance the equation H2 + O2 -> H2O"
```

```
  VERIFIED: 2 H2 + O2 -> 2 H2O   [confidence 100%]
  [route] routed to chem_balance. chemical equation with arrow detected Mapped: equation <- "H2 + O2 -> H2O"
```

### Example questions and real outputs

All outputs below were produced by `./target/release/reasoning-ai "<question>"`
after `cargo build --release` (run them yourself):

| Question | Output |
|---|---|
| `What is 12 * 7 + 5?` | `VERIFIED: 89   [confidence 100%]` |
| `Solve 2x + 3 = 11` | `VERIFIED: 4.0   [confidence 100%]` |
| `Factor x^2 + 5x + 6` | `VERIFIED: (x+2)(x+3)   [confidence 100%]` |
| `What is 5 choose 2?` | `VERIFIED: 10   [confidence 100%]` |
| `What is gcd(240, 46)?` | `VERIFIED: 240*(-193) + 46*(1007) = 2   [confidence 100%]` |
| `What is the determinant of [[1,2],[3,4]]?` | `VERIFIED: -2   [confidence 100%]` |
| `A car accelerates from rest at 3 m/s^2 for 5 s. What is its final velocity?` | `VERIFIED: 15.0 m/s   [confidence 100%]` |
| `A ball is dropped from 20 m. What is its impact speed?` | `VERIFIED: 19.79899 m/s   [confidence 100%]` |
| `Balance the equation H2 + O2 -> H2O` | `VERIFIED: 2 H2 + O2 -> 2 H2O   [confidence 100%]` |
| `What is the pH of a solution with [H+] = 1e-3?` | `VERIFIED: pH = 3.0   [confidence 100%]` |
| `Cross Aa x Aa. What is the phenotype ratio?` | `VERIFIED: dominant: 3/4, recessive: 1/4   [confidence 100%]` |
| `What day of the week was July 4, 1776?` | `VERIFIED: Thursday   [confidence 100%]` |
| `What is the angle between the hands of a clock at 3:30?` | `VERIFIED: 75.0 degrees   [confidence 100%]` |
| `Write a poem about the ocean` | `ABSTAINED: Could not map this question onto a verified domain (deterministic parser). no domain pattern matched this question` |

The last row is the design working as intended: no verifier, no answer.

---

## Workspace layout

Every Python package became a crate; every Python module became a Rust
module with the same name (`python/search/mcts.py` →
`crates/search/src/mcts.rs`).

| Crate | Ports (Python) | Purpose |
|---|---|---|
| `reasoning-common` | shared helpers | Python float semantics (banker's rounding, `str(float)`), exact i64 rationals (`Rat`), rational matrices (`RatMatrix`: determinant, solve, nullspace) |
| `reasoning-symbolic` | `sympy` usage | The mini-CAS replacing sympy (see below) |
| `reasoning-tensor` | `tensor/` | Phase 001 from-scratch N-D tensor abstraction |
| `reasoning-tokenizer` | `tokenizer/` | Phase 016 digit-level math tokenizer |
| `reasoning-verifier` | `verifier/` | The ground-truth signal: symbolic verifier, dimensional/units checks, cross-method disagreement flags, and the safe mini Python-interpreter sandbox |
| `reasoning-search` | `search/` | The `Domain` contract, MCTS, beam/best-first search, multi-path solving, pruning, and 16 concrete problem domains |
| `reasoning-science` | `science/` | Physics / chemistry / biology formula domains (16 modules), each re-verifying its own result through an independent path |
| `reasoning-puzzles` | `puzzles/` | Family-tree, direction, clock-angle, calendar, and mirror-image puzzle domains |
| `reasoning-memory` | `memory/` | Working / episodic / semantic / failure memory |
| `reasoning-uncertainty` | `uncertainty/` | The abstention gate, the Critic role, graded confidence |
| `reasoning-prm` | `prm/` | Hand-engineered process reward model (features + linear PRM + SGD), calibration fix, PRM-guided search policy |
| `reasoning-neural` | `neural/` | The neural PRM — a small Transformer encoder, **re-implemented from scratch over `Vec<f64>`** (replaces PyTorch; actually runs and is gradient-verified) |
| `reasoning-curriculum` | `curriculum/` | Verified problem generation, adaptive progression, template word problems, adaptive compute |
| `reasoning-nlp` | `nlp/` (parser, quantities, math_routes) | Deterministic quantity extraction with unit canonicalization, and the ordered domain routers |
| `reasoning-rl` | `rl/` | The RL stack: returns, GAE, PPO-clip, GRPO, KL penalty, entropy regularization, reward model, training loops, JSON checkpointing |
| `reasoning-selfplay` | `selfplay/` | The self-evolution loop and its calibrated retry |
| `reasoning-engine` | `apps/` + `nlp/solver.py` + `uncertainty/audit_log.py` + `curriculum/adversarial.py` | The integration layer: unified typed `solve()`, the `ask()` NL front door, science/puzzle routes, audit logging, adversarial stress tests, memory integration |
| `reasoning-evaluation` | `evaluation/` | Benchmark harnesses and the full-system `solve()` scorecard (`wrong_verified == 0` is the project's core correctness claim) |
| `reasoning-ai` (app crate) | `apps/*.py` | The CLI binary `reasoning-ai` plus the `train`, `train_rl`, and `api_entry` binaries |

Two Python package cycles (`apps↔nlp/curriculum/uncertainty`, and
`curriculum→apps`) were broken by this layering: the engine crate owns the
former `apps/` solve logic, and the app crate owns only the binaries.

---

## The sympy → `reasoning-symbolic` story

The Python original leaned on **sympy** for parsing, expansion,
simplification, solving, differentiation, matrices, and exact trigonometry.
There is no mature Rust equivalent, so `crates/symbolic` is a hand-written
**mini-CAS** covering exactly the surface this project uses:

- expression parsing with implicit multiplication (`3x`, `xy` → `x*y`) and
  both `^` and `**` powers;
- exact rational arithmetic (`Rat`, i64-based) — no floating-point drift in
  algebra;
- polynomial expansion and canonicalization (the `expand`/`simplify` subset);
- equation solving (linear and quadratic);
- symbolic differentiation;
- algebraic-equivalence checking: structural equality plus a **numeric
  cross-check** at sample points, so two structurally different forms only
  count as equal if they also agree numerically;
- exact trigonometry at standard angles (`sin(pi/6) = 1/2`, …) and an
  `nsimplify`-style rational reconstruction;
- rational matrices (`reasoning-common::RatMatrix`, Bareiss determinant +
  nullspace) — the `sympy.Matrix` subset, used by the chemistry
  equation-balancing domain.

The Python **code-execution sandbox** (Phase 031: parse candidate code with
the `ast` module, whitelist a tiny node set, `exec` with stripped builtins
under a step-count tracer) became a **safe mini interpreter** inside
`crates/verifier/src/code_sandbox.rs`: a hand-written tokenizer, parser, and
tree-walking evaluator that implements exactly the allowed subset and
rejects everything else with the same violation messages — no `exec`, no
host code execution, ever.

Similarly, the PyTorch neural PRM became `reasoning-neural`: same layer
shapes (token embedding → learned positional embedding → 2-layer Transformer
encoder → head), but a from-scratch forward/backward over plain `Vec<f64>`
parameter blocks with an Adam loop, verified against numerical gradients.
The PyTorch original never actually ran in its sandbox (no torch, no
network); the Rust version does.

---

## Binaries

All binaries live in the app crate (`crates/app`); build them with
`cargo build --release` or run them directly with `cargo run`.

| Binary | Run | What it does |
|---|---|---|
| `reasoning-ai` | `cargo run -q -p reasoning-ai -- "question"` / `./target/release/reasoning-ai` | The CLI: one-shot questions, interactive REPL, `--json`, `--explain`, `--no-color` |
| `train` | `cargo run -q --bin train` | Phase 047 training pipeline: curriculum → self-evolution (guided search → verify → retrain PRM) → before/after eval. Runs in ~0.3 s |
| `train_rl` | `cargo run -q --bin train_rl -- --rounds 5 --out /tmp/rl_run` | Phase 079 RL training CLI: curriculum-scheduled, MCTS-sourced GRPO with adaptive exploration and per-round JSON checkpoints |
| `api_entry` | `echo '{"question": "..."}' \| ./target/release/api_entry` | Phase 126 stdin/stdout JSON bridge for web servers — accepts `{"question": ...}` (NL) or `{"typed": {kind, payload}}` (structured); the LLM-fallback path of the web front end submits extractions here, and `solve()` still gates everything on verification |

The evaluation crate is a **library**, not a binary: its harnesses are
`pub fn` entry points (`run_full_system_benchmark` in
`crates/evaluation/src/full_system_benchmark.rs`,
`run_benchmark`/`run_full_benchmark`/`run_phase_1_30_benchmark`/
`run_phase_1_40_benchmark` in `crates/evaluation/src/benchmark.rs`,
`run_three_way_comparison` / `run_four_way_comparison` /
`run_five_way_comparison` / `run_low_budget_comparison` in
`crates/evaluation/src/rl_benchmark*.rs`), exercised by the integration
tests in `crates/evaluation/tests/`.

---

## Performance

The full evaluation suite — the Phase 96 cross-domain scorecard, the RL
three/four/five-way comparisons, the domain benchmarks, and the PRM
calibration report — runs in **~4.5 s** via
`cargo test -p reasoning-evaluation` (the Python original took roughly a
minute). One-shot CLI questions answer in milliseconds; the full 595-test
suite runs in ~16 s once compiled. No GPU, no
accelerated BLAS — the speedup is plain Rust plus the Phase 97
`is_terminal()` hot-fix (sympy `simplify` → `expand`), ported faithfully.

---

## Project documentation

| Document | Contents |
|---|---|
| [DESIGN.md](DESIGN.md) | The original design document: scope & honesty statement, architecture, mathematical foundations, phase plan |
| [RUN_LOCALLY.md](RUN_LOCALLY.md) | How to build, test, and use everything in this workspace |
| [RL_TRAINING.md](RL_TRAINING.md) | The RL training layer (Phases 51–90): equations, trainable policy, honest experiment results |
| [RELEASE.md](RELEASE.md) | Phases 1–100 final honest scorecard |
| [CAPSTONE.md](CAPSTONE.md) | Phases 1–50 retrospective, including the Phase 046 security finding |
| [EXPANSION_102_109.md](EXPANSION_102_109.md) | Phases 101–109: live-testing fixes, physics/chemistry/biology, puzzles |
| [EXPANSION_110_126.md](EXPANSION_110_126.md) | Phases 110–126: full science coverage, natural language, CLI, web front end |
| [CONVENTIONS.md](CONVENTIONS.md) | The porting rules the Rust crates follow (behavioral fidelity, error strings, Python float semantics) |

---

## Honest scope

- This is a **small, narrow-domain, verification-gated search system** —
  not, and never claimed to be, a competitor to frontier LLMs at general
  reasoning.
- Open-ended language reasoning, concept explanations, and subjective
  judgment have no reliable automatic verifier, so they are out of scope by
  design; the parser abstains on them (see the last row of the examples
  table).
- RL training of the tiny 5-parameter policy was built, tested, and honestly
  benchmarked: no statistically detectable improvement over pure search was
  found at the budgets tested. The null results are documented in
  [RL_TRAINING.md](RL_TRAINING.md), not smoothed over.
