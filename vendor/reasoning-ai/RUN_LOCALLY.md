# Running the Reasoning AI workspace locally (Rust)

> **Rust port:** this project was ported from Python to Rust. File paths use
> `crates/<name>/src/`; commands use `cargo`. The original Python layout
> (`python/<pkg>/<mod>.py`) maps to `crates/<pkg>/src/<mod>.rs`. See
> `README.md` for the current workspace layout and commands.

This document replaces the original Python one. The biggest change: the
PyTorch phases (017–018) that the Python sandbox could never execute are
**no longer unverified** — `crates/neural` re-implements the same
architecture from scratch over `Vec<f64>`, its backprop is checked against
numerical gradients, and its tests run in the normal suite. There is no
`pip install torch` step anywhere: **no external ML runtime is needed —
everything is pure Rust.**

## Setup

- **Rust 1.75+** via [rustup](https://rustup.rs) (`rustc --version` to check).
- No other prerequisites. The only external crates are `regex`, `rand`,
  `serde`, `serde_json` — `cargo` fetches them automatically on first build.

```bash
cd reasoning-ai-rust
cargo build --release          # ~1 min cold; binaries land in target/release/
cargo test                     # 595 tests, 0 failures
```

## Step 1 — sanity check the CLI wires up correctly

```bash
./target/release/reasoning-ai "What is 12 * 7 + 5?"
```

Expect:

```
  VERIFIED: 89   [confidence 100%]
```

(Equivalently: `cargo run -q -p reasoning-ai -- "What is 12 * 7 + 5?"`.)
Every answer is either `VERIFIED ...` or `ABSTAINED: <why>` — there is no
third mode.

## Step 2 — run the previously-unverifiable neural PRM (Phases 017–018)

In the Python original this section asked you to install torch and verify
the model yourself, because the sandbox couldn't. In the Rust port it's just
part of the test suite:

```bash
cargo test -p reasoning-neural
```

This exercises the Phase 017 neural PRM (a small Transformer encoder over
tokenized state text: token embedding → learned positional embedding →
2-layer self-attention blocks → mean-pool → head) and the Phase 018
training loop (`train_neural_prm` in `crates/neural/src/train_prm.rs`):
forward pass, hand-written backward pass, Adam updates, and a
finite-difference gradient check. The same layer shapes as the PyTorch
original, the same math — but implemented over `Vec<f64>` parameter
blocks, so it runs (and is verified) anywhere Rust does.

## Step 3 — the real test (self-evolution capacity, Phases 009/013)

The original's open question — does real model capacity break the linear
PRM's flat self-evolution curve? — is runnable end-to-end:

```bash
cargo test -p reasoning-selfplay     # Phases 009/013 loops
cargo run -q --bin train             # the full pipeline in ~0.3 s
```

The Rust port reproduces the Python finding: the linear-PRM self-evolution
curve stays flat across rounds (`[8, 8, 8, 8] / 15` on the held-out eval
set) — a capacity ceiling, not a pipeline failure. The neural-PRM
through-the-same-loop experiment remains the open follow-up it always was.

---

## Phases 110–126: natural language, CLI, and the JSON bridge

### Run the interactive CLI

```bash
cargo run -q -p reasoning-ai                       # REPL: plain-English questions
cargo run -q -p reasoning-ai -- "What is the weight of a 5 kg object?"
cargo run -q -p reasoning-ai -- --json "Balance H2 + O2 -> H2O"   # machine-readable output
cargo run -q -p reasoning-ai -- --explain "Solve 2x + 3 = 11"     # + routing details
```

Real one-shot outputs (after `cargo build --release`, run
`./target/release/reasoning-ai "<q>"`):

```
$ ./target/release/reasoning-ai "A car accelerates from rest at 3 m/s^2 for 5 s. What is its final velocity?"
  VERIFIED: 15.0 m/s   [confidence 100%]

$ ./target/release/reasoning-ai "Balance the equation H2 + O2 -> H2O"
  VERIFIED: 2 H2 + O2 -> 2 H2O   [confidence 100%]

$ ./target/release/reasoning-ai "Write a poem about the ocean"
  ABSTAINED: Could not map this question onto a verified domain (deterministic parser). no domain pattern matched this question
```

Inside the REPL, `help` prints ~40 example questions across math, physics,
chemistry, biology, and puzzles; `explain` toggles route explanations;
`quit` exits.

### Use the engine from Rust

```rust
use reasoning_engine::{ask, solve, Problem};

fn main() {
    // Natural-language front door: deterministic parse -> typed solve -> verify
    let r = ask("A ball is dropped from 20 m. What is its impact speed?", None, 0);
    println!("{}", r["verified"]);        // true
    println!("{}", r["answer"]);          // "19.79899 m/s"

    // Typed problem through the unified solve() API (same gate: verified-or-abstain)
    let payload = serde_json::json!({ "equation": "H2 + O2 -> H2O" });
    let p = Problem::new("chem_balance", serde_json::from_value(payload).unwrap());
    let ans = solve(&p, None, 0);
    println!("{}", ans.final_answer().unwrap_or("(abstained)"));
    // "2 H2 + O2 -> 2 H2O"
}
```

`ask()` lives in `crates/engine/src/nlp_solver.rs` and `solve()` /
`Problem` in `crates/engine/src/solve.rs` — the engine crate holds what the
Python layout spread over `apps/` and `nlp/solver.py` (the app crate is
binaries only).

### JSON bridge (for web servers)

```bash
echo '{"question": "Cross Aa x Aa. What is the phenotype ratio?"}' | ./target/release/api_entry
echo '{"typed": {"kind": "chem_molar_mass", "payload": {"formula": "H2SO4"}}}' | ./target/release/api_entry
```

Real outputs (one JSON object in, one out, always exit 0 with an `ok`
flag so abstention is distinguishable from failure):

```
{"answer":"dominant: 3/4, recessive: 1/4","confidence":1.0,"explanation":"Solved via 4-cell Punnett square enumeration (exact Fractions); independently re-verified (exhaustive gamete pairing, sum == 1).","ok":true,"parsed":{...},"question":"Cross Aa x Aa. What is the phenotype ratio?","source":"nl","verified":true}
{"answer":"98.09 g/mol","confidence":1.0,"explanation":"Solved via sum(atoms * atomic weight); independently re-verified (recursive-descent formula parse (IUPAC weights)).","kind":"chem_molar_mass","ok":true,"payload":{"formula":"H2SO4"},"source":"typed","verified":true}
```

The LLM-fallback path of the web front end submits structured extractions
to this same bridge — the LLM only ever fills in a typed form that
`solve()` still has to pass; it never produces an answer directly.

### Run the RL training CLI

```bash
cargo run -q --bin train_rl -- --rounds 2 --out /tmp/rl_run
```

Real run (2 rounds, default seed), curriculum-scheduled GRPO with
MCTS-sourced episodes, per-round JSON checkpoints in `--out`:

```
[train_rl] evaluating initial (untrained) policy on 10 problems...
[train_rl] before: solve_rate=0.50
[train_rl] round 0: level=1 action=hold solve_rate=0.38 entropy=1.00 next_epsilon=0.10 -> /tmp/rl_run/policy_round_000.json
[train_rl] round 1: level=1 action=hold solve_rate=0.75 entropy=1.00 next_epsilon=0.10 -> /tmp/rl_run/policy_round_001.json
[train_rl] evaluating final policy on 10 problems...
[train_rl] after: solve_rate=0.50
[train_rl] wrote /tmp/rl_run/policy_final.json and /tmp/rl_run/history.json
```

(See `RL_TRAINING.md` for what these numbers do and don't show — the
honest-null-result history is preserved, not re-run for a better number.)

### Run the new tests

```bash
cargo test -p reasoning-science         # science domains incl. the Phases 110-122 expansion
cargo test -p reasoning-engine          # NL + science routes, unified solve(), audit, memory
cargo test -p reasoning-nlp             # parser / quantities / math routes
cargo test -p reasoning-evaluation      # benchmark harnesses (~4.5 s)
cargo test                              # everything: 595 tests
```

The 38-question NL smoke run is `crates/engine/tests/nl_smoke.rs`
(`cargo test -p reasoning-engine --test nl_smoke`); the formal routing
suite is `crates/engine/tests/test_nl_and_science_routes.rs`.

### Run the benchmarks / scorecard

The evaluation crate is a library of `pub fn` harnesses, exercised by its
integration tests:

```bash
cargo test -p reasoning-evaluation                       # whole suite, ~4.5 s
```

Entry points (all in `crates/evaluation/src/`):
`run_full_system_benchmark` (Phase 96 full-system scorecard — the
`wrong_verified == 0` claim), `run_benchmark` / `run_full_benchmark` /
`run_phase_1_30_benchmark` / `run_phase_1_40_benchmark` (benchmark.rs),
`run_three_way_comparison` (rl_benchmark.rs, Phase 69),
`run_four_way_comparison` (rl_benchmark_v2.rs, Phase 75),
`run_low_budget_comparison` (rl_benchmark_low_budget.rs, Phase 82/83),
`run_five_way_comparison` (rl_benchmark_v3.rs, Phase 95). For comparison,
the Python original took roughly a minute for the same suite.

Full documentation of phases 110–126: `EXPANSION_110_126.md`.
