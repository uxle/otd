# Math Reasoning AI — Capstone Retrospective (Phases 1–50)

> **Rust port:** this project was ported from Python to Rust. File paths use
> `crates/<name>/src/`; commands use `cargo`. The original Python layout
> (`python/<pkg>/<mod>.py`) maps to `crates/<pkg>/src/<mod>.rs`. See
> `README.md` for the current workspace layout and commands.

## What this actually is

A verified, search-based reasoning system across 10+ math domains
(arithmetic, algebra, quadratics, calculus, number theory, combinatorics,
logic, statistics, geometry, code generation), built on the principle
that a small model + search + an external verifier can solve problems a
model alone can't — because it doesn't have to *know* the answer, it can
*derive and check* it. Every domain follows the same pattern: propose a
candidate, verify it independently, never trust a component's own claim
of success.

## The most important finding: Phase 046

Building a "prompt-injection resistance" test suite **found a real,
serious, two-layer code-execution vulnerability** in code that had shipped
since Phase 002 — the shared symbolic parser every single domain's
verification depends on. A crafted string like
`__import__("os").system(...)` executed arbitrary shell commands through
it. After patching the obvious hole (stripping `__builtins__`), a *second*
attack (`().__class__.__bases__[0].__subclasses__()`) still worked,
because it needs no builtin at all — just attribute access on a literal.
Both are now closed, verified with a permanent regression suite, and
confirmed at the actual public API surface — which itself had a third,
related bug: a crash instead of a graceful abstention.

**Why this matters more than any accuracy number in this document:** a
project that says "verification is the safety mechanism" is only as
trustworthy as the verifier's own security. This was found by building the
thing the design doc asked for (prompt-injection resistance testing),
not by hoping it wasn't a problem.

(In the Rust port the code-execution sandbox — `crates/verifier/src/code_sandbox.rs`
— no longer executes Python at all: it is a hand-written interpreter for
exactly the whitelisted subset, so the entire class of
`exec`-and-builtin attacks is structurally absent, while the
injection-resistance regression tests still run and pass.)

## Honest results, not cherry-picked

| Finding | Result |
|---|---|
| Search finds solutions with zero trained model | Solved 4,7,8,8→24 via MCTS + verifier alone (Phase 003) |
| PRM learns from verifier-grounded data | 4.4x better than baseline, generalizes to algebra too (Phase 004, 012) |
| PRM calibration bug found and fixed | Gap 0.660 → 0.133 (Phase 010, 011) |
| PRM-guided search helps | Real but modest: 11/30 vs 10/30 (Phase 007) |
| Adaptive compute tradeoff | 72–84% compute saved for 1-2 problem accuracy cost (Phase 008) |
| **Self-evolution (linear PRM)** | **Flat across 3 configurations — a proven capacity ceiling, not a bug** (Phase 009, 013) |
| Bounds-based pruning | Sound (verified against brute force), 105x waste reduction on hopeless problems (Phase 022) |
| Beam search heuristic weakness | Needed width ≥200, not 50, for a myopic heuristic — documented, not hidden (Phase 021) |
| Adversarial stress test | 100% (8/8) at tractable scale; documented scalability cliff above coefficient ~50 (Phase 039) |
| Security | **Found and fixed a real RCE vulnerability + a crash-vs-abstain bug** (Phase 046, 049) |

## What's real vs. what's honestly unverified

**Tested by me, in this sandbox, for real:** everything except Phases
017–018 (the PyTorch neural PRM). I ran every test shown in this project,
including ones that failed and needed fixing — the bugs and honest
negative results above are not exceptions to the process, they *are* the
process working correctly.

**Written but not run by me:** the PyTorch neural PRM and its training
loop (Phases 017–018), because this sandbox has no torch and no internet.
`RUN_LOCALLY.md` has exact verification steps. Given what Phase 046 found
in code I *could* test, treat the neural PRM code with proportionally
more scrutiny before trusting it, not less.

(Rust port update: that scrutiny has since been applied —
`crates/neural` re-implements the same architecture from scratch over
`Vec<f64>`, its backprop is checked against numerical gradients, and
`cargo test -p reasoning-neural` runs it as part of the normal suite. The
"written but never run" category is now empty.)

## The single most important open question

Phase 013 proved a linear PRM has no spare capacity — self-evolution
rounds teach it nothing new after round 1. That's the actual bottleneck
on everything downstream: adaptive compute, self-play data quality,
curriculum progression all inherit whatever ceiling the PRM has. The next
real experiment is running the neural PRM through the exact same
self-evolution test and seeing whether real capacity produces the
compounding improvement the design doc's self-evolution loop predicts —
or whether it's still flat for a different reason (data volume,
architecture), which would itself be an important, different finding.

## Scale, honestly

This is ~100 Python files, 10 math domains, and a real (if small)
demonstration that verified search beats single-shot generation on
checkable problems. It is not, and was never going to be, a system that
beats frontier LLMs on general reasoning — that was the honesty
commitment made in the very first design doc, and it still holds.

(The Rust port of that system is 19 crates, ~29,400 lines of Rust source
plus ~8,300 lines of tests — see `README.md`.)

---

## Addendum: Phases 51–70 (RL training layer)

Phases 51–60 implemented and unit-tested the RL math the design doc
promised but the code never had: discounted returns, GAE, PPO-clip,
GRPO, KL penalty, entropy regularization, reward composition/normalization
— all pure Python, all verified against known mathematical identities
(GAE's lambda=0/1 boundary cases, KL(P||P)=0, entropy of a uniform
distribution, etc.), then run end-to-end on real MCTS rollouts.

Phases 61–70 built the first genuinely *trainable* policy in this
project (a 5-parameter linear-softmax policy, analytic gradient checked
against finite differences), wired PPO/GRPO weight updates to it, ran a
real GRPO self-play loop, and benchmarked it honestly against the
existing PRM-guided baseline and a uniform-random baseline. Result: the
new machinery runs correctly end-to-end, but the one comparison run so
far (N=20 eval problems) is statistically underpowered to claim it beats
the baselines, and a real diagnosis (near-zero reward variance within
most GRPO training groups, because raw policy-sampling rarely succeeds at
all) explains why — see `RL_TRAINING.md` for full numbers and the
suggested fix (train from MCTS-generated episodes, not raw policy
samples).

Test count: 95/95 passing (Phases 51-60: 52 tests, Phases 61-70: 43 tests).

---

## Addendum: Phases 71–80 (acting on the Phase 70 diagnosis)

Phase 70 found GRPO training was signal-starved because raw-sampled
training episodes almost always failed. Phases 71-80 fixed that
(MCTS-sourced training episodes, Phase 71/73), added the statistics to
test it properly (Wilson intervals, Phase 74), reran the comparison at
adequate sample size (N=40, Phase 75), and built the infrastructure a
real training run needs (curriculum scheduling Phase 76, adaptive
exploration acting on Phase 68's diagnostics Phase 77, JSON checkpointing
Phase 78, a CLI Phase 79 — smoke-tested end-to-end).

Honest result (N=40, budget=250 MCTS sims): uniform 0.750, PRM-guided
0.575, GRPO-raw 0.750, GRPO-MCTS-sourced 0.750 — none of the trained
conditions is statistically distinguishable from the untrained uniform
baseline at this budget (Wilson 95% CIs all overlap). The Phase 71 fix
did measurably increase training-round solve rate (0.0→0.0→0.0 raw vs.
0.02→0.04→0.08 MCTS-sourced), it just wasn't enough training yet to beat
search-dominated MCTS evaluation. Full breakdown and the "what would make
this a fair test" discussion is in `RL_TRAINING.md` section 7.

Test count: 123/123 passing (Phases 51-60: 52, 61-68: 43, 71-78: 28).

---

## Addendum: Phases 81–90 (honest close-out + new capability)

Phases 81-84 finished the RL training question rather than leaving it
open: added power/effect-size analysis (Phase 81), reran the comparison
at low search budget (Phase 82/83) where rollout quality should matter
more. Result: still no statistically detectable improvement from GRPO
training over pure-search at any budget tested across three separate
experiments (N=20 and two N=40 runs). The training mechanism (Phase 71's
MCTS-sourced episodes) demonstrably works at the signal level; three
short rounds of a 5-parameter policy just isn't enough training yet.
Documented as an honest resting point, not chased further with more
seeds (that would be p-hacking) -- see RL_TRAINING.md section 9.

Phases 85-90 moved to new capability: trigonometry domain (Phase 85),
linear algebra domain — determinant/multiply/linear-system (Phase 86),
16 cross-checked tests for both (Phase 87), semantic memory distinct from
the existing episodic memory (Phase 88, structurally cannot store an
unverified fact), and wired all five new problem kinds into the existing
unified solve() API (Phase 89) with zero regressions in its existing
tests, including the injection-resistance test.

Test count: 170/170 passing across all Phase 51-90 test files.

---

## Addendum: Phases 91–100 (distillation, new domains, final release)

Phases 91-95 built supervised distillation from MCTS's own visit-count
distribution (the standard AlphaZero "expert iteration" recipe) as a
third training approach alongside PPO/GRPO, with a numerically-verified
gradient (finite-difference checked) and a real five-way comparison
(Phase 95). Result: same honest null finding as GRPO -- no statistically
detectable improvement over pure search at the budgets tested, though the
loss curve genuinely decreases each round (the mechanism works; three
rounds isn't enough to win yet).

Phases 96-100 closed the arc: a complete cross-domain benchmark (Phase
96) across all 9 solve() domains -- 0.87 overall solve rate, and
critically 0.00 wrong-verified rate across 54 real problems, the number
this entire project's correctness claim rests on. A real profiling pass
(Phase 97) found linear_equation's is_terminal() burning 95% of its
runtime in unnecessary full symbolic simplify() calls; fixed, verified
equivalent on real rollout states, measured 26-29x speedup. An
integration test (Phase 98) wiring episodic memory (cache hits skip
search entirely) and semantic memory (verified identities get
remembered) into the solve() pipeline for real, not just each module
tested in isolation. Full regression across everything built since Phase
51, plus the pre-existing Phase 1-50 suite -- zero regressions.
RELEASE.md is the final honest scorecard.

Test count: 197/197 passing across all Phase 51-98 test files
(this session's total across five "next 10" batches).

(Rust port: the entire history above was ported — the full suite now
runs as 595 Rust tests under `cargo test`, all passing.)
