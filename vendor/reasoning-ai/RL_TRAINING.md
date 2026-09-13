# RL Training Layer — Phases 51–70

> **Rust port:** this project was ported from Python to Rust. File paths use
> `crates/<name>/src/`; commands use `cargo`. The original Python layout
> (`python/<pkg>/<mod>.py`) maps to `crates/<pkg>/src/<mod>.rs` — e.g.
> `rl/reward_model.py` → `crates/rl/src/reward_model.rs`. See `README.md`
> for the current workspace layout and commands.

This document covers the two batches that turned the design doc's RL
section (2.5–2.6) into running, tested code: Phases 51–60 built the RL
*math* in isolation; Phases 61–70 built an actual trainable policy and
used that math to train it, for the first time in this project.

Every equation below has a corresponding unit test in
`crates/rl/tests/test_rl_phases_52_60.rs`,
`crates/rl/tests/test_rl_phases_61_63.rs`, and
`crates/rl/tests/test_rl_phases_64_68.rs` (run them all with
`cargo test -p reasoning-rl`). The experimental numbers in section 4
are from a real Python run, not illustrative placeholders; the Rust port
reproduces the same verified-answer semantics — and re-running the whole
benchmark suite now takes seconds (`cargo test -p reasoning-evaluation`)
instead of the minutes the Python harness needed.

## 1. Equations (Phases 51–60)

**Reward composition** (`crates/rl/src/reward_model.rs`)

```
reward = w_correct * correct
       + w_partial  * partial_credit
       + w_verify   * verification_passed
       + w_logic    * logically_consistent
       + w_tool     * used_tool_usefully
       - w_contra   * contradiction_detected
       - w_hallu    * hallucination_detected
       - w_invalid  * invalid_reasoning_detected
       - w_hack     * reward_hacking_detected
```
No term reads trajectory length or token count — the `RewardComponents`
struct has no such field, so a length-based reward cannot be expressed
in this schema, not merely "shouldn't" be.

**Discounted return** (`crates/rl/src/returns.rs`): `G_t = r_t + gamma * G_{t+1}`,
computed backward in O(T).

**TD error / advantage** (`crates/rl/src/advantage.rs`):
`delta_t = r_t + gamma*V(s_{t+1}) - V(s_t)`, `A_t = G_t - V(s_t)`.

**GAE** (`crates/rl/src/gae.rs`): `A_t = delta_t + (gamma*lambda) * A_{t+1}`.
Verified: lambda=0 recovers pure TD error; lambda=1 recovers `G_t - V(s_t)`.

**PPO-clip** (`crates/rl/src/ppo.rs`): `r_t = exp(logp_new - logp_old)`,
`L = min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)`.

**GRPO** (`crates/rl/src/grpo.rs`): `A_i = (r_i - mean(r_1..r_N)) / (std(r_1..r_N) + eps)`
for a group of N sampled solutions to the same problem — no separate
critic network needed. Verified: all-correct or all-wrong groups give
~zero gradient (nothing to learn from a batch already at 0% or 100%).

**KL penalty** (`crates/rl/src/kl_penalty.rs`): `KL(pi_new||pi_ref) = sum_a pi_new(a) * log(pi_new(a)/pi_ref(a))`,
estimated per-sample as `exp(logp_new) * (logp_new - logp_ref)`.

**Entropy bonus** (`crates/rl/src/entropy_reg.rs`): `H(pi) = -sum_a pi(a) * log(pi(a))`.
Verified: uniform distribution over N actions achieves the maximum,
`log(N)`.

**Reward normalization** (`crates/rl/src/reward_norm.rs`): hard clipping, plus a
Welford running mean/variance estimator for online normalization.

## 2. The trainable policy (Phases 61–65)

Phases 51–60 were pure math with synthetic or stand-in inputs. Nothing in
the codebase before Phase 61 actually *trained* with it — the existing
`crates/prm/src/guided_policy.rs` softmaxes over a **regression-trained value
function** (the PRM), which is a different thing from a policy trained by
policy gradient.

Phase 61 introduces a genuine parametric policy: a **linear-softmax
policy** over the same 5 hand-engineered features `crates/prm/src/features.rs`
already extracts (`bias, num_remaining, min_diff_to_target_norm,
depth_norm, has_exact_match`):

```
score(s, a) = theta . phi(s, a)
pi(a|s)     = softmax_a( score(s, a) )
```

Phase 62 derives the policy gradient via the standard log-derivative
identity:

```
d/d theta_k [ log pi(a|s) ] = phi(s,a)_k - E_{a'~pi(.|s)}[ phi(s,a')_k ]
```

and — per the project's rule against trusting derived math without
numerical verification — checks it against a central-difference numeric
gradient for every feature key, at both zero and non-zero weights
(`crates/rl/tests/test_rl_phases_61_63.rs`,
`test_policy_gradient_analytic_matches_numeric_at_zero_weights` and its
non-zero-weights sibling). Max analytic-vs-numeric
error across all tested cases: **< 1e-4**.

Phase 63/64 turn this into actual SGD update rules for PPO and GRPO,
including a hand-derived three-case gradient-masking rule for where the
PPO clip boundary zeroes the gradient — this is the part of the batch
most likely to contain a sign error, so it is the one backed by the most
tests (11 tests in the PPO/GRPO update files
`crates/rl/tests/test_rl_phases_64_68.rs`, including a direct
"nudge weights along the analytic gradient and confirm the target
action's probability actually increased" sanity check, not just a
finite-difference match).

Phase 65 bridges the trainable policy into two shapes: (a) an MCTS
`rollout_policy(state, legal) -> action` callable, matching
`crates/prm/src/guided_policy.rs`'s exact interface, so it's a drop-in replacement
during search; (b) a direct episode sampler (no tree search) for GRPO's
standard "sample N full completions from the current policy" recipe.

## 3. Self-play training loop (Phases 66–68)

`crates/rl/src/grpo_self_play.rs` (Phase 66): for each problem, sample N episodes
directly from the current policy (no search), verify each terminal state
with the existing Phase 002 verifier, compute GRPO group-relative
advantage, and take one weight-update step pooling every (state, action)
pair from every episode.

`crates/rl/src/grpo_training_loop.rs` (Phase 67): runs multiple rounds, evaluating
the policy (guided-MCTS, same budget style as
`crates/selfplay/src/self_evolution_v2.rs`) before
round 0 and after the final round.

`crates/rl/src/training_diagnostics.rs` (Phase 68): normalized policy entropy on a
fixed probe set (0 = deterministic, 1 = uniform) and KL-drift from a
reference policy, plus `detect_entropy_collapse` /
`detect_policy_stagnation` helpers — because a 5-parameter softmax policy
can saturate in very few gradient steps, this is measured every round
rather than assumed fine.

## 4. Honest results — Phase 69 (uniform vs. PRM-guided vs. GRPO-trained)

`crates/evaluation/src/rl_benchmark.rs` runs all three on the same 20-problem
held-out set with the same MCTS search budget. Two real runs, at two
different eval budgets:

| eval budget (MCTS sims) | uniform | PRM-guided (self_evolution_v2) | GRPO-trained |
|---|---|---|---|
| 60  | 0.05 | 0.00 | 0.05 |
| 250 | 0.35 | 0.25 | 0.45 |

**What this does and doesn't show:**

- At budget=60 the eval set is simply too hard for *any* policy to solve
  reliably (a follow-up sweep of the uniform baseline alone: 0.05 at
  budget 60, 0.35 at 250, 0.65 at 800 — solve rate is budget-bound, not
  policy-bound, at low budgets). That's a benchmark-configuration finding,
  not a finding about GRPO or the PRM.
- At budget=250, GRPO-trained (0.45) nominally beats uniform (0.35) and
  PRM-guided (0.25), but **with only 20 eval problems the standard error
  on a proportion this size is ≈0.10–0.11** — these three numbers are
  within roughly one standard error of each other. This run does **not**
  have the statistical power to claim GRPO training helped; it shows the
  new machinery runs end-to-end and produces a plausible policy, not that
  it's proven better. A real claim would need a substantially larger eval
  set (200+ problems) or repeated seeds with a variance estimate, which
  is flagged as follow-up work rather than run here.
- The PRM-guided baseline's history was flat across all 4 rounds
  (`[5, 5, 5, 5] / 20`, i.e. no measured improvement from training) — this
  matches the previously-diagnosed linear-PRM capacity ceiling, not a new
  finding.
- **A real diagnosis, not just a number:** GRPO's own training-round solve
  rates (direct policy sampling, no search) were `[0.0, 0.02, 0.0]` —
  almost every training episode fails outright. With that few
  successes, most GRPO groups have reward variance ≈ 0, which by
  construction (Phase 56's `grpo_all_correct_zero_gradient` property)
  means the gradient signal in most rounds is also ≈ 0. The training loop
  ran correctly; it just didn't get much to learn from in 3 short rounds
  of pure policy sampling. The likely fix — not yet implemented — is
  generating training episodes via MCTS (which can find some successes
  even when raw policy sampling can't) rather than raw sampling, so GRPO
  groups actually contain a mix of successes and failures to contrast.

In the Rust port this comparison is re-runnable as a test:
`cargo test -p reasoning-evaluation` (the three-way Phase 69, four-way
Phase 75, low-budget Phase 82/83, and five-way Phase 95 harnesses all run
as part of the suite, in seconds).

## 5. What is intentionally not built here

Per `DESIGN.md` section 3: *"World model, episodic memory, and
open-ended tool-calling are dropped for now — they don't have a verifier
and would just reintroduce unverifiable generation."* World model stays
out of scope for that reason; it was not overlooked.

The neural policy/PRM (`crates/neural/src/neural_prm.rs`,
`crates/neural/src/train_prm.rs`) required local PyTorch execution in the
Python original and remained unverified in that sandbox. In the Rust port
it is implemented from scratch over `Vec<f64>` — same layer shapes,
hand-written forward/backward, Adam — and it actually runs:
`cargo test -p reasoning-neural` exercises it, including a
numerical-gradient check of the backprop.

## 7. Phases 71–80: fixing the sparse-signal problem, and a properly powered re-test

Phase 70 diagnosed *why* GRPO training barely moved the policy: with raw
policy-sampled training episodes, almost every episode failed outright,
so most GRPO groups had ~zero reward variance and ~zero gradient. Phases
71–80 acted on that diagnosis rather than leaving it as a note:

- **Phase 71** (`crates/rl/src/mcts_episode_source.rs`) generates training episodes
  via MCTS instead of raw sampling — same policy, but the *actions taken*
  come from the AlphaZero-style visit-count-greedy path through the
  search tree, which finds successes far more often than blind sampling.
  Directly tested: on an untrained policy, MCTS-sourced episodes solve at
  least as often as raw-sampled ones over repeated trials
  (`test_mcts_episode_source_solve_rate_beats_raw_sampling_on_easy_problem`
  in `crates/rl/tests/test_rl_phases_71_74.rs`).
- **Phase 73** reruns the GRPO training loop on MCTS-sourced episodes.
- **Phase 74** (`crates/rl/src/stats.rs`) adds Wilson score confidence intervals and
  a two-proportion z-test, because Phase 69's original comparison (N=20)
  was flagged as statistically underpowered and that needed a real fix,
  not just an asterisk.
- **Phase 75** reran the comparison properly: N=40 eval problems, four
  conditions (uniform / PRM-guided / GRPO-raw-sampling / GRPO-MCTS-sourced),
  all with 95% Wilson intervals.
- **Phases 76–79** are infrastructure the above needed to be usable
  beyond a one-off script: curriculum-scheduled difficulty
  (`crates/rl/src/curriculum_grpo.rs`, reusing the existing Phase 014
  `CurriculumController` unchanged), an entropy/stagnation-triggered
  exploration boost (`crates/rl/src/adaptive_exploration.rs`, finally *acting on*
  Phase 68's diagnostics instead of just measuring them), JSON
  checkpointing (`crates/rl/src/checkpoint.rs` — plain JSON, deliberately not
  a binary format, for the same "don't evaluate untrusted input" reasoning
  the verifier already applies elsewhere), and a CLI
  (`crates/app/src/bin/train_rl.rs`,
  smoke-tested end-to-end — run it with
  `cargo run -q --bin train_rl -- --rounds 5 --out /tmp/rl_run`).

**Phase 75's real result (N=40, eval budget=250 MCTS simulations):**

| condition | solve rate | 95% Wilson CI |
|---|---|---|
| uniform (no training at all) | 0.750 | [0.598, 0.858] |
| PRM-guided (self_evolution_v2) | 0.575 | [0.422, 0.715] |
| GRPO, raw-sampled training episodes | 0.750 | [0.598, 0.858] |
| GRPO, MCTS-sourced training episodes | 0.750 | [0.598, 0.858] |

**None of the three trained conditions is statistically distinguishable
from the untrained uniform baseline at this sample size** — every
interval overlaps uniform's. That is the honest headline, not a footnote.

Two things are worth pulling out rather than burying in the null result,
though:

1. **The Phase 71 fix measurably worked at the mechanism level, even
   though it didn't yet move the final metric.** Training-round solve
   rates (raw policy sampling, no search) were flat at `[0.0, 0.0, 0.0]`
   for GRPO-raw, exactly reproducing Phase 70's diagnosis. Under
   MCTS-sourced episodes the same three rounds were
   `[0.02, 0.04, 0.08]` — a real, monotonically increasing signal. The
   fix did what it was supposed to do (produce groups with actual reward
   variance to learn from); three rounds and a 5-parameter policy just
   isn't enough training yet to detectably beat search-dominated MCTS
   evaluation at budget 250, where even a uniform-random rollout policy
   already solves 75% of this eval set through search alone.
2. **PRM-guided nominally scored lowest (0.575 vs. 0.750 uniform),
   though not significantly so.** This is consistent with, not a
   contradiction of, the project's earlier documented finding that the
   linear-PRM self-evolution loop shows no reliable improvement — it is
   reported here exactly as measured rather than smoothed over.

**What this means for future phases, stated plainly:** at MCTS budgets
this generous, search itself is doing most of the work, which makes it
hard for any rollout-policy improvement to show up in the final number —
the informative regime for testing whether GRPO/PPO training helps is
*low* search budgets (where rollout-policy quality matters most) or many
more training rounds. Phase 75's harness (`crates/evaluation/src/rl_benchmark_v2.rs`)
is reusable for either follow-up without modification, only different
arguments.

## 9. Phases 81–90: finishing the RL question honestly, then new capability

Phase 80 ended by naming the follow-up a fair test would need: either a
much larger eval set, or a low-search-budget regime where rollout-policy
quality matters more than search itself. Phases 81-84 did both rather than
letting that stay a suggestion:

- **Phase 81** (`crates/rl/src/effect_size.rs`) adds Cohen's h and a required-sample-size
  calculator, so "how big would N need to be" is computed, not guessed.
  It also *retroactively validates* Phase 75's null result: detecting the
  uniform-vs-PRM gap actually observed there (0.75 vs. 0.575) at 80% power
  would need well over 40 samples per group — confirming N=40 genuinely
  was underpowered for an effect that size, not just "probably was."
- **Phase 82/83** (`crates/evaluation/src/rl_benchmark_low_budget.rs`) reran the
  four-way comparison at low search budgets. Real results:
  - `eval_budget=25`: every condition hit 0/40 — a floor effect (the
    eval set is too hard for *any* policy at that little search), not a
    "policy doesn't matter" finding.
  - `eval_budget=50`: uniform 0.025, PRM-guided 0.075, GRPO-raw 0.075,
    GRPO-MCTS-sourced 0.050 — still all overlapping at N=40, this time
    because counts are too small near the floor to resolve anything,
    exactly what Phase 81's power calculation would predict.

**Honest conclusion on the RL training question, stated plainly rather
than left open indefinitely:** across three real experiments now (Phase
69 N=20, Phase 75 N=40 @ budget=250, Phase 82 N=40 @ budget=25/50), no
configuration has produced a statistically detectable improvement from
GRPO training over pure random-rollout MCTS search. The mechanism-level
fix from Phase 71 (MCTS-sourced training episodes) demonstrably works —
it turns a flat 0.0 training solve-rate into a rising one — but three
short rounds of a 5-parameter linear policy is not enough training signal
to clear the bar this evaluation methodology requires. Given the
diminishing returns of continuing to chase significance here without
running a much longer training regime (which risks p-hacking rather than
genuine progress), this line of experimentation is left at this honest
resting point rather than pushed further in this batch. The machinery
(Phases 51-79) is correct, tested, and reusable for a longer run whenever
that's worth the compute.

Phases 85-90 build new, independent capability rather than continuing to
probe the same question:

- **Phase 85** (`crates/search/src/trigonometry_domain.rs`) — trigonometric
  simplification and standard-angle evaluation, the "Trigonometry" item
  from design doc section 9 that had no domain until now. Same
  candidate-selection-plus-symbolic-ground-truth pattern as the existing
  calculus domain (Phase 041) — the ground truth comes from
  `reasoning-symbolic`'s exact trigonometry, the mini-CAS that replaces
  sympy in this port; sympy's simplification isn't re-implemented by hand.
- **Phase 86** (`crates/search/src/linear_algebra_domain.rs`) — determinant,
  matrix multiplication, and small linear-system solving via exact
  rational matrices, the "Linear algebra" item from the same list. (The
  Python original used `sympy.Matrix`; the Rust port's equivalent is the
  `RatMatrix` in `reasoning-common` — Bareiss determinant, exact solve,
  nullspace.)
- **Phase 87** — 16 tests for both, each cross-checked against a
  hand-computed value (e.g. `det([[3,8],[4,6]]) = 3*6-8*4 = -14`,
  `sin(pi/6)=1/2`) rather than only checking the CAS against itself.
- **Phase 88** (`crates/memory/src/semantic_memory.rs`) — the semantic-memory /
  episodic-memory distinction design doc section 14 draws but the
  codebase only had half of (episodic, Phase 026). Stores *general*
  verified facts/identities (not tied to one problem instance), and
  structurally cannot store an unverified one: `store()` requires
  equation form and independently re-checks it via the same algebraic
  verifier used everywhere else before accepting.
- **Phase 89** wires all five new problem kinds
  (`trig_simplify`, `trig_evaluate`, `matrix_determinant`,
  `matrix_multiply`, `linear_system`) into the existing unified
  `solve()` API (Phase 030), following its established pattern:
  explicit typed routing, verified-or-abstain, no free-text
  classification bolted on. Existing `solve()` tests (including the
  injection-resistance regression test) still pass unchanged.

## 10. On "better reasoning than other AI systems"

Per `CAPSTONE.md`'s own framing (written before this batch, still true
after it): this project is not, and was never going to be, a system that
outperforms frontier LLMs at general reasoning — it's a small,
verification-gated search system that is unusually hard to fool on the
narrow domains it covers (integer arithmetic, linear equations, some
number theory/combinatorics/calculus), because every claimed answer is
independently re-checked by a symbolic verifier rather than trusted from
generation. Phases 51–70 add real machinery for training a policy via
correct, tested RL math instead of pure search — Phase 69's results above
are a first, honest, statistically underpowered look at whether that
machinery helps, not a benchmark win.
