# Reasoning AI — Design Document (v0.3, math-first)

> **Rust port:** this project was ported from Python to Rust. File paths use
> `crates/<name>/src/`; commands use `cargo`. The original Python layout
> (`python/<pkg>/<mod>.py`) maps to `crates/<pkg>/src/<mod>.rs`. See
> `README.md` for the current workspace layout and commands.

## 0. Scope & Honesty Statement — read this first

Scope: **general reasoning, with mathematics as the flagship domain built
and proven first.** The core machinery — small policy model, search over
short steps, external verifier, self-evolution loop — is not inherently
math-specific. What changes per domain is only the *verifier*:

| Domain | Verifier | Status |
|---|---|---|
| Math | symbolic/numeric evaluation | primary, built first |
| Code | compile + run + test execution | same verifier pattern, easy add |
| Formal logic | SAT/SMT solver, truth-table check | straightforward add |
| Factual/scientific claims | retrieval + cross-check, unit consistency | harder — no clean ground truth |
| Open-ended language reasoning | no reliable automatic verifier | out of reach for this approach |

The reason math goes first isn't a hard restriction — it's that math has
the cleanest verifier, so we can prove the whole loop (generate -> search ->
verify -> self-evolve) actually works before spending effort on domains
where "correct" is fuzzier. Once the math pipeline is solid, extending to
code and formal logic is mostly swapping in a new verifier crate, not a
redesign. Domains without a reliable automatic verifier (open-ended
language reasoning, subjective judgment) are a real ceiling for this
architecture — no amount of search fixes an unverifiable domain, so I won't
claim this approach generalizes there.

There is real published precedent for the core idea: Microsoft's **rStar-Math** showed a 7B-parameter small
model, guided by Monte Carlo Tree Search and a process reward model, improved
from 58.8% to 90.0% on the MATH benchmark — surpassing OpenAI's o1-preview —
without ever being distilled from a bigger model. Each reasoning step
produced runnable code that was executed and checked, so wrong steps were
thrown out before they could compound (arXiv:2501.04519).

**Why math is special:** in open-ended language, "correct" is fuzzy. In
math, correctness is *decidable* — a numeric answer, an algebraic identity,
or a piece of code either checks out or it doesn't. That means we don't need
the neural net to "know" mathematics the way a large LLM does. We need it to
be a good **proposal generator** for the next small step, while an external,
non-neural **verifier** (symbolic engine + numeric evaluator) supplies ground
truth. The model's job shrinks from "know the answer" to "guess plausible
next steps often enough that search finds a verified path." That's a task a
genuinely tiny model can do well, if the search and verification around it
are strong.

**The "1k tokens, unbelievable reasoning" idea — how it actually works:**
1k tokens is not the *total* reasoning budget — it's the size of **one
step**. The system will:
- Generate many short (≤1k token) candidate next-steps, not one giant
  chain-of-thought.
- Search over a tree of these steps (MCTS), not commit to the first guess.
- Verify every step immediately via symbolic/numeric execution — a step
  that doesn't check out is pruned, never built on.
- Only a *verified* path becomes the final answer.

So total computation can be large (thousands of short generations + tree
search + symbolic checks) even though every individual generation is tiny.
This mirrors rStar-Math's real design, not a metaphor.

**What I will not do:** claim a specific benchmark number before running it,
claim this beats every model on every math topic (Olympiad geometry proofs
and open research problems are genuinely much harder than competition
algebra — I'll be explicit about which sub-domains are in reach first), or
silently drop the verification step to make demos look better.

---

## 1. System Architecture

```
INPUT
  |
UNDERSTAND        (tokenize, parse intent)
  |
MEMORY            (retrieve relevant facts/past solutions)
  |
GOAL              (extract explicit goal + constraints)
  |
PLAN              (decompose into subgoals)
  |
REASON            (generate candidate reasoning trajectories)
  |
SEARCH            (beam / best-first / MCTS over trajectories)
  |
TOOLS             (calculator, symbolic engine, code execution)
  |
VERIFY            (independent checkers: numeric, symbolic, logical)
  |
CRITIC            (compare candidates, detect contradictions)
  |
CORRECT           (revise if verification fails; backtrack)
  |
FINAL ANSWER
```

Training loop:

```
DATA -> SUPERVISED TRAINING -> REASONING -> VERIFICATION -> REWARD
     -> GRPO/PPO -> SELF-PLAY -> SYNTHETIC DATA -> CURRICULUM
     -> DISTILLATION -> EVALUATION -> REPEAT
```

---

## 2. Mathematical Foundations

### 2.1 Core tensor operations
- Scalar, vector, matrix, N-D tensor
- Matmul: `C[i,j] = sum_k A[i,k] * B[k,j]`
- Broadcasting rules (NumPy-style)
- Reductions: sum, mean, max over axes

### 2.2 Information theory
- Entropy: `H(p) = -sum_i p_i log p_i`
- Cross-entropy loss: `L = -sum_i y_i log(p_i)`
- KL divergence: `D_KL(P||Q) = sum_x P(x) log(P(x)/Q(x))`

### 2.3 Calculus / autodiff
- Chain rule: `d(f(g(x)))/dx = f'(g(x)) * g'(x)`
- Jacobian: matrix of all first-order partials
- Reverse-mode autodiff (backprop): build computation graph forward,
  propagate gradients backward via chain rule, one node at a time.

### 2.4 Probability & statistics
- Expected value: `E[X] = sum_x x * P(x)`
- Variance: `Var(X) = E[(X - E[X])^2]`
- Std dev: `sqrt(Var(X))`
- Bayes' rule: `P(A|B) = P(B|A) P(A) / P(B)`

### 2.5 RL mathematics
- MDP: `(S, A, P, R, gamma)`
- Bellman equation: `V(s) = E[R(s,a) + gamma * V(s')]`
- TD error: `delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)`
- GAE: `A_t = sum_{l=0}^inf (gamma*lambda)^l * delta_{t+l}`
- Policy ratio: `r_t = pi_theta(a_t|s_t) / pi_old(a_t|s_t)`
- PPO-clip objective:
  `L_CLIP = E[min(r_t * A_t, clip(r_t, 1-eps, 1+eps) * A_t)]`
- GRPO: for a group of N sampled completions to the same prompt, compute
  reward `r_i`, normalize `A_i = (r_i - mean(r)) / (std(r) + eps)`, and use
  `A_i` in place of a learned value baseline — removes the need for a
  separate critic network.

### 2.6 Reward composition (Phase 12 target)
```
reward = w1*correctness + w2*partial_credit + w3*verification_pass
       + w4*logical_consistency - w5*contradiction - w6*hallucination
       - w7*reward_hacking_penalty
```
No term rewards raw token count or chain-of-thought length directly.

### 2.7 Process Reward Model (PRM) — step-level scoring

Instead of only scoring a final answer, score every intermediate step so
search can prune bad branches early:
```
Q(s) = P(step s leads to a verified-correct final answer)
```
Trained from search rollouts: steps on paths that terminated in a verified
answer get positive labels; steps on paths that failed verification get
negative labels. No human step-labeling required — the symbolic verifier is
the label source (this is the "process preference model" idea from
rStar-Math).

### 2.8 MCTS selection (UCB)

At each node, choose the child that maximizes:
```
UCB(s,a) = Q(s,a) + c * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
```
where `Q` = average value from rollouts, `P` = policy model's prior
probability for that step, `N` = visit counts, `c` = exploration constant.
This balances exploiting known-good steps against exploring new ones —
identical in form to AlphaZero/AlphaGeometry-style search.

### 2.9 Self-evolution loop (bootstrapping without a bigger teacher)

```
round 0: seed policy model (small, weak) + symbolic verifier only
round k:
  1. Sample problems from curriculum
  2. Run MCTS: policy proposes steps, verifier checks each step
  3. Keep only trajectories that reach a VERIFIED correct final answer
  4. Label every step along verified paths with its rollout Q-value
  5. Retrain policy model on (step -> next-step) from verified paths
  6. Retrain PRM on (step -> Q-value) from all explored paths
  7. Increase curriculum difficulty where accuracy is already high
  -> round k+1
```
This is how a small model gets *better than its own starting ability*
without copying a bigger model: it manufactures its own verified training
data via search, and only search-verified data ever enters training —
never a hallucinated step.

---

## 3. Crate Layout (Rust workspace, math-first / domain-pluggable)

`verifier/` is a trait, not a fixed implementation — `symbolic` backs it for
math now; a `code_exec` or `sat_solver` backend can implement the same trait
later without touching search, PRM, or training. World model, episodic
memory, and open-ended tool-calling are dropped for now — they don't have a
verifier and would just reintroduce unverifiable generation.

```
math_reasoning_ai/
├── Cargo.toml                 # workspace
├── crates/
│   ├── tensor/                # Phase 1 — THIS PHASE
│   ├── autograd/
│   ├── nn/
│   ├── transformer/            # small policy model
│   ├── tokenizer/               # math-aware (numbers, symbols, LaTeX-ish)
│   ├── symbolic/                # expression trees, simplification, CAS
│   ├── verifier/                # numeric + symbolic + code-exec checkers
│   ├── search/                  # MCTS + beam + best-first
│   ├── prm/                     # process reward model
│   ├── rl/ (ppo/, grpo/)
│   ├── curriculum/              # difficulty-ordered problem sets
│   ├── selfplay/                # self-evolution loop (2.9)
│   ├── dataset/
│   ├── training/
│   ├── optimizer/
│   └── evaluation/              # MATH / GSM8K / AIME-style benchmark harness
└── apps/ (train/, infer/, evaluate/, benchmark/)
```

*How this landed:* the project was first built in Python (see section 5),
then ported to Rust. The shipped workspace (`/reasoning-ai-rust`, 19 crates)
realizes this plan with adjusted names — `symbolic` became the hand-written
mini-CAS `reasoning-symbolic` (sympy has no Rust equivalent), the
autograd/nn/optimizer split collapsed into `reasoning-neural` (the
Transformer PRM with from-scratch backprop), and the small policy model
stayed the 5-parameter linear-softmax policy in `reasoning-rl`. The full
crate table is in `README.md`.

## 3.5 Domain rollout order (easiest verified -> hardest)

Verification difficulty, not "intellectual difficulty," sets the order —
easy-to-verify domains come first so search and self-evolution have a clean
reward signal from day one.

1. Integer arithmetic, order of operations — trivial to verify (evaluate).
2. Linear equations / systems — verify by substitution.
3. Polynomial algebra, factoring — verify via symbolic expansion equality.
4. Word problems -> equation extraction -> solve — verify final numeric answer.
5. Number theory (GCD, modular arithmetic, primality) — verify computationally.
6. Combinatorics / counting — verify by brute-force enumeration at small N.
7. Calculus (derivatives, integrals, limits) — verify symbolically/numerically.
8. Competition-style multi-step problems (AMC/AIME-difficulty) — verify final
   numeric answer, steps scored by PRM.
9. Formal proof / Olympiad geometry — deferred; needs a proof checker
   (Lean/Coq-style), substantially harder engineering, flagged as future
   work rather than promised now.

## 4. Phase 001 (this session)

**Name:** Tensor Creation
**Objective:** Basic N-D tensor abstraction: shape, strides, data buffer,
construction, indexing, equality.
**Files:** `crates/tensor/src/tensor.rs`, `crates/tensor/src/lib.rs`
**Test:** create tensors of rank 0/1/2/3, verify shape, verify element
access, verify a shape-mismatch error path.
**Expected result:** all tests pass, numerically verified (not assumed).

*(This is one of the few sections where the paths were written for Rust from
the start — and they are exactly the paths that now exist: Phase 001 lives
at `crates/tensor/src/tensor.rs` in this workspace, tests in
`crates/tensor/tests/test_tensor.rs`.)*

## 5. Environment Constraint (read before continuing)

This chat sandbox has **no Rust toolchain and no network access** — I cannot
run `cargo build`/`cargo test` here, which breaks the project's own rule
that every phase must compile and pass tests before moving on. Options:

1. **I write full Rust source now**, you compile/test it on your own
   machine (I'll give exact commands). Fast for you long-term, but I can't
   verify Phase 001 myself before writing Phase 002.
2. **I implement the same phase in Python in parallel**, which I *can*
   actually execute and numerically verify right here, then port verified
   logic to Rust. Slower but keeps the "never assume correctness" rule
   honest.
3. Mix: Python for everything through the RL/math logic (fast iteration,
   verified), Rust only for the final production inference engine.

Tell me which and I'll proceed — I lean toward option 2/3 since it's the
only way I can actually keep the project's own testing discipline.

*(Resolution: option 2 was chosen — 126 phases were built and executed in
Python, then the whole system was ported to this Rust workspace, where every
phase compiles and its ported tests run under `cargo test` — 595 tests,
0 failures. The constraint is closed.)*
