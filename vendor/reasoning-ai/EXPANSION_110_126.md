# Phases 110–126: full science coverage, natural language, CLI, web front-end

> **Rust port:** this project was ported from Python to Rust. File paths use
> `crates/<name>/src/`; commands use `cargo`. The original Python layout
> (`python/<pkg>/<mod>.py`) maps to `crates/<pkg>/src/<mod>.rs`. See
> `README.md` for the current workspace layout and commands.

Continues directly from `EXPANSION_102_109.md`, which ended by flagging two
gaps explicitly: the science/puzzle modules were not wired into the unified
`solve()` API, and there was no natural-language front end. Both are now
closed, plus the science coverage was expanded from 3 topics to 13.

## Phases 110–114 — physics beyond kinematics

New modules (all in `crates/science/src/`), each following the house pattern
(textbook formula + an independent re-check inside the same call):

- **`forces_domain.rs`** — F=ma, weight W=mg, friction f=mu*N
  (flat surface), net acceleration with friction. Every solve re-plugs the
  result through the rearrangements NOT used to compute it; the friction
  path additionally checks the flat-surface identity f/m == mu*g.
- **`energy_domain.rs`** — KE, PE, work, power (W/t and F*v),
  drop-height <-> impact-speed. The impact-speed answer is computed from
  energy conservation and CROSS-CHECKED against the kinematics equation
  `v^2 = u^2 + 2*a*s` from the existing `physics_domain` — two different
  physical laws that must agree numerically before an answer is returned.
- **`momentum_domain.rs`** — p=mv, impulse-momentum, perfectly
  inelastic and elastic 1D collisions. Inelastic verifies momentum
  conservation AND the physical constraint KE_lost >= 0; elastic verifies
  BOTH conservation laws exactly (m1==m2 handled as the exact swap).
- **`electricity_domain.rs`** — Ohm's law, electrical power via
  all three algebraically independent forms (V*I, I^2*R, V^2/R — the
  function REQUIRES >=2 forms to agree rather than trusting one),
  series/parallel networks re-verified by Kirchhoff's voltage/current
  laws driven with a test source.
- **`density_domain.rs`** — rho=mV, P=F/A, hydrostatic P=rho*g*h
  cross-checked against the weight-of-column/area definition, Archimedes
  float test verified via submerged-fraction consistency.

## Phases 115–118 — chemistry beyond molar mass

- **`chem_balance_domain.rs`** — real equation balancing via the
  nullspace of the element-conservation matrix, scaled to the
  minimal positive integer vector (gcd==1 enforced), with element-by-element
  left/right equality verified explicitly before returning. Limiting
  reagent via extent-of-reaction comparison; verified by the limiting
  reactant's leftover being exactly 0 and no reactant going negative.
  Handles nesting: `C3H8 + O2 -> CO2 + H2O` -> `C3H8 + 5 O2 -> 3 CO2 + 4 H2O`.
  (The Python original computed the nullspace with `sympy.Matrix`; the
  Rust port computes it with the exact rational `RatMatrix` in
  `reasoning-common` — the sympy subset this domain needs, no
  dependency.)
- **`chem_gas_domain.rs`** — ideal gas law (solve any of P/V/n/T,
  verified by substituting the answer into the full identity), Boyle,
  Charles, Gay-Lussac, combined. Temperatures must be positive Kelvin;
  Celsius inputs are converted (+273.15) by the NL layer, not assumed.
- **`chem_solutions_domain.rs`** — molarity (with mass->moles via
  molar mass as an independent second path), dilution verified by moles
  conservation, percent composition verified by the sum-to-100% identity,
  percent yield REFUSES >100% as physically inconsistent.
- **`chem_ph_domain.rs`** — pH/pOH/[H+] with every answer checked
  by its inverse transform; the pH+pOH=14 pair is verified by the Kw
  identity [H+][OH-]==1e-14 (a different law than the sum rule);
  neutralization checks acid/base equivalents two ways.

## Phases 119–122 — biology beyond monohybrid crosses

- **`bio_dihybrid_domain.rs`** — exhaustive 16-cell Punnett
  enumeration with exact rationals, verified against the INDEPENDENT
  product rule over monohybrid crosses (independent assortment); the
  textbook AaBb x AaBb -> 9:3:3:1 is asserted as a special case.
- **`bio_popgen_domain.rs`** — Hardy-Weinberg from q^2 (recessive
  phenotype fraction) or from a raw allele census; verified by an
  exact-rational allele-counting round trip, plus a multi-generation
  stability walk (the actual predictive CONTENT of the law: frequencies
  must not drift).
- **`bio_dogma_domain.rs`** — transcription (verified per-position),
  reverse complement (verified by the involution property:
  rc(rc(x)) == x), translation with the standard genetic code table
  (stops terminate; unknown codons raise; no internal stops), base
  composition with Chargaff's law (A==T, G==C) enforced for
  double-stranded input.
- **`bio_ecology_domain.rs`** — discrete/continuous exponential
  growth (the two formulations are cross-checked per call through the
  equivalent-rate identity), doubling time (verified by growing exactly
  one doubling period), logistic growth (verified at its boundaries:
  N(0)==N0 and N stays below K), 10% trophic energy transfer (closed form
  vs level-by-level walk).

## Phase 123–125 — the natural language layer

The original design doc said free-text classification was "a much bigger
and riskier undertaking" than typed routing — deliberately deferred. This
batch builds it as a DETERMINISTIC, testable layer rather than a model:

- **`crates/nlp/src/quantities.rs`** — number+unit extraction with canonicalization:
  "3 minutes" -> 180 s, "72 km/h" -> 20 m/s, "500 mL" -> 0.5 L, "27 C" ->
  300.15 K at the solver. Units are converted BEFORE any formula runs, so
  the formula layer never guesses units. Composite units (m/s^2, kg*m/s,
  kg/m^3) are matched before their prefixes.
- **`crates/nlp/src/parser.rs`** — 22 ordered domain routes (chemistry, biology,
  physics, puzzles), each returning a typed payload + a rationale + the
  exact text spans each slot was read from. Underdetermined questions
  (e.g. kinematics with two unknown slots and no stated target) ABSTAIN
  at the parser level: the solver would verify the arithmetic of a
  different question, which is still a wrong answer to this one.
- **`crates/nlp/src/math_routes.rs`** — 8 math routes (arithmetic, linear, quadratic
  factoring, nCr/nPr, gcd, trig evaluate with degrees->radians, matrix
  determinant, word-problem templates with number-word digitization).
  Bare trig angles are interpreted as DEGREES (documented, and stated in
  the route rationale).
- **`crates/engine/src/nlp_solver.rs`** — `ask(text)` front door returning a JSON-able
  value (parse info + Answer + unit annotations). SI unit labels are appended
  to verified answers because the parser canonicalized the inputs. (The
  Python `nlp/solver.py` moved into the engine crate when the Python
  package cycles were broken for the port — the `ask()` there routes
  through the same `solve()`.)

A 38-question smoke suite (`crates/engine/tests/nl_smoke.rs`) passes 38/38 across all
domains; the formal suite (`crates/engine/tests/test_nl_and_science_routes.rs`) locks
routing AND hand-computed answers as test cases, including
abstention cases.

## Phase 126 — unified solve() wiring, CLI, web front-end

- **`crates/engine/src/science_solve.rs`** — 25 typed routes covering every new
  science + puzzle domain, reachable from the same
  `solve(Problem(kind, payload))` API as the math domains (closing the
  gap flagged at the end of phases 102–109). Each route re-verifies at
  this layer too where an independent check is available.
- **`crates/app/src/main.rs`** — the `reasoning-ai` binary: interactive REPL
  (`cargo run -q -p reasoning-ai`) and one-shot
  (`cargo run -q -p reasoning-ai -- "question"`) with `--json` output
  mode (plus `--explain` routing details).
- **`crates/app/src/bin/api_entry.rs`** — stdin/stdout JSON bridge so a web server can
  submit either `{"question": ...}` (NL) or `{"typed": {kind, payload}}`
  (structured) without ever bypassing verification. Run it as
  `echo '{"question": "..."}' | cargo run -q -p reasoning-ai --bin api_entry`.
- **Web chat UI (Next.js, separate repo layer)** — posts to
  `/api/solve`, which runs the deterministic parser FIRST and only on
  failure falls back to an LLM that must EXTRACT A STRUCTURED PROBLEM
  (kind + payload); the extraction is then solved AND verified by this
  engine. The LLM never produces an answer — it only ever fills in a
  form that the verified engine still has to pass. Answers show the
  verification badge, the domain, and how the question was routed;
  out-of-scope questions are shown as honest abstentions.

## Honest scope statement (unchanged in spirit)

Everything added is still in the "computable ground truth" slice: formula
problems, enumeration problems, and structural string transforms. No open
ended concept explanations are attempted, because there is still no
verifier for them. The NL parser is pattern-based and honestly refuses
phrasings it cannot map; the LLM fallback widens INPUT coverage without
weakening the output guarantee, because its extraction is only a form
that `solve()` still verifies.

## Test count

- 83 new tests in `crates/science/tests/test_science_expansion.rs` (+`_bio`) and
  `crates/engine/tests/test_nl_and_science_routes.rs` (34 additional subtests).
- Full suite: **548 passed** (465 pre-existing + 83 new), zero
  regressions, including the injection-resistance and calibration suites.
- Rust port: the entire suite runs as 595 Rust tests under `cargo test`,
  all passing (the Python and Rust totals count at different granularities
  — per the porting conventions, Python test methods were sometimes split
  into separate `#[test]` functions and parametrized loops condensed).
