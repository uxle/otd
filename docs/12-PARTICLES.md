# 12 — The Subatomic Layer: what the things you build are made of

Goal: when OTD3 says a cube of iron weighs 984 grams, the next question
is already loaded — *what is that, exactly?* This document explains the
answer the way the brief asked for everything else: philosophy first,
then the laws, then the numbers that test the laws, then (in the source
tree) the implementation. Everything below was checked against the
actual `src/` code — call sites are named so this stays falsifiable.

## 1. Design philosophy

Three principles, in priority order:

1. **The scene is the census.** The engine already measures every part's
   mass from its mesh volume and material density. That mass is all the
   information the subatomic layer needs: grams ÷ molar mass × N_A gives
   atoms, atoms × Z gives protons and electrons, atoms × (A−Z) gives
   neutrons. No analogies, no "imagine a billion of these" — an actual
   inventory, as real as the mass it came from. `simulate: atom` on the
   chess set counts 8.6×10²⁶ electrons, and the number is load-bearing.
2. **Famous numbers, verified, or not said.** Every quantity in this
   layer is either computed from first principles (Weizsäcker's formula,
   the Rydberg formula, the decay law) or looked up from a table whose
   entries are the textbook values (PDG masses, half-lives). The one
   number we knowingly fudge — Weizsäcker's under-binding of He-4 — is
   documented as the formula's known limitation in a test, not hidden.
3. **The card deck is complete.** The user asked for protons, neutrons,
   electrons, quarks, gluons, photons, and neutrinos. The Standard
   Model has exactly 17 fundamental particles; shipping 10 of them would
   be a gap in the deck. All 17 are in `FUNDAMENTALS`
   (`src/world/particles.rs`), and all 118 elements are in `ELEMENTS`
   (`src/world/atom.rs`) — the same completeness rule as the solar
   system table in P2230.

## 2. The laws

### 2.1 The census: grams → particles

For each visible part (`scene_census`, `src/world/atom.rs`):

```
atoms   = grams ÷ molar_mass × 6.02214076×10²³
protons = electrons = atoms × Z
neutrons           = atoms × (A − Z)
```

The engineering materials map to elements through `composition()` — a
mass-fraction table (steel = 98.5% Fe + 1.5% C, water = 11.2% H +
88.8% O, oak ≈ cellulose C₆H₁₀O₅…). This is why a wooden part answers
with carbon, oxygen AND hydrogen. The charge ledger that closes
`simulate: atom` falls out of the arithmetic: protons = electrons per
atom, so net charge is zero to the last electron — the universe balances
its books.

### 2.2 The quark model, and the famous accounting trick

Proton = u u d, neutron = u d d, and the quark charges (u = +2/3e,
d = −1/3e) sum to the measured hadron charges exactly — that's the
re-check in `hadron_card`. Then the accounting that most people never
meet:

```
m(u) + m(u) + m(d)   =  2.2 + 2.2 + 4.7  =  9.1 MeV
m(proton)            =  938.272 MeV
binding energy       =  929.2 MeV  =  99% of the proton's mass
```

**99% of the mass of you, this scene, and the Earth is not matter — it
is gluon field energy.** E = mc² made flesh. This is the single most
requested fact in the subatomic layer and it is computed, not quoted.

### 2.3 Weizsäcker's liquid drop: why iron kills stars

```
E_B = a_V·A − a_S·A^(2/3) − a_C·Z(Z−1)/A^(1/3) − a_A·(A−2Z)²/A ± a_P·A^(−1/2)
      volume   surface     Coulomb        asymmetry      pairing
```

with (a_V, a_S, a_C, a_A, a_P) = (15.8, 18.3, 0.714, 23.2, 11.5) MeV.
Five numbers fit the whole periodic table:

| Nucleus | formula | measured | error |
|---|---|---|---|
| Fe-56 | 8.76 MeV/A | 8.79 | 0.4% |
| Pb-208 | 7.83 | 7.87 | 0.5% |
| U-238 | 7.60 | 7.57 | 0.4% |
| He-4 | 5.49 | 7.07 | known limitation |

He-4 is where the formula famously fails (it's fit to medium and heavy
nuclei); the test documents this rather than pretending. Fe-56 sits at
the peak — which is why `simulate: atom` tells you an iron cube is where
fusion goes to die, and a uranium slug is above the peak, which is why
fission pays.

### 2.4 The decay law, and live radioactivity

```
λ = ln2 / T½          N(t) = N₀·2^(−t/T½)          A = λN
```

`simulate: decay` finds the radioactive materials (uranium, plutonium,
thorium — and the trace C-14 in any organic material, 1.2 per trillion
carbon atoms), computes N from the part's actual mass, and reports the
activity **in becquerels, right now**. The textbook check: 1 g of
U-238 → 12.4 kBq; the engine's own test asserts it within 4% (fresh
`half_life_s` = 4.468 Gy). A 955 g uranium slug in the atom tour
answers 11.88 MBq — twelve million α-decays per second inside a lump
sitting still on a table, and nothing you do can slow or speed them.
Rutherford's law has no snooze button.

### 2.5 Electron structure: Madelung + the 20 exceptions

Electron configurations follow the (n+ℓ, n) filling rule
(`electron_config`), with the 20 elements that break it corrected by
table (Cr, Cu, Nb, Mo, Ru, Rh, Pd, Ag, La, Ce, Gd, Pt, Au, Ac, Th, Pa,
U, Np, Cm, Lr). The exceptions' tails are **complete above the
noble-gas core** — Au carries its own 4f¹⁴, which a naive
"fill-to-Z-minus-tail" would leave at 4f¹² with a phantom 6s². That bug
was real, was caught by the question `electron configuration of gold`,
and is now pinned by tests in both OTD and the vendored reasoning-AI.

### 2.6 Photons, and the scene's own glow

E = hf = hc/λ. `simulate: particles` ties this to the scene: at its
temperature (default 293 K), everything radiates σT⁴ ≈ 419 W/m² peaking
at 9885 nm — infrared, felt but not seen. A 532 nm green photon carries
2.33 eV; the number is computed from h and c, not looked up.

### 2.7 Neutrinos: the honest ghost story

65 billion solar neutrinos per cm² per second pass through the scene;
336 relic neutrinos per cm³ are the oldest particles in existence. No
simulation of them is possible or needed — they pass through
everything, which is the fact. The briefing says so and moves on.

## 3. The numbers that test the laws

In `src/world/particles.rs` and `src/world/atom.rs` tests, plus
`tests/subatomic.rs` through the real front-end:

- quark charges sum to exactly +1e (proton) and 0e (neutron)
- β⁻ Q-value = 0.782 MeV (n − p − e masses, re-checked by rebuilding m_n)
- 1 eV electron's de Broglie λ = 1.227 nm; electron Compton λ = 2.426 pm
- Weizsäcker vs textbook: table above
- decay law: 25% remaining → exactly 2.000000 half-lives
- 1 g U-238 → 12.4 kBq; 1 g modern carbon → 0.23 Bq of C-14 (13.6 dpm)
- Rydberg: Hα 656.5 nm, Hβ 486.3 nm, Lyman-α 121.6 nm
- every element's configuration sums to Z — all 118
- the atom tour compiles with zero errors and the census runs

## 4. The language

```
particle: proton          # one card from the deck (fundamental or hadron)
particle: gluon
simulate: atom            # the census: shells, nucleons, binding, totals
simulate: decay           # half-lives + live activity (Bq) of this scene
simulate: particles       # the Standard Model briefing, tied to the scene
```

`particle` is keyword #78 (the budget test in `keywords.rs` moved with
it). The vendored reasoning-AI answers the same physics in plain
language — `physics_particles` route, NL-routed:

- "what is the quark composition of a proton" → VERIFIED: up + up + down
- "a sample has 25 percent carbon 14 remaining how old is it" → VERIFIED: 11460
- "binding energy of iron 56" → VERIFIED: 490.54
- "energy of a 500 nanometer photon" → VERIFIED: 2.48 eV
- "electron configuration of gold" → VERIFIED: …4f¹⁴ 5d¹⁰ 6s¹

Every answer verification-gated as always: VERIFIED or ABSTAINED, never
a guess.
