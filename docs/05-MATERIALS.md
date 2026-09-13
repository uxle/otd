# OTD 1.0 — Materials & Physics Database

Every material is predefined with **real engineering values** (typical, rounded
for education). Users never type physics — they type a name and the science
comes with it.

## Density (the star of the show)

`mass = volume × density`, computed automatically from the true mesh volume.

| Material | Density kg/m³ (g/cm³) | Color (hex) | Rough | Metal | Melts at °C |
|---|---|---|---|---|---|
| iron | 7874 (7.87) | #4e4e4e | 0.55 | 1 | 1538 |
| steel | 7850 (7.85) | #b8bcc2 | 0.35 | 1 | 1425 |
| stainless | 7900 (7.90) | #cfd2d6 | 0.25 | 1 | 1450 |
| aluminum | 2700 (2.70) | #c8cdd2 | 0.30 | 1 | 660 |
| copper | 8960 (8.96) | #b87333 | 0.25 | 1 | 1085 |
| brass | 8500 (8.50) | #c4a04d | 0.30 | 1 | 930 |
| bronze | 8800 (8.80) | #cd7f32 | 0.35 | 1 | 950 |
| gold | 19320 (19.32) | #ffd700 | 0.15 | 1 | 1064 |
| silver | 10490 (10.49) | #dcdcdc | 0.10 | 1 | 962 |
| titanium | 4506 (4.51) | #9fa3a6 | 0.40 | 1 | 1668 |
| zinc | 7135 (7.14) | #c8ccd0 | 0.35 | 1 | 420 |
| lead | 11340 (11.34) | #6a6d70 | 0.60 | 1 | 327 |
| chrome | 7190 (7.19) | #e8eaed | 0.05 | 1 | 1907 |
| tungsten | 19250 (19.25) | #8c9196 | 0.35 | 1 | 3422 |
| wood | 700 (0.70) | #8d6e63 | 0.85 | 0 | — |
| oak | 755 (0.76) | #a98153 | 0.80 | 0 | — |
| pine | 500 (0.50) | #d9c08c | 0.85 | 0 | — |
| teak | 660 (0.66) | #b08650 | 0.75 | 0 | — |
| glass | 2500 (2.50) | #cfe8ef* | 0.05 | 0 | 700 (softens) |
| plastic | 1050 (1.05) | #f0f0f0 | 0.50 | 0 | 100 (softens) |
| rubber | 1150 (1.15) | #2a2a2a | 0.95 | 0 | 180 |
| ceramic | 2400 (2.40) | #f5f0e6 | 0.30 | 0 | 1400 (fires) |
| concrete | 2400 (2.40) | #9a9a9a | 1.00 | 0 | — |
| marble | 2711 (2.71) | #f0ece8 | 0.25 | 0 | — |
| fabric | 300 (0.30) | #8a7f76 | 1.00 | 0 | — |
| carbon | 1600 (1.60) | #1b1b1b | 0.35 | 0 | 3600 (sublimes) |

*glass renders transparent.

## Full Physics Table

| Material | Thermal W/(m·K) | Electrical MS/m | Friction μ | Bounce | Magnetic | Notes |
|---|---|---|---|---|---|---|
| iron | 80 | 10.0 | 0.45 | 0.25 | yes | rusts |
| steel | 50 | 6.0 | 0.40 | 0.30 | yes | workhorse |
| stainless | 16 | 1.3 | 0.35 | 0.30 | no | kitchen |
| aluminum | 237 | 37.0 | 0.35 | 0.35 | no | light, conducts heat fast |
| copper | 401 | 59.6 | 0.30 | 0.30 | no | best everyday conductor |
| brass | 120 | 15.0 | 0.35 | 0.30 | no | instruments |
| bronze | 30 | 7.0 | 0.35 | 0.30 | no | statues, tools (history!) |
| gold | 317 | 44.0 | 0.20 | 0.30 | no | never tarnishes |
| silver | 429 | 63.0 | 0.20 | 0.30 | no | best conductor of all |
| titanium | 22 | 2.4 | 0.40 | 0.30 | no | strong + light |
| zinc | 116 | 16.9 | 0.35 | 0.30 | no | galvanizing |
| lead | 35 | 4.7 | 0.50 | 0.20 | no | very dense, very soft |
| chrome | 94 | 8.0 | 0.20 | 0.35 | no | mirror shine |
| tungsten | 170 | 18.0 | 0.45 | 0.25 | no | highest melting metal |
| wood | 0.15 | — | 0.50 | 0.30 | no | insulator |
| glass | 1.05 | — | 0.40 | 0.65 | no | bounces like a marble |
| plastic | 0.20 | — | 0.35 | 0.55 | no | kids' toys |
| rubber | 0.15 | — | 1.00 | 0.85 | no | the bounciest |
| ceramic | 1.50 | — | 0.50 | 0.40 | no | coffee cups |
| concrete | 1.00 | — | 0.80 | 0.20 | no | buildings |
| marble | 2.80 | — | 0.45 | 0.30 | no | palaces |
| fabric | 0.04 | — | 0.90 | 0.10 | no | softest landing |
| carbon | 10 | — | 0.30 | 0.30 | no | bikes, planes |

## Physics Constants

| Constant | Value | Used by |
|---|---|---|
| g (earth) | 9.81 m/s² | `simulate drop` |
| g (moon / mars) | 1.62 / 3.71 m/s² | gravity presets |
| water density | 1000 kg/m³ | `simulate float`, `ask "will it float?"` |
| water plane | y = 0 in `simulate float` | — |

## Worked Examples (verified in the test suite)

**Coffee cup** (ceramic, wall 3 mm): outer frustum 40/30 × 100 mm, inner 37/27 × 97 mm.
- outer V = π·100/3·(40² + 40·30 + 30²) = 387 463 mm³
- inner V = π·97/3·(37² + 37·27 + 27²) = 314 591 mm³
- shell V = 72 872 mm³ + handle torus 2π²·20·6² = 14 212 mm³ → **87 084 mm³**
- mass = 87.084 cm³ × 2.40 g/cm³ = **209 g**

**Steel cube 10 cm:** 1 000 000 mm³ × 7.85 g/cm³ = **7.85 kg**

**Gold sphere r = 5 cm:** 523 599 mm³ × 19.32 g/cm³ = **10.12 kg**

**Oak cube floating:** ρ = 755 < 1000 → floats with **75.5% submerged**
(Archimedes: fraction = ρ_object / ρ_water).

**Drop from 50 cm (earth):** t = √(2h/g) = 0.319 s, v = √(2gh) = 3.13 m/s.

## New in 2.3 — liquids & gases (real chemistry + STP densities)

**Liquids (6):** water 997 · oil 920 · mercury 13 546 (liquid metal, beads) ·
ethanol 789 · acetone 784 · glycerin 1261 kg/m³.
Miscibility is real: polar pairs (water/ethanol/acetone/glycerin) mix into
one phase; oil is non-polar and refuses them; mercury refuses everything.

**Gases (10, STP kg/m³):** hydrogen 0.0899 · helium 0.1786 · methane 0.717 ·
ammonia 0.769 · nitrogen 1.165 · air 1.225 · oxygen 1.429 · steam 0.598 ·
carbon_dioxide 1.977 · chlorine 3.214.
Gases are exempt from the solidity law — they interpenetrate by nature
(that is mixing). Reactive pairs (H2+O2, CH4+O2, H2+Cl2) react with
balanced, verified equations instead of mixing.

**Material table: 44 entries.**
