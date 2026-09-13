# OTD 2.3 "SOLIDITY" — start here

## Build & run (one command builds the engine + both intelligence tools)
    ./run.sh                    # builds, then serves the viewer on :6830

## The three new laws (try these first)
    ./target/release/otd --check examples/solidity-stack.otd
    #   things FALL and LAND; "zero interpenetrations" is verified, not hoped
    ./target/release/otd --check examples/environment-water.otd
    #   environment: water — oak & ice rise, concrete & steel sink
    ./target/release/otd --check examples/gas-mixing.otd
    #   gases rise/sink, then DIFFUSE into one phase — N2+O2 ≈ air
    ./target/release/otd --check examples/liquid-chemistry.otd
    #   liquids stack into density layers; `mix:` explains each verdict

## The engine understands real things (pre-added intelligence)
    ./target/release/otd --ai "balance the equation H2 + O2 -> H2O"
    #   → VERIFIED: 2 H2 + O2 -> 2 H2O   [confidence 100%]
    #   (the reasoning engine ABSTAINS when it cannot verify — by design)
    ./target/release/otd --perceive examples/solidity-stack.otd
    #   renders a stereo pair, AVC sees the scene:
    #   objects, metric centres/extents, relations (left_of, above, …)

## Render the demos
    ./target/release/otd --png examples/solidity-stack.otd stack.png --settle
    ./target/release/otd --png examples/liquid-chemistry.otd beaker.png

## What's new in 2.3 (full story in CHANGELOG.md)
- simulate: settle — THE floating fix + real solidity (nothing inserts into anything)
- simulate: solidity — static interpenetration audit
- environment: air | vacuum | water | oil | density N — buoyancy + drag
- simulate: gas + 10 real gases — mixing by diffusion, reactions with verified equations
- simulate: mix + mix: A + B + 3 new liquids — miscibility, layered beakers
- vendor/reasoning-ai + vendor/avc + the --ai / --perceive bridges
- material table 31 → 44 (liquids + gases at real STP values)

screenshots-2.3/ holds this release's demo renders and console transcripts.
