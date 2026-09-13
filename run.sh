#!/bin/sh
# OTD — build everything (engine + the two vendored intelligence toolchains)
# and start the 3D viewer. The vendor builds are one-time (~2 min each).
cd "$(dirname "$0")"
if [ ! -x target/release/otd ]; then
    echo "building the OTD engine (needs the Rust toolchain)…"
    cargo build --release
fi
# one-time: build the pre-added reasoning engine (verified answers)
if [ ! -x vendor/reasoning-ai/target/release/reasoning-ai ]; then
    echo "building the reasoning engine (vendor/reasoning-ai, one-time)…"
    cargo build --release --manifest-path vendor/reasoning-ai/Cargo.toml || \
        echo "  (skipped — --ai will say how to build it later)"
fi
# one-time: build the perception engine (stereo world model)
if [ ! -x vendor/avc/target/release/avc-process-pair ]; then
    echo "building the perception engine (vendor/avc, one-time)…"
    cargo build --release --manifest-path vendor/avc/Cargo.toml || \
        echo "  (skipped — --perceive will say how to build it later)"
fi
exec ./target/release/otd "$@"
