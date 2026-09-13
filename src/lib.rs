//! OTD — Open Three-Dimensional Language. Pure Rust + assembly, zero external libraries.
//! P0000: zero-dependency guarantee (see Cargo.toml — [dependencies] is empty).

pub mod phase;
pub mod units;
pub mod math3;
pub mod rng;
pub mod noise;
pub mod font;

pub mod vm;
pub mod lang;
pub mod geo;
pub mod world;
pub mod render;
pub mod simd;
pub mod net;
pub mod export;
pub mod content;
// OTD3 — PerceptAudio 2.0, native: full-spectrum ears with metric ranging
pub mod audio;
// OTD3.1 — the self-make expansion (P2200 series): nn.rs, OTD-Burn,
// statistics, astronomy, and the program synthesizer that writes OTD itself
pub mod nn;
pub mod burn;
pub mod ai;

/// Compile an OTD source string into a World (parts + stats + console + errors).
pub fn compile(src: &str) -> crate::world::World {
    crate::world::eval::compile(src)
}
