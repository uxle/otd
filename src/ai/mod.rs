//! P2240 — the self-make layer: AI that writes OTD. `synth` turns a plain
//! goal sentence into a complete, compiling .otd script — shapes, materials,
//! physics statements and all.

pub mod synth;



pub use synth::{synthesize, SynthReport};
