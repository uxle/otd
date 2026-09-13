pub mod camera;
pub mod raster;
pub mod png;
// OTD3 — real H.264/AVC video encoding + MP4 muxing (pure Rust)
pub mod video;
// P2230 — the eight-camera preview panel
pub mod octocam;
pub mod ggx;

mod render_impl;
pub use render_impl::*;
