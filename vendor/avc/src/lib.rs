//! # Artificial Visual Cortex (AVC) v3 - Rust Edition
//!
//! A real-time stereo vision perception engine that maintains a metric 3D world
//! model with honest uncertainty quantification at every level.
//!
//! Pipeline: rectification -> census/SGM stereo (half-pixel grid) -> 3D lifting
//! -> ground segmentation -> object detection -> Kalman tracking -> world model
//! -> spatial relations -> measurement engine, with visual odometry feeding the
//! world frame, and voxel point cloud / TSDF surface reconstruction.
//!
//! Design principles (inherited from AVC v1/v2):
//! 1. **No false precision.** Every number carries a sigma or a status flag.
//! 2. **Honest status matrix.** Modules self-report WORKING / PARTIAL / INCOMPLETE.
//! 3. **Ground-truth validation.** The bundled synthetic world renderer provides
//!    per-pixel ground truth; the demo measures every claim it makes.
//!
//! This crate compiles to a single native binary with no runtime dependency.

pub mod core;
pub mod camera;
pub mod stereo;
pub mod sim;
pub mod perception;
pub mod world;
pub mod slam;
pub mod engine;
pub mod export;

/// Crate version string (used by CLI apps and export headers).
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
