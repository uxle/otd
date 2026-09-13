//! Synthetic world with per-pixel ground truth.

pub mod renderer;
pub mod scene;

pub use renderer::{render, RenderResult, TriMesh, GROUND_ID};
pub use scene::{GtFrame, GtObjectState, ObjKind, ObjectSpec, RenderedPair, SceneSpec};
