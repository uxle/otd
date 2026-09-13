//! Camera models and stereo rectification.

pub mod model;
pub mod rectify;

pub use model::{look_at, Camera, Intrinsics, StereoRig};
pub use rectify::{apply_h, rectify_pair, remap_bilinear, RectifiedRig};
