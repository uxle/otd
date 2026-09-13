//! SLAM components: visual odometry (loop closure is a documented
//! INCOMPLETE roadmap item in this edition).

pub mod fast;
pub mod vo;

pub use fast::{fast9, nms};
pub use vo::{VoParams, VoResult, VoState};
