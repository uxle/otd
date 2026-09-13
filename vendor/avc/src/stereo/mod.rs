//! Stereo matching: two-scale census + SGM on a half-pixel cost grid,
//! v4 pyramid engine + motion-compensated temporal fusion.

pub mod census;
pub mod map;
pub mod post;
pub mod sgm;
pub mod temporal;

pub use map::{DisparityMap, MatchStats, Quality, SgmParams};
pub use sgm::{match_pair, match_pair_with_buffers, CostVolume, StereoBuffers, MATCH_BORDER};
pub use temporal::TemporalFuser;
