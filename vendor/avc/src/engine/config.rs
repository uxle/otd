//! Engine configuration.

use crate::camera::model::Intrinsics;
use crate::perception::detection::DetectParams;
use crate::perception::tracking::TrackerParams;
use crate::slam::vo::VoParams;
use crate::stereo::map::SgmParams;

#[derive(Debug, Clone)]
pub struct EngineConfig {
    pub intrinsics: Intrinsics,
    pub baseline: f64,
    pub sgm: SgmParams,
    pub detect: DetectParams,
    pub tracker: TrackerParams,
    pub vo: VoParams,
    /// disparity-map subsampling for 3D lifting
    pub lift_stride: usize,
    /// world point-cloud voxel size (m)
    pub cloud_voxel: f64,
    /// deterministic seed for RANSAC etc.
    pub seed: u64,
    /// propagate the VO pose covariance into point sigmas. Default FALSE:
    /// detection/tracking precision is local (per-frame measurement noise);
    /// the VO drift sigma is reported separately in the outputs (honest
    /// layering: relative geometry vs absolute world-frame drift).
    pub propagate_vo_cov: bool,
    /// v4: motion-compensated temporal depth fusion. Static-scene noise is
    /// averaged through the VO pose (3-sigma gate keeps moving objects on
    /// their fresh single-frame measurement); the fused sigma is inherited
    /// by lifting, detection and measurement.
    pub temporal_fusion: bool,
}

impl EngineConfig {
    /// VGA demo configuration (matches the synthetic validation world).
    /// v4: ACCURACY mode - full-resolution half-pixel-grid matcher (the
    /// pyramid is the fast/low-spec mode, see `fast()`); disparity range
    /// extended to 64 (the v3 validation failure on the 25 cm box was a
    /// range miss: its ~58 px disparity exceeded nd=52 at frames 10-20),
    /// lift stride 2, temporal fusion on.
    pub fn demo() -> Self {
        EngineConfig {
            intrinsics: Intrinsics::new(700.0, 700.0, 319.5, 239.5, 640, 480),
            baseline: 0.16,
            sgm: SgmParams {
                num_disparities: 64,
                disp_min: 2,
                pyramid: false,
                ..Default::default()
            },
            detect: DetectParams::default(),
            tracker: TrackerParams::default(),
            vo: VoParams::default(),
            lift_stride: 2,
            cloud_voxel: 0.02,
            seed: 12345,
            propagate_vo_cov: false,
            temporal_fusion: true,
        }
    }

    /// Faster QVGA configuration (real-time on modest CPUs). v4: PYRAMID
    /// matcher (dense search at half res + full-res +-3 px refinement) -
    /// the low-spec high-throughput mode; the measured accuracy trade-offs
    /// vs the full-resolution mode are documented in docs/VALIDATION.md.
    pub fn fast() -> Self {
        EngineConfig {
            intrinsics: Intrinsics::new(350.0, 350.0, 159.5, 119.5, 320, 240),
            baseline: 0.16,
            sgm: SgmParams {
                num_disparities: 48,
                p1: 8,
                p2_base: 72,
                paths: 4,
                pyramid: true,
                ..Default::default()
            },
            detect: DetectParams { min_points: 25, ..Default::default() },
            tracker: TrackerParams::default(),
            vo: VoParams { max_features: 200, search_radius: 10, ..Default::default() },
            lift_stride: 2,
            cloud_voxel: 0.03,
            seed: 12345,
            propagate_vo_cov: false,
            temporal_fusion: true,
        }
    }
}
