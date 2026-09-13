//! Core geometry and estimation math shared by every module.

pub mod geo;
pub mod rng;
pub mod se3;
pub mod uncertainty;

pub use geo::{kabsch, ransac_ground, ransac_plane, ransac_rigid, unit, wrap_pi, Plane, RigidFit};
pub use rng::GaussRng;
pub use se3::{skew, so3_log, Se3};
pub use uncertainty::{depth_sigma, point_covariance_through_pose, point_sigma_through_pose, sigma_of_distance};
