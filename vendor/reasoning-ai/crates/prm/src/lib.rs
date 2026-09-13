//! Reasoning AI — prm crate (Rust port of the Python `prm` package): the
//! hand-engineered process reward model (features + linear PRM + SGD),
//! tree-walk training-data extraction, root-depth calibration fix, the
//! algebra-domain feature extension, and the PRM-guided rollout policy.

pub mod calibration_fix;
pub mod data_extraction;
pub mod equation_data_extraction;
pub mod equation_features;
pub mod features;
pub mod guided_policy;
pub mod linear_prm;

pub use calibration_fix::{build_calibrated_training_set, collect_root_examples};
pub use data_extraction::{extract_training_examples, extract_training_examples_from_arena};
pub use equation_data_extraction::{
    extract_equation_training_examples, extract_equation_training_examples_from_arena,
};
pub use equation_features::extract_equation_features;
pub use features::{extract_features, FeatureMap, FEATURE_KEYS};
pub use guided_policy::PrmGuidedPolicy;
pub use linear_prm::LinearPRM;
