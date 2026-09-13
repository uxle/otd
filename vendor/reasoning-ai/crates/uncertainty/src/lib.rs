//! Reasoning AI — uncertainty crate (Rust port of the Python `uncertainty`
//! package): answer abstention, the Critic role, and graduated confidence.
//!
//! Python `uncertainty/audit_log.py` is NOT here — it wraps `apps.solve`,
//! which lives in the engine crate (ported later, per the workspace's
//! cycle-breaking layering).

pub mod abstention;
pub mod critic;
pub mod graded_confidence;

pub use abstention::{solve_with_abstention, Answer};
pub use critic::{critique, Constraint, CritiqueResult, INTEGER_VALUED, NON_NEGATIVE, POSITIVE};
pub use graded_confidence::{grade_number_target_confidence, GradedConfidence};
