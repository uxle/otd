//! Honest module status matrix - the AVC signature feature.
//!
//! Statuses are NOT hardcoded claims: the engine fills the `evidence`
//! strings from the actual run statistics wherever possible.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Status {
    Working,
    Partial,
    Incomplete,
}

impl Status {
    pub fn as_str(&self) -> &'static str {
        match self {
            Status::Working => "WORKING",
            Status::Partial => "PARTIAL",
            Status::Incomplete => "INCOMPLETE",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ModuleStatus {
    pub name: &'static str,
    pub status: Status,
    pub evidence: String,
}
