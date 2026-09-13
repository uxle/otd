//! Reasoning AI — memory crate (Rust port of the Python `memory` package,
//! design doc section 14): working / episodic / semantic / failure memory.
//!
//! No module here serializes to JSON (Python used plain dicts/strings held in
//! memory only), so no serde derives.

pub mod episodic_memory;
pub mod failure_memory;
pub mod semantic_memory;
pub mod working_memory;

pub use episodic_memory::{EpisodicMemory, EpisodicMemoryEntry};
pub use failure_memory::{FailureMemory, FailureMemoryDomain};
pub use semantic_memory::{SemanticFact, SemanticMemory};
pub use working_memory::{WorkingMemory, WorkingMemoryEntry, WmValue};
