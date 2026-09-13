//! Reasoning AI — science domains (Rust port of the python `science`
//! package's physics and chemistry modules; biology lives in sibling
//! modules of the same package).
//!
//! Every domain follows the same design: textbook formulas are the ground
//! truth, and each computed result is re-verified through an INDEPENDENT
//! path (a different formula, rearrangement, or physical law) before it
//! is returned; inputs that cannot pin an answer down are rejected with
//! an error rather than guessed at.

pub mod chemistry_domain;
pub mod chem_balance_domain;
pub mod chem_gas_domain;
pub mod chem_ph_domain;
pub mod chem_solutions_domain;
pub mod density_domain;
pub mod electricity_domain;
pub mod energy_domain;
pub mod forces_domain;
pub mod momentum_domain;
pub mod physics_domain;
// OTD3 science expansion
pub mod magnetism_domain;
pub mod waves_domain;
pub mod thermo_domain;
pub mod relativity_domain;
// OTD3.3 — the subatomic layer
pub mod particle_domain;

// biology modules (converted in parallel by the bio agent)
pub mod bio_dihybrid_domain;
pub mod bio_dogma_domain;
pub mod bio_ecology_domain;
pub mod bio_popgen_domain;
pub mod biology_domain;

mod val;

pub use val::{Val, ValMap};
