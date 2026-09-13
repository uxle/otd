//! Reasoning AI — search crate (Rust port of the Python `search` package):
//! the Domain contract, MCTS, best-first/beam search, multi-path solving,
//! pruning, and the concrete problem domains.

pub mod domain;
pub mod heuristic_search;
pub mod mcts;
pub mod multi_path;
pub mod number_target_domain;
pub mod pruning;
pub mod bezout_domain;
pub mod calculus_domain;
pub mod codegen_domain;
pub mod combinatorics_domain;
pub mod geometry_domain;
pub mod linear_algebra_domain;
pub mod linear_equation_domain;
pub mod logic_domain;
pub mod quadratic_factoring_domain;
pub mod statistics_domain;
pub mod trigonometry_domain;

pub use domain::{rng_choice, Domain, RolloutPolicy, UniformRollout};
pub use heuristic_search::{beam_search, best_first_search};
pub use pruning::{is_provably_unreachable, reachable_bound};
pub use mcts::{ucb_score, Mcts, SearchResult, TreeNode};
pub use multi_path::{solve_number_target_multi_path, MultiPathResult, PathResult};
pub use number_target_domain::{make_initial_state, NumberTargetDomain, NtAction, NtState};
pub use bezout_domain::{BezoutIdentityDomain, BezoutState};
pub use calculus_domain::{CalcState, DerivativeDomain};
pub use codegen_domain::{CodeGenDomain, CodeGenState};
pub use combinatorics_domain::{brute_force_count, CombinatoricsDomain, CombinatoricsState};
pub use geometry_domain::{exact_ground_truth, GeometryAction, GeometryDomain, GeometryState};
pub use linear_algebra_domain::{
    LinAlgState, LinearSystemDomain, MatrixDeterminantDomain, MatrixMultiplyDomain,
};
pub use linear_equation_domain::{EqState, LinEqAction, LinearEquationDomain};
pub use logic_domain::{
    brute_force_satisfying_assignments, LogicFormula, LogicState, SatisfiabilityDomain,
};
pub use quadratic_factoring_domain::{QFState, QuadraticFactoringDomain};
pub use statistics_domain::{
    exact_expected_value, exact_variance, ExpectedValueDomain, StatsAction, StatsState,
};
pub use trigonometry_domain::{TrigEvaluateDomain, TrigSimplifyDomain, TrigState};
