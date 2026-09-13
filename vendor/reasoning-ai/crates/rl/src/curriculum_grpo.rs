//! Phase 076 — Curriculum-aware GRPO training (design doc 20: "Difficulty
//! should adapt according to model performance"). // pending reasoning-curriculum
//!
//! This module wires `curriculum::progression::CurriculumController`
//! (Phase 014, ported by the `reasoning-curriculum` crate) into the
//! MCTS-sourced GRPO loop (Phase 073): problems for each round are drawn
//! at the controller's current level, solve rate at that level feeds
//! back into the controller, which advances or regresses the level for
//! the next round.
//!
//! STATUS: the `reasoning-curriculum` crate was still a placeholder when
//! this crate was ported, so the dependency is intentionally absent from
//! Cargo.toml and this module is excluded from `lib.rs`. To finish the
//! port: add `reasoning-curriculum = { path = "../curriculum" }` to
//! Cargo.toml, uncomment `pub mod curriculum_grpo;` in lib.rs, and
//! implement the three items below (ported from python/rl/curriculum_grpo.py):
//!   - `LEVEL_TO_NUM_COUNT` / `LEVEL_TO_MAX_DEPTH` ({1:2,2:3,3:4,4:5} /
//!     {1:2,2:4,3:6,4:8}, unknown level falls back to the max-key value),
//!   - `make_problems_at_level(level, n_problems, rng)` — random 1..=9
//!     numbers, target = sum of a random non-empty subset (guarantees
//!     solvability),
//!   - `CurriculumTrainingHistory {levels, solve_rates, actions,
//!     eval_solve_rate_before, eval_solve_rate_after}` and
//!     `run_curriculum_grpo_training(rounds, eval_problems,
//!     n_problems_per_round=8, n_samples_per_problem=6,
//!     train_num_simulations=150, eval_num_simulations=60, lr=0.2,
//!     seed=0)` — controller constructed with
//!     `CurriculumController::new(min_level=1, max_level=4,
//!     advance_threshold=0.7, regress_threshold=0.15,
//!     consecutive_needed=2)`, per-round rng
//!     `StdRng::seed_from_u64(seed + round*97)`, GRPO round seeded
//!     `seed + round*137`, evals seeded 8000, `controller.record(
//!     diag.solve_rate)` appended to history.actions.
//! Its tests (TestCurriculumGRPO in test_rl_phases_76_78.py) exercise
//! only `CurriculumController`'s advance/regress/hold semantics plus
//! this loop, and belong with the curriculum crate's own test suite.
