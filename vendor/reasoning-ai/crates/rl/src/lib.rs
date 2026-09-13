//! Reasoning AI — RL crate (Rust port of the Python `rl` package):
//! the RL math stack (Phases 051-059: returns, GAE, advantage, PPO-clip,
//! GRPO, KL penalty, entropy regularization, reward composition and
//! normalization), the trainable softmax policy (Phase 061) and its
//! gradient/update machinery (062-064), policy<->MCTS bridges (065, 071),
//! self-play training loops (066-067, 073, 076, 093-094), statistics
//! (074), effect sizes (081), checkpointing (078) and distillation
//! (091-092).
//!
//! `curriculum_grpo.rs` (Phase 076) is pending the `reasoning-curriculum`
//! crate — see that file for details.

pub mod adaptive_exploration;
pub mod advantage;
pub mod checkpoint;
pub mod distill_self_play;
pub mod distill_training_loop;
pub mod distill_update;
pub mod distillation;
pub mod effect_size;
pub mod entropy_reg;
pub mod gae;
pub mod grpo;
pub mod grpo_self_play;
pub mod grpo_training_loop;
pub mod grpo_training_loop_mcts;
pub mod grpo_update;
pub mod kl_penalty;
pub mod mcts_episode_source;
pub mod policy;
pub mod policy_grad;
pub mod ppo;
pub mod ppo_update;
pub mod reward_model;
pub mod reward_norm;
pub mod returns;
pub mod rl_loop;
pub mod stats;
pub mod trainable_rollout;
pub mod training_diagnostics;

// pub mod curriculum_grpo; // pending reasoning-curriculum (see file)

pub use adaptive_exploration::{AdaptiveExplorationController, ExplorationSummary};
pub use advantage::{advantages_from_returns, advantages_from_rewards, td_errors};
pub use checkpoint::{
    load_policy_checkpoint, load_training_history, save_policy_checkpoint,
    save_training_history,
};
pub use distill_self_play::{run_distill_round, DistillRoundDiagnostics};
pub use distill_training_loop::run_distill_training;
pub use distill_update::{distill_weight_update, DistillSample, DistillUpdateDiagnostics};
pub use distillation::{
    distillation_grad_log_probs, soft_target_cross_entropy, softmax,
    visit_counts_to_teacher_distribution,
};
pub use effect_size::{cohens_h, effect_size_label, required_n_for_power};
pub use entropy_reg::{
    apply_entropy_bonus, entropy_from_logprobs, entropy_trajectory_mean, max_entropy,
};
pub use gae::{gae, gae_returns};
pub use grpo::{grpo_all_correct_zero_gradient, grpo_loss, group_advantages};
pub use grpo_self_play::{run_grpo_round, RoundDiagnostics};
pub use grpo_training_loop::{evaluate_policy, run_grpo_training, TrainingHistory, _depth_fn};
pub use grpo_training_loop_mcts::{
    run_grpo_round_mcts, run_grpo_training_mcts, MCTSRoundDiagnostics,
};
pub use grpo_update::{grpo_weight_update, GRPOEpisode, GRPOEpisodeStep, GRPOUpdateDiagnostics};
pub use kl_penalty::{
    apply_kl_penalty, kl_divergence_from_logprobs, kl_sample_trajectory,
};
pub use mcts_episode_source::sample_episode_via_mcts;
pub use policy::{SoftmaxPolicy, FEATURE_KEYS};
pub use policy_grad::{grad_log_prob, max_abs_grad_diff, numerical_grad_log_prob};
pub use ppo::{clip_fraction, ppo_clip_loss, ppo_clip_loss_single};
pub use ppo_update::{
    ppo_gradient_coefficient, ppo_weight_update, PPOSample, PPOUpdateDiagnostics,
};
pub use reward_model::{
    compose_reward, compose_reward_full, reward_from_verification_result, RewardComponents,
    RewardWeights, DEFAULT_WEIGHTS,
};
pub use reward_norm::{clip_advantages, clip_rewards, normalize_advantages, RunningMeanStd};
pub use returns::{discounted_returns, discounted_returns_with_bootstrap};
pub use rl_loop::{compute_rl_batch, run_rollout, RLBatchResult, RolloutStep};
pub use stats::{
    proportions_overlap, required_n_for_margin, two_proportion_z_test, wilson_score_interval,
};
pub use trainable_rollout::{sample_episode, Episode, TrainableRollout};
pub use training_diagnostics::{
    detect_entropy_collapse, detect_policy_stagnation, kl_from_reference,
    policy_entropy_on_probe_states,
};

/// Shared alias: (domain, initial_state, max_depth) — the "problem" tuple
/// every training loop consumes. Type name is local to this crate.
pub type Problem = (reasoning_search::NumberTargetDomain, reasoning_search::NtState, usize);

/// `pub type FeatureMap` re-export from the PRM crate (HashMap<String, f64>).
pub use policy::FeatureMap;
