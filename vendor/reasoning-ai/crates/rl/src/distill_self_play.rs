//! Phase 093 — Distillation self-play round.
//!
//! For each problem: run MCTS with the current policy as rollout_policy
//! (exactly Phase 071's setup), then instead of extracting a single
//! GRPO-style episode with a scalar terminal reward, extract a
//! DistillSample at EVERY node along the greedy visit-count path — each
//! node's children's visit counts become that state's teacher
//! distribution (Phase 091). Uses strictly more of what MCTS already
//! computed per problem than GRPO does, which is exactly the efficiency
//! argument for trying this alongside GRPO rather than only in place of
//! it.

use std::collections::HashMap;
use std::rc::Rc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reasoning_search::{Domain, Mcts, NtAction, NtState, NumberTargetDomain};

use crate::distillation::visit_counts_to_teacher_distribution;
use crate::distill_update::{distill_weight_update, DistillSample};
use crate::policy::SoftmaxPolicy;
use crate::trainable_rollout::TrainableRollout;

use crate::Problem;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DistillRoundDiagnostics {
    pub mean_loss_before_update: f64,
    pub grad_norm: f64,
    pub solve_rate: f64,
    pub n_samples: usize,
}

/// Returns `(samples, passed)` — one DistillSample per node visited along
/// the greedy visit-count path, plus whether that path reached a
/// verified terminal state (for the round-level solve-rate diagnostic
/// only; distillation itself doesn't need this).
///
/// Python defaults: num_simulations=150, rollout_epsilon=0.1,
/// temperature=1.0. Errors (invalid temperature) propagate like the
/// Python ValueError.
#[allow(clippy::too_many_arguments)]
pub fn extract_distill_samples(
    policy: &SoftmaxPolicy,
    domain: &NumberTargetDomain,
    initial_state: &NtState,
    target: f64,
    max_depth: usize,
    depth_fn: Rc<dyn Fn(&NtState) -> usize>,
    rng: &mut StdRng,
    num_simulations: usize,
    rollout_epsilon: f64,
    temperature: f64,
) -> Result<(Vec<DistillSample>, bool), String> {
    let rollout = TrainableRollout::new(
        policy.clone(),
        domain.clone(),
        target,
        max_depth,
        depth_fn,
        rollout_epsilon,
    );
    let seed = rng.gen::<u64>();
    let mut mcts = Mcts::new(domain.clone(), max_depth, seed)
        .with_rollout_policy(Box::new(rollout));
    let result = mcts.search(initial_state.clone(), num_simulations);

    let mut samples: Vec<DistillSample> = Vec::new();
    let mut node_idx = result.root;
    let mut state = initial_state.clone();
    let mut depth = 0usize;
    loop {
        let children = result.tree[node_idx].children.clone();
        if children.is_empty() || depth >= max_depth {
            break;
        }
        let legal = domain.legal_actions(&state);
        let mut visit_counts = vec![0i64; legal.len()];
        let mut action_to_visits: HashMap<NtAction, i64> = HashMap::new();
        for &c in &children {
            let a = result.tree[c]
                .action_from_parent
                .clone()
                .expect("tree children always carry an action");
            action_to_visits.insert(a, result.tree[c].visit_count as i64);
        }
        for (i, a) in legal.iter().enumerate() {
            visit_counts[i] = action_to_visits.get(a).copied().unwrap_or(0);
        }
        if visit_counts.iter().sum::<i64>() > 0 {
            let teacher_dist =
                visit_counts_to_teacher_distribution(&visit_counts, temperature)?;
            samples.push(DistillSample {
                state: state.clone(),
                legal: legal.clone(),
                teacher_distribution: teacher_dist,
            });
        }

        // Python `max(node.children, key=visit_count)`: ties -> first.
        let mut best = children[0];
        for &c in &children[1..] {
            if result.tree[c].visit_count > result.tree[best].visit_count {
                best = c;
            }
        }
        if result.tree[best].visit_count == 0 {
            break;
        }
        state = result.tree[best].state.clone();
        node_idx = best;
        depth += 1;
        if domain.is_terminal(&state) {
            break;
        }
    }

    let passed =
        domain.is_terminal(&state) && domain.terminal_reward(&state) >= 0.999;
    Ok((samples, passed))
}

/// One distillation round. Python defaults: num_simulations=150, lr=0.3,
/// temperature=1.0, seed=0.
pub fn run_distill_round(
    policy: &SoftmaxPolicy,
    problems: &[Problem],
    depth_fn_factory: &dyn Fn(usize) -> Rc<dyn Fn(&NtState) -> usize>,
    num_simulations: usize,
    lr: f64,
    temperature: f64,
    seed: u64,
) -> Result<(SoftmaxPolicy, DistillRoundDiagnostics), String> {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut policy = policy.clone();
    let mut solved = 0usize;
    let mut total = 0usize;
    let mut n_samples_total = 0usize;
    let mut diag_loss_sum = 0.0f64;
    let mut diag_grad_sum = 0.0f64;
    let mut n_updates = 0usize;

    for (domain, initial_state, max_depth) in problems {
        let depth_fn = depth_fn_factory(*max_depth);
        // Python: rng=random.Random(rng.randint(0, 10**9)).
        let mut ep_rng = StdRng::seed_from_u64(rng.gen_range(0..=1_000_000_000u64));
        let (samples, passed) = extract_distill_samples(
            &policy,
            domain,
            initial_state,
            domain.target,
            *max_depth,
            depth_fn.clone(),
            &mut ep_rng,
            num_simulations,
            0.1,
            temperature,
        )?;
        solved += passed as usize;
        total += 1;
        n_samples_total += samples.len();
        if samples.is_empty() {
            continue;
        }
        // Update immediately on this problem's own samples with its own
        // apply_fn/target/max_depth before moving to the next problem —
        // same per-problem-group accumulation pattern as Phase 066's GRPO
        // round, since different problems can have different domains.
        let (new_policy, diag) = distill_weight_update(
            &policy,
            &samples,
            domain,
            domain.target,
            *max_depth,
            depth_fn.as_ref(),
            lr,
        )?;
        policy = new_policy;
        diag_loss_sum += diag.mean_loss_before_update;
        diag_grad_sum += diag.grad_norm;
        n_updates += 1;
    }

    let round_diag = DistillRoundDiagnostics {
        mean_loss_before_update: diag_loss_sum / (n_updates.max(1)) as f64,
        grad_norm: diag_grad_sum / (n_updates.max(1)) as f64,
        solve_rate: solved as f64 / (total.max(1)) as f64,
        n_samples: n_samples_total,
    };
    Ok((policy, round_diag))
}
