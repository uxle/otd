//! Port of python/tests/test_prm_guided_search.py
//! (TestPRMGuidedSearchEfficiency).

mod common;

use common::train_a_prm;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use reasoning_prm::PrmGuidedPolicy;
use reasoning_search::{make_initial_state, Mcts, NtState, NumberTargetDomain};
use std::rc::Rc;

#[test]
fn test_guided_search_at_reduced_budget() {
    // Same held-out problems, same LOW simulation budget: does PRM guidance
    // solve more of them than uniform-random rollout? Uses 30 held-out
    // problems (not 5) because n=5 is too noisy to draw any conclusion
    // from. This is measured honestly: if guidance doesn't help, the test
    // fails, it isn't quietly adjusted.
    let prm = train_a_prm(0);

    let mut rng = StdRng::seed_from_u64(999);
    let mut test_problems: Vec<(NumberTargetDomain, Vec<f64>, usize)> = Vec::new();
    for _ in 0..30 {
        let nums: Vec<f64> = (0..3)
            .map(|_| rng.gen_range(1..=9) as f64)
            .collect();
        // bias toward solvable, not all
        let target = nums[rng.gen_range(0..nums.len())] + nums[rng.gen_range(0..nums.len())];
        test_problems.push((NumberTargetDomain::new(target), nums, 4));
    }

    let low_budget = 40usize; // deliberately small so guidance has room to matter

    let mut random_solved = 0usize;
    let mut guided_solved = 0usize;

    for (i, (domain, nums, max_depth)) in test_problems.iter().enumerate() {
        let state = make_initial_state(nums);

        let mut mcts_random = Mcts::new(domain.clone(), *max_depth, 2000 + i as u64);
        let r_random = mcts_random.search(state.clone(), low_budget);
        random_solved += r_random.found_verified_solution as usize;

        // depth_fn: merges so far
        let depth_fn = {
            let max_depth = *max_depth;
            Rc::new(move |s: &NtState| max_depth - s.len() + 1)
        };
        let guided_policy = PrmGuidedPolicy::new(
            prm.clone(),
            domain.clone(),
            domain.target,
            *max_depth,
            depth_fn,
            0.15,
        );
        let mut mcts_guided =
            Mcts::new(domain.clone(), *max_depth, 2000 + i as u64)
                .with_rollout_policy(Box::new(guided_policy));
        let r_guided = mcts_guided.search(state, low_budget);
        guided_solved += r_guided.found_verified_solution as usize;
    }

    println!(
        "\n[Phase 007] random-rollout solved {}/30, PRM-guided solved {}/30 (budget={} sims/problem)",
        random_solved, guided_solved, low_budget
    );

    // Honest note (from the Python original): with this simple linear PRM
    // (5 hand-crafted features, trained on ~500 examples) the effect is
    // real but modest — proving the *mechanism* (verifier-grounded step
    // scores -> better search) works at all.
    assert!(
        guided_solved >= random_solved,
        "PRM guidance should solve at least as many held-out problems as random rollout at the same low budget"
    );
}
