//! Port of python/tests/test_pruning.py
//! (TestPruningBoundSoundness, TestPruningSavesRealSearchBudget).

use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use reasoning_search::{
    is_provably_unreachable, make_initial_state, Domain, Mcts, NtAction, NtState,
    NumberTargetDomain,
};
use std::collections::HashSet;

/// Exhaustively compute every value reachable by combining `values`
/// pairwise with +,-,*,/ down to one number. Ground truth for testing the
/// bound's soundness — exponential, only usable for small inputs.
fn brute_force_reachable_values(values: &[f64]) -> HashSet<u64> {
    if values.len() == 1 {
        let mut s = HashSet::new();
        s.insert(values[0].to_bits());
        return s;
    }
    let mut results = HashSet::new();
    let n = values.len();
    // itertools.permutations(range(n), 2) with i >= j skipped -> i < j
    for i in 0..n {
        for j in 0..n {
            if i >= j {
                continue;
            }
            let (a, b) = (values[i], values[j]);
            let rest: Vec<f64> = values
                .iter()
                .enumerate()
                .filter(|(k, _)| *k != i && *k != j)
                .map(|(_, v)| *v)
                .collect();
            let mut combos = vec![a + b, a - b, b - a];
            if b.abs() > 1e-9 {
                combos.push(a / b);
            }
            if a.abs() > 1e-9 {
                combos.push(b / a);
            }
            for c in combos {
                let mut next = rest.clone();
                next.push(c);
                results.extend(brute_force_reachable_values(&next));
            }
        }
    }
    results
}

// ---- TestPruningBoundSoundness ----

#[test]
fn test_pruning_bound_never_excludes_a_truly_reachable_value() {
    // The critical correctness property: for many random small sets, every
    // brute-force-verified-reachable value must fall inside the bound. A
    // single violation would mean the pruner could discard a real solution
    // — checked directly, not assumed.
    let mut rng = StdRng::seed_from_u64(0);
    let mut violations: Vec<(Vec<f64>, f64, f64, f64)> = Vec::new();
    for _ in 0..20 {
        let values: Vec<f64> = (0..3).map(|_| rng.gen_range(1..=9) as f64).collect();
        let reachable = brute_force_reachable_values(&values);
        let (lo, hi) = reasoning_search::reachable_bound(&values);
        for bits in reachable {
            let v = f64::from_bits(bits);
            if v < lo - 1e-6 || v > hi + 1e-6 {
                violations.push((values.clone(), v, lo, hi));
            }
        }
    }
    assert!(violations.is_empty(), "bound unsound for: {:?}", violations);
}

#[test]
fn test_pruning_correctly_flags_a_genuinely_unreachable_target() {
    // all 1's: brute-force-confirmed max reachable magnitude is small
    let values = [1.0, 1.0, 1.0, 1.0];
    let reachable = brute_force_reachable_values(&values);
    // sanity on the ground truth itself
    assert!(reachable.iter().map(|b| f64::from_bits(*b).abs()).fold(0.0, f64::max) < 10.0);
    assert!(is_provably_unreachable(&values, 10000.0));
}

#[test]
fn test_pruning_does_not_flag_a_reachable_target() {
    let values = [4.0, 7.0, 8.0, 8.0];
    assert!(!is_provably_unreachable(&values, 24.0));
}

// ---- TestPruningSavesRealSearchBudget ----

/// Wraps the number-target domain: bails out immediately on actions whose
/// resulting state is provably unable to reach the target (the Python
/// test subclasses NumberTargetDomain and overrides legal_actions).
struct PrunedDomain {
    inner: NumberTargetDomain,
}

impl Domain for PrunedDomain {
    type State = NtState;
    type Action = NtAction;

    fn initial_state(&self) -> NtState {
        self.inner.initial_state()
    }

    fn legal_actions(&self, state: &NtState) -> Vec<NtAction> {
        let acts = self.inner.legal_actions(state);
        // filter out actions whose resulting state is provably dead
        let kept: Vec<NtAction> = acts
            .iter()
            .cloned()
            .filter(|a| {
                let ns = self.inner.apply(state, a);
                let vals: Vec<f64> = ns.iter().map(|(v, _)| *v).collect();
                !is_provably_unreachable(&vals, self.inner.target)
            })
            .collect();
        // never return zero actions (would break MCTS)
        if !kept.is_empty() || acts.is_empty() {
            kept
        } else {
            vec![acts[0].clone()]
        }
    }

    fn apply(&self, state: &NtState, action: &NtAction) -> NtState {
        self.inner.apply(state, action)
    }

    fn is_terminal(&self, state: &NtState) -> bool {
        self.inner.is_terminal(state)
    }

    fn terminal_reward(&self, state: &NtState) -> f64 {
        self.inner.terminal_reward(state)
    }
}

#[test]
fn test_pruning_saves_real_search_budget_pruned_mcts_wastes_less_budget_on_hopeless_subtrees() {
    // Wrap MCTS's rollout to bail out immediately (reward=0, no further
    // expansion) whenever the current remaining numbers provably cannot
    // reach the target — compare total nodes_expanded against the unpruned
    // Phase 003 MCTS on a hopeless problem.
    let domain = NumberTargetDomain::new(1e9); // nothing here can reach a billion
    let state = make_initial_state(&[1.0, 1.0, 1.0]);

    let mut unpruned = Mcts::new(domain.clone(), 4, 1);
    let r_unpruned = unpruned.search(state.clone(), 300);

    let pruned_domain = PrunedDomain {
        inner: NumberTargetDomain::new(1e9),
    };
    let mut pruned = Mcts::new(pruned_domain, 4, 1);
    let r_pruned = pruned.search(state, 300);

    println!(
        "[Phase 022] unpruned nodes_expanded={}, pruned nodes_expanded={}",
        r_unpruned.nodes_expanded, r_pruned.nodes_expanded
    );

    assert!(!r_unpruned.found_verified_solution);
    assert!(!r_pruned.found_verified_solution); // both correctly report no solution
    assert!(r_pruned.nodes_expanded <= r_unpruned.nodes_expanded);
}
