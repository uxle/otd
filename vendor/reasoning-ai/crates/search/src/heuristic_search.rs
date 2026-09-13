//! Phase 021 — Beam search & best-first search (Rust port of
//! `search/heuristic_search.py`). Both use the same `Domain` contract as
//! MCTS; only a domain's own terminal_reward decides success.

use crate::domain::Domain;

/// Best-first search: always expand the single highest-scoring frontier
/// node. Returns (found_solution, terminal_state, nodes_expanded).
pub fn best_first_search<D: Domain>(
    domain: &D,
    root_state: D::State,
    heuristic_fn: impl Fn(&D::State) -> f64,
    max_expansions: usize,
    max_depth: usize,
) -> (bool, Option<D::State>, usize) {
    // (negated score, counter tie-breaker, state, depth)
    let mut frontier: Vec<(f64, usize, D::State, usize)> = Vec::new();
    let mut counter = 0usize;
    let mut expansions = 0usize;
    frontier.push((-heuristic_fn(&root_state), counter, root_state, 0));

    while !frontier.is_empty() && expansions < max_expansions {
        // pop the max-score (min of negated score), stable by insertion order
        let mut best_i = 0usize;
        for (i, item) in frontier.iter().enumerate() {
            if (item.0, item.1) < (frontier[best_i].0, frontier[best_i].1) {
                best_i = i;
            }
        }
        let (_, _, state, depth) = frontier.swap_remove(best_i);
        expansions += 1;

        if domain.is_terminal(&state) {
            if domain.terminal_reward(&state) >= 0.999 {
                return (true, Some(state), expansions);
            }
            continue;
        }

        if depth >= max_depth {
            continue;
        }

        for action in domain.legal_actions(&state) {
            let next_state = domain.apply(&state, &action);
            counter += 1;
            frontier.push((-heuristic_fn(&next_state), counter, next_state, depth + 1));
        }
    }

    (false, None, expansions)
}

/// Beam search: keep the top-K states at each depth, discard the rest.
/// Returns (found_solution, terminal_state, nodes_expanded).
pub fn beam_search<D: Domain>(
    domain: &D,
    root_state: D::State,
    heuristic_fn: impl Fn(&D::State) -> f64,
    beam_width: usize,
    max_depth: usize,
) -> (bool, Option<D::State>, usize) {
    let mut beam: Vec<D::State> = vec![root_state];
    let mut nodes_expanded = 0usize;

    for _depth in 0..max_depth {
        let mut candidates: Vec<D::State> = Vec::new();
        for state in &beam {
            if domain.is_terminal(state) {
                if domain.terminal_reward(state) >= 0.999 {
                    return (true, Some(state.clone()), nodes_expanded);
                }
                continue;
            }
            for action in domain.legal_actions(state) {
                let next_state = domain.apply(state, &action);
                nodes_expanded += 1;
                candidates.push(next_state);
            }
        }

        if candidates.is_empty() {
            break;
        }

        // sort by heuristic descending, keep top beam_width
        candidates.sort_by(|a, b| {
            heuristic_fn(b)
                .partial_cmp(&heuristic_fn(a))
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        beam = candidates.into_iter().take(beam_width).collect();
    }

    for state in &beam {
        if domain.is_terminal(state) && domain.terminal_reward(state) >= 0.999 {
            return (true, Some(state.clone()), nodes_expanded);
        }
    }

    (false, None, nodes_expanded)
}
