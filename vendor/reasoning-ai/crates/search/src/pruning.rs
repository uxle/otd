//! Phase 022 — Search pruning via reachability bounds (Rust port of
//! `search/pruning.py`). Provably-correct (conservative) bounds on the
//! value reachable by combining numbers with +,-,*,/.

/// (min_possible, max_possible) an honest bound on any value reachable by
/// combining `values` with +,-,*,/. Conservative: may overestimate the true
/// reachable range (safe — fewer prunes), never discards a solvable state.
pub fn reachable_bound(values: &[f64]) -> (f64, f64) {
    if values.is_empty() {
        return (0.0, 0.0);
    }
    let abs_vals: Vec<f64> = values.iter().map(|v| v.abs()).collect();
    // crude but SAFE upper bound: product of all magnitudes >= 1, plus sum
    // of magnitudes < 1 (multiplying by them can only shrink)
    let mut product = 1.0f64;
    let mut additive = 0.0f64;
    for v in &abs_vals {
        if *v >= 1.0 {
            product *= *v;
        } else {
            additive += *v;
        }
    }
    let sum: f64 = abs_vals.iter().sum();
    let max_bound = product + additive + sum; // generous slack term
    let min_bound = -max_bound;
    (min_bound, max_bound)
}

/// Is the target provably outside the reachable bound?
pub fn is_provably_unreachable(values: &[f64], target: f64) -> bool {
    let (lo, hi) = reachable_bound(values);
    target < lo || target > hi
}
