//! Hungarian (Jonker-Volgenant style) assignment, O(n^3).
//! Rows <= cols required (callers pad). Returns row -> column assignment.

/// Solve min-cost assignment. `cost[i][j]`, n rows, m cols, n <= m.
/// Returns Vec of length n with the assigned column per row.
pub fn hungarian(cost: &[Vec<f64>], n: usize, m: usize) -> Vec<usize> {
    assert!(n <= m, "hungarian requires n <= m (pad the cost matrix)");
    let inf = f64::INFINITY;
    // 1-indexed potentials / matching (e-maxx formulation)
    let mut u = vec![0.0f64; n + 1];
    let mut v = vec![0.0f64; m + 1];
    let mut p = vec![0usize; m + 1]; // p[j] = row matched to col j
    let mut way = vec![0usize; m + 1];

    for i in 1..=n {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; m + 1];
        let mut used = vec![false; m + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=m {
                if !used[j] {
                    let cur = cost[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=m {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        // augmenting path back
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }

    let mut ans = vec![usize::MAX; n];
    for j in 1..=m {
        if p[j] != 0 {
            ans[p[j] - 1] = j - 1;
        }
    }
    ans
}

#[cfg(test)]
mod tests {
    use super::*;

    fn brute_force(cost: &[Vec<f64>], n: usize, m: usize) -> f64 {
        // min over all injective row->col mappings (n <= m, small)
        let mut best = f64::INFINITY;
        let mut used = vec![false; m];
        fn rec(cost: &[Vec<f64>], used: &mut Vec<bool>, i: usize, acc: f64, best: &mut f64, m: usize) {
            if i == cost.len() {
                *best = best.min(acc);
                return;
            }
            for j in 0..m {
                if !used[j] {
                    used[j] = true;
                    rec(cost, used, i + 1, acc + cost[i][j], best, m);
                    used[j] = false;
                }
            }
        }
        rec(cost, &mut used, 0, 0.0, &mut best, m);
        best
    }

    fn hungarian_cost(cost: &[Vec<f64>], ans: &[usize]) -> f64 {
        ans.iter().enumerate().map(|(i, &j)| cost[i][j]).sum()
    }

    #[test]
    fn matches_brute_force_random() {
        let mut rng = crate::core::rng::GaussRng::new(9);
        for trial in 0..30 {
            let n = 1 + (trial % 4);
            let m = n + (trial % 3);
            let cost: Vec<Vec<f64>> = (0..n).map(|_| (0..m).map(|_| rng.uniform() * 10.0).collect()).collect();
            let ans = hungarian(&cost, n, m);
            // every row assigned, columns distinct
            let mut seen = std::collections::HashSet::new();
            for &j in &ans {
                assert!(j < m);
                assert!(seen.insert(j), "duplicate column");
            }
            let h = hungarian_cost(&cost, &ans);
            let b = brute_force(&cost, n, m);
            assert!((h - b).abs() < 1e-9, "trial {trial}: hungarian {h} vs brute {b}");
        }
    }

    #[test]
    fn obvious_assignment() {
        let cost = vec![vec![0.0, 100.0], vec![100.0, 0.0]];
        let ans = hungarian(&cost, 2, 2);
        assert_eq!(ans, vec![0, 1]);
        let cost2 = vec![vec![100.0, 0.0], vec![0.0, 100.0]];
        let ans2 = hungarian(&cost2, 2, 2);
        assert_eq!(ans2, vec![1, 0]);
    }

    #[test]
    fn rectangular_padded() {
        // 1 row, 3 cols
        let cost = vec![vec![5.0, 1.0, 9.0]];
        let ans = hungarian(&cost, 1, 3);
        assert_eq!(ans[0], 1);
    }
}
