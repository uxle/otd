//! Phase 027 — Combinatorics domain (Rust port of
//! `search/combinatorics_domain.py`, design doc 3.5 level 6: "verify by
//! brute-force enumeration at small N").
//!
//! Task: "how many ways can you choose k items from n, arrange r items
//! from n, etc." Search proposes a candidate answer, and the verifier is
//! a brute-force enumerator — for small combinatorics problems the actual
//! ground truth is just counting, so exhaustive enumeration IS the proof.

use crate::domain::Domain;

/// Python `brute_force_count` — exact count of the equivalent enumeration.
/// Python enumerates `itertools` combinations/permutations (identical to
/// the factorial formulas for every n/r in range); Python ints are
/// unbounded, so we use u128 internally and return i64 (saturating at
/// i64::MAX for absurd inputs — far beyond any search_radius this domain
/// uses).
pub fn brute_force_count(kind: &str, n: i64, r: i64) -> Result<i64, String> {
    match kind {
        "combinations" => {
            // itertools.combinations raises ValueError for negative r
            if r < 0 {
                return Err("r must be non-negative".to_string());
            }
            Ok(binom(n, r))
        }
        "permutations" => {
            if r < 0 {
                return Err("r must be non-negative".to_string());
            }
            Ok(perm(n, r))
        }
        "subsets" => {
            // sum(C(n,k) for k in range(n+1)); empty range for n < 0 -> 0
            if n < 0 {
                return Ok(0);
            }
            // 2^n (saturating far beyond any usable search_radius)
            if n > 62 {
                return Ok(i64::MAX);
            }
            Ok(1i64 << n)
        }
        _ => Err(format!("unknown kind {}", kind)),
    }
}

/// C(n, r) for r >= 0 (1 for r == 0, 0 when r > n or n < 0).
fn binom(n: i64, r: i64) -> i64 {
    if r == 0 {
        return 1;
    }
    if n < 0 || r > n {
        return 0;
    }
    let mut num: u128 = 1;
    let mut den: u128 = 1;
    for i in 1..=r as u128 {
        num *= (n as u128) - (r as u128) + i;
        den *= i;
    }
    (num / den).min(i64::MAX as u128) as i64
}

/// P(n, r) = n!/(n-r)! for r >= 0.
fn perm(n: i64, r: i64) -> i64 {
    if r == 0 {
        return 1;
    }
    if n < 0 || r > n {
        return 0;
    }
    let mut v: u128 = 1;
    for i in 0..r as u128 {
        v *= (n as u128) - i;
    }
    v.min(i64::MAX as u128) as i64
}

#[derive(Debug, Clone, PartialEq)]
pub struct CombinatoricsState {
    pub guess: Option<i64>,
}

pub struct CombinatoricsDomain {
    pub kind: String,
    pub n: i64,
    pub r: i64,
    pub ground_truth: i64,
    pub search_radius: i64,
}

impl CombinatoricsDomain {
    /// Python `__init__` computes the brute-force ground truth up front and
    /// can raise (unknown kind / negative r) — `try_new` propagates that.
    pub fn try_new(kind: &str, n: i64, r: i64, search_radius: i64) -> Result<Self, String> {
        let ground_truth = brute_force_count(kind, n, r)?;
        Ok(CombinatoricsDomain {
            kind: kind.to_string(),
            n,
            r,
            ground_truth,
            search_radius,
        })
    }

    /// Python defaults: `search_radius=300`.
    pub fn new(kind: &str, n: i64, r: i64) -> Self {
        Self::try_new(kind, n, r, 300).expect("valid combinatorics problem")
    }
}

impl Domain for CombinatoricsDomain {
    type State = CombinatoricsState;
    type Action = i64;

    fn initial_state(&self) -> CombinatoricsState {
        CombinatoricsState { guess: None }
    }

    fn legal_actions(&self, state: &CombinatoricsState) -> Vec<i64> {
        if state.guess.is_some() {
            return Vec::new();
        }
        // bound is a GENERIC upper limit (n^r), independent of the actual
        // answer — centering candidates on ground_truth would leak the
        // answer's location into the search space, making the search
        // trivial rather than a real test of finding it
        let exp = self.r.max(1) as u32;
        let pow = (self.n.unsigned_abs() as u128)
            .checked_pow(exp)
            .unwrap_or(u128::MAX);
        let upper_bound = pow.min(self.search_radius.max(0) as u128) as i64;
        (0..=upper_bound).collect()
    }

    fn apply(&self, _state: &CombinatoricsState, action: &i64) -> CombinatoricsState {
        CombinatoricsState {
            guess: Some(*action),
        }
    }

    fn is_terminal(&self, state: &CombinatoricsState) -> bool {
        state.guess.is_some()
    }

    fn terminal_reward(&self, state: &CombinatoricsState) -> f64 {
        assert!(self.is_terminal(state));
        if state.guess == Some(self.ground_truth) {
            1.0
        } else {
            0.0
        }
    }
}
