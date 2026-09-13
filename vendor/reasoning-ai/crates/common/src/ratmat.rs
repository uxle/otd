//! Rational matrix operations (subset of sympy.Matrix used by this project):
//! construction, multiplication, determinant (cofactor for small n /
//! fraction-free Gaussian), `solve` (A*x = b), and `nullspace` (rational
//! RREF basis), matching sympy semantics for the small matrices involved.

use crate::rat::Rat;

/// A row-major rational matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct RatMatrix {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<Rat>,
}

impl RatMatrix {
    pub fn from_rows(rows: &[Vec<Rat>]) -> RatMatrix {
        let n = rows.len();
        let m = if n > 0 { rows[0].len() } else { 0 };
        assert!(rows.iter().all(|r| r.len() == m), "ragged matrix");
        let mut data = Vec::with_capacity(n * m);
        for r in rows {
            data.extend_from_slice(r);
        }
        RatMatrix {
            rows: n,
            cols: m,
            data,
        }
    }

    pub fn at(&self, i: usize, j: usize) -> Rat {
        self.data[i * self.cols + j]
    }

    pub fn set(&mut self, i: usize, j: usize, v: Rat) {
        self.data[i * self.cols + j] = v;
    }

    pub fn from_f64_rows(rows: &[Vec<f64>]) -> RatMatrix {
        let rat_rows: Vec<Vec<Rat>> = rows
            .iter()
            .map(|r| r.iter().map(|&v| Rat::from_f64(v)).collect())
            .collect();
        RatMatrix::from_rows(&rat_rows)
    }

    /// to list-of-rows of f64 (like sympy Matrix.tolist() then float()).
    pub fn to_f64_rows(&self) -> Vec<Vec<f64>> {
        (0..self.rows)
            .map(|i| (0..self.cols).map(|j| self.at(i, j).to_f64()).collect())
            .collect()
    }

    /// Matrix multiplication (sympy Matrix * Matrix).
    pub fn mat_mul(&self, other: &RatMatrix) -> RatMatrix {
        assert_eq!(self.cols, other.rows, "shape mismatch in mat_mul");
        let mut out = vec![Rat::from_int(0); self.rows * other.cols];
        for i in 0..self.rows {
            for k in 0..self.cols {
                let a = self.at(i, k);
                if a.is_zero() {
                    continue;
                }
                for j in 0..other.cols {
                    let idx = i * other.cols + j;
                    out[idx] = out[idx] + a * other.at(k, j);
                }
            }
        }
        RatMatrix {
            rows: self.rows,
            cols: other.cols,
            data: out,
        }
    }

    /// Determinant via fraction-free Gaussian elimination (exact).
    /// Returns 0 for non-square or singular.
    pub fn det(&self) -> Rat {
        assert_eq!(self.rows, self.cols, "det requires a square matrix");
        let n = self.rows;
        if n == 0 {
            return Rat::from_int(1);
        }
        if n == 1 {
            return self.at(0, 0);
        }
        if n == 2 {
            return self.at(0, 0) * self.at(1, 1) - self.at(0, 1) * self.at(1, 0);
        }
        // fraction-free (Bareiss) elimination
        let mut m = self.clone();
        let mut sign = 1i64;
        let mut prev_piv = Rat::from_int(1);
        for k in 0..(n - 1) {
            // pivot: find nonzero entry in column k at row >= k
            let mut piv = k;
            while piv < n && m.at(piv, k).is_zero() {
                piv += 1;
            }
            if piv == n {
                return Rat::from_int(0);
            }
            if piv != k {
                for j in 0..n {
                    let a = m.at(k, j);
                    let b = m.at(piv, j);
                    m.set(k, j, b);
                    m.set(piv, j, a);
                }
                sign = -sign;
            }
            let piv_val = m.at(k, k);
            for i in (k + 1)..n {
                for j in (k + 1)..n {
                    // m[i][j] = (m[i][j]*piv - m[i][k]*m[k][j]) / prev_piv
                    let t = m.at(i, j) * piv_val - m.at(i, k) * m.at(k, j);
                    let t = t / prev_piv; // exact: Bareiss guarantees divisibility
                    m.set(i, j, t);
                }
                m.set(i, k, Rat::from_int(0));
            }
            prev_piv = piv_val;
        }
        let d = m.at(n - 1, n - 1);
        if sign < 0 {
            -d
        } else {
            d
        }
    }

    /// Solve A*x = b for a square, nonsingular A (sympy Matrix.solve).
    /// Returns column vector of Rats.
    pub fn solve(&self, b: &RatMatrix) -> Result<RatMatrix, String> {
        if self.rows != self.cols {
            return Err("solve requires a square matrix".to_string());
        }
        let n = self.rows;
        assert_eq!(b.rows, n, "b must be a column vector of length n");
        assert_eq!(b.cols, 1);
        // augmented Gaussian elimination with exact rationals
        let mut a = self.clone();
        let mut rhs: Vec<Rat> = (0..n).map(|i| b.at(i, 0)).collect();
        for col in 0..n {
            let mut piv = col;
            while piv < n && a.at(piv, col).is_zero() {
                piv += 1;
            }
            if piv == n {
                return Err("singular matrix".to_string());
            }
            if piv != col {
                for j in 0..n {
                    let t = a.at(col, j);
                    a.set(col, j, a.at(piv, j));
                    a.set(piv, j, t);
                }
                rhs.swap(col, piv);
            }
            let p = a.at(col, col);
            for j in 0..n {
                let v = a.at(col, j) / p;
                a.set(col, j, v);
            }
            rhs[col] = rhs[col] / p;
            for i in 0..n {
                if i != col && !a.at(i, col).is_zero() {
                    let f = a.at(i, col);
                    for j in 0..n {
                        let v = a.at(i, j) - f * a.at(col, j);
                        a.set(i, j, v);
                    }
                    rhs[i] = rhs[i] - f * rhs[col];
                }
            }
        }
        Ok(RatMatrix {
            rows: n,
            cols: 1,
            data: rhs,
        })
    }

    /// Rational RREF (used by nullspace). Returns (rref matrix, pivot columns).
    fn rref(&self) -> (RatMatrix, Vec<usize>) {
        let mut m = self.clone();
        let n_rows = self.rows;
        let n_cols = self.cols;
        let mut pivots = Vec::new();
        let mut r = 0usize;
        for c in 0..n_cols {
            if r >= n_rows {
                break;
            }
            // find pivot
            let mut piv = None;
            for i in r..n_rows {
                if !m.at(i, c).is_zero() {
                    piv = Some(i);
                    break;
                }
            }
            let Some(p) = piv else { continue };
            if p != r {
                for j in 0..n_cols {
                    let t = m.at(r, j);
                    m.set(r, j, m.at(p, j));
                    m.set(p, j, t);
                }
            }
            let pv = m.at(r, c);
            for j in 0..n_cols {
                let v = m.at(r, j) / pv;
                m.set(r, j, v);
            }
            for i in 0..n_rows {
                if i != r && !m.at(i, c).is_zero() {
                    let f = m.at(i, c);
                    for j in 0..n_cols {
                        let v = m.at(i, j) - f * m.at(r, j);
                        m.set(i, j, v);
                    }
                }
            }
            pivots.push(c);
            r += 1;
        }
        (m, pivots)
    }

    /// Nullspace basis (sympy Matrix.nullspace()): returns basis vectors as
    /// column vectors. Empty when the nullspace is trivial.
    pub fn nullspace(&self) -> Vec<RatMatrix> {
        let (r, pivots) = self.rref();
        let n_cols = self.cols;
        let rank = pivots.len();
        let pivot_set: std::collections::HashSet<usize> = pivots.iter().copied().collect();
        let mut free_cols: Vec<usize> = (0..n_cols).filter(|c| !pivot_set.contains(c)).collect();
        // sympy orders basis vectors by free column order
        free_cols.sort();
        let mut basis = Vec::new();
        for &fc in &free_cols {
            let mut vec = vec![Rat::from_int(0); n_cols];
            vec[fc] = Rat::from_int(1);
            for (row, &pc) in pivots.iter().enumerate() {
                // x[pc] = -rref[row][fc]
                vec[pc] = -r.at(row, fc);
            }
            basis.push(RatMatrix {
                rows: n_cols,
                cols: 1,
                data: vec,
            });
        }
        let _ = rank;
        basis
    }
}

/// Convenience: build from i64 rows.
pub fn rat_matrix_from_rows(rows: &[Vec<i64>]) -> RatMatrix {
    let rat_rows: Vec<Vec<Rat>> = rows
        .iter()
        .map(|r| r.iter().map(|&v| Rat::from_int(v)).collect())
        .collect();
    RatMatrix::from_rows(&rat_rows)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_det() {
        let m = rat_matrix_from_rows(&[vec![1, 2], vec![3, 4]]);
        assert_eq!(m.det(), Rat::from_int(-2));
        let m3 = rat_matrix_from_rows(&[vec![2, 0, 1], vec![1, 3, 2], vec![0, 5, 4]]);
        // 2*(12-10) + 1*(5-0) = 9
        assert_eq!(m3.det(), Rat::from_int(9));
    }

    #[test]
    fn test_mul() {
        let a = rat_matrix_from_rows(&[vec![1, 2], vec![3, 4]]);
        let b = rat_matrix_from_rows(&[vec![5, 6], vec![7, 8]]);
        let c = a.mat_mul(&b);
        assert_eq!(c.at(0, 0), Rat::from_int(19));
        assert_eq!(c.at(1, 1), Rat::from_int(50));
    }

    #[test]
    fn test_solve() {
        // x + y = 5; x - y = 1  -> x=3, y=2
        let a = rat_matrix_from_rows(&[vec![1, 1], vec![1, -1]]);
        let b = rat_matrix_from_rows(&[vec![5], vec![1]]);
        let x = a.solve(&b).unwrap();
        assert_eq!(x.at(0, 0), Rat::from_int(3));
        assert_eq!(x.at(1, 0), Rat::from_int(2));
    }

    #[test]
    fn test_nullspace() {
        // H2 + O2 -> H2O element matrix (H row, O row): [2,0,-2],[0,2,-1]
        let m = rat_matrix_from_rows(&[vec![2, 0, -2], vec![0, 2, -1]]);
        let ns = m.nullspace();
        assert_eq!(ns.len(), 1);
        let v = &ns[0];
        // free col is H2O (col 2): x3 = 1 -> x = (1, 1/2, 1)
        assert_eq!(v.at(0, 0), Rat::from_int(1));
        assert_eq!(v.at(1, 0), Rat::new(1, 2));
        assert_eq!(v.at(2, 0), Rat::from_int(1));
    }
}
