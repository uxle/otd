//! Phase 115 — Chemistry: balancing chemical equations and limiting
//! reagents (Rust port of python/science/chem_balance_domain.py)
//!
//! Balancing ground truth: a balanced equation is an integer vector of
//! coefficients in the nullspace of the element-conservation matrix (each
//! row = one element, each column = one species, products negated). The
//! minimal positive integer nullspace vector IS the balanced equation.
//! Verified explicitly: element conservation element-by-element, gcd == 1,
//! and all coefficients strictly positive.
//!
//! Limiting reagent: extent of reaction allowed by each reactant
//! (extent_i = moles_i / coeff_i); the minimum extent is limiting.
//! Verified: the limiting reactant's leftover is exactly 0 while every
//! other reactant's leftover is >= 0, and product moles recompute.

use std::collections::BTreeSet;

use reasoning_common::{py_round, rat_matrix_from_rows, Rat, RatMatrix};

use crate::chemistry_domain::{count_of, molar_mass, parse_formula};

// math.gcd equivalent (Python's result is always non-negative).
fn gcd(a: i64, b: i64) -> i64 {
    let (mut a, mut b) = (a.abs(), b.abs());
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

/// A balanced equation: species in input order with their integer
/// coefficients, plus the element list that was conserved.
#[derive(Debug, Clone, PartialEq)]
pub struct BalancedEquation {
    pub reactants: Vec<String>,
    pub products: Vec<String>,
    pub reactant_coeffs: Vec<i64>,
    pub product_coeffs: Vec<i64>,
    pub elements: Vec<String>,
}

impl BalancedEquation {
    pub fn formatted(&self) -> String {
        let left = self
            .reactant_coeffs
            .iter()
            .zip(self.reactants.iter())
            .map(|(c, f)| if *c != 1 { format!("{} {}", c, f) } else { f.clone() })
            .collect::<Vec<_>>()
            .join(" + ");
        let right = self
            .product_coeffs
            .iter()
            .zip(self.products.iter())
            .map(|(c, f)| if *c != 1 { format!("{} {}", c, f) } else { f.clone() })
            .collect::<Vec<_>>()
            .join(" + ");
        format!("{} -> {}", left, right)
    }
}

fn strip_coeff(s: &str) -> String {
    // strip any user-supplied leading coefficients like "2H2" -> "H2"
    let chars: Vec<char> = s.chars().collect();
    let mut i = 0usize;
    while i < chars.len() && chars[i].is_ascii_digit() {
        i += 1;
    }
    if i < chars.len() {
        chars[i..].iter().collect()
    } else {
        s.to_string()
    }
}

fn split_arrow(equation_str: &str) -> Result<(Vec<String>, Vec<String>), String> {
    let (left, right) = {
        let mut found = None;
        for arrow in ["->", "=>", "=", "→"] {
            if let Some(idx) = equation_str.find(arrow) {
                let (l, r) = equation_str.split_at(idx);
                found = Some((l, &r[arrow.len()..]));
                break;
            }
        }
        match found {
            Some(lr) => lr,
            None => {
                return Err("equation must contain an arrow: '->', '=>', '=', or '→'".to_string())
            }
        }
    };
    let reactants: Vec<String> = left
        .split('+')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().replace(' ', ""))
        .collect();
    let products: Vec<String> = right
        .split('+')
        .filter(|s| !s.trim().is_empty())
        .map(|s| s.trim().replace(' ', ""))
        .collect();
    if reactants.is_empty() || products.is_empty() {
        return Err("equation needs species on both sides of the arrow".to_string());
    }
    Ok((
        reactants.iter().map(|s| strip_coeff(s)).collect(),
        products.iter().map(|s| strip_coeff(s)).collect(),
    ))
}

fn element_matrix(
    reactants: &[String],
    products: &[String],
) -> Result<(RatMatrix, Vec<String>), String> {
    let mut species: Vec<String> = reactants.to_vec();
    species.extend_from_slice(products);
    let parsed: Vec<Vec<(String, i64)>> = species
        .iter()
        .map(|f| parse_formula(f))
        .collect::<Result<_, _>>()?;
    let mut element_set: BTreeSet<String> = BTreeSet::new(); // sorted(...) below
    for counts in &parsed {
        for (el, _) in counts {
            element_set.insert(el.clone());
        }
    }
    let elements: Vec<String> = element_set.into_iter().collect();
    let rows: Vec<Vec<i64>> = elements
        .iter()
        .map(|el| {
            parsed
                .iter()
                .enumerate()
                .map(|(i, counts)| {
                    let sign: i64 = if i < reactants.len() { 1 } else { -1 };
                    sign * count_of(counts, el)
                })
                .collect()
        })
        .collect();
    Ok((rat_matrix_from_rows(&rows), elements))
}

/// Balance 'A + B -> C' via the nullspace of the element matrix, with full
/// explicit verification of the result (see module docstring).
pub fn balance_equation(equation_str: &str) -> Result<BalancedEquation, String> {
    let (reactants, products) = split_arrow(equation_str)?;
    let (matrix, elements) = element_matrix(&reactants, &products)?;
    let nullspace = matrix.nullspace();
    if nullspace.is_empty() {
        return Err("no solution: this equation cannot be balanced (element matrix has trivial nullspace)".to_string());
    }
    let vec = &nullspace[0];
    // scale to integers: multiply by lcm of denominators, then reduce by gcd
    let mut lcm: i64 = 1;
    for j in 0..vec.rows {
        let d = vec.at(j, 0).den;
        lcm = lcm * d / gcd(lcm, d);
    }
    let mut ints: Vec<i64> = (0..vec.rows)
        .map(|j| (vec.at(j, 0) * Rat::from_int(lcm)).num)
        .collect();
    if ints.iter().any(|&v| v < 0) {
        for v in ints.iter_mut() {
            *v = -*v;
        }
    }
    if ints.iter().any(|&v| v == 0) {
        return Err("balanced solution contains a zero coefficient -- a species on one side is unused, rejecting".to_string());
    }
    let mut g: i64 = 0;
    for &v in &ints {
        g = gcd(g, v.abs());
    }
    if g == 0 {
        return Err("degenerate nullspace vector".to_string());
    }
    for v in ints.iter_mut() {
        *v /= g;
    }

    let n_r = reactants.len();
    let r_coeffs: Vec<i64> = ints[..n_r].to_vec();
    let p_coeffs: Vec<i64> = ints[n_r..].to_vec();

    // ---- explicit independent verification (element-by-element) ----
    let mut all_species: Vec<String> = reactants.clone();
    all_species.extend(products.iter().cloned());
    let parsed: Vec<Vec<(String, i64)>> = all_species
        .iter()
        .map(|f| parse_formula(f))
        .collect::<Result<_, _>>()?;
    for el in &elements {
        let left: i64 = r_coeffs
            .iter()
            .zip(parsed[..n_r].iter())
            .map(|(c, counts)| c * count_of(counts, el))
            .sum();
        let right: i64 = p_coeffs
            .iter()
            .zip(parsed[n_r..].iter())
            .map(|(c, counts)| c * count_of(counts, el))
            .sum();
        if left != right {
            return Err(format!(
                "verification failed: element {} unbalanced ({} left vs {} right)",
                el, left, right
            ));
        }
    }
    let mut g2: i64 = 0;
    for &v in &ints {
        g2 = gcd(g2, v);
    }
    if g2 != 1 {
        return Err("verification failed: coefficients share a common factor".to_string());
    }
    Ok(BalancedEquation {
        reactants,
        products,
        reactant_coeffs: r_coeffs,
        product_coeffs: p_coeffs,
        elements,
    })
}

/// Result of a limiting-reagent computation.
#[derive(Debug, Clone, PartialEq)]
pub struct LimitingResult {
    pub limiting_reactant: String,
    pub product_formula: String,
    pub moles_product: f64,
    pub mass_product: f64,
    /// reactant formula -> moles left over (insertion order preserved).
    pub leftovers: Vec<(String, f64)>,
}

impl LimitingResult {
    /// Moles left over for a reactant formula (Python: `leftovers[formula]`).
    pub fn leftover(&self, formula: &str) -> Option<f64> {
        self.leftovers
            .iter()
            .find(|(k, _)| k == formula)
            .map(|(_, v)| *v)
    }
}

/// Balance the equation, then find which reactant runs out first and how
/// much product that allows. `moles_by_formula` maps reactant formula to
/// available moles (any formula not mentioned is treated as absent, which
/// errors -- silently assuming zero would be a guess).
/// Verified: limiting reactant leftover == 0, all others >= 0, product
/// recomputed from the limiting extent agrees.
pub fn limiting_reagent(
    equation_str: &str,
    moles_by_formula: &[(&str, f64)],
    product_formula: &str,
) -> Result<LimitingResult, String> {
    let eq = balance_equation(equation_str)?;
    if !eq.products.iter().any(|p| p == product_formula) {
        return Err(format!(
            "'{}' is not a product of {}",
            product_formula,
            eq.formatted()
        ));
    }
    for r in &eq.reactants {
        if !moles_by_formula.iter().any(|(f, _)| f == r) {
            return Err(format!("no amount given for reactant '{}'", r));
        }
    }
    let moles_of = |f: &str| {
        moles_by_formula
            .iter()
            .find(|(k, _)| *k == f)
            .map(|(_, v)| *v)
            .unwrap()
    };
    let mut extents: Vec<f64> = Vec::with_capacity(eq.reactants.len());
    for (coeff, r) in eq.reactant_coeffs.iter().zip(eq.reactants.iter()) {
        let moles = moles_of(r);
        if moles < 0.0 {
            return Err("moles cannot be negative".to_string());
        }
        extents.push(moles / *coeff as f64);
    }
    // min(...) over the dict keys: first minimum in insertion order
    let mut limiting_idx = 0usize;
    for i in 1..extents.len() {
        if extents[i] < extents[limiting_idx] {
            limiting_idx = i;
        }
    }
    let limiting_reactant = eq.reactants[limiting_idx].clone();
    let extent = extents[limiting_idx];
    let p_index = eq.products.iter().position(|p| p == product_formula).unwrap();
    let p_coeff = eq.product_coeffs[p_index];
    let moles_product = extent * p_coeff as f64;
    let mass_product = moles_product * molar_mass(product_formula)?;

    let mut leftovers: Vec<(String, f64)> = Vec::with_capacity(eq.reactants.len());
    for (coeff, r) in eq.reactant_coeffs.iter().zip(eq.reactants.iter()) {
        let used = extent * *coeff as f64;
        leftovers.push((r.clone(), py_round(moles_of(r) - used, 10)));
    }
    let leftover_of = |f: &str| {
        leftovers
            .iter()
            .find(|(k, _)| k == f)
            .map(|(_, v)| *v)
            .unwrap()
    };
    if leftover_of(&limiting_reactant).abs() > 1e-9 {
        return Err("verification failed: limiting reactant leftover is not zero".to_string());
    }
    if leftovers.iter().any(|(_, v)| *v < -1e-9) {
        return Err(
            "verification failed: a reactant went negative (extent exceeds availability)"
                .to_string(),
        );
    }
    let back = extent * p_coeff as f64;
    if (back - moles_product).abs() > 1e-12 {
        return Err("verification failed: product moles do not recompute".to_string());
    }
    Ok(LimitingResult {
        limiting_reactant,
        product_formula: product_formula.to_string(),
        moles_product: py_round(moles_product, 6),
        mass_product: py_round(mass_product, 4),
        leftovers: leftovers
            .into_iter()
            .map(|(k, v)| (k, py_round(v, 6)))
            .collect(),
    })
}
