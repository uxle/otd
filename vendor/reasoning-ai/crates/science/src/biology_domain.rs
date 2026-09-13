//! Phase 104 — Biology: Mendelian genetics / Punnett squares (Promot:
//! "biology") (Rust port of python/science/biology_domain.py).
//!
//! Ground truth is exhaustive enumeration of a Punnett square (all gamete
//! combinations from two parent genotypes), not a shortcut formula --
//! enumeration-plus-counting over memorized results. Handles
//! single-gene (monohybrid) crosses; each parent's genotype is two
//! alleles, uppercase = dominant, lowercase = recessive, same letter for
//! both parents' gene (e.g. "Aa" x "Aa").

use std::collections::HashMap;

use reasoning_common::Rat;

pub(crate) fn _validate_genotype(genotype: &str) -> Result<(char, char), String> {
    let chars: Vec<char> = genotype.chars().collect();
    if chars.len() != 2 {
        return Err(format!(
            "genotype must be exactly 2 alleles, got '{}'",
            genotype
        ));
    }
    let (a, b) = (chars[0], chars[1]);
    if !a.to_lowercase().eq(b.to_lowercase()) {
        return Err(format!(
            "both alleles must be the same gene (e.g. 'Aa', not 'Ab'): '{}'",
            genotype
        ));
    }
    Ok((a, b))
}

/// Python's Punnett-square allele ordering: `sorted([a, b],
/// key=lambda c: (c.lower(), c.islower()))` -- uppercase sorts before
/// its lowercase twin ('Aa', never 'aA').
fn allele_sorted(a: char, b: char) -> String {
    let mut pair = [a, b];
    pair.sort_by(|x, y| {
        (x.to_lowercase().to_string(), x.is_lowercase())
            .cmp(&(y.to_lowercase().to_string(), y.is_lowercase()))
    });
    pair.iter().collect()
}

/// All 4 offspring genotype combinations (with repeats -- a real
/// Punnett square, not a deduplicated set), each allele-sorted so 'Aa'
/// and 'aA' count as the same genotype string.
pub fn punnett_square(genotype_a: &str, genotype_b: &str) -> Result<Vec<String>, String> {
    let alleles_a = _validate_genotype(genotype_a)?;
    let alleles_b = _validate_genotype(genotype_b)?;
    if !alleles_a
        .0
        .to_lowercase()
        .eq(alleles_b.0.to_lowercase())
    {
        return Err(format!(
            "genotypes must be for the same gene, got '{}' and '{}'",
            genotype_a, genotype_b
        ));
    }
    let mut offspring = Vec::new();
    // itertools.product over the two 2-allele gametes -> nested loops
    for a1 in [alleles_a.0, alleles_a.1] {
        for a2 in [alleles_b.0, alleles_b.1] {
            offspring.push(allele_sorted(a1, a2));
        }
    }
    Ok(offspring)
}

/// Exact fractions (not floats) for each distinct offspring genotype
/// -- counting from the full 4-cell Punnett square, matching how these
/// problems are actually taught and graded.
pub fn genotype_probabilities(
    genotype_a: &str,
    genotype_b: &str,
) -> Result<HashMap<String, Rat>, String> {
    let offspring = punnett_square(genotype_a, genotype_b)?;
    let mut counts: HashMap<String, i64> = HashMap::new();
    for g in &offspring {
        *counts.entry(g.clone()).or_insert(0) += 1;
    }
    let total = offspring.len() as i64;
    Ok(counts
        .into_iter()
        .map(|(g, c)| (g, Rat::new(c, total)))
        .collect())
}

/// Collapses genotype probabilities into phenotype probabilities:
/// any genotype containing the dominant (uppercase) allele shows the
/// dominant phenotype (for a simple complete-dominance gene); a
/// homozygous-recessive genotype shows the recessive phenotype.
/// (`dominant_wins` is accepted for signature parity with Python but is
/// unused there too.)
#[allow(unused_variables)]
pub fn phenotype_probability(
    genotype_a: &str,
    genotype_b: &str,
    dominant_wins: bool,
) -> Result<HashMap<String, Rat>, String> {
    let geno_probs = genotype_probabilities(genotype_a, genotype_b)?;
    let mut dominant_total = Rat::from_int(0);
    let mut recessive_total = Rat::from_int(0);
    for (g, p) in geno_probs {
        if g.chars().any(|c| c.is_uppercase()) {
            dominant_total = dominant_total + p;
        } else {
            recessive_total = recessive_total + p;
        }
    }
    let mut out = HashMap::new();
    out.insert("dominant".to_string(), dominant_total);
    out.insert("recessive".to_string(), recessive_total);
    Ok(out)
}
