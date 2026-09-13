//! Phase 119 — Biology: dihybrid crosses (two unlinked genes)
//! (Rust port of python/science/bio_dihybrid_domain.py).
//!
//! Ground truth: exhaustive enumeration of the 16-cell dihybrid Punnett
//! square (each parent contributes one of 4 gamete combinations), NOT
//! the memorized 9:3:3:1 ratio. Returns exact Fractions (Rat).
//!
//! INDEPENDENT verification (a genuinely different computation): for
//! unlinked genes the dihybrid distribution must equal the product of
//! the two single-gene monohybrid distributions (product rule of
//! independent assortment). The module computes both and compares --
//! enumeration vs. the multiplication law. All probabilities must sum
//! to exactly 1 (Fraction arithmetic, no rounding excuse).

use std::collections::HashMap;

use reasoning_common::Rat;

use crate::biology_domain::{_validate_genotype, phenotype_probability};

#[derive(Debug, Clone, PartialEq)]
pub struct DihybridResult {
    /// e.g. "AaBb" -> Fraction
    pub genotype_probs: HashMap<String, Rat>,
    /// e.g. "dominant-both" -> Fraction
    pub phenotype_probs: HashMap<String, Rat>,
    pub verify_method: String,
    pub verify_values: HashMap<String, Rat>,
}

fn _split_genotypes(genotype: &str) -> Result<(String, String), String> {
    let chars: Vec<char> = genotype.chars().collect();
    if chars.len() != 4 {
        return Err(format!(
            "dihybrid genotype must be 4 alleles like 'AaBb', got '{}'",
            genotype
        ));
    }
    let gene_a: String = chars[..2].iter().collect();
    let gene_b: String = chars[2..].iter().collect();
    _validate_genotype(&gene_a)?;
    _validate_genotype(&gene_b)?;
    if gene_a
        .chars()
        .next()
        .unwrap()
        .to_lowercase()
        .eq(gene_b.chars().next().unwrap().to_lowercase())
    {
        return Err(
            "the two genes must use different letters (e.g. AaBb, not AaAa)".to_string()
        );
    }
    Ok((gene_a, gene_b))
}

/// All 4 gamete allele combinations from a dihybrid parent, e.g.
/// AaBb -> [AB, Ab, aB, ab].
fn _gametes(genotype: &str) -> Result<Vec<String>, String> {
    let (gene_a, gene_b) = _split_genotypes(genotype)?;
    let mut out = Vec::new();
    for ga in gene_a.chars() {
        for gb in gene_b.chars() {
            out.push(format!("{}{}", ga, gb));
        }
    }
    Ok(out)
}

/// Combine two gametes into a 4-allele offspring genotype string,
/// each gene allele-sorted (Aa not aA), gene order preserved.
fn _combine(g1: &str, g2: &str) -> String {
    let c1: Vec<char> = g1.chars().collect();
    let c2: Vec<char> = g2.chars().collect();
    let sort_pair = |mut p: [char; 2]| {
        p.sort_by(|x, y| {
            (x.to_lowercase().to_string(), x.is_lowercase())
                .cmp(&(y.to_lowercase().to_string(), y.is_lowercase()))
        });
        p.iter().collect::<String>()
    };
    let gene_a = sort_pair([c1[0], c2[0]]);
    let gene_b = sort_pair([c1[1], c2[1]]);
    format!("{}{}", gene_a, gene_b)
}

/// Full 16-cell enumeration + product-rule verification.
pub fn dihybrid_cross(parent_a: &str, parent_b: &str) -> Result<DihybridResult, String> {
    let gametes_a = _gametes(parent_a)?;
    let gametes_b = _gametes(parent_b)?;
    let mut counts: HashMap<String, i64> = HashMap::new();
    for g1 in &gametes_a {
        for g2 in &gametes_b {
            let off = _combine(g1, g2);
            *counts.entry(off).or_insert(0) += 1;
        }
    }
    let total: i64 = 16;
    let geno: HashMap<String, Rat> = counts
        .iter()
        .map(|(k, v)| (k.clone(), Rat::new(*v, total)))
        .collect();

    // sanity: exactly 16 cells counted
    if counts.values().sum::<i64>() != total {
        return Err("enumeration error: did not count 16 cells".to_string());
    }

    // phenotype collapse (complete dominance at both genes)
    let mut pheno: HashMap<String, Rat> = HashMap::new();
    for key in [
        "dominant-both",
        "dominant-A-only",
        "dominant-B-only",
        "recessive-both",
    ] {
        pheno.insert(key.to_string(), Rat::from_int(0));
    }
    for (g, p) in &geno {
        let a_dom = g.chars().take(2).any(|c| c.is_uppercase());
        let b_dom = g.chars().skip(2).any(|c| c.is_uppercase());
        let key = if a_dom && b_dom {
            "dominant-both"
        } else if a_dom {
            "dominant-A-only"
        } else if b_dom {
            "dominant-B-only"
        } else {
            "recessive-both"
        };
        let entry = pheno.get_mut(key).expect("phenotype key pre-inserted");
        *entry = *entry + *p;
    }

    let sum = pheno
        .values()
        .fold(Rat::from_int(0), |acc, v| acc + *v);
    if sum != Rat::from_int(1) {
        return Err("phenotype probabilities do not sum to exactly 1".to_string());
    }

    // ---- independent verification: product rule over monohybrid crosses ----
    let (pa_gene1, pa_gene2) = _split_genotypes(parent_a)?; // parent A's two genes
    let (pb_gene1, pb_gene2) = _split_genotypes(parent_b)?; // parent B's two genes
    let mono_a = phenotype_probability(&pa_gene1, &pb_gene1, true)?; // gene-1 cross
    let mono_b = phenotype_probability(&pa_gene2, &pb_gene2, true)?; // gene-2 cross
    let mut expected: HashMap<String, Rat> = HashMap::new();
    expected.insert(
        "dominant-both".to_string(),
        mono_a["dominant"] * mono_b["dominant"],
    );
    expected.insert(
        "dominant-A-only".to_string(),
        mono_a["dominant"] * mono_b["recessive"],
    );
    expected.insert(
        "dominant-B-only".to_string(),
        mono_a["recessive"] * mono_b["dominant"],
    );
    expected.insert(
        "recessive-both".to_string(),
        mono_a["recessive"] * mono_b["recessive"],
    );
    for (k, v) in &expected {
        if *v != pheno[k] {
            return Err(format!(
                "product-rule verification failed for {}: enumeration {} vs product rule {}",
                k, pheno[k], v
            ));
        }
    }
    Ok(DihybridResult {
        genotype_probs: geno,
        phenotype_probs: pheno,
        verify_method: "monohybrid product rule (independent assortment)".to_string(),
        verify_values: expected,
    })
}

/// Convenience: just the phenotype fractions. The classic self-cross
/// AaBb x AaBb must give 9:3:3:1 -- asserted as a special case because it
/// is the textbook result (hand-checkable).
pub fn dihybrid_phenotype_ratio(
    parent_a: &str,
    parent_b: &str,
) -> Result<HashMap<String, Rat>, String> {
    let res = dihybrid_cross(parent_a, parent_b)?;
    if parent_a == "AaBb" && parent_b == "AaBb" {
        let expected: HashMap<String, Rat> = [
            ("dominant-both", Rat::new(9, 16)),
            ("dominant-A-only", Rat::new(3, 16)),
            ("dominant-B-only", Rat::new(3, 16)),
            ("recessive-both", Rat::new(1, 16)),
        ]
        .iter()
        .map(|(k, v)| (k.to_string(), *v))
        .collect();
        if res.phenotype_probs != expected {
            return Err("textbook 9:3:3:1 check failed for AaBb x AaBb".to_string());
        }
    }
    Ok(res.phenotype_probs)
}
