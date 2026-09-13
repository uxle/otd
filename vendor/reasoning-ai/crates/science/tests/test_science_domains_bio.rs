//! Port of the `TestGenetics` class from python/tests/test_science_domains.py
//! (the other classes live in test_science_domains.rs, owned by the
//! physics/chemistry conversion agent).

use std::collections::HashMap;

use reasoning_common::Rat;
use reasoning_science::biology_domain::{
    genotype_probabilities, phenotype_probability, punnett_square,
};

#[test]
fn test_genetics_punnett_square_heterozygous_cross_has_4_cells() {
    let offspring = punnett_square("Aa", "Aa").unwrap();
    assert_eq!(offspring.len(), 4);
}

#[test]
fn test_genetics_classic_monohybrid_3to1_ratio() {
    // Hand-known Mendel result: Aa x Aa -> 1 AA : 2 Aa : 1 aa (genotype),
    // 3 dominant : 1 recessive (phenotype)
    let geno = genotype_probabilities("Aa", "Aa").unwrap();
    assert_eq!(geno.get("AA"), Some(&Rat::new(1, 4)));
    assert_eq!(geno.get("Aa"), Some(&Rat::new(1, 2)));
    assert_eq!(geno.get("aa"), Some(&Rat::new(1, 4)));

    let pheno = phenotype_probability("Aa", "Aa", true).unwrap();
    assert_eq!(pheno.get("dominant"), Some(&Rat::new(3, 4)));
    assert_eq!(pheno.get("recessive"), Some(&Rat::new(1, 4)));
}

#[test]
fn test_genetics_homozygous_dominant_x_homozygous_recessive_all_heterozygous() {
    // Hand-known: AA x aa -> 100% Aa
    let geno = genotype_probabilities("AA", "aa").unwrap();
    let mut expected = HashMap::new();
    expected.insert("Aa".to_string(), Rat::new(1, 1));
    assert_eq!(geno, expected);
}

#[test]
fn test_genetics_testcross_1to1_ratio() {
    // Hand-known: Aa x aa -> 1/2 Aa, 1/2 aa -> phenotype 1/2 dominant, 1/2 recessive
    let pheno = phenotype_probability("Aa", "aa", true).unwrap();
    assert_eq!(pheno.get("dominant"), Some(&Rat::new(1, 2)));
    assert_eq!(pheno.get("recessive"), Some(&Rat::new(1, 2)));
}

#[test]
fn test_genetics_mismatched_gene_letters_raises() {
    punnett_square("Aa", "Bb").unwrap_err();
}

#[test]
fn test_genetics_wrong_length_genotype_raises() {
    punnett_square("A", "Aa").unwrap_err();
}
