//! Port of the biology classes (TestDihybrid, TestHardyWeinberg,
//! TestCentralDogma, TestEcology) from python/tests/test_science_expansion.py
//! (the physics/chemistry classes live in test_science_expansion.rs, owned
//! by the other conversion agent).
//!
//! `assertAlmostEqual(a, b)` -> `(a - b).abs() < 1e-7`
//! (`places=N` -> `< 1e-N`); `assertRaises(ValueError)` -> `.unwrap_err()`.

use reasoning_common::Rat;
use reasoning_science::bio_dihybrid_domain::{dihybrid_cross, dihybrid_phenotype_ratio};
use reasoning_science::bio_dogma_domain::{base_composition, reverse_complement, transcribe, translate};
use reasoning_science::bio_ecology_domain::{
    doubling_time, energy_transfer_10_percent, exponential_growth, logistic_growth,
};
use reasoning_science::bio_popgen_domain::{
    hardy_weinberg_from_allele_count, hardy_weinberg_from_recessive, hardy_weinberg_stability,
};

// -------------------------------------------------------------------- dihybrid

#[test]
fn test_dihybrid_classic_9_3_3_1() {
    let r = dihybrid_cross("AaBb", "AaBb").unwrap();
    assert_eq!(
        r.phenotype_probs.get("dominant-both"),
        Some(&Rat::new(9, 16))
    );
    assert_eq!(
        r.phenotype_probs.get("dominant-A-only"),
        Some(&Rat::new(3, 16))
    );
    assert_eq!(
        r.phenotype_probs.get("dominant-B-only"),
        Some(&Rat::new(3, 16))
    );
    assert_eq!(
        r.phenotype_probs.get("recessive-both"),
        Some(&Rat::new(1, 16))
    );
}

#[test]
fn test_dihybrid_product_rule_verification_is_built_in() {
    // the function itself verifies enumeration against the monohybrid
    // product rule and raises on any mismatch -- a passing call means
    // two independent derivations agreed
    dihybrid_cross("AaBb", "AaBb").unwrap();
    dihybrid_cross("AaBb", "aabb").unwrap(); // testcross: 1:1:1:1
    let v = dihybrid_cross("AaBb", "aabb")
        .unwrap()
        .phenotype_probs["dominant-both"]
        .to_f64();
    assert!((v - 0.25).abs() < 1e-7);
}

#[test]
fn test_dihybrid_bad_genotypes_raise() {
    dihybrid_cross("AaB", "AaBb").unwrap_err();
    dihybrid_cross("AaAa", "AaBb").unwrap_err();
}

#[test]
fn test_dihybrid_phenotype_ratio_textbook_check() {
    // dihybrid_phenotype_ratio asserts the textbook 9:3:3:1 for AaBb x AaBb
    let ratio = dihybrid_phenotype_ratio("AaBb", "AaBb").unwrap();
    assert_eq!(ratio.get("dominant-both"), Some(&Rat::new(9, 16)));
    assert_eq!(ratio.get("recessive-both"), Some(&Rat::new(1, 16)));
}

// --------------------------------------------------------------- hardy_weinberg

#[test]
fn test_hardy_weinberg_q_squared_16_percent_hand_computed() {
    // q = 0.4, p = 0.6 -> p^2=0.36, 2pq=0.48, q^2=0.16
    let r = hardy_weinberg_from_recessive(0.16).unwrap();
    assert!((r.q - 0.4).abs() < 1e-7);
    assert!((r.p - 0.6).abs() < 1e-7);
    assert!((r.freqs["2pq"] - 0.48).abs() < 1e-7);
}

#[test]
fn test_hardy_weinberg_allele_census_route() {
    // 8 dominant + 2 recessive alleles -> p=0.8, q=0.2
    let r = hardy_weinberg_from_allele_count(8, 2).unwrap();
    assert!((r.p - 0.8).abs() < 1e-7);
    assert!((r.freqs["q^2"] - 0.04).abs() < 1e-7);
}

#[test]
fn test_hardy_weinberg_stability_walk_is_the_law() {
    let r = hardy_weinberg_stability(0.7, 0.3, 3).unwrap();
    assert!((r.freqs["2pq"] - 0.42).abs() < 1e-6); // places=6
}

#[test]
fn test_hardy_weinberg_bad_inputs_raise() {
    hardy_weinberg_from_recessive(0.0).unwrap_err();
    hardy_weinberg_from_recessive(1.5).unwrap_err();
}

// ----------------------------------------------------------------- central_dogma

#[test]
fn test_central_dogma_transcription_hand_computed() {
    // coding strand ATGCAT -> mRNA UACGUA (A->U, T->A, G->C, C->G)
    let r = transcribe("ATGCAT").unwrap();
    assert_eq!(r.values["mRNA"].as_str(), Some("UACGUA"));
}

#[test]
fn test_central_dogma_reverse_complement_involution() {
    // hand check: GATTACA reversed = ACATTAG; complement = TGTAATC
    let r = reverse_complement("GATTACA").unwrap();
    assert_eq!(r.values["reverse_complement"].as_str(), Some("TGTAATC"));
    // involution: applying it twice returns the original (verified inside)
    reverse_complement("ATGCAT").unwrap();
}

#[test]
fn test_central_dogma_translation_with_stop() {
    // ATG GCC TAA -> Met Ala (stop terminates)
    let r = translate("ATGGCCTAA").unwrap();
    assert_eq!(r.values["protein"].as_str(), Some("MA"));
    // unknown codon raises rather than guessing
    translate("ZZZ").unwrap_err();
}

#[test]
fn test_central_dogma_chargaff_check() {
    // double-stranded: AAGC/TTTC complemented -> A=T, G=C holds
    base_composition("AAGCTT", true).unwrap(); // must not raise
    base_composition("AAAG", true).unwrap_err(); // A != T -> raises
}

#[test]
fn test_central_dogma_gc_content() {
    let r = base_composition("GCGC", false).unwrap();
    assert!((r.values["GC_percent"].as_f64().unwrap() - 100.0).abs() < 1e-7);
}

// ------------------------------------------------------------------------ ecology

#[test]
fn test_ecology_exponential_discrete_continuous_agree() {
    // 100 at 10%/yr for 5 yrs -> 100*1.1^5 = 161.051
    let r = exponential_growth(100.0, 0.1, 5.0, false).unwrap();
    assert!((r.values["population"] - 161.051).abs() < 1e-3); // places=3
    // the function cross-checks the continuous reformulation internally
}

#[test]
fn test_ecology_doubling_time_exact() {
    // rate 0.5 continuous -> t2 = ln2/0.5 = 1.386294...
    let r = doubling_time(0.5, true).unwrap();
    assert!((r.values["doubling_time"] - 1.386294).abs() < 1e-4); // places=4
}

#[test]
fn test_ecology_logistic_boundaries() {
    let r = logistic_growth(10.0, 500.0, 0.5, 10.0).unwrap();
    assert!(10.0 <= r.values["population"] && r.values["population"] < 500.0);
    // t=0 reproduces N0 (checked internally too)
    let r0 = logistic_growth(10.0, 500.0, 0.5, 0.0).unwrap();
    assert!((r0.values["population"] - 10.0).abs() < 1e-7);
}

#[test]
fn test_ecology_ten_percent_rule() {
    // 10000 kJ producers -> L2 1000 -> L3 100
    let r = energy_transfer_10_percent(10000.0, 3).unwrap();
    assert!((r.values["energy"] - 100.0).abs() < 1e-7);
    let r1 = energy_transfer_10_percent(10000.0, 1).unwrap();
    assert!((r1.values["energy"] - 10000.0).abs() < 1e-7);
}
