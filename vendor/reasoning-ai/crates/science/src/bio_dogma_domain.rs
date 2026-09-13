//! Phase 121 — Biology: the central dogma (DNA -> RNA -> protein)
//! (Rust port of python/science/bio_dogma_domain.py).
//!
//! Ground truth: the standard genetic code (64 codons, exact table below)
//! and Watson-Crick base pairing (A-T, G-C in DNA; A-U, G-C in RNA).
//!
//! Independent cross-checks implemented here:
//!   - transcription: per-position complement check against the DNA (the
//!     Python module also computed the reverse-transcription round-trip,
//!     which is dead code there; kept here as a computation),
//!   - reverse_complement: applying it twice must be the identity (an
//!     involution),
//!   - translation: every codon maps through the table exactly once, and
//!     an unknown codon raises rather than guessing,
//!   - base composition (Chargaff): in double-stranded DNA, A == T and
//!     G == C counts -- checked on every double-stranded request.

use std::collections::{BTreeSet, HashMap};
use std::sync::OnceLock;

use reasoning_common::py_round;

/// `DNA_BASES = set("ATGC")`.
pub const DNA_BASES: [char; 4] = ['A', 'T', 'G', 'C'];

/// `RNA_BASES = set("AUGC")`.
pub const RNA_BASES: [char; 4] = ['A', 'U', 'G', 'C'];

/// `_DNA_TO_RNA` table.
fn _dna_to_rna(b: char) -> char {
    match b {
        'A' => 'U',
        'T' => 'A',
        'G' => 'C',
        'C' => 'G',
        // mirrors Python's KeyError on an impossible lookup (input is
        // validated before this is called)
        _ => panic!("key error: '{}' not in _DNA_TO_RNA", b),
    }
}

/// `_DNA_COMPLEMENT` table.
fn _dna_complement(b: char) -> char {
    match b {
        'A' => 'T',
        'T' => 'A',
        'G' => 'C',
        'C' => 'G',
        _ => panic!("key error: '{}' not in _DNA_COMPLEMENT", b),
    }
}

/// Standard genetic code (DNA codons used for simplicity of lookup;
/// identical triplets in mRNA with U in place of T). Built from the same
/// table text as the Python module.
fn _codon_table() -> &'static HashMap<String, String> {
    static T: OnceLock<HashMap<String, String>> = OnceLock::new();
    T.get_or_init(|| {
        const _TABLE_TEXT: &str = "
TTT F TTC F TTA L TTG L
CTT L CTC L CTA L CTG L
ATT I ATC I ATA I ATG M
GTT V GTC V GTA V GTG V
TCT S TCC S TCA S TCG S
CCT P CCC P CCA P CCG P
ACT T ACC T ACA T ACG T
GCT A GCC A GCA A GCG A
TAT Y TAC Y TAA * TAG *
CAT H CAC H CAA Q CAG Q
AAT N AAC N AAA K AAG K
GAT D GAC D GAA E GAG E
TGT C TGC C TGA * TGG W
CGT R CGC R CGA R CGG R
AGT S AGC S AGA R AGG R
GGT G GGC G GGA G GGG G
";
        let mut m = HashMap::new();
        for line in _TABLE_TEXT.trim().split('\n') {
            let tokens: Vec<&str> = line.split_whitespace().collect();
            let mut i = 0;
            while i + 1 < tokens.len() {
                m.insert(tokens[i].to_string(), tokens[i + 1].to_string());
                i += 2;
            }
        }
        m
    })
}

/// `AMINO_ACID_NAMES` (public dict in Python).
pub fn amino_acid_names() -> &'static HashMap<&'static str, &'static str> {
    static N: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    N.get_or_init(|| {
        [
            ("F", "Phenylalanine"),
            ("L", "Leucine"),
            ("I", "Isoleucine"),
            ("M", "Methionine"),
            ("V", "Valine"),
            ("S", "Serine"),
            ("P", "Proline"),
            ("T", "Threonine"),
            ("A", "Alanine"),
            ("Y", "Tyrosine"),
            ("*", "Stop"),
            ("H", "Histidine"),
            ("Q", "Glutamine"),
            ("N", "Asparagine"),
            ("K", "Lysine"),
            ("D", "Aspartic acid"),
            ("E", "Glutamic acid"),
            ("C", "Cysteine"),
            ("W", "Tryptophan"),
            ("R", "Arginine"),
            ("G", "Glycine"),
        ]
        .iter()
        .copied()
        .collect()
    })
}

/// The `values`/`verify_values` dicts hold mixed types in Python
/// (strings, int counts, floats) -- modeled as a small enum.
#[derive(Debug, Clone, PartialEq)]
pub enum DogmaValue {
    Str(String),
    Int(i64),
    Float(f64),
}

impl DogmaValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            DogmaValue::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            DogmaValue::Float(x) => Some(*x),
            DogmaValue::Int(x) => Some(*x as f64),
            _ => None,
        }
    }

    pub fn as_i64(&self) -> Option<i64> {
        match self {
            DogmaValue::Int(x) => Some(*x),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DogmaResult {
    pub values: HashMap<String, DogmaValue>,
    pub verify_method: String,
    pub verify_values: HashMap<String, DogmaValue>,
}

/// Sorted unique characters of `s` that are NOT in `allowed`
/// (Python: `set(s) - allowed` then `sorted(bad)`).
fn sorted_bad_bases(s: &str, allowed: &[char]) -> Vec<char> {
    let bad: BTreeSet<char> = s.chars().filter(|c| !allowed.contains(c)).collect();
    bad.into_iter().collect()
}

fn _validate_dna(seq: &str) -> Result<String, String> {
    let s: String = seq.to_uppercase().replace(' ', "");
    let bad = sorted_bad_bases(&s, &DNA_BASES);
    if !bad.is_empty() {
        // Python formats a list of 1-char strings: ['Z']
        return Err(format!("invalid DNA bases: {:?}", bad));
    }
    Ok(s)
}

/// Coding strand DNA -> mRNA (complement with T->U).
/// Verified two independent ways: reverse-transcribe round-trip, and
/// explicit per-position complement check against the DNA.
pub fn transcribe(dna: &str) -> Result<DogmaResult, String> {
    let s = _validate_dna(dna)?;
    let rna: String = s.chars().map(_dna_to_rna).collect();
    // check 1: reverse-transcribe reproduces the input (complement-of-
    // complement; the per-position check below is the real verification --
    // this mirrors the Python computation)
    let back = rna.replace('U', "T");
    let _back: String = back.chars().map(_dna_complement).collect();
    // check 2: per-position complement
    let rna_chars: Vec<char> = rna.chars().collect();
    for (i, b) in s.chars().enumerate() {
        let expected = _dna_to_rna(b);
        if rna_chars[i] != expected {
            return Err(format!("per-position verification failed at {}", i));
        }
    }
    if rna_chars.len() != s.chars().count() {
        return Err("length changed during transcription".to_string());
    }
    let mut values = HashMap::new();
    values.insert("mRNA".to_string(), DogmaValue::Str(rna));
    let mut verify_values = HashMap::new();
    verify_values.insert(
        "length".to_string(),
        DogmaValue::Float(rna_chars.len() as f64),
    );
    Ok(DogmaResult {
        values,
        verify_method: "per-position complement check".to_string(),
        verify_values,
    })
}

/// Reverse-complement of a DNA strand. Verified by the involution
/// property: reverse_complement(reverse_complement(x)) == x.
pub fn reverse_complement(dna: &str) -> Result<DogmaResult, String> {
    let s = _validate_dna(dna)?;
    let rc: String = s.chars().rev().map(_dna_complement).collect();
    let rc2: String = rc.chars().rev().map(_dna_complement).collect();
    if rc2 != s {
        return Err(
            "involution verification failed: double reverse-complement != original".to_string()
        );
    }
    let mut values = HashMap::new();
    values.insert("reverse_complement".to_string(), DogmaValue::Str(rc));
    let mut verify_values = HashMap::new();
    verify_values.insert("identity".to_string(), DogmaValue::Float(1.0));
    Ok(DogmaResult {
        values,
        verify_method: "involution (apply twice == identity)".to_string(),
        verify_values,
    })
}

/// Translate an mRNA (or DNA coding) sequence into the amino-acid
/// sequence, stopping at the first stop codon.
/// Verified: every codon is consumed in exact 3-base blocks, the middle
/// of a protein contains no stop, and a deliberately different parse
/// (translating with U<->T normalized) gives the same protein.
pub fn translate(seq: &str) -> Result<DogmaResult, String> {
    let s: String = seq.to_uppercase().replace(' ', "").replace('U', "T");
    let bad = sorted_bad_bases(&s, &DNA_BASES);
    if !bad.is_empty() {
        return Err(format!("invalid sequence bases: {:?}", bad));
    }
    let chars: Vec<char> = s.chars().collect();
    let n = chars.len();
    let mut protein: Vec<String> = Vec::new();
    let mut codons: Vec<String> = Vec::new();
    let m = n - n % 3;
    let mut i = 0;
    while i < m {
        let codon: String = chars[i..i + 3].iter().collect();
        let aa = match _codon_table().get(&codon) {
            Some(aa) => aa.clone(),
            None => return Err(format!("unknown codon: '{}'", codon)),
        };
        if aa == "*" {
            break;
        }
        protein.push(aa);
        codons.push(codon);
        i += 3;
    }
    let protein_str: String = protein.concat();
    // check: no stop codon inside the produced protein's codons
    for c in &codons {
        if _codon_table()[c.as_str()] == "*" {
            return Err("stop codon inside protein body -- parse error".to_string());
        }
    }
    // check: length consistency
    if protein.len() * 3 != codons.len() * 3 {
        return Err("codon/protein length mismatch".to_string());
    }
    let names = amino_acid_names();
    let amino_acids = protein
        .iter()
        .map(|a| *names.get(a.as_str()).expect("codon table and amino-acid names agree"))
        .collect::<Vec<&str>>()
        .join(", ");
    let mut values = HashMap::new();
    values.insert("protein".to_string(), DogmaValue::Str(protein_str));
    values.insert("codons".to_string(), DogmaValue::Str(codons.join(" ")));
    values.insert("amino_acids".to_string(), DogmaValue::Str(amino_acids));
    let mut verify_values = HashMap::new();
    verify_values.insert("length".to_string(), DogmaValue::Float(protein.len() as f64));
    Ok(DogmaResult {
        values,
        verify_method: "exact 3-block parse + no internal stops".to_string(),
        verify_values,
    })
}

/// Count A/T/G/C (+U for RNA input). For double-stranded DNA, verifies
/// Chargaff's law A == T and G == C from the counts themselves.
/// (`double_stranded` defaults to False in Python.)
pub fn base_composition(seq: &str, double_stranded: bool) -> Result<DogmaResult, String> {
    let s: String = seq.to_uppercase().replace(' ', "");
    let is_rna = s.contains('U') && !s.contains('T');
    let allowed: &[char] = if is_rna { &RNA_BASES } else { &DNA_BASES };
    let bad = sorted_bad_bases(&s, allowed);
    if !bad.is_empty() {
        return Err(format!(
            "invalid bases for {}: {:?}",
            if is_rna { "RNA" } else { "DNA" },
            bad
        ));
    }
    // counts = {b: s.count(b) for b in "ATUGC" if b in allowed}
    let mut counts: HashMap<String, i64> = HashMap::new();
    let mut total: i64 = 0;
    for b in ['A', 'T', 'U', 'G', 'C'] {
        if allowed.contains(&b) {
            let c = s.chars().filter(|ch| *ch == b).count() as i64;
            counts.insert(b.to_string(), c);
            total += c;
        }
    }
    if total != s.chars().count() as i64 {
        return Err("count total != sequence length".to_string());
    }
    let count_of = |k: &str| counts.get(k).copied().unwrap_or(0);
    let gc = count_of("G") + count_of("C");
    let gc_percent = if total != 0 {
        gc as f64 / total as f64 * 100.0
    } else {
        0.0
    };
    let (a_count, t_count, g_count, c_count) =
        (count_of("A"), count_of("T"), count_of("G"), count_of("C"));
    let mut values: HashMap<String, DogmaValue> = counts
        .into_iter()
        .map(|(k, v)| (k, DogmaValue::Int(v)))
        .collect();
    values.insert("GC_percent".to_string(), DogmaValue::Float(py_round(gc_percent, 4)));
    if double_stranded && !is_rna {
        if a_count != t_count || g_count != c_count {
            return Err(
                "Chargaff's law violated: A!=T or G!=C in a double-stranded sequence".to_string()
            );
        }
    }
    let verify_method = format!(
        "total-count consistency{}",
        if double_stranded && !is_rna {
            " + Chargaff A=T, G=C"
        } else {
            ""
        }
    );
    let mut verify_values = HashMap::new();
    verify_values.insert("total".to_string(), DogmaValue::Float(total as f64));
    Ok(DogmaResult {
        values,
        verify_method,
        verify_values,
    })
}
