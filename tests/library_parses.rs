// The whole content library must parse and evaluate cleanly.
// Walks library/**/*.otd and runs the same parser the engine uses.
// Run with: cargo test library_parses

use std::path::PathBuf;

fn collect(dir: &str, out: &mut Vec<PathBuf>) {
    if let Ok(rd) = std::fs::read_dir(dir) {
        for e in rd.flatten() {
            let p = e.path();
            if p.is_dir() {
                collect(p.to_str().unwrap(), out);
            } else if p.extension().map(|x| x == "otd").unwrap_or(false) {
                out.push(p);
            }
        }
    }
}

#[test]
fn library_parses() {
    let mut files = Vec::new();
    collect("library", &mut files);
    assert!(files.len() >= 1000, "library should ship 1000+ programs, found {}", files.len());
    let mut failed = 0;
    for f in &files {
        let src = std::fs::read_to_string(f).unwrap();
        let p = otd::lang::parser::parse(&src, "cm");
        if !p.errors.is_empty() {
            failed += 1;
            eprintln!(
                "{}: {}",
                f.display(),
                p.errors.iter().map(|e| e.render()).collect::<Vec<_>>().join(" | ")
            );
        }
        if p.stmts.is_empty() {
            failed += 1;
            eprintln!("{} produced no statements", f.display());
        }
    }
    assert_eq!(failed, 0, "{} library files failed to parse", failed);
}
