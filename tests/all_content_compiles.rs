//! Every embedded lesson + example must compile cleanly under the current
//! parser (guards the syntax fixes against content regressions).
#[test]
fn all_lessons_compile() {
    let mut n = 0;
    for l in otd::content::LESSONS {
        let w = otd::compile(l.code);
        assert!(w.errors.is_empty(), "lesson {:?} failed: {:#?}",
            l.title, w.errors.iter().map(|e| e.render()).collect::<Vec<_>>());
        n += 1;
    }
    assert!(n >= 20, "expected 20+ lessons, found {}", n);
}

#[test]
fn all_examples_compile() {
    // Mechanics Edition: the off-focus examples were removed; this list is
    // the shipped set (steam-engine included — it must always compile).
    for f in ["table", "gear-system", "robot-arm", "chess-set", "cup",
              "staircase", "steam-engine"] {
        let src = std::fs::read_to_string(format!("examples/{}.otd", f)).unwrap();
        let w = otd::compile(&src);
        assert!(w.errors.is_empty(), "example {} failed: {:#?}", f,
            w.errors.iter().map(|e| e.render()).collect::<Vec<_>>());
    }
}
