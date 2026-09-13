#[test]
fn all_examples_parse() {
    let files = [
        "cup", "table", "gear-system", "robot-arm", "chess-set",
        "staircase", "steam-engine",
    ];
    for f in files {
        let src = std::fs::read_to_string(format!("examples/{}.otd", f)).unwrap();
        let p = otd::lang::parser::parse(&src, "cm");
        assert!(p.errors.is_empty(), "{} failed: {:?}", f,
            p.errors.iter().map(|e| e.render()).collect::<Vec<_>>());
        assert!(!p.stmts.is_empty(), "{} produced no statements", f);
    }
}
