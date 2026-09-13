use otd::world::eval::compile;

#[test]
fn the_coffee_cup_compiles() {
    let src = std::fs::read_to_string("examples/cup.otd").unwrap();
    let w = compile(&src);
    assert!(w.errors.is_empty(), "errors: {:#?}", w.errors.iter().map(|e| e.render()).collect::<Vec<_>>());
    assert!(!w.parts.is_empty());
    println!("parts: {}", w.parts.len());
    for p in &w.parts {
        println!("  {} — vol {:.1} mm³, mass {:.1} g, mat {}", p.name, p.volume_mm3, p.mass_g, p.material.map(|m| m.name).unwrap_or("?"));
    }
    println!("total mass: {:.1} g", w.stats.total_mass_g);
    println!("console:");
    for c in &w.console {
        println!("  [{:?}] {}", c.kind, c.text);
    }
    // analytic: cup wall ≈ 74.5 cm³ (3mm walls) + handle 14.2 cm³ → 88.7 cm³ × 2.4 g/cm³ ≈ 213 g
    assert!(w.stats.total_mass_g > 180.0 && w.stats.total_mass_g < 250.0,
        "cup mass {} g (analytic ≈ 213 g)", w.stats.total_mass_g);
}

#[test]
fn all_examples_compile() {
    // Mechanics Edition example set (off-focus examples removed)
    for f in ["table", "gear-system", "robot-arm", "chess-set", "steam-engine"] {
        let src = std::fs::read_to_string(format!("examples/{}.otd", f)).unwrap();
        let w = compile(&src);
        let errs: Vec<String> = w.errors.iter().map(|e| e.render()).collect();
        assert!(w.errors.is_empty(), "{}: {:#?}", f, errs);
        assert!(w.parts.iter().any(|p| !p.hidden && !p.mesh.is_empty()), "{}: no visible parts", f);
    }
}
