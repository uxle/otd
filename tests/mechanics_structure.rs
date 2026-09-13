// P1440 — the periodic table library: 118 element folders, 50 content files
// each (30 theory docs + 20 OTD samples), plus the folder README.
// The full OTD parse is covered by library_parses.rs; this test guards the
// STRUCTURE so nobody accidentally deletes an element's files.

use std::path::Path;

#[test]
fn periodic_table_complete() {
    let root = Path::new("library/elements");
    assert!(root.is_dir(), "library/elements must exist");
    let mut dirs: Vec<_> = std::fs::read_dir(root)
        .unwrap()
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.is_dir())
        .collect();
    dirs.sort();
    assert_eq!(dirs.len(), 118, "expected 118 element folders");

    for (i, d) in dirs.iter().enumerate() {
        let name = d.file_name().unwrap().to_string_lossy().to_string();
        let want_prefix = format!("{:03}-", i + 1);
        assert!(name.starts_with(&want_prefix), "folder {} mis-numbered: {}", i + 1, name);
        let files = std::fs::read_dir(d).unwrap().count();
        assert_eq!(files, 51, "{} should hold 51 files (README + 30 docs + 20 samples), has {}", name, files);
        for must in ["01-identity.md", "30-quantum-theory.md", "31-sample-cube.otd", "50-collection-item.otd"] {
            assert!(d.join(must).exists(), "{} is missing {}", name, must);
        }
    }
}

#[test]
fn parts_library_present() {
    // the mechanical parts the user asked for by name
    let parts = Path::new("library/parts");
    assert!(parts.is_dir());
    for must in [
        "nut_m8.otd", "bolt_m10.otd", "piston_v2.otd", "connecting_rod.otd",
        "crankshaft.otd", "gear_five.otd", "gear_six.otd", "gear_seven.otd",
        "gear_eight.otd", "gear_nine.otd", "gear_ten.otd", "gear_eleven.otd",
        "gear_twelve.otd", "flywheel.otd", "bearing_bronze.otd", "spring_coil.otd",
    ] {
        assert!(parts.join(must).exists(), "parts library missing {}", must);
    }
}

#[test]
fn steam_engine_example_exists() {
    let p = Path::new("examples/steam-engine.otd");
    assert!(p.exists(), "examples/steam-engine.otd must exist");
    let src = std::fs::read_to_string(p).unwrap();
    assert!(src.contains("flywheel"), "the engine needs its named flywheel");
    assert!(src.contains("boiler"), "the engine needs a boiler");
}

#[test]
fn off_focus_categories_removed() {
    // the Mechanics Edition drops houses, cars, furniture, games, art
    for gone in ["architecture", "furniture", "games", "art", "nature", "showcase", "patterns"] {
        assert!(!Path::new("library").join(gone).exists(), "library/{} should be removed", gone);
    }
}
