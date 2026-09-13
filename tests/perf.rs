use std::time::Instant;

#[test]
fn performance_targets() {
    // 1. parse speed
    let src = std::fs::read_to_string("examples/cup.otd").unwrap();
    let t = Instant::now();
    for _ in 0..100 {
        let _ = otd::lang::parser::parse(&src, "cm");
    }
    let parse_ms = t.elapsed().as_secs_f64() * 1000.0 / 100.0;
    println!("parse cup.otd: {:.3} ms", parse_ms);

    // 2. compile speed (CSG + hollow + patterns)
    for name in ["cup", "table", "gear-system", "robot-arm", "chess-set", "staircase", "steam-engine"] {
        let src = std::fs::read_to_string(format!("examples/{}.otd", name)).unwrap();
        let t = Instant::now();
        let w = otd::compile(&src);
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        println!("compile {}: {:.1} ms ({} parts, {:.1} g)", name, ms, w.parts.len(), w.stats.total_mass_g);
        assert!(ms < 8000.0, "{} took {} ms", name, ms);
    }

    // 3. render speed @ 2× SSAA
    let src = std::fs::read_to_string("examples/cup.otd").unwrap();
    let w = otd::compile(&src);
    let cam = otd::render::camera::Camera::fit(&w.stats.bbox);
    let opts = otd::render::RenderOpts { width: 900, height: 600, ssaa: 2, show_grid: true };
    let t = Instant::now();
    let png = otd::render::render_world(&w, &cam, &opts);
    let render_ms = t.elapsed().as_secs_f64() * 1000.0;
    println!("render cup 900×600 @2×SSAA: {:.0} ms ({} KB png)", render_ms, png.len() / 1024);
    assert!(render_ms < 2000.0, "render took {} ms", render_ms);
}
