#[test]
fn render_the_cup() {
    let src = std::fs::read_to_string("examples/cup.otd").unwrap();
    let w = otd::compile(&src);
    let cam = otd::render::camera::Camera::fit(&w.stats.bbox);
    let opts = otd::render::RenderOpts { width: 640, height: 440, ssaa: 2, show_grid: true };
    let png = otd::render::render_world(&w, &cam, &opts);
    std::fs::write("/tmp/otd_cup.png", &png).unwrap();
    assert!(png.len() > 1000);
}
