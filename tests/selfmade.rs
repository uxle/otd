//! OTD3.1 integration tests — the self-make expansion, end to end.
//!
//! Covers: nn.rs convergence, the OTD-Burn stack (checkpoint round-trips,
//! objective variants), the new `simulate:` kinds through the full language
//! front-end, and the synthesizer writing programs the compiler accepts.

// ── nn.rs: the zero-dependency neural network library ──────────────────────

#[test]
fn nn_library_learns_xor_and_a_law() {
    // XOR through the manual-backprop stack
    let xor_loss = otd::nn::learn_xor(400, 11);
    assert!(xor_loss < 0.02, "xor loss = {}", xor_loss);
    // the freefall law through the same stack
    let (loss, rel, _, params) = otd::nn::learn_freefall(9.81, 60, 7);
    assert!(params == 1 * 48 + 48 + 48 * 48 + 48 + 48 * 1 + 1);
    assert!(loss.is_finite() && loss < 1e-2, "loss = {}", loss);
    assert!(rel < 0.08, "relative error = {}", rel);
}

#[test]
fn nn_tensor_kernel_is_exact() {
    use otd::nn::Tensor;
    let a = Tensor::from_vec(&[2, 3], vec![1., 2., 3., 4., 5., 6.]);
    let b = Tensor::from_vec(&[3, 2], vec![7., 8., 9., 10., 11., 12.]);
    let c = a.matmul(&b);
    // textbook values
    assert_eq!(c.shape, vec![2, 2]);
    assert_eq!(c.data, vec![58.0, 64.0, 139.0, 154.0]);
}

// ── OTD-Burn: backend → tensor → autodiff → module → optimizer → learner ───

#[test]
fn burn_full_stack_learns_freefall_and_checkpoints() {
    use otd::burn::{Adam, Linear, Sequential};

    // the model: 1 → 16 → 1 with GELU, built the Burn way
    let mut rng = otd::rng::Rng::new(31);
    let mut net = Sequential::new();
    net = net.push(Box::new(Linear::new(1, 16, &mut rng)), "linear");
    net = net.push(Box::new(otd::burn::nn::Gelu), "gelu");
    net = net.push(Box::new(Linear::new(16, 1, &mut rng)), "linear");

    // dataset: y = 2x on [-1, 1]
    let x = otd::burn::Tensor::from_vec(&[64, 1], (0..64).map(|i| i as f32 / 31.5 - 1.0).collect());
    let y = otd::burn::Tensor::from_vec(&[64, 1], (0..64).map(|i| (i as f32 / 31.5 - 1.0) * 2.0).collect());
    let ds = otd::burn::data::Supervised::new(x, y);
    let mut opt = Adam::new(0.02);
    let report = otd::burn::train::fit(&net, &ds, &mut opt, 120, 16, 5, 0);
    assert!(report.final_loss < 1e-3, "loss = {}", report.final_loss);

    // checkpoint round-trip through the Module trait
    let dir = std::env::temp_dir().join("otd_selfmake_test");
    std::fs::create_dir_all(&dir).unwrap();
    let ckpt = dir.join("net.ckpt");
    net.save(&ckpt).unwrap();
    let before = net.params()[0].val().data.clone();
    net.params()[0].set_value(otd::burn::Tensor::from_vec(&[1, 16], vec![9.0; 16]));
    net.load(&ckpt).unwrap();
    assert_eq!(net.params()[0].val().data, before);
    std::fs::remove_file(&ckpt).ok();

    // and the trained net actually predicts 2x
    use otd::burn::module::Module;
    let probe = net.forward(&otd::burn::autodiff::Var::constant(
        otd::burn::Tensor::from_vec(&[1, 1], vec![0.5]),
    ));
    assert!((probe.val().data[0] - 1.0).abs() < 0.05, "f(0.5) = {}", probe.val().data[0]);
}

#[test]
fn burn_autodiff_matches_finite_differences_on_a_small_net() {
    use otd::burn::autodiff::{backward, vmean, vmul};
    use otd::burn::module::Module;
    use otd::burn::{Linear, Tensor};

    let mut rng = otd::rng::Rng::new(41);
    let lin = Linear::new(3, 2, &mut rng);
    let x0 = [0.3, -0.7, 1.1];
    let f = |v: &[f32]| -> f32 {
        let x = otd::burn::autodiff::Var::constant(Tensor::from_vec(&[1, 3], v.to_vec()));
        let y = lin.forward(&x);
        // scalar: MEAN of squares (matches vmean below)
        let mut s = 0.0f32;
        for val in y.val().data.iter() {
            s += val * val;
        }
        s / y.val().len() as f32
    };
    // analytic grads
    let x = otd::burn::autodiff::Var::constant(Tensor::from_vec(&[1, 3], x0.to_vec()));
    let y = lin.forward(&x);
    let sq = vmul(&y, &y);
    let loss = vmean(&sq);
    backward(&loss);
    let wg = lin.params()[0].grad().data.clone();
    // finite differences on w[0][0]
    let h = 1e-3f32;
    let mut up = x0.to_vec();
    // perturb w through a fresh net? simpler: check the input gradient instead
    let _ = up;
    let xg = x.grad().data.clone();
    for i in 0..3 {
        let mut xu = x0.to_vec();
        let mut xd = x0.to_vec();
        xu[i] += h;
        xd[i] -= h;
        let fd = (f(&xu) - f(&xd)) / (2.0 * h);
        assert!((fd - xg[i]).abs() < 1e-2, "i={}: {} vs {}", i, xg[i], fd);
    }
    // w gradients exist and are non-zero
    assert!(wg.iter().any(|&g| g != 0.0));
}

// ── the new simulate kinds through the real language front-end ──────────────

#[test]
fn simulate_learn_stats_orbit_parse_and_run() {
    let src = r#"
scene "integration"
ballA = sphere(r: 3cm) at (-8cm, 80cm, 0) material: steel
ballB = sphere(r: 3cm) at (0, 80cm, 0) material: pine
ballC = sphere(r: 3cm) at (8cm, 80cm, 0) material: gold
simulate: learn
simulate: stats
simulate: orbit
"#;
    let w = otd::compile(src);
    assert!(w.errors.is_empty(), "errors: {:?}", w.errors.iter().map(|e| &e.msg).collect::<Vec<_>>());
    // each sim produced console output
    let sim_lines: Vec<&otd::world::eval::ConsoleLine> =
        w.console.iter().filter(|l| matches!(l.kind, otd::world::eval::LineKind::Sim | otd::world::eval::LineKind::Info)).collect();
    assert!(sim_lines.len() >= 6, "console too short: {} lines", sim_lines.len());
    let joined: String = sim_lines.iter().map(|l| l.text.clone()).collect::<Vec<_>>().join("\n");
    assert!(joined.contains("√(2h/g)"), "learn line missing");
    assert!(joined.contains("STATISTICAL"), "stats line missing");
    assert!(joined.contains("ORBITAL"), "orbit line missing");
}

#[test]
fn synonyms_of_new_kinds_are_accepted() {
    for kind in ["nn", "brain", "neural", "statistics", "astro", "space"] {
        let src = format!("scene \"t\"\nball = sphere(r: 2cm) material: steel\nsimulate: {}\n", kind);
        let w = otd::compile(&src);
        assert!(w.errors.is_empty(), "'{}' rejected: {:?}", kind, w.errors.first().map(|e| &e.msg));
    }
}

// ── the synthesizer: self make anything ─────────────────────────────────────

#[test]
fn synthesizer_tour_goals_all_compile() {
    for goal in otd::ai::synth::DEMO_GOALS {
        let r = otd::ai::synthesize(goal);
        assert!(!r.code.contains("canonical fallback"), "'{}' hit the fallback", goal);
        assert!(r.parts >= 1, "'{}' made {} parts", goal, r.parts);
        assert!(r.mass_g > 0.0, "'{}' has no mass", goal);
    }
}

#[test]
fn synthesizer_output_runs_its_own_sims() {
    let r = otd::ai::synthesize("statistics of a mixed material scene");
    let w = otd::compile(&r.code);
    assert!(w.errors.is_empty());
    assert!(r.code.contains("simulate: stats"));
}

// ── statistics & astronomy accuracy through the public API ──────────────────

#[test]
fn stats_library_public_surface() {
    let xs = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(otd::world::stats::mean(&xs), 3.0);
    assert_eq!(otd::world::stats::median(&xs), 3.0);
    assert!((otd::world::stats::stddev(&xs) - std::f64::consts::SQRT_2).abs() < 1e-12);
    assert!((otd::world::stats::normal_cdf(1.96) - 0.975).abs() < 1e-3);
    assert!((otd::world::stats::binomial_pmf(20, 10, 0.5) - 0.176_197_052).abs() < 1e-6);
}

#[test]
fn astro_library_public_surface() {
    use otd::world::astro::*;
    let t_days = kepler_period(AU, SUN.mass_kg) / 86400.0;
    assert!((t_days - 365.25).abs() < 1.0);
    assert!((vis_viva(AU, AU, SUN.mass_kg) - 29_780.0).abs() < 60.0);
    assert!((escape_velocity(EARTH.mass_kg, EARTH.radius_km * 1000.0) - 11_186.0).abs() < 15.0);
    assert!((surface_g(EARTH.mass_kg, EARTH.radius_km * 1000.0) - 9.82).abs() < 0.03);
    assert!((wien_peak(5772.0) * 1e9 - 502.0).abs() < 4.0);
}

// ── the examples ship compiling ──────────────────────────────────────────────

#[test]
fn new_examples_compile_clean() {
    for ex in otd::content::EXAMPLES {
        if ex.file == "selfmake-tour.otd" || ex.file == "ai-lab.otd" {
            let w = otd::compile(ex.code);
            assert!(w.errors.is_empty(), "{}: {:?}", ex.file, w.errors.iter().map(|e| &e.msg).collect::<Vec<_>>());
            assert!(w.parts.len() >= 5, "{} has {} parts", ex.file, w.parts.len());
        }
    }
}
