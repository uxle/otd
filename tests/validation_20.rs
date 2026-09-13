//! P1290 — the 2.0 validation suite: every architectural claim, measured.
//! These numbers go straight into docs/09-TEST-REPORT.md.

use otd::math3::V3;

fn g_mm_ms() -> f64 { 0.00981 }

#[test]
fn validation_vm_parity() {
    // (the 2000-case fuzz lives in vm::tests::parity_fuzz — here we print
    // a spot-check table)
    let cases = [
        ("4cm + 1", 50.0, 1),
        ("3 * 2cm", 60.0, 1),
        ("90deg / 2", 45.0, 2),
        ("10cm / 2cm", 5.0, 0),
        ("pi * 2", 6.283_185_307_179_586, 0),
        ("sin(30)", 0.5, 0),
        ("cos(60)", 0.5, 0),
        ("sqrt(16)", 4.0, 0),
        ("min(3, 2, 5)", 2.0, 0),
    ];
    println!("== VM spot checks (value, unit-tag 0=plain 1=mm 2=deg) ==");
    for (src, want_v, want_tag) in cases {
        let prog = otd::lang::parser::parse(&format!("x = {}\n", src), "cm");
        let e = prog.stmts.iter().find_map(|s| match s {
            otd::lang::ast::Stmt::Assign(_, e, _) => Some(e.clone()),
            _ => None,
        }).unwrap();
        let p = otd::vm::compile::compile(&e, 10.0).unwrap();
        let r = otd::vm::interp::run(&p, &|_| None).unwrap();
        let ok = (r.v - want_v).abs() < 1e-9 && r.tag as u8 == want_tag as u8;
        println!("  {:<12} = {:<20} tag {} {}", src, r.v, r.tag, if ok { "OK" } else { "FAIL" });
        assert!(ok);
    }
}

#[test]
fn validation_rope_catenary() {
    println!("== XPBD rope vs analytic catenary ==");
    for (span, target) in [(100.0f64, 10.0f64), (600.0, 80.0), (300.0, 25.0)] {
        let rest = otd::world::xpbd::length_for_sag(span, target);
        let pts = otd::world::xpbd::solve_rope(
            V3::ZERO, V3::new(span, 0.0, 0.0), 48, rest, g_mm_ms());
        let sag = otd::world::xpbd::sag_of(&pts);
        let mut arc = 0.0;
        for i in 1..pts.len() { arc += pts[i].sub(&pts[i-1]).len(); }
        println!("  span {}mm target {}mm → measured {:.2}mm (err {:.2}%), arclength {:.2} vs rest {:.2} (Δ {:.3}%)",
            span, target, sag, (sag - target).abs() / target * 100.0, arc, rest,
            (arc - rest).abs() / rest * 100.0);
        assert!((sag - target).abs() / target < 0.03);
        assert!((arc - rest).abs() / rest < 0.005);
    }
}

#[test]
fn validation_subdiv_fairness() {
    println!("== Loop subdivision fairness (torus, valence 6) ==");
    let (m, _) = otd::geo::prims::torus(20.0, 5.0, 16);
    let torus_unfair = |m: &otd::geo::mesh::Mesh| -> f64 {
        let mut acc = 0.0;
        for t in 0..m.tris.len() {
            let [a, b, c] = m.tris[t];
            let (va, vb, vc) = (m.verts[a as usize], m.verts[b as usize], m.verts[c as usize]);
            let n = vb.sub(&va).cross(&vc.sub(&va));
            let p = va.add(&vb).add(&vc).mul(1.0/3.0).sub(&V3::new(0.0, 5.0, 0.0));
            let rho = (p.x()*p.x() + p.z()*p.z()).sqrt();
            let dir = V3::new(p.x()/rho, 0.0, p.z()/rho);
            let exact = p.sub(&dir.mul(20.0));
            acc += n.dot(&exact) / (n.len() * exact.len());
        }
        (acc / m.tris.len() as f64).acos().to_degrees()
    };
    let f0 = torus_unfair(&m);
    let s1 = otd::geo::subdiv::loop_subdivide(&m, 1);
    let f1 = torus_unfair(&s1);
    let s2 = otd::geo::subdiv::loop_subdivide(&m, 2);
    let f2 = torus_unfair(&s2);
    println!("  mean normal deviation: {:.2}° → {:.2}° → {:.2}° (tris {} → {} → {})",
        f0, f1, f2, m.tris.len(), s1.tris.len(), s2.tris.len());
    println!("  volume: {:.1} → {:.1} mm³ ({:.1} % change)", m.volume_signed(), s2.volume_signed(),
        (s2.volume_signed() - m.volume_signed()) / m.volume_signed() * 100.0);
    assert!(f2 < f0);
}

#[test]
fn validation_smooth_and_blend() {
    println!("== DEC smoothing (cube, Taubin) ==");
    let (m, _) = otd::geo::prims::cube(20.0, 20.0, 20.0);
    let v0 = m.volume_signed();
    let s = otd::geo::dec::smooth(&m, 4, 0.5);
    let v1 = s.volume_signed();
    println!("  cube volume {:.0} → {:.0} mm³ ({:.0} % melt), tris stay {}",
        v0, v1, (v0 - v1) / v0 * 100.0, s.tris.len());
    assert!(v1 < v0 && v1 > v0 * 0.25);

    println!("== SDF blend (two 10mm cubes, 5mm overlap, gap 1mm) ==");
    let (a, _) = otd::geo::prims::cube(10.0, 10.0, 10.0);
    let (b, _) = otd::geo::prims::cube(10.0, 10.0, 10.0);
    let mut b = b;
    b.transform(&otd::math3::M4::translate(5.0, 0.0, 0.0));
    let blended = otd::geo::sdf::blend(&a, &b, 1.0, 26);
    let he = otd::geo::halfedge::HalfEdges::build(&blended);
    let rep = he.manifold_report();
    println!("  union 1500 mm³ → blended {:.0} mm³ (+{:.0} % fillet), watertight: {}",
        blended.volume_signed(),
        (blended.volume_signed() - 1500.0) / 15.0, rep.closed);
    assert!(rep.closed);
}

#[test]
fn validation_inertia() {
    println!("== Inertia tensor (exact quadrature) ==");
    let (m, _) = otd::geo::prims::cube(20.0, 20.0, 20.0);
    let i = otd::geo::measure::inertia(&m, 1.0).unwrap();
    let want = 8.0 * (400.0 + 400.0) / 12.0;
    println!("  cube: Ixx = Iyy = Izz = {:.1} g·mm² (analytic m·(s²+s²)/12 = {:.1})", i.diag[0], want);
    assert!((i.diag[0] - want).abs() < 1e-6 * want.max(1.0));
    let (s, _) = otd::geo::prims::sphere(10.0, 32);
    let i = otd::geo::measure::inertia(&s, 2.7).unwrap();
    let m_a = 4.0/3.0*std::f64::consts::PI*1000.0/1000.0*2.7;
    let want = 0.4 * m_a * 100.0;
    println!("  sphere(32): mean I = {:.0} g·mm² (analytic 2/5·m·r² = {:.0}, {:.1} % facet deficit)",
        (i.diag[0]+i.diag[1]+i.diag[2])/3.0, want,
        (want - (i.diag[0]+i.diag[1]+i.diag[2])/3.0)/want*100.0);
}

#[test]
fn validation_corpus_compat() {
    println!("== 1.0 example corpus (backward compatibility) ==");
    for name in ["cup", "table", "gear-system", "robot-arm", "chess-set", "staircase", "steam-engine"] {
        let src = std::fs::read_to_string(format!("examples/{}.otd", name)).unwrap();
        let w = otd::compile(&src);
        let errs = w.errors.len();
        println!("  {:<12} {} parts, {:.1} g, {} errors {}", name, w.parts.len(), w.stats.total_mass_g, errs,
            if errs == 0 { "OK" } else { "REGRESSION" });
        assert_eq!(errs, 0, "{} regressed", name);
    }
}
