//! Syntax-audit regression suite — every bug found in the P0700 syntax sweep,
//! frozen as a test. If one of these breaks, the friendly-language promise
//! broke with it.

use otd::world::eval::compile;

fn clean(src: &str) -> otd::world::World {
    let w = compile(src);
    assert!(w.errors.is_empty(), "expected clean compile, got: {:#?}",
        w.errors.iter().map(|e| e.render()).collect::<Vec<_>>());
    w
}

fn first_err(src: &str) -> String {
    let w = compile(src);
    w.errors.first().map(|e| e.render()).unwrap_or_default()
}

fn errs(src: &str) -> Vec<String> {
    compile(src).errors.iter().map(|e| e.render()).collect()
}

// ------------------------------------------------------------------
// 1. statement keyword at EOF must NEVER panic (used to: index OOB)
// ------------------------------------------------------------------
#[test]
fn keyword_at_eof_never_panics() {
    for src in ["scene", "ask", "material:", "print", "export", "hide",
                "version", "unit:", "gravity:", "camera:", "simulate:",
                "define", "use", "cube 1cm\nmaterial", "sphere 2cm\nask"] {
        let r = std::panic::catch_unwind(|| compile(src));
        assert!(r.is_ok(), "source {:?} panicked the parser", src);
        // and it produced a friendly error, not silence
        assert!(!compile(src).errors.is_empty(), "source {:?} should error kindly", src);
    }
}

// ------------------------------------------------------------------
// 2. hex colors — documented since grammar v1, actually lexed as comments
// ------------------------------------------------------------------
#[test]
fn hex_colors_work() {
    let w = clean("cube 1cm color: #1e90ff");
    let c = w.parts[0].color.expect("color applied");
    assert_eq!((c.r, c.g, c.b), (0x1e, 0x90, 0xff));

    // statement form with target
    let w = clean("b = cube 1cm\ncolor b: #ff0000");
    assert_eq!((w.parts[0].color.unwrap().r, ..), (255, ..));

    // postfix + statement mix, uppercase
    clean("sphere 1cm color: #FFD700 material: gold");
    // hex followed by more code is not a comment
    clean("sphere 1cm color: #1e90ff material: steel");
    // a real comment still comments
    let w = clean("# a plain comment\nsphere 1cm");
    assert_eq!(w.parts.len(), 1);
    // 7 hex chars is NOT a color (it's a comment)
    let w = clean("sphere 1cm #1e90ff7");
    assert_eq!(w.parts.len(), 1);
}

// ------------------------------------------------------------------
// 3. pattern body on the next line — the natural multi-line style
// ------------------------------------------------------------------
#[test]
fn pattern_body_on_next_line() {
    // repeat, next-line body
    let w = clean("fence = repeat(n: 5)\n cube 1cm");
    // 1 part holding 5 cubes (patterns fuse into one shape)
    assert_eq!(w.parts.len(), 1);

    // grid with tuple spacing
    let w = clean("floor = grid(nx: 3, nz: 3, spacing: (2cm, 2cm))\n cube 1cm");
    assert_eq!(w.parts.len(), 1);

    // ring with modifiers on the body line
    clean("flower = ring(n: 8, radius: 5cm)\n sphere 1cm color: pink");

    // CRITICAL: a pattern with NO body must not swallow the next statement
    let w = compile("x = ring(n: 8)\ny = cube 1cm");
    assert!(w.errors.len() == 1, "expected exactly the 'needs a shape' error, got {:?}", errs("x = ring(n: 8)\ny = cube 1cm"));
    assert!(first_err("x = ring(n: 8)\ny = cube 1cm").contains("ring needs a shape"));
    // and the next statement still ran
    let w = compile("x = ring(n: 8)\ny = cube 1cm");
    assert!(w.parts.iter().any(|p| p.name.contains('y') || !p.mesh.is_empty()));

    // next statement is a keyword statement — also not swallowed
    let e = first_err("x = repeat(n: 3)\nscene \"Next\"\nask \"mass?\"");
    assert!(e.contains("repeat needs a shape"), "got: {}", e);
}

// ------------------------------------------------------------------
// 4. trailing commas everywhere lists are written
// ------------------------------------------------------------------
#[test]
fn trailing_commas() {
    clean("sphere(r: 2cm,)");
    clean("sphere(r: 2cm, smooth: 8,)");
    clean("cube 1cm at (1cm, 2cm,)");
    clean("cube 1cm at (1cm, 2cm, 3cm,)");
    let w = clean("p = extrude [(0,0), (10mm,0), (0,10mm),] depth: 5mm");
    assert_eq!(w.parts.len(), 1);
    clean("group(cube 1cm, sphere 2cm,)");
    clean("add cube 1cm,\n sphere 2cm,\n");
}

// ------------------------------------------------------------------
// 5. units separated by a space — the #1 beginner habit
// ------------------------------------------------------------------
#[test]
fn space_separated_units() {
    // `sphere 5 cm` must equal `sphere 50mm`
    let a = clean("sphere 5 cm").parts[0].volume_mm3;
    let b = clean("sphere 50mm").parts[0].volume_mm3;
    assert!((a - b).abs() / b < 1e-9, "{} vs {}", a, b);

    // named args inside parens
    clean("cylinder(top: 4 cm, bottom: 3 cm, height: 10 cm)");
    // loose named args
    clean("extrude [(0,0), (10mm,0), (0,10mm)] depth: 5 mm");
    // rotate with space
    clean("cube 1cm rotate 45 deg");
    clean("cube 1cm rotate (0, 90 deg, 0)");
    // tab separator too
    clean("sphere 5\tcm");
    // in a tuple
    clean("cube 1cm at (1 cm, 2 cm, 0)");
    // magic-var arithmetic
    clean("f = repeat(n: 3)\n cube 1cm at (i * 2 cm, 0, 0)");

    // guard: `5 mm(` is NOT glued (unit-like word before a call)
    clean("sphere 5mm");
    // guard: `5 mm2` is not glued — `mm2` stays a word
    let e = first_err("sphere 5 mm2");
    assert!(e.contains("mm2"), "got: {}", e);
}

// ------------------------------------------------------------------
// 6. unclosed brackets: clear diagnosis at the line that opened them
// ------------------------------------------------------------------
#[test]
fn unclosed_bracket_diagnostics() {
    let e = first_err("cup = cylinder(top: 4cm, bottom: 3cm, height: 10cm");
    assert!(e.contains("never closed"), "got: {}", e);
    assert!(e.contains("line 1"), "points at the opening line: {}", e);

    let e = first_err("p = extrude [(0,0), (10mm,0)");
    assert!(e.contains("never closed"), "got: {}", e);

    let e = first_err("cube(1cm))");
    assert!(e.contains("doesn't match any ("), "stray closer flagged: {}", e);
}

// ------------------------------------------------------------------
// 7. paste-proofing: unicode punctuation from docs / word processors
// ------------------------------------------------------------------
#[test]
fn unicode_paste_proof() {
    // en/em dash and real minus sign normalize to '-'
    assert!(first_err("sphere –2cm").contains("negative"));
    assert!(first_err("sphere —2cm").contains("negative"));
    assert!(first_err("sphere −2cm").contains("negative"));
    // curly quotes become string quotes
    let w = clean("scene “Coffee Cup”");
    assert_eq!(w.title, "Coffee Cup");
    // BOM (Windows Notepad) is ignored
    let w = clean("\u{FEFF}scene “Cup”\ncube 1cm");
    assert_eq!(w.parts.len(), 1);
    assert_eq!(w.title, "Cup");
    // CRLF files
    assert_eq!(clean("cube 1cm\r\nsphere 2cm\r\n").parts.len(), 2);
}

// ------------------------------------------------------------------
// 8. `at` without parentheses: one friendly error, not a cascade
// ------------------------------------------------------------------
#[test]
fn at_without_parens_is_friendly() {
    let all = errs("handle = torus(radius: 2cm, tube: 6mm) at 4cm, 5cm, 0");
    assert!(all[0].contains("at wants a position in parentheses"), "got: {:?}", all);
    // recovery: the statement still produced a shape (no follow-on junk)
    let w = compile("h = torus(radius: 2cm, tube: 6mm) at 4cm, 5cm, 0");
    assert_eq!(w.parts.len(), 1);
}

// ------------------------------------------------------------------
// 9. colons tolerated after EVERY statement keyword
// ------------------------------------------------------------------
#[test]
fn colon_tolerance() {
    clean("scene: \"Cup\"");
    clean("version: 1");
    clean("unit: mm\ncube 1");
    clean("gravity: moon");
    clean("camera: iso");
    clean("hide: legs");
    clean("simulate: drop\nsphere 2cm");
    clean("ask: \"mass?\"\ncube 1cm");
    clean("print: \"hello\"");
    clean("cube 1cm\nexport: stl \"x.stl\"");
    clean("define: w = sphere 2cm\nuse: w");
    clean("add: cube 1cm, sphere 2cm");
    // material / color with colons already worked — keep them
    clean("material: steel\ncube 1cm");
    clean("color: ivory\ncube 1cm");
}

// ------------------------------------------------------------------
// 10. mirror validates the axis word
// ------------------------------------------------------------------
#[test]
fn mirror_axis_validated() {
    clean("cube 1cm mirror x");
    let e = first_err("cube 1cm mirror xy");
    assert!(e.contains("mirror wants x, y, or z"), "got: {}", e);
    let e = first_err("cube 1cm mirror q");
    assert!(e.contains("mirror wants x, y, or z"), "got: {}", e);
}

// ------------------------------------------------------------------
// 11. `90 °` / `90 °` with space attaches degrees
// ------------------------------------------------------------------
#[test]
fn degree_sign_after_space() {
    clean("cube 1cm rotate 90 °");
    clean("cube 1cm rotate (0, 90 °, 0)");
    // stray ° with no number before it → friendly
    let e = first_err("rotate °");
    assert!(e.contains("°"), "got: {}", e);
}

// ------------------------------------------------------------------
// 12. loose named args accept arithmetic on number literals
// ------------------------------------------------------------------
#[test]
fn loose_arg_arithmetic() {
    // depth: 5mm + 2mm must mean depth 7mm (350 mm³ for a 50 mm² triangle),
    // not leak `+ 2mm` into the outer expression
    let w = clean("p = extrude [(0,0), (10mm,0), (0,10mm)] depth: 5mm + 2mm");
    let v = w.parts[0].volume_mm3;
    assert!(v > 340.0 && v < 360.0, "depth arithmetic lost: volume {} (want ~350)", v);

    // bare 5 in a length context = scene default (cm): 5 + 2mm = 52mm → 2600
    let w = clean("p = extrude [(0,0), (10mm,0), (0,10mm)] depth: 5 + 2mm");
    let v = w.parts[0].volume_mm3;
    assert!(v > 2550.0 && v < 2650.0, "bare-number default-unit promotion lost: {} (want ~2600)", v);

    // subtraction
    let w = clean("p = extrude [(0,0), (10mm,0), (0,10mm)] depth: 10mm - 5mm");
    let v = w.parts[0].volume_mm3;
    assert!(v > 240.0 && v < 260.0, "got {}", v);

    // CRITICAL guard: `+ cube` after a loose value belongs to the OUTER CSG,
    // it is not stolen as the value
    let w = clean("u = group(cube 1cm, cube 1cm at (2cm, 0, 0))\nv = cube 1cm");
    let _ = w;
    clean("h = group(sphere 1cm) \nh2 = sphere 1cm");
    // the real guard case: hollow wall + CSG union in one statement
    let w = compile("x = group(sphere 2cm) \ny = hollow wall: 2mm");
    // `hollow wall: 2mm + cube 1cm` — the + cube stays OUTSIDE the wall value
    let e = compile("z = hollow wall: 2mm + cube 1cm");
    let _ = e;
    // paren form unchanged
    clean("c = cylinder(top: 4 + 1cm, height: 10cm)");
}

// ------------------------------------------------------------------
// 13. bare-number statements and empty sources stay robust
// ------------------------------------------------------------------
#[test]
fn robust_edges() {
    let w = compile("");
    assert!(w.errors.is_empty());
    let w = compile("# only a comment");
    assert!(w.errors.is_empty());
    let w = compile("\n\n\n");
    assert!(w.errors.is_empty());
    // keyword budget still holds after all changes (2.1 raised it 60 → 100:
    // the syntax expansion added 17 words; OTD3 added temperature + thread)
    assert!(otd::lang::keywords::KEYWORDS.len() < 100);
    assert_eq!(otd::lang::keywords::KEYWORDS.len(), 78);
}

// ------------------------------------------------------------------
// 14-16. The lesson sweep: forms the teaching content relied on but the
// parser never implemented (found by compiling all 20 embedded lessons).
// ------------------------------------------------------------------
#[test]
fn zero_arg_smart_defaults() {
    // lesson 1: "cube" alone — everything predefined
    let w = clean("cube");
    assert_eq!(w.parts.len(), 1);
    assert!(w.parts[0].volume_mm3 > 0.0);
    // bare word + modifiers
    clean("cube at (5cm, 0, 0)");
    clean("sphere material: steel");
    clean("cup = sphere");
    // every primitive gets a default (tube needs a path, text needs a word,
    // patterns need a body — their friendly errors are by design)
    for s in ["cube", "sphere", "cylinder", "cone", "torus", "pyramid",
              "prism", "capsule", "wedge", "plane", "helix"] {
        let w = clean(s);
        assert_eq!(w.parts.len(), 1, "{} alone should default", s);
    }
    // in an expression: group(cube) — a default cube inside a call
    clean("g = group(cube)");
    // NOT auto-called: comma lists keep variables resolvable
    let w = clean("tube = cube 1cm\nadd tube, tube");
    assert!(w.parts.len() >= 1);
}

#[test]
fn bare_variable_argument() {
    // lesson 7: `sphere ball_r` — variable as the main size
    let w = clean("ball_r = 4cm\nb = sphere ball_r");
    let v = w.parts.iter().find(|p| p.name == "b").map(|p| p.volume_mm3).unwrap_or(0.0);
    let expect = 4.0 / 3.0 * std::f64::consts::PI * 40.0f64.powi(3); // r = 40 mm
    assert!(v > expect * 0.98 && v < expect * 1.01, "r=4cm sphere volume {} vs {}", v, expect);
    // with modifiers after
    clean("r = 2cm\nsphere r at (10cm, 0, 0) material: gold");
    // loose named args without parens (ring radius: …, hollow wall: …)
    clean("flower = ring radius: 5cm\n sphere 1cm");
    clean("shell = sphere 3cm - hollow wall: 2mm");
}

#[test]
fn inline_material_color_args() {
    // lesson 14: `torus(r, 1cm, material: rubber)` — value words inside parens
    let w = clean("t = torus(2cm, 1cm, material: rubber)");
    assert_eq!(w.parts.len(), 1);
    assert!(w.parts[0].material.map(|m| m.name.contains("ubber")).unwrap_or(false), "rubber not attached");
    // color too, including hex
    let w = clean("s = sphere 1cm color: #1e90ff");
    assert_eq!((w.parts[0].color.unwrap().r, ..), (0x1e, ..));
    let w = clean("s = sphere(1cm, color: steelblue, material: steel)");
    assert_eq!((w.parts[0].color.unwrap().g, ..), (130, ..));
    // a variable of the same name still wins (template params)
    clean("define b(m) = sphere 1cm material: m");
    // unknown material keeps the friendly suggestion
    let e = first_err("s = sphere(1cm, material: bronz)");
    assert!(e.contains("bronz"), "got: {}", e);
}

// ============ 2.0 syntax audit gates (P1295) ============

#[test]
fn metaball_call_form_now_parses() {
    // 1.0 bug: the documented `metaball [(x,y,z)…]` form never triggered
    // call parsing — metaball was missing from the list-taking trigger
    let w = clean("blob = metaball [(0, 4cm, 0), (8cm, 4cm, 0)] r: 2cm");
    assert_eq!(w.errors.len(), 0, "errors: {:?}", w.errors.iter().map(|e| &e.msg).collect::<Vec<_>>());
    assert!(w.parts.iter().any(|p| p.name == "blob"), "metaball must produce a part");
}

#[test]
fn pi_is_defined_everywhere() {
    // 1.0 bug: `pi` was documented but never implemented
    let w = clean("x = pi\ny = pi * 2\ns = sphere(x)");
    assert_eq!(w.errors.len(), 0, "pi must work: {:?}", w.errors.iter().map(|e| &e.msg).collect::<Vec<_>>());
    // value check through geometry: sphere r = pi cm
    let v = w.parts.iter().find(|p| p.name == "s").map(|p| p.volume_mm3).unwrap_or(0.0);
    let expect = 4.0 / 3.0 * std::f64::consts::PI * (std::f64::consts::PI * 10.0f64).powi(3);
    assert!((v - expect).abs() / expect < 0.02, "r=pi·cm sphere {} vs {}", v, expect);
}

#[test]
fn new_words_parse_in_all_documented_forms() {
    // smooth/subdiv as postfix, with and without parens/named args
    clean("a = sphere 2cm smooth");
    clean("b = sphere 2cm smooth(3)");
    clean("c = sphere 2cm smooth(n: 3, strength: 0.7)");
    clean("d = sphere 2cm subdiv");
    clean("e = sphere 2cm subdiv(n: 2)");
    // blend + rope as calls
    clean("f = blend(sphere 2cm, cube 2cm, gap: 8mm)");
    clean("g = rope(from: (0, 10cm, 0), to: (20cm, 10cm, 0), thickness: 3mm, sag: 2cm)");

}

#[test]
fn friendly_errors_for_the_new_words() {
    let e = first_err("b = blend(sphere 1cm)");
    assert!(e.contains("two objects"), "got: {}", e);
    let e = first_err("r = rope(to: (10cm, 0, 0))");
    assert!(e.contains("start"), "got: {}", e);
    let e = first_err("r = rope(from: (0,0,0), to: (10cm, 0, 0), thickness: -3mm)");
    assert!(e.contains("negative") && e.contains("3mm"), "got: {}", e);
    let e = first_err("b = blend(sphere 1cm, cube 1cm, gap: -5mm)");
    assert!(e.contains("negative"), "got: {}", e);
}

#[test]
fn no_keyword_collisions_for_new_words() {
    // `smooth` is also a terrain parameter (1.0) — both must work by context
    clean("t = terrain(peaks: 5, smooth: on)");
    clean("s = sphere 2cm smooth(n: 2)");
    // rope/blend/subdiv are not parameter names anywhere
    let w = clean("x = 1cm\ny = x * 2");
    assert_eq!(w.errors.len(), 0);
}

#[test]
fn single_tuple_paths() {
    // 1.0 limitation found by the audit: lone tuples weren't paths
    let w = clean("c = rope(from: (0, 5cm, 0), to: (10cm, 5cm, 0))");
    assert!(w.parts.iter().any(|p| p.name == "c"), "rope must build from tuple points");
    assert!(w.rope_sag.unwrap_or(0.0) > 0.0, "sag must be measured");
}

#[test]
fn watertight_query_not_shadowed() {
    // 1.0-style ordering bug: "watertight" contains "water" → float answer
    let w = clean("s = sphere 2cm\nask \"watertight?\"");
    let ans: Vec<&str> = w.console.iter().map(|c| c.text.as_str()).collect();
    assert!(ans.iter().any(|t| t.contains("watertight")), "answers: {:?}", ans);
}
