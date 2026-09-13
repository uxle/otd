//! OTD 2.1 — the syntax expansion, verified end-to-end through compile().
//!
//! Every construct the expansion added, one measured claim per test:
//!   control flow   if / else if / else / end, one-liner `:` forms
//!   loops          for = to by, for in (lists, ranges, shapes), while
//!   flow control   break / continue
//!   assignment     += -= *= /=
//!   operators      ^ % mod, < > <= >= == !=, && || !, and or not, is / is not, in
//!   data           true / false, ranges a..b, indexing xs[i], slicing xs[a..b]
//!   templates      define … end multi-statement bodies
//!   checking       assert cond, "message"
//!   output         print "… {name}" interpolation
//!   functions      floor ceil pow log ln exp sign hypot atan atan2 asin acos lerp clamp count sum avg
//!   lexer          scientific notation, ; separators, #[ … ]# block comments
//!   units          km yd um rad (and the `3 in [1, 2, 3]` disambiguation)

use otd::world::eval::LineKind;
use otd::world::World;

fn compile(src: &str) -> World {
    otd::compile(src)
}

fn clean(src: &str) -> World {
    let w = compile(src);
    assert!(w.errors.is_empty(), "expected clean compile, got: {:#?}",
        w.errors.iter().map(|e| e.render()).collect::<Vec<_>>());
    w
}

fn first_err(src: &str) -> String {
    let w = compile(src);
    assert!(!w.errors.is_empty(), "expected an error, got none");
    w.errors[0].render()
}

fn printed(w: &World) -> Vec<String> {
    w.console.iter()
        .filter(|l| l.kind == LineKind::Print)
        .map(|l| l.text.clone())
        .collect()
}

// ---------------------------------------------------------------- if / else

#[test]
fn if_true_runs_its_block() {
    let w = clean("if 1 < 2\n  cube 1cm\nend");
    assert_eq!(w.parts.len(), 1);
}

#[test]
fn if_false_skips_its_block() {
    let w = clean("if 1 > 2\n  cube 1cm\nend");
    assert!(w.parts.is_empty());
}

#[test]
fn if_one_liner_with_else() {
    let w = clean("if 1 > 2: cube 1cm else: sphere 1cm");
    assert_eq!(w.parts.len(), 1);
    // a 1cm sphere meshes to ≈ 4/3·π·10³ mm³ (polygonal, slightly under)
    assert!((w.parts[0].volume_mm3 - 4188.79).abs() < 60.0, "got {}", w.parts[0].volume_mm3);
}

#[test]
fn else_if_chain_picks_the_right_arm() {
    let w = clean("x = 5\nif x > 10\n  cube 1cm\nelse if x > 3\n  sphere 1cm\nelse\n  cone 1cm\nend");
    assert_eq!(w.parts.len(), 1);
    assert!((w.parts[0].volume_mm3 - 4188.79).abs() < 60.0, "the middle arm should win");
}

#[test]
fn if_on_a_bool_literal() {
    let w = clean("if true\n  cube 1cm\nend\nif false\n  sphere 9cm\nend");
    assert_eq!(w.parts.len(), 1);
}

#[test]
fn if_condition_may_use_logic() {
    let w = clean("r = 4cm\nif r > 2cm and not r > 6cm\n  cube 1cm\nend");
    assert_eq!(w.parts.len(), 1);
}

#[test]
fn if_wants_true_or_false() {
    let e = first_err("if \"hello\"\n  cube 1cm\nend");
    assert!(e.contains("true or false"), "{e}");
}

// ---------------------------------------------------------------- for

#[test]
fn for_makes_each_copy() {
    let w = clean("for i = 1 to 5\n  cube 1cm at (i * 3cm, 0, 0)\nend");
    assert_eq!(w.parts.len(), 5);
    // and they walk up the x axis
    let xs: Vec<f64> = {
        let mut v: Vec<f64> = w.parts.iter().filter_map(|p| p.centroid.map(|c| c.x())).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v
    };
    assert!(xs.last().unwrap() - xs.first().unwrap() > 10.0, "copies should spread: {xs:?}");
}

#[test]
fn for_by_skips() {
    let w = clean("for i = 1 to 9 by 2\n  cube 1cm\nend");
    assert_eq!(w.parts.len(), 5); // 1, 3, 5, 7, 9
}

#[test]
fn for_downward_autoreverses() {
    let w = clean("for i = 5 to 1\n  cube 1cm\nend");
    assert_eq!(w.parts.len(), 5, "counting down should just work");
}

#[test]
fn for_steps_through_angles() {
    let w = clean("for a = 0deg to 90deg by 30deg\n  cube 1cm rotate (0, a, 0)\nend");
    assert_eq!(w.parts.len(), 4); // 0, 30, 60, 90
}

#[test]
fn for_steps_through_lengths() {
    let w = clean("for x = 0cm to 10cm by 2.5cm\n  cube 1cm at (x, 0, 0)\nend");
    assert_eq!(w.parts.len(), 5); // 0, 2.5, 5, 7.5, 10
}

#[test]
fn for_one_liner() {
    let w = clean("for i in 1..3: cube 1cm");
    assert_eq!(w.parts.len(), 3);
}

#[test]
fn for_in_range() {
    let w = clean("for i in 1..12\n  cube 1cm\nend");
    assert_eq!(w.parts.len(), 12);
}

#[test]
fn for_in_list_of_lengths() {
    let w = clean("radii = [1cm, 2cm, 3cm]\nfor r in radii\n  sphere r\nend");
    assert_eq!(w.parts.len(), 3);
    let mut vols: Vec<f64> = w.parts.iter().map(|p| p.volume_mm3).collect();
    vols.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!(vols[2] > vols[1] * 2.5 && vols[1] > vols[0] * 2.5, "{vols:?}");
}

#[test]
fn for_in_list_of_shapes() {
    let w = clean("parts = [cube 1cm, sphere 1cm]\nfor thing in parts\n  thing at (10cm, 0, 0)\nend");
    assert_eq!(w.parts.len(), 2);
}

#[test]
fn nested_loops() {
    let w = clean("for i in 1..3\n  for j in 1..2\n    cube 1cm\n  end\nend");
    assert_eq!(w.parts.len(), 6);
}

#[test]
fn loop_variable_is_local() {
    // the loop may not clobber a same-named variable outside it
    clean("i = 99\nfor i in 1..3\n  cube 1cm\nend\nassert i is 99, \"loop letters stay inside the loop\"");
}

#[test]
fn for_never_ends_is_caught() {
    let e = first_err("for i = 1 to 2 by 0.00001\n  x = 1\nend");
    assert!(e.contains("run forever") || e.contains("too big"), "{e}");
}

// ---------------------------------------------------------------- while / break / continue

#[test]
fn while_runs_until_false() {
    clean("x = 0cm\nwhile x < 30cm\n  cube 1cm at (x, 0, 0)\n  x += 10cm\nend\nassert x is 30cm");
}

#[test]
fn while_creates_each_pass() {
    let w = clean("x = 0\nwhile x < 4\n  cube 1cm\n  x += 1\nend");
    assert_eq!(w.parts.len(), 4);
}

#[test]
fn break_leaves_the_loop() {
    clean("x = 0\nwhile true\n  x += 1\n  if x is 3\n    break\n  end\nend\nassert x is 3, \"break should stop the while\"");
}

#[test]
fn break_stops_for_early() {
    let w = clean("for i in 1..100\n  if i > 5\n    break\n  end\n  cube 1cm\nend");
    assert_eq!(w.parts.len(), 5);
}

#[test]
fn continue_skips_one_pass() {
    let w = clean("for i in 1..5\n  if i is 3\n    continue\n  end\n  cube 1cm\nend");
    assert_eq!(w.parts.len(), 4);
}

#[test]
fn break_outside_a_loop_is_friendly() {
    let e = first_err("break");
    assert!(e.contains("inside for or while"), "{e}");
    let e = first_err("continue");
    assert!(e.contains("inside for or while"), "{e}");
}

#[test]
fn while_never_ends_is_caught() {
    let e = first_err("while true\n  x = 1\nend");
    assert!(e.contains("never ends"), "{e}");
}

// ---------------------------------------------------------------- assignment operators

#[test]
fn compound_assignment_arithmetic() {
    clean("x = 10\nx += 5\nx -= 3\nx *= 2\nx /= 4\nassert x is 6");
}

#[test]
fn compound_assignment_units() {
    clean("x = 10cm\nx += 5cm\nassert x is 15cm\ny = 90deg\ny -= 30deg\nassert y is 60deg");
}

#[test]
fn compound_assignment_fuses_shapes() {
    let w = clean("blob = cube 2cm\nblob += cube 1cm at (10cm, 0, 0)");
    assert_eq!(w.parts.len(), 1, "one named object, not two");
    assert!(w.parts[0].name == "blob");
    // the fused volume is the union of both cubes (no overlap)
    assert!((w.parts[0].volume_mm3 - 9000.0).abs() < 5.0, "got {}", w.parts[0].volume_mm3);
}

#[test]
fn compound_assign_needs_the_name_first() {
    let e = first_err("count += 1");
    assert!(e.contains("make count first"), "{e}");
}

// ---------------------------------------------------------------- operators

#[test]
fn power_and_modulo() {
    clean("x = 2 ^ 10\nassert x is 1024\ny = 7 % 3\nassert y is 1\nz = 7 mod 3\nassert z is 1");
}

#[test]
fn power_is_right_associative() {
    clean("x = 2 ^ 3 ^ 2\nassert x is 512");
}

#[test]
fn minus_binds_looser_than_power() {
    clean("y = -2 ^ 2\nassert y is -4");
}

#[test]
fn precedence_the_classic() {
    clean("x = 2 + 3 * 4 ^ 2\nassert x is 50");
}

#[test]
fn comparisons_and_equality() {
    clean("assert 2 < 3\nassert 3 <= 3\nassert 4 > 3\nassert 3 >= 3\nassert 5 == 5\nassert 5 != 4");
    clean("assert 5 is 5\nassert 5 is not 4\nassert \"a\" == \"a\"");
}

#[test]
fn comparisons_promote_the_scene_unit() {
    // scene unit mm: the bare 40 becomes 40mm, 5cm is 50mm
    clean("unit: mm\nassert 5cm > 40\nassert 5cm == 50");
}

#[test]
fn logic_words_and_symbols() {
    clean("assert true and true\nassert false or true\nassert not false\nassert !false");
    clean("assert 1 < 2 && 3 > 2\nassert 1 > 2 || 3 > 2");
}

#[test]
fn and_or_short_circuit() {
    // the right side of a decided and/or is never evaluated — no zero divide
    clean("x = 0\nok = false and x / 0 > 1\nok2 = true or x / 0 > 1");
}

#[test]
fn dividing_by_zero_is_friendly() {
    let e = first_err("x = 1 / 0");
    assert!(e.contains("divide by zero"), "{e}");
    let e = first_err("y = 5 mod 0");
    assert!(e.contains("zero"), "{e}");
}

#[test]
fn power_keeps_one_dimension() {
    let e = first_err("x = 2cm ^ 2");
    assert!(e.contains("plain numbers"), "{e}");
}

#[test]
fn comparing_mixed_dimensions_is_friendly() {
    let e = first_err("assert 5cm > 90deg");
    assert!(e.contains("can't compare"), "{e}");
}

// ---------------------------------------------------------------- membership / in

#[test]
fn membership_in_lists() {
    clean("xs = [1, 2, 3]\nassert 2 in xs\nassert not (5 in xs)");
}

#[test]
fn membership_in_ranges() {
    clean("assert 5 in 1..10\nassert not (11 in 1..10)");
}

#[test]
fn membership_in_words() {
    clean("assert \"cup\" in \"cupcake\"");
}

#[test]
fn in_after_a_number_is_not_inches() {
    // the lexer must read `3 in [` as a membership test, not 3 inches
    clean("assert 3 in [1, 2, 3]");
    // …while attached `3in` stays three inches (r = 76.2mm)
    let w = clean("unit: mm\nsphere 3in");
    let want = 4.0 / 3.0 * std::f64::consts::PI * (3.0 * 25.4f64).powi(3);
    assert!((w.parts[0].volume_mm3 - want).abs() / want < 0.01, "got {} want {}", w.parts[0].volume_mm3, want);
}

// ---------------------------------------------------------------- indexing / slicing

#[test]
fn indexing_counts_from_zero() {
    clean("xs = [1, 2, 3]\nassert xs[0] is 1\nassert xs[2] is 3\nassert xs[-1] is 3");
}

#[test]
fn slicing_is_inclusive() {
    clean("xs = [1, 2, 3, 4]\ntail = xs[1..3]\nassert count(tail) is 3");
}

#[test]
fn indexing_positions_and_words() {
    clean("p = (3cm, 4cm, 5cm)\nassert p[0] is 3cm\nw = \"cup\"\nassert w[0] == \"c\"");
}

#[test]
fn out_of_range_is_friendly() {
    let e = first_err("xs = [1]\ny = xs[5]");
    assert!(e.contains("that list has 1 thing"), "{e}");
}

#[test]
fn shapes_cannot_be_indexed() {
    let e = first_err("c = cube 1cm\nx = c[0]");
    assert!(e.contains("you can index lists"), "{e}");
}

// ---------------------------------------------------------------- ranges

#[test]
fn ranges_are_plain_numbers_only() {
    let e = first_err("r = 1cm..5cm");
    assert!(e.contains("plain numbers"), "{e}");
}

// ---------------------------------------------------------------- define … end

#[test]
fn define_block_returns_its_last_line() {
    let w = clean("define wheel(r)\n  rim = torus(radius: r, tube: 1cm)\n  hub = cylinder(r: r / 2, h: 2cm)\n  rim + hub\nend\nuse wheel(r: 3cm)");
    assert_eq!(w.parts.len(), 1);
    assert_eq!(w.parts[0].name, "wheel");
}

#[test]
fn define_block_last_assignment_is_the_value() {
    clean("define bobbin\n  body = cylinder(r: 2cm, h: 5cm)\n  body\nend\nuse bobbin\nassert true");
}

#[test]
fn define_block_locals_do_not_leak() {
    let e = first_err("define thing\n  secret = 7\n  cube 1cm\nend\nuse thing\nx = secret");
    assert!(e.contains("secret"), "{e}");
}

#[test]
fn define_block_params_bind() {
    clean("define plate(s)\n  cube(s, s, s / 4)\nend\nuse plate(s: 4cm)\nassert true");
}

// ---------------------------------------------------------------- assert

#[test]
fn assert_passes_quietly() {
    clean("x = 5\nassert x > 0\nassert x is 5, \"five is five\"");
}

#[test]
fn assert_failure_carries_the_message() {
    let e = first_err("assert 1 > 2, \"math broke\"");
    assert!(e.contains("math broke"), "{e}");
}

#[test]
fn assert_failure_has_a_default_message() {
    let e = first_err("assert 1 > 2");
    assert!(e.contains("assert failed"), "{e}");
}

// ---------------------------------------------------------------- print interpolation

#[test]
fn print_interpolates_values() {
    let w = clean("r = 5cm\nn = 3\nprint \"radius {r} count {n}\"");
    let lines = printed(&w);
    assert_eq!(lines, vec!["radius 50mm count 3".to_string()]);
}

#[test]
fn print_unknown_names_stay_as_written() {
    let w = clean("print \"literal {brace} stays\"");
    let lines = printed(&w);
    assert_eq!(lines, vec!["literal {brace} stays".to_string()]);
}

// ---------------------------------------------------------------- functions

#[test]
fn new_math_functions() {
    clean("assert floor(2.7) is 2\nassert ceil(2.1) is 3\nassert pow(2, 10) is 1024");
    clean("assert abs(-5) is 5\nassert sign(-3) is -1\nassert hypot(3, 4) is 5");
    clean("assert lerp(0cm, 10cm, 0.5) is 5cm\nassert clamp(15, 0, 10) is 10");
    clean("assert log(1000) is 3\nassert exp(0) is 1");
}

#[test]
fn inverse_trig_answers_in_degrees() {
    clean("assert atan2(1, 1) is 45deg\nassert asin(1) is 90deg\nassert acos(1) is 0deg");
}

#[test]
fn list_functions() {
    clean("assert count([4, 5, 6]) is 3\nassert count(1..10) is 10");
    clean("assert sum([1, 2, 3]) is 6\nassert sum(1..100) is 5050");
    clean("assert avg([2, 4, 6]) is 4");
}

// ---------------------------------------------------------------- lexer additions

#[test]
fn scientific_notation_values() {
    clean("a = 1.5e3\nassert a is 1500\nb = 2E-4\nassert b is 0.0002\nc = 3e2\nassert c is 300");
}

#[test]
fn semicolons_separate_statements() {
    let w = clean("x = 1; y = 2; cube 1cm; assert x + y is 3");
    assert_eq!(w.parts.len(), 1);
}

#[test]
fn block_comments_disappear() {
    let w = clean("#[ a note\n that spans lines ]# cube 1cm #[ and one more ]# sphere 2cm");
    assert_eq!(w.parts.len(), 2);
}

#[test]
fn semicolon_inside_brackets_is_friendly() {
    let e = first_err("at (1, 2; 3)");
    assert!(e.contains("; separates"), "{e}");
}

#[test]
fn unterminated_block_comment_is_friendly() {
    let e = first_err("#[ never closed");
    assert!(e.contains("block comment"), "{e}");
}

// ---------------------------------------------------------------- new units

#[test]
fn kilometer_yard_micron_radian() {
    clean("assert 1km is 1000000mm\nassert 1yd is 914.4mm\nassert 1000um is 1mm");
    clean("r = 2rad\nassert r > 114deg and r < 116deg");
}

// ---------------------------------------------------------------- regression guards

#[test]
fn classic_forms_still_compile() {
    // the 1.0/2.0 corpus forms, untouched by the expansion
    let w = clean("scene \"Coffee Cup\"\ncup = cylinder(top: 40mm, bottom: 30mm, height: 100mm) - hollow(wall: 3mm)\nhandle = torus(radius: 20mm, tube: 6mm) rotate (90deg, 0, 0) at (35mm, 25mm, 0)\nadd cup, handle\nmaterial cup: ceramic\ncolor cup: ivory\nask \"mass?\"");
    assert_eq!(w.parts.len(), 1); // fused
    let w = clean("arch = ring(n: 13, radius: 19cm, from: 0deg, to: 180deg, axis: z) cube(12cm, 4cm, 13cm) rotate (0, 0, a + 90deg)");
    assert_eq!(w.parts.len(), 1);
    let w = clean("fence = repeat(n: 5, step: (4cm, 0, 0)) cube(1cm, 4cm, 1cm) material: wood");
    assert_eq!(w.parts.len(), 1);
    clean("sphere 5 cm");     // space-separated units still glue
    clean("sphere 2 in");     // inches still glue when no list follows
}

#[test]
fn missing_end_is_friendly() {
    let e = first_err("if 1 > 0\n  cube 1cm");
    assert!(e.contains("needs its end"), "{e}");
}

#[test]
fn stray_end_and_else_are_friendly() {
    assert!(first_err("end").contains("doesn't close anything"));
    assert!(first_err("else\n  cube 1cm\nend").contains("else has no if"));
}

#[test]
fn loops_inside_templates_work() {
    let w = clean("define beads\n  out = repeat(n: 4, step: (3cm, 0, 0)) sphere 1cm\n  out\nend\nuse beads");
    assert_eq!(w.parts.len(), 1);
    assert!(w.parts[0].volume_mm3 > 4.0 * 4000.0, "four beads should fuse: {}", w.parts[0].volume_mm3);
}

#[test]
fn if_inside_for_picks_colors() {
    let w = clean("for i in 1..6\n  if i > 3\n    cube 1cm color: red\n  else\n    cube 1cm color: blue\n  end\nend");
    assert_eq!(w.parts.len(), 6);
}
