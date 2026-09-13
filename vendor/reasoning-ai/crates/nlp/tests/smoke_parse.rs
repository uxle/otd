use reasoning_nlp::parse;

#[test]
fn smoke_math_routes() {
    let p = parse("What is 12 * 7 + 5?");
    assert!(p.ok);
    assert_eq!(p.kind.as_deref(), Some("arithmetic"));
    assert_eq!(
        p.payload.as_ref().unwrap().get("expr").unwrap().as_str().unwrap(),
        "12 * 7 + 5"
    );

    let p = parse("Solve 2x + 3 = 11");
    assert!(p.ok, "rationale: {}", p.rationale);
    assert_eq!(p.kind.as_deref(), Some("linear_equation"));
    assert_eq!(
        p.payload.as_ref().unwrap().get("equation").unwrap().as_str().unwrap(),
        "2*x+3=11"
    );

    let p = parse("Factor x^2 + 5x + 6");
    assert!(p.ok, "rationale: {}", p.rationale);
    assert_eq!(p.kind.as_deref(), Some("quadratic_factoring"));

    let p = parse("What is 5 choose 2?");
    assert!(p.ok, "rationale: {}", p.rationale);
    assert_eq!(p.kind.as_deref(), Some("combinatorics"));

    let p = parse("What is gcd(240, 46)?");
    assert!(p.ok, "rationale: {}", p.rationale);
    assert_eq!(p.kind.as_deref(), Some("gcd_bezout"));
}

#[test]
fn smoke_science_routes() {
    let p = parse("A car accelerates from rest at 3 m/s^2 for 5 s. What is its final velocity?");
    assert!(p.ok, "rationale: {}", p.rationale);
    assert_eq!(p.kind.as_deref(), Some("physics_kinematics"));
    let payload = p.payload.as_ref().unwrap();
    assert_eq!(payload.get("a").unwrap().as_f64().unwrap(), 3.0);
    assert_eq!(payload.get("t").unwrap().as_f64().unwrap(), 5.0);
    assert_eq!(payload.get("u").unwrap().as_f64().unwrap(), 0.0);
    assert_eq!(payload.get("find").unwrap().as_str().unwrap(), "v");

    let p = parse("What is the molar mass of C6H12O6?");
    assert!(p.ok, "rationale: {}", p.rationale);
    assert_eq!(p.kind.as_deref(), Some("chem_molar_mass"));

    let p = parse("Balance the equation H2 + O2 -> H2O");
    assert!(p.ok, "rationale: {}", p.rationale);
    assert_eq!(p.kind.as_deref(), Some("chem_balance"));
    assert_eq!(
        p.payload.as_ref().unwrap().get("equation").unwrap().as_str().unwrap(),
        "H2 + O2 -> H2O"
    );

    let p = parse("What is the pH of a solution with [H+] = 1e-3?");
    assert!(p.ok, "rationale: {}", p.rationale);
    assert_eq!(p.kind.as_deref(), Some("chem_ph"));

    let p = parse("Cross Aa x Aa. What is the phenotype ratio?");
    assert!(p.ok, "rationale: {}", p.rationale);
    assert_eq!(p.kind.as_deref(), Some("bio_monohybrid"));

    let p = parse("What day of the week was July 4, 1776?");
    assert!(p.ok, "rationale: {}", p.rationale);
    assert_eq!(p.kind.as_deref(), Some("calendar_day"));
    let payload = p.payload.as_ref().unwrap();
    assert_eq!(payload.get("year").unwrap().as_i64().unwrap(), 1776);
    assert_eq!(payload.get("month").unwrap().as_u64().unwrap(), 7);
    assert_eq!(payload.get("day").unwrap().as_u64().unwrap(), 4);
}

#[test]
fn smoke_quantities_and_abstain() {
    let p = parse("What is the meaning of life?");
    assert!(!p.ok);
    assert!(!p.candidates.is_empty() || p.rationale.contains("no domain pattern"));

    // quantity canonicalization: "for 5 minutes" -> t = 300 s
    let p = parse("A car accelerates from rest at 2 m/s^2 for 5 minutes. What is its final velocity?");
    assert!(p.ok);
    let payload = p.payload.as_ref().unwrap();
    assert_eq!(payload.get("t").unwrap().as_f64().unwrap(), 300.0);
}
