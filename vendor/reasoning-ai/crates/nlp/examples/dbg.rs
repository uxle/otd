fn main() {
    let p = reasoning_nlp::parse("What is 12 * 7 + 5?");
    println!("arithmetic: ok={} kind={:?} rationale={}", p.ok, p.kind, p.rationale);
    let p = reasoning_nlp::parse("What is the pH of a solution with [H+] = 1e-3?");
    println!("ph: ok={} kind={:?} rationale={}", p.ok, p.kind, p.rationale);
    let qs = reasoning_nlp::extract_quantities("solution with [H+] = 1e-3");
    for q in qs { println!("q: {:?}", q); }
}
