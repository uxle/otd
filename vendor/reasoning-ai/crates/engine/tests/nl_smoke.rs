//! Port of python/tests/nl_smoke.py — quick NL smoke test across all
//! domains: every question goes through ask() and must come back with the
//! right routed kind, a verified answer containing the expected substring.

use reasoning_engine::ask;

#[test]
fn nl_smoke_all_domains() {
    // (question, want_kind, want_answer_substring) — None means only the
    // kind and verified flag are checked.
    let questions: &[(&str, &str, Option<&str>)] = &[
        // physics
        ("A car accelerates from rest at 3 m/s^2 for 5 s. What is its final velocity?", "physics_kinematics", Some("15")),
        ("A ball is dropped from 20 m. What is its impact speed?", "physics_energy", Some("19.798")),
        ("A 2 kg block is pushed with a force of 10 N. What is the acceleration?", "physics_force", Some("5")),
        ("What is the weight of a 5 kg object?", "physics_force", Some("49")),
        ("A 4 kg box slides with coefficient of friction 0.3. What is the friction force?", "physics_force", Some("11.76")),
        ("How much kinetic energy does a 2 kg object moving at 3 m/s have?", "physics_energy", Some("9")),
        ("A 1000 kg car moves at 20 m/s. What is its momentum?", "physics_momentum", Some("20000")),
        ("A 2 kg cart moving at 3 m/s collides with a stationary 3 kg cart and they stick together. What is the final velocity?", "physics_momentum", Some("1.2")),
        ("What is the current through a 12 V circuit with 4 ohm resistance?", "physics_electricity", Some("3")),
        ("Three resistors 4 ohm, 6 ohm and 12 ohm are in parallel. What is the total resistance?", "physics_electricity", Some("2")),
        ("What is the density of a 5 kg object with volume 2 m^3?", "physics_density", Some("2.5")),
        ("A car travels 120 km in 2 hours. What is its average speed?", "physics_speed", Some("16.666")),
        // chemistry
        ("Balance the equation H2 + O2 -> H2O", "chem_balance", Some("2 H2 + O2 -> 2 H2O")),
        ("What is the molar mass of C6H12O6?", "chem_molar_mass", Some("180.18")),
        ("In N2 + H2 -> NH3, if 2 moles of N2 react with 3 moles of H2, how many moles of NH3 form?", "chem_limiting", Some("2")),
        ("What is the pH of a solution with [H+] = 1e-3?", "chem_ph", Some("3")),
        ("What is the volume of 1 mole of gas at 1 atm and 273 K?", "chem_gas", Some("22.4")),
        ("What is the molarity of 4 grams of NaOH in 0.5 L?", "chem_solutions", Some("0.2")),
        ("What is the percent composition of H2O?", "chem_solutions", Some("88.79")),
        // biology
        ("Cross Aa x Aa. What is the phenotype ratio?", "bio_monohybrid", Some("3/4")),
        ("Cross AaBb x AaBb. What is the phenotype ratio?", "bio_dihybrid", Some("9/16")),
        ("In a Hardy-Weinberg population, 16% shows the recessive trait. What is q?", "bio_hardy_weinberg", Some("0.4")),
        ("Transcribe the DNA sequence ATGCAT", "bio_dogma", Some("UACGUA")),
        ("Translate the sequence ATGGCCTAA", "bio_dogma", Some("MA")),
        ("A population of 100 bacteria grows at 10% per hour for 5 hours. What is the population?", "bio_ecology", Some("161")),
        // puzzles
        ("What day of the week was July 4, 1776?", "calendar_day", Some("Thursday")),
        ("What is the angle between the hands of a clock at 3:30?", "clock_angle", Some("75")),
        ("What is the mirror image of the word MATH?", "mirror_image", None),
        ("Alice is the father of Bob. Bob is the father of Carol. How is Alice related to Carol?", "family_tree", Some("grandfather")),
        ("A man walks north 3 km then east 4 km. How far is he from the start?", "direction_walk", Some("5")),
        // math
        ("What is 12 * 7 + 5?", "arithmetic", Some("89")),
        ("Solve 2x + 3 = 11", "linear_equation", Some("4")),
        ("Factor x^2 + 5x + 6", "quadratic_factoring", Some("(x+2)(x+3)")),
        ("What is 5 choose 2?", "combinatorics", Some("10")),
        ("What is gcd(240, 46)?", "gcd_bezout", Some("2")),
        ("What is sin(30)?", "trig_evaluate", Some("1/2")),
        ("What is the determinant of [[1,2],[3,4]]?", "matrix_determinant", Some("-2")),
        ("Five times a number plus 3 is 18.", "word_problem", Some("3")),
    ];

    let mut passed = 0usize;
    let mut failures: Vec<String> = Vec::new();
    for &(q, want_kind, want_ans) in questions {
        let r = ask(q, None, 0);
        let got_kind = r["parsed"]["kind"].as_str().unwrap_or("NONE");
        let answer = r["answer"].as_str().unwrap_or_default();
        let ok_kind = got_kind == want_kind;
        let ok_ans = want_ans.map(|w| !answer.is_empty() && answer.contains(w)).unwrap_or(true);
        if ok_kind && ok_ans && r["verified"].as_bool().unwrap_or(false) {
            passed += 1;
            println!("PASS [{:<20}] {:<60} -> {}", got_kind, &q[..q.len().min(60)], answer);
        } else {
            let mut msg = format!(
                "FAIL [{:<20}] {}",
                if got_kind.is_empty() { "NONE" } else { got_kind },
                &q[..q.len().min(60)]
            );
            msg.push_str(&format!("\n     want kind={} ans~{:?}; got {:?}", want_kind, want_ans, r["answer"]));
            msg.push_str(&format!(
                "\n     parse: {}",
                r["parsed"]["explain"].as_str().unwrap_or_default().chars().take(140).collect::<String>()
            ));
            failures.push(msg);
        }
    }
    println!("\n{} passed, {} failed of {}", passed, failures.len(), questions.len());
    assert!(
        failures.is_empty(),
        "{} smoke question(s) failed:\n{}",
        failures.len(),
        failures.join("\n")
    );
}
