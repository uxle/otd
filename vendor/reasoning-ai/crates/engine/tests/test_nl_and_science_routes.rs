//! Port of python/tests/test_nl_and_science_routes.py — Phase 110-123
//! integration tests: the expanded science + puzzle domains through the
//! unified solve() API, plus the natural-language layer via ask().
//!
//! Only the solve()/ask()-level classes are ported here (per the task
//! split): TestUnifiedScienceRoutes and TestNLAnswers. The parse-level
//! classes of the Python file (TestQuantityExtraction, TestNLRouter) are
//! covered by the nlp crate's tests/smoke_parse.rs.

use reasoning_engine::solve::{solve, Problem};
use reasoning_engine::ask;
use serde_json::{json, Value};

fn p(kind: &str, payload: Value) -> Problem {
    Problem::new(kind, serde_json::from_value(payload).unwrap())
}

// ---------------- TestUnifiedScienceRoutes ----------------

#[test]
fn test_all_new_kinds_solve_and_verify() {
    let cases: Vec<(Problem, &str)> = vec![
        (p("physics_kinematics", json!({"find": "v", "u": 0, "a": 3, "t": 5})), "15"),
        (p("physics_speed", json!({"find": "speed", "distance": 120, "time": 2})), "60"),
        (p("physics_force", json!({"kind": "weight", "mass": 5})), "49"),
        (p("physics_force", json!({"kind": "friction", "find": "friction", "mass": 4, "mu": 0.3})), "11.76"),
        (p("physics_force", json!({"find": "acceleration", "force": 10, "mass": 2})), "5"),
        (p("physics_energy", json!({"form": "kinetic", "mass": 2, "velocity": 3})), "9"),
        (p("physics_energy", json!({"form": "impact_speed", "height": 20})), "19.79"),
        (p("physics_momentum", json!({"kind": "momentum", "mass": 1000, "velocity": 20})), "20000"),
        (p("physics_momentum", json!({"kind": "elastic", "m1": 2, "v1": 3, "m2": 2, "v2": -1})), "-1"),
        (p("physics_electricity", json!({"kind": "ohm", "find": "current", "voltage": 12, "resistance": 4})), "3"),
        (p("physics_electricity", json!({"kind": "parallel", "resistances": [4, 6, 12], "voltage": 12})), "2"),
        (p("physics_density", json!({"kind": "density", "mass": 5, "volume": 2})), "2.5"),
        (p("physics_density", json!({"kind": "pressure", "rho": 1000, "height": 10, "area": 2})), "98000"),
        (p("chem_molar_mass", json!({"formula": "C6H12O6"})), "180.18"),
        (p("chem_balance", json!({"equation": "C3H8 + O2 -> CO2 + H2O"})), "5"),
        (
            p("chem_limiting", json!({"equation": "N2 + H2 -> NH3", "moles": {"N2": 2.0, "H2": 3.0}, "product": "NH3"})),
            "H2",
        ),
        (p("chem_gas", json!({"kind": "ideal", "find": "volume", "pressure": 1, "moles": 1, "temperature": 273})), "22.4"),
        (p("chem_solutions", json!({"kind": "percent_yield", "actual": 8, "theoretical": 10})), "80"),
        (p("chem_ph", json!({"kind": "from_concentration", "h": 1e-3})), "3"),
        (p("bio_monohybrid", json!({"parent_a": "Aa", "parent_b": "Aa"})), "3/4"),
        (p("bio_dihybrid", json!({"parent_a": "AaBb", "parent_b": "AaBb"})), "9/16"),
        (p("bio_hardy_weinberg", json!({"q_squared": 0.16})), "0.4"),
        (p("bio_dogma", json!({"kind": "translate", "seq": "ATGGCCTAA"})), "MA"),
        (p("bio_ecology", json!({"kind": "energy_transfer", "producer_energy": 10000, "trophic_level": 3})), "100"),
        (
            p("family_tree", json!({"facts": [["A", "father", "B"], ["B", "father", "C"]], "who": "A", "whom": "C"})),
            "grandfather",
        ),
        (p("clock_angle", json!({"hour": 3, "minute": 30})), "75"),
        (p("calendar_day", json!({"year": 1776, "month": 7, "day": 4})), "Thursday"),
        (p("direction_walk", json!({"moves": [["N", 3000], ["E", 4000]]})), "5000"),
    ];
    for (problem, expected) in &cases {
        let ans = solve(problem, None, 0);
        assert!(
            ans.verified,
            "{} should verify: {}",
            problem.kind,
            ans.explanation
        );
        let answer_expr = ans.answer_expr.as_deref().unwrap_or_default();
        assert!(
            answer_expr.contains(expected),
            "{}: answer {:?} should contain {:?}",
            problem.kind,
            answer_expr,
            expected
        );
    }
}

#[test]
fn test_math_kinds_still_route_first() {
    let ans = solve(&p("linear_equation", json!({"equation": "2*x + 3 = 11"})), None, 0);
    assert!(ans.verified);
    assert!(ans.answer_expr.as_deref().unwrap_or_default().contains("4"));
}

#[test]
fn test_abstention_not_guess() {
    let ans = solve(&p("physics_kinematics", json!({"find": "v", "u": 0})), None, 0); // insufficient
    assert!(!ans.verified);
    assert!(ans.final_answer().is_none());
}

// ---------------- TestNLAnswers ----------------
// End-to-end: question -> verified, hand-checkable answer.

fn ask_check(text: &str, kind: &str, answer_sub: &str) {
    let r = ask(text, None, 0);
    assert!(
        r["verified"].as_bool().unwrap_or(false),
        "should verify: {:?} -> {:?}",
        text,
        r["explanation"].as_str()
    );
    assert_eq!(r["parsed"]["kind"].as_str(), Some(kind), "text: {:?}", text);
    let answer = r["answer"].as_str().unwrap_or_default();
    assert!(
        answer.contains(answer_sub),
        "answer {:?} should contain {:?} ({:?})",
        answer,
        answer_sub,
        text
    );
}

#[test]
fn test_kinematics() {
    // v = 0 + 3*5 = 15 m/s
    ask_check(
        "A car accelerates from rest at 3 m/s^2 for 5 s. What is its final velocity?",
        "physics_kinematics",
        "15",
    );
}

#[test]
fn test_free_fall_via_energy_conservation() {
    // sqrt(2*9.8*20) = 19.799 m/s, cross-checked against kinematics
    ask_check(
        "A ball is dropped from 20 m. What is its impact speed?",
        "physics_energy",
        "19.79899",
    );
}

#[test]
fn test_friction() {
    // 0.3 * 4 * 9.8 = 11.76 N
    ask_check(
        "A 4 kg box slides with coefficient of friction 0.3. What is the friction force?",
        "physics_force",
        "11.76",
    );
}

#[test]
fn test_inelastic_collision() {
    // (2*3 + 3*0)/5 = 1.2 m/s
    ask_check(
        "A 2 kg cart moving at 3 m/s collides with a stationary 3 kg cart and they stick together. What is the final velocity?",
        "physics_momentum",
        "1.2",
    );
}

#[test]
fn test_limiting_reagent() {
    // H2 limiting (extents 2 vs 1) -> 2 mol NH3
    ask_check(
        "In N2 + H2 -> NH3, if 2 moles of N2 react with 3 moles of H2, how many moles of NH3 form?",
        "chem_limiting",
        "2.0 mol",
    );
}

#[test]
fn test_gas_law() {
    ask_check(
        "What is the volume of 1 mole of gas at 1 atm and 273 K?",
        "chem_gas",
        "22.4",
    );
}

#[test]
fn test_molarity_from_mass() {
    // 4 g NaOH / 40 g/mol / 0.5 L = 0.2 M
    ask_check(
        "What is the molarity of 4 grams of NaOH in 0.5 L?",
        "chem_solutions",
        "0.2",
    );
}

#[test]
fn test_ecology() {
    // 100 * 1.1^5 = 161.05
    ask_check(
        "A population of 100 bacteria grows at 10% per hour for 5 hours. What is the population?",
        "bio_ecology",
        "161.05",
    );
}

#[test]
fn test_directions() {
    // 3-4-5 triangle (km -> m canonicalized)
    ask_check(
        "A man walks north 3 km then east 4 km. How far is he from the start?",
        "direction_walk",
        "5000",
    );
}

#[test]
fn test_trig_degrees() {
    ask_check("What is sin(30)?", "trig_evaluate", "1/2");
}

#[test]
fn test_word_problem_digitized() {
    ask_check("Five times a number plus 3 is 18.", "word_problem", "3");
}

#[test]
fn test_mirror_clock() {
    ask_check(
        "What does a digital clock showing 4:40 look like in a mirror?",
        "mirror_image",
        "07:20",
    );
}

#[test]
fn test_underdetermined_kinematics_abstains() {
    // two unknowns and no stated target -> must NOT guess
    let r = ask("An object moves at 5 m/s for a while. What happens?", None, 0);
    assert!(!r["verified"].as_bool().unwrap_or(false));
}
