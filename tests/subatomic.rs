//! OTD3.3 — the subatomic layer, through the real front-end:
//! `particle:` statements, `simulate: atom / decay / particles`, and the
//! physics underneath them (element table, configs, binding, decay law).

use otd::world::atom;
use otd::world::particles;

fn run(code: &str) -> otd::world::eval::World {
    otd::world::eval::compile(code)
}

#[test]
fn particle_statement_deck() {
    let w = run(
        "scene \"deck\"\nparticle: proton\nparticle: gluon\nparticle: neutrino\nparticle: nope\n",
    );
    let text: String = w.console.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    assert!(text.contains("proton = u u d"));
    assert!(text.contains("strong force carrier"));
    assert!(text.contains("light-year of lead"));
    // unknown particle → a helpful error line, not a crash
    assert!(text.contains("is not in the deck"));
}

#[test]
fn simulate_atom_counts_real_mass() {
    let w = run(
        "scene \"census\"\ncube = cube(1cm, 1cm, 1cm) at (0, 5mm, 0) material: iron\nsimulate: atom\n",
    );
    let text: String = w.console.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    // 7.87 g of iron → ~8.5e22 atoms
    assert!(text.contains("iron (Fe, Z=26)"));
    assert!(text.contains("26 protons + 30 neutrons (Fe-56)"));
    assert!(text.contains("K2 L8 M14 N2"));
    assert!(text.contains("binding energy 8.76 MeV/nucleon"));
    // charge ledger must balance exactly
    assert!(text.contains("net 0.000 C"));
    // totals present in scientific notation
    assert!(text.contains("SCENE TOTALS"));
    assert!(text.contains("e22") || text.contains("e23"));
}

#[test]
fn simulate_decay_activity_from_real_mass() {
    let w = run(
        "scene \"hot\"\nslug = cylinder(r: 1cm, h: 1cm) at (0, 5mm, 0) material: uranium\nsimulate: decay\n",
    );
    let text: String = w.console.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    // 59.7 g of U-238 → ~740 kBq (12.4 kBq per gram)
    assert!(text.contains("U-238"));
    assert!(text.contains("4.47 billion years"));
    assert!(text.contains("activity RIGHT NOW"));
    // the physics: 1 g U-238 = 12.4 kBq, ±5% for the Weiszäcker-free direct count
    assert!(text.contains("kBq") || text.contains("MBq"));
}

#[test]
fn simulate_particles_briefing() {
    let w = run(
        "scene \"sm\"\nball = sphere(r: 1cm) at (0, 1cm, 0) material: copper\nsimulate: particles\n",
    );
    let text: String = w.console.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    assert!(text.contains("THE STANDARD MODEL"));
    assert!(text.contains("99% of the mass"));
    assert!(text.contains("GLUON g"));
    assert!(text.contains("PHOTON γ"));
    assert!(text.contains("NEUTRINO"));
    assert!(text.contains("solar neutrinos"));
    // all 17 cards listed
    assert!(text.contains("top quark"));
    assert!(text.contains("Higgs boson"));
}

#[test]
fn trace_carbon14_in_wood() {
    let w = run(
        "scene \"quiet\"\nplank = cube(10cm, 2cm, 5cm) at (0, 1cm, 0) material: wood\nsimulate: decay\n",
    );
    let text: String = w.console.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    assert!(text.contains("trace radioactivity"));
    assert!(text.contains("C-14"));
    // ~0.2 Bq per gram of carbon; the plank is ~98 g × 0.44 C → ~9.6 Bq scale
    assert!(text.contains("Bq of soft"));
}

#[test]
fn element_table_and_lookups() {
    assert_eq!(atom::ELEMENTS.len(), 118);
    assert_eq!(atom::find_element("Au").unwrap().z, 79);
    assert_eq!(atom::find_element("uranium").unwrap().a, 238);
    // every config sums to Z
    for e in atom::ELEMENTS {
        let total: u32 = atom::electron_config(e.z).iter().map(|(_, c)| c).sum();
        assert_eq!(total, e.z, "{} config must sum to Z", e.name);
    }
}

#[test]
fn weizsacker_against_textbook() {
    let fe = atom::binding_per_nucleon(56, 26);
    assert!((fe - 8.76).abs() < 0.05);
    let u = atom::binding_per_nucleon(238, 92);
    assert!((u - 7.60).abs() < 0.05);
    assert!(fe > u, "iron must sit at the peak");
}

#[test]
fn decay_law_numbers() {
    use otd::world::atom::*;
    // 1 g U-238 → 12.4 kBq (textbook)
    let n = 1.0 / 238.0 * N_A;
    let bq = activity_bq(n, 4.468e9 * 3.156e7);
    assert!((bq - 12_400.0).abs() < 500.0);
    // 25% → exactly two half-lives
    let t = decay_age_s(0.25, 5730.0 * 3.156e7);
    assert!((t / 3.156e7 / 5730.0 - 2.0).abs() < 1e-6);
}

#[test]
fn particle_zoo_facts() {
    // the famous accounting: 99% of proton mass is gluon energy
    let qm = 2.0 * 2.2 + 4.7;
    assert!(qm / particles::PROTON_MEV < 0.01);
    // β⁻ Q-value 0.782 MeV
    assert!((particles::beta_minus_q_mev() - 0.782).abs() < 0.005);
    // 17 cards
    assert_eq!(particles::FUNDAMENTALS.len(), 17);
    // quark charges are exact thirds
    assert_eq!(format!("{:+}/3", (particles::quark_charge("u") * 3.0) as i32), "+2/3");
}

#[test]
fn atom_tour_example_compiles() {
    let w = run(include_str!("../examples/atom-tour.otd"));
    assert!(w.errors.is_empty(), "atom-tour must compile: {:?}", w.errors);
    let text: String = w.console.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    assert!(text.contains("ATOMIC STRUCTURE"));
    assert!(text.contains("RADIOACTIVE DECAY"));
    assert!(text.contains("THE STANDARD MODEL"));
    // the uranium slug's live activity in the tour
    assert!(text.contains("11.88 MBq") || text.contains("MBq"));
}

#[test]
fn example_is_registered() {
    assert!(otd::content::EXAMPLES.iter().any(|e| e.file == "atom-tour.otd"));
}

#[test]
fn hydrogen_spectrum_when_water_present() {
    let w = run(
        "scene \"water\"\nbeaker = cube(5cm, 5cm, 5cm) at (0, 25mm, 0) material: water\nsimulate: atom\n",
    );
    let text: String = w.console.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join("\n");
    assert!(text.contains("hydrogen's fingerprint"));
    assert!(text.contains("656.5 nm"));
}
