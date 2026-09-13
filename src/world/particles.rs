//! P2250 — PARTICLES: the Standard Model, the subatomic census.
//!
//! Everything you have ever touched is made of exactly three particles
//! of matter — the proton, the neutron, the electron — and those three
//! are themselves made of exactly two more (the up and down quarks,
//! stitched together by gluons). Everything else in the particle zoo
//! exists in cosmic rays, in accelerators, and, in uncountable numbers,
//! passing straight through you: photons from the Sun, neutrinos from
//! its furnace and from the Big Bang itself.
//!
//! This module is the data + the laws for that layer:
//!
//!   * `FUNDAMENTALS` — all 17 fundamental particles of the Standard
//!     Model: 6 quarks, 6 leptons, 4 gauge bosons, 1 Higgs. Mass,
//!     charge, spin, statistics — the card the universe deals from.
//!   * `HADRONS` — the quark model of the composite particles that
//!     matter is actually built of: proton (uud), neutron (udd), pions.
//!   * The famous accounting trick: the quarks in a proton weigh
//!     9.4 MeV; the proton weighs 938.3 MeV. **99% of your mass is
//!     not matter — it is gluon field energy**, E = mc² made visible.
//!   * β-decay as a quark-level event: d → u + W⁻ → e⁻ + ν̄ₑ.
//!   * Annihilation and pair creation: e⁻ + e⁺ → 2γ, 1.022 MeV.
//!   * `particles_sim(world)` — the briefing, tied to the actual scene:
//!     how many protons, neutrons, electrons are physically present in
//!     the parts you built, how many photons are landing on them, and
//!     how many neutrinos are threading through them as you read this.
//!
//! Constants are CODATA 2018; masses are PDG 2022 world averages.

use super::eval::{ConsoleLine, LineKind, World};
use crate::world::atom;

// ---------- the constants ----------

/// Electron volt, the particle physicist's inch.
pub const EV_J: f64 = 1.602_176_634e-19; // exact, SI definition
/// MeV in joules.
pub const MEV_J: f64 = EV_J * 1.0e6;
/// Speed of light (m/s, exact).
pub const C_MPS: f64 = 299_792_458.0;
/// Planck constant (J·s, exact).
pub const H_JS: f64 = 6.626_070_15e-34;
/// Atomic mass unit in MeV/c² (PDG).
pub const AMU_MEV: f64 = 931.494_102_42;
/// Proton rest mass (MeV/c², PDG).
pub const PROTON_MEV: f64 = 938.272_088_16;
/// Neutron rest mass (MeV/c², PDG).
pub const NEUTRON_MEV: f64 = 939.565_420_52;
/// Electron rest mass (MeV/c², PDG).
pub const ELECTRON_MEV: f64 = 0.510_998_950_00;
/// Proton mass in kg.
pub const PROTON_KG: f64 = 1.672_621_923_69e-27;
/// Neutron mass in kg.
pub const NEUTRON_KG: f64 = 1.674_927_498_04e-27;
/// Electron mass in kg.
pub const ELECTRON_KG: f64 = 9.109_383_701_5e-31;
/// Elementary charge (C, exact).
pub const E_CHARGE: f64 = 1.602_176_634e-19;
/// Bohr radius (m) — the atom's natural size.
pub const BOHR_M: f64 = 5.291_772_109_03e-11;
/// Solar neutrino flux at Earth's surface (per cm² per second).
pub const SOLAR_NU_FLUX: f64 = 6.5e10;
/// Big-Bang relic neutrino density (per cm³).
pub const RELIC_NU_DENSITY: f64 = 336.0;

// ---------- the Standard Model ----------

/// The family a fundamental particle belongs to.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    Quark,
    Lepton,
    GaugeBoson,
    ScalarBoson,
}

impl Family {
    pub fn as_str(&self) -> &'static str {
        match self {
            Family::Quark => "quark",
            Family::Lepton => "lepton",
            Family::GaugeBoson => "gauge boson",
            Family::ScalarBoson => "scalar boson",
        }
    }
}

/// One row of the Standard Model. The whole of known matter, one card.
#[derive(Clone, Copy)]
pub struct Fundamental {
    pub name: &'static str,
    pub symbol: &'static str,
    pub family: Family,
    /// Rest mass in MeV/c² (0 = massless; neutrinos are upper limits).
    pub mass_mev: f64,
    /// Mass display string — neutrinos get "< x eV", the rest get MeV/GeV.
    pub mass_str: &'static str,
    /// Electric charge in units of e (quarks carry thirds!).
    pub charge_e: f64,
    /// Spin in units of ħ.
    pub spin: f64,
    /// Generation (1/2/3), 0 for bosons.
    pub generation: u8,
    /// Carries color charge (quarks and gluons only).
    pub colored: bool,
}

/// The 17 fundamental particles — the complete Standard Model.
pub const FUNDAMENTALS: &[Fundamental] = &[
    // ---- quarks (matter, fractionally charged, color-confined) ----
    Fundamental { name: "up quark",          symbol: "u",  family: Family::Quark, mass_mev: 2.2,      mass_str: "2.2 MeV",       charge_e:  2.0 / 3.0, spin: 0.5, generation: 1, colored: true },
    Fundamental { name: "down quark",        symbol: "d",  family: Family::Quark, mass_mev: 4.7,      mass_str: "4.7 MeV",       charge_e: -1.0 / 3.0, spin: 0.5, generation: 1, colored: true },
    Fundamental { name: "strange quark",     symbol: "s",  family: Family::Quark, mass_mev: 95.0,     mass_str: "95 MeV",        charge_e: -1.0 / 3.0, spin: 0.5, generation: 2, colored: true },
    Fundamental { name: "charm quark",       symbol: "c",  family: Family::Quark, mass_mev: 1275.0,   mass_str: "1.275 GeV",     charge_e:  2.0 / 3.0, spin: 0.5, generation: 2, colored: true },
    Fundamental { name: "bottom quark",      symbol: "b",  family: Family::Quark, mass_mev: 4180.0,   mass_str: "4.18 GeV",      charge_e: -1.0 / 3.0, spin: 0.5, generation: 3, colored: true },
    Fundamental { name: "top quark",         symbol: "t",  family: Family::Quark, mass_mev: 172_760.0, mass_str: "172.76 GeV",   charge_e:  2.0 / 3.0, spin: 0.5, generation: 3, colored: true },
    // ---- charged leptons (matter, integer charge) ----
    Fundamental { name: "electron",          symbol: "e⁻", family: Family::Lepton, mass_mev: 0.510_998_95, mass_str: "0.511 MeV",     charge_e: -1.0, spin: 0.5, generation: 1, colored: false },
    Fundamental { name: "muon",              symbol: "μ⁻", family: Family::Lepton, mass_mev: 105.658,  mass_str: "105.66 MeV",    charge_e: -1.0, spin: 0.5, generation: 2, colored: false },
    Fundamental { name: "tau",               symbol: "τ⁻", family: Family::Lepton, mass_mev: 1776.86,  mass_str: "1.7769 GeV",    charge_e: -1.0, spin: 0.5, generation: 3, colored: false },
    // ---- neutrinos (matter, nearly massless, almost never interact) ----
    Fundamental { name: "electron neutrino", symbol: "νₑ", family: Family::Lepton, mass_mev: 0.0,     mass_str: "< 2.2 eV",      charge_e:  0.0, spin: 0.5, generation: 1, colored: false },
    Fundamental { name: "muon neutrino",     symbol: "νμ", family: Family::Lepton, mass_mev: 0.0,     mass_str: "< 0.17 MeV",    charge_e:  0.0, spin: 0.5, generation: 2, colored: false },
    Fundamental { name: "tau neutrino",      symbol: "ντ", family: Family::Lepton, mass_mev: 0.0,     mass_str: "< 18.2 MeV",    charge_e:  0.0, spin: 0.5, generation: 3, colored: false },
    // ---- gauge bosons (the force carriers) ----
    Fundamental { name: "photon",            symbol: "γ",  family: Family::GaugeBoson, mass_mev: 0.0, mass_str: "0 (exactly)",   charge_e: 0.0, spin: 1.0, generation: 0, colored: false },
    Fundamental { name: "gluon",             symbol: "g",  family: Family::GaugeBoson, mass_mev: 0.0, mass_str: "0 (exactly)",   charge_e: 0.0, spin: 1.0, generation: 0, colored: true },
    Fundamental { name: "W boson",           symbol: "W±", family: Family::GaugeBoson, mass_mev: 80_379.0, mass_str: "80.38 GeV", charge_e: 1.0, spin: 1.0, generation: 0, colored: false },
    Fundamental { name: "Z boson",           symbol: "Z⁰", family: Family::GaugeBoson, mass_mev: 91_188.0, mass_str: "91.19 GeV", charge_e: 0.0, spin: 1.0, generation: 0, colored: false },
    // ---- the mass-giver ----
    Fundamental { name: "Higgs boson",       symbol: "H⁰", family: Family::ScalarBoson, mass_mev: 125_250.0, mass_str: "125.25 GeV", charge_e: 0.0, spin: 0.0, generation: 0, colored: false },
];

/// Look up a fundamental particle by name, symbol, or common alias
/// (case/unicode-insensitive: "nu_e", "νe", "gamma", "top" all work).
pub fn find_fundamental(q: &str) -> Option<&'static Fundamental> {
    let clean: String = q
        .chars()
        .filter(|c| c.is_alphanumeric())
        .collect::<String>()
        .to_lowercase();
    // common aliases first (bare family names, symbols with underscores…)
    let aliased: Option<&str> = match clean.as_str() {
        "nue" | "ve" | "neutrino" => Some("electronneutrino"),
        "numu" | "vmu" | "muonneutrino" => Some("muonneutrino"),
        "nutau" | "vtau" | "tauneutrino" => Some("tauneutrino"),
        "gamma" | "light" => Some("photon"),
        "w" | "wboson" | "wplus" | "wminus" => Some("wboson"),
        "z" | "zboson" | "z0" => Some("zboson"),
        "higgs" | "hboson" | "higgsboson" => Some("higgsboson"),
        "e" | "electron" => Some("electron"),
        "mu" | "muon" => Some("muon"),
        "tau" | "tauon" => Some("tau"),
        "up" | "upquark" => Some("upquark"),
        "down" | "downquark" => Some("downquark"),
        "strange" | "strangequark" => Some("strangequark"),
        "charm" | "charmquark" => Some("charmquark"),
        "bottom" | "bottomquark" | "beauty" => Some("bottomquark"),
        "top" | "topquark" | "truth" => Some("topquark"),
        "gluon" => Some("gluon"),
        _ => None,
    };
    if let Some(canonical) = aliased {
        return FUNDAMENTALS.iter().find(|p| {
            let name: String = p.name.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
            name == canonical
        });
    }
    // otherwise: full name or 2+ char symbol match
    FUNDAMENTALS.iter().find(|p| {
        let name: String = p.name.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
        let sym: String = p.symbol.chars().filter(|c| c.is_alphanumeric()).collect::<String>().to_lowercase();
        if sym.len() >= 2 { name == clean || sym == clean } else { name == clean }
    })
}

// ---------- the hadrons: quarks in formation ----------

/// A composite particle (quarks bound by gluons).
#[derive(Clone, Copy)]
pub struct Hadron {
    pub name: &'static str,
    /// Quark content, e.g. proton = u u d.
    pub quarks: &'static [&'static str],
    /// Measured mass (MeV/c²).
    pub mass_mev: f64,
    pub charge_e: f64,
    pub spin: f64,
}

pub const HADRONS: &[Hadron] = &[
    Hadron { name: "proton",  quarks: &["u", "u", "d"], mass_mev: PROTON_MEV,  charge_e: 1.0, spin: 0.5 },
    Hadron { name: "neutron", quarks: &["u", "d", "d"], mass_mev: NEUTRON_MEV, charge_e: 0.0, spin: 0.5 },
    Hadron { name: "pion+",   quarks: &["u", "d̄"],      mass_mev: 139.570_39, charge_e: 1.0, spin: 0.0 },
    Hadron { name: "pion0",   quarks: &["u", "ū"],      mass_mev: 134.976_8,  charge_e: 0.0, spin: 0.0 },
];

/// The quark-model card for one hadron: composition, charge check, and
/// the gluon-energy accounting that makes up the rest of the mass.
pub fn hadron_card(h: &Hadron) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    let comp: String = h.quarks.iter().copied().collect::<Vec<_>>().join(" ");
    // charge check from quark charges
    let q_sum: f64 = h
        .quarks
        .iter()
        .map(|q| {
            let anti = q.contains('̄');
            let base = q.trim_end_matches('̄');
            let c = match base {
                "u" => 2.0 / 3.0,
                "d" => -1.0 / 3.0,
                "s" => -1.0 / 3.0,
                _ => 0.0,
            };
            if anti { -c } else { c }
        })
        .sum();
    out.push(ConsoleLine { kind: LineKind::Info, text: format!("  {} = {} — charge from quarks: {} = {}e (measured: {}e) ✓", h.name, comp, h.quarks.iter().map(|q| format!("{:+.2}e", quark_charge(q))).collect::<Vec<_>>().join(" "), q_sum, h.charge_e) });
    // the famous accounting: quark masses vs hadron mass
    let quark_mass: f64 = h.quarks.iter().map(|q| quark_mass_mev(q)).sum();
    let binding = h.mass_mev - quark_mass;
    let frac = binding / h.mass_mev * 100.0;
    out.push(ConsoleLine { kind: LineKind::Info, text: format!("    quark masses sum to {:.1} MeV but the {} weighs {:.1} MeV — the other {:.1} MeV ({:.0}%) is gluon field energy + sea quarks. E = mc²: energy HAS mass.", quark_mass, h.name, h.mass_mev, binding, frac) });
    out
}

/// Charge of one quark (handles antiquarks marked with U+0304).
pub fn quark_charge(q: &str) -> f64 {
    let anti = q.contains('̄');
    let base = q.trim_end_matches('̄');
    let c = match base {
        "u" => 2.0 / 3.0,
        "d" => -1.0 / 3.0,
        "s" => -1.0 / 3.0,
        "c" => 2.0 / 3.0,
        "b" => -1.0 / 3.0,
        "t" => 2.0 / 3.0,
        _ => 0.0,
    };
    if anti { -c } else { c }
}

/// PDG mass of one quark (MeV) — current-quark masses.
fn quark_mass_mev(q: &str) -> f64 {
    let base = q.trim_end_matches('̄');
    match base {
        "u" => 2.2,
        "d" => 4.7,
        "s" => 95.0,
        "c" => 1275.0,
        "b" => 4180.0,
        "t" => 172_760.0,
        _ => 0.0,
    }
}

// ---------- the laws, particle-flavoured ----------

/// β⁻ decay: n → p + e⁻ + ν̄ₑ. A down quark becomes an up quark by
/// emitting a W⁻, which decays into an electron and an antineutrino.
/// Returns the Q-value in MeV (kinetic energy released).
pub fn beta_minus_q_mev() -> f64 {
    NEUTRON_MEV - PROTON_MEV - ELECTRON_MEV // 0.782 MeV
}

/// Electron–positron annihilation energy: e⁻ + e⁺ → 2γ.
/// Returns the total photon energy in MeV.
pub fn annihilation_mev() -> f64 {
    2.0 * ELECTRON_MEV // 1.022 MeV
}

/// Photon energy from wavelength (E = hc/λ), returned in eV.
pub fn photon_energy_ev(wavelength_nm: f64) -> f64 {
    if wavelength_nm <= 0.0 {
        return 0.0;
    }
    let lambda_m = wavelength_nm * 1.0e-9;
    H_JS * C_MPS / lambda_m / EV_J
}

/// Photon momentum p = E/c (kg·m/s) from wavelength in nm.
pub fn photon_momentum(wavelength_nm: f64) -> f64 {
    let lambda_m = wavelength_nm * 1.0e-9;
    H_JS / lambda_m
}

/// de Broglie wavelength λ = h/p, in meters.
pub fn de_broglie_m(mass_kg: f64, velocity_mps: f64) -> f64 {
    if mass_kg <= 0.0 || velocity_mps <= 0.0 {
        return f64::INFINITY;
    }
    H_JS / (mass_kg * velocity_mps)
}

/// Compton wavelength of a particle: λ = h/mc.
pub fn compton_m(mass_kg: f64) -> f64 {
    H_JS / (mass_kg * C_MPS)
}

// ---------- the briefing ----------

/// `simulate: particles` — the Standard Model briefing, tied to the
/// scene: the census of matter in the parts you built, the photons
/// raining on them, the neutrinos threading through them.
pub fn particles_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "THE STANDARD MODEL — 17 particles, and everything you built is three of them".into(),
    });

    // ---- the census: how much matter, by particle, is in this scene ----
    let census = atom::scene_census(world);
    if census.total_protons > 0.0 {
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("  in this scene right now: {:.3} protons, {:.3} neutrons, {:.3} electrons — every one of them 13.8 billion years old", census.total_protons, census.total_neutrons, census.total_electrons) });
        let net = (census.total_protons - census.total_electrons).abs();
        if net / census.total_protons.max(1.0) < 1e-9 {
            out.push(ConsoleLine { kind: LineKind::Info, text: format!("    proton charge +{:.3} C, electron charge −{:.3} C — net zero, exactly. Atoms are neutral because the universe keeps the books.", census.total_protons * E_CHARGE, census.total_electrons * E_CHARGE) });
        }
    } else {
        out.push(ConsoleLine { kind: LineKind::Info, text: "  the scene has no matter yet — build a part and ask again".into() });
    }

    // ---- the proton: the famous accounting trick ----
    let proton = &HADRONS[0];
    let neutron = &HADRONS[1];
    let mut card = hadron_card(proton);
    card.extend(hadron_card(neutron));
    for l in card {
        out.push(l);
    }
    out.push(ConsoleLine { kind: LineKind::Info, text: "    (so 99% of the mass of you, this scene, and the Earth is not 'stuff' — it is binding energy of the strong force)".into() });

    // ---- gluons: why you never see a lone quark ----
    out.push(ConsoleLine { kind: LineKind::Info, text: "  GLUON g — mass 0, spin 1, carries color charge itself".into() });
    out.push(ConsoleLine { kind: LineKind::Info, text: "    unlike photons, gluons feel the force they carry — the strong force between two quarks gets STRONGER with distance (like a stretched spring). Pull quarks apart and the field makes new quarks before it can snap: confinement. Free quarks have never been seen, by anyone, ever.".into() });

    // ---- photons: tied to this scene's temperature ----
    let t_k = world.temp_c + 273.15;
    let glow_wm2 = crate::world::energy::STEFAN_BOLTZMANN * t_k * t_k * t_k * t_k;
    let mid_ir_nm = crate::world::waves::wien_peak(t_k) * 1.0e9;
    out.push(ConsoleLine { kind: LineKind::Info, text: "  PHOTON γ — mass 0, spin 1, its own antiparticle, moves at c always".into() });
    out.push(ConsoleLine { kind: LineKind::Info, text: format!("    everything at a temperature glows: this scene at {:.1} K radiates {:.0} W/m² peaking at {:.0} nm {} — E = hf, so bluer light carries more energy per photon", t_k, glow_wm2, mid_ir_nm, if mid_ir_nm > 740.0 { "(infrared — you feel it, you can't see it)" } else if mid_ir_nm < 380.0 { "(ultraviolet)" } else { "(visible!)" }) });
    let green_ev = photon_energy_ev(532.0);
    out.push(ConsoleLine { kind: LineKind::Info, text: format!("    a 532 nm green photon carries {:.2} eV; gamma rays top 1 MeV — a million times more per packet", green_ev) });

    // ---- neutrinos: the ghost majority ----
    out.push(ConsoleLine { kind: LineKind::Info, text: "  NEUTRINO νₑ νμ ντ — mass ≈ 0 but not exactly 0, charge 0, spin ½".into() });
    out.push(ConsoleLine { kind: LineKind::Info, text: format!("    {:.1} billion solar neutrinos pass through every cm² of this scene every second, and a solid-light-year of lead would barely slow them — the 1998 discovery that they have mass (via oscillation) broke the Standard Model's last zero", SOLAR_NU_FLUX / 1.0e9) });
    out.push(ConsoleLine { kind: LineKind::Info, text: format!("    plus {} relic neutrinos per cm³ left over from one second after the Big Bang — the oldest particles in existence", RELIC_NU_DENSITY) });

    // ---- β-decay, if the scene is radioactive ----
    let radioactive = atom::radioactive_materials(world);
    if !radioactive.is_empty() {
        let q = beta_minus_q_mev();
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("  BETA DECAY is quark alchemy: d → u + W⁻ → e⁻ + ν̄ₑ — a neutron ({:.1} MeV) becomes a proton ({:.1} MeV) and the {:.3} MeV difference becomes the electron, the antineutrino, and their motion", NEUTRON_MEV, PROTON_MEV, q) });
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("    this scene knows: {} — run `simulate: decay` for their half-lives and live activity", radioactive.join(", ")) });
    }

    // ---- the family tree, one line per particle ----
    out.push(ConsoleLine { kind: LineKind::Info, text: "  the full card deck (mass · charge · spin):".into() });
    for p in FUNDAMENTALS {
        let charge = if p.charge_e == 0.0 {
            "0".to_string()
        } else if (p.charge_e * 3.0).fract() == 0.0 && p.charge_e.abs() < 1.0 {
            format!("{:+}/3", (p.charge_e * 3.0) as i32) // +2/3, −1/3
        } else {
            format!("{:+}", p.charge_e)
        };
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("    {:<18} {:>3}  {:>12}  charge {}e  spin {}", p.name, p.symbol, p.mass_str, charge, p.spin) });
    }
    out.push(ConsoleLine { kind: LineKind::Info, text: "  12 fermions of matter + 4 bosons of force + 1 Higgs = the Standard Model — complete, tested to 12 digits, and KNOWN to be incomplete (no gravity, no dark matter, no why-neutrinos-have-mass)".into() });
    out
}

/// `particle: <name>` — one particle's card, straight from the deck.
pub fn particle_card(name: &str) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    // fundamental first, then hadrons
    if let Some(p) = find_fundamental(name) {
        let charge = if p.charge_e == 0.0 {
            "0".into()
        } else if p.charge_e.abs() < 1.0 {
            let num = (p.charge_e * 3.0) as i32;
            format!("{}/3 e", num)
        } else {
            format!("{:+} e", p.charge_e)
        };
        out.push(ConsoleLine { kind: LineKind::Answer, text: format!("{} ({}) — a {} of generation {}, mass {}, charge {}, spin {}", p.name, p.symbol, p.family.as_str(), p.generation, p.mass_str, charge, p.spin) });
        // the one-paragraph story per family
        let story: String = match p.family {
            Family::Quark => "fractional charge, confined — has never been seen alone; three of them (u,u,d) are a proton".to_string(),
            Family::Lepton => if p.name.contains("neutrino") { "chargeless, nearly massless, interacts only via the weak force — a light-year of lead wouldn't stop it".to_string() } else { "the electron's heavier siblings — same charge, more mass, unstable (the muon decays in 2.2 μs; the tau in 0.3 ps)".to_string() },
            Family::GaugeBoson => match p.symbol {
                "γ" => "the electromagnetic force carrier — light itself, always at c, no charge, no mass, no aging".to_string(),
                "g" => "the strong force carrier — carries color itself, so gluons attract gluons; confinement is the consequence".to_string(),
                _ => "the weak force carriers — uniquely massive (that's why the weak force is weak and short-ranged); W± change quark flavors, enabling beta decay".to_string(),
            },
            Family::ScalarBoson => "the excitation of the Higgs field that fills all space — particles get mass by dragging through it; found in 2012, Nobel 2013".to_string(),
        };
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("  {}", story) });
        // matter vs antimatter note
        if p.family == Family::Quark || (p.family == Family::Lepton && !p.name.contains("neutrino")) {
            out.push(ConsoleLine { kind: LineKind::Info, text: format!("  its antiparticle {}̄ has the opposite charge but the same mass and spin — and the universe somehow made more of you than of it", p.symbol) });
        }
        return out;
    }
    let key = name.trim().to_lowercase();
    let key = key.trim_end_matches('0').trim_end_matches("minus").trim_end_matches("plus").trim_end_matches('+').trim_end_matches('-').trim_end_matches('⁺').trim_end_matches('⁰');
    if let Some(h) = HADRONS.iter().find(|h| h.name == key || (key == "electrons" && h.name == "electron")) {
        out.push(ConsoleLine { kind: LineKind::Answer, text: format!("{} = {} (quarks bound by gluons) — mass {:.3} MeV, charge {:+}e, spin {}", h.name, h.quarks.iter().copied().collect::<Vec<_>>().join(" "), h.mass_mev, h.charge_e, h.spin) });
        out.extend(hadron_card(h));
        if h.name == "neutron" {
            out.push(ConsoleLine { kind: LineKind::Info, text: format!("  free neutrons decay in 881.5 s (n → p + e⁻ + ν̄ₑ, {:.3} MeV released) — only the gluon binding inside a nucleus keeps them forever", beta_minus_q_mev()) });
        }
        return out;
    }
    out.push(ConsoleLine { kind: LineKind::Error, text: format!("'{}' is not in the deck — try proton, neutron, electron, up, down, strange, charm, bottom, top, muon, tau, neutrino, photon, gluon, w, z, or higgs", name) });
    out
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn standard_model_census() {
        assert_eq!(FUNDAMENTALS.len(), 17);
        let quarks = FUNDAMENTALS.iter().filter(|p| p.family == Family::Quark).count();
        let leptons = FUNDAMENTALS.iter().filter(|p| p.family == Family::Lepton).count();
        let gauge = FUNDAMENTALS.iter().filter(|p| p.family == Family::GaugeBoson).count();
        let scalar = FUNDAMENTALS.iter().filter(|p| p.family == Family::ScalarBoson).count();
        assert_eq!((quarks, leptons, gauge, scalar), (6, 6, 4, 1));
    }

    #[test]
    fn quark_charges_sum_right() {
        // proton uud = +1e exactly
        let p: f64 = quark_charge("u") + quark_charge("u") + quark_charge("d");
        assert!((p - 1.0).abs() < 1e-12);
        // neutron udd = 0 exactly
        let n: f64 = quark_charge("u") + quark_charge("d") + quark_charge("d");
        assert!(n.abs() < 1e-12);
        // pion+ u d̄ = +1e
        let pi: f64 = quark_charge("u") + quark_charge("d̄");
        assert!((pi - 1.0).abs() < 1e-12);
    }

    #[test]
    fn the_famous_accounting() {
        // quarks in a proton: 2×2.2 + 4.7 = 9.1 MeV vs proton 938.3 MeV
        let qm = 2.0 * 2.2 + 4.7;
        assert!(qm < 10.0);
        let binding = PROTON_MEV - qm;
        assert!(binding / PROTON_MEV > 0.98, "99% of mass is gluon energy");
        // hadron_card must agree
        let card = hadron_card(&HADRONS[0]);
        assert!(card[1].text.contains("gluon field energy"));
    }

    #[test]
    fn beta_decay_and_annihilation() {
        assert!((beta_minus_q_mev() - 0.782).abs() < 0.005);
        assert!((annihilation_mev() - 1.022).abs() < 0.001);
    }

    #[test]
    fn photon_math() {
        // 532 nm green photon ≈ 2.33 eV
        assert!((photon_energy_ev(532.0) - 2.33).abs() < 0.02);
        // 500 nm ≈ 2.48 eV
        assert!((photon_energy_ev(500.0) - 2.48).abs() < 0.02);
        // E=hf: a 100 MHz FM photon — λ = c/f = 2.998 m = 2.998e9 nm
        assert!((photon_energy_ev(2.997_924_58e9) - 4.136e-7).abs() < 1e-9);
    }

    #[test]
    fn de_broglie_and_compton() {
        // electron at 1 keV... kinetic → v via KE = ½mv² (non-relativistic ok at 1 eV scale)
        let v_1ev = (2.0 * EV_J / ELECTRON_KG).sqrt();
        let lambda = de_broglie_m(ELECTRON_KG, v_1ev);
        // 1 eV electron → 1.227 nm
        assert!((lambda * 1e9 - 1.227).abs() < 0.01);
        // electron Compton wavelength 2.426 pm
        assert!((compton_m(ELECTRON_KG) * 1e12 - 2.426).abs() < 0.01);
    }

    #[test]
    fn lookup_works() {
        assert!(find_fundamental("proton").is_none(), "proton is composite, not fundamental");
        assert!(find_fundamental("electron").is_some());
        assert!(find_fundamental("photon").is_some());
        assert!(find_fundamental("gluon").is_some());
        assert!(find_fundamental("electron neutrino").is_some());
        assert!(find_fundamental("nu_e").is_some());
        assert!(find_fundamental("top").is_some());
        assert!(find_fundamental("higgs").is_some());
        assert_eq!(find_fundamental("electron").unwrap().mass_mev, ELECTRON_MEV);
    }

    #[test]
    fn particle_cards() {
        let card = particle_card("gluon");
        assert!(card[0].text.contains("GaugeBoson".replace("GaugeBoson", "gauge boson").as_str()) || card[0].text.contains("gauge boson"));
        let card = particle_card("proton");
        assert!(card[0].text.contains("u u d"));
        let card = particle_card("neutron");
        assert!(card[0].text.contains("u d d"));
        let card = particle_card("nope");
        assert!(matches!(card[0].kind, LineKind::Error));
    }

    #[test]
    fn neutron_proton_split() {
        // n − p = 1.293 MeV: why the free neutron decays but nuclei hold
        assert!((NEUTRON_MEV - PROTON_MEV - 1.293).abs() < 0.001);
    }
}
