//! P2100 — OTD3 ENERGY — the complete taxonomy of energy.
//!
//! There are two main categories of energy: energy that is STORED, and
//! energy that is MOVING. From these two categories, science breaks energy
//! down into nine specific forms:
//!
//!   KINETIC (energy in motion — 5 forms)
//!     1. Mechanical   — moving objects: a rolling ball, a spinning turbine
//!     2. Thermal      — moving atoms & molecules; faster = hotter
//!     3. Radiant      — waves through space: light, X-rays, radio
//!     4. Electrical   — moving electrons (current)
//!     5. Sound        — vibrations through air, water, solids
//!
//!   POTENTIAL (stored energy — 4 forms)
//!     1. Chemical     — bonds of atoms: food, wood, batteries, gasoline
//!     2. Gravitational— stored by height: a rock at the top of a hill
//!     3. Nuclear      — the nucleus of atoms: fission and fusion
//!     4. Elastic      — stretched or compressed: springs, rubber bands
//!
//! Energy is conserved: it never appears or disappears, only transforms.
//! `simulate: energy` audits every part of the scene and reports the
//! full ledger, form by form, with the law that computed each number.

use super::eval::{ConsoleLine, LineKind, World};
use crate::units::MM3_PER_M3;

/// One line of the energy ledger.
#[derive(Clone, Debug)]
pub struct EnergyForm {
    /// kinetic | potential
    pub category: &'static str,
    /// mechanical, thermal, radiant, electrical, sound,
    /// chemical, gravitational, nuclear, elastic
    pub form: &'static str,
    /// joules
    pub joules: f64,
    /// the physical law that produced the number
    pub law: String,
}

/// Energy density of fuels & food (J/kg) — chemical energy per kilogram.
/// Textbook values (approximate, rounded for education).
pub fn chemical_energy_density(mat: &str) -> f64 {
    match mat {
        // fuels
        "gasoline" => 46_000_000.0,
        "diesel" => 45_600_000.0,
        "kerosene" => 43_000_000.0,
        "propane" => 50_300_000.0,
        "hydrogen" => 142_000_000.0, // the champion: H₂ per kg
        "methane" => 55_500_000.0,   // natural gas
        "wood" | "oak" | "pine" | "teak" => 16_000_000.0,
        "coal" => 30_000_000.0,
        // battery metals: lithium-ion cells deliver ~0.9 MJ/kg packaged
        "lithium" => 900_000.0, // Li-ion cell energy density (packaged)
        // oxidation reserves for metals that burn/rust (iron → Fe₂O₃ is
        // exothermic; aluminium burns hotter than thermite shows)
        "iron" | "steel" | "stainless" => 7_200_000.0,
        "aluminum" => 8_400_000.0,   // why thermite cuts rails
        "copper" | "brass" | "bronze" | "silver" | "gold" => 500_000.0,
        "plastic" => 46_000_000.0,   // ~45 MJ/kg — plastics ARE fuels
        "rubber" => 35_000_000.0,    // tyre fires are real chemistry
        "fabric" => 17_000_000.0,
        "carbon" => 32_800_000.0,    // burning graphite
        // default: every material stores SOME bond energy; the honest
        // baseline for quiet materials is ~0.2 MJ/kg of reaction reserve
        _ => 200_000.0,
    }
}

/// Nuclear energy available per kilogram by fission/fusion of the material.
/// E = mc² sets the absolute ceiling; real fuels release only the binding
/// energy difference. Values in J/kg (fission unless noted).
pub fn nuclear_energy_density(mat: &str) -> f64 {
    match mat {
        // U-235 complete fission: ~8.2e13 J/kg (0.09% of mc²)
        "uranium" => 8.2e13,
        // Pu-239: ~8.6e13 J/kg
        "plutonium" => 8.6e13,
        // thorium fuel cycle: ~7.9e13 J/kg
        "thorium" => 7.9e13,
        // deuterium/tritium fusion is ~3.4e14 J/kg — the Sun's economy
        "hydrogen" => 3.4e14, // D-T fusion of pure fuel mass
        // ordinary matter's mc² — the absolute ceiling, if you could
        // annihilate it (only antimatter lets you actually do this)
        _ => 9.0e16 * 0.0, // default handled by caller via E=mc² flag
    }
}

/// Food-class chemical energy (rough guide, J/kg): fat 3.7e7, carbs 1.7e7,
/// protein 1.7e7.
pub const FAT_J_KG: f64 = 37_000_000.0;
pub const CARB_J_KG: f64 = 17_000_000.0;

/// Stefan–Boltzmann constant, W/(m²·K⁴) — the radiant-energy law.
pub const STEFAN_BOLTZMANN: f64 = 5.670374419e-8;
/// Speed of light in vacuum, m/s — the universe's speed limit.
pub const C_LIGHT: f64 = 299_792_458.0;
/// Speed of sound in air at 20 °C, m/s.
pub const SPEED_OF_SOUND_AIR: f64 = 343.0;
/// Atomic mass unit in kg (for E=mc² bookkeeping).
pub const AMU_KG: f64 = 1.66053906860e-27;

/// Audit ONE part and return every energy form it carries.
///
/// The philosophy is unchanged since 2.0: users never configure physics.
/// Geometry gives mass and height; materials give everything else; the
/// laws do the arithmetic.
pub fn audit_part(part: &super::eval::Part, g: f64, scene_temp_c: f64) -> Vec<EnergyForm> {
    let mut out = Vec::new();
    let mass_kg = part.mass_g / 1000.0;
    if mass_kg <= 0.0 {
        return out;
    }
    let mat_name = part.material.map(|m| m.name).unwrap_or("plastic");
    let bb = part.mesh.bbox();
    // height of the centre of mass above the scene floor (mm → m)
    let h_m = part.centroid.map(|c| c.y().max(0.0) / 1000.0)
        .unwrap_or(bb.center().y().max(0.0) / 1000.0);

    // ---------- POTENTIAL (stored) ----------

    // 1. Gravitational — PE = m·g·h. The rock on the hill.
    let pe_g = mass_kg * g * h_m;
    out.push(EnergyForm {
        category: "potential",
        form: "gravitational",
        joules: pe_g,
        law: format!("PE = m·g·h = {:.3} kg × {} × {:.2} m", mass_kg, g, h_m),
    });

    // 2. Chemical — the bonds of atoms. Every material carries its bonds'
    //    energy; fuels carry spectacular amounts, quiet materials a small
    //    but real reaction reserve.
    let chem_j_kg = chemical_energy_density(mat_name);
    let pe_chem = mass_kg * chem_j_kg;
    out.push(EnergyForm {
        category: "potential",
        form: "chemical",
        joules: pe_chem,
        law: format!("E = m·e_d  ({:.2} MJ/kg stored in {} bonds)", chem_j_kg / 1e6, mat_name),
    });

    // 3. Nuclear — the nucleus. E = mc² is the ceiling; fuels release the
    //    binding-energy difference (fission ~0.09% of mc², fusion ~0.4%).
    let nuc_j_kg = nuclear_energy_density(mat_name);
    let pe_nuc = if nuc_j_kg > 0.0 {
        mass_kg * nuc_j_kg
    } else {
        // every mass has the mc² ceiling — report it for emphasis
        mass_kg * C_LIGHT * C_LIGHT
    };
    let nuclear_note = if nuc_j_kg > 0.0 { "fissionable" } else { "E = mc² ceiling" };
    out.push(EnergyForm {
        category: "potential",
        form: "nuclear",
        joules: pe_nuc,
        law: format!("{} — {} ({:.3} TJ/kg equivalent)", nuclear_note, mat_name, pe_nuc / mass_kg / 1e12),
    });

    // 4. Elastic — stored in stretch & compression. A part resting on the
    //    ground stores elastic energy in its own compression: the weight
    //    above it (its own height of material) strains the bottom face.
    //    Estimate via the material's stiffness (Young's modulus, MPa).
    let young_mpa = youngs_modulus(mat_name);
    if young_mpa > 0.0 {
        // self-load compression: strain ≈ ρ·g·h / E, energy ≈ ½·strain²·E·V
        let rho = part.material.map(|m| m.density).unwrap_or(1050.0);
        let h = h_m;
        let stress_pa = rho * g * h; // Pa
        let strain = (stress_pa / (young_mpa * 1e6)).min(0.05); // cap at 5%
        let volume_m3 = part.volume_mm3 / MM3_PER_M3;
        let pe_elastic = 0.5 * strain * stress_pa * volume_m3;
        out.push(EnergyForm {
            category: "potential",
            form: "elastic",
            joules: pe_elastic,
            law: format!("U = ½·σ·ε·V (self-load strain {:.4}%)", strain * 100.0),
        });
    }

    // ---------- KINETIC (moving) ----------

    // 1. Thermal — moving atoms & molecules. Internal energy above absolute
    //    zero, U ≈ m·c_p·T (a solid stores ~3kB per atom: Dulong–Petit).
    let cp = specific_heat(mat_name); // J/(kg·K)
    let u_thermal = mass_kg * cp * (scene_temp_c + 273.15);
    out.push(EnergyForm {
        category: "kinetic",
        form: "thermal",
        joules: u_thermal,
        law: format!("U = m·c_p·T = {:.3} kg × {} J/kgK × {} K", mass_kg, cp as u32, (scene_temp_c + 273.15) as u32),
    });

    // 2. Radiant — energy the part EMITS right now, by its temperature
    //    (Stefan–Boltzmann). Every warm object glows; hot ones visibly.
    let area_m2 = part.area_mm2 * 1e-6;
    let t_kelvin = scene_temp_c + 273.15;
    let radiant_w = STEFAN_BOLTZMANN * area_m2 * t_kelvin * t_kelvin * t_kelvin * t_kelvin;
    // report the emission RATE (W) converted to energy per second
    out.push(EnergyForm {
        category: "kinetic",
        form: "radiant",
        joules: radiant_w, // J per second of emission
        law: format!("P = εσA·T⁴ = {} W of glowing (mostly infrared now)", radiant_w as u32),
    });

    // 3. Electrical — moving electrons. A conductor at rest stores charge
    //    energy only under voltage; report its conductive capacity as the
    //    energy a 1-second, 1-amp flow through it dissipates at 1 V/m.
    if let Some(cond_ms) = part.material.and_then(|m| m.conductive) {
        // resistive loss along the longest axis at 1 A: P = I²·R, R = L/(σA)
        let sz = bb.size();
        let len_m = sz.x().max(sz.y()).max(sz.z()) / 1000.0;
        let cross_m2 = if len_m > 0.0 { (part.volume_mm3 / MM3_PER_M3) / len_m } else { 0.0 };
        let r_ohm = len_m / (cond_ms * 1e6 * cross_m2.max(1e-9));
        let e_elec = 1.0 * 1.0 * r_ohm; // 1 A for 1 s at that resistance
        out.push(EnergyForm {
            category: "kinetic",
            form: "electrical",
            joules: e_elec,
            law: format!("P = I²·R ({} MS/m conductor, R ≈ {:.6} Ω end-to-end)", cond_ms, r_ohm),
        });
    }

    // 4. Sound — vibrations. A still part is silent, but any impact of THIS
    //    mass from THIS height converts a slice of its PE into sound. The
    //    classic fraction radiated by an impact on a hard surface ~1%.
    let e_impact = pe_g; // gravitational → kinetic at the floor
    let e_sound = 0.01 * e_impact;
    out.push(EnergyForm {
        category: "kinetic",
        form: "sound",
        joules: e_sound,
        law: "≈1% of impact energy radiates as sound (hard-surface rule)".into(),
    });

    // 5. Mechanical — the energy of motion itself. A static scene stores
    //    none in translation; report the kinetic energy this mass carries
    //    at free-fall impact speed after dropping its own height (the
    //    motion it is ABOUT to have under gravity).
    let v = (2.0 * g * h_m.max(0.001)).sqrt();
    let ke = 0.5 * mass_kg * v * v;
    out.push(EnergyForm {
        category: "kinetic",
        form: "mechanical",
        joules: ke,
        law: format!("KE = ½mv² at impact, v = √(2gh) = {:.2} m/s", v),
    });

    out
}

/// Young's modulus, MPa (approximate, textbook).
pub fn youngs_modulus(mat: &str) -> f64 {
    match mat {
        "steel" | "stainless" => 200_000.0,
        "iron" => 190_000.0,
        "titanium" => 110_000.0,
        "copper" => 120_000.0,
        "brass" | "bronze" => 100_000.0,
        "aluminum" => 69_000.0,
        "gold" => 78_000.0,
        "silver" => 83_000.0,
        "zinc" => 85_000.0,
        "lead" => 16_000.0,
        "tungsten" => 400_000.0,
        "chrome" => 250_000.0,
        "glass" => 70_000.0,
        "marble" | "ceramic" => 60_000.0,
        "concrete" => 30_000.0,
        "wood" | "oak" | "teak" => 11_000.0,
        "pine" => 9_000.0,
        "plastic" => 3_000.0,
        "rubber" => 0.01 * 1000.0, // ~0.01–0.1 MPa — the squish champion
        "ice" => 9_000.0,
        "carbon" => 150_000.0, // carbon fiber laminate
        "foam" => 10.0,
        _ => 3_000.0,
    }
}

/// Specific heat capacity, J/(kg·K) (approximate, textbook).
pub fn specific_heat(mat: &str) -> f64 {
    match mat {
        "water" => 4186.0, // the famous one — water rules heat storage
        "ice" => 2093.0,
        "hydrogen" => 14_304.0,
        "helium" => 5_193.0,
        "aluminum" => 900.0,
        "copper" => 385.0,
        "gold" => 129.0,
        "silver" => 235.0,
        "iron" | "steel" | "stainless" => 450.0,
        "lead" => 128.0,
        "titanium" => 523.0,
        "tungsten" => 134.0,
        "zinc" => 388.0,
        "brass" | "bronze" => 380.0,
        "glass" => 840.0,
        "ceramic" => 1_050.0,
        "marble" => 880.0,
        "concrete" => 880.0,
        "wood" | "oak" | "teak" => 1_700.0,
        "pine" => 1_500.0,
        "plastic" => 1_300.0,
        "rubber" => 1_500.0,
        "carbon" => 710.0,
        "mercury" => 139.0,
        "ethanol" => 2_440.0,
        "acetone" => 2_150.0,
        "glycerin" => 2_430.0,
        "oil" => 1_970.0,
        "foam" => 1_300.0,
        "fabric" => 1_300.0,
        _ => 1_000.0,
    }
}

/// Format joules into a human-friendly string with the right SI prefix.
pub fn fmt_j(j: f64) -> String {
    let a = j.abs();
    if a >= 1e12 { format!("{:.2} TJ", j / 1e12) }
    else if a >= 1e9 { format!("{:.2} GJ", j / 1e9) }
    else if a >= 1e6 { format!("{:.2} MJ", j / 1e6) }
    else if a >= 1e3 { format!("{:.2} kJ", j / 1e3) }
    else if a >= 1e-3 { format!("{:.2} J", j) }
    else if a >= 1e-6 { format!("{:.2} mJ", j / 1e-3) }
    else { format!("{:.2e} J", j) }
}

/// The full-scene audit: `simulate: energy`.
pub fn energy_sim(world: &World, g: f64, scene_temp_c: f64) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine {
        kind: LineKind::Sim,
        text: "ENERGY AUDIT — two categories, nine forms, one law: energy is never created or destroyed".into(),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: "  KINETIC (in motion): mechanical, thermal, radiant, electrical, sound".into(),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: "  POTENTIAL (stored):  chemical, gravitational, nuclear, elastic".into(),
    });

    if world.parts.iter().all(|p| p.hidden) {
        out.push(ConsoleLine { kind: LineKind::Warn, text: "nothing to audit — every part is hidden".into() });
        return out;
    }

    let mut totals: Vec<EnergyForm> = Vec::new();
    for part in &world.parts {
        if part.hidden || part.mass_g <= 0.0 {
            continue;
        }
        let forms = audit_part(part, g, scene_temp_c);
        let name = &part.name;
        // per-part digest: top 3 forms by magnitude
        let mut sorted = forms.clone();
        sorted.sort_by(|a, b| b.joules.partial_cmp(&a.joules).unwrap_or(std::cmp::Ordering::Equal));
        let top: Vec<String> = sorted.iter().take(3)
            .map(|f| format!("{} {}", f.form, fmt_j(f.joules)))
            .collect();
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("  {} ({:.1} g): {}", name, part.mass_g, top.join(" · ")),
        });
        for f in forms {
            if let Some(t) = totals.iter_mut().find(|t| t.form == f.form) {
                t.joules += f.joules;
            } else {
                totals.push(f);
            }
        }
    }

    // the ledger, grouped by category
    out.push(ConsoleLine { kind: LineKind::Sim, text: "---- THE LEDGER (whole scene) ----".into() });
    let mut kinetic = 0.0;
    let mut potential = 0.0;
    for f in &totals {
        let prefix = if f.category == "kinetic" { "kinetic" } else { "potential" };
        out.push(ConsoleLine {
            kind: LineKind::Info,
            text: format!("  {:<14} {:<14} {:>12}   — {}", prefix, f.form, fmt_j(f.joules), f.law),
        });
        if f.category == "kinetic" { kinetic += f.joules; } else { potential += f.joules; }
    }
    out.push(ConsoleLine {
        kind: LineKind::Answer,
        text: format!("  KINETIC total {}  +  POTENTIAL total {}  =  {} of energy riding in this scene",
            fmt_j(kinetic), fmt_j(potential), fmt_j(kinetic + potential)),
    });
    out.push(ConsoleLine {
        kind: LineKind::Info,
        text: "  (nuclear dominates every ledger — that is why the Sun out-shines every campfire)".into(),
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::units::G_EARTH;
    use crate::world::eval::compile;

    #[test]
    fn rock_on_a_hill() {
        // 1 kg of iron 1 m up: PE = mgh = 9.81 J (plus change)
        let w = compile("scene \"t\"\nrock = cube 10cm at (0, 1m, 0) material: iron");
        let forms = audit_part(&w.parts[0], G_EARTH, 20.0);
        let pe = forms.iter().find(|f| f.form == "gravitational").unwrap();
        let mass = w.parts[0].mass_g / 1000.0;
        // `at` parks the cube's base at 1 m; its centroid rides 5 cm higher
        assert!((pe.joules - mass * G_EARTH * 1.05).abs() < 0.05, "PE=mgh for the real centroid height, got {}", pe.joules);
    }

    #[test]
    fn kinetic_from_height() {
        let w = compile("scene \"t\"\nb = sphere 3cm at (0, 2m, 0) material: steel");
        let forms = audit_part(&w.parts[0], G_EARTH, 20.0);
        let ke = forms.iter().find(|f| f.form == "mechanical").unwrap();
        let mass = w.parts[0].mass_g / 1000.0;
        let h = w.parts[0].centroid.unwrap().y() / 1000.0;
        // v = √(2·g·h); KE = ½mv² = m·g·h exactly (h at the real centroid)
        assert!((ke.joules - mass * G_EARTH * h).abs() < 1e-6, "KE at impact = mgh");
    }

    #[test]
    fn water_stores_the_most_heat() {
        assert!(specific_heat("water") > specific_heat("iron"));
        assert!(specific_heat("water") > specific_heat("aluminum"));
    }

    #[test]
    fn all_nine_forms_present() {
        let w = compile("scene \"t\"\nthing = cylinder r: 1cm h: 5cm at (0, 50cm, 0) material: copper");
        let forms = audit_part(&w.parts[0], G_EARTH, 20.0);
        let kinds: Vec<&str> = forms.iter().map(|f| f.form).collect();
        for want in ["mechanical", "thermal", "radiant", "electrical", "sound",
                     "chemical", "gravitational", "nuclear", "elastic"] {
            assert!(kinds.contains(&want), "missing energy form: {}", want);
        }
    }

    #[test]
    fn audit_reports_ledger() {
        let w = compile("scene \"t\"\na = cube 5cm at (0, 20cm, 0) material: wood");
        let lines = energy_sim(&w, G_EARTH, 20.0);
        assert!(lines.iter().any(|l| l.text.contains("THE LEDGER")));
        assert!(lines.iter().any(|l| l.text.contains("gravitational")));
        assert!(lines.iter().any(|l| l.text.contains("thermal")));
    }
}
