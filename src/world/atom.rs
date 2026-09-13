//! P2260 — ATOM: protons, neutrons, electrons, and the nucleus that binds them.
//!
//! The scene is measured in grams; this module converts grams into
//! PARTICLES — real proton, neutron and electron counts from the actual
//! mesh mass and the chemical composition of each material. A 220 g
//! ceramic cup becomes ~10²⁴ electrons, and every one of them is
//! seventeen orders of magnitude smaller than the cup it lives in.
//!
//!   * `ELEMENTS` — all 118 elements: Z, symbol, name, dominant isotope A,
//!     molar mass. The whole periodic table, one array.
//!   * Madelung electron configurations (the (n+ℓ, n) filling rule) with
//!     the 20 famous exceptions (Cr, Cu, Pd, Au, the actinides…) — so
//!     `electron configuration of gold` answers 5d¹⁰ 6s¹, not the
//!     textbook-naive 6s² 5d⁹.
//!   * `composition()` — the engineering material (steel, brass, wood,
//!     water…) mapped to elements with mass fractions, so the census is
//!     chemistry-aware, not fantasy.
//!   * Weizsäcker's semi-empirical mass formula — binding energy from
//!     volume + surface + Coulomb + asymmetry + pairing, checked against
//!     Fe-56 (8.76 vs 8.79 MeV/nucleon measured) and U-238.
//!   * `ISOTOPES` — the famous radioactive ones, half-lives and modes:
//!     C-14 (5730 y), U-238 (4.47 Gy), Pu-239, K-40, Cs-137…
//!   * The decay law N(t) = N₀·2^(−t/T½), and the Rydberg formula for
//!     hydrogen's spectral lines.
//!   * `atom_sim` / `decay_sim` — the narrated briefings; `scene_census`
//!     feeds `simulate: particles` with the real matter count.

use super::eval::{ConsoleLine, LineKind, Part, World};

/// Avogadro's number (exact, SI 2019).
pub const N_A: f64 = 6.022_140_76e23;
/// Rydberg constant for hydrogen (m⁻¹).
pub const RYDBERG_H: f64 = 1.096_775_8e7;
/// Nuclear radius scaling: r = R0·A^(1/3), R0 = 1.2 fm.
pub const R0_FM: f64 = 1.2;

// ---------- the periodic table ----------

#[derive(Clone, Copy)]
pub struct Element {
    pub z: u32,
    pub sym: &'static str,
    pub name: &'static str,
    /// Dominant / most relevant isotope's mass number (hydrogen 1, U 238…).
    pub a: u32,
    /// Molar mass (g/mol) — standard atomic weight, or isotope mass for
    /// the radioactive tail of the table.
    pub mass_u: f64,
    /// Half-life string for naturally radioactive elements ("" = stable).
    pub note: &'static str,
}

macro_rules! el {
    ($z:literal, $sym:literal, $name:literal, $a:literal, $m:literal) => {
        Element { z: $z, sym: $sym, name: $name, a: $a, mass_u: $m, note: "" }
    };
    ($z:literal, $sym:literal, $name:literal, $a:literal, $m:literal, $note:literal) => {
        Element { z: $z, sym: $sym, name: $name, a: $a, mass_u: $m, note: $note }
    };
}

/// All 118 elements. Radioactive ones carry their half-life in `note`.
pub const ELEMENTS: &[Element] = &[
    el!(1, "H", "hydrogen", 1, 1.008), el!(2, "He", "helium", 4, 4.0026),
    el!(3, "Li", "lithium", 7, 6.94), el!(4, "Be", "beryllium", 9, 9.0122),
    el!(5, "B", "boron", 11, 10.81), el!(6, "C", "carbon", 12, 12.011),
    el!(7, "N", "nitrogen", 14, 14.007), el!(8, "O", "oxygen", 16, 15.999),
    el!(9, "F", "fluorine", 19, 18.998), el!(10, "Ne", "neon", 20, 20.180),
    el!(11, "Na", "sodium", 23, 22.990), el!(12, "Mg", "magnesium", 24, 24.305),
    el!(13, "Al", "aluminum", 27, 26.982), el!(14, "Si", "silicon", 28, 28.085),
    el!(15, "P", "phosphorus", 31, 30.974), el!(16, "S", "sulfur", 32, 32.06),
    el!(17, "Cl", "chlorine", 35, 35.45), el!(18, "Ar", "argon", 40, 39.948),
    el!(19, "K", "potassium", 39, 39.098, "K-40: T½ 1.25 Gy, β⁻/EC — every banana is faintly radioactive"),
    el!(20, "Ca", "calcium", 40, 40.078), el!(21, "Sc", "scandium", 45, 44.956),
    el!(22, "Ti", "titanium", 48, 47.867), el!(23, "V", "vanadium", 51, 50.942),
    el!(24, "Cr", "chromium", 52, 51.996), el!(25, "Mn", "manganese", 55, 54.938),
    el!(26, "Fe", "iron", 56, 55.845, "Fe-56: the most tightly bound nucleus — fusion's dead end, 8.79 MeV/nucleon"),
    el!(27, "Co", "cobalt", 59, 58.933), el!(28, "Ni", "nickel", 58, 58.693),
    el!(29, "Cu", "copper", 63, 63.546), el!(30, "Zn", "zinc", 64, 65.38),
    el!(31, "Ga", "gallium", 69, 69.723), el!(32, "Ge", "germanium", 74, 72.630),
    el!(33, "As", "arsenic", 75, 74.922), el!(34, "Se", "selenium", 80, 78.971),
    el!(35, "Br", "bromine", 79, 79.904), el!(36, "Kr", "krypton", 84, 83.798),
    el!(37, "Rb", "rubidium", 85, 85.468, "Rb-87: T½ 48.8 Gy, β⁻ — a rock-dating clock"),
    el!(38, "Sr", "strontium", 88, 87.62), el!(39, "Y", "yttrium", 89, 88.906),
    el!(40, "Zr", "zirconium", 90, 91.224), el!(41, "Nb", "niobium", 93, 92.906),
    el!(42, "Mo", "molybdenum", 98, 95.95), el!(43, "Tc", "technetium", 98, 98.0, "no stable isotope at all — the gap Mendeleev predicted"),
    el!(44, "Ru", "ruthenium", 102, 101.07), el!(45, "Rh", "rhodium", 103, 102.91),
    el!(46, "Pd", "palladium", 106, 106.42), el!(47, "Ag", "silver", 107, 107.87),
    el!(48, "Cd", "cadmium", 114, 112.41), el!(49, "In", "indium", 115, 114.82, "In-115: T½ 4.4e14 y — radioactive, but the universe isn't patient enough"),
    el!(50, "Sn", "tin", 120, 118.71), el!(51, "Sb", "antimony", 121, 121.76),
    el!(52, "Te", "tellurium", 130, 127.60), el!(53, "I", "iodine", 127, 126.90),
    el!(54, "Xe", "xenon", 132, 131.29), el!(55, "Cs", "cesium", 133, 132.91),
    el!(56, "Ba", "barium", 138, 137.33), el!(57, "La", "lanthanum", 139, 138.91),
    el!(58, "Ce", "cerium", 140, 140.12), el!(59, "Pr", "praseodymium", 141, 140.91),
    el!(60, "Nd", "neodymium", 142, 144.24), el!(61, "Pm", "promethium", 145, 145.0, "no stable isotope — the second gap"),
    el!(62, "Sm", "samarium", 152, 150.36), el!(63, "Eu", "europium", 153, 151.96),
    el!(64, "Gd", "gadolinium", 158, 157.25), el!(65, "Tb", "terbium", 159, 158.93),
    el!(66, "Dy", "dysprosium", 164, 162.50), el!(67, "Ho", "holmium", 165, 164.93),
    el!(68, "Er", "erbium", 166, 167.26), el!(69, "Tm", "thulium", 169, 168.93),
    el!(70, "Yb", "ytterbium", 174, 173.05), el!(71, "Lu", "lutetium", 175, 174.97),
    el!(72, "Hf", "hafnium", 180, 178.49), el!(73, "Ta", "tantalum", 181, 180.95),
    el!(74, "W", "tungsten", 184, 183.84), el!(75, "Re", "rhenium", 187, 186.21, "Re-187: T½ 41.6 Gy — half the age of the universe"),
    el!(76, "Os", "osmium", 192, 190.23), el!(77, "Ir", "iridium", 193, 192.22),
    el!(78, "Pt", "platinum", 195, 195.08), el!(79, "Au", "gold", 197, 196.97),
    el!(80, "Hg", "mercury", 202, 200.59), el!(81, "Tl", "thallium", 205, 204.38),
    el!(82, "Pb", "lead", 208, 207.2, "Pb-208: doubly magic (82 protons, 126 neutrons) — where all decay chains end"),
    el!(83, "Bi", "bismuth", 209, 208.98, "Bi-209: T½ 1.9e19 y — 'stable' until 2003, α-decays eventually"),
    el!(84, "Po", "polonium", 209, 209.0, "Po-209: T½ 124 y, α — Curie's element"),
    el!(85, "At", "astatine", 210, 210.0, "At-210: T½ 8.1 h — the rarest natural element (~1 g in the whole crust)"),
    el!(86, "Rn", "radon", 222, 222.0, "Rn-222: T½ 3.82 d, α — the basement gas"),
    el!(87, "Fr", "francium", 223, 223.0, "Fr-223: T½ 22 min — at most ~30 g in the crust at any instant"),
    el!(88, "Ra", "radium", 226, 226.0, "Ra-226: T½ 1600 y, α — the Curies' glow-in-the-dark"),
    el!(89, "Ac", "actinium", 227, 227.0, "Ac-227: T½ 21.8 y, β⁻"),
    el!(90, "Th", "thorium", 232, 232.04, "Th-232: T½ 14.05 Gy — every atom you hold predates the Earth"),
    el!(91, "Pa", "protactinium", 231, 231.04, "Pa-231: T½ 32.8 ky"),
    el!(92, "U", "uranium", 238, 238.03, "U-238: T½ 4.47 Gy — the age of the Earth, one half-life"),
    el!(93, "Np", "neptunium", 237, 237.0, "Np-237: T½ 2.14 My"),
    el!(94, "Pu", "plutonium", 239, 239.05, "Pu-239: T½ 24.1 ky — born in reactors, not in stars (well, almost none)"),
    el!(95, "Am", "americium", 243, 243.0, "Am-241: T½ 432 y — the smoke detector's whisper"),
    el!(96, "Cm", "curium", 247, 247.0, "Cm-247: T½ 15.6 My"),
    el!(97, "Bk", "berkelium", 247, 247.0, "Bk-247: T½ 1.4 ky"),
    el!(98, "Cf", "californium", 251, 251.0, "Cf-251: T½ 900 y"),
    el!(99, "Es", "einsteinium", 252, 252.0, "Es-252: T½ 471 d"),
    el!(100, "Fm", "fermium", 257, 257.0, "Fm-257: T½ 100 d"),
    el!(101, "Md", "mendelevium", 258, 258.0, "Md-258: T½ 51.5 d"),
    el!(102, "No", "nobelium", 259, 259.0, "No-259: T½ 58 min"),
    el!(103, "Lr", "lawrencium", 266, 266.0, "Lr-266: T½ 11 h"),
    el!(104, "Rf", "rutherfordium", 267, 267.0, "seconds"),
    el!(105, "Db", "dubnium", 268, 268.0, "days at most"),
    el!(106, "Sg", "seaborgium", 269, 269.0, "minutes"),
    el!(107, "Bh", "bohrium", 270, 270.0, "minutes"),
    el!(108, "Hs", "hassium", 277, 277.0, "seconds"),
    el!(109, "Mt", "meitnerium", 278, 278.0, "seconds"),
    el!(110, "Ds", "darmstadtium", 281, 281.0, "seconds"),
    el!(111, "Rg", "roentgenium", 282, 282.0, "seconds"),
    el!(112, "Cn", "copernicium", 285, 285.0, "seconds"),
    el!(113, "Nh", "nihonium", 286, 286.0, "seconds"),
    el!(114, "Fl", "flerovium", 289, 289.0, "seconds"),
    el!(115, "Mc", "moscovium", 290, 290.0, "under a second"),
    el!(116, "Lv", "livermorium", 293, 293.0, "under a second"),
    el!(117, "Ts", "tennessine", 294, 294.0, "under a second"),
    el!(118, "Og", "oganesson", 294, 294.0, "Og-294: T½ 0.7 ms — five atoms ever made"),
];

/// Find an element by Z, symbol, or name (case-insensitive).
pub fn find_element(q: &str) -> Option<&'static Element> {
    let q = q.trim().to_lowercase();
    if let Ok(z) = q.parse::<u32>() {
        return ELEMENTS.iter().find(|e| e.z == z);
    }
    // "carbon-14" / "carbon14" → "carbon"
    let base = q.split('-').next().unwrap_or(&q);
    let base: String = base.chars().filter(|c| c.is_alphabetic()).collect();
    ELEMENTS
        .iter()
        .find(|e| e.sym.to_lowercase() == q || e.name == base)
        .or_else(|| ELEMENTS.iter().find(|e| e.name == q))
}

// ---------- electron structure: the Madelung rule + its exceptions ----------

/// The (n+ℓ, n) filling order — the Madelung rule.
const SUBSHELL_ORDER: [(&str, u32); 19] = [
    ("1s", 2), ("2s", 2), ("2p", 6), ("3s", 2), ("3p", 6), ("4s", 2), ("3d", 10),
    ("4p", 6), ("5s", 2), ("4d", 10), ("5p", 6), ("6s", 2), ("4f", 14), ("5d", 10),
    ("6p", 6), ("7s", 2), ("5f", 14), ("6d", 10), ("7p", 6),
];

/// Elements whose true configuration breaks the Madelung rule
/// (a half-filled or filled d/f shell wins stability, or an electron
/// hops from s to d). Tails are COMPLETE above the noble-gas core —
/// Pt and Au must carry their 4f¹⁴, which a bare "5d¹⁰ 6s¹" tail would
/// leave half-filled in the Madelung core.
const CONFIG_EXCEPTIONS: &[(u32, &str)] = &[
    (24, "3d5 4s1"), (29, "3d10 4s1"), (41, "4d4 5s1"), (42, "4d5 5s1"),
    (44, "4d7 5s1"), (45, "4d8 5s1"), (46, "4d10"), (47, "4d10 5s1"),
    (57, "5d1 6s2"), (58, "4f1 5d1 6s2"), (64, "4f7 5d1 6s2"),
    (78, "4f14 5d9 6s1"), (79, "4f14 5d10 6s1"), (89, "6d1 7s2"), (90, "6d2 7s2"),
    (91, "5f2 6d1 7s2"), (92, "5f3 6d1 7s2"), (93, "5f4 6d1 7s2"),
    (96, "5f7 6d1 7s2"), (103, "5f14 7s2 7p1"),
];

/// The noble gases — where the Madelung rule is exactly right.
const NOBLE_GAS_Z: [u32; 7] = [2, 10, 18, 36, 54, 86, 118];

/// Parse a subshell string like "3d" → (n=3, capacity=10).
fn subshell(cap: &str) -> (u32, u32) {
    let n: u32 = cap.chars().next().unwrap_or('1').to_digit(10).unwrap_or(1);
    let l_cap = match cap.chars().nth(1).unwrap_or('s') {
        's' => 2, 'p' => 6, 'd' => 10, _ => 14,
    };
    (n, l_cap)
}

/// Electron configuration of element Z, as a list of (subshell, electrons).
/// Madelung order, with the 20 known exceptions corrected.
pub fn electron_config(z: u32) -> Vec<(String, u32)> {
    if let Some((_, fixed)) = CONFIG_EXCEPTIONS.iter().find(|(ez, _)| *ez == z) {
        // exception: the core is the largest noble gas below Z (Madelung
        // is exact for every noble gas), then the corrected tail — so
        // Pt/Au get their full [Xe] 4f¹⁴ core, not a truncated Madelung fill
        let noble_z = NOBLE_GAS_Z.iter().rev().find(|&&n| n < z).copied().unwrap_or(0);
        let tail: Vec<(String, u32)> = fixed
            .split_whitespace()
            .map(|s| {
                // "3d10" → ("3d", 10): subshell names are exactly n + letter
                let (name, count) = s.split_at(2);
                (name.to_string(), count.parse().unwrap_or(0))
            })
            .collect();
        let core = madelung_fill(noble_z);
        let mut out = core;
        out.extend(tail);
        return out;
    }
    madelung_fill(z)
}

/// Plain Madelung filling for `z` electrons.
fn madelung_fill(z: u32) -> Vec<(String, u32)> {
    let mut out = Vec::new();
    let mut left = z;
    for (name, cap) in SUBSHELL_ORDER {
        if left == 0 {
            break;
        }
        let take = left.min(cap);
        out.push((name.to_string(), take));
        left -= take;
    }
    out
}

/// Shell populations (K, L, M, …) from a configuration — electrons per
/// principal quantum number.
pub fn shells(config: &[(String, u32)]) -> Vec<(u32, u32)> {
    let mut per_n: Vec<(u32, u32)> = Vec::new();
    for (name, count) in config {
        let (n, _) = subshell(name);
        if let Some(slot) = per_n.iter_mut().find(|(en, _)| *en == n) {
            slot.1 += count;
        } else {
            per_n.push((n, *count));
        }
    }
    per_n.sort_by_key(|(n, _)| *n);
    per_n
}

/// The shell letters: K=1, L=2, M=3, N=4, O=5, P=6, Q=7.
pub fn shell_letter(n: u32) -> char {
    (b'K' + (n - 1) as u8) as char
}

/// Format a configuration: "1s² 2s² 2p⁶ …" with proper superscript
/// counts, subshells grouped by shell (the textbook display order:
/// 3d before 4s, even though 4s fills first).
pub fn fmt_config(config: &[(String, u32)]) -> String {
    let mut sorted = config.to_vec();
    sorted.sort_by_key(|(name, _)| {
        let (n, _) = subshell(name);
        let l = match name.chars().nth(1).unwrap_or('s') {
            's' => 0, 'p' => 1, 'd' => 2, _ => 3,
        };
        (n, l)
    });
    sorted
        .iter()
        .map(|(s, c)| format!("{}{}", s, superscript(*c)))
        .collect::<Vec<_>>()
        .join(" ")
}

/// Unicode superscripts for 0–14 (subshell counts).
fn superscript(n: u32) -> String {
    const SUP: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    n.to_string().chars().map(|d| SUP[d.to_digit(10).unwrap() as usize]).collect()
}

// ---------- the nucleus ----------

/// Weizsäcker semi-empirical mass formula: total binding energy in MeV.
/// E_B = aV·A − aS·A^(2/3) − aC·Z(Z−1)/A^(1/3) − aA·(A−2Z)²/A ± δ(pairing).
/// Fits Fe-56 to 0.4% — remarkable for five numbers.
pub fn binding_energy_mev(a: u32, z: u32) -> f64 {
    let (av, as_, ac, aa, ap) = (15.8, 18.3, 0.714, 23.2, 11.5);
    let af = a as f64;
    let zf = z as f64;
    let vol = av * af;
    let surf = as_ * af.powf(2.0 / 3.0);
    let coul = ac * zf * (zf - 1.0) / af.powf(1.0 / 3.0);
    let asym = aa * (af - 2.0 * zf).powi(2) / af;
    // pairing: even-even +δ, odd-A 0, odd-odd −δ
    let n = af - zf;
    let pair = if af % 2.0 == 0.0 && n % 2.0 == 0.0 {
        ap / af.sqrt()
    } else if af % 2.0 == 1.0 {
        0.0
    } else {
        -ap / af.sqrt()
    };
    (vol - surf - coul - asym + pair).max(0.0)
}

/// Binding energy per nucleon (MeV) — the curve that decides fission,
/// fusion, and why iron kills stars.
pub fn binding_per_nucleon(a: u32, z: u32) -> f64 {
    binding_energy_mev(a, z) / a as f64
}

/// Nuclear radius in femtometers: r = 1.2·A^(1/3) fm.
pub fn nuclear_radius_fm(a: u32) -> f64 {
    R0_FM * (a as f64).powf(1.0 / 3.0)
}

/// The most tightly bound nucleus (the peak of the curve).
pub const MOST_BOUND: (u32, u32) = (56, 26); // Fe-56, 8.79 MeV/nucleon measured

// ---------- radioactivity ----------

#[derive(Clone, Copy)]
pub struct Isotope {
    /// e.g. "carbon", "uranium"
    pub element: &'static str,
    /// mass number
    pub a: u32,
    /// half-life in seconds
    pub half_life_s: f64,
    /// human-readable half-life
    pub half_life_str: &'static str,
    /// decay mode narrative
    pub mode: &'static str,
    /// where it decays to ("" if simple)
    pub chain: &'static str,
}

const YEAR_S: f64 = 3.156e7;
const DAY_S: f64 = 86_400.0;
const HOUR_S: f64 = 3_600.0;
const MIN_S: f64 = 60.0;

/// The famous radioactive isotopes (the ones with stories).
pub const ISOTOPES: &[Isotope] = &[
    Isotope { element: "hydrogen", a: 3, half_life_s: 12.32 * YEAR_S, half_life_str: "12.32 years", mode: "β⁻", chain: "→ He-3" },
    Isotope { element: "carbon", a: 14, half_life_s: 5_730.0 * YEAR_S, half_life_str: "5,730 years", mode: "β⁻ (d → u: the atom becomes nitrogen)", chain: "→ N-14" },
    Isotope { element: "potassium", a: 40, half_life_s: 1.248e9 * YEAR_S, half_life_str: "1.25 billion years", mode: "β⁻ 89% / electron capture 11%", chain: "→ Ca-40 / Ar-40" },
    Isotope { element: "cobalt", a: 60, half_life_s: 5.271 * YEAR_S, half_life_str: "5.27 years", mode: "β⁻ + two γ (1.17, 1.33 MeV)", chain: "→ Ni-60" },
    Isotope { element: "strontium", a: 90, half_life_s: 28.79 * YEAR_S, half_life_str: "28.8 years", mode: "β⁻", chain: "→ Y-90 → Zr-90" },
    Isotope { element: "iodine", a: 131, half_life_s: 8.025 * DAY_S, half_life_str: "8.02 days", mode: "β⁻", chain: "→ Xe-131" },
    Isotope { element: "cesium", a: 137, half_life_s: 30.08 * YEAR_S, half_life_str: "30.1 years", mode: "β⁻ → Ba-137m → γ 662 keV (the Chernobyl signature)", chain: "→ Ba-137" },
    Isotope { element: "bismuth", a: 209, half_life_s: 2.01e19 * YEAR_S, half_life_str: "1.9e19 years", mode: "α", chain: "→ Tl-205" },
    Isotope { element: "radium", a: 226, half_life_s: 1_600.0 * YEAR_S, half_life_str: "1,600 years", mode: "α", chain: "→ Rn-222 → Pb-206 (in 4 more steps)" },
    Isotope { element: "radon", a: 222, half_life_s: 3.8235 * DAY_S, half_life_str: "3.82 days", mode: "α", chain: "→ Po-218" },
    Isotope { element: "thorium", a: 232, half_life_s: 14.05e9 * YEAR_S, half_life_str: "14.05 billion years", mode: "α", chain: "→ Pb-208 after 10 steps (6α + 4β⁻)" },
    Isotope { element: "uranium", a: 235, half_life_s: 704e6 * YEAR_S, half_life_str: "704 million years", mode: "α", chain: "→ Pb-207 after 11 steps (7α + 4β⁻) — the actinium series" },
    Isotope { element: "uranium", a: 238, half_life_s: 4.468e9 * YEAR_S, half_life_str: "4.47 billion years", mode: "α", chain: "→ Pb-206 after 14 steps (8α + 6β⁻) — the age of the Earth" },
    Isotope { element: "plutonium", a: 239, half_life_s: 24_110.0 * YEAR_S, half_life_str: "24,100 years", mode: "α", chain: "→ U-235 → Pb-207" },
    Isotope { element: "americium", a: 241, half_life_s: 432.2 * YEAR_S, half_life_str: "432 years", mode: "α", chain: "→ Np-237 → Pb-209" },
];

/// Look up an isotope: ("carbon", 14) or by dominant (element, a).
pub fn find_isotope(element: &str, a: u32) -> Option<&'static Isotope> {
    ISOTOPES.iter().find(|i| i.element == element && i.a == a)
}

/// Decay law: fraction remaining after `elapsed_s`.
pub fn decay_fraction(elapsed_s: f64, half_life_s: f64) -> f64 {
    if half_life_s <= 0.0 {
        return 1.0;
    }
    (-elapsed_s / half_life_s).exp2()
}

/// Age from remaining fraction: t = −T½·log₂(fraction).
pub fn decay_age_s(fraction: f64, half_life_s: f64) -> f64 {
    if fraction <= 0.0 || fraction > 1.0 {
        return f64::NAN;
    }
    -half_life_s * fraction.log2()
}

/// Activity in becquerels: A = λN = ln2·N/T½.
pub fn activity_bq(n_atoms: f64, half_life_s: f64) -> f64 {
    if half_life_s <= 0.0 {
        return 0.0;
    }
    std::f64::consts::LN_2 * n_atoms / half_life_s
}

// ---------- material → element composition ----------

/// One ingredient of a material's chemical make-up.
#[derive(Clone, Copy)]
pub struct Ingredient {
    /// element symbol, e.g. "Fe"
    pub sym: &'static str,
    /// mass fraction 0..1
    pub frac: f64,
    /// what it is, for the narrative
    pub note: &'static str,
}

/// The chemical composition of every OTD material. Mass fractions sum
/// to ~1.0 (alloys rounded to their dominant grades, organics to their
/// representative molecules: cellulose, SiO₂, polyethylene…).
pub fn composition(material: &str) -> &'static [Ingredient] {
    use Ingredient as I;
    match material {
        "iron" => &[I { sym: "Fe", frac: 1.0, note: "elemental" }],
        "steel" => &[I { sym: "Fe", frac: 0.985, note: "the metal" }, I { sym: "C", frac: 0.015, note: "the 1.5% that makes it steel" }],
        "stainless" => &[I { sym: "Fe", frac: 0.74, note: "the metal" }, I { sym: "Cr", frac: 0.18, note: "the rust-proofing oxide layer" }, I { sym: "Ni", frac: 0.08, note: "the toughener" }],
        "aluminum" => &[I { sym: "Al", frac: 1.0, note: "elemental" }],
        "copper" => &[I { sym: "Cu", frac: 1.0, note: "elemental" }],
        "brass" => &[I { sym: "Cu", frac: 0.65, note: "the metal" }, I { sym: "Zn", frac: 0.35, note: "the cheap half" }],
        "bronze" => &[I { sym: "Cu", frac: 0.88, note: "the metal" }, I { sym: "Sn", frac: 0.12, note: "the Bronze Age's difference" }],
        "gold" => &[I { sym: "Au", frac: 1.0, note: "elemental — forged in neutron star collisions" }],
        "silver" => &[I { sym: "Ag", frac: 1.0, note: "elemental" }],
        "titanium" => &[I { sym: "Ti", frac: 1.0, note: "elemental" }],
        "zinc" => &[I { sym: "Zn", frac: 1.0, note: "elemental" }],
        "lead" => &[I { sym: "Pb", frac: 1.0, note: "elemental — where decay chains go to rest" }],
        "chrome" => &[I { sym: "Cr", frac: 1.0, note: "elemental" }],
        "tungsten" => &[I { sym: "W", frac: 1.0, note: "elemental — the highest melting point (3422 °C)" }],
        "lithium" => &[I { sym: "Li", frac: 1.0, note: "elemental — one of the three Big Bang elements" }],
        "mercury" => &[I { sym: "Hg", frac: 1.0, note: "elemental — the only liquid metal at room temperature" }],
        "hydrogen" => &[I { sym: "H", frac: 1.0, note: "H₂ — the Big Bang's element, 74% of all matter by mass" }],
        "helium" => &[I { sym: "He", frac: 1.0, note: "the second Big Bang element — its nucleus is the α-particle" }],
        "nitrogen" => &[I { sym: "N", frac: 1.0, note: "N₂ — 78% of the air you breathe" }],
        "oxygen" => &[I { sym: "O", frac: 1.0, note: "O₂ — forged in stars, the third Big Bang element" }],
        "chlorine" => &[I { sym: "Cl", frac: 1.0, note: "Cl₂" }],
        "carbon" => &[I { sym: "C", frac: 1.0, note: "elemental — the atom life chose" }],
        "uranium" => &[I { sym: "U", frac: 1.0, note: "U-238 99.3% / U-235 0.7% — natural mix" }],
        "plutonium" => &[I { sym: "Pu", frac: 1.0, note: "Pu-239 — reactor-born" }],
        "thorium" => &[I { sym: "Th", frac: 1.0, note: "Th-232 — three times more common than uranium" }],
        "wood" | "oak" | "pine" | "teak" | "fabric" => &[I { sym: "C", frac: 0.44, note: "cellulose (C₆H₁₀O₅) backbone" }, I { sym: "O", frac: 0.49, note: "hydroxyl oxygen" }, I { sym: "H", frac: 0.07, note: "hydrogen" }],
        "glass" => &[I { sym: "Si", frac: 0.467, note: "the tetrahedra" }, I { sym: "O", frac: 0.533, note: "the bridges" }],
        "plastic" | "foam" => &[I { sym: "C", frac: 0.857, note: "polyethylene chains (CH₂)ₙ" }, I { sym: "H", frac: 0.143, note: "hydrogen" }],
        "rubber" => &[I { sym: "C", frac: 0.88, note: "isoprene backbone" }, I { sym: "H", frac: 0.12, note: "hydrogen" }],
        "ceramic" => &[I { sym: "Al", frac: 0.529, note: "alumina Al₂O₃" }, I { sym: "O", frac: 0.471, note: "oxygen" }],
        "concrete" => &[I { sym: "O", frac: 0.50, note: "oxides" }, I { sym: "Ca", frac: 0.30, note: "calcium silicate cement" }, I { sym: "Si", frac: 0.15, note: "aggregate silica" }, I { sym: "Al", frac: 0.03, note: "traces" }, I { sym: "Fe", frac: 0.02, note: "traces" }],
        "marble" => &[I { sym: "Ca", frac: 0.40, note: "CaCO₃" }, I { sym: "C", frac: 0.12, note: "carbonate carbon" }, I { sym: "O", frac: 0.48, note: "carbonate oxygen" }],
        "ice" | "water" | "steam" => &[I { sym: "H", frac: 0.112, note: "the H of H₂O" }, I { sym: "O", frac: 0.888, note: "the O — every atom forged in a star" }],
        "oil" | "gasoline" => &[I { sym: "C", frac: 0.84, note: "hydrocarbon chains" }, I { sym: "H", frac: 0.16, note: "hydrogen" }],
        "coal" => &[I { sym: "C", frac: 0.92, note: "the carbon" }, I { sym: "H", frac: 0.04, note: "volatiles" }, I { sym: "O", frac: 0.02, note: "moisture/oxides" }, I { sym: "S", frac: 0.02, note: "the sulfur (the acid-rain part)" }],
        "ethanol" => &[I { sym: "C", frac: 0.522, note: "C₂H₆O" }, I { sym: "O", frac: 0.348, note: "the hydroxyl" }, I { sym: "H", frac: 0.130, note: "hydrogen" }],
        "acetone" => &[I { sym: "C", frac: 0.621, note: "C₃H₆O" }, I { sym: "O", frac: 0.276, note: "the ketone" }, I { sym: "H", frac: 0.103, note: "hydrogen" }],
        "glycerin" => &[I { sym: "C", frac: 0.391, note: "C₃H₈O₃" }, I { sym: "O", frac: 0.522, note: "three hydroxyls" }, I { sym: "H", frac: 0.087, note: "hydrogen" }],
        "methane" => &[I { sym: "C", frac: 0.75, note: "CH₄" }, I { sym: "H", frac: 0.25, note: "hydrogen" }],
        "ammonia" => &[I { sym: "N", frac: 0.824, note: "NH₃" }, I { sym: "H", frac: 0.176, note: "hydrogen" }],
        "air" => &[I { sym: "N", frac: 0.755, note: "N₂" }, I { sym: "O", frac: 0.232, note: "O₂" }, I { sym: "Ar", frac: 0.013, note: "argon — the lazy 1%" }],
        _ => &[],
    }
}

// ---------- the census: grams → particles ----------

/// The subatomic census of the whole scene.
pub struct Census {
    pub total_protons: f64,
    pub total_neutrons: f64,
    pub total_electrons: f64,
    /// (element, grams) per element, scene-wide
    pub per_element: Vec<(&'static Element, f64)>,
}

/// Count the actual protons, neutrons and electrons in the scene, from
/// each part's measured mass and its material's composition. This is the
/// number `simulate: particles` quotes — not an analogy, an inventory.
pub fn scene_census(world: &World) -> Census {
    let mut per_element: Vec<(&'static Element, f64)> = Vec::new();
    for part in &world.parts {
        if part.hidden || part.mass_g <= 0.0 {
            continue;
        }
        let mat = match part.material {
            Some(m) => m.name,
            None => "plastic", // the default material when none is typed
        };
        for ing in composition(mat) {
            if let Some(el) = ELEMENTS.iter().find(|e| e.sym == ing.sym) {
                let grams = part.mass_g * ing.frac;
                if let Some(slot) = per_element.iter_mut().find(|(e, _)| e.sym == el.sym) {
                    slot.1 += grams;
                } else {
                    per_element.push((el, grams));
                }
            }
        }
    }
    // grams → moles → atoms → nucleons
    let mut total_protons = 0.0;
    let mut total_neutrons = 0.0;
    let mut total_electrons = 0.0;
    for (el, grams) in &per_element {
        let atoms = grams / el.mass_u * N_A;
        total_protons += atoms * el.z as f64;
        total_neutrons += atoms * (el.a - el.z) as f64;
        total_electrons += atoms * el.z as f64;
    }
    Census { total_protons, total_neutrons, total_electrons, per_element }
}

/// Materials in the scene whose elements are radioactive (drives
/// `simulate: decay`).
pub fn radioactive_materials(world: &World) -> Vec<String> {
    let mut out = Vec::new();
    for part in &world.parts {
        if part.hidden {
            continue;
        }
        let name = match part.material {
            Some(m) => m.name,
            None => continue,
        };
        let radio = composition(name).iter().any(|i| {
            ELEMENTS.iter().find(|e| e.sym == i.sym).map_or(false, |e| is_radioactive(e))
        });
        if radio && !out.iter().any(|n| n == name) {
            out.push(name.to_string());
        }
    }
    out
}

/// Does this element have a naturally famous radioactive isotope that
/// drives our decay narrative? (U, Pu, Th, K, C-14 traces.)
pub fn is_radioactive(el: &Element) -> bool {
    matches!(el.sym, "U" | "Pu" | "Th" | "Ra" | "Rn" | "Po" | "At" | "Fr" | "Ac" | "Pa" | "Np" | "Am" | "Tc" | "Pm" | "Cm" | "Bk" | "Cf" | "Es" | "Fm" | "Md" | "No" | "Lr")
        || matches!(el.sym, "K" | "C" | "Rb" | "In" | "Re" | "La" | "Nd" | "Sm" | "Gd" | "Lu" | "Yb" | "Bi")
}

// ---------- the spectral lines ----------

/// Rydberg: wavelength (nm) of the hydrogen transition n_hi → n_lo.
pub fn rydberg_nm(n_lo: u32, n_hi: u32) -> f64 {
    if n_lo >= n_hi || n_lo == 0 {
        return f64::NAN;
    }
    let inv = RYDBERG_H * (1.0 / (n_lo as f64).powi(2) - 1.0 / (n_hi as f64).powi(2));
    1.0e9 / inv
}

/// Named hydrogen series (the ones in the textbooks).
pub const HYDROGEN_SERIES: &[(&str, u32, &str)] = &[
    ("Lyman", 1, "ultraviolet — the series that ionizes DNA"),
    ("Balmer", 2, "visible — the red Hα, cyan Hβ, blue-violet Hγ"),
    ("Paschen", 3, "infrared"),
    ("Brackett", 4, "infrared"),
    ("Pfund", 5, "far infrared"),
];

// ---------- the sims ----------

/// `simulate: atom` — the atomic census: every material's element,
/// electron shells, nucleon counts, binding energy, and the size ratio
/// that makes matter mostly empty space.
pub fn atom_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine { kind: LineKind::Sim, text: "ATOMIC STRUCTURE — grams become particles: the census of this scene".into() });

    let census = scene_census(world);
    if census.per_element.is_empty() {
        out.push(ConsoleLine { kind: LineKind::Info, text: "  no measurable matter — build a part with a material and ask again".into() });
        return out;
    }

    for (el, grams) in &census.per_element {
        let atoms = grams / el.mass_u * N_A;
        let config = electron_config(el.z);
        let shell_str = shells(&config)
            .iter()
            .map(|(n, c)| format!("{}{}", shell_letter(*n), c))
            .collect::<Vec<_>>()
            .join(" ");
        let z = el.z;
        let a = el.a;
        let n_n = a - z;
        let be = binding_per_nucleon(a, z);
        let fe_peak = binding_per_nucleon(MOST_BOUND.0, MOST_BOUND.1);
        let r_atom = 0.5 + z as f64 * 0.045; // ~Å, crude but honest to an order
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("  {} ({}, Z={}): {:.3} g = {} atoms", el.name, el.sym, z, grams, fmt_sci(atoms)) });
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("    nucleus: {} protons + {} neutrons ({}-{}) — {} electrons around it: {}  [{}]", z, n_n, el.sym, a, z, shell_str, fmt_config(&config)) });
        let binding_line = if be == 0.0 {
            "a lone proton has nothing to bind — hydrogen fuses with everything (the Sun's first step)".to_string()
        } else {
            format!("binding energy {:.2} MeV/nucleon ({:+.2}% vs iron-56's {:.2} — {}) ", be, (be / fe_peak - 1.0) * 100.0, fe_peak, if be < fe_peak - 0.3 { "this nucleus could still yield energy by fusing toward iron" } else if be < fe_peak + 0.3 { "sitting at the peak — fusion and fission both lose from here" } else { "above the peak — fission pays" })
        };
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("    {}", binding_line) });
        let r_nuc = nuclear_radius_fm(a);
        let ratio = 1.0e5 / r_nuc.max(1.0); // atom ≈ 1 Å = 1e5 fm
        let marble_m = 0.01 * ratio; // nucleus as a 1 cm marble → electrons at
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("    size: nucleus {:.1} fm vs atom ~{:.2} Å — 1:{:.0}. If the nucleus were a 1 cm marble on your desk, the electrons would be {} away. Matter is overwhelmingly empty; solidity is electric fields refusing to overlap (the Pauli principle).", r_nuc, r_atom, ratio, if marble_m >= 990.0 { "a kilometer".to_string() } else if marble_m >= 90.0 { format!("{} m", (marble_m / 10.0).round() * 10.0) } else { "a city block".to_string() }) });
        if !el.note.is_empty() {
            out.push(ConsoleLine { kind: LineKind::Info, text: format!("    note: {}", el.note) });
        }
    }

    // hydrogen bonus: the spectral fingerprint
    if census.per_element.iter().any(|(e, _)| e.sym == "H") {
        let ha = rydberg_nm(2, 3);
        let hb = rydberg_nm(2, 4);
        out.push(ConsoleLine { kind: LineKind::Info, text: format!("  hydrogen's fingerprint (Rydberg, 1/λ = R(1/n₁²−1/n₂²)): Hα {:.1} nm (the red of nebulae), Hβ {:.1} nm (cyan) — Balmer's 1885 formula, the Rosetta stone of atoms", ha, hb) });
    }

    // the totals
    let e = super::particles::E_CHARGE;
    out.push(ConsoleLine { kind: LineKind::Info, text: format!("  SCENE TOTALS: {} protons, {} neutrons, {} electrons", fmt_sci(census.total_protons), fmt_sci(census.total_neutrons), fmt_sci(census.total_electrons)) });
    let net = (census.total_protons - census.total_electrons) * e;
    out.push(ConsoleLine { kind: LineKind::Info, text: format!("    charge ledger: +{:.0} C of proton charge, −{:.0} C of electron charge — net {} C, exact to the last electron. The universe balances its books.", census.total_protons * e, census.total_electrons * e, if net == 0.0 { "0.000".to_string() } else { format!("{:.3e}", net) }) });
    out.push(ConsoleLine { kind: LineKind::Info, text: "    every proton and electron is 13.8 billion years old — older than the Earth, the Sun, and every atom of them was inside a star (or the first three minutes) first".into() });
    out
}

/// `simulate: decay` — half-lives, decay modes, and the LIVE activity of
/// every radioactive part, computed from its actual atom count.
pub fn decay_sim(world: &World) -> Vec<ConsoleLine> {
    let mut out = Vec::new();
    out.push(ConsoleLine { kind: LineKind::Sim, text: "RADIOACTIVE DECAY — the half-lives and live activity of this scene".into() });

    let mut any = false;
    for part in &world.parts {
        if part.hidden || part.mass_g <= 0.0 {
            continue;
        }
        let mat = match part.material {
            Some(m) => m.name,
            None => continue,
        };
        // the radioactive isotope this material carries
        let iso: Option<&'static Isotope> = match mat {
            "uranium" => find_isotope("uranium", 238),
            "plutonium" => find_isotope("plutonium", 239),
            "thorium" => find_isotope("thorium", 232),
            _ => None,
        };
        if let Some(iso) = iso {
            any = true;
            if let Some(el) = find_element(iso.element) {
                let molar_g = iso.a as f64; // grams per mole ≈ mass number
                let n_atoms = part.mass_g / molar_g * N_A;
                let bq = activity_bq(n_atoms, iso.half_life_s);
                out.push(ConsoleLine { kind: LineKind::Info, text: format!("  '{}' ({} {:.1} g of {}-{}):", part.name, mat, part.mass_g, el.sym, iso.a) });
                out.push(ConsoleLine { kind: LineKind::Info, text: format!("    {} atoms of {}-{}, T½ {} — {} {}", fmt_sci(n_atoms), el.sym, iso.a, iso.half_life_str, iso.mode, iso.chain) });
                out.push(ConsoleLine { kind: LineKind::Info, text: format!("    activity RIGHT NOW: A = λN = {} — {} decays per second are happening inside this part as it sits there, and nothing you do can slow or speed them", fmt_bq(bq), fmt_count(bq)) });
                // a time-travel line
                let t100 = 100.0 * YEAR_S;
                let frac = decay_fraction(t100, iso.half_life_s);
                out.push(ConsoleLine { kind: LineKind::Info, text: format!("    after 100 years: {:.4}% still {}-{} ({:.6} half-lives) — decay is the one clock with no snooze button", frac * 100.0, el.sym, iso.a, 100.0 * YEAR_S / iso.half_life_s) });
            }
        }
    }

    // trace radioactivity in organic materials (C-14, K-40)
    let organic: Vec<&Part> = world
        .parts
        .iter()
        .filter(|p| {
            let m = p.material.map(|mm| mm.name).unwrap_or("");
            matches!(m, "wood" | "oak" | "pine" | "teak" | "fabric" | "coal" | "plastic" | "rubber" | "oil" | "gasoline" | "ethanol" | "acetone" | "glycerin" | "water" | "ice")
        })
        .collect();
    if !organic.is_empty() {
        let mut carbon_g = 0.0;
        let mut potassium_g = 0.0;
        for p in &organic {
            for ing in composition(p.material.map(|m| m.name).unwrap_or("")) {
                match ing.sym {
                    "C" => carbon_g += p.mass_g * ing.frac,
                    "K" => potassium_g += p.mass_g * ing.frac,
                    _ => {}
                }
            }
        }
        if carbon_g > 0.0 {
            any = true;
            let c14_fraction = 1.2e-12; // modern carbon
            let n_c = carbon_g / 12.011 * N_A;
            let n_c14 = n_c * c14_fraction;
            let iso = find_isotope("carbon", 14).unwrap();
            let bq = activity_bq(n_c14, iso.half_life_s);
            out.push(ConsoleLine { kind: LineKind::Info, text: format!("  trace radioactivity (the quiet kind): {:.1} g of modern carbon → {} C-14 atoms (1.2 per trillion) → {} of soft β⁻ — this is the clock archaeologists read; after 5,730 years, half of it is gone", carbon_g, fmt_sci(n_c14), fmt_bq(bq)) });
        }
        let _ = potassium_g; // K-40 appears in concrete/wood ash narratives later
    }

    if !any {
        out.push(ConsoleLine { kind: LineKind::Info, text: "  nothing radioactive here — try material: uranium (or plutonium, thorium), or a wooden part for its trace C-14".into() });
    } else {
        out.push(ConsoleLine { kind: LineKind::Info, text: "  the quantum deck has no memory: each nucleus decays with fixed probability per second, regardless of age, temperature, or pressure — Rutherford found the law in 1900 and nothing since has bent it".into() });
    }
    out
}

/// Human-friendly big counts: 2.42e24 rather than 2416004942686255948234752.000.
fn fmt_sci(x: f64) -> String {
    if x == 0.0 {
        return "0".into();
    }
    let exp = x.abs().log10().floor() as i32;
    let mant = x / 10f64.powi(exp);
    format!("{:.2}e{}", mant, exp)
}

/// Human-friendly becquerels.
fn fmt_bq(bq: f64) -> String {
    if bq >= 1e9 { format!("{:.2} GBq", bq / 1e9) }
    else if bq >= 1e6 { format!("{:.2} MBq", bq / 1e6) }
    else if bq >= 1e3 { format!("{:.2} kBq", bq / 1e3) }
    else if bq >= 1.0 { format!("{:.1} Bq", bq) }
    else { format!("{:.2} Bq", bq) }
}

/// Human-friendly decay rate words.
fn fmt_count(bq: f64) -> String {
    if bq >= 1e9 { format!("{:.0} billion", bq / 1e9) }
    else if bq >= 1e6 { format!("{:.0} million", bq / 1e6) }
    else if bq >= 1e3 { format!("{:.0} thousand", bq / 1e3) }
    else { format!("{:.0}", bq) }
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn table_complete() {
        assert_eq!(ELEMENTS.len(), 118);
        for (i, e) in ELEMENTS.iter().enumerate() {
            assert_eq!(e.z as usize, i + 1, "Z must be contiguous");
        }
    }

    #[test]
    fn element_lookup() {
        assert_eq!(find_element("Fe").unwrap().z, 26);
        assert_eq!(find_element("iron").unwrap().z, 26);
        assert_eq!(find_element("26").unwrap().z, 26);
        assert_eq!(find_element("gold").unwrap().z, 79);
        assert_eq!(find_element("carbon-14").unwrap().z, 6);
        assert!(find_element("unobtainium").is_none());
    }

    #[test]
    fn configs_madelung_and_exceptions() {
        // H: 1s1
        assert_eq!(fmt_config(&electron_config(1)), "1s¹");
        // He: 1s2
        assert_eq!(fmt_config(&electron_config(2)), "1s²");
        // Fe: ...4s2 3d6 in FILL order; shells K2 L8 M14 N2
        let fe = electron_config(26);
        assert_eq!(fe.last().unwrap().0, "3d");
        assert_eq!(fe.last().unwrap().1, 6);
        let fe_shells = shells(&fe);
        assert_eq!(fe_shells, vec![(1, 2), (2, 8), (3, 14), (4, 2)]);
        // display order: 3d groups before 4s (textbook convention)
        let fe_disp = fmt_config(&fe);
        assert!(fe_disp.ends_with("3d⁶ 4s²"), "Fe displays as …3d⁶ 4s², got {}", fe_disp);
        // Cu exception: 3d10 4s1, not 3d9 4s2
        let cu = electron_config(29);
        assert_eq!(cu.iter().rev().take(2).map(|(s, c)| (s.as_str(), *c)).collect::<Vec<_>>(), vec![("4s", 1), ("3d", 10)]);
        // Pd exception: ends 4d10, no 5s
        let pd = electron_config(46);
        assert_eq!(pd.last().unwrap().0, "4d");
        assert_eq!(pd.last().unwrap().1, 10);
        // Au: [Xe] 4f14 5d10 6s1 — the exception tail carries its own 4f14
        let au = electron_config(79);
        assert_eq!(au.last().unwrap().0, "6s");
        assert_eq!(au.last().unwrap().1, 1);
        let au_disp = fmt_config(&au);
        assert!(au_disp.contains("4f¹⁴") && au_disp.ends_with("5d¹⁰ 6s¹"), "Au config: {}", au_disp);
        assert!(!au_disp.contains("6s²"), "Au must have 6s¹ only: {}", au_disp);
        // Pt: [Xe] 4f14 5d9 6s1
        let pt_disp = fmt_config(&electron_config(78));
        assert!(pt_disp.contains("4f¹⁴") && pt_disp.ends_with("5d⁹ 6s¹"), "Pt config: {}", pt_disp);
        // total electrons = Z for all elements
        for e in ELEMENTS {
            let total: u32 = electron_config(e.z).iter().map(|(_, c)| c).sum();
            assert_eq!(total, e.z, "config of {} must have {} electrons", e.name, e.z);
        }
    }

    #[test]
    fn shell_letters() {
        assert_eq!(shell_letter(1), 'K');
        assert_eq!(shell_letter(4), 'N');
        assert_eq!(shell_letter(7), 'Q');
    }

    #[test]
    fn weizsacker_matches_textbook() {
        // Fe-56: 8.76 MeV/A predicted vs 8.79 measured (< 0.5% error)
        let fe = binding_per_nucleon(56, 26);
        assert!((fe - 8.76).abs() < 0.05, "Fe-56 binding: {}", fe);
        // U-238: 7.60 vs 7.57 measured
        let u = binding_per_nucleon(238, 92);
        assert!((u - 7.60).abs() < 0.05, "U-238 binding: {}", u);
        // Pb-208 (doubly magic): 7.83 vs 7.87 measured
        let pb = binding_per_nucleon(208, 82);
        assert!((pb - 7.83).abs() < 0.05, "Pb-208 binding: {}", pb);
        // He-4: Weizsäcker famously under-binds the lightest nuclei
        // (it's fit to medium/heavy A) — 5.49 vs 7.07 measured is the
        // KNOWN limitation, documented, not a bug
        let he = binding_per_nucleon(4, 2);
        assert!((he - 5.49).abs() < 0.1, "He-4 via Weizsäcker (under-binds light nuclei): {}", he);
        // Fe-56 sits at the max of the curve among these
        assert!(fe > u && fe > pb);
        // nuclear radii: U-238 ≈ 7.4 fm
        assert!((nuclear_radius_fm(238) - 7.44).abs() < 0.1);
    }

    #[test]
    fn decay_math() {
        // one half-life → exactly half
        assert!((decay_fraction(5730.0 * YEAR_S, 5730.0 * YEAR_S) - 0.5).abs() < 1e-12);
        // 25% left → two half-lives
        let age = decay_age_s(0.25, 5730.0 * YEAR_S);
        assert!((age / YEAR_S / 5730.0 - 2.0).abs() < 1e-9);
        // 1 g U-238 → 12.4 kBq (the textbook number)
        let n = 1.0 / 238.0 * N_A;
        let bq = activity_bq(n, 4.468e9 * YEAR_S);
        assert!((bq - 12_400.0).abs() < 500.0, "U-238 specific activity: {} Bq/g", bq);
        // 1 g modern carbon → ~0.23 Bq of C-14 (13.6 dpm)
        let nc = 1.0 / 12.011 * N_A * 1.2e-12;
        let bq_c = activity_bq(nc, 5730.0 * YEAR_S);
        assert!((bq_c - 0.23).abs() < 0.02, "C-14 activity: {} Bq/g", bq_c);
    }

    #[test]
    fn rydberg_lines() {
        // Hα 656.3 nm (Balmer 3→2)
        assert!((rydberg_nm(2, 3) - 656.5).abs() < 0.5);
        // Hβ 486.1 nm (4→2)
        assert!((rydberg_nm(2, 4) - 486.3).abs() < 0.5);
        // Lyman-α 121.6 nm (2→1)
        assert!((rydberg_nm(1, 2) - 121.6).abs() < 0.2);
    }

    #[test]
    fn composition_sums() {
        // every composition's fractions should sum to ~1
        for mat in ["iron", "steel", "stainless", "brass", "wood", "glass", "water", "air", "concrete", "marble", "coal", "ethanol"] {
            let sum: f64 = composition(mat).iter().map(|i| i.frac).sum();
            assert!((sum - 1.0).abs() < 0.01, "{} fractions sum to {}", mat, sum);
        }
        assert!(composition("unobtainium").is_empty());
        // wood has carbon
        assert!(composition("wood").iter().any(|i| i.sym == "C"));
    }

    #[test]
    fn isotope_table() {
        assert_eq!(find_isotope("carbon", 14).unwrap().a, 14);
        assert_eq!(find_isotope("uranium", 238).unwrap().half_life_s / (1e9 * YEAR_S), 4.468);
        assert!(find_isotope("iron", 56).is_none(), "Fe-56 is stable — not in the decay table");
        // 5730 y in the table is really 5730 years in seconds
        let c14 = find_isotope("carbon", 14).unwrap();
        assert!((c14.half_life_s / YEAR_S - 5730.0).abs() < 1.0);
    }

    #[test]
    fn radioactive_detection() {
        let u = find_element("uranium").unwrap();
        let fe = find_element("iron").unwrap();
        assert!(is_radioactive(u));
        assert!(!is_radioactive(fe));
    }
}
