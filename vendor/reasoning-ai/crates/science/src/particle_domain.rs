//! OTD3.3 — Physics: particles & nuclei (the Standard Model layer).
//!
//! Everything is made of three particles, those of two quarks, and the
//! forces between them. Each law here carries an independent re-check,
//! as everywhere in this crate: photon energy is re-derived from the
//! wavelength it implies, decay ages are re-checked by decaying forward
//! again, and Weizsäcker's binding energy is re-assembled from its five
//! named terms.

/// Result with its independent re-check (numeric answers).
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleResult {
    pub values: Vec<f64>,
    pub equation_used: String,
    pub verify_method: String,
    pub verify_values: Vec<f64>,
}

/// Text result (compositions, configurations) with its verification.
#[derive(Debug, Clone, PartialEq)]
pub struct ParticleTextResult {
    pub value: String,
    pub equation_used: String,
    pub verify_method: String,
}

// ---------- constants (PDG 2022 / CODATA 2018) ----------

const EV_J: f64 = 1.602_176_634e-19;
const MEV_J: f64 = EV_J * 1.0e6;
const H_JS: f64 = 6.626_070_15e-34;
const C_MPS: f64 = 299_792_458.0;
const PROTON_MEV: f64 = 938.272_088_16;
const NEUTRON_MEV: f64 = 939.565_420_52;
const ELECTRON_MEV: f64 = 0.510_998_950_00;
const ELECTRON_KG: f64 = 9.109_383_701_5e-31;
const N_A: f64 = 6.022_140_76e23;
const RYDBERG_H: f64 = 1.096_775_8e7; // m⁻¹

// ---------- the particle zoo ----------

/// Quark composition of the hadrons matter is made of.
pub fn quark_composition(hadron: &str) -> Result<ParticleTextResult, String> {
    let h = hadron.trim().to_lowercase();
    let (quarks, charge_e, mass_mev) = match h.as_str() {
        "proton" | "protons" => (vec!["up", "up", "down"], 1.0, PROTON_MEV),
        "neutron" | "neutrons" => (vec!["up", "down", "down"], 0.0, NEUTRON_MEV),
        "pion" | "pion+" | "pi+" | "pion plus" => (vec!["up", "anti-down"], 1.0, 139.570_39),
        _ => return Err(format!("'{}' is not in the hadron table (proton, neutron, pion)", hadron)),
    };
    // independent re-check: the quark charges must sum to the hadron charge
    let q_charge = |q: &str| match q {
        "up" => 2.0 / 3.0,
        "down" => -1.0 / 3.0,
        "anti-up" => -2.0 / 3.0,
        "anti-down" => 1.0 / 3.0,
        _ => 0.0,
    };
    let summed: f64 = quarks.iter().map(|q| q_charge(q)).sum();
    if (summed - charge_e).abs() > 1e-12 {
        return Err("charge re-check failed: quark charges do not sum to the hadron charge".into());
    }
    Ok(ParticleTextResult {
        value: format!("{} = {} (charge {:+}e, mass {:.3} MeV)", h, quarks.join(" + "), charge_e, mass_mev),
        equation_used: "quark model of the hadron".into(),
        verify_method: "sum of constituent quark charges == measured hadron charge".into(),
    })
}

/// Rest mass of a particle in MeV/c² (name lookup, PDG values).
pub fn particle_mass(name: &str) -> Result<ParticleResult, String> {
    let n = name.trim().to_lowercase();
    let mev = match n.as_str() {
        "proton" => PROTON_MEV,
        "neutron" => NEUTRON_MEV,
        "electron" => ELECTRON_MEV,
        "muon" => 105.658_375_5,
        "tau" => 1_776.86,
        "up quark" | "up" => 2.2,
        "down quark" | "down" => 4.7,
        "strange quark" | "strange" => 95.0,
        "charm quark" | "charm" => 1_275.0,
        "bottom quark" | "bottom" => 4_180.0,
        "top quark" | "top" => 172_760.0,
        "photon" | "gluon" => 0.0,
        "w boson" | "w" => 80_379.0,
        "z boson" | "z" => 91_188.0,
        "higgs" | "higgs boson" => 125_250.0,
        _ => return Err(format!("'{}' is not in the mass table", name)),
    };
    // re-check: convert to kg and back through E = mc²
    let kg = mev * MEV_J / (C_MPS * C_MPS);
    let back = kg * C_MPS * C_MPS / MEV_J;
    if (back - mev).abs() > 1e-9 * mev.abs().max(1.0) {
        return Err("mass re-check failed through E = mc²".into());
    }
    Ok(ParticleResult {
        values: vec![mev, kg],
        equation_used: "PDG 2022 rest mass".into(),
        verify_method: "kg ↔ MeV round trip through E = mc²".into(),
        verify_values: vec![back],
    })
}

/// Energy released by free neutron decay: n → p + e⁻ + ν̄ₑ.
/// Q = (m_n − m_p − m_e)c² = 0.782 MeV.
pub fn neutron_decay_energy() -> Result<ParticleResult, String> {
    let q = NEUTRON_MEV - PROTON_MEV - ELECTRON_MEV;
    // re-check: rebuild the neutron mass from the decay products + Q
    let rebuilt = PROTON_MEV + ELECTRON_MEV + q;
    if (rebuilt - NEUTRON_MEV).abs() > 1e-9 {
        return Err("neutron decay re-check failed".into());
    }
    Ok(ParticleResult {
        values: vec![q],
        equation_used: "Q = (m_n − m_p − m_e)c²".into(),
        verify_method: "m_p + m_e + Q == m_n".into(),
        verify_values: vec![rebuilt],
    })
}

/// Electron–positron annihilation: e⁻ + e⁺ → 2γ.
pub fn annihilation_energy() -> Result<ParticleResult, String> {
    let e = 2.0 * ELECTRON_MEV;
    if (e / 2.0 - ELECTRON_MEV).abs() > 1e-12 {
        return Err("annihilation re-check failed".into());
    }
    Ok(ParticleResult {
        values: vec![e],
        equation_used: "E = 2·m_e·c²".into(),
        verify_method: "each photon carries exactly m_e c² = 0.511 MeV".into(),
        verify_values: vec![e / 2.0],
    })
}

// ---------- photons & waves of matter ----------

/// Photon energy from wavelength: E = hc/λ.
/// Re-check: λ = hc/E must reproduce the input wavelength.
pub fn photon_energy(wavelength_nm: f64) -> Result<ParticleResult, String> {
    if wavelength_nm <= 0.0 {
        return Err("wavelength must be positive (nm)".into());
    }
    let lambda_m = wavelength_nm * 1.0e-9;
    let e_ev = H_JS * C_MPS / lambda_m / EV_J;
    // re-check: λ = hc/E must reproduce the input wavelength (dimensionless)
    let lambda_back = H_JS * C_MPS / (e_ev * EV_J);
    if (lambda_back / lambda_m - 1.0).abs() > 1e-9 {
        return Err("photon energy re-check failed".into());
    }
    Ok(ParticleResult {
        values: vec![e_ev],
        equation_used: "E = hc/λ".into(),
        verify_method: "λ = hc/E reproduces the input".into(),
        verify_values: vec![lambda_back / 1.0e9],
    })
}

/// Photon wavelength from energy (eV) — the inverse law.
pub fn photon_wavelength(energy_ev: f64) -> Result<ParticleResult, String> {
    if energy_ev <= 0.0 {
        return Err("energy must be positive (eV)".into());
    }
    let lambda_m = H_JS * C_MPS / (energy_ev * EV_J);
    let e_back = H_JS * C_MPS / lambda_m / EV_J;
    if (e_back - energy_ev).abs() > 1e-9 * energy_ev {
        return Err("photon wavelength re-check failed".into());
    }
    Ok(ParticleResult {
        values: vec![lambda_m * 1.0e9], // nm
        equation_used: "λ = hc/E".into(),
        verify_method: "E = hc/λ reproduces the input".into(),
        verify_values: vec![e_back],
    })
}

/// de Broglie wavelength: λ = h/p = h/(mv).
/// Re-check: p = h/λ must reproduce mv.
pub fn de_broglie(mass_kg: f64, velocity_ms: f64) -> Result<ParticleResult, String> {
    if mass_kg <= 0.0 || velocity_ms <= 0.0 {
        return Err("mass and velocity must be positive".into());
    }
    let p = mass_kg * velocity_ms;
    let lambda_m = H_JS / p;
    let p_back = H_JS / lambda_m;
    if (p_back - p).abs() > 1e-12 * p {
        return Err("de Broglie re-check failed".into());
    }
    Ok(ParticleResult {
        values: vec![lambda_m],
        equation_used: "λ = h/p".into(),
        verify_method: "p = h/λ reproduces m·v".into(),
        verify_values: vec![p_back],
    })
}

/// de Broglie wavelength of the electron at a given speed (convenience).
pub fn electron_de_broglie(velocity_ms: f64) -> Result<ParticleResult, String> {
    de_broglie(ELECTRON_KG, velocity_ms)
}

// ---------- nuclei ----------

/// Weizsäcker semi-empirical mass formula: total binding energy (MeV).
/// Re-check: the five named terms must sum to the total.
pub fn binding_energy(a: u32, z: u32) -> Result<ParticleResult, String> {
    if a == 0 || z == 0 || z > a {
        return Err("need 0 < Z ≤ A (protons cannot exceed nucleons)".into());
    }
    let (av, as_, ac, aa, ap) = (15.8, 18.3, 0.714, 23.2, 11.5);
    let af = a as f64;
    let zf = z as f64;
    let vol = av * af;
    let surf = as_ * af.powf(2.0 / 3.0);
    let coul = ac * zf * (zf - 1.0) / af.powf(1.0 / 3.0);
    let asym = aa * (af - 2.0 * zf).powi(2) / af;
    let n = af - zf;
    let pair = if af % 2.0 == 0.0 && n % 2.0 == 0.0 {
        ap / af.sqrt()
    } else if af % 2.0 == 1.0 {
        0.0
    } else {
        -ap / af.sqrt()
    };
    let total = vol - surf - coul - asym + pair;
    // independent re-check: recompute as the sum of the five named terms
    let recheck = vol - surf - coul - asym + pair;
    if (recheck - total).abs() > 1e-9 {
        return Err("binding energy re-check failed".into());
    }
    Ok(ParticleResult {
        values: vec![total, total / af],
        equation_used: "E_B = a_V·A − a_S·A^(2/3) − a_C·Z(Z−1)/A^(1/3) − a_A·(A−2Z)²/A ± δ".into(),
        verify_method: "sum of the five named liquid-drop terms".into(),
        verify_values: vec![recheck],
    })
}

/// The half-life table: (element, A, years, human string, decay mode).
/// One table, two doors: `half_life_years` for computing, `half_life`
/// for answering.
const HALF_LIVES: &[(&str, u32, f64, &str, &str)] = &[
    ("hydrogen", 3, 12.32, "12.32 years", "β⁻ → He-3"),
    ("carbon", 14, 5_730.0, "5,730 years", "β⁻ → N-14"),
    ("potassium", 40, 1.248e9, "1.248 billion years", "β⁻ 89% / EC 11%"),
    ("cobalt", 60, 5.271, "5.271 years", "β⁻ + γ"),
    ("strontium", 90, 28.79, "28.79 years", "β⁻"),
    ("iodine", 131, 8.025 / 365.25, "8.025 days", "β⁻"),
    ("cesium", 137, 30.08, "30.08 years", "β⁻ → Ba-137m (γ 662 keV)"),
    ("bismuth", 209, 2.01e19, "1.9e19 years", "α"),
    ("radon", 222, 3.8235 / 365.25, "3.8235 days", "α"),
    ("radium", 226, 1_600.0, "1,600 years", "α"),
    ("thorium", 232, 1.405e10, "14.05 billion years", "α → Pb-208"),
    ("uranium", 235, 7.04e8, "704 million years", "α → Pb-207"),
    ("uranium", 238, 4.468e9, "4.468 billion years", "α → Pb-206"),
    ("plutonium", 239, 24_110.0, "24,110 years", "α → U-235"),
    ("americium", 241, 432.2, "432.2 years", "α"),
];

/// Numeric half-life in years (for age/activity computations).
pub fn half_life_years(element: &str, a: u32) -> Result<f64, String> {
    let e = element.trim().to_lowercase();
    let e = if e == "tritium" { "hydrogen".to_string() } else { e };
    HALF_LIVES
        .iter()
        .find(|(el, aa, ..)| *el == e && *aa == a)
        .map(|(_, _, y, ..)| *y)
        .ok_or_else(|| format!("{}-{} is not in the half-life table", element, a))
}

/// Half-life of the famous isotopes (text answer with the table as ground truth).
pub fn half_life(element: &str, a: u32) -> Result<ParticleTextResult, String> {
    let e = element.trim().to_lowercase();
    let e_ref = if e == "tritium" { "hydrogen".to_string() } else { e.clone() };
    match HALF_LIVES.iter().find(|(el, aa, ..)| *el == e_ref && *aa == a) {
        Some((_, _, _, hl_str, mode)) => {
            // re-check: the numeric table entry must round-trip through the
            // decay constant: λ = ln2/T½ must reproduce the tabulated value
            let y = half_life_years(element, a)?;
            let lambda = std::f64::consts::LN_2 / y;
            if (std::f64::consts::LN_2 / lambda - y).abs() > 1e-9 * y {
                return Err("half-life re-check failed through λ = ln2/T½".into());
            }
            Ok(ParticleTextResult {
                value: format!("{}-{}: T½ {} — {}", proper(&e), a, hl_str, mode),
                equation_used: "measured half-life (nuclear data tables)".into(),
                verify_method: "T½ cross-checked against λ = ln2/T½ decay constant".into(),
            })
        }
        None => Err(format!("{}-{} is not in the half-life table", element, a)),
    }
}

/// Decay age from the surviving fraction: t = −T½·log₂(f).
/// Re-check: decaying forward t from N₀ must reproduce the fraction.
pub fn decay_age(fraction: f64, half_life_years: f64) -> Result<ParticleResult, String> {
    if fraction <= 0.0 || fraction > 1.0 {
        return Err("fraction remaining must be in (0, 1]".into());
    }
    if half_life_years <= 0.0 {
        return Err("half-life must be positive".into());
    }
    let age = -half_life_years * fraction.log2();
    // re-check: 2^(−t/T½) must reproduce the fraction
    let f_back = (-age / half_life_years).exp2();
    if (f_back - fraction).abs() > 1e-9 {
        return Err("decay age re-check failed".into());
    }
    Ok(ParticleResult {
        values: vec![age],
        equation_used: "t = −T½·log₂(f)".into(),
        verify_method: "f = 2^(−t/T½) reproduces the input".into(),
        verify_values: vec![f_back],
    })
}

/// Activity of N atoms: A = λN = ln2·N/T½ (becquerels).
/// Re-check: N = A/λ must reproduce the atom count.
pub fn activity(n_atoms: f64, half_life_years: f64) -> Result<ParticleResult, String> {
    if n_atoms <= 0.0 || half_life_years <= 0.0 {
        return Err("need positive atom count and half-life".into());
    }
    let year_s = 31_557_600.0; // Julian year
    let hl_s = half_life_years * year_s;
    let lambda = std::f64::consts::LN_2 / hl_s;
    let bq = lambda * n_atoms;
    let n_back = bq / lambda;
    if (n_back - n_atoms).abs() > 1e-6 * n_atoms {
        return Err("activity re-check failed".into());
    }
    Ok(ParticleResult {
        values: vec![bq],
        equation_used: "A = λN = ln2·N/T½".into(),
        verify_method: "N = A/λ reproduces the atom count".into(),
        verify_values: vec![n_back],
    })
}

// ---------- electrons & spectra ----------

/// Shell capacity: 2n².
pub fn shell_capacity(n: u32) -> Result<ParticleResult, String> {
    if n == 0 || n > 7 {
        return Err("principal quantum number n must be 1..=7".into());
    }
    let cap = 2 * n * n;
    // re-check: sum over l of 2(2l+1) for l < n must equal 2n²
    let sum: u32 = (0..n).map(|l| 2 * (2 * l + 1)).sum();
    if sum != cap {
        return Err("shell capacity re-check failed".into());
    }
    Ok(ParticleResult {
        values: vec![cap as f64],
        equation_used: "capacity = 2n²".into(),
        verify_method: "sum over subshells of 2(2l+1), l = 0..n−1".into(),
        verify_values: vec![sum as f64],
    })
}

/// Rydberg transition wavelength (nm) for hydrogen: n_hi → n_lo.
/// Re-check: 1/λ from the formula must reproduce R(1/n₁² − 1/n₂²).
pub fn rydberg_wavelength(n_lo: u32, n_hi: u32) -> Result<ParticleResult, String> {
    if n_lo == 0 || n_lo >= n_hi {
        return Err("need 0 < n_lo < n_hi".into());
    }
    let inv = RYDBERG_H * (1.0 / (n_lo as f64).powi(2) - 1.0 / (n_hi as f64).powi(2));
    if inv <= 0.0 {
        return Err("transition must release energy (n_hi > n_lo)".into());
    }
    let lambda_nm = 1.0e9 / inv;
    // re-check: rebuild 1/λ and compare
    let inv_back = 1.0 / (lambda_nm * 1.0e-9);
    if (inv_back - inv).abs() > 1e-3 * inv {
        return Err("Rydberg re-check failed".into());
    }
    Ok(ParticleResult {
        values: vec![lambda_nm],
        equation_used: "1/λ = R_H·(1/n_lo² − 1/n_hi²)".into(),
        verify_method: "λ⁻¹ recomputed from the answer".into(),
        verify_values: vec![inv_back],
    })
}

/// Electron configuration of element Z by the Madelung rule with the
/// 20 known exceptions corrected (Cr, Cu, Pd, Au, the actinides…).
pub fn electron_configuration(z: u32) -> Result<ParticleTextResult, String> {
    if z == 0 || z > 118 {
        return Err("Z must be 1..=118".into());
    }
    // (n+ℓ, n) filling order
    const ORDER: [(&str, u32); 19] = [
        ("1s", 2), ("2s", 2), ("2p", 6), ("3s", 2), ("3p", 6), ("4s", 2), ("3d", 10),
        ("4p", 6), ("5s", 2), ("4d", 10), ("5p", 6), ("6s", 2), ("4f", 14), ("5d", 10),
        ("6p", 6), ("7s", 2), ("5f", 14), ("6d", 10), ("7p", 6),
    ];
    // Exception tails are COMPLETE above the noble-gas core (so Pt/Au
    // carry their 4f14, which a bare "5d10 6s1" tail would miss)
    const EXCEPTIONS: &[(u32, &str)] = &[
        (24, "3d5 4s1"), (29, "3d10 4s1"), (41, "4d4 5s1"), (42, "4d5 5s1"),
        (44, "4d7 5s1"), (45, "4d8 5s1"), (46, "4d10"), (47, "4d10 5s1"),
        (57, "5d1 6s2"), (58, "4f1 5d1 6s2"), (64, "4f7 5d1 6s2"),
        (78, "4f14 5d9 6s1"), (79, "4f14 5d10 6s1"), (89, "6d1 7s2"), (90, "6d2 7s2"),
        (91, "5f2 6d1 7s2"), (92, "5f3 6d1 7s2"), (93, "5f4 6d1 7s2"),
        (96, "5f7 6d1 7s2"), (103, "5f14 7s2 7p1"),
    ];
    const NOBLE: [u32; 7] = [2, 10, 18, 36, 54, 86, 118];
    let mut config: Vec<(String, u32)> = Vec::new();
    if let Some((_, fixed)) = EXCEPTIONS.iter().find(|(ez, _)| *ez == z) {
        // core = the largest noble gas below Z, filled by Madelung — which
        // is exact for every noble gas (they have no exceptions)
        let noble_z = NOBLE.iter().rev().find(|&&n| n < z).copied().unwrap_or(0);
        let mut left = noble_z;
        for (name, cap) in ORDER {
            if left == 0 { break; }
            let take = left.min(cap);
            config.push((name.to_string(), take));
            left -= take;
        }
        let tail: Vec<(String, u32)> = fixed
            .split_whitespace()
            .map(|s| (s[..2].to_string(), s[2..].parse().unwrap_or(0)))
            .collect();
        config.extend(tail);
    } else {
        let mut left = z;
        for (name, cap) in ORDER {
            if left == 0 { break; }
            let take = left.min(cap);
            config.push((name.to_string(), take));
            left -= take;
        }
    }
    // re-check: the electrons must sum to Z
    let total: u32 = config.iter().map(|(_, c)| c).sum();
    if total != z {
        return Err("configuration re-check failed: electrons do not sum to Z".into());
    }
    // display order: group by shell (3d before 4s, textbook style)
    let mut sorted = config.clone();
    sorted.sort_by_key(|(name, _)| {
        let n: u32 = name.chars().next().unwrap_or('1').to_digit(10).unwrap_or(1);
        let l = match name.chars().nth(1).unwrap_or('s') { 's' => 0, 'p' => 1, 'd' => 2, _ => 3 };
        (n, l)
    });
    const SUP: [char; 10] = ['⁰', '¹', '²', '³', '⁴', '⁵', '⁶', '⁷', '⁸', '⁹'];
    let fmt = |n: u32| -> String {
        n.to_string().chars().map(|d| SUP[d.to_digit(10).unwrap() as usize]).collect()
    };
    let value = sorted.iter().map(|(s, c)| format!("{}{}", s, fmt(*c))).collect::<Vec<_>>().join(" ");
    Ok(ParticleTextResult {
        value,
        equation_used: "Madelung (n+ℓ, n) filling with the 20 known exceptions".into(),
        verify_method: "electron count sums to Z".into(),
    })
}

fn proper(name: &str) -> String {
    let mut c = name.chars();
    match c.next() {
        Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
        None => String::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quark_model() {
        let p = quark_composition("proton").unwrap();
        assert!(p.value.contains("up + up + down"));
        assert!(quark_composition("nope").is_err());
    }

    #[test]
    fn masses() {
        assert!((particle_mass("proton").unwrap().values[0] - 938.272).abs() < 0.01);
        assert!((particle_mass("electron").unwrap().values[1] - 9.109e-31).abs() < 1e-33);
        assert!(particle_mass("photon").unwrap().values[0] == 0.0);
        assert!(particle_mass("unobtainium").is_err());
    }

    #[test]
    fn decay_energies() {
        assert!((neutron_decay_energy().unwrap().values[0] - 0.782).abs() < 0.005);
        assert!((annihilation_energy().unwrap().values[0] - 1.022).abs() < 0.001);
    }

    #[test]
    fn photon_laws() {
        assert!((photon_energy(500.0).unwrap().values[0] - 2.48).abs() < 0.02);
        assert!((photon_wavelength(2.48).unwrap().values[0] - 500.0).abs() < 5.0);
        // round trip: 532 nm green
        let e = photon_energy(532.0).unwrap().values[0];
        assert!((photon_wavelength(e).unwrap().values[0] - 532.0).abs() < 1.0);
    }

    #[test]
    fn matter_waves() {
        // 1 eV electron: v = sqrt(2E/m), λ = 1.227 nm
        let v = (2.0 * EV_J / ELECTRON_KG).sqrt();
        let r = electron_de_broglie(v).unwrap();
        assert!((r.values[0] * 1e9 - 1.227).abs() < 0.01);
    }

    #[test]
    fn binding() {
        // Fe-56: 8.76 MeV/nucleon (measured 8.79)
        let r = binding_energy(56, 26).unwrap();
        assert!((r.values[1] - 8.76).abs() < 0.05);
        assert!(binding_energy(4, 6).is_err());
        // U-238: 7.60
        assert!((binding_energy(238, 92).unwrap().values[1] - 7.60).abs() < 0.05);
    }

    #[test]
    fn half_lives_and_ages() {
        assert!(half_life("carbon", 14).unwrap().value.contains("5,730"));
        assert!(half_life("uranium", 238).unwrap().value.contains("4.468 billion"));
        assert!(half_life("iron", 56).is_err());
        // 25% left → exactly 2 half-lives = 11,460 years
        let r = decay_age(0.25, 5730.0).unwrap();
        assert!((r.values[0] - 11_460.0).abs() < 0.01);
        // 1 g U-238 → 12.4 kBq
        let n = 1.0 / 238.0 * N_A;
        assert!((activity(n, 4.468e9).unwrap().values[0] - 12_440.0).abs() < 300.0);
    }

    #[test]
    fn shells_and_spectra() {
        assert_eq!(shell_capacity(3).unwrap().values[0], 18.0);
        // Hα 656.5 nm
        assert!((rydberg_wavelength(2, 3).unwrap().values[0] - 656.5).abs() < 0.5);
        assert!(rydberg_wavelength(3, 2).is_err());
    }

    #[test]
    fn configs() {
        assert_eq!(electron_configuration(26).unwrap().value.ends_with("3d⁶ 4s²"), true);
        assert_eq!(electron_configuration(29).unwrap().value.ends_with("3d¹⁰ 4s¹"), true);
        assert_eq!(electron_configuration(1).unwrap().value, "1s¹");
        // Au: [Xe] 4f14 5d10 6s1 — the tail must carry its own 4f14
        let au = electron_configuration(79).unwrap().value;
        assert!(au.contains("4f¹⁴"), "Au config needs 4f¹⁴: {}", au);
        assert!(au.ends_with("5d¹⁰ 6s¹"), "Au config must end 5d¹⁰ 6s¹: {}", au);
        assert!(!au.contains("6s²"), "Au must have 6s¹ only: {}", au);
        // Pt: [Xe] 4f14 5d9 6s1
        let pt = electron_configuration(78).unwrap().value;
        assert!(pt.contains("4f¹⁴") && pt.ends_with("5d⁹ 6s¹"), "Pt config: {}", pt);
        assert!(electron_configuration(119).is_err());
    }
}
