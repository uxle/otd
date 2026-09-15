//! P0520 — 27 materials with REAL engineering values (typical, rounded for
//! education). Users never type physics — the science ships with the name.

#[derive(Clone, Copy, Debug)]
pub struct Material {
    pub name: &'static str,
    /// kg/m³
    pub density: f64,
    /// base color RGB 0-255
    pub color: [u8; 3],
    /// 0 (mirror) .. 1 (matte)
    pub roughness: f64,
    pub metal: bool,
    /// melting point °C (None = burns/decomposes before melting)
    pub melt_c: Option<f64>,
    /// thermal conductivity W/(m·K)
    pub thermal: f64,
    /// friction coefficient μ
    pub friction: f64,
    /// bounce (restitution 0..1)
    pub bounce: f64,
    pub magnetic: bool,
    /// compressive strength MPa (approximate, for `simulate collapse`)
    pub compressive_mpa: f64,
    /// electrical conductivity MS/m (None = insulator)
    pub conductive: Option<f64>,
    /// P1400 — 1.0 = opaque, 0.0 = invisible. Glass/water/ice let light
    /// through; the renderer blends them so a glass cube shows what's behind it.
    pub opacity: f64,
}

macro_rules! mat {
    ($name:literal, $rho:literal, $c:expr, $rough:literal, $metal:literal, $melt:expr,
     $thermal:literal, $fric:literal, $bounce:literal, $mag:literal, $comp:literal, $cond:expr,
     $op:expr) => {
        Material {
            name: $name,
            density: $rho,
            color: $c,
            roughness: $rough,
            metal: $metal,
            melt_c: $melt,
            thermal: $thermal,
            friction: $fric,
            bounce: $bounce,
            magnetic: $mag,
            compressive_mpa: $comp,
            conductive: $cond,
            opacity: $op,
        }
    };
}

pub const MATERIALS: &[Material] = &[
    // ---- metals ----
    mat!("iron",      7874.0, [0x4e,0x4e,0x4e], 0.55, true,  Some(1538.0), 80.0,  0.45, 0.25, true,  170.0, Some(10.0),  1.0),
    mat!("steel",     7850.0, [0xb8,0xbc,0xc2], 0.35, true,  Some(1425.0), 50.0,  0.40, 0.30, true,  250.0, Some(6.0),   1.0),
    mat!("stainless", 7900.0, [0xcf,0xd2,0xd6], 0.25, true,  Some(1450.0), 16.0,  0.35, 0.30, false, 500.0, Some(1.3),   1.0),
    mat!("aluminum",  2700.0, [0xc8,0xcd,0xd2], 0.30, true,  Some(660.0),  237.0, 0.35, 0.35, false, 300.0, Some(37.0),  1.0),
    mat!("copper",    8960.0, [0xb8,0x73,0x33], 0.25, true,  Some(1085.0), 401.0, 0.30, 0.30, false, 210.0, Some(59.6),  1.0),
    mat!("brass",     8500.0, [0xc4,0xa0,0x4d], 0.30, true,  Some(930.0),  120.0, 0.35, 0.30, false, 340.0, Some(15.0),  1.0),
    mat!("bronze",    8800.0, [0xcd,0x7f,0x32], 0.35, true,  Some(950.0),  30.0,  0.35, 0.30, false, 350.0, Some(7.0),   1.0),
    mat!("gold",      19320.0,[0xff,0xd7,0x00], 0.15, true,  Some(1064.0), 317.0, 0.20, 0.30, false, 200.0, Some(44.0),  1.0),
    mat!("silver",    10490.0,[0xdc,0xdc,0xdc], 0.10, true,  Some(962.0),  429.0, 0.20, 0.30, false, 170.0, Some(63.0),  1.0),
    mat!("titanium",  4506.0, [0x9f,0xa3,0xa6], 0.40, true,  Some(1668.0), 22.0,  0.40, 0.30, false, 950.0, Some(2.4),   1.0),
    mat!("zinc",      7135.0, [0xc8,0xcc,0xd0], 0.35, true,  Some(420.0),  116.0, 0.35, 0.30, false, 250.0, Some(16.9),  1.0),
    mat!("lead",      11340.0,[0x6a,0x6d,0x70], 0.60, true,  Some(327.0),  35.0,  0.50, 0.20, false, 40.0,  Some(4.7),   1.0),
    mat!("chrome",    7190.0, [0xe8,0xea,0xed], 0.05, true,  Some(1907.0), 94.0,  0.20, 0.35, false, 530.0, Some(8.0),   1.0),
    mat!("tungsten",  19250.0,[0x8c,0x91,0x96], 0.35, true,  Some(3422.0), 170.0, 0.45, 0.25, false, 1300.0,Some(18.0),  1.0),
    // ---- woods ----
    mat!("wood",      700.0,  [0x8d,0x6e,0x63], 0.85, false, None,        0.15,  0.50, 0.30, false, 40.0,  None,        1.0),
    mat!("oak",       755.0,  [0xa9,0x81,0x53], 0.80, false, None,        0.17,  0.50, 0.30, false, 50.0,  None,        1.0),
    mat!("pine",      500.0,  [0xd9,0xc0,0x8c], 0.85, false, None,        0.12,  0.50, 0.30, false, 35.0,  None,        1.0),
    mat!("teak",      660.0,  [0xb0,0x86,0x50], 0.75, false, None,        0.15,  0.45, 0.30, false, 55.0,  None,        1.0),
    // ---- non-metals (glass/ice/water carry real opacity — see through them) ----
    mat!("glass",     2500.0, [0xcf,0xe8,0xef], 0.05, false, Some(1400.0), 1.05, 0.40, 0.65, false, 50.0,  None,        0.34),
    mat!("plastic",   1050.0, [0xf0,0xf0,0xf0], 0.50, false, Some(100.0),  0.20,  0.35, 0.55, false, 45.0,  None,        1.0),
    mat!("rubber",    1150.0, [0x2a,0x2a,0x2a], 0.95, false, Some(180.0),  0.15,  1.00, 0.85, false, 25.0,  None,        1.0),
    mat!("ceramic",   2400.0, [0xf5,0xf0,0xe6], 0.30, false, Some(1400.0), 1.50,  0.50, 0.40, false, 400.0, None,        1.0),
    mat!("concrete",  2400.0, [0x9a,0x9a,0x9a], 1.00, false, None,        1.00,  0.80, 0.20, false, 30.0,  None,        1.0),
    mat!("marble",    2711.0, [0xf0,0xec,0xe8], 0.25, false, None,        2.80,  0.45, 0.30, false, 100.0, None,        1.0),
    mat!("fabric",    300.0,  [0x8a,0x7f,0x76], 1.00, false, None,        0.04,  0.90, 0.10, false, 5.0,   None,        1.0),
    mat!("carbon",    1600.0, [0x1b,0x1b,0x1b], 0.35, false, Some(3600.0), 10.0,  0.30, 0.30, false, 1500.0,None,        1.0),
    mat!("ice",       917.0,  [0xc8,0xe8,0xf5], 0.10, false, Some(0.0),    2.2,   0.05, 0.10, false, 10.0,  None,        0.55),
    mat!("foam",      60.0,   [0xff,0xff,0xff], 0.90, false, Some(80.0),   0.03,  0.60, 0.20, false, 0.5,   None,        1.0),
    // ---- liquids (P1400: water is a first-class material — splash physics,
    // buoyancy, transparency) ----
    mat!("water",     997.0,  [0x2f,0x8f,0xd0], 0.12, false, Some(0.0),    0.60,  0.02, 0.02, false, 1.0,   None,        0.40),
    mat!("oil",       920.0,  [0xc8,0xa4,0x3c], 0.20, false, Some(-30.0),  0.17,  0.04, 0.03, false, 1.0,   None,        0.55),
    mat!("mercury",   13546.0,[0xb8,0xbf,0xc4], 0.08, true,  Some(-39.0),  8.3,   0.03, 0.02, false, 1.0,   Some(1.0),   1.0),
    // ---- more liquids (P1430 chemistry mixing) ----
    mat!("ethanol",   789.0,  [0xdc,0xe8,0xdc], 0.10, false, Some(-114.0), 0.17,  0.02, 0.02, false, 1.0,   None,        0.35),
    mat!("acetone",   784.0,  [0xe2,0xee,0xf0], 0.10, false, Some(-95.0),  0.16,  0.02, 0.02, false, 1.0,   None,        0.35),
    mat!("glycerin",  1261.0, [0xe8,0xe2,0xd2], 0.15, false, Some(18.0),   0.28,  0.30, 0.01, false, 1.0,   None,        0.45),
    // ---- gases (P1440: real STP densities kg/m³; they rise, sink and MIX) ----
    mat!("hydrogen",  0.0899, [0xd0,0xe0,0xff], 1.00, false, Some(-259.0), 0.18,  0.0,  0.0,  false, 0.1,   None,        0.12),
    mat!("helium",    0.1786, [0xd8,0xe4,0xf0], 1.00, false, Some(-272.0), 0.15,  0.0,  0.0,  false, 0.1,   None,        0.12),
    mat!("methane",   0.717,  [0xd0,0xf0,0xd8], 1.00, false, Some(-182.0), 0.03,  0.0,  0.0,  false, 0.1,   None,        0.12),
    mat!("ammonia",   0.769,  [0xd8,0xf0,0xd0], 1.00, false, Some(-78.0),  0.02,  0.0,  0.0,  false, 0.1,   None,        0.12),
    mat!("nitrogen",  1.165,  [0xd4,0xdc,0xf0], 1.00, false, Some(-210.0), 0.03,  0.0,  0.0,  false, 0.1,   None,        0.12),
    mat!("air",       1.225,  [0xd8,0xe0,0xec], 1.00, false, Some(-213.0), 0.03,  0.0,  0.0,  false, 0.1,   None,        0.10),
    mat!("oxygen",    1.429,  [0xd0,0xd8,0xff], 1.00, false, Some(-218.0), 0.03,  0.0,  0.0,  false, 0.1,   None,        0.12),
    mat!("steam",     0.598,  [0xe8,0xf0,0xf2], 1.00, false, Some(0.0),    0.02,  0.0,  0.0,  false, 0.1,   None,        0.18),
    mat!("carbon_dioxide", 1.977, [0xd4,0xe0,0xd8], 1.00, false, Some(-78.0), 0.02, 0.0, 0.0, false, 0.1, None,     0.12),
    mat!("chlorine",  3.214,  [0xd0,0xe8,0xb8], 1.00, false, Some(-101.0), 0.01,  0.0,  0.0,  false, 0.1,   None,        0.20),
    // ---- OTD3 energy materials (P2100): nuclear fuels, battery metal, fuels ----
    mat!("uranium",   19050.0,[0x5a,0x6e,0x5a], 0.55, true,  Some(1132.0), 27.0,  0.45, 0.25, false, 400.0, Some(0.35),  1.0),
    mat!("plutonium", 19816.0,[0x6e,0x5a,0x64], 0.55, true,  Some(640.0),  6.7,   0.45, 0.25, false, 400.0, Some(0.67),  1.0),
    mat!("thorium",   11724.0,[0x6a,0x6a,0x72], 0.50, true,  Some(1750.0), 54.0,  0.45, 0.25, false, 350.0, Some(0.7),   1.0),
    mat!("lithium",     534.0, [0xd8,0xd0,0xc8], 0.45, true,  Some(180.5),  85.0,  0.40, 0.30, false, 90.0,  Some(10.8),  1.0),
    mat!("coal",       1350.0, [0x1f,0x1f,0x22], 0.90, false, None,         0.3,   0.50, 0.25, false, 20.0,  None,        1.0),
    mat!("gasoline",    690.0, [0xd8,0xd4,0x9a], 0.15, false, Some(-40.0),  0.12,  0.02, 0.02, false, 1.0,   None,        0.55),
];

/// P1440 — the state of matter a material is in at room conditions.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum State {
    Solid,
    Liquid,
    Gas,
}

impl State {
    pub fn name(self) -> &'static str {
        match self {
            State::Solid => "solid",
            State::Liquid => "liquid",
            State::Gas => "gas",
        }
    }
}

/// The liquid and gas names (used by the chemistry and settle engines).
pub const LIQUIDS: &[&str] = &["water", "oil", "mercury", "ethanol", "acetone", "glycerin", "gasoline"];
pub const GASES: &[&str] = &[
    "hydrogen", "helium", "methane", "ammonia", "nitrogen", "air", "oxygen", "steam", "carbon_dioxide", "chlorine",
    "uranium", "plutonium", "thorium", "lithium", "coal", "gasoline",
];

pub fn state(m: &Material) -> State {
    if GASES.contains(&m.name) {
        State::Gas
    } else if LIQUIDS.contains(&m.name) {
        State::Liquid
    } else {
        State::Solid
    }
}

/// P1420b — air at 20 °C, 1 atm (kg/m³). The default medium: buoyancy and
/// drag are computed against it, so a helium ball rises and a stone barely
/// notices. `environment: vacuum` sets the medium density to 0.
pub const AIR_DENSITY: f64 = 1.204;

pub fn find(name: &str) -> Option<&'static Material> {
    MATERIALS.iter().find(|m| m.name == name)
}

pub const NAMES: &[&str] = &[
    "iron", "steel", "stainless", "aluminum", "copper", "brass", "bronze", "gold",
    "silver", "titanium", "zinc", "lead", "chrome", "tungsten",
    "wood", "oak", "pine", "teak",
    "glass", "plastic", "rubber", "ceramic", "concrete", "marble", "fabric",
    "carbon", "ice", "foam",
    "water", "oil", "mercury", "ethanol", "acetone", "glycerin",
    "hydrogen", "helium", "methane", "ammonia", "nitrogen", "air", "oxygen",
    "steam", "carbon_dioxide", "chlorine",
    "uranium", "plutonium", "thorium", "lithium", "coal", "gasoline",
];

/// P1400 — true if this material lets the background show through.
pub fn is_transparent(m: &Material) -> bool {
    m.opacity < 0.999
}

/// Suggest a material name (friendly errors).
pub fn suggest(name: &str) -> Option<String> {
    crate::lang::errors::suggest(name, NAMES)
}

/// OTD4 — suggest a material by intended use case. When the user hasn't
/// specified a material, we can recommend one based on what they're trying
/// to make. Returns (name, reason) so the user understands the suggestion.
pub fn suggest_by_use(use_case: &str) -> Option<(&'static str, &'static str)> {
    match use_case.to_lowercase().as_str() {
        // structural
        "structure" | "frame" | "beam" | "column" | "load_bearing" | "load-bearing" => {
            Some(("steel", "strong, cheap, welds easily; the default structural metal"))
        }
        "lightweight" | "light" | "drone" | "airplane" | "aerospace" => {
            Some(("aluminum", "2700 kg/m³ — 1/3 the weight of steel, fully recyclable"))
        }
        "strong_light" | "strong-light" | "bicycle" | "sport" => {
            Some(("titanium", "4500 kg/m³, ~steel strength at half the weight, biocompatible"))
        }
        // magnets
        "magnet" | "magnetic" | "permanent_magnet" | "permanent-magnet" => {
            Some(("iron", "ferromagnetic at room temperature — the easiest permanent magnet"))
        }
        "electromagnet" | "coil" | "inductor" | "transformer" => {
            Some(("iron", "high permeability — concentrates field lines in solenoids and transformers"))
        }
        // conductors
        "wire" | "conductor" | "cable" | "electrical" => {
            Some(("copper", "59.6 MS/m — the second-best conductor, and far cheaper than silver"))
        }
        // thermal
        "heat_sink" | "heat-sink" | "radiator" | "cooling" => {
            Some(("aluminum", "237 W/mK — the heatsink champion; cheap and easy to extrude"))
        }
        "pan" | "cookware" | "pot" => {
            Some(("copper", "401 W/mK — spreads heat evenly; the chef's choice (often lined with tin)"))
        }
        // liquids
        "fluid" | "liquid" | "water" | "coolant" => {
            Some(("water", "997 kg/m³, 4.18 J/gK — the universal solvent and coolant"))
        }
        "lubricant" | "lubrication" => {
            Some(("oil", "920 kg/m³ — slips between moving parts; floats on water"))
        }
        // optical
        "lens" | "window" | "transparent" | "optical" => {
            Some(("glass", "transmits 66% of light; the oldest optical material"))
        }
        // decorative
        "jewelry" | "ornament" | "luxury" => {
            Some(("gold", "19320 kg/m³ — never tarnishes, the eternal metal"))
        }
        // wood-class
        "furniture" | "wood" | "floor" | "cabinet" => {
            Some(("oak", "755 kg/m³ — hard, durable, the furniture wood"))
        }
        // thermal insulator
        "insulator" | "insulation" => {
            Some(("foam", "60 kg/m³, 0.03 W/mK — traps air, blocks heat"))
        }
        // heavy
        "heavy" | "ballast" | "weight" => {
            Some(("lead", "11340 kg/m³ — dense, soft, easy to shape; the weight champion"))
        }
        // nuclear
        "nuclear" | "reactor" | "radioactive" => {
            Some(("uranium", "19050 kg/m³ — the heaviest natural metal; powers reactors"))
        }
        // hard
        "cutting" | "tool" | "drill" | "hard" => {
            Some(("tungsten", "4.5 GPa hardness — the cutting-tool metal; highest melting point of any metal"))
        }
        // soft / protective
        "tire" | "seal" | "gasket" | "shock" | "bounce" => {
            Some(("rubber", "1150 kg/m³, 0.85 restitution — the bounce and seal material"))
        }
        // food / drink
        "cup" | "mug" | "dish" | "plate" => {
            Some(("ceramic", "2400 kg/m³ — fired clay, holds heat, dishwasher-safe"))
        }
        _ => None,
    }
}

/// OTD4 — list all material names that match a given property predicate
/// (e.g., "all ferromagnetic materials", "all materials denser than water").
pub fn list_by_property(prop: &str) -> Vec<&'static str> {
    MATERIALS
        .iter()
        .filter(|m| match prop.to_lowercase().as_str() {
            "magnetic" | "ferromagnetic" => m.magnetic,
            "metal" => m.metal,
            "transparent" => m.opacity < 0.999,
            "liquid" => LIQUIDS.contains(&m.name),
            "gas" => GASES.contains(&m.name),
            "solid" => !LIQUIDS.contains(&m.name) && !GASES.contains(&m.name),
            "conductor" | "conductive" => m.conductive.is_some(),
            "insulator" | "nonconductive" => m.conductive.is_none(),
            "sinks" | "denser_than_water" => m.density > 1000.0,
            "floats" | "lighter_than_water" => m.density < 1000.0,
            _ => false,
        })
        .map(|m| m.name)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_materials_resolvable() {
        assert_eq!(MATERIALS.len(), NAMES.len());
        for n in NAMES {
            assert!(find(n).is_some(), "missing material {}", n);
        }
    }

    #[test]
    fn transparent_materials_are_transparent() {
        // P1400: glass, ice, water, oil are see-through; metals are not
        assert!(is_transparent(find("glass").unwrap()));
        assert!(is_transparent(find("ice").unwrap()));
        assert!(is_transparent(find("water").unwrap()));
        assert!(is_transparent(find("oil").unwrap()));
        assert!(!is_transparent(find("steel").unwrap()));
        assert!(!is_transparent(find("wood").unwrap()));
    }

    #[test]
    fn water_is_water() {
        let w = find("water").unwrap();
        assert!((w.density - 997.0).abs() < 1.0);
        assert!(w.melt_c.unwrap().abs() < 0.01);
        // mercury is the only liquid metal — and it is HEAVY
        let hg = find("mercury").unwrap();
        assert!(hg.density > 13000.0 && hg.metal);
        // oil floats on water, water floats on mercury
        assert!(find("oil").unwrap().density < w.density);
        assert!(w.density < hg.density);
    }

    #[test]
    fn densities_are_real() {
        // spot-check real-world values (educational truth)
        assert!((find("gold").unwrap().density - 19320.0).abs() < 1.0);
        assert!((find("oak").unwrap().density - 755.0).abs() < 1.0);
        assert!((find("water_like").map(|m| m.density).unwrap_or(0.0) - 0.0).abs() < 1.0);
        // gold must be ~19.3 g/cm³
        assert!(find("gold").unwrap().density > 19000.0);
        // oak floats (density < water)
        assert!(find("oak").unwrap().density < 1000.0);
    }
}
