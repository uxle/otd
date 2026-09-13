//! P2240 — THE SYNTHESIZER: "self make anything", literally.
//!
//! Feed `synthesize` a plain-language goal — "a titanium ball dropping into
//! water", "two balls racing, lead and cork", "a brass nut screwing onto a
//! steel bolt" — and it writes a complete .otd program: parts with real
//! materials, positions, gravity, environment, the right `simulate:` calls,
//! even the CLI hints for video and screw animation. Then it *compiles its
//! own output* through the full OTD front-end and repairs it (fallback
//! sizes, fallback materials) until the parser reports zero errors.
//!
//! How it decides:
//!   1. SCAN   — bag of words over the goal: materials (synonyms folded to
//!               the 50-material library), shapes, sizes with units, counts,
//!               heights, action verbs, modifiers (hot / spinning / racing).
//!   2. PLAN   — an action picks a template: drop, float, melt, race, magnet,
//!               orbit, collapse, learn, stats, screw… each knows its
//!               statements, its defaults, and its ask line.
//!   3. EMIT   — the script, in idiomatic OTD with comments explaining the
//!               physics the user is about to see.
//!   4. VERIFY — `otd::compile` runs; on error, fall back (steel instead of
//!               the unknown alloy, safe sizes) and re-emit. Up to 3 rounds.
//!
//! The result is always a program the engine itself has already accepted —
//! self-made, self-checked. `--make "goal"` on the CLI is the front door.

use crate::world::materials;

/// What came back from one act of self-making.
pub struct SynthReport {
    /// the goal sentence, as given
    pub goal: String,
    /// the generated .otd source (verified: compiles with zero errors)
    pub code: String,
    /// number of solid parts in the compiled scene
    pub parts: usize,
    /// total mass, grams
    pub mass_g: f64,
    /// CLI hints the goal implied (video, screw animation, spin)
    pub hints: Vec<String>,
    /// repair rounds it took to reach a clean compile
    pub repairs: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1 — SCAN: the goal becomes structured intent
// ─────────────────────────────────────────────────────────────────────────────

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ShapeKind {
    Sphere,
    Cube,
    Cylinder,
    Torus,
    Cone,
    Pyramid,
    Capsule,
    Plate,
    NutBolt,
}

impl ShapeKind {
    fn default_mm(&self) -> f64 {
        match self {
            ShapeKind::Sphere => 30.0,   // r
            ShapeKind::Cube => 60.0,     // side
            ShapeKind::Cylinder => 15.0, // r
            ShapeKind::Torus => 40.0,    // R
            ShapeKind::Cone => 25.0,     // r
            ShapeKind::Pyramid => 40.0,  // base
            ShapeKind::Capsule => 15.0,  // r
            ShapeKind::Plate => 120.0,   // w
            ShapeKind::NutBolt => 60.0,  // shaft len
        }
    }
}

/// Material words folded onto the library, with everyday synonyms.
fn material_of(word: &str) -> Option<&'static str> {
    let w = word.trim_end_matches(',');
    let canonical: &'static str = match w {
        // direct library hits
        "steel" => "steel",
        "iron" => "iron",
        "wood" => "wood",
        "oak" => "oak",
        "pine" => "pine",
        "teak" => "teak",
        "aluminum" | "aluminium" => "aluminum",
        "brass" => "brass",
        "bronze" => "bronze",
        "copper" => "copper",
        "lead" => "lead",
        "gold" => "gold",
        "silver" => "silver",
        "zinc" => "zinc",
        "titanium" => "titanium",
        "tungsten" => "tungsten",
        "chrome" => "chrome",
        "stainless" => "stainless",
        "glass" => "glass",
        "rubber" => "rubber",
        "plastic" => "plastic",
        "ceramic" => "ceramic",
        "marble" => "marble",
        "concrete" => "concrete",
        "ice" => "ice",
        "water" => "water",
        "mercury" => "mercury",
        "uranium" => "uranium",
        "plutonium" => "plutonium",
        "thorium" => "thorium",
        "lithium" => "lithium",
        "coal" => "coal",
        "gasoline" => "gasoline",
        "carbon" => "carbon",
        "foam" => "foam",
        "fabric" => "fabric",
        "oil" => "oil",
        // synonyms & adjective forms
        "wooden" => "wood",
        "golden" => "gold",
        "silvern" | "silvery" => "silver",
        "metal" => "steel",
        "stone" | "rock" | "granite" => "marble",
        "cork" | "balsa" => "pine",
        "plasticky" => "plastic",
        "glassy" => "glass",
        "icy" => "ice",
        "cardboard" | "paper" => "foam",
        _ => return None,
    };
    if canonical == "pine" && (w == "cork" || w == "balsa") {
        // cork/balsa fold onto pine — the lightest wood in the library
        return Some("pine");
    }
    if materials::find(canonical).is_some() {
        Some(canonical)
    } else {
        None
    }
}

fn shape_of(word: &str) -> Option<ShapeKind> {
    Some(match word.trim_end_matches(',') {
        "ball" | "balls" | "sphere" | "spheres" | "orb" | "orbs" | "marble" | "marbles" | "sphere-like" => ShapeKind::Sphere,
        "cube" | "cubes" | "box" | "boxes" | "block" | "blocks" | "dice" | "die" => ShapeKind::Cube,
        "cylinder" | "cylinders" | "rod" | "rods" | "bar" | "bars" | "can" | "canister" | "pillar" | "pipe" => ShapeKind::Cylinder,
        "donut" | "donuts" | "doughnut" | "ring" | "rings" | "torus" | "hoop" | "hoops" => ShapeKind::Torus,
        "cone" | "cones" => ShapeKind::Cone,
        "pyramid" | "pyramids" => ShapeKind::Pyramid,
        "capsule" | "capsules" | "pill" | "pills" => ShapeKind::Capsule,
        "plate" | "plates" | "sheet" | "sheets" | "pan" | "table" | "floor" => ShapeKind::Plate,
        "nut" | "nuts" | "bolt" | "bolts" | "screw" | "screws" => ShapeKind::NutBolt,
        _ => return None,
    })
}

fn number_word(word: &str) -> Option<usize> {
    Some(match word.trim_end_matches(',') {
        "a" | "an" | "one" | "1" => 1,
        "two" | "2" | "pair" | "both" => 2,
        "three" | "3" | "trio" => 3,
        "four" | "4" => 4,
        "five" | "5" => 5,
        "six" | "6" => 6,
        _ => return None,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Action {
    Drop,
    Float,
    Melt,
    Race,
    Magnet,
    Orbit,
    Collapse,
    Learn,
    Stats,
    Screw,
}

impl Action {
    fn simulate_lines(&self) -> &'static [&'static str] {
        match self {
            Action::Drop => &["simulate: drop", "simulate: time"],
            Action::Float => &["simulate: settle", "simulate: float"],
            Action::Melt => &["simulate: heat", "simulate: energy"],
            Action::Race => &["simulate: drop"],
            Action::Magnet => &["simulate: magnet"],
            Action::Orbit => &["simulate: orbit"],
            Action::Collapse => &["simulate: collapse"],
            Action::Learn => &["simulate: learn", "simulate: time"],
            Action::Stats => &["simulate: stats", "simulate: energy"],
            Action::Screw => &["simulate: energy"],
        }
    }

    fn verb(&self) -> &'static str {
        match self {
            Action::Drop => "dropping",
            Action::Float => "floating",
            Action::Melt => "melting",
            Action::Race => "racing",
            Action::Magnet => "magnetising",
            Action::Orbit => "orbiting",
            Action::Collapse => "collapsing",
            Action::Learn => "learning",
            Action::Stats => "measuring",
            Action::Screw => "screwing",
        }
    }
}

fn pick_action(words: &[&str]) -> Action {
    let has = |w: &str| words.contains(&w);
    // screw is the most specific — it wins whenever nuts/bolts appear
    if has("screwing") || has("screw") || has("nut") || has("bolt") {
        return Action::Screw;
    }
    if has("learn") || has("learned") || has("neural") || has("ai") || has("brain") {
        return Action::Learn;
    }
    if has("stats") || has("statistics") || has("dataset") {
        return Action::Stats;
    }
    if has("orbit") || has("orbits") || has("space") || has("satellite") || has("orbiting") {
        return Action::Orbit;
    }
    if has("magnet") || has("magnetism") || has("magnetic") || has("attract") || has("lifting") {
        return Action::Magnet;
    }
    if has("melt") || has("melting") || has("melted") || has("furnace") || has("hot") | has("heat") | has("glowing") {
        return Action::Melt;
    }
    if has("race") || has("racing") || has("versus") || has("vs") {
        return Action::Race;
    }
    if has("float") || has("floating") || has("floats") || has("sink") || has("sinks") || has("swim") || has("buoyancy") {
        return Action::Float;
    }
    if has("drop") || has("dropping") || has("drops") || has("dropped") || has("fall") || has("falling") || has("falls") || has("fell") {
        return Action::Drop;
    }
    if has("collapse") || has("tower") || has("stack") || has("topple") {
        return Action::Collapse;
    }
    Action::Drop
}

/// Extract "3cm", "50mm", "1m", "2in" — the first explicit size, in mm.
fn size_from_words(words: &[&str]) -> Option<f64> {
    for w in words {
        let w = w.trim_end_matches(',');
        let (num, unit): (String, String) = {
            let split = w.find(|c: char| !c.is_ascii_digit() && c != '.');
            match split {
                Some(i) => (w[..i].to_string(), w[i..].to_string()),
                None => continue,
            }
        };
        let v: f64 = match num.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        let mm = match unit.as_str() {
            "mm" => v,
            "cm" => v * 10.0,
            "m" => v * 1000.0,
            "in" | "inch" | "inches" => v * 25.4,
            "ft" | "foot" | "feet" => v * 304.8,
            _ => continue,
        };
        if mm > 0.5 && mm < 4000.0 {
            return Some(mm);
        }
    }
    None
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2 + 3 — PLAN and EMIT
// ─────────────────────────────────────────────────────────────────────────────

struct Intent {
    materials: Vec<&'static str>,
    shape: ShapeKind,
    count: usize,
    size_mm: Option<f64>,
    height_mm: Option<f64>,
    action: Action,
    environment: Option<&'static str>,
    gravity: Option<&'static str>,
    temperature_c: Option<f64>,
    wants_video: bool,
    spin: bool,
}

fn scan(goal: &str) -> Intent {
    let lower = goal.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    // materials: every recognised material word, in order of appearance
    let mut materials: Vec<&'static str> = Vec::new();
    for (i, w) in words.iter().enumerate() {
        // "into water" / "in oil" name the medium, not the part — skip those
        let prev = if i > 0 { words[i - 1] } else { "" };
        if ["water", "oil"].contains(w) && ["into", "in", "on", "of", "underwater"].contains(&prev) {
            continue;
        }
        if let Some(m) = material_of(w) {
            if !materials.contains(&m) {
                materials.push(m);
            }
        }
    }
    if materials.is_empty() {
        materials.push(if lower.contains("melt") || lower.contains("hot") || lower.contains("glow") {
            "iron"
        } else {
            "steel"
        });
    }

    // shape: first recognised shape word
    let mut shape = ShapeKind::Sphere;
    for w in &words {
        if let Some(s) = shape_of(w) {
            shape = s;
            break;
        }
    }

    // count: explicit number word, or the number of distinct materials for
    // races (racing needs at least two competitors)
    let mut count = number_word(words.first().copied().unwrap_or("a")).unwrap_or(1);
    let action = pick_action(&words);
    if action == Action::Race && count < 2 {
        count = materials.len().max(2);
    }
    count = count.clamp(1, 6);

    // heights: "from 2m", "at 80cm", "1m high" — a metre-ish number
    // following a position word (or preceding high/tall) becomes height
    let mut height_mm = None;
    let mut height_idx: Option<usize> = None;
    for (i, w) in words.iter().enumerate() {
        let w = w.trim_end_matches(',');
        if w == "from" || w == "at" || w == "off" {
            if let Some(h) = words.get(i + 1).and_then(|nw| size_from_words(&[*nw])) {
                if h > 100.0 {
                    height_mm = Some(h);
                    height_idx = Some(i + 1);
                }
            }
        }
    }
    // "80cm high/tall/deep"
    for i in 1..words.len() {
        let w = words[i].trim_end_matches(',');
        if ["high", "tall", "deep", "up"].contains(&w) {
            if let Some(h) = size_from_words(&[words[i - 1]]) {
                if h > 100.0 && height_mm.is_none() {
                    height_mm = Some(h);
                    height_idx = Some(i - 1);
                }
            }
        }
    }
    // the size is the first unit-number that is NOT the height token
    let size_mm = (0..words.len())
        .find(|&i| {
            height_idx.map(|hi| i != hi).unwrap_or(true) && size_from_words(&[words[i]]).is_some()
        })
        .and_then(|i| size_from_words(&[words[i]]));

    let environment = if words.iter().any(|w| ["water", "pool", "tank", "underwater", "lake", "ocean", "sea"].contains(w)) {
        Some("water")
    } else if words.iter().any(|w| ["vacuum", "space", "moon", "mars", "orbit"].contains(w)) {
        Some("vacuum")
    } else if words.iter().any(|w| ["oil", "oil-tank"].contains(w)) {
        Some("oil")
    } else {
        None
    };

    let gravity = if words.iter().any(|w| ["moon", "lunar"].contains(w)) {
        Some("moon")
    } else if words.iter().any(|w| ["mars", "martian"].contains(w)) {
        Some("mars")
    } else if words.iter().any(|w| ["space", "orbit", "weightless", "zero-g"].contains(w)) {
        Some("off")
    } else {
        None
    };

    let temperature_c = if action == Action::Melt {
        // choose a temperature that actually melts the first material
        let m = materials[0];
        Some(materials::find(m).and_then(|mat| mat.melt_c).map(|mp| mp + 150.0).unwrap_or(1600.0))
    } else if words.iter().any(|w| ["hot", "glowing", "heated", "warm"].contains(w)) {
        Some(600.0)
    } else if words.iter().any(|w| ["cold", "frozen", "freezing"].contains(w)) {
        Some(-30.0)
    } else {
        None
    };

    let wants_video = words.iter().any(|w| ["video", "film", "movie", "camera", "cameras", "animation", "animate", "clip"].contains(w));
    let spin = words.iter().any(|w| ["spin", "spinning", "spins", "rotate", "rotating", "turning", "twirl"].contains(w));

    Intent { materials, shape, count, size_mm, height_mm, action, environment, gravity, temperature_c, wants_video, spin }
}

/// mm → the cleanest unit string for a generated script.
fn mm_str(mm: f64) -> String {
    if (mm / 10.0).fract().abs() < 1e-9 && mm >= 10.0 && mm < 1000.0 {
        format!("{}cm", (mm / 10.0) as i64)
    } else if mm >= 1000.0 && (mm / 1000.0).fract().abs() < 1e-9 {
        format!("{}m", (mm / 1000.0) as i64)
    } else {
        format!("{}mm", mm as i64)
    }
}

fn shape_expr(kind: ShapeKind, size: f64) -> String {
    match kind {
        ShapeKind::Sphere => format!("sphere(r: {})", mm_str(size)),
        ShapeKind::Cube => format!("cube(w: {}, d: {}, h: {})", mm_str(size), mm_str(size), mm_str(size)),
        ShapeKind::Cylinder => format!("cylinder(r: {}, h: {})", mm_str(size), mm_str(size * 3.0)),
        ShapeKind::Torus => format!("torus(R: {}, tube: {})", mm_str(size), mm_str((size * 0.3).max(4.0))),
        ShapeKind::Cone => format!("cone(r: {}, h: {})", mm_str(size), mm_str(size * 2.0)),
        ShapeKind::Pyramid => format!("pyramid(w: {}, h: {})", mm_str(size), mm_str(size)),
        ShapeKind::Capsule => format!("capsule(r: {}, h: {})", mm_str(size), mm_str(size * 2.0)),
        ShapeKind::Plate => format!("plane(w: {}, d: {}, t: {})", mm_str(size), mm_str(size), mm_str((size * 0.05).max(1.0))),
        ShapeKind::NutBolt => format!("cylinder(r: {}, h: {})", mm_str(size * 0.15), mm_str(size)),
    }
}

fn emit(intent: &Intent, fallback_material: bool, safe_sizes: bool) -> String {
    let mut c = String::new();
    let action = intent.action;
    let mats: Vec<&str> = if fallback_material {
        vec!["steel"; intent.materials.len().max(1)]
    } else {
        intent.materials.clone()
    };
    let base = intent.size_mm.unwrap_or_else(|| {
        let d = intent.shape.default_mm();
        if safe_sizes { d } else { d }
    });
    let size = if safe_sizes {
        base.clamp(2.0, 300.0)
    } else {
        base.clamp(0.5, 2000.0)
    };
    let height = intent.height_mm.unwrap_or(if action == Action::Drop || action == Action::Race { 700.0 } else { 120.0 });
    let title = make_title(intent);

    c.push_str(&format!("# Self-made by the OTD3 synthesizer from the goal:\n#   \"{}\"\n", intent_words(intent)));
    c.push_str(&format!("# {} — the engine below was written, compiled and accepted by OTD itself.\n\n", action_comment(intent)));
    c.push_str(&format!("scene \"{}\"\n\n", title));

    if let Some(g) = intent.gravity {
        c.push_str(&format!("gravity: {}\n", g));
    }
    if let Some(e) = intent.environment {
        c.push_str(&format!("environment: {}\n", e));
    }
    if let Some(t) = intent.temperature_c {
        c.push_str(&format!("temperature: {}\n", t as i64));
    }
    if intent.gravity.is_some() || intent.environment.is_some() || intent.temperature_c.is_some() {
        c.push('\n');
    }

    // parts
    if intent.shape == ShapeKind::NutBolt {
        c.push_str("# the bolt: hex head + shaft + ISO thread; the nut rides down it\n");
        let pitch = 1.75;
        c.push_str(&format!(
            "head = prism(sides: 6, r: {}, h: {}) at (0, 0, 0) material: {}\n",
            mm_str((size * 0.2).max(8.0)),
            mm_str((size * 0.15).max(5.0)),
            mats[0]
        ));
        c.push_str(&format!(
            "shaft = cylinder(r: {}, h: {}) at (0, {}, 0) material: {}\n",
            mm_str((size * 0.12).max(5.0)),
            mm_str(size),
            mm_str((size * 0.15).max(5.0)),
            mats[0]
        ));
        let mat_nut = mats.get(1).copied().unwrap_or("brass");
        if mat_nut == mats[0] && mats.len() == 1 {
            // distinct nut material reads better
        }
        c.push_str(&format!(
            "threads = thread(radius: {}, pitch: {}, turns: 12) at (0, {}, 0) material: {}\n",
            mm_str((size * 0.11).max(4.5)),
            mm_str(pitch),
            mm_str((size * 0.18).max(6.0)),
            mats[0]
        ));
        c.push_str(&format!(
            "nut = subtract(prism(sides: 6, r: {}, h: {}) at (0, {}, 0), cylinder(r: {}, h: {}) at (0, {}, 0)) material: {}\n",
            mm_str((size * 0.18).max(9.0)),
            mm_str((size * 0.12).max(5.0)),
            mm_str((size * 0.55).max(20.0)),
            mm_str((size * 0.13).max(5.5)),
            mm_str((size * 0.14).max(6.0)),
            mm_str((size * 0.54).max(19.0)),
            mat_nut
        ));
    } else {
        let n = intent.count;
        let spacing = size * 3.0 + 20.0;
        for i in 0..n {
            let mat = mats[i % mats.len().min(n.max(1))].to_string();
            let name = format!("{}{}", mat, match intent.shape {
                ShapeKind::Sphere => "Ball",
                ShapeKind::Cube => "Cube",
                ShapeKind::Cylinder => "Rod",
                ShapeKind::Torus => "Ring",
                ShapeKind::Cone => "Cone",
                ShapeKind::Pyramid => "Pyramid",
                ShapeKind::Capsule => "Pill",
                ShapeKind::Plate => "Plate",
                ShapeKind::NutBolt => "Bolt",
            });
            let x = if n > 1 { (i as f64 - (n - 1) as f64 / 2.0) * spacing } else { 0.0 };
            let y = if action == Action::Float { height } else { height };
            c.push_str(&format!(
                "{} = {} at ({}, {}, 0) material: {}\n",
                name,
                shape_expr(intent.shape, size),
                mm_str(x),
                mm_str(y),
                mat
            ));
        }
        // water surface for float scenes
        if action == Action::Float && intent.environment.is_none() {
            c.push_str(&format!(
                "pool = plane(w: {}, d: {}, t: 2mm) at (0, {}, 0) color: deepskyblue\n",
                mm_str(spacing * (n.max(1) as f64) + 80.0),
                mm_str(spacing * (n.max(1) as f64) + 80.0),
                mm_str((height * 0.35).max(20.0))
            ));
        }
    }
    c.push('\n');

    // statements
    for s in action.simulate_lines() {
        c.push_str(&format!("{}\n", s));
    }
    let ask = match action {
        Action::Drop => "impact speed?",
        Action::Float => "will it float?",
        Action::Melt => "heat to melt?",
        Action::Race => "which lands first?",
        Action::Magnet => "force on the iron?",
        Action::Orbit => "orbital period?",
        Action::Collapse => "energy released?",
        Action::Learn => "how close to the law?",
        Action::Stats => "mean mass?",
        Action::Screw => "mechanical advantage?",
    };
    c.push_str(&format!("ask \"{}\"\n", ask));
    c
}

fn make_title(intent: &Intent) -> String {
    let m = intent.materials[0];
    let shape_word = match intent.shape {
        ShapeKind::Sphere => "ball",
        ShapeKind::Cube => "cube",
        ShapeKind::Cylinder => "rod",
        ShapeKind::Torus => "ring",
        ShapeKind::Cone => "cone",
        ShapeKind::Pyramid => "pyramid",
        ShapeKind::Capsule => "capsule",
        ShapeKind::Plate => "plate",
        ShapeKind::NutBolt => "nut & bolt",
    };
    let mut t = format!("{} {} — {}", m, shape_word, intent.action.verb());
    if let Some(e) = intent.environment {
        t.push_str(&format!(" in {}", e));
    }
    t[0..1].to_uppercase() + &t[1..]
}

fn action_comment(intent: &Intent) -> String {
    match intent.action {
        Action::Drop => "freefall: v = √(2gh), no mass anywhere in the formula".into(),
        Action::Float => "Archimedes: floats when ρ_body < ρ_fluid".into(),
        Action::Melt => "phase change: the material's real melting point does the deciding".into(),
        Action::Race => "the Galileo race — different masses, same arrival time".into(),
        Action::Magnet => "dipole fields falling off as 1/r⁴".into(),
        Action::Orbit => "Kepler III: T² = 4π²a³/GM".into(),
        Action::Collapse => "potential energy becoming motion and noise".into(),
        Action::Learn => "a neural net rediscovering t = √(2h/g) from samples alone".into(),
        Action::Stats => "the scene reading itself: mean, σ, correlation, the CLT live".into(),
        Action::Screw => "one turn = one pitch of travel: Archimedes' 2300-year-old machine".into(),
    }
}

fn intent_words(intent: &Intent) -> String {
    // reconstruct a clean one-line goal for the header comment
    let mut s = String::new();
    if intent.count > 1 {
        s.push_str(&format!("{} ", intent.count));
    }
    s.push_str(&format!("{} {}", intent.materials.join("/"), match intent.shape {
        ShapeKind::Sphere => "ball",
        ShapeKind::Cube => "cube",
        ShapeKind::Cylinder => "rod",
        ShapeKind::Torus => "ring",
        ShapeKind::Cone => "cone",
        ShapeKind::Pyramid => "pyramid",
        ShapeKind::Capsule => "capsule",
        ShapeKind::Plate => "plate",
        ShapeKind::NutBolt => "nut and bolt",
    }));
    s.push_str(&format!(", {}", intent.action.verb()));
    s
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4 — VERIFY: compile, repair, repeat
// ─────────────────────────────────────────────────────────────────────────────

/// Turn a goal into a verified .otd program. The synthesizer never ships
/// code its own engine has not already accepted.
pub fn synthesize(goal: &str) -> SynthReport {
    let intent = scan(goal);
    let mut repairs = 0usize;
    let mut code = emit(&intent, false, false);

    for round in 0..4 {
        let w = crate::compile(&code);
        if w.errors.is_empty() {
            let parts = w.parts.iter().filter(|p| !p.hidden).count();
            let mut hints = Vec::new();
            if intent.wants_video {
                if intent.action == Action::Screw {
                    hints.push("otd --video out.mp4 --screw nut=6@y:1.75 --frames 48 --cams 8 out.otd".into());
                } else if intent.spin {
                    let name = w.parts.first().map(|p| p.name.clone()).unwrap_or_else(|| "part".into());
                    hints.push(format!("otd --video out.mp4 --spin {}=360 --frames 36 --cams 8 out.otd", name));
                } else {
                    hints.push("otd --video out.mp4 --cams 8 --frames 36 out.otd".into());
                }
            } else if intent.spin {
                let name = w.parts.first().map(|p| p.name.clone()).unwrap_or_else(|| "part".into());
                hints.push(format!("otd --png out.png --spin {}=90 out.otd", name));
            }
            if intent.action == Action::Screw && !hints.iter().any(|h| h.contains("--screw")) {
                hints.push("otd --video out.mp4 --screw nut=6@y:1.75 --frames 48 out.otd".into());
            }
            return SynthReport {
                goal: goal.to_string(),
                code,
                parts,
                mass_g: w.stats.total_mass_g,
                hints,
                repairs,
            };
        }
        // repair: round 1 → safe sizes; round 2 → fallback materials; round 3 → both
        repairs += 1;
        let safe_sizes = round >= 1;
        let fallback_material = round >= 2;
        code = emit(&intent, fallback_material, safe_sizes);
    }
    // last resort: the always-compiles canonical drop scene
    repairs += 1;
    code = "# canonical fallback: one steel ball, one metre of freefall\nscene \"Steel ball — dropping\"\n\nball = sphere(r: 3cm) at (0, 1m, 0) material: steel\n\nsimulate: drop\nask \"impact speed?\"\n".to_string();
    let w = crate::compile(&code);
    SynthReport {
        goal: goal.to_string(),
        code,
        parts: w.parts.iter().filter(|p| !p.hidden).count(),
        mass_g: w.stats.total_mass_g,
        hints: Vec::new(),
        repairs,
    }
}

/// A tour of goals for docs and tests — every one must compile clean.
pub const DEMO_GOALS: &[&str] = &[
    "make a titanium ball dropping into water from 1m",
    "two balls racing to the ground, lead and pine",
    "a golden sphere floating in water",
    "a hot iron cube glowing in a furnace",
    "nut and bolt screwing animation in brass and steel",
    "8 camera video of a spinning copper torus",
    "an iron cube and a magnet attracting",
    "a satellite orbiting earth in space",
    "teach a neural network freefall physics",
    "statistics of a mixed material scene",
    "three glass marbles dropping from 80cm",
    "a wooden block tower collapsing",
    "a uranium sphere on the moon",
    "a big aluminum donut",
    "melt a lead cube",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_demo_goal_compiles_clean() {
        for goal in DEMO_GOALS {
            let r = synthesize(goal);
            assert!(
                !r.code.starts_with("# canonical fallback"),
                "'{}' fell back to the canonical scene",
                goal
            );
            assert!(r.parts >= 1, "'{}' produced {} parts", goal, r.parts);
        }
    }

    #[test]
    fn titanium_ball_drop_has_water_and_drop() {
        let r = synthesize("make a titanium ball dropping into water from 1m");
        assert!(r.code.contains("material: titanium"));
        assert!(r.code.contains("environment: water"));
        assert!(r.code.contains("simulate: drop"));
        assert!(r.code.contains("titaniumBall"));
        // height 1m respected
        assert!(r.code.contains("1m"));
    }

    #[test]
    fn race_uses_two_materials() {
        let r = synthesize("two balls racing to the ground, lead and pine");
        assert!(r.code.contains("material: lead"));
        assert!(r.code.contains("material: pine"));
        assert!(r.code.contains("simulate: drop"));
    }

    #[test]
    fn melt_sets_temperature_above_melting_point() {
        let r = synthesize("melt a lead cube");
        // lead melts at 327.5 °C — the synthesizer must overshoot it
        assert!(r.code.contains("temperature:"));
        let t: f64 = r
            .code
            .lines()
            .find(|l| l.starts_with("temperature:"))
            .and_then(|l| l["temperature:".len()..].trim().parse().ok())
            .unwrap();
        assert!(t > 327.5, "temperature {} must exceed lead's melting point", t);
        assert!(r.code.contains("material: lead"));
    }

    #[test]
    fn screw_goal_makes_thread_and_hint() {
        let r = synthesize("nut and bolt screwing animation in brass and steel");
        assert!(r.code.contains("thread("));
        assert!(r.code.contains("subtract("));
        assert!(r.hints.iter().any(|h| h.contains("--screw")), "hints: {:?}", r.hints);
    }

    #[test]
    fn video_goal_yields_avc_hint() {
        let r = synthesize("8 camera video of a spinning copper torus");
        assert!(r.code.contains("material: copper"));
        assert!(r.code.contains("torus("));
        assert!(r.hints.iter().any(|h| h.contains("--cams 8")), "hints: {:?}", r.hints);
        assert!(r.hints.iter().any(|h| h.contains("--spin")), "hints: {:?}", r.hints);
    }

    #[test]
    fn orbit_and_learn_goals_route_correctly() {
        let r = synthesize("a satellite orbiting earth in space");
        assert!(r.code.contains("simulate: orbit"));
        assert!(r.code.contains("environment: vacuum"));
        let r2 = synthesize("teach a neural network freefall physics");
        assert!(r2.code.contains("simulate: learn"));
    }

    #[test]
    fn moon_gravity_recognised() {
        let r = synthesize("a uranium sphere on the moon");
        assert!(r.code.contains("gravity: moon"));
        assert!(r.code.contains("material: uranium"));
    }

    #[test]
    fn sizes_with_units_are_respected() {
        let r = synthesize("a 8cm brass sphere");
        assert!(r.code.contains("sphere(r: 8cm)"), "code: {}", r.code);
    }

    #[test]
    fn garbage_goal_still_returns_compiling_code() {
        let r = synthesize("xyzzy plugh flurble");
        assert!(r.parts >= 1);
        let w = crate::compile(&r.code);
        assert!(w.errors.is_empty());
    }

    #[test]
    fn synonyms_fold_to_library_materials() {
        assert_eq!(material_of("wooden"), Some("wood"));
        assert_eq!(material_of("golden"), Some("gold"));
        assert_eq!(material_of("stone"), Some("marble"));
        assert_eq!(material_of("cork"), Some("pine"));
        assert_eq!(material_of("unobtainium"), None);
        assert_eq!(material_of("titanium"), Some("titanium"));
    }
}
