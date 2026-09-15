//! P0130 — the keyword registry. OTD 2.1 shipped 75 keywords (< 100 budget).
//! OTD3 adds the science expansion: `temperature` (the scene thermometer).
//! OTD3.3 adds `particle` (the Standard Model card statement).
//! OTD4 adds the dynamics + assembly expansion: `include` (multi-part
//! assembly), `magnetize` (mark a part as a permanent magnet), `strict`
//! (turn silent-wrongness into loud errors), and `overlap` (overlap-check
//! statement that fails compilation when solids interpenetrate).
//!
//! Counting follows the SVG model: structural words are keywords; parameter
//! names are attributes; materials/colors are value libraries.
//!
//! Every word here is reserved: it can never be an object name.

/// The 75 OTD 2.1 words + 2 OTD3 words + 1 OTD3.3 word + 4 OTD4 words + 1 OTD6 word = 83 keywords.
pub const KEYWORDS: &[&str] = &[
    // scene & output (7)
    "scene", "unit", "version", "gravity", "camera", "hide", "show",
    // OTD3 science (1) — the scene thermometer
    "temperature",
    // OTD3.3 subatomic (1) — the particle card statement
    "particle",
    // OTD4 dynamics + assembly (4) — multi-part, magnet, strict-overlap
    "include", "magnetize", "strict", "overlap",
    // OTD6 (1) — connect: electrical connectivity between parts
    "connect",
    // primitives (14 — +rope in 2.0, +thread in OTD3)
    "sphere", "cube", "cylinder", "cone", "torus", "pyramid",
    "prism", "capsule", "wedge", "plane", "tube", "helix", "rope", "thread",
    // advanced builders (11 — +blend in 2.0)
    "extrude", "revolve", "sweep", "loft", "text", "import",
    "terrain", "metaball", "hollow", "group", "blend",
    // deep-tier modifiers (2 — smooth, subdiv)
    "smooth", "subdiv",
    // booleans (3)
    "add", "subtract", "intersect",
    // transforms (4)
    "at", "rotate", "scale", "mirror",
    // patterns (4)
    "repeat", "grid", "ring", "scatter",
    // parts & appearance (4)
    "define", "use", "material", "color",
    // physics, query, export (4)
    "simulate", "ask", "export", "print",
    // units (6)
    "mm", "cm", "m", "in", "ft", "deg",
    // ---- 2.1 syntax expansion: logic (7) ----
    "and", "or", "not", "true", "false", "is", "mod",
    // ---- 2.1 syntax expansion: control flow (9; `in` already counted with units) ----
    "if", "else", "end", "for", "to", "by", "while", "break", "continue",
    // ---- 2.1 syntax expansion: checking (1) ----
    "assert",
];

pub const PRIMITIVES: &[&str] = &[
    "sphere", "cube", "cylinder", "cone", "torus", "pyramid",
    "prism", "capsule", "wedge", "plane", "tube", "helix", "rope", "thread",
];

pub const BUILDERS: &[&str] = &[
    "extrude", "revolve", "sweep", "loft", "text", "import",
    "terrain", "metaball", "hollow", "group", "blend",
];

pub const PATTERNS: &[&str] = &["repeat", "grid", "ring", "scatter"];

pub const FUNCS: &[&str] = &[
    // 1.0 math
    "cos", "sin", "tan", "sqrt", "abs", "min", "max", "round",
    // 2.1 additions
    "floor", "ceil", "pow", "log", "ln", "exp", "sign", "hypot",
    "atan", "atan2", "asin", "acos", "lerp", "clamp",
    "count", "sum", "avg",
    // OTD6 #1: self-describing
    "help", "functions", "materials", "keywords", "shapes", "sims",
    // OTD6 #3: electrodynamics functions (callable from scripts)
    "ohm_v", "ohm_i", "ohm_r", "power_vi", "power_ir",
    "cap_energy", "ind_energy", "rc_tau", "lc_omega",
];

pub const UNITS: &[&str] = &["mm", "cm", "m", "in", "ft", "deg", "km", "yd", "um", "rad"];

/// Statement words that open a block terminated by `end` (2.1).
pub const BLOCK_WORDS: &[&str] = &["if", "for", "while", "define"];

/// Words that end or branch a block (2.1).
pub const BLOCK_ENDERS: &[&str] = &["end", "else"];

pub fn is_keyword(w: &str) -> bool { KEYWORDS.contains(&w) }
pub fn is_primitive(w: &str) -> bool { PRIMITIVES.contains(&w) }
pub fn is_builder(w: &str) -> bool { BUILDERS.contains(&w) }
pub fn is_pattern(w: &str) -> bool { PATTERNS.contains(&w) }
pub fn is_func(w: &str) -> bool { FUNCS.contains(&w) }
pub fn is_unit(w: &str) -> bool { UNITS.contains(&w) }

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn keyword_budget() {
        assert_eq!(KEYWORDS.len(), 83, "OTD6 ships 83 keywords (75 + temperature + thread + particle + include + magnetize + strict + overlap + connect)");
        assert!(KEYWORDS.len() < 100, "hard budget: fewer than 100 keywords (two cheat-sheet sides)");
        // no duplicates
        let mut sorted = KEYWORDS.to_vec();
        sorted.sort();
        for i in 1..sorted.len() {
            assert_ne!(sorted[i - 1], sorted[i], "duplicate keyword");
        }
    }
    #[test]
    fn syntax_expansion_words_are_reserved() {
        for w in ["if", "else", "end", "for", "to", "in", "by", "while",
                  "break", "continue", "and", "or", "not", "true", "false",
                  "is", "mod", "assert"] {
            assert!(is_keyword(w), "{w} must be reserved in 2.1");
        }
    }
    #[test]
    fn func_table_grew() {
        assert_eq!(FUNCS.len(), 40, "8 original + 17 math/list + 6 self-describing + 9 electrodynamics = 40");
    }
    #[test]
    fn unit_table_grew() {
        assert_eq!(UNITS.len(), 10, "6 original + km, yd, um, rad");
    }
}
