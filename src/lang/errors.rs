//! P0140 — friendly errors with "did you mean?" suggestions (Levenshtein).
//!
//! OTD4 — expanded with a library of corrective error templates: every
//! common mistake (negative radius, missing material, wrong unit, unknown
//! shape, unsupported boolean, bad axis, etc.) now comes with both a
//! diagnostic message AND a concrete fix suggestion. The point: an AI
//! working without a human in the loop must be able to read the error and
//! know exactly what to change.

#[derive(Clone, Debug)]
pub struct Error {
    pub line: usize,
    pub msg: String,
    pub hint: Option<String>,
}

impl Error {
    pub fn new(line: usize, msg: impl Into<String>) -> Error {
        Error { line, msg: msg.into(), hint: None }
    }
    pub fn with_hint(mut self, hint: impl Into<String>) -> Error {
        self.hint = Some(hint.into());
        self
    }
    /// Render for the console / browser.
    pub fn render(&self) -> String {
        match &self.hint {
            Some(h) => format!("line {}: {} — {}", self.line, self.msg, h),
            None => format!("line {}: {}", self.line, self.msg),
        }
    }
}

/// Classic Levenshtein edit distance (case-insensitive inputs compared lowercased).
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.to_lowercase().chars().collect();
    let b: Vec<char> = b.to_lowercase().chars().collect();
    if a.is_empty() { return b.len(); }
    if b.is_empty() { return a.len(); }
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0usize; b.len() + 1];
    for i in 1..=a.len() {
        cur[0] = i;
        for j in 1..=b.len() {
            let cost = if a[i - 1] == b[j - 1] { 0 } else { 1 };
            cur[j] = (prev[j] + 1).min(cur[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// Best suggestion from a candidate list, or None if nothing is close.
pub fn suggest(word: &str, candidates: &[&str]) -> Option<String> {
    let mut best: Option<(usize, &str)> = None;
    for c in candidates {
        let d = levenshtein(word, c);
        let max = (word.len() / 3).max(1);
        if d <= max {
            match best {
                Some((bd, _)) if bd <= d => {}
                _ => best = Some((d, c)),
            }
        }
    }
    best.map(|(_, c)| c.to_string())
}

// =============================================================================
// OTD4 — corrective error templates
// =============================================================================
//
// Every common OTD mistake now has a ready-made (msg, hint) pair. The hint is
// always a concrete fix an AI or human can apply. The eval layer calls these
// helpers to keep error messages consistent across the codebase.
//
// Convention: each helper returns an `Error` ready to push into the errors
// vector. The `line` parameter is the source line of the offending statement.

/// "radius cannot be negative" — with a fix suggesting the absolute value.
pub fn err_negative_radius(line: usize, got: f64, unit: &str) -> Error {
    Error::new(line, format!("radius cannot be negative ({}{})", got, unit))
        .with_hint(format!("did you mean {}{}? (radii are positive — use abs() if the sign came from a calculation)", got.abs(), unit))
}

/// "negative length" — for wall thicknesses, heights, depths, etc.
pub fn err_negative_length(line: usize, what: &str, got: f64, unit: &str) -> Error {
    Error::new(line, format!("{} cannot be negative ({}{})", what, got, unit))
        .with_hint(format!("use {}{} (a positive value), or check the calculation that produced this number", got.abs(), unit))
}

/// "unknown material" — with the closest-matching material suggestion.
pub fn err_unknown_material(line: usize, name: &str) -> Error {
    let hint = crate::world::materials::suggest(name)
        .map(|s| format!("did you mean {}?", s))
        .unwrap_or_else(|| "known materials: steel, aluminum, copper, gold, oak, glass, water, ceramic, iron, ... (see docs/05-MATERIALS.md)".into());
    Error::new(line, format!("'{}' is not a material I know", name))
        .with_hint(hint)
}

/// "unknown color" — with the closest-matching color suggestion.
pub fn err_unknown_color(line: usize, name: &str) -> Error {
    let hint = crate::world::colors::suggest(name)
        .map(|s| format!("did you mean {}?", s))
        .unwrap_or_else(|| "colors are words like ivory, steelblue, crimson, or hex like #1e90ff".into());
    Error::new(line, format!("'{}' is not a color I know", name))
        .with_hint(hint)
}

/// "unknown shape" — with the list of known primitives.
pub fn err_unknown_shape(line: usize, name: &str) -> Error {
    let hint = crate::lang::errors::suggest(name, crate::lang::keywords::PRIMITIVES)
        .map(|s| format!("did you mean {}?", s))
        .unwrap_or_else(|| "primitives: sphere, cube, cylinder, cone, torus, pyramid, prism, capsule, wedge, plane, tube, helix, rope, thread".into());
    Error::new(line, format!("'{}' is not a shape I know", name))
        .with_hint(hint)
}

/// "unknown unit" — with the list of known units.
pub fn err_unknown_unit(line: usize, name: &str) -> Error {
    let hint = crate::lang::errors::suggest(name, crate::lang::keywords::UNITS)
        .map(|s| format!("did you mean {}?", s))
        .unwrap_or_else(|| "units: mm, cm, m, km, in, ft, yd, um, deg, rad".into());
    Error::new(line, format!("'{}' is not a unit I know", name))
        .with_hint(hint)
}

/// "wrong number of args" — for shape calls that need a fixed count.
pub fn err_arg_count(line: usize, shape: &str, expected: usize, got: usize, example: &str) -> Error {
    Error::new(line, format!("{} needs {} argument(s), got {}", shape, expected, got))
        .with_hint(format!("example: {}", example))
}

/// "missing required arg" — for shape calls with a mandatory named arg.
pub fn err_missing_arg(line: usize, shape: &str, arg: &str, example: &str) -> Error {
    Error::new(line, format!("{} is missing the required `{}` argument", shape, arg))
        .with_hint(format!("example: {}", example))
}

/// "wrong axis" — for mirror/rotate that take x/y/z only.
pub fn err_bad_axis(line: usize, got: &str) -> Error {
    Error::new(line, format!("axis must be x, y, or z (got '{}')", got))
        .with_hint("use one of: x, y, z (lowercase, single letter)".to_string())
}

/// "wrong boolean operator" — for CSG mistakes.
pub fn err_bad_boolean(line: usize, got: &str) -> Error {
    Error::new(line, format!("'{}' is not a boolean operator", got))
        .with_hint("use + (fuse), - (cut), & (overlap) between shapes".to_string())
}

/// "object not found" — for material/color/hide applied to a non-existent name.
pub fn err_object_not_found(line: usize, name: &str, candidates: &[&str]) -> Error {
    let hint = crate::lang::errors::suggest(name, candidates)
        .map(|s| format!("did you mean {}?", s))
        .unwrap_or_else(|| "make the part first with `name = shape ...`, then apply material/color/transform".into());
    Error::new(line, format!("I can't find an object named '{}'", name))
        .with_hint(hint)
}

/// "type mismatch" — for argument type errors (e.g., a string where a number is expected).
pub fn err_type_mismatch(line: usize, what: &str, expected: &str, got: &str) -> Error {
    Error::new(line, format!("{} expects {} but got {}", what, expected, got))
        .with_hint(format!("check the value's type — common causes: passing a string where a number is needed, or a single value where a tuple is needed"))
}

/// "out of range" — for values that must be in a specific range.
pub fn err_out_of_range(line: usize, what: &str, got: f64, min: f64, max: f64, hint: &str) -> Error {
    Error::new(line, format!("{} = {} is out of range [{}, {}]", what, got, min, max))
        .with_hint(hint.to_string())
}

/// "syntax error" — generic with a fix suggestion.
pub fn err_syntax(line: usize, what: &str, example: &str) -> Error {
    Error::new(line, format!("syntax error: {}", what))
        .with_hint(format!("example: {}", example))
}

/// "overlap detected" — the OTD4 strict-mode error.
pub fn err_overlap(line: usize, a: &str, b: &str, depth_mm: f64, axis: &str) -> Error {
    Error::new(line, format!("strict overlap: {} and {} interpenetrate by {:.2} mm", a, b, depth_mm))
        .with_hint(format!(
            "fix: move {} along {} by {:.2} mm, or fuse with `add {}, {}` if they are meant to be one part",
            b, axis, depth_mm, a, b
        ))
}

/// "missing file" — for include/import of non-existent files.
pub fn err_file_not_found(line: usize, file: &str, kind: &str) -> Error {
    Error::new(line, format!("I can't find the {} file \"{}\"", kind, file))
        .with_hint(format!("put it next to your main .otd file, in examples/, or in library/parts/"))
}

/// "deprecated syntax" — for old syntax that still works but is discouraged.
pub fn err_deprecated(line: usize, old: &str, new: &str) -> Error {
    Error::new(line, format!("'{}' is deprecated", old))
        .with_hint(format!("use '{}' instead (the old form still works for now)", new))
}

/// "no parts" — when a simulate is called on an empty scene.
pub fn err_empty_scene(line: usize, sim: &str) -> Error {
    Error::new(line, format!("simulate: {} has nothing to work on — the scene is empty", sim))
        .with_hint("make a part first:  my_part = cube 5cm at (0, 5cm, 0) material: steel")
}

/// "no material" — when a physics simulation needs a material but the part has none.
pub fn err_no_material(line: usize, part: &str) -> Error {
    Error::new(line, format!("part '{}' has no material — physics needs density, conductivity, etc.", part))
        .with_hint(format!("add `material {}: steel` (or another material) before the simulate", part))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn levenshtein_basic() {
        assert_eq!(levenshtein("cermic", "ceramic"), 1);
        assert_eq!(levenshtein("sphre", "sphere"), 1);
        assert_eq!(levenshtein("abc", "abc"), 0);
    }
    #[test]
    fn suggestions() {
        assert_eq!(suggest("cermic", &["ceramic", "iron", "oak"]).unwrap(), "ceramic");
        assert!(suggest("zzzzzz", &["ceramic", "iron"]).is_none());
    }
    #[test]
    fn friendly_negative_radius() {
        let e = Error::new(3, "radius cannot be negative (-5mm)").with_hint("did you mean 5mm?");
        assert!(e.render().contains("did you mean 5mm?"));
    }
    #[test]
    fn err_templates_have_hints() {
        // every corrective template must produce a non-empty hint
        let e1 = err_negative_radius(1, -5.0, "mm");
        assert!(e1.hint.is_some());
        let e2 = err_bad_axis(1, "X");
        assert!(e2.hint.is_some());
        let e3 = err_bad_boolean(1, "*");
        assert!(e3.hint.is_some());
        let e4 = err_overlap(1, "a", "b", 2.5, "Y");
        assert!(e4.hint.is_some());
        let e5 = err_empty_scene(1, "drop");
        assert!(e5.hint.is_some());
    }
}
