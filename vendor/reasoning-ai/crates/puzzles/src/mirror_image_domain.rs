//! Phase 109 — Mirror image and water image (Promot: "mirror image")
//! (Rust port of python/puzzles/mirror_image_domain.py).
//!
//! Ground truth: an explicit, hand-verified lookup table of how each
//! character looks when reflected -- there is no formula to derive this
//! from, it's a fact about glyph shapes, so it's stated as data. Only
//! characters with an unambiguous, commonly-agreed mirror form are
//! included; anything else raises rather than guessing.
//!
//! Mirror image (vertical mirror): reverses left-right AND flips each
//! character horizontally. Water image (horizontal mirror): flips each
//! character vertically, does NOT reverse character order.
//!
//! Digits and uppercase letters only -- lowercase and most
//! non-alphanumeric glyphs have no clean, universally-agreed mirror form
//! at a monospace-text level, and this domain refuses to guess at ones
//! that don't (raises, doesn't pass the original character through).

/// Vertical mirror (left-right flip) of each character. Absence means
/// "not supported", not "maps to itself".
fn _mirror_map_lookup(key: char) -> Option<char> {
    Some(match key {
        'A' => 'A',
        'H' => 'H',
        'I' => 'I',
        'M' => 'M',
        'O' => 'O',
        'T' => 'T',
        'U' => 'U',
        'V' => 'V',
        'W' => 'W',
        'X' => 'X',
        'Y' => 'Y',
        '0' => '0',
        '1' => '1',
        '8' => '8',
        ' ' => ' ',
        _ => return None,
    })
}

/// Water (up-down flip) mirror of each character.
fn _water_map_lookup(key: char) -> Option<char> {
    Some(match key {
        'B' => 'B',
        'C' => 'C',
        'D' => 'D',
        'E' => 'E',
        'H' => 'H',
        'I' => 'I',
        'K' => 'K',
        'O' => 'O',
        'X' => 'X',
        '0' => '0',
        '1' => '1',
        '3' => 'E',
        '8' => '8',
        ' ' => ' ',
        _ => return None,
    })
}

/// `_CLOCK_DIGIT_MIRROR` (defined but unused in the Python module too;
/// `mirror_clock_time` uses the clock-face geometry rule instead).
#[allow(dead_code)]
const _CLOCK_DIGIT_MIRROR: &[(char, char)] = &[
    ('0', '0'),
    ('1', '1'),
    ('2', '2'),
    ('5', '5'),
    ('8', '8'),
    (':', ':'),
];

/// Python `str.upper()` for one char, as a single-char key: Some(c) when
/// the uppercase mapping is exactly one char (it always is for ASCII).
fn py_upper_char(ch: char) -> Option<char> {
    let mut it = ch.to_uppercase();
    let first = it.next()?;
    if it.next().is_some() {
        None
    } else {
        Some(first)
    }
}

/// Reverses the string AND mirrors each character -- like looking at
/// the text in a mirror to its right/left.
pub fn mirror_image(text: &str) -> Result<String, String> {
    let reversed: Vec<char> = text.chars().rev().collect();
    let mut out = String::new();
    for ch in reversed {
        let key = py_upper_char(ch);
        let mirrored = match key.and_then(_mirror_map_lookup) {
            Some(m) => m,
            None => return Err(format!("no defined mirror image for character '{}'", ch)),
        };
        // mirrored if ch.isupper() or not ch.isalpha(), else mirrored.lower()
        if ch.is_uppercase() || !ch.is_alphabetic() {
            out.push(mirrored);
        } else {
            out.push(mirrored.to_ascii_lowercase());
        }
    }
    Ok(out)
}

/// Flips each character vertically, keeps left-to-right order -- like
/// looking at a reflection in water below the text.
pub fn water_image(text: &str) -> Result<String, String> {
    let mut out = String::new();
    for ch in text.chars() {
        let key = py_upper_char(ch);
        let mirrored = match key.and_then(_water_map_lookup) {
            Some(m) => m,
            None => return Err(format!("no defined water image for character '{}'", ch)),
        };
        if ch.is_uppercase() || !ch.is_alphabetic() {
            out.push(mirrored);
        } else {
            out.push(mirrored.to_ascii_lowercase());
        }
    }
    Ok(out)
}

/// Classic aptitude puzzle: what does a digital clock showing HH:MM look
/// like in a mirror? Answer = 11:60 minus the time (mirror of a 12-hour
/// analog-style face), NOT a character-by-character flip -- a
/// digital-clock mirror puzzle is really asking about the clock face
/// geometry, so this uses the standard "sum to 11:60" rule.
pub fn mirror_clock_time(hh_mm: &str) -> Result<String, String> {
    let parts: Vec<&str> = hh_mm.split(':').collect();
    let (h, m): (i32, i32) = match parts.as_slice() {
        [hs, ms] => {
            let h: i32 = hs
                .trim()
                .parse()
                .map_err(|_| {
                    format!(
                        "expected HH:MM format, got '{}': invalid literal for int() with base 10: '{}'",
                        hh_mm,
                        hs.trim()
                    )
                })?;
            let m: i32 = ms
                .trim()
                .parse()
                .map_err(|_| {
                    format!(
                        "expected HH:MM format, got '{}': invalid literal for int() with base 10: '{}'",
                        hh_mm,
                        ms.trim()
                    )
                })?;
            (h, m)
        }
        _ => {
            let reason = if parts.len() < 2 {
                format!("not enough values to unpack (expected 2, got {})", parts.len())
            } else {
                "too many values to unpack (expected 2)".to_string()
            };
            return Err(format!("expected HH:MM format, got '{}': {}", hh_mm, reason));
        }
    };
    if !(0 <= h && h <= 12 && 0 <= m && m < 60) {
        return Err(format!(
            "time out of range for a 12-hour clock face: '{}'",
            hh_mm
        ));
    }
    let total_minutes = h * 60 + m;
    // Python's `%` is floored -> rem_euclid
    let mirrored_total = (11 * 60 + 60 - total_minutes).rem_euclid(12 * 60);
    let (mut mh, mm) = (mirrored_total / 60, mirrored_total % 60);
    if mh == 0 {
        mh = 12;
    }
    Ok(format!("{:02}:{:02}", mh, mm))
}
