//! P0010 — units and dimension checking.
//! Internal canonical unit: millimeters (f64). Angles: degrees in the language,
//! radians internally where trig needs them.

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Dim {
    /// plain number
    Plain,
    /// millimeters
    Length,
    /// degrees
    Angle,
}

impl Dim {
    pub fn mul(self, other: Dim) -> Result<Dim, String> {
        use Dim::*;
        match (self, other) {
            (Plain, x) | (x, Plain) => Ok(x),
            (Length, Length) => Err("mm × mm would be an area — OTD keeps one length dimension. Try dividing instead?".into()),
            (Angle, Angle) => Err("deg × deg — angles multiply into nothing useful. Did you mean i * 30deg?".into()),
            (Length, Angle) | (Angle, Length) => Err("length × angle is not a thing OTD knows. Maybe you mixed up a value?".into()),
        }
    }
    pub fn div(self, other: Dim) -> Result<Dim, String> {
        use Dim::*;
        match (self, other) {
            (x, Plain) => Ok(x),
            (Plain, Length) | (Plain, Angle) => Ok(Plain), // 1 / 5cm — allow, treat as plain
            (Length, Length) => Ok(Plain),
            (Angle, Angle) => Ok(Plain),
            (Length, Angle) | (Angle, Length) => Err("length / angle is not a length. Check the units?".into()),
        }
    }
}

/// A number with a dimension, e.g. `40mm`, `30deg`, `4`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Qty {
    pub v: f64,
    pub dim: Dim,
}

impl Qty {
    pub fn plain(v: f64) -> Qty { Qty { v, dim: Dim::Plain } }
    pub fn mm(v: f64) -> Qty { Qty { v, dim: Dim::Length } }
    pub fn deg(v: f64) -> Qty { Qty { v, dim: Dim::Angle } }

    /// Value as millimeters (angle/length mix-ups are caught at * and / or by
    /// the caller where it matters).
    pub fn as_mm(&self) -> f64 { self.v }
    pub fn as_deg(&self) -> f64 { self.v }

    pub fn expect_len(&self, _what: &str) -> f64 { self.v }
    pub fn expect_angle(&self, _what: &str) -> f64 { self.v }
    pub fn expect_plain(&self, _what: &str) -> f64 { self.v }
}

/// Parse an attached unit suffix to millimeters / degrees multiplier.
/// Returns (multiplier_to_mm, Dim) or None if unknown.
pub fn unit_factor(unit: &str) -> Option<(f64, Dim)> {
    match unit {
        "um" => Some((0.001, Dim::Length)),
        "mm" => Some((1.0, Dim::Length)),
        "cm" => Some((10.0, Dim::Length)),
        "m" => Some((1000.0, Dim::Length)),
        "km" => Some((1_000_000.0, Dim::Length)),
        "in" | "inch" | "inches" => Some((25.4, Dim::Length)),
        "ft" | "foot" | "feet" => Some((304.8, Dim::Length)),
        "yd" | "yard" | "yards" => Some((914.4, Dim::Length)),
        "deg" | "°" => Some((1.0, Dim::Angle)),
        "rad" => Some((57.29577951308232, Dim::Angle)),
        _ => None,
    }
}

/// Physics constants (P0540).
pub const G_EARTH: f64 = 9.81; // m/s²
pub const G_MOON: f64 = 1.62;
pub const G_MARS: f64 = 3.71;
pub const WATER_DENSITY: f64 = 1000.0; // kg/m³
/// mm³ in one m³
pub const MM3_PER_M3: f64 = 1e9;

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn units_convert() {
        assert_eq!(unit_factor("cm").unwrap().0, 10.0);
        assert_eq!(unit_factor("in").unwrap().0, 25.4);
        assert_eq!(unit_factor("ft").unwrap().0, 304.8);
        // 2.1 additions
        assert_eq!(unit_factor("km").unwrap().0, 1_000_000.0);
        assert_eq!(unit_factor("yd").unwrap().0, 914.4);
        assert_eq!(unit_factor("um").unwrap().0, 0.001);
        assert_eq!(unit_factor("rad").unwrap().1, Dim::Angle);
    }
    #[test]
    fn dims_multiply() {
        let a = Qty::mm(5.0).dim.mul(Qty::plain(2.0).dim).unwrap();
        assert_eq!(a, Dim::Length);
        assert!(Qty::mm(5.0).dim.mul(Qty::mm(5.0).dim).is_err());
    }
}
