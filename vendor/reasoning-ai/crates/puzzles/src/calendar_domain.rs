//! Phase 108 — Calendar / day-of-week reasoning (Promot: "time")
//! (Rust port of python/puzzles/calendar_domain.py).
//!
//! Ground truth: Zeller's congruence, a classic closed-form algorithm --
//! implemented explicitly because THIS is the kind of "reasoning" these
//! puzzles test, but every result is cross-checked against a
//! from-first-principles proleptic-Gregorian calendar (the Rust stand-in
//! for Python's `datetime.date`, which the original used) before being
//! trusted. A `day_of_week` call that disagrees with that calendar panics
//! (Python raised `AssertionError`) rather than silently returning the
//! formula's answer.

/// Zeller's per-month codes (kept for parity with the Python module; the
/// closed-form `(13*(m+1))//5` term below embeds them, so the table is
/// unused by the implementation).
#[allow(dead_code)]
const _ZELLER_MONTH_CODE: &[(i32, i32)] = &[
    (3, 3),
    (4, 6),
    (5, 2),
    (6, 5),
    (7, 0),
    (8, 3),
    (9, 6),
    (10, 1),
    (11, 4),
    (12, 6),
    (1, 1),
    (2, 4),
];

/// Zeller's own weekday convention: index 0=Saturday .. 6=Friday.
const _DAY_NAMES: [&str; 7] = [
    "Saturday",
    "Sunday",
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
];

/// Minimal proleptic-Gregorian date — the stand-in for Python's
/// `datetime.date` (same validation, same `weekday()` Monday=0
/// convention, same year range 1..=9999).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PyDate {
    pub year: i32,
    pub month: i32,
    pub day: i32,
}

impl PyDate {
    /// `datetime.date(y, m, d)` with the same ValueError messages.
    pub fn new(year: i32, month: i32, day: i32) -> Result<PyDate, String> {
        if !(1..=9999).contains(&year) {
            return Err(format!("year {} is out of range", year));
        }
        if !(1..=12).contains(&month) {
            return Err("month must be in 1..12".to_string());
        }
        if !(1..=days_in_month(year, month)).contains(&day) {
            return Err("day is out of range for month".to_string());
        }
        Ok(PyDate { year, month, day })
    }

    /// Monday=0 .. Sunday=6 (like `datetime.date.weekday()`).
    pub fn weekday(&self) -> i32 {
        let days = days_from_civil(self.year, self.month, self.day);
        // 1970-01-01 was a Thursday (weekday index 3, Monday=0).
        (days as i32 + 3).rem_euclid(7)
    }

    /// `strftime("%A")` equivalent: full weekday name.
    pub fn weekday_name(&self) -> &'static str {
        const NAMES: [&str; 7] = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ];
        NAMES[self.weekday() as usize]
    }
}

fn is_leap(year: i32) -> bool {
    year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)
}

fn days_in_month(year: i32, month: i32) -> i32 {
    match month {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => {
            if is_leap(year) {
                29
            } else {
                28
            }
        }
        _ => 0,
    }
}

/// Days since 1970-01-01 for a proleptic-Gregorian civil date
/// (Howard Hinnant's `days_from_civil` algorithm; negative for older dates).
fn days_from_civil(y: i32, m: i32, d: i32) -> i64 {
    let y = if m <= 2 { y - 1 } else { y } as i64;
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = y - era * 400; // [0, 399]
    let mp = (if m > 2 { m - 3 } else { m + 9 }) as i64; // [0, 11]
    let doy = (153 * mp + 2) / 5 + d as i64 - 1; // [0, 365]
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    era * 146097 + doe - 719468
}

/// Inverse of `days_from_civil` (Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i32, i32, i32) {
    let z = z + 719468;
    let era = if z >= 0 { z } else { z - 146096 } / 146097;
    let doe = z - era * 146097; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365; // [0, 399]
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = (doy - (153 * mp + 2) / 5 + 1) as i32; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 } as i32; // [1, 12]
    let year = if m <= 2 { y + 1 } else { y } as i32;
    (year, m, d)
}

/// Zeller's congruence for the Gregorian calendar. Returns 0=Saturday
/// .. 6=Friday (Zeller's own convention).
fn _zeller_day_index(year: i32, month: i32, day: i32) -> i32 {
    let (mut y, mut m) = (year, month);
    if m < 3 {
        y -= 1;
        m += 12;
    }
    let k = y.rem_euclid(100);
    let j = y.div_euclid(100);
    (day + (13 * (m + 1)) / 5 + k + k / 4 + j / 4 + 5 * j) % 7
}

/// Returns the weekday name, independently cross-checked against the
/// proleptic-Gregorian calendar (Python's own `datetime.date.weekday()`)
/// -- panics rather than trusting Zeller's formula alone if the two ever
/// disagree (e.g. an out-of-range date `datetime` itself would reject).
pub fn day_of_week(year: i32, month: i32, day: i32) -> Result<String, String> {
    let reference = PyDate::new(year, month, day)
        .map_err(|e| format!("not a valid calendar date: {}-{}-{} ({})", year, month, day, e))?;

    let zeller_idx = _zeller_day_index(year, month, day);
    let zeller_name = _DAY_NAMES[zeller_idx as usize];

    let ref_name = reference.weekday_name();
    assert!(
        zeller_name == ref_name,
        "Zeller's congruence ({}) disagreed with datetime ({}) for {}-{}-{} \
         -- refusing to report either as authoritative.",
        zeller_name,
        ref_name,
        year,
        month,
        day
    );
    Ok(ref_name.to_string())
}

/// Signed day count from date1 to date2, via exact proleptic-Gregorian
/// calendar arithmetic (already authoritative -- no need to reinvent
/// this the way day-of-week was worth deriving explicitly).
pub fn days_between(
    year1: i32,
    month1: i32,
    day1: i32,
    year2: i32,
    month2: i32,
    day2: i32,
) -> Result<i64, String> {
    let d1 = PyDate::new(year1, month1, day1)?;
    let d2 = PyDate::new(year2, month2, day2)?;
    Ok(days_from_civil(d2.year, d2.month, d2.day) - days_from_civil(d1.year, d1.month, d1.day))
}

/// `datetime.date(y, m, d) + datetime.timedelta(days=n)`.
pub fn add_days(year: i32, month: i32, day: i32, n: i64) -> Result<PyDate, String> {
    let d = PyDate::new(year, month, day)?;
    let days = days_from_civil(d.year, d.month, d.day) + n;
    let (y, m, dd) = civil_from_days(days);
    // datetime raises (OverflowError) outside years 1..9999
    if !(1..=9999).contains(&y) {
        return Err("date value out of range".to_string());
    }
    Ok(PyDate { year: y, month: m, day: dd })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zeller_agrees_with_civil_calendar() {
        // spot-check the two independent derivations across a wide range
        for &(y, m, d) in &[
            (1776, 7, 4),
            (1900, 1, 1),
            (2000, 1, 1),
            (2024, 2, 29),
            (2050, 6, 15),
            (1, 1, 1),
            (9999, 12, 31),
        ] {
            let date = PyDate::new(y, m, d).unwrap();
            // convert Monday=0 civil index to Zeller's Saturday=0 index
            let zeller_from_civil = (date.weekday() + 2) % 7;
            assert_eq!(
                _zeller_day_index(y, m, d),
                zeller_from_civil,
                "mismatch for {}-{}-{}",
                y,
                m,
                d
            );
        }
    }
}
