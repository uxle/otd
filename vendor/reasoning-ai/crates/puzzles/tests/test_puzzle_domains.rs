//! Port of python/tests/test_puzzle_domains.py (all classes).
//!
//! `assertAlmostEqual(a, b)` -> `(a - b).abs() < 1e-7`
//! (`places=N` -> `< 1e-N`); `assertRaises(ValueError)` -> `.unwrap_err()`.

use reasoning_puzzles::calendar_domain::{add_days, day_of_week, days_between, PyDate};
use reasoning_puzzles::clock_domain::{
    angle_between_hands, hour_hand_angle, is_right_angle, is_straight_line, minute_hand_angle,
    DEFAULT_TOLERANCE,
};
use reasoning_puzzles::direction_domain::{turn, walk, Move};
use reasoning_puzzles::family_tree_domain::FamilyTree;
use reasoning_puzzles::mirror_image_domain::{mirror_clock_time, mirror_image, water_image};

// ---------------------------------------------------------------- family_tree

#[test]
fn test_family_tree_simple_father_relationship() {
    let mut ft = FamilyTree::new();
    ft.add_fact("John", "father", "Mary").unwrap();
    ft.add_fact("Mary", "daughter", "John").unwrap(); // states Mary's gender explicitly
    assert_eq!(ft.relationship("John", "Mary"), Some("father".to_string()));
    assert_eq!(ft.relationship("Mary", "John"), Some("daughter".to_string()));
}

#[test]
fn test_family_tree_child_relationship_without_stated_gender_is_honestly_unknown() {
    // If gender was never stated, the domain should say 'child'
    // rather than guess 'son' or 'daughter'.
    let mut ft = FamilyTree::new();
    ft.add_fact("John", "father", "Mary").unwrap();
    assert_eq!(ft.relationship("Mary", "John"), Some("child".to_string()));
}

#[test]
fn test_family_tree_grandfather_relationship() {
    let mut ft = FamilyTree::new();
    ft.add_fact("John", "father", "Mary").unwrap();
    ft.add_fact("Mary", "mother", "Sue").unwrap();
    assert_eq!(ft.relationship("John", "Sue"), Some("grandfather".to_string()));
}

#[test]
fn test_family_tree_siblings_share_parents() {
    let mut ft = FamilyTree::new();
    ft.add_fact("John", "father", "Mary").unwrap();
    ft.add_fact("John", "father", "Tom").unwrap();
    ft.add_fact("Mary", "sister", "Tom").unwrap(); // states Mary's gender explicitly
    ft.add_fact("Tom", "brother", "Mary").unwrap(); // states Tom's gender explicitly
    assert_eq!(ft.relationship("Mary", "Tom"), Some("sister".to_string()));
    assert_eq!(ft.relationship("Tom", "Mary"), Some("brother".to_string()));
}

#[test]
fn test_family_tree_uncle_relationship() {
    let mut ft = FamilyTree::new();
    ft.add_fact("John", "father", "Mary").unwrap();
    ft.add_fact("John", "father", "Tom").unwrap();
    ft.add_fact("Tom", "son", "John").unwrap(); // states Tom's gender explicitly
    ft.add_fact("Mary", "mother", "Sue").unwrap();
    // Tom is Mary's brother, Sue is Mary's daughter -> Tom is Sue's uncle
    assert_eq!(ft.relationship("Tom", "Sue"), Some("uncle".to_string()));
    // Sue's own gender unstated
    assert_eq!(ft.relationship("Sue", "Tom"), Some("nephew/niece".to_string()));
}

#[test]
fn test_family_tree_husband_wife() {
    let mut ft = FamilyTree::new();
    ft.add_fact("John", "husband", "Ann").unwrap();
    assert_eq!(ft.relationship("John", "Ann"), Some("husband".to_string()));
    assert_eq!(ft.relationship("Ann", "John"), Some("wife".to_string()));
}

#[test]
fn test_family_tree_unrelated_returns_none() {
    let mut ft = FamilyTree::new();
    ft.add_fact("John", "father", "Mary").unwrap();
    assert!(ft.relationship("John", "Zzz").is_none());
}

#[test]
fn test_family_tree_unsupported_relation_raises() {
    let mut ft = FamilyTree::new();
    ft.add_fact("John", "cousin_by_marriage_twice_removed", "Mary")
        .unwrap_err();
}

// ----------------------------------------------------------- direction_distance

#[test]
fn test_direction_distance_straight_line_north() {
    let result = walk(&[Move::new("N", 10.0)]).unwrap();
    assert!((result.final_x - 0.0).abs() < 1e-7);
    assert!((result.final_y - 10.0).abs() < 1e-7);
    assert!((result.straight_line_distance - 10.0).abs() < 1e-7);
    assert!((result.bearing_degrees - 0.0).abs() < 1e-7);
}

#[test]
fn test_direction_distance_east_then_north_pythagorean() {
    // 3 east, 4 north -> straight-line distance = 5 (3-4-5 triangle)
    let result = walk(&[Move::new("E", 3.0), Move::new("N", 4.0)]).unwrap();
    assert!((result.straight_line_distance - 5.0).abs() < 1e-7);
}

#[test]
fn test_direction_distance_bearing_east_is_90() {
    let result = walk(&[Move::new("E", 5.0)]).unwrap();
    assert!((result.bearing_degrees - 90.0).abs() < 1e-7);
}

#[test]
fn test_direction_distance_return_to_start_zero_distance() {
    let result = walk(&[Move::new("N", 5.0), Move::new("S", 5.0)]).unwrap();
    assert!((result.straight_line_distance - 0.0).abs() < 1e-8); // places=8
}

#[test]
fn test_direction_distance_unsupported_direction_raises() {
    walk(&[Move::new("NE", 5.0)]).unwrap_err();
}

#[test]
fn test_direction_distance_turn_right_from_north_is_east() {
    assert_eq!(turn("N", "right", 90).unwrap(), "E");
}

#[test]
fn test_direction_distance_turn_left_from_north_is_west() {
    assert_eq!(turn("N", "left", 90).unwrap(), "W");
}

#[test]
fn test_direction_distance_turn_180() {
    assert_eq!(turn("N", "right", 180).unwrap(), "S");
}

// ------------------------------------------------------------------ clock_angles

#[test]
fn test_clock_angles_twelve_oclock_zero_angle() {
    assert!((angle_between_hands(12, 0).unwrap() - 0.0).abs() < 1e-7);
}

#[test]
fn test_clock_angles_three_oclock_ninety_degrees() {
    assert!((angle_between_hands(3, 0).unwrap() - 90.0).abs() < 1e-7);
}

#[test]
fn test_clock_angles_six_oclock_180_degrees() {
    assert!((angle_between_hands(6, 0).unwrap() - 180.0).abs() < 1e-7);
}

#[test]
fn test_clock_angles_hour_hand_creeps_with_minutes() {
    // Hand-known: at 3:30, hour hand is halfway between 3 and 4 = 105 degrees
    assert!((hour_hand_angle(3, 30).unwrap() - 105.0).abs() < 1e-7);
}

#[test]
fn test_clock_angles_classic_4_10_angle() {
    // Hand-known textbook example: at 4:10, angle = |4*30+10*0.5 - 10*6| = 65
    let h = hour_hand_angle(4, 10).unwrap();
    let m = minute_hand_angle(10).unwrap();
    let expected = f64::min((h - m).abs(), 360.0 - (h - m).abs());
    assert!((angle_between_hands(4, 10).unwrap() - expected).abs() < 1e-7);
    assert!((angle_between_hands(4, 10).unwrap() - 65.0).abs() < 1e-7);
}

#[test]
fn test_clock_angles_is_straight_line_at_six() {
    assert!(is_straight_line(6, 0, DEFAULT_TOLERANCE).unwrap());
}

#[test]
fn test_clock_angles_is_right_angle_at_three() {
    assert!(is_right_angle(3, 0, DEFAULT_TOLERANCE).unwrap());
}

#[test]
fn test_clock_angles_invalid_minute_raises() {
    hour_hand_angle(3, 60).unwrap_err();
}

// ----------------------------------------------------------------------- calendar

#[test]
fn test_calendar_known_historical_date() {
    // July 4, 1776 was a Thursday (well-documented historical fact)
    assert_eq!(day_of_week(1776, 7, 4).unwrap(), "Thursday");
}

#[test]
fn test_calendar_matches_datetime_for_arbitrary_dates() {
    // Python computed `datetime.date(y, m, d).strftime("%A")`; the
    // expected names are pinned here.
    for (y, m, d, expected) in [
        (2000, 1, 1, "Saturday"),
        (2024, 2, 29, "Thursday"),
        (1999, 12, 31, "Friday"),
        (2050, 6, 15, "Wednesday"),
    ] {
        assert_eq!(day_of_week(y, m, d).unwrap(), expected);
    }
}

#[test]
fn test_calendar_invalid_date_raises() {
    day_of_week(2023, 2, 30).unwrap_err(); // Feb 30 doesn't exist
}

#[test]
fn test_calendar_days_between_hand_computed() {
    assert_eq!(days_between(2024, 1, 1, 2024, 1, 11).unwrap(), 10);
}

#[test]
fn test_calendar_add_days_wraps_month() {
    let result = add_days(2024, 1, 30, 5).unwrap();
    assert_eq!(result, PyDate::new(2024, 2, 4).unwrap());
}

// ------------------------------------------------------------------- mirror_image

#[test]
fn test_mirror_image_mirror_single_symmetric_letter() {
    assert_eq!(mirror_image("A").unwrap(), "A");
}

#[test]
fn test_mirror_image_mirror_reverses_order() {
    // "HAT": T, A, H are all mirror-safe -> check order reversal
    assert_eq!(mirror_image("HAT").unwrap(), "TAH");
}

#[test]
fn test_mirror_image_mirror_unsupported_char_raises() {
    mirror_image("F").unwrap_err();
}

#[test]
fn test_mirror_image_water_image_keeps_order() {
    // all vertically symmetric, order preserved
    assert_eq!(water_image("BOX").unwrap(), "BOX");
}

#[test]
fn test_mirror_image_water_image_unsupported_char_raises() {
    water_image("A").unwrap_err(); // A is not in the water-safe set here
}

#[test]
fn test_mirror_image_mirror_clock_time_classic_example() {
    // Hand-known classic puzzle: 4:40 in a mirror looks like 7:20
    assert_eq!(mirror_clock_time("4:40").unwrap(), "07:20");
}

#[test]
fn test_mirror_image_mirror_clock_time_noon() {
    assert_eq!(mirror_clock_time("12:00").unwrap(), "12:00");
}

#[test]
fn test_mirror_image_mirror_clock_time_invalid_format_raises() {
    mirror_clock_time("four forty").unwrap_err();
}
