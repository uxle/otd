//! Phase 014 — Adaptive curriculum progression (design doc section 20:
//! "Difficulty should adapt according to model performance")
//! (Rust port of python/curriculum/progression.py).
//!
//! A small state machine: stay at the current level until measured accuracy
//! on a held-out sample at that level crosses a threshold for enough
//! consecutive checks, then advance. Regress one level if accuracy craters
//! (protects against a level being too hard to learn from, matching design
//! doc 13's "difficulty estimation" spirit, applied at the curriculum level
//! instead of per-problem).

/// One `history` entry (Python appended plain dicts with these keys).
#[derive(Debug, Clone, PartialEq)]
pub struct HistoryEntry {
    pub level_before: i64,
    pub accuracy: f64,
    pub action: String,
    pub level_after: i64,
}

/// Python `@dataclass CurriculumController`.
#[derive(Debug, Clone)]
pub struct CurriculumController {
    pub min_level: i64,
    pub max_level: i64,
    pub advance_threshold: f64,
    pub regress_threshold: f64,
    pub consecutive_needed: i64,
    /// Python `level: int = field(init=False)` set in `__post_init__`.
    pub level: i64,
    consecutive_good: i64,
    pub history: Vec<HistoryEntry>,
}

impl CurriculumController {
    /// Full dataclass surface — Python field defaults are applied by the
    /// callers that don't pass them (advance 0.8 / regress 0.2 / needed 2).
    pub fn new(
        min_level: i64,
        max_level: i64,
        advance_threshold: f64,
        regress_threshold: f64,
        consecutive_needed: i64,
    ) -> Self {
        CurriculumController {
            min_level,
            max_level,
            advance_threshold,
            regress_threshold,
            consecutive_needed,
            level: min_level,
            consecutive_good: 0,
            history: Vec::new(),
        }
    }

    /// Python `record(accuracy) -> str`: feed in a measured accuracy at the
    /// current level; returns the action taken: 'advance', 'regress', or
    /// 'hold'.
    pub fn record(&mut self, accuracy: f64) -> String {
        let mut action = "hold".to_string();
        if accuracy >= self.advance_threshold {
            self.consecutive_good += 1;
            if self.consecutive_good >= self.consecutive_needed && self.level < self.max_level {
                self.level += 1;
                self.consecutive_good = 0;
                action = "advance".to_string();
            }
        } else if accuracy <= self.regress_threshold {
            self.consecutive_good = 0;
            if self.level > self.min_level {
                self.level -= 1;
                action = "regress".to_string();
            }
        } else {
            self.consecutive_good = 0;
        }

        self.history.push(HistoryEntry {
            level_before: if action != "advance" && action != "regress" {
                self.level
            } else if action == "advance" {
                self.level - 1
            } else {
                self.level + 1
            },
            accuracy,
            action: action.clone(),
            level_after: self.level,
        });
        action
    }
}

/// Python `run_curriculum(controller, accuracy_fn, max_checks=20)`: drive the
/// controller by repeatedly measuring accuracy at whatever level it's
/// currently at, via a caller-supplied accuracy_fn(level). (Python returned
/// the same controller it mutated; the Rust caller keeps its reference.)
pub fn run_curriculum<F: Fn(i64) -> f64>(
    controller: &mut CurriculumController,
    accuracy_fn: F,
    max_checks: usize,
) {
    for _ in 0..max_checks {
        let acc = accuracy_fn(controller.level);
        controller.record(acc);
    }
}
