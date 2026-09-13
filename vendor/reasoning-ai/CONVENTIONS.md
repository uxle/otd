# Rust Port — Shared Conventions (READ FIRST, applies to every crate)

This workspace is a faithful Rust port of the Python `reasoning-ai` project
(`/home/z/my-project/upload/reasoning_ai_extracted/reasoning-ai/python/`).
Every Python module maps to a Rust module with the same name
(`science/physics_domain.py` -> `crates/science/src/physics_domain.rs`).

## Hard rules

1. **Behavioral fidelity over idiom.** Port the *exact* algorithms, constants,
   thresholds, formulas, regex patterns, error messages, and doc comments
   (condensed where needed). Where Python code says `round(x, 4)`, use
   `round_dp(x, 4)` from `reasoning-science` helpers or replicate inline
   (see "Python float semantics" below). Do NOT "improve" logic.
2. **Errors:** Python `raise ValueError("msg")` -> Rust `Err(anyhow::Error)`?
   NO. Use `Result<T, String>` with the SAME message string, e.g.
   `return Err(format!("unrecognized element symbol: {:?}", tok));`.
   Call sites that Python guards with `except ValueError` map to
   `match`/`if let Err` and ignore or record the message.
   (Tests sometimes assert on the message — keep them identical.)
3. **Rust 2021, no external crates** unless the crate's Cargo.toml already
   lists them (`regex`, `rand`, `serde`, `serde_json` are pre-approved).
4. **Float semantics (Python):**
   - `round(x, n)` (banker's rounding, half-to-even): use helper `py_round(x, n)`.
   - `str(float)` formatting (shortest repr, `5.0` not `5`, `1e-07` for small):
     use helper `py_float_str(x)`.
   - Python `abs`, `min`, `max`, `sum` map to `.abs()`, `f64::min`, etc.
   - Integer division `//` -> `(a as f64 / b as f64).floor()` or i64 ops as
     appropriate; `%` on floats -> `a.rem_euclid(b)` only if Python used
     positive-modulo semantics — Python's `%` IS floored: use
     `((a % b) + b) % b` when b can be negative.
5. **Dataclasses** -> Rust `#[derive(Debug, Clone, PartialEq)]` structs
   (add `frozen=True` semantics naturally). `@dataclass` field defaults ->
   struct + `impl` constructor or `Default`. Don't use serde derives unless
   the module serialized to JSON in Python.
6. **Dicts/sets** -> `std::collections::HashMap`/`HashSet` (or BTreeMap when
   iteration order matters — Python dict preserves insertion order; use
   `Vec<(K,V)>` or IndexMap-like manual ordering when exact order affects
   output strings).
7. **No panics in library code** except where Python `assert`ed (keep as
   `assert!` / `debug_assert!`). Parse failures return `Err`.
8. **Module docstring** -> `//! ...` at file top (keep the phase number and
   key design notes, condensed to a few lines is OK).
9. **Function names:** Python `snake_case` stays `snake_case`. Class
   `MolarMass` -> struct `MolarMass`; methods keep names. Class methods
   become `impl` methods; `@staticmethod`/`@classmethod` -> free functions
   or associated functions.
10. **Tuples as records** -> tuples; `Tuple[float, str]` -> `(f64, String)`.
11. **`re` module:** use the `regex` crate. Python `re.I` ->
    `(?i)` inline or `RegexBuilder::case_insensitive`. Python
    `re.findall` -> `regex.find_iter().map(...)`. IMPORTANT: port patterns
    character-for-character, only changing Python-only syntax:
    - `\d` stays `\d`, `\b` stays `\b`
    - `(?:...)` stays
    - named groups `(?P<name>...)` -> `(?P<name>...)` (supported)
    - Python `(?!...)` lookahead is NOT supported by regex crate — if you
      encounter it, reimplement that specific check manually.
    - Empty pattern edge cases: guard with `Regex::new(...).unwrap()` in a
      `OnceLock`/`lazy` pattern (std `OnceLock` + fn).
    Compile each regex ONCE with `std::sync::OnceLock<Regex>`:
    `fn time_pat() -> &'static Regex { static R: OnceLock<Regex> = OnceLock::new(); R.get_or_init(|| Regex::new(r"...").unwrap()) }`
12. **random:** `random.Random(seed)` -> `rand::rngs::StdRng::seed_from_u64(seed as u64)`.
    `rng.random()` -> `rng.gen::<f64>()` (uniform [0,1)),
    `rng.choice(&v)` -> `v[rng.gen_range(0..v.len())]`,
    `rng.shuffle(&mut v)` -> `v.shuffle(&mut rng)` (rand 0.8).
    Exact RNG streams do NOT need to match Python (search is
    verification-driven, tests assert outcomes not paths).
13. **Tests:** port the matching Python test file(s) into the crate's
    `tests/` directory as integration tests using `#[test]` fns (split
    Python test classes/methods into separate `#[test]` fns named
    `test_classname_methodname`, condensed when a Python test is
    parametrized loops). pytest `approx(a, b)` -> `(a - b).abs() < 1e-9`
    (or the tolerance the test used). `pytest.raises(ValueError, match="...")`
    -> `let err = f(...).unwrap_err(); assert!(err.contains("..."))`.
14. Keep every magic number, tolerance (`1e-9`, `1e-6`, `0.999`), and
    threshold EXACTLY as in Python.
15. **Comments:** keep the insightful "why" comments (shortened OK), drop
    pure-Python mechanics commentary.

## File header template

```rust
//! Phase NNN — <title> (Rust port of python/<pkg>/<mod>.py)
//!
//! <1-3 lines of the original docstring's key points>
```
