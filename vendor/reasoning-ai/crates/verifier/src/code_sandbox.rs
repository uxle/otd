//! Phase 031 — Safe code-execution verifier (design doc section 17:
//! "Never execute arbitrary generated code directly on the host system")
//! (Rust port of python/verifier/code_sandbox.py).
//!
//! The Python original sandboxes via CPython's own `ast` module: parse the
//! candidate code, reject anything outside a small allowed-node set (no
//! imports, no attribute/dunder access, no exec/eval/open/__import__, no
//! comprehensions/f-strings/assert/...), then `exec` in a namespace with
//! stripped builtins under a step-count tracer. This port reproduces that
//! security posture with a hand-written interpreter for exactly the allowed
//! subset: everything the whitelist admits is implemented, everything it
//! rejects is rejected with the same violation messages ("X is not in the
//! allowed node set" / "attribute access not allowed: .x" / "forbidden
//! name: eval" / "only direct name calls are allowed (no method calls)"),
//! and a hard step budget stops allowed-syntax runaway loops.
//!
//! Two-layer structure mirrors the original: a tokenizer + recursive-descent
//! parser that builds a sandbox AST (raising SandboxError::Violation for
//! every construct outside the whitelist), then a tree-walking interpreter
//! with Python value semantics (`/` true division, `//` floor division,
//! int-preserving `**` for non-negative exponents, cross-type `==`).
//! A step budget (each statement execution and each loop iteration counts)
//! replaces sys.settrace; the interpreter itself never touches the OS.

use std::cmp::Ordering;
use std::collections::HashMap;
use std::fmt;
use std::rc::Rc;

// ---------------------------------------------------------------------------
// Public value + error types
// ---------------------------------------------------------------------------

/// A runtime value in the sandbox. Python semantics: `Int(1) == Float(1.0)`
/// is true, `Bool` participates in arithmetic as 0/1, `List`/`Tuple` hold
/// heterogeneous elements.
#[derive(Debug, Clone)]
pub enum SandboxValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    Str(String),
    List(Vec<SandboxValue>),
    Tuple(Vec<SandboxValue>),
    None,
}

impl SandboxValue {
    pub fn type_name(&self) -> &'static str {
        match self {
            SandboxValue::Int(_) => "int",
            SandboxValue::Float(_) => "float",
            SandboxValue::Bool(_) => "bool",
            SandboxValue::Str(_) => "str",
            SandboxValue::List(_) => "list",
            SandboxValue::Tuple(_) => "tuple",
            SandboxValue::None => "NoneType",
        }
    }

    /// Python truthiness: 0, 0.0, "", [], (), None, False are falsy
    /// (NaN is truthy, like Python).
    pub fn truthy(&self) -> bool {
        match self {
            SandboxValue::Int(v) => *v != 0,
            SandboxValue::Float(v) => *v != 0.0,
            SandboxValue::Bool(v) => *v,
            SandboxValue::Str(s) => !s.is_empty(),
            SandboxValue::List(xs) => !xs.is_empty(),
            SandboxValue::Tuple(xs) => !xs.is_empty(),
            SandboxValue::None => false,
        }
    }

    /// Python `repr()` rendering, used for messages and debugging.
    pub fn py_repr(&self) -> String {
        match self {
            SandboxValue::Int(v) => format!("{}", v),
            SandboxValue::Float(v) => reasoning_common::py_float_str(*v),
            SandboxValue::Bool(v) => if *v { "True" } else { "False" }.to_string(),
            SandboxValue::Str(s) => format!("'{}'", escape_str(s)),
            SandboxValue::List(xs) => {
                let parts: Vec<String> = xs.iter().map(|x| x.py_repr()).collect();
                format!("[{}]", parts.join(", "))
            }
            SandboxValue::Tuple(xs) => {
                let parts: Vec<String> = xs.iter().map(|x| x.py_repr()).collect();
                if parts.len() == 1 {
                    format!("({},)", parts[0])
                } else {
                    format!("({})", parts.join(", "))
                }
            }
            SandboxValue::None => "None".to_string(),
        }
    }
}

fn escape_str(s: &str) -> String {
    let mut out = String::new();
    for c in s.chars() {
        match c {
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            '\'' => out.push_str("\\'"),
            _ => out.push(c),
        }
    }
    out
}

impl fmt::Display for SandboxValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.py_repr())
    }
}

/// Python `==`: cross-type numeric equality (True == 1, 1 == 1.0), element
/// equality for lists/tuples, type mismatch otherwise False (never an error).
impl PartialEq for SandboxValue {
    fn eq(&self, other: &SandboxValue) -> bool {
        py_eq(self, other)
    }
}

fn py_eq(a: &SandboxValue, b: &SandboxValue) -> bool {
    use SandboxValue::*;
    match (a, b) {
        (Int(x), Int(y)) => x == y,
        (Bool(x), Bool(y)) => x == y,
        (Float(x), Float(y)) => x == y,
        (Int(x), Bool(y)) => *x == i64::from(*y),
        (Bool(x), Int(y)) => i64::from(*x) == *y,
        (Int(x), Float(y)) => int_float_eq(*x, *y),
        (Float(x), Int(y)) => int_float_eq(*y, *x),
        (Bool(x), Float(y)) => int_float_eq(i64::from(*x), *y),
        (Float(x), Bool(y)) => int_float_eq(i64::from(*y), *x),
        (Str(x), Str(y)) => x == y,
        (List(x), List(y)) => x.len() == y.len() && x.iter().zip(y.iter()).all(|(u, v)| py_eq(u, v)),
        (Tuple(x), Tuple(y)) => {
            x.len() == y.len() && x.iter().zip(y.iter()).all(|(u, v)| py_eq(u, v))
        }
        (None, None) => true,
        _ => false,
    }
}

/// Exact int/float equality like Python (1 == 1.0, 2**53+1 != 2.0**53).
fn int_float_eq(x: i64, y: f64) -> bool {
    if y.is_finite() && y.fract() == 0.0 && y.abs() <= 9.2e18 {
        // exactly representable integer: compare exactly
        y as i64 == x
    } else {
        x as f64 == y
    }
}

fn py_cmp(a: &SandboxValue, b: &SandboxValue) -> Result<Ordering, SandboxError> {
    use SandboxValue::*;
    if matches!(a, Int(_) | Bool(_) | Float(_)) && matches!(b, Int(_) | Bool(_) | Float(_)) {
        if let (Some(x), Some(y)) = (as_int(a), as_int(b)) {
            return Ok(x.cmp(&y));
        }
        let (x, y) = (as_float(a).unwrap(), as_float(b).unwrap());
        return Ok(x.partial_cmp(&y).unwrap_or(Ordering::Equal));
    }
    match (a, b) {
        (Str(x), Str(y)) => Ok(x.cmp(y)),
        (List(x), List(y)) | (Tuple(x), Tuple(y)) => {
            for (u, v) in x.iter().zip(y.iter()) {
                match py_cmp(u, v)? {
                    Ordering::Equal => continue,
                    o => return Ok(o),
                }
            }
            Ok(x.len().cmp(&y.len()))
        }
        _ => Err(SandboxError::Violation(format!(
            "'<' not supported between instances of '{}' and '{}'",
            a.type_name(),
            b.type_name()
        ))),
    }
}

fn as_int(v: &SandboxValue) -> Option<i64> {
    match v {
        SandboxValue::Int(i) => Some(*i),
        SandboxValue::Bool(b) => Some(i64::from(*b)),
        _ => None,
    }
}

fn as_float(v: &SandboxValue) -> Option<f64> {
    match v {
        SandboxValue::Int(i) => Some(*i as f64),
        SandboxValue::Float(f) => Some(*f),
        SandboxValue::Bool(b) => Some(i64::from(*b) as f64),
        _ => None,
    }
}

fn is_float_typed(v: &SandboxValue) -> bool {
    matches!(v, SandboxValue::Float(_))
}

/// Errors from the sandbox. `Violation` = SandboxViolation (rejected
/// construct or runtime misbehavior), `Timeout` = ExecutionTimeout (step
/// budget exceeded), `Syntax` = the "syntax error: ..." wrapper.
#[derive(Debug, Clone, PartialEq)]
pub enum SandboxError {
    Violation(String),
    Timeout(String),
    Syntax(String),
}

impl fmt::Display for SandboxError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SandboxError::Violation(m) => write!(f, "{}", m),
            SandboxError::Timeout(m) => write!(f, "{}", m),
            SandboxError::Syntax(m) => write!(f, "{}", m),
        }
    }
}

impl std::error::Error for SandboxError {}

// ---------------------------------------------------------------------------
// Binary/unary/comparison operations (Python semantics)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    FloorDiv,
    Mod,
    Pow,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum UnOp {
    Neg,
    Pos,
    Not,
}

#[derive(Debug, Clone, Copy, PartialEq)]
enum CmpOp {
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
}

/// Cap for `range()` materialization and `str/list * int` repetition.
/// Python is lazy/unbounded here; the interpreter's value model is eager,
/// so pathological sizes fail closed instead of exhausting memory.
const MAX_MATERIALIZE: i64 = 1_000_000;

fn binop(op: BinOp, a: SandboxValue, b: SandboxValue) -> Result<SandboxValue, SandboxError> {
    match op {
        BinOp::Add => add_values(a, b),
        BinOp::Sub => {
            if let (Some(x), Some(y)) = (as_int(&a), as_int(&b)) {
                return match x.checked_sub(y) {
                    Some(d) => Ok(SandboxValue::Int(d)),
                    None => Ok(SandboxValue::Float(x as f64 - y as f64)),
                };
            }
            if let (Some(x), Some(y)) = (as_float(&a), as_float(&b)) {
                return Ok(SandboxValue::Float(x - y));
            }
            Err(SandboxError::Violation(format!(
                "unsupported operand type(s) for -: '{}' and '{}'",
                a.type_name(),
                b.type_name()
            )))
        }
        BinOp::Mul => {
            if let (Some(x), Some(y)) = (as_int(&a), as_int(&b)) {
                return match x.checked_mul(y) {
                    Some(p) => Ok(SandboxValue::Int(p)),
                    None => Ok(SandboxValue::Float(x as f64 * y as f64)),
                };
            }
            if let (Some(x), Some(y)) = (as_float(&a), as_float(&b)) {
                return Ok(SandboxValue::Float(x * y));
            }
            // str/list/tuple repetition
            if let (SandboxValue::Str(s), _) = (&a, &b) {
                if let Some(n) = as_int(&b) {
                    return repeat_str(s, n);
                }
            }
            if let (Some(n), SandboxValue::Str(s)) = (as_int(&a), &b) {
                return repeat_str(s, n);
            }
            if let (SandboxValue::List(xs), Some(n)) = (&a, as_int(&b)) {
                return repeat_seq(xs, n, false);
            }
            if let (Some(n), SandboxValue::List(xs)) = (as_int(&a), &b) {
                return repeat_seq(xs, n, false);
            }
            if let (SandboxValue::Tuple(xs), Some(n)) = (&a, as_int(&b)) {
                return repeat_seq(xs, n, true);
            }
            if let (Some(n), SandboxValue::Tuple(xs)) = (as_int(&a), &b) {
                return repeat_seq(xs, n, true);
            }
            Err(SandboxError::Violation(format!(
                "unsupported operand type(s) for *: '{}' and '{}'",
                a.type_name(),
                b.type_name()
            )))
        }
        BinOp::Div => {
            let (x, y) = (as_float(&a), as_float(&b));
            if let (Some(x), Some(y)) = (x, y) {
                if y == 0.0 {
                    return Err(SandboxError::Violation(
                        if is_float_typed(&a) || is_float_typed(&b) {
                            "float division by zero".to_string()
                        } else {
                            "division by zero".to_string()
                        },
                    ));
                }
                return Ok(SandboxValue::Float(x / y));
            }
            Err(SandboxError::Violation(format!(
                "unsupported operand type(s) for /: '{}' and '{}'",
                a.type_name(),
                b.type_name()
            )))
        }
        BinOp::FloorDiv => {
            if let (Some(x), Some(y)) = (as_int(&a), as_int(&b)) {
                if y == 0 {
                    return Err(SandboxError::Violation(
                        "integer division or modulo by zero".to_string(),
                    ));
                }
                return match x.checked_div(y) {
                    Some(q) => Ok(SandboxValue::Int(q)),
                    None => Err(SandboxError::Violation(
                        "integer division result too large for sandbox".to_string(),
                    )),
                };
            }
            if let (Some(x), Some(y)) = (as_float(&a), as_float(&b)) {
                if y == 0.0 {
                    return Err(SandboxError::Violation(
                        "float floor division by zero".to_string(),
                    ));
                }
                return Ok(SandboxValue::Float((x / y).floor()));
            }
            Err(SandboxError::Violation(format!(
                "unsupported operand type(s) for //: '{}' and '{}'",
                a.type_name(),
                b.type_name()
            )))
        }
        BinOp::Mod => {
            if let (Some(x), Some(y)) = (as_int(&a), as_int(&b)) {
                if y == 0 {
                    return Err(SandboxError::Violation(
                        "integer division or modulo by zero".to_string(),
                    ));
                }
                return Ok(SandboxValue::Int(py_mod_i(x, y)));
            }
            if let (Some(x), Some(y)) = (as_float(&a), as_float(&b)) {
                if y == 0.0 {
                    return Err(SandboxError::Violation("float modulo".to_string()));
                }
                let r = x % y;
                let r = if r != 0.0 && (r < 0.0) != (y < 0.0) { r + y } else { r };
                return Ok(SandboxValue::Float(r));
            }
            Err(SandboxError::Violation(format!(
                "unsupported operand type(s) for %: '{}' and '{}'",
                a.type_name(),
                b.type_name()
            )))
        }
        BinOp::Pow => {
            if let (Some(x), Some(y)) = (as_int(&a), as_int(&b)) {
                if y >= 0 {
                    if y <= u32::MAX as i64 {
                        if let Some(p) = x.checked_pow(y as u32) {
                            return Ok(SandboxValue::Int(p));
                        }
                    }
                    return Ok(SandboxValue::Float((x as f64).powf(y as f64)));
                }
                // negative exponent: Python 2**-1 == 0.5 (float)
                if x == 0 {
                    return Err(SandboxError::Violation(
                        "0.0 cannot be raised to a negative power".to_string(),
                    ));
                }
                return Ok(SandboxValue::Float((x as f64).powi(y as i32)));
            }
            if let (Some(x), Some(y)) = (as_float(&a), as_float(&b)) {
                return Ok(SandboxValue::Float(x.powf(y)));
            }
            Err(SandboxError::Violation(format!(
                "unsupported operand type(s) for ** or pow(): '{}' and '{}'",
                a.type_name(),
                b.type_name()
            )))
        }
    }
}

fn add_values(a: SandboxValue, b: SandboxValue) -> Result<SandboxValue, SandboxError> {
    if let (Some(x), Some(y)) = (as_int(&a), as_int(&b)) {
        return Ok(match x.checked_add(y) {
            Some(s) => SandboxValue::Int(s),
            None => SandboxValue::Float(x as f64 + y as f64),
        });
    }
    if let (Some(x), Some(y)) = (as_float(&a), as_float(&b)) {
        return Ok(SandboxValue::Float(x + y));
    }
    match (&a, &b) {
        (SandboxValue::Str(x), SandboxValue::Str(y)) => {
            Ok(SandboxValue::Str(format!("{}{}", x, y)))
        }
        (SandboxValue::List(x), SandboxValue::List(y)) => {
            let mut xs = x.clone();
            xs.extend(y.iter().cloned());
            Ok(SandboxValue::List(xs))
        }
        (SandboxValue::Tuple(x), SandboxValue::Tuple(y)) => {
            let mut xs = x.clone();
            xs.extend(y.iter().cloned());
            Ok(SandboxValue::Tuple(xs))
        }
        _ => Err(SandboxError::Violation(format!(
            "unsupported operand type(s) for +: '{}' and '{}'",
            a.type_name(),
            b.type_name()
        ))),
    }
}

/// Python floored modulo (result takes the sign of the divisor).
fn py_mod_i(a: i64, b: i64) -> i64 {
    let r = a % b;
    if r != 0 && (r < 0) != (b < 0) {
        r + b
    } else {
        r
    }
}

fn repeat_str(s: &str, n: i64) -> Result<SandboxValue, SandboxError> {
    if n <= 0 {
        return Ok(SandboxValue::Str(String::new()));
    }
    if n > MAX_MATERIALIZE || s.len() as i64 * n > MAX_MATERIALIZE * 16 {
        return Err(SandboxError::Violation(
            "multiplication result too large for sandbox".to_string(),
        ));
    }
    Ok(SandboxValue::Str(s.repeat(n as usize)))
}

fn repeat_seq(xs: &[SandboxValue], n: i64, tuple: bool) -> Result<SandboxValue, SandboxError> {
    if n <= 0 {
        return Ok(if tuple {
            SandboxValue::Tuple(vec![])
        } else {
            SandboxValue::List(vec![])
        });
    }
    let total = xs.len() as i64 * n;
    if total > MAX_MATERIALIZE {
        return Err(SandboxError::Violation(
            "multiplication result too large for sandbox".to_string(),
        ));
    }
    let mut out: Vec<SandboxValue> = Vec::with_capacity(total as usize);
    for _ in 0..n {
        out.extend(xs.iter().cloned());
    }
    Ok(if tuple {
        SandboxValue::Tuple(out)
    } else {
        SandboxValue::List(out)
    })
}

fn compare_op(op: CmpOp, a: &SandboxValue, b: &SandboxValue) -> Result<bool, SandboxError> {
    Ok(match op {
        CmpOp::Eq => py_eq(a, b),
        CmpOp::NotEq => !py_eq(a, b),
        CmpOp::Lt => matches!(py_cmp(a, b)?, Ordering::Less),
        CmpOp::LtE => !matches!(py_cmp(a, b)?, Ordering::Greater),
        CmpOp::Gt => matches!(py_cmp(a, b)?, Ordering::Greater),
        CmpOp::GtE => !matches!(py_cmp(a, b)?, Ordering::Less),
    })
}

fn norm_index(i: i64, len: i64, kind: &str) -> Result<usize, SandboxError> {
    let u = if i < 0 { i + len } else { i };
    if u < 0 || u >= len {
        return Err(SandboxError::Violation(format!(
            "{} index out of range",
            kind
        )));
    }
    Ok(u as usize)
}

fn index_value(cont: &SandboxValue, idx: &SandboxValue) -> Result<SandboxValue, SandboxError> {
    let i = match as_int(idx) {
        Some(i) => i,
        None => {
            return Err(SandboxError::Violation(format!(
                "{} indices must be integers or slices, not {}",
                cont.type_name(),
                idx.type_name()
            )))
        }
    };
    match cont {
        SandboxValue::List(xs) => {
            let u = norm_index(i, xs.len() as i64, "list")?;
            Ok(xs[u].clone())
        }
        SandboxValue::Tuple(xs) => {
            let u = norm_index(i, xs.len() as i64, "tuple")?;
            Ok(xs[u].clone())
        }
        SandboxValue::Str(s) => {
            let u = norm_index(i, s.chars().count() as i64, "string")?;
            Ok(SandboxValue::Str(s.chars().nth(u).unwrap().to_string()))
        }
        _ => Err(SandboxError::Violation(format!(
            "'{}' object is not subscriptable",
            cont.type_name()
        ))),
    }
}

fn slice_bound(v: Option<&SandboxValue>, default: i64, len: i64, upper: bool) -> Result<i64, SandboxError> {
    match v {
        None => Ok(default),
        Some(x) => {
            let i = as_int(x).ok_or_else(|| {
                SandboxError::Violation(format!(
                    "slice indices must be integers or None or have an __index__ method, not {}",
                    x.type_name()
                ))
            })?;
            let j = if i < 0 { i + len } else { i };
            if upper {
                // stop bound clamps into [-1, len] semantics
                Ok(j.clamp(-1, len))
            } else {
                Ok(j.clamp(0, len))
            }
        }
    }
}

fn slice_value(
    cont: &SandboxValue,
    lo: Option<&SandboxValue>,
    hi: Option<&SandboxValue>,
    st: Option<&SandboxValue>,
) -> Result<SandboxValue, SandboxError> {
    let step = match st {
        None => 1i64,
        Some(x) => as_int(x).ok_or_else(|| {
            SandboxError::Violation(format!(
                "slice indices must be integers or None or have an __index__ method, not {}",
                x.type_name()
            ))
        })?,
    };
    if step == 0 {
        return Err(SandboxError::Violation(
            "slice step cannot be zero".to_string(),
        ));
    }
    // materialize the sequence as a Vec of items
    let (items, kind): (Vec<SandboxValue>, u8) = match cont {
        SandboxValue::List(xs) => (xs.clone(), b'l'),
        SandboxValue::Tuple(xs) => (xs.clone(), b't'),
        SandboxValue::Str(s) => (
            s.chars().map(|c| SandboxValue::Str(c.to_string())).collect(),
            b's',
        ),
        _ => {
            return Err(SandboxError::Violation(format!(
                "'{}' object is not subscriptable",
                cont.type_name()
            )))
        }
    };
    let len = items.len() as i64;
    let mut picked: Vec<SandboxValue> = Vec::new();
    if step > 0 {
        let start = slice_bound(lo, 0, len, false)?;
        let stop = slice_bound(hi, len, len, false)?;
        let mut i = start;
        while i < stop {
            picked.push(items[i as usize].clone());
            i = match i.checked_add(step) {
                Some(v) => v,
                None => break,
            };
        }
    } else {
        let start = slice_bound(lo, len - 1, len, true)?;
        let stop = slice_bound(hi, -1, len, true)?;
        let mut i = start;
        while i > stop {
            picked.push(items[i as usize].clone());
            i = match i.checked_add(step) {
                Some(v) => v,
                None => break,
            };
        }
    }
    Ok(match kind {
        b't' => SandboxValue::Tuple(picked),
        b's' => SandboxValue::Str(
            picked
                .iter()
                .map(|v| match v {
                    SandboxValue::Str(s) => s.clone(),
                    _ => String::new(),
                })
                .collect(),
        ),
        _ => SandboxValue::List(picked),
    })
}

fn iter_values(v: SandboxValue) -> Result<Vec<SandboxValue>, SandboxError> {
    match v {
        SandboxValue::List(xs) => Ok(xs),
        SandboxValue::Tuple(xs) => Ok(xs),
        SandboxValue::Str(s) => Ok(s.chars().map(|c| SandboxValue::Str(c.to_string())).collect()),
        other => Err(SandboxError::Violation(format!(
            "'{}' object is not iterable",
            other.type_name()
        ))),
    }
}

// ---------------------------------------------------------------------------
// Builtins (the small safe allowlist from the Python sandbox)
// ---------------------------------------------------------------------------

const _FORBIDDEN_NAMES: [&str; 17] = [
    "__import__",
    "eval",
    "exec",
    "open",
    "compile",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "__builtins__",
    "__class__",
    "__bases__",
    "__subclasses__",
    "__globals__",
];

const _ALLOWED_CALL_NAMES: [&str; 7] = ["range", "len", "sum", "min", "max", "abs", "sorted"];

fn call_builtin(name: &str, args: &[SandboxValue]) -> Result<SandboxValue, SandboxError> {
    match name {
        "range" => {
            if args.is_empty() {
                return Err(SandboxError::Violation(
                    "range expected 1 argument, got 0".to_string(),
                ));
            }
            if args.len() > 3 {
                return Err(SandboxError::Violation(format!(
                    "range expected at most 3 arguments, got {}",
                    args.len()
                )));
            }
            let ints: Vec<i64> = args
                .iter()
                .map(|v| {
                    as_int(v).ok_or_else(|| {
                        SandboxError::Violation(format!(
                            "'{}' object cannot be interpreted as an integer",
                            v.type_name()
                        ))
                    })
                })
                .collect::<Result<_, _>>()?;
            let (start, stop, step) = match ints.len() {
                1 => (0i64, ints[0], 1i64),
                2 => (ints[0], ints[1], 1),
                _ => (ints[0], ints[1], ints[2]),
            };
            if step == 0 {
                return Err(SandboxError::Violation(
                    "range() arg 3 must not be zero".to_string(),
                ));
            }
            let count: i128 = if step > 0 {
                if stop <= start {
                    0
                } else {
                    ((stop - start) as i128 + (step as i128) - 1) / (step as i128)
                }
            } else if stop >= start {
                0
            } else {
                ((start - stop) as i128 + (-(step as i128)) - 1) / (-(step as i128))
            };
            if count > MAX_MATERIALIZE as i128 {
                return Err(SandboxError::Violation(
                    "range() result too large for sandbox".to_string(),
                ));
            }
            let mut out: Vec<SandboxValue> = Vec::new();
            let mut v = start;
            for _ in 0..count {
                out.push(SandboxValue::Int(v));
                v = match v.checked_add(step) {
                    Some(n) => n,
                    None => break,
                };
            }
            Ok(SandboxValue::List(out))
        }
        "len" => {
            if args.len() != 1 {
                return Err(SandboxError::Violation(format!(
                    "len() takes exactly one argument ({} given)",
                    args.len()
                )));
            }
            match &args[0] {
                SandboxValue::List(xs) => Ok(SandboxValue::Int(xs.len() as i64)),
                SandboxValue::Tuple(xs) => Ok(SandboxValue::Int(xs.len() as i64)),
                SandboxValue::Str(s) => Ok(SandboxValue::Int(s.chars().count() as i64)),
                v => Err(SandboxError::Violation(format!(
                    "object of type '{}' has no len()",
                    v.type_name()
                ))),
            }
        }
        "sum" => {
            if args.is_empty() {
                return Err(SandboxError::Violation(
                    "sum expected at least 1 argument, got 0".to_string(),
                ));
            }
            if args.len() > 2 {
                return Err(SandboxError::Violation(format!(
                    "sum expected at most 2 arguments, got {}",
                    args.len()
                )));
            }
            let items = iter_values(args[0].clone())?;
            let mut acc = if args.len() == 2 {
                args[1].clone()
            } else {
                SandboxValue::Int(0)
            };
            for it in items {
                acc = add_values(acc, it)?;
            }
            Ok(acc)
        }
        "min" | "max" => {
            if args.is_empty() {
                return Err(SandboxError::Violation(format!(
                    "{} expected 1 argument, got 0",
                    name
                )));
            }
            let candidates: Vec<SandboxValue> = if args.len() == 1 {
                iter_values(args[0].clone())?
            } else {
                args.to_vec()
            };
            if candidates.is_empty() {
                return Err(SandboxError::Violation(format!(
                    "{}() arg is an empty sequence",
                    name
                )));
            }
            let mut best = candidates[0].clone();
            for c in &candidates[1..] {
                let ord = py_cmp(c, &best)?;
                if (name == "min" && ord == Ordering::Less)
                    || (name == "max" && ord == Ordering::Greater)
                {
                    best = c.clone();
                }
            }
            Ok(best)
        }
        "abs" => {
            if args.len() != 1 {
                return Err(SandboxError::Violation(format!(
                    "abs() takes exactly one argument ({} given)",
                    args.len()
                )));
            }
            match &args[0] {
                SandboxValue::Int(i) => {
                    if *i == i64::MIN {
                        Ok(SandboxValue::Float(-(i64::MIN as f64)))
                    } else {
                        Ok(SandboxValue::Int(i.abs()))
                    }
                }
                SandboxValue::Float(f) => Ok(SandboxValue::Float(f.abs())),
                SandboxValue::Bool(b) => Ok(SandboxValue::Int(i64::from(*b))),
                v => Err(SandboxError::Violation(format!(
                    "bad operand type for abs(): '{}'",
                    v.type_name()
                ))),
            }
        }
        "sorted" => {
            if args.len() != 1 {
                return Err(SandboxError::Violation(format!(
                    "sorted expected 1 argument, got {}",
                    args.len()
                )));
            }
            let mut items: Vec<SandboxValue> = match &args[0] {
                SandboxValue::List(xs) => xs.clone(),
                SandboxValue::Tuple(xs) => xs.clone(),
                SandboxValue::Str(s) => {
                    s.chars().map(|c| SandboxValue::Str(c.to_string())).collect()
                }
                v => {
                    return Err(SandboxError::Violation(format!(
                        "'{}' object is not iterable",
                        v.type_name()
                    )))
                }
            };
            // stable insertion sort (Python's sort is stable; mixed
            // incomparable elements raise, like Python's TypeError)
            for i in 1..items.len() {
                let mut j = i;
                while j > 0 {
                    match py_cmp(&items[j], &items[j - 1])? {
                        Ordering::Less => {
                            items.swap(j, j - 1);
                            j -= 1;
                        }
                        _ => break,
                    }
                }
            }
            Ok(SandboxValue::List(items))
        }
        _ => Err(SandboxError::Violation(format!(
            "name '{}' is not defined",
            name
        ))),
    }
}

// ---------------------------------------------------------------------------
// Tokenizer
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
enum Kw {
    Def,
    Return,
    If,
    Elif,
    Else,
    For,
    In,
    While,
    Break,
    Continue,
    Pass,
    Import,
    From,
    Class,
    Lambda,
    Assert,
    Try,
    Except,
    Finally,
    With,
    As,
    Raise,
    Del,
    Global,
    Nonlocal,
    Yield,
    Await,
    Async,
    Is,
    Not,
    And,
    Or,
    True,
    False,
    None,
}

fn kw_of(s: &str) -> Option<Kw> {
    Some(match s {
        "def" => Kw::Def,
        "return" => Kw::Return,
        "if" => Kw::If,
        "elif" => Kw::Elif,
        "else" => Kw::Else,
        "for" => Kw::For,
        "in" => Kw::In,
        "while" => Kw::While,
        "break" => Kw::Break,
        "continue" => Kw::Continue,
        "pass" => Kw::Pass,
        "import" => Kw::Import,
        "from" => Kw::From,
        "class" => Kw::Class,
        "lambda" => Kw::Lambda,
        "assert" => Kw::Assert,
        "try" => Kw::Try,
        "except" => Kw::Except,
        "finally" => Kw::Finally,
        "with" => Kw::With,
        "as" => Kw::As,
        "raise" => Kw::Raise,
        "del" => Kw::Del,
        "global" => Kw::Global,
        "nonlocal" => Kw::Nonlocal,
        "yield" => Kw::Yield,
        "await" => Kw::Await,
        "async" => Kw::Async,
        "is" => Kw::Is,
        "not" => Kw::Not,
        "and" => Kw::And,
        "or" => Kw::Or,
        "True" => Kw::True,
        "False" => Kw::False,
        "None" => Kw::None,
        _ => return None,
    })
}

#[derive(Debug, Clone, PartialEq)]
enum Tok {
    Int(i64),
    Float(f64),
    Str(String),
    Name(String),
    Kw(Kw),
    Plus,
    Minus,
    Star,
    Slash,
    DoubleSlash,
    Percent,
    DoubleStar,
    LParen,
    RParen,
    LBracket,
    RBracket,
    LBrace,
    RBrace,
    Comma,
    Colon,
    Semi,
    Dot,
    At,
    Tilde,
    Amp,
    Pipe,
    Caret,
    Shl,
    Shr,
    Arrow,
    Assign,
    PlusA,
    MinusA,
    StarA,
    SlashA,
    DoubleSlashA,
    PercentA,
    DoubleStarA,
    Eq,
    NotEq,
    Lt,
    LtE,
    Gt,
    GtE,
    Newline,
    Indent,
    Dedent,
    Eof,
}

fn syntax_err(msg: impl fmt::Display) -> SandboxError {
    SandboxError::Syntax(format!("syntax error: {}", msg))
}

fn violation(msg: impl fmt::Display) -> SandboxError {
    SandboxError::Violation(format!("{}", msg))
}

/// Tokenize Python-subset source with INDENT/DEDENT tracking, implicit line
/// joining inside brackets, comments, string prefixes (f-strings rejected),
/// and decimal/hex/octal/binary/float literals.
fn tokenize(src: &str) -> Result<Vec<Tok>, SandboxError> {
    let src = src.replace("\r\n", "\n").replace('\r', "\n");
    let chars: Vec<char> = src.chars().collect();
    let mut toks: Vec<Tok> = Vec::new();
    let mut stack: Vec<usize> = vec![0];
    let mut i = 0usize;
    let n = chars.len();
    let mut paren = 0usize;
    let mut at_line_start = true;

    while i < n {
        if at_line_start && paren == 0 {
            // measure indentation
            let mut j = i;
            let mut width = 0usize;
            while j < n && chars[j] == ' ' {
                width += 1;
                j += 1;
            }
            if j < n && chars[j] == '\t' {
                return Err(syntax_err("tabs are not supported in indentation"));
            }
            // blank or comment-only line: skip entirely (no NEWLINE token)
            if j >= n || chars[j] == '\n' || chars[j] == '#' {
                while j < n && chars[j] != '\n' {
                    j += 1;
                }
                i = if j < n { j + 1 } else { j };
                continue;
            }
            if width > *stack.last().unwrap() {
                stack.push(width);
                toks.push(Tok::Indent);
            } else if width < *stack.last().unwrap() {
                while *stack.last().unwrap() > width {
                    stack.pop();
                    toks.push(Tok::Dedent);
                }
                if *stack.last().unwrap() != width {
                    return Err(syntax_err(
                        "unindent does not match any outer indentation level",
                    ));
                }
            }
            at_line_start = false;
            i = j;
            continue;
        }
        let c = chars[i];
        match c {
            ' ' | '\t' => {
                i += 1;
            }
            '#' => {
                while i < n && chars[i] != '\n' {
                    i += 1;
                }
            }
            '\\' => {
                // explicit line continuation
                if i + 1 < n && chars[i + 1] == '\n' {
                    i += 2;
                } else {
                    return Err(syntax_err("unexpected character '\\'"));
                }
            }
            '\n' => {
                if paren > 0 {
                    i += 1; // implicit line joining inside brackets
                } else {
                    toks.push(Tok::Newline);
                    i += 1;
                    at_line_start = true;
                }
            }
            '\r' => {
                i += 1;
            }
            '\'' | '"' => {
                let (s, next) = scan_string(&chars, i, false)?;
                toks.push(Tok::Str(s));
                i = next;
            }
            '(' => {
                paren += 1;
                toks.push(Tok::LParen);
                i += 1;
            }
            ')' => {
                if paren == 0 {
                    return Err(syntax_err("unmatched ')'"));
                }
                paren -= 1;
                toks.push(Tok::RParen);
                i += 1;
            }
            '[' => {
                paren += 1;
                toks.push(Tok::LBracket);
                i += 1;
            }
            ']' => {
                if paren == 0 {
                    return Err(syntax_err("unmatched ']'"));
                }
                paren -= 1;
                toks.push(Tok::RBracket);
                i += 1;
            }
            '{' => {
                paren += 1;
                toks.push(Tok::LBrace);
                i += 1;
            }
            '}' => {
                if paren == 0 {
                    return Err(syntax_err("unmatched '}'"));
                }
                paren -= 1;
                toks.push(Tok::RBrace);
                i += 1;
            }
            ',' => {
                toks.push(Tok::Comma);
                i += 1;
            }
            ':' => {
                toks.push(Tok::Colon);
                i += 1;
            }
            ';' => {
                toks.push(Tok::Semi);
                i += 1;
            }
            '.' => {
                toks.push(Tok::Dot);
                i += 1;
            }
            '~' => {
                toks.push(Tok::Tilde);
                i += 1;
            }
            '+' => {
                if chars.get(i + 1) == Some(&'=') {
                    toks.push(Tok::PlusA);
                    i += 2;
                } else {
                    toks.push(Tok::Plus);
                    i += 1;
                }
            }
            '-' => {
                if chars.get(i + 1) == Some(&'>') {
                    toks.push(Tok::Arrow);
                    i += 2;
                } else if chars.get(i + 1) == Some(&'=') {
                    toks.push(Tok::MinusA);
                    i += 2;
                } else {
                    toks.push(Tok::Minus);
                    i += 1;
                }
            }
            '*' => {
                if chars.get(i + 1) == Some(&'*') {
                    if chars.get(i + 2) == Some(&'=') {
                        toks.push(Tok::DoubleStarA);
                        i += 3;
                    } else {
                        toks.push(Tok::DoubleStar);
                        i += 2;
                    }
                } else if chars.get(i + 1) == Some(&'=') {
                    toks.push(Tok::StarA);
                    i += 2;
                } else {
                    toks.push(Tok::Star);
                    i += 1;
                }
            }
            '/' => {
                if chars.get(i + 1) == Some(&'/') {
                    if chars.get(i + 2) == Some(&'=') {
                        toks.push(Tok::DoubleSlashA);
                        i += 3;
                    } else {
                        toks.push(Tok::DoubleSlash);
                        i += 2;
                    }
                } else if chars.get(i + 1) == Some(&'=') {
                    toks.push(Tok::SlashA);
                    i += 2;
                } else {
                    toks.push(Tok::Slash);
                    i += 1;
                }
            }
            '%' => {
                if chars.get(i + 1) == Some(&'=') {
                    toks.push(Tok::PercentA);
                    i += 2;
                } else {
                    toks.push(Tok::Percent);
                    i += 1;
                }
            }
            '=' => {
                if chars.get(i + 1) == Some(&'=') {
                    toks.push(Tok::Eq);
                    i += 2;
                } else {
                    toks.push(Tok::Assign);
                    i += 1;
                }
            }
            '!' => {
                if chars.get(i + 1) == Some(&'=') {
                    toks.push(Tok::NotEq);
                    i += 2;
                } else {
                    return Err(syntax_err("invalid syntax"));
                }
            }
            '<' => {
                if chars.get(i + 1) == Some(&'=') {
                    toks.push(Tok::LtE);
                    i += 2;
                } else if chars.get(i + 1) == Some(&'<') {
                    toks.push(Tok::Shl);
                    i += 2;
                } else {
                    toks.push(Tok::Lt);
                    i += 1;
                }
            }
            '>' => {
                if chars.get(i + 1) == Some(&'=') {
                    toks.push(Tok::GtE);
                    i += 2;
                } else if chars.get(i + 1) == Some(&'>') {
                    toks.push(Tok::Shr);
                    i += 2;
                } else {
                    toks.push(Tok::Gt);
                    i += 1;
                }
            }
            '&' => {
                toks.push(Tok::Amp);
                i += 1;
            }
            '|' => {
                toks.push(Tok::Pipe);
                i += 1;
            }
            '^' => {
                toks.push(Tok::Caret);
                i += 1;
            }
            '@' => {
                toks.push(Tok::At);
                i += 1;
            }
            '0'..='9' => {
                let (t, next) = scan_number(&chars, i)?;
                toks.push(t);
                i = next;
            }
            _ if c.is_alphabetic() || c == '_' => {
                let start = i;
                while i < n && (chars[i].is_alphanumeric() || chars[i] == '_') {
                    i += 1;
                }
                let word: String = chars[start..i].iter().collect();
                // string prefix? (r"", f"", b"", u"", combos) — adjacent quote
                if word.len() <= 2
                    && word.chars().all(|ch| matches!(ch, 'r' | 'b' | 'f' | 'u' | 'R' | 'B' | 'F' | 'U'))
                    && i < n
                    && (chars[i] == '\'' || chars[i] == '"')
                {
                    let has_f = word.contains('f') || word.contains('F');
                    let has_r = word.contains('r') || word.contains('R');
                    if has_f {
                        // f-string: ast.JoinedStr — not in the allowed set
                        return Err(violation("JoinedStr is not in the allowed node set"));
                    }
                    let (s, next) = scan_string(&chars, i, has_r)?;
                    toks.push(Tok::Str(s));
                    i = next;
                    continue;
                }
                if let Some(kw) = kw_of(&word) {
                    toks.push(Tok::Kw(kw));
                } else {
                    toks.push(Tok::Name(word));
                }
            }
            other => {
                return Err(syntax_err(format!("unexpected character '{}'", other)));
            }
        }
    }
    // ensure the last logical line is terminated
    if !toks.is_empty() && !matches!(toks.last(), Some(Tok::Newline)) {
        toks.push(Tok::Newline);
    }
    while stack.len() > 1 {
        stack.pop();
        toks.push(Tok::Dedent);
    }
    toks.push(Tok::Eof);
    Ok(toks)
}

fn scan_number(chars: &[char], start: usize) -> Result<(Tok, usize), SandboxError> {
    let n = chars.len();
    let mut i = start;
    // hex / octal / binary
    if chars[i] == '0' && i + 1 < n && matches!(chars[i + 1], 'x' | 'X' | 'o' | 'O' | 'b' | 'B') {
        let base = match chars[i + 1] {
            'x' | 'X' => 16,
            'o' | 'O' => 8,
            _ => 2,
        };
        i += 2;
        let ds = i;
        while i < n && (chars[i].is_ascii_alphanumeric() || chars[i] == '_') {
            i += 1;
        }
        let text: String = chars[ds..i].iter().filter(|&&c| c != '_').collect();
        let v = i64::from_str_radix(&text, base).map_err(|_| syntax_err("invalid literal"))?;
        return Ok((Tok::Int(v), i));
    }
    let mut saw_dot = false;
    let mut saw_exp = false;
    while i < n && (chars[i].is_ascii_digit() || chars[i] == '_') {
        i += 1;
    }
    if i < n && chars[i] == '.' {
        if i + 1 < n && chars[i + 1].is_ascii_digit() {
            saw_dot = true;
            i += 1;
            while i < n && (chars[i].is_ascii_digit() || chars[i] == '_') {
                i += 1;
            }
        } else if i + 1 < n && (chars[i + 1].is_alphabetic() || chars[i + 1] == '_') {
            // "1.x" — stop the number here; the '.' becomes an attribute
            // dot that the parser rejects (Python: syntax error on bare
            // int-literal attribute access, rejection either way)
        } else {
            saw_dot = true; // "1." trailing dot
            i += 1;
        }
    }
    if i < n && (chars[i] == 'e' || chars[i] == 'E') {
        let mut j = i + 1;
        if j < n && (chars[j] == '+' || chars[j] == '-') {
            j += 1;
        }
        if j < n && chars[j].is_ascii_digit() {
            while j < n && chars[j].is_ascii_digit() {
                j += 1;
            }
            i = j;
            saw_exp = true;
        }
    }
    let text: String = chars[start..i].iter().filter(|&&c| c != '_').collect();
    if saw_dot || saw_exp {
        let v: f64 = text.parse().map_err(|_| syntax_err("invalid literal"))?;
        Ok((Tok::Float(v), i))
    } else {
        match text.parse::<i64>() {
            Ok(v) => Ok((Tok::Int(v), i)),
            Err(_) => {
                // huge int literal: Python ints are unbounded; keep the value
                // as a float rather than failing (deviation, documented)
                let v: f64 = text.parse().map_err(|_| syntax_err("invalid literal"))?;
                Ok((Tok::Float(v), i))
            }
        }
    }
}

fn scan_string(chars: &[char], start: usize, raw: bool) -> Result<(String, usize), SandboxError> {
    let quote = chars[start];
    let mut i = start + 1;
    let triple = chars.get(i) == Some(&quote) && chars.get(i + 1) == Some(&quote);
    if triple {
        i += 2;
    }
    let mut out = String::new();
    loop {
        let c = *chars
            .get(i)
            .ok_or_else(|| syntax_err("unterminated string literal"))?;
        if triple {
            if c == quote
                && chars.get(i + 1) == Some(&quote)
                && chars.get(i + 2) == Some(&quote)
            {
                return Ok((out, i + 3));
            }
        } else {
            if c == quote {
                return Ok((out, i + 1));
            }
            if c == '\n' {
                return Err(syntax_err("EOL while scanning string literal"));
            }
        }
        if c == '\\' {
            if raw {
                out.push('\\');
                let nc = *chars
                    .get(i + 1)
                    .ok_or_else(|| syntax_err("unterminated string literal"))?;
                out.push(nc);
                i += 2;
                continue;
            }
            let e = *chars
                .get(i + 1)
                .ok_or_else(|| syntax_err("unterminated string literal"))?;
            i += 2;
            match e {
                'n' => out.push('\n'),
                't' => out.push('\t'),
                'r' => out.push('\r'),
                'a' => out.push('\u{07}'),
                'b' => out.push('\u{08}'),
                'f' => out.push('\u{0c}'),
                'v' => out.push('\u{0b}'),
                '0' => out.push('\0'),
                '\\' => out.push('\\'),
                '\'' => out.push('\''),
                '"' => out.push('"'),
                '\n' => {} // line continuation inside string
                'x' => {
                    let (ch, consumed) = hex_escape(chars, i, 2)?;
                    out.push(ch);
                    i = consumed;
                }
                'u' => {
                    let (ch, consumed) = hex_escape(chars, i, 4)?;
                    out.push(ch);
                    i = consumed;
                }
                other => {
                    out.push('\\');
                    out.push(other);
                }
            }
        } else {
            out.push(c);
            i += 1;
        }
    }
}

fn hex_escape(chars: &[char], at: usize, count: usize) -> Result<(char, usize), SandboxError> {
    let mut v: u32 = 0;
    for k in 0..count {
        let c = *chars
            .get(at + k)
            .ok_or_else(|| syntax_err("invalid \\x/\\u escape"))?;
        let d = c
            .to_digit(16)
            .ok_or_else(|| syntax_err("invalid hex escape in string"))?;
        v = v * 16 + d;
    }
    let ch = char::from_u32(v).ok_or_else(|| syntax_err("invalid unicode escape"))?;
    Ok((ch, at + count))
}

// ---------------------------------------------------------------------------
// Sandbox AST
// ---------------------------------------------------------------------------

#[derive(Debug, Clone)]
enum Expr {
    Int(i64),
    Float(f64),
    Str(String),
    BoolLit(bool),
    NoneLit,
    Name(String),
    ListLit(Vec<Expr>),
    TupleLit(Vec<Expr>),
    Bin(BinOp, Box<Expr>, Box<Expr>),
    Unary(UnOp, Box<Expr>),
    BoolAnd(Vec<Expr>),
    BoolOr(Vec<Expr>),
    /// Chained comparison: `a < b <= c`
    Compare(Box<Expr>, Vec<(CmpOp, Expr)>),
    /// Call — the callee is always a plain name (parser-enforced).
    Call(String, Vec<Expr>),
    /// `base[index]` or `base[lo:hi:step]`
    Sub(Box<Expr>, Box<SIdx>),
}

#[derive(Debug, Clone)]
enum SIdx {
    Idx(Box<Expr>),
    Slice(Option<Expr>, Option<Expr>, Option<Expr>),
}

#[derive(Debug, Clone)]
enum Target {
    Name(String),
    Tuple(Vec<Target>),
    List(Vec<Target>),
    Sub(Box<Expr>, SIdx),
}

#[derive(Debug, Clone)]
enum Stmt {
    FuncDef(Rc<FuncDef>),
    Return(Option<Expr>),
    Assign { targets: Vec<Target>, value: Expr },
    AugAssign { op: BinOp, target: Target, value: Expr },
    If {
        arms: Vec<(Expr, Vec<Stmt>)>,
        else_body: Option<Vec<Stmt>>,
    },
    For {
        target: Target,
        iter: Expr,
        body: Vec<Stmt>,
    },
    While { cond: Expr, body: Vec<Stmt> },
    Break,
    Continue,
    ExprStmt(Expr),
    /// `;`-separated simple statements on one line
    Seq(Vec<Stmt>),
}

#[derive(Debug, Clone)]
struct FuncDef {
    name: String,
    params: Vec<Param>,
    body: Vec<Stmt>,
}

#[derive(Debug, Clone)]
struct Param {
    name: String,
    default: Option<Expr>,
}

// ---------------------------------------------------------------------------
// Parser (recursive descent, whitelist enforcement inline)
// ---------------------------------------------------------------------------

const MAX_PARSE_DEPTH: usize = 200;
const MAX_CALL_DEPTH: usize = 512;

struct Parser {
    toks: Vec<Tok>,
    pos: usize,
    depth: usize,
}

fn parse_module(code: &str) -> Result<Vec<Stmt>, SandboxError> {
    let toks = tokenize(code)?;
    let mut p = Parser {
        toks,
        pos: 0,
        depth: 0,
    };
    p.parse_program()
}

/// Parse an eval-mode expression (the `call_expr` argument).
fn parse_expression(src: &str) -> Result<Expr, SandboxError> {
    let toks = tokenize(src)?;
    let mut p = Parser {
        toks,
        pos: 0,
        depth: 0,
    };
    let e = p.parse_expr_tuple()?;
    while matches!(p.peek(), Some(Tok::Newline)) {
        p.pos += 1;
    }
    match p.peek() {
        Some(Tok::Eof) | None => Ok(e),
        _ => Err(syntax_err("unexpected trailing tokens in expression")),
    }
}

impl Parser {
    fn peek(&self) -> Option<&Tok> {
        self.toks.get(self.pos)
    }

    fn peek2(&self) -> Option<&Tok> {
        self.toks.get(self.pos + 1)
    }

    fn next(&mut self) -> Option<Tok> {
        let t = self.toks.get(self.pos).cloned();
        if t.is_some() {
            self.pos += 1;
        }
        t
    }

    fn expect_colon(&mut self) -> Result<(), SandboxError> {
        match self.next() {
            Some(Tok::Colon) => Ok(()),
            _ => Err(syntax_err("expected ':'")),
        }
    }

    fn parse_program(&mut self) -> Result<Vec<Stmt>, SandboxError> {
        let mut out = Vec::new();
        while !matches!(self.peek(), Some(Tok::Eof) | None) {
            out.push(self.parse_statement()?);
        }
        Ok(out)
    }

    fn parse_statement(&mut self) -> Result<Stmt, SandboxError> {
        self.depth += 1;
        let r = if self.depth > MAX_PARSE_DEPTH {
            Err(syntax_err("too deeply nested"))
        } else {
            self.parse_statement_inner()
        };
        self.depth -= 1;
        r
    }

    fn parse_statement_inner(&mut self) -> Result<Stmt, SandboxError> {
        match self.peek() {
            Some(Tok::Kw(Kw::Def)) => self.parse_def(),
            Some(Tok::Kw(Kw::If)) => self.parse_if(),
            Some(Tok::Kw(Kw::While)) => {
                self.pos += 1;
                let cond = self.parse_expr_tuple()?;
                self.expect_colon()?;
                let body = self.parse_block()?;
                Ok(Stmt::While { cond, body })
            }
            Some(Tok::Kw(Kw::For)) => {
                self.pos += 1;
                let target = self.parse_for_target()?;
                match self.next() {
                    Some(Tok::Kw(Kw::In)) => {}
                    _ => return Err(syntax_err("expected 'in' after for-loop target")),
                }
                let iter = self.parse_expr_tuple()?;
                self.expect_colon()?;
                let body = self.parse_block()?;
                Ok(Stmt::For { target, iter, body })
            }
            _ => {
                let stmts = self.parse_simple_stmt_list()?;
                if stmts.len() == 1 {
                    Ok(stmts.into_iter().next().unwrap())
                } else {
                    Ok(Stmt::Seq(stmts))
                }
            }
        }
    }

    fn parse_def(&mut self) -> Result<Stmt, SandboxError> {
        self.pos += 1; // 'def'
        let name = match self.next() {
            Some(Tok::Name(n)) => n,
            _ => return Err(syntax_err("expected function name after 'def'")),
        };
        if _FORBIDDEN_NAMES.contains(&name.as_str()) {
            return Err(violation(format!("forbidden name: {}", name)));
        }
        match self.next() {
            Some(Tok::LParen) => {}
            _ => return Err(syntax_err("expected '(' after function name")),
        }
        let mut params: Vec<Param> = Vec::new();
        if !matches!(self.peek(), Some(Tok::RParen)) {
            loop {
                match self.peek() {
                    Some(Tok::Name(pn)) => {
                        let pname = pn.clone();
                        self.pos += 1;
                        let default = if matches!(self.peek(), Some(Tok::Assign)) {
                            self.pos += 1;
                            Some(self.parse_expr()?)
                        } else {
                            None
                        };
                        if _FORBIDDEN_NAMES.contains(&pname.as_str()) {
                            return Err(violation(format!("forbidden name: {}", pname)));
                        }
                        params.push(Param { name: pname, default });
                    }
                    Some(Tok::Star) => {
                        return Err(violation("Starred is not in the allowed node set"));
                    }
                    Some(Tok::DoubleStar) => {
                        return Err(violation("keyword is not in the allowed node set"));
                    }
                    _ => return Err(syntax_err("expected parameter name")),
                }
                match self.peek() {
                    Some(Tok::Comma) => {
                        self.pos += 1;
                        if matches!(self.peek(), Some(Tok::RParen)) {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        }
        match self.next() {
            Some(Tok::RParen) => {}
            _ => return Err(syntax_err("expected ')' after parameters")),
        }
        if matches!(self.peek(), Some(Tok::Arrow)) {
            return Err(syntax_err("return annotations are not supported"));
        }
        self.expect_colon()?;
        let body = self.parse_block()?;
        // no nested defs in this subset (the Python whitelist technically
        // permits them; rejecting them keeps the interpreter flat-scoped)
        for s in &body {
            if matches!(s, Stmt::FuncDef(_)) {
                return Err(syntax_err("nested function definitions are not supported"));
            }
        }
        Ok(Stmt::FuncDef(Rc::new(FuncDef { name, params, body })))
    }

    fn parse_if(&mut self) -> Result<Stmt, SandboxError> {
        self.pos += 1; // 'if' or 'elif'
        let cond = self.parse_expr_tuple()?;
        self.expect_colon()?;
        let then = self.parse_block()?;
        let mut arms = vec![(cond, then)];
        let mut else_body = None;
        loop {
            match self.peek() {
                Some(Tok::Kw(Kw::Elif)) => {
                    self.pos += 1;
                    let c = self.parse_expr_tuple()?;
                    self.expect_colon()?;
                    let b = self.parse_block()?;
                    arms.push((c, b));
                }
                Some(Tok::Kw(Kw::Else)) => {
                    self.pos += 1;
                    self.expect_colon()?;
                    else_body = Some(self.parse_block()?);
                    break;
                }
                _ => break,
            }
        }
        Ok(Stmt::If { arms, else_body })
    }

    fn parse_block(&mut self) -> Result<Vec<Stmt>, SandboxError> {
        match self.peek() {
            Some(Tok::Newline) => {
                self.pos += 1;
                if !matches!(self.peek(), Some(Tok::Indent)) {
                    return Err(syntax_err("expected an indented block"));
                }
                self.pos += 1;
                let mut out: Vec<Stmt> = Vec::new();
                loop {
                    match self.peek() {
                        Some(Tok::Dedent) => {
                            self.pos += 1;
                            break;
                        }
                        Some(Tok::Eof) => break,
                        _ => out.push(self.parse_statement()?),
                    }
                }
                Ok(out)
            }
            _ => self.parse_simple_stmt_list(),
        }
    }

    fn parse_simple_stmt_list(&mut self) -> Result<Vec<Stmt>, SandboxError> {
        let mut out = vec![self.parse_simple_stmt()?];
        while matches!(self.peek(), Some(Tok::Semi)) {
            self.pos += 1;
            if matches!(self.peek(), Some(Tok::Newline) | Some(Tok::Eof) | None) {
                break;
            }
            out.push(self.parse_simple_stmt()?);
        }
        match self.next() {
            Some(Tok::Newline) | Some(Tok::Eof) => Ok(out),
            _ => Err(syntax_err("expected end of statement")),
        }
    }

    fn parse_simple_stmt(&mut self) -> Result<Stmt, SandboxError> {
        match self.peek() {
            Some(Tok::Kw(Kw::Return)) => {
                self.pos += 1;
                let value = if matches!(
                    self.peek(),
                    Some(Tok::Newline) | Some(Tok::Semi) | Some(Tok::Eof) | None
                ) {
                    None
                } else {
                    Some(self.parse_expr_tuple()?)
                };
                Ok(Stmt::Return(value))
            }
            Some(Tok::Kw(Kw::Break)) => {
                self.pos += 1;
                Ok(Stmt::Break)
            }
            Some(Tok::Kw(Kw::Continue)) => {
                self.pos += 1;
                Ok(Stmt::Continue)
            }
            Some(Tok::Kw(Kw::Pass)) => Err(violation("Pass is not in the allowed node set")),
            Some(Tok::Kw(Kw::Import)) => Err(violation("Import is not in the allowed node set")),
            Some(Tok::Kw(Kw::From)) => {
                Err(violation("ImportFrom is not in the allowed node set"))
            }
            Some(Tok::Kw(Kw::Class)) => Err(violation("ClassDef is not in the allowed node set")),
            Some(Tok::Kw(Kw::Assert)) => Err(violation("Assert is not in the allowed node set")),
            Some(Tok::Kw(Kw::Try)) => Err(violation("Try is not in the allowed node set")),
            Some(Tok::Kw(Kw::Except)) | Some(Tok::Kw(Kw::Finally)) => {
                Err(violation("Try is not in the allowed node set"))
            }
            Some(Tok::Kw(Kw::With)) => Err(violation("With is not in the allowed node set")),
            Some(Tok::Kw(Kw::Raise)) => Err(violation("Raise is not in the allowed node set")),
            Some(Tok::Kw(Kw::Del)) => Err(violation("Delete is not in the allowed node set")),
            Some(Tok::Kw(Kw::Global)) => Err(violation("Global is not in the allowed node set")),
            Some(Tok::Kw(Kw::Nonlocal)) => {
                Err(violation("Nonlocal is not in the allowed node set"))
            }
            Some(Tok::Kw(Kw::Yield)) => Err(violation("Yield is not in the allowed node set")),
            Some(Tok::Kw(Kw::Async)) => {
                Err(violation("AsyncFunctionDef is not in the allowed node set"))
            }
            Some(Tok::Kw(Kw::Await)) => Err(violation("Await is not in the allowed node set")),
            Some(Tok::Kw(Kw::Lambda)) => Err(violation("Lambda is not in the allowed node set")),
            Some(_) | None => {
                let e = self.parse_expr_tuple()?;
                match self.peek() {
                    Some(Tok::Assign) => {
                        let mut targets = vec![check_target(e)?];
                        self.pos += 1;
                        loop {
                            let v = self.parse_expr_tuple()?;
                            if matches!(self.peek(), Some(Tok::Assign)) {
                                self.pos += 1;
                                targets.push(check_target(v)?);
                                continue;
                            }
                            return Ok(Stmt::Assign { targets, value: v });
                        }
                    }
                    Some(t) if is_aug_token(t) => {
                        let op = aug_op_of(t);
                        self.pos += 1;
                        let target = check_aug_target(e)?;
                        let value = self.parse_expr_tuple()?;
                        Ok(Stmt::AugAssign { op, target, value })
                    }
                    _ => Ok(Stmt::ExprStmt(e)),
                }
            }
        }
    }

    // ----- expression parsing (Python precedence) -----

    /// A `for` loop target: a name, or a comma-separated (optionally
    /// parenthesized) tuple of names. Parsed WITHOUT the general
    /// expression parser so the following `in` keyword is never
    /// consumed as a comparison operator.
    fn parse_for_target(&mut self) -> Result<Target, SandboxError> {
        let paren = matches!(self.peek(), Some(Tok::LParen));
        if paren {
            self.pos += 1;
        }
        let mut names: Vec<Target> = Vec::new();
        loop {
            match self.next() {
                Some(Tok::Name(n)) => names.push(Target::Name(n)),
                other => {
                    return Err(syntax_err(&format!(
                        "invalid for-loop target {:?}",
                        other
                    )))
                }
            }
            if matches!(self.peek(), Some(Tok::Comma)) {
                self.pos += 1;
                continue;
            }
            break;
        }
        if paren {
            match self.next() {
                Some(Tok::RParen) => {}
                _ => return Err(syntax_err("expected ')' after tuple target")),
            }
        }
        if names.len() == 1 && !paren {
            Ok(names.pop().unwrap())
        } else {
            Ok(Target::Tuple(names))
        }
    }

    /// `a, b, c` → tuple (top level only: statements, returns, parens)
    fn parse_expr_tuple(&mut self) -> Result<Expr, SandboxError> {
        let first = self.parse_expr()?;
        if !matches!(self.peek(), Some(Tok::Comma)) {
            return Ok(first);
        }
        let mut items = vec![first];
        while matches!(self.peek(), Some(Tok::Comma)) {
            self.pos += 1;
            if matches!(
                self.peek(),
                Some(Tok::Newline) | Some(Tok::Semi) | Some(Tok::Eof) | Some(Tok::Eq) | None
            ) {
                break;
            }
            items.push(self.parse_expr()?);
        }
        Ok(Expr::TupleLit(items))
    }

    /// Full expression, no top-level comma. Ternary is rejected (IfExp).
    fn parse_expr(&mut self) -> Result<Expr, SandboxError> {
        let e = self.parse_or()?;
        if matches!(self.peek(), Some(Tok::Kw(Kw::If))) {
            return Err(violation("IfExp is not in the allowed node set"));
        }
        Ok(e)
    }

    fn parse_or(&mut self) -> Result<Expr, SandboxError> {
        let mut operands = vec![self.parse_and()?];
        while matches!(self.peek(), Some(Tok::Kw(Kw::Or))) {
            self.pos += 1;
            operands.push(self.parse_and()?);
        }
        if operands.len() == 1 {
            Ok(operands.pop().unwrap())
        } else {
            Ok(Expr::BoolOr(operands))
        }
    }

    fn parse_and(&mut self) -> Result<Expr, SandboxError> {
        let mut operands = vec![self.parse_not()?];
        while matches!(self.peek(), Some(Tok::Kw(Kw::And))) {
            self.pos += 1;
            operands.push(self.parse_not()?);
        }
        if operands.len() == 1 {
            Ok(operands.pop().unwrap())
        } else {
            Ok(Expr::BoolAnd(operands))
        }
    }

    fn parse_not(&mut self) -> Result<Expr, SandboxError> {
        if matches!(self.peek(), Some(Tok::Kw(Kw::Not))) {
            self.pos += 1;
            return Ok(Expr::Unary(UnOp::Not, Box::new(self.parse_not()?)));
        }
        self.parse_comparison()
    }

    fn parse_comparison(&mut self) -> Result<Expr, SandboxError> {
        let lhs = self.parse_arith()?;
        let mut ops: Vec<(CmpOp, Expr)> = Vec::new();
        loop {
            let op = match self.peek() {
                Some(Tok::Eq) => CmpOp::Eq,
                Some(Tok::NotEq) => CmpOp::NotEq,
                Some(Tok::Lt) => CmpOp::Lt,
                Some(Tok::LtE) => CmpOp::LtE,
                Some(Tok::Gt) => CmpOp::Gt,
                Some(Tok::GtE) => CmpOp::GtE,
                Some(Tok::Kw(Kw::In)) => {
                    return Err(violation("In is not in the allowed node set"))
                }
                Some(Tok::Kw(Kw::Is)) => {
                    return Err(violation(if matches!(self.peek2(), Some(Tok::Kw(Kw::Not))) {
                        "IsNot is not in the allowed node set"
                    } else {
                        "Is is not in the allowed node set"
                    }));
                }
                Some(Tok::Kw(Kw::Not)) => {
                    return Err(violation("NotIn is not in the allowed node set"))
                }
                _ => break,
            };
            self.pos += 1;
            let rhs = self.parse_arith()?;
            ops.push((op, rhs));
        }
        if ops.is_empty() {
            Ok(lhs)
        } else {
            Ok(Expr::Compare(Box::new(lhs), ops))
        }
    }

    fn parse_arith(&mut self) -> Result<Expr, SandboxError> {
        let mut lhs = self.parse_term()?;
        loop {
            match self.peek() {
                Some(Tok::Plus) => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    lhs = Expr::Bin(BinOp::Add, Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Minus) => {
                    self.pos += 1;
                    let rhs = self.parse_term()?;
                    lhs = Expr::Bin(BinOp::Sub, Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Amp) => {
                    return Err(violation("BitAnd is not in the allowed node set"))
                }
                Some(Tok::Pipe) => {
                    return Err(violation("BitOr is not in the allowed node set"))
                }
                Some(Tok::Caret) => {
                    return Err(violation("BitXor is not in the allowed node set"))
                }
                Some(Tok::Shl) => {
                    return Err(violation("LShift is not in the allowed node set"))
                }
                Some(Tok::Shr) => {
                    return Err(violation("RShift is not in the allowed node set"))
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<Expr, SandboxError> {
        let mut lhs = self.parse_factor()?;
        loop {
            match self.peek() {
                Some(Tok::Star) => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    lhs = Expr::Bin(BinOp::Mul, Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Slash) => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    lhs = Expr::Bin(BinOp::Div, Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::DoubleSlash) => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    lhs = Expr::Bin(BinOp::FloorDiv, Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::Percent) => {
                    self.pos += 1;
                    let rhs = self.parse_factor()?;
                    lhs = Expr::Bin(BinOp::Mod, Box::new(lhs), Box::new(rhs));
                }
                Some(Tok::At) => {
                    return Err(violation("MatMult is not in the allowed node set"))
                }
                Some(Tok::Amp) => {
                    return Err(violation("BitAnd is not in the allowed node set"))
                }
                Some(Tok::Pipe) => {
                    return Err(violation("BitOr is not in the allowed node set"))
                }
                Some(Tok::Caret) => {
                    return Err(violation("BitXor is not in the allowed node set"))
                }
                Some(Tok::Shl) => {
                    return Err(violation("LShift is not in the allowed node set"))
                }
                Some(Tok::Shr) => {
                    return Err(violation("RShift is not in the allowed node set"))
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_factor(&mut self) -> Result<Expr, SandboxError> {
        match self.peek() {
            Some(Tok::Minus) => {
                self.pos += 1;
                Ok(Expr::Unary(UnOp::Neg, Box::new(self.parse_factor()?)))
            }
            Some(Tok::Plus) => {
                self.pos += 1;
                Ok(Expr::Unary(UnOp::Pos, Box::new(self.parse_factor()?)))
            }
            Some(Tok::Tilde) => Err(violation("Invert is not in the allowed node set")),
            _ => self.parse_power(),
        }
    }

    fn parse_power(&mut self) -> Result<Expr, SandboxError> {
        let base = self.parse_atom_postfix()?;
        if matches!(self.peek(), Some(Tok::DoubleStar)) {
            self.pos += 1;
            let exp = self.parse_factor()?;
            return Ok(Expr::Bin(BinOp::Pow, Box::new(base), Box::new(exp)));
        }
        Ok(base)
    }

    fn parse_atom_postfix(&mut self) -> Result<Expr, SandboxError> {
        let mut e = self.parse_atom()?;
        loop {
            match self.peek() {
                Some(Tok::Dot) => {
                    self.pos += 1;
                    let attr = match self.next() {
                        Some(Tok::Name(n)) => n,
                        _ => return Err(syntax_err("expected attribute name after '.'")),
                    };
                    // Python's AST walk hits the Call node before the
                    // Attribute for `x.f(...)`: method calls get the
                    // method-call message, plain attribute reads get the
                    // attribute message.
                    if matches!(self.peek(), Some(Tok::LParen)) {
                        return Err(violation(
                            "only direct name calls are allowed (no method calls)",
                        ));
                    }
                    return Err(violation(format!("attribute access not allowed: .{}", attr)));
                }
                Some(Tok::LBracket) => {
                    self.pos += 1;
                    let idx = self.parse_subscript()?;
                    match self.next() {
                        Some(Tok::RBracket) => {}
                        _ => return Err(syntax_err("expected ']'")),
                    }
                    e = Expr::Sub(Box::new(e), Box::new(idx));
                }
                Some(Tok::LParen) => match e {
                    Expr::Name(ref n) => {
                        let n = n.clone();
                        self.pos += 1;
                        let args = self.parse_call_args()?;
                        e = Expr::Call(n, args);
                    }
                    _ => {
                        return Err(violation(
                            "only direct name calls are allowed (no method calls)",
                        ))
                    }
                },
                _ => break,
            }
        }
        Ok(e)
    }

    fn parse_subscript(&mut self) -> Result<SIdx, SandboxError> {
        let lower = if matches!(self.peek(), Some(Tok::Colon) | Some(Tok::RBracket)) {
            None
        } else {
            Some(self.parse_expr()?)
        };
        if matches!(self.peek(), Some(Tok::Colon)) {
            self.pos += 1;
            let upper = if matches!(self.peek(), Some(Tok::Colon) | Some(Tok::RBracket)) {
                None
            } else {
                Some(self.parse_expr()?)
            };
            let step = if matches!(self.peek(), Some(Tok::Colon)) {
                self.pos += 1;
                if matches!(self.peek(), Some(Tok::RBracket)) {
                    None
                } else {
                    Some(self.parse_expr()?)
                }
            } else {
                None
            };
            Ok(SIdx::Slice(lower, upper, step))
        } else {
            match lower {
                Some(e) => Ok(SIdx::Idx(Box::new(e))),
                None => Err(syntax_err("expected index expression")),
            }
        }
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, SandboxError> {
        let mut args = Vec::new();
        if !matches!(self.peek(), Some(Tok::RParen)) {
            loop {
                match self.peek() {
                    Some(Tok::DoubleStar) => {
                        return Err(violation("keyword is not in the allowed node set"))
                    }
                    Some(Tok::Star) => {
                        return Err(violation("Starred is not in the allowed node set"))
                    }
                    Some(Tok::Name(_)) if matches!(self.peek2(), Some(Tok::Assign)) => {
                        return Err(violation("keyword is not in the allowed node set"))
                    }
                    _ => {}
                }
                args.push(self.parse_expr()?);
                match self.peek() {
                    Some(Tok::Comma) => {
                        self.pos += 1;
                        if matches!(self.peek(), Some(Tok::RParen)) {
                            break;
                        }
                    }
                    _ => break,
                }
            }
        }
        match self.next() {
            Some(Tok::RParen) => Ok(args),
            _ => Err(syntax_err("expected ')' after call arguments")),
        }
    }

    fn parse_atom(&mut self) -> Result<Expr, SandboxError> {
        self.depth += 1;
        let r = if self.depth > MAX_PARSE_DEPTH {
            Err(syntax_err("too deeply nested"))
        } else {
            self.parse_atom_inner()
        };
        self.depth -= 1;
        r
    }

    fn parse_atom_inner(&mut self) -> Result<Expr, SandboxError> {
        match self.next() {
            Some(Tok::Int(v)) => Ok(Expr::Int(v)),
            Some(Tok::Float(v)) => Ok(Expr::Float(v)),
            Some(Tok::Str(s)) => Ok(Expr::Str(s)),
            Some(Tok::Kw(Kw::True)) => Ok(Expr::BoolLit(true)),
            Some(Tok::Kw(Kw::False)) => Ok(Expr::BoolLit(false)),
            Some(Tok::Kw(Kw::None)) => Ok(Expr::NoneLit),
            Some(Tok::Name(n)) => {
                if _FORBIDDEN_NAMES.contains(&n.as_str()) {
                    return Err(violation(format!("forbidden name: {}", n)));
                }
                Ok(Expr::Name(n))
            }
            Some(Tok::LParen) => {
                if matches!(self.peek(), Some(Tok::RParen)) {
                    self.pos += 1;
                    return Ok(Expr::TupleLit(vec![]));
                }
                let e = self.parse_expr_tuple()?;
                if matches!(self.peek(), Some(Tok::Kw(Kw::For))) {
                    return Err(violation("GeneratorExp is not in the allowed node set"));
                }
                match self.next() {
                    Some(Tok::RParen) => Ok(e),
                    _ => Err(syntax_err("expected ')'")),
                }
            }
            Some(Tok::LBracket) => {
                if matches!(self.peek(), Some(Tok::RBracket)) {
                    self.pos += 1;
                    return Ok(Expr::ListLit(vec![]));
                }
                let e = self.parse_expr()?;
                if matches!(self.peek(), Some(Tok::Kw(Kw::For))) {
                    return Err(violation("ListComp is not in the allowed node set"));
                }
                let mut items = vec![e];
                while matches!(self.peek(), Some(Tok::Comma)) {
                    self.pos += 1;
                    if matches!(self.peek(), Some(Tok::RBracket)) {
                        break;
                    }
                    items.push(self.parse_expr()?);
                }
                match self.next() {
                    Some(Tok::RBracket) => Ok(Expr::ListLit(items)),
                    _ => Err(syntax_err("expected ']'")),
                }
            }
            Some(Tok::LBrace) => {
                if matches!(self.peek(), Some(Tok::RBrace)) {
                    self.pos += 1;
                    return Err(violation("Dict is not in the allowed node set"));
                }
                let _e = self.parse_expr()?;
                if matches!(self.peek(), Some(Tok::Colon)) {
                    return Err(violation("Dict is not in the allowed node set"));
                }
                if matches!(self.peek(), Some(Tok::Kw(Kw::For))) {
                    return Err(violation("SetComp is not in the allowed node set"));
                }
                Err(violation("Set is not in the allowed node set"))
            }
            Some(Tok::Kw(Kw::Not)) => Err(syntax_err("unexpected keyword 'not'")),
            Some(Tok::Kw(Kw::Lambda)) => Err(violation("Lambda is not in the allowed node set")),
            Some(Tok::Kw(Kw::Await)) => Err(violation("Await is not in the allowed node set")),
            Some(Tok::Kw(Kw::Yield)) => Err(violation("Yield is not in the allowed node set")),
            Some(Tok::Kw(k)) => Err(syntax_err(format!("unexpected keyword '{:?}'", k))),
            other => Err(syntax_err(format!("unexpected token {:?}", other))),
        }
    }
}

fn is_aug_token(t: &Tok) -> bool {
    matches!(
        t,
        Tok::PlusA
            | Tok::MinusA
            | Tok::StarA
            | Tok::SlashA
            | Tok::DoubleSlashA
            | Tok::PercentA
            | Tok::DoubleStarA
    )
}

fn aug_op_of(t: &Tok) -> BinOp {
    match t {
        Tok::PlusA => BinOp::Add,
        Tok::MinusA => BinOp::Sub,
        Tok::StarA => BinOp::Mul,
        Tok::SlashA => BinOp::Div,
        Tok::DoubleSlashA => BinOp::FloorDiv,
        Tok::PercentA => BinOp::Mod,
        _ => BinOp::Pow,
    }
}

fn check_target(e: Expr) -> Result<Target, SandboxError> {
    match e {
        Expr::Name(n) => Ok(Target::Name(n)),
        Expr::TupleLit(es) => Ok(Target::Tuple(
            es.into_iter().map(check_target).collect::<Result<_, _>>()?,
        )),
        Expr::ListLit(es) => Ok(Target::List(
            es.into_iter().map(check_target).collect::<Result<_, _>>()?,
        )),
        Expr::Sub(base, idx) => Ok(Target::Sub(base, *idx)),
        _ => Err(syntax_err("cannot assign to this expression")),
    }
}

fn check_aug_target(e: Expr) -> Result<Target, SandboxError> {
    match check_target(e)? {
        t @ Target::Name(_) => Ok(t),
        t @ Target::Sub(_, _) => Ok(t),
        _ => Err(syntax_err("illegal expression for augmented assignment")),
    }
}

// ---------------------------------------------------------------------------
// Interpreter
// ---------------------------------------------------------------------------

#[derive(Clone)]
enum Binding {
    Val(SandboxValue),
    Func(Rc<RuntimeFunc>),
}

struct RuntimeFunc {
    def: Rc<FuncDef>,
    /// default parameter values, evaluated once at definition time
    defaults: Vec<Option<SandboxValue>>,
}

enum Flow {
    Normal,
    Return(Option<SandboxValue>),
    Break,
    Continue,
}

type Locals = HashMap<String, Binding>;

struct Interp {
    globals: Locals,
    steps: usize,
    max_steps: usize,
    depth: usize,
}

impl Interp {
    fn tick(&mut self) -> Result<(), SandboxError> {
        self.steps += 1;
        if self.steps > self.max_steps {
            return Err(SandboxError::Timeout(format!(
                "exceeded {} execution steps",
                self.max_steps
            )));
        }
        Ok(())
    }

    fn lookup<'a>(&'a self, name: &str, locals: Option<&'a Locals>) -> Option<&'a Binding> {
        if let Some(l) = locals {
            if let Some(b) = l.get(name) {
                return Some(b);
            }
        }
        self.globals.get(name)
    }

    fn store(&mut self, name: &str, b: Binding, locals: Option<&mut Locals>) {
        match locals {
            Some(l) => {
                l.insert(name.to_string(), b);
            }
            None => {
                self.globals.insert(name.to_string(), b);
            }
        }
    }

    fn exec_block(
        &mut self,
        stmts: &[Stmt],
        mut locals: Option<&mut Locals>,
    ) -> Result<Flow, SandboxError> {
        for s in stmts {
            match self.exec_stmt(s, locals.as_deref_mut())? {
                Flow::Normal => {}
                f => return Ok(f),
            }
        }
        Ok(Flow::Normal)
    }

    fn exec_stmt(
        &mut self,
        s: &Stmt,
        mut locals: Option<&mut Locals>,
    ) -> Result<Flow, SandboxError> {
        self.tick()?;
        match s {
            Stmt::FuncDef(def) => {
                let mut defaults = Vec::with_capacity(def.params.len());
                for p in &def.params {
                    defaults.push(match &p.default {
                        Some(d) => Some(self.eval(d, locals.as_deref_mut())?),
                        None => None,
                    });
                }
                self.store(
                    &def.name,
                    Binding::Func(Rc::new(RuntimeFunc {
                        def: def.clone(),
                        defaults,
                    })),
                    locals,
                );
                Ok(Flow::Normal)
            }
            Stmt::Return(e) => {
                let v = match e {
                    Some(e) => Some(self.eval(e, locals.as_deref_mut())?),
                    None => None,
                };
                Ok(Flow::Return(v))
            }
            Stmt::Assign { targets, value } => {
                let v = self.eval(value, locals.as_deref_mut())?;
                for t in targets {
                    self.assign_target(t, v.clone(), locals.as_deref_mut())?;
                }
                Ok(Flow::Normal)
            }
            Stmt::AugAssign { op, target, value } => {
                let val = self.eval(value, locals.as_deref_mut())?;
                match target {
                    Target::Name(n) => {
                        let cur = match self.lookup(n, locals.as_deref()) {
                            Some(Binding::Val(v)) => v.clone(),
                            Some(Binding::Func(_)) => {
                                return Err(violation(format!(
                                    "unsupported operand type(s) for augmented assignment to function '{}'",
                                    n
                                )))
                            }
                            None => {
                                if _ALLOWED_CALL_NAMES.contains(&n.as_str()) {
                                    return Err(violation(format!(
                                        "name '{}' is not defined",
                                        n
                                    )));
                                }
                                return Err(violation(format!("name '{}' is not defined", n)));
                            }
                        };
                        let new = binop(*op, cur, val)?;
                        self.store(n, Binding::Val(new), locals);
                    }
                    Target::Sub(base, idx) => {
                        // read current element, apply op, write back
                        let cur = self.eval(
                            &Expr::Sub(base.clone(), Box::new(match idx.clone() {
                                SIdx::Idx(e) => SIdx::Idx(e),
                                _ => return Err(violation("illegal slice target for augmented assignment")),
                            })),
                            locals.as_deref_mut(),
                        )?;
                        let new = binop(*op, cur, val)?;
                        self.assign_target(&Target::Sub(base.clone(), idx.clone()), new, locals.as_deref_mut())?;
                    }
                    _ => return Err(violation("illegal target for augmented assignment")),
                }
                Ok(Flow::Normal)
            }
            Stmt::If { arms, else_body } => {
                for (cond, body) in arms {
                    let c = self.eval(cond, locals.as_deref_mut())?;
                    if c.truthy() {
                        return self.exec_block(body, locals);
                    }
                }
                if let Some(b) = else_body {
                    return self.exec_block(b, locals);
                }
                Ok(Flow::Normal)
            }
            Stmt::While { cond, body } => {
                loop {
                    self.tick()?;
                    let c = self.eval(cond, locals.as_deref_mut())?;
                    if !c.truthy() {
                        break;
                    }
                    match self.exec_block(body, locals.as_deref_mut())? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::For { target, iter, body } => {
                let iter_v = self.eval(iter, locals.as_deref_mut())?;
                let items = iter_values(iter_v)?;
                for item in items {
                    self.tick()?;
                    self.assign_target(target, item, locals.as_deref_mut())?;
                    match self.exec_block(body, locals.as_deref_mut())? {
                        Flow::Break => break,
                        Flow::Return(v) => return Ok(Flow::Return(v)),
                        _ => {}
                    }
                }
                Ok(Flow::Normal)
            }
            Stmt::Break => Ok(Flow::Break),
            Stmt::Continue => Ok(Flow::Continue),
            Stmt::ExprStmt(e) => {
                self.eval(e, locals.as_deref_mut())?;
                Ok(Flow::Normal)
            }
            Stmt::Seq(stmts) => {
                for s in stmts {
                    match self.exec_stmt(s, locals.as_deref_mut())? {
                        Flow::Normal => {}
                        f => return Ok(f),
                    }
                }
                Ok(Flow::Normal)
            }
        }
    }

    fn assign_target(
        &mut self,
        t: &Target,
        value: SandboxValue,
        mut locals: Option<&mut Locals>,
    ) -> Result<(), SandboxError> {
        match t {
            Target::Name(n) => {
                self.store(n, Binding::Val(value), locals);
                Ok(())
            }
            Target::Tuple(ts) | Target::List(ts) => {
                let items = match value {
                    SandboxValue::List(xs) => xs,
                    SandboxValue::Tuple(xs) => xs,
                    other => {
                        return Err(violation(format!(
                            "cannot unpack non-iterable {} object",
                            other.type_name()
                        )))
                    }
                };
                if items.len() > ts.len() {
                    return Err(violation(format!(
                        "too many values to unpack (expected {})",
                        ts.len()
                    )));
                }
                if items.len() < ts.len() {
                    return Err(violation(format!(
                        "not enough values to unpack (expected {}, got {})",
                        ts.len(),
                        items.len()
                    )));
                }
                for (t2, v) in ts.iter().zip(items) {
                    self.assign_target(t2, v, locals.as_deref_mut())?;
                }
                Ok(())
            }
            Target::Sub(base, idx) => {
                let name = match base.as_ref() {
                    Expr::Name(n) => n.clone(),
                    _ => {
                        return Err(violation(
                            "only simple subscript targets (name[index]) are supported",
                        ))
                    }
                };
                match idx {
                    SIdx::Idx(ie) => {
                        let iv = self.eval(ie, locals.as_deref_mut())?;
                        self.assign_sub_value(&name, &iv, value, locals)
                    }
                    SIdx::Slice(..) => {
                        Err(syntax_err("slice assignment is not supported"))
                    }
                }
            }
        }
    }

    fn assign_sub_value(
        &mut self,
        name: &str,
        idx: &SandboxValue,
        value: SandboxValue,
        locals: Option<&mut Locals>,
    ) -> Result<(), SandboxError> {
        let i = match as_int(idx) {
            Some(i) => i,
            None => {
                return Err(violation(format!(
                    "list indices must be integers or slices, not {}",
                    idx.type_name()
                )))
            }
        };
        let holds = match locals.as_deref() {
            Some(l) => l.contains_key(name),
            None => false,
        };
        if holds {
            if let Some(l) = locals {
                let b = l.get_mut(name).unwrap();
                return set_sub_index(b, i, value);
            }
            unreachable!();
        }
        match self.globals.get_mut(name) {
            Some(b) => set_sub_index(b, i, value),
            None => Err(violation(format!("name '{}' is not defined", name))),
        }
    }

    fn eval(
        &mut self,
        e: &Expr,
        mut locals: Option<&mut Locals>,
    ) -> Result<SandboxValue, SandboxError> {
        match e {
            Expr::Int(v) => Ok(SandboxValue::Int(*v)),
            Expr::Float(v) => Ok(SandboxValue::Float(*v)),
            Expr::Str(s) => Ok(SandboxValue::Str(s.clone())),
            Expr::BoolLit(b) => Ok(SandboxValue::Bool(*b)),
            Expr::NoneLit => Ok(SandboxValue::None),
            Expr::Name(n) => match self.lookup(n, locals.as_deref()) {
                Some(Binding::Val(v)) => Ok(v.clone()),
                Some(Binding::Func(_)) => Err(violation(format!(
                    "function '{}' cannot be used as a value in the sandbox",
                    n
                ))),
                None => {
                    if _ALLOWED_CALL_NAMES.contains(&n.as_str()) {
                        Err(violation(format!(
                            "builtin function '{}' cannot be used as a value in the sandbox",
                            n
                        )))
                    } else {
                        Err(violation(format!("name '{}' is not defined", n)))
                    }
                }
            },
            Expr::ListLit(es) => {
                let mut out = Vec::with_capacity(es.len());
                for x in es {
                    out.push(self.eval(x, locals.as_deref_mut())?);
                }
                Ok(SandboxValue::List(out))
            }
            Expr::TupleLit(es) => {
                let mut out = Vec::with_capacity(es.len());
                for x in es {
                    out.push(self.eval(x, locals.as_deref_mut())?);
                }
                Ok(SandboxValue::Tuple(out))
            }
            Expr::Bin(op, a, b) => {
                let av = self.eval(a, locals.as_deref_mut())?;
                let bv = self.eval(b, locals.as_deref_mut())?;
                binop(*op, av, bv)
            }
            Expr::Unary(op, a) => {
                let av = self.eval(a, locals.as_deref_mut())?;
                match op {
                    UnOp::Not => Ok(SandboxValue::Bool(!av.truthy())),
                    UnOp::Neg => match av {
                        SandboxValue::Int(i) => {
                            if i == i64::MIN {
                                Ok(SandboxValue::Float(-(i64::MIN as f64)))
                            } else {
                                Ok(SandboxValue::Int(-i))
                            }
                        }
                        SandboxValue::Float(f) => Ok(SandboxValue::Float(-f)),
                        SandboxValue::Bool(b) => Ok(SandboxValue::Int(-i64::from(b))),
                        v => Err(violation(format!(
                            "bad operand type for unary -: '{}'",
                            v.type_name()
                        ))),
                    },
                    UnOp::Pos => match av {
                        SandboxValue::Int(i) => Ok(SandboxValue::Int(i)),
                        SandboxValue::Float(f) => Ok(SandboxValue::Float(f)),
                        SandboxValue::Bool(b) => Ok(SandboxValue::Int(i64::from(b))),
                        v => Err(violation(format!(
                            "bad operand type for unary +: '{}'",
                            v.type_name()
                        ))),
                    },
                }
            }
            Expr::BoolAnd(ops) => {
                let mut last = SandboxValue::None;
                for o in ops {
                    let v = self.eval(o, locals.as_deref_mut())?;
                    if !v.truthy() {
                        return Ok(v);
                    }
                    last = v;
                }
                Ok(last)
            }
            Expr::BoolOr(ops) => {
                let mut last = SandboxValue::None;
                for o in ops {
                    let v = self.eval(o, locals.as_deref_mut())?;
                    if v.truthy() {
                        return Ok(v);
                    }
                    last = v;
                }
                Ok(last)
            }
            Expr::Compare(lhs, ops) => {
                let mut cur = self.eval(lhs, locals.as_deref_mut())?;
                let mut result = true;
                for (op, rhs_e) in ops {
                    let rhs = self.eval(rhs_e, locals.as_deref_mut())?;
                    if !compare_op(*op, &cur, &rhs)? {
                        result = false;
                        break;
                    }
                    cur = rhs;
                }
                Ok(SandboxValue::Bool(result))
            }
            Expr::Call(name, args) => {
                let mut vals = Vec::with_capacity(args.len());
                for a in args {
                    vals.push(self.eval(a, locals.as_deref_mut())?);
                }
                match self.lookup(name, locals.as_deref()).cloned() {
                    Some(Binding::Func(f)) => {
                        Ok(self.call_function(&f, &vals)?.unwrap_or(SandboxValue::None))
                    }
                    Some(Binding::Val(v)) => Err(violation(format!(
                        "'{}' object is not callable",
                        v.type_name()
                    ))),
                    None => {
                        if _ALLOWED_CALL_NAMES.contains(&name.as_str()) {
                            call_builtin(name, &vals)
                        } else {
                            Err(violation(format!("name '{}' is not defined", name)))
                        }
                    }
                }
            }
            Expr::Sub(base, idx) => {
                let cont = self.eval(base, locals.as_deref_mut())?;
                match idx.as_ref() {
                    SIdx::Idx(ie) => {
                        let iv = self.eval(ie, locals.as_deref_mut())?;
                        index_value(&cont, &iv)
                    }
                    SIdx::Slice(lo, hi, st) => {
                        let lo_v = match lo {
                            Some(e) => Some(self.eval(e, locals.as_deref_mut())?),
                            None => None,
                        };
                        let hi_v = match hi {
                            Some(e) => Some(self.eval(e, locals.as_deref_mut())?),
                            None => None,
                        };
                        let st_v = match st {
                            Some(e) => Some(self.eval(e, locals.as_deref_mut())?),
                            None => None,
                        };
                        slice_value(&cont, lo_v.as_ref(), hi_v.as_ref(), st_v.as_ref())
                    }
                }
            }
        }
    }

    fn call_function(
        &mut self,
        f: &RuntimeFunc,
        args: &[SandboxValue],
    ) -> Result<Option<SandboxValue>, SandboxError> {
        if self.depth >= MAX_CALL_DEPTH {
            return Err(violation("maximum recursion depth exceeded"));
        }
        self.depth += 1;
        let result = self.call_function_inner(f, args);
        self.depth -= 1;
        result
    }

    fn call_function_inner(
        &mut self,
        f: &RuntimeFunc,
        args: &[SandboxValue],
    ) -> Result<Option<SandboxValue>, SandboxError> {
        let params = &f.def.params;
        if args.len() > params.len() {
            return Err(violation(format!(
                "{}() takes {} positional argument{} but {} were given",
                f.def.name,
                params.len(),
                if params.len() == 1 { "" } else { "s" },
                args.len()
            )));
        }
        let mut locals: Locals = HashMap::new();
        for (i, p) in params.iter().enumerate() {
            let val = if i < args.len() {
                args[i].clone()
            } else {
                match &f.defaults[i] {
                    Some(v) => v.clone(),
                    None => {
                        return Err(violation(format!(
                            "{}() missing 1 required positional argument: '{}'",
                            f.def.name, p.name
                        )))
                    }
                }
            };
            locals.insert(p.name.clone(), Binding::Val(val));
        }
        let flow = self.exec_block(&f.def.body, Some(&mut locals))?;
        Ok(match flow {
            Flow::Return(v) => v,
            _ => None,
        })
    }
}

fn set_sub_index(b: &mut Binding, i: i64, value: SandboxValue) -> Result<(), SandboxError> {
    match b {
        Binding::Val(SandboxValue::List(xs)) => {
            let n = xs.len() as i64;
            let u = if i < 0 { i + n } else { i };
            if u < 0 || u >= n {
                return Err(violation("list assignment index out of range"));
            }
            xs[u as usize] = value;
            Ok(())
        }
        Binding::Val(SandboxValue::Tuple(_)) => {
            Err(violation("'tuple' object does not support item assignment"))
        }
        Binding::Val(SandboxValue::Str(_)) => {
            Err(violation("'str' object does not support item assignment"))
        }
        Binding::Val(v) => Err(violation(format!(
            "'{}' object does not support item assignment",
            v.type_name()
        ))),
        Binding::Func(_) => Err(violation("'function' object does not support item assignment")),
    }
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Parses + validates `code` (expected to define one or more functions),
/// then evaluates `call_expr` (e.g. "my_func(5)") in a namespace with NO
/// builtins except a small safe allowlist, under a hard step budget.
/// Mirrors Python `run_sandboxed` (default max_steps there: 100_000).
pub fn run_sandboxed(
    code: &str,
    call_expr: &str,
    max_steps: usize,
) -> Result<SandboxValue, SandboxError> {
    let stmts = parse_module(code)?;
    let mut interp = Interp {
        globals: HashMap::new(),
        steps: 0,
        max_steps,
        depth: 0,
    };
    interp.exec_block(&stmts, None)?;
    let expr = parse_expression(call_expr)?;
    interp.eval(&expr, None)
}

