//! Phase 001 — Tensor Creation (Rust port of python/tensor/tensor.py)
//!
//! A minimal N-D tensor abstraction built from scratch (no numpy):
//! shape metadata, flat contiguous data buffer + row-major strides,
//! construction from nested lists, element access via a full index tuple,
//! shape-mismatch detection (ragged input rejected), and equality.

pub enum NestedVal {
    /// Python float.
    Num(f64),
    /// Python int.
    Int(i64),
    List(Vec<NestedVal>),
}

/// Python `Number = Union[int, float]` — a flat tensor element. Equality is
/// Python semantics (`5 == 5.0`).
#[derive(Debug, Clone, Copy)]
pub enum Number {
    Int(i64),
    Float(f64),
}

impl Number {
    pub fn as_f64(&self) -> f64 {
        match *self {
            Number::Int(i) => i as f64,
            Number::Float(f) => f,
        }
    }
}

impl PartialEq for Number {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Number::Int(a), Number::Int(b)) => a == b,
            (Number::Float(a), Number::Float(b)) => a == b,
            // Python: 5 == 5.0
            (Number::Int(a), Number::Float(b)) => (*a as f64) == *b,
            (Number::Float(a), Number::Int(b)) => *a == (*b as f64),
        }
    }
}

/// Raised when a tensor is constructed from ragged/inconsistent nested data,
/// or when an index doesn't match the tensor's rank (Python `ShapeError`,
/// a `ValueError` subclass).
#[derive(Debug, Clone, PartialEq)]
pub struct ShapeError(pub String);

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ShapeError {}

/// Errors from element access: rank mismatch is Python's `ShapeError`,
/// out-of-bounds is Python's `IndexError`.
#[derive(Debug, Clone, PartialEq)]
pub enum TensorError {
    Shape(ShapeError),
    Index(String),
}

impl std::fmt::Display for TensorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TensorError::Shape(e) => write!(f, "{}", e.0),
            TensorError::Index(msg) => write!(f, "{}", msg),
        }
    }
}

impl std::error::Error for TensorError {}

/// Index elements accepted by `Tensor::get`/`set` — signed so the Python
/// negative-index `IndexError` path is expressible, with `usize` also
/// accepted (all Python indices are checked against `0 <= idx < dim`).
pub trait TensorIndex: Copy {
    fn to_index(self) -> i64;
}

impl TensorIndex for i64 {
    fn to_index(self) -> i64 {
        self
    }
}

impl TensorIndex for i32 {
    fn to_index(self) -> i64 {
        self as i64
    }
}

impl TensorIndex for usize {
    fn to_index(self) -> i64 {
        self as i64
    }
}

/// Python tuple repr of a shape: `()`, `(4,)`, `(2, 3)`.
fn fmt_shape(shape: &[usize]) -> String {
    let items: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
    if shape.len() == 1 {
        format!("({},)", items[0])
    } else {
        format!("({})", items.join(", "))
    }
}

/// Python `str(float)` (shortest repr; plain notation for -4 <= exp < 16,
/// otherwise scientific). Inlined from the workspace's Python-float rules
/// because this crate intentionally has no dependencies.
fn py_float_repr(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf".to_string() } else { "-inf".to_string() };
    }
    if x == 0.0 {
        return if x.is_sign_negative() { "-0.0".to_string() } else { "0.0".to_string() };
    }
    let neg = x < 0.0;
    let ax = x.abs();
    let e_str = format!("{:e}", ax);
    let (mant, exp_str) = e_str.split_once('e').expect("{:e} always has 'e'");
    let exp: i32 = exp_str.parse().unwrap();
    let mut digits: String = mant.chars().filter(|c| *c != '.').collect();
    while digits.len() > 1 && digits.ends_with('0') {
        digits.pop();
    }
    let n = digits.len() as i32;

    let mut out = String::new();
    if neg {
        out.push('-');
    }
    if exp >= -4 && exp < 16 {
        if exp >= 0 {
            if exp >= n - 1 {
                out.push_str(&digits);
                for _ in 0..(exp - (n - 1)) {
                    out.push('0');
                }
                out.push_str(".0");
            } else {
                out.push_str(&digits[..(exp + 1) as usize]);
                out.push('.');
                out.push_str(&digits[(exp + 1) as usize..]);
            }
        } else {
            out.push_str("0.");
            for _ in 0..(-exp - 1) {
                out.push('0');
            }
            out.push_str(&digits);
        }
    } else {
        out.push_str(&digits[..1]);
        if digits.len() > 1 {
            out.push('.');
            out.push_str(&digits[1..]);
        }
        out.push('e');
        out.push(if exp < 0 { '-' } else { '+' });
        let a = exp.abs();
        if a < 10 {
            out.push('0');
        }
        out.push_str(&a.to_string());
    }
    out
}

fn fmt_data(data: &[Number]) -> String {
    let items: Vec<String> = data
        .iter()
        .map(|v| match v {
            Number::Int(i) => i.to_string(),
            Number::Float(f) => py_float_repr(*f),
        })
        .collect();
    format!("[{}]", items.join(", "))
}

/// Infer the shape of a (possibly ragged) nested list, erroring on ragged
/// input with Python's exact message.
fn infer_shape(data: &NestedVal) -> Result<Vec<usize>, ShapeError> {
    match data {
        NestedVal::Num(_) | NestedVal::Int(_) => Ok(Vec::new()), // scalar leaf
        NestedVal::List(items) => {
            if items.is_empty() {
                return Ok(vec![0]);
            }
            let child_shapes: Vec<Vec<usize>> = items
                .iter()
                .map(infer_shape)
                .collect::<Result<_, _>>()?;
            let first = &child_shapes[0];
            for (i, s) in child_shapes.iter().enumerate() {
                if s != first {
                    return Err(ShapeError(format!(
                        "Ragged nested list: element 0 has shape {} but element {} has shape {}",
                        fmt_shape(first),
                        i,
                        fmt_shape(s)
                    )));
                }
            }
            let mut shape = Vec::with_capacity(1 + first.len());
            shape.push(items.len());
            shape.extend_from_slice(first);
            Ok(shape)
        }
    }
}

fn flatten(data: &NestedVal, out: &mut Vec<Number>) {
    match data {
        NestedVal::Num(f) => out.push(Number::Float(*f)),
        NestedVal::Int(i) => out.push(Number::Int(*i)),
        NestedVal::List(items) => {
            for x in items {
                flatten(x, out);
            }
        }
    }
}

fn row_major_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1usize; shape.len()];
    let mut acc = 1usize;
    for i in (0..shape.len()).rev() {
        strides[i] = acc;
        acc *= shape[i];
    }
    strides
}

#[derive(Debug, Clone)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub strides: Vec<usize>,
    pub data: Vec<Number>,
}

impl Tensor {
    /// Python `Tensor(data: NestedList)` — rejects ragged input.
    pub fn new(data: NestedVal) -> Result<Tensor, ShapeError> {
        let shape = infer_shape(&data)?;
        let mut flat: Vec<Number> = Vec::new();
        flatten(&data, &mut flat);
        let expected_size = shape.iter().product::<usize>();
        if flat.len() != expected_size {
            // Unreachable if infer_shape didn't already error — hard
            // invariant check kept from the Python original.
            return Err(ShapeError(format!(
                "Flattened data length {} does not match shape {}",
                flat.len(),
                fmt_shape(&shape)
            )));
        }
        Ok(Tensor {
            strides: row_major_strides(&shape),
            shape,
            data: flat,
        })
    }

    // ---- constructors ----

    pub fn zeros(shape: &[usize]) -> Tensor {
        Tensor::filled(shape, Number::Int(0))
    }

    pub fn ones(shape: &[usize]) -> Tensor {
        Tensor::filled(shape, Number::Int(1))
    }

    pub fn full(shape: &[usize], value: Number) -> Tensor {
        Tensor::filled(shape, value)
    }

    fn filled(shape: &[usize], value: Number) -> Tensor {
        let size = shape.iter().product::<usize>();
        Tensor {
            shape: shape.to_vec(),
            strides: row_major_strides(shape),
            data: vec![value; size],
        }
    }

    // ---- metadata ----

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    pub fn size(&self) -> usize {
        self.shape.iter().product::<usize>()
    }

    // ---- access ----

    fn flat_index<I: TensorIndex>(&self, index: &[I]) -> Result<usize, TensorError> {
        if index.len() != self.ndim() {
            return Err(TensorError::Shape(ShapeError(format!(
                "Index rank {} does not match tensor rank {}",
                index.len(),
                self.ndim()
            ))));
        }
        let mut flat = 0usize;
        for (i, (&idx, &dim)) in index.iter().zip(self.shape.iter()).enumerate() {
            let idx = idx.to_index();
            if !(0 <= idx && (idx as usize) < dim) {
                return Err(TensorError::Index(format!(
                    "Index {} out of bounds for axis {} with size {}",
                    idx, i, dim
                )));
            }
            flat += idx as usize * self.strides[i];
        }
        Ok(flat)
    }

    /// Python `__getitem__(index)` (single int or tuple).
    pub fn get<I: TensorIndex>(&self, index: &[I]) -> Result<Number, TensorError> {
        let flat = self.flat_index(index)?;
        Ok(self.data[flat])
    }

    /// Convenience: `get` as f64.
    pub fn get_f64<I: TensorIndex>(&self, index: &[I]) -> Result<f64, TensorError> {
        Ok(self.get(index)?.as_f64())
    }

    /// Python `__setitem__(index, value)`.
    pub fn set<I: TensorIndex>(&mut self, index: &[I], value: Number) -> Result<(), TensorError> {
        let flat = self.flat_index(index)?;
        self.data[flat] = value;
        Ok(())
    }
}

impl PartialEq for Tensor {
    /// Python `__eq__`: shape + data equality (strides excluded).
    fn eq(&self, other: &Self) -> bool {
        self.shape == other.shape && self.data == other.data
    }
}

impl std::fmt::Display for Tensor {
    /// Python `__repr__`: `Tensor(shape=(2, 3), data=[1, 2, 3, 4, 5, 6])`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Tensor(shape={}, data={})",
            fmt_shape(&self.shape),
            fmt_data(&self.data)
        )
    }
}

impl From<i64> for Number {
    fn from(i: i64) -> Number {
        Number::Int(i)
    }
}

impl From<f64> for Number {
    fn from(f: f64) -> Number {
        Number::Float(f)
    }
}
