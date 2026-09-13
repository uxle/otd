//! Rust port of python/tests/test_tensor.py.

use reasoning_tensor::{NestedVal, Number, ShapeError, Tensor, TensorError};

/// Build a `NestedVal` from integer literals (test convenience).
fn nested(data: impl Into<NestedData>) -> NestedVal {
    data.into().0
}

struct NestedData(NestedVal);

impl From<Vec<i64>> for NestedData {
    fn from(v: Vec<i64>) -> NestedData {
        NestedData(NestedVal::List(v.into_iter().map(NestedVal::Int).collect()))
    }
}

impl From<Vec<Vec<i64>>> for NestedData {
    fn from(v: Vec<Vec<i64>>) -> NestedData {
        NestedData(NestedVal::List(
            v.into_iter().map(|r| NestedData::from(r).0).collect(),
        ))
    }
}

impl From<Vec<Vec<Vec<i64>>>> for NestedData {
    fn from(v: Vec<Vec<Vec<i64>>>) -> NestedData {
        NestedData(NestedVal::List(
            v.into_iter().map(|r| NestedData::from(r).0).collect(),
        ))
    }
}

// ── TestTensorCreation ──────────────────────────────────────────────────
#[test]
fn test_rank0_scalar() {
    let t = Tensor::new(NestedVal::Int(5)).unwrap();
    assert_eq!(t.shape, Vec::<usize>::new());
    assert_eq!(t.ndim(), 0);
    assert_eq!(t.size(), 1);
    assert_eq!(t.data, vec![Number::Int(5)]);
}

#[test]
fn test_rank1_vector() {
    let t = Tensor::new(nested(vec![1, 2, 3, 4])).unwrap();
    assert_eq!(t.shape, vec![4]);
    assert_eq!(t.ndim(), 1);
    assert_eq!(t.size(), 4);
    assert_eq!(t.get(&[0]).unwrap(), Number::Int(1));
    assert_eq!(t.get(&[3]).unwrap(), Number::Int(4));
}

#[test]
fn test_rank2_matrix() {
    let t = Tensor::new(nested(vec![vec![1, 2, 3], vec![4, 5, 6]])).unwrap();
    assert_eq!(t.shape, vec![2, 3]);
    assert_eq!(t.ndim(), 2);
    assert_eq!(t.size(), 6);
    assert_eq!(t.get(&[0, 0]).unwrap(), Number::Int(1));
    assert_eq!(t.get(&[0, 2]).unwrap(), Number::Int(3));
    assert_eq!(t.get(&[1, 0]).unwrap(), Number::Int(4));
    assert_eq!(t.get(&[1, 2]).unwrap(), Number::Int(6));
}

#[test]
fn test_rank3_tensor() {
    let t = Tensor::new(nested(vec![
        vec![vec![1, 2], vec![3, 4]],
        vec![vec![5, 6], vec![7, 8]],
    ]))
    .unwrap();
    assert_eq!(t.shape, vec![2, 2, 2]);
    assert_eq!(t.ndim(), 3);
    assert_eq!(t.size(), 8);
    assert_eq!(t.get(&[0, 0, 0]).unwrap(), Number::Int(1));
    assert_eq!(t.get(&[0, 1, 1]).unwrap(), Number::Int(4));
    assert_eq!(t.get(&[1, 0, 0]).unwrap(), Number::Int(5));
    assert_eq!(t.get(&[1, 1, 1]).unwrap(), Number::Int(8));
}

#[test]
fn test_strides_row_major() {
    let t = Tensor::new(nested(vec![vec![1, 2, 3], vec![4, 5, 6]])).unwrap();
    // row-major: strides should be (3, 1) for shape (2, 3)
    assert_eq!(t.strides, vec![3, 1]);
}

#[test]
fn test_setitem() {
    let mut t = Tensor::new(nested(vec![vec![0, 0], vec![0, 0]])).unwrap();
    t.set(&[1, 1], Number::Int(99)).unwrap();
    assert_eq!(t.get(&[1, 1]).unwrap(), Number::Int(99));
    assert_eq!(t.get(&[0, 0]).unwrap(), Number::Int(0));
}

#[test]
fn test_zeros_ones_full() {
    let z = Tensor::zeros(&[2, 3]);
    assert_eq!(z.shape, vec![2, 3]);
    assert!(z.data.iter().all(|v| *v == Number::Int(0)));

    let o = Tensor::ones(&[3]);
    assert_eq!(o.data, vec![Number::Int(1), Number::Int(1), Number::Int(1)]);

    let f = Tensor::full(&[2, 2], Number::Int(7));
    assert_eq!(
        f.data,
        vec![Number::Int(7), Number::Int(7), Number::Int(7), Number::Int(7)]
    );
}

#[test]
fn test_equality() {
    let a = Tensor::new(nested(vec![vec![1, 2], vec![3, 4]])).unwrap();
    let b = Tensor::new(nested(vec![vec![1, 2], vec![3, 4]])).unwrap();
    let c = Tensor::new(nested(vec![vec![1, 2], vec![3, 5]])).unwrap();
    assert_eq!(a, b);
    assert_ne!(a, c);
}

// ---- error paths: must actually raise, not silently "pretend to work" ----

#[test]
fn test_ragged_input_raises() {
    // inconsistent row lengths
    let err = Tensor::new(nested(vec![vec![1, 2, 3], vec![4, 5]])).unwrap_err();
    assert_eq!(
        err,
        ShapeError(
            "Ragged nested list: element 0 has shape (3,) but element 1 has shape (2,)"
                .to_string()
        )
    );
}

#[test]
fn test_index_rank_mismatch_raises() {
    let t = Tensor::new(nested(vec![vec![1, 2], vec![3, 4]])).unwrap();
    // rank-1 index into a rank-2 tensor
    match t.get(&[0]).unwrap_err() {
        TensorError::Shape(e) => {
            assert_eq!(e.0, "Index rank 1 does not match tensor rank 2".to_string())
        }
        other => panic!("expected ShapeError, got {:?}", other),
    }
}

#[test]
fn test_index_out_of_bounds_raises() {
    let t = Tensor::new(nested(vec![1, 2, 3])).unwrap();
    match t.get(&[3]).unwrap_err() {
        TensorError::Index(msg) => assert_eq!(msg, "Index 3 out of bounds for axis 0 with size 3"),
        other => panic!("expected IndexError, got {:?}", other),
    }
    // negative indices not supported in phase 1
    match t.get(&[-1i64]).unwrap_err() {
        TensorError::Index(msg) => assert_eq!(msg, "Index -1 out of bounds for axis 0 with size 3"),
        other => panic!("expected IndexError, got {:?}", other),
    }
}

// ---- extras exercising the ported repr (Python `__repr__`) ----

#[test]
fn test_display_matches_python_repr() {
    let t = Tensor::new(nested(vec![vec![1, 2, 3], vec![4, 5, 6]])).unwrap();
    assert_eq!(
        t.to_string(),
        "Tensor(shape=(2, 3), data=[1, 2, 3, 4, 5, 6])".to_string()
    );
    let s = Tensor::new(NestedVal::Num(5.0)).unwrap();
    assert_eq!(s.to_string(), "Tensor(shape=(), data=[5.0])".to_string());
}

#[test]
fn test_int_float_equality_is_python_semantics() {
    // Python: Tensor([1, 2]) == Tensor([1.0, 2.0])
    let a = Tensor::new(nested(vec![1, 2])).unwrap();
    let b = Tensor::new(NestedVal::List(vec![
        NestedVal::Num(1.0),
        NestedVal::Num(2.0),
    ]))
    .unwrap();
    assert_eq!(a, b);
}
