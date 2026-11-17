// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! Python semantics on decoded values: what `int()`, `float()`, `len()`,
//! indexing, iteration and truthiness do to a value `VCF.parse_field`
//! returned, so the ported functions treat an undeclared (string) or
//! unusually declared field the way the Python worker does.
//!
//! A Python value is `Option<&Value>`: `None` (and `Value::Null`, the
//! decoded `.`) is Python's `None`.

use std::fmt;

use crate::value::{DecodeError, Value, py_float, py_int};

/// A Python exception: `ValueError` / `TypeError` (which the Python code
/// catches where it says so) or anything else (`AttributeError`,
/// `OverflowError`, ...: uncaught, the shard fails).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyErr {
    Value(String),
    Type(String),
    Other(String),
}

impl fmt::Display for PyErr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PyErr::Value(m) => write!(f, "ValueError: {m}"),
            PyErr::Type(m) => write!(f, "TypeError: {m}"),
            PyErr::Other(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for PyErr {}

impl PyErr {
    /// Caught by `except (ValueError, TypeError)`.
    pub fn is_value_or_type(&self) -> bool {
        matches!(self, PyErr::Value(_) | PyErr::Type(_))
    }
}

fn from_decode(e: DecodeError) -> PyErr {
    match e {
        DecodeError::Malformed(m) => PyErr::Value(m),
        DecodeError::Fatal(m) => PyErr::Other(m),
    }
}

/// `Value::Null` and an absent key are both Python's `None`.
pub fn pv(o: &Option<Value>) -> Option<&Value> {
    match o {
        None | Some(Value::Null) => None,
        Some(v) => Some(v),
    }
}

fn kind(v: Option<&Value>) -> &'static str {
    match v {
        None | Some(Value::Null) => "NoneType",
        Some(Value::Bool(_)) => "bool",
        Some(Value::Int(_)) => "int",
        Some(Value::Float(_)) => "float",
        Some(Value::Str(_)) => "str",
        Some(_) => "list",
    }
}

/// `bool(v)`.
pub fn truthy(v: Option<&Value>) -> bool {
    match v {
        None | Some(Value::Null) => false,
        Some(Value::Bool(b)) => *b,
        Some(Value::Int(i)) => *i != 0,
        Some(Value::Float(x)) => *x != 0.0,
        Some(Value::Str(s)) => !s.is_empty(),
        Some(Value::Bools(x)) => !x.is_empty(),
        Some(Value::Ints(x)) => !x.is_empty(),
        Some(Value::Floats(x)) => !x.is_empty(),
        Some(Value::Strs(x)) => !x.is_empty(),
    }
}

/// `int(x)` of a float: truncation; NaN is a `ValueError`, an infinity an
/// `OverflowError`.
pub fn int_of_float(x: f64) -> Result<i64, PyErr> {
    if x.is_nan() {
        return Err(PyErr::Value("cannot convert float NaN to integer".into()));
    }
    if x.is_infinite() {
        return Err(PyErr::Other(
            "OverflowError: cannot convert float infinity to integer".into(),
        ));
    }
    let t = x.trunc();
    // [-2**63, 2**63): the i64 range.
    if t < -9_223_372_036_854_775_808.0 || t >= 9_223_372_036_854_775_808.0 {
        return Err(PyErr::Other(format!("an integer beyond 64 bits: int({x})")));
    }
    Ok(t as i64)
}

/// `int(v)`.
pub fn to_int(v: Option<&Value>) -> Result<i64, PyErr> {
    match v {
        Some(Value::Int(i)) => Ok(*i),
        Some(Value::Bool(b)) => Ok(i64::from(*b)),
        Some(Value::Float(x)) => int_of_float(*x),
        Some(Value::Str(s)) => py_int(s.as_bytes()).map_err(from_decode),
        other => Err(PyErr::Type(format!(
            "int() argument must be a string, a bytes-like object or a real \
             number, not '{}'",
            kind(other)
        ))),
    }
}

/// `float(v)`.
pub fn to_float(v: Option<&Value>) -> Result<f64, PyErr> {
    match v {
        Some(Value::Float(x)) => Ok(*x),
        // float(int) is correctly rounded, as `as f64` is.
        Some(Value::Int(i)) => Ok(*i as f64),
        Some(Value::Bool(b)) => Ok(if *b { 1.0 } else { 0.0 }),
        Some(Value::Str(s)) => py_float(s.as_bytes()).map_err(from_decode),
        other => Err(PyErr::Type(format!(
            "float() argument must be a string or a real number, not '{}'",
            kind(other)
        ))),
    }
}

/// `len(v)`.
pub fn py_len(v: Option<&Value>) -> Result<usize, PyErr> {
    match v {
        Some(Value::Str(s)) => Ok(s.chars().count()),
        Some(Value::Bools(x)) => Ok(x.len()),
        Some(Value::Ints(x)) => Ok(x.len()),
        Some(Value::Floats(x)) => Ok(x.len()),
        Some(Value::Strs(x)) => Ok(x.len()),
        other => Err(PyErr::Type(format!(
            "object of type '{}' has no len()",
            kind(other)
        ))),
    }
}

/// The items of an iterable (`for x in v`): a string's characters, a
/// list's elements; anything else is a `TypeError`.
pub fn items(v: &Value) -> Result<Vec<Value>, PyErr> {
    Ok(match v {
        Value::Str(s) => s.chars().map(|c| Value::Str(c.to_string())).collect(),
        Value::Bools(x) => x.iter().map(|&b| Value::Bool(b)).collect(),
        Value::Ints(x) => x.iter().map(|&i| Value::Int(i)).collect(),
        Value::Floats(x) => x.iter().map(|&f| Value::Float(f)).collect(),
        Value::Strs(x) => x.iter().map(|s| Value::Str(s.clone())).collect(),
        other => {
            return Err(PyErr::Type(format!(
                "'{}' object is not iterable",
                kind(Some(other))
            )));
        }
    })
}

/// `v[i]` for `i < len(v)` (the callers check the length first).
pub fn item(v: &Value, i: usize) -> Result<Value, PyErr> {
    Ok(match v {
        Value::Str(s) => Value::Str(
            s.chars()
                .nth(i)
                .ok_or_else(|| PyErr::Other("IndexError: string index out of range".into()))?
                .to_string(),
        ),
        Value::Bools(x) => Value::Bool(x[i]),
        Value::Ints(x) => Value::Int(x[i]),
        Value::Floats(x) => Value::Float(x[i]),
        Value::Strs(x) => Value::Str(x[i].clone()),
        other => {
            return Err(PyErr::Type(format!(
                "'{}' object is not subscriptable",
                kind(Some(other))
            )));
        }
    })
}

/// `int(v[i])` with the list fast paths.
pub fn int_item(v: &Value, i: usize) -> Result<i64, PyErr> {
    match v {
        Value::Ints(x) => Ok(x[i]),
        _ => to_int(Some(&item(v, i)?)),
    }
}

/// `float(v[i])` with the list fast paths.
pub fn float_item(v: &Value, i: usize) -> Result<f64, PyErr> {
    match v {
        Value::Floats(x) => Ok(x[i]),
        Value::Ints(x) => Ok(x[i] as f64),
        _ => to_float(Some(&item(v, i)?)),
    }
}

/// `sum(xs)` of a list of floats as CPython (3.12 and later) computes it:
/// the int start 0 plus the first item, then Neumaier's compensated
/// summation over the rest, the compensation added at the end when it is
/// non-zero and finite (`builtin_sum_impl`'s float path).
pub fn py_fsum_builtin(xs: &[f64]) -> f64 {
    let Some((&first, rest)) = xs.split_first() else {
        return 0.0;
    };
    // int 0 + float: float(0) + x.
    let mut f = 0.0 + first;
    let mut c = 0.0f64;
    for &x in rest {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// Python's `str.upper()` of one character (ASCII is all the QC asks).
pub fn upper(c: char) -> String {
    c.to_uppercase().collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn conversions() {
        assert_eq!(to_int(Some(&Value::Float(-2.7))), Ok(-2));
        assert_eq!(to_int(Some(&Value::Str(" 12 ".into()))), Ok(12));
        assert!(matches!(
            to_int(Some(&Value::Str("1.5".into()))),
            Err(PyErr::Value(_))
        ));
        assert!(matches!(to_int(None), Err(PyErr::Type(_))));
        assert!(matches!(
            to_int(Some(&Value::Ints(vec![1]))),
            Err(PyErr::Type(_))
        ));
        assert!(matches!(
            to_int(Some(&Value::Float(f64::INFINITY))),
            Err(PyErr::Other(_))
        ));
        assert_eq!(to_float(Some(&Value::Int(3))), Ok(3.0));
        assert_eq!(py_len(Some(&Value::Str("ab,c".into()))), Ok(4));
        assert!(py_len(Some(&Value::Int(3))).is_err());
        assert!(!truthy(Some(&Value::Int(0))));
        assert!(truthy(Some(&Value::Float(f64::NAN))));
        assert!(!truthy(Some(&Value::Null)));
        assert_eq!(
            item(&Value::Str("12".into()), 1),
            Ok(Value::Str("2".into()))
        );
    }

    #[test]
    fn builtin_sum_is_compensated() {
        // Neumaier: 1e100 + 1.0 - 1e100 is 1.0 (a left fold gives 0.0).
        assert_eq!(py_fsum_builtin(&[1e100, 1.0, -1e100]), 1.0);
        assert_eq!(py_fsum_builtin(&[]), 0.0);
        assert_eq!(py_fsum_builtin(&[0.1, 0.2, 0.3]), 0.6);
        // -0.0 alone: int 0 + -0.0 is 0.0.
        assert_eq!(py_fsum_builtin(&[-0.0]).to_bits(), 0.0f64.to_bits());
    }
}
