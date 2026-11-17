// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! Typed field decoding with `vcflib`'s `VCF.parse_field` semantics, and
//! Python-compatible `int()` / `float()` / `bool()` parses.

/// A decoded INFO or FORMAT value, as `VCF.parse_field` returns it.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// `.` or a list of only `.`.
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    Bools(Vec<bool>),
    Ints(Vec<i64>),
    Floats(Vec<f64>),
    Strs(Vec<String>),
}

/// `Type` of a header declaration, as `VCF.decoders` maps it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ty {
    Integer,
    Float,
    Flag,
    /// Anything else: decoded as a string (`str`).
    Str,
}

/// A header declaration's `Type` and `Number` (`None` when the header
/// line lacks the key: vcflib raises `KeyError` when such a field occurs).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Decl {
    pub ty: Option<Ty>,
    pub number: Option<String>,
}

impl Decl {
    pub fn from_strs(ty: Option<&str>, number: Option<&str>) -> Decl {
        Decl {
            ty: ty.map(|t| match t {
                "Integer" => Ty::Integer,
                "Float" => Ty::Float,
                "Flag" => Ty::Flag,
                _ => Ty::Str,
            }),
            number: number.map(str::to_string),
        }
    }
}

/// Why a field did not decode.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    /// `int()` / `float()` raised `ValueError` in Python: the record is
    /// dropped and counted.
    Malformed(String),
    /// Python would raise something other than `ValueError` (a missing
    /// `Type` / `Number`, an `int` beyond i64, a list-valued flag): the
    /// shard fails.
    Fatal(String),
}

fn is_py_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c | 0x1c..=0x1f)
}

/// `str.strip()` over ASCII whitespace (as Python's `int()` / `float()`
/// strip their argument).
pub fn py_strip(s: &[u8]) -> &[u8] {
    let mut a = 0;
    let mut b = s.len();
    while a < b && is_py_space(s[a]) {
        a += 1;
    }
    while b > a && is_py_space(s[b - 1]) {
        b -= 1;
    }
    &s[a..b]
}

/// Validate PEP 515 underscores (only single ones between digits) and
/// return the digits without them; `None` if misplaced.
fn strip_underscores(s: &[u8]) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len());
    for (i, &c) in s.iter().enumerate() {
        if c == b'_' {
            let prev = i.checked_sub(1).map(|p| s[p]);
            let next = s.get(i + 1).copied();
            match (prev, next) {
                (Some(p), Some(n)) if p.is_ascii_digit() && n.is_ascii_digit() => {}
                _ => return None,
            }
        } else {
            out.push(c);
        }
    }
    Some(out)
}

/// Python's `int(s)` for a decimal string: surrounding whitespace, a sign,
/// digits with PEP 515 underscores. Beyond i64 is `Fatal` (Python would
/// return a big int).
pub fn py_int(s: &[u8]) -> Result<i64, DecodeError> {
    let bad = || DecodeError::Malformed(format!("invalid literal for int(): {:?}", lossy(s)));
    let t = py_strip(s);
    let (neg, digits) = match t.first() {
        Some(b'-') => (true, &t[1..]),
        Some(b'+') => (false, &t[1..]),
        _ => (false, t),
    };
    if digits.is_empty() || !digits[0].is_ascii_digit() {
        return Err(bad());
    }
    let mut v: i128 = 0;
    let mut prev_us = false;
    for &c in digits {
        if c == b'_' {
            if prev_us {
                return Err(bad());
            }
            prev_us = true;
            continue;
        }
        if !c.is_ascii_digit() {
            return Err(bad());
        }
        prev_us = false;
        v = v * 10 + i128::from(c - b'0');
        if v > i128::from(i64::MAX) + 1 {
            return Err(DecodeError::Fatal(format!(
                "integer beyond 64 bits: {:?}",
                lossy(s)
            )));
        }
    }
    if prev_us {
        return Err(bad());
    }
    let v = if neg { -v } else { v };
    i64::try_from(v)
        .map_err(|_| DecodeError::Fatal(format!("integer beyond 64 bits: {:?}", lossy(s))))
}

/// Python's `float(s)`: surrounding whitespace, PEP 515 underscores, and
/// the `inf` / `infinity` / `nan` spellings (which `str::parse` accepts
/// too). The parse is correctly rounded on both sides.
pub fn py_float(s: &[u8]) -> Result<f64, DecodeError> {
    let bad =
        || DecodeError::Malformed(format!("could not convert string to float: {:?}", lossy(s)));
    let t = py_strip(s);
    let owned;
    let t = if t.contains(&b'_') {
        owned = strip_underscores(t).ok_or_else(bad)?;
        &owned[..]
    } else {
        t
    };
    let st = std::str::from_utf8(t).map_err(|_| bad())?;
    // Rust also accepts "infinity"/"inf"/"nan" with a sign, case-free,
    // as Python does; neither accepts hex or an empty string.
    st.parse::<f64>().map_err(|_| bad())
}

fn lossy(s: &[u8]) -> String {
    String::from_utf8_lossy(s).into_owned()
}

/// The raw text of a field: `Some(bytes)` after `=` (or a FORMAT value),
/// `None` for an INFO key without `=` (vcflib's `parse_kv` gives `True`).
pub type Raw<'a> = Option<&'a [u8]>;

fn to_string(s: &[u8]) -> String {
    // The line was checked to be UTF-8 before tokenizing.
    String::from_utf8_lossy(s).into_owned()
}

/// `VCF.parse_field(desc, (k, v))` for one key: `decl` is the header's
/// declaration of the key (`None`: undeclared).
pub fn decode(decl: Option<&Decl>, raw: Raw<'_>) -> Result<Value, DecodeError> {
    let Some(d) = decl else {
        // Undeclared: the raw string, or True for a bare INFO key.
        return Ok(match raw {
            Some(s) => Value::Str(to_string(s)),
            None => Value::Bool(true),
        });
    };
    let ty =
        d.ty.ok_or_else(|| DecodeError::Fatal("header declaration without Type".into()))?;
    let number = d
        .number
        .as_deref()
        .ok_or_else(|| DecodeError::Fatal("header declaration without Number".into()))?;
    let scalar = number == "0" || number == "1";
    let Some(s) = raw else {
        // v is True: `True == '.'` is false; a list Number calls
        // `True.split` (AttributeError); a scalar applies cvt(True).
        if !scalar {
            return Err(DecodeError::Fatal(
                "a value-less INFO key declared with a list Number".into(),
            ));
        }
        return Ok(match ty {
            Ty::Integer => Value::Int(1),
            Ty::Float => Value::Float(1.0),
            Ty::Flag => Value::Bool(true),
            Ty::Str => Value::Str("True".into()),
        });
    };
    if s == b"." {
        return Ok(Value::Null);
    }
    if !scalar {
        let parts = s.split(|&c| c == b',');
        if s.split(|&c| c == b',').all(|x| x == b".") {
            return Ok(Value::Null);
        }
        return Ok(match ty {
            Ty::Integer => Value::Ints(parts.map(py_int).collect::<Result<_, _>>()?),
            Ty::Float => Value::Floats(parts.map(py_float).collect::<Result<_, _>>()?),
            Ty::Flag => Value::Bools(parts.map(|x| !x.is_empty()).collect()),
            Ty::Str => Value::Strs(parts.map(to_string).collect()),
        });
    }
    Ok(match ty {
        Ty::Integer => Value::Int(py_int(s)?),
        Ty::Float => Value::Float(py_float(s)?),
        Ty::Flag => Value::Bool(!s.is_empty()),
        Ty::Str => Value::Str(to_string(s)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn d(ty: &str, n: &str) -> Decl {
        Decl::from_strs(Some(ty), Some(n))
    }

    fn malformed(r: Result<Value, DecodeError>) -> bool {
        matches!(r, Err(DecodeError::Malformed(_)))
    }

    #[test]
    fn python_int() {
        assert_eq!(py_int(b"12"), Ok(12));
        assert_eq!(py_int(b" -7 "), Ok(-7));
        assert_eq!(py_int(b"+3"), Ok(3));
        assert_eq!(py_int(b"007"), Ok(7));
        assert_eq!(py_int(b"1_000"), Ok(1000));
        assert!(py_int(b"1__0").is_err());
        assert!(py_int(b"_1").is_err());
        assert!(py_int(b"1_").is_err());
        assert!(py_int(b"").is_err());
        assert!(py_int(b".").is_err());
        assert!(py_int(b"1.0").is_err());
        assert!(py_int(b"0x10").is_err());
        assert_eq!(py_int(b"-9223372036854775808"), Ok(i64::MIN));
        assert!(matches!(
            py_int(b"9223372036854775808"),
            Err(DecodeError::Fatal(_))
        ));
    }

    #[test]
    fn python_float() {
        assert_eq!(py_float(b"1.5"), Ok(1.5));
        assert_eq!(py_float(b" 2e3 "), Ok(2000.0));
        assert_eq!(py_float(b"1_0.5"), Ok(10.5));
        assert_eq!(py_float(b".5"), Ok(0.5));
        assert_eq!(py_float(b"5."), Ok(5.0));
        assert_eq!(py_float(b"-inf"), Ok(f64::NEG_INFINITY));
        assert_eq!(py_float(b"Infinity"), Ok(f64::INFINITY));
        assert!(py_float(b"nan").unwrap().is_nan());
        assert!(py_float(b".").is_err());
        assert!(py_float(b"").is_err());
        assert!(py_float(b"1e").is_err());
        assert!(py_float(b"0x1p3").is_err());
        // Correctly rounded: the shortest repr round-trips.
        assert_eq!(py_float(b"0.1").unwrap().to_bits(), 0.1f64.to_bits());
    }

    #[test]
    fn scalars_and_dots() {
        assert_eq!(
            decode(Some(&d("Integer", "1")), Some(b"5")),
            Ok(Value::Int(5))
        );
        assert_eq!(
            decode(Some(&d("Integer", "1")), Some(b".")),
            Ok(Value::Null)
        );
        assert_eq!(decode(Some(&d("Float", "1")), Some(b".")), Ok(Value::Null));
        assert_eq!(
            decode(Some(&d("Float", "1")), Some(b"0.5")),
            Ok(Value::Float(0.5))
        );
        assert_eq!(
            decode(Some(&d("String", "1")), Some(b"0/1")),
            Ok(Value::Str("0/1".into()))
        );
        // Number 0 is a scalar too.
        assert_eq!(
            decode(Some(&d("Integer", "0")), Some(b"3")),
            Ok(Value::Int(3))
        );
        assert!(malformed(decode(Some(&d("Integer", "1")), Some(b"x"))));
        assert!(malformed(decode(Some(&d("Float", "1")), Some(b"x"))));
        // A scalar never splits on ',': "1,2" is not an int.
        assert!(malformed(decode(Some(&d("Integer", "1")), Some(b"1,2"))));
    }

    #[test]
    fn lists_by_number() {
        for n in ["A", "R", "G", ".", "2", "3"] {
            assert_eq!(
                decode(Some(&d("Integer", n)), Some(b"1,2,3")),
                Ok(Value::Ints(vec![1, 2, 3])),
                "{n}"
            );
            // A single value is still a list.
            assert_eq!(
                decode(Some(&d("Integer", n)), Some(b"4")),
                Ok(Value::Ints(vec![4]))
            );
        }
        assert_eq!(
            decode(Some(&d("Float", "G")), Some(b"0.5,1")),
            Ok(Value::Floats(vec![0.5, 1.0]))
        );
        assert_eq!(
            decode(Some(&d("String", ".")), Some(b"a,.,b")),
            Ok(Value::Strs(vec!["a".into(), ".".into(), "b".into()]))
        );
    }

    #[test]
    fn dot_lists() {
        // All '.' -> None; mixed '.' in a numeric list -> ValueError.
        assert_eq!(
            decode(Some(&d("Integer", "R")), Some(b".,.")),
            Ok(Value::Null)
        );
        assert_eq!(
            decode(Some(&d("Integer", "R")), Some(b".")),
            Ok(Value::Null)
        );
        assert_eq!(
            decode(Some(&d("String", "R")), Some(b".,.,.")),
            Ok(Value::Null)
        );
        assert!(malformed(decode(Some(&d("Integer", "R")), Some(b"1,."))));
        assert!(malformed(decode(Some(&d("Float", "G")), Some(b".,0.5"))));
        assert!(malformed(decode(Some(&d("Integer", "R")), Some(b"1,,2"))));
    }

    #[test]
    fn undeclared_and_flags() {
        assert_eq!(decode(None, Some(b"12")), Ok(Value::Str("12".into())));
        assert_eq!(decode(None, Some(b".")), Ok(Value::Str(".".into())));
        assert_eq!(decode(None, None), Ok(Value::Bool(true)));
        assert_eq!(decode(Some(&d("Flag", "0")), None), Ok(Value::Bool(true)));
        assert_eq!(
            decode(Some(&d("Flag", "0")), Some(b"")),
            Ok(Value::Bool(false))
        );
        // cvt(True) for a declared scalar without '='.
        assert_eq!(decode(Some(&d("Integer", "1")), None), Ok(Value::Int(1)));
        assert_eq!(
            decode(Some(&d("String", "1")), None),
            Ok(Value::Str("True".into()))
        );
        assert!(matches!(
            decode(Some(&d("Integer", "A")), None),
            Err(DecodeError::Fatal(_))
        ));
        assert!(matches!(
            decode(Some(&Decl::from_strs(Some("Integer"), None)), Some(b"1")),
            Err(DecodeError::Fatal(_))
        ));
    }
}
