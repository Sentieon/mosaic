// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! Numbers as the Python engine printed them: `str(float)`
//! (`py_str_float`), `round(x, 3) + 0.0` (`py_round`, the rule that no
//! INFO float is written as a negative zero) and `'%4.2f' % x` (`py_fmt_2f`). Rust's `{}` / `{:?}` / `{:.N}`
//! are not these (`1e16` is positional in Rust; the exponent has no sign
//! or padding). Pinned by the expectation tables Python generated
//! (`rust/tools/gen_pyfmt_tables.py`, `tests/data/pyfmt_*.tsv`).

/// Python's `repr(x)` / `str(x)` of a float: the shortest digits that
/// round-trip, positional when the decimal exponent is in [-4, 16) (an
/// integral value gets `.0`), else `d[.ddd]e±XX` (at least two exponent
/// digits); `nan`, `inf`, `-inf`; `-0.0` keeps its sign.
pub fn py_str_float(x: f64) -> String {
    if x.is_nan() {
        return "nan".to_string();
    }
    if x.is_infinite() {
        return if x > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    // Rust's `{:e}` without a precision gives the shortest round-trip
    // digits (the same digits as Python's repr), as `d.ddde<exp>`.
    let s = format!("{x:e}");
    let (mant, exp) = s.split_once('e').expect("`{:e}` has an exponent");
    let exp: i32 = exp.parse().expect("`{:e}` exponent is an integer");
    let (neg, mant) = match mant.strip_prefix('-') {
        Some(m) => (true, m),
        None => (false, mant),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    let mut out = String::with_capacity(24);
    if neg {
        out.push('-');
    }
    if (-4..16).contains(&exp) {
        if exp >= 0 {
            let int_len = exp as usize + 1;
            if digits.len() <= int_len {
                out.push_str(&digits);
                out.extend(std::iter::repeat_n('0', int_len - digits.len()));
                out.push_str(".0");
            } else {
                out.push_str(&digits[..int_len]);
                out.push('.');
                out.push_str(&digits[int_len..]);
            }
        } else {
            out.push_str("0.");
            out.extend(std::iter::repeat_n('0', (-exp - 1) as usize));
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
        out.push_str(&format!("{:02}", exp.unsigned_abs()));
    }
    out
}

/// Round `x` at `ndigits` decimals as the correctly rounded decimal
/// rounding of its exact binary value, half to even on an exact tie (what
/// Python's `round` and `'%.Nf'` do), as the decimal text. An exact tie at
/// `ndigits` decimals is a value whose `x · 2 · 10^ndigits` is an odd
/// integer, i.e. `x · 2^(ndigits + 1)` odd (the 5^ndigits factor cannot
/// divide a dyadic fraction's odd part otherwise): there the text is built
/// from the even neighbor; elsewhere Rust's exact formatting
/// (`{:.N}`, correctly rounded from the binary value) is the answer.
fn fixed_half_even(x: f64, ndigits: u32) -> String {
    let scaled = x * f64::from(1u32 << (ndigits + 1)); // exact: a power of two
    if scaled.fract() == 0.0 && scaled.abs() < 9.0e15 && (scaled as i64) % 2 != 0 {
        // A tie: x · 10^ndigits = k + 1/2 exactly; take the even of k, k + 1
        // (by magnitude: Python rounds the absolute value, the sign kept).
        let t = x.abs() * 10f64.powi(ndigits as i32); // exact: k + 0.5 < 2^53
        let k = t.floor() as i64;
        let n = if k % 2 == 0 { k } else { k + 1 };
        let p = 10i64.pow(ndigits);
        let sign = if x.is_sign_negative() { "-" } else { "" };
        if ndigits == 0 {
            return format!("{sign}{n}");
        }
        return format!("{sign}{}.{:0w$}", n / p, n % p, w = ndigits as usize);
    }
    format!("{x:.prec$}", prec = ndigits as usize)
}

/// `round(x, 3) + 0.0`: Python's correctly rounded
/// decimal rounding of the exact binary value, half to even on an exact
/// tie, parsed back to the nearest double; the `+ 0.0` turns a `-0.0` (and
/// a negative value that rounds to zero) into `0.0`. `nan` / `inf` pass.
pub fn py_round(x: f64) -> f64 {
    if !x.is_finite() {
        return x + 0.0;
    }
    let r: f64 = fixed_half_even(x, 3)
        .parse()
        .expect("fixed-point text parses");
    r + 0.0
}

/// `'%4.2f' % x`: the exact binary value rounded half to even at two
/// decimals, right-aligned in four characters; `nan`, `inf`, `-inf` as
/// Python prints them; a negative value that rounds to zero keeps its
/// sign (`-0.00`), as C's `%f` does.
pub fn py_fmt_2f(x: f64) -> String {
    let s = if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        fixed_half_even(x, 2)
    };
    format!("{s:>4}")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn table(name: &str) -> Vec<(u64, String)> {
        let p = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("tests/data")
            .join(name);
        let text = std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        text.lines()
            .filter(|l| !l.starts_with('#') && !l.is_empty())
            .map(|l| {
                let (bits, want) = l.split_once('\t').expect("bits<TAB>expected");
                (
                    u64::from_str_radix(bits, 16).expect("hex bits"),
                    want.to_string(),
                )
            })
            .collect()
    }

    fn check(name: &str, f: impl Fn(f64) -> String) -> usize {
        let rows = table(name);
        assert!(rows.len() > 1000, "{name}: {} rows", rows.len());
        let bad: Vec<String> = rows
            .iter()
            .filter_map(|(bits, want)| {
                let x = f64::from_bits(*bits);
                let got = f(x);
                (got != *want).then(|| format!("{x:e} ({bits:016x}): got {got:?}, want {want:?}"))
            })
            .collect();
        assert!(
            bad.is_empty(),
            "{name}: {} of {} differ:\n{}",
            bad.len(),
            rows.len(),
            bad[..bad.len().min(20)].join("\n")
        );
        rows.len()
    }

    #[test]
    fn str_float_table() {
        check("pyfmt_str.tsv", py_str_float);
    }

    #[test]
    fn round_table() {
        // The table holds `repr(round(x, 3) + 0.0)`: compared as the bits
        // of the result through `py_str_float` (the text is the
        // round-trip identity of the double).
        check("pyfmt_round.tsv", |x| py_str_float(py_round(x)));
    }

    #[test]
    fn fmt_2f_table() {
        check("pyfmt_2f.tsv", py_fmt_2f);
    }

    /// Replaces `tests/unit/test_joint_call_info.py::test_round3_never_negative_zero`
    /// (ported when the Python engine was removed): `_round3` is `py_round`;
    /// the same inputs, the Python results as bits and as `str()`.
    #[test]
    fn round3_python_cases() {
        for x in [-0.0, -0.0004, -0.00049, -1e-12] {
            let r = py_round(x);
            assert_eq!(r.to_bits(), 0.0f64.to_bits(), "{x:e}");
            assert_eq!(py_str_float(r), "0.0", "{x:e}");
        }
        // Python's round of the binary value: half to even on an exact tie.
        for (x, want) in [
            (0.0625, 0.062),
            (2.675, 2.675),
            (1.23456, 1.235),
            (-1.23456, -1.235),
            (3.0, 3.0),
        ] {
            assert_eq!(py_round(x).to_bits(), f64::to_bits(want), "{x}");
        }
        assert!(py_round(f64::NAN).is_nan());
    }

    #[test]
    fn worked_examples() {
        assert_eq!(py_str_float(1e16), "1e+16");
        assert_eq!(py_str_float(1e-5), "1e-05");
        assert_eq!(py_str_float(1.5e20), "1.5e+20");
        assert_eq!(py_str_float(0.0001), "0.0001");
        assert_eq!(py_str_float(-0.0), "-0.0");
        assert_eq!(py_str_float(123.0), "123.0");
        assert_eq!(py_round(0.0625).to_bits(), 0.062f64.to_bits());
        assert_eq!(py_round(2.675).to_bits(), 2.675f64.to_bits());
        assert_eq!(py_round(-0.0004).to_bits(), 0.0f64.to_bits());
        assert_eq!(py_round(-0.0).to_bits(), 0.0f64.to_bits());
        assert_eq!(py_fmt_2f(0.125), "0.12");
        assert_eq!(py_fmt_2f(1.005), "1.00");
        assert_eq!(py_fmt_2f(0.375), "0.38");
        assert_eq!(py_fmt_2f(5.0), "5.00");
    }
}
