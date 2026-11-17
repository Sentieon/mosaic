// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! `is_ref_block`, `trim_shared_suffix`, `format_alt_allele`: the
//! allele encoding (originally a port of the Python ingest worker's
//! functions of those names). Strings are compared by character, as
//! Python's `str`. Shared by the ingest worker and the joint caller (moved
//! from the worker).

/// `is_ref_block` on a record's REF and ALTs: REF `.`, or a single ALT
/// that is `<NON_REF>` or `*`. The worker's `record::is_ref_block` applies
/// it to a parsed record.
pub fn is_ref_block_alleles<S: AsRef<str>>(ref_: &str, alt: &[S]) -> bool {
    ref_ == "." || (alt.len() == 1 && (alt[0].as_ref() == "<NON_REF>" || alt[0].as_ref() == "*"))
}

/// Python `len(s)`.
pub fn clen(s: &str) -> usize {
    if s.is_ascii() {
        s.len()
    } else {
        s.chars().count()
    }
}

/// `trim_shared_suffix`: drop the common suffix of REF and ALT.
pub fn trim_shared_suffix(r: &str, a: &str) -> (String, String) {
    if r.is_empty() || a.is_empty() {
        return (r.to_string(), a.to_string());
    }
    let rc: Vec<char> = r.chars().collect();
    let ac: Vec<char> = a.chars().collect();
    let min_len = rc.len().min(ac.len());
    let mut i = 0;
    while i < min_len && rc[rc.len() - 1 - i] == ac[ac.len() - 1 - i] {
        i += 1;
    }
    (
        rc[..rc.len() - i].iter().collect(),
        ac[..ac.len() - i].iter().collect(),
    )
}

/// `format_alt_allele`: `*` for `<NON_REF>`; else `x<alt>` (SNV / MNP,
/// printing the `Found MNP` line for an MNP), `+<ins>` or `-<del>` after
/// the suffix trim.
pub fn format_alt_allele(r: &str, a: &str) -> String {
    if a == "<NON_REF>" {
        return "*".to_string();
    }
    let (t_ref, t_alt) = trim_shared_suffix(r, a);
    let (lr, la) = (clen(&t_ref), clen(&t_alt));
    if lr == la {
        if la > 1 {
            eprintln!("Found MNP: {r}\t{a}\t{t_ref}\t{t_alt}");
        }
        format!("x{t_alt}")
    } else if la > lr {
        format!("+{t_alt}")
    } else {
        format!("-{t_ref}")
    }
}

#[cfg(test)]
mod tests {
    //! `tests/unit/test_allele_encoding.py`'s encoding cases.
    use super::*;

    #[test]
    fn trim() {
        for (r, a, want) in [
            ("A", "T", ("A", "T")),
            ("AT", "GT", ("A", "G")),
            ("ATG", "AG", ("AT", "A")),
            ("A", "AT", ("A", "AT")),
            ("ACGT", "ACGT", ("", "")),
            ("", "A", ("", "A")),
            ("A", "", ("A", "")),
        ] {
            let got = trim_shared_suffix(r, a);
            assert_eq!((got.0.as_str(), got.1.as_str()), want, "{r} {a}");
        }
    }

    #[test]
    fn format() {
        for (r, a, want) in [
            ("A", "G", "xG"),
            ("AC", "GT", "xGT"),
            ("A", "AT", "+AT"),
            ("ATG", "A", "-ATG"),
            ("A", "<NON_REF>", "*"),
            ("A", "*", "x*"),
            ("ATG", "CA", "-ATG"),
            ("CT", "CCT", "+C"),
        ] {
            assert_eq!(format_alt_allele(r, a), want, "{r} {a}");
        }
    }

    #[test]
    fn ref_block_alleles() {
        assert!(is_ref_block_alleles("A", &["<NON_REF>"]));
        assert!(is_ref_block_alleles("A", &["*"]));
        assert!(is_ref_block_alleles(".", &["G"]));
        assert!(!is_ref_block_alleles("A", &["G", "<NON_REF>"]));
        assert!(!is_ref_block_alleles::<&str>("A", &[]));
    }

    // Replaces tests/unit/test_allele_encoding.py::test_trim_shared_suffix (K6).
    #[test]
    fn trim_python_cases() {
        for (r, a, want) in [
            ("A", "T", ("A", "T")),
            ("AT", "GT", ("A", "G")),
            ("ATG", "ATG", ("", "")),
            ("CAT", "AT", ("C", "")),
            ("", "A", ("", "A")),
            ("A", "", ("A", "")),
        ] {
            let got = trim_shared_suffix(r, a);
            assert_eq!((got.0.as_str(), got.1.as_str()), want, "{r} {a}");
        }
    }

    // Replaces tests/unit/test_allele_encoding.py::test_format_alt_allele (K6).
    #[test]
    fn format_python_cases() {
        for (r, a, want) in [
            ("A", "<NON_REF>", "*"),
            ("A", "T", "xT"),
            ("AT", "GT", "xG"),
            ("A", "AT", "+AT"),
            ("AT", "A", "-AT"),
            ("ATG", "A", "-ATG"),
        ] {
            assert_eq!(format_alt_allele(r, a), want, "{r} {a}");
        }
    }

    // Replaces tests/unit/test_allele_encoding.py::test_is_ref_block (K6).
    #[test]
    fn ref_block_python_cases() {
        let cases: [(&str, &[&str], bool); 5] = [
            (".", &[], true),
            ("A", &["<NON_REF>"], true),
            ("A", &["*"], true),
            ("A", &["T"], false),
            ("A", &["T", "<NON_REF>"], false),
        ];
        for (r, alt, want) in cases {
            assert_eq!(is_ref_block_alleles(r, alt), want, "{r} {alt:?}");
        }
    }
}
