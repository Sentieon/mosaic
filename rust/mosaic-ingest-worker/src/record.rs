// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The record tokenizer: one gVCF line to the fields ingest reads, decoded
//! by the header's declarations (`VCF.parse` semantics, lazily: only the
//! needed keys are decoded).

use memchr::{memchr, memchr_iter};

use crate::header::Header;
use crate::value::{DecodeError, Value, decode, py_float, py_int};

/// The INFO keys ingest reads, in the dump's order.
pub const INFO_KEYS: [&str; 8] = [
    "DP",
    "END",
    "MQ",
    "MQSQ",
    "RAW_MQ",
    "RAW_MQandDP",
    "GTC",
    "SB",
];
/// The first sample's FORMAT keys ingest reads, in the dump's order.
pub const FMT_KEYS: [&str; 9] = ["GT", "AD", "DP", "MIN_DP", "SB", "PL", "F1R2", "F2R1", "GQ"];
pub const INFO_END: usize = 1;
pub const FMT_GT: usize = 0;

/// The decoded fields of one record (a `vcflib` `Variant`, restricted to
/// what ingest reads).
#[derive(Debug, Clone, PartialEq)]
pub struct Record {
    /// 0-based (`v.pos`).
    pub pos: i64,
    /// INFO `END`, else `pos + len(REF)` (`v.end`).
    pub end: i64,
    pub ref_: String,
    /// `ALT` split on `,`; `.` is none.
    pub alt: Vec<String>,
    /// `QUAL`; `.` is None.
    pub qual: Option<f64>,
    /// `FILTER` split on `;`; `.` is empty.
    pub filter: Vec<String>,
    /// `INFO_KEYS` by index: `None` absent, else the decoded value.
    pub info: [Option<Value>; 8],
    /// The first sample's `FMT_KEYS` (None: the record has no sample).
    pub s0: Option<[Option<Value>; 9]>,
    /// Samples whose GT is `0/0`, `0|0` or `0` (`_extract_rb_range`).
    pub homref: u32,
    /// The first sample's dict is non-empty (Python `if sample:`): the
    /// FORMAT column lists at least one key (vcflib zips the keys with the
    /// sample column, which splits into at least one value).
    pub s0_nonempty: bool,
    /// The decoded `GT` of the second and later samples (`None`: the
    /// sample has no `GT` key), for the allele counts over every sample
    /// (`collect_variant_group_data`). Empty for a one-sample file.
    pub rest_gts: Vec<Option<Value>>,
}

/// Why a line did not become a record.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ParseError {
    /// vcflib raises `ValueError`: grouper drops and counts it.
    Malformed(String),
    /// vcflib raises something else: the shard fails.
    Fatal(String),
}

impl From<DecodeError> for ParseError {
    fn from(e: DecodeError) -> Self {
        match e {
            DecodeError::Malformed(m) => ParseError::Malformed(m),
            DecodeError::Fatal(m) => ParseError::Fatal(m),
        }
    }
}

const NONE_VALUE: Option<Value> = None;

fn info_key_index(k: &[u8]) -> Option<usize> {
    INFO_KEYS.iter().position(|x| x.as_bytes() == k)
}

fn fmt_key_index(k: &[u8]) -> Option<usize> {
    FMT_KEYS.iter().position(|x| x.as_bytes() == k)
}

fn is_rstrip_space(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r' | 0x0b | 0x0c | 0x1c..=0x1f)
}

fn to_string(s: &[u8]) -> String {
    String::from_utf8_lossy(s).into_owned()
}

fn split_strings(s: &[u8], sep: u8) -> Vec<String> {
    s.split(|&c| c == sep).map(to_string).collect()
}

/// A tokenizer with reusable scratch space (one per reader).
#[derive(Default)]
pub struct Tokenizer {
    fields: Vec<(usize, usize)>,
    fmt_idx: Vec<Option<usize>>,
}

/// Is a decoded GT one of `0/0`, `0|0`, `0` (Python `gt in (...)`: only a
/// string can be equal)?
fn is_homref(v: &Value) -> bool {
    matches!(v, Value::Str(s) if s == "0/0" || s == "0|0" || s == "0")
}

impl Tokenizer {
    /// Parse one line (without its newline). Returns the CHROM bytes and
    /// the record. As `VCF.__next__` + `VCF.parse`: the line must be UTF-8
    /// and is `rstrip`ped first.
    pub fn parse<'a>(
        &mut self,
        line: &'a [u8],
        hdr: &Header,
    ) -> Result<(&'a [u8], Record), ParseError> {
        if std::str::from_utf8(line).is_err() {
            return Err(ParseError::Malformed("line is not UTF-8".into()));
        }
        let mut n = line.len();
        while n > 0 && is_rstrip_space(line[n - 1]) {
            n -= 1;
        }
        let line = &line[..n];
        self.fields.clear();
        let mut a = 0;
        for t in memchr_iter(b'\t', line) {
            self.fields.push((a, t));
            a = t + 1;
        }
        self.fields.push((a, line.len()));
        let nf = self.fields.len();
        if nf < 7 {
            return Err(ParseError::Fatal(format!(
                "a VCF line with {nf} column(s): {}",
                to_string(line)
            )));
        }
        let fld = |i: usize| -> &'a [u8] {
            let (s, e) = self.fields[i];
            &line[s..e]
        };

        // INFO: every occurrence of a needed key is decoded (a repeated
        // key: the last wins, as dict() keeps it).
        let mut info: [Option<Value>; 8] = [NONE_VALUE; 8];
        if nf >= 8 && fld(7) != b"." {
            for item in fld(7).split(|&c| c == b';') {
                let (k, v) = match memchr(b'=', item) {
                    Some(e) => (&item[..e], Some(&item[e + 1..])),
                    None => (item, None),
                };
                if let Some(i) = info_key_index(k) {
                    info[i] = Some(decode(hdr.info[i].as_ref(), v)?);
                }
            }
        }

        // FORMAT keys and the samples (zip: the shorter side truncates).
        self.fmt_idx.clear();
        if nf >= 9 && fld(8) != b"." {
            for k in fld(8).split(|&c| c == b':') {
                self.fmt_idx.push(fmt_key_index(k));
            }
        }
        let mut s0: Option<[Option<Value>; 9]> = None;
        let mut homref = 0u32;
        let mut rest_gts: Vec<Option<Value>> = Vec::new();
        for si in 9..nf {
            let col = fld(si);
            if si == 9 {
                let mut vals: [Option<Value>; 9] = [NONE_VALUE; 9];
                for (k, v) in self.fmt_idx.iter().zip(col.split(|&c| c == b':')) {
                    if let Some(i) = *k {
                        vals[i] = Some(decode(hdr.format[i].as_ref(), Some(v))?);
                    }
                }
                if vals[FMT_GT].as_ref().is_some_and(is_homref) {
                    homref += 1;
                }
                s0 = Some(vals);
            } else {
                let mut gt: Option<Value> = None;
                for (k, v) in self.fmt_idx.iter().zip(col.split(|&c| c == b':')) {
                    if *k == Some(FMT_GT) {
                        gt = Some(decode(hdr.format[FMT_GT].as_ref(), Some(v))?);
                    }
                }
                if gt.as_ref().is_some_and(is_homref) {
                    homref += 1;
                }
                rest_gts.push(gt);
            }
        }
        let s0_nonempty = !self.fmt_idx.is_empty();

        let pos = py_int(fld(1))? - 1;
        let alt = if fld(4) == b"." {
            Vec::new()
        } else {
            split_strings(fld(4), b',')
        };
        let qual = if fld(5) == b"." {
            None
        } else {
            Some(py_float(fld(5))?)
        };
        let filter = if fld(6) == b"." {
            Vec::new()
        } else {
            split_strings(fld(6), b';')
        };
        let ref_ = to_string(fld(3));
        let end = match &info[INFO_END] {
            None => pos + ref_.chars().count() as i64,
            Some(Value::Int(e)) => *e,
            Some(other) => {
                return Err(ParseError::Fatal(format!(
                    "INFO/END is not an integer ({other:?}): {}",
                    to_string(line)
                )));
            }
        };
        Ok((
            fld(0),
            Record {
                pos,
                end,
                ref_,
                alt,
                qual,
                filter,
                info,
                s0,
                homref,
                s0_nonempty,
                rest_gts,
            },
        ))
    }
}

/// `is_ref_block`: REF `.`, or a single ALT that is
/// `<NON_REF>` or `*` (`mosaic_core::alleles::is_ref_block_alleles`).
pub fn is_ref_block(v: &Record) -> bool {
    mosaic_core::alleles::is_ref_block_alleles(&v.ref_, &v.alt)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hdr() -> Header {
        let lines: Vec<String> = [
            "##INFO=<ID=END,Number=1,Type=Integer,Description=\"x\">",
            "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"x\">",
            "##INFO=<ID=MQ,Number=1,Type=Float,Description=\"x\">",
            "##INFO=<ID=GTC,Number=G,Type=Integer,Description=\"x\">",
            "##INFO=<ID=SB,Number=.,Type=Float,Description=\"x\">",
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"x\">",
            "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=SB,Number=4,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=F1R2,Number=A,Type=Integer,Description=\"x\">",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        Header::parse(&lines).unwrap()
    }

    fn parse(line: &str) -> Result<Record, ParseError> {
        Tokenizer::default()
            .parse(line.as_bytes(), &hdr())
            .map(|(_, r)| r)
    }

    #[test]
    fn ref_block() {
        let r = parse(
            "chr21\t101\t.\tA\t<NON_REF>\t.\t.\tEND=150\tGT:AD:DP:GQ:PL\t0/0:10,0:12:30:0,30,450\t0|0:1,0",
        )
        .unwrap();
        assert_eq!((r.pos, r.end), (100, 150));
        assert_eq!(r.alt, vec!["<NON_REF>"]);
        assert_eq!(r.qual, None);
        assert!(r.filter.is_empty());
        assert_eq!(r.info[INFO_END], Some(Value::Int(150)));
        assert_eq!(r.info[0], None);
        let s0 = r.s0.unwrap();
        assert_eq!(s0[0], Some(Value::Str("0/0".into())));
        assert_eq!(s0[1], Some(Value::Ints(vec![10, 0])));
        assert_eq!(s0[5], Some(Value::Ints(vec![0, 30, 450])));
        assert_eq!(r.homref, 2);
    }

    #[test]
    fn variant_fields() {
        let r = parse(
            "chr21\t200\t.\tAC\tA,ACC,<NON_REF>\t50.25\tPASS;LowQual\tDP=20;MQ=59.5;GTC=.,.;SB=1.5,.;X=1\tGT:AD:DP\t./.:.:.\t1/1",
        );
        // SB is a Float list holding a '.': float('.') raises -> dropped.
        assert!(matches!(r, Err(ParseError::Malformed(_))));
        let r = parse(
            "chr21\t200\t.\tAC\tA,ACC,<NON_REF>\t50.25\tPASS;LowQual\tDP=20;MQ=59.5;GTC=.,.;X=1;FLAGX\tGT:AD:DP\t./.:.:.\t1/1",
        )
        .unwrap();
        assert_eq!((r.pos, r.end), (199, 201)); // no END: pos + len(REF)
        assert_eq!(r.alt, vec!["A", "ACC", "<NON_REF>"]);
        assert_eq!(r.qual.unwrap().to_bits(), 50.25f64.to_bits());
        assert_eq!(r.filter, vec!["PASS", "LowQual"]);
        assert_eq!(r.info[0], Some(Value::Int(20)));
        assert_eq!(r.info[2], Some(Value::Float(59.5)));
        assert_eq!(r.info[6], Some(Value::Null)); // all-'.' list
        let s0 = r.s0.unwrap();
        assert_eq!(s0[0], Some(Value::Str("./.".into())));
        assert_eq!(s0[1], Some(Value::Null));
        assert_eq!(s0[2], Some(Value::Null));
        assert_eq!(r.homref, 0);
    }

    #[test]
    fn short_sample_column_truncates() {
        let r = parse("chr21\t5\t.\tA\t<NON_REF>\t.\t.\tEND=9\tGT:AD:DP:GQ\t0/0:3").unwrap();
        let s0 = r.s0.unwrap();
        assert_eq!(s0[0], Some(Value::Str("0/0".into())));
        assert_eq!(s0[1], Some(Value::Ints(vec![3])));
        assert_eq!(s0[2], None);
        assert_eq!(s0[8], None);
        // FORMAT without a sample column: no s0.
        let r = parse("chr21\t5\t.\tA\t<NON_REF>\t.\t.\tEND=9\tGT:AD").unwrap();
        assert_eq!(r.s0, None);
        // No FORMAT column at all.
        let r = parse("chr21\t5\t.\tA\t.\t.\t.\t.").unwrap();
        assert_eq!(r.s0, None);
        assert!(r.alt.is_empty());
        assert_eq!(r.end, 5);
    }

    #[test]
    fn undeclared_keys_stay_strings() {
        // RAW_MQ is not declared in this header.
        let r = parse("chr21\t5\t.\tA\tG\t3\t.\tRAW_MQ=12.5;MQSQ\tGT:MIN_DP\t0/1:x").unwrap();
        assert_eq!(r.info[4], Some(Value::Str("12.5".into())));
        assert_eq!(r.info[3], Some(Value::Bool(true)));
        // MIN_DP undeclared too: 'x' stays a string, no drop.
        assert_eq!(r.s0.unwrap()[3], Some(Value::Str("x".into())));
    }

    #[test]
    fn malformed_needed_fields_drop() {
        for line in [
            "chr21\tx\t.\tA\tG\t3\t.\t.",                        // POS
            "chr21\t5\t.\tA\tG\tq\t.\t.",                        // QUAL
            "chr21\t5\t.\tA\tG\t3\t.\tDP=1.5",                   // INFO Integer
            "chr21\t5\t.\tA\t<NON_REF>\t.\t.\tEND=x",            // END
            "chr21\t5\t.\tA\tG\t3\t.\t.\tGT:AD\t0/1:1,x",        // FORMAT list
            "chr21\t5\t.\tA\tG\t3\t.\t.\tGT:DP\t0/1:1:2\t0/1:.", // fine below
        ] {
            let r = parse(line);
            if line.ends_with("0/1:.") {
                assert!(r.is_ok(), "{line}");
            } else {
                assert!(matches!(r, Err(ParseError::Malformed(_))), "{line}");
            }
        }
        // An unneeded malformed field is kept (the accepted divergence).
        assert!(parse("chr21\t5\t.\tA\tG\t3\t.\tFOO=1\tGT:XX\t0/1:abc").is_ok());
        // A second sample's malformed DP is not read either.
        assert!(parse("chr21\t5\t.\tA\tG\t3\t.\t.\tGT:DP\t0/1:3\t0/1:z").is_ok());
        // Too few columns is fatal, as vcflib's IndexError.
        assert!(matches!(
            parse("chr21\t5\t.\tA\tG\t3"),
            Err(ParseError::Fatal(_))
        ));
        // Not UTF-8: UnicodeDecodeError is a ValueError.
        let mut t = Tokenizer::default();
        assert!(matches!(
            t.parse(b"chr21\t5\t.\tA\tG\t3\t.\tX=\xff", &hdr()),
            Err(ParseError::Malformed(_))
        ));
    }

    #[test]
    fn rstrip_and_repeated_keys() {
        let r = parse("chr21\t5\t.\tA\tG\t3\t.\tDP=1;DP=7\t\t ").unwrap();
        assert_eq!(r.info[0], Some(Value::Int(7)));
        assert_eq!(r.s0, None); // the trailing tabs were stripped
    }

    #[test]
    fn is_ref_block_on_records() {
        let h = Header::default();
        let mut t = Tokenizer::default();
        let mut rec = |line: &str| t.parse(line.as_bytes(), &h).unwrap().1;
        assert!(is_ref_block(&rec("c\t1\t.\tA\t<NON_REF>\t.\t.\t.")));
        assert!(is_ref_block(&rec("c\t1\t.\tA\t*\t.\t.\t.")));
        assert!(is_ref_block(&rec("c\t1\t.\t.\tG\t.\t.\t.")));
        assert!(!is_ref_block(&rec("c\t1\t.\tA\tG,<NON_REF>\t.\t.\t.")));
        assert!(!is_ref_block(&rec("c\t1\t.\tA\t.\t.\t.\t.")));
    }
}
