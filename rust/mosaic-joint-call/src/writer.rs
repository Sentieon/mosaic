// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The VCF writer: vcflib's `VCF.format` reproduced exactly,
//! the header lines as the driver built them, and the output as a plain
//! file or, when the path ends in `.gz`, BGZF (noodles-bgzf with
//! libdeflate: each chunk of a unit's text compressed into blocks of its
//! own, the blocks laid down in unit order by `ordered::OrderedWriter`, the
//! EOF block appended once). No tabix index.

use std::fs::File;
use std::io::{BufWriter, Read, Write};

use anyhow::{Context, Result, bail};
use noodles_bgzf as bgzf;

use crate::call::Info;
use crate::format::Call;
use crate::pyfmt::{py_fmt_2f, py_str_float};

fn join_ints(v: &[i64]) -> String {
    v.iter().map(i64::to_string).collect::<Vec<_>>().join(",")
}

/// The INFO column: the present keys in sorted order (`AC`, `AF`, `AN`,
/// `DP`, `FS`, `InbreedingCoeff`, `MQ`, `QD`, `SOR`), `key=value`, lists
/// joined by `,` (no ALT: no `AC` and no `AF`, K5), integers as `str(int)`, floats as
/// `str(float)`.
pub fn format_info(ac: &[i64], an: i64, af: &[f64], info: &Info) -> String {
    let mut f = Vec::with_capacity(9);
    // A record without ALTs carries neither AC nor AF (as bcftools writes a record whose last ALT it trims; an empty
    // `AC=` makes htslib warn and `--trim-alt-alleles` rewrite AN).
    if !ac.is_empty() {
        f.push(format!("AC={}", join_ints(ac)));
        f.push(format!(
            "AF={}",
            af.iter()
                .map(|&x| py_str_float(x))
                .collect::<Vec<_>>()
                .join(",")
        ));
    }
    f.push(format!("AN={an}"));
    if let Some(dp) = info.dp {
        f.push(format!("DP={dp}"));
    }
    if let Some(x) = info.fs {
        f.push(format!("FS={}", py_str_float(x)));
    }
    if let Some(x) = info.inbreeding_coeff {
        f.push(format!("InbreedingCoeff={}", py_str_float(x)));
    }
    if let Some(x) = info.mq {
        f.push(format!("MQ={}", py_str_float(x)));
    }
    if let Some(x) = info.qd {
        f.push(format!("QD={}", py_str_float(x)));
    }
    if let Some(x) = info.sor {
        f.push(format!("SOR={}", py_str_float(x)));
    }
    f.join(";")
}

/// A record's site columns.
#[derive(Debug, Clone, Copy)]
pub struct Record<'a> {
    pub chrom: &'a str,
    pub pos: i32,
    pub reference: &'a str,
    pub alts: &'a [String],
    pub qual: Option<f64>,
    pub info: &'a str,
}

/// One record line (newline-terminated): `CHROM POS . REF ALT QUAL . INFO`, then,
/// with samples, `FORMAT` (`GT` plus the keys any sample carries, sorted:
/// `AD`, `DP`, `F1R2`, `F2R1`, `PL`) and one column per sample, a missing
/// key `.`. ALT `.` when none; QUAL `'%4.2f'` or `.`.
pub fn format_record(out: &mut String, rec: &Record<'_>, calls: &[Call]) {
    use std::fmt::Write as _;
    let Record {
        chrom,
        pos,
        reference,
        alts,
        qual,
        info,
    } = *rec;
    let alt = if alts.is_empty() {
        ".".to_string()
    } else {
        alts.join(",")
    };
    let qual = qual.map_or_else(|| ".".to_string(), py_fmt_2f);
    let _ = write!(
        out,
        "{chrom}\t{pos}\t.\t{reference}\t{alt}\t{qual}\t.\t{info}"
    );
    if !calls.is_empty() {
        let has = |k: fn(&Call) -> bool| calls.iter().any(k);
        let (ad, dp, f1, f2, pl) = (
            has(|c| c.ad.is_some()),
            has(|c| c.dp.is_some()),
            has(|c| c.f1r2.is_some()),
            has(|c| c.f2r1.is_some()),
            has(|c| c.pl.is_some()),
        );
        out.push_str("\tGT");
        for (on, key) in [
            (ad, "AD"),
            (dp, "DP"),
            (f1, "F1R2"),
            (f2, "F2R1"),
            (pl, "PL"),
        ] {
            if on {
                out.push(':');
                out.push_str(key);
            }
        }
        let list = |v: &Option<Vec<i64>>| v.as_deref().map_or_else(|| ".".to_string(), join_ints);
        for c in calls {
            out.push('\t');
            out.push_str(&c.gt);
            if ad {
                let _ = write!(out, ":{}", list(&c.ad));
            }
            if dp {
                match c.dp {
                    Some(d) => {
                        let _ = write!(out, ":{d}");
                    }
                    None => out.push_str(":."),
                }
            }
            if f1 {
                let _ = write!(out, ":{}", list(&c.f1r2));
            }
            if f2 {
                let _ = write!(out, ":{}", list(&c.f2r1));
            }
            if pl {
                let _ = write!(out, ":{}", list(&c.pl));
            }
        }
    }
    out.push('\n');
}

/// The output file: plain, or BGZF when the path ends in `.gz`.
pub struct Output {
    file: BufWriter<File>,
    bgzf: bool,
    /// Uncompressed bytes written (the header and the records).
    pub text_bytes: u64,
}

impl Output {
    pub fn create(path: &str) -> Result<Output> {
        let f = File::create(path).with_context(|| format!("{path}: create the output"))?;
        Ok(Output {
            file: BufWriter::with_capacity(1 << 20, f),
            bgzf: path.ends_with(".gz"),
            text_bytes: 0,
        })
    }

    /// Compress `text` into BGZF blocks of its own (no EOF block).
    pub fn compress(text: &[u8]) -> Result<Vec<u8>> {
        let mut w = bgzf::io::Writer::new(Vec::with_capacity(text.len() / 3 + 64));
        w.write_all(text)?;
        w.flush()?;
        // The blocks without the EOF marker: `into_inner` does not finish.
        Ok(w.into_inner())
    }

    /// Whether the output is BGZF (the path ends in `.gz`).
    pub fn is_bgzf(&self) -> bool {
        self.bgzf
    }

    /// Write bytes already in the output's form (a chunk's BGZF blocks, or
    /// its text for a plain output) holding `text_bytes` uncompressed
    /// bytes, in order.
    pub fn write_raw(&mut self, bytes: &[u8], text_bytes: u64) -> Result<()> {
        self.text_bytes += text_bytes;
        self.file.write_all(bytes)?;
        Ok(())
    }

    /// Copy `len` bytes already in the output's form from `r` (a spill
    /// file) through the output's buffer, in order.
    pub fn copy_from(&mut self, r: &mut impl Read, len: u64, text_bytes: u64) -> Result<()> {
        let copied = std::io::copy(&mut r.take(len), &mut self.file)?;
        if copied != len {
            bail!("copied {copied} of {len} bytes");
        }
        self.text_bytes += text_bytes;
        Ok(())
    }

    /// Write one chunk of text (the header, a unit's records) in order.
    pub fn write_text(&mut self, text: &[u8]) -> Result<()> {
        self.text_bytes += text.len() as u64;
        if self.bgzf {
            let blocks = Output::compress(text)?;
            self.file.write_all(&blocks)?;
        } else {
            self.file.write_all(text)?;
        }
        Ok(())
    }

    /// Append the BGZF EOF block (once) and flush.
    pub fn finish(mut self) -> Result<()> {
        if self.bgzf {
            let eof = bgzf::io::Writer::new(Vec::new()).finish()?;
            self.file.write_all(&eof)?;
        }
        self.file.flush().context("flush the output")?;
        Ok(())
    }
}

/// The header text: every line as given, newline-terminated.
pub fn header_text(lines: &[String]) -> String {
    let mut s = String::new();
    for l in lines {
        s.push_str(l);
        s.push('\n');
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn call(gt: &str, ad: Option<Vec<i64>>, dp: Option<i64>, pl: Option<Vec<i64>>) -> Call {
        Call {
            gt: gt.to_string(),
            ad,
            dp,
            f1r2: None,
            f2r1: None,
            pl,
        }
    }

    #[test]
    fn info_keys_sorted_and_numbers_as_python() {
        let info = Info {
            dp: Some(15),
            qd: Some(2.033),
            mq: Some(60.0),
            fs: Some(0.0),
            sor: Some(0.693),
            inbreeding_coeff: Some(-0.25),
        };
        assert_eq!(
            format_info(&[1, 0], 4, &[0.25, 0.0], &info),
            "AC=1,0;AF=0.25,0.0;AN=4;DP=15;FS=0.0;InbreedingCoeff=-0.25;MQ=60.0;QD=2.033;SOR=0.693"
        );
        // No ALT (K5): neither AC nor AF; AN and the site's fields as ever.
        assert_eq!(format_info(&[], 0, &[], &Info::default()), "AN=0");
        let dp = Info {
            dp: Some(9),
            ..Info::default()
        };
        assert_eq!(format_info(&[], 2, &[], &dp), "AN=2;DP=9");
        assert_eq!(
            format_info(&[1], 2, &[1e-05], &Info::default()),
            "AC=1;AF=1e-05;AN=2"
        );
    }

    #[test]
    fn format_keys_are_the_union_and_missing_is_dot() {
        let calls = vec![
            call("0/1", Some(vec![3, 4]), Some(7), Some(vec![50, 0, 60])),
            call("./.", None, None, None),
            Call {
                f2r1: Some(vec![1, 2]),
                ..call("0/0", Some(vec![5, 0]), None, Some(vec![0, 15, 990]))
            },
        ];
        let alts = vec!["C".to_string()];
        let rec = Record {
            chrom: "chr21",
            pos: 100,
            reference: "T",
            alts: &alts,
            qual: Some(0.125),
            info: "AN=4",
        };
        let mut out = String::new();
        format_record(&mut out, &rec, &calls);
        assert_eq!(
            out,
            "chr21\t100\t.\tT\tC\t0.12\t.\tAN=4\tGT:AD:DP:F2R1:PL\t\
             0/1:3,4:7:.:50,0,60\t./.:.:.:.:.\t0/0:5,0:.:1,2:0,15,990\n"
        );
        // No ALT, no QUAL; GT alone.
        let mut out = String::new();
        let rec = Record {
            alts: &[],
            qual: None,
            ..rec
        };
        format_record(&mut out, &rec, &[call("./.", None, None, None)]);
        assert_eq!(out, "chr21\t100\t.\tT\t.\t.\t.\tAN=4\tGT\t./.\n");
        // No samples: no FORMAT column.
        let mut out = String::new();
        format_record(&mut out, &rec, &[]);
        assert_eq!(out, "chr21\t100\t.\tT\t.\t.\t.\tAN=4\n");
    }

    #[test]
    fn bgzf_blocks_concatenate_with_one_eof() {
        use std::io::Read;
        let dir = std::env::temp_dir().join(format!("mosaic-jc-bgzf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.vcf.gz");
        let path = path.to_str().unwrap();
        let mut o = Output::create(path).unwrap();
        let big = "x".repeat(200_000) + "\n";
        for chunk in ["##h\n", big.as_str(), "", "chr1\t1\n"] {
            o.write_text(chunk.as_bytes()).unwrap();
        }
        o.finish().unwrap();
        let bytes = std::fs::read(path).unwrap();
        // Ends with exactly one EOF block.
        let eof = bgzf::io::Writer::new(Vec::new()).finish().unwrap();
        assert!(bytes.ends_with(&eof));
        assert!(!bytes[..bytes.len() - eof.len()].ends_with(&eof));
        let mut text = String::new();
        bgzf::io::Reader::new(&bytes[..])
            .read_to_string(&mut text)
            .unwrap();
        assert_eq!(text, format!("##h\n{big}chr1\t1\n"));
        // A plain path: the text as is.
        let plain = dir.join("out.vcf");
        let mut o = Output::create(plain.to_str().unwrap()).unwrap();
        o.write_text(b"a\n").unwrap();
        o.finish().unwrap();
        assert_eq!(std::fs::read(&plain).unwrap(), b"a\n");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
