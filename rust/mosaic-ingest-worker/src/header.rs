// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The gVCF header, parsed once per input the way `VCF.parse_header` does:
//! the contigs (ID, length), the sample names and the `Type` / `Number`
//! of the INFO and FORMAT keys the worker decodes.

use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::record::{FMT_KEYS, INFO_KEYS};
use crate::value::Decl;

#[derive(Debug, Clone, Default)]
pub struct Header {
    /// `##contig` lines in header order: (ID, `length` if present).
    pub contigs: Vec<(String, Option<i64>)>,
    /// The `#CHROM` line's columns from the tenth on.
    pub samples: Vec<String>,
    /// Declarations of `INFO_KEYS`, by index (`None`: undeclared).
    pub info: Vec<Option<Decl>>,
    /// Declarations of `FMT_KEYS`, by index.
    pub format: Vec<Option<Decl>>,
}

/// `VCF.kvpat.findall` (`(.*?)=(".*?"|.*?)(?:,|$)`) over the text between
/// the line's first `<` and first `>`, into a dict (a repeated key: the
/// last wins).
pub fn parse_kv_line(line: &str) -> Result<HashMap<String, String>> {
    let (Some(s), Some(e)) = (line.find('<'), line.find('>')) else {
        bail!("header line without <...>: {line}");
    };
    let body = if s < e { &line[s + 1..e] } else { "" };
    let b = body.as_bytes();
    let mut out = HashMap::new();
    let mut p = 0;
    while p <= b.len() {
        // (.*?)= : the key runs to the first '=' at or after p.
        let Some(eq) = b[p..].iter().position(|&c| c == b'=').map(|x| x + p) else {
            break;
        };
        let q = eq + 1;
        let mut end = None; // (value end, match end)
        if b.get(q) == Some(&b'"') {
            // ".*?" followed by ',' or the end; the lazy match tries each
            // closing quote in turn.
            let mut r = q + 1;
            while let Some(off) = b[r..].iter().position(|&c| c == b'"') {
                let close = r + off;
                if close + 1 == b.len() {
                    end = Some((close + 1, close + 1));
                    break;
                }
                if b[close + 1] == b',' {
                    end = Some((close + 1, close + 2));
                    break;
                }
                r = close + 1;
            }
        }
        let (vend, mend) = end.unwrap_or_else(|| match b[q..].iter().position(|&c| c == b',') {
            Some(off) => (q + off, q + off + 1),
            None => (b.len(), b.len()),
        });
        out.insert(body[p..eq].to_string(), body[q..vend].to_string());
        if mend >= b.len() {
            break;
        }
        p = mend;
    }
    Ok(out)
}

fn decl_of(d: &HashMap<String, String>) -> Decl {
    Decl::from_strs(
        d.get("Type").map(String::as_str),
        d.get("Number").map(String::as_str),
    )
}

impl Header {
    /// Parse the header lines (each `rstrip`ped, `#`-prefixed).
    pub fn parse(lines: &[String]) -> Result<Header> {
        let mut h = Header {
            info: vec![None; INFO_KEYS.len()],
            format: vec![None; FMT_KEYS.len()],
            ..Header::default()
        };
        let mut contig_pos: HashMap<String, usize> = HashMap::new();
        for line in lines {
            if let Some(rest) = line.strip_prefix("##contig=") {
                let d = parse_kv_line(rest)?;
                let Some(id) = d.get("ID") else {
                    bail!("##contig without ID: {line}");
                };
                let len = match d.get("length") {
                    Some(l) => Some(l.trim().parse::<i64>()?),
                    None => None,
                };
                // An OrderedDict assignment: a repeated ID keeps its slot.
                match contig_pos.get(id) {
                    Some(&i) => h.contigs[i].1 = len,
                    None => {
                        contig_pos.insert(id.clone(), h.contigs.len());
                        h.contigs.push((id.clone(), len));
                    }
                }
            } else if let Some(rest) = line.strip_prefix("##INFO=") {
                let d = parse_kv_line(rest)?;
                let Some(id) = d.get("ID") else {
                    bail!("##INFO without ID: {line}");
                };
                if let Some(i) = INFO_KEYS.iter().position(|k| k == id) {
                    h.info[i] = Some(decl_of(&d));
                }
            } else if let Some(rest) = line.strip_prefix("##FORMAT=") {
                let d = parse_kv_line(rest)?;
                let Some(id) = d.get("ID") else {
                    bail!("##FORMAT without ID: {line}");
                };
                if let Some(i) = FMT_KEYS.iter().position(|k| k == id) {
                    h.format[i] = Some(decl_of(&d));
                }
            } else if line.starts_with("#CHROM") {
                h.samples = line[1..].split('\t').skip(9).map(str::to_string).collect();
            }
        }
        Ok(h)
    }

    pub fn contig_length(&self, chrom: &str) -> Option<i64> {
        self.contigs
            .iter()
            .find(|(c, _)| c == chrom)
            .and_then(|(_, l)| *l)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::value::Ty;

    #[test]
    fn kv_line() {
        let d = parse_kv_line("<ID=AD,Number=R,Type=Integer,Description=\"Allelic depths, a, b\">")
            .unwrap();
        assert_eq!(d["ID"], "AD");
        assert_eq!(d["Number"], "R");
        assert_eq!(d["Type"], "Integer");
        assert_eq!(d["Description"], "\"Allelic depths, a, b\"");
        let d = parse_kv_line("<ID=chr21,length=46709983>").unwrap();
        assert_eq!(d["length"], "46709983");
    }

    #[test]
    fn header_decls() {
        let lines: Vec<String> = [
            "##fileformat=VCFv4.2",
            "##contig=<ID=chr21,length=46709983>",
            "##contig=<ID=chr22,length=50818468>",
            "##INFO=<ID=END,Number=1,Type=Integer,Description=\"x\">",
            "##INFO=<ID=MQ,Number=1,Type=Float,Description=\"x\">",
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"x\">",
            "##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"x\">",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        let h = Header::parse(&lines).unwrap();
        assert_eq!(h.samples, vec!["S1", "S2"]);
        assert_eq!(h.contig_length("chr22"), Some(50818468));
        assert_eq!(h.info[1].as_ref().unwrap().ty, Some(Ty::Integer));
        assert_eq!(h.info[0], None); // DP undeclared
        assert_eq!(h.format[5].as_ref().unwrap().number.as_deref(), Some("G"));
    }
}
