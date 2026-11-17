// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The genotype matrix (`batch_genotypes`): `genotype_quality`,
//! `parse_gt`, `_star_ends` and `GenotypeRowBuilder` (originally a port
//! of the Python ingest worker's), and `encode_gt_blob` of
//! `mosaic/genotypes.py`.

use std::rc::Rc;

use anyhow::{Result, bail};

use crate::pyops::{PyErr, pv, py_len, to_int, truthy};
use crate::record::Record;
use crate::record::is_ref_block;
use crate::ref_blocks::DelState;
use crate::value::Value;
use mosaic_core::alleles::clen;

pub const GT_NO_CALL: i64 = -1;
pub const GQ_UNKNOWN: i64 = 255;
pub const GQ_CAP: i64 = 99;
pub const STAR_ALLELE: &str = "*";
pub const MAX_SITE_ALLELES: usize = 32767;
pub const GT_NARROW_MAX: i64 = 127;

/// Ref-block classes (`RB_OTHER`, `RB_HOM_REF`, `RB_NO_CALL`).
pub const RB_OTHER: u8 = 0;
pub const RB_HOM_REF: u8 = 1;
pub const RB_NO_CALL: u8 = 2;

/// FORMAT key indices of `Record::s0` (`record::FMT_KEYS`).
pub const F_GT: usize = 0;
pub const F_AD: usize = 1;
pub const F_DP: usize = 2;
pub const F_MIN_DP: usize = 3;
pub const F_SB: usize = 4;
pub const F_PL: usize = 5;
pub const F_F1R2: usize = 6;
pub const F_F2R1: usize = 7;
pub const F_GQ: usize = 8;

/// A sample's FORMAT values (the first sample's dict).
pub type Sample = [Option<Value>; 9];

/// `v.samples[0] if v.samples else None` (the list's truthiness).
pub fn samples0(v: &Record) -> Option<&Sample> {
    v.s0.as_ref()
}

/// The same, then `if sample:` (the dict's truthiness): None for an empty
/// dict too.
pub fn dict0(v: &Record) -> Option<&Sample> {
    match &v.s0 {
        Some(s) if v.s0_nonempty => Some(s),
        _ => None,
    }
}

/// `genotype_quality(sample)`: FORMAT/GQ, else PL's second-smallest minus
/// smallest, else GQ_UNKNOWN; clamped to 0..99. `sample` is the truthy
/// dict (None for a missing or empty one).
pub fn genotype_quality(sample: Option<&Sample>) -> Result<i64, PyErr> {
    let Some(s) = sample else {
        return Ok(GQ_UNKNOWN);
    };
    let gq_v: Value;
    let gq: Option<&Value> = match pv(&s[F_GQ]) {
        Some(g) => Some(g),
        None => {
            let pl = pv(&s[F_PL]);
            if !truthy(pl) || py_len(pl)? < 2 {
                return Ok(GQ_UNKNOWN);
            }
            // sorted(pl); lo[1] - lo[0] (a TypeError: unknown).
            gq_v = match pl.expect("truthy") {
                Value::Ints(x) => {
                    let mut lo = x.clone();
                    lo.sort_unstable();
                    Value::Int(lo[1] - lo[0])
                }
                Value::Floats(x) => {
                    let mut lo = x.clone();
                    lo.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                    Value::Float(lo[1] - lo[0])
                }
                Value::Bools(x) => {
                    let mut lo: Vec<i64> = x.iter().map(|&b| i64::from(b)).collect();
                    lo.sort_unstable();
                    Value::Int(lo[1] - lo[0])
                }
                _ => return Ok(GQ_UNKNOWN),
            };
            Some(&gq_v)
        }
    };
    let gq = match to_int(gq) {
        Ok(g) => g,
        Err(e) if e.is_value_or_type() => return Ok(GQ_UNKNOWN),
        Err(e) => return Err(e),
    };
    Ok(gq.clamp(0, GQ_CAP))
}

/// Split a GT string: `gt.replace("|", "/").split("/")`.
pub fn gt_fields_str(gt: &str) -> impl Iterator<Item = &str> {
    gt.split(['/', '|'])
}

/// `parse_gt(gt)`: `'0/1'` -> `[0, 1]`, `'./.'` -> `[None, None]`; None
/// for a missing or unparseable GT (ValueError / AttributeError).
pub fn parse_gt(gt: Option<&Value>) -> Result<Option<Vec<Option<i64>>>, PyErr> {
    if !truthy(gt) {
        return Ok(None);
    }
    let Some(Value::Str(s)) = gt else {
        return Ok(None); // AttributeError: no .replace
    };
    let mut out = Vec::new();
    for x in gt_fields_str(s) {
        if x == "." {
            out.push(None);
        } else {
            match to_int(Some(&Value::Str(x.to_string()))) {
                Ok(i) => out.push(Some(i)),
                Err(PyErr::Value(_)) => return Ok(None),
                Err(e) => return Err(e),
            }
        }
    }
    Ok(Some(out))
}

/// `_star_ends(v, gt_fields, pos_1based)`.
pub fn star_ends(v: &Record, gt_fields: &[Option<i64>], pos_1based: i64) -> (i64, i64) {
    let n_alt = v.alt.len() as i64;
    let lref = clen(&v.ref_) as i64;
    let mut ends: Vec<i64> = Vec::with_capacity(2);
    for x in gt_fields.iter().take(2) {
        let mut e = 0;
        if let Some(x) = *x
            && 0 < x
            && x <= n_alt
        {
            let alt = &v.alt[(x - 1) as usize];
            if alt != "<NON_REF>" && (clen(alt) as i64) < lref {
                e = pos_1based + lref - clen(alt) as i64;
            }
        }
        ends.push(e);
    }
    match ends.len() {
        0 => (0, 0),
        1 => (ends[0], -1),
        _ => (ends[0], ends[1]),
    }
}

/// `encode_gt_blob(codes, wide)`.
pub fn encode_gt_blob(codes: &[i64], wide: bool) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(codes.len() * if wide { 2 } else { 1 });
    for &c in codes {
        if wide {
            let c = i16::try_from(c).map_err(|_| anyhow::anyhow!("gt code {c} beyond int16"))?;
            out.extend_from_slice(&c.to_le_bytes());
        } else {
            let c = i8::try_from(c).map_err(|_| anyhow::anyhow!("gt code {c} beyond int8"))?;
            out.push(c as u8);
        }
    }
    Ok(out)
}

/// One `batch_genotypes` row (before the batch / chromosome columns).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BgtRow {
    pub pos: i64,
    pub n_slots: i64,
    pub alleles: Vec<String>,
    pub gt_1: Vec<u8>,
    pub gt_2: Vec<u8>,
    pub gq: Vec<u8>,
}

/// The cached `(gt_1, gt_2, gq, class)` of a stream's last record.
pub type BlockInfo = (i64, i64, i64, u8);

const NO_INFO: BlockInfo = (GT_NO_CALL, GT_NO_CALL, GQ_UNKNOWN, RB_OTHER);

/// `GenotypeRowBuilder`.
pub struct GenotypeRowBuilder {
    slot_of_k: Vec<i64>,
    pub n_slots: usize,
    pub c_idx: i64,
    pub batch_id: i64,
    rec: Vec<Option<Rc<Record>>>,
    info: Vec<BlockInfo>,
    pub rows: Vec<BgtRow>,
    pub pending: bool,
    pos: i64,
    g1: Vec<i64>,
    g2: Vec<i64>,
    gq: Vec<i64>,
    s1: Vec<Option<String>>,
    s2: Vec<Option<String>>,
    rule1: Vec<bool>,
    /// `gt_matrix_conflicts`.
    pub conflicts: u64,
}

impl GenotypeRowBuilder {
    pub fn new(
        sample_ids: &[Option<i64>],
        slot_sample_ids: &[i64],
        c_idx: i64,
        batch_id: i64,
    ) -> GenotypeRowBuilder {
        // {sid: i}: a repeated id keeps its last index, as the dict does.
        let mut slot_of_sid = std::collections::HashMap::new();
        for (i, &sid) in slot_sample_ids.iter().enumerate() {
            slot_of_sid.insert(sid, i as i64);
        }
        let slot_of_k = sample_ids
            .iter()
            .map(|s| s.and_then(|s| slot_of_sid.get(&s).copied()).unwrap_or(-1))
            .collect();
        let n = sample_ids.len();
        GenotypeRowBuilder {
            slot_of_k,
            n_slots: slot_sample_ids.len(),
            c_idx,
            batch_id,
            rec: vec![None; n],
            info: vec![NO_INFO; n],
            rows: Vec::new(),
            pending: false,
            pos: 0,
            g1: Vec::new(),
            g2: Vec::new(),
            gq: Vec::new(),
            s1: Vec::new(),
            s2: Vec::new(),
            rule1: Vec::new(),
            conflicts: 0,
        }
    }

    /// `block_info(k, v)`, cached by record identity.
    pub fn block_info(&mut self, k: usize, v: &Rc<Record>) -> Result<BlockInfo, PyErr> {
        if let Some(r) = &self.rec[k]
            && Rc::ptr_eq(r, v)
        {
            return Ok(self.info[k]);
        }
        let mut info = NO_INFO;
        if is_ref_block(v) {
            let sample = samples0(v);
            let gt = sample.and_then(|s| pv(&s[F_GT]));
            let gq_of = || genotype_quality(dict0(v));
            if let Some(Value::Str(g)) = gt {
                if g == "0/0" || g == "0|0" {
                    info = (0, 0, gq_of()?, RB_HOM_REF);
                } else if g == "0" {
                    info = (0, GT_NO_CALL, gq_of()?, RB_HOM_REF);
                } else if g == "./." || g == ".|." || g == "." {
                    info = (GT_NO_CALL, GT_NO_CALL, gq_of()?, RB_NO_CALL);
                }
            }
        }
        self.rec[k] = Some(v.clone());
        self.info[k] = info;
        Ok(info)
    }

    /// `begin_site(pos_1based, grp, sample_del_state)`.
    pub fn begin_site(
        &mut self,
        pos_1based: i64,
        grp: &[Option<Rc<Record>>],
        del_state: &[Option<DelState>],
    ) -> Result<(), PyErr> {
        let n = self.n_slots;
        self.g1 = vec![GT_NO_CALL; n];
        self.g2 = vec![GT_NO_CALL; n];
        self.gq = vec![GQ_UNKNOWN; n];
        self.s1 = vec![None; n];
        self.s2 = vec![None; n];
        self.rule1 = vec![false; n];
        for (k, v) in grp.iter().enumerate() {
            let s = self.slot_of_k.get(k).copied().unwrap_or(-1);
            if s < 0 {
                continue;
            }
            let s = s as usize;
            if let Some(Some(d)) = del_state.get(k)
                && (d.e1 >= pos_1based || d.e2 >= pos_1based)
            {
                self.gq[s] = d.gq;
                if d.e2 < 0 {
                    self.s1[s] = Some(STAR_ALLELE.to_string());
                } else if d.e1 >= pos_1based && d.e2 >= pos_1based {
                    self.s1[s] = Some(STAR_ALLELE.to_string());
                    self.s2[s] = Some(STAR_ALLELE.to_string());
                } else {
                    self.g1[s] = 0;
                    self.s2[s] = Some(STAR_ALLELE.to_string());
                }
                continue;
            }
            let Some(v) = v else {
                continue;
            };
            let t = self.block_info(k, v)?;
            self.g1[s] = t.0;
            self.g2[s] = t.1;
            self.gq[s] = t.2;
        }
        self.pending = true;
        self.pos = pos_1based;
        Ok(())
    }

    /// `add_record(k, v, fmts, gt_fields)`: rule 1.
    pub fn add_record(
        &mut self,
        k: usize,
        v: &Record,
        fmts: &[String],
        gt_fields: Option<&[Option<i64>]>,
    ) -> Result<(), PyErr> {
        let s = self.slot_of_k.get(k).copied().unwrap_or(-1);
        if s < 0 {
            return Ok(());
        }
        let s = s as usize;
        if self.rule1[s] {
            self.conflicts += 1;
            return Ok(());
        }
        self.rule1[s] = true;
        self.s1[s] = None;
        self.s2[s] = None;
        let n_fmt = fmts.len() as i64;
        for i in 0..2 {
            let x = match gt_fields {
                Some(g) if !g.is_empty() && i < g.len() => g[i],
                _ => None,
            };
            let (g, strs) = if i == 0 {
                (&mut self.g1, &mut self.s1)
            } else {
                (&mut self.g2, &mut self.s2)
            };
            match x {
                None => g[s] = GT_NO_CALL,
                Some(x) if x < 0 || x >= n_fmt => g[s] = GT_NO_CALL,
                Some(0) => g[s] = 0,
                Some(x) => strs[s] = Some(fmts[x as usize].clone()),
            }
        }
        self.gq[s] = genotype_quality(dict0(v))?;
        Ok(())
    }

    /// `end_site(site_alleles)`.
    pub fn end_site(&mut self, site_alleles: &[String]) -> Result<()> {
        let star = self
            .s1
            .iter()
            .chain(self.s2.iter())
            .any(|a| a.as_deref() == Some(STAR_ALLELE));
        let mut alleles: Vec<String> = Vec::with_capacity(site_alleles.len() + 1);
        alleles.push(site_alleles[0].clone());
        alleles.extend(
            site_alleles[1..]
                .iter()
                .filter(|a| *a != STAR_ALLELE)
                .cloned(),
        );
        if star {
            alleles.push(STAR_ALLELE.to_string());
        }
        if alleles.len() > MAX_SITE_ALLELES {
            bail!(
                "batch {}, site {}:{}: {} alleles, more than the {} a \
                 genotype-matrix slot can index",
                self.batch_id,
                self.c_idx,
                self.pos,
                alleles.len(),
                MAX_SITE_ALLELES
            );
        }
        // {a: i}: the last index of a repeated string.
        let mut idx: std::collections::HashMap<&str, i64> =
            std::collections::HashMap::with_capacity(alleles.len());
        for (i, a) in alleles.iter().enumerate() {
            idx.insert(a.as_str(), i as i64);
        }
        let mut wide = false;
        for (strs, g) in [(&self.s1, &mut self.g1), (&self.s2, &mut self.g2)] {
            for (s, a) in strs.iter().enumerate() {
                if let Some(a) = a {
                    let code = *idx.get(a.as_str()).ok_or_else(|| {
                        anyhow::anyhow!("KeyError: allele {a} not in the site's list")
                    })?;
                    if code > GT_NARROW_MAX {
                        wide = true;
                    }
                    g[s] = code;
                }
            }
        }
        let gq: Vec<u8> = self
            .gq
            .iter()
            .map(|&q| u8::try_from(q).map_err(|_| anyhow::anyhow!("gq {q} beyond a byte")))
            .collect::<Result<_>>()?;
        self.rows.push(BgtRow {
            pos: self.pos,
            n_slots: self.n_slots as i64,
            alleles,
            gt_1: encode_gt_blob(&self.g1, wide)?,
            gt_2: encode_gt_blob(&self.g2, wide)?,
            gq,
        });
        self.pending = false;
        Ok(())
    }
}

#[cfg(test)]
pub mod tests {
    //! `tests/unit/test_genotype_builder.py`'s builder cases.
    use super::*;
    use crate::header::Header;
    use crate::record::Tokenizer;

    pub fn hdr() -> Header {
        let lines: Vec<String> = [
            "##INFO=<ID=END,Number=1,Type=Integer,Description=\"x\">",
            "##INFO=<ID=DP,Number=1,Type=Integer,Description=\"x\">",
            "##INFO=<ID=MQ,Number=1,Type=Float,Description=\"x\">",
            "##INFO=<ID=MQSQ,Number=2,Type=Float,Description=\"x\">",
            "##INFO=<ID=RAW_MQandDP,Number=2,Type=Float,Description=\"x\">",
            "##INFO=<ID=RAW_MQ,Number=1,Type=Float,Description=\"x\">",
            "##INFO=<ID=GTC,Number=G,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"x\">",
            "##FORMAT=<ID=AD,Number=R,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=DP,Number=1,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=MIN_DP,Number=1,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=GQ,Number=1,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=PL,Number=G,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=SB,Number=4,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=F1R2,Number=R,Type=Integer,Description=\"x\">",
            "##FORMAT=<ID=F2R1,Number=R,Type=Integer,Description=\"x\">",
            "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1",
        ]
        .iter()
        .map(|s| s.to_string())
        .collect();
        Header::parse(&lines).unwrap()
    }

    /// A record from a VCF line (1-based POS).
    pub fn rec(line: &str) -> Rc<Record> {
        Rc::new(
            Tokenizer::default()
                .parse(line.as_bytes(), &hdr())
                .unwrap()
                .1,
        )
    }

    /// A one-sample record: `chr21 <pos> . <ref> <alt> 50 PASS <info>
    /// <fmt> <sample>`.
    pub fn var(pos: i64, r: &str, alt: &str, info: &str, fmt: &str, sample: &str) -> Rc<Record> {
        rec(&format!(
            "chr21\t{pos}\t.\t{r}\t{alt}\t50\tPASS\t{info}\t{fmt}\t{sample}"
        ))
    }

    pub fn block(pos: i64, end: i64, sample: &str) -> Rc<Record> {
        rec(&format!(
            "chr21\t{pos}\t.\tA\t<NON_REF>\t.\t.\tEND={end}\tGT:AD:DP:GQ:MIN_DP:PL\t{sample}"
        ))
    }

    fn s(fmt: &str, vals: &str) -> Rc<Record> {
        var(10, "A", "G", ".", fmt, vals)
    }

    #[test]
    fn gq_rule() {
        let gq = |r: &Rc<Record>| genotype_quality(dict0(r)).unwrap();
        assert_eq!(gq(&s("GT:GQ", "0/1:37")), 37);
        assert_eq!(gq(&s("GT:GQ", "0/1:150")), 99);
        assert_eq!(gq(&s("GT:GQ", "0/1:-3")), 0);
        assert_eq!(gq(&s("GT:PL", "0/1:30,0,200")), 30);
        assert_eq!(gq(&s("GT:PL", "0/1:0")), GQ_UNKNOWN);
        assert_eq!(gq(&s("GT", "0/1")), GQ_UNKNOWN);
        assert_eq!(gq(&s("GT:GQ:PL", "0/1:.:5,0,9")), 5);
        assert_eq!(genotype_quality(None).unwrap(), GQ_UNKNOWN);
    }

    #[test]
    fn parse_gt_keeps_ploidy_and_missing() {
        let p = |g: &str| parse_gt(Some(&Value::Str(g.into()))).unwrap();
        assert_eq!(p("0/1"), Some(vec![Some(0), Some(1)]));
        assert_eq!(p("1|1"), Some(vec![Some(1), Some(1)]));
        assert_eq!(p("./."), Some(vec![None, None]));
        assert_eq!(p("./1"), Some(vec![None, Some(1)]));
        assert_eq!(p("1"), Some(vec![Some(1)]));
        assert_eq!(p("a/1"), None);
        assert_eq!(parse_gt(None).unwrap(), None);
        assert_eq!(parse_gt(Some(&Value::Str(String::new()))).unwrap(), None);
    }

    #[test]
    fn star_ends_per_slot() {
        let v = var(100, "ATTT", "A,AT,G", ".", "GT", "1/2");
        assert_eq!(star_ends(&v, &[Some(1), Some(2)], 100), (103, 102));
        assert_eq!(star_ends(&v, &[Some(3), Some(0)], 100), (103, 0));
        assert_eq!(star_ends(&v, &[Some(1)], 100), (103, -1));
        assert_eq!(star_ends(&v, &[], 100), (0, 0));
    }

    fn fmts(xs: &[&str]) -> Vec<String> {
        xs.iter().map(|s| s.to_string()).collect()
    }

    #[test]
    fn slot_rules_and_rule1_by_string() {
        // Streams: 0 -> sid 7, 1 -> sid 3, 2 -> no id. Slots ascending.
        let sids = [Some(7), Some(3), None];
        let mut b = GenotypeRowBuilder::new(&sids, &[3, 7], 21, 5);
        let hr = block(90, 200, "0/0:10,0:10:40:9:0,30,300");
        let nc = block(90, 200, "./.:0,0:0:0:0:0,0,0");
        let grp = vec![Some(hr.clone()), Some(nc.clone()), None];
        b.begin_site(100, &grp, &[None, None, None]).unwrap();
        // Stream 0 has its own record: rule 1 maps 'xG' by string.
        let v = var(100, "A", "C,G", ".", "GT:GQ", "2/0:12");
        b.add_record(0, &v, &fmts(&["A", "xC", "xG"]), Some(&[Some(2), Some(0)]))
            .unwrap();
        // A second record of slot 1 (stream 0) loses and is counted.
        b.add_record(0, &v, &fmts(&["A", "xC"]), Some(&[Some(1), Some(1)]))
            .unwrap();
        assert_eq!(b.conflicts, 1);
        b.end_site(&fmts(&["A", "xT", "xG", "*"])).unwrap();
        let r = &b.rows[0];
        assert_eq!(r.alleles, fmts(&["A", "xT", "xG"])); // '*' unused: dropped
        // slot 0 = sid 3 = stream 1 (./. block); slot 1 = sid 7 (rule 1).
        assert_eq!(r.gt_1, vec![0xFF, 2]);
        assert_eq!(r.gt_2, vec![0xFF, 0]);
        assert_eq!(r.gq, vec![0, 12]);
    }

    #[test]
    fn deletion_star_and_haploid() {
        let sids = [Some(1), Some(2)];
        let mut b = GenotypeRowBuilder::new(&sids, &[1, 2], 0, 1);
        let dv = var(100, "ATT", "A", ".", "GT", "0/1");
        let mk = |e1, e2| {
            Some(DelState {
                del_end: 102,
                del_dp: 5,
                e1,
                e2,
                gq: 33,
                spans: Rc::new(crate::ref_blocks::DelSpans::new((1, 0, 100), dv.clone())),
            })
        };
        // Slot 0: het deletion (0, *); slot 1: haploid deletion (*, -).
        let ds = vec![mk(0, 102), mk(102, -1)];
        b.begin_site(101, &[None, None], &ds).unwrap();
        b.end_site(&fmts(&["T", "xC"])).unwrap();
        let r = &b.rows[0];
        assert_eq!(r.alleles, fmts(&["T", "xC", "*"]));
        assert_eq!(r.gt_1, vec![0, 2]);
        assert_eq!(r.gt_2, vec![2, 0xFF]);
        assert_eq!(r.gq, vec![33, 33]);
    }

    #[test]
    fn blob_widths_and_the_limit() {
        // 129 alleles (code 128 used): two bytes per slot.
        let mut site: Vec<String> = vec!["A".into()];
        for i in 0..128 {
            site.push(format!("x{i}"));
        }
        let mut b = GenotypeRowBuilder::new(&[Some(1)], &[1], 0, 1);
        b.begin_site(10, &[None], &[None]).unwrap();
        let v = var(10, "A", "G", ".", "GT", "1/1");
        b.add_record(0, &v, &site, Some(&[Some(128), Some(1)]))
            .unwrap();
        b.end_site(&site).unwrap();
        assert_eq!(b.rows[0].gt_1, vec![128, 0]);
        assert_eq!(b.rows[0].gt_2, vec![1, 0]);
        // 128 alleles, code 127: one byte.
        let site128 = &site[..128];
        b.begin_site(11, &[None], &[None]).unwrap();
        b.add_record(0, &v, site128, Some(&[Some(127), Some(0)]))
            .unwrap();
        b.end_site(site128).unwrap();
        assert_eq!(b.rows[1].gt_1, vec![127]);
        // Above MAX_SITE_ALLELES: an error naming the site.
        let big: Vec<String> = (0..=MAX_SITE_ALLELES).map(|i| format!("x{i}")).collect();
        b.begin_site(12, &[None], &[None]).unwrap();
        let e = b.end_site(&big).unwrap_err().to_string();
        assert!(e.contains("site 0:12"), "{e}");
        assert!(e.contains("32768 alleles"), "{e}");
    }

    // ── K6 ports of tests/unit/test_genotype_builder.py ─────────────────
    // Python's `Rec(pos, ...)` takes a 0-based pos; these helpers take it
    // too and write POS = pos + 1.

    /// A one-sample variant record at 0-based `pos0`: FORMAT GT[:GQ].
    fn vrec(pos0: i64, r: &str, alts: &[&str], gt: &str, gq: Option<i64>) -> Rc<Record> {
        let (f, s) = match gq {
            Some(q) => ("GT:GQ".to_string(), format!("{gt}:{q}")),
            None => ("GT".to_string(), gt.to_string()),
        };
        var(pos0 + 1, r, &alts.join(","), ".", &f, &s)
    }

    /// Python's `block(pos, end_1based, gt, gq)`: a `<NON_REF>` block.
    fn blk(pos0: i64, end_1b: i64, gt: &str, gq: Option<i64>) -> Rc<Record> {
        let (f, s) = match gq {
            Some(q) => ("GT:GQ".to_string(), format!("{gt}:{q}")),
            None => ("GT".to_string(), gt.to_string()),
        };
        rec(&format!(
            "chr21\t{}\t.\tA\t<NON_REF>\t.\t.\tEND={end_1b}\t{f}\t{s}",
            pos0 + 1
        ))
    }

    /// `fmts_of(v)`: REF's first base, then each ALT formatted.
    fn fmts_of(v: &Record) -> Vec<String> {
        let mut out = vec![v.ref_.chars().next().unwrap().to_string()];
        out.extend(
            v.alt
                .iter()
                .map(|a| mosaic_core::alleles::format_alt_allele(&v.ref_, a)),
        );
        out
    }

    fn gt_of(v: &Record) -> Option<Vec<Option<i64>>> {
        parse_gt(pv(&v.s0.as_ref().unwrap()[F_GT])).unwrap()
    }

    /// `decode_gt_blob(blob, n)`: one or two bytes per slot.
    fn dec(blob: &[u8], n: usize) -> Vec<i64> {
        if blob.len() == n {
            blob.iter().map(|&b| i64::from(b as i8)).collect()
        } else {
            assert_eq!(blob.len(), 2 * n);
            blob.chunks(2)
                .map(|c| i64::from(i16::from_le_bytes([c[0], c[1]])))
                .collect()
        }
    }

    type Decoded = (i64, i64, Vec<String>, Vec<i64>, Vec<i64>, Vec<i64>);

    fn decode(r: &BgtRow) -> Decoded {
        let n = r.n_slots as usize;
        (
            r.pos,
            r.n_slots,
            r.alleles.clone(),
            dec(&r.gt_1, n),
            dec(&r.gt_2, n),
            r.gq.iter().map(|&q| i64::from(q)).collect(),
        )
    }

    /// `build_site(...)`: one site through the builder; (row, conflicts).
    fn build_site(
        grp: &[Option<Rc<Record>>],
        records: &[(usize, Rc<Record>)],
        sids: &[i64],
        site_pos0: i64,
        site_alleles: &[&str],
        del_state: &[Option<DelState>],
    ) -> (BgtRow, u64) {
        let mut slots = sids.to_vec();
        slots.sort_unstable();
        let ids: Vec<Option<i64>> = sids.iter().map(|&s| Some(s)).collect();
        let mut b = GenotypeRowBuilder::new(&ids, &slots, 0, 7);
        b.begin_site(site_pos0 + 1, grp, del_state).unwrap();
        for (k, v) in records {
            b.add_record(*k, v, &fmts_of(v), gt_of(v).as_deref())
                .unwrap();
        }
        b.end_site(&fmts(site_alleles)).unwrap();
        assert_eq!(b.rows.len(), 1);
        (b.rows[0].clone(), b.conflicts)
    }

    fn del_state_of(grp: &[Option<Rc<Record>>], sids: &[i64]) -> Vec<Option<DelState>> {
        let ids: Vec<Option<i64>> = sids.iter().map(|&s| Some(s)).collect();
        let mut st = vec![None; grp.len()];
        crate::ref_blocks::update_sample_del_state(grp, &ids, 0, &mut st).unwrap();
        st
    }

    // Replaces tests/unit/test_genotype_builder.py::test_genotype_quality (K6).
    #[test]
    fn genotype_quality_python_cases() {
        let gq = |r: &Rc<Record>| genotype_quality(dict0(r)).unwrap();
        assert_eq!(gq(&s("GQ:PL", "30:0,3,45")), 30); // GQ wins over PL
        assert_eq!(gq(&s("GQ", "120")), 99);
        assert_eq!(gq(&s("GQ", "-3")), 0);
        assert_eq!(gq(&s("PL", "0,3,45")), 3);
        assert_eq!(gq(&s("PL", "37,0,1,70,6,70")), 1);
        assert_eq!(gq(&s("PL", "0,200,300")), 99); // derived, capped
        assert_eq!(gq(&s("PL", "5")), GQ_UNKNOWN);
        assert_eq!(gq(&s("GT", "0/0")), GQ_UNKNOWN);
        assert_eq!(genotype_quality(None).unwrap(), GQ_UNKNOWN);
    }

    // Replaces tests/unit/test_genotype_builder.py::test_parse_gt_keeps_ploidy_and_missing (K6).
    #[test]
    fn parse_gt_python_cases() {
        let p = |g: &str| parse_gt(Some(&Value::Str(g.into()))).unwrap();
        assert_eq!(p("0/1"), Some(vec![Some(0), Some(1)]));
        assert_eq!(p("1|1"), Some(vec![Some(1), Some(1)]));
        assert_eq!(p("./."), Some(vec![None, None]));
        assert_eq!(p("./1"), Some(vec![None, Some(1)]));
        assert_eq!(p("1"), Some(vec![Some(1)]));
        assert_eq!(parse_gt(None).unwrap(), None);
        assert_eq!(p("x/y"), None);
    }

    // Replaces tests/unit/test_genotype_builder.py::test_slot_rules_table (K6).
    #[test]
    fn slot_rules_table() {
        let site = 1000; // 0-based; 1-based 1001
        let snp = vrec(site, "A", &["T", "<NON_REF>"], "0/1", Some(40)); // rule 1
        let het_del = vrec(995, "AGGGGGGG", &["A", "<NON_REF>"], "0/1", Some(31));
        let hom_del = vrec(996, "AGGGGGGG", &["A", "<NON_REF>"], "1/1", Some(32));
        let hom_ref = blk(990, 1100, "0/0", Some(25)); // rule 3
        let hap_ref = blk(992, 1100, "0", Some(27)); // rule 3, haploid
        let no_call = blk(999, 1005, "./.", Some(0)); // rule 4
        let carried = vrec(998, "ACG", &["TTT", "<NON_REF>"], "0/1", None); // rule 5
        let sids = [10, 11, 12, 13, 14, 15, 16, 17];
        // The deletion state, from the real tracker on the deletions.
        let del_state = del_state_of(
            &[
                None,
                Some(het_del),
                Some(hom_del),
                None,
                None,
                None,
                None,
                None,
            ],
            &sids,
        );
        // At the site the deletion samples' entries are their ref blocks;
        // rule 2 must win over rule 3.
        let grp = vec![
            Some(snp.clone()),
            Some(blk(996, 1050, "0/0", Some(60))),
            Some(blk(997, 1050, "0/0", Some(61))),
            Some(hom_ref),
            Some(no_call),
            None,
            Some(carried),
            Some(hap_ref),
        ];
        let (row, conflicts) = build_site(
            &grp,
            &[(0, snp)],
            &sids,
            site,
            &["A", "xT", "*"],
            &del_state,
        );
        let (pos, n, alleles, g1, g2, gq) = decode(&row);
        assert_eq!((pos, n), (site + 1, 8));
        assert_eq!(alleles, fmts(&["A", "xT", "*"])); // '*' kept
        assert_eq!(g1, vec![0, 0, 2, 0, -1, -1, -1, 0]);
        assert_eq!(g2, vec![1, 2, 2, 0, -1, -1, -1, -1]);
        assert_eq!(gq, vec![40, 31, 32, 25, 0, 255, 255, 27]);
        assert_eq!(conflicts, 0);
    }

    // Replaces tests/unit/test_genotype_builder.py::test_deletion_coverage_is_per_allele (K6).
    #[test]
    fn deletion_coverage_is_per_allele() {
        // 1/2 at GAAA -> G (3 bp) and GAA (1 bp).
        let r = vrec(100, "GAAA", &["G", "GAA", "<NON_REF>"], "1/2", Some(9));
        let del_state = del_state_of(&[Some(r)], &[1]);
        for (site, want) in [
            (101, Some((vec![2], vec![2]))),
            (102, Some((vec![0], vec![2]))),
            (104, None),
        ] {
            let (row, _) = build_site(
                &[Some(blk(site, site + 5, "0/0", None))],
                &[],
                &[1],
                site,
                &["A", "xC"],
                &del_state,
            );
            let (_, _, alleles, g1, g2, _) = decode(&row);
            match want {
                None => {
                    // Past the deletion: the block; no slot uses '*'.
                    assert_eq!(alleles, fmts(&["A", "xC"]), "{site}");
                    assert_eq!((g1, g2), (vec![0], vec![0]), "{site}");
                }
                Some(w) => {
                    assert_eq!(alleles, fmts(&["A", "xC", "*"]), "{site}");
                    assert_eq!((g1, g2), w, "{site}");
                }
            }
        }
    }

    // Replaces tests/unit/test_genotype_builder.py::test_haploid_deletion_star (K6).
    #[test]
    fn haploid_deletion_star() {
        let r = vrec(100, "ATT", &["A", "<NON_REF>"], "1", Some(50));
        let del_state = del_state_of(&[Some(r)], &[1]);
        let (row, _) = build_site(&[None], &[], &[1], 101, &["A", "xC"], &del_state);
        let (_, _, alleles, g1, g2, gq) = decode(&row);
        assert_eq!(alleles, fmts(&["A", "xC", "*"]));
        assert_eq!((g1, g2, gq), (vec![2], vec![-1], vec![50]));
    }

    // Replaces tests/unit/test_genotype_builder.py::test_slot_order_is_ascending_sample_id (K6).
    #[test]
    fn slot_order_is_ascending_sample_id() {
        let site = 50;
        let r0 = vrec(site, "C", &["T", "<NON_REF>"], "1/1", Some(11)); // sample 30
        let r1 = vrec(site, "C", &["G", "<NON_REF>"], "0/1", Some(12)); // sample 10
        let grp = vec![
            Some(r0.clone()),
            Some(r1.clone()),
            Some(blk(40, 60, "0/0", Some(13))), // sample 20 hom-ref
        ];
        let (row, _) = build_site(
            &grp,
            &[(0, r0), (1, r1)],
            &[30, 10, 20],
            site,
            &["C", "xT", "*", "xG"],
            &[None, None, None],
        );
        let (_, _, alleles, g1, g2, gq) = decode(&row);
        assert_eq!(alleles, fmts(&["C", "xT", "xG"])); // '*' dropped, order kept
        assert_eq!(
            (g1, g2, gq),
            (vec![0, 0, 1], vec![2, 0, 1], vec![12, 13, 11])
        );
    }

    // Replaces tests/unit/test_genotype_builder.py::test_rule1_maps_by_string_and_first_record_wins (K6).
    #[test]
    fn rule1_maps_by_string_and_first_record_wins() {
        let site = 200;
        let deletion = vrec(site, "AT", &["A", "<NON_REF>"], "0/1", Some(40));
        let snp = vrec(site, "A", &["T", "<NON_REF>"], "1/1", Some(41));
        let (row, conflicts) = build_site(
            &[Some(deletion.clone())],
            &[(0, deletion), (0, snp)],
            &[5],
            site,
            &["A", "xT", "*", "-AT"],
            &[None],
        );
        let (_, _, alleles, g1, g2, gq) = decode(&row);
        assert_eq!(alleles, fmts(&["A", "xT", "-AT"]));
        assert_eq!((g1, g2, gq), (vec![0], vec![2], vec![40]));
        assert_eq!(conflicts, 1);
    }

    /// `n` distinct insertion ALTs of REF `A`: `AC`, `ACC`, ...
    fn insertions(n: usize) -> Vec<String> {
        (0..n).map(|i| format!("A{}", "C".repeat(i + 1))).collect()
    }

    // Replaces tests/unit/test_genotype_builder.py::test_wide_blob_above_127_alleles (K6).
    #[test]
    fn wide_blob_above_127_alleles() {
        let alts = insertions(130);
        let mut all: Vec<&str> = alts.iter().map(String::as_str).collect();
        all.push("<NON_REF>");
        let r = vrec(10, "A", &all, "1/130", Some(30));
        let f = fmts_of(&r);
        let uniq: std::collections::HashSet<&String> = f.iter().collect();
        assert_eq!(uniq.len(), 132);
        let mut b = GenotypeRowBuilder::new(&[Some(1), Some(2), Some(3)], &[1, 2, 3], 0, 1);
        b.begin_site(
            11,
            &[Some(r.clone()), Some(blk(0, 50, "0/0", Some(20))), None],
            &[None, None, None],
        )
        .unwrap();
        b.add_record(0, &r, &f, Some(&[Some(1), Some(130)]))
            .unwrap();
        b.end_site(&f).unwrap();
        let row = &b.rows[0];
        assert_eq!((row.gt_1.len(), row.gt_2.len(), row.gq.len()), (6, 6, 3));
        let (_, _, alleles, g1, g2, gq) = decode(row);
        assert_eq!(alleles, f[..f.len() - 1].to_vec()); // '*' dropped
        assert_eq!(
            (g1, g2.clone(), gq),
            (vec![1, 0, -1], vec![130, 0, -1], vec![30, 20, 255])
        );
        assert_eq!(
            alleles[g2[0] as usize],
            mosaic_core::alleles::format_alt_allele("A", alts.last().unwrap())
        );
        assert_eq!(b.conflicts, 0);
    }

    // Replaces tests/unit/test_genotype_builder.py::test_narrow_blob_at_128_alleles (K6).
    #[test]
    fn narrow_blob_at_128_alleles() {
        let alts = insertions(127);
        let mut all: Vec<&str> = alts.iter().map(String::as_str).collect();
        all.push("<NON_REF>");
        let r = vrec(10, "A", &all, "127/127", None);
        let f = fmts_of(&r);
        let mut b = GenotypeRowBuilder::new(&[Some(1)], &[1], 0, 1);
        b.begin_site(11, &[Some(r.clone())], &[None]).unwrap();
        b.add_record(0, &r, &f, Some(&[Some(127), Some(127)]))
            .unwrap();
        b.end_site(&f).unwrap();
        let row = &b.rows[0];
        assert_eq!(row.alleles.len(), 128);
        assert_eq!((row.gt_1.len(), row.gt_2.len()), (1, 1));
        let (_, _, _, g1, g2, _) = decode(row);
        assert_eq!((g1, g2), (vec![127], vec![127]));
    }

    // Replaces tests/unit/test_genotype_builder.py::test_too_many_site_alleles_raises (K6).
    #[test]
    fn too_many_site_alleles_raises() {
        let mut site: Vec<String> = vec!["A".into()];
        site.extend((0..MAX_SITE_ALLELES).map(|i| format!("+{i}")));
        let mut b = GenotypeRowBuilder::new(&[Some(1)], &[1], 0, 1);
        b.begin_site(11, &[None], &[None]).unwrap();
        let e = b.end_site(&site).unwrap_err().to_string();
        assert!(e.contains("alleles"), "{e}");
        let mut b = GenotypeRowBuilder::new(&[Some(1)], &[1], 0, 1);
        b.begin_site(11, &[None], &[None]).unwrap();
        b.end_site(&site[..site.len() - 1]).unwrap(); // exactly the max: fine
    }

    // The worker's own blob encoder on test_genotype_builder.py's
    // test_blob_round_trip / test_wide_blob_round_trip codes (those tests
    // pin mosaic/genotypes.py's encode_gt_blob, which stays; K6).
    #[test]
    fn encode_gt_blob_bytes() {
        assert_eq!(
            encode_gt_blob(&[-1, 0, 1, 2, 127, -1], false).unwrap(),
            vec![0xFF, 0, 1, 2, 127, 0xFF]
        );
        assert!(encode_gt_blob(&[128], false).is_err());
        let w = encode_gt_blob(&[-1, 0, 128, 300, 32767, -1], true).unwrap();
        assert_eq!(w.len(), 12);
        assert_eq!(&w[..2], &[0xFF, 0xFF]);
        assert_eq!(&w[4..6], &[128, 0]);
        assert_eq!(&w[8..10], &[0xFF, 0x7F]);
        assert_eq!(dec(&w, 6), vec![-1, 0, 128, 300, 32767, -1]);
        assert!(encode_gt_blob(&[32768], true).is_err());
    }
}
