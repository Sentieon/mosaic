// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! Variant sites: `extract_site_info`, `quantize_gtc`, `parse_int_list`,
//! `collect_variant_group_data`, `merge_site_alleles` (originally a port
//! of the Python ingest worker's functions of those names), with Python's
//! float semantics (and `sum()`'s compensated summation,
//! `pyops::py_fsum_builtin`).

use std::collections::HashMap;
use std::rc::Rc;

use crate::matrix::{F_AD, F_DP, F_F1R2, F_F2R1, F_GT, F_PL, F_SB, dict0, samples0};
use crate::pyops::{
    PyErr, float_item, int_item, items, pv, py_fsum_builtin, py_len, to_float, to_int, truthy,
};
use crate::record::Record;
use crate::record::is_ref_block;
use crate::ref_blocks::{
    DelSpans, DelState, I_DP, I_GTC, I_MQ, I_MQSQ, I_RAW_MQ, I_RAW_MQANDDP, I_SB, gt_indices,
};
use crate::value::Value;
use mosaic_core::alleles::{clen, format_alt_allele};

/// `GTC_FIXED_BITS`.
pub const GTC_FIXED_BITS: u32 = 32;

/// The shard's bad-input counters (`parse_stats`).
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ParseStats {
    pub records_dropped: u64,
    pub site_info_parse_errors: u64,
    pub gt_parse_errors: u64,
    pub gt_matrix_conflicts: u64,
}

impl ParseStats {
    pub fn is_empty(&self) -> bool {
        self.records_dropped == 0
            && self.site_info_parse_errors == 0
            && self.gt_parse_errors == 0
            && self.gt_matrix_conflicts == 0
    }
}

/// One record's `extract_site_info` result: (qual, dp, mq, sb, gtc).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RecordSi {
    pub qual: f64,
    pub dp: i64,
    pub mq0: f64,
    pub mq1: i64,
    pub sb: [i64; 4],
    pub gtc: [f64; 3],
}

/// A site's summed site_info (gtc in fixed point).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SiteInfo {
    pub qual: f64,
    pub dp: i64,
    pub mq0: f64,
    pub mq1: i64,
    pub sb: [i64; 4],
    pub gtc: [i64; 3],
}

/// Run `f`; a ValueError / TypeError counts a parse error and stops (the
/// parts already added are kept, as in the Python `try` blocks).
fn guarded(stats: &mut ParseStats, f: impl FnOnce() -> Result<(), PyErr>) -> Result<(), PyErr> {
    match f() {
        Ok(()) => Ok(()),
        Err(e) if e.is_value_or_type() => {
            stats.site_info_parse_errors += 1;
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// `t is not None and len(t) >= n` (len's TypeError is not caught):
/// `Some(t)` when it holds.
fn len_at_least(t: Option<&Value>, n: usize) -> Result<Option<&Value>, PyErr> {
    match t {
        Some(v) if py_len(Some(v))? >= n => Ok(Some(v)),
        _ => Ok(None),
    }
}

/// `extract_site_info(variant, stats)`.
pub fn extract_site_info(v: &Record, stats: &mut ParseStats) -> Result<RecordSi, PyErr> {
    let mut qual = 0.0f64;
    let mut dp = 0i64;
    let mut mq0 = 0.0f64;
    let mut mq1 = 0i64;
    let mut sb = [0i64; 4];
    let mut gtc = [0.0f64; 3];
    let s0 = samples0(v);
    let zero = Value::Int(0);

    // DP and QUAL.
    let info_dp = pv(&v.info[I_DP]);
    let t = if truthy(info_dp) {
        info_dp
    } else {
        match s0 {
            Some(s) => pv(&s[F_DP]),
            None => Some(&zero),
        }
    };
    if let Some(t) = t {
        guarded(stats, || {
            dp += to_int(Some(t))?;
            Ok(())
        })?;
        if let Some(q) = v.qual {
            qual += q;
        }
    }

    // MQSQ: info.get("MQSQ", info.get("RAW_MQandDP")).
    let t = match &v.info[I_MQSQ] {
        Some(x) => pv(&Some(x.clone())).cloned(),
        None => pv(&v.info[I_RAW_MQANDDP]).cloned(),
    };
    let raw_mq = pv(&v.info[I_RAW_MQ]);
    let mq = pv(&v.info[I_MQ]);
    if let Some(t) = len_at_least(t.as_ref(), 2)? {
        guarded(stats, || {
            mq0 += float_item(t, 0)?;
            mq1 += int_item(t, 1)?;
            Ok(())
        })?;
    } else if truthy(raw_mq) && truthy(info_dp) {
        guarded(stats, || {
            mq0 += to_float(raw_mq)?;
            mq1 += to_int(info_dp)?;
            Ok(())
        })?;
    } else if truthy(mq) && truthy(info_dp) {
        guarded(stats, || {
            let v_mq = to_float(mq)?;
            let v_dp = to_int(info_dp)?;
            mq0 += v_mq * v_mq * (v_dp as f64);
            mq1 += v_dp;
            Ok(())
        })?;
    }

    // SB: FORMAT/SB when the record has a sample, else INFO/SB.
    let t = match s0 {
        Some(s) => pv(&s[F_SB]),
        None => pv(&v.info[I_SB]),
    };
    if let Some(t) = len_at_least(t, 4)? {
        guarded(stats, || {
            for (i, x) in sb.iter_mut().enumerate() {
                *x += int_item(t, i)?;
            }
            Ok(())
        })?;
    }

    // GTC: INFO/GTC, else from the first sample's PL.
    let t = pv(&v.info[I_GTC]);
    let pl = s0.and_then(|s| pv(&s[F_PL]));
    if let Some(t) = len_at_least(t, 3)? {
        guarded(stats, || {
            for (i, x) in gtc.iter_mut().enumerate() {
                *x += float_item(t, i)?;
            }
            Ok(())
        })?;
    } else if let (Some(_), Some(t)) = (s0, pl) {
        // `if t and all(x is not None for x in t)`: list items are never
        // None; iterating a non-iterable truthy value raises TypeError.
        if truthy(Some(t)) {
            let xs = items(t).map_err(|e| PyErr::Other(e.to_string()))?;
            guarded(stats, || {
                let mut t_float = Vec::with_capacity(xs.len());
                for x in &xs {
                    let e = -0.1 * to_float(Some(x))?;
                    // black_box: LLVM rewrites pow(10, x) with a constant
                    // base into exp10(x), which rounds differently from
                    // the libm pow that math.pow calls.
                    let p = std::hint::black_box(10.0f64).powf(e);
                    // math.pow raises OverflowError for a finite argument
                    // whose result overflows.
                    if p.is_infinite() && e.is_finite() {
                        return Err(PyErr::Other("OverflowError: math range error".into()));
                    }
                    t_float.push(p);
                }
                let s = py_fsum_builtin(&t_float);
                if s > 0.0 {
                    let n_a = v.alt.len() + 1;
                    let n_g = n_a * (n_a + 1) / 2;
                    if t_float.len() == n_g {
                        // g = [(y, x) for x in a for y in a[:x + 1]]
                        let mut j = 0;
                        for x in 0..n_a {
                            for y in 0..=x {
                                let z = t_float[j] / s;
                                j += 1;
                                if y == 0 && x == 0 {
                                    gtc[0] += z;
                                } else if y == 0 {
                                    gtc[1] += z;
                                } else {
                                    gtc[2] += z;
                                }
                            }
                        }
                    }
                }
                Ok(())
            })?;
        }
    }
    Ok(RecordSi {
        qual,
        dp,
        mq0,
        mq1,
        sb,
        gtc,
    })
}

/// `quantize_gtc`: round(g * 2**32) half to even, as int64.
pub fn quantize_gtc(gtc: &[f64; 3]) -> Result<[i64; 3], PyErr> {
    let mut out = [0i64; 3];
    for (o, &g) in out.iter_mut().zip(gtc) {
        let p = g * (1u64 << GTC_FIXED_BITS) as f64;
        if !p.is_finite() {
            return Err(PyErr::Other(format!(
                "round() of a non-finite gtc product: {p}"
            )));
        }
        *o = p.round_ties_even() as i64;
    }
    Ok(out)
}

/// `parse_int_list(value)`.
pub fn parse_int_list(v: Option<&Value>) -> Result<Vec<i64>, PyErr> {
    if !truthy(v) {
        return Ok(Vec::new());
    }
    let v = v.expect("truthy");
    match v {
        Value::Int(i) => Ok(vec![*i]),
        Value::Bool(b) => Ok(vec![i64::from(*b)]),
        Value::Ints(x) => Ok(x.clone()),
        other => {
            let xs = items(other).map_err(|e| PyErr::Other(e.to_string()))?;
            xs.iter()
                .map(|x| to_int(Some(x)).map_err(|e| PyErr::Other(e.to_string())))
                .collect()
        }
    }
}

/// One `sample_site_info` row as `collect_variant_group_data` builds it.
#[derive(Debug, Clone)]
pub struct SiteRow {
    pub sample_id: i64,
    pub alleles: Vec<String>,
    pub gt_1: Option<i64>,
    pub gt_2: Option<i64>,
    /// `[pos, excl_end)` of the carried deletion (del_range_str).
    pub del_range: Option<(i64, i64)>,
    pub dp: i64,
    pub ad: Vec<i64>,
    pub f1r2: Vec<i64>,
    pub f2r1: Vec<i64>,
    pub sb: [i64; 4],
    pub pl: Vec<i64>,
    pub qual: f64,
    pub site_dp: i64,
    pub mqsq_1: f64,
    pub mqsq_2: i64,
    pub gtc: [f64; 3],
    /// The DelSpans shared with the deletion state (None: not its active
    /// deletion).
    pub del_spans: Option<Rc<DelSpans>>,
}

/// A record of the site (`records_out`): (k, v, fmts).
pub struct VarRec {
    pub k: usize,
    pub v: Rc<Record>,
    pub fmts: Vec<String>,
}

/// `collect_variant_group_data`'s result: the site's unified (allele,
/// count) list, its site_info and the per-sample rows.
pub struct GroupSite {
    pub alleles: Vec<(String, i64)>,
    pub site_info: SiteInfo,
    pub rows: Vec<SiteRow>,
}

/// `collect_variant_group_data(pos, grp, ...)` for the shard's chromosome;
/// `processed` holds the records already processed at this position (by
/// identity), `seed` the previous same-site yield's allele list.
#[allow(clippy::too_many_arguments)]
pub fn collect_variant_group_data(
    pos: i64,
    grp: &[Option<Rc<Record>>],
    processed: &mut Vec<Rc<Record>>,
    sample_ids: &[Option<i64>],
    stats: &mut ParseStats,
    seed: Option<&[(String, i64)]>,
    records_out: &mut Vec<VarRec>,
    del_state: &[Option<DelState>],
) -> Result<Option<GroupSite>, PyErr> {
    if grp.iter().all(Option::is_none) {
        return Ok(None);
    }
    let pos_1based = pos + 1;
    let mut variant_grp: Vec<(usize, &Rc<Record>)> = Vec::new();
    for (k, v) in grp.iter().enumerate() {
        let Some(v) = v else { continue };
        if is_ref_block(v) || v.pos != pos {
            continue;
        }
        if processed.iter().any(|p| Rc::ptr_eq(p, v)) {
            continue;
        }
        processed.push(v.clone());
        variant_grp.push((k, v));
    }
    if variant_grp.is_empty() {
        return Ok(None);
    }

    // The unified allele list (insertion order) and its index.
    let mut unified: Vec<String> = Vec::new();
    let mut to_idx: HashMap<String, usize> = HashMap::new();
    let ref_formatted: String = match seed {
        Some(s) if !s.is_empty() => {
            for (a, _) in s {
                to_idx.insert(a.clone(), unified.len());
                unified.push(a.clone());
            }
            s[0].0.clone()
        }
        _ => {
            let r0 = variant_grp[0]
                .1
                .ref_
                .chars()
                .next()
                .ok_or_else(|| PyErr::Other("IndexError: an empty REF".into()))?
                .to_string();
            to_idx.insert(r0.clone(), 0);
            unified.push(r0.clone());
            r0
        }
    };
    // Per record: local index -> unified index; its own allele list.
    let mut l2u: Vec<Vec<usize>> = Vec::with_capacity(variant_grp.len());
    let mut own: Vec<Vec<String>> = Vec::with_capacity(variant_grp.len());
    for (k, v) in &variant_grp {
        let mut map = vec![0usize];
        let mut fmts = vec![ref_formatted.clone()];
        for alt in &v.alt {
            let f = format_alt_allele(&v.ref_, alt);
            let ui = match to_idx.get(&f) {
                Some(&ui) => ui,
                None => {
                    let ui = unified.len();
                    to_idx.insert(f.clone(), ui);
                    unified.push(f.clone());
                    ui
                }
            };
            map.push(ui);
            fmts.push(f);
        }
        records_out.push(VarRec {
            k: *k,
            v: (*v).clone(),
            fmts: fmts.clone(),
        });
        l2u.push(map);
        own.push(fmts);
    }

    // Allele counts over every sample of every record.
    let mut counts: Vec<i64> = vec![0; unified.len()];
    for (j, (_k, v)) in variant_grp.iter().enumerate() {
        let gts =
            v.s0.as_ref()
                .map(|s| &s[F_GT])
                .into_iter()
                .chain(v.rest_gts.iter());
        for gt in gts {
            let gt = pv(gt);
            if !truthy(gt) {
                continue;
            }
            match gt_indices(gt.expect("truthy")) {
                Ok(gis) => {
                    for gi in gis {
                        if gi >= 0
                            && let Some(&ui) = l2u[j].get(gi as usize)
                        {
                            counts[ui] += 1;
                        }
                    }
                }
                Err(e) if e.is_value_or_type() => stats.gt_parse_errors += 1,
                Err(e) => return Err(e),
            }
        }
    }
    let alleles: Vec<(String, i64)> = unified.iter().cloned().zip(counts).collect();

    // site_info: left folds in variant_grp order.
    let mut si = SiteInfo {
        qual: 0.0,
        dp: 0,
        mq0: 0.0,
        mq1: 0,
        sb: [0; 4],
        gtc: [0; 3],
    };
    let mut record_si: Vec<RecordSi> = Vec::with_capacity(variant_grp.len());
    for (_k, v) in &variant_grp {
        let r = extract_site_info(v, stats)?;
        si.qual += r.qual;
        si.dp += r.dp;
        si.mq0 += r.mq0;
        si.mq1 += r.mq1;
        for i in 0..4 {
            si.sb[i] += r.sb[i];
        }
        for (i, q) in quantize_gtc(&r.gtc)?.iter().enumerate() {
            si.gtc[i] += q;
        }
        record_si.push(r);
    }

    // The per-sample rows.
    let mut rows = Vec::new();
    for (j, (k, v)) in variant_grp.iter().enumerate() {
        let Some(sample_id) = sample_ids.get(*k).copied().flatten() else {
            continue;
        };
        let Some(sample) = dict0(v) else { continue };
        let own_j = &own[j];
        let gt = pv(&sample[F_GT]);
        let mut gt_parts: Vec<i64> = Vec::new();
        if truthy(gt) {
            match gt_indices(gt.expect("truthy")) {
                Ok(g) => gt_parts = g,
                Err(e) if e.is_value_or_type() => {}
                Err(e) => return Err(e),
            }
        }
        let local = |i: i64| -> Option<i64> {
            if 0 <= i && (i as usize) < own_j.len() {
                Some(i)
            } else {
                None
            }
        };
        let gt_1 = gt_parts.first().and_then(|&i| local(i));
        let gt_2 = gt_parts.get(1).and_then(|&i| local(i));
        let lref = clen(&v.ref_) as i64;
        let mut del_range = None;
        let mut max_del_end = -1i64;
        for &g in &gt_parts {
            if g > 0 && g <= v.alt.len() as i64 {
                let alt = &v.alt[(g - 1) as usize];
                let ref_size = (lref - clen(alt) as i64).max(0);
                if ref_size > 0 {
                    let excl_end = pos_1based + ref_size + 1;
                    if excl_end > max_del_end {
                        max_del_end = excl_end;
                        del_range = Some((pos_1based, excl_end));
                    }
                }
            }
        }
        let dp = match pv(&sample[F_DP]) {
            Some(d) => to_int(Some(d)).map_err(|e| PyErr::Other(e.to_string()))?,
            None => 0,
        };
        let r = record_si[j];
        let del_spans = match del_state.get(*k) {
            Some(Some(d)) if Rc::ptr_eq(&d.spans.record, v) => Some(d.spans.clone()),
            _ => None,
        };
        rows.push(SiteRow {
            sample_id,
            alleles: own_j.clone(),
            gt_1,
            gt_2,
            del_range,
            dp,
            ad: parse_int_list(pv(&sample[F_AD]))?,
            f1r2: parse_int_list(pv(&sample[F_F1R2]))?,
            f2r1: parse_int_list(pv(&sample[F_F2R1]))?,
            sb: r.sb,
            pl: parse_int_list(pv(&sample[F_PL]))?,
            qual: r.qual,
            site_dp: r.dp,
            mqsq_1: r.mq0,
            mqsq_2: r.mq1,
            gtc: r.gtc,
            del_spans,
        });
    }
    Ok(Some(GroupSite {
        alleles,
        site_info: si,
        rows,
    }))
}

/// `merge_site_alleles(existing, new)`: counts add by allele string, new
/// strings append in order.
pub fn merge_site_alleles(existing: &[(String, i64)], new: &[(String, i64)]) -> Vec<(String, i64)> {
    let mut out: Vec<(String, i64)> = existing.to_vec();
    let mut idx: HashMap<&str, usize> = HashMap::with_capacity(out.len());
    for (i, (a, _)) in existing.iter().enumerate() {
        idx.insert(a.as_str(), i);
    }
    for (a, c) in new {
        match idx.get(a.as_str()) {
            Some(&i) => out[i].1 += c,
            None => {
                idx.insert(a.as_str(), out.len());
                out.push((a.clone(), *c));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    //! `test_allele_encoding.py`'s extract_site_info cases and
    //! `test_removal_substrate.py`'s per-row contribution cases.
    use super::*;
    use crate::matrix::tests::var;

    fn si(info: &str, fmt: &str, sample: &str) -> RecordSi {
        let v = var(100, "A", "G,<NON_REF>", info, fmt, sample);
        extract_site_info(&v, &mut ParseStats::default()).unwrap()
    }

    #[test]
    fn dp_and_qual() {
        let r = si("DP=20", "GT:DP", "0/1:9");
        assert_eq!((r.dp, r.qual), (20, 50.0));
        let r = si(".", "GT:DP", "0/1:9");
        assert_eq!(r.dp, 9);
        // No DP anywhere: t is None, QUAL is not added either.
        let r = si(".", "GT", "0/1");
        assert_eq!((r.dp, r.qual), (0, 0.0));
    }

    #[test]
    fn mq_sources() {
        let r = si("DP=10;MQSQ=36000.5,10", "GT", "0/1");
        assert_eq!((r.mq0, r.mq1), (36000.5, 10));
        let r = si("DP=10;RAW_MQandDP=3600,12", "GT", "0/1");
        assert_eq!((r.mq0, r.mq1), (3600.0, 12));
        let r = si("DP=10;RAW_MQ=1800", "GT", "0/1");
        assert_eq!((r.mq0, r.mq1), (1800.0, 10));
        let r = si("DP=10;MQ=60", "GT", "0/1");
        assert_eq!((r.mq0, r.mq1), (60.0 * 60.0 * 10.0, 10));
        // MQSQ present but '.': the fallbacks apply.
        let r = si("DP=10;MQSQ=.;MQ=59.5", "GT", "0/1");
        assert_eq!(r.mq0.to_bits(), (59.5f64 * 59.5 * 10.0).to_bits());
    }

    #[test]
    fn sb_and_gtc() {
        let r = si(".", "GT:SB", "0/1:1,2,3,4");
        assert_eq!(r.sb, [1, 2, 3, 4]);
        let r = si(".", "GT:SB", "0/1:1,2,3");
        assert_eq!(r.sb, [0; 4]);
        let r = si("GTC=1,2,3", "GT", "0/1");
        assert_eq!(r.gtc, [1.0, 2.0, 3.0]);
        // From PL: two alleles' three genotypes.
        let r = si(".", "GT:PL", "0/1:0,10,100,20,30,40");
        let t: Vec<f64> = [0.0f64, 10.0, 100.0, 20.0, 30.0, 40.0]
            .iter()
            .map(|x| std::hint::black_box(10f64).powf(-0.1 * x))
            .collect();
        let s = py_fsum_builtin(&t);
        assert_eq!(r.gtc[0].to_bits(), (t[0] / s).to_bits());
        assert_eq!(r.gtc[1].to_bits(), (t[1] / s + t[3] / s).to_bits());
        assert_eq!(
            r.gtc[2].to_bits(),
            ((t[2] / s + t[4] / s) + t[5] / s).to_bits()
        );
        // A PL of the wrong length: nothing.
        let r = si(".", "GT:PL", "0/1:0,10,100");
        assert_eq!(r.gtc, [0.0; 3]);
    }

    #[test]
    fn undeclared_string_values_and_parse_errors() {
        // RAW_MQ declared Float above; an undeclared key would stay a
        // string: use FORMAT keys the header lacks to show it. MQSQ is a
        // declared list: a malformed list drops the record at parse, so
        // the errors here come from int() of a float string.
        let v = var(100, "A", "G", "DP=10;MQSQ=1.5,2.5", "GT", "0/1");
        let mut st = ParseStats::default();
        let r = extract_site_info(&v, &mut st).unwrap();
        // float('1.5') added, int(2.5) of a float truncates: no error.
        assert_eq!((r.mq0, r.mq1), (1.5, 2));
        assert_eq!(st.site_info_parse_errors, 0);
    }

    #[test]
    fn quantize() {
        assert_eq!(
            quantize_gtc(&[0.0, 1.0, 0.5]).unwrap(),
            [0, 1 << 32, 1 << 31]
        );
        // Half to even at 2**-33.
        let h = 2f64.powi(-33);
        assert_eq!(quantize_gtc(&[h, 3.0 * h, 0.0]).unwrap(), [0, 2, 0]);
    }

    #[test]
    fn rows_carry_their_own_list_and_seed() {
        let a = var(
            100,
            "A",
            "G,<NON_REF>",
            "DP=5",
            "GT:AD:DP:PL",
            "0/1:3,2,0:5:30,0,40,90,90,90",
        );
        let b = var(100, "A", "T,G,<NON_REF>", "DP=6", "GT:DP", "2/2:6");
        let grp = vec![Some(a), Some(b)];
        let mut processed = vec![];
        let mut recs = vec![];
        let g = collect_variant_group_data(
            99,
            &grp,
            &mut processed,
            &[Some(1), Some(2)],
            &mut ParseStats::default(),
            None,
            &mut recs,
            &[None, None],
        )
        .unwrap()
        .unwrap();
        let names: Vec<&str> = g.alleles.iter().map(|(a, _)| a.as_str()).collect();
        assert_eq!(names, vec!["A", "xG", "*", "xT"]);
        let counts: Vec<i64> = g.alleles.iter().map(|(_, c)| *c).collect();
        assert_eq!(counts, vec![1, 3, 0, 0]);
        assert_eq!(g.rows[1].alleles, vec!["A", "xT", "xG", "*"]);
        assert_eq!((g.rows[1].gt_1, g.rows[1].gt_2), (Some(2), Some(2)));
        assert_eq!(g.rows[0].ad, vec![3, 2, 0]);
        assert_eq!(g.site_info.dp, 11);
        // Processed again at the same position: nothing new.
        let again = collect_variant_group_data(
            99,
            &grp,
            &mut processed,
            &[Some(1), Some(2)],
            &mut ParseStats::default(),
            Some(&g.alleles),
            &mut recs,
            &[None, None],
        )
        .unwrap();
        assert!(again.is_none());
        let m = merge_site_alleles(&g.alleles, &[("A".into(), 1), ("+C".into(), 2)]);
        assert_eq!(m.last().unwrap(), &("+C".to_string(), 2));
        assert_eq!(m[0].1, 2);
    }

    // ── K6 ports: test_allele_encoding.py's extract_site_info cases ──────
    // (FakeVariant: no FORMAT column means no sample; QUAL '.' is None.)

    fn line_rec(pos_1b: i64, r: &str, alt: &str, qual: &str, info: &str, fs: &str) -> Rc<Record> {
        crate::matrix::tests::rec(&format!(
            "chr21\t{pos_1b}\t.\t{r}\t{alt}\t{qual}\tPASS\t{info}{fs}"
        ))
    }

    fn esi(v: &Record) -> RecordSi {
        extract_site_info(v, &mut ParseStats::default()).unwrap()
    }

    // Replaces tests/unit/test_allele_encoding.py::test_extract_site_info_dp_qual_from_info (K6).
    #[test]
    fn esi_dp_qual_from_info() {
        let r = esi(&line_rec(1, ".", ".", "123.0", "DP=30", ""));
        assert_eq!(r.dp, 30);
        assert_eq!(r.qual, 123.0);
    }

    // Replaces tests/unit/test_allele_encoding.py::test_extract_site_info_dp_from_sample_when_info_absent (K6).
    #[test]
    fn esi_dp_from_sample_when_info_absent() {
        let r = esi(&line_rec(1, ".", ".", "50.0", ".", "\tDP\t20"));
        assert_eq!(r.dp, 20);
    }

    // Replaces tests/unit/test_allele_encoding.py::test_extract_site_info_mqsq_direct (K6).
    #[test]
    fn esi_mqsq_direct() {
        let r = esi(&line_rec(1, ".", ".", ".", "DP=10;MQSQ=1234.5,30", ""));
        assert_eq!((r.mq0, r.mq1), (1234.5, 30));
    }

    // Replaces tests/unit/test_allele_encoding.py::test_extract_site_info_mqsq_from_raw_mqanddp (K6).
    #[test]
    fn esi_mqsq_from_raw_mqanddp() {
        let r = esi(&line_rec(
            1,
            ".",
            ".",
            ".",
            "DP=40;RAW_MQandDP=5000.0,40",
            "",
        ));
        assert_eq!((r.mq0, r.mq1), (5000.0, 40));
    }

    // Replaces tests/unit/test_allele_encoding.py::test_extract_site_info_mq_fallback_squares (K6).
    #[test]
    fn esi_mq_fallback_squares() {
        let r = esi(&line_rec(1, ".", ".", ".", "MQ=60.0;DP=30", ""));
        assert_eq!((r.mq0, r.mq1), (60.0 * 60.0 * 30.0, 30));
    }

    // Replaces tests/unit/test_allele_encoding.py::test_extract_site_info_sb_from_sample (K6).
    #[test]
    fn esi_sb_from_sample() {
        let r = esi(&line_rec(1, ".", ".", ".", "DP=8", "\tSB\t1,2,3,4"));
        assert_eq!(r.sb, [1, 2, 3, 4]);
    }

    // Replaces tests/unit/test_allele_encoding.py::test_extract_site_info_gtc_from_info (K6).
    #[test]
    fn esi_gtc_from_info() {
        let r = esi(&line_rec(1, ".", ".", ".", "DP=8;GTC=5,3,2", ""));
        assert_eq!(r.gtc, [5.0, 3.0, 2.0]);
    }

    // Replaces tests/unit/test_allele_encoding.py::test_extract_site_info_gtc_from_pl (K6).
    #[test]
    fn esi_gtc_from_pl() {
        let r = esi(&line_rec(1, "A", "T", ".", "DP=30", "\tPL\t0,10,100"));
        assert!((r.gtc.iter().sum::<f64>() - 1.0).abs() <= 1e-6);
        assert!(r.gtc[0] > 0.9);
        assert_eq!(r.gtc[0], r.gtc.iter().cloned().fold(f64::MIN, f64::max));
    }

    // ── K6 ports: test_removal_substrate.py's per-row contribution ───────

    const PL6: &str = "0,30,300,33,330,333";

    /// One site of one-record streams at 0-based `pos` (one grouper yield:
    /// the records are in different streams).
    fn one_site(pos: i64, recs: Vec<Rc<Record>>, sids: &[i64]) -> GroupSite {
        let n = recs.len();
        let grp: Vec<Option<Rc<Record>>> = recs.into_iter().map(Some).collect();
        let ids: Vec<Option<i64>> = sids.iter().map(|&s| Some(s)).collect();
        collect_variant_group_data(
            pos,
            &grp,
            &mut vec![],
            &ids,
            &mut ParseStats::default(),
            None,
            &mut vec![],
            &vec![None; n],
        )
        .unwrap()
        .unwrap()
    }

    fn q32(g: f64) -> i64 {
        (g * 4294967296.0).round_ties_even() as i64
    }

    // Replaces tests/unit/test_removal_substrate.py::test_row_carries_extract_site_info (K6).
    #[test]
    fn row_carries_extract_site_info() {
        // (info, FORMAT keys after GT:PL, their values, site_dp, (mq0, mq1),
        // sb, Some(gtc) when INFO/GTC); Python's 6 cases.
        type Case<'a> = (
            &'a str,
            &'a str,
            &'a str,
            i64,
            (f64, i64),
            [i64; 4],
            Option<[f64; 3]>,
        );
        let cases: [Case; 6] = [
            (
                "DP=30;MQSQ=1234.5,30",
                ":DP:SB",
                ":25:1,2,3,4",
                30,
                (1234.5, 30),
                [1, 2, 3, 4],
                None,
            ),
            (
                "DP=40;RAW_MQandDP=72000.0,20",
                ":DP",
                ":38",
                40,
                (72000.0, 20),
                [0; 4],
                None,
            ),
            (
                "DP=7;RAW_MQ=25200.0",
                ":DP",
                ":7",
                7,
                (25200.0, 7),
                [0; 4],
                None,
            ),
            (
                "DP=12;MQ=59.7",
                ":DP:SB",
                ":11:5,0,6,1",
                12,
                (59.7 * 59.7 * 12.0, 12),
                [5, 0, 6, 1],
                None,
            ),
            (
                "DP=9;GTC=1,0,0;MQ=60.0",
                ":DP",
                ":9",
                9,
                (60.0 * 60.0 * 9.0, 9),
                [0; 4],
                Some([1.0, 0.0, 0.0]),
            ),
            // No INFO/DP: site dp falls back to FORMAT/DP; MQ needs INFO/DP.
            ("MQ=50.0", ":DP", ":17", 17, (0.0, 0), [0; 4], None),
        ];
        for (info, fk, fv, site_dp, mq, sb, gtc) in cases {
            let v = line_rec(
                1001,
                "A",
                "T,<NON_REF>",
                "51.25",
                info,
                &format!("\tGT:PL{fk}\t0/1:{PL6}{fv}"),
            );
            let want = esi(&v);
            let g = one_site(1000, vec![v], &[7]);
            assert_eq!(g.rows.len(), 1, "{info}");
            let row = &g.rows[0];
            assert_eq!(row.qual.to_bits(), want.qual.to_bits(), "{info}");
            assert_eq!(row.qual, 51.25, "{info}");
            assert_eq!(row.site_dp, want.dp, "{info}");
            assert_eq!(row.site_dp, site_dp, "{info}");
            assert_eq!(row.mqsq_1.to_bits(), want.mq0.to_bits(), "{info}");
            assert_eq!((row.mqsq_1, row.mqsq_2), mq, "{info}");
            assert_eq!(row.mqsq_2, want.mq1, "{info}");
            assert_eq!(row.gtc, want.gtc, "{info}");
            assert_eq!(row.sb, want.sb, "{info}");
            assert_eq!(row.sb, sb, "{info}");
            assert!(row.del_spans.is_none(), "{info}"); // not a deletion
            // One record: its contribution is the site's site_info.
            let si = &g.site_info;
            assert_eq!(
                (si.qual, si.dp, si.mq0, si.mq1),
                (row.qual, row.site_dp, row.mqsq_1, row.mqsq_2),
                "{info}"
            );
            assert_eq!(si.sb, row.sb, "{info}");
            assert_eq!(si.gtc, row.gtc.map(q32), "{info}");
            match gtc {
                Some(x) => assert_eq!(row.gtc, x, "{info}"),
                // PL-derived, fractional.
                None => assert!(row.gtc.iter().any(|&x| 0.0 < x && x < 1.0), "{info}"),
            }
        }
    }

    // Replaces tests/unit/test_removal_substrate.py::test_site_dp_is_info_dp_and_dp_stays_format_dp (K6).
    #[test]
    fn site_dp_is_info_dp_and_dp_stays_format_dp() {
        let v = line_rec(1001, "A", "T,<NON_REF>", "10.0", "DP=30", "\tGT:DP\t0/1:25");
        let g = one_site(1000, vec![v], &[7]);
        assert_eq!((g.rows[0].dp, g.rows[0].site_dp), (25, 30));
    }

    // Replaces tests/unit/test_removal_substrate.py::test_rows_sum_to_the_site_info (K6).
    #[test]
    fn rows_sum_to_the_site_info() {
        let fs = format!("\tGT:PL:SB\t{{}}:{PL6}:1,2,3,4");
        let mk = |r: &str, alt: &str, q: &str, info: &str, gt: &str| {
            line_rec(501, r, alt, q, info, &fs.replace("{}", gt))
        };
        let a = mk("A", "T,<NON_REF>", "40.5", "DP=30;MQ=60.0", "0/1");
        let b = mk("AT", "A,<NON_REF>", "70.0", "DP=20;MQ=58.0", "1/1");
        let c = mk("A", "T,<NON_REF>", "9.5", "DP=11;MQ=41.0", "1/1");
        let g = one_site(500, vec![a, b, c], &[1, 2, 3]);
        let rows = &g.rows;
        assert_eq!(rows.len(), 3);
        let si = &g.site_info;
        let close = |x: f64, y: f64| (x - y).abs() <= 1e-9 * x.abs().max(y.abs());
        assert!(close(si.qual, rows.iter().map(|r| r.qual).sum()));
        assert_eq!(si.dp, rows.iter().map(|r| r.site_dp).sum::<i64>());
        assert!(close(si.mq0, rows.iter().map(|r| r.mqsq_1).sum()));
        assert_eq!(si.mq1, rows.iter().map(|r| r.mqsq_2).sum::<i64>());
        for j in 0..4 {
            assert_eq!(si.sb[j], rows.iter().map(|r| r.sb[j]).sum::<i64>());
        }
        for j in 0..3 {
            assert_eq!(si.gtc[j], rows.iter().map(|r| q32(r.gtc[j])).sum::<i64>());
        }
    }

    // Replaces tests/unit/test_removal_substrate.py::test_sb_is_the_summed_value (K6).
    #[test]
    fn sb_is_the_summed_value() {
        for (sb, want) in [
            (None, [0, 0, 0, 0]),
            (Some("3,4"), [0, 0, 0, 0]),
            (Some("1,2,3,4,5"), [1, 2, 3, 4]),
        ] {
            let fs = match sb {
                Some(s) => format!("\tGT:SB\t0/1:{s}"),
                None => "\tGT\t0/1".to_string(),
            };
            let v = line_rec(1001, "A", "T,<NON_REF>", "1.0", "DP=5", &fs);
            let g = one_site(1000, vec![v], &[7]);
            assert_eq!(g.rows[0].sb, want, "{sb:?}");
            assert_eq!(g.site_info.sb, want, "{sb:?}");
        }
    }

    // ── K6 ports: the record-derived deletion span (test_allele_catalog.py) ─

    // Replaces the ingest half of
    // tests/unit/test_allele_catalog.py::test_del_range_matches_ingest_span
    // (K6): the allele string ingest lists for the record and the span the
    // record gives its carrier's row, `[pos, pos + len(REF) - len(ALT) + 1)`
    // for a deletion, none otherwise. (`allele_del_range` of
    // mosaic/aggregate.py, which derives the catalog's span from the
    // string, stays tested in Python.)
    #[test]
    fn del_range_of_the_carried_allele() {
        for (r, alt, allele) in [
            ("AT", "A", "-AT"),
            ("ATG", "A", "-ATG"),
            ("ATGC", "AGC", "-AT"),
            ("ATTTTTTTTTTTG", "A", "-ATTTTTTTTTTTG"),
            ("A", "AT", "+AT"),
            ("A", "T", "xT"),
            ("AT", "GC", "xGC"),
            ("A", "<NON_REF>", "*"),
        ] {
            let alts = if alt == "<NON_REF>" {
                "T,<NON_REF>".to_string()
            } else {
                format!("{alt},<NON_REF>")
            };
            let v = line_rec(1001, r, &alts, ".", ".", "\tGT\t0/1");
            let g = one_site(1000, vec![v], &[1]);
            assert!(g.alleles.iter().any(|(a, _)| a == allele), "{r} {alt}");
            let row = &g.rows[0];
            assert!(row.alleles.iter().any(|a| a == allele), "{r} {alt}");
            if allele.starts_with('-') {
                let want = (1001, 1001 + clen(r) as i64 - clen(alt) as i64 + 1);
                assert_eq!(row.del_range, Some(want), "{r} {alt}");
                // The same span the catalog derives from the string:
                // [pos, pos + len(allele) - 1).
                assert_eq!(want.1, 1001 + clen(allele) as i64 - 1, "{r} {alt}");
            } else {
                assert_eq!(row.del_range, None, "{r} {alt}");
            }
        }
    }

    // ── K6 port: the same-site second yield (test_genotype_builder.py) ───

    struct RecSource(std::vec::IntoIter<Record>);

    impl crate::grouper::Source for RecSource {
        type Rec = Record;
        fn next_rec(&mut self) -> anyhow::Result<Option<Record>> {
            Ok(self.0.next())
        }
    }

    // Replaces tests/unit/test_genotype_builder.py::test_same_site_second_yield_keeps_its_own_list (K6).
    #[test]
    fn same_site_second_yield_keeps_its_own_list() {
        let snp = line_rec(
            101,
            "A",
            "T,<NON_REF>",
            ".",
            ".",
            "\tGT:AD:PL\t0/1:10,10,0:50,0,50,80,80,160",
        );
        let dele = line_rec(
            101,
            "AT",
            "A,<NON_REF>",
            ".",
            ".",
            "\tGT:AD:PL\t0/1:10,8,0:40,0,45,80,80,160",
        );
        let src = RecSource(vec![(*snp).clone(), (*dele).clone()].into_iter());
        let mut g = crate::grouper::Grouper::new(vec![Some(src)]).unwrap();
        let mut alleles: Option<Vec<(String, i64)>> = None;
        let mut rows: Vec<SiteRow> = Vec::new();
        let mut processed = vec![];
        let mut yields = 0;
        while let Some(pos) = g.next_group().unwrap() {
            yields += 1;
            let res = collect_variant_group_data(
                pos,
                &g.grp,
                &mut processed,
                &[Some(1)],
                &mut ParseStats::default(),
                alleles.as_deref(),
                &mut vec![],
                &[None],
            )
            .unwrap()
            .unwrap();
            alleles = Some(match alleles {
                None => res.alleles,
                Some(a) => merge_site_alleles(&a, &res.alleles),
            });
            rows.extend(res.rows);
        }
        assert_eq!(yields, 2);
        let merged: Vec<&str> = alleles
            .as_ref()
            .unwrap()
            .iter()
            .map(|(a, _)| a.as_str())
            .collect();
        assert_eq!(merged, vec!["A", "xT", "*", "-AT"]);
        assert_eq!(rows.len(), 2);
        let (s, d) = (&rows[0], &rows[1]);
        assert_eq!(s.alleles, vec!["A", "xT", "*"]);
        assert_eq!(d.alleles, vec!["A", "-AT", "*"]);
        let at = |r: &SiteRow, g: Option<i64>| r.alleles[g.unwrap() as usize].clone();
        assert_eq!((at(s, s.gt_1), at(s, s.gt_2)), ("A".into(), "xT".into()));
        assert_eq!((at(d, d.gt_1), at(d, d.gt_2)), ("A".into(), "-AT".into()));
        assert_eq!(d.ad, vec![10, 8, 0]);
    }
}
