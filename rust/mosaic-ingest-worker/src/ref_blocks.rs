// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The per-sample ref blocks and the deletion split: `_ref_block_dp`,
//! `_extract_rb_range`, `_ref_block_format`,
//! `collect_sample_ref_block_data`, `_record_del_span`, `_record_site_dp`,
//! `update_sample_del_state`, `DelSpans`, `merge_spans` (originally a
//! port of the Python ingest worker's functions of those names).

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;

use crate::banding::RefBlockBander;
use crate::matrix::{
    F_AD, F_DP, F_GT, F_MIN_DP, F_PL, dict0, genotype_quality, gt_fields_str, parse_gt, samples0,
    star_ends,
};
use crate::pyops::{PyErr, int_item, pv, py_len, to_int, truthy};
use crate::record::Record;
use crate::record::is_ref_block;
use crate::value::Value;
use mosaic_core::alleles::clen;

/// INFO key indices of `Record::info` (`record::INFO_KEYS`).
pub const I_DP: usize = 0;
pub const I_MQ: usize = 2;
pub const I_MQSQ: usize = 3;
pub const I_RAW_MQ: usize = 4;
pub const I_RAW_MQANDDP: usize = 5;
pub const I_GTC: usize = 6;
pub const I_SB: usize = 7;

/// A `sample_ref_blocks` row: (sample_id, chromosome_index,
/// position_start, position_end, ad_1, ad_2, dp, pl_1, pl_2, pl_3,
/// sample_counts).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RbRow {
    pub sample_id: i64,
    pub c_idx: i64,
    pub start: i64,
    pub end: i64,
    pub ad_1: i64,
    pub ad_2: i64,
    pub dp: i64,
    pub pl_1: i64,
    pub pl_2: i64,
    pub pl_3: i64,
    pub sc: i64,
}

/// A batch_ref_blocks contribution: (c_idx, start, end, dp,
/// sample_counts), an inclusive 1-based range.
pub type RbRange = (i64, i64, i64, i64, i64);

/// A block seen for the first time (`new_blocks_out`): (k, v, start, end).
pub struct NewBlock {
    pub k: usize,
    pub v: Rc<Record>,
    pub start: i64,
    pub end: i64,
}

/// `DelSpans`: the ref-block spans one deletion record's split overrode.
#[derive(Debug)]
pub struct DelSpans {
    /// (sample_id, chromosome_index, position) of the record's row.
    pub owner: (i64, i64, i64),
    pub record: Rc<Record>,
    pub spans: RefCell<Vec<i64>>,
}

impl DelSpans {
    pub fn new(owner: (i64, i64, i64), record: Rc<Record>) -> DelSpans {
        DelSpans {
            owner,
            record,
            spans: RefCell::new(Vec::new()),
        }
    }
}

/// One stream's active deletion: (del_end, del_dp, star_end_1,
/// star_end_2, gq, DelSpans).
#[derive(Debug, Clone)]
pub struct DelState {
    pub del_end: i64,
    pub del_dp: i64,
    pub e1: i64,
    pub e2: i64,
    pub gq: i64,
    pub spans: Rc<DelSpans>,
}

/// `foreign_del_spans`: {(sample_id, c_idx, pos): flat spans} in insertion
/// order.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ForeignSpans {
    pub order: Vec<(i64, i64, i64)>,
    pub map: HashMap<(i64, i64, i64), Vec<i64>>,
}

impl ForeignSpans {
    pub fn extend(&mut self, key: (i64, i64, i64), spans: &[i64]) {
        self.map
            .entry(key)
            .or_insert_with(|| {
                self.order.push(key);
                Vec::new()
            })
            .extend_from_slice(spans);
    }
}

/// `merge_spans`: sorted, overlapping or adjacent inclusive spans merged;
/// None for none.
pub fn merge_spans(flat: &[i64]) -> Option<Vec<i64>> {
    if flat.is_empty() {
        return None;
    }
    let mut pairs: Vec<(i64, i64)> = flat.chunks(2).map(|p| (p[0], p[1])).collect();
    pairs.sort_unstable();
    let mut out = vec![pairs[0].0, pairs[0].1];
    for &(s, e) in &pairs[1..] {
        let last = out.len() - 1;
        if s <= out[last] + 1 {
            if e > out[last] {
                out[last] = e;
            }
        } else {
            out.push(s);
            out.push(e);
        }
    }
    Some(out)
}

/// `_ref_block_dp(v)`: FORMAT/MIN_DP, else INFO/DP, else 0.
pub fn ref_block_dp(v: &Record) -> Result<i64, PyErr> {
    if let Some(s) = dict0(v)
        && let Some(m) = pv(&s[F_MIN_DP])
    {
        return to_int(Some(m));
    }
    match pv(&v.info[I_DP]) {
        Some(d) => to_int(Some(d)),
        None => Ok(0),
    }
}

/// `_ref_block_format(sample)`: (ad_1, ad_2, pl_1, pl_2, pl_3), 0 where
/// missing. `sample` is the truthy dict.
pub fn ref_block_format(sample: Option<&crate::matrix::Sample>) -> Result<[i64; 5], PyErr> {
    let mut out = [0i64; 5];
    if let Some(s) = sample {
        let ad = pv(&s[F_AD]);
        if truthy(ad) && py_len(ad)? >= 2 {
            let ad = ad.expect("truthy");
            let (a, b) = (int_item(ad, 0)?, int_item(ad, 1)?);
            out[0] = a;
            out[1] = b;
        }
        let pl = pv(&s[F_PL]);
        if truthy(pl) && py_len(pl)? >= 3 {
            let pl = pl.expect("truthy");
            let (a, b, c) = (int_item(pl, 0)?, int_item(pl, 1)?, int_item(pl, 2)?);
            out[2] = a;
            out[3] = b;
            out[4] = c;
        }
    }
    Ok(out)
}

/// `_record_site_dp(v)`: INFO/DP (falsy: FORMAT/DP), else 0; an
/// unparseable value is 0.
pub fn record_site_dp(v: &Record) -> Result<i64, PyErr> {
    let zero = Value::Int(0);
    let info_dp = pv(&v.info[I_DP]);
    let t = if truthy(info_dp) {
        info_dp
    } else {
        match samples0(v) {
            Some(s) => pv(&s[F_DP]),
            None => Some(&zero),
        }
    };
    let Some(t) = t else {
        return Ok(0);
    };
    match to_int(Some(t)) {
        Ok(i) => Ok(i),
        Err(e) if e.is_value_or_type() => Ok(0),
        Err(e) => Err(e),
    }
}

/// `[int(x) for x in gt.replace("|", "/").split("/") if x != "."]` of a
/// truthy GT: Err(Value / Type) where the Python code catches it, the
/// AttributeError of a non-string GT as `PyErr::Other`.
pub fn gt_indices(gt: &Value) -> Result<Vec<i64>, PyErr> {
    let Value::Str(s) = gt else {
        return Err(PyErr::Other(
            "AttributeError: a non-string GT has no attribute 'replace'".into(),
        ));
    };
    let mut out = Vec::with_capacity(2);
    for x in gt_fields_str(s) {
        if x != "." {
            out.push(to_int(Some(&Value::Str(x.to_string())))?);
        }
    }
    Ok(out)
}

/// `update_sample_del_state(grp, sample_ids, contig_map, state)` for the
/// shard's chromosome (`c_idx`); `state` is indexed by stream.
pub fn update_sample_del_state(
    grp: &[Option<Rc<Record>>],
    sample_ids: &[Option<i64>],
    c_idx: i64,
    state: &mut [Option<DelState>],
) -> Result<(), PyErr> {
    for (k, v) in grp.iter().enumerate() {
        let Some(v) = v else { continue };
        if is_ref_block(v) {
            continue;
        }
        let Some(sid) = sample_ids.get(k).copied().flatten() else {
            continue;
        };
        let pos_1based = v.pos + 1;
        let Some(sample) = dict0(v) else { continue };
        let gt = pv(&sample[F_GT]);
        if !truthy(gt) {
            continue;
        }
        let gis = match gt_indices(gt.expect("truthy")) {
            Ok(g) => g,
            Err(e) if e.is_value_or_type() => continue,
            Err(e) => return Err(e),
        };
        let lref = clen(&v.ref_) as i64;
        let mut found_del = false;
        for gi in gis {
            if gi > 0 && gi <= v.alt.len() as i64 {
                let allele = &v.alt[(gi - 1) as usize];
                if allele != "<NON_REF>" && (clen(allele) as i64) < lref {
                    let del_end = pos_1based + lref - 1;
                    if state[k].as_ref().is_none_or(|d| del_end > d.del_end) {
                        let fields = parse_gt(gt)?.unwrap_or_default();
                        let (e1, e2) = star_ends(v, &fields, pos_1based);
                        state[k] = Some(DelState {
                            del_end,
                            del_dp: record_site_dp(v)?,
                            e1,
                            e2,
                            gq: genotype_quality(Some(sample))?,
                            spans: Rc::new(DelSpans::new((sid, c_idx, pos_1based), v.clone())),
                        });
                    }
                    found_del = true;
                    break;
                }
            }
        }
        if !found_del
            && let Some(d) = &state[k]
            && pos_1based > d.del_end
        {
            state[k] = None;
        }
    }
    Ok(())
}

/// `_record_del_span`: the overlapped span on the deletion's row when the
/// row is this shard's, else in `foreign`.
fn record_del_span(
    holder: &DelSpans,
    start: i64,
    end: i64,
    shard_bounds: Option<(i64, i64)>,
    foreign: &mut ForeignSpans,
) {
    let (mut start, mut end) = (start, end);
    if let Some((lo, hi)) = shard_bounds {
        start = start.max(lo);
        end = end.min(hi);
    }
    if start > end {
        return;
    }
    if shard_bounds.is_none_or(|(lo, _)| holder.owner.2 >= lo) {
        holder.spans.borrow_mut().extend_from_slice(&[start, end]);
    } else {
        foreign.extend(holder.owner, &[start, end]);
    }
}

/// The outputs of `collect_sample_ref_block_data` for one group.
pub struct RbOut<'a> {
    pub rows: &'a mut Vec<RbRow>,
    pub new_blocks: &'a mut Vec<NewBlock>,
    pub partials: &'a mut Vec<RbRange>,
    pub foreign: &'a mut ForeignSpans,
}

/// `collect_sample_ref_block_data` for the shard's chromosome `c_idx`;
/// `last_start` and `del_state` are indexed by stream.
#[allow(clippy::too_many_arguments)]
pub fn collect_sample_ref_block_data(
    grp: &[Option<Rc<Record>>],
    sample_ids: &[Option<i64>],
    c_idx: i64,
    last_start: &mut [i64],
    del_state: &[Option<DelState>],
    shard_bounds: Option<(i64, i64)>,
    mut bander: Option<&mut RefBlockBander>,
    out: RbOut<'_>,
) -> Result<(), PyErr> {
    for (k, v) in grp.iter().enumerate() {
        let Some(v) = v else { continue };
        if !is_ref_block(v) {
            continue;
        }
        // _extract_rb_range: 1-based start; end = INFO/END, else
        // v_pos + len(REF) - 1 (= v.end either way).
        let start = v.pos + 1;
        let end = v.end;
        let rb_dp = ref_block_dp(v)?;
        let rb_sc = i64::from(v.homref);
        if last_start[k] >= start {
            continue;
        }
        last_start[k] = start;
        let sid = sample_ids.get(k).copied().flatten();
        if sid.is_some() {
            out.new_blocks.push(NewBlock {
                k,
                v: v.clone(),
                start,
                end,
            });
        }
        let mut rest_start = start;
        if let Some(d) = &del_state[k]
            && start <= d.del_end
        {
            let ovl_end = end.min(d.del_end);
            out.partials.push((c_idx, start, ovl_end, d.del_dp, 0));
            record_del_span(&d.spans, start, ovl_end, shard_bounds, out.foreign);
            rest_start = ovl_end + 1;
        }
        if rest_start > end {
            continue;
        }
        let sample = dict0(v);
        if let (Some(b), Some(sid)) = (bander.as_deref_mut(), sid) {
            let f = ref_block_format(sample)?;
            b.add(
                k,
                sid,
                c_idx,
                rest_start,
                end,
                rb_sc,
                genotype_quality(sample)?,
                if sample.is_some() { rb_dp } else { 0 },
                f,
                out.rows,
                Some(out.partials),
            );
            continue;
        }
        out.partials.push((c_idx, rest_start, end, rb_dp, rb_sc));
        let Some(sid) = sid else { continue };
        let f = ref_block_format(sample)?;
        out.rows.push(RbRow {
            sample_id: sid,
            c_idx,
            start: rest_start,
            end,
            ad_1: f[0],
            ad_2: f[1],
            dp: if sample.is_some() { rb_dp } else { 0 },
            pl_1: f[2],
            pl_2: f[3],
            pl_3: f[4],
            sc: rb_sc,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    //! `tests/unit/test_removal_substrate.py`'s ref-block cases.
    use super::*;
    use crate::matrix::tests::{block, var};

    fn run(
        grps: &[Vec<Option<Rc<Record>>>],
        sids: &[Option<i64>],
        bounds: Option<(i64, i64)>,
    ) -> (
        Vec<RbRow>,
        Vec<RbRange>,
        ForeignSpans,
        Vec<Option<DelState>>,
    ) {
        let n = sids.len();
        let mut last = vec![-1; n];
        let mut ds: Vec<Option<DelState>> = vec![None; n];
        let (mut rows, mut parts, mut fs, mut nb) =
            (vec![], vec![], ForeignSpans::default(), vec![]);
        for g in grps {
            update_sample_del_state(g, sids, 20, &mut ds).unwrap();
            collect_sample_ref_block_data(
                g,
                sids,
                20,
                &mut last,
                &ds,
                bounds,
                None,
                RbOut {
                    rows: &mut rows,
                    new_blocks: &mut nb,
                    partials: &mut parts,
                    foreign: &mut fs,
                },
            )
            .unwrap();
        }
        (rows, parts, fs, ds)
    }

    #[test]
    fn merge_spans_cases() {
        assert_eq!(merge_spans(&[]), None);
        assert_eq!(merge_spans(&[5, 7]), Some(vec![5, 7]));
        assert_eq!(merge_spans(&[8, 9, 5, 7]), Some(vec![5, 9]));
        assert_eq!(merge_spans(&[5, 7, 9, 10]), Some(vec![5, 7, 9, 10]));
        assert_eq!(merge_spans(&[5, 10, 6, 7]), Some(vec![5, 10]));
    }

    #[test]
    fn split_overlap_takes_site_dp_and_rest_keeps_the_block() {
        // A deletion at 100 (REF ATT: del_end 102), INFO/DP 30; then the
        // sample's block 101-110 (MIN_DP 12).
        let d = var(100, "ATT", "A,<NON_REF>", "DP=30", "GT:DP", "0/1:25");
        let b = block(101, 110, "0/0:9,0:12:40:12:0,30,300");
        let (rows, parts, _fs, ds) = run(&[vec![Some(d)], vec![Some(b)]], &[Some(4)], None);
        assert_eq!(parts, vec![(20, 101, 102, 30, 0), (20, 103, 110, 12, 1)]);
        assert_eq!(rows.len(), 1);
        assert_eq!(
            (rows[0].start, rows[0].end, rows[0].dp, rows[0].sc),
            (103, 110, 12, 1)
        );
        assert_eq!(rows[0].ad_1, 9);
        assert_eq!(rows[0].pl_2, 30);
        let spans = ds[0].as_ref().unwrap().spans.spans.borrow().clone();
        assert_eq!(spans, vec![101, 102]);
    }

    #[test]
    fn split_dp_falls_back_to_format_dp() {
        let d = var(100, "ATT", "A,<NON_REF>", ".", "GT:DP", "0/1:25");
        let b = block(101, 110, "0/0:9,0:12:40:12:0,30,300");
        let (_rows, parts, _fs, _ds) = run(&[vec![Some(d)], vec![Some(b)]], &[Some(4)], None);
        assert_eq!(parts[0], (20, 101, 102, 25, 0));
    }

    #[test]
    fn block_inside_the_deletion_has_no_row() {
        let d = var(100, "ATTTT", "A,<NON_REF>", "DP=30", "GT", "1/1");
        let b = block(101, 103, "0/0:9,0:12:40:12:0,30,300");
        let (rows, parts, _fs, _ds) = run(&[vec![Some(d)], vec![Some(b)]], &[Some(4)], None);
        assert!(rows.is_empty());
        assert_eq!(parts, vec![(20, 101, 103, 30, 0)]);
    }

    #[test]
    fn no_call_block_counts_no_sample() {
        let b = block(101, 110, "./.:0,0:0:0:0:0,0,0");
        let (rows, parts, _fs, _ds) = run(&[vec![Some(b)]], &[Some(4)], None);
        assert_eq!(parts, vec![(20, 101, 110, 0, 0)]);
        assert_eq!(rows[0].sc, 0);
    }

    #[test]
    fn cross_shard_span_goes_to_the_foreign_map() {
        // The deletion's row is at 100, before the shard [101, 200].
        let d = var(100, "ATTT", "A,<NON_REF>", "DP=30", "GT", "0/1");
        let b = block(101, 110, "0/0:9,0:12:40:12:0,30,300");
        let (_rows, _parts, fs, ds) = run(
            &[vec![Some(d)], vec![Some(b)]],
            &[Some(4)],
            Some((101, 200)),
        );
        assert_eq!(fs.order, vec![(4, 20, 100)]);
        assert_eq!(fs.map[&(4, 20, 100)], vec![101, 103]);
        assert!(ds[0].as_ref().unwrap().spans.spans.borrow().is_empty());
    }

    #[test]
    fn block_after_own_variant_is_counted() {
        // Stream 0: block 90-99, SNV at 100, block 101-150; stream 1 a
        // long block 90-200. Every block of stream 0 is emitted once.
        let a1 = block(90, 99, "0/0:1,0:5:40:5:0,3,30");
        let snv = var(100, "A", "G,<NON_REF>", "DP=7", "GT", "0/1");
        let a2 = block(101, 150, "0/0:1,0:6:40:6:0,3,30");
        let long = block(90, 200, "0/0:1,0:9:40:9:0,3,30");
        let grps = vec![
            vec![Some(a1.clone()), Some(long.clone())],
            vec![Some(snv), Some(long.clone())],
            vec![Some(a2), Some(long)],
        ];
        let (rows, parts, _fs, _ds) = run(&grps, &[Some(1), Some(2)], None);
        assert_eq!(
            parts,
            vec![
                (20, 90, 99, 5, 1),
                (20, 90, 200, 9, 1),
                (20, 101, 150, 6, 1)
            ]
        );
        assert_eq!(rows.len(), 3);
    }

    #[test]
    fn stream_without_an_id_only_contributes() {
        let b = block(101, 110, "0/0:9,0:12:40:12:0,30,300");
        let (rows, parts, _fs, _ds) = run(&[vec![Some(b)]], &[None], None);
        assert!(rows.is_empty());
        assert_eq!(parts, vec![(20, 101, 110, 12, 1)]);
    }

    #[test]
    fn site_dp_rule() {
        let v = var(5, "A", "G", "DP=0", "GT:DP", "0/1:9");
        assert_eq!(record_site_dp(&v).unwrap(), 9); // INFO/DP 0 falls back
        let v = var(5, "A", "G", "DP=4", "GT:DP", "0/1:9");
        assert_eq!(record_site_dp(&v).unwrap(), 4);
        let v = var(5, "A", "G", ".", "GT", "0/1");
        assert_eq!(record_site_dp(&v).unwrap(), 0);
    }
    struct VecSource(std::vec::IntoIter<Rc<Record>>);

    impl crate::grouper::Source for VecSource {
        type Rec = Record;
        fn next_rec(&mut self) -> anyhow::Result<Option<Record>> {
            Ok(self.0.next().map(|r| (*r).clone()))
        }
    }

    fn rb(start: i64, end: i64, gq: i64, dp: i64, ad: (i64, i64)) -> Rc<Record> {
        crate::matrix::tests::rec(&format!(
            "chr21\t{start}\t.\tA\t<NON_REF>\t.\t.\tEND={end}\tGT:AD:GQ:MIN_DP:PL\t\
             0/0:{},{}:{gq}:{dp}:0,{gq},{}",
            ad.0,
            ad.1,
            10 * gq
        ))
    }

    /// `test_ref_block_banding.py`'s `collect`: the shard loop's ref-block
    /// path over whole streams, the bander closed at the end.
    fn collect(
        streams: Vec<Vec<Rc<Record>>>,
        sids: &[Option<i64>],
        spec: Option<&str>,
    ) -> (Vec<RbRow>, Vec<RbRange>) {
        let n = streams.len();
        let mut g = crate::grouper::Grouper::new(
            streams
                .into_iter()
                .map(|s| Some(VecSource(s.into_iter())))
                .collect(),
        )
        .unwrap();
        let mut bander = spec
            .map(|s| RefBlockBander::new(crate::banding::RefBandSpec::parse(s).unwrap(), 0, None));
        let (mut ds, mut last) = (vec![None; n], vec![-1; n]);
        let (mut rows, mut parts, mut fs, mut nb) =
            (vec![], vec![], ForeignSpans::default(), vec![]);
        while g.next_group().unwrap().is_some() {
            update_sample_del_state(&g.grp, sids, 20, &mut ds).unwrap();
            collect_sample_ref_block_data(
                &g.grp,
                sids,
                20,
                &mut last,
                &ds,
                Some((1, 1_000_000_000)),
                bander.as_mut(),
                RbOut {
                    rows: &mut rows,
                    new_blocks: &mut nb,
                    partials: &mut parts,
                    foreign: &mut fs,
                },
            )
            .unwrap();
        }
        if let Some(b) = bander.as_mut() {
            let (r, p) = b.close_all();
            rows.extend(r);
            parts.extend(p);
        }
        (rows, parts)
    }

    #[test]
    fn with_a_bander_the_split_rest_merges_with_the_next_block() {
        let del = var(
            11,
            "GAAA",
            "G,<NON_REF>",
            "DP=44",
            "GT:DP:GQ:PL",
            "0/1:40:35:30,0,400,60,420,480",
        );
        let stream = vec![
            rb(1, 10, 30, 12, (10, 0)),
            del,
            rb(12, 20, 35, 14, (14, 0)),
            rb(21, 30, 50, 11, (11, 1)),
        ];
        let (plain_rows, plain) = collect(vec![stream.clone()], &[Some(3)], None);
        let spans: Vec<(i64, i64)> = plain_rows.iter().map(|r| (r.start, r.end)).collect();
        assert_eq!(spans, vec![(1, 10), (15, 20), (21, 30)]);
        let (rows, mut parts) = collect(vec![stream], &[Some(3)], Some("20,60"));
        let row = |s, e, a1, a2, dp, p2, p3| RbRow {
            sample_id: 3,
            c_idx: 20,
            start: s,
            end: e,
            ad_1: a1,
            ad_2: a2,
            dp,
            pl_1: 0,
            pl_2: p2,
            pl_3: p3,
            sc: 1,
        };
        // dp (6 * 14 + 10 * 11 + 8) // 16 = 12; AD / PL of the GQ-35 block.
        assert_eq!(
            rows,
            vec![
                row(1, 10, 10, 0, 12, 30, 300),
                row(15, 30, 14, 0, 12, 35, 350)
            ]
        );
        let split = (20, 12, 14, 44, 0);
        let mut want = vec![split, (20, 1, 10, 12, 1), (20, 15, 30, 12, 1)];
        want.sort_unstable();
        parts.sort_unstable();
        assert_eq!(parts, want);
        assert!(plain.contains(&split));
    }

    #[test]
    fn a_bander_does_not_change_a_stream_without_a_sample_id() {
        let stream = vec![rb(1, 4, 30, 10, (10, 0)), rb(5, 9, 35, 10, (10, 0))];
        let (rows, parts) = collect(vec![stream], &[None], Some("all"));
        assert!(rows.is_empty());
        assert_eq!(parts, vec![(20, 1, 4, 10, 1), (20, 5, 9, 10, 1)]);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_without_a_bander_one_row_per_raw_block (K6).
    #[test]
    fn k6_without_a_bander_one_row_per_raw_block() {
        let stream = vec![
            rb(1, 4, 30, 10, (10, 0)),
            rb(5, 9, 35, 10, (10, 0)),
            rb(10, 12, 70, 10, (10, 0)),
        ];
        let (rows, parts) = collect(vec![stream], &[Some(7)], None);
        let spans: Vec<(i64, i64)> = rows.iter().map(|r| (r.start, r.end)).collect();
        assert_eq!(spans, vec![(1, 4), (5, 9), (10, 12)]);
        let ranges: Vec<RbRange> = rows
            .iter()
            .map(|r| (r.c_idx, r.start, r.end, r.dp, r.sc))
            .collect();
        assert_eq!(parts, ranges);
        let pls: Vec<(i64, i64, i64)> = rows.iter().map(|r| (r.pl_1, r.pl_2, r.pl_3)).collect();
        assert_eq!(pls, vec![(0, 30, 300), (0, 35, 350), (0, 70, 700)]);
    }

    // ── K6: test_removal_substrate.py's ref-block section, value for value
    // (chromosome index 0, as there).

    const C0: i64 = 0;

    /// The Python test's `block(start_1b, end_1b, gt="0/0", min_dp=10)`.
    fn blk0(start: i64, end: i64, gt: &str, min_dp: i64) -> Rc<Record> {
        crate::matrix::tests::rec(&format!(
            "chr21\t{start}\t.\tA\t<NON_REF>\t.\tPASS\tEND={end}\tGT:MIN_DP\t{gt}:{min_dp}"
        ))
    }

    /// The Python test's `_deletion(pos_1b, ref_len, info_dp=44, gt="0/1",
    /// fmt_dp=40)`.
    fn deletion0(pos: i64, ref_len: usize, info_dp: Option<i64>, fmt_dp: i64) -> Rc<Record> {
        let info = info_dp.map_or(".".to_string(), |d| format!("DP={d}"));
        crate::matrix::tests::rec(&format!(
            "chr21\t{pos}\t.\tG{}\tG,<NON_REF>\t30\tPASS\t{info}\tGT:DP\t0/1:{fmt_dp}",
            "A".repeat(ref_len - 1)
        ))
    }

    fn one_rows(
        v: Rc<Record>,
        ds: &[Option<DelState>],
        bounds: Option<(i64, i64)>,
    ) -> (Vec<RbRow>, Vec<RbRange>, ForeignSpans) {
        let (mut rows, mut parts, mut fs, mut nb) =
            (vec![], vec![], ForeignSpans::default(), vec![]);
        collect_sample_ref_block_data(
            &[Some(v)],
            &[Some(5)],
            C0,
            &mut [-1],
            ds,
            bounds,
            None,
            RbOut {
                rows: &mut rows,
                new_blocks: &mut nb,
                partials: &mut parts,
                foreign: &mut fs,
            },
        )
        .unwrap();
        (rows, parts, fs)
    }

    fn del_state(d: Rc<Record>) -> Vec<Option<DelState>> {
        let mut ds = vec![None];
        update_sample_del_state(&[Some(d)], &[Some(5)], C0, &mut ds).unwrap();
        ds
    }

    fn own_spans(ds: &[Option<DelState>]) -> Vec<i64> {
        ds[0].as_ref().unwrap().spans.spans.borrow().clone()
    }

    // Replaces tests/unit/test_removal_substrate.py::test_split_overlap_takes_site_dp_and_rest_keeps_the_block (K6).
    #[test]
    fn k6_split_overlap_takes_site_dp_and_rest_keeps_the_block() {
        let ds = del_state(deletion0(100, 4, Some(44), 40)); // deletes 101..103
        let d = ds[0].as_ref().unwrap();
        assert_eq!((d.del_end, d.del_dp), (103, 44)); // site dp: INFO/DP
        let (rows, parts, _) = one_rows(blk0(101, 120, "0/0", 31), &ds, None);
        assert_eq!(parts, vec![(C0, 101, 103, 44, 0), (C0, 104, 120, 31, 1)]);
        assert_eq!(
            rows,
            vec![RbRow {
                sample_id: 5,
                c_idx: C0,
                start: 104,
                end: 120,
                ad_1: 0,
                ad_2: 0,
                dp: 31,
                pl_1: 0,
                pl_2: 0,
                pl_3: 0,
                sc: 1,
            }]
        );
        assert_eq!(own_spans(&ds), vec![101, 103]);
    }

    // Replaces tests/unit/test_removal_substrate.py::test_split_dp_falls_back_to_format_dp (K6).
    #[test]
    fn k6_split_dp_falls_back_to_format_dp() {
        let dele = deletion0(100, 3, None, 27);
        assert_eq!(record_site_dp(&dele).unwrap(), 27);
        let ds = del_state(dele);
        assert_eq!(ds[0].as_ref().unwrap().del_dp, 27);
    }

    // Replaces tests/unit/test_removal_substrate.py::test_block_inside_the_deletion_has_no_row (K6).
    #[test]
    fn k6_block_inside_the_deletion_has_no_row() {
        let ds = del_state(deletion0(100, 6, Some(44), 40)); // del_end 105
        let (rows, parts, _) = one_rows(blk0(101, 103, "0/0", 10), &ds, None);
        assert!(rows.is_empty());
        assert_eq!(parts, vec![(C0, 101, 103, 44, 0)]);
        assert_eq!(own_spans(&ds), vec![101, 103]);
    }

    // Replaces tests/unit/test_removal_substrate.py::test_no_call_block_counts_no_sample (K6).
    #[test]
    fn k6_no_call_block_counts_no_sample() {
        let (rows, parts, _) = one_rows(blk0(10, 20, "./.", 0), &[None], None);
        assert_eq!(parts, vec![(C0, 10, 20, 0, 0)]);
        assert_eq!(rows[0].sc, 0);
    }

    // Replaces tests/unit/test_removal_substrate.py::test_cross_shard_span_goes_to_the_foreign_map (K6).
    #[test]
    fn k6_cross_shard_span_goes_to_the_foreign_map() {
        let ds = del_state(deletion0(100, 5, Some(44), 40)); // del_end 104
        let (_rows, parts, fs) = one_rows(blk0(101, 110, "0/0", 10), &ds, Some((102, 1000)));
        assert_eq!(fs.order, vec![(5, C0, 100)]);
        assert_eq!(fs.map.len(), 1);
        assert_eq!(fs.map[&(5, C0, 100)], vec![102, 104]);
        assert!(own_spans(&ds).is_empty());
        // The partial is unclipped here; the shard loop clips it.
        assert_eq!(parts[0], (C0, 101, 104, 44, 0));
    }

    // Replaces tests/unit/test_removal_substrate.py::test_merge_spans (K6).
    // Its `None` input case has no Rust counterpart (the argument is a
    // slice; the worker never passes None).
    #[test]
    fn k6_merge_spans() {
        assert_eq!(merge_spans(&[]), None);
        assert_eq!(merge_spans(&[5, 9]), Some(vec![5, 9]));
        assert_eq!(
            merge_spans(&[20, 22, 5, 9, 10, 12]),
            Some(vec![5, 12, 20, 22])
        );
        assert_eq!(merge_spans(&[5, 9, 7, 8, 9, 11]), Some(vec![5, 11]));
    }

    /// The Python test's `_ref_partials(streams, sample_ids, shard)`: per
    /// group the deletion state update and the first-sight emission, the
    /// contributions clipped to the shard and summed per position.
    fn ref_partials(
        streams: Vec<Vec<Rc<Record>>>,
        sids: &[Option<i64>],
        shard: (i64, i64),
    ) -> std::collections::BTreeMap<i64, (i64, i64)> {
        let n = streams.len();
        let mut g = crate::grouper::Grouper::new(
            streams
                .into_iter()
                .map(|s| Some(VecSource(s.into_iter())))
                .collect(),
        )
        .unwrap();
        let (mut ds, mut last) = (vec![None; n], vec![-1; n]);
        let mut segs = crate::windows::Segments::new(C0);
        while g.next_group().unwrap().is_some() {
            update_sample_del_state(&g.grp, sids, C0, &mut ds).unwrap();
            let (mut rows, mut parts, mut fs, mut nb) =
                (vec![], vec![], ForeignSpans::default(), vec![]);
            collect_sample_ref_block_data(
                &g.grp,
                sids,
                C0,
                &mut last,
                &ds,
                Some(shard),
                None,
                RbOut {
                    rows: &mut rows,
                    new_blocks: &mut nb,
                    partials: &mut parts,
                    foreign: &mut fs,
                },
            )
            .unwrap();
            for (_c, s, e, dp, sc) in parts {
                let (s, e) = (s.max(shard.0), e.min(shard.1));
                if s <= e {
                    segs.add(s, e, dp, sc);
                }
            }
        }
        let (p, dp, sc) = crate::windows::sum_segments(&segs);
        p.into_iter().zip(dp.into_iter().zip(sc)).collect()
    }

    /// The Python test's `_per_sample(*blocks)`.
    fn per_sample(blocks: &[(i64, i64, i64, i64)]) -> std::collections::BTreeMap<i64, (i64, i64)> {
        let mut out = std::collections::BTreeMap::new();
        for &(s, e, dp, sc) in blocks {
            for q in s..=e {
                let a: &mut (i64, i64) = out.entry(q).or_default();
                a.0 += dp;
                a.1 += sc;
            }
        }
        out.retain(|_, v| *v != (0, 0));
        out
    }

    const WHOLE: (i64, i64) = (1, 1_000_000_000);
    const IDS12: [Option<i64>; 2] = [Some(1), Some(2)];

    // Replaces tests/unit/test_removal_substrate.py::test_block_after_own_variant_is_counted (K6).
    #[test]
    fn k6_block_after_own_variant_is_counted() {
        let s0 = vec![
            blk0(1, 9, "0/0", 20),
            var(10, "A", "T,<NON_REF>", "DP=22", "GT", "0/1"),
            blk0(11, 40, "0/0", 21),
        ];
        let s1 = vec![blk0(1, 15, "0/0", 30), blk0(16, 40, "0/0", 31)];
        let got = ref_partials(vec![s0, s1], &IDS12, WHOLE);
        assert_eq!(
            got,
            per_sample(&[
                (1, 9, 20, 1),
                (11, 40, 21, 1),
                (1, 15, 30, 1),
                (16, 40, 31, 1)
            ])
        );
        assert_eq!(got[&12], (51, 2));
    }

    // Replaces tests/unit/test_removal_substrate.py::test_block_tail_after_a_gap_is_counted (K6).
    #[test]
    fn k6_block_tail_after_a_gap_is_counted() {
        let s0 = vec![blk0(1, 40, "0/0", 20), blk0(41, 45, "0/0", 22)];
        let s1 = vec![blk0(1, 15, "0/0", 30)];
        let got = ref_partials(vec![s0, s1], &IDS12, WHOLE);
        assert_eq!(
            got,
            per_sample(&[(1, 40, 20, 1), (41, 45, 22, 1), (1, 15, 30, 1)])
        );
    }

    // Replaces tests/unit/test_removal_substrate.py::test_split_inside_a_multi_sample_batch (K6).
    #[test]
    fn k6_split_inside_a_multi_sample_batch() {
        let s0 = vec![
            blk0(1, 9, "0/0", 20),
            deletion0(10, 4, Some(44), 40),
            blk0(11, 30, "0/0", 21),
        ];
        let s1 = vec![blk0(1, 30, "0/0", 30)];
        let got = ref_partials(vec![s0, s1], &IDS12, WHOLE);
        assert_eq!(
            got,
            per_sample(&[
                (1, 9, 20, 1),
                (11, 13, 44, 0),
                (14, 30, 21, 1),
                (1, 30, 30, 1)
            ])
        );
    }

    // Replaces tests/unit/test_removal_substrate.py::test_shards_add_up_to_the_whole (K6).
    #[test]
    fn k6_shards_add_up_to_the_whole() {
        let s0 = vec![
            blk0(1, 9, "0/0", 20),
            deletion0(10, 6, Some(44), 40),
            blk0(11, 30, "0/0", 21),
        ];
        let s1 = vec![blk0(1, 30, "0/0", 30)];
        let whole = ref_partials(vec![s0.clone(), s1.clone()], &IDS12, WHOLE);
        let left = ref_partials(vec![s0.clone(), s1.clone()], &IDS12, (1, 12));
        // A shard's readers yield the records overlapping it.
        let tail = |s: &[Rc<Record>]| -> Vec<Rc<Record>> {
            s.iter().filter(|r| r.end > 12).cloned().collect()
        };
        let right = ref_partials(vec![tail(&s0), tail(&s1)], &IDS12, (13, 1_000_000_000));
        assert!(left.keys().all(|k| !right.contains_key(k)));
        let mut both = left;
        both.extend(right);
        assert_eq!(both, whole);
    }
}
