// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! `bulk_upload` (named after the Python worker's function it was
//! ported from): the shard's batch_site_info /
//! batch_allele_counts partial rows and its per-sample site-info rows by
//! window.

use std::collections::{BTreeMap, HashMap, HashSet};

use crate::ref_blocks::merge_spans;
use crate::sites::{SiteInfo, SiteRow};
use crate::windows::{WinKey, iceberg_window};

/// A `sample_site_info` row as the local file holds it.
#[derive(Debug, Clone, PartialEq)]
pub struct SiOut {
    pub sample_id: i64,
    pub c_idx: i64,
    pub pos: i64,
    pub del_start: Option<i64>,
    pub del_end: Option<i64>,
    pub alleles: Vec<String>,
    pub gt_1: Option<i64>,
    pub gt_2: Option<i64>,
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
    pub del_ref_spans: Option<Vec<i64>>,
}

/// A `batch_site_info` row (without batch_id).
#[derive(Debug, Clone, PartialEq)]
pub struct BsiRow {
    pub c_idx: i64,
    pub pos: i64,
    pub si: SiteInfo,
    pub n_records: i64,
}

/// A `batch_allele_counts` row (without batch_id).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BacRow {
    pub c_idx: i64,
    pub pos: i64,
    pub allele: String,
    pub count: i64,
    pub n_listed: i64,
}

/// One buffered variant site.
pub struct SiteBuf {
    pub c_idx: i64,
    pub pos: i64,
    pub alleles: Vec<(String, i64)>,
    pub site_info: SiteInfo,
    pub rows: Vec<SiteRow>,
}

/// `bulk_upload(...)`: appends to `bsi`, `bac` and the per-sample rows'
/// windows.
pub fn bulk_upload(
    sites: &[SiteBuf],
    pbs: i64,
    si_by_window: &mut BTreeMap<WinKey, Vec<SiOut>>,
    bsi: &mut Vec<BsiRow>,
    bac: &mut Vec<BacRow>,
) {
    for s in sites {
        let mut listed: HashMap<&str, i64> = HashMap::new();
        for r in &s.rows {
            let set: HashSet<&str> = r.alleles.iter().map(String::as_str).collect();
            for a in set {
                *listed.entry(a).or_insert(0) += 1;
            }
        }
        for (a, count) in &s.alleles {
            bac.push(BacRow {
                c_idx: s.c_idx,
                pos: s.pos,
                allele: a.clone(),
                count: *count,
                n_listed: listed.get(a.as_str()).copied().unwrap_or(0),
            });
        }
    }
    for s in sites {
        bsi.push(BsiRow {
            c_idx: s.c_idx,
            pos: s.pos,
            si: s.site_info,
            n_records: s.rows.len() as i64,
        });
    }
    for s in sites {
        for r in &s.rows {
            let (del_start, del_end) = match r.del_range {
                Some((a, b)) => (Some(a), Some(b - 1)),
                None => (None, None),
            };
            let spans = r
                .del_spans
                .as_ref()
                .and_then(|d| merge_spans(&d.spans.borrow()));
            si_by_window
                .entry(iceberg_window(s.c_idx, s.pos, pbs))
                .or_default()
                .push(SiOut {
                    sample_id: r.sample_id,
                    c_idx: s.c_idx,
                    pos: s.pos,
                    del_start,
                    del_end,
                    alleles: r.alleles.clone(),
                    gt_1: r.gt_1,
                    gt_2: r.gt_2,
                    dp: r.dp,
                    ad: r.ad.clone(),
                    f1r2: r.f1r2.clone(),
                    f2r1: r.f2r1.clone(),
                    sb: r.sb,
                    pl: r.pl.clone(),
                    qual: r.qual,
                    site_dp: r.site_dp,
                    mqsq_1: r.mqsq_1,
                    mqsq_2: r.mqsq_2,
                    gtc: r.gtc,
                    del_ref_spans: spans,
                });
        }
    }
}

#[cfg(test)]
mod tests {
    //! `test_allele_catalog.py`'s partial-row case.
    use super::*;
    use crate::matrix::tests::var;
    use crate::sites::{ParseStats, collect_variant_group_data};

    #[test]
    fn every_allele_gets_a_row_with_n_listed() {
        let a = var(100, "A", "G,<NON_REF>", "DP=5", "GT", "0/1");
        let b = var(100, "A", "T,<NON_REF>", "DP=6", "GT", "0/0");
        let grp = vec![Some(a), Some(b)];
        let g = collect_variant_group_data(
            99,
            &grp,
            &mut vec![],
            &[Some(1), Some(2)],
            &mut ParseStats::default(),
            None,
            &mut vec![],
            &[None, None],
        )
        .unwrap()
        .unwrap();
        let sites = vec![SiteBuf {
            c_idx: 20,
            pos: 100,
            alleles: g.alleles,
            site_info: g.site_info,
            rows: g.rows,
        }];
        let (mut si, mut bsi, mut bac) = (BTreeMap::new(), vec![], vec![]);
        bulk_upload(&sites, 100_000, &mut si, &mut bsi, &mut bac);
        let got: Vec<(&str, i64, i64)> = bac
            .iter()
            .map(|r| (r.allele.as_str(), r.count, r.n_listed))
            .collect();
        assert_eq!(
            got,
            vec![("A", 3, 2), ("xG", 1, 1), ("*", 0, 2), ("xT", 0, 1)]
        );
        assert_eq!(bsi[0].n_records, 2);
        assert_eq!(si[&(20, 0)].len(), 2);
        assert_eq!(si[&(20, 0)][1].alleles, vec!["A", "xT", "*"]);
    }

    // Replaces tests/unit/test_allele_catalog.py::test_bulk_upload_writes_local_indices_and_every_allele (K6).
    #[test]
    fn bulk_upload_writes_local_indices_and_every_allele() {
        // Three samples at one site (0-based 100), two record shapes: a het
        // SNP, a half call on the deletion, a no call.
        let snp = var(101, "A", "T,<NON_REF>", ".", "GT:AD", "0/1:10,10,0");
        let dele = var(101, "AT", "A,<NON_REF>", ".", "GT:AD", "1/.:2,8,0");
        let nocall = var(101, "A", "T,<NON_REF>", ".", "GT", "./.");
        let grp = vec![Some(snp), Some(dele), Some(nocall)];
        let g = collect_variant_group_data(
            100,
            &grp,
            &mut vec![],
            &[Some(11), Some(12), Some(13)],
            &mut ParseStats::default(),
            None,
            &mut vec![],
            &[None, None, None],
        )
        .unwrap()
        .unwrap();
        let names: Vec<&str> = g.alleles.iter().map(|(a, _)| a.as_str()).collect();
        assert_eq!(names, vec!["A", "xT", "*", "-AT"]);
        let pos_1b = 101;
        let sites = vec![SiteBuf {
            c_idx: 0,
            pos: pos_1b,
            alleles: g.alleles,
            site_info: g.site_info,
            rows: g.rows,
        }];
        let (mut si, mut bsi, mut bac) = (BTreeMap::new(), vec![], vec![]);
        bulk_upload(&sites, 100_000, &mut si, &mut bsi, &mut bac);
        // Every listed allele has a partial row, the uncalled * with count
        // 0; n_listed counts the rows whose own list holds the allele.
        let got: Vec<(i64, i64, &str, i64, i64)> = bac
            .iter()
            .map(|r| (r.c_idx, r.pos, r.allele.as_str(), r.count, r.n_listed))
            .collect();
        assert_eq!(
            got,
            vec![
                (0, pos_1b, "A", 1, 3),
                (0, pos_1b, "xT", 1, 2),
                (0, pos_1b, "*", 0, 3),
                (0, pos_1b, "-AT", 1, 1),
            ]
        );
        assert_eq!(bsi.len(), 1);
        assert_eq!(bsi[0].n_records, 3);
        assert_eq!(si.len(), 1);
        let rows = si.values().next().unwrap();
        let by: HashMap<i64, &SiOut> = rows.iter().map(|r| (r.sample_id, r)).collect();
        let mut ids: Vec<i64> = by.keys().copied().collect();
        ids.sort_unstable();
        assert_eq!(ids, vec![11, 12, 13]);
        assert_eq!(by[&11].alleles, vec!["A", "xT", "*"]);
        assert_eq!(by[&12].alleles, vec!["A", "-AT", "*"]);
        assert_eq!(by[&13].alleles, vec!["A", "xT", "*"]);
        let gts = |s: i64| (by[&s].gt_1, by[&s].gt_2);
        assert_eq!(gts(11), (Some(0), Some(1)));
        assert_eq!(gts(12), (Some(1), None));
        assert_eq!(gts(13), (None, None));
        let dec = |s: i64| {
            let r = by[&s];
            [r.gt_1, r.gt_2].map(|g| g.map(|g| r.alleles[g as usize].clone()))
        };
        assert_eq!(dec(11), [Some("A".to_string()), Some("xT".to_string())]);
        assert_eq!(dec(12), [Some("-AT".to_string()), None]);
        assert_eq!(dec(13), [None, None]);
        // The deletion carrier's per-sample deletion span (the Python row's
        // [pos, pos + 1) as del_start / inclusive del_end).
        assert_eq!(
            (by[&12].del_start, by[&12].del_end),
            (Some(pos_1b), Some(pos_1b + 1))
        );
        assert_eq!(by[&12].ad, vec![2, 8, 0]); // AD as the record carries it
    }
}
