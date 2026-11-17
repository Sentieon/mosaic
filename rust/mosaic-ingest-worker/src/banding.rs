// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! Ref-block banding: `RefBandSpec` (as `mosaic/ref_banding.py` defines
//! it) and `RefBlockBander` (originally a 1:1 port of the Python
//! bander): pending blocks keyed by (stream, chromosome), closed in their
//! insertion order.

use std::collections::HashMap;

use anyhow::{Result, bail};

use crate::matrix::GQ_CAP;
use crate::ref_blocks::{RbRange, RbRow};

pub const UNKNOWN_GQ_BAND: i64 = -1;
const NO_WINDOW_END: i64 = 1 << 62;

/// A GQ band spec: ascending exclusive upper bounds in [1, 99]; empty is
/// one band (`all`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RefBandSpec {
    pub bounds: Vec<i64>,
}

impl RefBandSpec {
    /// `RefBandSpec.parse`: `20,60` or `all`.
    pub fn parse(text: &str) -> Result<RefBandSpec> {
        // str.strip(): Unicode whitespace; the manifest holds the
        // normalized spec, so ASCII is what reaches here.
        let s = text.trim();
        if s == "all" {
            return Ok(RefBandSpec { bounds: vec![] });
        }
        if s.is_empty() {
            bail!("an empty GQ band spec");
        }
        let mut bounds = Vec::new();
        for item in s.split(',') {
            let item = item.trim();
            if item.is_empty() || !item.bytes().all(|c| c.is_ascii_digit()) {
                bail!("GQ band bound {item:?} is not an integer (spec {text:?})");
            }
            bounds.push(
                item.parse::<i64>().map_err(|_| {
                    anyhow::anyhow!("GQ band bound {item} is outside [1, {GQ_CAP}]")
                })?,
            );
        }
        for &b in &bounds {
            if !(1..=GQ_CAP).contains(&b) {
                bail!("GQ band bound {b} is outside [1, {GQ_CAP}]");
            }
        }
        if bounds.windows(2).any(|w| w[1] <= w[0]) {
            bail!("GQ band bounds must be strictly ascending (no repeat): {s}");
        }
        Ok(RefBandSpec { bounds })
    }

    /// The normalized text (Python's `RefBandSpec.text`).
    #[cfg(test)]
    pub fn text(&self) -> String {
        if self.bounds.is_empty() {
            return "all".into();
        }
        self.bounds
            .iter()
            .map(|b| b.to_string())
            .collect::<Vec<_>>()
            .join(",")
    }

    /// `band(gq)`: `bisect_right(bounds, gq)`, or UNKNOWN_GQ_BAND outside
    /// 0..99.
    pub fn band(&self, gq: i64) -> i64 {
        if !(0..=GQ_CAP).contains(&gq) {
            return UNKNOWN_GQ_BAND;
        }
        self.bounds.partition_point(|&b| b <= gq) as i64
    }
}

/// `_Pending`.
#[derive(Debug, Clone)]
struct Pending {
    sample_id: i64,
    c_idx: i64,
    start: i64,
    end: i64,
    window_end: i64,
    sc: i64,
    band: i64,
    dp_sum: i64,
    ad_1: i64,
    ad_2: i64,
    pl_1: i64,
    pl_2: i64,
    pl_3: i64,
    gq: i64,
    /// Insertion order in the Python dict.
    seq: u64,
}

impl Pending {
    fn dp(&self) -> i64 {
        let n = self.end - self.start + 1;
        (self.dp_sum + n.div_euclid(2)).div_euclid(n)
    }

    fn row(&self) -> RbRow {
        RbRow {
            sample_id: self.sample_id,
            c_idx: self.c_idx,
            start: self.start,
            end: self.end,
            ad_1: self.ad_1,
            ad_2: self.ad_2,
            dp: self.dp(),
            pl_1: self.pl_1,
            pl_2: self.pl_2,
            pl_3: self.pl_3,
            sc: self.sc,
        }
    }

    fn range(&self) -> RbRange {
        (self.c_idx, self.start, self.end, self.dp(), self.sc)
    }
}

/// `RefBlockBander`.
#[derive(Debug)]
pub struct RefBlockBander {
    pub spec: RefBandSpec,
    pub pos_batch_size: i64,
    pub shard_bounds: Option<(i64, i64)>,
    pending: HashMap<(usize, i64), Pending>,
    seq: u64,
}

fn close(p: &Pending, rows: &mut Vec<RbRow>, ranges: Option<&mut Vec<RbRange>>) {
    rows.push(p.row());
    if let Some(r) = ranges {
        r.push(p.range());
    }
}

impl RefBlockBander {
    pub fn new(
        spec: RefBandSpec,
        pos_batch_size: i64,
        shard_bounds: Option<(i64, i64)>,
    ) -> RefBlockBander {
        RefBlockBander {
            spec,
            pos_batch_size,
            shard_bounds,
            pending: HashMap::new(),
            seq: 0,
        }
    }

    fn window_end(&self, pos: i64) -> i64 {
        let pbs = self.pos_batch_size;
        if pbs <= 0 {
            return NO_WINDOW_END;
        }
        pos.div_euclid(pbs) * pbs + pbs - 1
    }

    /// `add(k, sample_id, c_idx, start, end, sc, gq, dp, ad_1, ad_2, pl_1,
    /// pl_2, pl_3, rows_out, ranges_out)`; `f` is (ad_1, ad_2, pl_1, pl_2,
    /// pl_3).
    #[allow(clippy::too_many_arguments)]
    pub fn add(
        &mut self,
        k: usize,
        sample_id: i64,
        c_idx: i64,
        start: i64,
        end: i64,
        sc: i64,
        gq: i64,
        dp: i64,
        f: [i64; 5],
        rows: &mut Vec<RbRow>,
        mut ranges: Option<&mut Vec<RbRange>>,
    ) {
        let (mut start, mut end) = (start, end);
        if let Some((lo, hi)) = self.shard_bounds {
            start = start.max(lo);
            end = end.min(hi);
            if start > end {
                return;
            }
        }
        let key = (k, c_idx);
        let band = self.spec.band(gq);
        if let Some(p) = self.pending.get_mut(&key) {
            if start == p.end + 1 && sc == p.sc && band == p.band && start <= p.window_end {
                let merged_end = end.min(p.window_end);
                p.dp_sum += dp * (merged_end - start + 1);
                p.end = merged_end;
                if gq < p.gq {
                    p.ad_1 = f[0];
                    p.ad_2 = f[1];
                    p.pl_1 = f[2];
                    p.pl_2 = f[3];
                    p.pl_3 = f[4];
                    p.gq = gq;
                }
                if end <= p.window_end {
                    return;
                }
                start = p.window_end + 1;
            }
            let p = self.pending.remove(&key).expect("present");
            close(&p, rows, ranges.as_deref_mut());
        }
        loop {
            let window_end = self.window_end(start);
            let part_end = end.min(window_end);
            let q = Pending {
                sample_id,
                c_idx,
                start,
                end: part_end,
                window_end,
                sc,
                band,
                dp_sum: dp * (part_end - start + 1),
                ad_1: f[0],
                ad_2: f[1],
                pl_1: f[2],
                pl_2: f[3],
                pl_3: f[4],
                gq,
                seq: 0,
            };
            if part_end == end {
                self.seq += 1;
                self.pending.insert(key, Pending { seq: self.seq, ..q });
                return;
            }
            close(&q, rows, ranges.as_deref_mut());
            start = part_end + 1;
        }
    }

    fn close_keys(&mut self, mut keys: Vec<(u64, (usize, i64))>) -> (Vec<RbRow>, Vec<RbRange>) {
        keys.sort_unstable();
        let mut rows = Vec::with_capacity(keys.len());
        let mut ranges = Vec::with_capacity(keys.len());
        for (_, key) in keys {
            let p = self.pending.remove(&key).expect("present");
            close(&p, &mut rows, Some(&mut ranges));
        }
        (rows, ranges)
    }

    /// `close_before(c_idx, head_1b)`.
    pub fn close_before(&mut self, c_idx: i64, head_1b: i64) -> (Vec<RbRow>, Vec<RbRange>) {
        if self.pos_batch_size <= 0 {
            return (vec![], vec![]);
        }
        let done: Vec<(u64, (usize, i64))> = self
            .pending
            .iter()
            .filter(|(_, p)| p.c_idx < c_idx || (p.c_idx == c_idx && p.window_end < head_1b))
            .map(|(key, p)| (p.seq, *key))
            .collect();
        self.close_keys(done)
    }

    /// `close_all()`.
    pub fn close_all(&mut self) -> (Vec<RbRow>, Vec<RbRange>) {
        let all: Vec<(u64, (usize, i64))> =
            self.pending.iter().map(|(key, p)| (p.seq, *key)).collect();
        self.close_keys(all)
    }
}

#[cfg(test)]
mod tests {
    //! `tests/unit/test_ref_block_banding.py`'s spec and bander cases.
    use super::*;

    const F: [i64; 5] = [1, 0, 0, 3, 30];

    fn spec(s: &str) -> RefBandSpec {
        RefBandSpec::parse(s).unwrap()
    }

    #[test]
    fn spec_parse_and_errors() {
        assert_eq!(spec("20,60").bounds, vec![20, 60]);
        assert_eq!(spec(" 20 , 60 ").text(), "20,60");
        assert_eq!(spec("all").text(), "all");
        assert_eq!(spec("1").bounds, vec![1]);
        for bad in ["", "60,20", "20,20", "0", "100", "2a", "1.5", "-3"] {
            assert!(RefBandSpec::parse(bad).is_err(), "{bad}");
        }
    }

    #[test]
    fn band_boundaries() {
        let s = spec("20,60");
        for (gq, b) in [
            (0, 0),
            (19, 0),
            (20, 1),
            (59, 1),
            (60, 2),
            (99, 2),
            (255, -1),
            (-1, -1),
        ] {
            assert_eq!(s.band(gq), b, "{gq}");
        }
        assert_eq!(spec("all").band(5), 0);
        assert_eq!(spec("all").band(255), UNKNOWN_GQ_BAND);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_band_boundaries and
    // ::test_band_all_and_single_bound (K6: RefBandSpec.band() went with the
    // Python bander); band_boundaries above has the rest of their cases.
    #[test]
    fn k6_band_python_cases() {
        let s = spec("20,60");
        for (gq, b) in [(0, 0), (19, 0), (20, 1), (59, 1), (60, 2), (98, 2), (99, 2)] {
            assert_eq!(s.band(gq), b, "{gq}");
        }
        assert_eq!(s.band(255), UNKNOWN_GQ_BAND);
        let one = spec("all");
        assert!((0..100).all(|g| one.band(g) == 0));
        assert_eq!(one.band(255), UNKNOWN_GQ_BAND);
        let single = spec("20");
        let got: Vec<i64> = [0, 19, 20, 99].iter().map(|&g| single.band(g)).collect();
        assert_eq!(got, vec![0, 0, 1, 1]);
        assert!((0..100).all(|g| single.band(g) != UNKNOWN_GQ_BAND));
    }

    fn bander(s: &str, pbs: i64) -> RefBlockBander {
        RefBlockBander::new(spec(s), pbs, None)
    }

    #[allow(clippy::too_many_arguments)]
    fn add(
        b: &mut RefBlockBander,
        k: usize,
        start: i64,
        end: i64,
        sc: i64,
        gq: i64,
        dp: i64,
        f: [i64; 5],
        rows: &mut Vec<RbRow>,
        ranges: &mut Vec<RbRange>,
    ) {
        b.add(k, 9, 0, start, end, sc, gq, dp, f, rows, Some(ranges));
    }

    #[test]
    fn merges_and_breaks() {
        let (mut rows, mut ranges) = (vec![], vec![]);
        let mut b = bander("20,60", 0);
        add(&mut b, 0, 1, 5, 1, 30, 10, F, &mut rows, &mut ranges);
        add(&mut b, 0, 6, 9, 1, 40, 10, F, &mut rows, &mut ranges); // merges
        add(&mut b, 0, 11, 12, 1, 40, 10, F, &mut rows, &mut ranges); // gap
        add(&mut b, 0, 13, 14, 0, 40, 10, F, &mut rows, &mut ranges); // class
        add(&mut b, 0, 15, 16, 0, 10, 10, F, &mut rows, &mut ranges); // band
        add(&mut b, 1, 17, 18, 0, 10, 10, F, &mut rows, &mut ranges); // stream
        let (r2, g2) = b.close_all();
        rows.extend(r2);
        ranges.extend(g2);
        let spans: Vec<(i64, i64)> = rows.iter().map(|r| (r.start, r.end)).collect();
        assert_eq!(spans, vec![(1, 9), (11, 12), (13, 14), (15, 16), (17, 18)]);
        assert_eq!(ranges.len(), rows.len());
    }

    #[test]
    fn merged_values_mean_and_lowest_gq() {
        let (mut rows, mut ranges) = (vec![], vec![]);
        let mut b = bander("all", 0);
        // dp 10 over 3 and 13 over 2: (30 + 26 + 2) // 5 = 11.
        add(
            &mut b,
            0,
            1,
            3,
            1,
            50,
            10,
            [1, 1, 1, 1, 1],
            &mut rows,
            &mut ranges,
        );
        add(
            &mut b,
            0,
            4,
            5,
            1,
            20,
            13,
            [2, 2, 2, 2, 2],
            &mut rows,
            &mut ranges,
        );
        add(
            &mut b,
            0,
            6,
            6,
            1,
            20,
            11,
            [3, 3, 3, 3, 3],
            &mut rows,
            &mut ranges,
        ); // tie: earliest
        let (r, g) = b.close_all();
        // (30 + 26 + 11 + 3) // 6 = 11.
        assert_eq!(r[0].dp, 11);
        assert_eq!(r[0].ad_1, 2);
        assert_eq!(g[0], (0, 1, 6, 11, 1));
        // Rounding half up: dp 1 and 2 over one each: (3 + 1) // 2 = 2.
        let mut b = bander("all", 0);
        add(&mut b, 0, 1, 1, 1, 50, 1, F, &mut rows, &mut ranges);
        add(&mut b, 0, 2, 2, 1, 50, 2, F, &mut rows, &mut ranges);
        assert_eq!(b.close_all().0[0].dp, 2);
    }

    #[test]
    fn unknown_gq_is_its_own_band_and_window_cut() {
        let (mut rows, mut ranges) = (vec![], vec![]);
        let mut b = bander("all", 10);
        add(&mut b, 0, 1, 5, 1, 255, 4, F, &mut rows, &mut ranges);
        add(&mut b, 0, 6, 8, 1, 30, 4, F, &mut rows, &mut ranges);
        assert_eq!(rows.len(), 1); // the unknown block closed
        // A raw block over two windows: the first part closes at once.
        add(&mut b, 0, 9, 25, 1, 30, 4, F, &mut rows, &mut ranges);
        let spans: Vec<(i64, i64)> = rows.iter().map(|r| (r.start, r.end)).collect();
        assert_eq!(spans, vec![(1, 5), (6, 9), (10, 19)]);
        // close_before: the window [20, 29] is not behind head 25.
        assert!(b.close_before(0, 25).0.is_empty());
        let (r, _) = b.close_before(0, 30);
        assert_eq!((r[0].start, r[0].end), (20, 25));
    }

    #[test]
    fn parts_are_clipped_to_the_shard_and_insertion_order_kept() {
        let mut b = RefBlockBander::new(spec("all"), 0, Some((10, 20)));
        let (mut rows, mut ranges) = (vec![], vec![]);
        b.add(1, 7, 0, 1, 12, 1, 30, 6, F, &mut rows, Some(&mut ranges));
        b.add(0, 8, 0, 15, 30, 1, 30, 6, F, &mut rows, Some(&mut ranges));
        b.add(1, 7, 0, 25, 30, 1, 30, 6, F, &mut rows, Some(&mut ranges)); // outside
        let (r, _) = b.close_all();
        assert_eq!(
            r.iter()
                .map(|r| (r.sample_id, r.start, r.end))
                .collect::<Vec<_>>(),
            vec![(7, 10, 12), (8, 15, 20)]
        );
    }

    // ── K6: test_ref_block_banding.py's bander cases, value for value ──

    const SID: i64 = 7;
    const C: i64 = 0;
    /// `mosaic.genotypes.GQ_UNKNOWN`.
    const GQ_UNKNOWN: i64 = 255;

    /// The Python test's `blk(start, end, gq, dp=10, sc=1, ad=(5, 0),
    /// pl=(0, 30, 300), k=0, c=C)`: sample id SID + k.
    #[derive(Clone, Copy)]
    struct Blk {
        k: usize,
        c: i64,
        start: i64,
        end: i64,
        sc: i64,
        gq: i64,
        dp: i64,
        f: [i64; 5],
    }

    fn blk(start: i64, end: i64, gq: i64) -> Blk {
        Blk {
            k: 0,
            c: C,
            start,
            end,
            sc: 1,
            gq,
            dp: 10,
            f: [5, 0, 0, 30, 300],
        }
    }

    impl Blk {
        fn dp(self, dp: i64) -> Blk {
            Blk { dp, ..self }
        }
        fn sc(self, sc: i64) -> Blk {
            Blk { sc, ..self }
        }
        fn k(self, k: usize) -> Blk {
            Blk { k, ..self }
        }
        fn c(self, c: i64) -> Blk {
            Blk { c, ..self }
        }
        fn adpl(self, ad: (i64, i64), pl: (i64, i64, i64)) -> Blk {
            Blk {
                f: [ad.0, ad.1, pl.0, pl.1, pl.2],
                ..self
            }
        }
        fn add_to(&self, b: &mut RefBlockBander, rows: &mut Vec<RbRow>, ranges: &mut Vec<RbRange>) {
            b.add(
                self.k,
                SID + self.k as i64,
                self.c,
                self.start,
                self.end,
                self.sc,
                self.gq,
                self.dp,
                self.f,
                rows,
                Some(ranges),
            );
        }
    }

    /// A row as the Python tuple: (sample_id, c, start, end, ad_1, ad_2,
    /// dp, pl_1, pl_2, pl_3, sample_counts).
    fn tup(r: &RbRow) -> [i64; 11] {
        [
            r.sample_id,
            r.c_idx,
            r.start,
            r.end,
            r.ad_1,
            r.ad_2,
            r.dp,
            r.pl_1,
            r.pl_2,
            r.pl_3,
            r.sc,
        ]
    }

    /// The Python test's `band(spec, blocks, pbs, close)`.
    fn band_run(
        s: &str,
        blocks: &[Blk],
        pbs: i64,
        close_all: bool,
    ) -> (Vec<RbRow>, Vec<RbRange>, RefBlockBander) {
        let mut b = RefBlockBander::new(spec(s), pbs, None);
        let (mut rows, mut ranges) = (vec![], vec![]);
        for x in blocks {
            x.add_to(&mut b, &mut rows, &mut ranges);
        }
        if close_all {
            let (r, g) = b.close_all();
            rows.extend(r);
            ranges.extend(g);
            assert_eq!(b.pending.len(), 0);
        }
        (rows, ranges, b)
    }

    fn spans(rows: &[RbRow]) -> Vec<(i64, i64)> {
        rows.iter().map(|r| (r.start, r.end)).collect()
    }

    fn row_ranges(rows: &[RbRow]) -> Vec<RbRange> {
        rows.iter()
            .map(|r| (r.c_idx, r.start, r.end, r.dp, r.sc))
            .collect()
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_same_band_merges (K6).
    #[test]
    fn k6_same_band_merges() {
        let (rows, ranges, _) = band_run("20,60", &[blk(1, 10, 30), blk(11, 20, 45)], 0, true);
        assert_eq!(spans(&rows), vec![(1, 20)]);
        assert_eq!(ranges, row_ranges(&rows));
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_a_gap_breaks_a_run (K6).
    #[test]
    fn k6_a_gap_breaks_a_run() {
        let (rows, ranges, _) = band_run("all", &[blk(1, 10, 30), blk(12, 20, 30)], 0, true);
        assert_eq!(spans(&rows), vec![(1, 10), (12, 20)]);
        assert_eq!(ranges, row_ranges(&rows));
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_a_class_change_breaks_a_run (K6).
    #[test]
    fn k6_a_class_change_breaks_a_run() {
        let (rows, _, _) = band_run("all", &[blk(1, 10, 30), blk(11, 20, 30).sc(0)], 0, true);
        assert_eq!(spans(&rows), vec![(1, 10), (11, 20)]);
        assert_eq!(rows.iter().map(|r| r.sc).collect::<Vec<_>>(), vec![1, 0]);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_a_band_change_breaks_a_run (K6).
    #[test]
    fn k6_a_band_change_breaks_a_run() {
        let blocks = [blk(1, 10, 30), blk(11, 20, 70), blk(21, 30, 65)];
        assert_eq!(
            spans(&band_run("20,60", &blocks, 0, true).0),
            vec![(1, 10), (11, 30)]
        );
        assert_eq!(spans(&band_run("all", &blocks, 0, true).0), vec![(1, 30)]);
        assert_eq!(spans(&band_run("20", &blocks, 0, true).0), vec![(1, 30)]);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_streams_do_not_merge (K6).
    #[test]
    fn k6_streams_do_not_merge() {
        let (rows, _, _) = band_run("all", &[blk(1, 10, 30), blk(11, 20, 30).k(1)], 0, true);
        let mut got: Vec<(i64, i64, i64)> =
            rows.iter().map(|r| (r.sample_id, r.start, r.end)).collect();
        got.sort_unstable();
        assert_eq!(got, vec![(SID, 1, 10), (SID + 1, 11, 20)]);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_merged_values (K6).
    #[test]
    fn k6_merged_values() {
        // (4 * 12 + 5 * 15 + 3 * 9) / 12 = 12.5 -> 13; AD / PL of the
        // lowest-GQ constituent.
        let blocks = [
            blk(1, 4, 50).dp(12).adpl((12, 0), (0, 50, 500)),
            blk(5, 9, 31).dp(15).adpl((15, 1), (0, 31, 310)),
            blk(10, 12, 44).dp(9).adpl((9, 0), (0, 44, 440)),
        ];
        let (rows, ranges, _) = band_run("20,60", &blocks, 0, true);
        assert_eq!(
            rows.iter().map(tup).collect::<Vec<_>>(),
            vec![[SID, C, 1, 12, 15, 1, 13, 0, 31, 310, 1]]
        );
        assert_eq!(ranges, vec![(C, 1, 12, 13, 1)]);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_a_shared_dp_is_kept_exactly (K6).
    #[test]
    fn k6_a_shared_dp_is_kept_exactly() {
        for dp in [0, 1, 17, 250] {
            let blocks: Vec<Blk> = [(1, 1), (2, 8), (9, 11), (12, 61), (62, 62)]
                .iter()
                .map(|&(s, e)| blk(s, e, 30).dp(dp))
                .collect();
            let (rows, _, _) = band_run("all", &blocks, 0, true);
            assert_eq!(spans(&rows), vec![(1, 62)], "{dp}");
            assert_eq!(rows[0].dp, dp);
        }
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_mean_rounds_half_up (K6).
    #[test]
    fn k6_mean_rounds_half_up() {
        let cases: [(&[i64], i64); 6] = [
            (&[10, 11], 11),
            (&[11, 10], 11),
            (&[0, 1], 1),
            (&[10, 10, 11], 10),
            (&[10, 11, 11], 11),
            (&[3, 4, 4, 3], 4),
        ];
        for (dps, want) in cases {
            let blocks: Vec<Blk> = dps
                .iter()
                .enumerate()
                .map(|(i, &d)| blk(i as i64 + 1, i as i64 + 1, 30).dp(d))
                .collect();
            let (rows, ranges, _) = band_run("all", &blocks, 0, true);
            assert_eq!(
                rows.iter().map(|r| r.dp).collect::<Vec<_>>(),
                vec![want],
                "{dps:?}"
            );
            assert_eq!(ranges, row_ranges(&rows));
        }
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_mixed_length_run (K6).
    #[test]
    fn k6_mixed_length_run() {
        let (rows, _, _) = band_run(
            "all",
            &[blk(1, 1, 30).dp(30), blk(2, 100, 30).dp(10)],
            0,
            true,
        );
        assert_eq!((rows[0].start, rows[0].end, rows[0].dp), (1, 100, 10));
        let (rows, _, _) = band_run(
            "all",
            &[blk(1, 99, 30).dp(0), blk(100, 100, 30).dp(100)],
            0,
            true,
        );
        assert_eq!(rows[0].dp, 1);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_tie_keeps_the_earliest (K6).
    #[test]
    fn k6_tie_keeps_the_earliest() {
        let blocks = [
            blk(1, 4, 30).adpl((1, 0), (0, 30, 301)),
            blk(5, 9, 30).adpl((2, 0), (0, 30, 302)),
        ];
        let (rows, _, _) = band_run("all", &blocks, 0, true);
        assert_eq!((rows[0].ad_1, rows[0].ad_2, rows[0].pl_3), (1, 0, 301));
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_unknown_gq_is_a_band_of_its_own (K6).
    #[test]
    fn k6_unknown_gq_is_a_band_of_its_own() {
        let blocks = [
            blk(1, 4, GQ_UNKNOWN),
            blk(5, 9, GQ_UNKNOWN),
            blk(10, 12, 0),
            blk(13, 15, GQ_UNKNOWN),
        ];
        let (rows, _, _) = band_run("all", &blocks, 0, true);
        assert_eq!(spans(&rows), vec![(1, 9), (10, 12), (13, 15)]);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_run_split_at_a_window_end (K6).
    #[test]
    fn k6_run_split_at_a_window_end() {
        let blocks = [blk(90, 95, 30).dp(8), blk(96, 105, 40).dp(11)];
        let (rows, ranges, _) = band_run("all", &blocks, 100, true);
        assert_eq!(spans(&rows), vec![(90, 99), (100, 105)]);
        // (6 * 8 + 4 * 11) / 10 = 9.2: the merged part only up to the end.
        assert_eq!(rows.iter().map(|r| r.dp).collect::<Vec<_>>(), vec![9, 11]);
        assert_eq!(ranges, row_ranges(&rows));
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_parts_are_clipped_to_the_shard (K6).
    #[test]
    fn k6_parts_are_clipped_to_the_shard() {
        let mut b = RefBlockBander::new(spec("all"), 0, Some((100, 200)));
        let (mut rows, mut ranges) = (vec![], vec![]);
        blk(80, 99, 30)
            .dp(50)
            .add_to(&mut b, &mut rows, &mut ranges);
        blk(90, 110, 30)
            .dp(10)
            .k(1)
            .add_to(&mut b, &mut rows, &mut ranges);
        blk(111, 120, 30)
            .dp(20)
            .k(1)
            .add_to(&mut b, &mut rows, &mut ranges);
        blk(195, 230, 30)
            .dp(7)
            .k(2)
            .add_to(&mut b, &mut rows, &mut ranges);
        let (r, mut g) = b.close_all();
        let mut got: Vec<(i64, i64, i64)> = rows
            .iter()
            .chain(r.iter())
            .map(|x| (x.start, x.end, x.dp))
            .collect();
        got.sort_unstable();
        assert_eq!(got, vec![(100, 120, 15), (195, 200, 7)]);
        let mut want = row_ranges(&r);
        g.sort_unstable();
        want.sort_unstable();
        assert_eq!(g, want);
        let (unclipped, _, _) = band_run(
            "all",
            &[blk(90, 110, 30).dp(10), blk(111, 120, 30).dp(20)],
            0,
            true,
        );
        assert_eq!(unclipped[0].dp, 13);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_adjacent_blocks_in_two_windows_do_not_merge (K6).
    #[test]
    fn k6_adjacent_blocks_in_two_windows_do_not_merge() {
        let (rows, _, _) = band_run("all", &[blk(90, 99, 30), blk(100, 110, 30)], 100, true);
        assert_eq!(spans(&rows), vec![(90, 99), (100, 110)]);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_a_raw_block_spanning_windows (K6).
    #[test]
    fn k6_a_raw_block_spanning_windows() {
        let (rows, _, b) = band_run("20,60", &[blk(95, 105, 30)], 100, false);
        assert_eq!(spans(&rows), vec![(95, 99)]);
        assert_eq!(b.pending.len(), 1);
        let (rows, ranges, _) = band_run("20,60", &[blk(95, 105, 30)], 100, true);
        assert_eq!(spans(&rows), vec![(95, 99), (100, 105)]);
        assert_eq!(ranges, row_ranges(&rows));
        let (rows, _, b) = band_run("all", &[blk(50, 250, 30)], 100, false);
        assert_eq!(spans(&rows), vec![(50, 99), (100, 199)]);
        assert_eq!(b.pending.len(), 1);
        // Without a window, nothing is cut.
        let (rows, _, _) = band_run("all", &[blk(50, 250, 30)], 0, true);
        assert_eq!(spans(&rows), vec![(50, 250)]);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_close_before_emits_only_the_windows_behind_the_head (K6).
    #[test]
    fn k6_close_before_emits_only_the_windows_behind_the_head() {
        let mut b = RefBlockBander::new(spec("all"), 100, None);
        let (mut rows, mut ranges) = (vec![], vec![]);
        blk(10, 20, 30).add_to(&mut b, &mut rows, &mut ranges);
        blk(150, 160, 30)
            .k(1)
            .add_to(&mut b, &mut rows, &mut ranges);
        blk(5, 9, 30)
            .k(2)
            .c(C + 1)
            .add_to(&mut b, &mut rows, &mut ranges);
        assert!(rows.is_empty());
        assert_eq!(b.pending.len(), 3);
        // The head inside window [100, 199] of chromosome C: window [0, 99]
        // is complete, [100, 199] is not (and chromosome C + 1 is ahead).
        let (r, g) = b.close_before(C, 150);
        assert_eq!(spans(&r), vec![(10, 20)]);
        assert_eq!(g, row_ranges(&r));
        assert_eq!(b.pending.len(), 2);
        let (r, g) = b.close_before(C, 199);
        assert!(r.is_empty() && g.is_empty());
        let (r, _) = b.close_before(C, 200);
        assert_eq!(spans(&r), vec![(150, 160)]);
        // A chromosome behind the head: all of it.
        let (r, _) = b.close_before(C + 2, 1);
        assert_eq!(spans(&r), vec![(5, 9)]);
        assert_eq!(b.pending.len(), 0);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_close_before_leaves_the_current_window_mergeable (K6).
    #[test]
    fn k6_close_before_leaves_the_current_window_mergeable() {
        let mut b = RefBlockBander::new(spec("all"), 100, None);
        let (mut rows, mut ranges) = (vec![], vec![]);
        blk(100, 120, 30).add_to(&mut b, &mut rows, &mut ranges);
        let (r, g) = b.close_before(C, 121);
        assert!(r.is_empty() && g.is_empty());
        blk(121, 130, 30).add_to(&mut b, &mut rows, &mut ranges);
        let (r, _) = b.close_all();
        rows.extend(r);
        assert_eq!(spans(&rows), vec![(100, 130)]);
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_pos_batch_size_zero (K6).
    #[test]
    fn k6_pos_batch_size_zero() {
        let mut b = RefBlockBander::new(spec("all"), 0, None);
        let (mut rows, mut ranges) = (vec![], vec![]);
        blk(10, 2_000_000, 30).add_to(&mut b, &mut rows, &mut ranges);
        blk(2_000_001, 5_000_000, 70).add_to(&mut b, &mut rows, &mut ranges);
        assert!(rows.is_empty());
        let (r, g) = b.close_before(C, 1_000_000_000); // never window-complete
        assert!(r.is_empty() && g.is_empty());
        let (r, g) = b.close_before(C + 1, 1);
        assert!(r.is_empty() && g.is_empty());
        let (r, g) = b.close_all();
        assert_eq!(spans(&r), vec![(10, 5_000_000)]);
        assert_eq!(g, row_ranges(&r));
    }

    // Replaces tests/unit/test_ref_block_banding.py::test_ranges_out_none (K6).
    #[test]
    fn k6_ranges_out_none() {
        let mut b = RefBlockBander::new(spec("all"), 100, None);
        let mut rows = vec![];
        for (s, e) in [(10, 20), (30, 40)] {
            let x = blk(s, e, 30);
            b.add(
                x.k, SID, x.c, x.start, x.end, x.sc, x.gq, x.dp, x.f, &mut rows, None,
            );
        }
        assert_eq!(spans(&rows), vec![(10, 20)]);
    }
}
