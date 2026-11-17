// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! Position windows: `_iceberg_window`, `_split_range_to_windows`,
//! `RefBlockSegments`, `sum_ref_block_segments`,
//! `_bucket_ref_block_partials`, `_route_ref_block_output` and the
//! completeness test of the `_flush_*` functions (originally a port of
//! the Python ingest worker's; `RefBlockSegments` and
//! `sum_ref_block_segments` remain in `mosaic/ingest.py`, where
//! `mosaic remove` uses them).

use std::collections::BTreeMap;

use crate::ref_blocks::{RbRange, RbRow};

pub type WinKey = (i64, i64);

/// `_iceberg_window(c_idx, pos, pos_batch_size)`.
pub fn iceberg_window(c_idx: i64, pos: i64, pbs: i64) -> WinKey {
    if pbs > 0 {
        (c_idx, pos.div_euclid(pbs) * pbs)
    } else {
        (-1, 0)
    }
}

/// `_split_range_to_windows`: (window, (c_idx, seg_start, seg_end, dp,
/// sc)) per window the inclusive range touches.
pub fn split_range_to_windows(
    c_idx: i64,
    start: i64,
    end: i64,
    dp: i64,
    sc: i64,
    pbs: i64,
) -> Vec<(WinKey, RbRange)> {
    if pbs <= 0 {
        return vec![(
            iceberg_window(c_idx, start, pbs),
            (c_idx, start, end, dp, sc),
        )];
    }
    let mut out = Vec::new();
    let mut seg_start = start;
    while seg_start <= end {
        let win_start = seg_start.div_euclid(pbs) * pbs;
        let seg_end = end.min(win_start + pbs - 1);
        out.push(((c_idx, win_start), (c_idx, seg_start, seg_end, dp, sc)));
        seg_start = seg_end + 1;
    }
    out
}

/// `RefBlockSegments`: one window's inclusive ranges.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Segments {
    pub c_idx: i64,
    pub starts: Vec<i64>,
    pub ends: Vec<i64>,
    pub dp: Vec<i64>,
    pub sc: Vec<i64>,
}

impl Segments {
    pub fn new(c_idx: i64) -> Segments {
        Segments {
            c_idx,
            ..Segments::default()
        }
    }

    pub fn add(&mut self, start: i64, end: i64, dp: i64, sc: i64) {
        self.starts.push(start);
        self.ends.push(end);
        self.dp.push(dp);
        self.sc.push(sc);
    }

    pub fn len(&self) -> usize {
        self.starts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.starts.is_empty()
    }
}

/// `sum_ref_block_segments`: (positions, dp, sample_counts) for every
/// position whose sum is not (0, 0), ascending; an i64 difference array
/// over the span (the Python bincounts are exact in float64 there).
pub fn sum_segments(seg: &Segments) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    if seg.is_empty() {
        return (vec![], vec![], vec![]);
    }
    let lo = *seg.starts.iter().min().expect("non-empty");
    let hi = *seg.ends.iter().max().expect("non-empty");
    let span = (hi - lo + 1) as usize;
    let mut ddp = vec![0i64; span + 1];
    let mut dsc = vec![0i64; span + 1];
    for i in 0..seg.len() {
        let a = (seg.starts[i] - lo) as usize;
        let b = (seg.ends[i] - lo + 1) as usize;
        ddp[a] += seg.dp[i];
        ddp[b] -= seg.dp[i];
        dsc[a] += seg.sc[i];
        dsc[b] -= seg.sc[i];
    }
    let mut pos = Vec::new();
    let mut dp = Vec::new();
    let mut sc = Vec::new();
    let (mut cd, mut cs) = (0i64, 0i64);
    for j in 0..span {
        cd += ddp[j];
        cs += dsc[j];
        if cd != 0 || cs != 0 {
            pos.push(lo + j as i64);
            dp.push(cd);
            sc.push(cs);
        }
    }
    (pos, dp, sc)
}

/// The flushes' completeness test: `c_idx < flush_c_idx`, or the same
/// chromosome and `win_start + pos_batch_size <= flush_pos`. `None` is the
/// shard end (`float("inf")`): every window.
pub fn is_complete(w: WinKey, flush: Option<(i64, i64)>, pbs: i64) -> bool {
    match flush {
        None => true,
        Some((fc, fp)) => w.0 < fc || (w.0 == fc && w.1 + pbs <= fp),
    }
}

/// The shard's window buffers.
#[derive(Default)]
pub struct WindowBufs {
    pub rb: BTreeMap<WinKey, Vec<RbRow>>,
    pub si: BTreeMap<WinKey, Vec<crate::partials::SiOut>>,
    pub brb: BTreeMap<WinKey, Segments>,
}

/// `_bucket_ref_block_partials`: clip to the shard, split into windows.
/// Returns the ranges kept.
pub fn bucket_partials(
    partials: &[RbRange],
    shard: (i64, i64),
    pbs: i64,
    brb: &mut BTreeMap<WinKey, Segments>,
) -> i64 {
    let mut kept = 0;
    for &(c_idx, start, end, dp, sc) in partials {
        let start = start.max(shard.0);
        let end = end.min(shard.1);
        if start > end {
            continue;
        }
        for (wk, (_c, s, e, w_dp, w_sc)) in split_range_to_windows(c_idx, start, end, dp, sc, pbs) {
            brb.entry(wk)
                .or_insert_with(|| Segments::new(c_idx))
                .add(s, e, w_dp, w_sc);
        }
        kept += 1;
    }
    kept
}

/// `_route_ref_block_output`: the ranges through `bucket_partials`, the
/// rows clipped to the shard and bucketed by the window of their clipped
/// start. Returns the ranges kept.
pub fn route_ref_block_output(
    rows: &[RbRow],
    partials: &[RbRange],
    shard: (i64, i64),
    pbs: i64,
    bufs: &mut WindowBufs,
) -> i64 {
    let kept = if partials.is_empty() {
        0
    } else {
        bucket_partials(partials, shard, pbs, &mut bufs.brb)
    };
    for r in rows {
        let cs = r.start.max(shard.0);
        let ce = r.end.min(shard.1);
        if cs <= ce {
            let wk = iceberg_window(r.c_idx, cs, pbs);
            bufs.rb.entry(wk).or_default().push(RbRow {
                start: cs,
                end: ce,
                ..*r
            });
        }
    }
    kept
}

#[cfg(test)]
mod tests {
    //! `test_ref_block_windows.py` / `test_ref_window_fold.py`'s split and
    //! segment-sum cases.
    use super::*;

    #[test]
    fn split_boundaries() {
        let starts = |s, e, pbs| -> Vec<i64> {
            split_range_to_windows(3, s, e, 5, 1, pbs)
                .iter()
                .map(|(w, _)| w.1)
                .collect()
        };
        assert_eq!(starts(1, 99, 100), vec![0]);
        assert_eq!(starts(99, 100, 100), vec![0, 100]);
        assert_eq!(starts(100, 100, 100), vec![100]);
        assert_eq!(starts(50, 350, 100), vec![0, 100, 200, 300]);
        let segs = split_range_to_windows(3, 50, 350, 5, 1, 100);
        let ranges: Vec<(i64, i64)> = segs.iter().map(|(_, r)| (r.1, r.2)).collect();
        assert_eq!(ranges, vec![(50, 99), (100, 199), (200, 299), (300, 350)]);
        assert_eq!(
            split_range_to_windows(3, 7, 900, 2, 1, 0),
            vec![((-1, 0), (3, 7, 900, 2, 1))]
        );
    }

    #[test]
    fn sum_of_overlapping_ranges_and_the_net_rule() {
        let mut s = Segments::new(0);
        s.add(10, 14, 3, 1);
        s.add(12, 20, 4, 1);
        s.add(30, 31, 0, 0); // (0, 0): not kept
        s.add(40, 41, 5, 0);
        s.add(40, 41, -5, 0); // nets to (0, 0)
        s.add(50, 50, -2, -1);
        let (p, dp, sc) = sum_segments(&s);
        let mut want_p = vec![];
        let mut want_dp = vec![];
        for x in 10..=20 {
            want_p.push(x);
            want_dp.push(if x < 12 {
                3
            } else if x <= 14 {
                7
            } else {
                4
            });
        }
        want_p.push(50);
        want_dp.push(-2);
        assert_eq!(p, want_p);
        assert_eq!(dp, want_dp);
        assert_eq!(sc[2], 2);
        assert_eq!(*sc.last().unwrap(), -1);
        assert_eq!(sum_segments(&Segments::new(0)), (vec![], vec![], vec![]));
    }

    #[test]
    fn complete_windows() {
        assert!(is_complete((1, 0), Some((2, 5)), 100));
        assert!(is_complete((2, 0), Some((2, 100)), 100));
        assert!(!is_complete((2, 0), Some((2, 99)), 100));
        assert!(is_complete((2, 900), None, 100));
    }

    // ── K6: test_ref_block_windows.py's split cases, value for value ──

    fn positions(segs: &[(WinKey, RbRange)]) -> Vec<i64> {
        segs.iter().flat_map(|(_, r)| r.1..=r.2).collect()
    }

    // Replaces tests/unit/test_ref_block_windows.py::test_split_boundaries (K6).
    #[test]
    fn k6_split_boundaries() {
        // (c_idx, start, end, pbs, expected window starts)
        let cases: [(i64, i64, i64, i64, &[i64]); 7] = [
            (3, 110, 150, 100, &[100]),
            (3, 100, 150, 100, &[100]),
            (3, 110, 199, 100, &[100]),
            (3, 110, 200, 100, &[100, 200]),
            (3, 150, 350, 100, &[100, 200, 300]),
            (3, 200, 200, 100, &[200]),
            (3, 199, 199, 100, &[100]),
        ];
        for (c_idx, start, end, pbs, win_starts) in cases {
            let segs = split_range_to_windows(c_idx, start, end, 7, 2, pbs);
            let keys: Vec<WinKey> = segs.iter().map(|(w, _)| *w).collect();
            let want: Vec<WinKey> = win_starts.iter().map(|&ws| (c_idx, ws)).collect();
            assert_eq!(keys, want, "{start}-{end}");
            for (wk, (sc_idx, s, e, dp, sc)) in &segs {
                assert_eq!(s.div_euclid(pbs), e.div_euclid(pbs));
                assert_eq!(*wk, iceberg_window(c_idx, *s, pbs));
                assert_eq!(*wk, (c_idx, s.div_euclid(pbs) * pbs));
                assert_eq!((*sc_idx, *dp, *sc), (c_idx, 7, 2));
            }
            assert_eq!(positions(&segs), (start..=end).collect::<Vec<_>>());
        }
    }

    // Replaces tests/unit/test_ref_block_windows.py::test_split_pos_batch_size_zero_is_whole_range (K6).
    #[test]
    fn k6_split_pos_batch_size_zero_is_whole_range() {
        let segs = split_range_to_windows(5, 50, 500, 9, 3, 0);
        assert_eq!(segs, vec![((-1, 0), (5, 50, 500, 9, 3))]);
        assert_eq!(segs[0].0, iceberg_window(5, 50, 0));
        assert_eq!(positions(&segs), (50..=500).collect::<Vec<_>>());
    }

    // Replaces tests/unit/test_ref_block_windows.py::test_split_then_sum_equals_direct_sum (K6).
    // The same pbs values, ranges and checks; Python's Mersenne Twister
    // (random.Random(1234 + pbs)) is replaced by a fixed-seed xorshift over
    // the same distributions (c_idx 0..=4, start 1..=200,000, length
    // 0..=5,000, dp 0..=500, sc 0..=100, (0, 0) bumped to sc 1).
    #[test]
    fn k6_split_then_sum_equals_direct_sum() {
        for pbs in [1i64, 7, 100, 25_000, 100_000] {
            let mut x: u64 = 0x9E37_79B9_7F4A_7C15 ^ (1234 + pbs as u64);
            let mut randint = |lo: i64, hi: i64| -> i64 {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                lo + (x % (hi - lo + 1) as u64) as i64
            };
            for _ in 0..200 {
                let c_idx = randint(0, 4);
                let start = randint(1, 200_000);
                let end = start + randint(0, 5_000);
                let dp = randint(0, 500);
                let mut sc = randint(0, 100);
                if dp == 0 && sc == 0 {
                    sc = 1;
                }
                let mut whole = Segments::new(c_idx);
                whole.add(start, end, dp, sc);
                let direct = sum_segments(&whole);
                let segs = split_range_to_windows(c_idx, start, end, dp, sc, pbs);
                let mut by_window: BTreeMap<WinKey, Segments> = BTreeMap::new();
                for (wk, (c, s, e, d, n)) in &segs {
                    by_window
                        .entry(*wk)
                        .or_insert_with(|| Segments::new(*c))
                        .add(*s, *e, *d, *n);
                }
                let (mut p, mut d, mut n) = (vec![], vec![], vec![]);
                for seg in by_window.values() {
                    let (a, b, c) = sum_segments(seg);
                    p.extend(a);
                    d.extend(b);
                    n.extend(c);
                }
                assert_eq!((p, d.clone(), n.clone()), direct);
                assert_eq!(direct.0, (start..=end).collect::<Vec<_>>());
                assert!(d.iter().all(|&v| v == dp) && n.iter().all(|&v| v == sc));
            }
        }
    }

    // Replaces tests/unit/test_ref_block_windows.py::test_split_segments_are_contiguous_and_disjoint (K6).
    #[test]
    fn k6_split_segments_are_contiguous_and_disjoint() {
        let segs = split_range_to_windows(2, 137, 9_001, 1, 1, 1_000);
        let bounds: Vec<(i64, i64)> = segs.iter().map(|(_, r)| (r.1, r.2)).collect();
        assert_eq!(bounds[0].0, 137);
        assert_eq!(bounds.last().unwrap().1, 9_001);
        for w in bounds.windows(2) {
            assert_eq!(w[1].0, w[0].1 + 1);
        }
    }
}
