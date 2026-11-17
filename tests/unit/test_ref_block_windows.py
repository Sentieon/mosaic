# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the dense ref-block summation (``RefBlockSegments``,
``sum_ref_block_segments``; shared by ``mosaic remove``'s derive).

sum_ref_block_segments adds every sample's (overlapping) ranges of a window
per position: checked against a per-position Python sum, with (0, 0)
positions dropped as the partial always has.

The position-window split tests this file held earlier
(``_split_range_to_windows``, the earlier Python ingest worker's) are ``cargo test``s of the Rust worker's ``windows.rs``
(``k6_split_*``) with the same cases.
"""

import random

import pytest

from mosaic.ingest import RefBlockSegments, sum_ref_block_segments

pytestmark = pytest.mark.unit


def _summed_rows(ranges, batch_id):
    """Dense (batch_id, c_idx, pos, dp, sc) rows of the per-position sums of
    (c_idx, start, end, dp, sc) ranges (one chromosome), via
    sum_ref_block_segments."""
    if not ranges:
        return []
    segs = RefBlockSegments(ranges[0][0])
    for _c, start, end, dp, sc in ranges:
        segs.add(start, end, dp, sc)
    pos, dp, sc = sum_ref_block_segments(segs)
    return [
        (batch_id, segs.c_idx, int(p), int(d), int(c))
        for p, d, c in zip(pos, dp, sc)
    ]


def test_sum_of_overlapping_ranges_matches_per_position_sum():
    """Overlapping ranges of many samples (a window's segments) sum per
    position; a position whose sum is (0, 0) gets no row."""
    rng = random.Random(77)
    for _ in range(50):
        ranges = []
        for _r in range(rng.randint(1, 60)):
            start = rng.randint(1_000, 1_400)
            end = start + rng.randint(0, 80)
            ranges.append(
                (3, start, end, rng.choice([0, 0, 5, 31]), rng.randint(0, 1))
            )
        want = {}
        for _c, start, end, dp, sc in ranges:
            for q in range(start, end + 1):
                a = want.get(q, (0, 0))
                want[q] = (a[0] + dp, a[1] + sc)
        want_rows = [
            (4, 3, q, d, c)
            for q, (d, c) in sorted(want.items())
            if (d, c) != (0, 0)
        ]
        assert _summed_rows(ranges, 4) == want_rows


def test_sum_of_no_segments_is_empty():
    pos, dp, sc = sum_ref_block_segments(RefBlockSegments(0))
    assert len(pos) == len(dp) == len(sc) == 0
