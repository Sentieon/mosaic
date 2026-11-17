# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the ref_blocks window fold (``mosaic.aggregate._WindowFold``).

A ref_blocks window unit sums every covering batch's dense ``batch_ref_blocks``
rows for one (chromosome, position window) into numpy arrays indexed by
position, then emits the dense root (every position whose net sum is not
(0, 0)) and the sparse root (those that are variant sites). These tests pin
its arithmetic on synthetic record batches: overlapping batches, nulls, a
``dp = 0, sample_counts = 0`` row, inclusive window edges, an empty window,
the net (0, 0) rule (E3: a removal batch's negative partial cancelling a
position), the sparse gather, and agreement with the Arrow SUM-by-key reducer
the tier tree used.
"""

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import mosaic.aggregate as agg
from mosaic.aggregate import (
    DEFAULT_REF_SHARD_SIZE,
    SPECS,
    _BufferedSum,
    _dense_root_table,
    _flatten,
    _position_windows,
    _ref_sparse_result,
    _staging_table,
    _unflatten,
    _WindowFold,
    _write_dense_root,
)

pytestmark = pytest.mark.unit

REF = SPECS["ref_blocks"]


def _batch(rows):
    """A batch_ref_blocks-shaped record batch: (position, dp, sample_counts),
    dp / sample_counts int32 as ingest writes them."""
    return pa.record_batch(
        {
            "position": pa.array([r[0] for r in rows], pa.int32()),
            "dp": pa.array([r[1] for r in rows], pa.int32()),
            "sample_counts": pa.array([r[2] for r in rows], pa.int32()),
        }
    )


def _fold(lo, hi, batches):
    f = _WindowFold(lo, hi)
    for rows in batches:
        f.add(_batch(rows))
    return f


def _as_rows(triple):
    pos, dp, sc = triple
    return list(zip(pos.tolist(), dp.tolist(), sc.tolist()))


def test_two_batches_overlapping_positions_sum():
    f = _fold(
        100,
        199,
        [
            [(100, 10, 1), (101, 11, 1), (102, 12, 1)],
            [(101, 5, 2), (102, 6, 2), (150, 7, 2)],
        ],
    )
    assert _as_rows(f.rows()) == [
        (100, 10, 1),
        (101, 16, 3),
        (102, 18, 3),
        (150, 7, 2),
    ]
    assert f.n_live == 4


def test_batch_order_does_not_matter():
    a = [(100, 10, 1), (105, 3, 1)]
    b = [(105, 4, 2), (199, 1, 1)]
    assert _as_rows(_fold(100, 199, [a, b]).rows()) == _as_rows(
        _fold(100, 199, [b, a]).rows()
    )


def test_nulls_count_as_zero():
    f = _fold(1, 10, [[(3, None, 1), (4, 2, None), (5, None, None)]])
    assert _as_rows(f.rows()) == [(3, 0, 1), (4, 2, 0)]


def test_net_zero_position_has_no_root_row():
    """A position yields a root row iff its net (dp, sample_counts) is not
    (0, 0). Ingest's partials never hold a (0, 0) row (sum_ref_block_segments
    omits them), so for all-positive data this is the old "touched" rule."""
    f = _fold(1, 10, [[(7, 0, 0), (8, 0, 1), (9, 3, 0)]])
    assert _as_rows(f.rows()) == [(8, 0, 1), (9, 3, 0)]
    assert _as_rows(f.at(np.array([7, 8, 9]))) == [(8, 0, 1), (9, 3, 0)]
    assert f.n_live == 2


def test_negative_source_cancels_positions():
    """A removal batch's negative partial (the removed samples' coverage,
    negated): a position only they covered nets to (0, 0) and leaves the
    dense and the sparse root; a shared one keeps the rest's sum."""
    ingest = [(100, 30, 2), (101, 30, 2), (102, 12, 1)]
    removal = [(100, -12, -1), (102, -12, -1), (103, -5, 0)]
    other = [(103, 5, 0)]
    f = _fold(100, 199, [ingest, removal, other])
    assert _as_rows(f.rows()) == [(100, 18, 1), (101, 30, 2)]
    assert _as_rows(f.at(np.array([100, 102, 103]))) == [(100, 18, 1)]
    # Any order: the sum is exact.
    g = _fold(100, 199, [removal, other, ingest])
    assert _as_rows(g.rows()) == _as_rows(f.rows())


def test_untouched_positions_have_no_row_even_at_sites():
    f = _fold(1, 10, [[(2, 5, 1)]])
    assert _as_rows(f.rows()) == [(2, 5, 1)]
    assert _as_rows(f.at(np.array([1, 2, 3]))) == [(2, 5, 1)]


def test_window_edges_are_inclusive():
    f = _fold(1000, 1999, [[(1000, 1, 1), (1999, 2, 1)]])
    assert _as_rows(f.rows()) == [(1000, 1, 1), (1999, 2, 1)]
    assert _as_rows(f.at(np.array([999, 1000, 1999, 2000]))) == [
        (1000, 1, 1),
        (1999, 2, 1),
    ]


@pytest.mark.parametrize("position", [999, 2000])
def test_row_outside_window_is_an_error(position):
    """The scan's row filter keeps every row inside the window; a row outside
    it is a bug, not data to drop (or to wrap around with a negative index)."""
    f = _WindowFold(1000, 1999)
    with pytest.raises(ValueError, match="outside the window"):
        f.add(_batch([(1500, 1, 1), (position, 1, 1)]))


def test_empty_window():
    f = _WindowFold(1, 1_000_000)
    f.add(_batch([]))
    assert f.n_live == 0
    assert _as_rows(f.rows()) == []
    assert _as_rows(f.at(np.array([1, 500_000]))) == []


def test_single_base_window():
    f = _fold(42, 42, [[(42, 3, 1)], [(42, 4, 1)]])
    assert _as_rows(f.rows()) == [(42, 7, 2)]


def test_duplicate_positions_within_one_batch_sum():
    """np.bincount is duplicate-safe (plain fancy-index += is not)."""
    f = _fold(1, 10, [[(5, 1, 1), (5, 2, 1), (5, 3, 1), (6, 1, 1)]])
    assert _as_rows(f.rows()) == [(5, 6, 3), (6, 1, 1)]


def test_int32_partials_widen_to_int64():
    big = 2**31 - 1
    f = _fold(1, 2, [[(1, big, big)], [(1, big, big)], [(1, big, big)]])
    assert _as_rows(f.rows()) == [(1, 3 * big, 3 * big)]
    assert f.dp.dtype == np.int64 and f.sc.dtype == np.int64


def test_rows_slices_partition_the_dense_root():
    f = _fold(10, 109, [[(p, p, 1) for p in range(10, 110, 3)]])
    whole = _as_rows(f.rows())
    parts = []
    for start in range(0, 100, 7):
        parts += _as_rows(f.rows(start, start + 7))
    assert parts == whole


def test_sparse_gather_matches_site_filter():
    """at(sites) == the dense root restricted to site positions (the old
    leaf filter), and ignores sites outside the window."""
    rng = np.random.default_rng(11)
    lo, hi = 5_000, 14_999
    rows = [
        (int(p), int(rng.integers(0, 50)), int(rng.integers(0, 3)))
        for p in rng.choice(np.arange(lo, hi + 1), 3_000, replace=False)
    ]
    f = _fold(lo, hi, [sorted(rows)])
    sites = np.unique(rng.integers(lo - 1_000, hi + 1_000, 2_000))
    site_set = set(sites.tolist())
    want = [r for r in _as_rows(f.rows()) if r[0] in site_set]
    assert _as_rows(f.at(sites)) == want
    assert want  # non-vacuous


def test_matches_arrow_sum_by_key():
    """The fold equals the Arrow group-by SUM the ref_blocks tier tree ran,
    over random overlapping batches with nulls and duplicates."""
    rng = np.random.default_rng(5)
    lo, hi = 1, 20_000
    batches = []
    for _ in range(6):
        n = int(rng.integers(1, 4_000))
        pos = rng.integers(lo, hi + 1, n)
        dp = [None if x < 5 else int(x) for x in rng.integers(0, 200, n)]
        sc = [None if x == 0 else int(x) for x in rng.integers(0, 4, n)]
        batches.append(list(zip(pos.tolist(), dp, sc)))
    f = _fold(lo, hi, batches)

    acc = _BufferedSum(REF, buffer_rows=1_000)
    for rows in batches:
        b = _batch(rows)
        acc.add(
            _flatten(
                REF,
                pa.record_batch(
                    {
                        "chromosome_index": pa.array(
                            [3] * b.num_rows, pa.int32()
                        ),
                        "position": b.column("position"),
                        "dp": b.column("dp"),
                        "sample_counts": b.column("sample_counts"),
                    }
                ),
            )
        )
    want = [
        (p, dp, sc)
        for _, p, dp, sc in _rows(_unflatten(REF, acc.result()))
        if (dp, sc) != (0, 0)
    ]
    assert _as_rows(f.rows()) == want


def _rows(table):
    return [tuple(r.values()) for r in table.to_pylist()]


# ── Arrow / Parquet output ───────────────────────────────────────────────────


def test_dense_root_table_layout():
    f = _fold(10, 20, [[(10, 1, 1), (15, 2, 0)]])
    t = _dense_root_table(7, 3, 2, *f.rows())
    assert t.schema == agg._DENSE_ROOT_SCHEMA
    assert _rows(t) == [(7, 3, 2, 10, 1, 1), (7, 3, 2, 15, 2, 0)]


def test_sparse_result_is_a_ref_blocks_staging_group():
    f = _fold(10, 20, [[(10, 1, 1), (15, 2, 0)]])
    result = _ref_sparse_result(3, *f.at(np.array([15, 18])))
    assert result.schema == REF.agg_schema()
    staged = _staging_table(REF, result, 7, 1, 2)
    assert staged.schema == REF.staging_schema()
    assert _rows(staged) == [(7, 1, 2, 3, 15, 2, 0)]


def test_write_dense_root_row_groups(tmp_path, monkeypatch):
    """The dense root is written slice by slice (one row group per slice of
    window positions), sorted by position, and reads back as rows()."""
    monkeypatch.setattr(agg, "_DENSE_WRITE_CHUNK", 16)
    f = _fold(1, 100, [[(p, p * 2, 1) for p in range(1, 101, 3)]])
    path = str(tmp_path / "dense.parquet")
    n = _write_dense_root(path, f, 9, 4, 1)
    assert n == f.n_live == 34
    meta = pq.ParquetFile(path).metadata
    assert meta.num_row_groups == 7  # ceil(100 / 16) slices, all non-empty
    position = meta.row_group(0).column(3)
    assert position.path_in_schema == "position"
    assert "DELTA_BINARY_PACKED" in position.encodings
    back = pq.read_table(path)
    assert back.schema == agg._DENSE_ROOT_SCHEMA
    assert [(r[3], r[4], r[5]) for r in _rows(back)] == _as_rows(f.rows())
    assert {(r[0], r[1], r[2]) for r in _rows(back)} == {(9, 4, 1)}


# ── windows ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "length,size,expected",
    [
        (46_709_983, DEFAULT_REF_SHARD_SIZE, 3),
        (46_709_983, 0, 1),
        (46_709_983, 5_000_000, 10),
        (20_000_000, 20_000_000, 1),
        (20_000_001, 20_000_000, 2),
    ],
)
def test_position_windows_tile_the_chromosome(length, size, expected):
    windows = _position_windows(length, size)
    assert len(windows) == expected
    assert windows[0][0] == 1 and windows[-1][1] == length
    for (_, end), (start, _) in zip(windows, windows[1:]):
        assert start == end + 1
    if size > 0:
        assert all(hi - lo + 1 <= size for lo, hi in windows)
