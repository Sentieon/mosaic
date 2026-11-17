# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for `mosaic remove` (mosaic/remove.py).

* The derive: derive_site_partials on crafted rows (two records of one
  sample at one position, n_records / n_listed, zero-hit alleles, an allele
  listed twice in one row, the negation); derive_ref_partial
  (blocks + deletion-split spans, a span crossing the window edge, the
  span-margin error, the (0, 0) omission, the negation).
* The window planner, the decision state machine (fresh / resume at derive
  or delete / another unfinished set / already removed / a non-complete
  batch / unknown), the fault-hook parser.
* The PostgreSQL DDL of ingestion_progress.kind and removal_progress
  (idempotent), RefBlockSegments.from_arrays, the sample_filter pushdown
  rule.

Against ingest's own partials for the same records (sign +1): the Rust
worker binary's
(``tests/unit/test_worker_bin_files.py::
test_site_partials_equal_the_workers_for_the_same_records``; until then the
Python worker's ``bulk_upload`` here).
"""

import numpy as np
import pyarrow as pa
import pytest

from mosaic import remove as R
from mosaic.ingest import RefBlockSegments, sum_ref_block_segments
from mosaic.init_db import (
    INGESTION_KIND_COLUMN_DDL,
    _create_removal_schema,
)
from mosaic.tombstones import NOT_IN_PUSHDOWN_MAX, sample_filter
from sample_slices import (
    SAMPLE_REF_BLOCKS_ARROW_SCHEMA,
    SAMPLE_SITE_INFO_ARROW_SCHEMA,
)

pytestmark = pytest.mark.unit

C = 0


def _site_rows(rows):
    """A sample_site_info table from dicts (missing columns null)."""
    cols = {n: [] for n in SAMPLE_SITE_INFO_ARROW_SCHEMA.names}
    for r in rows:
        for n in cols:
            cols[n].append(r.get(n))
    return pa.Table.from_pydict(cols, schema=SAMPLE_SITE_INFO_ARROW_SCHEMA)


def _row(sid, pos, alleles, gt_1, gt_2, **kw):
    base = {
        "sample_id": sid,
        "chromosome_index": C,
        "position": pos,
        "alleles": alleles,
        "gt_1": gt_1,
        "gt_2": gt_2,
        "qual": 1.0,
        "site_dp": 10,
        "mqsq_1": 100.0,
        "mqsq_2": 10,
        "sb": [1, 2, 3, 4],
        "gtc": [0.0, 1.0, 0.0],
    }
    base.update(kw)
    return base


# ── derive_site_partials ─────────────────────────────────────────────────────


def test_site_partials_sum_negate_and_count_records():
    """Two records of one sample at one position count twice (n_records 2)
    and their contributions add; everything is negated."""
    rows = _site_rows(
        [
            _row(
                5, 100, ["A", "xT", "*"], 0, 1, qual=2.5, gtc=[0.25, 0.75, 0]
            ),
            _row(5, 100, ["A", "-AT", "*"], 1, None, qual=4.0, site_dp=7),
            _row(6, 200, ["G", "xC", "*"], 1, 1, sb=[0, 0, 5, 5]),
        ]
    )
    bsi, bac = R.derive_site_partials(rows, batch_id=9)
    assert bsi.schema == R.BATCH_SITE_INFO_ARROW_SCHEMA
    got = {r["position"]: r for r in bsi.to_pylist()}
    assert set(got) == {100, 200}
    a = got[100]
    assert a["batch_id"] == 9
    assert (a["qual"], a["dp"], a["mqsq_1"], a["mqsq_2"]) == (
        -6.5,
        -17,
        -200.0,
        -20,
    )
    assert a["sb"] == [-2, -4, -6, -8]
    # Fixed point: each row's gtc quantized (2**32 units), summed, negated.
    assert a["gtc"] == [-(1 << 30), -(7 << 30), 0]
    assert a["n_records"] == -2
    assert got[200]["n_records"] == -1 and got[200]["sb"] == [0, 0, -5, -5]

    al = {(r["position"], r["allele"]): r for r in bac.to_pylist()}
    # (count, n_listed), negated: A listed by both records at 100 (hit by
    # gt_1 of the first), xT once and hit once, -AT hit once, * listed by
    # both and never hit (a zero-hit row is still written).
    want = {
        (100, "A"): (-1, -2),
        (100, "xT"): (-1, -1),
        (100, "-AT"): (-1, -1),
        (100, "*"): (0, -2),
        (200, "G"): (0, -1),
        (200, "xC"): (-2, -1),
        (200, "*"): (0, -1),
    }
    assert {k: (r["count"], r["n_listed"]) for k, r in al.items()} == want
    assert [r["position"] for r in bsi.to_pylist()] == [100, 200]


def test_site_partials_allele_listed_twice_in_a_row_counts_once():
    """A record whose two ALTs encode alike (ATG>CA and ATG>A both -ATG):
    one listing, two hits."""
    rows = _site_rows([_row(5, 10, ["A", "-ATG", "-ATG", "*"], 1, 2)])
    _bsi, bac = R.derive_site_partials(rows, batch_id=1, sign=1)
    al = {r["allele"]: (r["count"], r["n_listed"]) for r in bac.to_pylist()}
    assert al == {"A": (0, 1), "-ATG": (2, 1), "*": (0, 1)}


def test_site_partials_empty():
    bsi, bac = R.derive_site_partials(_site_rows([]), batch_id=1)
    assert bsi.num_rows == 0 and bac.num_rows == 0
    assert bsi.schema == R.BATCH_SITE_INFO_ARROW_SCHEMA
    assert bac.schema == R.BATCH_ALLELE_COUNTS_ARROW_SCHEMA


# ── derive_ref_partial ───────────────────────────────────────────────────────


def _blocks(rows):
    """sample_ref_blocks rows (sid, start, end, dp, sample_counts)."""
    return pa.Table.from_pydict(
        {
            "sample_id": [r[0] for r in rows],
            "chromosome_index": [C] * len(rows),
            "position_start": [r[1] for r in rows],
            "position_end": [r[2] for r in rows],
            "ad_1": [None] * len(rows),
            "ad_2": [None] * len(rows),
            "dp": [r[3] for r in rows],
            "pl_1": [None] * len(rows),
            "pl_2": [None] * len(rows),
            "pl_3": [None] * len(rows),
            "sample_counts": [r[4] for r in rows],
        },
        schema=SAMPLE_REF_BLOCKS_ARROW_SCHEMA,
    )


def _dense(table):
    return {
        r["position"]: (r["dp"], r["sample_counts"]) for r in table.to_pylist()
    }


def test_ref_partial_blocks_and_spans_clipped_and_negated():
    blocks = _blocks([(5, 95, 104, 10, 1), (6, 100, 102, 3, 0)])
    # A deletion row at 98 (before the window) whose split span 99..101
    # reaches into it; a row at 103 with the span 104..105 (past hi).
    sites = _site_rows(
        [
            _row(
                5,
                98,
                ["G", "-GAA", "*"],
                0,
                1,
                site_dp=44,
                del_start=98,
                del_end=101,
                del_ref_spans=[99, 101],
            ),
            _row(
                6,
                103,
                ["G", "-GAA", "*"],
                0,
                1,
                site_dp=7,
                del_start=103,
                del_end=105,
                del_ref_spans=[104, 105],
            ),
        ]
    )
    t = R.derive_ref_partial(blocks, sites, C, 100, 104, 50, batch_id=3)
    assert t.schema == R.BATCH_REF_BLOCKS_ARROW_SCHEMA
    assert set(t.column("batch_id").to_pylist()) == {3}
    assert _dense(t) == {
        100: (-(10 + 3 + 44), -1),
        101: (-(10 + 3 + 44), -1),
        102: (-(10 + 3), -1),
        103: (-10, -1),
        104: (-(10 + 7), -1),
    }


def test_ref_partial_omits_zero_positions():
    """A ./. block with dp 0 (0, 0) and a split span with site dp 0 leave
    no row, as ingest's partial has none."""
    blocks = _blocks([(5, 10, 12, 0, 0), (5, 13, 13, 4, 1)])
    t = R.derive_ref_partial(blocks, _site_rows([]), C, 1, 100, 10, 1)
    assert _dense(t) == {13: (-4, -1)}


def test_ref_partial_sign_plus_equals_the_segment_sum():
    blocks = _blocks([(5, 1, 50, 7, 1), (6, 20, 80, 2, 1)])
    t = R.derive_ref_partial(blocks, _site_rows([]), C, 1, 100, 10, 1, 1)
    seg = RefBlockSegments(C)
    seg.add(1, 50, 7, 1)
    seg.add(20, 80, 2, 1)
    pos, dp, sc = sum_ref_block_segments(seg)
    assert _dense(t) == dict(zip(pos.tolist(), zip(dp.tolist(), sc.tolist())))


def test_span_beyond_the_margin_is_an_error():
    sites = _site_rows(
        [
            _row(
                5,
                100,
                ["G", "-G" + "A" * 300, "*"],
                0,
                1,
                del_start=100,
                del_end=400,
                del_ref_spans=[101, 400],
            ),
        ]
    )
    with pytest.raises(R.SpanMarginError, match="--span-margin >= 300"):
        R.derive_ref_partial(_blocks([]), sites, C, 1, 1000, 299, 1)
    # Within the margin: counted.
    t = R.derive_ref_partial(_blocks([]), sites, C, 1, 1000, 300, 1)
    assert len(_dense(t)) == 300
    # A row before the window is checked by its own window, not this one.
    t = R.derive_ref_partial(_blocks([]), sites, C, 350, 1000, 10, 1)
    assert sorted(_dense(t)) == list(range(350, 401))


def test_window_split_sums_to_the_whole():
    """Two adjacent windows (with the margin) give the whole-range sums."""
    blocks = _blocks([(5, 1, 30, 5, 1), (6, 18, 40, 2, 0)])
    sites = _site_rows(
        [
            _row(
                5,
                15,
                ["G", "-GAAAAAA", "*"],
                0,
                1,
                site_dp=9,
                del_start=15,
                del_end=22,
                del_ref_spans=[16, 22],
            ),
        ]
    )

    def part(lo, hi):
        rows = sites.filter(
            pa.compute.and_(
                pa.compute.greater_equal(sites.column("position"), lo - 10),
                pa.compute.less_equal(sites.column("position"), hi),
            )
        )
        bl = blocks.filter(
            pa.compute.and_(
                pa.compute.less_equal(blocks.column("position_start"), hi),
                pa.compute.greater_equal(blocks.column("position_end"), lo),
            )
        )
        return _dense(R.derive_ref_partial(bl, rows, C, lo, hi, 10, 1))

    whole = part(1, 40)
    assert {**part(1, 19), **part(20, 40)} == whole
    assert whole[20] == (-(5 + 2 + 9), -1)


# ── planner, decisions, hook ─────────────────────────────────────────────────


def test_plan_units_tiles_every_contig():
    units = R.plan_units({0: 25, 1: 10}, 10)
    assert units == [
        (0, 0, 1, 10),
        (0, 1, 11, 20),
        (0, 2, 21, 25),
        (1, 0, 1, 10),
    ]
    with pytest.raises(ValueError):
        R.plan_units({0: 25}, 0)


def _samples(**status):
    return {
        n: R.SampleRow(i + 1, n, st)
        for i, (n, st) in enumerate(sorted(status.items()))
    }


S3 = _samples(a="active", b="active", c="active")


def test_fresh_removal():
    plan = R.decide_removal(["b", "a", "a"], S3, [], [])
    assert (plan.state, plan.phase) == ("fresh", "derive")
    assert plan.names == ("a", "b") and plan.sample_ids == (1, 2)


def test_unknown_sample_refused():
    with pytest.raises(R.RemovalRefused, match="unknown sample.*zz"):
        R.decide_removal(["a", "zz"], S3, [], [])


def test_already_removed_refused():
    s = _samples(a="removed", b="active")
    with pytest.raises(R.RemovalRefused, match="already removed: a"):
        R.decide_removal(["a", "b"], s, [], [])


def _unfinished(names, status, batch_status, rid=4, bid=11):
    return R.RemovalRow(rid, (1, 2), tuple(names), bid, status, batch_status)


@pytest.mark.parametrize(
    "status,batch_status,phase",
    [
        ("in_progress", "in_progress", "derive"),
        ("failed", "failed", "derive"),
        ("failed", "complete", "delete"),
        ("registered", "complete", "delete"),
    ],
)
def test_unfinished_same_set_resumes(status, batch_status, phase):
    s = _samples(a="removed", b="removed", c="active")
    r = _unfinished(["a", "b"], status, batch_status)
    plan = R.decide_removal(["b", "a"], s, [r], [])
    assert (plan.state, plan.phase, plan.removal) == ("resume", phase, r)


def test_unfinished_other_set_refused():
    s = _samples(a="removed", b="removed", c="active")
    r = _unfinished(["a", "b"], "failed", "failed")
    for names in (["c"], ["a"], ["a", "b", "c"]):
        with pytest.raises(
            R.RemovalRefused, match="removal 4 of a, b is unfinished"
        ):
            R.decide_removal(names, s, [r], [])


@pytest.mark.parametrize(
    "status,hint",
    [
        ("uploaded", "run mosaic register"),
        ("in_progress", "finish it with mosaic ingest"),
        ("failed", "finish it with mosaic ingest"),
    ],
)
def test_sample_in_a_non_complete_batch_refused(status, hint):
    batches = [
        R.BatchRow(1, ("a", "c"), "complete"),
        R.BatchRow(2, ("b", "c"), status),
    ]
    assert R.decide_removal(["a"], S3, [], batches).state == "fresh"
    with pytest.raises(R.RemovalRefused, match=f"batch 2 .*{hint}"):
        R.decide_removal(["a", "b"], S3, [], batches)


def test_hook_parser():
    assert R.parse_fail_remove(None) is None
    assert R.parse_fail_remove("") is None
    for phase in (
        "pre_register",
        "post_register",
        "pre_delete",
        "post_delete",
    ):
        assert R.parse_fail_remove(phase) == phase
    for bad in ("pre_commit", "post_register:x", "PRE_DELETE"):
        with pytest.raises(ValueError, match="MOSAIC_TEST_FAIL_REMOVE"):
            R.parse_fail_remove(bad)
    with pytest.raises(RuntimeError, match="injected crash @pre_delete"):
        R._maybe_fail_remove("pre_delete", "pre_delete")
    R._maybe_fail_remove("pre_delete", "post_delete")  # another phase: inert


# ── DDL, segments, pushdown ──────────────────────────────────────────────────


class _Cursor:
    def __init__(self):
        self.sql = []

    def execute(self, sql, params=None):
        self.sql.append(" ".join(sql.split()))


def test_removal_schema_ddl_is_idempotent():
    cur = _Cursor()
    _create_removal_schema(cur)
    alter, create = cur.sql
    assert alter.startswith("ALTER TABLE ingestion_progress ADD COLUMN IF NOT")
    assert " ".join(INGESTION_KIND_COLUMN_DDL.split()) in alter
    assert "DEFAULT 'ingest'" in alter and "'removal'" in alter
    assert create.startswith("CREATE TABLE IF NOT EXISTS removal_progress")
    for col in ("sample_ids INTEGER[] NOT NULL", "batch_id INTEGER NOT NULL"):
        assert col in create
    for status in ("in_progress", "registered", "complete", "failed"):
        assert f"'{status}'" in create


def test_segments_from_arrays():
    seg = RefBlockSegments.from_arrays(
        C, np.array([1, 5]), np.array([3, 6]), [10, -2], [1, 0]
    )
    assert len(seg) == 2
    pos, dp, sc = sum_ref_block_segments(seg)
    assert pos.tolist() == [1, 2, 3, 5, 6]
    assert dp.tolist() == [10, 10, 10, -2, -2]
    with pytest.raises(ValueError):
        RefBlockSegments.from_arrays(C, [1], [2, 3], [1], [1])


def test_sample_filter_pushdown_rule():
    table = pa.table({"sample_id": pa.array([1, 2, 3, 2], pa.int32())})
    push, mask = sample_filter([2, 3])
    assert push is not None and "sample_id" in repr(push)
    assert mask(table).column("sample_id").to_pylist() == [2, 3, 2]
    push, mask = sample_filter(range(NOT_IN_PUSHDOWN_MAX + 1))
    assert push is None
    assert mask(table).num_rows == 4


# ── the delete line ──────────────────────────────────────────────────────────


class _File:
    def __init__(self, rows, size):
        self.record_count = rows
        self.file_size_in_bytes = size


def test_delete_line_counts_files_as_counted():
    # A copy-on-write rewrite that splits one file into two (the round-2
    # cloud run printed "-1 data file(s) dropped" here): 1 removed, 2
    # written, 1 kept.
    before = {"a": _File(100, 1000), "k": _File(7, 70)}
    after = {"k": before["k"], "b": _File(30, 300), "c": _File(20, 200)}
    assert R.delete_line("sample_ref_blocks", before, after, 0.5) == (
        "  delete sample_ref_blocks: 1 data file(s) removed, 2 written (50 "
        "rows of other samples, 500 bytes), 1 kept; 1000 bytes and 50 rows "
        "removed (0.50s)"
    )


def test_delete_line_dropped_rewritten_kept_and_nothing():
    # One file dropped whole, one rewritten, one kept.
    before = {"x": _File(10, 100), "y": _File(40, 400), "z": _File(5, 50)}
    after = {"z": before["z"], "y2": _File(25, 250)}
    assert R.delete_line("sample_site_info", before, after, 1.234) == (
        "  delete sample_site_info: 2 data file(s) removed, 1 written (25 "
        "rows of other samples, 250 bytes), 1 kept; 500 bytes and 25 rows "
        "removed (1.23s)"
    )
    # A resumed delete: nothing matched.
    assert R.delete_line("sample_site_info", after, after, 0.0) == (
        "  delete sample_site_info: 0 data file(s) removed, 0 written (0 "
        "rows of other samples, 0 bytes), 2 kept; 0 bytes and 0 rows "
        "removed (0.00s)"
    )
    # An empty table.
    assert R.delete_line("sample_site_info", {}, {}, 0.0).startswith(
        "  delete sample_site_info: 0 data file(s) removed, 0 written"
    )
