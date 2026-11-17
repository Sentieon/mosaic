# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for incremental `mosaic aggregate` (no Docker).

A run whose completed-batch set extends the latest complete run (its parent)
reduces only the delta batches and folds them onto the parent's retained
roots. These tests pin the pure pieces:

* parent resolution and the resume key (``_choose_parent`` / ``_decide_run``)
  on synthetic aggregation_progress rows;
* the multi-source SUM (``_sum_sources``) a fold unit runs over this run's
  delta root plus the parent's root, with overlapping keys and list columns;
* the fold unit's and the window unit's sources (``_fold_sources``,
  ``_window_sources``) and the window fold over the parent's dense root plus
  delta rows (``_fold_window``);
* the staging prune filter (``_nonroot_tiers`` / ``_prune_filter``).
"""

import numpy as np
import pyarrow as pa
import pytest
from pyiceberg.expressions.visitors import bind
from pyiceberg.io.pyarrow import expression_to_pyarrow

from mosaic.aggregate import (
    DENSE_REF_TABLE,
    SPECS,
    RunRow,
    _choose_parent,
    _decide_run,
    _dense_root_table,
    _fold_sources,
    _fold_window,
    _nonroot_tiers,
    _position_windows,
    _prune_filter,
    _staging_table,
    _sum_sources,
    _window_sources,
    _WindowFold,
)
from mosaic.init_db import tier_table_schemas

pytestmark = pytest.mark.unit

SITE = SPECS["site_info"]
ALLELE = SPECS["allele_counts"]


# ── parent resolution ────────────────────────────────────────────────────────


def _run(
    agg_run_id,
    batch_ids,
    status="complete",
    pruned=False,
    parent=None,
    fan_in=100,
    ref_shard_size=20_000_000,
    batch_set_hash=None,
    prune_started=False,
):
    return RunRow(
        agg_run_id=agg_run_id,
        batch_set_hash=batch_set_hash or ",".join(map(str, batch_ids)),
        batch_ids=tuple(batch_ids),
        fan_in=fan_in,
        ref_shard_size=ref_shard_size,
        status=status,
        parent_run_id=parent,
        staging_pruned=pruned,
        prune_started=prune_started,
    )


def _decide(runs, completed, rebuild=False, fan_in=100, ref=20_000_000):
    return _decide_run(
        runs,
        ",".join(map(str, completed)),
        completed,
        fan_in,
        ref,
        rebuild,
    )


def test_parent_is_the_latest_complete_unpruned_subset():
    runs = [
        _run(1, [1]),
        _run(2, [1, 2]),
        _run(3, [1, 2], status="failed"),
    ]
    assert _choose_parent(runs, [1, 2, 3]).agg_run_id == 2
    # Highest agg_run_id wins, not the largest batch set.
    runs.append(_run(4, [1]))
    assert _choose_parent(runs, [1, 2, 3]).agg_run_id == 4


def test_pruned_run_is_never_a_parent():
    runs = [_run(1, [1]), _run(2, [1, 2], pruned=True)]
    assert _choose_parent(runs, [1, 2, 3]).agg_run_id == 1
    runs = [_run(1, [1], pruned=True), _run(2, [1, 2], pruned=True)]
    assert _choose_parent(runs, [1, 2, 3]) is None


def test_parent_must_be_a_proper_subset():
    # A batch outside the completed set disqualifies (no removal support).
    assert _choose_parent([_run(1, [1, 4])], [1, 2, 3]) is None
    # The whole set is not a parent (the delta would be empty).
    assert _choose_parent([_run(1, [1, 2, 3])], [1, 2, 3]) is None
    assert _choose_parent([_run(1, [2])], [1, 2, 3]).agg_run_id == 1


@pytest.mark.parametrize("status", ["in_progress", "failed"])
def test_unfinished_runs_are_never_parents(status):
    assert _choose_parent([_run(1, [1], status=status)], [1, 2]) is None


def test_no_runs_is_a_full_run():
    d = _decide([], [1, 2])
    assert (d.state, d.run, d.parent) == ("fresh", None, None)


def test_incremental_run_gets_the_parent():
    d = _decide([_run(1, [1, 2])], [1, 2, 3])
    assert d.state == "fresh" and d.run is None
    assert d.parent.agg_run_id == 1


def test_complete_run_for_the_set_is_up_to_date():
    runs = [_run(1, [1]), _run(2, [1, 2], parent=1)]
    d = _decide(runs, [1, 2])
    assert d.state == "current" and d.run.agg_run_id == 2


def test_rebuild_ignores_the_parent():
    d = _decide([_run(1, [1, 2])], [1, 2, 3], rebuild=True)
    assert (d.state, d.run, d.parent) == ("fresh", None, None)


def test_resume_needs_the_same_parent():
    parent = _run(1, [1, 2])
    failed = _run(2, [1, 2, 3], status="failed", parent=1)
    d = _decide([parent, failed], [1, 2, 3])
    assert d.state == "resume" and d.run.agg_run_id == 2
    assert d.parent.agg_run_id == 1

    # A failed FULL run of the set does not resume while a parent exists: a
    # resume must fold onto the same parent as the run it continues.
    full = _run(2, [1, 2, 3], status="failed", parent=None)
    d = _decide([parent, full], [1, 2, 3])
    assert d.state == "fresh" and d.parent.agg_run_id == 1

    # ... and does resume once no parent qualifies.
    d = _decide([_run(1, [1, 2], pruned=True), full], [1, 2, 3])
    assert d.state == "resume" and d.run.agg_run_id == 2
    assert d.parent is None


@pytest.mark.parametrize(
    "fan_in,ref,resumes",
    [(100, 20_000_000, True), (2, 20_000_000, False), (100, 5, False)],
)
def test_resume_key_fan_in_and_ref_shard_size(fan_in, ref, resumes):
    runs = [_run(1, [1, 2]), _run(2, [1, 2, 3], status="failed", parent=1)]
    d = _decide(runs, [1, 2, 3], fan_in=fan_in, ref=ref)
    assert (d.state == "resume") is resumes
    assert d.parent.agg_run_id == 1


def test_pruned_unfinished_run_is_not_resumed():
    """A failed run whose staging another run's completion deleted has
    checkpoints with nothing behind them: start over."""
    runs = [_run(1, [1, 2], status="failed", pruned=True)]
    d = _decide(runs, [1, 2])
    assert (d.state, d.run, d.parent) == ("fresh", None, None)


def _interrupted_prune(prune_started, pruned=False, with_parent=True):
    """Run 2 (+{b3} onto run 1) crashed after its prune marked run 1
    staging_pruned: run 1 no longer qualifies as a parent, and no other run
    does."""
    runs = [_run(0, [1], pruned=True)] if with_parent else []
    if with_parent:
        runs.append(_run(1, [1, 2], pruned=True))
    runs.append(
        _run(
            2,
            [1, 2, 3],
            status="failed",
            parent=1,
            pruned=pruned,
            prune_started=prune_started,
        )
    )
    return runs


def test_prune_started_run_resumes_onto_its_pruned_parent():
    """(a) A run that began its prune pruned its own parent; it resumes
    anyway, carrying that STORED parent (its delta and plan)."""
    d = _decide(_interrupted_prune(prune_started=True), [1, 2, 3])
    assert d.state == "resume" and d.run.agg_run_id == 2
    assert d.parent.agg_run_id == 1 and d.parent.staging_pruned


def test_run_whose_parent_was_pruned_otherwise_is_not_resumed():
    """(b) Without its own prune record, a run whose parent is pruned is
    not resumed: a full run starts (no parent qualifies)."""
    d = _decide(_interrupted_prune(prune_started=False), [1, 2, 3])
    assert (d.state, d.run, d.parent) == ("fresh", None, None)


def test_prune_record_does_not_save_a_pruned_run():
    """(c) Another run's completion deleted this run's staging: its own
    prune record does not make its checkpoints trustworthy again."""
    runs = _interrupted_prune(prune_started=True, pruned=True)
    d = _decide(runs, [1, 2, 3])
    assert (d.state, d.run, d.parent) == ("fresh", None, None)


def test_prune_started_run_without_its_parent_row_is_not_resumed():
    """(d) The stored parent's row is gone: nothing fixes the run's delta,
    so it is not resumed."""
    runs = _interrupted_prune(prune_started=True, with_parent=False)
    d = _decide(runs, [1, 2, 3])
    assert (d.state, d.run, d.parent) == ("fresh", None, None)


@pytest.mark.parametrize("stored_parent", [None, 0])
def test_prune_started_run_yields_to_a_newer_candidate(stored_parent):
    """A prune-started failed run (full, or incremental onto run 0) is NOT
    resumed when an unpruned complete candidate parent exists: a new run
    folds onto that candidate.

    The failed run marked every run that existed when it began its prune
    staging_pruned (in the same transaction as its prune record), so an
    unpruned complete candidate (run 3) can only have completed after the
    crash, and its finalizes swapped its partitions in after the failed
    run's. Resuming would skip the failed run's checkpointed finalizes,
    leave PostgreSQL holding run 3's partitions and still mark the failed
    run complete for its own batch set: a wrong cohort_generation.
    """
    runs = [
        _run(0, [1], pruned=True),
        _run(
            2,
            [1, 2, 3],
            status="failed",
            parent=stored_parent,
            prune_started=True,
        ),
        _run(3, [1, 2]),
    ]
    d = _decide(runs, [1, 2, 3])
    assert (d.state, d.run) == ("fresh", None)
    assert d.parent.agg_run_id == 3


# ── the multi-source fold (sparse metrics) ──────────────────────────────────


def _site(rows):
    """tier_site_info-shaped rows: (chrom, pos, qual, dp, mqsq_1, mqsq_2,
    sb, gtc[, n_records]), staging columns included (the fold reads staging
    groups); gtc is in fixed point (int64), n_records defaults to 1."""
    table = pa.table(
        {
            "chromosome_index": pa.array([r[0] for r in rows], pa.int32()),
            "position": pa.array([r[1] for r in rows], pa.int32()),
            "qual": pa.array([r[2] for r in rows], pa.float64()),
            "dp": pa.array([r[3] for r in rows], pa.int64()),
            "mqsq_1": pa.array([r[4] for r in rows], pa.float64()),
            "mqsq_2": pa.array([r[5] for r in rows], pa.int64()),
            "sb": pa.array([r[6] for r in rows], pa.list_(pa.int64())),
            "gtc": pa.array([r[7] for r in rows], pa.list_(pa.int64())),
            "n_records": pa.array(
                [r[8] if len(r) > 8 else 1 for r in rows], pa.int64()
            ),
        }
    )
    return _staging_table(SITE, table, 9, 1, 0)


def _scan_of(tables):
    """An in-memory stand-in for _scan_table: {table name: [record batch]},
    ignoring the row filter (each fake table holds one source's rows)."""

    def scan(name, row_filter, fields):
        for batch in tables[name]:
            yield batch.select(list(fields))

    return scan


def _rows(table):
    return [tuple(r.values()) for r in table.to_pylist()]


DELTA_ROOT = [
    (1, 100, 1.5, 10, 2.0, 3, [1, 0, 2, 0], [1, 1, 0]),
    (1, 200, 0.5, 4, 1.0, 1, [0, 1, 0, 1], [0, 1, 0]),
]
PARENT_ROOT = [
    (1, 100, 2.5, 20, 3.0, 5, [2, 2, 2, 2], [3, 0, 1]),
    (1, 150, 7.0, 8, 4.0, 2, None, [0, 0, 2]),
    (1, 200, 1.0, None, 0.5, 2, [1, 1, 1, 1], None),
]


def test_fold_sums_two_sources_with_overlapping_keys():
    sources = [("delta", None), ("parent", None)]
    scan = _scan_of(
        {
            "delta": _site(DELTA_ROOT).to_batches(),
            "parent": _site(PARENT_ROOT).to_batches(),
        }
    )
    out = _sum_sources(SITE, sources, scan)
    assert out.schema == SITE.agg_schema()
    assert _rows(out) == [
        (1, 100, 4.0, 30, 5.0, 8, [3, 2, 4, 2], [4, 1, 1], 2),
        (1, 150, 7.0, 8, 4.0, 2, [0, 0, 0, 0], [0, 0, 2], 1),
        (1, 200, 1.5, 4, 1.5, 3, [1, 2, 1, 2], [0, 1, 0], 2),
    ]
    # Source order and buffer flushes change nothing (integers exact; these
    # floats are exact in binary).
    for order in (sources, sources[::-1]):
        for bound in (1, 2, 1000):
            assert _rows(_sum_sources(SITE, order, scan, bound)) == _rows(out)


def test_fold_of_one_source_rewrites_it_unchanged():
    """An empty-delta chromosome folds the parent's root alone: the new root
    equals it (nulls filled with 0, as every root is)."""
    scan = _scan_of({"parent": _site(PARENT_ROOT[:2]).to_batches()})
    out = _sum_sources(SITE, [("parent", None)], scan)
    assert _rows(out) == [
        (1, 100, 2.5, 20, 3.0, 5, [2, 2, 2, 2], [3, 0, 1], 1),
        (1, 150, 7.0, 8, 4.0, 2, [0, 0, 0, 0], [0, 0, 2], 1),
    ]


def test_fold_allele_counts_string_keys():
    def alleles(rows):
        return pa.record_batch(
            {
                "chromosome_index": pa.array([1] * len(rows), pa.int32()),
                "position": pa.array([r[0] for r in rows], pa.int32()),
                "allele": pa.array([r[1] for r in rows], pa.string()),
                "count": pa.array([r[2] for r in rows], pa.int64()),
                "n_listed": pa.array([1] * len(rows), pa.int64()),
            }
        )

    scan = _scan_of(
        {
            "delta": [alleles([(5, "xA", 1), (5, "+T", 2)])],
            "parent": [alleles([(5, "xA", 4), (6, "xG", 1)])],
        }
    )
    out = _sum_sources(ALLELE, [("delta", None), ("parent", None)], scan)
    assert _rows(out) == [
        (1, 5, "+T", 2, 1),
        (1, 5, "xA", 5, 2),
        (1, 6, "xG", 1, 1),
    ]


def test_fold_sources_delta_and_parent_roots():
    both = _fold_sources(
        SITE, 3, agg_run_id=7, delta_root=2, parent_run_id=4, parent_root=1
    )
    assert [t for t, _ in both] == [SITE.staging_table] * 2
    assert _filter_terms(both[0][1]) == {
        ("agg_run_id", 7),
        ("chromosome_index", 3),
        ("tier", 2),
    }
    assert _filter_terms(both[1][1]) == {
        ("agg_run_id", 4),
        ("chromosome_index", 3),
        ("tier", 1),
    }
    # Every group of a root is read: no group_id term.
    assert "group_id" not in repr(both[0][1])
    # No delta batch covers the chromosome (T = 0): the parent alone.
    only_parent = _fold_sources(SITE, 3, 7, 0, 4, 2)
    assert len(only_parent) == 1
    assert ("agg_run_id", 4) in _filter_terms(only_parent[0][1])
    # The parent had no unit there: the delta alone.
    only_delta = _fold_sources(SITE, 3, 7, 1, 4, None)
    assert len(only_delta) == 1
    assert ("agg_run_id", 7) in _filter_terms(only_delta[0][1])


def _filter_terms(expr):
    """{(column, value)} of the EqualTo terms of an And tree."""
    from pyiceberg.expressions import And, EqualTo

    if isinstance(expr, And):
        return _filter_terms(expr.left) | _filter_terms(expr.right)
    if isinstance(expr, EqualTo):
        return {(expr.term.name, expr.literal.value)}
    return set()


# ── the window fold with a parent dense root ─────────────────────────────────


def _ref_batch(rows):
    """batch_ref_blocks rows (position, dp, sample_counts), int32 values."""
    return pa.record_batch(
        {
            "position": pa.array([r[0] for r in rows], pa.int32()),
            "dp": pa.array([r[1] for r in rows], pa.int32()),
            "sample_counts": pa.array([r[2] for r in rows], pa.int32()),
        }
    )


def _dense_rows(fold):
    pos, dp, sc = fold.rows()
    return list(zip(pos.tolist(), dp.tolist(), sc.tolist()))


def _parent_root(rows, window_id=0):
    """A parent's dense-root rows in the tier_ref_blocks_dense schema."""
    pos = np.array([r[0] for r in rows], dtype=np.int64)
    return _dense_root_table(
        4,
        1,
        window_id,
        pos,
        np.array([r[1] for r in rows], dtype=np.int64),
        np.array([r[2] for r in rows], dtype=np.int64),
    )


def test_window_sources_parent_by_position_not_window():
    sources = _window_sources(1, 101, 200, [5, 6], parent_run_id=4)
    assert [t for t, _ in sources] == [DENSE_REF_TABLE, "batch_ref_blocks"]
    parent_filter = repr(sources[0][1])
    assert "window_id" not in parent_filter
    for term in ("agg_run_id", "chromosome_index", "position"):
        assert term in parent_filter
    assert "batch_id" in repr(sources[1][1])
    # A full run: no parent source. An empty delta: no batch source.
    assert [t for t, _ in _window_sources(1, 101, 200, [5], None)] == [
        "batch_ref_blocks"
    ]
    assert [t for t, _ in _window_sources(1, 101, 200, [], 4)] == [
        DENSE_REF_TABLE
    ]


def test_window_fold_parent_rows_plus_delta_rows():
    parent = [(100, 30, 3), (101, 0, 0), (150, 12, 2)]
    delta = [(100, 5, 1), (120, 7, 1), (150, 1, 1)]
    fold = _WindowFold(100, 199)
    n_parent = _fold_window(
        fold,
        _window_sources(1, 100, 199, [9], 4),
        _scan_of(
            {
                DENSE_REF_TABLE: _parent_root(parent).to_batches(),
                "batch_ref_blocks": [_ref_batch(delta)],
            }
        ),
    )
    assert n_parent == 3
    assert _dense_rows(fold) == [
        (100, 35, 4),
        # (101, 0, 0): a net (0, 0) position has no root row
        (120, 7, 1),
        (150, 13, 3),
    ]


def test_window_fold_parent_rows_only_rewrites_them():
    parent = [(100, 30, 3), (101, 0, 1), (199, 12, 2)]
    fold = _WindowFold(100, 199)
    _fold_window(
        fold,
        _window_sources(1, 100, 199, [], 4),
        _scan_of({DENSE_REF_TABLE: _parent_root(parent).to_batches()}),
    )
    assert _dense_rows(fold) == parent


def test_window_fold_parent_root_equals_full_fold_across_window_sizes():
    """Fold b1 + b2 into a parent dense root with one window size, then fold
    that root (re-cut by position into other windows) plus b3: the result
    equals folding b1 + b2 + b3 directly, window by window."""
    rng = np.random.default_rng(3)
    length = 10_000
    batches = []
    for _ in range(3):
        pos = np.sort(rng.choice(np.arange(1, length + 1), 4_000, False))
        batches.append(
            [
                (int(p), int(rng.integers(0, 40)), int(rng.integers(0, 3)))
                for p in pos
            ]
        )

    def fold_windows(window_size, parent_rows, delta):
        out = []
        for lo, hi in _position_windows(length, window_size):
            f = _WindowFold(lo, hi)
            tables = {"batch_ref_blocks": []}
            if parent_rows is not None:
                inside = [r for r in parent_rows if lo <= r[0] <= hi]
                tables[DENSE_REF_TABLE] = _parent_root(inside).to_batches()
            for rows in delta:
                inside = [r for r in rows if lo <= r[0] <= hi]
                tables["batch_ref_blocks"].append(_ref_batch(inside))
            _fold_window(
                f,
                _window_sources(
                    1, lo, hi, [1] if delta else [], 4 if parent_rows else None
                ),
                _scan_of(tables),
            )
            out += _dense_rows(f)
        return out

    parent = fold_windows(3_000, None, batches[:2])
    incremental = fold_windows(2_500, parent, batches[2:])
    full = fold_windows(4_000, None, batches)
    assert incremental == full
    assert len(full) > 6_000


# ── staging pruning ──────────────────────────────────────────────────────────


def test_nonroot_tiers_groups_chromosomes_by_root():
    assert _nonroot_tiers({0: 0, 1: 1, 2: 2, 3: 3, 4: 2}) == {
        2: [2, 4],
        3: [3],
    }
    assert _nonroot_tiers({0: 0, 1: 1}) == {}


def _deleted(expr, rows):
    """The (agg_run_id, tier, chromosome_index) rows ``expr`` deletes from
    tier_site_info."""
    schema = dict(tier_table_schemas())["tier_site_info"]
    table = pa.table(
        {
            "agg_run_id": pa.array([r[0] for r in rows], pa.int32()),
            "tier": pa.array([r[1] for r in rows], pa.int32()),
            "chromosome_index": pa.array([r[2] for r in rows], pa.int32()),
        }
    )
    hit = table.filter(expression_to_pyarrow(bind(schema, expr, True)))
    return set(zip(*(hit.column(c).to_pylist() for c in hit.column_names)))


def test_prune_filter_keeps_only_this_runs_roots():
    this = 7
    # chromosome 1: root tier 3 (tiers 1, 2 non-root); chromosome 2: root 1;
    # chromosome 3: root 2 (an incremental fold over a one-tier delta).
    nonroot = _nonroot_tiers({1: 3, 2: 1, 3: 2})
    rows = [
        (this, 1, 1),
        (this, 2, 1),
        (this, 3, 1),
        (this, 1, 2),
        (this, 1, 3),
        (this, 2, 3),
        (6, 1, 1),
        (6, 2, 3),
        (8, 1, 2),
    ]
    deleted = _deleted(_prune_filter(this, nonroot), rows)
    assert deleted == {
        (this, 1, 1),
        (this, 2, 1),
        (this, 1, 3),
        (6, 1, 1),
        (6, 2, 3),
        (8, 1, 2),
    }
    # The ref_blocks tables keep all of this run's rows.
    assert _deleted(_prune_filter(this, {}), rows) == {
        (6, 1, 1),
        (6, 2, 3),
        (8, 1, 2),
    }


def _data_file(schema, bounds):
    """A one-unit staging DataFile whose columns ``bounds`` = {name: (lo,
    hi)} carry Parquet-style statistics (no nulls)."""
    from pyiceberg.conversions import to_bytes
    from pyiceberg.manifest import DataFile, DataFileContent, FileFormat
    from pyiceberg.typedef import Record
    from pyiceberg.types import IntegerType

    ids = {n: schema.find_field(n).field_id for n in bounds}
    return DataFile.from_args(
        content=DataFileContent.DATA,
        file_path="s3://bucket/staging/data.parquet",
        file_format=FileFormat.PARQUET,
        partition=Record(),
        record_count=100,
        file_size_in_bytes=1000,
        value_counts={i: 100 for i in ids.values()},
        null_value_counts={i: 0 for i in ids.values()},
        lower_bounds={
            ids[n]: to_bytes(IntegerType(), lo)
            for n, (lo, _) in bounds.items()
        },
        upper_bounds={
            ids[n]: to_bytes(IntegerType(), hi)
            for n, (_, hi) in bounds.items()
        },
    )


@pytest.mark.parametrize("table", ["tier_site_info", "tier_ref_blocks_dense"])
def test_prune_decides_every_staging_file_from_its_bounds(table):
    """pyiceberg's delete drops a file the strict metrics evaluator proves
    fully matched, keeps one the inclusive evaluator proves unmatched, and
    reads + rewrites any other. Every staging file is one unit's output, so
    the prune filter must land each file in one of the first two cases: a
    rewrite would read this run's kept roots whole for nothing."""
    from pyiceberg.expressions import NotEqualTo
    from pyiceberg.expressions.visitors import (
        _InclusiveMetricsEvaluator,
        _StrictMetricsEvaluator,
    )

    schema = dict(tier_table_schemas())[table]
    this = 7
    tiered = table == "tier_site_info"
    # chromosome 1: root tier 3; chromosome 2: root tier 1.
    nonroot = _nonroot_tiers({1: 3, 2: 1}) if tiered else {}
    expr = _prune_filter(this, nonroot)

    def unit(run, chrom, tier=1):
        bounds = {"agg_run_id": (run, run), "chromosome_index": (chrom, chrom)}
        if tiered:
            bounds["tier"] = (tier, tier)
        return _data_file(schema, bounds)

    deleted = [unit(6, 1), unit(8, 2), unit(1, 1, 3)]
    kept = [unit(this, 1, 3), unit(this, 2, 1)]
    if tiered:
        deleted += [unit(this, 1, 1), unit(this, 1, 2)]
    strict = _StrictMetricsEvaluator(schema, expr).eval
    inclusive = _InclusiveMetricsEvaluator(schema, expr).eval
    for f in deleted:
        assert strict(f), f  # dropped whole
    for f in kept:
        assert not inclusive(f), f  # untouched, never read
    # The trap the filter avoids: "!= this" might match every file.
    naive = _InclusiveMetricsEvaluator(schema, NotEqualTo("agg_run_id", this))
    assert naive.eval(kept[0])
