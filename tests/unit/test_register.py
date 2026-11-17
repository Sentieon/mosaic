# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the deferred-registration pieces that need no Docker.

`mosaic register` (mosaic/register.py) groups the pending ingest_files rows
per Iceberg table and chunks each table's files into commits of at most
``--files-per-commit``; it completes a batch when its last pending row, in
any table, is marked. The run loop itself (Iceberg first, PostgreSQL second,
the leftover path, the advisory lock) needs the stack and is covered by
tests/integration/test_deferred_register.py.
"""

import argparse
from unittest import mock

import pytest

from mosaic import cli
from mosaic.genotypes import BATCH_GENOTYPES_TABLE
from mosaic.ingest import INGEST_ICEBERG_TABLES, record_ingest_files
from mosaic.register import (
    DEFAULT_FILES_PER_COMMIT,
    REGISTERABLE_TABLES,
    CompletionTracker,
    PendingFile,
    _maybe_fail_register,
    chunk_files,
    files_per_commit_type,
    group_by_table,
    parse_fail_register,
)

pytestmark = pytest.mark.unit


def _files(table, batch_id, n, start=0):
    return [
        PendingFile(table, f"s3://b/{table}/{batch_id}/{i:03d}", batch_id)
        for i in range(start, start + n)
    ]


# ── grouping + chunking into per-table commits ───────────────────────────────


def test_group_by_table_keeps_first_seen_table_order_and_row_order():
    rows = (
        _files("batch_allele_counts", 1, 2)
        + _files("batch_genotypes", 1, 1)
        + _files("batch_genotypes", 2, 2)
        + _files("sample_site_info", 2, 1)
    )
    grouped = group_by_table(rows)
    assert [t for t, _ in grouped] == [
        "batch_allele_counts",
        "batch_genotypes",
        "sample_site_info",
    ]
    assert dict(grouped)["batch_genotypes"] == rows[2:5]
    assert sum(len(f) for _, f in grouped) == len(rows)


def test_chunks_respect_the_files_per_commit_boundary_in_order():
    files = _files("sample_ref_blocks", 1, 5)
    chunks = chunk_files(files, 2)
    assert [len(c) for c in chunks] == [2, 2, 1]
    assert [f for c in chunks for f in c] == files


def test_exact_multiple_has_no_empty_tail():
    chunks = chunk_files(_files("sample_ref_blocks", 1, 6), 3)
    assert [len(c) for c in chunks] == [3, 3]


def test_table_with_fewer_rows_than_the_chunk_is_one_commit():
    files = _files("batch_site_info", 4, 3)
    assert chunk_files(files, DEFAULT_FILES_PER_COMMIT) == [files]


def test_one_file_per_commit():
    files = _files("batch_ref_blocks", 1, 3)
    assert chunk_files(files, 1) == [[f] for f in files]


def test_nothing_pending_is_no_commit():
    assert chunk_files([], 500) == []


def test_chunk_size_below_one_is_refused():
    with pytest.raises(ValueError):
        chunk_files(_files("batch_ref_blocks", 1, 2), 0)


# ── batch completion ────────────────────────────────────────────────────────


def test_batch_completes_only_when_its_last_row_in_any_table_is_marked():
    b1_rb = _files("sample_ref_blocks", 1, 2)
    b1_gt = _files("batch_genotypes", 1, 1)
    b2_rb = _files("sample_ref_blocks", 2, 1)
    tracker = CompletionTracker(b1_rb + b1_gt + b2_rb)
    # Batch 1's first table done, its genotype file still pending: not yet.
    assert tracker.mark(b1_rb[:1]) == []
    assert tracker.mark(b1_rb[1:] + b2_rb) == [2]
    # Its last pending row, in another table: now.
    assert tracker.mark(b1_gt) == [1]
    assert all(n == 0 for n in tracker.pending.values())


def test_a_chunk_can_complete_several_batches():
    rows = _files("batch_site_info", 3, 1) + _files("batch_site_info", 1, 2)
    tracker = CompletionTracker(rows)
    assert tracker.mark(rows) == [1, 3]


def test_marking_a_row_twice_is_an_error():
    rows = _files("batch_site_info", 1, 1)
    tracker = CompletionTracker(rows)
    assert tracker.mark(rows) == [1]
    with pytest.raises(ValueError):
        tracker.mark(rows)


def test_a_batch_absent_from_the_pending_set_never_completes():
    tracker = CompletionTracker(_files("batch_site_info", 1, 1))
    with pytest.raises(ValueError):
        tracker.mark(_files("batch_site_info", 9, 1))


# ── the MOSAIC_TEST_FAIL_REGISTER hook ───────────────────────────────────────


@pytest.mark.parametrize("spec", [None, ""])
def test_hook_unset_is_inert(spec):
    assert parse_fail_register(spec) is None
    _maybe_fail_register(None, "sample_ref_blocks", "pre_commit")


@pytest.mark.parametrize(
    "spec, want",
    [
        (
            "sample_ref_blocks:post_commit",
            ("sample_ref_blocks", "post_commit"),
        ),
        ("batch_genotypes:pre_commit", ("batch_genotypes", "pre_commit")),
    ],
)
def test_hook_parses(spec, want):
    assert parse_fail_register(spec) == want


@pytest.mark.parametrize(
    "spec",
    [
        "sample_ref_blocks",
        "sample_ref_blocks:",
        ":pre_commit",
        "sample_ref_blocks:during_commit",
        "sample_ref_blocks:pre_ckpt",
    ],
)
def test_malformed_hook_is_refused(spec):
    with pytest.raises(ValueError):
        parse_fail_register(spec)


def test_hook_fires_only_on_its_table_and_phase():
    hook = parse_fail_register("sample_ref_blocks:post_commit")
    _maybe_fail_register(hook, "sample_ref_blocks", "pre_commit")
    _maybe_fail_register(hook, "sample_site_info", "post_commit")
    with pytest.raises(RuntimeError, match="sample_ref_blocks @post_commit"):
        _maybe_fail_register(hook, "sample_ref_blocks", "post_commit")


# ── --files-per-commit ───────────────────────────────────────────────────────


@pytest.mark.parametrize("value, want", [("1", 1), ("500", 500)])
def test_files_per_commit_accepts_positive_integers(value, want):
    assert files_per_commit_type(value) == want


@pytest.mark.parametrize("value", ["0", "-3", "x", "1.5"])
def test_files_per_commit_refuses_the_rest(value):
    with pytest.raises(argparse.ArgumentTypeError):
        files_per_commit_type(value)


def test_register_subcommand_parses_and_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["register"])
    assert args.files_per_commit == DEFAULT_FILES_PER_COMMIT == 500
    args = parser.parse_args(["register", "--files-per-commit", "7"])
    assert args.files_per_commit == 7
    with pytest.raises(SystemExit):
        parser.parse_args(["register", "--files-per-commit", "0"])


# ── the ingest side ──────────────────────────────────────────────────────────


def test_ingest_defer_register_flag():
    parser = cli.build_parser()
    assert not parser.parse_args(["ingest", "a.g.vcf.gz"]).defer_register
    assert parser.parse_args(
        ["ingest", "a.g.vcf.gz", "--defer-register"]
    ).defer_register


def test_registerable_tables_are_the_six_ingest_tables():
    names = [name for name, _ in INGEST_ICEBERG_TABLES]
    assert len(names) == len(set(names)) == 6
    assert set(names) == REGISTERABLE_TABLES
    assert BATCH_GENOTYPES_TABLE in REGISTERABLE_TABLES


def test_record_ingest_files_offers_one_row_per_table_uri():
    cur = mock.MagicMock()
    with mock.patch("mosaic.ingest.psycopg2.extras.execute_values") as ev:
        n = record_ingest_files(
            cur,
            7,
            [
                ("sample_ref_blocks", ["s3://b/rb/1", "s3://b/rb/2"]),
                ("batch_genotypes", []),
                ("batch_site_info", ["s3://b/bsi/1"]),
            ],
        )
    assert n == 3
    (call,) = ev.call_args_list
    sql, rows = call.args[1], call.args[2]
    assert "ON CONFLICT DO NOTHING" in sql
    assert rows == [
        ("sample_ref_blocks", "s3://b/rb/1", 7),
        ("sample_ref_blocks", "s3://b/rb/2", 7),
        ("batch_site_info", "s3://b/bsi/1", 7),
    ]


def test_record_ingest_files_with_nothing_uploaded_writes_nothing():
    with mock.patch("mosaic.ingest.psycopg2.extras.execute_values") as ev:
        assert record_ingest_files(mock.MagicMock(), 7, [("x", [])]) == 0
    ev.assert_not_called()
