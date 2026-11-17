# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the Iceberg-registration scaling fix.

The fix removes the per-batch,
O(files-already-in-table) work from the *fresh* registration path: a fresh
batch/run's data-file URIs cannot already be registered (the S3 key folds in
the SERIAL batch_id / agg_run_id), so the pre-registration full-table scan
(``_get_registered_uris``) and pyiceberg's ``check_duplicate_files`` dedup are
both skipped. This module guards that behaviour directly — the end-to-end
bit-exact resume/idempotency suites would still pass if the skip were reverted,
so they cannot catch a regression of the optimization itself.
"""

from unittest import mock

import pytest

from mosaic.ingest import (
    _COMMIT_FAILED_EXC,
    _add_files_with_retry,
    _init_iceberg_writer,
)

pytestmark = pytest.mark.unit


# ── _add_files_with_retry: check_duplicate_files gating ──────────────────────


def test_fresh_add_skips_dupcheck_on_happy_path():
    """A fresh batch's single successful add_files runs with the dedup off."""
    table = mock.MagicMock()
    _add_files_with_retry(table, ["s3://b/x"], check_duplicate_files=False)
    table.add_files.assert_called_once_with(
        file_paths=["s3://b/x"], check_duplicate_files=False
    )


def test_resume_add_keeps_dupcheck_from_first_attempt():
    """A resume passes check_duplicate_files=True, so it is on from attempt 0."""
    table = mock.MagicMock()
    _add_files_with_retry(table, ["s3://b/x"], check_duplicate_files=True)
    table.add_files.assert_called_once_with(
        file_paths=["s3://b/x"], check_duplicate_files=True
    )


def test_fresh_add_forces_dupcheck_on_conflict_retry():
    """The ambiguous-commit guard: a conflict retry re-enables the dedup even
    on a fresh batch, so a commit that landed on the catalog but reported a
    retriable error cannot silently double-register the data file."""
    table = mock.MagicMock()
    table.add_files.side_effect = [_COMMIT_FAILED_EXC("conflict"), None]
    with mock.patch("mosaic.ingest.time.sleep"):
        _add_files_with_retry(table, ["s3://b/x"], check_duplicate_files=False)
    assert table.add_files.call_count == 2
    dup_flags = [
        c.kwargs["check_duplicate_files"]
        for c in table.add_files.call_args_list
    ]
    assert dup_flags == [False, True]
    table.refresh.assert_called_once()


# ── _init_iceberg_writer: the fresh-path scan skip ───────────────────────────


def _run_writer(is_resume, registered_return=frozenset({"s3://b/old"})):
    """Build the writer with the catalog / S3 / scan calls mocked out."""
    catalog = mock.MagicMock()
    catalog.load_table.side_effect = lambda name: mock.MagicMock(name=name)
    with (
        mock.patch("mosaic.ingest.load_catalog", return_value=catalog),
        mock.patch("mosaic.ingest.boto3.client"),
        mock.patch("mosaic.ingest.build_s3_client_kwargs", return_value={}),
        mock.patch(
            "mosaic.ingest._get_registered_uris",
            return_value=set(registered_return),
        ) as scan,
    ):
        writer = _init_iceberg_writer(["s1", "s2"], is_resume)
    return writer, scan


_REG_KEYS = [
    "rb_registered",
    "si_registered",
    "brb_registered",
    "bsi_registered",
    "bac_registered",
    "bgt_registered",
]


def test_fresh_writer_skips_all_six_scans():
    writer, scan = _run_writer(is_resume=False)
    assert scan.call_count == 0
    for key in _REG_KEYS:
        assert writer[key] == set()
    # Each table's set must be an independent object (no shared-mutation alias).
    sets = [writer[k] for k in _REG_KEYS]
    assert len({id(s) for s in sets}) == len(_REG_KEYS)


def test_resume_writer_scans_all_six_tables():
    writer, scan = _run_writer(is_resume=True)
    assert scan.call_count == 6
    for key in _REG_KEYS:
        assert writer[key] == {"s3://b/old"}
