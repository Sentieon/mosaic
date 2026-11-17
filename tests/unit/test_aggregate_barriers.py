# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the aggregate's barrier registration (F2; no Docker).

The reduce and finalize units only upload; after each barrier the main
process registers the barrier's files with one add_files per Iceberg table,
then commits the done units' tier_completion rows in one transaction, and
after a metric's finalize barrier replaces the agg_* mirror in one commit.
These tests pin, with fake catalogs, tables, connections and S3 clients:

* ``group_barrier_uris``: URIs grouped per table, None skipped, order kept;
* ``register_barrier``: one commit per table, no scan and no duplicate check
  on a fresh run, one scan per table and a dedup on a resume, no commit for
  an empty barrier or a fully registered table;
* ``_finish_barrier``: Iceberg before PostgreSQL, only the done units'
  checkpoint keys, and the three MOSAIC_TEST_FAIL_UNIT phases;
* ``parse_fail_unit`` / ``_maybe_crash_unit``;
* the mirror's keys, the skipped unit's listing and ``_replace_mirror``.
"""

from contextlib import contextmanager

import psycopg2.extras
import pytest

from mosaic import aggregate as A

pytestmark = pytest.mark.unit


class FakeTable:
    def __init__(self, name, log, registered=(), has_snapshot=True):
        self._name = name
        self.log = log
        self.registered = set(registered)
        self.has_snapshot = has_snapshot

    def name(self):
        return ("ns", self._name)

    def add_files(self, file_paths, check_duplicate_files=True):
        self.log.append(
            ("add_files", self._name, list(file_paths), check_duplicate_files)
        )

    def refresh(self):
        pass


class FakeCatalog:
    def __init__(self, tables):
        self.tables = tables
        self.loaded = []

    def load_table(self, full_name):
        name = full_name.split(".", 1)[1]
        self.loaded.append(name)
        return self.tables[name]


@pytest.fixture
def scans(monkeypatch):
    """Record every registered-URI scan; a table's scan returns its
    ``registered`` set."""
    calls = []

    def fake_scan(table):
        calls.append(table._name)
        return set(table.registered)

    monkeypatch.setattr(A, "_get_registered_uris", fake_scan)
    return calls


# ── group_barrier_uris ───────────────────────────────────────────────────────


def test_group_barrier_uris_per_table_none_skipped():
    files = [
        (("tier_ref_blocks_dense", "s3://b/d0"), ("tier_ref_blocks", None)),
        (),  # a skipped unit
        (
            ("tier_ref_blocks_dense", "s3://b/d1"),
            ("tier_ref_blocks", "s3://b/s1"),
        ),
        (("tier_ref_blocks_dense", None), ("tier_ref_blocks", None)),
    ]
    assert A.group_barrier_uris(files) == {
        "tier_ref_blocks_dense": ["s3://b/d0", "s3://b/d1"],
        "tier_ref_blocks": ["s3://b/s1"],
    }
    # Table order: first seen.
    assert list(A.group_barrier_uris(files)) == [
        "tier_ref_blocks_dense",
        "tier_ref_blocks",
    ]
    assert A.group_barrier_uris([(("t", None),), ()]) == {}
    assert A.group_barrier_uris([]) == {}


# ── register_barrier ─────────────────────────────────────────────────────────


def _catalog(log, **registered):
    names = ("tier_site_info", "tier_ref_blocks", "tier_ref_blocks_dense")
    return FakeCatalog(
        {n: FakeTable(n, log, registered.get(n, ())) for n in names}
    )


def test_register_barrier_fresh_one_commit_per_table_no_scan(scans):
    log = []
    catalog = _catalog(log)
    got = A.register_barrier(
        catalog,
        {
            "tier_ref_blocks_dense": ["s3://b/d0", "s3://b/d1"],
            "tier_ref_blocks": ["s3://b/s1"],
        },
        fresh=True,
    )
    assert got == (3, 2)
    assert scans == []
    assert log == [
        (
            "add_files",
            "tier_ref_blocks_dense",
            ["s3://b/d0", "s3://b/d1"],
            False,
        ),
        ("add_files", "tier_ref_blocks", ["s3://b/s1"], False),
    ]


def test_register_barrier_resume_scans_once_and_dedups(scans):
    log = []
    catalog = _catalog(
        log,
        tier_site_info={"s3://b/a", "s3://b/other"},
        tier_ref_blocks={"s3://b/s1"},
    )
    got = A.register_barrier(
        catalog,
        {
            "tier_site_info": ["s3://b/a", "s3://b/b", "s3://b/c"],
            "tier_ref_blocks": ["s3://b/s1"],
        },
        fresh=False,
    )
    # One scan per table; the registered URIs dropped; a table whose URIs
    # are all registered makes no commit; the rest keep the duplicate check.
    assert sorted(scans) == ["tier_ref_blocks", "tier_site_info"]
    assert log == [
        ("add_files", "tier_site_info", ["s3://b/b", "s3://b/c"], True),
    ]
    assert got == (2, 1)


def test_register_barrier_empty_makes_no_commit(scans):
    log = []
    catalog = _catalog(log)
    assert A.register_barrier(catalog, {}, fresh=True) == (0, 0)
    assert A.register_barrier(catalog, {}, fresh=False) == (0, 0)
    assert log == [] and scans == [] and catalog.loaded == []


# ── _finish_barrier ──────────────────────────────────────────────────────────


class FakeConn:
    def __init__(self, log):
        self.log = log

    @contextmanager
    def cursor(self):
        yield "cur"

    def commit(self):
        self.log.append(("commit",))


@pytest.fixture
def checkpoints(monkeypatch):
    """Record execute_values' rows instead of running SQL."""

    def fake_execute_values(cur, sql, rows):
        assert "INSERT INTO tier_completion" in sql
        assert "ON CONFLICT DO NOTHING" in sql
        log.append(("checkpoints", list(rows)))

    log = []
    monkeypatch.setattr(psycopg2.extras, "execute_values", fake_execute_values)
    return log


def _done():
    return [
        ((7, "site_info", 1, 1, 0), (("tier_site_info", "s3://b/g0"),)),
        ((7, "site_info", 1, 1, 1), (("tier_site_info", None),)),
        ((7, "site_info", 2, 1, 0), (("tier_site_info", "s3://b/c2g0"),)),
    ]


def test_finish_barrier_iceberg_then_one_checkpoint_transaction(
    scans, checkpoints, monkeypatch
):
    monkeypatch.delenv("MOSAIC_TEST_FAIL_UNIT", raising=False)
    log = checkpoints
    catalog = _catalog(log)
    got = A._finish_barrier(FakeConn(log), catalog, _done(), fresh=True)
    assert got == (2, 1)
    assert log == [
        ("add_files", "tier_site_info", ["s3://b/g0", "s3://b/c2g0"], False),
        (
            "checkpoints",
            [
                (7, "site_info", 1, 1, 0),
                (7, "site_info", 1, 1, 1),  # no file, still checkpointed
                (7, "site_info", 2, 1, 0),
            ],
        ),
        ("commit",),
    ]


def test_finish_barrier_nothing_done(scans, checkpoints, monkeypatch):
    """Every unit skipped: no registration, no checkpoint statement."""
    monkeypatch.setenv("MOSAIC_TEST_FAIL_UNIT", "site_info:*:1:0:pre_register")
    log = checkpoints
    assert A._finish_barrier(FakeConn(log), _catalog(log), [], False) == (0, 0)
    assert log == [("commit",)]


@pytest.mark.parametrize(
    "phase,expect",
    [
        ("pre_register", []),
        ("pre_ckpt", ["add_files"]),
        ("post_ckpt", ["add_files", "checkpoints", "commit"]),
    ],
)
def test_finish_barrier_fault_phases(
    scans, checkpoints, monkeypatch, phase, expect
):
    monkeypatch.setenv("MOSAIC_TEST_FAIL_UNIT", f"site_info:*:1:1:{phase}")
    log = checkpoints
    with pytest.raises(RuntimeError, match=f"site_info c1 t1 g1 @{phase}"):
        A._finish_barrier(FakeConn(log), _catalog(log), _done(), fresh=True)
    assert [e[0] for e in log] == expect


def test_finish_barrier_fault_needs_the_unit_to_have_run(
    scans, checkpoints, monkeypatch
):
    """A named unit that was skipped (not in ``done``) raises nothing."""
    monkeypatch.setenv("MOSAIC_TEST_FAIL_UNIT", "site_info:*:2:0:pre_ckpt")
    log = checkpoints
    A._finish_barrier(FakeConn(log), _catalog(log), _done(), fresh=True)
    assert [e[0] for e in log] == ["add_files", "checkpoints", "commit"]


# ── MOSAIC_TEST_FAIL_UNIT ────────────────────────────────────────────────────


@pytest.mark.parametrize("phase", ["pre_register", "pre_ckpt", "post_ckpt"])
def test_parse_fail_unit_phases(phase):
    assert A.parse_fail_unit(f"ref_blocks:*:1:3:{phase}") == (
        "ref_blocks",
        "*",
        "1",
        "3",
        phase,
    )
    assert A.parse_fail_unit(f"prune:*:-1:0:{phase}")[2] == "-1"


@pytest.mark.parametrize("spec", [None, ""])
def test_parse_fail_unit_unset(spec):
    assert A.parse_fail_unit(spec) is None


@pytest.mark.parametrize(
    "spec",
    [
        "site_info:*:1:0",
        "site_info:*:1:0:pre_ckpt:x",
        "site_info:*:1:0:pre_commit",
        "site_info:*::0:pre_ckpt",
        "site_info",
    ],
)
def test_parse_fail_unit_malformed(spec):
    with pytest.raises(ValueError, match="MOSAIC_TEST_FAIL_UNIT"):
        A.parse_fail_unit(spec)


def test_maybe_crash_unit_matching(monkeypatch):
    monkeypatch.setenv("MOSAIC_TEST_FAIL_UNIT", "site_info:3:2:0:pre_ckpt")
    A._maybe_crash_unit("site_info", 3, 2, 0, "post_ckpt")  # other phase
    A._maybe_crash_unit("site_info", 4, 2, 0, "pre_ckpt")  # other chrom
    A._maybe_crash_unit("site_info", 3, 1, 0, "pre_ckpt")  # other tier
    A._maybe_crash_unit("site_info", 3, 2, 1, "pre_ckpt")  # other group
    A._maybe_crash_unit("allele_counts", 3, 2, 0, "pre_ckpt")
    with pytest.raises(RuntimeError, match="injected crash site_info c3"):
        A._maybe_crash_unit("site_info", 3, 2, 0, "pre_ckpt")
    monkeypatch.setenv("MOSAIC_TEST_FAIL_UNIT", "site_info:*:2:0:pre_ckpt")
    with pytest.raises(RuntimeError):
        A._maybe_crash_unit("site_info", 9, 2, 0, "pre_ckpt")
    monkeypatch.delenv("MOSAIC_TEST_FAIL_UNIT")
    A._maybe_crash_unit("site_info", 3, 2, 0, "pre_ckpt")


# ── the agg_* mirror ─────────────────────────────────────────────────────────


def test_mirror_keys_under_the_table_prefix():
    assert (
        A._mirror_s3_key("agg_site_info", 12, 3, 0)
        == "agg_site_info/12/c3/0.parquet"
    )
    assert A._mirror_prefix("agg_site_info", 12, 1) == "agg_site_info/12/c1/"
    # Covered by the per-table prefixes `mosaic clean` deletes.
    from mosaic.init_db import ICEBERG_TABLES

    for spec in A.SPECS.values():
        assert spec.agg_table in ICEBERG_TABLES
        assert A._mirror_s3_key(spec.agg_table, 1, 0, 0).startswith(
            f"{spec.agg_table}/"
        )


class FakePaginator:
    def __init__(self, keys):
        self.keys = keys

    def paginate(self, Bucket, Prefix):
        hits = [k for k in self.keys if k.startswith(Prefix)]
        # Two pages, to exercise the pagination.
        yield {"Contents": [{"Key": k} for k in hits[:2]]}
        yield {"Contents": [{"Key": k} for k in hits[2:]]}
        yield {}


class FakeS3:
    def __init__(self, keys):
        self.keys = keys

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        return FakePaginator(self.keys)


def test_list_mirror_uris_sorted_by_chunk_index():
    keys = [
        "agg_site_info/5/c1/10.parquet",
        "agg_site_info/5/c1/2.parquet",
        "agg_site_info/5/c1/0.parquet",
        "agg_site_info/5/c1/1.parquet",
        "agg_site_info/5/c10/0.parquet",  # another chromosome
        "agg_site_info/6/c1/0.parquet",  # another run
        "agg_site_info/5/c1/tmp.txt",
    ]
    got = A._list_mirror_uris(FakeS3(keys), "agg_site_info", 5, 1)
    b = A.S3_BUCKET
    assert got == [
        f"s3://{b}/agg_site_info/5/c1/{i}.parquet" for i in (0, 1, 2, 10)
    ]
    assert A._list_mirror_uris(FakeS3(keys), "agg_site_info", 5, 7) == []


class FakeTxn:
    def __init__(self, log):
        self.log = log

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is None:
            self.log.append(("commit",))

    def delete(self, expr):
        self.log.append(("delete", type(expr).__name__))

    def add_files(self, file_paths, check_duplicate_files=True):
        self.log.append(("add_files", list(file_paths), check_duplicate_files))


class FakeMirrorTable:
    def __init__(self, log, snapshot):
        self.log = log
        self.snapshot = snapshot

        class M:
            snapshots = []

        self.metadata = M()

    def name(self):
        return ("ns", "agg_site_info")

    def current_snapshot(self):
        return self.snapshot

    def transaction(self):
        return FakeTxn(self.log)

    def refresh(self):
        pass


@pytest.mark.parametrize("snapshot", [None, object()])
def test_replace_mirror_one_transaction(snapshot):
    log = []
    table = FakeMirrorTable(log, snapshot)
    catalog = FakeCatalog({"agg_site_info": table})
    A._replace_mirror(catalog, "agg_site_info", ["s3://b/1", "s3://b/2"])
    deletes = [("delete", "AlwaysTrue")] if snapshot is not None else []
    assert log == deletes + [
        ("add_files", ["s3://b/1", "s3://b/2"], False),
        ("commit",),
    ]
