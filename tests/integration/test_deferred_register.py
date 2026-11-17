# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Deferred Iceberg registration: `mosaic ingest --defer-register` + `mosaic
register`.

A deferred batch uploads its Parquet, records the file list in PostgreSQL
ingest_files and stays 'uploaded' (invisible to `mosaic aggregate` and the
readers); `mosaic register` registers the pending files with one committer per
table and completes the batches. These tests check that the deferred path
yields exactly the inline path's Iceberg content (the same data-file URIs:
the S3 keys are deterministic in batch_id, which restarts at 1 after each
reset) and the same aggregates and joint-called VCF, the crash windows of the
registrar (MOSAIC_TEST_FAIL_REGISTER), chunked commits, the no-op re-runs,
a mix of inline and deferred batches, and the registrar lock.

Fixture: two small chr21 batches (two 300 kb windows of the two test
samples), ingested inline once for the baseline and again, deferred or mixed,
by each test.
"""

import os

import pytest

import jc_testlib as L
from mosaic.register import REGISTER_LOCK_KEY

pytestmark = pytest.mark.integration

REGIONS = ["chr21:10000000-10300000", "chr21:20000000-20300000"]


def _ingest(subsets, defer, check=True):
    """Ingest each region's subset pair as one batch; ``defer`` is a bool or
    one bool per batch."""
    flags = defer if isinstance(defer, list) else [defer] * len(subsets)
    return [
        L.run_ingest(
            gvcfs,
            region=region,
            threads=2,
            defer_register=flag,
            check=check,
        )
        for (region, gvcfs), flag in zip(subsets, flags)
    ]


def _fresh(subsets, defer):
    """reset + init + ingest both batches."""
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    return _ingest(subsets, defer)


def _ingest_state(conn):
    """The PostgreSQL state a no-op ingest must not touch."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id, status, updated_at FROM ingestion_progress "
            "ORDER BY batch_id"
        )
        batches = cur.fetchall()
        cur.execute("SELECT count(*) FROM sample_qc")
        n_qc = cur.fetchone()[0]
        cur.execute("SELECT count(*) FROM samples")
        n_samples = cur.fetchone()[0]
    return batches, L.ingest_files_rows(conn), n_qc, n_samples


@pytest.fixture(scope="module")
def subsets(docker_stack, tmp_path_factory):
    work = tmp_path_factory.mktemp("register_stage")
    out = []
    for i, region in enumerate(REGIONS):
        gvcfs = []
        for j, src in enumerate(L.GVCFS):
            dst = os.path.join(str(work), f"b{i}_s{j}.g.vcf.gz")
            L.subset_gvcf(src, region, dst)
            gvcfs.append(dst)
        out.append((region, gvcfs))
    return out


@pytest.fixture(scope="module")
def inline(subsets, tmp_path_factory):
    """The all-inline baseline: Iceberg data files, snapshots and row
    counts after ingest, then the aggregates and the joint-called VCF."""
    _fresh(subsets, defer=False)
    catalog = L.get_catalog()
    files = L.iceberg_data_files(catalog)
    snaps = L.iceberg_snapshot_counts(catalog)
    rows = L.count_iceberg(catalog, L.INGEST_TABLES)
    L.run_aggregate(threads=2)
    conn = L.get_pg_conn()
    try:
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"] * 2
        assert L.ingest_files_rows(conn) == []
        pg = L.snapshot_pg(conn)
    finally:
        conn.close()
    vcf = os.path.join(
        str(tmp_path_factory.mktemp("register_vcf")), "i.vcf.gz"
    )
    L.run_joint_call(vcf)
    # Inline: one commit per task per table (two batches, each one file per
    # table on these 300 kb windows).
    assert snaps == {t: 2 for t in L.INGEST_TABLES}, snaps
    assert all(len(f) == 2 for f in files.values()), files
    return {
        "files": files,
        "rows": rows,
        "pg": pg,
        "vcf": L.snapshot_vcf(vcf),
    }


def _assert_iceberg_equals_inline(catalog, inline):
    assert L.iceberg_data_files(catalog) == inline["files"]
    assert L.count_iceberg(catalog, L.INGEST_TABLES) == inline["rows"]


def test_deferred_equals_inline(subsets, inline, tmp_path):
    procs = _fresh(subsets, defer=True)
    for p in procs:
        assert "awaits" not in p.stderr
        assert "uploaded: " in p.stderr and "mosaic register" in p.stderr
    catalog = L.get_catalog()
    conn = L.get_pg_conn()
    try:
        assert [s for _, s in L.batch_statuses(conn)] == ["uploaded"] * 2
        # One pending row per uploaded file per table: exactly the files
        # the inline path registered.
        rows = L.ingest_files_rows(conn)
        assert not any(reg for *_, reg in rows)
        recorded = {
            t: sorted(u for tt, u, *_ in rows if tt == t)
            for t in L.INGEST_TABLES
        }
        assert recorded == inline["files"]
        assert {b for _, _, b, _ in rows} == {1, 2}
        # No Iceberg table gained a snapshot.
        assert L.iceberg_snapshot_counts(catalog) == {
            t: 0 for t in L.INGEST_TABLES
        }

        # aggregate sees no complete batch, and says why.
        proc = L.run_aggregate(threads=2)
        assert (
            "2 batch(es) uploaded but not registered; run mosaic register"
            in proc.stderr
        )
        assert "No completed batches to aggregate." in proc.stderr
        assert L.agg_statuses(conn) == []

        proc = L.run_register()
        assert "batches completed: 2 (1, 2)" in proc.stderr
        assert all(reg for *_, reg in L.ingest_files_rows(conn))
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"] * 2
        # One snapshot per table for the whole registrar run.
        assert L.iceberg_snapshot_counts(catalog) == {
            t: 1 for t in L.INGEST_TABLES
        }
        _assert_iceberg_equals_inline(catalog, inline)

        proc = L.run_aggregate(threads=2)
        assert "uploaded but not registered" not in proc.stderr
        L.assert_pg_bitexact(L.snapshot_pg(conn), inline["pg"])
    finally:
        conn.close()
    vcf = str(tmp_path / "d.vcf.gz")
    L.run_joint_call(vcf)
    assert L.snapshot_vcf(vcf) == inline["vcf"]


def _assert_crashed_register(conn, catalog, phase):
    """State after the hook fired on sample_ref_blocks' first commit. The
    registrar goes table by table in name order: the batch_* tables are
    registered and marked, sample_ref_blocks is registered (post_commit) or
    not (pre_commit) but unmarked either way, sample_site_info untouched;
    no batch is complete."""
    rows = L.ingest_files_rows(conn)
    marked = {t for t, _, _, reg in rows if reg}
    assert marked == {t for t in L.INGEST_TABLES if t.startswith("batch_")}
    assert [s for _, s in L.batch_statuses(conn)] == ["uploaded"] * 2
    snaps = L.iceberg_snapshot_counts(catalog)
    files = L.iceberg_data_files(catalog)
    recorded = sorted(u for t, u, *_ in rows if t == "sample_ref_blocks")
    if phase == "post_commit":
        assert snaps["sample_ref_blocks"] == 1
        assert files["sample_ref_blocks"] == recorded
    else:
        assert snaps["sample_ref_blocks"] == 0
    assert snaps["sample_site_info"] == 0
    for t in L.INGEST_TABLES:
        if t.startswith("batch_"):
            assert snaps[t] == 1, (t, snaps)


@pytest.mark.restart
@pytest.mark.parametrize("phase", ["post_commit", "pre_commit"])
def test_register_crash_then_rerun(subsets, inline, phase):
    _fresh(subsets, defer=True)
    catalog = L.get_catalog()
    conn = L.get_pg_conn()
    try:
        proc = L.run_register(
            extra_env={
                "MOSAIC_TEST_FAIL_REGISTER": f"sample_ref_blocks:{phase}"
            },
            check=False,
        )
        assert proc.returncode != 0
        assert "MOSAIC_TEST_FAIL_REGISTER" in proc.stderr
        _assert_crashed_register(conn, catalog, phase)

        proc = L.run_register()
        srb = next(
            line
            for line in proc.stderr.splitlines()
            if line.strip().startswith("sample_ref_blocks:")
        )
        if phase == "post_commit":
            # The leftovers: marked from the scan, not re-added.
            assert "leftovers_marked=2 commits=0 registered=0" in srb, srb
        else:
            assert "leftovers_marked=0 commits=1 registered=2" in srb, srb
        assert "batches completed: 2 (1, 2)" in proc.stderr
        assert all(reg for *_, reg in L.ingest_files_rows(conn))
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"] * 2
        # One commit per table across both runs: no duplicate data file.
        assert L.iceberg_snapshot_counts(catalog) == {
            t: 1 for t in L.INGEST_TABLES
        }
        _assert_iceberg_equals_inline(catalog, inline)
    finally:
        conn.close()


def test_one_file_per_commit(subsets, inline):
    _fresh(subsets, defer=True)
    proc = L.run_register(files_per_commit=1)
    for t in L.INGEST_TABLES:
        assert f"{t}: pending=2 leftovers_marked=0 commits=2 " in proc.stderr
    catalog = L.get_catalog()
    assert L.iceberg_snapshot_counts(catalog) == {
        t: 2 for t in L.INGEST_TABLES
    }
    _assert_iceberg_equals_inline(catalog, inline)
    conn = L.get_pg_conn()
    try:
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"] * 2
    finally:
        conn.close()


def test_nothing_pending_and_reingest_of_an_uploaded_batch(subsets):
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    proc = L.run_register()
    assert "nothing to register" in proc.stderr
    # A database initialized before ingest_files: the registrar has nothing
    # pending, and the next ingest adds the table (ensure_ingest_schema).
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE ingest_files")
    finally:
        conn.close()
    proc = L.run_register()
    assert "nothing to register" in proc.stderr

    _ingest(subsets[:1], defer=True)
    catalog = L.get_catalog()
    conn = L.get_pg_conn()
    try:
        before = _ingest_state(conn)
        assert [b[1] for b in before[0]] == ["uploaded"]
        objects = L.bucket_listing()
        # A re-run, with or without the flag, finds the uploaded batch and
        # does nothing: no upload, no PostgreSQL write.
        for defer in (True, False):
            (proc,) = _ingest(subsets[:1], defer=defer)
            assert "awaits `mosaic register`" in proc.stderr
            assert "Uploaded aggregated" not in proc.stderr
            assert _ingest_state(conn) == before
            assert L.bucket_listing() == objects
        assert L.iceberg_snapshot_counts(catalog) == {
            t: 0 for t in L.INGEST_TABLES
        }

        L.run_register()
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"]
        proc = L.run_register()
        assert "nothing to register" in proc.stderr
        assert L.iceberg_snapshot_counts(catalog) == {
            t: 1 for t in L.INGEST_TABLES
        }
    finally:
        conn.close()


def test_mixed_inline_and_deferred(subsets, inline):
    _fresh(subsets, defer=[False, True])
    conn = L.get_pg_conn()
    try:
        assert [s for _, s in L.batch_statuses(conn)] == [
            "complete",
            "uploaded",
        ]
        # Only the deferred batch's files are pending.
        assert {b for _, _, b, _ in L.ingest_files_rows(conn)} == {2}
        # Aggregating now covers the complete batch, with the warning ...
        proc = L.run_aggregate(threads=2)
        assert "1 batch(es) uploaded but not registered" in proc.stderr
        assert L.agg_run(conn, L.latest_agg_run_id(conn))["batch_ids"] == [1]
        # ... and register + aggregate folds the other one in.
        L.run_register()
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"] * 2
        proc = L.run_aggregate(threads=2)
        assert "delta=1" in proc.stderr
        L.assert_pg_bitexact(L.snapshot_pg(conn), inline["pg"])
    finally:
        conn.close()
    catalog = L.get_catalog()
    _assert_iceberg_equals_inline(catalog, inline)
    assert L.iceberg_snapshot_counts(catalog) == {
        t: 2 for t in L.INGEST_TABLES
    }


def test_register_refuses_while_another_holds_the_lock(subsets):
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    _ingest(subsets[:1], defer=True)
    catalog = L.get_catalog()
    holder = L.get_pg_conn()
    conn = L.get_pg_conn()
    try:
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (REGISTER_LOCK_KEY,))
        proc = L.run_register(check=False)
        assert proc.returncode != 0
        assert "registrar lock" in proc.stderr
        assert not any(reg for *_, reg in L.ingest_files_rows(conn))
        assert [s for _, s in L.batch_statuses(conn)] == ["uploaded"]
        assert L.iceberg_snapshot_counts(catalog) == {
            t: 0 for t in L.INGEST_TABLES
        }
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (REGISTER_LOCK_KEY,))
        L.run_register()
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"]
    finally:
        holder.close()
        conn.close()
