# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Iceberg maintenance: `mosaic maintain`.

`mosaic maintain` sets the table properties, rewrites each table's manifests
(a merge-append of no files), expires the snapshots the retention policy
drops and deletes every file only they needed (plus the files the retained
snapshots name only as DELETED: the staging the aggregate pruned), and with
--orphans sweeps the upload prefixes and table locations for objects nothing
references. These tests check that it is invisible to every reader (the data
files, the PG aggregates, the joint-called VCF, the next aggregate), that the
pruned staging leaves the bucket, the manifest rewrite, the orphan sweep and
its guards (pending ingest_files rows, the grace window, --dry-run), the
crash windows of MOSAIC_TEST_FAIL_MAINTAIN, the lock and the
in-progress-aggregation guard, and the table properties.

Fixture: three small chr21 batches (three 300 kb windows of the two test
samples); each test stages what it needs from scratch (reset + init +
ingest), since maintenance is destructive.
"""

import os

import pytest

import jc_testlib as L
from mosaic.init_db import ICEBERG_TABLE_PROPERTIES
from mosaic.register import REGISTER_LOCK_KEY

pytestmark = pytest.mark.integration

REGIONS = [
    "chr21:10000000-10300000",
    "chr21:20000000-20300000",
    "chr21:25000000-25300000",
]
HOOK = "MOSAIC_TEST_FAIL_MAINTAIN"
PROPERTY = next(iter(ICEBERG_TABLE_PROPERTIES))


@pytest.fixture(scope="module")
def subsets(docker_stack, tmp_path_factory):
    work = tmp_path_factory.mktemp("maintain_stage")
    out = []
    for i, region in enumerate(REGIONS):
        gvcfs = []
        for j, src in enumerate(L.GVCFS):
            dst = os.path.join(str(work), f"b{i}_s{j}.g.vcf.gz")
            L.subset_gvcf(src, region, dst)
            gvcfs.append(dst)
        out.append((region, gvcfs))
    return out


def _stage(subsets, defer=True, files_per_commit=None):
    """reset + init + ingest one batch per subset (deferred batches then
    registered)."""
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    for region, gvcfs in subsets:
        L.run_ingest(gvcfs, region=region, threads=2, defer_register=defer)
    if defer:
        L.run_register(files_per_commit=files_per_commit)


def _line(proc, table):
    """The per-table line of a `mosaic maintain` run."""
    return next(
        line.strip()
        for line in proc.stderr.splitlines()
        if line.strip().startswith(f"{table}:")
    )


def _all_objects(catalog):
    """{uri: (etag, last_modified)} of every object of every table."""
    out = {}
    for t in L.ICEBERG_TABLES:
        out.update(L.lake_objects(catalog, t))
    return out


def _two_runs(conn):
    """Run 1 on batch 1, run 2 on batches 1-2 (incremental: its completion
    prunes run 1's staging)."""
    L.set_completed_batches(conn, [1])
    L.run_aggregate(threads=2)
    L.set_completed_batches(conn, [1, 2])
    proc = L.run_aggregate(threads=2)
    assert "parent=1 delta=1" in proc.stderr, proc.stderr[-2000:]


# ── 1. invisible to readers ──────────────────────────────────────────────────


def test_maintain_changes_nothing_a_reader_sees(subsets, tmp_path):
    _stage(subsets, defer=True)
    # --fan-in 2: two tiers and a prune, so the tier-tree staging tables
    # have more than one snapshot (since F2 each barrier commits once: at
    # the default fan-in a three-batch run is one snapshot per table).
    L.run_aggregate(threads=2, fan_in=2)
    catalog = L.get_catalog()
    files = L.iceberg_data_files(catalog, L.ICEBERG_TABLES)
    rows = L.count_iceberg(catalog, L.ICEBERG_TABLES)
    before = L.iceberg_snapshot_counts(catalog, L.ICEBERG_TABLES)
    assert all(n >= 1 for n in before.values()), before
    assert sum(before.values()) > len(L.ICEBERG_TABLES), before
    conn = L.get_pg_conn()
    try:
        pg = L.snapshot_pg(conn)
        run = L.latest_agg_run_id(conn)
        vcf = str(tmp_path / "before.vcf.gz")
        L.run_joint_call(vcf)
        vcf_before = L.snapshot_vcf(vcf)

        proc = L.run_maintain(older_than_hours=0)
        assert L.iceberg_snapshot_counts(catalog, L.ICEBERG_TABLES) == {
            t: 1 for t in L.ICEBERG_TABLES
        }
        assert L.iceberg_data_files(catalog, L.ICEBERG_TABLES) == files
        assert L.count_iceberg(catalog, L.ICEBERG_TABLES) == rows
        assert "total: 13 table(s)" in proc.stderr
        # Only what the remaining snapshot references is left.
        L.assert_lake_exact(catalog)
        assert L.snapshot_pg(conn) == pg
        vcf = str(tmp_path / "after.vcf.gz")
        L.run_joint_call(vcf)
        assert L.snapshot_vcf(vcf) == vcf_before

        proc = L.run_aggregate(threads=2, fan_in=2)
        assert f"up to date (agg_run_id={run})" in proc.stderr
        # The batch_* partials are intact: a full rebuild is bit-exact.
        L.run_aggregate(threads=2, fan_in=2, rebuild=True)
        L.assert_pg_bitexact(L.snapshot_pg(conn), pg)
    finally:
        conn.close()


# ── 2. the pruned staging is reclaimed ───────────────────────────────────────


def test_pruned_staging_is_reclaimed(subsets):
    _stage(subsets, defer=False)
    catalog = L.get_catalog()
    conn = L.get_pg_conn()
    try:
        L.set_completed_batches(conn, [1])
        L.run_aggregate(threads=2)
        run1 = L.iceberg_data_files(catalog, L.STAGING_TABLES)
        L.set_completed_batches(conn, [1, 2])
        proc = L.run_aggregate(threads=2)
        assert "parent=1 delta=1" in proc.stderr
        run2 = L.iceberg_data_files(catalog, L.STAGING_TABLES)
        for t in L.STAGING_TABLES:
            # Run 2's completion pruned all of run 1's staging (metadata
            # only: the objects are still in the bucket).
            assert run1[t] and run2[t], t
            assert not set(run1[t]) & set(run2[t]), t
            assert set(run1[t]) <= set(L.lake_objects(catalog, t)), t
        for t in ("tier_site_info", "tier_allele_counts"):
            # Run 2's own non-root tier (its delta root under the fold
            # unit), pruned at its completion, is in the bucket too.
            uploads = set(L.bucket_listing(L.TEST_BUCKET, f"{t}/"))
            uploads = {f"s3://{L.TEST_BUCKET}/{k}" for k in uploads}
            assert uploads - set(run1[t]) - set(run2[t]), t
        refs_before = {
            t: L.table_references(catalog, t) for t in L.ICEBERG_TABLES
        }

        proc = L.run_maintain(older_than_hours=0)
        for t in L.STAGING_TABLES:
            line = _line(proc, t)
            assert "reclaimed 0 files" not in line, line
            # The bucket holds run 2's roots and nothing else.
            data = set(L.bucket_listing(L.TEST_BUCKET, f"{t}/"))
            assert {f"s3://{L.TEST_BUCKET}/{k}" for k in data} == set(
                run2[t]
            ), t
        assert L.iceberg_data_files(catalog, L.STAGING_TABLES) == run2
        for t in L.ICEBERG_TABLES:
            # The expired snapshots' lists and manifests are gone from the
            # metadata prefix; the retained snapshot's are there.
            after = L.table_references(catalog, t)
            kept = after["lists"] | after["manifests"]
            old = refs_before[t]["lists"] | refs_before[t]["manifests"]
            listing = set(L.location_listing(catalog, t, "metadata/"))
            assert kept <= listing, t
            assert not (old - kept) & listing, t
        L.assert_lake_exact(catalog)

        # Run 2's roots are still readable: run 3 folds batch 3 onto them,
        # bit-exact against a rebuild.
        L.set_completed_batches(conn, [1, 2, 3])
        proc = L.run_aggregate(threads=2)
        assert "parent=2 delta=1" in proc.stderr
        incremental = L.snapshot_pg(conn)
        L.run_aggregate(threads=2, rebuild=True)
        L.assert_pg_bitexact(L.snapshot_pg(conn), incremental)
    finally:
        conn.close()


# ── 3. manifest rewrite ──────────────────────────────────────────────────────


def test_manifest_rewrite(subsets):
    # One file per commit: three snapshots and three manifests per ingest
    # table.
    _stage(subsets, defer=True, files_per_commit=1)
    catalog = L.get_catalog()
    files = L.iceberg_data_files(catalog)
    rows = L.count_iceberg(catalog, L.INGEST_TABLES)
    assert L.manifest_counts(catalog) == {t: 3 for t in L.INGEST_TABLES}
    assert L.iceberg_snapshot_counts(catalog) == {
        t: 3 for t in L.INGEST_TABLES
    }

    # --no-rewrite-manifests, default retention (all young): nothing to do.
    proc = L.run_maintain(rewrite_manifests=False)
    for t in L.ICEBERG_TABLES:
        assert _line(proc, t) == f"{t}: nothing to do"
    assert L.manifest_counts(catalog) == {t: 3 for t in L.INGEST_TABLES}

    proc = L.run_maintain()
    for t in L.INGEST_TABLES:
        line = _line(proc, t)
        assert "manifests 3 -> 1; snapshots 3 -> 4; reclaimed 0 files" in line
    assert L.manifest_counts(catalog) == {t: 1 for t in L.INGEST_TABLES}
    assert L.iceberg_data_files(catalog) == files
    assert L.count_iceberg(catalog, L.INGEST_TABLES) == rows

    # Already packed: nothing to do.
    proc = L.run_maintain()
    for t in L.ICEBERG_TABLES:
        assert _line(proc, t) == f"{t}: nothing to do"
    assert L.iceberg_snapshot_counts(catalog) == {
        t: 4 for t in L.INGEST_TABLES
    }

    # By count only: the three pre-rewrite snapshots (and their small
    # manifests) go.
    proc = L.run_maintain(older_than_hours=0)
    for t in L.INGEST_TABLES:
        line = _line(proc, t)
        assert "manifests 1 -> 1; snapshots 4 -> 1" in line, line
        assert "(data 0, manifests 3, lists 3)" in line, line
    assert L.iceberg_snapshot_counts(catalog) == {
        t: 1 for t in L.INGEST_TABLES
    }
    assert L.manifest_counts(catalog) == {t: 1 for t in L.INGEST_TABLES}
    assert L.iceberg_data_files(catalog) == files
    assert L.count_iceberg(catalog, L.INGEST_TABLES) == rows
    L.assert_lake_exact(catalog)


# ── 4. orphan sweep ──────────────────────────────────────────────────────────


def test_orphan_sweep(subsets):
    _stage(subsets[:1], defer=True)
    L.run_aggregate(threads=2)
    # A deferred batch left uploaded: its files are pending, referenced by
    # no snapshot.
    L.run_ingest(
        subsets[1][1], region=subsets[1][0], threads=2, defer_register=True
    )
    catalog = L.get_catalog()
    conn = L.get_pg_conn()
    s3 = L.s3_client()
    try:
        rows = [r for r in L.ingest_files_rows(conn) if not r[3]]
        assert rows and {b for _, _, b, _ in rows} == {2}
        pending = [u for _, u, _, _ in rows]
        stray_data = (
            f"s3://{L.TEST_BUCKET}/sample_ref_blocks/stray/data.parquet"
        )
        s3.put_object(
            Bucket=L.TEST_BUCKET,
            Key="sample_ref_blocks/stray/data.parquet",
            Body=b"stray",
        )
        location = catalog.load_table(
            f"{L.TEST_NAMESPACE}.sample_site_info"
        ).location()
        stray_meta = f"{location}/metadata/zz.metadata.json"
        bucket, _, key = stray_meta.split("://", 1)[1].partition("/")
        s3.put_object(Bucket=bucket, Key=key, Body=b"{}")
        strays = {stray_data, stray_meta}
        objects = _all_objects(catalog)
        assert strays <= set(objects) and set(pending) <= set(objects)
        snaps = L.iceberg_snapshot_ids(catalog)

        proc = L.run_maintain(orphans=True, orphan_grace_hours=0, dry_run=True)
        assert "DRY RUN" in proc.stderr
        named = {
            line.strip().split("would delete ", 1)[1]
            for line in proc.stderr.splitlines()
            if line.strip().startswith("would delete s3://")
        }
        assert named == strays, proc.stderr[-3000:]
        assert ", would delete 2 (" in proc.stderr
        assert f"pending {len(pending)}," in proc.stderr
        assert _all_objects(catalog) == objects
        assert L.iceberg_snapshot_ids(catalog) == snaps

        proc = L.run_maintain(orphans=True, orphan_grace_hours=0)
        assert ", deleted 2 (" in proc.stderr, proc.stderr[-3000:]
        left = _all_objects(catalog)
        assert not strays & set(left)
        assert set(left) == set(objects) - strays
        for t, uris in L.iceberg_data_files(catalog, L.ICEBERG_TABLES).items():
            assert set(uris) <= set(left), t
        assert set(pending) <= set(left)

        # The pending batch registers and folds in.
        proc = L.run_register()
        assert "batches completed: 1 (2)" in proc.stderr
        proc = L.run_aggregate(threads=2)
        assert "parent=1 delta=1" in proc.stderr
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"] * 2

        # Default grace: a young stray stays.
        s3.put_object(
            Bucket=L.TEST_BUCKET,
            Key="sample_site_info/stray/data.parquet",
            Body=b"stray",
        )
        proc = L.run_maintain(orphans=True)
        assert "young 1, deleted 0 (0 bytes)" in proc.stderr, proc.stderr
        assert "sample_site_info/stray/data.parquet" in L.bucket_listing()
    finally:
        conn.close()


# ── 5. crash and re-run ──────────────────────────────────────────────────────


@pytest.mark.restart
@pytest.mark.parametrize("phase", ["pre_expire", "post_expire"])
def test_maintain_crash_then_rerun(subsets, phase):
    _stage(subsets[:2], defer=False)
    catalog = L.get_catalog()
    conn = L.get_pg_conn()
    try:
        _two_runs(conn)
    finally:
        conn.close()
    crashed = "tier_site_info"
    order = L.ICEBERG_TABLES
    later = order[order.index(crashed) + 1 :]
    ids = L.iceberg_snapshot_ids(catalog)
    assert len(ids[crashed]) > 1
    objects = L.lake_objects(catalog, crashed)

    proc = L.run_maintain(
        older_than_hours=0, extra_env={HOOK: f"{crashed}:{phase}"}, check=False
    )
    assert proc.returncode != 0
    assert HOOK in proc.stderr
    after = L.iceberg_snapshot_ids(catalog)
    # The tables before it were maintained, those after it untouched.
    for t in order[: order.index(crashed)]:
        assert len(after[t]) == 1, t
    for t in later:
        assert after[t] == ids[t], t
    # Nothing of its files deleted either way.
    assert set(objects) <= set(L.lake_objects(catalog, crashed))
    if phase == "pre_expire":
        # Nothing expired (the rewrite, which precedes, may have committed).
        assert ids[crashed] <= after[crashed]
    else:
        assert len(after[crashed]) == 1

    proc = L.run_maintain(older_than_hours=0)
    line = _line(proc, crashed)
    if phase == "pre_expire":
        assert "reclaimed 0 files" not in line, line
        L.assert_lake_exact(catalog)
    else:
        # The re-run has nothing more to expire for it; its files are
        # unreferenced now, which the sweep reclaims.
        assert line == f"{crashed}: nothing to do", line
        leftovers = set(L.lake_objects(catalog, crashed))
        refs = L.table_references(catalog, crashed)
        unreferenced = leftovers - set().union(*refs.values())
        assert unreferenced
        L.assert_lake_exact(catalog, [t for t in order if t != crashed])
        proc = L.run_maintain(orphans=True, orphan_grace_hours=0)
        assert f", deleted {len(unreferenced)} (" in proc.stderr, proc.stderr
        L.assert_lake_exact(catalog)
    assert L.iceberg_snapshot_counts(catalog, L.ICEBERG_TABLES) == {
        t: 1 for t in order
    }


# ── 6. guards ────────────────────────────────────────────────────────────────


def test_lock_and_in_progress_aggregation_guard(subsets):
    # Two inline batches: two snapshots per ingest table.
    _stage(subsets[:2], defer=False)
    L.run_aggregate(threads=2)
    catalog = L.get_catalog()
    ids = L.iceberg_snapshot_ids(catalog)
    assert any(len(v) > 1 for v in ids.values())
    objects = _all_objects(catalog)
    holder = L.get_pg_conn()
    conn = L.get_pg_conn()
    try:
        run = L.latest_agg_run_id(conn)
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE aggregation_progress SET status = 'in_progress' "
                "WHERE agg_run_id = %s",
                (run,),
            )
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (REGISTER_LOCK_KEY,))

        # --dry-run takes neither the lock nor the guard.
        proc = L.run_maintain(older_than_hours=0, orphans=True, dry_run=True)
        assert "would reclaim" in proc.stderr

        proc = L.run_maintain(older_than_hours=0, check=False)
        assert proc.returncode == 1
        assert "metadata-writer lock" in proc.stderr

        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (REGISTER_LOCK_KEY,))
        proc = L.run_maintain(older_than_hours=0, check=False)
        assert proc.returncode == 1
        assert f"aggregation run(s) [{run}] in progress" in proc.stderr
        assert L.iceberg_snapshot_ids(catalog) == ids
        assert _all_objects(catalog) == objects

        proc = L.run_maintain(older_than_hours=0, force=True)
        assert "continuing (--force)" in proc.stderr
        assert L.iceberg_snapshot_counts(catalog, L.ICEBERG_TABLES) == {
            t: 1 for t in L.ICEBERG_TABLES
        }
        L.assert_lake_exact(catalog)
    finally:
        holder.close()
        conn.close()


# ── 7. table properties ──────────────────────────────────────────────────────


def test_table_properties(docker_stack, caplog):
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    catalog = L.get_catalog()
    for t in L.ICEBERG_TABLES:
        props = catalog.load_table(f"{L.TEST_NAMESPACE}.{t}").properties
        assert props.get(PROPERTY) == "true", t

    # Keep two previous versions on one table: after four more commits at
    # most three metadata.json objects remain. Both the REST server and
    # pyiceberg delete the old versions; that logs no warning.
    name = "agg_site_info"
    table = catalog.load_table(f"{L.TEST_NAMESPACE}.{name}")
    with table.transaction() as txn:
        txn.set_properties({"write.metadata.previous-versions-max": "2"})
    with caplog.at_level("WARNING"):
        for i in range(4):
            with table.transaction() as txn:
                txn.set_properties({"mosaic.test.commit": str(i)})
    assert not [r for r in caplog.records if "Failed to delete" in r.message]
    jsons = [
        uri
        for uri in L.location_listing(catalog, name, "metadata/")
        if uri.endswith(".metadata.json")
    ]
    assert 1 <= len(jsons) <= 3, jsons

    # A table without the property gets it back.
    with table.transaction() as txn:
        txn.remove_properties(PROPERTY)
    assert (
        PROPERTY
        not in catalog.load_table(f"{L.TEST_NAMESPACE}.{name}").properties
    )
    proc = L.run_maintain(dry_run=True)
    assert _line(proc, name).startswith(f"{name}: properties would be set;")
    proc = L.run_maintain()
    assert _line(proc, name).startswith(f"{name}: properties set;")
    for t in L.ICEBERG_TABLES:
        if t != name:
            assert _line(proc, t) == f"{t}: nothing to do"
    table = catalog.load_table(f"{L.TEST_NAMESPACE}.{name}")
    assert table.properties.get(PROPERTY) == "true"
    proc = L.run_maintain()
    assert _line(proc, name) == f"{name}: nothing to do"

    # A per-sample table without the codec properties (a lake from before
    # they were introduced) gets them back.
    codec = {
        "write.parquet.compression-codec": "brotli",
        "write.parquet.compression-level": "5",
    }
    name = "sample_site_info"
    table = catalog.load_table(f"{L.TEST_NAMESPACE}.{name}")
    assert {k: table.properties.get(k) for k in codec} == codec
    with table.transaction() as txn:
        txn.remove_properties(*codec)
    proc = L.run_maintain()
    assert _line(proc, name).startswith(f"{name}: properties set;")
    table = catalog.load_table(f"{L.TEST_NAMESPACE}.{name}")
    assert {k: table.properties.get(k) for k in codec} == codec
    proc = L.run_maintain()
    assert _line(proc, name) == f"{name}: nothing to do"
