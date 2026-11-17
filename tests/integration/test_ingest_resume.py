# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Crash + resume correctness for ingestion (a shard checkpointed before
the batch upload).

A mid-ingest crash leaves some shards checkpointed (shard_completion rows, which
the main process commits as each shard's result arrives) but with no per-sample
Iceberg data, because the batch's Iceberg upload happens once, after every shard
returns. The historical bug skipped those committed shards on resume, leaving
PostgreSQL complete but Iceberg missing whole spans. The fix reprocesses
committed shards Iceberg-only on resume.

Oracle: a clean run is the truth. After a crash+resume the per-sample Iceberg row
counts and the decoded joint-called VCF must equal a clean single-pass run, and the
PG aggregates must be bit-exact (compared by genomic coordinate + allele string).
"""

import os

import pytest

import jc_testlib as L

pytestmark = [pytest.mark.integration, pytest.mark.restart]

# A 5 Mbp window cut into 1 Mbp shards => ~5 shards over 2 samples, so a mid-run
# crash leaves several shards committed and several not.
REGION = "chr21:5000000-10000000"
SHARD_SIZE = 1_000_000
# Crash on the first shard whose 0-based start is >= this (a middle shard), so a
# few earlier shards commit to PG before the run aborts.
FAIL_THRESHOLD = "chr21:7000000"


def _subset_pair(tmp_path):
    a = os.path.join(tmp_path, "a.g.vcf.gz")
    b = os.path.join(tmp_path, "b.g.vcf.gz")
    L.subset_gvcf(L.GVCFS[0], REGION, a)
    L.subset_gvcf(L.GVCFS[1], REGION, b)
    return [a, b]


def _full_clean_run(gvcfs, out_vcf, pos_batch_size=None, files=None):
    """init -> ingest -> aggregate -> joint_call; return (pg, iceberg,
    vcfhash). ``files`` (a dict) gets the ingest tables' data files with
    their ETags (L.iceberg_file_etags) under "etags"."""
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(
        gvcfs,
        region=REGION,
        shard_size=SHARD_SIZE,
        threads=2,
        pos_batch_size=pos_batch_size,
    )
    if files is not None:
        files["etags"] = L.iceberg_file_etags(L.get_catalog())
    L.run_aggregate(fan_in=100, threads=2)
    L.run_joint_call(out_vcf)
    conn = L.get_pg_conn()
    try:
        pg = L.snapshot_pg(conn)
    finally:
        conn.close()
    ice = L.count_iceberg(L.get_catalog())
    return pg, ice, L.snapshot_vcf(out_vcf)


# pos_batch_size=None is the default (window >> shard, flush only at shard end);
# 25_000 forces ~40 in-loop window flushes per 1 Mbp shard, so a mid-shard crash
# lands after some windows' parquet has been written locally but not uploaded —
# resume must reprocess and regenerate identical files.
@pytest.mark.parametrize("pos_batch_size", [None, 25_000])
def test_ingest_resume_hook(clean_state, tmp_path, pos_batch_size):
    """Hook-injected crash after a shard's PG commit, then resume → bit-exact."""
    gvcfs = _subset_pair(str(tmp_path))

    # 1. Baseline clean run.
    base_files = {}
    base_pg, base_ice, base_vcf = _full_clean_run(
        gvcfs,
        os.path.join(str(tmp_path), "baseline.vcf.gz"),
        pos_batch_size=pos_batch_size,
        files=base_files,
    )
    assert base_ice["sample_site_info"] > 0

    # 2. Fresh start, crash mid-ingest.
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    proc = L.run_ingest(
        gvcfs,
        region=REGION,
        shard_size=SHARD_SIZE,
        threads=2,
        pos_batch_size=pos_batch_size,
        extra_env={"MOSAIC_TEST_FAIL_SHARD": FAIL_THRESHOLD},
        check=False,
    )
    assert (
        proc.returncode != 0
    ), "ingest should have crashed via MOSAIC_TEST_FAIL_SHARD"

    # The hazard: PG progress recorded, but no per-sample Iceberg data uploaded.
    conn = L.get_pg_conn()
    try:
        assert (
            L.shard_completion_count(conn) > 0
        ), "no shards committed before crash"
        statuses = dict(L.batch_statuses(conn))
        assert statuses.get(1) in ("in_progress", "failed")
    finally:
        conn.close()
    crash_ice = L.count_iceberg(L.get_catalog())
    assert (
        crash_ice["sample_site_info"] < base_ice["sample_site_info"]
    ), "Iceberg should be short of the committed PG shards after a crash"

    # 3. Resume (same args, no fault) — must reprocess committed shards Iceberg-only.
    L.run_ingest(
        gvcfs,
        region=REGION,
        shard_size=SHARD_SIZE,
        threads=2,
        pos_batch_size=pos_batch_size,
    )
    # The resumed batch produced the same data files, byte for byte: the
    # same keys (batch_id 1 again after the reset), and for the
    # whole-sample per-sample tables the same packing.
    assert L.iceberg_file_etags(L.get_catalog()) == base_files["etags"]
    L.run_aggregate(fan_in=100, threads=2)
    out = os.path.join(str(tmp_path), "resumed.vcf.gz")
    L.run_joint_call(out)

    # 4. Bit-exact to the clean baseline + Iceberg fully repopulated + cleaned up.
    conn = L.get_pg_conn()
    try:
        L.assert_pg_bitexact(L.snapshot_pg(conn), base_pg)
        assert (
            L.shard_completion_count(conn) == 0
        ), "shard_completion not cleaned up"
        assert dict(L.batch_statuses(conn)).get(1) == "complete"
    finally:
        conn.close()
    assert L.count_iceberg(L.get_catalog()) == base_ice
    assert L.snapshot_vcf(out) == base_vcf


def test_ingest_resume_sigkill(clean_state, tmp_path):
    """Black-box SIGKILL of the ingest process group, then resume → bit-exact.

    Defense-in-depth alongside the hook test: an un-ignorable hard kill (Sharder
    workers ignore SIGINT) at an uncontrolled point in the run.
    """
    import signal

    gvcfs = _subset_pair(str(tmp_path))

    base_files = {}
    base_pg, base_ice, base_vcf = _full_clean_run(
        gvcfs, os.path.join(str(tmp_path), "baseline.vcf.gz"), files=base_files
    )

    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    proc = L.popen_pipeline(
        "ingest",
        [
            *gvcfs,
            "--region",
            REGION,
            "--shard-size",
            str(SHARD_SIZE),
            "--threads",
            "2",
        ],
    )

    def _at_least_one_shard():
        if proc.poll() is not None:  # process already exited
            return True
        c = L.get_pg_conn()
        try:
            return L.shard_completion_count(c) >= 1
        finally:
            c.close()

    assert L.poll_until(
        _at_least_one_shard, timeout=120, interval=0.05
    ), "ingest never committed a shard"
    # Hard-kill the whole process group (workers included).
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
    proc.wait(timeout=30)

    conn = L.get_pg_conn()
    try:
        assert (
            dict(L.batch_statuses(conn)).get(1) != "complete"
        ), "killed too late — batch already completed; cannot test resume"
    finally:
        conn.close()

    # Resume to completion: the same data files.
    L.run_ingest(gvcfs, region=REGION, shard_size=SHARD_SIZE, threads=2)
    assert L.iceberg_file_etags(L.get_catalog()) == base_files["etags"]
    L.run_aggregate(fan_in=100, threads=2)
    out = os.path.join(str(tmp_path), "resumed.vcf.gz")
    L.run_joint_call(out)

    conn = L.get_pg_conn()
    try:
        L.assert_pg_bitexact(L.snapshot_pg(conn), base_pg)
        assert dict(L.batch_statuses(conn)).get(1) == "complete"
    finally:
        conn.close()
    assert L.count_iceberg(L.get_catalog()) == base_ice
    assert L.snapshot_vcf(out) == base_vcf


def test_batch_partials_no_duplicate_rows(clean_state, tmp_path):
    """The per-batch partials must have NO duplicate rows per key after a
    crash+resume ingest (which reprocesses committed shards to regenerate their
    Iceberg parquet).

    This is the invariant that lets `mosaic aggregate` drop its tier-0 dedup
    `seen` set: each (batch, key) contribution is written -- and therefore
    scanned by the reduce -- exactly once, even across a resume, because each
    data file registers once and ingest writes each position once. A regression
    here (a shard's rows regenerated without replacing the old file, say) would
    make the dedup-free aggregate double-count. Asserting the invariant directly
    catches that even though the bit-exact resume tests above would too.
    """
    gvcfs = _subset_pair(str(tmp_path))
    L.run_init(L.GVCFS[0], "chr21")
    # Crash mid-ingest (some shards committed), then resume to completion.
    proc = L.run_ingest(
        gvcfs,
        region=REGION,
        shard_size=SHARD_SIZE,
        threads=2,
        extra_env={"MOSAIC_TEST_FAIL_SHARD": FAIL_THRESHOLD},
        check=False,
    )
    assert proc.returncode != 0, "ingest should have crashed"
    L.run_ingest(gvcfs, region=REGION, shard_size=SHARD_SIZE, threads=2)
    conn = L.get_pg_conn()
    try:
        assert dict(L.batch_statuses(conn)).get(1) == "complete"
    finally:
        conn.close()

    catalog = L.get_catalog()
    keyed = {
        "batch_ref_blocks": ["batch_id", "chromosome_index", "position"],
        "batch_site_info": ["batch_id", "chromosome_index", "position"],
        "batch_allele_counts": [
            "batch_id",
            "chromosome_index",
            "position",
            "allele",
        ],
    }
    for tbl, keys in keyed.items():
        arrow = (
            catalog.load_table(f"{L.TEST_NAMESPACE}.{tbl}").scan().to_arrow()
        )
        n = arrow.num_rows
        assert n > 0, f"{tbl} unexpectedly empty"
        distinct = (
            arrow.group_by(keys).aggregate([(keys[0], "count")]).num_rows
        )
        assert n == distinct, (
            f"{tbl}: {n - distinct} duplicate key rows across {n} -- resume "
            f"regenerated overlapping rows; dedup-free aggregate would "
            f"double-count"
        )
