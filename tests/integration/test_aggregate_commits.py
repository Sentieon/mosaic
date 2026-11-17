# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The aggregate's Iceberg commits per barrier (the batched aggregate
commits).

The reduce and finalize units only upload; the main process makes every
commit: one add_files per staging table per barrier (a tier, a fold, the
ref_blocks windows), one replace (delete-all + add_files, one transaction) per
agg_* mirror per finalize barrier, and the completion prune's deletes. So at
--threads 4 no commit can conflict with another of the run's, and each table
gains exactly the planned number of snapshots, whatever the thread count.

Fixture: four 2-sample batches over four disjoint chr21 regions, init chr20 +
chr21 (chr20 covered by no batch). A full run over b1-b3 at --fan-in 2 (tier
0->1: two groups, tier 1->2: one) with 5 Mbp ref windows (10 windows on
chr21), then an incremental run adding b4 (one delta tier and a fold), at
--threads 4 and at --threads 1: the same snapshot counts, no "Iceberg commit
conflict" line, and the same PG aggregates.
"""

import os

import pytest

import jc_testlib as L

pytestmark = [pytest.mark.integration]

REGIONS = [
    "chr21:10000000-10500000",
    "chr21:20000000-20500000",
    "chr21:25000000-25500000",
    "chr21:15000000-15500000",
]
FAN_IN = 2
REF_SHARD = 5_000_000

# Snapshots each table gains (reset_aggregate_runs leaves every table with a
# current snapshot and no live file, so a delete that matches nothing makes
# none).
# Full run over b1-b3: two tier barriers + the prune of its own tier 1 on the
# tier-tree tables; one windows barrier on the ref_blocks tables (the prune
# matches nothing there); one add on each (empty) mirror.
FULL = {
    "tier_site_info": 3,
    "tier_allele_counts": 3,
    "tier_ref_blocks": 1,
    "tier_ref_blocks_dense": 1,
    "agg_site_info": 1,
    "agg_allele_counts": 1,
    "agg_ref_blocks": 1,
}
# Incremental +b4: one delta tier barrier + one fold barrier + one prune
# delete (the parent's rows and its own delta root); one windows barrier +
# one prune delete; each mirror a delete-all and an add in one commit.
INCREMENTAL = {
    "tier_site_info": 3,
    "tier_allele_counts": 3,
    "tier_ref_blocks": 2,
    "tier_ref_blocks_dense": 2,
    "agg_site_info": 2,
    "agg_allele_counts": 2,
    "agg_ref_blocks": 2,
}


@pytest.fixture(scope="module")
def staged(docker_stack, tmp_path_factory):
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr20,chr21")
    work = tmp_path_factory.mktemp("agg_commits_stage")
    for i, region in enumerate(REGIONS):
        gvcfs = []
        for j, src in enumerate(L.GVCFS):
            dst = os.path.join(str(work), f"b{i}_s{j}.g.vcf.gz")
            L.subset_gvcf(src, region, dst)
            gvcfs.append(dst)
        L.run_ingest(gvcfs, region=region, threads=2)
    conn = L.get_pg_conn()
    try:
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"] * 4
    finally:
        conn.close()
    return REGIONS


def _delta(before, after):
    return {t: after[t] - before[t] for t in before}


def _lineage(threads):
    """Full over b1-b3, then +b4, at ``threads``; returns each run's PG
    snapshot, snapshot deltas and stderr."""
    conn = L.get_pg_conn()
    catalog = L.get_catalog()
    out = {}
    try:
        L.reset_aggregate_runs(conn, catalog)
        for name, batches in (("full", [1, 2, 3]), ("incr", [1, 2, 3, 4])):
            L.set_completed_batches(conn, batches)
            conn.commit()
            before = L.iceberg_snapshot_counts(catalog, L.AGGREGATE_TABLES)
            proc = L.run_aggregate(
                fan_in=FAN_IN, threads=threads, ref_shard_size=REF_SHARD
            )
            after = L.iceberg_snapshot_counts(catalog, L.AGGREGATE_TABLES)
            out[name] = {
                "pg": L.snapshot_pg(conn),
                "snapshots": _delta(before, after),
                "stderr": proc.stderr,
            }
        L.set_completed_batches(conn, [1, 2, 3, 4])
        conn.commit()
    finally:
        conn.close()
    return out


@pytest.fixture(scope="module")
def one_thread(staged):
    return _lineage(1)


@pytest.fixture(scope="module")
def four_threads(staged, one_thread):
    return _lineage(4)


@pytest.mark.parametrize("threads", [1, 4])
def test_one_commit_per_barrier(one_thread, four_threads, threads):
    runs = one_thread if threads == 1 else four_threads
    full, incr = runs["full"], runs["incr"]
    assert f"threads={threads} " in full["stderr"]
    assert " full " in full["stderr"]
    assert "parent=1 delta=1" in incr["stderr"]
    for run in (full, incr):
        assert "Iceberg commit conflict" not in run["stderr"]
    assert full["snapshots"] == FULL
    assert incr["snapshots"] == INCREMENTAL
    # The barriers' own log lines agree: tiers 0->1 and 1->2 on each
    # tier-tree metric and the windows barrier each registered in one
    # commit (two for the windows: dense and sparse root).
    assert full["stderr"].count("in 1 commit(s)") == 4
    assert full["stderr"].count("in 2 commit(s)") == 1
    assert incr["stderr"].count("in 1 commit(s)") == 4  # tier + fold, x2
    assert incr["stderr"].count("in 2 commit(s)") == 1
    for metric in ("site_info", "allele_counts", "ref_blocks"):
        assert f"agg_{metric} replaced by" in incr["stderr"]


def test_threads_do_not_change_the_result(one_thread, four_threads):
    for run in ("full", "incr"):
        L.compare_pg_snapshots(four_threads[run]["pg"], one_thread[run]["pg"])
        L.assert_pg_integer_equal(
            four_threads[run]["pg"], one_thread[run]["pg"]
        )
