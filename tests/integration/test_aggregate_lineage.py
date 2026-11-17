# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Aggregation run lineage, the "up to date" exit, and output gating.

* Every run records its lineage in aggregation_progress: a full run has
  parent_run_id NULL and delta_batch_ids = batch_ids, staging_pruned false.
* The cohort_generation view exposes the latest run.
* Without --rebuild, a batch set whose latest run is complete is up to date:
  `mosaic aggregate` says so and exits 0 without creating a run.
* joint_call.py refuses (exit 2) to call from aggregates whose latest run is not
  complete or does not cover every completed ingest batch, unless
  --allow-stale is given.

Fixture: two small chr21 batches (two 300 kb windows of the two test samples).
"""

import os
import subprocess

import pytest

import jc_testlib as L

pytestmark = pytest.mark.integration

REGIONS = ["chr21:10000000-10300000", "chr21:20000000-20300000"]


@pytest.fixture(scope="module")
def staged(docker_stack, tmp_path_factory):
    work = tmp_path_factory.mktemp("lineage_stage")
    L.stage_batches(REGIONS, str(work))
    return REGIONS


def _run_row(conn, agg_run_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM aggregation_progress WHERE agg_run_id=%s",
            (agg_run_id,),
        )
        row = cur.fetchone()
        return dict(zip([d[0] for d in cur.description], row))


def _count(conn, table):
    with conn.cursor() as cur:
        cur.execute(f"SELECT count(*) FROM {table}")
        return cur.fetchone()[0]


def test_run_lineage_and_cohort_generation(staged):
    conn = L.get_pg_conn()
    try:
        if L.latest_agg_run_id(conn) is None:
            assert L.cohort_generation(conn) is None
        L.run_aggregate(fan_in=100, threads=2, rebuild=True)
        run = L.latest_agg_run_id(conn)

        gen = L.cohort_generation(conn)
        assert gen["agg_run_id"] == run
        assert gen["batch_ids"] == [1, 2]
        assert gen["parent_run_id"] is None
        assert gen["status"] == "complete"
        assert gen["updated_at"] is not None

        row = _run_row(conn, run)
        assert row["parent_run_id"] is None
        assert row["delta_batch_ids"] == row["batch_ids"] == [1, 2]
        assert row["staging_pruned"] is False
        assert row["fan_in"] == 100
        assert row["ref_shard_size"] == 20_000_000
    finally:
        conn.close()


def test_plain_rerun_is_up_to_date(staged):
    """A second plain run (any --fan-in, any --ref-shard-size) creates no run
    and touches nothing; --rebuild still starts a new one."""
    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=100, threads=2, rebuild=True)
        run = L.latest_agg_run_id(conn)
        runs = _count(conn, "aggregation_progress")
        units = _count(conn, "tier_completion")
        layout = L.partition_layout(conn)
        snap = L.snapshot_pg(conn)

        for fan_in, ref_shard_size in ((100, None), (2, 5_000_000)):
            proc = L.run_aggregate(
                fan_in=fan_in, threads=2, ref_shard_size=ref_shard_size
            )
            assert f"up to date (agg_run_id={run})" in proc.stderr
            assert "Aggregation complete" not in proc.stdout
            assert _count(conn, "aggregation_progress") == runs
            assert _count(conn, "tier_completion") == units
            assert L.partition_layout(conn) == layout
            assert L.cohort_generation(conn)["agg_run_id"] == run
        L.assert_pg_bitexact(L.snapshot_pg(conn), snap)

        proc = L.run_aggregate(fan_in=100, threads=2, rebuild=True)
        assert L.latest_agg_run_id(conn) > run
        assert "up to date" not in proc.stderr
    finally:
        conn.close()


def _vcf_records(path):
    proc = subprocess.run(
        ["bcftools", "view", "-H", path],
        check=True,
        capture_output=True,
        text=True,
    )
    return len(proc.stdout.splitlines())


def test_joint_call_refuses_stale_aggregates(staged, tmp_path):
    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=100, threads=2, rebuild=True)
        run = L.latest_agg_run_id(conn)
        out = str(tmp_path / "joint.vcf.gz")

        # The latest run failed (some chromosomes may hold its generation,
        # others the previous one): refuse, before writing any output.
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE aggregation_progress SET status='failed' "
                "WHERE agg_run_id=%s",
                (run,),
            )
        proc = L.run_joint_call(out, check=False)
        assert proc.returncode == 2, proc.stderr[-2000:]
        assert f"(agg_run_id={run}) is failed" in proc.stderr
        assert "--allow-stale" in proc.stderr
        assert not os.path.exists(out)

        # --allow-stale overrides, with a warning.
        proc = L.run_joint_call(out, allow_stale=True)
        assert "Warning (--allow-stale)" in proc.stderr
        assert _vcf_records(out) > 100
        os.unlink(out)

        # The latest run is complete but misses a completed batch.
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE aggregation_progress SET status='complete', "
                "batch_ids='{1}' WHERE agg_run_id=%s",
                (run,),
            )
        proc = L.run_joint_call(out, check=False)
        assert proc.returncode == 2
        assert "not aggregated: [2]" in proc.stderr

        # No aggregation run at all.
        with conn.cursor() as cur:
            cur.execute("DELETE FROM aggregation_progress")
        assert L.cohort_generation(conn) is None
        proc = L.run_joint_call(out, check=False)
        assert proc.returncode == 2
        assert "no aggregation run exists" in proc.stderr
        assert not os.path.exists(out)
    finally:
        conn.close()
