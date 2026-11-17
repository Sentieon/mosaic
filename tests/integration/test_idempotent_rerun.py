# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Idempotency: re-running a completed step changes nothing.

  * Re-ingesting the same samples+region detects the 'complete' batch and skips
    (no double-write).
  * Re-aggregating (--rebuild) reproduces bit-exact PG aggregates, and a plain
    re-run of an aggregated batch set is "up to date" (no new run).

Both are the steady-state companions to the crash/resume tests: not only does a
killed run recover, a redundant run is harmless.
"""

import os

import pytest

import jc_testlib as L

pytestmark = [pytest.mark.integration, pytest.mark.restart]

REGION = "chr21:10000000-10500000"


def _subset_pair(tmp_path):
    gvcfs = []
    for j, src in enumerate(L.GVCFS):
        dst = os.path.join(str(tmp_path), f"s{j}.g.vcf.gz")
        L.subset_gvcf(src, REGION, dst)
        gvcfs.append(dst)
    return gvcfs


def test_reingest_same_batch_is_noop(clean_state, tmp_path):
    gvcfs = _subset_pair(tmp_path)
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(gvcfs, region=REGION, threads=2)

    conn = L.get_pg_conn()
    try:
        snap1 = L.snapshot_pg(conn)
    finally:
        conn.close()
    ice1 = L.count_iceberg(L.get_catalog())

    # Second identical ingest must detect the completed batch and skip.
    proc = L.run_ingest(gvcfs, region=REGION, threads=2)
    assert "already completed" in proc.stderr.lower()

    conn = L.get_pg_conn()
    try:
        L.assert_pg_bitexact(L.snapshot_pg(conn), snap1)
        assert dict(L.batch_statuses(conn)) == {1: "complete"}
    finally:
        conn.close()
    assert L.count_iceberg(L.get_catalog()) == ice1


def test_reaggregate_is_idempotent(clean_state, tmp_path):
    gvcfs = _subset_pair(tmp_path)
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(gvcfs, region=REGION, threads=2)

    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=100, threads=2)
        snap1 = L.snapshot_pg(conn)
        vcf1 = os.path.join(str(tmp_path), "first.vcf.gz")
        L.run_joint_call(vcf1)
        # A second aggregate (--rebuild: a new run) recomputes from the
        # immutable Iceberg partials; the result must be bit-exact, and so
        # must the joint-called VCF (joint_call.py synthesizes its allele
        # ids per window from the rebuilt catalog).
        L.run_aggregate(fan_in=100, threads=2, rebuild=True)
        L.assert_pg_bitexact(L.snapshot_pg(conn), snap1)
        vcf2 = os.path.join(str(tmp_path), "second.vcf.gz")
        L.run_joint_call(vcf2)
        assert L.snapshot_vcf(vcf2) == L.snapshot_vcf(vcf1)
        # A plain re-run of the same batch set is up to date: no new run.
        runs = L.agg_statuses(conn)
        proc = L.run_aggregate(fan_in=100, threads=2)
        assert f"up to date (agg_run_id={runs[-1][0]})" in proc.stderr
        assert L.agg_statuses(conn) == runs
        L.assert_pg_bitexact(L.snapshot_pg(conn), snap1)
    finally:
        conn.close()
