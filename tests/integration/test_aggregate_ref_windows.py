# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The ref_blocks window units and the dense root they write.

`mosaic aggregate` reduces ref_blocks as one unit per (chromosome, position
window of --ref-shard-size bases). A unit sums every covering batch's dense
batch_ref_blocks rows for its window and uploads the dense root (every covered
position) for tier_ref_blocks_dense and the sparse root (covered variant sites)
for tier_ref_blocks at tier 1, group_id = window; after the windows barrier the
main process registers every window's files (one add_files per table) and then
commits their tier_completion checkpoints (tier 1, group = window) in one
transaction.

Fixture: the 3 chr21 batches of test_aggregate_resume over a chr20 + chr21
init. At --ref-shard-size 12 Mbp chr21 (46.7 Mbp) has 4 windows: windows 0, 1
and 2 hold the 10-10.5M, 20-20.5M and 25-25.5M batches, window 3 is empty;
chr20 is covered by no batch and gets no window units.
"""

import os
import re

import pyarrow.compute as pc
import pytest
from pyiceberg.expressions import EqualTo

import jc_testlib as L

pytestmark = [pytest.mark.integration, pytest.mark.restart]

REGIONS = [
    "chr21:10000000-10500000",
    "chr21:20000000-20500000",
    "chr21:25000000-25500000",
]
REF_SHARD = 12_000_000
N_WINDOWS = 4  # ceil(46,709,983 / 12,000,000)
C_EMPTY, C_DATA = 0, 1  # chr20 (no batch), chr21


@pytest.fixture(scope="module")
def staged(docker_stack, tmp_path_factory):
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr20,chr21")
    work = tmp_path_factory.mktemp("ref_windows_stage")
    for i, region in enumerate(REGIONS):
        gvcfs = []
        for j, src in enumerate(L.GVCFS):
            dst = os.path.join(str(work), f"b{i}_s{j}.g.vcf.gz")
            L.subset_gvcf(src, region, dst)
            gvcfs.append(dst)
        L.run_ingest(gvcfs, region=region, threads=2)
    conn = L.get_pg_conn()
    try:
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"] * 3
    finally:
        conn.close()
    return REGIONS


@pytest.fixture(scope="module")
def baseline(staged):
    """A clean run at REF_SHARD: its PG snapshot and its dense root (without
    the agg_run_id column, so other runs' roots compare equal)."""
    conn = L.get_pg_conn()
    try:
        L.run_aggregate(
            fan_in=100, threads=2, rebuild=True, ref_shard_size=REF_SHARD
        )
        run = L.latest_agg_run_id(conn)
        return {
            "pg": L.snapshot_pg(conn),
            "dense": _dense(run),
        }
    finally:
        conn.close()


def _dense(agg_run_id):
    return L.dense_root(L.get_catalog(), agg_run_id).drop_columns(
        ["agg_run_id"]
    )


def _units(conn, agg_run_id, c_idx):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tier, group_id FROM tier_completion WHERE agg_run_id=%s "
            "AND metric='ref_blocks' AND chromosome_index=%s AND tier >= 0",
            (agg_run_id, c_idx),
        )
        return set(cur.fetchall())


def test_dense_root_is_every_covered_position(staged):
    """The dense root holds exactly the distinct covered positions of the
    batch_ref_blocks partials, with their summed values; the sparse root is
    the dense root at the variant sites, and equals PG ref_blocks."""
    conn = L.get_pg_conn()
    catalog = L.get_catalog()
    try:
        proc = L.run_aggregate(
            fan_in=100, threads=2, rebuild=True, ref_shard_size=REF_SHARD
        )
        assert f"ref_windows={N_WINDOWS}" in proc.stderr
        run = L.latest_agg_run_id(conn)
        assert _units(conn, run, C_DATA) == {(1, w) for w in range(N_WINDOWS)}
        assert _units(conn, run, C_EMPTY) == set()

        dense = L.dense_root(catalog, run)
        partials = (
            catalog.load_table(f"{L.TEST_NAMESPACE}.batch_ref_blocks")
            .scan()
            .to_arrow()
        )
        truth = (
            partials.group_by(["chromosome_index", "position"])
            .aggregate([("dp", "sum"), ("sample_counts", "sum")])
            .sort_by(
                [("chromosome_index", "ascending"), ("position", "ascending")]
            )
        )
        # Row count == distinct covered positions (one chromosome here).
        assert dense.num_rows == truth.num_rows
        assert dense.num_rows == len(pc.unique(partials["position"]))
        assert dense.num_rows > 1_000_000
        assert dense["position"].to_pylist() == truth["position"].to_pylist()
        assert set(dense["chromosome_index"].to_pylist()) == {C_DATA}
        for col in ("dp", "sample_counts"):
            want = pc.fill_null(truth[f"{col}_sum"], 0).to_pylist()
            assert dense[col].to_pylist() == want, col
        windows = pc.divide(pc.subtract(dense["position"], 1), REF_SHARD)
        assert dense["window_id"].to_pylist() == windows.to_pylist()
        # One data file per non-empty window: deterministic keys, no dupes.
        dense_table = catalog.load_table(
            f"{L.TEST_NAMESPACE}.tier_ref_blocks_dense"
        )
        files = list(
            dense_table.scan(
                row_filter=EqualTo("agg_run_id", run)
            ).plan_files()
        )
        assert len(files) == 3

        # Sparse root (tier 1, group = window) == PG ref_blocks == the dense
        # root at the PG site_info positions.
        sparse = (
            catalog.load_table(f"{L.TEST_NAMESPACE}.tier_ref_blocks")
            .scan(row_filter=EqualTo("agg_run_id", run))
            .to_arrow()
        )
        assert set(sparse["tier"].to_pylist()) == {1}
        pg = L.snapshot_pg(conn)
        sparse_rows = {
            (c, p): (dp, sc)
            for c, p, dp, sc in zip(
                sparse["chromosome_index"].to_pylist(),
                sparse["position"].to_pylist(),
                sparse["dp"].to_pylist(),
                sparse["sample_counts"].to_pylist(),
            )
        }
        assert sparse_rows == pg["ref_blocks"]
        sites = set(pg["site_info_int"])
        at_sites = {
            (c, p): (dp, sc)
            for c, p, dp, sc in zip(
                dense["chromosome_index"].to_pylist(),
                dense["position"].to_pylist(),
                dense["dp"].to_pylist(),
                dense["sample_counts"].to_pylist(),
            )
            if (c, p) in sites
        }
        assert at_sites == pg["ref_blocks"]
        assert len(at_sites) > 1000
    finally:
        conn.close()


@pytest.mark.parametrize("phase", ["pre_register", "pre_ckpt", "post_ckpt"])
def test_window_unit_crash_resume(baseline, phase):
    """Crash at the ref_blocks windows barrier, named by chr21's window 0:
    before the barrier registers (nothing registered), between registration
    and checkpoint, or after the checkpoint transaction. The resume re-runs
    the windows (their files overwritten at the same keys and registered
    once) or skips them (post_ckpt), and ends bit-exact with a clean run of
    the same agg_run_id, with the same dense root and the same staging and
    mirror files (URI, ETag)."""
    conn = L.get_pg_conn()
    catalog = L.get_catalog()
    try:
        L.reset_aggregate_runs(conn, catalog)
        L.run_aggregate(
            fan_in=100, threads=2, rebuild=True, ref_shard_size=REF_SHARD
        )
        clean_files = L.aggregate_file_etags(catalog)
        L.reset_aggregate_runs(conn, catalog)
        proc = L.run_aggregate(
            fan_in=100,
            threads=2,
            rebuild=True,
            ref_shard_size=REF_SHARD,
            extra_env={"MOSAIC_TEST_FAIL_UNIT": f"ref_blocks:*:1:0:{phase}"},
            check=False,
        )
        assert proc.returncode != 0, "aggregate should have crashed"
        assert "MOSAIC_TEST_FAIL_UNIT" in proc.stderr
        crashed = L.latest_agg_run_id(conn)
        assert dict(L.agg_statuses(conn))[crashed] == "failed"
        assert L.tier_completion_has(conn, "ref_blocks", 1, 0) == (
            phase == "post_ckpt"
        )
        # Window 0's dense root is registered from pre_ckpt on (the barrier
        # registers before it checkpoints), not at pre_register.
        base_w0 = baseline["dense"].filter(
            pc.equal(baseline["dense"]["window_id"], 0)
        )
        dense = _dense(crashed)
        crashed_w0 = dense.filter(pc.equal(dense["window_id"], 0))
        if phase == "pre_register":
            assert dense.num_rows == 0
        else:
            assert crashed_w0.equals(base_w0)
        # The ref_blocks finalize never ran.
        assert L.finalize_markers(conn, "ref_blocks", crashed) == set()

        proc = L.run_aggregate(fan_in=100, threads=2, ref_shard_size=REF_SHARD)
        assert f"agg_run_id={crashed} resume" in proc.stderr
        m = re.search(
            r"ref_blocks windows: (\d+) unit\(s\) over 1 chrom "
            r"\((\d+) done, (\d+) skipped",
            proc.stderr,
        )
        assert m, proc.stderr[-2000:]
        assert int(m.group(1)) == N_WINDOWS
        if phase == "post_ckpt":
            assert int(m.group(3)) == N_WINDOWS  # the whole barrier
        else:
            assert int(m.group(2)) == N_WINDOWS  # every window re-ran
        L.assert_pg_bitexact(L.snapshot_pg(conn), baseline["pg"])
        assert _dense(crashed).equals(baseline["dense"])
        assert L.aggregate_file_etags(catalog) == clean_files
    finally:
        conn.close()


def test_resume_needs_the_same_ref_shard_size(baseline):
    """Window ids only mean something for one window size, so a re-run with a
    different --ref-shard-size starts a fresh run instead of resuming the
    failed one (whose checkpointed window ids would cover other ranges)."""
    conn = L.get_pg_conn()
    try:
        L.clear_pg_aggregates(conn)
        proc = L.run_aggregate(
            fan_in=100,
            threads=2,
            rebuild=True,
            ref_shard_size=REF_SHARD,
            extra_env={"MOSAIC_TEST_FAIL_UNIT": "ref_blocks:*:1:0:post_ckpt"},
            check=False,
        )
        assert proc.returncode != 0
        crashed = L.latest_agg_run_id(conn)

        proc = L.run_aggregate(fan_in=100, threads=2, ref_shard_size=5_000_000)
        fresh = L.latest_agg_run_id(conn)
        assert fresh > crashed
        assert f"agg_run_id={fresh} fresh" in proc.stderr
        assert "ref_windows=10" in proc.stderr
        assert dict(L.agg_statuses(conn))[crashed] == "failed"
        L.assert_pg_bitexact(L.snapshot_pg(conn), baseline["pg"])
        # Same dense root, cut into other windows.
        assert (
            _dense(fresh)
            .drop_columns(["window_id"])
            .equals(baseline["dense"].drop_columns(["window_id"]))
        )
    finally:
        conn.close()
