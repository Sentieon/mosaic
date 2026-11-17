# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The aggregate tier tree is planned per chromosome.

A chromosome's tree covers only the completed batches whose ingest region covers
it (``ingestion_progress.region`` NULL = whole genome, else its chromosome part).
Fixture, over the two-chromosome HG00099 slice (init chr21 + chr22):

  batch 1: chr21 window A (+ HG00100), --region chr21:...  -> covers chr21 only
  batch 2: chr21 window B, --region chr21:...              -> covers chr21 only
  batch 3: chr21 window A + a chr22 window, no --region (whole genome)

At --fan-in 2, chr21 plans [[1,2],[3]] -> [[0,1]] (two tiers) while chr22 plans
[[3]] (one tier). The old global plan gave chr22 the same two tiers as every
chromosome; here chr22 must have exactly the whole-genome batch, no tier-2
units, and both chromosomes must finalize -- to the values the pure-Python
oracle computes from the partials. (Only HG00099 has chr22 data, and no sample's
own ref blocks cover its variant sites, so chr22's sparse ref_blocks are empty
by construction; batch 1's second sample, HG00100, gives chr21 ref_blocks rows.)

An incremental run plans per chromosome too: after a full run over batches 1
and 3, batch 2 (chr21 only) leaves chr22 with an empty delta, whose
partitions are rebuilt from the parent's roots alone.
"""

import os

import pyarrow.compute as pc
import pytest

import jc_testlib as L

pytestmark = pytest.mark.integration

SLICE = os.path.join(
    L.REPO_ROOT,
    "test_data/downloads/HG00099/"
    "HG00099.chr21_1-15M.chr22_10.5-20M.final.hard-filtered.gvcf.gz",
)
HG00100 = os.path.join(
    L.REPO_ROOT,
    "test_data/downloads/HG00100/HG00100.chr21.final.hard-filtered.gvcf.gz",
)
CHR21_A = "chr21:10000000-10300000"
CHR21_B = "chr21:14000000-14300000"
CHR22 = "chr22:15000000-15300000"


@pytest.fixture(scope="module")
def staged(docker_stack, tmp_path_factory):
    for path in (SLICE, HG00100):
        if not os.path.exists(path):
            pytest.skip(f"missing test input {path}")
    work = str(tmp_path_factory.mktemp("per_chrom_stage"))
    L.reset_state()
    L.run_init(SLICE, "chr21,chr22")
    for name, sources, region, ingest_region in [
        ("a", [SLICE, HG00100], CHR21_A, CHR21_A),
        ("b", [SLICE], CHR21_B, CHR21_B),
        ("wg", [SLICE], f"{CHR21_A},{CHR22}", None),
    ]:
        gvcfs = []
        for j, src in enumerate(sources):
            dst = os.path.join(work, f"{name}_{j}.g.vcf.gz")
            L.subset_gvcf(src, region, dst)
            gvcfs.append(dst)
        L.run_ingest(gvcfs, region=ingest_region, threads=2)
    conn = L.get_pg_conn()
    try:
        assert [s for _, s in L.batch_statuses(conn)] == ["complete"] * 3
        with conn.cursor() as cur:
            cur.execute("SELECT chromosome, id FROM contigs")
            contigs = dict(cur.fetchall())
    finally:
        conn.close()
    return contigs


def _units(conn, metric, c_idx):
    """{(tier, group_id)} checkpointed reduce units of the latest run (tier
    >= 0: the per-chromosome finalize checkpoint is tier -1)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT tier, group_id FROM tier_completion WHERE agg_run_id = "
            "(SELECT max(agg_run_id) FROM aggregation_progress) AND "
            "metric=%s AND chromosome_index=%s AND tier >= 0",
            (metric, c_idx),
        )
        return set(cur.fetchall())


def _rows_per_chrom(conn, table):
    with conn.cursor() as cur:
        cur.execute(
            f"SELECT chromosome_index, count(*) FROM {table} "
            "GROUP BY chromosome_index"
        )
        return dict(cur.fetchall())


def test_tier_plan_is_per_chromosome(staged):
    c21, c22 = staged["chr21"], staged["chr22"]
    conn = L.get_pg_conn()
    try:
        L.clear_pg_aggregates(conn)
        proc = L.run_aggregate(fan_in=2, threads=2, rebuild=True)
        assert (
            f"plan chr21 (c_idx {c21}): batches=[1,2,3] tiers=2" in proc.stderr
        )
        assert f"plan chr22 (c_idx {c22}): batches=[3] tiers=1" in proc.stderr

        for metric in ("allele_counts", "site_info"):
            # chr21: tier 0->1 (groups 0, 1) then 1->2 (root group 0).
            assert _units(conn, metric, c21) == {(1, 0), (1, 1), (2, 0)}
            # chr22: the whole-genome batch alone, one tier, no tier-2 unit.
            assert _units(conn, metric, c22) == {(1, 0)}
        # ref_blocks has no tier tree: one tier-1 window unit per default
        # 20 Mbp window on every covered chromosome (chr21 and chr22 are both
        # 40-60 Mbp), each reading all of that chromosome's batches.
        for c_idx in (c21, c22):
            assert _units(conn, "ref_blocks", c_idx) == {
                (1, 0),
                (1, 1),
                (1, 2),
            }

        # Both chromosomes finalized, to the oracle's values.
        site_rows = _rows_per_chrom(conn, "site_info")
        assert site_rows.get(c21, 0) > 0 and site_rows.get(c22, 0) > 0
        assert _rows_per_chrom(conn, "ref_blocks").get(c21, 0) > 0
        with conn.cursor() as cur:
            cur.execute(
                "SELECT chromosome_index, sum(count) FROM alleles "
                "GROUP BY chromosome_index"
            )
            sums = dict(cur.fetchall())
        assert sums.get(c21, 0) > 0 and sums.get(c22, 0) > 0
        L.assert_matches_oracle(conn, L.get_catalog())
    finally:
        conn.close()


def test_per_chrom_plan_single_pass_matches_two_tiers(staged):
    """fan_in 100 (one tier on both chromosomes) gives the same integers."""
    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=2, threads=2, rebuild=True)
        deep = L.snapshot_pg(conn)
        proc = L.run_aggregate(fan_in=100, threads=2, rebuild=True)
        assert "tiers=1 " in proc.stderr
        L.assert_pg_integer_equal(L.snapshot_pg(conn), deep)
    finally:
        conn.close()


def _chrom(snapshot, c_idx):
    """The part of a snapshot_pg on one chromosome."""
    return {
        name: {k: v for k, v in rows.items() if k[0] == c_idx}
        for name, rows in snapshot.items()
    }


def test_incremental_empty_delta_chromosome(staged):
    """Full run over the whole-genome batch + chr21 window A (chr21 and
    chr22); then the chr21-only batch arrives. The incremental run has an
    empty delta on chr22, so chr22's fold unit and window units rebuild its
    partitions from the parent's roots alone -- equal to a full rebuild."""
    c21, c22 = staged["chr21"], staged["chr22"]
    conn = L.get_pg_conn()
    catalog = L.get_catalog()
    try:
        L.clear_lineage(conn, catalog)
        L.set_completed_batches(conn, [1, 3])
        L.run_aggregate(fan_in=2, threads=2)
        parent = L.latest_agg_run_id(conn)

        L.set_completed_batches(conn, [1, 2, 3])
        proc = L.run_aggregate(fan_in=2, threads=2)
        child = L.latest_agg_run_id(conn)
        assert f"agg_run_id={child} fresh parent={parent} delta=1 " in (
            proc.stderr
        )
        assert f"plan chr22 (c_idx {c22}): batches=[] tiers=0 fold=1" in (
            proc.stderr
        )
        assert f"plan chr21 (c_idx {c21}): batches=[2] tiers=1 fold=2" in (
            proc.stderr
        )
        for metric in ("allele_counts", "site_info"):
            assert _units(conn, metric, c22) == {(1, 0)}  # the fold alone
            assert _units(conn, metric, c21) == {(1, 0), (2, 0)}
        assert _units(conn, "ref_blocks", c22) == {(1, 0), (1, 1), (1, 2)}
        incremental = L.snapshot_pg(conn)
        dense = L.dense_root(catalog, child).drop_columns(["agg_run_id"])
        assert len(_chrom(incremental, c22)["site_info_int"]) > 0
        assert len(_chrom(incremental, c22)["alleles"]) > 0
        L.assert_matches_oracle(conn, catalog)

        L.run_aggregate(fan_in=2, threads=2, rebuild=True)
        rebuild = L.snapshot_pg(conn)
        L.assert_pg_bitexact(_chrom(incremental, c22), _chrom(rebuild, c22))
        L.assert_pg_bitexact(incremental, rebuild)
        full = L.dense_root(catalog, L.latest_agg_run_id(conn))
        assert dense.equals(full.drop_columns(["agg_run_id"]))
        on_c22 = pc.equal(dense["chromosome_index"], c22)
        assert pc.sum(on_c22).as_py() > 10_000  # chr22's dense root kept
    finally:
        L.set_completed_batches(conn, [1, 2, 3])
        conn.close()
