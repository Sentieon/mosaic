# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""--pos-batch-size must not change pipeline output (INCREMENTAL_FLUSH Phase 1).

pos_batch_size controls only *how* ingest buckets/flushes per-sample and
per-batch ref-block data to parquet (memory vs. file count) — never *what* is
produced. This runs the full init -> ingest -> aggregate -> joint_call pipeline
at several pos_batch_size values and asserts identical results:

  * PG aggregates bit-exact (site_info / ref_blocks / alleles, by coordinate).
  * Iceberg row counts identical — including batch_ref_blocks, whose dense
    per-position row multiset must be invariant to the windowing (the direct
    check that the window split drops/duplicates no position).
  * The decoded joint-called VCF identical.

pos_batch_size=0 exercises the single-sentinel-window path; 25k and 100k force
many in-loop window flushes per shard (shard_size 1 Mbp).
"""

import os

import pytest

import jc_testlib as L

pytestmark = [pytest.mark.integration]

REGION = "chr21:5000000-7000000"
SHARD_SIZE = 1_000_000
PBS_VALUES = [0, 25_000, 100_000]

_ICEBERG_TABLES = [
    "sample_ref_blocks",
    "sample_site_info",
    "batch_ref_blocks",
    "batch_site_info",
    "batch_allele_counts",
]


def _run_pipeline(gvcfs, out_vcf, pos_batch_size):
    """A fresh init -> ingest -> aggregate -> joint_call at one pos_batch_size."""
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(
        gvcfs,
        region=REGION,
        shard_size=SHARD_SIZE,
        threads=2,
        pos_batch_size=pos_batch_size,
    )
    L.run_aggregate(fan_in=100, threads=2)
    L.run_joint_call(out_vcf)
    conn = L.get_pg_conn()
    try:
        pg = L.snapshot_pg(conn)
    finally:
        conn.close()
    ice = L.count_iceberg(L.get_catalog(), tables=_ICEBERG_TABLES)
    return pg, ice, L.snapshot_vcf(out_vcf)


def test_pos_batch_size_equivalence(clean_state, tmp_path):
    a = os.path.join(str(tmp_path), "a.g.vcf.gz")
    b = os.path.join(str(tmp_path), "b.g.vcf.gz")
    L.subset_gvcf(L.GVCFS[0], REGION, a)
    L.subset_gvcf(L.GVCFS[1], REGION, b)
    gvcfs = [a, b]

    results = {}
    for pbs in PBS_VALUES:
        out = os.path.join(str(tmp_path), f"out_{pbs}.vcf.gz")
        results[pbs] = _run_pipeline(gvcfs, out, pbs)

    base = PBS_VALUES[0]
    base_pg, base_ice, base_vcf = results[base]
    # Sanity: the fixture region actually produced data.
    assert base_ice["sample_site_info"] > 0
    assert base_ice["batch_ref_blocks"] > 0

    for pbs in PBS_VALUES[1:]:
        pg, ice, vcf = results[pbs]
        L.assert_pg_bitexact(pg, base_pg)
        assert ice == base_ice, (
            f"Iceberg row counts differ at pos_batch_size={pbs} "
            f"(vs {base}): {ice} vs {base_ice}"
        )
        assert (
            vcf == base_vcf
        ), f"joint-call VCF differs at pos_batch_size={pbs} (vs {base})"
