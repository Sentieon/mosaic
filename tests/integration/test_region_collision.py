# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Regression: per-region ingests of one sample set must not collide on S3.

The pipeline fans out ingestion per chromosome, running the SAME sample batch against
every chromosome into the SAME shared Iceberg tables. Historically each aggregated
parquet file's S3 key covered only (table prefix, sorted sample names, aggregation
index) — NOT the region/batch — so two per-region tasks for the same samples computed
identical keys and silently overwrote each other's data files, losing a region's rows.

This reproduces it end-to-end: ingest two NON-overlapping chr21 windows under the same
sample set, then assert rows from BOTH windows survive in BOTH shared Iceberg tables.

Migrated from the root-level standalone script; the destructive ``clean_db.py`` call
is replaced by the harness's surgical ``reset_state`` (the ``clean_state`` fixture).
"""

import os

import pytest

import jc_testlib as L

pytestmark = pytest.mark.integration

REGION_A = "chr21:10000000-10200000"
REGION_B = "chr21:20000000-20200000"
MIDPOINT = 15_000_000  # between the two windows


def _positions(catalog, table_short, pos_column):
    tbl = catalog.load_table(f"{L.TEST_NAMESPACE}.{table_short}")
    return tbl.scan().to_arrow().column(pos_column).to_pylist()


def test_per_region_ingests_do_not_collide(clean_state, tmp_path):
    L.run_init(L.GVCFS[0], "chr21")

    a_gvcfs, b_gvcfs = [], []
    for i, src in enumerate(L.GVCFS):
        a = os.path.join(str(tmp_path), f"A_{i}.g.vcf.gz")
        b = os.path.join(str(tmp_path), f"B_{i}.g.vcf.gz")
        L.subset_gvcf(src, REGION_A, a)
        L.subset_gvcf(src, REGION_B, b)
        a_gvcfs.append(a)
        b_gvcfs.append(b)

    # Same samples, different region, same shared Iceberg tables.
    L.run_ingest(a_gvcfs, region=REGION_A, threads=2)
    L.run_ingest(b_gvcfs, region=REGION_B, threads=2)

    catalog = L.get_catalog()
    for table_short, pos_column in [
        ("sample_ref_blocks", "position_start"),
        ("sample_site_info", "position"),
    ]:
        positions = _positions(catalog, table_short, pos_column)
        n_a = sum(1 for p in positions if p < MIDPOINT)
        n_b = sum(1 for p in positions if p >= MIDPOINT)
        assert n_a > 0 and n_b > 0, (
            f"{table_short} lost a region (A={n_a}, B={n_b}); the second batch "
            "overwrote the first on S3 (key collision)"
        )
