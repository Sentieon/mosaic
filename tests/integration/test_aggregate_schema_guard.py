# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""`mosaic aggregate` refuses a database initialized by an older `mosaic init`.

The finalize swaps chromosome partitions of site_info / ref_blocks /
alleles, so it cannot run against the old shapes (site_info a plain table;
alleles the plain, append-once identity table ingest used to write), nor
against an aggregation_progress without the run lineage columns. There is no
migration path: the run must fail at startup, before it registers a run or
touches any table, with a message that says to re-initialize.
"""

import pytest

import jc_testlib as L

pytestmark = pytest.mark.integration

# site_info as `mosaic init` created it before the aggregate tables were
# partitioned (a plain table with an FK to contigs).
OLD_SITE_INFO = """
    CREATE TABLE site_info (
        chromosome_index INTEGER,
        position INTEGER,
        qual REAL DEFAULT 0.0,
        dp BIGINT DEFAULT 0,
        mqsq_1 REAL DEFAULT 0.0,
        mqsq_2 BIGINT DEFAULT 0,
        sb BIGINT[] CONSTRAINT sq_length CHECK (array_length(sb, 1) = 4),
        gtc BIGINT[] CONSTRAINT gtc_length CHECK (array_length(gtc, 1) = 3),
        PRIMARY KEY (chromosome_index, position),
        FOREIGN KEY (chromosome_index) REFERENCES contigs(id)
    )
"""


def _refused(conn, expected):
    proc = L.run_aggregate(fan_in=2, threads=2, check=False)
    assert proc.returncode != 0
    assert expected in proc.stderr, proc.stderr[-2000:]
    assert "no migration path" in proc.stderr
    assert "mosaic init" in proc.stderr
    assert "Traceback" not in proc.stderr  # a clear message, not a crash
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM aggregation_progress")
        assert cur.fetchone()[0] == 0  # refused before registering a run


def test_refuses_unpartitioned_site_info(clean_state):
    L.run_init(L.GVCFS[0], "chr21")
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE site_info CASCADE")
            cur.execute(OLD_SITE_INFO)
        _refused(conn, "site_info is not partitioned")
    finally:
        conn.close()


# alleles as `mosaic init` created it before the derived catalog (ingest
# inserted the identities; the counts lived in allele_counts).
OLD_ALLELES = """
    CREATE TABLE alleles (
        id BIGSERIAL PRIMARY KEY,
        chromosome_index INTEGER,
        position INTEGER,
        del_range INT4RANGE,
        allele TEXT,
        FOREIGN KEY (chromosome_index) REFERENCES contigs(id),
        UNIQUE (chromosome_index, position, allele)
    )
"""


def test_refuses_plain_alleles(clean_state):
    L.run_init(L.GVCFS[0], "chr21")
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP TABLE alleles CASCADE")
            cur.execute(OLD_ALLELES)
        _refused(conn, "alleles is not partitioned")
    finally:
        conn.close()


def test_refuses_aggregation_progress_without_lineage_columns(clean_state):
    """A database initialized before aggregation_progress gained the run
    lineage columns (it also lacks the dense ref_blocks staging table)."""
    L.run_init(L.GVCFS[0], "chr21")
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("DROP VIEW cohort_generation")
            cur.execute(
                "ALTER TABLE aggregation_progress DROP COLUMN parent_run_id, "
                "DROP COLUMN delta_batch_ids, DROP COLUMN staging_pruned, "
                "DROP COLUMN ref_shard_size"
            )
        _refused(
            conn,
            "aggregation_progress lacks parent_run_id, delta_batch_ids, "
            "staging_pruned, ref_shard_size",
        )
    finally:
        conn.close()
