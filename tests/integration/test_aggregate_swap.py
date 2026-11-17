# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Build-then-attach finalize: PostgreSQL is never updated in place.

`mosaic aggregate` finalizes each (metric, chromosome) by building a fresh
generation table ``{table}_c{c}_r{agg_run_id}``, indexing it, and swapping it
in for the attached partition (DETACH old, ATTACH new, DROP old, checkpoint
row) in one transaction. These tests check the partition layout the swaps
leave behind, that a crash before the swap commits keeps the previous
generation readable (and one after it is skipped on resume), the derived
alleles catalog (which replaced the append-once alleles table and the
allele_counts table), and `mosaic index`.

Fixture: the 3 chr21 batches of test_aggregate_resume over a chr20 + chr21
init, so every metric has a chromosome with data (chr21, c_idx 1) and one no
batch covers (chr20, c_idx 0, whose partitions are attached empty).
"""

import os

import pytest

import jc_testlib as L

pytestmark = [pytest.mark.integration, pytest.mark.restart]

REGIONS = [
    "chr21:10000000-10500000",
    "chr21:20000000-20500000",
    "chr21:25000000-25500000",
]
FAN_IN = 2
C_DATA = 1  # chr21 (contig ids follow the header order of the init filter)


@pytest.fixture(scope="module")
def staged(docker_stack, tmp_path_factory):
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr20,chr21")
    work = tmp_path_factory.mktemp("swap_stage")
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
        with conn.cursor() as cur:
            cur.execute("SELECT chromosome, id FROM contigs")
            assert dict(cur.fetchall()) == {"chr20": 0, "chr21": C_DATA}
    finally:
        conn.close()
    return REGIONS


def _site_rows(conn):
    """site_info as {(c, pos): row} -- what a reader sees right now."""
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM site_info")
        return {(r[0], r[1]): r for r in cur.fetchall()}


def _assert_loaded_frozen(conn):
    """Every attached generation partition has autovacuum off, and its
    n_live_tup (set by the finalize's explicit ANALYZE, since autovacuum
    never analyzes it) equals its row count."""
    layout = L.partition_layout(conn)
    with conn.cursor() as cur:
        for table, parts in layout.items():
            for c_idx, name in parts.items():
                cur.execute(
                    "SELECT c.reloptions, s.n_live_tup FROM pg_class c "
                    "JOIN pg_stat_user_tables s ON s.relid = c.oid "
                    "WHERE c.relname = %s",
                    (name,),
                )
                reloptions, n_live = cur.fetchone()
                assert "autovacuum_enabled=false" in (reloptions or []), (
                    name,
                    reloptions,
                )
                cur.execute(f"SELECT count(*) FROM {name}")
                assert n_live == cur.fetchone()[0], (name, n_live)


def test_init_creates_generation_zero(staged):
    """Before any aggregate, init's empty r0 partitions are attached."""
    conn = L.get_pg_conn()
    try:
        if L.latest_agg_run_id(conn) is None:
            L.assert_single_generation(conn, 0)
    finally:
        conn.close()


def test_one_generation_after_run_rebuild_and_resume(staged):
    """After a run, a --rebuild and a resume, each aggregate table has exactly
    one partition per contig -- the latest run's generation -- and no
    detached generation tables are left."""
    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
        first = L.latest_agg_run_id(conn)
        L.assert_single_generation(conn, first)
        _assert_loaded_frozen(conn)
        base = L.snapshot_pg(conn)

        L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
        second = L.latest_agg_run_id(conn)
        assert second > first
        L.assert_single_generation(conn, second)
        L.assert_pg_bitexact(L.snapshot_pg(conn), base)

        # Crash after chr21's ref_blocks swap commits: site_info is done,
        # ref_blocks is (partly) done, the alleles finalize never started.
        proc = L.run_aggregate(
            fan_in=FAN_IN,
            threads=2,
            rebuild=True,
            extra_env={
                "MOSAIC_TEST_FAIL_FINALIZE": f"ref_blocks:{C_DATA}:post_commit"
            },
            check=False,
        )
        assert proc.returncode != 0
        third = L.latest_agg_run_id(conn)
        layout = L.partition_layout(conn)
        assert layout["site_info"][C_DATA] == f"site_info_c{C_DATA}_r{third}"
        assert layout["ref_blocks"][C_DATA] == f"ref_blocks_c{C_DATA}_r{third}"
        assert layout["alleles"][C_DATA] == f"alleles_c{C_DATA}_r{second}"

        proc = L.run_aggregate(fan_in=FAN_IN, threads=2)  # resume
        assert f"agg_run_id={third} resume" in proc.stderr
        assert "site_info finalize: skipped" in proc.stderr
        L.assert_single_generation(conn, third)
        _assert_loaded_frozen(conn)
        L.assert_pg_bitexact(L.snapshot_pg(conn), base)
    finally:
        conn.close()


@pytest.mark.parametrize("contig", ["*", str(C_DATA)])
def test_pre_swap_crash_keeps_old_generation_readable(staged, contig):
    """A crash just before the site_info swap commits rolls the swap back: the
    previous generation stays attached and readable, the new one is left
    detached, and no checkpoint row exists. The resume rebuilds it, swaps it
    in, and leaves no leftover; the result is bit-exact."""
    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
        old_run = L.latest_agg_run_id(conn)
        base = L.snapshot_pg(conn)
        old_sites = _site_rows(conn)
        assert len(old_sites) > 1000

        proc = L.run_aggregate(
            fan_in=FAN_IN,
            threads=2,
            rebuild=True,
            extra_env={
                "MOSAIC_TEST_FAIL_FINALIZE": f"site_info:{contig}:pre_commit"
            },
            check=False,
        )
        assert proc.returncode != 0
        assert "MOSAIC_TEST_FAIL_FINALIZE" in proc.stderr
        new_run = L.latest_agg_run_id(conn)
        assert new_run > old_run
        assert C_DATA not in L.finalize_markers(conn, "site_info")
        # Readers still see the first run's site_info, from its partition.
        assert _site_rows(conn) == old_sites
        layout = L.partition_layout(conn)
        assert layout["site_info"][C_DATA] == (
            f"site_info_c{C_DATA}_r{old_run}"
        )
        gens = L.generation_tables(conn)
        assert gens.get(f"site_info_c{C_DATA}_r{new_run}") is False
        detached = sorted(n for n, att in gens.items() if not att)
        assert all(n.startswith("site_info_c") for n in detached), detached
        assert all(n.endswith(f"_r{new_run}") for n in detached), detached

        proc = L.run_aggregate(fan_in=FAN_IN, threads=2)  # resume
        assert f"agg_run_id={new_run} resume" in proc.stderr
        L.assert_single_generation(conn, new_run)
        L.assert_pg_bitexact(L.snapshot_pg(conn), base)
    finally:
        conn.close()


def test_post_swap_crash_is_skipped_on_resume(staged):
    """A crash just after the site_info swap commits: the new generation is
    attached and checkpointed, the old one dropped; the resume skips it."""
    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
        old_run = L.latest_agg_run_id(conn)
        base = L.snapshot_pg(conn)

        proc = L.run_aggregate(
            fan_in=FAN_IN,
            threads=2,
            rebuild=True,
            extra_env={"MOSAIC_TEST_FAIL_FINALIZE": "site_info:*:post_commit"},
            check=False,
        )
        assert proc.returncode != 0
        new_run = L.latest_agg_run_id(conn)
        assert L.finalize_marker_exists(conn, "site_info")
        layout = L.partition_layout(conn)
        for c_idx, name in layout["site_info"].items():
            assert name == f"site_info_c{c_idx}_r{new_run}"
        for table in ("ref_blocks", "alleles"):
            for c_idx, name in layout[table].items():
                assert name == f"{table}_c{c_idx}_r{old_run}"
        assert not [
            n for n, att in L.generation_tables(conn).items() if not att
        ]
        L.assert_pg_bitexact(L.snapshot_pg(conn), base)

        proc = L.run_aggregate(fan_in=FAN_IN, threads=2)  # resume
        assert "site_info finalize: skipped" in proc.stderr
        L.assert_single_generation(conn, new_run)
        L.assert_pg_bitexact(L.snapshot_pg(conn), base)
    finally:
        conn.close()


def test_rebuild_reclaims_detached_generations(staged):
    """A --rebuild after a pre-swap crash (instead of a resume) drops the
    crashed run's detached generation tables along with its staging."""
    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
        base = L.snapshot_pg(conn)
        proc = L.run_aggregate(
            fan_in=FAN_IN,
            threads=2,
            rebuild=True,
            extra_env={"MOSAIC_TEST_FAIL_FINALIZE": "site_info:*:pre_commit"},
            check=False,
        )
        assert proc.returncode != 0
        crashed = L.latest_agg_run_id(conn)
        assert [n for n, att in L.generation_tables(conn).items() if not att]

        proc = L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
        assert f"dropping detached site_info_c{C_DATA}_r{crashed}" in (
            proc.stderr
        )
        L.assert_single_generation(conn, L.latest_agg_run_id(conn))
        L.assert_pg_bitexact(L.snapshot_pg(conn), base)
    finally:
        conn.close()


def test_alleles_is_the_derived_catalog(staged):
    """alleles is a partitioned catalog with no id: every allele any batch
    listed (zero counts kept), its summed count and derived del_range
    (assert_matches_oracle checks all three); there is no allele_counts
    table; each attached partition carries the del_range GiST, attached to
    the parent's index."""
    conn = L.get_pg_conn()
    try:
        L.clear_pg_aggregates(conn)
        L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_schema = current_schema() "
                "AND table_name = 'alleles'"
            )
            columns = {c for (c,) in cur.fetchall()}
            cur.execute("SELECT to_regclass('allele_counts')")
            counts_table = cur.fetchone()[0]
            cur.execute("SELECT count(*), min(count) FROM alleles")
            n_rows, min_count = cur.fetchone()
            cur.execute(
                "SELECT count(*) FROM pg_inherits "
                "WHERE inhparent = 'alleles_delrange_index'::regclass"
            )
            n_gist = cur.fetchone()[0]
        assert columns == {
            "chromosome_index",
            "position",
            "allele",
            "del_range",
            "count",
        }
        assert counts_table is None
        assert n_rows > 1000
        assert min_count == 0
        assert n_gist == len(L.partition_layout(conn)["alleles"]) == 2
        L.assert_matches_oracle(conn, L.get_catalog())
    finally:
        conn.close()


def _indexes(conn):
    """Index names on the tables that are not partitions (a partitioned
    parent's index stands for its per-partition children)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT i.relname FROM pg_index x "
            "JOIN pg_class i ON i.oid = x.indexrelid "
            "JOIN pg_class t ON t.oid = x.indrelid "
            "WHERE t.relnamespace = current_schema()::regnamespace "
            "AND NOT t.relispartition"
        )
        return {n for (n,) in cur.fetchall()}


def test_index_creates_only_the_gist_and_drops_redundant(staged):
    """`mosaic index` re-creates only the parent alleles_delrange_index when
    it is missing, drops the two redundant indexes earlier versions
    created, and is idempotent (nothing to do on a current database); a
    re-aggregate after it still swaps cleanly."""
    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
        base = L.snapshot_pg(conn)
        with conn.cursor() as cur:
            cur.execute("DROP INDEX IF EXISTS alleles_delrange_index")
        before = _indexes(conn)
        with conn.cursor() as cur:
            cur.execute(
                "CREATE INDEX alleles_pos_index "
                "ON alleles (chromosome_index, position)"
            )
            cur.execute(
                "CREATE INDEX ref_block_pos_index "
                "ON ref_blocks (chromosome_index, position)"
            )
        assert {"alleles_pos_index", "ref_block_pos_index"} <= _indexes(conn)

        L.run_index()
        after = _indexes(conn)
        assert after == before | {"alleles_delrange_index"}
        proc = L.run_index()
        assert "Nothing to do" in proc.stdout
        assert _indexes(conn) == after

        L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
        L.assert_single_generation(conn, L.latest_agg_run_id(conn))
        L.assert_pg_bitexact(L.snapshot_pg(conn), base)
        assert "alleles_delrange_index" in _indexes(conn)
    finally:
        conn.close()
