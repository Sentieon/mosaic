# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Local allele indices in sample_site_info and the derived alleles catalog.

Ingest writes no allele identities to PostgreSQL: each sample_site_info row
carries its record's own allele strings (since E3; before, its site's merged
list), with gt_1 / gt_2 as local indices into them, and every allele of the
site's merged list gets a batch_allele_counts row (count 0 included). `mosaic aggregate` derives the PostgreSQL ``alleles`` catalog
(identities, summed counts, del_range) from those partials, as a
LIST-partitioned table swapped in per chromosome like site_info and
ref_blocks. The shard workers open no PostgreSQL connection: the main process
commits each shard's checkpoint (the crash/resume semantics are covered by
test_ingest_resume.py).

Truth for the decoded genotypes is ``bcftools query`` over the same subset
files (test_genotype_matrix's record parser).
"""

import os
import subprocess

import pytest
from pyiceberg.schema import Schema
from pyiceberg.types import IntegerType, ListType, NestedField

import jc_testlib as L
from test_genotype_matrix import query_records

pytestmark = pytest.mark.integration

REGION = "chr21:5000000-10000000"
SHARD_SIZE = 1_000_000  # several shard boundaries inside the region
REF_BASES = set("ACGTN")


@pytest.fixture(scope="module")
def ingested(docker_stack, tmp_path_factory):
    """One batch of the two fixture samples over REGION, ingested only."""
    work = str(tmp_path_factory.mktemp("local_alleles"))
    subsets = []
    for j, src in enumerate(L.GVCFS):
        dst = os.path.join(work, f"s{j}.g.vcf.gz")
        L.subset_gvcf(src, REGION, dst)
        subsets.append(dst)
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    proc = L.run_ingest(
        subsets, region=REGION, shard_size=SHARD_SIZE, threads=2
    )
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT sample_name, id FROM samples")
            sid_of = dict(cur.fetchall())
    finally:
        conn.close()
    recs = {}
    for path in subsets:
        name = subprocess.run(
            ["bcftools", "query", "-l", path],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        recs[sid_of[name]] = query_records(path)
    ssi = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.sample_site_info")
        .scan()
        .to_arrow()
        .to_pylist()
    )
    return {"proc": proc, "recs": recs, "ssi": ssi}


def test_ingest_writes_no_allele_rows(ingested):
    """After ingest every alleles partition is empty (ingest wrote no
    identities), the shard checkpoints were committed by the main process
    and cleared at batch completion."""
    stderr = ingested["proc"].stderr
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT length FROM contigs WHERE chromosome='chr21'")
            n_shards = -(-cur.fetchone()[0] // SHARD_SIZE)
        assert (
            f"Checkpointed {n_shards} shard(s) in the main process" in stderr
        ), stderr[-3000:]
        assert "0 checkpointed by an earlier attempt" in stderr
        layout = L.partition_layout(conn)
        assert layout["alleles"] == {0: "alleles_c0_r0"}
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM alleles")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT to_regclass('allele_counts')")
            assert cur.fetchone()[0] is None
        assert L.shard_completion_count(conn) == 0
        assert dict(L.batch_statuses(conn)) == {1: "complete"}
    finally:
        conn.close()


def test_rows_carry_allele_strings_and_local_indices(ingested):
    rows = ingested["ssi"]
    assert len(rows) > 1000
    n_star = 0
    for r in rows:
        alleles = r["alleles"]
        assert alleles and all(isinstance(a, str) for a in alleles), r
        assert alleles[0] in REF_BASES, r  # the ref base first
        assert all(a[0] in "x+-*" for a in alleles[1:]), r
        assert len(set(alleles)) == len(alleles), r
        n_star += "*" in alleles
        for g in (r["gt_1"], r["gt_2"]):
            assert g is None or 0 <= g < len(alleles), r
    assert n_star > 0


def test_local_gt_decodes_to_the_gvcf_gt(ingested):
    """Each sample's variant record at a site decodes, through the row's
    allele list, to the alleles bcftools reports for its GT. The row lists
    exactly the record's own alleles in its order (E3): the ref base, then
    each ALT prefix-encoded (``*`` for <NON_REF>), and its GT indices are
    the record's own."""
    by_key = {}
    for r in ingested["ssi"]:
        by_key.setdefault((r["sample_id"], r["position"]), []).append(r)
    checked = 0
    for sid, recs in ingested["recs"].items():
        n_at = {}
        for rec in recs:
            if not rec.block:
                n_at[rec.pos] = n_at.get(rec.pos, 0) + 1
        for rec in recs:
            if rec.block or n_at[rec.pos] != 1:
                continue  # one record per (sample, site) only
            (row,) = by_key[(sid, rec.pos)]
            alleles = row["alleles"]
            assert alleles == [
                rec.allele(i) for i in range(len(rec.alts) + 1)
            ], (sid, rec.pos, alleles, rec.alts)
            assert [
                g for g in (row["gt_1"], row["gt_2"]) if g is not None
            ] == [x for x in rec.gt_fields if x is not None][:2]
            got = sorted(
                alleles[g] for g in (row["gt_1"], row["gt_2"]) if g is not None
            )
            want = sorted(
                rec.allele(x) for x in rec.gt_fields if x is not None
            )
            assert got == want, (sid, rec.pos, row, rec.gt)
            checked += 1
    assert checked > 1000


def test_aggregate_derives_the_catalog(ingested):
    """`mosaic aggregate` builds alleles from the partials: one row per
    (chromosome, position, allele) any partial lists, the summed count
    (zero rows kept), del_range derived from the string (all three checked
    by assert_matches_oracle), the GiST adopted by the attached partition,
    and every allele a sample row lists is in the catalog."""
    L.run_aggregate(fan_in=100, threads=2)
    conn = L.get_pg_conn()
    try:
        counts = L.assert_matches_oracle(conn, L.get_catalog())
        assert counts["alleles"] > 1000
        run = L.latest_agg_run_id(conn)
        L.assert_single_generation(conn, run)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FILTER (WHERE count = 0), "
                "count(*) FILTER (WHERE allele = '*' AND count = 0), "
                "count(*) FILTER (WHERE del_range IS NOT NULL), "
                "count(*) FILTER (WHERE allele LIKE '-%%') FROM alleles"
            )
            zero, zero_star, n_ranged, n_del = cur.fetchone()
            cur.execute(
                "SELECT chromosome_index, position, allele FROM alleles"
            )
            catalog = set(cur.fetchall())
            cur.execute("SELECT position FROM site_info")
            sites = {p for (p,) in cur.fetchall()}
            # The partition's GiST is attached to the parent's index.
            cur.execute(
                "SELECT c.relname FROM pg_inherits i "
                "JOIN pg_class c ON c.oid = i.inhrelid "
                "WHERE i.inhparent = 'alleles_delrange_index'::regclass"
            )
            gist_children = [n for (n,) in cur.fetchall()]
            cur.execute(
                "SELECT indexdef FROM pg_indexes WHERE tablename = %s",
                (f"alleles_c0_r{run}",),
            )
            defs = [d for (d,) in cur.fetchall()]
    finally:
        conn.close()
    assert zero > 0 and zero_star > 0  # listed-but-uncalled alleles kept
    assert n_ranged == n_del > 0
    assert {p for _c, p, _a in catalog} == sites
    for r in ingested["ssi"]:
        for a in r["alleles"]:
            assert (r["chromosome_index"], r["position"], a) in catalog, r
    assert len(gist_children) == 1
    assert any(
        "USING gist (del_range) WHERE (del_range IS NOT NULL)" in d
        for d in defs
    ), defs
    assert any("PRIMARY KEY" in d or "_pkey" in d for d in defs), defs


def test_ingest_refuses_a_lake_with_allele_ids(clean_state, tmp_path):
    """A lake initialized before the change (sample_site_info.alleles a
    list of PG allele ids) is refused at ingest start with the reset
    instructions, before anything is written."""
    L.run_init(L.GVCFS[0], "chr21")
    catalog = L.get_catalog()
    name = f"{L.TEST_NAMESPACE}.sample_site_info"
    fields = [
        (
            NestedField(
                6,
                "alleles",
                ListType(
                    element_id=21,
                    element_type=IntegerType(),
                    element_required=False,
                ),
                required=False,
            )
            if f.name == "alleles"
            else f
        )
        for f in catalog.load_table(name).schema().fields
    ]
    catalog.drop_table(name)
    catalog.create_table(name, schema=Schema(*fields))

    gvcf = os.path.join(str(tmp_path), "s0.g.vcf.gz")
    L.subset_gvcf(L.GVCFS[0], "chr21:10000000-10100000", gvcf)
    proc = L.run_ingest([gvcf], region="chr21", threads=1, check=False)
    assert proc.returncode != 0
    assert "sample_site_info.alleles" in proc.stderr, proc.stderr[-2000:]
    assert "no migration path" in proc.stderr
    assert "mosaic init" in proc.stderr
    assert "Traceback" not in proc.stderr
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM ingestion_progress")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT count(*) FROM samples")
            assert cur.fetchone()[0] == 0
    finally:
        conn.close()
