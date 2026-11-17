# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The removal substrate (the per-record contribution columns, the lake
oracle and the tombstone).

* The lake oracle (jc_testlib.assert_partials_derivable_from_lake, the E1
  acceptance test): every batch_site_info, batch_allele_counts and
  batch_ref_blocks row is recomputed from sample_site_info and
  sample_ref_blocks alone and must match, on the three-batch chr21 fixture,
  on a run whose shard boundary falls inside a deletion's span (the
  cross-shard deletion-split spans), and on HG00096 + HG00097 when present
  (``./.`` blocks, many deletions).
* The regression the ref-block bug escaped: the same samples ingested as one
  joint batch and as one batch per sample give the same per-position
  batch_ref_blocks sums, and both equal the lake recompute.
* The tombstone (samples.status = 'removed', set by SQL: nothing in the
  pipeline sets it, sample removal is not implemented): joint_call.py drops
  the sample's header column, FORMAT values, ``*`` carriage and ref-block
  fill while the INFO fields from the PostgreSQL aggregates stay (AC / AN /
  AF, from the emitted genotypes since J0, count the remaining sample); with no
  tombstone the VCF is byte-identical; gwas_benchmark.py's subset recount
  drops it; `mosaic ingest` refuses it before doing any work.
* Old stores: a lake without the contribution columns is refused at ingest
  start; a database without samples.status gets it from `mosaic init` and
  from ensure_ingest_schema (so `mosaic ingest`), and joint_call.py exits
  with the remedy.
"""

import gzip
import os
import subprocess
import sys

import pytest
from pyiceberg.schema import Schema

import jc_testlib as L
from mosaic.init_db import (
    DEL_REF_SPANS_COLUMN,
    SAMPLE_SITE_CONTRIBUTION_COLUMNS,
    ensure_ingest_schema,
)

pytestmark = pytest.mark.integration

REGIONS = [
    "chr21:10000000-10300000",
    "chr21:20000000-20300000",
    "chr21:25000000-25300000",
]
SHARD_SIZE = 100_000  # shard boundaries inside every region
# A shard boundary at 0-based 10,001,908 (4 x 2,500,477): inside the span of
# the two fixture samples' deletion GAAA>G at 10,001,907 (record span
# 10,001,907-10,001,910), with their ref blocks from 10,001,908.
CROSS_SHARD_SIZE = 2_500_477
CROSS_DEL_POS = 10_001_907
HG00096 = os.path.join(
    L.REPO_ROOT,
    "test_data/input_gvcfs5/HG00096.chr21.final.hard-filtered.gvcf.gz",
)
HG00097 = os.path.join(
    L.REPO_ROOT,
    "test_data/input_gvcfs5/HG00097.chr21.final.hard-filtered.gvcf.gz",
)
INFO_FROM_AGGREGATES = ("DP", "QD", "MQ", "FS", "SOR")


def _subsets(work, region, gvcfs, tag):
    out = []
    for j, src in enumerate(gvcfs):
        dst = os.path.join(work, f"{tag}_s{j}.g.vcf.gz")
        if not os.path.exists(dst):
            L.subset_gvcf(src, region, dst)
        out.append(dst)
    return out


def _set_status(name, status):
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE samples SET status = %s WHERE sample_name = %s",
                (status, name),
            )
            assert cur.rowcount == 1
    finally:
        conn.close()


def _sample_names():
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT sample_name FROM samples ORDER BY id")
            return [n for (n,) in cur.fetchall()]
    finally:
        conn.close()


def _vcf(path):
    """(text, header sample names, {(chrom, pos, ref): fields})."""
    with gzip.open(path, "rt") as f:
        text = f.read()
    samples = []
    recs = {}
    for line in text.splitlines():
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
        elif not line.startswith("#"):
            f = line.split("\t")
            recs[(f[0], int(f[1]), f[3])] = f
    return text, samples, recs


def _call(fmt, value):
    """One sample's FORMAT values as {key: value}, missing ('.') dropped."""
    return {k: v for k, v in zip(fmt.split(":"), value.split(":")) if v != "."}


def _info(field):
    out = {}
    for kv in field.split(";"):
        k, _, v = kv.partition("=")
        out[k] = v
    return out


@pytest.fixture(scope="module")
def three_batches(docker_stack, tmp_path_factory):
    """The two fixture samples as three batches (three chr21 windows),
    ingested inline, aggregated and joint-called. Its tests come first in
    this module: the later tests reset the stores."""
    work = str(tmp_path_factory.mktemp("removal_substrate"))
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    for i, region in enumerate(REGIONS):
        L.run_ingest(
            _subsets(work, region, L.GVCFS, f"b{i}"),
            region=region,
            shard_size=SHARD_SIZE,
            threads=2,
        )
    L.run_aggregate(threads=2)
    vcf = os.path.join(work, "baseline.vcf.gz")
    L.run_joint_call(vcf)
    return {"work": work, "vcf": vcf, "names": _sample_names()}


# ── the lake oracle ──────────────────────────────────────────────────────────


def test_partials_derivable_from_the_lake(three_batches):
    conn = L.get_pg_conn()
    try:
        stats = L.assert_partials_derivable_from_lake(conn, L.get_catalog())
    finally:
        conn.close()
    assert stats["batches"] == 3
    assert stats["sites"] > 500 and stats["alleles"] > stats["sites"]
    assert stats["zero_count_alleles"] > 0
    assert stats["ref_positions"] > 500_000
    # The deletion split is exercised...
    assert stats["deletion_rows_with_spans"] > 0
    assert stats["split_positions"] >= stats["deletion_rows_with_spans"]
    # ...and FORMAT/DP would not reproduce batch_site_info.dp (so the rows
    # carry site_dp, INFO/DP).
    assert stats["format_dp_differs"] > 0


# ── the tombstone ────────────────────────────────────────────────────────────


def test_tombstone_hides_the_sample_from_joint_call(three_batches, tmp_path):
    keep, gone = three_batches["names"]
    base_text, base_samples, base = _vcf(three_batches["vcf"])
    assert base_samples == sorted([keep, gone])
    _set_status(gone, "removed")
    try:
        out = str(tmp_path / "tomb.vcf.gz")
        proc = L.run_joint_call(out)
        assert "Excluding 1 tombstoned sample(s)" in proc.stderr
        _text, samples, recs = _vcf(out)
    finally:
        _set_status(gone, "active")
    assert samples == [keep]
    # Site selection comes from PostgreSQL: the same records.
    assert set(recs) == set(base)
    ki = base_samples.index(keep)
    for key, f in recs.items():
        b = base[key]
        assert len(f) == 10  # one sample column
        bi, ti = _info(b[7]), _info(f[7])
        for k in INFO_FROM_AGGREGATES:
            assert bi.get(k) == ti.get(k), (key, k, bi, ti)
        if b[4] == f[4]:
            # Same ALT list: the remaining sample's call is unchanged (the
            # FORMAT keys are the union over the record's samples).
            assert _call(f[8], f[9]) == _call(b[8], b[9 + ki]), key
        else:
            # Only a `*` the tombstoned sample carried can go.
            assert "*" in b[4].split(",") and "*" not in f[4].split(","), key
        # Since J0 AC / AN come from the emitted genotypes: the remaining
        # sample's call alone (the tombstoned one's no longer counts).
        n_alts = 0 if f[4] == "." else len(f[4].split(","))
        gt = f[9].split(":")[0].split("/")
        called = [int(a) for a in gt if a != "."]
        ac = [sum(1 for a in called if a == i) for i in range(1, n_alts + 1)]
        assert ti["AN"] == str(len(called)), key
        assert ti["AC"] == ",".join(map(str, ac)), key
    # Cleared again: byte-identical to the VCF before the tombstone.
    again = str(tmp_path / "again.vcf.gz")
    L.run_joint_call(again)
    assert _vcf(again)[0] == base_text


def _gwas(samples):
    proc = subprocess.run(
        [
            sys.executable,
            "scripts/gwas_benchmark.py",
            "--af-cutoff",
            "0.2",
            "--samples",
            *samples,
        ],
        cwd=L.REPO_ROOT,
        env=L.test_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    return proc


def test_tombstone_in_gwas_benchmark(three_batches):
    """With one sample tombstoned, a subset naming both recounts exactly as
    the subset of the other one alone (the global AF, from the aggregates,
    is unchanged)."""
    keep, gone = three_batches["names"]
    alone = _gwas([keep])
    assert len(alone.stdout.splitlines()) > 50
    _set_status(gone, "removed")
    try:
        both = _gwas([keep, gone])
    finally:
        _set_status(gone, "active")
    assert both.stdout == alone.stdout
    assert f"tombstoned samples left out of the subset: {gone}" in (
        both.stderr
    )
    assert "Subset size: 1 samples" in both.stderr


def test_ingest_refuses_a_removed_sample(three_batches):
    keep, gone = three_batches["names"]
    region = "chr21:15000000-15100000"
    subsets = _subsets(three_batches["work"], region, L.GVCFS, "r")
    catalog = L.get_catalog()
    files = L.iceberg_data_files(catalog)
    conn = L.get_pg_conn()
    try:
        before = (L.batch_statuses(conn), _sample_names())
    finally:
        conn.close()
    _set_status(gone, "removed")
    try:
        proc = L.run_ingest(subsets, region=region, check=False)
    finally:
        _set_status(gone, "active")
    assert proc.returncode == 1
    assert f"sample(s) {gone} were removed" in proc.stderr
    assert "not supported yet" in proc.stderr
    assert "Traceback" not in proc.stderr
    conn = L.get_pg_conn()
    try:
        assert (L.batch_statuses(conn), _sample_names()) == before
    finally:
        conn.close()
    assert L.iceberg_data_files(catalog) == files


# ── the lake oracle on other layouts (each resets the stores) ────────────────


def test_cross_shard_deletion_spans(docker_stack, tmp_path):
    """A shard boundary inside a deletion's span: the next shard's part of
    the split is added to the deletion row the previous shard wrote."""
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    proc = L.run_ingest(
        _subsets(str(tmp_path), REGIONS[0], L.GVCFS, "x"),
        region=REGIONS[0],
        shard_size=CROSS_SHARD_SIZE,
        threads=2,
    )
    assert "Added cross-shard deletion-split spans to 2" in proc.stderr
    from pyiceberg.expressions import EqualTo

    rows = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.sample_site_info")
        .scan(row_filter=EqualTo("position", CROSS_DEL_POS))
        .to_arrow()
        .to_pylist()
    )
    assert len(rows) == 2
    for r in rows:
        assert r[DEL_REF_SPANS_COLUMN] == [
            CROSS_DEL_POS + 1,
            CROSS_DEL_POS + 3,
        ]
    conn = L.get_pg_conn()
    try:
        L.assert_partials_derivable_from_lake(conn, L.get_catalog())
    finally:
        conn.close()


@pytest.mark.skipif(
    not (os.path.exists(HG00096) and os.path.exists(HG00097)),
    reason="test_data/input_gvcfs5 not present",
)
def test_partials_derivable_with_no_call_blocks(docker_stack, tmp_path):
    region = "chr21:5000000-10000000"
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(
        _subsets(str(tmp_path), region, [HG00096, HG00097], "g5"),
        region=region,
        shard_size=1_000_000,
        threads=2,
    )
    catalog = L.get_catalog()
    conn = L.get_pg_conn()
    try:
        stats = L.assert_partials_derivable_from_lake(conn, catalog)
    finally:
        conn.close()
    assert stats["deletion_rows_with_spans"] > 100
    counts = (
        catalog.load_table(f"{L.TEST_NAMESPACE}.sample_ref_blocks")
        .scan()
        .to_arrow()
        .column("sample_counts")
        .to_pylist()
    )
    assert 0 in counts and 1 in counts  # ./. and 0/0 blocks


def _ref_sums():
    """{(chromosome_index, position): (dp, sample_counts)} summed over every
    batch's batch_ref_blocks rows."""
    rows = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.batch_ref_blocks")
        .scan()
        .to_arrow()
        .to_pylist()
    )
    out = {}
    for r in rows:
        k = (r["chromosome_index"], r["position"])
        a = out.get(k, (0, 0))
        out[k] = (a[0] + r["dp"], a[1] + r["sample_counts"])
    return out


def _allele_sums():
    """{(chromosome_index, position, allele): (count, n_listed)} summed
    over every batch."""
    out = {}
    rows = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.batch_allele_counts")
        .scan()
        .to_arrow()
        .to_pylist()
    )
    for r in rows:
        k = (r["chromosome_index"], r["position"], r["allele"])
        a = out.get(k, (0, 0))
        out[k] = (a[0] + r["count"], a[1] + r["n_listed"])
    return out


def _sample_rows():
    """{sample name: (its sample_site_info rows, its sample_ref_blocks
    rows)}, each a sorted list of value tuples (sample_id dropped)."""
    names = {i: n for n, i in _sample_ids().items()}
    out = {}
    for table in ("sample_site_info", "sample_ref_blocks"):
        rows = (
            L.get_catalog()
            .load_table(f"{L.TEST_NAMESPACE}.{table}")
            .scan()
            .to_arrow()
            .to_pylist()
        )
        for r in rows:
            name = names[r.pop("sample_id")]
            key = tuple(
                tuple(v) if isinstance(v, list) else v for v in r.values()
            )
            out.setdefault(name, ([], []))[
                0 if table == "sample_site_info" else 1
            ].append(key)
    return {n: (sorted(a), sorted(b)) for n, (a, b) in out.items()}


def _sample_ids():
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT sample_name, id FROM samples")
            return dict(cur.fetchall())
    finally:
        conn.close()


def test_split_batches_equal_one_joint_batch(docker_stack, tmp_path):
    """One joint batch of both samples and one batch per sample must give
    the same per-position batch_ref_blocks sums (the bug the old ref-block
    sweep had only showed across batches) and allele (count, n_listed)
    sums, each layout must equal the lake recompute, and (E3, option B:
    each row holds its own record's allele list) every sample's per-sample
    rows are identical whether it was ingested alone or with a batch-mate,
    so the joint-called VCF is byte-identical too."""
    region = REGIONS[0]
    subsets = _subsets(str(tmp_path), region, L.GVCFS, "j")
    layouts = {"joint": [subsets], "split": [[s] for s in subsets]}
    sums = {}
    for name, batches in layouts.items():
        L.reset_state()
        L.run_init(L.GVCFS[0], "chr21")
        for gvcfs in batches:
            L.run_ingest(gvcfs, region=region, shard_size=SHARD_SIZE)
        conn = L.get_pg_conn()
        try:
            L.assert_partials_derivable_from_lake(conn, L.get_catalog())
        finally:
            conn.close()
        L.run_aggregate(threads=2)
        vcf = str(tmp_path / f"{name}.vcf.gz")
        L.run_joint_call(vcf)
        sums[name] = (_ref_sums(), _allele_sums(), _sample_rows(), _vcf(vcf))
    joint, split = sums["joint"], sums["split"]
    assert len(joint[0]) > 100_000
    diff = sorted(
        k
        for k in set(joint[0]) | set(split[0])
        if joint[0].get(k) != split[0].get(k)
    )
    assert not diff, [(k, joint[0].get(k), split[0].get(k)) for k in diff[:5]]
    assert joint[1] == split[1]  # zero counts and listings included
    # The independence property: each sample's rows do not depend on its
    # batch-mates.
    assert sorted(joint[2]) == sorted(split[2]) == ["HG00245", "HG00420"]
    for sample in joint[2]:
        for j, table in enumerate(("sample_site_info", "sample_ref_blocks")):
            a, b = joint[2][sample][j], split[2][sample][j]
            assert len(a) > 100 and a == b, (sample, table)
    assert joint[3][0] == split[3][0]  # the joint-called VCF, byte for byte
    assert len(joint[3][2]) > 100


# ── old stores ───────────────────────────────────────────────────────────────


@pytest.mark.parametrize("table", ["sample_site_info", "sample_ref_blocks"])
def test_ingest_refuses_a_lake_without_the_contribution(
    clean_state, tmp_path, table
):
    L.run_init(L.GVCFS[0], "chr21")
    catalog = L.get_catalog()
    name = f"{L.TEST_NAMESPACE}.{table}"
    drop = set(SAMPLE_SITE_CONTRIBUTION_COLUMNS) | {
        DEL_REF_SPANS_COLUMN,
        "sample_counts",
    }
    fields = [
        f
        for f in catalog.load_table(name).schema().fields
        if f.name not in drop
    ]
    catalog.drop_table(name)
    catalog.create_table(name, schema=Schema(*fields))

    gvcf = os.path.join(str(tmp_path), "s0.g.vcf.gz")
    L.subset_gvcf(L.GVCFS[0], "chr21:10000000-10100000", gvcf)
    proc = L.run_ingest([gvcf], region="chr21", threads=1, check=False)
    assert proc.returncode != 0
    assert f"{table}." in proc.stderr, proc.stderr[-2000:]
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


def _has_status(conn):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM information_schema.columns "
            "WHERE table_name = 'samples' AND column_name = 'status'"
        )
        return cur.fetchone()[0] == 1


def _drop_status(conn):
    """Make the database look initialized before the tombstone: no
    samples.status, and no sample_qc_summary view (since E3 it reads the
    column; the older view did not)."""
    with conn.cursor() as cur:
        cur.execute("DROP VIEW IF EXISTS sample_qc_summary")
        cur.execute("ALTER TABLE samples DROP COLUMN status")
    assert not _has_status(conn)


def test_older_database_gets_samples_status(clean_state, tmp_path):
    L.run_init(L.GVCFS[0], "chr21")
    region = "chr21:10000000-10100000"
    subsets = _subsets(str(tmp_path), region, L.GVCFS, "o")
    L.run_ingest(subsets, region=region)
    conn = L.get_pg_conn()
    try:
        # joint_call.py on a database without the column: the remedy.
        _drop_status(conn)
        jc = L.run_joint_call(str(tmp_path / "o.vcf.gz"), check=False)
        assert jc.returncode == 1
        assert "no status column" in jc.stderr
        assert "mosaic init" in jc.stderr
        assert "Traceback" not in jc.stderr

        # `mosaic init` adds it (then stops at the populated contigs).
        proc = L.run_init(L.GVCFS[0], "chr21", check=False)
        assert proc.returncode != 0
        assert "already contains" in proc.stderr
        assert _has_status(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT status FROM samples")
            assert cur.fetchall() == [("active",)]
            # The CHECK constraint came with it.
            with pytest.raises(Exception):
                cur.execute("UPDATE samples SET status = 'gone'")

        # ensure_ingest_schema adds it...
        _drop_status(conn)
        ensure_ingest_schema(conn)
        assert _has_status(conn)

        # ...so any `mosaic ingest` does (a no-op re-ingest here).
        _drop_status(conn)
        again = L.run_ingest(subsets, region=region)
        assert "already completed" in again.stderr
        assert _has_status(conn)
    finally:
        conn.close()
