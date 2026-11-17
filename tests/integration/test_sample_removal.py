# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""`mosaic remove` + `mosaic aggregate` (sample removal).

* The cohort - R oracle, the central test: the six D2 samples (b0 =
  HG00245 + HG00420, b1 = HG00099 + HG00100, b2 = HG00101 + HG00102) over
  a chr21 slice; remove HG00100 (the second sample of b1) and the whole of
  b2 in one `mosaic remove`; the incremental aggregate. Then a fresh ingest
  of b0 and HG00099 alone. PostgreSQL integers equal row for row (the dead
  sites and alleles gone; gtc too: it is summed in fixed point), qual /
  mqsq_1 within a float32 ULP, the per-sample row counts equal, the
  joint-called VCF byte-identical; --rebuild equals the incremental result
  exactly. Also with the removed sample the FIRST of its batch (HG00099).
  While the removal cohort is live: the gate, the lake oracle in net mode,
  the dense root vs the partials, physical deletion (no data file holds the
  samples; a rewritten file keeps ingest's row-group cap, the
  write.parquet.row-group-limit `mosaic init` sets) and `mosaic maintain`
  reclaiming the replaced files, the genotype matrix / gwas_benchmark /
  sample_qc_summary.
* On a small fixture (the two input_gvcfs4 samples as three 300 kb
  batches; HG00420 is in all three): crash / resume at each
  MOSAIC_TEST_FAIL_REMOVE window equals a clean removal (the negative
  partial's files by URI and ETag, the PG aggregates, the VCF); the
  tombstone takes effect at T0; the refusals; a cross-shard deletion span;
  two chromosomes (the downloaded HG00099 slice, when present); PG gtc ==
  the integer part of the per-sample fixed-point sum.
"""

import gzip
import os

import pyarrow.parquet as pq
import pytest
from pyiceberg.expressions import EqualTo
from pyiceberg.schema import Schema
from pyiceberg.types import ListType, LongType, NestedField

import jc_testlib as L
from mosaic.config import PARQUET_ROW_GROUP_ROWS, PARQUET_SITE_ROW_GROUP_ROWS
from mosaic.genotypes import read_genotype_matrix
from mosaic.register import REGISTER_LOCK_KEY

pytestmark = pytest.mark.integration

REPO = L.REPO_ROOT


def _dl(s):
    return (
        f"{REPO}/test_data/downloads/{s}/{s}.chr21.final.hard-filtered.gvcf.gz"
    )


D2 = {
    "HG00245": L.GVCFS[0],
    "HG00420": L.GVCFS[1],
    "HG00099": _dl("HG00099"),
    "HG00100": _dl("HG00100"),
    "HG00101": _dl("HG00101"),
    "HG00102": _dl("HG00102"),
}
HAVE_D2 = all(os.path.exists(p) for p in D2.values())
# A 7 Mbp slice of the D2 inputs (the measurements use the whole of chr21):
# it holds the star-call worked example (chr21:9,002,912).
SLICE = "chr21:9000000-16000000"
D2_SHARD = 2_000_000
B0 = ["HG00245", "HG00420"]
B1 = ["HG00099", "HG00100"]
B2 = ["HG00101", "HG00102"]
R_CENTRAL = ["HG00100", "HG00101", "HG00102"]

SMALL_REGIONS = [
    "chr21:10000000-10300000",
    "chr21:20000000-20300000",
    "chr21:25000000-25300000",
]
SMALL_SHARD = 100_000
GONE = "HG00420"
PARTIALS = ["batch_ref_blocks", "batch_site_info", "batch_allele_counts"]
SAMPLE_TABLES = ["sample_site_info", "sample_ref_blocks"]


def _subset(work, src, region, tag):
    dst = os.path.join(work, f"{tag}.g.vcf.gz")
    if not os.path.exists(dst):
        L.subset_gvcf(src, region, dst)
    return dst


def _stage_d2(work, batches):
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    for names in batches:
        L.run_ingest(
            [_subset(work, D2[n], SLICE, n) for n in names],
            region=SLICE,
            shard_size=D2_SHARD,
            threads=2,
        )


def _stage_small(work, names=("HG00245", "HG00420")):
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    for i, region in enumerate(SMALL_REGIONS):
        L.run_ingest(
            [_subset(work, D2[n], region, f"{n}_s{i}") for n in names],
            region=region,
            shard_size=SMALL_SHARD,
            threads=2,
        )


def _vcf_text(path):
    with gzip.open(path, "rt") as f:
        return f.read()


def _vcf_samples(path):
    for line in _vcf_text(path).splitlines():
        if line.startswith("#CHROM"):
            return line.split("\t")[9:]
    return []


def _state(work, tag):
    """The comparable state of the store: PG snapshot, per-sample row
    counts, the joint-called VCF text."""
    conn = L.get_pg_conn()
    try:
        pg = L.snapshot_pg(conn)
    finally:
        conn.close()
    catalog = L.get_catalog()
    vcf = os.path.join(work, f"{tag}.vcf.gz")
    L.run_joint_call(vcf)
    return {
        "pg": pg,
        "counts": L.count_iceberg(catalog, SAMPLE_TABLES),
        "vcf": _vcf_text(vcf),
        "vcf_path": vcf,
    }


def _assert_same_cohort(removed, fresh):
    stats = L.compare_pg_snapshots(removed["pg"], fresh["pg"])
    assert removed["counts"] == fresh["counts"]
    assert removed["vcf"] == fresh["vcf"]
    return stats


def _sample_ids():
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT sample_name, id FROM samples")
            return dict(cur.fetchall())
    finally:
        conn.close()


# ── the central test: six D2 samples, remove HG00100 + the whole of b2 ─────


@pytest.fixture(scope="module")
def removed(docker_stack, tmp_path_factory):
    if not HAVE_D2:
        pytest.skip("test_data/downloads (the D2 inputs) not present")
    work = str(tmp_path_factory.mktemp("removal_central"))
    _stage_d2(work, [B0, B1, B2])
    L.run_aggregate(threads=2)
    catalog = L.get_catalog()
    files_before = L.iceberg_data_files(catalog, SAMPLE_TABLES)
    ids = _sample_ids()
    proc = L.run_remove(R_CENTRAL, shard_size=D2_SHARD)
    stale = L.run_joint_call(os.path.join(work, "stale.vcf.gz"), check=False)
    agg = L.run_aggregate(threads=2)
    state = _state(work, "removed")
    return {
        "work": work,
        "remove": proc,
        "stale": stale,
        "aggregate": agg,
        "files_before": files_before,
        "ids": ids,
        **state,
    }


def test_removal_runs_and_applies_incrementally(removed):
    err = removed["remove"].stderr
    assert "fresh" in err and "samples=HG00100,HG00101,HG00102" in err
    for table in ("sample_site_info", "sample_ref_blocks"):
        assert f"delete {table}:" in err
    assert "negative partial: batch_ref_blocks" in err
    # Before the aggregate the cohort is stale: joint_call.py exits 2.
    assert removed["stale"].returncode == 2, removed["stale"].stderr[-2000:]
    banner = removed["aggregate"].stderr
    assert "parent=1 delta=1 removals=1" in banner
    # The finalize dropped the dead sites and alleles.
    dead = [
        int(line.split("(")[1].split(" dead")[0])
        for line in banner.splitlines()
        if "dead root row(s) dropped" in line
        and ("site_info finalize" in line or "allele_counts finalize" in line)
    ]
    assert sum(dead) > 1000
    conn = L.get_pg_conn()
    try:
        rows = L.removal_rows(conn)
        gen = L.cohort_generation(conn)
    finally:
        conn.close()
    assert rows == [(1, sorted(R_CENTRAL), 4, "complete", "complete")]
    assert gen["status"] == "complete" and 4 in gen["batch_ids"]
    assert _vcf_samples(removed["vcf_path"]) == sorted(
        set(D2) - set(R_CENTRAL)
    )


def test_lake_oracle_in_net_mode(removed):
    conn = L.get_pg_conn()
    try:
        stats = L.assert_partials_derivable_from_lake(conn, L.get_catalog())
        L.assert_matches_oracle(conn, L.get_catalog())
    finally:
        conn.close()
    # b0 alone; b1, b2 and the removal batch together.
    assert stats["plain_batches"] == 1 and stats["net_batches"] == 3
    assert stats["dead_sites"] > 1000 and stats["dead_alleles"] > 1000


def test_dense_root_is_the_net_of_the_partials(removed):
    """The dense ref_blocks root of the run that applied the removal is the
    net per-position sum of every partial, negative batch included, with
    the net (0, 0) positions gone; the sparse root (PG) is it at the live
    sites."""
    catalog = L.get_catalog()
    conn = L.get_pg_conn()
    try:
        run = L.latest_agg_run_id(conn)
        pg = L.snapshot_pg(conn)
    finally:
        conn.close()
    dense = L.dense_root(catalog, run)
    partials = (
        catalog.load_table(f"{L.TEST_NAMESPACE}.batch_ref_blocks")
        .scan()
        .to_arrow()
    )
    assert min(partials["dp"].to_pylist()) < 0  # the negative batch
    net = {}
    for c, p, dp, sc in zip(
        partials["chromosome_index"].to_pylist(),
        partials["position"].to_pylist(),
        partials["dp"].to_pylist(),
        partials["sample_counts"].to_pylist(),
    ):
        a = net.get((c, p), (0, 0))
        net[(c, p)] = (a[0] + dp, a[1] + sc)
    want = {k: v for k, v in net.items() if v != (0, 0)}
    assert len(want) < len(net)  # positions only the removed covered
    got = {
        (c, p): (dp, sc)
        for c, p, dp, sc in zip(
            dense["chromosome_index"].to_pylist(),
            dense["position"].to_pylist(),
            dense["dp"].to_pylist(),
            dense["sample_counts"].to_pylist(),
        )
    }
    assert got == want
    sites = set(pg["site_info_int"])
    assert pg["ref_blocks"] == {k: v for k, v in want.items() if k in sites}


def test_no_data_file_holds_a_removed_sample(removed):
    catalog = L.get_catalog()
    ids = [removed["ids"][n] for n in R_CENTRAL]
    for table in SAMPLE_TABLES:
        t = catalog.load_table(f"{L.TEST_NAMESPACE}.{table}")
        for sid in ids:
            # No data file can hold the sample (its statistics prune all).
            scan = t.scan(row_filter=EqualTo("sample_id", sid))
            assert list(scan.plan_files()) == [], (table, sid)
    # b2's file of each table held only HG00101 + HG00102: dropped, not
    # rewritten; b1's held HG00099 too: rewritten; b0's: kept. The line
    # counts the files removed (both) and written (b1's rewrite).
    err = removed["remove"].stderr
    for table in SAMPLE_TABLES:
        assert (
            f"delete {table}: 2 data file(s) removed, 1 written" in err
        ), err[-3000:]
    # The rewritten file (pyiceberg's) keeps ingest's row-group cap: the
    # write.parquet.row-group-limit `mosaic init` set from the same config.
    caps = {
        "sample_ref_blocks": PARQUET_ROW_GROUP_ROWS,
        "sample_site_info": PARQUET_SITE_ROW_GROUP_ROWS,
    }
    now = L.iceberg_data_files(catalog, SAMPLE_TABLES)
    for table in SAMPLE_TABLES:
        t = catalog.load_table(f"{L.TEST_NAMESPACE}.{table}")
        assert t.properties["write.parquet.row-group-limit"] == str(
            caps[table]
        )
        new = [
            u for u in now[table] if u not in removed["files_before"][table]
        ]
        assert len(new) == 1, (table, new)
        with t.io.new_input(new[0]).open() as f:
            meta = pq.ParquetFile(f).metadata
        rows = [meta.row_group(i).num_rows for i in range(meta.num_row_groups)]
        assert sum(rows) > caps[table] and max(rows) <= caps[table], (
            table,
            rows,
        )


def test_rebuild_equals_incremental(removed):
    L.run_aggregate(threads=2, rebuild=True)
    conn = L.get_pg_conn()
    try:
        pg = L.snapshot_pg(conn)
    finally:
        conn.close()
    stats = L.compare_pg_snapshots(removed["pg"], pg)
    # Integers (gtc included) exact; only REAL may move a float32 ULP.
    assert pg["site_info_int"] == removed["pg"]["site_info_int"]
    print(f"\n--rebuild vs incremental: {stats}")


def test_genotype_matrix_gwas_and_sample_qc(removed):
    catalog = L.get_catalog()
    conn = L.get_pg_conn()
    try:
        m = read_genotype_matrix(catalog, conn, 0, 9_000_000, 16_000_000)
        full = read_genotype_matrix(
            catalog, conn, 0, 9_000_000, 16_000_000, include_removed=True
        )
        with conn.cursor() as cur:
            cur.execute("SELECT sample_id FROM sample_qc_summary")
            qc = sorted(r[0] for r in cur.fetchall())
    finally:
        conn.close()
    gone = {removed["ids"][n] for n in R_CENTRAL}
    live = sorted(set(removed["ids"].values()) - gone)
    assert sorted(m.sample_ids.tolist()) == live
    assert sorted(full.sample_ids.tolist()) == sorted(removed["ids"].values())
    assert m.positions.size > 1000 and m.gt_1.shape[1] == len(live)
    assert qc == live
    proc = L._run(
        [
            "scripts/gwas_benchmark.py",
            "--af-cutoff",
            "0.2",
            "--samples",
            *B0,
        ]
    )
    assert "Total samples in DB: 3" in proc.stderr
    assert len(proc.stdout.splitlines()) > 50


def test_maintain_reclaims_the_replaced_files(removed):
    """The data files the deletes dropped or rewrote stay in S3 until
    `mosaic maintain` expires the snapshots that name them."""
    catalog = L.get_catalog()
    now = L.iceberg_data_files(catalog, SAMPLE_TABLES)
    replaced = [
        uri
        for t in SAMPLE_TABLES
        for uri in removed["files_before"][t]
        if uri not in now[t]
    ]
    assert len(replaced) == 4  # b1's and b2's file of each table
    listing = L.bucket_listing()

    def key(uri):
        return uri[len(f"s3://{L.TEST_BUCKET}/") :]

    # The replaced files are ingest uploads (the test bucket); a rewritten
    # file is pyiceberg's, under the table location.
    assert all(key(u) in listing for u in replaced)
    L.run_maintain(older_than_hours=0)
    listing = L.bucket_listing()
    assert not any(key(u) in listing for u in replaced)
    s3 = L.s3_client()
    for uri in (u for t in SAMPLE_TABLES for u in now[t]):
        bucket, k = uri[len("s3://") :].split("/", 1)
        s3.head_object(Bucket=bucket, Key=k)  # still there


@pytest.fixture(scope="module")
def fresh(removed, tmp_path_factory):
    """b0 and HG00099 alone, ingested fresh (after the live-state tests of
    the removal cohort: this resets the stores)."""
    work = str(tmp_path_factory.mktemp("removal_fresh"))
    _stage_d2(work, [B0, ["HG00099"]])
    L.run_aggregate(threads=2)
    return _state(work, "fresh")


def test_cohort_minus_r_equals_a_fresh_ingest(removed, fresh):
    stats = _assert_same_cohort(removed, fresh)
    print(f"\ncohort - R vs fresh: {stats}")
    assert stats["site_info"] > 10_000 and stats["alleles"] > 30_000


def test_removing_the_first_sample_of_a_batch(docker_stack, tmp_path):
    """HG00099 is the first sample of b1: before E3 its batch-mate's rows
    listed its alleles first (the merged list); under per-record lists the
    result equals a fresh ingest exactly as for the second sample."""
    if not HAVE_D2:
        pytest.skip("test_data/downloads (the D2 inputs) not present")
    work = str(tmp_path)
    _stage_d2(work, [B0, B1, B2])
    L.run_aggregate(threads=2)
    L.run_remove(["HG00099"], shard_size=D2_SHARD)
    L.run_aggregate(threads=2)
    removed = _state(work, "removed1")
    _stage_d2(work, [B0, ["HG00100"], B2])
    L.run_aggregate(threads=2)
    fresh = _state(work, "fresh1")
    stats = _assert_same_cohort(removed, fresh)
    print(f"\nfirst sample removed vs fresh: {stats}")


# ── the small fixture: crash / resume, refusals, the tombstone ──────────────


@pytest.fixture(scope="module")
def clean_removal(docker_stack, tmp_path_factory):
    """A clean removal of HG00420 from the three small batches: the
    reference the crash/resume runs must equal."""
    work = str(tmp_path_factory.mktemp("removal_small"))
    _stage_small(work)
    L.run_aggregate(threads=2)
    L.run_remove([GONE], shard_size=SMALL_SHARD)
    files = L.iceberg_file_etags(L.get_catalog(), PARTIALS)
    L.run_aggregate(threads=2)
    return {"work": work, "files": files, **_state(work, "clean")}


def _removal_hook(phase):
    return {"MOSAIC_TEST_FAIL_REMOVE": phase}


@pytest.mark.parametrize(
    "phase", ["pre_register", "post_register", "pre_delete", "post_delete"]
)
def test_crash_and_resume(clean_removal, phase):
    work = clean_removal["work"]
    _stage_small(work)
    L.run_aggregate(threads=2)
    catalog = L.get_catalog()
    sid = _sample_ids()[GONE]
    proc = L.run_remove(
        [GONE],
        shard_size=SMALL_SHARD,
        extra_env=_removal_hook(phase),
        check=False,
    )
    assert proc.returncode == 1, proc.stderr[-2000:]
    assert f"injected crash @{phase}" in proc.stderr
    conn = L.get_pg_conn()
    try:
        ((rid, names, bid, status, bstatus),) = L.removal_rows(conn)
    finally:
        conn.close()
    assert (names, status) == ([GONE], "failed")
    registered = phase != "pre_register"
    t1 = phase in ("pre_delete", "post_delete")
    assert bstatus == ("complete" if t1 else "failed")
    have = L.iceberg_data_files(catalog, PARTIALS)
    want = {
        t: [u for u, _e in rows] for t, rows in clean_removal["files"].items()
    }
    for t in PARTIALS:
        if registered:
            assert have[t] == want[t], t  # each once
        else:
            assert len(have[t]) == len(want[t]) - 1, t  # not the removal's
    rows_left = (
        catalog.load_table(f"{L.TEST_NAMESPACE}.sample_site_info")
        .scan(row_filter=EqualTo("sample_id", sid))
        .to_arrow()
        .num_rows
    )
    assert (rows_left == 0) == (phase == "post_delete")
    # The tombstone holds from T0 whatever failed after it.
    ingest = L.run_ingest(
        [_subset(work, D2[GONE], "chr21:15000000-15100000", "re")],
        region="chr21:15000000-15100000",
        check=False,
    )
    assert ingest.returncode == 1 and "were removed" in ingest.stderr
    gate = L.run_joint_call(
        os.path.join(work, f"gate_{phase}.vcf.gz"), check=False
    )
    if t1:
        # The negative batch is complete and not aggregated: stale.
        assert gate.returncode == 2, gate.stderr[-2000:]
        stale = os.path.join(work, f"stale_{phase}.vcf.gz")
        L.run_joint_call(stale, allow_stale=True)
        assert GONE not in _vcf_samples(stale)
    else:
        # The failed batch is not complete: the gate counts complete
        # batches only, and the tombstone already hides the sample.
        assert gate.returncode == 0, gate.stderr[-2000:]
        assert GONE not in _vcf_samples(
            os.path.join(work, f"gate_{phase}.vcf.gz")
        )

    resumed = L.run_remove([GONE], shard_size=SMALL_SHARD)
    assert "resume at failed" in resumed.stderr
    conn = L.get_pg_conn()
    try:
        assert L.removal_rows(conn) == [
            (rid, [GONE], bid, "complete", "complete")
        ]
    finally:
        conn.close()
    assert L.iceberg_file_etags(catalog, PARTIALS) == clean_removal["files"]
    L.run_aggregate(threads=2)
    state = _state(work, f"resumed_{phase}")
    assert state["pg"] == clean_removal["pg"]  # same order: bit-exact
    assert state["counts"] == clean_removal["counts"]
    assert state["vcf"] == clean_removal["vcf"]


def test_refusals(clean_removal, tmp_path):
    """Each refusal exits 1 and changes nothing."""
    work = clean_removal["work"]
    _stage_small(work)
    L.run_aggregate(threads=2)

    def snapshot():
        conn = L.get_pg_conn()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT sample_name, status FROM samples")
                samples = sorted(cur.fetchall())
            return (L.batch_statuses(conn), L.removal_rows(conn), samples)
        finally:
            conn.close()

    def refused(names, needle, extra_env=None):
        before = snapshot()
        proc = L.run_remove(
            names, shard_size=SMALL_SHARD, extra_env=extra_env, check=False
        )
        assert proc.returncode == 1, proc.stderr[-2000:]
        assert needle in proc.stderr, proc.stderr[-2000:]
        assert "Traceback" not in proc.stderr
        assert snapshot() == before

    refused(["NOSUCH"], "unknown sample(s): NOSUCH")

    # The lock held by another session (`mosaic register` / `maintain`).
    holder = L.get_pg_conn()
    try:
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s)", (REGISTER_LOCK_KEY,))
        refused([GONE], "metadata-writer lock")
    finally:
        with holder.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s)", (REGISTER_LOCK_KEY,))
        holder.close()

    # A sample in an uploaded (deferred, not registered) batch.
    region = "chr21:15000000-15100000"
    L.run_ingest(
        [_subset(work, D2["HG00245"], region, "up")],
        region=region,
        defer_register=True,
    )
    refused(["HG00245"], "is uploaded")
    refused(["HG00245"], "run mosaic register")

    # An unfinished removal of another set.
    crash = L.run_remove(
        [GONE],
        shard_size=SMALL_SHARD,
        extra_env=_removal_hook("pre_register"),
        check=False,
    )
    assert crash.returncode == 1
    L.run_register()
    refused(["HG00245"], f"removal 1 of {GONE} is unfinished")
    # ...finished, the same sample is then already removed.
    L.run_remove([GONE], shard_size=SMALL_SHARD)
    refused([GONE], f"already removed: {GONE}")


def test_pre_e3_lake_refused(clean_state, tmp_path):
    """A lake whose batch_site_info predates E3 (no n_records; its gtc
    list<long> held each batch's truncated sum): `mosaic remove`, `mosaic
    aggregate` and `mosaic ingest` refuse it with the reset remedy."""
    L.run_init(L.GVCFS[0], "chr21")
    catalog = L.get_catalog()
    name = f"{L.TEST_NAMESPACE}.batch_site_info"
    old = catalog.load_table(name).schema()
    fields = []
    for f in old.fields:
        if f.name == "n_records":
            continue
        if f.name == "gtc":
            f = NestedField(
                f.field_id,
                "gtc",
                ListType(
                    element_id=32,
                    element_type=LongType(),
                    element_required=False,
                ),
                required=False,
            )
        fields.append(f)
    catalog.drop_table(name)
    catalog.create_table(name, schema=Schema(*fields))
    for proc in (
        L.run_remove([GONE], check=False),
        L.run_aggregate(check=False),
        L.run_ingest(
            [_subset(str(tmp_path), D2[GONE], SMALL_REGIONS[0], "x")],
            region=SMALL_REGIONS[0],
            check=False,
        ),
    ):
        assert proc.returncode == 1, proc.stderr[-2000:]
        assert "predates sample removal" in proc.stderr, proc.stderr[-2000:]
        assert "batch_site_info lacks n_records" in proc.stderr
        assert "mosaic clean" in proc.stderr


def test_cross_shard_deletion_span(docker_stack, tmp_path):
    """A shard boundary inside a deletion (the E1 fixture: GAAA>G at
    10,001,907 in both samples, a shard edge at 10,001,908): the removed
    sample's negative partial nets out exactly what the lake recomputes for
    it, checked with its rows still in the lake (pre_delete) and after."""
    region = SMALL_REGIONS[0]
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    proc = L.run_ingest(
        [_subset(str(tmp_path), D2[n], region, n) for n in ("HG00245", GONE)],
        region=region,
        shard_size=2_500_477,
        threads=2,
    )
    assert "Added cross-shard deletion-split spans to 2" in proc.stderr
    # Derive windows cut inside the deletion too (the same edge).
    crash = L.run_remove(
        [GONE],
        shard_size=2_500_477,
        extra_env=_removal_hook("pre_delete"),
        check=False,
    )
    assert crash.returncode == 1
    conn = L.get_pg_conn()
    try:
        stats = L.assert_partials_derivable_from_lake(conn, L.get_catalog())
        assert stats["deletion_rows_with_spans"] > 0
        L.run_remove([GONE], shard_size=2_500_477)
        L.assert_partials_derivable_from_lake(conn, L.get_catalog())
    finally:
        conn.close()


HG00099_SLICE = os.path.join(
    REPO,
    "test_data/downloads/HG00099/"
    "HG00099.chr21_1-15M.chr22_10.5-20M.final.hard-filtered.gvcf.gz",
)


@pytest.mark.skipif(
    not os.path.exists(HG00099_SLICE), reason="the HG00099 slice not present"
)
def test_two_chromosomes(docker_stack, tmp_path):
    """HG00099 ingested per chromosome in two batches (chr21 with HG00245,
    chr22 alone) and removed in one `mosaic remove`: both chromosomes' rows
    go and the result equals a fresh ingest of HG00245 alone."""
    work = str(tmp_path)
    r21, r22 = "chr21:10000000-10500000", "chr22:16000000-16500000"
    hg99_21 = _subset(work, HG00099_SLICE, r21, "h21")
    hg99_22 = _subset(work, HG00099_SLICE, r22, "h22")
    hg245 = _subset(work, D2["HG00245"], r21, "g21")

    def init():
        L.reset_state()
        L.run_init(HG00099_SLICE, "chr21,chr22")

    init()
    L.run_ingest([hg99_21, hg245], region=r21, threads=2)
    L.run_ingest([hg99_22], region=r22, threads=2)
    L.run_aggregate(threads=2)
    sid = _sample_ids()["HG00099"]
    L.run_remove(["HG00099"])
    catalog = L.get_catalog()
    for table in SAMPLE_TABLES:
        t = catalog.load_table(f"{L.TEST_NAMESPACE}.{table}")
        assert (
            t.scan(row_filter=EqualTo("sample_id", sid)).to_arrow().num_rows
            == 0
        )
    L.run_aggregate(threads=2)
    conn = L.get_pg_conn()
    try:
        L.assert_partials_derivable_from_lake(conn, catalog)
        with conn.cursor() as cur:
            cur.execute(
                "SELECT count(*) FROM site_info WHERE chromosome_index = 1"
            )
            assert cur.fetchone()[0] == 0  # chr22 had only HG00099
    finally:
        conn.close()
    removed = _state(work, "two_chrom_removed")
    init()
    L.run_ingest([hg245], region=r21, threads=2)
    L.run_aggregate(threads=2)
    _assert_same_cohort(removed, _state(work, "two_chrom_fresh"))


def test_pg_gtc_is_the_fixed_point_sum(clean_state, tmp_path):
    """Without removals, three batches: PG site_info.gtc is the integer part
    of the sum of the per-sample rows' fixed-point gtc, (sum of round(g *
    2**32)) >> 32, exactly at every site (E3)."""
    _stage_small(str(tmp_path))
    L.run_aggregate(threads=2)
    rows = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.sample_site_info")
        .scan()
        .to_arrow()
        .to_pylist()
    )
    sums = {}
    for r in rows:
        k = (r["chromosome_index"], r["position"])
        acc = sums.setdefault(k, [0, 0, 0])
        for j in range(3):
            acc[j] += L.gtc_fixed(r["gtc"][j])
    conn = L.get_pg_conn()
    try:
        pg = L.snapshot_pg(conn)["site_info_int"]
    finally:
        conn.close()
    assert set(pg) == set(sums)
    for k, acc in sums.items():
        assert pg[k][3] == tuple(q >> 32 for q in acc), (k, pg[k][3], acc)
    fractional = sum(1 for acc in sums.values() for q in acc if q % (1 << 32))
    assert len(sums) > 1000 and fractional > 1000
