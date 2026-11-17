# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Ref-block banding (`mosaic ingest --ref-gq-bands`) on the two
input_gvcfs4 samples
over chr21:10,000,000-10,300,000, with 50 kb position windows and 130 kb
shards (no shard edge on a window edge), so merged runs meet both.

* For off, ``20,60`` and ``all`` on the same inputs: under banding the lake
  oracle holds (the partials are derivable from the banded rows); fewer
  sample_ref_blocks rows (``all`` < ``20,60`` < off); every stored block
  lies within one position window and one shard; the genotype matrix and
  sample_qc equal the unbanded ingest's exactly; per sample the banded rows
  cover the same positions with the same sample_counts, and each banded
  row's dp is the rounded length-weighted mean of the unbanded dp it covers
  (inside their [min, max], equal when they agree); after `mosaic
  aggregate`, PG site_info and alleles equal and ref_blocks.sample_counts
  equal at every site (dp moves both ways); the joint-called VCF has the
  same records, the same GT / AC / AN at every record and sample, and its
  differences confined to INFO/DP, QD and the FORMAT DP / AD / PL of
  ref-filled samples.
* ingestion_progress.ref_gq_bands holds the normalized spec (NULL off) and
  the banner prints it.
* Resume: a banded ingest crashed after a shard checkpoint; a re-run with
  another spec, or none, exits 1 before any work naming the stored value;
  a re-run with the same spec completes and its data files equal the clean
  banded run's.
* Removal under banding: two banded batches (the two fixture samples, and
  HG00099 + HG00100 from test_data/downloads when present), `mosaic remove`
  of HG00420, `mosaic aggregate`: equal to a fresh banded ingest of the
  remaining samples (PG, per-sample row counts, the VCF byte for byte).

The default (off) is proven by the rest of the suite and the golden e2e,
which do not pass the flag.
"""

import gzip
import os

import pytest

import jc_testlib as L
from mosaic.genotypes import read_genotype_matrix

pytestmark = pytest.mark.integration

REGION = "chr21:10000000-10300000"
REGION2 = "chr21:20000000-20300000"
LO, HI = 10_000_000, 10_300_000
# Shard edges (0-based starts) at 10,010,000, 10,140,000, 10,270,000; window
# edges at multiples of 50,000.
SHARD_SIZE = 130_000
POS_BATCH = 50_000
# {spec label: the --ref-gq-bands value passed (None: off)}; "20,60" is
# passed un-normalized.
SPECS = {"off": None, "20,60": " 20, 060", "all": "all"}
FAIL_AT = "chr21:10140000"
SAMPLE_TABLES = ["sample_site_info", "sample_ref_blocks"]
# What banding may change in the joint-called VCF.
INFO_MAY_CHANGE = {"DP", "QD"}
FORMAT_MAY_CHANGE = {"DP", "AD", "PL", "GQ"}


def _subsets(work, region, gvcfs, tag):
    out = []
    for j, src in enumerate(gvcfs):
        dst = os.path.join(work, f"{tag}_s{j}.g.vcf.gz")
        if not os.path.exists(dst):
            L.subset_gvcf(src, region, dst)
        out.append(dst)
    return out


def _ingest(gvcfs, spec, region=REGION, **kw):
    return L.run_ingest(
        gvcfs,
        region=region,
        shard_size=SHARD_SIZE,
        pos_batch_size=POS_BATCH,
        threads=2,
        ref_gq_bands=spec,
        **kw,
    )


def _progress():
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT batch_id, status, ref_gq_bands FROM "
                "ingestion_progress ORDER BY batch_id"
            )
            return cur.fetchall()
    finally:
        conn.close()


def _vcf_text(path):
    with gzip.open(path, "rt") as f:
        return f.read()


def _vcf_records(path):
    """(header sample names, {(chrom, pos, ref, alt): fields})."""
    samples, recs = [], {}
    for line in _vcf_text(path).splitlines():
        if line.startswith("#CHROM"):
            samples = line.split("\t")[9:]
        elif not line.startswith("#"):
            f = line.split("\t")
            recs[(f[0], int(f[1]), f[3], f[4])] = f
    return samples, recs


def _info(field):
    out = {}
    for kv in field.split(";"):
        k, _, v = kv.partition("=")
        out[k] = v
    return out


def _call(fmt, value):
    return dict(zip(fmt.split(":"), value.split(":")))


@pytest.fixture(scope="module")
def runs(docker_stack, tmp_path_factory):
    """The same two samples ingested off, ``20,60`` and ``all`` (each on
    a reset store: batch 1 every time), then aggregated and joint-called."""
    work = str(tmp_path_factory.mktemp("ref_banding"))
    gvcfs = _subsets(work, REGION, L.GVCFS, "r")
    out = {}
    for label, spec in SPECS.items():
        L.reset_state()
        L.run_init(L.GVCFS[0], "chr21")
        proc = _ingest(gvcfs, spec)
        catalog = L.get_catalog()
        conn = L.get_pg_conn()
        try:
            stats = L.assert_partials_derivable_from_lake(conn, catalog)
            matrix = L.genotype_matrix_cells(
                read_genotype_matrix(catalog, conn, 0, LO, HI)
            )
            qc = L.sample_qc_rows(conn)
        finally:
            conn.close()
        srb = (
            catalog.load_table(f"{L.TEST_NAMESPACE}.sample_ref_blocks")
            .scan()
            .to_arrow()
        )
        etags = L.iceberg_file_etags(catalog)
        progress = _progress()
        L.run_aggregate(threads=2)
        conn = L.get_pg_conn()
        try:
            pg = L.snapshot_pg(conn)
        finally:
            conn.close()
        vcf = os.path.join(work, f"{label.replace(',', '_')}.vcf.gz")
        L.run_joint_call(vcf)
        out[label] = {
            "stderr": proc.stderr,
            "stats": stats,
            "matrix": matrix,
            "qc": qc,
            "srb": srb,
            "etags": etags,
            "progress": progress,
            "pg": pg,
            "vcf": vcf,
        }
    out["work"] = work
    out["gvcfs"] = gvcfs
    return out


# ── 1. what banding changes, and what it does not ──────────────────────────


@pytest.mark.parametrize("label", ["20,60", "all"])
def test_lake_oracle_holds_on_banded_rows(runs, label):
    stats = runs[label]["stats"]
    assert stats["batches"] == 1 and stats["plain_batches"] == 1
    assert stats["ref_positions"] > 100_000
    # The deletion split is exercised: its parts stay out of the rows.
    assert stats["deletion_rows_with_spans"] > 0
    assert runs["off"]["stats"]["ref_positions"] == stats["ref_positions"]


def test_fewer_rows_the_coarser_the_bands(runs):
    n = {label: runs[label]["srb"].num_rows for label in SPECS}
    print(f"\nsample_ref_blocks rows: {n}")
    assert n["all"] < n["20,60"] < n["off"]
    assert n["off"] > 2 * n["20,60"]


@pytest.mark.parametrize("label", ["off", "20,60", "all"])
def test_every_block_lies_in_one_window_and_one_shard(runs, label):
    srb = runs[label]["srb"]
    start = srb.column("position_start").to_numpy()
    end = srb.column("position_end").to_numpy()
    assert (start <= end).all()
    # Shards (0-based [i * size, (i + 1) * size)) clip every layout.
    assert ((start - 1) // SHARD_SIZE == (end - 1) // SHARD_SIZE).all()
    in_window = start // POS_BATCH == end // POS_BATCH
    if label == "off":
        # Unbanded rows are kept whole: some cross a window edge.
        assert not in_window.all()
    else:
        assert in_window.all()


def test_coverage_sample_counts_and_the_dp_envelope(runs):
    """Per sample, the banded rows cover exactly the unbanded rows'
    positions (each once), with the same sample_counts at every position.
    Each banded row's dp lies within the [min, max] of the unbanded rows'
    dp over its span, equal to it when they all agree; it is exactly the
    length-weighted mean of them, rounded half up (the bander clips its
    parts to the shard, so a row's mean is over the positions it holds)."""

    def per_sample(srb):
        out = {}
        for r in srb.to_pylist():
            d = out.setdefault(r["sample_id"], {})
            for q in range(r["position_start"], r["position_end"] + 1):
                assert q not in d, (r, q)
                d[q] = (r["dp"], r["sample_counts"])
        return out

    base = per_sample(runs["off"]["srb"])
    for label in ("20,60", "all"):
        got = per_sample(runs[label]["srb"])
        assert set(got) == set(base)
        for sid, want in base.items():
            have = got[sid]
            assert set(have) == set(want), (label, sid)
            for q, (_dp, sc) in want.items():
                assert have[q][1] == sc, (label, sid, q)
        moved = 0
        for r in runs[label]["srb"].to_pylist():
            want = base[r["sample_id"]]
            raw = [
                want[q][0]
                for q in range(r["position_start"], r["position_end"] + 1)
            ]
            n, total = len(raw), sum(raw)
            assert min(raw) <= r["dp"] <= max(raw), (label, r, raw)
            if min(raw) == max(raw):
                assert r["dp"] == raw[0], (label, r)
            assert r["dp"] == (total + n // 2) // n, (label, r, raw)
            moved += r["dp"] not in raw
        assert moved > 0, label  # some rows hold a dp no raw block had


@pytest.mark.parametrize("label", ["20,60", "all"])
def test_genotype_matrix_and_sample_qc_unchanged(runs, label):
    assert len(runs["off"]["matrix"]) > 400
    assert runs[label]["matrix"] == runs["off"]["matrix"]
    assert runs["off"]["qc"] and runs[label]["qc"] == runs["off"]["qc"]


@pytest.mark.parametrize("label", ["20,60", "all"])
def test_pg_aggregates(runs, label):
    off, got = runs["off"]["pg"], runs[label]["pg"]
    for key in ("site_info_int", "site_info_float", "alleles"):
        assert got[key] == off[key], key
    lower = higher = 0
    for k in set(off["ref_blocks"]) | set(got["ref_blocks"]):
        a = off["ref_blocks"].get(k, (0, 0))
        g = got["ref_blocks"].get(k, (0, 0))
        assert g[1] == a[1], (k, g, a)  # sample_counts at every site
        lower += g[0] < a[0]
        higher += g[0] > a[0]
    assert lower + higher > 0
    print(
        f"\n{label}: ref_blocks.dp lower at {lower}, higher at {higher} of "
        f"{len(off['ref_blocks'])} sites"
    )


@pytest.mark.parametrize("label", ["20,60", "all"])
def test_joint_called_vcf(runs, label):
    samples, base = _vcf_records(runs["off"]["vcf"])
    got_samples, got = _vcf_records(runs[label]["vcf"])
    assert got_samples == samples and len(samples) == 2
    assert set(got) == set(base)
    assert len(base) > 200
    changed = {"INFO/DP": 0, "FORMAT": 0}
    for key, b in base.items():
        g = got[key]
        assert g[:7] == b[:7], key  # CHROM .. FILTER, QUAL included
        bi, gi = _info(b[7]), _info(g[7])
        assert set(bi) == set(gi), key
        for k in bi:
            if k not in INFO_MAY_CHANGE:
                assert gi[k] == bi[k], (key, k)  # AC, AN, AF, MQ, FS, ...
        if "DP" in bi:
            changed["INFO/DP"] += gi["DP"] != bi["DP"]
        assert g[8] == b[8], key
        for j in range(len(samples)):
            bc, gc = _call(b[8], b[9 + j]), _call(g[8], g[9 + j])
            assert gc["GT"] == bc["GT"], (key, j)
            for k in bc:
                if k not in FORMAT_MAY_CHANGE:
                    assert gc[k] == bc[k], (key, j, k)
            if gc != bc:
                # Only a ref-filled sample's call can change.
                assert bc["GT"] in ("0/0", "0|0"), (key, j, bc, gc)
                changed["FORMAT"] += 1
    print(f"\n{label}: {changed} of {len(base)} records")
    assert changed["INFO/DP"] > 0 and changed["FORMAT"] > 0


# ── 2. provenance ───────────────────────────────────────────────────────────


def test_ingestion_progress_records_the_normalized_spec(runs):
    assert runs["off"]["progress"] == [(1, "complete", None)]
    assert runs["20,60"]["progress"] == [(1, "complete", "20,60")]
    assert runs["all"]["progress"] == [(1, "complete", "all")]
    assert "ref_gq_bands=none" in runs["off"]["stderr"]
    assert "ref_gq_bands=20,60" in runs["20,60"]["stderr"]
    assert "ref_gq_bands=all" in runs["all"]["stderr"]


# ── 3. resume ───────────────────────────────────────────────────────────────


def test_resume_needs_the_same_spec(runs):
    gvcfs = runs["gvcfs"]
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    crash = _ingest(
        gvcfs,
        "20,60",
        extra_env={"MOSAIC_TEST_FAIL_SHARD": FAIL_AT},
        check=False,
    )
    assert crash.returncode != 0
    assert "MOSAIC_TEST_FAIL_SHARD" in crash.stderr
    conn = L.get_pg_conn()
    try:
        shards = L.shard_completion_count(conn)
    finally:
        conn.close()
    assert shards > 0
    before = _progress()
    assert before == [(1, "failed", "20,60")]

    for other in ("all", None, "20"):
        proc = _ingest(gvcfs, other, check=False)
        assert proc.returncode == 1, proc.stderr[-2000:]
        assert "was begun with ref_gq_bands=20,60" in proc.stderr
        assert (
            f"this run has ref_gq_bands={other or 'none'}" in proc.stderr
        ), proc.stderr[-2000:]
        assert "--ref-gq-bands 20,60" in proc.stderr
        assert "Nothing was ingested" in proc.stderr
        assert "Traceback" not in proc.stderr
        assert _progress() == before
        conn = L.get_pg_conn()
        try:
            assert L.shard_completion_count(conn) == shards
        finally:
            conn.close()

    resumed = _ingest(gvcfs, "20,60")
    assert "Resuming batch 1" in resumed.stderr
    assert _progress() == [(1, "complete", "20,60")]
    assert L.iceberg_file_etags(L.get_catalog()) == runs["20,60"]["etags"]


def test_resume_of_an_unbanded_batch_refuses_a_spec(runs):
    gvcfs = runs["gvcfs"]
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    crash = _ingest(
        gvcfs,
        None,
        extra_env={"MOSAIC_TEST_FAIL_SHARD": FAIL_AT},
        check=False,
    )
    assert crash.returncode != 0
    proc = _ingest(gvcfs, "all", check=False)
    assert proc.returncode == 1
    assert "was begun with ref_gq_bands=none" in proc.stderr
    assert "Resume it without --ref-gq-bands" in proc.stderr
    assert _progress() == [(1, "failed", None)]
    _ingest(gvcfs, None)
    assert L.iceberg_file_etags(L.get_catalog()) == runs["off"]["etags"]


# ── 4. removal under banding ────────────────────────────────────────────────


def _dl(name):
    return os.path.join(
        L.REPO_ROOT,
        f"test_data/downloads/{name}/{name}.chr21.final.hard-filtered.gvcf.gz",
    )


D2 = [_dl("HG00099"), _dl("HG00100")]


def _state(work, tag):
    conn = L.get_pg_conn()
    try:
        pg = L.snapshot_pg(conn)
    finally:
        conn.close()
    vcf = os.path.join(work, f"{tag}.vcf.gz")
    L.run_joint_call(vcf)
    return {
        "pg": pg,
        "counts": L.count_iceberg(L.get_catalog(), SAMPLE_TABLES),
        "vcf": _vcf_text(vcf),
    }


@pytest.mark.skipif(
    not all(os.path.exists(p) for p in D2),
    reason="test_data/downloads (HG00099, HG00100) not present",
)
def test_removal_under_banding_equals_a_fresh_banded_ingest(runs):
    """b0 = HG00245 + HG00420 and b1 = HG00099 + HG00100, both banded
    (20,60); remove HG00420 and aggregate; equals a fresh banded ingest of
    HG00245 and b1: the negative partial, re-derived from the banded rows,
    cancels exactly what the banded ingest summed."""
    work = runs["work"]
    b0 = runs["gvcfs"]
    b1 = _subsets(work, REGION, D2, "d2")
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    _ingest(b0, "20,60")
    _ingest(b1, "20,60")
    L.run_aggregate(threads=2)
    L.run_remove(["HG00420"], shard_size=SHARD_SIZE)
    L.run_aggregate(threads=2)
    conn = L.get_pg_conn()
    try:
        stats = L.assert_partials_derivable_from_lake(conn, L.get_catalog())
    finally:
        conn.close()
    # b0 with the removal batch; b1 alone.
    assert stats["net_batches"] == 2 and stats["plain_batches"] == 1
    removed = _state(work, "banded_removed")

    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    _ingest(b0[:1], "20,60")
    _ingest(b1, "20,60")
    L.run_aggregate(threads=2)
    fresh = _state(work, "banded_fresh")

    counts = L.compare_pg_snapshots(removed["pg"], fresh["pg"])
    print(f"\nbanded cohort - HG00420 vs fresh: {counts}")
    assert counts["ref_blocks"] > 100
    assert removed["counts"] == fresh["counts"]
    assert removed["vcf"] == fresh["vcf"]
