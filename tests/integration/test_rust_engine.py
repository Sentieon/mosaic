# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The Rust ingest worker (``mosaic-ingest-worker``) against its frozen
signatures (earlier against the Python worker, which is how the
signatures were recorded: each only after the two workers' comparison
passed in the same test).

1. The one-shard row oracle: the binary run from a manifest on one shard
   (every fixture set, both window settings, banding, ``pos_batch_size``
   0, the 150-ALT site, the cross-shard span, one worker over four
   shards): its local files (names, schemas, slice indexes, every column's
   values with floats as bits) and its result dicts against
   ``tests/golden/ingest_signatures.json`` (``row_oracle/...``).
2. Three 2-sample batches at the CLI defaults, the workflow's settings and
   ``--ref-gq-bands 20,60``: the six tables' data files
   (``parquet_file_signature``), sample_qc and the banner lines against
   their signatures (``batches/...``); the lake oracle; the joint-called VCF
   against its joint-call golden.
3. Resume and refusals: a crashed batch resumes to the clean run's files; a
   batch begun under the removed Python worker is refused; the missing
   binary writes nothing; ``MOSAIC_INGEST_ENGINE`` is read nowhere.
4. Threads and the crash paths (R3): ``--threads`` 1 / 2 / 4 give the
   same data files and the same per-shard lines (and those against their
   signature); ``MOSAIC_TEST_FAIL_SHARD``, a SIGKILL of the worker binary
   and ``MOSAIC_TEST_FAIL_RUST_SHARD`` fail the batch and resume to the
   clean run's files; the binary's output cut short (missing shards, a
   truncated line) fails the batch.
5. The combine in the worker (R5): the seam's combine form and a SIGKILL
   of the worker during the combine fail the batch and resume to the clean
   run's files (ETags: byte identity is the same-worker contract).

The tests that need the binary skip with a visible reason when it is
absent (``cargo build --release`` in ``rust/``).
"""

import json
import os
import re
import struct
import subprocess
import sys
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import pyarrow.parquet as pq  # type: ignore[import-untyped]
import pytest

import jc_testlib as L
import sample_slices
from mosaic import ingest
from mosaic.ref_banding import RefBandSpec

try:
    from vcflib import vcf  # type: ignore[import-untyped]
except ImportError:  # pragma: no cover
    vcf = None

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not L.rust_worker_available(), reason=L.RUST_WORKER_SKIP_REASON
    ),
]

TD = os.path.join(L.REPO_ROOT, "test_data")
GVCFS4 = L.GVCFS
DOWNLOADS = [
    os.path.join(
        TD,
        "downloads/HG00099/"
        "HG00099.chr21_1-15M.chr22_10.5-20M.final.hard-filtered.gvcf.gz",
    ),
] + [
    os.path.join(TD, f"downloads/{s}/{s}.chr21.final.hard-filtered.gvcf.gz")
    for s in ("HG00100", "HG00101", "HG00102")
]
DOWNLOADS_CHR21 = [
    os.path.join(TD, f"downloads/{s}/{s}.chr21.final.hard-filtered.gvcf.gz")
    for s in ("HG00099", "HG00100", "HG00101", "HG00102")
]
OVERLAPPED = [
    os.path.join(
        TD,
        f"overlapped_deletions/{s}.final.hard-filtered_chr21_overlapping_"
        "no-ambig.gvcf.gz",
    )
    for s in ("HG00245", "HG00420")
]
GVCFS5 = [
    os.path.join(TD, f"input_gvcfs5/{s}.chr21.final.hard-filtered.gvcf.gz")
    for s in ("HG00096", "HG00097")
]

BATCH_ID = 7


# ── the one-shard row oracle ────────────────────────────────────────────────


def _contig_map(path: str) -> Dict[str, int]:
    vf = vcf.VCF(path)
    return {c: i for i, c in enumerate(vf.contigs)}


def _inputs(
    paths: List[str],
) -> Tuple[List[Any], List[Optional[int]], List[str]]:
    vfs = [vcf.VCF(p) for p in paths]
    sample_ids: List[Optional[int]] = [
        (k + 1) if vf.samples else None for k, vf in enumerate(vfs)
    ]
    names = [vf.samples[0] for vf, s in zip(vfs, sample_ids) if s is not None]
    return vfs, sample_ids, names


def run_rust_shards(
    paths: List[str],
    shards: List[Tuple[str, int, int]],
    pos_batch_size: int,
    ref_gq_bands: Optional[str],
    tmpdir: str,
    committed: frozenset = frozenset(),
    extra_args: Optional[List[str]] = None,
) -> List[Dict[str, Any]]:
    """The binary on the shards, through the production manifest and
    delivery code (results in shard order)."""
    vfs, sample_ids, names = _inputs(paths)
    manifest = ingest.rust_manifest(
        BATCH_ID,
        paths,
        vfs,
        sample_ids,
        _contig_map(paths[0]),
        sorted({s for s in sample_ids if s is not None}),
        shards,
        pos_batch_size,
        ref_gq_bands,
        tmpdir,
        committed,
        names,
    )
    mpath = os.path.join(tmpdir, "manifest.json")
    with open(mpath, "w") as fh:
        json.dump(manifest, fh)
    worker = L.rust_worker_path()
    assert worker is not None
    if extra_args:
        proc = subprocess.run(
            [worker, "--manifest", mpath, *extra_args],
            capture_output=True,
            text=True,
            check=True,
        )
        out = []
        for line in proc.stdout.splitlines():
            out.append(ingest.parse_rust_result(line)[1])
        return out

    def collect(results, rv=None):
        if rv is None:
            return []
        results.append(rv)
        return results

    return ingest.run_rust_shards(worker, mpath, 1, len(shards), collect)


def _canon(x: Any) -> Any:
    """Floats as their IEEE 754 bits (so -0.0, NaN and every ULP count)."""
    if isinstance(x, float):
        return ("f64", struct.pack("<d", x))
    if isinstance(x, list):
        return [_canon(v) for v in x]
    if isinstance(x, tuple):
        return tuple(_canon(v) for v in x)
    if isinstance(x, dict):
        return {k: _canon(v) for k, v in x.items()}
    return x


PATH_KEYS = (
    "rb_paths",
    "si_paths",
    "brb_paths",
    "bsi_paths",
    "bac_paths",
    "bgt_paths",
)


def row_oracle(
    paths: List[str],
    shard: Tuple[str, int, int],
    pos_batch_size: int,
    keep: str,
    ref_gq_bands: Optional[str] = None,
    committed: frozenset = frozenset(),
    check: Optional[Any] = None,
) -> Dict[str, Any]:
    """Run the binary on one shard into a fresh temp dir; copy its local
    files into ``keep`` and return its result. ``check(rs)`` runs while
    the files exist."""
    with tempfile.TemporaryDirectory() as b:
        (rs,) = run_rust_shards(
            paths, [shard], pos_batch_size, ref_gq_bands, b, committed
        )
        _keep_rust_files(rs, keep)
        if check is not None:
            check(rs)
        return rs


# ── the frozen signatures ────────────────────────────────────────────────────
#
# The worker's result against its recorded signature
# (tests/golden/ingest_signatures.json, jc_testlib.assert_ingest_signature),
# recorded in K6 stage A while the Python worker gave the same rows.


def _keep_rust_files(rs: Dict[str, Any], keep: str) -> None:
    """Copy the Rust worker's local files of one shard into ``keep`` (they
    are deleted with the row oracle's temp dirs)."""
    import shutil

    os.makedirs(keep, exist_ok=True)
    for key in PATH_KEYS:
        for p in rs[key]:
            shutil.copyfile(p, os.path.join(keep, os.path.basename(p)))


def _local_table(path: str) -> Tuple[Any, Any]:
    """(pyarrow Table, slice meta) of a local file."""
    if path.endswith(".slices"):
        index = sample_slices.read_sample_slice_index(path)
        meta = [[s.chromosome_index, s.sample_id, s.num_rows] for s in index]
        return sample_slices.read_sample_slices(path), meta
    table = pq.read_table(path)
    return table, [table.num_rows]


def shard_signature(
    rs: Dict[str, Any], keep: str
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """(detail, counts) of one shard's Rust result whose local files are
    in ``keep``: the result dict (paths as file names) and per file its
    schema string, slice index and every column's digest (values with
    floats as bits, nulls kept); the counts are files and rows per kind."""
    result = {k: v for k, v in rs.items() if k not in PATH_KEYS}
    files, counts = {}, {}
    for key in PATH_KEYS:
        names = [os.path.basename(p) for p in rs[key]]
        result[key] = names
        rows = 0
        for name in names:
            table, meta = _local_table(os.path.join(keep, name))
            rows += table.num_rows
            files[name] = {
                "schema": str(table.schema),
                "slices": meta,
                "columns": L.table_column_digests(table),
            }
        counts[key] = [len(names), rows]
    return {"result": result, "files": files}, counts


def _rows_against_copy(keep: str) -> Any:
    """A mismatch's first differing rows: each file of the output (in
    ``keep``) against the golden copy's file of that name."""

    def compare(golden: Any, cdir: str) -> List[str]:
        out = []
        names = sorted(set(os.listdir(keep)) | set(os.listdir(cdir)))
        for name in names:
            if name == "detail.json.gz":
                continue
            a, b = os.path.join(cdir, name), os.path.join(keep, name)
            if not (os.path.exists(a) and os.path.exists(b)):
                out.append(
                    f"  {name}: only in the "
                    f"{'golden' if os.path.exists(a) else 'output'}"
                )
                continue
            (ta, ma), (tb, mb) = _local_table(a), _local_table(b)
            if str(ta.schema) != str(tb.schema):
                out.append(f"  {name}: schema {ta.schema} != {tb.schema}")
            if ma != mb:
                out.append(f"  {name}: slices {ma[:5]} != {mb[:5]}")
            for col in ta.schema.names:
                va = _canon(ta.column(col).to_pylist())
                vb = (
                    _canon(tb.column(col).to_pylist())
                    if (col in tb.schema.names)
                    else []
                )
                if va != vb:
                    i = next(
                        (i for i, (x, y) in enumerate(zip(va, vb)) if x != y),
                        min(len(va), len(vb)),
                    )
                    out.append(
                        f"  {name}: column {col} row {i} ({len(va)} vs "
                        f"{len(vb)} rows): golden "
                        f"{va[i] if i < len(va) else None!r} != output "
                        f"{vb[i] if i < len(vb) else None!r}"
                    )
        return out

    return compare


def assert_shard_signature(case: str, rs: Dict[str, Any], keep: str) -> None:
    detail, counts = shard_signature(rs, keep)
    L.assert_ingest_signature(
        case,
        detail,
        counts,
        files={n: os.path.join(keep, n) for n in os.listdir(keep)},
        reference=L.INGEST_WORKER_REFERENCE
        + " (the one-shard row oracle, test_rust_engine.py::row_oracle)",
        compare=_rows_against_copy(keep),
    )


@pytest.fixture(scope="module")
def synth150(tmp_path_factory):
    """The 150-ALT synthetic gVCF of test_genotype_matrix_wide.py and its
    real slice."""
    import test_genotype_matrix_wide as W

    work = str(tmp_path_factory.mktemp("rust_wide"))
    real = os.path.join(work, "real.g.vcf.gz")
    L.subset_gvcf(L.GVCFS[0], W.REGION, real)
    synth = os.path.join(work, "synth.g.vcf.gz")
    W._write_synthetic(L.GVCFS[0], W._hom_ref_block_start(real), synth)
    return [real, synth]


SHARD_500K = ("chr21", 10_000_000, 10_500_000)
# A 130 kb shard starting inside the span of the two input_gvcfs4 samples'
# deletion GAAA>G at 10,001,907 (test_removal_substrate_lake.py's
# CROSS_SHARD_SIZE boundary): its deletion-split spans belong to the
# previous shard's row (foreign_del_spans).
SHARD_130K = ("chr21", 10_001_908, 10_131_908)

# (id, inputs, shard, pos_batch_size, ref_gq_bands)
ORACLE_CASES = [
    (
        "gvcfs4-defaults",
        GVCFS4,
        ("chr21", 10_000_000, 20_000_000),
        10**7,
        None,
    ),
    ("gvcfs4-500k", GVCFS4, SHARD_500K, 100_000, None),
    ("gvcfs4-130k", GVCFS4, SHARD_130K, 50_000, None),
    ("gvcfs4-pbs0", GVCFS4, SHARD_500K, 0, None),
    ("gvcfs4-bands20,60", GVCFS4, SHARD_500K, 100_000, "20,60"),
    ("gvcfs4-bandsall", GVCFS4, SHARD_500K, 100_000, "all"),
    ("gvcfs4-130k-bands20,60", GVCFS4, SHARD_130K, 50_000, "20,60"),
    ("gvcfs4-pbs0-bandsall", GVCFS4, SHARD_500K, 0, "all"),
    (
        "downloads-chr22",
        DOWNLOADS,
        ("chr22", 10_500_000, 11_500_000),
        100_000,
        None,
    ),
    (
        "downloads-chr21",
        DOWNLOADS,
        ("chr21", 10_000_000, 12_000_000),
        10**7,
        None,
    ),
    ("downloads-chr21-bands", DOWNLOADS_CHR21, SHARD_500K, 100_000, "20,60"),
    (
        "overlapped",
        OVERLAPPED,
        ("chr21", 10_000_000, 10_500_000),
        100_000,
        None,
    ),
    (
        "overlapped-pbs0",
        OVERLAPPED,
        ("chr21", 10_000_000, 20_000_000),
        0,
        None,
    ),
    ("gvcfs5", GVCFS5, ("chr21", 10_000_000, 11_000_000), 100_000, None),
    (
        "gvcfs5-bandsall",
        GVCFS5,
        ("chr21", 10_000_000, 10_500_000),
        50_000,
        "all",
    ),
]


@pytest.mark.parametrize(
    "paths,shard,pbs,bands",
    [c[1:] for c in ORACLE_CASES],
    ids=[c[0] for c in ORACLE_CASES],
)
def test_row_oracle(paths, shard, pbs, bands, request, tmp_path):
    for p in paths:
        if not os.path.exists(p):
            pytest.skip(f"fixture {p} not present")
    if bands is not None:
        bands = RefBandSpec.parse(bands).text
    keep = str(tmp_path / "rust")
    rs = row_oracle(paths, shard, pbs, keep, bands)
    assert rs["group_count"] > 0 or "overlapped" in shard[0]
    case = request.node.callspec.id
    assert_shard_signature(f"row_oracle/{case}", rs, keep)


def test_row_oracle_cross_shard_span(tmp_path):
    """A 130 kb shard whose deletion-split spans belong to a deletion row
    of the previous shard (foreign_del_spans), and the reprocessed
    label."""
    keep = str(tmp_path / "rust")
    rs = row_oracle(
        GVCFS4,
        SHARD_130K,
        50_000,
        keep,
        committed=frozenset({SHARD_130K[:2]}),
    )
    assert rs["foreign_del_spans"], "the shard has no cross-shard span"
    assert_shard_signature("row_oracle/cross_shard_span", rs, keep)


def test_row_oracle_wide(synth150, tmp_path):
    """The 150-ALT site: its batch_genotypes row two bytes per slot."""
    widths = []
    keep = str(tmp_path / "rust")

    def check(rs):
        t = pq.read_table(rs["bgt_paths"][0])
        for blob, n in zip(
            t.column("gt_1").to_pylist(), t.column("n_samples").to_pylist()
        ):
            widths.append(len(blob) // n)

    rs = row_oracle(
        synth150, ("chr21", 9_000_000, 9_500_000), 100_000, keep, check=check
    )
    assert widths.count(2) == 1 and set(widths) == {1, 2}
    assert_shard_signature("row_oracle/wide", rs, keep)


def test_row_oracle_multi_shard_run(tmp_path):
    """One worker process over consecutive shards (the readers re-seeked
    per shard): each shard's result against its signature (recorded while
    it equalled the Python worker's run of that shard alone)."""
    shards = [
        ("chr21", 10_000_000 + i * 130_000, 10_000_000 + (i + 1) * 130_000)
        for i in range(4)
    ]
    with tempfile.TemporaryDirectory() as b:
        rs = run_rust_shards(GVCFS4, shards, 50_000, None, b)
        for i, r in enumerate(rs):
            keep = str(tmp_path / f"rust{i}")
            _keep_rust_files(r, keep)
            assert_shard_signature(f"row_oracle/multi_shard_run/{i}", r, keep)


def test_timing_flag_prints_buckets():
    with tempfile.TemporaryDirectory() as b:
        vfs, sample_ids, names = _inputs(GVCFS4)
        manifest = ingest.rust_manifest(
            BATCH_ID,
            GVCFS4,
            vfs,
            sample_ids,
            _contig_map(GVCFS4[0]),
            sorted(s for s in sample_ids if s is not None),
            [SHARD_130K],
            50_000,
            None,
            b,
            frozenset(),
            names,
        )
        mpath = os.path.join(b, "manifest.json")
        with open(mpath, "w") as fh:
            json.dump(manifest, fh)
        proc = subprocess.run(
            [L.rust_worker_path(), "--manifest", mpath, "--timing"],
            capture_output=True,
            text=True,
            check=True,
        )
        assert "dense-brb-write" in proc.stderr
        assert len(proc.stdout.splitlines()) == 1


# ── batches: S3 files, PG, downstream ───────────────────────────────────────


def _d2_batches() -> List[List[str]]:
    """The three 2-sample batches of the six chr21 samples (the D2 inputs):
    b0 = HG00245 + HG00420 (input_gvcfs4), b1 = HG00099 + HG00100, b2 =
    HG00101 + HG00102."""
    return [GVCFS4, DOWNLOADS_CHR21[:2], DOWNLOADS_CHR21[2:]]


def _ingest_all(
    batches: List[List[str]],
    shard_size: Optional[int] = None,
    pos_batch_size: Optional[int] = None,
    ref_gq_bands: Optional[str] = None,
) -> Dict[str, Any]:
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    stderr = []
    for gv in batches:
        p = L.run_ingest(
            gv,
            region="chr21",
            threads=2,
            shard_size=shard_size,
            pos_batch_size=pos_batch_size,
            ref_gq_bands=ref_gq_bands,
        )
        stderr.append(p.stderr)
    conn = L.get_pg_conn()
    try:
        qc = L.sample_qc_rows(conn)
        with conn.cursor() as cur:
            cur.execute("SELECT batch_id, engine FROM ingestion_progress")
            engines = dict(cur.fetchall())
    finally:
        conn.close()
    return {
        "etags": L.iceberg_file_etags(L.get_catalog()),
        "qc": qc,
        "engines": engines,
        "done": [_line(s, "Done. ") for s in stderr],
        "spans": [_line(s, "  Added cross-shard") for s in stderr],
    }


def _line(text: str, prefix: str) -> Optional[str]:
    for line in text.splitlines():
        if line.startswith(prefix):
            return line
    return None


def _downstream(tmp: str, name: str) -> Dict[str, Any]:
    L.run_aggregate()
    conn = L.get_pg_conn()
    try:
        snap = L.snapshot_pg(conn)
        L.assert_partials_derivable_from_lake(conn, L.get_catalog())
    finally:
        conn.close()
    vcf_path = os.path.join(tmp, f"{name}.vcf.gz")
    # The uncapped output, the golden's (K3: the cap's default is 6).
    proc = L.run_joint_call(vcf_path, max_alt_alleles=0)
    with open(vcf_path, "rb") as fh:
        vcf_bytes = fh.read()
    return {
        "pg": snap,
        "vcf": vcf_bytes,
        "vcf_path": vcf_path,
        "jc_stderr": proc.stderr,
    }


# The joint-call golden each case's VCF is checked against (K6, decision 8):
# the CLI defaults and the workflow's settings give the six-sample cohort's
# golden; banding moves INFO/DP and the ref-filled FORMAT values, so its
# own scenario, recorded while both ingest workers' lakes gave this VCF.
BATCHES_VCF_GOLDEN = {
    "defaults": "fresh_cohort_chr21",
    "workflow": "fresh_cohort_chr21",
    "bands20,60": "six_samples_bands20_60",
}


@pytest.mark.parametrize(
    "shard_size,pos_batch_size,bands",
    [(None, None, None), (500_000, 100_000, None), (None, None, "20,60")],
    ids=["defaults", "workflow", "bands20,60"],
)
def test_batches_row_for_row(
    docker_stack, tmp_path, request, shard_size, pos_batch_size, bands
):
    """The three 2-sample batches: the data files (URIs; per file the
    schema string, rows with floats as bits, row groups, sort declaration,
    statistics columns), sample_qc and the Done / cross-shard lines against
    the case's frozen signature (recorded in K6 stage A while the Python
    worker's batches were equal row for row); the lake oracle; the
    joint-called VCF against its joint-call golden (the defaults and the
    workflow's settings: the six-sample cohort's; banding: its own
    scenario, recorded while both workers' lakes gave it). PostgreSQL has
    no frozen form (its REAL sums are order-dependent at the last ULP): the
    aggregate oracle and the VCF's INFO fields cover it."""
    for p in _d2_batches()[1] + _d2_batches()[2]:
        if not os.path.exists(p):
            pytest.skip(f"fixture {p} not present")
    b = _ingest_all(_d2_batches(), shard_size, pos_batch_size, bands)
    b["files"] = L.iceberg_file_signatures(L.get_catalog(), str(tmp_path))
    b.update(_downstream(str(tmp_path), "rust"))
    assert set(b["engines"].values()) == {"rust"}
    case = request.node.callspec.id
    L.assert_golden(BATCHES_VCF_GOLDEN[case], b["vcf_path"], b["jc_stderr"])
    assert_batches_signature(f"batches/{case}", b)


def assert_batches_signature(case: str, b: Dict[str, Any]) -> None:
    """The Rust batches' frozen signature: per table and URI the data
    file's ``parquet_file_signature``, the sample_qc rows and the Done /
    cross-shard banner lines."""
    detail = {
        "files": b["files"],
        "qc": b["qc"],
        "done": b["done"],
        "spans": b["spans"],
    }
    counts = {
        t: [len(fs), sum(sig["num_rows"] for sig in fs.values())]
        for t, fs in sorted(b["files"].items())
    }

    def compare(golden: Any, cdir: str) -> List[str]:
        cur = L.canonical(detail)
        out = L.compare_iceberg_signatures(golden["files"], cur["files"])
        for key in ("qc", "done", "spans"):
            if golden[key] != cur[key]:
                out += L._first_detail_differences(golden[key], cur[key], key)
        return out

    L.assert_ingest_signature(
        case,
        detail,
        counts,
        reference=L.INGEST_WORKER_REFERENCE
        + " (three 2-sample batches row for row, "
        "test_rust_engine.py::test_batches_row_for_row)",
        compare=compare,
    )


# ── resume and refusals ─────────────────────────────────────────────────────

CRASH_SHARD = {"MOSAIC_TEST_FAIL_SHARD": "chr21:10000000"}


def test_rust_crash_resume_equals_clean_rust_run(docker_stack):
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    L.run_ingest(GVCFS4, region="chr21")
    clean = L.iceberg_file_etags(L.get_catalog())

    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    p = L.run_ingest(
        GVCFS4,
        region="chr21",
        extra_env=CRASH_SHARD,
        check=False,
    )
    assert p.returncode != 0
    assert "MOSAIC_TEST_FAIL_SHARD" in p.stderr
    assert L.batch_statuses(L.get_pg_conn()) == [(1, "failed")]
    p = L.run_ingest(GVCFS4, region="chr21")
    assert "Resuming batch 1" in p.stderr
    assert "reprocessed for Iceberg" in p.stderr
    assert L.iceberg_file_etags(L.get_catalog()) == clean


@pytest.mark.parametrize("stored", ["python", None])
def test_resume_of_a_python_worker_batch_is_refused(docker_stack, stored):
    """A batch begun under the Python worker (removed in K6; its stored
    engine ``python``, or NULL on a row older than the column) cannot be
    resumed: exit 1 before any work, the batch, the stored engine and the
    remedy named, nothing changed."""
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    p = L.run_ingest(
        GVCFS4, region="chr21", extra_env=CRASH_SHARD, check=False
    )
    assert p.returncode != 0
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("UPDATE ingestion_progress SET engine = %s", (stored,))
        conn.commit()
        before = (L.batch_statuses(conn), L.shard_completion_count(conn))
    finally:
        conn.close()
    etags = L.iceberg_file_etags(L.get_catalog())
    p = L.run_ingest(GVCFS4, region="chr21", check=False)
    assert p.returncode == 1
    assert "Error: batch 1 (was 'failed'" in p.stderr
    assert "was begun with engine=python, the Python ingest worker" in (
        p.stderr
    )
    assert "`mosaic ingest --engine python`" in p.stderr
    assert "mosaic clean, mosaic init, and ingest again" in p.stderr
    assert "Nothing was ingested." in p.stderr
    conn = L.get_pg_conn()
    try:
        assert (
            L.batch_statuses(conn),
            L.shard_completion_count(conn),
        ) == before
        with conn.cursor() as cur:
            cur.execute("SELECT engine FROM ingestion_progress")
            assert cur.fetchall() == [(stored,)]
    finally:
        conn.close()
    assert L.iceberg_file_etags(L.get_catalog()) == etags


def test_missing_binary_writes_nothing(docker_stack, tmp_path):
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    env = {
        "MOSAIC_INGEST_WORKER": "/nonexistent",
        "PATH": "/usr/bin:/bin",
    }
    p = L.run_ingest(GVCFS4, region="chr21", extra_env=env, check=False)
    assert p.returncode == 1
    assert "mosaic-ingest-worker" in p.stderr
    assert "Nothing was ingested" in p.stderr
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM ingestion_progress")
            assert cur.fetchone()[0] == 0
            cur.execute("SELECT count(*) FROM samples")
            assert cur.fetchone()[0] == 0
    finally:
        conn.close()


def test_the_engine_variable_is_read_nowhere(docker_stack):
    """``MOSAIC_INGEST_ENGINE`` (removed in K6) selects nothing: with it
    set to ``python`` the batch runs the worker and records ``rust``."""
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    p = L.run_ingest(
        GVCFS4,
        region="chr21:10000000-10500000",
        extra_env={"MOSAIC_INGEST_ENGINE": "python"},
    )
    assert "engine=rust" in p.stderr
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT engine FROM ingestion_progress")
            assert cur.fetchall() == [("rust",)]
    finally:
        conn.close()


# ── R3: threads in the worker, the crash paths ─────────────────────────────
#
# input_gvcfs4 (chr21:1-30M) under --region chr21 at 2 Mbp shards: 24
# shards (--region restricts sharding to the chromosome), 14 of them with
# data, several per thread at --threads 4.

R3_REGION = "chr21"
R3_SHARD = 2_000_000
R3_SHARDS = 24
_SHARD_LINE = re.compile(
    r"^  Shard (\S+): (\d+) groups, (\d+) variants, (\d+) ref-blocks in "
    r"[0-9.]+s \(stage [0-9.]+s\)(.*)$"
)


def _r3_ingest(threads: int, **kw: Any) -> subprocess.CompletedProcess:
    return L.run_ingest(
        GVCFS4,
        region=R3_REGION,
        shard_size=R3_SHARD,
        threads=threads,
        **kw,
    )


def _shard_lines(stderr: str) -> List[Tuple[str, ...]]:
    """The per-shard lines without their seconds, sorted."""
    out = []
    for line in stderr.splitlines():
        m = _SHARD_LINE.match(line)
        if m:
            out.append(m.groups())
        else:
            assert not line.startswith("  Shard "), line
    return sorted(out)


@pytest.fixture(scope="module")
def r3_clean(docker_stack):
    """A clean Rust ingest of the R3 batch at --threads 2: its data files
    and its per-shard lines."""
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    p = _r3_ingest(2)
    assert f"Created {R3_SHARDS} shard(s)" in p.stderr
    return {
        "etags": L.iceberg_file_etags(L.get_catalog()),
        "lines": _shard_lines(p.stderr),
    }


@pytest.mark.parametrize("threads", [1, 4])
def test_threads_give_identical_files(r3_clean, threads):
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    p = _r3_ingest(threads)
    assert f"using {threads} thread(s)" in p.stderr
    assert L.iceberg_file_etags(L.get_catalog()) == r3_clean["etags"]
    assert _shard_lines(p.stderr) == r3_clean["lines"]


def test_shard_lines_signature(r3_clean):
    """The clean run's per-shard lines (without their seconds) against
    their signature (recorded while the Python worker printed the same
    lines)."""
    assert len(r3_clean["lines"]) == 14  # the shards with data
    L.assert_ingest_signature(
        "shard_lines/gvcfs4_chr21_2mb",
        r3_clean["lines"],
        {"lines": len(r3_clean["lines"])},
    )


def test_rust_crash_resume_threads(r3_clean):
    """MOSAIC_TEST_FAIL_SHARD at --threads 4, resumed at --threads 2: the
    clean run's files."""
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    p = _r3_ingest(
        4,
        extra_env={"MOSAIC_TEST_FAIL_SHARD": "chr21:6000000"},
        check=False,
    )
    assert p.returncode != 0
    assert "MOSAIC_TEST_FAIL_SHARD" in p.stderr
    conn = L.get_pg_conn()
    try:
        assert L.batch_statuses(conn) == [(1, "failed")]
        assert L.shard_completion_count(conn) == 4
    finally:
        conn.close()
    p = _r3_ingest(2)
    assert "Resuming batch 1" in p.stderr
    # Shards 0 and 1 (chr21:0-4M) hold no record and print no line.
    assert p.stderr.count("reprocessed for Iceberg") == 2
    assert L.iceberg_file_etags(L.get_catalog()) == r3_clean["etags"]


def _worker_pids(parent: int) -> List[int]:
    p = subprocess.run(
        ["pgrep", "-P", str(parent), "-f", "mosaic-ingest-worker"],
        capture_output=True,
        text=True,
    )
    return [int(x) for x in p.stdout.split()]


def test_sigkill_of_the_worker_fails_the_batch_and_resumes(r3_clean, tmp_path):
    """SIGKILL of the mosaic-ingest-worker process (not of mosaic ingest)
    once a shard is checkpointed: the batch is marked failed, and a resume
    gives the clean run's files."""
    import signal

    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    log = tmp_path / "ingest.err"
    with open(log, "w") as err:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mosaic.cli",
                "ingest",
                *GVCFS4,
                "--region",
                R3_REGION,
                "--shard-size",
                str(R3_SHARD),
                "--threads",
                "1",
            ],
            cwd=L.REPO_ROOT,
            env=L.test_env(),
            stdout=subprocess.DEVNULL,
            stderr=err,
        )
        killed: List[int] = []

        def _kill_after_a_checkpoint() -> bool:
            if proc.poll() is not None:
                return True
            c = L.get_pg_conn()
            try:
                if L.shard_completion_count(c) < 1:
                    return False
            finally:
                c.close()
            pids = _worker_pids(proc.pid)
            for pid in pids:
                os.kill(pid, signal.SIGKILL)
            killed.extend(pids)
            return bool(pids)

        assert L.poll_until(
            _kill_after_a_checkpoint, timeout=120, interval=0.02
        )
        rc = proc.wait(timeout=60)
    stderr = log.read_text()
    assert killed, "the worker finished before it could be killed"
    assert rc == 1, stderr[-2000:]
    assert "mosaic-ingest-worker exited with status -9" in stderr
    conn = L.get_pg_conn()
    try:
        assert L.batch_statuses(conn) == [(1, "failed")]
        n = L.shard_completion_count(conn)
        assert 1 <= n < R3_SHARDS
    finally:
        conn.close()
    p = _r3_ingest(2)
    assert "Resuming batch 1" in p.stderr
    assert L.iceberg_file_etags(L.get_catalog()) == r3_clean["etags"]


def test_fail_rust_shard_hook(r3_clean):
    """MOSAIC_TEST_FAIL_RUST_SHARD fails one shard inside the worker: exit
    non-zero, the shard named on stderr, the batch failed; the resume
    without it gives the clean run's files."""
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    p = _r3_ingest(
        4,
        extra_env={"MOSAIC_TEST_FAIL_RUST_SHARD": "chr21:10000000"},
        check=False,
    )
    assert p.returncode == 1
    assert "Error: shard chr21:10000000-12000000" in p.stderr
    assert (
        "MOSAIC_TEST_FAIL_RUST_SHARD: injected failure of shard "
        "chr21:10000000" in p.stderr
    )
    assert "mosaic-ingest-worker exited with status 1" in p.stderr
    conn = L.get_pg_conn()
    try:
        assert L.batch_statuses(conn) == [(1, "failed")]
        # Shards are checkpointed in order: none at or past the failed one.
        assert L.shard_completion_count(conn) <= 5
    finally:
        conn.close()
    p = _r3_ingest(4)
    assert "Resuming batch 1" in p.stderr
    assert L.iceberg_file_etags(L.get_catalog()) == r3_clean["etags"]


def test_malformed_fail_rust_shard_hook(docker_stack):
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    p = _r3_ingest(
        2,
        extra_env={"MOSAIC_TEST_FAIL_RUST_SHARD": "chr21:x"},
        check=False,
    )
    assert p.returncode == 1
    assert 'MOSAIC_TEST_FAIL_RUST_SHARD="chr21:x"' in p.stderr
    conn = L.get_pg_conn()
    try:
        assert L.batch_statuses(conn) == [(1, "failed")]
        assert L.shard_completion_count(conn) == 0
    finally:
        conn.close()


def test_fail_rust_combine_hook(r3_clean):
    """MOSAIC_TEST_FAIL_RUST_SHARD=combine:<table>:<index> at --threads 4
    fails that combined file inside the worker: exit non-zero, the file
    named on stderr, the batch failed with every shard checkpointed; the
    resume without it gives the clean run's files."""
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    p = _r3_ingest(
        4,
        extra_env={
            "MOSAIC_TEST_FAIL_RUST_SHARD": "combine:sample_site_info:0"
        },
        check=False,
    )
    assert p.returncode == 1
    assert "Error: combine sample_site_info file 0" in p.stderr
    assert "injected failure of combine sample_site_info file 0" in p.stderr
    assert "mosaic-ingest-worker exited with status 1" in p.stderr
    conn = L.get_pg_conn()
    try:
        assert L.batch_statuses(conn) == [(1, "failed")]
        assert L.shard_completion_count(conn) == R3_SHARDS
    finally:
        conn.close()
    p = _r3_ingest(4)
    assert "Resuming batch 1" in p.stderr
    assert L.iceberg_file_etags(L.get_catalog()) == r3_clean["etags"]


def test_sigkill_during_the_combine_fails_and_resumes(docker_stack, tmp_path):
    """SIGKILL of the worker during its combine (after the first Uploaded
    line; a 1-byte --aggregate-parquet-size makes ~60 files, so the gate
    holds the worker until the uploads catch up): the batch is marked
    failed, and a resume gives the clean run's files."""
    import signal

    size = 1
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    _r3_ingest(2, aggregate_parquet_size=size)
    clean = L.iceberg_file_etags(L.get_catalog())
    assert sum(len(v) for v in clean.values()) > 20

    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    log = tmp_path / "ingest.err"
    with open(log, "w") as err:
        proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "mosaic.cli",
                "ingest",
                *GVCFS4,
                "--region",
                R3_REGION,
                "--shard-size",
                str(R3_SHARD),
                "--threads",
                "2",
                "--aggregate-parquet-size",
                str(size),
            ],
            cwd=L.REPO_ROOT,
            env=L.test_env(),
            stdout=subprocess.DEVNULL,
            stderr=err,
        )
        killed: List[int] = []

        def _kill_after_an_upload() -> bool:
            if proc.poll() is not None:
                return True
            if "  Uploaded " not in log.read_text():
                return False
            pids = _worker_pids(proc.pid)
            for pid in pids:
                os.kill(pid, signal.SIGKILL)
            killed.extend(pids)
            return bool(pids)

        assert L.poll_until(_kill_after_an_upload, timeout=120, interval=0.02)
        rc = proc.wait(timeout=60)
    stderr = log.read_text()
    assert killed, "the worker finished before it could be killed"
    assert rc == 1, stderr[-2000:]
    assert "mosaic-ingest-worker exited with status -9" in stderr
    conn = L.get_pg_conn()
    try:
        assert L.batch_statuses(conn) == [(1, "failed")]
        assert L.shard_completion_count(conn) == R3_SHARDS
    finally:
        conn.close()
    p = _r3_ingest(2, aggregate_parquet_size=size)
    assert "Resuming batch 1" in p.stderr
    assert L.iceberg_file_etags(L.get_catalog()) == clean


def _wrapper(tmp_path: Any, pipe: str) -> str:
    """An executable running the real binary with its stdout through
    ``pipe``: the pipeline exits 0 whatever the worker does."""
    exe = tmp_path / "wrapped_worker"
    exe.write_text(f'#!/bin/sh\n"{L.rust_worker_path()}" "$@" | {pipe}\n')
    exe.chmod(0o755)
    return str(exe)


@pytest.mark.parametrize(
    "pipe,message",
    [
        ("head -n 3", "reported 3 of 24 shard(s)"),
        ("head -c 300", "malformed mosaic-ingest-worker result line"),
    ],
    ids=["missing-shards", "truncated-line"],
)
def test_worker_output_cut_short_fails_the_batch(
    docker_stack, tmp_path, pipe, message
):
    """The real binary's result stream cut short behind an exit 0 (a worker
    that stops early, a truncated last line): the batch is failed."""
    L.reset_state()
    L.run_init(GVCFS4[0], "chr21")
    p = _r3_ingest(
        2,
        extra_env={"MOSAIC_INGEST_WORKER": _wrapper(tmp_path, pipe)},
        check=False,
    )
    assert p.returncode == 1
    assert message in p.stderr
    conn = L.get_pg_conn()
    try:
        assert L.batch_statuses(conn) == [(1, "failed")]
    finally:
        conn.close()
