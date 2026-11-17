# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The whole-sample per-sample layout on the chr21 fixture: data files
that
hold whole samples of one batch and one chromosome, rows sorted (position,
sample_id) inside them.

Five chr21 samples over two 300 kb windows, as two batches (R1: one shard,
one position window, so a sample's rows of a table are one blob; R2: 100 kb
shards and 50 kb windows, many blobs per sample).

* ``singles`` (`--aggregate-parquet-size 1`: one sample per file): every
  file holds one sample; a sample-subset scan plans exactly the subset's
  files (``plan_files``); a pyiceberg ``delete(EqualTo("sample_id", s))``
  (copy-on-write) removes exactly ``s``'s files, one per batch, writes none
  and leaves every other file as it was. A one-sample R1 file has one row
  group whose compressed bytes equal its blob's: the packing weights.
* ``packed`` (fresh stores; R1 at a target between those weights, R2 at the
  default): the files per partition are exactly the packing rule's
  (``_pack``, the rule restated: K6) of the weights; every file holds whole samples, rows sorted (position,
  sample_id), row groups within the table's cap and spanning samples; the
  E1 lake oracle holds; a delete of a sample that shares its files rewrites
  exactly those files (the packing, not the row order, sets a delete's
  cost).

Last (it resets the stores): `mosaic ingest` refuses a lake whose
per-sample table is not partitioned.
"""

import os

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest
from pyiceberg.expressions import And, EqualTo, In
from pyiceberg.schema import Schema

import jc_testlib as L
from mosaic.config import PARQUET_ROW_GROUP_ROWS, PARQUET_SITE_ROW_GROUP_ROWS

pytestmark = pytest.mark.integration

# Five distinct samples: the fixture's HG00245 (input_gvcfs4) and four
# single-sample chr21:1-15M gVCFs (input_gvcfs; its HG00245 file holds two
# other samples).
FIVE = [L.GVCFS[0]] + [
    os.path.join(
        L.REPO_ROOT,
        f"test_data/input_gvcfs/{s}.final.hard-filtered_chr21_1_15000000_"
        "no-ambig.gvcf.gz",
    )
    for s in ("HG00420", "HG00658", "HG02604", "HG02972")
]
# Inside one 10 Mbp shard and position window (1-based 10.1M-10.4M), so a
# sample's rows are one blob per table.
R1 = "chr21:10100000-10400000"
R2 = "chr21:14500000-14800000"
R2_START = 13_000_000  # between the windows
R2_INGEST = {"shard_size": 100_000, "pos_batch_size": 50_000}
PER_SAMPLE = ("sample_ref_blocks", "sample_site_info")
POSITION = {
    "sample_ref_blocks": "position_start",
    "sample_site_info": "position",
}
CAP = {
    "sample_ref_blocks": PARQUET_ROW_GROUP_ROWS,
    "sample_site_info": PARQUET_SITE_ROW_GROUP_ROWS,
}


def _subsets(work, region, tag):
    out = []
    for j, src in enumerate(FIVE):
        dst = os.path.join(work, f"{tag}_s{j}.g.vcf.gz")
        L.subset_gvcf(src, region, dst)
        out.append(dst)
    return out


def _table(table):
    return L.get_catalog().load_table(f"{L.TEST_NAMESPACE}.{table}")


def _data_files(table, row_filter=None):
    """{uri: DataFile} of the current snapshot (``row_filter``: as planned
    for that scan)."""
    tbl = _table(table)
    scan = (
        tbl.scan() if row_filter is None else tbl.scan(row_filter=row_filter)
    )
    return {t.file.file_path: t.file for t in scan.plan_files()}


def _read(uri):
    """A data file's bytes as a ParquetFile."""
    bucket, key = uri[len("s3://") :].split("/", 1)
    body = L.s3_client().get_object(Bucket=bucket, Key=key)["Body"].read()
    return pq.ParquetFile(pa.BufferReader(body))


def _file_samples(table):
    """{uri: sorted sample ids} of every data file of ``table``, from its
    rows."""
    out = {}
    for uri in _data_files(table):
        ids = _read(uri).read(columns=["sample_id"]).column(0)
        out[uri] = sorted(set(ids.to_pylist()))
    return out


def _batch(df):
    """1 or 2: which window (batch) a data file's rows are in."""
    lo = df.lower_bounds[3]  # the position key: field id 3 in both tables
    return 2 if int.from_bytes(lo, "little", signed=True) >= R2_START else 1


def _batch_files(table, batch):
    """{uri: sorted sample ids} of one batch's files."""
    files = _data_files(table)
    return {
        u: ss
        for u, ss in _file_samples(table).items()
        if _batch(files[u]) == batch
    }


def _sample_ids():
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id FROM samples ORDER BY id")
            return [i for (i,) in cur.fetchall()]
    finally:
        conn.close()


def _fresh_lake(r1, r2, r1_target, r2_target):
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(r1, region=R1, threads=2, aggregate_parquet_size=r1_target)
    L.run_ingest(
        r2, region=R2, threads=2, aggregate_parquet_size=r2_target, **R2_INGEST
    )


@pytest.fixture(scope="module")
def inputs(docker_stack, tmp_path_factory):
    if not all(os.path.exists(p) for p in FIVE):
        pytest.skip("test_data/input_gvcfs: the chr21 gVCFs are missing")
    work = str(tmp_path_factory.mktemp("whole_sample"))
    return _subsets(work, R1, "r1"), _subsets(work, R2, "r2")


@pytest.fixture(scope="module")
def singles(inputs):
    """One sample per file (a 1-byte target) in both batches, and the
    per-sample packing weights of R1."""
    _fresh_lake(*inputs, r1_target=1, r2_target=1)
    ids = _sample_ids()
    assert len(ids) == 5
    weights = {}
    for t in PER_SAMPLE:
        weights[t] = {}
        samples = _file_samples(t)
        for uri, df in _data_files(t).items():
            if _batch(df) != 1:
                continue
            md = _read(uri).metadata
            assert md.num_row_groups == 1  # one blob's rows
            (sid,) = samples[uri]
            rg = md.row_group(0)
            weights[t][sid] = sum(
                rg.column(i).total_compressed_size
                for i in range(rg.num_columns)
            )
    return {"ids": ids, "weights": weights}


# ── one sample per file ──────────────────────────────────────────────────────


@pytest.mark.parametrize("table", PER_SAMPLE)
def test_one_file_per_sample_and_batch(singles, table):
    for batch in (1, 2):
        assert sorted(_batch_files(table, batch).values()) == [
            [s] for s in singles["ids"]
        ]
    for df in _data_files(table).values():
        assert tuple(df.partition) == (0,)


@pytest.mark.parametrize("table", PER_SAMPLE)
def test_subset_scan_plans_only_the_subsets_files(singles, table):
    ids = singles["ids"]
    samples = _file_samples(table)
    assert len(samples) == 10
    chrom = EqualTo("chromosome_index", 0)
    assert len(_data_files(table, chrom)) == 10
    for subset in ([ids[0], ids[-1]], [ids[2]]):
        planned = _data_files(
            table, And(chrom, In("sample_id", [int(s) for s in subset]))
        )
        assert len(planned) == 2 * len(subset)  # one file per batch
        assert set(planned) == {
            u for u, ss in samples.items() if set(ss) & set(subset)
        }
    one = _data_files(table, EqualTo("sample_id", int(ids[2])))
    assert {tuple(samples[u]) for u in one} == {(ids[2],)}


@pytest.mark.parametrize("table", PER_SAMPLE)
def test_sample_delete_removes_only_its_files(singles, table):
    """Copy-on-write on one-sample files: the sample's files (one per
    batch) leave the table, nothing is rewritten, every other file stays.
    Runs last on this lake (it deletes)."""
    s = singles["ids"][1]
    before = _data_files(table)
    samples = _file_samples(table)
    holding = {u for u, ss in samples.items() if ss == [s]}
    assert len(holding) == 2
    n_s = sum(before[u].record_count for u in holding)

    _table(table).delete(EqualTo("sample_id", int(s)))

    after = _data_files(table)
    assert set(before) - set(after) == holding
    assert set(after) - set(before) == set()  # nothing rewritten
    for u in after:
        assert after[u].file_size_in_bytes == before[u].file_size_in_bytes
    rest = _table(table).scan().to_arrow()
    assert (
        rest.num_rows == sum(df.record_count for df in before.values()) - n_s
    )
    assert not pc.any(pc.equal(rest.column("sample_id"), s)).as_py()


# ── packed files ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def packed(inputs, singles):
    """R1 packed to a target between the singles' weights, R2 at the
    default target (one file per table)."""
    w = singles["weights"]["sample_ref_blocks"]
    s1, s2, s3 = sorted(w)[:3]
    target = w[s1] + w[s2] + w[s3] // 2
    _fresh_lake(*inputs, r1_target=target, r2_target=None)
    assert _sample_ids() == singles["ids"]
    return {"target": target}


def _pack(weights, target):
    """The packing rule, restated test-side (K6: the Python combine's
    ``pack_sample_files`` is gone; the Rust worker's ``combine.rs`` packs):
    one chromosome's samples in ascending ``sample_id``, a new file opened
    when adding the next sample would take the current, non-empty file over
    the byte ``target``; so a sample over the target is alone in its file
    and no file splits a sample. ``weights`` {sample_id: bytes}; returns the
    files' sample lists in order."""
    files, cur, cur_bytes = [], [], 0
    for s in sorted(weights):
        if cur and cur_bytes + weights[s] > target:
            files.append(cur)
            cur, cur_bytes = [], 0
        cur.append(s)
        cur_bytes += weights[s]
    if cur:
        files.append(cur)
    return files


@pytest.mark.parametrize("table", PER_SAMPLE)
def test_files_per_partition_follow_the_packing(singles, packed, table):
    weights = singles["weights"][table]
    want = _pack(weights, packed["target"])
    got = sorted(_batch_files(table, 1).values())
    assert got == want
    if table == "sample_ref_blocks":
        # The target splits the five samples, and some file holds more
        # than one.
        assert 1 < len(got) < 5 and max(map(len, got)) > 1
    # The default target: one file holding the five samples.
    assert list(_batch_files(table, 2).values()) == [singles["ids"]]


@pytest.mark.parametrize("table", PER_SAMPLE)
def test_rows_sorted_by_position_then_sample(singles, packed, table):
    multi_sample_groups = 0
    for uri in _data_files(table):
        pf = _read(uri)
        data = pf.read(columns=[POSITION[table], "sample_id"])
        keys = list(
            zip(data.column(0).to_pylist(), data.column(1).to_pylist())
        )
        assert keys == sorted(keys), uri
        md = pf.metadata
        sid = [md.schema.column(i).path for i in range(md.num_columns)].index(
            "sample_id"
        )
        for g in range(md.num_row_groups):
            rg = md.row_group(g)
            assert rg.num_rows <= CAP[table]
            st = rg.column(sid).statistics
            multi_sample_groups += st.min != st.max
    assert multi_sample_groups > 0  # row groups span samples
    # Whole samples: a sample is in exactly one file per batch.
    for batch in (1, 2):
        held = [s for ss in _batch_files(table, batch).values() for s in ss]
        assert sorted(held) == singles["ids"]


def test_lake_oracle_holds(packed):
    conn = L.get_pg_conn()
    try:
        stats = L.assert_partials_derivable_from_lake(conn, L.get_catalog())
    finally:
        conn.close()
    assert stats["batches"] == 2


@pytest.mark.parametrize("table", PER_SAMPLE)
def test_shared_file_delete_rewrites_exactly_the_files_holding_it(
    packed, table
):
    """A sample that shares its files: each file holding it is rewritten
    without it (pyiceberg keeps the remaining rows in their order), every
    other file stays. Runs last on this lake (it deletes)."""
    before = _data_files(table)
    samples = _file_samples(table)
    shared = [u for u, ss in _batch_files(table, 1).items() if len(ss) > 1]
    s = samples[shared[0]][0]
    holding = {u for u, ss in samples.items() if s in ss}
    assert len(holding) == 2  # its batch-1 file and the batch-2 file

    _table(table).delete(EqualTo("sample_id", int(s)))

    after = _data_files(table)
    assert set(before) - set(after) == holding
    added = set(after) - set(before)
    assert len(added) == len(holding)
    now = _file_samples(table)
    left = sorted(x for u in added for x in now[u])
    assert left == sorted(x for u in holding for x in samples[u] if x != s)
    for u in added:
        data = _read(u).read(columns=[POSITION[table], "sample_id"])
        keys = list(
            zip(data.column(0).to_pylist(), data.column(1).to_pylist())
        )
        assert keys == sorted(keys)
    for u in set(before) - holding:
        assert after[u].file_size_in_bytes == before[u].file_size_in_bytes


# ── an old lake (resets the stores) ──────────────────────────────────────────


@pytest.mark.parametrize("table", PER_SAMPLE)
def test_ingest_refuses_an_unpartitioned_per_sample_table(
    clean_state, tmp_path, table
):
    """A lake from before E2 (every column present, no partition spec) is
    refused before anything is written."""
    L.run_init(L.GVCFS[0], "chr21")
    catalog = L.get_catalog()
    name = f"{L.TEST_NAMESPACE}.{table}"
    fields = catalog.load_table(name).schema().fields
    catalog.drop_table(name)
    catalog.create_table(name, schema=Schema(*fields))
    gvcf = os.path.join(str(tmp_path), "s0.g.vcf.gz")
    L.subset_gvcf(L.GVCFS[0], "chr21:10100000-10200000", gvcf)
    proc = L.run_ingest([gvcf], region="chr21", threads=1, check=False)
    assert proc.returncode != 0
    assert f"{table}: not partitioned by chromosome_index" in proc.stderr
    assert "no migration path" in proc.stderr and "mosaic init" in proc.stderr
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
