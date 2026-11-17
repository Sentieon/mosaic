# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Binary-level tests of the Rust ingest worker's local files and of the
removal derive against them.

* ``test_removal_substrate.py::
  test_local_sample_site_info_has_the_contribution_columns`` drove the
  Python worker's ``update_sample_del_state`` +
  ``collect_variant_group_data`` + ``collect_sample_ref_block_data`` +
  ``bulk_upload`` + ``_write_parquet_local`` on a deletion and a SNP of two
  samples at one site: here the same records go through
  ``mosaic-ingest-worker`` and the local ``sample_site_info`` rows carry
  the deletion's ``del_ref_spans``, its INFO DP as ``site_dp`` and FORMAT
  DP as ``dp``, and the SNP's PL-derived ``gtc`` (frozen from the Python
  worker's ``extract_site_info`` before its deletion).
* The local files' schemas equal the Iceberg tables' (``mosaic init``'s):
  the check ``test_iceberg_schemas_carry_the_contribution`` made against
  the Python worker's Arrow schemas, now against what the worker writes.
* ``test_remove.py::test_site_partials_equal_ingests_for_the_same_records``
  compared ``derive_site_partials`` (sign +1) with the Python worker's
  ``bulk_upload`` partials of the same records: here with the worker's
  local ``batch_site_info`` / ``batch_allele_counts`` rows. The ref-block
  side (``derive_ref_partial`` against the worker's ``batch_ref_blocks``)
  is added on a crafted batch with a deletion split and a ``./.`` block.
* ``test_sample_major_layout.py::
  test_combine_keys_are_deterministic_and_fold_chromosome_and_index``: the
  S3 keys are the driver's (``rust_combined_s3_key``), which needs no
  binary; the worker's half (the same bytes, one (chromosome, index) per
  file) is ``combine.rs``'s
  ``combine_is_deterministic_one_file_per_sample_at_a_one_byte_target``.

Positions in ``VRec`` are 1-based, as in the VCF text.
"""

import math

import pyarrow as pa
import pytest

from mosaic import ingest
from mosaic import remove as R
from mosaic.init_db import sample_ref_blocks_schema, sample_site_info_schema
from worker_harness import (
    VRec,
    read_local,
    ref_block,
    requires_worker,
    run_worker,
    write_gvcf,
)

pytestmark = pytest.mark.unit

C = 0  # the harness's chromosome index of chr21
SHARD = ("chr21", 0, 1_000_000)
PL6 = [0, 30, 300, 33, 330, 333]  # ref, alt, <NON_REF>
BATCH_ID = 7  # worker_harness.BATCH_ID


def _table(paths):
    tables = [read_local(p) for p in paths]
    return pa.concat_tables(tables) if tables else None


def _norm(t):
    """An Arrow type with list / large_list and string / large_string
    identified (pyiceberg's ``as_arrow`` uses the large forms)."""
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        return ("list", _norm(t.value_type))
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    return str(t)


def _shape(schema):
    return [(f.name, _norm(f.type), f.nullable) for f in schema]


def _run(tmp_path, samples, sample_ids):
    paths = [
        write_gvcf(str(tmp_path / f"{name}.g.vcf.gz"), name, recs)
        for name, recs in samples
    ]
    out = tmp_path / "out"
    out.mkdir()
    (res,) = run_worker(paths, [SHARD], str(out), sample_ids=sample_ids)
    return res


# ── the local files' rows and schemas ────────────────────────────────────────


# Replaces tests/unit/test_removal_substrate.py::
# test_local_sample_site_info_has_the_contribution_columns (K6).
@requires_worker
def test_local_sample_site_info_has_the_contribution_columns(tmp_path):
    # Python's _deletion(100, 4): POS 100, GAAA>G (deletes 101..103), INFO
    # DP 44, FORMAT DP 40; then sample 5's block 101..110, which the
    # deletion splits. Sample 6: a SNP at POS 100 with a PL-derived GTC.
    dele = VRec(
        100, "GAAA", ["G", "<NON_REF>"], "0/1", {"DP": 44}, 30.0, DP=40
    )
    snp = VRec(100, "G", ["T", "<NON_REF>"], "0/1", {"DP": 12}, 3.0, PL=PL6)
    res = _run(
        tmp_path,
        [("a", [dele, ref_block(101, 110, ref="A")]), ("b", [snp])],
        [5, 6],
    )
    table = _table(res["si_paths"])
    by_sid = {r["sample_id"]: r for r in table.to_pylist()}
    assert set(by_sid) == {5, 6}
    assert by_sid[5]["del_ref_spans"] == [101, 103]
    assert by_sid[6]["del_ref_spans"] is None
    assert by_sid[5]["site_dp"] == 44 and by_sid[5]["dp"] == 40
    # extract_site_info(snp)[4] of the Python worker (frozen): the GTC
    # from PL [0, 30, 300] (the <NON_REF> slots dropped).
    want = [0.998501062951534, 0.0014989370484661036, 1e-30]
    assert by_sid[6]["gtc"] == pytest.approx(want, rel=1e-12, abs=0)


@requires_worker
def test_local_files_have_the_iceberg_schemas_and_sample_counts(tmp_path):
    """The two per-sample local files' columns (names, types, nullability)
    are the Iceberg tables'; a 0/0 block counts 1 sample, a ./. block 0."""
    snp = VRec(60, "A", ["T", "<NON_REF>"], "0/1", {"DP": 9}, 5.0, PL=PL6)
    res = _run(
        tmp_path,
        [
            (
                "a",
                [
                    ref_block(1, 20),
                    ref_block(21, 59, gt="./.", min_dp=0),
                    snp,
                    ref_block(61, 80),
                ],
            )
        ],
        [3],
    )
    rb = _table(res["rb_paths"])
    si = _table(res["si_paths"])
    assert _shape(rb.schema) == _shape(sample_ref_blocks_schema().as_arrow())
    assert _shape(si.schema) == _shape(sample_site_info_schema().as_arrow())
    blocks = {
        (r["position_start"], r["position_end"]): r["sample_counts"]
        for r in rb.to_pylist()
    }
    assert blocks == {(1, 20): 1, (21, 59): 0, (61, 80): 1}


# ── the removal derive against the worker's partials ─────────────────────────


# Replaces tests/unit/test_remove.py::
# test_site_partials_equal_ingests_for_the_same_records (K6).
@requires_worker
def test_site_partials_equal_the_workers_for_the_same_records(tmp_path):
    """sign +1 over a batch's own rows reproduces the worker's
    batch_site_info and batch_allele_counts: the lake alone gives the
    partial. Three samples, two record shapes, a no call, PL-derived
    fractional gtc (the Python test's records; its Rec(100) is POS 101)."""
    snp = VRec(
        101,
        "A",
        ["T", "<NON_REF>"],
        "0/1",
        {"DP": 30},
        40.5,
        PL=PL6,
        SB=[1, 2, 3, 4],
    )
    dele = VRec(
        101,
        "AT",
        ["A", "<NON_REF>"],
        "1/1",
        {"DP": 20, "MQ": 58.0},
        70.0,
        SB=[1, 2, 3, 4],
    )
    nocall = VRec(101, "A", ["C", "<NON_REF>"], "./.", {"DP": 3}, 1.0, PL=PL6)
    res = _run(
        tmp_path,
        [("s1", [snp]), ("s2", [dele]), ("s3", [nocall])],
        [1, 2, 3],
    )
    lake = _table(res["si_paths"])
    assert lake.num_rows == 3
    (bsi,) = _table(res["bsi_paths"]).to_pylist()
    bac = _table(res["bac_paths"]).to_pylist()
    got_si, got_ac = R.derive_site_partials(lake, batch_id=BATCH_ID, sign=1)
    (g,) = got_si.to_pylist()
    for k in ("batch_id", "chromosome_index", "position"):
        assert g[k] == bsi[k], k
    assert g["position"] == 101
    for k in ("dp", "mqsq_2", "sb", "n_records"):
        assert g[k] == bsi[k], k
    assert g["n_records"] == 3
    assert g["gtc"] == bsi["gtc"]  # fixed point: exactly the worker's sum
    assert any(q % (1 << 32) for q in bsi["gtc"])  # fractional (PL-derived)
    for k in ("qual", "mqsq_1"):
        assert math.isclose(g[k], bsi[k], rel_tol=1e-12, abs_tol=1e-12), k
    assert {
        (r["position"], r["allele"]): (r["count"], r["n_listed"])
        for r in got_ac.to_pylist()
    } == {
        (r["position"], r["allele"]): (r["count"], r["n_listed"]) for r in bac
    }


@requires_worker
def test_ref_partial_equals_the_workers_batch_ref_blocks(tmp_path):
    """derive_ref_partial (sign +1) over a batch's own sample_ref_blocks
    and sample_site_info rows equals the worker's dense batch_ref_blocks:
    plain 0/0 blocks, a ./. block (sample count 0), and a deletion whose
    split part of the carrier's next block is counted with the deletion's
    site dp."""
    s1 = [
        ref_block(1, 100, min_dp=12),
        VRec(101, "A", ["T", "<NON_REF>"], "0/1", {"DP": 30}, 40.5, PL=PL6),
        ref_block(102, 200, min_dp=15),
    ]
    s2 = [
        ref_block(1, 100, min_dp=8),
        VRec(101, "ATTT", ["A", "<NON_REF>"], "0/1", {"DP": 25}, 70.0, DP=21),
        ref_block(102, 200, min_dp=9),
    ]
    s3 = [
        ref_block(1, 50, gt="./.", min_dp=4),
        ref_block(51, 100, min_dp=6),
        VRec(101, "A", ["C", "<NON_REF>"], "./.", {"DP": 3}, 1.0, PL=PL6),
        ref_block(102, 150, min_dp=7),
    ]
    res = _run(tmp_path, [("s1", s1), ("s2", s2), ("s3", s3)], [1, 2, 3])
    blocks = _table(res["rb_paths"])
    sites = _table(res["si_paths"])
    (spans,) = [
        r["del_ref_spans"] for r in sites.to_pylist() if r["sample_id"] == 2
    ]
    assert spans == [102, 104]  # the split is exercised
    got = R.derive_ref_partial(blocks, sites, C, 1, 1000, 100, BATCH_ID, 1)
    dense = {
        r["position"]: (r["dp"], r["sample_counts"]) for r in got.to_pylist()
    }
    want = {
        r["position"]: (r["dp"], r["sample_counts"])
        for r in _table(res["brb_paths"]).to_pylist()
    }
    assert dense == want
    # s1's and s3's blocks; s2's deletion (its site dp, no sample).
    assert want[102] == (15 + 25 + 7, 2)
    assert want[105] == (15 + 9 + 7, 3)
    assert want[30] == (12 + 8 + 4, 2)  # the ./. block counts no sample


# ── the combined files' S3 keys (the driver) ─────────────────────────────────


# Replaces tests/unit/test_sample_major_layout.py::
# test_combine_keys_are_deterministic_and_fold_chromosome_and_index (K6),
# the key half; no binary needed.
def test_combined_keys_fold_chromosome_index_and_batch():
    names = ["x", "y"]

    def keys(batch_id, table="sample_ref_blocks"):
        return [
            ingest.rust_combined_s3_key(
                {"table": table, "chromosome_index": c, "index": i},
                names,
                batch_id,
                "chr21",
            )
            for c in (0, 1)
            for i in range(3)
        ]

    first = keys(7)
    assert keys(7) == first  # deterministic
    assert len(set(first)) == len(first) == 6  # 2 chromosomes x 3 files
    assert not set(keys(8)) & set(first)  # another batch: other keys
    assert not set(keys(7, "sample_site_info")) & set(first)
    assert all(k.startswith("sample_ref_blocks/") for k in first)
