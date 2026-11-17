# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""batch_genotypes / sample_qc across batch layouts, resume, partitions.

* Two single-sample batches vs one two-sample batch: every cell the split
  layout resolves equals the joint layout's; the rest are UNRESOLVED (a
  batch has no row where none of its samples has a variant record).
* Crash + resume (MOSAIC_TEST_FAIL_SHARD): the matrix and sample_qc equal a
  clean run's and batch_genotypes has no duplicate (batch, chrom, pos) rows.
* Whole-genome ingest of a two-chromosome slice: batch_genotypes and the
  two (whole-sample) per-sample tables are partitioned by chromosome_index,
  every data file holds one chromosome, and a chromosome filter prunes to
  that partition's files.
* The same-site merge: one sample with two records at one POS whose alt
  orders differ is stored with each record's own allele list and GT/AD in
  that record's order (sample_site_info, since E3) and by allele string in
  the site's merged list (batch_genotypes).
"""

import os
import struct
import subprocess

import pytest

import jc_testlib as L
from pyiceberg.expressions import EqualTo

from mosaic.genotypes import read_genotype_matrix
from mosaic.init_db import SAMPLE_QC_COLUMNS

pytestmark = pytest.mark.integration

SLICE_2CHR = os.path.join(
    L.REPO_ROOT,
    "test_data/downloads/HG00099/"
    "HG00099.chr21_1-15M.chr22_10.5-20M.final.hard-filtered.gvcf.gz",
)


def _subsets(tmp_path, region):
    out = []
    for j, src in enumerate(L.GVCFS):
        dst = os.path.join(str(tmp_path), f"s{j}.g.vcf.gz")
        L.subset_gvcf(src, region, dst)
        out.append(dst)
    return out


def _matrix_by_name(chrom_index=0):
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, sample_name FROM samples")
            name = dict(cur.fetchall())
        m = read_genotype_matrix(L.get_catalog(), conn, chrom_index, 1, 10**9)
    finally:
        conn.close()
    cells = {
        (pos, name[sid]): v
        for (pos, sid), v in L.genotype_matrix_cells(m).items()
    }
    return m, cells


def _bgt_keys():
    arrow = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.batch_genotypes")
        .scan()
        .to_arrow()
    )
    return list(
        zip(
            arrow.column("batch_id").to_pylist(),
            arrow.column("chromosome_index").to_pylist(),
            arrow.column("position").to_pylist(),
        )
    )


def test_two_single_sample_batches_vs_one_batch(clean_state, tmp_path):
    region = "chr21:10000000-10500000"
    a, b = _subsets(tmp_path, region)

    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest([a, b], region=region, threads=2)
    joint_m, joint = _matrix_by_name()

    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest([a], region=region, threads=2)
    L.run_ingest([b], region=region, threads=2)
    split_m, split = _matrix_by_name()

    assert sorted(set(split_m.batch_ids.tolist())) == [1, 2]
    assert split_m.positions.tolist() == joint_m.positions.tolist()
    assert set(split) == set(joint)
    unresolved = 0
    for key, cell in split.items():
        if cell[0] == "?":
            assert cell == ("?", "?", 255)
            unresolved += 1
        else:
            assert cell == joint[key], key
    # Each sample is unresolved exactly where only the other sample has a
    # variant record.
    assert 0 < unresolved < len(split)
    print(
        f"\n  two single-sample batches: {unresolved} of {len(split)} cells "
        f"UNRESOLVED ({len(split_m.positions)} sites x 2 samples)"
    )


@pytest.mark.restart
def test_resume_matrix_and_qc_match_clean_run(clean_state, tmp_path):
    region = "chr21:5000000-10000000"
    gvcfs = _subsets(tmp_path, region)
    kwargs = dict(region=region, shard_size=1_000_000, threads=2)

    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(gvcfs, **kwargs)
    _, base_cells = _matrix_by_name()
    conn = L.get_pg_conn()
    try:
        base_qc = L.sample_qc_rows(conn)
    finally:
        conn.close()
    assert base_cells and len(base_qc) == 2

    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    proc = L.run_ingest(
        gvcfs,
        extra_env={"MOSAIC_TEST_FAIL_SHARD": "chr21:7000000"},
        check=False,
        **kwargs,
    )
    assert proc.returncode != 0, "ingest should have crashed"
    conn = L.get_pg_conn()
    try:
        assert L.shard_completion_count(conn) > 0
        assert L.sample_qc_rows(conn) == {}, "QC is written only on completion"
    finally:
        conn.close()

    L.run_ingest(gvcfs, **kwargs)
    _, cells = _matrix_by_name()
    conn = L.get_pg_conn()
    try:
        assert dict(L.batch_statuses(conn)) == {1: "complete"}
        assert L.sample_qc_rows(conn) == base_qc
    finally:
        conn.close()
    assert cells == base_cells
    keys = _bgt_keys()
    assert len(keys) == len(set(keys)), "duplicate batch_genotypes rows"


def test_partitioned_whole_genome_ingest(clean_state):
    if not os.path.exists(SLICE_2CHR):
        pytest.skip(
            f"{SLICE_2CHR} not downloaded (bcftools view -r "
            f"chr21:1-15000000,chr22:10500000-20000000 of HG00099)"
        )
    L.run_init(SLICE_2CHR, "chr21,chr22")
    L.run_ingest([SLICE_2CHR], threads=2)  # no --region: whole genome

    table = L.get_catalog().load_table(f"{L.TEST_NAMESPACE}.batch_genotypes")
    assert [f.name for f in table.spec().fields] == ["chromosome_index"]
    files = [t.file for t in table.scan().plan_files()]
    by_chrom = {}
    for f in files:
        part = f.partition[0]
        lo = struct.unpack("<i", f.lower_bounds[2])[0]
        hi = struct.unpack("<i", f.upper_bounds[2])[0]
        assert lo == hi == part, (f.file_path, part, lo, hi)
        by_chrom.setdefault(part, []).append(f.file_path)
    assert sorted(by_chrom) == [0, 1]  # chr21, chr22

    for c_idx in (0, 1):
        pruned = table.scan(
            row_filter=EqualTo("chromosome_index", c_idx)
        ).plan_files()
        assert sorted(t.file.file_path for t in pruned) == sorted(
            by_chrom[c_idx]
        )
        m, _ = _matrix_by_name(c_idx)
        lo, hi = (1, 15_000_000) if c_idx == 0 else (10_500_000, 20_000_000)
        assert len(m.positions) > 1000
        assert lo - 1000 <= m.positions.min() and m.positions.max() <= hi

    # The per-sample tables (E2): one file per chromosome by construction
    # (the whole-sample combine packs per chromosome), its partition value
    # the file's chromosome_index statistics.
    for name in ("sample_ref_blocks", "sample_site_info"):
        tbl = L.get_catalog().load_table(f"{L.TEST_NAMESPACE}.{name}")
        assert [f.name for f in tbl.spec().fields] == ["chromosome_index"]
        parts = []
        for t in tbl.scan().plan_files():
            f = t.file
            lo = struct.unpack("<i", f.lower_bounds[2])[0]
            hi = struct.unpack("<i", f.upper_bounds[2])[0]
            assert lo == hi == f.partition[0], (name, f.file_path, lo, hi)
            parts.append(f.partition[0])
        assert sorted(parts) == [0, 1], (name, parts)

    bsi = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.batch_site_info")
        .scan()
        .to_arrow()
    )
    bsi_keys = set(
        zip(
            bsi.column("batch_id").to_pylist(),
            bsi.column("chromosome_index").to_pylist(),
            bsi.column("position").to_pylist(),
        )
    )
    assert set(_bgt_keys()) == bsi_keys


def _same_site_gvcf(tmp_path):
    """One sample (SAMESITE1): a SNP record and a deletion record at the
    same POS, alt orders [T, <NON_REF>] and [A, <NON_REF>]."""
    header = subprocess.run(
        ["bcftools", "view", "-h", L.GVCFS[0]],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    header[-1] = "\t".join(header[-1].split("\t")[:9] + ["SAMESITE1"])
    body = [
        "chr21\t5000000\t.\tA\t<NON_REF>\t.\tPASS\tEND=5000099\t"
        "GT:AD:DP:GQ:MIN_DP:PL\t0/0:10,0:10:30:10:0,30,300",
        "chr21\t5000100\t.\tA\tT,<NON_REF>\t50\tPASS\tDP=20\t"
        "GT:AD:DP:GQ:PL:SB\t0/1:10,10,0:20:50:50,0,50,80,80,160:5,5,5,5",
        "chr21\t5000100\t.\tAT\tA,<NON_REF>\t40\tPASS\tDP=18\t"
        "GT:AD:DP:GQ:PL:SB\t0/1:10,8,0:18:40:40,0,45,80,80,160:5,5,4,4",
        "chr21\t5000101\t.\tT\t<NON_REF>\t.\tPASS\tEND=5000200\t"
        "GT:AD:DP:GQ:MIN_DP:PL\t0/0:10,0:10:30:10:0,30,300",
    ]
    plain = os.path.join(str(tmp_path), "samesite.g.vcf")
    with open(plain, "w") as fh:
        fh.write("\n".join(header + body) + "\n")
    out = plain + ".gz"
    subprocess.run(
        ["bcftools", "view", "--write-index=tbi", "-Oz", "-o", out, plain],
        check=True,
        capture_output=True,
    )
    return out


def test_same_site_merge_maps_by_allele(clean_state, tmp_path):
    gvcf = _same_site_gvcf(tmp_path)
    L.run_init(L.GVCFS[0], "chr21")
    proc = L.run_ingest([gvcf], region="chr21", threads=1)
    assert "1 genotype-matrix slot conflicts" in proc.stderr

    ssi = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.sample_site_info")
        .scan()
        .to_arrow()
        .to_pylist()
    )
    assert len(ssi) == 2  # one row per record
    # Each row carries its own record's allele strings (E3); gt_1 / gt_2
    # index them and AD is in that record's order.
    got = sorted(
        (
            r["alleles"],
            r["alleles"][r["gt_1"]],
            r["alleles"][r["gt_2"]],
            r["ad"],
        )
        for r in ssi
    )
    # Before D1's fix the deletion row read gt_2 = xT and AD [10, 8, 0]
    # (its own [A, -AT, *] order) against the merged allele list; from D1
    # to E3 its AD was re-laid out in the merged order [10, 0, 0, 8].
    assert got == [
        (["A", "-AT", "*"], "A", "-AT", [10, 8, 0]),
        (["A", "xT", "*"], "A", "xT", [10, 10, 0]),
    ]
    # The partial lists the site's merged alleles, each with its listing
    # records (n_listed) and hits.
    bac = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.batch_allele_counts")
        .scan()
        .to_arrow()
        .to_pylist()
    )
    assert sorted((r["allele"], r["count"], r["n_listed"]) for r in bac) == [
        ("*", 0, 2),
        ("-AT", 1, 1),
        ("A", 2, 2),
        ("xT", 1, 1),
    ]

    m, cells = _matrix_by_name()
    assert m.positions.tolist() == [5000100]
    assert m.alleles == [["A", "xT", "-AT"]]
    assert cells == {(5000100, "SAMESITE1"): ("A", "xT", 50)}  # first wins
    conn = L.get_pg_conn()
    try:
        ((_, row),) = L.sample_qc_rows(conn).items()
    finally:
        conn.close()
    qc = dict(zip(SAMPLE_QC_COLUMNS, row))
    assert (qc["n_var_records"], qc["n_snv"], qc["n_del"]) == (2, 1, 1)
    assert qc["rb_bases"] == 200
