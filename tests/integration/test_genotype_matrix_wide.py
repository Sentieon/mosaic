# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""batch_genotypes rows two bytes per slot (genotype indices wider than
int8).

A synthetic single-sample gVCF written here holds one record with 150
distinct insertion ALTs (GT ``1/150``, ``<NON_REF>`` last) at a chr21
position where the real fixture sample HG00245 has a hom-ref block. Both are
ingested as one batch over a 500 kb slice. The site's allele list has 152
alleles, so its codes exceed 127: its ``gt_1`` / ``gt_2`` blobs hold two
bytes per slot (before F1 the code 150 was stored as a no call), and every
other row keeps one byte. The matrix read returns the call; the lake oracle
and the aggregate (whose partials never read the matrix) hold.
"""

import gzip
import os
import subprocess

import pytest

import jc_testlib as L
from mosaic.genotypes import dosage, read_genotype_matrix
from truth_alleles import format_alt_allele

pytestmark = pytest.mark.integration

REGION = "chr21:9000000-9500000"
N_ALTS = 150
SAMPLE = "SYNTH150"


def _hom_ref_block_start(path: str) -> int:
    """POS of HG00245's first hom-ref block in the slice past 9.2 Mb whose
    REF is ``A`` (the synthetic record's REF), covering a few bases."""
    out = subprocess.run(
        [
            "bcftools",
            "query",
            "-r",
            "chr21:9200000-9500000",
            "-f",
            "%POS\t%REF\t%ALT\t%INFO/END\t[%GT]\n",
            path,
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    for line in out.splitlines():
        pos, ref, alt, end, gt = line.split("\t")
        if (
            alt == "<NON_REF>"
            and ref == "A"
            and gt in ("0/0", "0|0")
            and end != "."
            and int(end) > int(pos) + 2
        ):
            return int(pos)
    raise AssertionError("no hom-ref block with REF A in the slice")


def _write_synthetic(template: str, pos: int, dst: str) -> list:
    """Write the 150-ALT single-sample gVCF (header from ``template``, the
    sample renamed), bgzip and index it; return its ALT strings."""
    alts = ["A" + "C" * (i + 1) for i in range(N_ALTS)]
    n_alleles = N_ALTS + 2  # ref, the insertions, <NON_REF>
    ad = [0] * n_alleles
    ad[1] = ad[N_ALTS] = 10
    n_gt = n_alleles * (n_alleles + 1) // 2
    pl = [99] * n_gt
    pl[N_ALTS * (N_ALTS + 1) // 2 + 1] = 0  # genotype (1, 150)
    with gzip.open(template, "rt") as fh:
        header = []
        for line in fh:
            if not line.startswith("#"):
                break
            header.append(line)
    header[-1] = "\t".join(header[-1].rstrip("\n").split("\t")[:9])
    header[-1] += f"\t{SAMPLE}\n"
    record = "\t".join(
        [
            "chr21",
            str(pos),
            ".",
            "A",
            ",".join(alts + ["<NON_REF>"]),
            "50",
            "PASS",
            "DP=20;MQ=60",
            "GT:AD:DP:GQ:PL",
            ":".join(
                [
                    f"1/{N_ALTS}",
                    ",".join(map(str, ad)),
                    "20",
                    "40",
                    ",".join(map(str, pl)),
                ]
            ),
        ]
    )
    plain = dst[: -len(".gz")]
    with open(plain, "w") as fh:
        fh.writelines(header)
        fh.write(record + "\n")
    subprocess.run(
        ["bcftools", "view", "--write-index=tbi", "-Oz", "-o", dst, plain],
        check=True,
        capture_output=True,
        text=True,
    )
    return alts


@pytest.fixture(scope="module")
def wide(docker_stack, tmp_path_factory):
    work = str(tmp_path_factory.mktemp("gtm_wide"))
    real = os.path.join(work, "real.g.vcf.gz")
    L.subset_gvcf(L.GVCFS[0], REGION, real)
    pos = _hom_ref_block_start(real)
    synth = os.path.join(work, "synth.g.vcf.gz")
    alts = _write_synthetic(L.GVCFS[0], pos, synth)
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest([real, synth], region=REGION, threads=2)
    conn = L.get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT sample_name, id FROM samples")
            sid_of = dict(cur.fetchall())
        matrix = read_genotype_matrix(L.get_catalog(), conn, 0, 1, 10**9)
    finally:
        conn.close()
    bgt = (
        L.get_catalog()
        .load_table(f"{L.TEST_NAMESPACE}.batch_genotypes")
        .scan()
        .to_arrow()
    )
    yield {
        "pos": pos,
        "alts": [format_alt_allele("A", a) for a in alts],
        "sid_of": sid_of,
        "matrix": matrix,
        "bgt": bgt,
    }


def test_wide_row_at_the_site_only(wide):
    bgt = wide["bgt"]
    pos = bgt.column("position").to_pylist()
    n = bgt.column("n_samples").to_pylist()
    assert set(n) == {2}
    assert len(pos) > 100
    for name in ("gt_1", "gt_2"):
        widths = [len(b) // k for b, k in zip(bgt.column(name).to_pylist(), n)]
        assert {p for p, w in zip(pos, widths) if w == 2} == {wide["pos"]}
        assert set(widths) == {1, 2}
    i = pos.index(wide["pos"])
    assert len(bgt.column("alleles")[i].as_py()) == N_ALTS + 1  # '*' unused
    assert [len(b) for b in bgt.column("gq").to_pylist()] == [2] * len(pos)


def test_matrix_returns_the_call(wide):
    m = wide["matrix"]
    assert m.gt_1.dtype.name == "int16"
    i = m.positions.tolist().index(wide["pos"])
    alleles = m.alleles[i]
    assert alleles[0] == "A" and alleles[1:] == wide["alts"]
    cols = {int(s): j for j, s in enumerate(m.sample_ids.tolist())}
    j_syn = cols[wide["sid_of"][SAMPLE]]
    j_real = 1 - j_syn
    assert (m.gt_1[i, j_syn], m.gt_2[i, j_syn]) == (1, N_ALTS)
    assert alleles[m.gt_2[i, j_syn]] == wide["alts"][-1]
    assert m.gq[i, j_syn] == 40
    assert (m.gt_1[i, j_real], m.gt_2[i, j_real]) == (0, 0)  # its block
    d = dosage(m, i, wide["alts"][-1])
    assert d[j_syn] == 1 and d[j_real] == 0
    assert dosage(m, i, wide["alts"][0])[j_syn] == 1
    assert dosage(m, i, wide["alts"][70])[j_syn] == 0
    # The other sites are narrow rows read as before: the real sample's
    # calls, the synthetic sample uncovered there (a no call).
    others = [k for k in range(len(m.positions)) if k != i]
    assert (m.gt_1[others, j_syn] == -1).all()
    assert (m.gt_1[others, j_real] >= 0).any()


def test_lake_oracle_and_aggregate_unaffected(wide):
    conn = L.get_pg_conn()
    try:
        stats = L.assert_partials_derivable_from_lake(conn, L.get_catalog())
        assert stats["sites"] > 100
        L.run_aggregate()
        L.assert_matches_oracle(conn, L.get_catalog())
        with conn.cursor() as cur:
            cur.execute(
                "SELECT allele, count FROM alleles "
                "WHERE chromosome_index = 0 AND position = %s",
                (wide["pos"],),
            )
            counts = dict(cur.fetchall())
    finally:
        conn.close()
    assert set(wide["alts"]) <= set(counts)
    assert counts[wide["alts"][0]] == 1 and counts[wide["alts"][-1]] == 1
    assert sum(counts[a] for a in wide["alts"][1:-1]) == 0
