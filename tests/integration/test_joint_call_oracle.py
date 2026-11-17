# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""joint_call.py as a self-consistent oracle (AC / AN / AF from the
emitted genotypes), on the six chr21 samples as three 2-sample batches, one
ingest + aggregate and several joint calls (the Rust engine, the only
one):

  (a) ``bcftools +fill-tags -t AC,AN,AF`` reproduces the written AC and AN
      at every record, AF within 0.0005 (fill-tags prints more digits);
  (b) ``bcftools stats`` completes (exit 0; it stopped at the first record
      whose AN / AC disagreed before J0);
  (c) the text holds no ``=-0.0`` (FS at p = 1 before J0) and no ``nan``;
  (d) ``--region`` equals the whole run's slice (the header identical);
  (e) (removed in K1 with ``--step-size``, the Python engine's window:
      the window and thread independence is the fresh cohort's golden at
      three windows and three thread counts, test_rust_joint_call.py);
  (f) a one-site region inside a spanning deletion equals the whole run's
      record (a one-site window loaded no spanning deletion before J0);
  (g) the ref-block fill honors the gVCF's call: ``./.`` inside a sample's
      ``./.`` <NON_REF> block (``sample_counts`` 0), ``0/0`` inside a 0/0
      block, a star call the only other GT there.

The joint calls run under the default ALT cap (6) and (a) and (b) also at
``--max-alt-alleles 3``: a capped record's AC / AN are its projected GTs'.
"""

import bisect
import gzip
import os
import re
import subprocess
from typing import Dict, List

import pytest

import jc_testlib as L

pytestmark = [pytest.mark.integration]

TD = os.path.join(L.REPO_ROOT, "test_data")
DOWNLOADS_CHR21 = [
    os.path.join(TD, f"downloads/{s}/{s}.chr21.final.hard-filtered.gvcf.gz")
    for s in ("HG00099", "HG00100", "HG00101", "HG00102")
]
REGION = "chr21:10000000-12000000"
REGION_LO, REGION_HI = 10_000_000, 12_000_000


@pytest.fixture(scope="module")
def cohort(docker_stack, tmp_path_factory):
    """The D2 cohort ingested and aggregated once; the whole run's joint
    call and a region run."""
    missing = [p for p in DOWNLOADS_CHR21 if not os.path.exists(p)]
    if missing:
        pytest.skip(f"chr21 downloads not present: {missing}")
    tmp = str(tmp_path_factory.mktemp("jc_oracle"))
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    for batch in (L.GVCFS, DOWNLOADS_CHR21[:2], DOWNLOADS_CHR21[2:]):
        L.run_ingest(batch, region="chr21", threads=2)
    L.run_aggregate()
    out: Dict[str, str] = {}
    stderr: Dict[str, str] = {}
    for name, region, cap in (
        ("whole", None, None),
        ("region", REGION, None),
        ("cap3", None, 3),
    ):
        path = os.path.join(tmp, f"{name}.vcf.gz")
        proc = L.run_joint_call(path, region=region, max_alt_alleles=cap)
        out[name] = path
        stderr[name] = proc.stderr
    # (f) The first record of the whole run with a `*` ALT, alone.
    star_pos = next(
        int(f[1])
        for f in (r.split("\t", 5) for r in _split(_text(out["whole"]))[1])
        if "*" in f[4].split(",")
    )
    path = os.path.join(tmp, "one_site.vcf.gz")
    L.run_joint_call(path, region=f"chr21:{star_pos}-{star_pos}")
    out["one_site"] = path
    return {"vcf": out, "stderr": stderr, "star_pos": star_pos}


def _text(path: str) -> str:
    with gzip.open(path, "rt") as fh:
        return fh.read()


def _split(text: str):
    lines = text.splitlines()
    header = [x for x in lines if x.startswith("#")]
    records = [x for x in lines if not x.startswith("#")]
    return header, records


def _query(cmd: List[str], stdin: bytes = b"") -> List[List[str]]:
    proc = subprocess.run(cmd, input=stdin, capture_output=True, check=True)
    return [
        line.split("\t") for line in proc.stdout.decode().splitlines() if line
    ]


@pytest.mark.parametrize("name", ["whole", "cap3"])
def test_fill_tags_reproduces_ac_an_af(cohort, name):
    """At the default cap (6) and at 3: some records are capped."""
    path = cohort["vcf"][name]
    cap, counts = L.joint_call_cap_counts(cohort["stderr"][name])
    assert cap == (6 if name == "whole" else 3)
    assert counts["records_capped"] > (300 if cap == 6 else 1800)
    fmt = "%CHROM\t%POS\t%AC\t%AN\t%AF\n"
    written = _query(["bcftools", "query", "-f", fmt, path])
    filled_vcf = subprocess.run(
        ["bcftools", "+fill-tags", path, "-Ov", "--", "-t", "AC,AN,AF"],
        capture_output=True,
        check=True,
    ).stdout
    filled = _query(["bcftools", "query", "-f", fmt, "-"], filled_vcf)
    assert len(written) == len(filled) > 100_000
    bad_ac_an = []
    worst_af = 0.0
    for w, f in zip(written, filled):
        assert w[:2] == f[:2]
        if w[2] != f[2] or w[3] != f[3]:
            bad_ac_an.append((w, f))
        for a, b in zip(w[4].split(","), f[4].split(",")):
            worst_af = max(worst_af, abs(float(a) - float(b)))
        assert sum(int(c) for c in w[2].split(",")) <= int(w[3])
    assert bad_ac_an == []
    assert worst_af <= 0.0005


@pytest.mark.parametrize("name", ["whole", "cap3"])
def test_bcftools_stats_completes(cohort, name):
    proc = subprocess.run(
        ["bcftools", "stats", cohort["vcf"][name]],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "number of records:" in proc.stdout


def test_no_negative_zero_or_nan(cohort):
    text = _text(cohort["vcf"]["whole"])
    _, records = _split(text)
    body = "\n".join(records)
    assert re.search(r"[=,:]-0\.0(?![0-9])", body) is None
    assert re.search(r"(?<![A-Za-z])nan(?![A-Za-z])", body, re.I) is None
    # FS at p = 1 is written 0.0 (it was -0.0 at 40,395 records before J0).
    assert "FS=0.0;" in body


def test_inbreeding_coeff_header_present(cohort):
    header, _ = _split(_text(cohort["vcf"]["whole"]))
    assert any(
        h.startswith("##INFO=<ID=InbreedingCoeff,Number=1,Type=Float,")
        for h in header
    )


def test_region_equals_slice_of_whole_run(cohort):
    whole_header, whole = _split(_text(cohort["vcf"]["whole"]))
    region_header, region = _split(_text(cohort["vcf"]["region"]))
    assert region_header == whole_header
    expected = [
        r for r in whole if REGION_LO <= int(r.split("\t", 2)[1]) <= REGION_HI
    ]
    assert len(expected) > 1000
    assert region == expected


def test_catalog_cross_check_line(cohort):
    err = cohort["stderr"]["whole"]
    lines = err.splitlines()
    i = next(
        k for k, x in enumerate(lines) if x.startswith("Completed! Total")
    )
    assert lines[i + 1].startswith("Catalog cross-check: ")
    # A fresh, fully aggregated cohort: no difference without a reason
    # (the fixture's 3 come from calls written ./. at a <NON_REF> allele).
    assert not any(x.startswith("  other: ") for x in lines[i + 1 :])


def test_one_site_region_keeps_the_spanning_deletion(cohort):
    pos = cohort["star_pos"]
    whole_header, whole = _split(_text(cohort["vcf"]["whole"]))
    header, records = _split(_text(cohort["vcf"]["one_site"]))
    assert header == whole_header
    expected = [r for r in whole if int(r.split("\t", 2)[1]) == pos]
    assert len(expected) == 1
    assert "*" in expected[0].split("\t")[4].split(",")
    assert records == expected


def _gvcf_blocks(path: str, gt: str):
    """The gVCF's blocks with an END and the given GT, as sorted
    (start, end) pairs, 1-based inclusive."""
    rows = _query(
        [
            "bcftools",
            "query",
            "-f",
            "%POS\t%INFO/END\t[%GT]\n",
            "-r",
            "chr21",
            path,
        ]
    )
    return sorted((int(p), int(e)) for p, e, g in rows if e != "." and g == gt)


def _inside(blocks, pos: int) -> bool:
    i = bisect.bisect_right(blocks, (pos, float("inf"))) - 1
    return i >= 0 and blocks[i][0] <= pos <= blocks[i][1]


def test_ref_block_fill_honors_the_gvcf_call(cohort):
    """Inside one of HG00099's ./. blocks the record writes ./. for it and
    inside one of its 0/0 blocks 0/0, unless the sample is a `*` carrier
    there: a star call takes precedence over the ref-block fill (DRAGEN
    writes the positions a sample's own deletion covers as blocks of their
    own, and the ported star rule carries `*` from any non-ref call at a
    spanning deletion's site), so the only other GT allowed is a star call,
    0/s or s/s with s the record's `*` index (787 of 58,255 in-block
    records on this cohort)."""
    gvcf = DOWNLOADS_CHR21[0]
    no_call = _gvcf_blocks(gvcf, "./.")
    hom_ref = _gvcf_blocks(gvcf, "0/0")
    header, records = _split(_text(cohort["vcf"]["whole"]))
    col = header[-1].split("\t").index("HG00099")
    n_no_call = n_hom_ref = n_star = 0
    bad = []
    for r in records:
        f = r.split("\t")
        pos, gt = int(f[1]), f[col].split(":")[0]
        alts = f[4].split(",")
        star_gts = set()
        if "*" in alts:
            s = alts.index("*") + 1
            star_gts = {f"0/{s}", f"{s}/{s}"}
        if _inside(no_call, pos):
            n_no_call += 1
            if gt in star_gts:
                n_star += 1
            elif gt != "./.":
                bad.append(("./.", pos, gt))
        elif _inside(hom_ref, pos):
            n_hom_ref += 1
            if gt in star_gts:
                n_star += 1
            elif gt != "0/0":
                bad.append(("0/0", pos, gt))
    assert bad == []
    assert n_no_call >= 200
    assert n_hom_ref >= 1000
    assert n_star >= 100
