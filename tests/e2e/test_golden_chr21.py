# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Golden end-to-end regression: full pipeline output vs the gVCFtyper truth VCF.

Runs init -> ingest -> aggregate -> joint_call -> bcftools trim+norm -> vcfdiff over a
chr21 window and asserts the vcfdiff summary against a recorded baseline. Replaces the
stale, assertion-free test_site.py (which also invoked a non-existent ``ingest_gvcf.py
init``; the real initializer is ``init_db.py``).

vcfdiff summary line:  ``summary < <extra> > <missing> +/- <diff> = <match>``
  * extra/missing == 0  is the core correctness property (exact variant set);
  * diff is dominated by last-ULP REAL differences (QUAL etc., not in the compare
    keys) so it is bounded, not pinned;
  * extra+missing+diff+match (the compared-variant universe) is pinned.

Baselines are per region (a fast subset by default; the full chr21:1-30M run is the
documented `< 0 > 0 +/- 118 = 56654`). To re-record after an intended change, run with
MOSAIC_GOLDEN_RECORD=1 and copy the printed tuple into BASELINES.
"""

import os
import re
import subprocess
import sys

import pytest

import jc_testlib as L

pytestmark = pytest.mark.e2e

# region -> (extra, missing, diff, match) from a known-good run.
BASELINES = {
    "chr21:10000000-12000000": (0, 0, 11, 3486),
}
REGION = "chr21:10000000-12000000"
# Allow a little slack on the float-ULP-driven diff count across platforms.
DIFF_SLACK = 5

TRUTH = os.path.join(
    L.REPO_ROOT,
    "test_data/gvcftyper_output/gvcftyper_n2_chr21-1-30000000-2.vcf.gz",
)


def _normalize(in_vcf, out_vcf):
    cmd = (
        f"bcftools view --trim-alt-alleles {in_vcf} | "
        f"bcftools norm --atom-overlaps . -m +any --write-index=tbi "
        f"-Oz -o {out_vcf} -"
    )
    subprocess.run(cmd, shell=True, check=True, capture_output=True, text=True)


def _vcfdiff_summary(query, truth):
    proc = subprocess.run(
        [
            sys.executable,
            "vcfdiff.py",
            query,
            truth,
            "level=4",
            "qualkeys=",
            "formatkeys=GT,PL,DP",
            "infokeys=DP,AC,AF,AN",
            "header=0",
        ],
        cwd=L.REPO_ROOT,
        env=L.test_env(),
        capture_output=True,
        text=True,
    )
    text = proc.stderr + "\n" + proc.stdout
    m = re.search(r"summary < (\d+) > (\d+) \+/- (\d+) = (\d+)", text)
    assert m, f"could not parse vcfdiff summary from:\n{text[-3000:]}"
    return tuple(int(x) for x in m.groups())  # (extra, missing, diff, match)


def _run_golden(region, work):
    L.reset_state()
    gvcfs, truth = [], os.path.join(work, "truth.vcf.gz")
    for j, src in enumerate(L.GVCFS):
        dst = os.path.join(work, f"s{j}.g.vcf.gz")
        L.subset_gvcf(src, region, dst)
        gvcfs.append(dst)
    L.subset_gvcf(TRUTH, region, truth)

    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(gvcfs, region=region, threads=2)
    L.run_aggregate(fan_in=100, threads=2)
    out = os.path.join(work, "joint.vcf.gz")
    # The default ALT cap (6; K3): print its closing line (the records it
    # capped).
    proc = L.run_joint_call(out)
    print(
        f"\njoint call (default cap): {L.joint_call_cap_counts(proc.stderr)}"
    )
    norm = os.path.join(work, "joint_norm.vcf.gz")
    _normalize(out, norm)
    return _vcfdiff_summary(norm, truth)


def test_golden_chr21_subset(docker_stack, tmp_path):
    extra, missing, diff, match = _run_golden(REGION, str(tmp_path))
    print(f"\nvcfdiff summary < {extra} > {missing} +/- {diff} = {match}")

    if os.environ.get("MOSAIC_GOLDEN_RECORD"):
        pytest.skip(
            f"RECORD mode: baseline for {REGION} = "
            f"({extra}, {missing}, {diff}, {match})"
        )

    b_extra, b_missing, b_diff, b_match = BASELINES[REGION]
    # Exact variant set: no extra, no missing (the documented < 0 > 0 property).
    assert extra == b_extra, f"extra variants regressed: {extra} != {b_extra}"
    assert (
        missing == b_missing
    ), f"missing variants regressed: {missing} != {b_missing}"
    # The compared-variant universe is pinned; diff/match may trade a few at the ULP.
    assert diff + match == b_diff + b_match
    assert (
        diff <= b_diff + DIFF_SLACK
    ), f"field differences grew: {diff} > {b_diff}"
