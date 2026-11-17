# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Print the expectations of the joint caller's slice fixture under the ALT
cap (``--max-alt-alleles``) and under a sample subset (``--samples``):
for each cap N, the SHA-256, record and
byte counts of ``tests/vcf_project.project(<expected.vcf.gz's text>, N)``
(the test side's projection of the frozen uncapped expectation, the cap's
header line included) and the projection's four counts, as the Rust
constants of ``run.rs``'s ``slice_tests::CAP_DIGESTS``; then for each
(subset, cap) of SUBSETS the same of
``project(project_samples(<the text>, subset), cap)`` (cap 0: the subset
alone, no cap line), as ``slice_tests::SUBSET_DIGESTS``; then (the trim
and ``--min-ac``) for each (subset, trim, cap, min-ac) of OPTIONS the same of the chain
``project_min_ac(project(project_trim(project_samples(<the text>,
subset)), cap), min_ac)``, each step only when set, with the trim's two
counts, the cap's four and the records left out, as
``slice_tests::OPTION_DIGESTS``. No store is read and no file is written;
``expected.vcf.gz`` is the checked-in one. Since the sample subset both
projections apply the one PL rule (a PL is shifted only when its source
held a 0); the cap's digests did not move under it.

Run from the repo root with any Python 3:
    python rust/tools/gen_jc_slice_cap_digests.py [N ...]   (default: 1 3)
"""

import argparse
import gzip
import hashlib
import os
import sys

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(REPO, "tests"))

import vcf_project  # noqa: E402

EXPECTED = os.path.join(
    REPO, "rust", "mosaic-joint-call", "tests", "data", "jc_slice",
    "expected.vcf.gz",
)  # fmt: skip


# (subset, cap) pairs: two samples of different batches, two of one batch,
# one sample, and a subset under a cap.
SUBSETS = (
    (("HG00100", "HG00245"), 0),
    (("HG00101", "HG00102"), 0),
    (("HG00099",), 0),
    (("HG00100", "HG00245"), 3),
)

# The trim and --min-ac: (subset or None, trim, cap, min-ac): each option alone, both, both
# under the cap, and under subsets.
OPTIONS = (
    (None, True, 0, 0),
    (None, False, 0, 1),
    (None, False, 0, 2),
    (None, True, 0, 1),
    (None, True, 3, 2),
    (("HG00100", "HG00245"), True, 0, 1),
    (("HG00099",), True, 6, 1),
    (("HG00101", "HG00102"), False, 1, 1),
)

COUNT_KEYS = (
    "records_capped",
    "genotypes_no_called",
    "dropped_alt_observations",
    "kept_alt_observations_lost",
)


def _digest_fields(out: str, counts: dict) -> None:
    data = out.encode()
    records = sum(1 for x in out.split("\n") if x and not x.startswith("#"))
    print(f'        sha256: "{hashlib.sha256(data).hexdigest()}",')
    print(f"        records: {records},")
    print(f"        bytes: {len(data)},")
    print(
        "        counts: ["
        + ", ".join(str(counts[k]) for k in COUNT_KEYS)
        + "],"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("caps", nargs="*", type=int, default=[1, 3])
    args = ap.parse_args()
    with gzip.open(EXPECTED, "rt") as fh:
        text = fh.read()
    print("const CAP_DIGESTS: [CapDigest; %d] = [" % len(args.caps))
    for n in args.caps:
        out, counts = vcf_project.project(text, n)
        print("    CapDigest {")
        print(f"        cap: {n},")
        _digest_fields(out, counts)
        print("    },")
    print("];")
    print()
    print("const SUBSET_DIGESTS: [SubsetDigest; %d] = [" % len(SUBSETS))
    for names, n in SUBSETS:
        out, counts = vcf_project.project(
            vcf_project.project_samples(text, names), n
        )
        print("    SubsetDigest {")
        print("        names: &[" + ", ".join(f'"{x}"' for x in names) + "],")
        print(f"        cap: {n},")
        _digest_fields(out, counts)
        print("    },")
    print("];")
    print()
    print("const OPTION_DIGESTS: [OptionDigest; %d] = [" % len(OPTIONS))
    for names, trim, n, k in OPTIONS:
        out = text
        if names is not None:
            out = vcf_project.project_samples(out, names)
        tc = {"records_trimmed": 0, "alts_dropped": 0}
        if trim:
            out, tc = vcf_project.project_trim(out)
        out, counts = vcf_project.project(out, n)
        out, mc = vcf_project.project_min_ac(out, k)
        print("    OptionDigest {")
        print(
            "        names: &["
            + ", ".join(f'"{x}"' for x in (names or ()))
            + "],"
        )
        print(f"        trim: {'true' if trim else 'false'},")
        print(f"        cap: {n},")
        print(f"        min_ac: {k},")
        _digest_fields(out, counts)
        print(
            f"        trim_counts: [{tc['records_trimmed']}, "
            f"{tc['alts_dropped']}],"
        )
        print(f"        left_out: {mc['records_left_out']},")
        print("    },")
    print("];")


if __name__ == "__main__":
    main()
