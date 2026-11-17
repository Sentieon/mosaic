# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The allele encoding of the ``bcftools query`` truth.

``test_genotype_matrix.py``, ``test_genotype_matrix_wide.py`` and
``test_local_alleles.py`` compare what the ingest worker wrote against
``bcftools query`` over the same gVCFs, so they need the lake's allele
encoding of a VCF (REF, ALT) pair. This module **is the truth's encoder: a
restatement of the encoding rule** (the allele encoding convention in
docs/ARCHITECTURE.md), and it must not import worker code: until the
earlier Python ingest worker was removed the truth called its own ``format_alt_allele``, so it was not
independent of the code it checked. The production encoder is the Rust
worker's (``rust/mosaic-core/src/alleles.rs``, whose ``trim_python_cases``
/ ``format_python_cases`` carry the same case tables as
``tests/unit/test_allele_encoding.py``, which pins this module).

The rule: ``<NON_REF>`` is ``*``; otherwise REF and ALT lose their shared
suffix, and the trimmed pair is ``x`` + ALT when the lengths are equal (an
SNV or MNP), ``+`` + ALT when ALT is longer (an insertion), ``-`` + REF when
REF is longer (a deletion).
"""

from typing import Tuple


def trim_shared_suffix(ref: str, alt: str) -> Tuple[str, str]:
    """REF and ALT without their common suffix (an empty side: as is)."""
    n = 0
    while n < min(len(ref), len(alt)) and ref[-1 - n] == alt[-1 - n]:
        n += 1
    return (ref[: len(ref) - n], alt[: len(alt) - n]) if n else (ref, alt)


def format_alt_allele(ref: str, alt: str) -> str:
    """The lake's encoding of the VCF ALT ``alt`` against ``ref``."""
    if alt == "<NON_REF>":
        return "*"
    t_ref, t_alt = trim_shared_suffix(ref, alt)
    if len(t_ref) == len(t_alt):
        return "x" + t_alt
    if len(t_alt) > len(t_ref):
        return "+" + t_alt
    return "-" + t_ref
