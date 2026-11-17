# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests of the allele encoding the ``bcftools query`` truth uses
(``tests/truth_alleles.py``, the test-side restatement of the encoding
rule).

Until the earlier Python ingest worker was removed these case tables tested the Python ingest worker's encoder; the
production encoder is the Rust worker's (``rust/mosaic-core/src/alleles.rs``,
whose ``trim_python_cases`` / ``format_python_cases`` carry the same tables,
and ``ref_block_python_cases`` the ``is_ref_block`` cases this file held).
``extract_site_info``'s cases are ``rust/mosaic-ingest-worker/src/sites.rs``'s
``esi_*`` tests; the encode / decode round trip is
``rust/mosaic-joint-call/src/call.rs`` ``encode_decode_roundtrip`` (K1).
"""

import pytest

from truth_alleles import format_alt_allele, trim_shared_suffix

pytestmark = pytest.mark.unit


# ── trim_shared_suffix ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ref,alt,expected",
    [
        ("A", "T", ("A", "T")),  # no shared suffix
        ("AT", "GT", ("A", "G")),  # shared 'T'
        ("ATG", "ATG", ("", "")),  # fully shared
        ("CAT", "AT", ("C", "")),  # shared 'AT'
        ("", "A", ("", "A")),  # empty ref
        ("A", "", ("A", "")),  # empty alt
    ],
)
def test_trim_shared_suffix(ref, alt, expected):
    assert trim_shared_suffix(ref, alt) == expected


# ── format_alt_allele ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "ref,alt,expected",
    [
        ("A", "<NON_REF>", "*"),  # symbolic non-ref
        ("A", "T", "xT"),  # SNV
        ("AT", "GT", "xG"),  # SNV after suffix trim
        ("A", "AT", "+AT"),  # insertion
        ("AT", "A", "-AT"),  # deletion
        ("ATG", "A", "-ATG"),  # longer deletion
    ],
)
def test_format_alt_allele(ref, alt, expected):
    assert format_alt_allele(ref, alt) == expected
