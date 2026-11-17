# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the derived allele catalog's deletion span.

``allele_del_range``: the deletion span the aggregate finalize derives from a
catalog allele string, for ordinary deletions the record formula
``[pos, pos + len(REF) - len(ALT) + 1)``, and NULL for insertions, SNPs, MNPs
and ``*``; the Arrow-vectorized form against the scalar helper.

The catalog's inputs are frozen literals: earlier each case ran the Python
ingest worker's site path
(``grouper`` -> ``collect_variant_group_data`` -> ``merge_site_alleles``) on
a record at 0-based 1000 and compared against the span it recorded; the
literals below are that path's output, checked equal before it was deleted.
The worker side (the carrier row's ``del_start`` / ``del_end``) is
``rust/mosaic-ingest-worker/src/sites.rs::del_range_of_the_carried_allele``;
``bulk_upload``'s rows and partials are
``partials.rs::bulk_upload_writes_local_indices_and_every_allele``, and the
dictionary-encoded ``sample_site_info.alleles`` column is
``tests/unit/test_worker_bin_sites.py``.
"""

import pyarrow as pa
import pytest

from mosaic.aggregate import allele_del_range, allele_del_range_text

pytestmark = pytest.mark.unit


def _text(span):
    return None if span is None else f"[{span[0]},{span[1]})"


# ── allele_del_range vs the record's span ────────────────────────────────────


@pytest.mark.parametrize(
    "ref,alt,allele,span",
    [
        ("AT", "A", "-AT", (1001, 1003)),  # 1-base deletion
        ("ATG", "A", "-ATG", (1001, 1004)),  # 2-base deletion
        ("ATGC", "AGC", "-AT", (1001, 1003)),  # suffix-trimmed: == AT>A
        ("ATTTTTTTTTTTG", "A", "-ATTTTTTTTTTTG", (1001, 1014)),  # long
        ("A", "AT", "+AT", None),  # insertion: no span
        ("A", "T", "xT", None),  # SNP
        ("AT", "GC", "xGC", None),  # MNP
        ("A", "<NON_REF>", "*", None),  # the symbolic non-ref allele
    ],
)
def test_del_range_matches_ingest_span(ref, alt, allele, span):
    # The site at 1-based 1001; ``span`` is what ingest recorded there.
    pos_1b = 1001
    assert allele_del_range(pos_1b, allele) == span
    if allele.startswith("-"):
        # the record formula on the untrimmed record
        assert span == (pos_1b, pos_1b + len(ref) - len(alt) + 1)


def test_star_has_no_span_even_after_a_long_ref():
    """The catalog gives ``*`` no span. Ingest's old record formula
    (len(REF) - len("<NON_REF>")) gave it one whenever the record's REF was
    longer than nine bases (at 1-based 501: ``[501,506)``); that span was
    never a deletion, and the Rust worker computes none."""
    assert allele_del_range(501, "*") is None


def test_arrow_form_equals_scalar_helper():
    positions = [1, 10, 10, 99, 2**31 - 20, 7, 7, 42]
    alleles = ["A", "xT", "-ATG", "+AT", "-ACGTACGTAC", "*", "xGC", "-A"]
    got = allele_del_range_text(
        pa.array(positions, pa.int32()), pa.array(alleles)
    ).to_pylist()
    want = [_text(allele_del_range(p, a)) for p, a in zip(positions, alleles)]
    assert got == want
    assert want[2] == "[10,13)" and want[4] == f"[{2**31 - 20},{2**31 - 10})"
