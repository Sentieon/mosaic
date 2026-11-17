# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the per-batch genotype matrix's blob format and reader.

The blob encoding at both widths (one byte per slot, or two at a site whose
codes exceed 127; ``encode_gt_blob`` stays in ``mosaic/genotypes.py`` as the
decoder's inverse), ``dosage``, and the reader's int16 matrix over rows of
mixed widths, its refusals and the tombstone column drop.

The builder tests this file held earlier (the GQ rule, ``parse_gt``, the sample_qc
classification, ``GenotypeRowBuilder``'s slot rules, the by-string remap,
the blob widths at 127 / 128 / above 32,767 alleles, the same-site seeded
allele list) tested the Python ingest worker; they are ``cargo test``s of
the Rust worker with the same inputs and expected values
(``rust/mosaic-ingest-worker/src/matrix.rs``, ``qc.rs``, ``sites.rs``).
"""

import numpy as np
import pytest

from mosaic.genotypes import (
    GT_NO_CALL,
    MAX_SITE_ALLELES,
    UNRESOLVED,
    GenotypeMatrix,
    decode_gq_blob,
    decode_gt_blob,
    dosage,
    encode_gt_blob,
    read_genotype_matrix,
)

pytestmark = pytest.mark.unit


# ── blob encoding and the reader's dosage ────────────────────────────────────


def test_blob_round_trip():
    codes = [-1, 0, 1, 2, 127, -1]
    blob = encode_gt_blob(codes)
    assert blob == bytes([0xFF, 0, 1, 2, 127, 0xFF])
    assert decode_gt_blob(blob, 6).tolist() == codes
    assert decode_gt_blob(blob, 6).dtype == np.int8
    assert decode_gq_blob(bytes([0, 99, 255])).tolist() == [0, 99, 255]
    assert GT_NO_CALL == -1
    with pytest.raises(OverflowError):
        encode_gt_blob([128])


def test_wide_blob_round_trip():
    codes = [-1, 0, 128, 300, 32767, -1]
    blob = encode_gt_blob(codes, wide=True)
    assert len(blob) == 12
    assert blob[:2] == b"\xff\xff" and blob[4:6] == bytes([128, 0])
    assert blob[8:10] == bytes([0xFF, 0x7F])  # little-endian
    got = decode_gt_blob(blob, 6)
    assert got.dtype == np.int16 and got.tolist() == codes
    assert decode_gt_blob(encode_gt_blob([1, 2], wide=True), 2).tolist() == [
        1,
        2,
    ]
    with pytest.raises(ValueError, match="1 or 2 bytes"):
        decode_gt_blob(bytes(9), 3)
    with pytest.raises(OverflowError):
        encode_gt_blob([32768], wide=True)


def test_dosage():
    m = GenotypeMatrix(
        chromosome_index=0,
        positions=np.array([10], dtype=np.int32),
        alleles=[["A", "xT", "*"]],
        sample_ids=np.arange(5, dtype=np.int32),
        batch_ids=np.zeros(5, dtype=np.int32),
        gt_1=np.array([[0, 1, -1, UNRESOLVED, 1]], dtype=np.int16),
        gt_2=np.array([[1, 1, -1, UNRESOLVED, -1]], dtype=np.int16),
        gq=np.zeros((1, 5), dtype=np.uint8),
    )
    assert dosage(m, 0, "xT").tolist() == [1, 2, -1, -1, 1]
    assert dosage(m, 0, "-AT").tolist() == [0, 0, -1, -1, 0]


# ── the reader hides removed samples' columns (E3) ──────────────────────────


class _Cur:
    """Answers the reader's three queries: the contig, the batches' slot
    orders (two ingest batches) and the removed sample ids."""

    def __init__(self, removed):
        self.removed = removed
        self.rows = []

    def execute(self, sql, params=None):
        if "FROM contigs" in sql:
            self.rows = [("chr21",)]
        elif "FROM ingestion_progress" in sql:
            assert "kind = 'ingest'" in sql  # a removal batch has no slots
            self.rows = [(1, [11, 12], "chr21"), (2, [13], None)]
        elif "FROM samples" in sql:
            self.rows = [(s,) for s in self.removed]
        else:
            raise AssertionError(sql)

    def fetchone(self):
        return self.rows[0]

    def fetchall(self):
        return self.rows

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _Conn:
    def __init__(self, removed):
        self.removed = removed

    def cursor(self):
        return _Cur(self.removed)


class _Catalog:
    def __init__(self, table):
        self.table = table

    def load_table(self, name):
        table = self.table

        class _T:
            def scan(self, row_filter=None):
                class _S:
                    def to_arrow(self):
                        return table

                return _S()

        return _T()


def _bgt_table():
    import pyarrow as pa

    from sample_slices import BATCH_GENOTYPES_ARROW_SCHEMA

    rows = [
        (1, 0, 50, 2, ["A", "xT", "*"], [0, 1], [1, 1], [30, 40]),
        (2, 0, 50, 1, ["A", "xG", "*"], [1], [1], [50]),
    ]
    return pa.Table.from_pydict(
        {
            "batch_id": [r[0] for r in rows],
            "chromosome_index": [r[1] for r in rows],
            "position": [r[2] for r in rows],
            "n_samples": [r[3] for r in rows],
            "alleles": [r[4] for r in rows],
            "gt_1": [encode_gt_blob(r[5]) for r in rows],
            "gt_2": [encode_gt_blob(r[6]) for r in rows],
            "gq": [bytes(r[7]) for r in rows],
        },
        schema=BATCH_GENOTYPES_ARROW_SCHEMA,
    )


def test_reader_drops_removed_samples_columns():
    cat = _Catalog(_bgt_table())
    full = read_genotype_matrix(
        cat, _Conn([12]), 0, 1, 100, include_removed=True
    )
    assert full.sample_ids.tolist() == [11, 12, 13]
    m = read_genotype_matrix(cat, _Conn([12]), 0, 1, 100)
    assert m.sample_ids.tolist() == [11, 13]
    assert m.batch_ids.tolist() == [1, 2]
    assert m.alleles == full.alleles  # the batch's row still lists xT
    assert m.gt_1.tolist() == [[0, 2]] and m.gt_2.tolist() == [[1, 2]]
    assert m.gq.tolist() == [[30, 50]]
    assert dosage(m, 0, "xG").tolist() == [0, 2]
    # Nothing removed: every column.
    none = read_genotype_matrix(cat, _Conn([]), 0, 1, 100)
    assert none.sample_ids.tolist() == [11, 12, 13]


# ── the reader over rows of mixed blob widths (F1) ───────────────────────────


def _bgt_rows_table(rows):
    """A batch_genotypes Arrow table from ``(batch, pos, n, alleles, g1, g2,
    gq)`` rows, each blob encoded wide iff a code exceeds 127."""
    import pyarrow as pa

    from sample_slices import BATCH_GENOTYPES_ARROW_SCHEMA

    def enc(codes):
        return encode_gt_blob(codes, wide=max(codes) > 127)

    return pa.Table.from_pydict(
        {
            "batch_id": [r[0] for r in rows],
            "chromosome_index": [0] * len(rows),
            "position": [r[1] for r in rows],
            "n_samples": [r[2] for r in rows],
            "alleles": [r[3] for r in rows],
            "gt_1": [enc(r[4]) for r in rows],
            "gt_2": [enc(r[5]) for r in rows],
            "gq": [bytes(r[6]) for r in rows],
        },
        schema=BATCH_GENOTYPES_ARROW_SCHEMA,
    )


def test_reader_mixed_widths_union_above_128():
    """Batch 1 (slots 11, 12) lists 100 alts at site 50 in a one-byte row;
    batch 2 (slot 13) lists 140 others in a two-byte row. The union holds
    241 alleles; the int16 matrix indexes all of them. Batch 1's narrow row
    at site 60 sits beside batch 2's wide one."""
    a_alts = [f"+A{i}" for i in range(100)]
    b_alts = [f"+B{i}" for i in range(140)]
    rows = [
        (1, 50, 2, ["A"] + a_alts, [0, 100], [5, -1], [30, 40]),
        (2, 50, 1, ["A"] + b_alts, [140], [1], [50]),
        (1, 60, 2, ["C", "xT"], [0, 1], [1, 1], [10, 20]),
        (2, 60, 1, ["C"] + [f"+C{i}" for i in range(129)], [129], [0], [9]),
    ]
    tbl = _bgt_rows_table(rows)
    assert [len(b) for b in tbl.column("gt_1").to_pylist()] == [2, 2, 2, 2]
    m = read_genotype_matrix(_Catalog(tbl), _Conn([]), 0, 1, 100)
    assert m.gt_1.dtype == np.int16 and m.gt_2.dtype == np.int16
    assert m.positions.tolist() == [50, 60]
    assert m.alleles[0] == ["A"] + a_alts + b_alts
    assert m.gt_1[0].tolist() == [0, 100, 1 + 100 + 139]
    assert m.gt_2[0].tolist() == [5, -1, 1 + 100]
    assert m.alleles[0][m.gt_1[0, 2]] == "+B139"
    assert m.gt_1[1].tolist() == [0, 1, 2 + 128]
    assert m.gt_2[1].tolist() == [1, 1, 0]
    assert m.alleles[1][130] == "+C128"
    assert m.gq.tolist() == [[30, 40, 50], [10, 20, 9]]
    assert dosage(m, 0, "+B139").tolist() == [0, 0, 1]
    assert dosage(m, 0, "+A99").tolist() == [0, 1, 0]
    assert dosage(m, 0, "+A4").tolist() == [1, 0, 0]
    assert dosage(m, 1, "+C128").tolist() == [0, 0, 1]


def test_reader_union_above_32767_raises():
    half = MAX_SITE_ALLELES // 2 + 1
    rows = [
        (
            1,
            50,
            2,
            ["A"] + [f"+A{i}" for i in range(half)],
            [0, 0],
            [0, 0],
            [1, 1],
        ),
        (2, 50, 1, ["A"] + [f"+B{i}" for i in range(half)], [0], [0], [1]),
    ]
    with pytest.raises(ValueError, match="0:50"):
        read_genotype_matrix(
            _Catalog(_bgt_rows_table(rows)), _Conn([]), 0, 1, 100
        )


def test_reader_rejects_a_blob_of_another_width():
    tbl = _bgt_rows_table([(1, 50, 2, ["A", "xT"], [0, 1], [1, 1], [1, 1])])
    import pyarrow as pa

    tbl = tbl.set_column(
        tbl.schema.get_field_index("gt_1"),
        tbl.schema.field("gt_1"),
        pa.array([bytes(3)], type=tbl.schema.field("gt_1").type),
    )
    with pytest.raises(ValueError, match="1 or 2 bytes"):
        read_genotype_matrix(_Catalog(tbl), _Conn([]), 0, 1, 100)
