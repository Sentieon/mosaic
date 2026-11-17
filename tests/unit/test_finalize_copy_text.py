# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the finalize's Arrow -> PostgreSQL COPY text encoder.

The vectorized finalize renders each Arrow chunk with ``pyarrow.csv`` instead of
formatting Python tuples. The reference is the pre-vectorization row encoders
(``_pg_copy_allele`` / ``_pg_copy_site`` / ``_pg_copy_ref`` at commit e2d0fc7),
kept here verbatim (renamed ``_old_*``) and fed the rows the old finalize loop
built for the same chunk. Every integer, string and array field must match
byte-for-byte. Float fields (qual / mqsq_1) are written as Arrow's shortest
round-trip decimal (``15``) where Python wrote ``repr`` (``15.0``): the two
strings denote the same decimal number, so PostgreSQL parses the same REAL;
the test asserts that equality of value.

Since E3 the finalize first maps each chunk to its PostgreSQL rows
(``spec.pg_rows``): a site whose net ``n_records`` is 0 and an allele whose
net ``n_listed`` is 0 are dropped (only removed samples' records had them),
and site_info's fixed-point gtc becomes its integer part (an arithmetic shift
by GTC_FIXED_BITS) in PG's BIGINT[]; the
agg_* mirror keeps the chunk itself. Tested at the end.
"""

import io

import pyarrow as pa
import pytest

from mosaic.aggregate import SPECS, _copy_arrow, _final_rows, _pg_copy_text

pytestmark = pytest.mark.unit


class _FakeCursor:
    """Captures what an encoder hands to psycopg2."""

    def __init__(self):
        self.calls = []

    def copy_from(self, buf, table, columns):
        self.calls.append(("copy_from", table, tuple(columns), buf.read()))

    def copy_expert(self, sql, buf):
        self.calls.append(("copy_expert", sql, buf.read()))


# ── the pre-vectorization encoders (reference; e2d0fc7) ──────────────────────


def _old_pg_copy_allele(cur, rows):
    buf = io.StringIO()
    for ci, pos, allele, cnt in rows:
        buf.write(f"{ci}\t{pos}\t{allele}\t{cnt}\n")
    buf.seek(0)
    cur.copy_from(
        buf, "tmp_allele_counts", columns=("chrom_idx", "pos", "allele", "cnt")
    )


def _old_pg_copy_site(cur, rows):
    buf = io.StringIO()
    for ci, pos, qual, dp, m1, m2, sb, gtc in rows:
        sb_s = "{" + ",".join(str(int(x)) for x in sb) + "}"
        gtc_s = "{" + ",".join(str(int(x)) for x in gtc) + "}"
        buf.write(f"{ci}\t{pos}\t{qual}\t{dp}\t{m1}\t{m2}\t{sb_s}\t{gtc_s}\n")
    buf.seek(0)
    cur.copy_from(
        buf,
        "site_info",
        columns=(
            "chromosome_index",
            "position",
            "qual",
            "dp",
            "mqsq_1",
            "mqsq_2",
            "sb",
            "gtc",
        ),
    )


def _old_pg_copy_ref(cur, rows):
    buf = io.StringIO()
    for ci, pos, dp, sc in rows:
        buf.write(f"{ci}\t{pos}\t{dp}\t{sc}\n")
    buf.seek(0)
    cur.copy_from(
        buf,
        "ref_blocks",
        columns=("chromosome_index", "position", "dp", "sample_counts"),
    )


def _old(encoder, rows):
    cur = _FakeCursor()
    encoder(cur, rows)
    _, table, columns, text = cur.calls[0]
    return table, columns, text


# The finalize COPYs into a per-chromosome generation table named like this.
_TARGET = "x_c1_r7"


def _new(spec, staged):
    """The new path: staging rows -> _final_rows -> pg_rows -> COPY text via
    a cursor."""
    chunk = _final_rows(spec, staged)
    cur = _FakeCursor()
    spec.pg_copy(cur, _TARGET, spec.pg_rows(chunk))
    _, sql, text = cur.calls[0]
    return sql, text.decode()


def _assert_same_copy_text(old_text, new_text, float_fields=()):
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    assert len(old_lines) == len(new_lines)
    for old_line, new_line in zip(old_lines, new_lines):
        old_f = old_line.split("\t")
        new_f = new_line.split("\t")
        assert len(old_f) == len(new_f)
        for i, (a, b) in enumerate(zip(old_f, new_f)):
            if i in float_fields:
                assert float(a) == float(b), (i, a, b)
            else:
                assert a == b, (i, a, b)


def _copy_sql(table, columns):
    return f"COPY {table} ({', '.join(columns)}) FROM STDIN"


# ── per-metric round trips ───────────────────────────────────────────────────


def test_allele_copy_text_matches_old():
    """The alleles catalog row: the old key / count fields byte-for-byte,
    plus the derived del_range as INT4RANGE text (``\\N`` = NULL) between
    the allele and the count."""
    staged = pa.table(
        {
            "chromosome_index": pa.array([1, 1, 2, 3, 3, 4], pa.int32()),
            "position": pa.array(
                [10, 10, 99, 2**31 - 1, 7, 2**31 - 5], pa.int32()
            ),
            "allele": ["xT", "-ATG", "+AT", "*", "A", "-AT"],
            "count": pa.array([3, None, 5_000_000_000, 0, 0, 1], pa.int64()),
            "n_listed": pa.array([1, 2, 3, 4, 5, 6], pa.int64()),
        }
    )
    # The old finalize loop built (ci, pos, allele, int(count or 0)).
    rows = [
        (
            r["chromosome_index"],
            r["position"],
            r["allele"],
            int(r["count"] or 0),
        )
        for r in staged.to_pylist()
    ]
    _, _, old_text = _old(_old_pg_copy_allele, rows)
    sql, new_text = _new(SPECS["allele_counts"], staged)
    # The target is an alleles generation table: the keys, the derived
    # del_range, then the count.
    assert sql == _copy_sql(
        _TARGET,
        ("chromosome_index", "position", "allele", "del_range", "count"),
    )
    old_lines = old_text.splitlines()
    new_lines = new_text.splitlines()
    assert len(new_lines) == len(old_lines)
    del_ranges = []
    for old_line, new_line in zip(old_lines, new_lines):
        fields = new_line.split("\t")
        assert len(fields) == 5
        assert "\t".join(fields[:3] + fields[4:]) == old_line
        del_ranges.append(fields[3])
    assert del_ranges == [
        "\\N",
        "[10,13)",
        "\\N",
        "\\N",
        "\\N",
        f"[{2**31 - 5},{2**31 - 3})",
    ]


def test_ref_copy_text_matches_old():
    staged = pa.table(
        {
            "chromosome_index": pa.array([1, 1, 2], pa.int32()),
            "position": pa.array([5, 6, 7], pa.int32()),
            "dp": pa.array([30, None, 9_000_000_000], pa.int64()),
            "sample_counts": pa.array([None, 2, 3], pa.int64()),
        }
    )
    rows = [
        (
            r["chromosome_index"],
            r["position"],
            int(r["dp"] or 0),
            int(r["sample_counts"] or 0),
        )
        for r in staged.to_pylist()
    ]
    _, columns, old_text = _old(_old_pg_copy_ref, rows)
    sql, new_text = _new(SPECS["ref_blocks"], staged)
    assert sql == _copy_sql(_TARGET, columns)
    assert new_text == old_text


def test_site_copy_text_matches_old():
    staged = pa.table(
        {
            "chromosome_index": pa.array([1, 1, 2, 2], pa.int32()),
            "position": pa.array([100, 101, 7, 8], pa.int32()),
            "qual": pa.array(
                [15.0, 0.1 + 0.2, None, 1.2345678901234567e19], pa.float64()
            ),
            "dp": pa.array([5, None, 9, 2**40], pa.int64()),
            "mqsq_1": pa.array([3.0, 1e-05, 1.5e-07, None], pa.float64()),
            "mqsq_2": pa.array([7, 0, None, 1], pa.int64()),
            "sb": pa.array(
                [[3, 3, 3, 3], None, [1, None, 3, 4], [0, 0, 0, 2**35]],
                pa.list_(pa.int64()),
            ),
            # Fixed point: 2**32 units (GTC_FIXED_BITS).
            "gtc": pa.array(
                [
                    [1 << 32, 1 << 32, 0],
                    [0, 0, (1 << 32) + 5],
                    None,
                    [None, 2 << 32, None],
                ],
                pa.list_(pa.int64()),
            ),
            "n_records": pa.array([1, 2, 3, 4], pa.int64()),
        }
    )

    # The old reduce never emitted nulls (its accumulator started at 0 / a
    # zero list), so the null-free rows it would have produced are the reference.
    def nz(x, zero):
        return zero if x is None else x

    rows = []
    for r in staged.to_pylist():
        sb = [nz(x, 0) for x in nz(r["sb"], [0, 0, 0, 0])]
        gtc = [nz(x, 0) >> 32 for x in nz(r["gtc"], [0, 0, 0])]
        rows.append(
            (
                r["chromosome_index"],
                r["position"],
                nz(r["qual"], 0.0),
                nz(r["dp"], 0),
                nz(r["mqsq_1"], 0.0),
                nz(r["mqsq_2"], 0),
                sb,
                gtc,
            )
        )
    _, columns, old_text = _old(_old_pg_copy_site, rows)
    sql, new_text = _new(SPECS["site_info"], staged)
    assert sql == _copy_sql(_TARGET, columns)
    # qual and mqsq_1 (fields 2 and 4) are compared by value.
    _assert_same_copy_text(old_text, new_text, float_fields=(2, 4))
    # Array literals are exact, e.g. the first row's.
    assert new_text.splitlines()[0].split("\t")[6:] == ["{3,3,3,3}", "{1,1,0}"]


def test_float_text_is_shortest_round_trip():
    """Arrow writes the shortest decimal that round-trips the double -- the
    same digits as Python's repr -- so no precision is lost on the way to PG.
    """
    values = [15.0, 0.1 + 0.2, 1e-05, 1.5e-07, 123456789.123, 1e20, 2.5]
    chunk = pa.table({"q": pa.array(values, pa.float64())})
    lines = _pg_copy_text(chunk).decode().splitlines()
    assert [float(s) for s in lines] == values
    assert lines[0] == "15" and lines[1] == "0.30000000000000004"


def test_copy_text_refuses_structural_characters():
    """quoting_style="none" raises instead of writing a field COPY would split."""
    chunk = pa.table({"allele": ["x\tT"]})
    with pytest.raises(pa.ArrowInvalid):
        _pg_copy_text(chunk)


def test_empty_chunk_is_empty_text():
    spec = SPECS["site_info"]
    assert _pg_copy_text(spec.agg_schema().empty_table()) == b""
    cur = _FakeCursor()
    _copy_arrow(cur, "site_info", ("a",), spec.agg_schema().empty_table())
    assert cur.calls == [("copy_expert", "COPY site_info (a) FROM STDIN", b"")]


# ── the PostgreSQL rows: dead rows dropped, gtc's integer part (E3) ─────────


def _site_staged(gtc, n_records):
    n = len(gtc)
    return pa.table(
        {
            "chromosome_index": pa.array([1] * n, pa.int32()),
            "position": pa.array(list(range(10, 10 + n)), pa.int32()),
            "qual": pa.array([1.0] * n, pa.float64()),
            "dp": pa.array([2] * n, pa.int64()),
            "mqsq_1": pa.array([3.0] * n, pa.float64()),
            "mqsq_2": pa.array([4] * n, pa.int64()),
            "sb": pa.array([[0, 0, 0, 0]] * n, pa.list_(pa.int64())),
            "gtc": pa.array(gtc, pa.list_(pa.int64())),
            "n_records": pa.array(n_records, pa.int64()),
        }
    )


def test_site_gtc_is_the_integer_part_of_the_fixed_point_sum():
    """gtc is summed in fixed point (2**32 units); PG holds sum >> 32, the
    floor (a negative net, which a correct removal never leaves, floors
    too)."""
    one = 1 << 32
    chunk = _final_rows(
        SPECS["site_info"],
        _site_staged(
            [
                [3 * one - 1, one // 2, 7 * one],
                [0, 1, 12 * one + 3 * one // 4],
                [0, 0, -1],
            ],
            [1, 1, 1],
        ),
    )
    pg = SPECS["site_info"].pg_rows(chunk)
    assert pg.column("gtc").type == pa.list_(pa.int64())
    assert pg.column("gtc").to_pylist() == [[2, 0, 7], [0, 0, 12], [0, 0, -1]]
    assert pg.column_names == [
        "chromosome_index",
        "position",
        "qual",
        "dp",
        "mqsq_1",
        "mqsq_2",
        "sb",
        "gtc",
    ]


def test_site_with_no_live_record_is_dropped():
    chunk = _final_rows(
        SPECS["site_info"],
        _site_staged([[1 << 32, 0, 0]] * 3, [2, 0, 1]),
    )
    pg = SPECS["site_info"].pg_rows(chunk)
    assert pg.column("position").to_pylist() == [10, 12]
    # The mirror chunk keeps every root row (and the fixed-point gtc).
    assert chunk.num_rows == 3
    assert chunk.column("gtc").to_pylist()[0] == [1 << 32, 0, 0]


def test_allele_no_live_record_lists_is_dropped():
    """n_listed 0: only removed records listed it (dropped); count 0 with a
    live listing record: kept, as before."""
    staged = pa.table(
        {
            "chromosome_index": pa.array([1, 1, 1], pa.int32()),
            "position": pa.array([5, 5, 5], pa.int32()),
            "allele": ["A", "xC", "xG"],
            "count": pa.array([2, 0, 0], pa.int64()),
            "n_listed": pa.array([1, 1, 0], pa.int64()),
        }
    )
    sql, text = _new(SPECS["allele_counts"], staged)
    assert [line.split("\t")[2] for line in text.splitlines()] == ["A", "xC"]
    assert SPECS["ref_blocks"].pg_rows(staged) is staged
