# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the aggregate.py Arrow SUM-by-key reducers.

These are the pure core of the tier-tree reduction. Their order-independence is the
property that underwrites restart correctness (a resumed partial tree must equal a
clean tree) and fan-in invariance — proven end-to-end in
tests/integration/test_fanin_associativity.py and unit-level here, including across
the reducer's bounded-buffer flushes (``_BufferedSum``).
"""

import itertools

import numpy as np
import pyarrow as pa
import pytest

from mosaic.aggregate import (
    GTC_FIXED_BITS,
    SPECS,
    _BufferedSum,
    _flatten,
    _staging_table,
    _unflatten,
)

pytestmark = pytest.mark.unit

ALLELE = SPECS["allele_counts"]
SITE = SPECS["site_info"]
REF = SPECS["ref_blocks"]


def _batch_allele(rows):
    """(chromosome_index, position, allele, count[, n_listed]) rows; n_listed
    defaults to 1 (one listing record)."""
    return pa.record_batch(
        {
            "chromosome_index": pa.array([r[0] for r in rows], pa.int32()),
            "position": pa.array([r[1] for r in rows], pa.int32()),
            "allele": pa.array([r[2] for r in rows], pa.string()),
            "count": pa.array([r[3] for r in rows], pa.int64()),
            "n_listed": pa.array(
                [r[4] if len(r) > 4 else 1 for r in rows], pa.int64()
            ),
        }
    )


def _batch_ref(rows, int_type=pa.int32()):
    # batch_ref_blocks stores dp / sample_counts as int32 (tier_* as int64).
    return pa.record_batch(
        {
            "chromosome_index": pa.array([r[0] for r in rows], pa.int32()),
            "position": pa.array([r[1] for r in rows], pa.int32()),
            "dp": pa.array([r[2] for r in rows], int_type),
            "sample_counts": pa.array([r[3] for r in rows], int_type),
        }
    )


def _batch_site(rows):
    """(chromosome_index, position, qual, dp, mqsq_1, mqsq_2, sb, gtc[,
    n_records]) rows (batch_site_info: gtc list<int64>, fixed point);
    n_records defaults
    to 1."""
    return pa.record_batch(
        {
            "chromosome_index": pa.array([r[0] for r in rows], pa.int32()),
            "position": pa.array([r[1] for r in rows], pa.int32()),
            "qual": pa.array([r[2] for r in rows], pa.float64()),
            "dp": pa.array([r[3] for r in rows], pa.int64()),
            "mqsq_1": pa.array([r[4] for r in rows], pa.float64()),
            "mqsq_2": pa.array([r[5] for r in rows], pa.int64()),
            "sb": pa.array([r[6] for r in rows], pa.list_(pa.int64())),
            "gtc": pa.array([r[7] for r in rows], pa.list_(pa.int64())),
            "n_records": pa.array(
                [r[8] if len(r) > 8 else 1 for r in rows], pa.int64()
            ),
        }
    )


def _reduce(spec, batches, buffer_rows=None):
    """Run batches through the reducer exactly as _reduce_group does."""
    acc = _BufferedSum(spec, buffer_rows)
    for batch in batches:
        acc.add(_flatten(spec, batch))
    return _unflatten(spec, acc.result())


def _rows(table):
    return [tuple(r.values()) for r in table.to_pylist()]


def _split(batch, sizes):
    out, start = [], 0
    for n in sizes:
        out.append(batch.slice(start, n))
        start += n
    return out


# ── allele_counts ────────────────────────────────────────────────────────────

ALLELE_ROWS = [
    (1, 100, "xA", 5),
    (1, 100, "xA", 3),
    (1, 200, "xT", 2),
    (2, 100, "xA", 1),
]


def test_allele_sum_by_key():
    out = _reduce(ALLELE, [_batch_allele(ALLELE_ROWS)])
    assert _rows(out) == [
        (1, 100, "xA", 8, 2),
        (1, 200, "xT", 2, 1),
        (2, 100, "xA", 1, 1),
    ]


@pytest.mark.parametrize("buffer_rows", [1, 2, 3, 1000])
def test_allele_order_independent_across_flushes(buffer_rows):
    """Any input order and any buffer-flush boundary give the same result."""
    one = _rows(_reduce(ALLELE, [_batch_allele(ALLELE_ROWS)]))
    for perm in itertools.permutations(ALLELE_ROWS):
        batches = _split(_batch_allele(list(perm)), [1, 2, 1])
        assert _rows(_reduce(ALLELE, batches, buffer_rows)) == one


def test_allele_string_keys_distinct_and_sorted():
    """Alleles at one position are separate keys, ordered bytewise (as the old
    Python sort of the dict keys)."""
    rows = [
        (1, 50, "xT", 1),
        (1, 50, "-ATG", 2),
        (1, 50, "*", 3),
        (1, 50, "+AT", 4),
        (1, 50, "A", 5),
        (1, 50, "xT", 6),
    ]
    out = _rows(_reduce(ALLELE, [_batch_allele(rows)]))
    assert [r[:4] for r in out] == [
        (1, 50, "*", 3),
        (1, 50, "+AT", 4),
        (1, 50, "-ATG", 2),
        (1, 50, "A", 5),
        (1, 50, "xT", 7),
    ]


def test_allele_bigint_no_overflow():
    big = 5_000_000_000  # > int32
    rows = [(1, 1, "xA", big), (1, 1, "xA", big)]
    out = _reduce(ALLELE, _split(_batch_allele(rows), [1, 1]), buffer_rows=1)
    assert _rows(out) == [(1, 1, "xA", 2 * big, 2)]


def test_allele_null_count_is_zero():
    rows = [(1, 1, "xA", None), (1, 1, "xA", 4), (1, 2, "xT", None, None)]
    out = _reduce(ALLELE, [_batch_allele(rows)])
    assert _rows(out) == [(1, 1, "xA", 4, 2), (1, 2, "xT", 0, 0)]
    assert out.column("count").null_count == 0
    assert out.column("n_listed").null_count == 0


def test_output_sorted_by_key_whatever_the_input_order():
    rows = [(2, 1, "xA", 1), (1, 9, "xA", 1), (1, 2, "xC", 1), (1, 2, "xA", 1)]
    out = _rows(_reduce(ALLELE, [_batch_allele(rows)]))
    assert [r[:3] for r in out] == sorted(r[:3] for r in rows)


# ── site_info ────────────────────────────────────────────────────────────────

SITE_ROWS = [
    (1, 100, 10.0, 3, 1.0, 3, [1, 1, 1, 1], [1, 0, 0]),
    (1, 100, 5.0, 2, 2.0, 4, [2, 2, 2, 2], [0, 1, 0]),
    (2, 50, 7.0, 9, 4.0, 8, [0, 0, 0, 0], [0, 0, 1]),
]


def test_site_sum_arrays_and_scalars():
    out = _reduce(SITE, [_batch_site(SITE_ROWS[:2])])
    assert _rows(out) == [
        (1, 100, 15.0, 5, 3.0, 7, [3, 3, 3, 3], [1, 1, 0], 2)
    ]
    assert out.schema == SITE.agg_schema()


@pytest.mark.parametrize("buffer_rows", [1, 2, 1000])
def test_site_order_independent(buffer_rows):
    one = _rows(_reduce(SITE, [_batch_site(SITE_ROWS)]))
    rev = _batch_site(SITE_ROWS[::-1])
    two = _rows(_reduce(SITE, _split(rev, [2, 1]), buffer_rows))
    assert one == two


def test_site_list_columns_summed_element_wise_with_nulls():
    """sb / gtc sum element-wise; a null list or a null element counts as 0."""
    rows = [
        (1, 7, 1.0, 1, 1.0, 1, [1, 2, 3, 4], [1, 2, 3]),
        (1, 7, 1.0, 1, 1.0, 1, None, [10, None, 30]),
        (1, 7, None, None, None, None, [100, 200, 300, 400], None, None),
    ]
    out = _reduce(SITE, _split(_batch_site(rows), [1, 2]), buffer_rows=1)
    assert _rows(out) == [
        (1, 7, 2.0, 2, 2.0, 2, [101, 202, 303, 404], [11, 2, 33], 2)
    ]
    for name in SITE.value_fields:
        assert out.column(name).null_count == 0


def test_site_float_sum_is_left_fold_independent_of_flushes():
    """qual / mqsq_1 equal the sequential Python float sum bit-for-bit, for any
    buffer bound: the running result is folded first, then the new rows in
    arrival order (the old dict accumulator's order)."""
    rng = np.random.default_rng(7)
    n = 500
    quals = (rng.random(n) * 1e3).tolist()
    rows = [
        (1, int(p), q, 1, q / 3.0, 1, [0, 0, 0, 0], [0, 0, 0])
        for p, q in zip(rng.integers(0, 20, n), quals)
    ]
    expect = {}
    for r in rows:
        expect[r[1]] = expect.get(r[1], 0.0) + r[2]
    batch = _batch_site(rows)
    for bound in (1, 7, 64, 10_000):
        out = _reduce(SITE, _split(batch, [100, 150, 250]), buffer_rows=bound)
        got = dict(zip(out["position"].to_pylist(), out["qual"].to_pylist()))
        assert got == expect  # exact float equality


def test_site_flat_schema_types_list_elements():
    """The flat layout explodes sb and gtc as int64 elements (gtc in fixed
    point, GTC_FIXED_BITS: a removal's fractional gtc nets out exactly)."""
    flat = SITE.flat_schema()
    assert [flat.field(f"sb_{j}").type for j in range(4)] == [pa.int64()] * 4
    assert [flat.field(f"gtc_{j}").type for j in range(3)] == [pa.int64()] * 3
    assert flat.field("n_records").type == pa.int64()
    assert SITE.agg_schema().field("gtc").type == pa.list_(pa.int64())


def test_site_fixed_point_gtc_sums_exactly_in_any_order():
    """Per-record GTC in fixed point (2**32 units): the int64 sums are the
    same in every order and with a negative (removal) row, unlike the
    double sums (0.9999999999999999 + 1.0 rounds to 2.0, and minus 1.0 it
    is 1.0, not 0.9999999999999999). A double within 2**-33 of an integer
    quantizes to it."""
    scale = 1 << GTC_FIXED_BITS
    assert round(0.9999999999999999 * scale) == scale
    q = [round(0.7 * scale), round(0.9 * scale), round(0.6 * scale)]
    rows = [
        (1, 9, 1.0, 1, 1.0, 1, [0, 0, 0, 0], [0, q[0], q[1]]),
        (1, 9, 1.0, 1, 1.0, 1, [0, 0, 0, 0], [0, q[2], q[2]]),
        (1, 9, 1.0, 1, 1.0, 1, [0, 0, 0, 0], [0, -q[2], -q[2]]),
    ]
    sums = set()
    for perm in itertools.permutations(rows):
        out = _reduce(SITE, _split(_batch_site(list(perm)), [1, 1, 1]), 1)
        sums.add(tuple(out.column("gtc").to_pylist()[0]))
    assert sums == {(0, q[0], q[1])}


# ── negative (removal) partials ──────────────────────────────────────────────


@pytest.mark.parametrize("buffer_rows", [1, 1000])
def test_negative_site_row_nets_out(buffer_rows):
    """A removal batch's site row is the removed records' contribution,
    negated: summed with the positive partial it leaves the rest exactly, and
    n_records 0 where only removed records were."""
    keep = (1, 100, 10.0, 3, 1.5, 3, [1, 2, 3, 4], [1 << 32, 0, 0], 1)
    gone = (1, 100, 5.0, 2, 2.0, 4, [2, 2, 2, 2], [0, 1 << 32, 0], 1)
    only = (1, 200, 7.0, 9, 4.0, 8, [1, 1, 1, 1], [0, 1 << 31, 1 << 31], 1)

    def neg(r):
        return r[:2] + tuple(
            [-x for x in v] if isinstance(v, list) else -v for v in r[2:]
        )

    positive = [
        (1, 100, 15.0, 5, 3.5, 7, [3, 4, 5, 6], [1 << 32, 1 << 32, 0], 2),
        only,
    ]
    removal = [neg(gone), neg(only)]
    out = _reduce(
        SITE,
        _split(_batch_site(positive + removal), [1, 1, 1, 1]),
        buffer_rows,
    )
    got = {r[1]: r for r in _rows(out)}
    assert got[100] == keep
    assert got[200][2:] == (0.0, 0, 0.0, 0, [0, 0, 0, 0], [0, 0, 0], 0)


def test_negative_allele_row_nets_out():
    """An allele only removed records listed nets to n_listed 0 (count 0);
    one a live record lists keeps its live count."""
    rows = [
        (1, 5, "xA", 3, 2),
        (1, 5, "xC", 1, 1),
        (1, 5, "xA", -1, -1),
        (1, 5, "xC", -1, -1),
    ]
    out = _rows(_reduce(ALLELE, _split(_batch_allele(rows), [2, 2]), 1))
    assert out == [(1, 5, "xA", 2, 1), (1, 5, "xC", 0, 0)]


# ── ref_blocks ───────────────────────────────────────────────────────────────
# The ref_blocks reduce is the numpy window fold (test_ref_window_fold.py); the
# Arrow SUM-by-key over its spec is the reference that fold is checked against.


def test_ref_sum_by_key():
    rows = [(1, 100, 10, 1), (1, 100, 20, 2), (1, 200, 5, 1)]
    out = _reduce(REF, [_batch_ref(rows)])
    assert _rows(out) == [(1, 100, 30, 3), (1, 200, 5, 1)]


def test_ref_int32_partials_widen_to_int64():
    big = 2**31 - 1
    rows = [(1, 1, big, big), (1, 1, big, big)]
    out = _reduce(REF, [_batch_ref(rows, pa.int32())])
    assert _rows(out) == [(1, 1, 2 * big, 2 * big)]
    assert out.schema.field("dp").type == pa.int64()


def test_ref_null_values_are_zero():
    rows = [(1, 5, None, 1), (1, 5, None, None)]
    out = _reduce(REF, [_batch_ref(rows)])
    assert _rows(out) == [(1, 5, 0, 1)]


# ── staging layout ───────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", sorted(SPECS))
def test_empty_result_keeps_schema(name):
    spec = SPECS[name]
    result = _reduce(spec, [])
    assert result.schema == spec.agg_schema()
    staged = _staging_table(spec, result, 7, 2, 3)
    assert staged.schema == spec.staging_schema()
    assert staged.num_rows == 0


def test_staging_table_constant_columns():
    out = _reduce(ALLELE, [_batch_allele(ALLELE_ROWS)])
    staged = _staging_table(ALLELE, out, 7, 2, 3)
    assert set(staged["agg_run_id"].to_pylist()) == {7}
    assert set(staged["tier"].to_pylist()) == {2}
    assert set(staged["group_id"].to_pylist()) == {3}
    assert staged.num_rows == out.num_rows
