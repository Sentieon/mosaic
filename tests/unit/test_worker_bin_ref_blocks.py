# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Binary-level ref-block banding tests of the Rust ingest worker (K6).

Replaces ``test_ref_block_banding.py::
test_random_stream_keeps_coverage_and_sample_counts``, which drove the
Python worker's ``grouper`` + ``update_sample_del_state`` +
``collect_sample_ref_block_data`` + ``RefBlockBander`` over random streams.
Here the same streams (the same generator, ``random.Random`` seeded with
the same string, the same draws in the same order) are written as gVCFs and
run through ``mosaic-ingest-worker`` with and without ``ref_gq_bands``; the
same properties are asserted on the local ``sample_ref_blocks`` rows and the
dense ``batch_ref_blocks`` partial (which the worker writes summed per
position, so the Python test's "the rows' ranges plus the same split parts"
becomes: banded partial = unbanded partial - unbanded rows + banded rows,
at every position).
"""

import random

import pytest

from worker_harness import (
    VRec,
    local_rows,
    requires_worker,
    run_worker,
    write_gvcf,
)

pytestmark = [pytest.mark.unit, requires_worker]

SID = 7
SHARD = ("chr21", 0, 1_000_000)


def rb(start_1b, end_1b, gq, dp=10, gt="0/0", ad=(10, 0)):
    """test_ref_block_banding.rb: a ref block with GQ, MIN_DP, AD, PL."""
    return VRec(
        start_1b,
        "A",
        ["<NON_REF>"],
        gt,
        {"END": end_1b},
        None,
        GQ=gq,
        MIN_DP=dp,
        AD=list(ad),
        PL=[0, gq, 10 * gq],
    )


def deletion(pos_1b, ref_len, info_dp=44):
    """test_ref_block_banding.deletion."""
    return VRec(
        pos_1b,
        "G" + "A" * (ref_len - 1),
        ["G", "<NON_REF>"],
        "0/1",
        {"DP": info_dp},
        30.0,
        DP=40,
        GQ=35,
        PL=[30, 0, 400, 60, 420, 480],
    )


def snv(pos_1b):
    """test_ref_block_banding.snv."""
    return VRec(
        pos_1b,
        "A",
        ["T", "<NON_REF>"],
        "0/1",
        {"DP": 20},
        9.0,
        GQ=40,
        PL=[40, 0, 300, 90, 330, 400],
    )


def _random_stream(rng, n):
    """test_ref_block_banding._random_stream, draw for draw."""
    out, pos = [], 1
    while len(out) < n:
        u = rng.random()
        if u < 0.08:
            out.append(snv(pos))
            pos += 1
        elif u < 0.12:
            n_del = rng.randint(2, 6)
            out.append(deletion(pos, n_del, info_dp=rng.randint(5, 60)))
            pos += 1  # the deleted bases are covered by the next blocks
        else:
            length = rng.choice([1, 1, 2, 2, 3, 5, 8, 20, 70])
            gq = rng.choice([0, 5, 19, 20, 33, 59, 60, 81, 99])
            r = rb(
                pos,
                pos + length - 1,
                gq,
                dp=rng.randint(0, 40),
                gt="./." if rng.random() < 0.05 else "0/0",
            )
            if rng.random() < 0.03:
                del r.fmt["GQ"], r.fmt["PL"]
            out.append(r)
            pos += length
    return out


def _run(tmp_path, name, streams, ids, spec, pbs):
    d = tmp_path / name
    d.mkdir()
    paths = [
        write_gvcf(str(d / f"s{sid}.g.vcf.gz"), f"S{sid}", s)
        for sid, s in zip(ids, streams)
    ]
    (res,) = run_worker(
        paths,
        [SHARD],
        str(d),
        pos_batch_size=pbs,
        ref_gq_bands=spec,
        sample_ids=ids,
    )
    rows = [
        (
            r["sample_id"],
            r["chromosome_index"],
            r["position_start"],
            r["position_end"],
            r["ad_1"],
            r["ad_2"],
            r["dp"],
            r["pl_1"],
            r["pl_2"],
            r["pl_3"],
            r["sample_counts"],
        )
        for r in local_rows(res["rb_paths"])
    ]
    dense = {}
    for r in local_rows(res["brb_paths"]):
        assert r["position"] not in dense, r
        dense[r["position"]] = (r["dp"], r["sample_counts"])
    return rows, dense


def row_ranges(rows):
    """test_ref_block_banding.row_ranges: (c_idx, start, end, dp, sc)."""
    return [(r[1], r[2], r[3], r[6], r[10]) for r in rows]


def _per_position(ranges):
    """{position: (dp, sample_counts)} of one sample's rows; a position
    covered twice fails."""
    out = {}
    for _c, s, e, dp, sc in ranges:
        for q in range(s, e + 1):
            assert q not in out, q
            out[q] = (dp, sc)
    return out


def _summed(ranges):
    """{position: (dp, sc)} of the ranges summed (every touched position)."""
    out = {}
    for _c, s, e, dp, sc in ranges:
        for q in range(s, e + 1):
            a = out.get(q, (0, 0))
            out[q] = (a[0] + dp, a[1] + sc)
    return out


@pytest.mark.parametrize("spec", ["20", "20,60", "all"])
@pytest.mark.parametrize("pbs", [0, 50])
def test_random_stream_keeps_coverage_and_sample_counts(tmp_path, spec, pbs):
    """Against the unbanded rows: the same positions covered (each once) and
    sample_counts equal at every position; each banded row's dp is the
    rounded length-weighted mean of the raw dp over its span, so inside
    their [min, max] and equal to it when they agree; per sample the depth
    mass is kept up to each row's rounding (n // 2 for a row of n bp). The
    dense partial is the unbanded one with the rows' ranges swapped (the
    split parts unchanged), so sample_counts is equal at every position;
    under windows no row crosses one."""
    rng = random.Random(f"{spec}/{pbs}")
    streams = [_random_stream(rng, 600) for _ in range(2)]
    ids = [SID, SID + 1]
    plain_rows, plain = _run(tmp_path, "plain", streams, ids, None, pbs)
    rows, partial = _run(tmp_path, "banded", streams, ids, spec, pbs)
    assert len(rows) < len(plain_rows)
    moved = 0
    for sid in ids:
        a = _per_position(row_ranges([r for r in plain_rows if r[0] == sid]))
        b = _per_position(row_ranges([r for r in rows if r[0] == sid]))
        assert a and set(a) == set(b)
        assert all(b[q][1] == a[q][1] for q in a)
        mass_diff = slack = 0
        for r in rows:
            if r[0] != sid:
                continue
            raw = [a[q][0] for q in range(r[2], r[3] + 1)]
            n, total = len(raw), sum(raw)
            assert r[6] == (total + n // 2) // n, (r, raw)
            assert min(raw) <= r[6] <= max(raw), (r, raw)
            if min(raw) == max(raw):
                assert r[6] == raw[0]
            moved += r[6] not in raw
            mass_diff += r[6] * n - total
            slack += n // 2
        assert abs(mass_diff) <= slack, (sid, mass_diff, slack)
    assert moved > 0  # some merged rows hold a dp no raw block had
    # The partial: the unbanded one minus its rows' ranges plus the banded
    # rows' ranges, at every position (a (0, 0) position has no row).
    plain_rr = _summed(row_ranges(plain_rows))
    banded_rr = _summed(row_ranges(rows))
    for q in set(plain) | set(partial) | set(plain_rr) | set(banded_rr):
        p, pr, br = (
            plain.get(q, (0, 0)),
            plain_rr.get(q, (0, 0)),
            banded_rr.get(q, (0, 0)),
        )
        want = (p[0] - pr[0] + br[0], p[1] - pr[1] + br[1])
        assert partial.get(q, (0, 0)) == want, q
        assert partial.get(q, (0, 0))[1] == p[1], q
    if pbs:
        assert all(r[2] // pbs == r[3] // pbs for r in rows)


def test_split_dp_is_the_rows_site_dp(tmp_path):
    """test_removal_substrate.py::test_split_dp_falls_back_to_format_dp's
    cross-module half: the deletion split's dp is the deletion row's own
    site_dp (extract_site_info's rule: no INFO/DP -> FORMAT/DP, 27), so the
    split's partial and the row's contribution agree."""
    dele = VRec(100, "GAA", ["G", "<NON_REF>"], "0/1", {}, 30.0, DP=27)
    stream = [
        VRec(1, "A", ["<NON_REF>"], "0/0", {"END": 99}, None, MIN_DP=20),
        dele,
        VRec(101, "A", ["<NON_REF>"], "0/0", {"END": 120}, None, MIN_DP=31),
    ]
    path = write_gvcf(str(tmp_path / "s.g.vcf.gz"), "S5", stream)
    (res,) = run_worker([path], [SHARD], str(tmp_path), sample_ids=[5])
    (site,) = local_rows(res["si_paths"])
    assert (site["position"], site["site_dp"], site["dp"]) == (100, 27, 27)
    assert site["del_ref_spans"] == [101, 102]
    dense = {
        r["position"]: (r["dp"], r["sample_counts"])
        for r in local_rows(res["brb_paths"])
    }
    assert dense[101] == dense[102] == (27, 0)
    assert dense[99] == (20, 1) and dense[103] == (31, 1)
    assert 100 not in dense  # the variant's own position: no block
    spans = [
        (r["position_start"], r["position_end"], r["dp"])
        for r in local_rows(res["rb_paths"])
    ]
    assert spans == [(1, 99, 20), (103, 120, 31)]
