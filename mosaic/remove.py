# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""`mosaic remove`: remove samples from the cohort.

Removal is a negative batch.
This command re-derives the removed samples' exact contribution to the
batch partials from their own per-sample rows (E1 stores it on every row),
writes it NEGATED as one ordinary ``ingestion_progress`` batch of kind
``removal`` (``batch_site_info``, ``batch_allele_counts``,
``batch_ref_blocks``), and deletes the samples' rows from
``sample_site_info`` / ``sample_ref_blocks`` (E2 makes that copy-on-write
delete rewrite only the files that hold them). Nothing is subtracted in
place: the next ``mosaic aggregate`` folds the negative batch onto the
previous run's roots like any added batch, and its finalize drops the sites
and alleles only removed samples' records had (net ``n_records`` /
``n_listed`` 0) and the positions only they covered (net ``(0, 0)``), so the
PostgreSQL aggregates equal a fresh ingest of the remaining samples'.
Until that run ``joint_call.py`` refuses the stale cohort (exit 2).

Steps, with their resume points:

1. **T0** (one transaction): tombstone the samples (``samples.status =
   'removed'``: from here on the readers hide their rows and ``mosaic
   ingest`` refuses them, whatever happens to the rest), insert the removal
   batch (``in_progress``) and the ``removal_progress`` row
   (``in_progress``).
2. **Derive**, one unit per (chromosome, ``--shard-size`` window) in a spawn
   ProcessPoolExecutor (no worker opens PostgreSQL): the samples' rows in the
   window -> the negated partial rows, written locally (derive_site_partials,
   derive_ref_partial). The main process combines and uploads them with the
   partials' concatenation (deterministic keys folding the batch id) and
   registers them with Iceberg, one ``add_files`` commit per table
   (register_batch_uploads, the path cmd_ingest takes). Redone in full on a
   resume; a table an earlier attempt already registered is left as it is.
3. **T1**: the removal batch ``complete``, ``removal_progress``
   ``registered``.
4. **Delete** the samples' rows from both per-sample tables, one
   copy-on-write commit per table (idempotent: a resume redoes it).
5. **T2**: ``removal_progress`` ``complete``.

A failure marks the removal ``failed`` (and its batch, before T1) and
re-raises; ``mosaic remove`` with the same samples resumes it, at the derive
when its batch is not complete, else at the delete. The Iceberg
metadata-writer lock (``register.REGISTER_LOCK_KEY``, which ``mosaic
register`` and ``mosaic maintain`` take too) is held throughout.

The only test seam: ``MOSAIC_TEST_FAIL_REMOVE`` (parse_fail_remove).
"""

import argparse
import concurrent.futures
import functools
import multiprocessing
import os
import shutil
import sys
import tempfile
import time
import traceback
import warnings
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import boto3  # type: ignore[import-untyped]
import numpy as np
import psycopg2
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import (
    And,
    EqualTo,
    GreaterThanOrEqual,
    In,
    LessThanOrEqual,
)

from .aggregate import _iceberg_commit_with_retry, _position_windows
from .config import (
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
    ICEBERG_NAMESPACE,
    S3_BUCKET,
    build_s3_client_kwargs,
)
from .ingest import (
    BATCH_ALLELE_COUNTS_ARROW_SCHEMA,
    BATCH_REF_BLOCKS_ARROW_SCHEMA,
    BATCH_SITE_INFO_ARROW_SCHEMA,
    PARQUET_KWARGS,
    RefBlockSegments,
    _aggregate_and_upload_parquet,
    _get_registered_uris,
    combined_s3_uri,
    register_batch_uploads,
    sum_ref_block_segments,
)
from .init_db import (
    BATCH_KIND_REMOVAL,
    GTC_FIXED_BITS,
    SAMPLE_STATUS_REMOVED,
    SchemaError,
    check_ingest_lake_schema,
    ensure_ingest_schema,
)
from .register import REGISTER_LOCK_KEY
from .tombstones import sample_filter

# The partial tables a removal batch writes, in registration order (the
# batch_* partials ingest writes, less batch_genotypes: a removed sample keeps
# its genotype-matrix slot, which the reader hides).
PARTIAL_TABLES: Tuple[Tuple[str, Any], ...] = (
    ("batch_ref_blocks", BATCH_REF_BLOCKS_ARROW_SCHEMA),
    ("batch_site_info", BATCH_SITE_INFO_ARROW_SCHEMA),
    ("batch_allele_counts", BATCH_ALLELE_COUNTS_ARROW_SCHEMA),
)
# The per-sample tables the removed samples' rows are deleted from.
SAMPLE_TABLES = ("sample_site_info", "sample_ref_blocks")

DEFAULT_SHARD_SIZE = 10_000_000
DEFAULT_AGGREGATE_PARQUET_SIZE = 268_435_456
# How far before a window the derive reads deletion rows, for their
# del_ref_spans reaching into the window; a span ending further than this
# past its row's position is an error (raise the margin), not dropped.
DEFAULT_SPAN_MARGIN = 100_000

_HOOK_ENV = "MOSAIC_TEST_FAIL_REMOVE"
_HOOK_PHASES = ("pre_register", "post_register", "pre_delete", "post_delete")


def parse_fail_remove(spec: Optional[str]) -> Optional[str]:
    """Parse ``MOSAIC_TEST_FAIL_REMOVE="<phase>"`` (``pre_register``,
    ``post_register``, ``pre_delete`` or ``post_delete``); None when unset.
    A malformed value raises ValueError rather than silently doing nothing
    (a test relying on it would otherwise pass vacuously)."""
    if not spec:
        return None
    if spec not in _HOOK_PHASES:
        raise ValueError(
            f"{_HOOK_ENV}={spec!r}: expected one of {', '.join(_HOOK_PHASES)}"
        )
    return spec


def _maybe_fail_remove(hook: Optional[str], phase: str) -> None:
    """Test-only fault injection, inert unless MOSAIC_TEST_FAIL_REMOVE is
    set: raise just before the first add_files commit (``pre_register``:
    the partials are uploaded, nothing registered), just after the last one
    (``post_register``: registered, T1 not committed), just before the first
    delete commit (``pre_delete``) or just after the last (``post_delete``:
    the rows are gone, T2 not committed)."""
    if hook == phase:
        raise RuntimeError(f"{_HOOK_ENV}: injected crash @{phase}")


# ── the derive: a sample set's rows -> its negated partial rows ─────────────


def _empty(schema: Any) -> Any:
    return schema.empty_table()


def derive_site_partials(
    rows: Any, batch_id: int, sign: int = -1
) -> Tuple[Any, Any]:
    """The batch_site_info and batch_allele_counts rows of a set of
    sample_site_info rows (one window of the removed samples), times
    ``sign`` (-1: the removal batch's negative partials).

    Per site (chromosome_index, position): ``n_records`` = the rows there (a
    sample with two records at one position counts twice, as ingest's
    partial does) and the sums of the rows' own contribution columns
    (``qual``, ``site_dp`` -> ``dp``, ``mqsq_1``, ``mqsq_2``, ``sb[4]``, and
    ``gtc[3]`` in fixed point: each row's doubles quantized as ingest
    quantized its record, ``rint(g * 2**GTC_FIXED_BITS)``, so the negation
    nets ingest's sum out exactly). Per (site, allele) that any row lists:
    ``n_listed`` = the rows whose (own record's) allele list holds it, and
    ``count`` = the rows' gt_1 / gt_2 hits on it through their own lists,
    zero hits included (n_listed is what makes those rows matter). The
    rules of tests/jc_testlib.py's lake oracle. Both tables sorted by key.
    """
    if rows.num_rows == 0:
        return (
            _empty(BATCH_SITE_INFO_ARROW_SCHEMA),
            _empty(BATCH_ALLELE_COUNTS_ARROW_SCHEMA),
        )
    rows = rows.combine_chunks()
    keys = ["chromosome_index", "position"]
    n = rows.num_rows

    def filled(col: Any, typ: Any) -> Any:
        return pc.fill_null(pc.cast(col, typ), 0)

    flat = {k: rows.column(k) for k in keys}
    flat["qual"] = filled(rows.column("qual"), pa.float64())
    flat["dp"] = filled(rows.column("site_dp"), pa.int64())
    flat["mqsq_1"] = filled(rows.column("mqsq_1"), pa.float64())
    flat["mqsq_2"] = filled(rows.column("mqsq_2"), pa.int64())
    for j in range(4):
        flat[f"sb_{j}"] = filled(
            pc.list_element(rows.column("sb"), j), pa.int64()
        )
    for j in range(3):
        # Each row's gtc in fixed point, as ingest quantized the record
        # (quantize_gtc; np.rint rounds half to even like Python round, and
        # the product is exact), so the sum is exactly ingest's.
        g = np.asarray(
            filled(
                pc.list_element(rows.column("gtc"), j), pa.float64()
            ).to_numpy(zero_copy_only=False),
            np.float64,
        )
        flat[f"gtc_{j}"] = pa.array(
            np.rint(g * float(1 << GTC_FIXED_BITS)).astype(np.int64)
        )
    flat["n_records"] = pa.array(np.ones(n, dtype=np.int64))
    values = [k for k in flat if k not in keys]
    site = (
        pa.table(flat)
        .group_by(keys, use_threads=False)
        .aggregate([(v, "sum") for v in values])
        .sort_by([(k, "ascending") for k in keys])
    )
    m = site.num_rows

    def signed(name: str, typ: Any) -> Any:
        return pc.multiply(
            pc.cast(site.column(f"{name}_sum"), typ), pa.scalar(sign, typ)
        )

    def signed_list(name: str, width: int, typ: Any) -> Any:
        elems = np.column_stack(
            [
                np.asarray(
                    signed(f"{name}_{j}", typ).to_numpy(),
                    typ.to_pandas_dtype(),
                )
                for j in range(width)
            ]
        ).reshape(-1)
        offsets = np.arange(0, m * width + 1, width, dtype=np.int32)
        return pa.ListArray.from_arrays(
            pa.array(offsets), pa.array(elems, typ)
        )

    bsi = pa.Table.from_arrays(
        [
            pa.array(np.full(m, batch_id, dtype=np.int32)),
            site.column("chromosome_index"),
            site.column("position"),
            signed("qual", pa.float64()),
            signed("dp", pa.int64()),
            signed("mqsq_1", pa.float64()),
            signed("mqsq_2", pa.int64()),
            signed_list("sb", 4, pa.int64()),
            signed_list("gtc", 3, pa.int64()),
            signed("n_records", pa.int64()),
        ],
        schema=BATCH_SITE_INFO_ARROW_SCHEMA,
    )

    # Alleles: each row's own list, one listing per distinct allele per row.
    alleles = rows.column("alleles")
    lengths = np.asarray(
        pc.fill_null(pc.list_value_length(alleles), 0).to_numpy(), np.int64
    )
    starts = np.zeros(n, dtype=np.int64)
    np.cumsum(lengths[:-1], out=starts[1:])
    values_flat = pc.list_flatten(alleles)
    parent = np.asarray(pc.list_parent_indices(alleles).to_numpy(), np.int64)
    c_col = rows.column("chromosome_index").to_numpy()
    p_col = rows.column("position").to_numpy()
    listing = (
        pa.table(
            {
                "row": pa.array(parent),
                "chromosome_index": pa.array(c_col[parent], pa.int32()),
                "position": pa.array(p_col[parent], pa.int32()),
                "allele": values_flat,
            }
        )
        .group_by(["row", "chromosome_index", "position", "allele"])
        .aggregate([])
    )
    parts = [
        pa.table(
            {
                "chromosome_index": listing.column("chromosome_index"),
                "position": listing.column("position"),
                "allele": listing.column("allele"),
                "count": pa.array(np.zeros(listing.num_rows, dtype=np.int64)),
                "n_listed": pa.array(
                    np.ones(listing.num_rows, dtype=np.int64)
                ),
            }
        )
    ]
    for gt in ("gt_1", "gt_2"):
        g = rows.column(gt)
        valid = np.asarray(pc.is_valid(g).to_numpy(zero_copy_only=False))
        idx = np.flatnonzero(valid)
        if idx.size == 0:
            continue
        local = np.asarray(
            pc.fill_null(g, 0).to_numpy(zero_copy_only=False), np.int64
        )[idx]
        if np.any(local < 0) or np.any(local >= lengths[idx]):
            raise ValueError(f"{gt} outside its row's allele list")
        hit = values_flat.take(pa.array(starts[idx] + local))
        parts.append(
            pa.table(
                {
                    "chromosome_index": pa.array(c_col[idx], pa.int32()),
                    "position": pa.array(p_col[idx], pa.int32()),
                    "allele": hit,
                    "count": pa.array(np.ones(idx.size, dtype=np.int64)),
                    "n_listed": pa.array(np.zeros(idx.size, dtype=np.int64)),
                }
            )
        )
    akeys = ["chromosome_index", "position", "allele"]
    allele = (
        pa.concat_tables(parts)
        .group_by(akeys, use_threads=False)
        .aggregate([("count", "sum"), ("n_listed", "sum")])
        .sort_by([(k, "ascending") for k in akeys])
    )
    k = allele.num_rows
    bac = pa.Table.from_arrays(
        [
            pa.array(np.full(k, batch_id, dtype=np.int32)),
            allele.column("chromosome_index"),
            allele.column("position"),
            allele.column("allele"),
            pc.multiply(allele.column("count_sum"), sign),
            pc.multiply(allele.column("n_listed_sum"), sign),
        ],
        schema=BATCH_ALLELE_COUNTS_ARROW_SCHEMA,
    )
    return bsi, bac


class SpanMarginError(ValueError):
    """A deletion row's del_ref_spans reach further past its position than
    --span-margin: a window's scan could miss them."""


def derive_ref_partial(
    blocks: Any,
    site_rows: Any,
    c_idx: int,
    lo: int,
    hi: int,
    span_margin: int,
    batch_id: int,
    sign: int = -1,
) -> Any:
    """The batch_ref_blocks rows of a set of samples over the inclusive
    window [lo, hi] of chromosome ``c_idx``, times ``sign``: per position,
    the sums of their sample_ref_blocks rows' (``dp``, ``sample_counts``)
    and of their deletion rows' ``del_ref_spans`` (the row's ``site_dp``,
    sample count 0), each range clipped to the window (the rule of the lake
    oracle). Positions whose sum is (0, 0) are omitted, as ingest's partial
    omits them. Summed with ingest's own sum_ref_block_segments.

    ``blocks``: the samples' sample_ref_blocks rows overlapping the window.
    ``site_rows``: their sample_site_info rows at positions [lo -
    span_margin, hi], so a span of an earlier row reaching into the window
    is counted. Every span of a row inside the window must end within
    ``span_margin`` of the row's position, else SpanMarginError: that is
    what guarantees the next window's scan sees every span reaching into
    it.
    """
    starts = [np.asarray(blocks.column("position_start").to_numpy(), np.int64)]
    ends = [np.asarray(blocks.column("position_end").to_numpy(), np.int64)]
    dps = [
        np.asarray(pc.fill_null(blocks.column("dp"), 0).to_numpy(), np.int64)
    ]
    scs = [
        np.asarray(
            pc.fill_null(blocks.column("sample_counts"), 0).to_numpy(),
            np.int64,
        )
    ]
    spans = site_rows.filter(pc.is_valid(site_rows.column("del_ref_spans")))
    if spans.num_rows:
        flat = np.asarray(
            pc.list_flatten(spans.column("del_ref_spans")).to_numpy(), np.int64
        )
        parent = np.asarray(
            pc.list_parent_indices(spans.column("del_ref_spans")).to_numpy(),
            np.int64,
        )
        if flat.size % 2:
            raise ValueError("del_ref_spans with an odd number of values")
        pos = np.asarray(spans.column("position").to_numpy(), np.int64)
        site_dp = np.asarray(
            pc.fill_null(spans.column("site_dp"), 0).to_numpy(), np.int64
        )
        s, e, owner = flat[0::2], flat[1::2], parent[0::2]
        own = (pos[owner] >= lo) & (pos[owner] <= hi)
        reach = e - pos[owner]
        if np.any(own & (reach > span_margin)):
            worst = int(reach[own].max())
            raise SpanMarginError(
                f"a deletion's del_ref_spans end {worst} bp past its "
                f"position on chromosome {c_idx}, beyond --span-margin "
                f"{span_margin}: re-run with --span-margin >= {worst}"
            )
        starts.append(s)
        ends.append(e)
        dps.append(site_dp[owner])
        scs.append(np.zeros(s.size, dtype=np.int64))
    st = np.maximum(np.concatenate(starts), lo)
    en = np.minimum(np.concatenate(ends), hi)
    keep = st <= en
    segments = RefBlockSegments.from_arrays(
        c_idx,
        st[keep],
        en[keep],
        np.concatenate(dps)[keep],
        np.concatenate(scs)[keep],
    )
    positions, dp, sc = sum_ref_block_segments(segments)
    n = int(positions.size)
    return pa.Table.from_arrays(
        [
            pa.array(np.full(n, batch_id, dtype=np.int32)),
            pa.array(np.full(n, c_idx, dtype=np.int32)),
            pa.array(positions.astype(np.int32)),
            pa.array(sign * dp, pa.int64()).cast(pa.int32()),
            pa.array(sign * sc, pa.int64()).cast(pa.int32()),
        ],
        schema=BATCH_REF_BLOCKS_ARROW_SCHEMA,
    )


def plan_units(
    contigs: Dict[int, int], shard_size: int
) -> List[Tuple[int, int, int, int]]:
    """(chromosome_index, window_id, lo, hi) of every derive unit: each
    contig in id order, cut into inclusive windows of ``shard_size`` bp (the
    aggregate's _position_windows)."""
    if shard_size < 1:
        raise ValueError(f"shard_size must be >= 1: {shard_size}")
    return [
        (int(c), w, lo, hi)
        for c in sorted(contigs)
        for w, (lo, hi) in enumerate(
            _position_windows(int(contigs[c]), shard_size)
        )
    ]


# ── what to do: refuse, start fresh or resume ───────────────────────────────


@dataclass(frozen=True)
class SampleRow:
    """A samples row."""

    sample_id: int
    name: str
    status: str


@dataclass(frozen=True)
class RemovalRow:
    """An unfinished removal_progress row (status in_progress, registered
    or failed) with the status of its removal batch."""

    removal_id: int
    sample_ids: Tuple[int, ...]
    sample_names: Tuple[str, ...]
    batch_id: int
    status: str
    batch_status: str


@dataclass(frozen=True)
class BatchRow:
    """An ingest batch (ingestion_progress, kind 'ingest')."""

    batch_id: int
    sample_names: Tuple[str, ...]
    status: str


@dataclass(frozen=True)
class RemovalPlan:
    """``state`` "fresh" (T0 to come) or "resume" (``removal`` is the
    unfinished removal of exactly these samples); ``phase`` where the work
    starts: "derive" (the negative batch is not complete) or "delete"."""

    state: str
    phase: str
    names: Tuple[str, ...]
    sample_ids: Tuple[int, ...]
    removal: Optional[RemovalRow] = None


class RemovalRefused(Exception):
    """A check before any change failed: `mosaic remove` exits 1 with the
    message and changes nothing."""


def decide_removal(
    names: Sequence[str],
    samples: Dict[str, SampleRow],
    unfinished: Sequence[RemovalRow],
    batches: Sequence[BatchRow],
) -> RemovalPlan:
    """Check a removal request against the database state (pure: the unit
    tests drive it). Refused (RemovalRefused), in this order:

    * a name not in ``samples``;
    * an unfinished removal of another sample set (finish it first);
    * a sample already removed (unless the unfinished removal of exactly
      these samples is being resumed);
    * a sample in an ingest batch that is not complete (in_progress /
      failed / uploaded): its rows are not all in the lake.

    Otherwise a fresh removal, or the resume of the unfinished removal of
    exactly this sample set: at its delete step when its batch is complete
    (T1 committed), else at its derive.
    """
    want = tuple(sorted(set(names)))
    if not want:
        raise RemovalRefused("no sample named")
    unknown = [n for n in want if n not in samples]
    if unknown:
        raise RemovalRefused(f"unknown sample(s): {', '.join(unknown)}")
    ids = tuple(sorted(samples[n].sample_id for n in want))
    mine = [r for r in unfinished if tuple(sorted(r.sample_names)) == want]
    others = [r for r in unfinished if r not in mine]
    if others:
        r = others[0]
        raise RemovalRefused(
            f"removal {r.removal_id} of {', '.join(r.sample_names)} is "
            f"unfinished ({r.status}); finish it first: mosaic remove "
            f"{' '.join(r.sample_names)}"
        )
    if not mine:
        gone = [n for n in want if samples[n].status == SAMPLE_STATUS_REMOVED]
        if gone:
            raise RemovalRefused(f"already removed: {', '.join(gone)}")
    open_batches = [
        b
        for b in batches
        if b.status != "complete" and set(b.sample_names) & set(want)
    ]
    if open_batches:
        b = open_batches[0]
        hint = (
            "run mosaic register"
            if b.status == "uploaded"
            else "finish it with mosaic ingest"
        )
        held = sorted(set(b.sample_names) & set(want))
        raise RemovalRefused(
            f"ingest batch {b.batch_id} holding {', '.join(held)} is "
            f"{b.status}, so their rows are not all in the lake; {hint} "
            "first"
        )
    if mine:
        r = mine[0]
        phase = "delete" if r.batch_status == "complete" else "derive"
        return RemovalPlan("resume", phase, want, ids, r)
    return RemovalPlan("fresh", "derive", want, ids)


def _load_state(
    conn: Any,
) -> Tuple[Dict[str, SampleRow], List[RemovalRow], List[BatchRow]]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, sample_name, status FROM samples")
        samples = {
            name: SampleRow(int(i), name, status)
            for i, name, status in cur.fetchall()
        }
        cur.execute(
            "SELECT r.removal_id, r.sample_ids, r.sample_names, r.batch_id, "
            "r.status, p.status FROM removal_progress r "
            "JOIN ingestion_progress p ON p.batch_id = r.batch_id "
            "WHERE r.status <> 'complete' ORDER BY r.removal_id"
        )
        unfinished = [
            RemovalRow(
                int(rid),
                tuple(int(s) for s in sids),
                tuple(names),
                int(bid),
                status,
                bstatus,
            )
            for rid, sids, names, bid, status, bstatus in cur.fetchall()
        ]
        cur.execute(
            "SELECT batch_id, sample_names, status FROM ingestion_progress "
            "WHERE kind = 'ingest' ORDER BY batch_id"
        )
        batches = [
            BatchRow(int(b), tuple(n), st) for b, n, st in cur.fetchall()
        ]
    conn.commit()
    return samples, unfinished, batches


# ── the derive units (spawn workers: no PostgreSQL) ─────────────────────────

_WORKER: Dict[str, Any] = {}


def _init_worker() -> None:
    _WORKER["catalog"] = load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)


def _worker_table(name: str) -> Any:
    """A per-sample table, loaded once per worker (not per unit: ~300 units
    genome-wide). The derive runs before the removal's deletes, so every
    unit reads the snapshot the first load saw."""
    key = f"table:{name}"
    if key not in _WORKER:
        _WORKER[key] = _WORKER["catalog"].load_table(
            f"{ICEBERG_NAMESPACE}.{name}"
        )
    return _WORKER[key]


def _scan(table: Any, row_filter: Any, ids: Sequence[int]) -> Tuple[Any, int]:
    """The rows of the sample set ``ids`` matching ``row_filter`` (the In
    pushdown or the mask, sample_filter's rule), and the planned data
    files' bytes."""
    from pyiceberg.io.pyarrow import ArrowScan

    pushdown, mask = sample_filter(ids)
    if pushdown is not None:
        row_filter = And(row_filter, pushdown)
    scan = table.scan(row_filter=row_filter)
    tasks = list(scan.plan_files())  # planned once (to_arrow plans again)
    planned = sum(t.file.file_size_in_bytes for t in tasks)
    arrow = ArrowScan(
        scan.table_metadata,
        scan.io,
        scan.projection(),
        scan.row_filter,
        scan.case_sensitive,
        scan.limit,
    ).to_table(tasks)
    return mask(arrow), planned


def _derive_unit(task: Tuple) -> Dict[str, Any]:
    """One (chromosome, window) unit: read the removed samples' rows there
    and write the negated partial rows to local Parquet. Top-level and
    picklable for the spawn ProcessPoolExecutor."""
    c_idx, window_id, lo, hi, ids, batch_id, span_margin, tmpdir = task
    if "catalog" not in _WORKER:
        _init_worker()
    t0 = time.perf_counter()
    site_table = _worker_table("sample_site_info")
    rb_table = _worker_table("sample_ref_blocks")
    # pyiceberg's pydantic stubs reject positional args (mypy).
    on_chrom: Any = EqualTo("chromosome_index", int(c_idx))  # type: ignore
    site_rows, site_bytes = _scan(
        site_table,
        And(
            on_chrom,
            And(
                GreaterThanOrEqual(  # type: ignore[misc, call-arg]
                    "position", max(1, int(lo) - int(span_margin))
                ),
                LessThanOrEqual("position", int(hi)),  # type: ignore
            ),
        ),
        ids,
    )
    blocks, rb_bytes = _scan(
        rb_table,
        And(
            on_chrom,
            And(
                LessThanOrEqual("position_start", int(hi)),  # type: ignore
                GreaterThanOrEqual("position_end", int(lo)),  # type: ignore
            ),
        ),
        ids,
    )
    in_window = site_rows.filter(
        pc.and_(
            pc.greater_equal(site_rows.column("position"), lo),
            pc.less_equal(site_rows.column("position"), hi),
        )
    )
    bsi, bac = derive_site_partials(in_window, batch_id)
    brb = derive_ref_partial(
        blocks, site_rows, c_idx, lo, hi, span_margin, batch_id
    )
    paths: Dict[str, Optional[str]] = {}
    rows: Dict[str, int] = {}
    for (name, _schema), table in zip(PARTIAL_TABLES, (brb, bsi, bac)):
        rows[name] = table.num_rows
        if table.num_rows == 0:
            paths[name] = None
            continue
        path = os.path.join(tmpdir, f"{name}_c{c_idx}_w{window_id}.parquet")
        pq.write_table(table, path, **PARQUET_KWARGS)
        paths[name] = path
    return {
        "c_idx": c_idx,
        "window_id": window_id,
        "lo": lo,
        "hi": hi,
        "paths": paths,
        "rows": rows,
        "site_rows": in_window.num_rows,
        "block_rows": blocks.num_rows,
        "bytes_planned": site_bytes + rb_bytes,
        "seconds": time.perf_counter() - t0,
    }


# ── the orchestration (main process) ────────────────────────────────────────


def get_db_connection() -> Any:
    return psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]


def _t0(conn: Any, plan: RemovalPlan) -> Tuple[int, int]:
    """T0: tombstone the samples, insert the removal batch and its
    removal_progress row, in one transaction. Returns (removal_id,
    batch_id)."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE samples SET status = %s WHERE id = ANY(%s)",
            (SAMPLE_STATUS_REMOVED, list(plan.sample_ids)),
        )
        cur.execute(
            "INSERT INTO ingestion_progress (sample_names, region, status, "
            "sample_ids, kind) VALUES (%s, NULL, 'in_progress', NULL, %s) "
            "RETURNING batch_id",
            (list(plan.names), BATCH_KIND_REMOVAL),
        )
        batch_id = int(cur.fetchone()[0])
        cur.execute(
            "INSERT INTO removal_progress (sample_ids, sample_names, "
            "batch_id, status) VALUES (%s, %s, %s, 'in_progress') "
            "RETURNING removal_id",
            (list(plan.sample_ids), list(plan.names), batch_id),
        )
        removal_id = int(cur.fetchone()[0])
    conn.commit()
    return removal_id, batch_id


def _set_status(
    conn: Any,
    removal_id: int,
    status: str,
    batch_id: Optional[int] = None,
    batch_status: Optional[str] = None,
) -> None:
    """removal_progress.status (and the removal batch's status), one
    transaction."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE removal_progress SET status = %s, updated_at = now() "
            "WHERE removal_id = %s",
            (status, removal_id),
        )
        if batch_id is not None and batch_status is not None:
            cur.execute(
                "UPDATE ingestion_progress SET status = %s, "
                "updated_at = now() WHERE batch_id = %s",
                (batch_status, batch_id),
            )
    conn.commit()


def _mark_failed(conn: Any, removal_id: int, batch_id: int) -> None:
    """The removal failed: removal_progress 'failed', and its batch 'failed'
    unless T1 already completed it. Best effort."""
    try:
        conn.rollback()
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE removal_progress SET status = 'failed', "
                "updated_at = now() WHERE removal_id = %s",
                (removal_id,),
            )
            cur.execute(
                "UPDATE ingestion_progress SET status = 'failed', "
                "updated_at = now() WHERE batch_id = %s "
                "AND status <> 'complete'",
                (batch_id,),
            )
        conn.commit()
    except Exception:
        pass


def _contig_lengths(conn: Any) -> Dict[int, int]:
    with conn.cursor() as cur:
        cur.execute("SELECT id, length FROM contigs ORDER BY id")
        out = {int(c): int(n) for c, n in cur.fetchall()}
    conn.commit()
    return out


def _derive(
    units: List[Tuple[int, int, int, int]],
    ids: Sequence[int],
    batch_id: int,
    span_margin: int,
    threads: int,
    tmpdir: str,
    contig_names: Dict[int, str],
) -> List[Dict[str, Any]]:
    """Run the derive units in a spawn ProcessPoolExecutor (spawn: no
    fork-after-threads; a dead worker raises BrokenProcessPool instead of
    hanging). Results come back in plan (genomic) order."""
    tasks = [
        (c, w, lo, hi, list(ids), batch_id, span_margin, tmpdir)
        for c, w, lo, hi in units
    ]
    out: List[Dict[str, Any]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max(1, threads),
        mp_context=multiprocessing.get_context("spawn"),
        initializer=_init_worker,
    ) as executor:
        for r in executor.map(_derive_unit, tasks):
            rows = r["rows"]
            print(
                f"  unit {contig_names.get(r['c_idx'], r['c_idx'])}:"
                f"{r['lo']}-{r['hi']}: {r['site_rows']} site rows, "
                f"{r['block_rows']} ref-block rows -> "
                f"{rows['batch_site_info']} site, "
                f"{rows['batch_allele_counts']} allele, "
                f"{rows['batch_ref_blocks']} ref-block partial rows "
                f"({r['seconds']:.1f}s)",
                file=sys.stderr,
            )
            out.append(r)
    return out


def _file_list(table: Any) -> Dict[str, Any]:
    """The current snapshot's data files of an Iceberg table, by path."""
    if table.current_snapshot() is None:
        return {}
    return {t.file.file_path: t.file for t in table.scan().plan_files()}


def delete_line(
    name: str, before: Dict[str, Any], after: Dict[str, Any], secs: float
) -> str:
    """The per-table delete line from the table's data files (by path)
    before and after the delete: the files removed and written, each as
    counted (a copy-on-write rewrite may split one file into two, so which
    removed file was dropped whole and which was rewritten is not known),
    the files kept, then the bytes and rows removed."""
    gone = [before[u] for u in before if u not in after]
    new = [after[u] for u in after if u not in before]
    kept = len(before) - len(gone)
    new_rows = sum(f.record_count for f in new)
    gone_rows = sum(f.record_count for f in gone)
    return (
        f"  delete {name}: {len(gone)} data file(s) removed, "
        f"{len(new)} written ({new_rows} rows of other samples, "
        f"{sum(f.file_size_in_bytes for f in new)} bytes), {kept} kept; "
        f"{sum(f.file_size_in_bytes for f in gone)} bytes and "
        f"{gone_rows - new_rows} rows removed ({secs:.2f}s)"
    )


def _delete_rows(
    catalog: Any, ids: Sequence[int], hook: Optional[str]
) -> float:
    """Delete the samples' rows from both per-sample tables, one
    copy-on-write commit per table (idempotent: nothing matches on a
    re-run), printing per table the data files removed, written and kept
    (``delete_line``). Returns the seconds spent."""
    t0 = time.perf_counter()
    expr = In("sample_id", [int(s) for s in ids])  # type: ignore
    for i, name in enumerate(SAMPLE_TABLES):
        table = catalog.load_table(f"{ICEBERG_NAMESPACE}.{name}")
        before = _file_list(table)
        if i == 0:
            _maybe_fail_remove(hook, "pre_delete")
        t = time.perf_counter()
        if before:
            with warnings.catch_warnings():
                # A resumed delete finds nothing left to delete.
                warnings.filterwarnings(
                    "ignore",
                    message="Delete operation did not match any records",
                )
                _iceberg_commit_with_retry(
                    table, functools.partial(table.delete, expr)
                )
            table.refresh()
        after = _file_list(table)
        print(
            delete_line(name, before, after, time.perf_counter() - t),
            file=sys.stderr,
        )
    _maybe_fail_remove(hook, "post_delete")
    return time.perf_counter() - t0


def run_remove(
    names: Sequence[str],
    threads: int,
    shard_size: int = DEFAULT_SHARD_SIZE,
    aggregate_parquet_size: int = DEFAULT_AGGREGATE_PARQUET_SIZE,
    span_margin: int = DEFAULT_SPAN_MARGIN,
) -> int:
    """Remove the named samples (see the module docstring). Returns the
    exit code: 0 done, 1 refused or locked (nothing changed). Raises on any
    failure after T0 (the removal is then marked failed: re-run to
    resume)."""
    hook = parse_fail_remove(os.environ.get(_HOOK_ENV))
    if shard_size < 1 or span_margin < 0 or aggregate_parquet_size < 1:
        raise ValueError(
            "--shard-size and --aggregate-parquet-size must be "
            ">= 1 and --span-margin >= 0"
        )
    conn = get_db_connection()
    locked = False
    try:
        ensure_ingest_schema(conn)
        catalog = load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)
        try:
            check_ingest_lake_schema(catalog)
        except SchemaError as e:
            print(f"Error: {e}", file=sys.stderr)
            return 1
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s)", (REGISTER_LOCK_KEY,)
            )
            locked = bool(cur.fetchone()[0])
        conn.commit()
        if not locked:
            print(
                "Error: `mosaic register` or `mosaic maintain` holds the "
                "Iceberg metadata-writer lock (pg advisory lock "
                f"{REGISTER_LOCK_KEY:#x}); `mosaic remove` takes it too. "
                "Nothing done.",
                file=sys.stderr,
            )
            return 1
        try:
            plan = decide_removal(names, *_load_state(conn))
        except RemovalRefused as e:
            print(f"Error: {e}. Nothing done.", file=sys.stderr)
            return 1
        return _run_plan(
            conn,
            catalog,
            plan,
            threads,
            shard_size,
            aggregate_parquet_size,
            span_margin,
            hook,
        )
    finally:
        if locked:
            try:
                conn.rollback()
                with conn.cursor() as cur:
                    cur.execute(
                        "SELECT pg_advisory_unlock(%s)", (REGISTER_LOCK_KEY,)
                    )
                conn.commit()
            except Exception:
                pass  # closing the session releases the lock anyway
        conn.close()


def _run_plan(
    conn: Any,
    catalog: Any,
    plan: RemovalPlan,
    threads: int,
    shard_size: int,
    aggregate_parquet_size: int,
    span_margin: int,
    hook: Optional[str],
) -> int:
    t_start = time.perf_counter()
    if plan.state == "fresh":
        removal_id, batch_id = _t0(conn, plan)
        how = "fresh"
    else:
        assert plan.removal is not None
        removal_id, batch_id = plan.removal.removal_id, plan.removal.batch_id
        how = f"resume at {plan.removal.status}"
        if plan.phase == "derive":
            _set_status(
                conn, removal_id, "in_progress", batch_id, "in_progress"
            )
        else:
            _set_status(conn, removal_id, "registered")
    contigs = _contig_lengths(conn)
    with conn.cursor() as cur:
        cur.execute("SELECT id, chromosome FROM contigs")
        contig_names = {int(c): n for c, n in cur.fetchall()}
    conn.commit()
    units = plan_units(contigs, shard_size)
    print(
        f"removal_id={removal_id} batch_id={batch_id} "
        f"samples={','.join(plan.names)} {how} threads={threads} "
        f"shard_size={shard_size} units={len(units)}",
        file=sys.stderr,
    )
    times = {"derive": 0.0, "upload": 0.0, "register": 0.0, "delete": 0.0}
    sizes: Dict[str, Tuple[int, int]] = {}
    try:
        if plan.phase == "derive":
            sizes = _derive_upload_register(
                catalog,
                plan,
                batch_id,
                plan.state == "resume",
                units,
                threads,
                aggregate_parquet_size,
                span_margin,
                contig_names,
                hook,
                times,
            )
            # T1: the negative batch is complete (aggregate sees it).
            _set_status(conn, removal_id, "registered", batch_id, "complete")
        times["delete"] = _delete_rows(catalog, plan.sample_ids, hook)
        _set_status(conn, removal_id, "complete")
    except Exception:
        _mark_failed(conn, removal_id, batch_id)
        raise
    partial = "; ".join(
        f"{name} {rows} rows {nbytes} bytes"
        for name, (rows, nbytes) in sizes.items()
    )
    print(
        f"removed {','.join(plan.names)} (removal_id={removal_id}, "
        f"batch_id={batch_id}) in {time.perf_counter() - t_start:.1f}s: "
        + " ".join(f"{k} {v:.2f}s" for k, v in times.items())
        + (f"; negative partial: {partial}" if partial else "")
        + ". Run `mosaic aggregate` to apply it.",
        file=sys.stderr,
    )
    return 0


def _derive_upload_register(
    catalog: Any,
    plan: RemovalPlan,
    batch_id: int,
    is_resume: bool,
    units: List[Tuple[int, int, int, int]],
    threads: int,
    aggregate_parquet_size: int,
    span_margin: int,
    contig_names: Dict[int, str],
    hook: Optional[str],
    times: Dict[str, float],
) -> Dict[str, Tuple[int, int]]:
    """Steps 2 of the module docstring: derive, upload, register. Returns
    {table: (rows, bytes)} of the negative partial."""
    tmpdir = tempfile.mkdtemp(prefix="mosaic_remove_")
    try:
        t = time.perf_counter()
        results = _derive(
            units,
            plan.sample_ids,
            batch_id,
            span_margin,
            threads,
            tmpdir,
            contig_names,
        )
        times["derive"] = time.perf_counter() - t
        print(
            f"  derived {sum(r['site_rows'] for r in results)} site rows and "
            f"{sum(r['block_rows'] for r in results)} ref-block rows of "
            f"{len(plan.names)} sample(s) "
            f"({sum(r['bytes_planned'] for r in results)} data-file bytes "
            f"planned) in {times['derive']:.1f}s",
            file=sys.stderr,
        )

        s3_client = boto3.client(
            "s3", **build_s3_client_kwargs()
        )  # type: ignore[call-overload]
        t = time.perf_counter()
        uploads: List[Tuple[str, Any, set, List[str]]] = []
        sizes: Dict[str, Tuple[int, int]] = {}
        for name, _schema in PARTIAL_TABLES:
            table = catalog.load_table(f"{ICEBERG_NAMESPACE}.{name}")
            paths = [r["paths"][name] for r in results if r["paths"][name]]
            rows = sum(r["rows"][name] for r in results)
            registered = _get_registered_uris(table) if is_resume else set()
            if (
                is_resume
                and paths
                and combined_s3_uri(name, list(plan.names), batch_id, None)
                in registered
            ):
                # Registered by an earlier attempt (one add_files commit
                # per table, so all of it): leave its files as they are.
                print(
                    f"  {name}: registered by an earlier attempt; kept",
                    file=sys.stderr,
                )
                for p in paths:
                    os.remove(p)
                sizes[name] = (rows, 0)
                continue
            uris = _aggregate_and_upload_parquet(
                paths,
                name,
                s3_client,
                S3_BUCKET,
                aggregate_parquet_size,
                list(plan.names),
                None,
                batch_id,
            )
            nbytes = 0
            for uri in uris:
                key = uri.split("/", 3)[3]
                nbytes += int(
                    s3_client.head_object(Bucket=S3_BUCKET, Key=key)[
                        "ContentLength"
                    ]
                )
            sizes[name] = (rows, nbytes)
            uploads.append((name, table, registered, uris))
        times["upload"] = time.perf_counter() - t

        _maybe_fail_remove(hook, "pre_register")
        times["register"] = register_batch_uploads(uploads, is_resume)
        print(
            f"  registered {sum(len(u[3]) for u in uploads)} data file(s) "
            f"of batch {batch_id} in {times['register']:.2f}s",
            file=sys.stderr,
        )
        _maybe_fail_remove(hook, "post_register")
        return sizes
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# ── CLI ──────────────────────────────────────────────────────────────────────


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def _non_negative_int(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "samples", nargs="+", help="Names of the samples to remove"
    )
    parser.add_argument(
        "--threads",
        type=_positive_int,
        default=multiprocessing.cpu_count(),
        help="Parallel derive workers (default: number of CPUs)",
    )
    parser.add_argument(
        "--shard-size",
        type=_positive_int,
        default=DEFAULT_SHARD_SIZE,
        dest="shard_size",
        help="Genomic window in bp of one derive unit (default "
        f"{DEFAULT_SHARD_SIZE:,}, as `mosaic ingest --shard-size`)",
    )
    parser.add_argument(
        "--aggregate-parquet-size",
        type=_positive_int,
        default=DEFAULT_AGGREGATE_PARQUET_SIZE,
        dest="aggregate_parquet_size",
        help="Target size in bytes of the negative partial's data files "
        f"(default {DEFAULT_AGGREGATE_PARQUET_SIZE}, as `mosaic ingest`)",
    )
    parser.add_argument(
        "--span-margin",
        type=_non_negative_int,
        default=DEFAULT_SPAN_MARGIN,
        dest="span_margin",
        help="How far (bp) before a window the derive reads deletion rows "
        "for their deletion-split spans reaching into it (default "
        f"{DEFAULT_SPAN_MARGIN:,}); a span reaching further is an error, "
        "not dropped",
    )


def run(args: Any) -> None:
    try:
        rc = run_remove(
            args.samples,
            args.threads,
            args.shard_size,
            args.aggregate_parquet_size,
            args.span_margin,
        )
    except Exception as e:
        traceback.print_exc()
        print(
            f"Error during removal: {e} (the removal is marked failed; "
            "re-run `mosaic remove` with the same samples to resume it)",
            file=sys.stderr,
        )
        sys.exit(1)
    if rc:
        sys.exit(rc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Remove samples: tombstone them, write their re-derived "
            "contribution as a negative batch for `mosaic aggregate`, and "
            "delete their per-sample rows."
        )
    )
    configure_parser(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
