# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""
Aggregate the per-batch Iceberg partials and finalize the cross-sample
aggregates to PostgreSQL, using a materialized tier tree.

Runs once after all ingest batches register, before `mosaic index`:

    mosaic init -> mosaic ingest (all batches)
                -> mosaic aggregate -> mosaic index

Each metric is a uniform SUM-by-key reduction into its own PG table:

    allele_counts  SUM(count)             by (chrom, pos, allele) -> alleles
    site_info      SUM(...)               by (chrom, pos)         -> site_info
    ref_blocks     SUM(dp, sample_counts) by (chrom, pos) at sites -> ref_blocks

`alleles` is the allele catalog: every allele any batch listed (ingest writes
a batch_allele_counts row for each, count 0 included), its summed count, and
the deletion span ``del_range`` derived from the allele string at the
finalize (allele_del_range). Ingest writes no allele identities to PG.

PostgreSQL is never updated in place. The three tables are LIST-partitioned by
chromosome, and the finalize is one unit per (metric, chromosome): it COPYs the
chromosome's root-tier rows into a fresh standalone table, builds its primary
key in one pass, and swaps it in for the attached partition (DETACH old, ATTACH
new, DROP old) in one short transaction. Readers never see a half-built table,
and the transient space is one chromosome's old plus new partition.

The reduce and the finalize are vectorized: each Iceberg data file's record
batches are summed with an Arrow hash aggregate (``Table.group_by``) into a
bounded buffer (site_info, allele_counts) or into dense numpy arrays
(ref_blocks), and the finalize renders Arrow chunks straight to PostgreSQL
``COPY`` text. No per-row Python work remains on either path.

The sparse metrics (site_info, allele_counts) reduce through a **materialized
tier tree**, planned **per chromosome**: a chromosome's tree covers only the
completed batches whose ingest region covers it (``ingestion_progress.region``
NULL = whole genome, else its chromosome part). Its per-batch partials
(batch_*, tier 0) are reduced in groups of `--fan-in`; each group's subtotal is
persisted to an Iceberg staging table (tier_*), and tiers reduce upward until
one root group remains. The root is materialized to agg_* and finalized to PG.
Degenerates to a single pass per chromosome when that chromosome's batch count
is <= fan_in (one tier, one group).

ref_blocks has no tier tree: it is one **window unit per (chromosome, position
window)** of `--ref-shard-size` bases. A unit reads every covering batch's dense
batch_ref_blocks rows for its window, sums them into arrays indexed by
position, and writes two roots: the **dense root** (every covered position,
tier_ref_blocks_dense) and the **sparse root** (the covered positions that are
variant sites in PG site_info, tier_ref_blocks at tier 1, group_id = window).
The finalize reads the sparse root; the dense root is kept so a later run can
fold new batches into the older batches' coverage at newly discovered sites.

Incremental by default: a run whose completed-batch set extends the batch set
of the latest complete run (its **parent**) reduces only the **delta** batches
and folds them onto the parent's retained roots. Per chromosome, the sparse
metrics' tier tree covers the delta batches, and one **fold unit** (one tier
above the delta root, group 0) sums the delta root and the parent's root; each
ref_blocks window unit adds the parent's dense root rows for its position range
before the delta batches' partials. SUM is associative, so the result is the
full rebuild's (integers exact, floats within the last-ULP contract). When a
run completes it deletes its own lower tiers and every other run's staging
(marking those runs `staging_pruned`), so only the newest complete run's roots
survive -- exactly what the next run needs as its parent. `--rebuild` always
makes a full run from the batch_* partials.

Independent units run in parallel via a spawn-based ProcessPoolExecutor with
per-worker DB + catalog connections (spawn avoids fork-after-threads worker
crashes; the executor fails loud on an abrupt worker death rather than
hanging). The workers make no Iceberg commit (the registrar pattern): a reduce
unit uploads its staging file(s) to deterministic S3 keys and returns the
URIs, and after each barrier (a tier, a fold, the ref_blocks windows) the main
process registers them with one add_files per staging table and then commits
the barrier's checkpoints in one PG transaction; a finalize unit uploads its
agg_* mirror chunks and the main process replaces the mirror in one commit
after the metric's finalize barrier. Concurrent workers therefore never
contend for an Iceberg commit.

Restart: a PG `aggregation_progress` run (keyed by the completed-batch set hash +
fan_in + ref_shard_size + parent run) plus a `tier_completion` per-unit checkpoint let a
killed run resume. A barrier registers its units' staging output in Iceberg
FIRST (idempotent: deterministic S3 key + a resume's registered-URI dedup),
then commits their checkpoints SECOND — so a checkpointed unit is provably
durable and safely skipped (the inverse of the ingest-resume hazard where a
checkpoint outran its data). Each (metric, chromosome)
finalize unit records a sentinel `tier_completion` row (tier -1) in the SAME PG
transaction as its partition swap, so a resume skips a swap that already
committed and rebuilds one that did not. The completion prune commits its own
sentinel ("prune", tier -1) together with the staging_pruned marks on the
other runs, so a run that crashed after beginning its prune resumes (skipping
every unit, redoing the deletes) even though its parent is now marked pruned,
unless a newer complete run has since appeared to fold onto.
Without --rebuild, a batch set whose latest run is complete is up to date: the
command exits without a new run.

Order: allele_counts and site_info first; site_info is finalized to PG before
ref_blocks (whose sparse root reads the variant-site positions from PG
site_info). Integer columns are exact; qual/mqsq_1 (REAL) are summed in float64,
so tier order can shift the last ULP (accepted; not in the vcfdiff keys).
"""

import sys
import io
import os
import time
import argparse
import functools
import hashlib
import random
import tempfile
import traceback
import warnings
import multiprocessing
import concurrent.futures
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import psycopg2
import psycopg2.extras
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
import pyarrow.csv as pacsv  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]
import boto3  # type: ignore[import-untyped]

from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import (
    AlwaysTrue,
    And,
    EqualTo,
    GreaterThan,
    GreaterThanOrEqual,
    In,
    LessThan,
    LessThanOrEqual,
    Or,
)

from .config import (
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
    ICEBERG_NAMESPACE,
    S3_BUCKET,
    build_s3_client_kwargs,
)
from .ingest import (
    _COMMIT_FAILED_EXC,
    _get_registered_uris,
    register_iceberg_uris,
)
from .init_db import (
    AGGREGATE_PG_TABLES,
    AGGREGATION_RUN_COLUMNS,
    ALLELES_DEL_RANGE_INDEX_DEF,
    DENSE_REF_TABLE,
    GTC_FIXED_BITS,
    SchemaError,
    check_removal_lake_schema,
    detached_generation_tables,
    generation_table_name,
)


def get_db_connection() -> Any:
    return psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]


def get_catalog() -> Any:
    return load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)


# The phases of MOSAIC_TEST_FAIL_UNIT, in the order a barrier passes them.
FAIL_UNIT_PHASES = ("pre_register", "pre_ckpt", "post_ckpt")


def parse_fail_unit(
    spec: Optional[str],
) -> Optional[Tuple[str, str, str, str, str]]:
    """Parse ``MOSAIC_TEST_FAIL_UNIT`` = ``<metric>:<c_idx|*>:<tier>:<group>:
    <phase>`` into its five fields (None when unset or empty). ``phase`` is
    one of FAIL_UNIT_PHASES. A malformed value is an error, not ignored (a
    typo would otherwise make a crash test pass without crashing)."""
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) != 5 or parts[4] not in FAIL_UNIT_PHASES or not all(parts):
        raise ValueError(
            f"MOSAIC_TEST_FAIL_UNIT={spec!r}: expected "
            "<metric>:<c_idx|*>:<tier>:<group>:<phase> with phase one of "
            + ", ".join(FAIL_UNIT_PHASES)
        )
    metric, c_idx, tier, group, phase = parts
    return metric, c_idx, tier, group, phase


def _maybe_crash_unit(
    metric: str, c_idx: int, out_tier: int, group_id: int, phase: str
) -> None:
    """Test-only fault injection; a no-op unless ``MOSAIC_TEST_FAIL_UNIT`` is set.

    ``MOSAIC_TEST_FAIL_UNIT="<metric>:<c_idx>:<tier>:<group>:<phase>"`` makes
    the main process raise at the barrier that holds the matching reduce unit
    (a tier of _run_metric_tiers, a fold of _run_fold, the ref_blocks windows
    of _run_ref_windows; tier 1, group = window for a window unit), and only
    when that unit ran (was not skipped) in this attempt: _finish_barrier
    calls this for each of the barrier's done units. ``c_idx`` may be ``*``
    to match any chromosome. ``phase``:

    * ``pre_register``: after the barrier's units ran and uploaded their
      files, before any registration (nothing of the barrier registered);
    * ``pre_ckpt``: after the barrier's Iceberg registration, before its
      tier_completion transaction (registered, not checkpointed: the
      durable-but-unchecked window);
    * ``post_ckpt``: after the checkpoint transaction (checkpointed: skipped
      on resume).

    These bracket the register-Iceberg-first / checkpoint-second window so
    resume tests can prove a checkpointed unit is skipped and an
    un-checkpointed one is harmlessly recomputed (no double-count). The
    completion prune uses it as ``prune:*:-1:0:<phase>`` (_prune_staging:
    ``pre_register`` and ``pre_ckpt`` raise before its sentinel commits,
    ``post_ckpt`` after). Inert in production.
    """
    want = parse_fail_unit(os.environ.get("MOSAIC_TEST_FAIL_UNIT"))
    if want is None:
        return
    want_metric, want_c, want_tier, want_group, want_phase = want
    if want_metric != metric or want_phase != phase:
        return
    if want_c != "*" and want_c != str(c_idx):
        return
    if want_tier != str(out_tier) or want_group != str(group_id):
        return
    raise RuntimeError(
        f"MOSAIC_TEST_FAIL_UNIT: injected crash {metric} c{c_idx} "
        f"t{out_tier} g{group_id} @{phase}"
    )


def _maybe_crash_barrier(keys: Sequence[Tuple[Any, ...]], phase: str) -> None:
    """_maybe_crash_unit for each done unit of a barrier: ``keys`` are their
    tier_completion keys (agg_run_id, metric, chromosome_index, tier,
    group_id)."""
    for _run, metric, c_idx, tier, group_id in keys:
        _maybe_crash_unit(metric, c_idx, tier, group_id, phase)


def _maybe_kill_worker(
    metric: str, c_idx: int, out_tier: int, group_id: int
) -> None:
    """Test-only ABRUPT worker death; a no-op unless ``MOSAIC_TEST_KILL_WORKER``
    is set.

    ``MOSAIC_TEST_KILL_WORKER="<metric>:<c_idx>:<tier>:<group>"`` makes the matching
    reduce unit ``os._exit()`` -- an abrupt, signal-like death (no exception, no
    result) standing in for an OOM SIGKILL or segfault. The old
    ``multiprocessing.Pool`` could not detect this (it hung forever in
    ``imap_unordered``); ``ProcessPoolExecutor`` surfaces it as
    ``BrokenProcessPool``, so a test can prove the aggregate fails promptly instead
    of hanging. ``c_idx`` may be ``*`` to match any chromosome. Inert in production.
    """
    spec = os.environ.get("MOSAIC_TEST_KILL_WORKER")
    if not spec:
        return
    parts = spec.split(":")
    if len(parts) != 4:
        return
    want_metric, want_c, want_tier, want_group = parts
    if want_metric != metric:
        return
    if want_c != "*" and want_c != str(c_idx):
        return
    if want_tier != str(out_tier) or want_group != str(group_id):
        return
    os._exit(137)


def _maybe_crash_finalize(metric: str, c_idx: int, phase: str) -> None:
    """Test-only fault injection; a no-op unless ``MOSAIC_TEST_FAIL_FINALIZE``
    is set.

    ``MOSAIC_TEST_FAIL_FINALIZE="<metric>:<c_idx>:<phase>"`` makes the matching
    (metric, chromosome) finalize unit raise. ``c_idx`` may be ``*`` to match
    any chromosome. ``phase`` is ``pre_commit`` (after the new generation table
    is built and committed, and after the swap statements and the checkpoint
    row, just before the swap commits -- so the swap rolls back, the old
    partition stays attached and the new table is left detached) or
    ``post_commit`` (just after the swap commits -- the unit is durable and
    must be skipped on resume). Mirrors ``_maybe_crash_unit``. Inert in
    production.
    """
    spec = os.environ.get("MOSAIC_TEST_FAIL_FINALIZE")
    if not spec:
        return
    parts = spec.split(":")
    if len(parts) != 3:
        return
    want_metric, want_c, want_phase = parts
    if want_metric != metric or want_phase != phase:
        return
    if want_c != "*" and want_c != str(c_idx):
        return
    raise RuntimeError(
        f"MOSAIC_TEST_FAIL_FINALIZE: injected crash {metric} c{c_idx} "
        f"finalize @{phase}"
    )


def _load_contig_names(conn: Any) -> Dict[int, str]:
    """contig id -> chromosome name, in id order (aggregate.py is a full
    rebuild over every contig; the names match ``ingestion_progress.region``
    for the per-chromosome tier plan)."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, chromosome FROM contigs ORDER BY id")
        return {cid: name for cid, name in cur.fetchall()}


def _load_chrom_lengths(conn: Any) -> Dict[int, int]:
    """contig id -> length (drives the ref_blocks position windows)."""
    with conn.cursor() as cur:
        cur.execute("SELECT id, length FROM contigs")
        return {cid: length for cid, length in cur.fetchall()}


# Default --ref-shard-size: the ref_blocks window length. A window unit holds
# ~16 bytes per base (dp + sample_counts int64), so 20 Mbp is ~320 MB per
# worker; one window per chromosome would be ~4 GB on chr1.
DEFAULT_REF_SHARD_SIZE = 20_000_000

# The tier a ref_blocks window unit writes its sparse root to (group_id =
# window index); the ref_blocks finalize reads every group at this tier.
_REF_ROOT_TIER = 1


def _position_windows(length: int, shard_size: int) -> List[Tuple[int, int]]:
    """Split [1, length] (1-based, inclusive) into windows of <= shard_size bp.

    Returns a single [(1, length)] window when shard_size <= 0 or the chromosome
    fits in one window. The window's index is the ref_blocks window unit's
    group_id (its tier_completion key and its sparse-root staging group).
    """
    if shard_size <= 0 or length <= shard_size:
        return [(1, max(length, 1))]
    out: List[Tuple[int, int]] = []
    start = 1
    while start <= length:
        end = min(start + shard_size - 1, length)
        out.append((start, end))
        start = end + 1
    return out


# ── Arrow SUM-by-key reduce ──────────────────────────────────────────────────
#
# A reduce unit streams its input one Iceberg data file at a time and folds each
# record batch into a _BufferedSum: batches are buffered as Arrow tables and,
# whenever the buffer exceeds _REDUCE_BUFFER_ROWS rows, summed together with the
# running result in one Arrow hash aggregate. Peak memory is ~one data file +
# the buffer + the distinct-key result, instead of a Python object per key.
#
# Values are summed in a "flat" layout: the key columns plus one scalar column
# per value, with site_info's fixed-width list columns exploded element-wise
# (sb -> sb_0..sb_3, gtc -> gtc_0..gtc_2, int64: gtc is in fixed point,
# init_db.GTC_FIXED_BITS) and rebuilt as lists
# for output. Every value may be negative: a `mosaic remove` batch's partials
# are the removed samples' contribution negated, summed like any other.

# Input rows buffered before they are folded into the running result. Larger
# = fewer group_by passes over the running result (CPU), smaller = lower peak
# memory. Override via MOSAIC_REDUCE_BUFFER_ROWS (unit tests pass small bounds
# directly to exercise many flushes).
_REDUCE_BUFFER_ROWS = max(
    1, int(os.environ.get("MOSAIC_REDUCE_BUFFER_ROWS", "4000000"))
)

# Canonical Arrow types of the key columns (staging / final tables + PG).
_KEY_TYPES: Dict[str, Any] = {
    "chromosome_index": pa.int32(),
    "position": pa.int32(),
    "allele": pa.string(),
}


def _flat_value_names(
    value_fields: Sequence[str], list_widths: Sequence[Tuple[str, int]]
) -> Tuple[str, ...]:
    """Value column names in the flat SUM layout (lists exploded)."""
    widths = dict(list_widths)
    out: List[str] = []
    for v in value_fields:
        if v in widths:
            out.extend(f"{v}_{j}" for j in range(widths[v]))
        else:
            out.append(v)
    return tuple(out)


def _sum_by_key(
    key_fields: Sequence[str], value_fields: Sequence[str], tables: List[Any]
) -> Any:
    """SUM every flat value column by key over the concatenation of ``tables``.

    One Arrow hash aggregate, single-threaded (``use_threads=False``): Arrow then
    folds each group's values in input-row order, so with the running result
    placed first (see _BufferedSum) a float sum is the same left fold as the old
    per-row Python accumulator, whatever the buffer-flush boundaries. Integer
    sums are int64 (int32 partials widen; no overflow). Nulls are skipped and an
    all-null group is filled with 0, matching the old ``x or 0``: staging and
    final output never carry nulls.
    """
    table = tables[0] if len(tables) == 1 else pa.concat_tables(tables)
    grouped = table.group_by(list(key_fields), use_threads=False).aggregate(
        [(v, "sum") for v in value_fields]
    )
    cols = [grouped.column(k) for k in key_fields] + [
        pc.fill_null(grouped.column(f"{v}_sum"), 0) for v in value_fields
    ]
    return pa.Table.from_arrays(
        cols, names=list(key_fields) + list(value_fields)
    )


# ── metric specs ─────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MetricSpec:
    name: str
    tier0_table: str
    staging_table: str
    agg_table: str
    s3_prefix: str
    key_fields: Tuple[str, ...]
    value_fields: Tuple[str, ...]
    # Arrow type of each value column in the staging / final tables: int64 and
    # float64 sums, or a fixed-width list of element-wise sums (site_info's
    # sb list<int64>, gtc list<int64> in fixed point).
    value_types: Tuple[Any, ...]
    # Width of each list-valued column (site_info sb=4, gtc=3; PG CHECKs the
    # same lengths). Lists are summed element-wise as scalar columns of the
    # list's element type.
    list_widths: Tuple[Tuple[str, int], ...]
    reduce: Callable  # (flat tables) -> flat SUM-by-key table
    # finalize: the PG parent table (LIST-partitioned by chromosome_index;
    # its primary key is key_fields) and the COPY of one Arrow chunk into a
    # named generation table: pg_copy(cur, target, chunk, freeze=False).
    # pg_rows(chunk) -> the chunk's rows as PostgreSQL holds them (dead rows
    # dropped, gtc's integer part; the agg_* mirror gets the chunk itself).
    pg_table: str
    pg_copy: Callable
    pg_rows: Callable
    # Secondary indexes (``USING ... (...)`` clauses) each generation table
    # builds after its primary key, before it is attached: the parent
    # declares the same definitions, so the ATTACH adopts them.
    pg_indexes: Tuple[str, ...] = ()

    def flat_schema(self) -> Any:
        """Arrow schema of the flat SUM layout (see _flatten)."""
        widths = dict(self.list_widths)
        fields = [pa.field(k, _KEY_TYPES[k]) for k in self.key_fields]
        for v, t in zip(self.value_fields, self.value_types):
            if v in widths:
                fields.extend(
                    pa.field(f"{v}_{j}", t.value_type)
                    for j in range(widths[v])
                )
            else:
                fields.append(pa.field(v, t))
        return pa.schema(fields)

    def _key_value_fields(self) -> List[Any]:
        return [
            pa.field(k, _KEY_TYPES[k], nullable=False) for k in self.key_fields
        ] + [
            pa.field(v, t, nullable=True)
            for v, t in zip(self.value_fields, self.value_types)
        ]

    def staging_schema(self) -> Any:
        """Arrow schema of a tier_* staging file (matches init_db's tables)."""
        return pa.schema(
            [
                pa.field("agg_run_id", pa.int32(), nullable=False),
                pa.field("tier", pa.int32(), nullable=False),
                pa.field("group_id", pa.int32(), nullable=False),
            ]
            + self._key_value_fields()
        )

    def agg_schema(self) -> Any:
        """Arrow schema of the agg_* mirror and of a finalize chunk."""
        return pa.schema(self._key_value_fields())


def _flatten(spec: MetricSpec, data: Any) -> Any:
    """Project a scanned record batch (or table) onto the flat SUM layout: the
    key columns in their canonical types plus one scalar value column each, with
    list columns exploded element-wise. Nulls stay null (the sum skips them).
    """
    widths = dict(spec.list_widths)
    cols: List[Any] = []
    names: List[str] = []
    for k in spec.key_fields:
        cols.append(data.column(k).cast(_KEY_TYPES[k]))
        names.append(k)
    for v, typ in zip(spec.value_fields, spec.value_types):
        col = data.column(v)
        if v in widths:
            for j in range(widths[v]):
                cols.append(pc.list_element(col, j).cast(typ.value_type))
                names.append(f"{v}_{j}")
        else:
            cols.append(col.cast(typ))
            names.append(v)
    return pa.Table.from_arrays(cols, names=names)


def _fill_values(spec: MetricSpec, flat: Any) -> Any:
    """Replace nulls in a flat table's value columns with 0 (the old ``x or 0``)."""
    keys = set(spec.key_fields)
    return pa.Table.from_arrays(
        [
            col if name in keys else pc.fill_null(col, 0)
            for name, col in zip(flat.column_names, flat.columns)
        ],
        names=flat.column_names,
    )


def _unflatten(spec: MetricSpec, flat: Any, sort: bool = True) -> Any:
    """Inverse of _flatten for a null-free flat table: the key + value columns
    in the agg_* schema, list columns rebuilt from their element columns with
    numpy (no Python lists). ``sort`` orders the rows by key so a unit's staging
    file is deterministic (same rows, same order as the old sorted dict)."""
    if sort:
        flat = flat.sort_by([(k, "ascending") for k in spec.key_fields])
    flat = flat.combine_chunks()
    widths = dict(spec.list_widths)
    n = flat.num_rows
    cols: List[Any] = [flat.column(k) for k in spec.key_fields]
    for v, typ in zip(spec.value_fields, spec.value_types):
        if v in widths:
            w = widths[v]
            elem_type = typ.value_type
            elems = np.column_stack(
                [
                    np.asarray(
                        flat.column(f"{v}_{j}").to_numpy(),
                        elem_type.to_pandas_dtype(),
                    )
                    for j in range(w)
                ]
            ).reshape(-1)
            offsets = np.arange(0, n * w + 1, w, dtype=np.int32)
            cols.append(
                pa.ListArray.from_arrays(
                    pa.array(offsets), pa.array(elems, elem_type)
                )
            )
        else:
            cols.append(flat.column(v).cast(typ))
    return pa.Table.from_arrays(cols, schema=spec.agg_schema())


def _staging_table(
    spec: MetricSpec, result: Any, run_id: int, tier: int, gid: int
) -> Any:
    """A reduce unit's tier_* rows: its (sorted) result with the constant
    agg_run_id / tier / group_id columns prepended."""
    n = result.num_rows
    head = [
        pa.array(np.full(n, value, dtype=np.int32))
        for value in (run_id, tier, gid)
    ]
    return pa.Table.from_arrays(
        head + list(result.columns), schema=spec.staging_schema()
    )


class _BufferedSum:
    """Streaming SUM-by-key over flat tables with a bounded input buffer.

    ``add`` buffers a flat table; once more than ``buffer_rows`` input rows are
    buffered they are summed together with the running result (placed FIRST, so
    each key's float sum stays a left fold in arrival order). ``result`` flushes
    and returns the flat SUM-by-key table (unsorted).
    """

    def __init__(self, spec: MetricSpec, buffer_rows: Optional[int] = None):
        self._spec = spec
        self._bound = max(1, buffer_rows or _REDUCE_BUFFER_ROWS)
        self._acc: Optional[Any] = None
        self._pending: List[Any] = []
        self._pending_rows = 0

    def add(self, flat: Any) -> None:
        if flat.num_rows == 0:
            return
        self._pending.append(flat)
        self._pending_rows += flat.num_rows
        if self._pending_rows >= self._bound:
            self._flush()

    def _flush(self) -> None:
        if not self._pending:
            return
        parts = ([self._acc] if self._acc is not None else []) + self._pending
        self._acc = self._spec.reduce(parts)
        self._pending = []
        self._pending_rows = 0

    def result(self) -> Any:
        self._flush()
        if self._acc is None:
            return self._spec.flat_schema().empty_table()
        return self._acc


# ── ref_blocks window fold ───────────────────────────────────────────────────
#
# batch_ref_blocks is dense: one row per covered (batch, position). Summing it
# over one position window is plain array arithmetic indexed by position, so a
# ref_blocks window unit folds every covering batch's rows into numpy arrays
# (no hash group-by, no site filter while reading) and then emits the dense
# root (every covered position) and the sparse root (covered variant sites).


class _WindowFold:
    """Dense SUM of ``dp`` / ``sample_counts`` per position over one inclusive
    position window [lo, hi] of one chromosome.

    A position yields a root row iff its net sum is not (0, 0) (the dense
    root and the sparse root alike). Ingest's partials omit (0, 0)
    positions, so while every partial is positive this is the old "touched
    by any partial row" rule; a `mosaic remove` batch's negative partials
    net a position only removed samples covered to (0, 0), and it goes, as
    a fresh ingest of the remaining samples would never have written it.
    Nulls count as 0. Memory is 16 bytes per window base (two int64
    arrays).
    """

    def __init__(self, lo: int, hi: int):
        if hi < lo:
            raise ValueError(f"empty window [{lo}, {hi}]")
        self.lo = int(lo)
        self.hi = int(hi)
        n = self.hi - self.lo + 1
        self.dp = np.zeros(n, dtype=np.int64)
        self.sc = np.zeros(n, dtype=np.int64)

    def _live(self, start: int = 0, stop: Optional[int] = None) -> Any:
        """Mask of the positions in [start, stop) whose net sum is not
        (0, 0)."""
        return (self.dp[start:stop] != 0) | (self.sc[start:stop] != 0)

    @property
    def n_live(self) -> int:
        """Positions whose net sum is not (0, 0): the dense root's rows."""
        return int(np.count_nonzero(self._live()))

    def add(self, batch: Any) -> None:
        """Add a record batch (or table) of ``position``, ``dp``,
        ``sample_counts`` rows, all inside the window. Values may be
        negative (a removal batch).

        ``np.bincount`` sums duplicate positions correctly (``np.add.at`` is
        duplicate-safe too but far slower; plain fancy-index ``+=`` is not
        duplicate-safe). It runs over the batch's own position span, not the
        whole window, so the cost per batch is O(rows + span). Its float64
        weights are exact while a position's per-batch |sum| stays below
        2**53.
        """
        if batch.num_rows == 0:
            return
        idx = (
            np.asarray(
                batch.column("position").to_numpy(zero_copy_only=False),
                dtype=np.int64,
            )
            - self.lo
        )
        first, last = int(idx.min()), int(idx.max())
        if first < 0 or last >= self.dp.size:
            raise ValueError(
                f"position {first + self.lo if first < 0 else last + self.lo}"
                f" outside the window [{self.lo}, {self.hi}]"
            )
        rel = idx - first
        span = last - first + 1
        for acc, name in ((self.dp, "dp"), (self.sc, "sample_counts")):
            values = pc.fill_null(batch.column(name), 0).to_numpy(
                zero_copy_only=False
            )
            acc[first : last + 1] += np.bincount(
                rel, weights=values, minlength=span
            ).astype(np.int64)

    def rows(
        self, start: int = 0, stop: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position, dp, sample_counts) of the positions whose window
        offset is in [start, stop) and whose net sum is not (0, 0), in
        position order: the dense root."""
        t = np.flatnonzero(self._live(start, stop)) + start
        return t + self.lo, self.dp[t], self.sc[t]

    def at(
        self, positions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(position, dp, sample_counts) at the positions among
        ``positions`` (sorted, e.g. the window's variant sites) whose net sum
        is not (0, 0): the sparse root. Positions outside the window are
        ignored."""
        rel = np.asarray(positions, dtype=np.int64) - self.lo
        rel = rel[(rel >= 0) & (rel < self.dp.size)]
        rel = rel[(self.dp[rel] != 0) | (self.sc[rel] != 0)]
        return rel + self.lo, self.dp[rel], self.sc[rel]


# Arrow schema of a tier_ref_blocks_dense data file (matches init_db's table).
_DENSE_ROOT_SCHEMA = pa.schema(
    [
        pa.field("agg_run_id", pa.int32(), nullable=False),
        pa.field("chromosome_index", pa.int32(), nullable=False),
        pa.field("window_id", pa.int32(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("dp", pa.int64(), nullable=True),
        pa.field("sample_counts", pa.int64(), nullable=True),
    ]
)

# Window positions per dense-root write (and so per Parquet row group): bounds
# the Arrow copy of the dense root to one slice instead of the whole window.
_DENSE_WRITE_CHUNK = 1 << 20

# Parquet options of the dense root. Its positions are unique and nearly
# consecutive, so dictionary encoding fails and plain + snappy leaves ~4.5
# bytes per row (89% of the file on chr21); DELTA_BINARY_PACKED stores them in
# well under a byte. Measured on six chr21 samples: 5.1 -> 0.56 bytes per row.
_DENSE_PARQUET_OPTIONS: Dict[str, Any] = {
    "use_dictionary": [
        f.name for f in _DENSE_ROOT_SCHEMA if f.name != "position"
    ],
    "column_encoding": {"position": "DELTA_BINARY_PACKED"},
}


def _dense_root_table(
    agg_run_id: int,
    c_idx: int,
    window_id: int,
    positions: np.ndarray,
    dp: np.ndarray,
    sc: np.ndarray,
) -> Any:
    """Dense-root rows in the tier_ref_blocks_dense schema."""
    n = int(positions.size)
    head = [
        pa.array(np.full(n, value, dtype=np.int32))
        for value in (agg_run_id, c_idx, window_id)
    ]
    return pa.Table.from_arrays(
        head
        + [
            pa.array(positions.astype(np.int32)),
            pa.array(dp, pa.int64()),
            pa.array(sc, pa.int64()),
        ],
        schema=_DENSE_ROOT_SCHEMA,
    )


def _write_dense_root(
    path: str, fold: _WindowFold, agg_run_id: int, c_idx: int, window_id: int
) -> int:
    """Write a window's dense root to a local Parquet file, one row group
    per _DENSE_WRITE_CHUNK window positions, sorted by position. Returns the
    row count."""
    rows = 0
    with pq.ParquetWriter(
        path, _DENSE_ROOT_SCHEMA, **_DENSE_PARQUET_OPTIONS
    ) as writer:
        for start in range(0, fold.dp.size, _DENSE_WRITE_CHUNK):
            positions, dp, sc = fold.rows(start, start + _DENSE_WRITE_CHUNK)
            if positions.size:
                writer.write_table(
                    _dense_root_table(
                        agg_run_id, c_idx, window_id, positions, dp, sc
                    )
                )
                rows += int(positions.size)
    return rows


def _ref_sparse_result(
    c_idx: int, positions: np.ndarray, dp: np.ndarray, sc: np.ndarray
) -> Any:
    """Sparse-root rows in the ref_blocks agg_* schema (sorted by position
    when ``positions`` is), ready for _staging_table."""
    n = int(positions.size)
    return pa.Table.from_arrays(
        [
            pa.array(np.full(n, c_idx, dtype=np.int32)),
            pa.array(positions.astype(np.int32)),
            pa.array(dp, pa.int64()),
            pa.array(sc, pa.int64()),
        ],
        schema=SPECS["ref_blocks"].agg_schema(),
    )


# ── PG finalize COPY, streamed in chunks ─────────────────────────────────────
#
# _finalize_chrom COPYs a chromosome's root-tier rows into its new generation
# table one Arrow chunk at a time (pg_copy per chunk), so a worker never holds
# the chromosome's whole result. PG content is keyed by coordinate, so chunk
# boundaries and row order do not affect it.

# Arrow CSV options that produce PostgreSQL COPY text: tab-delimited, no header,
# no quoting. quoting_style="none" raises if a value contains a tab, newline or
# quote, instead of writing a field COPY would misparse.
_COPY_CSV_OPTIONS = pacsv.WriteOptions(
    include_header=False, delimiter="\t", quoting_style="none"
)


def _pg_copy_text(chunk: Any) -> bytes:
    """Render a null-free Arrow chunk as PostgreSQL COPY text.

    Numbers are written as text (floats as the shortest round-trip decimal:
    ``15`` where Python wrote ``15.0`` -- the same value) and list columns as
    array literals ``{a,b,...}`` built with Arrow string kernels. The chunk must
    be null-free (_final_rows fills 0): an empty CSV field is not a COPY NULL
    and would fail on a numeric column.
    """
    cols = []
    for col in chunk.columns:
        if pa.types.is_list(col.type) or pa.types.is_large_list(col.type):
            col = pc.binary_join_element_wise(
                "{",
                pc.binary_join(pc.cast(col, pa.list_(pa.string())), ","),
                "}",
                "",
            )
        cols.append(col)
    buf = io.BytesIO()
    pacsv.write_csv(
        pa.Table.from_arrays(cols, names=chunk.column_names),
        buf,
        _COPY_CSV_OPTIONS,
    )
    return buf.getvalue()


def _copy_arrow(
    cur: Any,
    target: str,
    columns: Sequence[str],
    chunk: Any,
    freeze: bool = False,
) -> None:
    """COPY one Arrow chunk (columns in ``columns`` order) into ``target``.

    ``freeze`` adds ``WITH (FREEZE)``: the rows are written frozen and their
    pages all-visible. PostgreSQL accepts it only when ``target`` was created
    or truncated in the current transaction."""
    with_freeze = " WITH (FREEZE)" if freeze else ""
    cur.copy_expert(
        f"COPY {target} ({', '.join(columns)}) FROM STDIN{with_freeze}",
        io.BytesIO(_pg_copy_text(chunk)),
    )


def allele_del_range(position: int, allele: str) -> Optional[Tuple[int, int]]:
    """The half-open span ``[lo, hi)`` a catalog allele deletes, or None.

    A deletion allele is ``-`` plus the trimmed reference including its
    anchor base (format_alt_allele), so its span is ``[position, position +
    len(allele) - 1)``. Ingest used to store ``[pos, pos + (len(REF) -
    len(ALT)) + 1)`` from the untrimmed record: the same span whenever the
    trimmed ALT is the single anchor base, i.e. every ordinary deletion. A
    complex substitution (``ATG>CA``) encodes as ``-ATG`` too and gets the
    ordinary span (a known limitation of the encoding). Every other allele
    (ref base, ``x``, ``+``, ``*``) has none.
    """
    if not allele.startswith("-"):
        return None
    return int(position), int(position) + len(allele) - 1


def allele_del_range_text(positions: Any, alleles: Any) -> Any:
    """Arrow-vectorized allele_del_range: the PostgreSQL INT4RANGE text
    ``[lo,hi)`` of each deletion allele, null for every other allele."""
    lo = pc.cast(positions, pa.int64())
    hi = pc.add(
        lo, pc.cast(pc.subtract(pc.utf8_length(alleles), 1), pa.int64())
    )
    text = pc.binary_join_element_wise(
        "[", pc.cast(lo, pa.string()), ",", pc.cast(hi, pa.string()), ")", ""
    )
    return pc.if_else(
        pc.starts_with(alleles, "-"), text, pa.scalar(None, pa.string())
    )


# COPY text's NULL marker (text format).
_COPY_NULL = "\\N"


def _alleles_copy_chunk(chunk: Any) -> Any:
    """An allele_counts chunk (agg_* schema) -> the alleles catalog's COPY
    columns: the keys, the derived del_range (``\\N`` where null) and the
    summed count."""
    del_range = allele_del_range_text(
        chunk.column("position"), chunk.column("allele")
    )
    return pa.Table.from_arrays(
        [
            chunk.column("chromosome_index"),
            chunk.column("position"),
            chunk.column("allele"),
            pc.fill_null(del_range, _COPY_NULL),
            chunk.column("count"),
        ],
        names=["chromosome_index", "position", "allele", "del_range", "count"],
    )


def _pg_copy_allele(
    cur: Any, target: str, chunk: Any, freeze: bool = False
) -> None:
    _copy_arrow(
        cur,
        target,
        ("chromosome_index", "position", "allele", "del_range", "count"),
        _alleles_copy_chunk(chunk),
        freeze,
    )


def _pg_copy_site(
    cur: Any, target: str, chunk: Any, freeze: bool = False
) -> None:
    _copy_arrow(
        cur,
        target,
        (
            "chromosome_index",
            "position",
            "qual",
            "dp",
            "mqsq_1",
            "mqsq_2",
            "sb",
            "gtc",
        ),
        chunk,
        freeze,
    )


def _pg_copy_ref(
    cur: Any, target: str, chunk: Any, freeze: bool = False
) -> None:
    _copy_arrow(
        cur,
        target,
        ("chromosome_index", "position", "dp", "sample_counts"),
        chunk,
        freeze,
    )


def _ref_site_positions(
    conn: Any, c_idx: int, window: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """Sorted variant-site positions on a chromosome (where a ref_blocks
    window unit gathers its sparse root), optionally restricted to an
    inclusive position window [start, end].

    Streamed with ``COPY (...) TO STDOUT`` and parsed by Arrow's CSV reader, so a
    chromosome's ~10M positions never become Python ints."""
    with conn.cursor() as cur:
        if window is None:
            query = cur.mogrify(
                "SELECT position FROM site_info WHERE chromosome_index = %s "
                "ORDER BY position",
                (int(c_idx),),
            )
        else:
            query = cur.mogrify(
                "SELECT position FROM site_info WHERE chromosome_index = %s "
                "AND position BETWEEN %s AND %s ORDER BY position",
                (int(c_idx), int(window[0]), int(window[1])),
            )
        buf = io.BytesIO()
        cur.copy_expert(f"COPY ({query.decode()}) TO STDOUT", buf)
    data = buf.getvalue()
    if not data:
        return np.empty(0, dtype=np.int64)
    table = pacsv.read_csv(
        io.BytesIO(data),
        read_options=pacsv.ReadOptions(column_names=["position"]),
        convert_options=pacsv.ConvertOptions(
            column_types={"position": pa.int64()}
        ),
    )
    return np.asarray(table.column("position").to_numpy(), dtype=np.int64)


def _integer_part(col: Any) -> Any:
    """A list<int64> column of fixed-point sums (GTC_FIXED_BITS fraction
    bits) -> list<int64> of their integer parts: an arithmetic shift, so
    the floor (a negative net sum, which a correct removal never leaves,
    would floor too). The arrays are null-free (_final_rows)."""
    arr = col.combine_chunks() if isinstance(col, pa.ChunkedArray) else col
    values = np.asarray(
        pc.list_flatten(arr).to_numpy(zero_copy_only=False), np.int64
    )
    lengths = np.asarray(
        pc.list_value_length(arr).to_numpy(zero_copy_only=False), np.int64
    )
    offsets = np.zeros(len(lengths) + 1, dtype=np.int32)
    np.cumsum(lengths, out=offsets[1:])
    return pa.ListArray.from_arrays(
        pa.array(offsets), pa.array(values >> GTC_FIXED_BITS, pa.int64())
    )


def _site_pg_rows(chunk: Any) -> Any:
    """site_info root rows (agg_* schema) -> the rows PostgreSQL site_info
    holds: a site whose net n_records is 0 is dropped (every record there
    belonged to a removed sample: a fresh ingest of the rest has no such
    site), and gtc, summed in fixed point (each record quantized once at
    ingest: exact and order-independent), becomes its integer part in PG's
    BIGINT[] (a one-batch cohort's PG gtc equals the old per-batch
    truncation away from an integer boundary; a multi-batch cohort's is now
    the integer part of the cohort sum instead of the sum of each batch's).
    The agg_site_info mirror keeps every root row and the fixed-point
    sums."""
    chunk = chunk.filter(pc.not_equal(chunk.column("n_records"), 0))
    return pa.Table.from_arrays(
        [chunk.column(k) for k in ("chromosome_index", "position", "qual")]
        + [chunk.column(k) for k in ("dp", "mqsq_1", "mqsq_2", "sb")]
        + [_integer_part(chunk.column("gtc"))],
        names=[
            "chromosome_index",
            "position",
            "qual",
            "dp",
            "mqsq_1",
            "mqsq_2",
            "sb",
            "gtc",
        ],
    )


def _allele_pg_rows(chunk: Any) -> Any:
    """allele_counts root rows -> the rows of the PostgreSQL alleles
    catalog: an allele whose net n_listed is 0 is dropped (only removed
    samples' records listed it). A count-0 allele a live record lists stays,
    as before. The agg_allele_counts mirror keeps every root row."""
    return chunk.filter(pc.not_equal(chunk.column("n_listed"), 0))


def _identity_rows(chunk: Any) -> Any:
    """ref_blocks: the sparse root is already the PG rows (the window fold
    emits no net (0, 0) position, and gathers only PG site_info sites)."""
    return chunk


_SITE_LIST_WIDTHS = (("sb", 4), ("gtc", 3))
_SITE_VALUES = ("qual", "dp", "mqsq_1", "mqsq_2", "sb", "gtc", "n_records")

SPECS: Dict[str, MetricSpec] = {
    "allele_counts": MetricSpec(
        name="allele_counts",
        tier0_table="batch_allele_counts",
        staging_table="tier_allele_counts",
        agg_table="agg_allele_counts",
        s3_prefix="tier_allele_counts",
        key_fields=("chromosome_index", "position", "allele"),
        value_fields=("count", "n_listed"),
        value_types=(pa.int64(), pa.int64()),
        list_widths=(),
        reduce=functools.partial(
            _sum_by_key,
            ("chromosome_index", "position", "allele"),
            ("count", "n_listed"),
        ),
        # The allele catalog (zero counts kept: every allele a live record
        # lists).
        pg_table="alleles",
        pg_copy=_pg_copy_allele,
        pg_rows=_allele_pg_rows,
        pg_indexes=(ALLELES_DEL_RANGE_INDEX_DEF,),
    ),
    "site_info": MetricSpec(
        name="site_info",
        tier0_table="batch_site_info",
        staging_table="tier_site_info",
        agg_table="agg_site_info",
        s3_prefix="tier_site_info",
        key_fields=("chromosome_index", "position"),
        value_fields=_SITE_VALUES,
        value_types=(
            pa.float64(),
            pa.int64(),
            pa.float64(),
            pa.int64(),
            pa.list_(pa.int64()),
            pa.list_(pa.int64()),
            pa.int64(),
        ),
        list_widths=_SITE_LIST_WIDTHS,
        reduce=functools.partial(
            _sum_by_key,
            ("chromosome_index", "position"),
            _flat_value_names(_SITE_VALUES, _SITE_LIST_WIDTHS),
        ),
        pg_table="site_info",
        pg_copy=_pg_copy_site,
        pg_rows=_site_pg_rows,
    ),
    "ref_blocks": MetricSpec(
        name="ref_blocks",
        tier0_table="batch_ref_blocks",
        staging_table="tier_ref_blocks",
        agg_table="agg_ref_blocks",
        s3_prefix="tier_ref_blocks",
        key_fields=("chromosome_index", "position"),
        value_fields=("dp", "sample_counts"),
        value_types=(pa.int64(), pa.int64()),
        list_widths=(),
        reduce=functools.partial(
            _sum_by_key,
            ("chromosome_index", "position"),
            ("dp", "sample_counts"),
        ),
        pg_table="ref_blocks",
        pg_copy=_pg_copy_ref,
        pg_rows=_identity_rows,
    ),
}

# The tier-tree metrics, reduced first; site_info is finalized to PG before the
# ref_blocks window units (their sparse root reads PG site_info).
PRE_METRICS = ("allele_counts", "site_info")


# ── Iceberg staging I/O ──────────────────────────────────────────────────────


def _make_s3_client() -> Any:
    """boto3 S3 client (MinIO endpoint+creds for local rest catalog; else IAM)."""
    s3_kwargs = build_s3_client_kwargs()
    return boto3.client("s3", **s3_kwargs)  # type: ignore[call-overload]


def _iter_scan_record_batches(
    table: Any, row_filter: Any, fields: Tuple[str, ...]
) -> Any:
    """Yield Arrow record batches for (row_filter, fields), bounded to ONE data
    file at a time.

    ``table.scan(...).to_arrow_batch_reader()`` hands ALL of ``plan_files()`` to
    ``ArrowScan.to_record_batches`` in a single call, which ``executor.map``s over
    every file and buffers the completed files (pyiceberg 0.11.0
    io/pyarrow.py:1759) -- with no backpressure the peak is ~the whole scan, which
    OOMs the dense ref_blocks reduce at scale. Feeding ``plan_files()`` one task
    at a time caps the peak at a single data file. Same files, same order, same
    bound residual filter and projection as ``to_arrow()`` /
    ``to_arrow_batch_reader()`` (it reuses the identical ``ArrowScan`` /
    per-file read path), so downstream results are byte-identical.
    """
    from pyiceberg.io.pyarrow import ArrowScan

    scan = table.scan(row_filter=row_filter, selected_fields=fields)
    arrow_scan = ArrowScan(
        scan.table_metadata,
        scan.io,
        scan.projection(),
        scan.row_filter,
        scan.case_sensitive,
        scan.limit,
    )
    for task in scan.plan_files():
        yield from arrow_scan.to_record_batches([task])


def _tier_s3_key(
    s3_prefix: str, agg_run_id: int, c_idx: int, out_tier: int, group_id: int
) -> str:
    """Deterministic S3 key of one unit's staging file: re-running the unit
    (a resume) overwrites the same key instead of adding a second file."""
    key_hash = hashlib.sha256(
        f"{agg_run_id}:{s3_prefix}:c{c_idx}:t{out_tier}:g{group_id}".encode()
    ).hexdigest()[:24]
    return f"{s3_prefix}/{key_hash}/data.parquet"


def _upload_parquet(
    write: Callable[[str], Any], s3_client: Any, s3_key: str
) -> str:
    """Write a Parquet file locally with ``write(path)``, upload it to
    ``s3_key`` and remove the local copy. Returns the s3:// URI."""
    tmp = tempfile.NamedTemporaryFile(suffix=".parquet", delete=False).name
    try:
        write(tmp)
        s3_client.upload_file(tmp, S3_BUCKET, s3_key)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return f"s3://{S3_BUCKET}/{s3_key}"


def _upload_tier_parquet(
    arrow_table: Any,
    s3_client: Any,
    s3_prefix: str,
    agg_run_id: int,
    c_idx: int,
    out_tier: int,
    group_id: int,
) -> str:
    """Write one reduce unit's staging parquet to a deterministic S3 key.

    Re-running the same unit overwrites the same key (idempotent), so resume +
    register-time URI dedup give exactly-once. Returns the s3:// URI.
    """
    return _upload_parquet(
        lambda path: pq.write_table(arrow_table, path),
        s3_client,
        _tier_s3_key(s3_prefix, agg_run_id, c_idx, out_tier, group_id),
    )


# A unit's uploaded files: (Iceberg table name, s3:// URI or None) pairs, in
# registration order. None = the unit wrote no file for that table (a result
# with no rows, a window with no live position).
UnitFiles = Tuple[Tuple[str, Optional[str]], ...]


def group_barrier_uris(files: Sequence[UnitFiles]) -> Dict[str, List[str]]:
    """{table: URIs} over a barrier's done units' files, in first-seen table
    order and unit order, None skipped; a table with no URI is absent."""
    out: Dict[str, List[str]] = {}
    for unit_files in files:
        for table_name, uri in unit_files:
            if uri is not None:
                out.setdefault(table_name, []).append(uri)
    return out


def register_barrier(
    catalog: Any, uris_by_table: Dict[str, List[str]], fresh: bool
) -> Tuple[int, int]:
    """Register a barrier's staging files: one add_files commit per table
    (the registrar pattern: the units only upload). Returns (files added,
    commits).

    Fresh run: every unit's S3 key folds in the SERIAL agg_run_id, so no URI
    can already be registered -> no registered-URI scan, no duplicate check
    (the aggregate half of the O(N^2) registration fix).
    A resume may have registered some of these URIs before crashing (between
    this registration and the barrier's checkpoint transaction), so it scans
    each table's registered URIs once, drops those, and keeps
    check_duplicate_files on the rest. A table left with nothing to add makes
    no commit."""
    added = commits = 0
    for table_name, uris in uris_by_table.items():
        staging = catalog.load_table(f"{ICEBERG_NAMESPACE}.{table_name}")
        registered = set() if fresh else _get_registered_uris(staging)
        todo = [u for u in uris if u not in registered]
        if not todo:
            continue
        register_iceberg_uris(
            staging, set(), todo, check_duplicate_files=not fresh
        )
        added += len(todo)
        commits += 1
    return added, commits


def _unit_checkpointed(conn: Any, key: Tuple[Any, ...]) -> bool:
    """True iff the tier_completion row ``key`` = (agg_run_id, metric,
    chromosome_index, tier, group_id) exists: that unit's output is durable."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM tier_completion WHERE agg_run_id=%s AND "
            "metric=%s AND chromosome_index=%s AND tier=%s AND group_id=%s",
            key,
        )
        return cur.fetchone() is not None


def _commit_checkpoints(conn: Any, keys: Sequence[Tuple[Any, ...]]) -> None:
    """Insert the tier_completion rows ``keys`` and commit, in one
    transaction (none: no statement, but whatever else is open on ``conn``
    commits, as _prune_staging relies on). Called only after the units'
    staging output is registered."""
    if keys:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur,
                "INSERT INTO tier_completion (agg_run_id, metric, "
                "chromosome_index, tier, group_id) VALUES %s "
                "ON CONFLICT DO NOTHING",
                [tuple(k) for k in keys],
            )
    conn.commit()


def _commit_checkpoint(conn: Any, key: Tuple[Any, ...]) -> None:
    """_commit_checkpoints of one key."""
    _commit_checkpoints(conn, [key])


def _finish_barrier(
    conn: Any,
    catalog: Any,
    done: Sequence[Tuple[Tuple[Any, ...], UnitFiles]],
    fresh: bool,
) -> Tuple[int, int]:
    """After a barrier's executor.map is drained, in the main process:
    register the done units' files (register_barrier: one add_files per
    staging table), THEN commit their tier_completion rows in one
    PostgreSQL transaction -- Iceberg first, PostgreSQL second, so a
    checkpointed unit is provably durable. ``done`` is (checkpoint key,
    files) per unit that ran in this attempt (skipped units have nothing to
    register or checkpoint). The MOSAIC_TEST_FAIL_UNIT phases bracket the
    two steps. Returns register_barrier's (files added, commits)."""
    keys = [key for key, _ in done]
    _maybe_crash_barrier(keys, "pre_register")
    counts = register_barrier(
        catalog, group_barrier_uris([f for _, f in done]), fresh
    )
    _maybe_crash_barrier(keys, "pre_ckpt")
    _commit_checkpoints(conn, keys)
    _maybe_crash_barrier(keys, "post_ckpt")
    return counts


# ── parallel reduce workers ──────────────────────────────────────────────────

_WORKER: Dict[str, Any] = {}


def _init_worker() -> None:
    _WORKER["catalog"] = get_catalog()
    _WORKER["s3"] = _make_s3_client()


# A reduce unit's input: a list of (Iceberg table name, row filter) sources,
# summed together. An ordinary tier-tree unit has one source (tier-0 batch_*
# partials, or a lower tier's staging groups); the fold unit of an incremental
# run has two (this run's delta root and the parent run's root).
Source = Tuple[str, Any]


def _scan_table(
    catalog: Any, table_name: str, row_filter: Any, fields: Tuple[str, ...]
) -> Any:
    """Record batches of ``table_name`` (in ICEBERG_NAMESPACE) for
    (row_filter, fields), one data file at a time."""
    table = catalog.load_table(f"{ICEBERG_NAMESPACE}.{table_name}")
    return _iter_scan_record_batches(table, row_filter, fields)


def _sum_sources(
    spec: MetricSpec,
    sources: Sequence[Source],
    scan: Callable[[str, Any, Tuple[str, ...]], Any],
    buffer_rows: Optional[int] = None,
) -> Any:
    """SUM-by-key over every source's rows: the agg_* schema, sorted by key.

    ``scan(table_name, row_filter, fields)`` yields the record batches of one
    source (``_scan_table`` in a worker; unit tests pass an in-memory scan).
    Sources are read in order, so a float sum is the same left fold whatever
    the buffer bound. The filter columns (batch_id, agg_run_id, tier,
    group_id) are read by pyiceberg for the residual filter but need not be
    projected.
    """
    acc = _BufferedSum(spec, buffer_rows)
    fields = spec.key_fields + spec.value_fields
    for table_name, row_filter in sources:
        for batch in scan(table_name, row_filter, fields):
            acc.add(_flatten(spec, batch))
    return _unflatten(spec, acc.result())


def _batch_source(
    spec: MetricSpec, c_idx: int, batch_ids: List[int]
) -> Source:
    """Tier-0 source: the batch_* partials of ``batch_ids`` on ``c_idx``."""
    return (
        spec.tier0_table,
        And(
            EqualTo(  # type: ignore[misc, call-arg]
                "chromosome_index", int(c_idx)
            ),
            In(  # type: ignore[misc, call-arg]
                "batch_id", [int(b) for b in batch_ids]
            ),
        ),
    )


def _staging_source(
    spec: MetricSpec,
    c_idx: int,
    agg_run_id: int,
    tier: int,
    group_ids: Optional[List[int]] = None,
) -> Source:
    """Staging source: run ``agg_run_id``'s tier ``tier`` rows on ``c_idx``,
    restricted to ``group_ids`` (None = every group, as a root is read)."""
    row_filter = And(
        EqualTo("agg_run_id", int(agg_run_id)),  # type: ignore[misc, call-arg]
        And(
            EqualTo(  # type: ignore[misc, call-arg]
                "chromosome_index", int(c_idx)
            ),
            EqualTo("tier", int(tier)),  # type: ignore[misc, call-arg]
        ),
    )
    if group_ids is not None:
        row_filter = And(
            row_filter,
            In(  # type: ignore[misc, call-arg]
                "group_id", [int(g) for g in group_ids]
            ),
        )
    return (spec.staging_table, row_filter)


def _fold_sources(
    spec: MetricSpec,
    c_idx: int,
    agg_run_id: int,
    delta_root: int,
    parent_run_id: int,
    parent_root: Optional[int],
) -> List[Source]:
    """The fold unit's sources on ``c_idx``: this run's delta root (tier
    ``delta_root``, skipped when 0: no delta batch covers the chromosome) and
    the parent's root (tier ``parent_root``, skipped when None: the parent
    had no unit there). Every group of each root tier is read."""
    sources: List[Source] = []
    if delta_root >= 1:
        sources.append(_staging_source(spec, c_idx, agg_run_id, delta_root))
    if parent_root is not None:
        sources.append(
            _staging_source(spec, c_idx, parent_run_id, parent_root)
        )
    return sources


def _reduce_group(task: Tuple) -> Tuple:
    """Reduce one (metric, chromosome, out_tier, group) unit of a tier-tree
    metric (site_info, allele_counts): SUM its sources into one staging group.

    ``task`` is ``(metric, c_idx, out_tier, group_id, sources,
    agg_run_id)`` where ``sources`` is a list of (Iceberg table, row filter):
    one for an ordinary tier unit, the delta root plus the parent's root for
    an incremental run's fold unit. Top-level + picklable for the spawn
    ProcessPoolExecutor (pyiceberg expressions pickle).

    Uploads its staging file to its deterministic S3 key and returns
    ``(status, metric, c_idx, out_tier, group_id, files)``: ``files`` is
    ((staging table, URI or None),), None when the result has no rows. It
    neither registers nor checkpoints: the main process does both for the
    whole barrier (_finish_barrier: Iceberg first, then one tier_completion
    transaction). A unit already checkpointed returns "skipped" before it
    computes; its catalog is used for reads only.
    """
    (
        metric_name,
        c_idx,
        out_tier,
        group_id,
        sources,
        agg_run_id,
    ) = task
    spec = SPECS[metric_name]
    if "catalog" not in _WORKER:
        _init_worker()
    catalog = _WORKER["catalog"]
    s3_client = _WORKER["s3"]
    key = (agg_run_id, metric_name, c_idx, out_tier, group_id)

    conn = psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]
    try:
        if _unit_checkpointed(conn, key):
            return ("skipped", metric_name, c_idx, out_tier, group_id, ())

        # Test-only: abruptly kill this worker (simulates an OOM/segfault) to
        # prove the executor fails the run instead of hanging. No-op in prod.
        _maybe_kill_worker(metric_name, c_idx, out_tier, group_id)

        # Stream each source one DATA FILE at a time (_iter_scan_record_batches)
        # and fold each record batch into a bounded Arrow buffer: peak is ~one
        # data file + the buffer + the distinct-key result (bounded by OUTPUT
        # keys), never the whole scan.
        #
        # Exactly-once: every tier-0 input row is summed once because (a) each
        # batch_* data file is registered exactly once (_get_registered_uris
        # filter + add_files check_duplicate_files=True in ingest), and (b)
        # ingest writes each (batch_id, key) exactly once. There is no
        # row-level dedup; test_batch_partials_no_duplicate_rows guards the
        # invariant. A fold's two sources are disjoint batch sets (the delta
        # is completed - parent.batch_ids).
        result = _sum_sources(
            spec,
            sources,
            functools.partial(_scan_table, catalog),
        )

        # Upload only (idempotent: deterministic key); the barrier registers
        # and checkpoints.
        s3_uri: Optional[str] = None
        if result.num_rows:
            arrow = _staging_table(
                spec, result, agg_run_id, out_tier, group_id
            )
            s3_uri = _upload_tier_parquet(
                arrow,
                s3_client,
                spec.s3_prefix,
                agg_run_id,
                c_idx,
                out_tier,
                group_id,
            )
        files: UnitFiles = ((spec.staging_table, s3_uri),)
        return ("done", metric_name, c_idx, out_tier, group_id, files)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _window_sources(
    c_idx: int,
    lo: int,
    hi: int,
    member_ids: Sequence[int],
    parent_run_id: Optional[int],
) -> List[Source]:
    """A ref_blocks window unit's sources over [lo, hi] on ``c_idx``, in fold
    order: the parent's dense root rows (when there is a parent), then the
    delta batches' batch_ref_blocks rows (when any covers the chromosome).

    The parent's rows are selected by POSITION range, never by window_id: the
    parent may have used another --ref-shard-size, so its windows need not
    line up with this run's (the dense root's per-file and per-row-group
    position statistics prune the read either way).
    """
    in_window = And(
        EqualTo(  # type: ignore[misc, call-arg]
            "chromosome_index", int(c_idx)
        ),
        And(
            GreaterThanOrEqual("position", lo),  # type: ignore[misc, call-arg]
            LessThanOrEqual("position", hi),  # type: ignore[misc, call-arg]
        ),
    )
    sources: List[Source] = []
    if parent_run_id is not None:
        sources.append(
            (
                DENSE_REF_TABLE,
                And(
                    EqualTo(  # type: ignore[misc, call-arg]
                        "agg_run_id", int(parent_run_id)
                    ),
                    in_window,
                ),
            )
        )
    if member_ids:
        sources.append(
            (
                SPECS["ref_blocks"].tier0_table,
                And(
                    In(  # type: ignore[misc, call-arg]
                        "batch_id", [int(m) for m in member_ids]
                    ),
                    in_window,
                ),
            )
        )
    return sources


def _fold_window(
    fold: _WindowFold,
    sources: Sequence[Source],
    scan: Callable[[str, Any, Tuple[str, ...]], Any],
) -> int:
    """Add every source's (position, dp, sample_counts) rows to ``fold``.
    Returns the number of rows read from the first source when it is the
    parent's dense root (0 otherwise), for the log."""
    fields = ("position", "dp", "sample_counts")
    parent_rows = 0
    for i, (table_name, row_filter) in enumerate(sources):
        for batch in scan(table_name, row_filter, fields):
            fold.add(batch)
            if i == 0 and table_name == DENSE_REF_TABLE:
                parent_rows += batch.num_rows
    return parent_rows


def _reduce_ref_window(task: Tuple) -> Tuple:
    """Reduce one ref_blocks (chromosome, position window) unit.

    Top-level + picklable for the spawn ProcessPoolExecutor. ``task`` is
    ``("ref_blocks", c_idx, window_id, (lo, hi), member_batch_ids,
    parent_run_id, agg_run_id)``; the unit's checkpoint key is
    (agg_run_id, "ref_blocks", c_idx, tier 1, group_id = window_id), so the
    MOSAIC_TEST_FAIL_UNIT / MOSAIC_TEST_KILL_WORKER hooks address it as tier
    1, group window_id.

    1. Fold into dense arrays (_WindowFold), one data file at a time: the
       parent run's dense root rows in [lo, hi] (an incremental run; None =
       a full run), then every member (delta) batch's batch_ref_blocks rows
       in [lo, hi]. Positions are disjoint across windows, so the windows'
       union is the chromosome's full sum; the parent's dense root carries
       every older batch's coverage, including at sites only the delta
       batches reveal.
    2. Upload the dense root (every position whose net sum is not (0, 0))
       for tier_ref_blocks_dense and the sparse root (those of them that are
       variant sites in PG site_info) for tier_ref_blocks at tier 1,
       group_id = window_id, each to a deterministic S3 key.
    3. Return ``(status, metric, c_idx, 1, window_id, n_dense, n_sparse,
       n_parent, files)``, ``files`` = ((tier_ref_blocks_dense, URI or
       None), (tier_ref_blocks, URI or None)). The main process registers
       every window's files (one add_files per table) and then commits the
       windows' checkpoints in one transaction (_run_ref_windows ->
       _finish_barrier). A window with no live position uploads neither
       file and is still checkpointed.
    """
    (
        metric_name,
        c_idx,
        window_id,
        window,
        member_ids,
        parent_run_id,
        agg_run_id,
    ) = task
    spec = SPECS[metric_name]
    lo, hi = int(window[0]), int(window[1])
    out_tier = _REF_ROOT_TIER
    if "catalog" not in _WORKER:
        _init_worker()
    catalog = _WORKER["catalog"]
    s3_client = _WORKER["s3"]
    key = (agg_run_id, metric_name, c_idx, out_tier, window_id)

    conn = psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]
    try:
        if _unit_checkpointed(conn, key):
            return (
                "skipped",
                metric_name,
                c_idx,
                out_tier,
                window_id,
                0,
                0,
                0,
                (),
            )

        # Test-only: abruptly kill this worker. No-op in production.
        _maybe_kill_worker(metric_name, c_idx, out_tier, window_id)

        # Exactly-once as in _reduce_group: each batch_ref_blocks data file is
        # registered once and holds each (batch_id, position) once, and the
        # parent's dense root holds each position once (one file per parent
        # window, deterministic key). np.bincount would sum a duplicate
        # anyway, so a duplicate would double-count, not be lost;
        # test_batch_partials_no_duplicate_rows guards the invariant.
        fold = _WindowFold(lo, hi)
        n_parent = _fold_window(
            fold,
            _window_sources(c_idx, lo, hi, member_ids, parent_run_id),
            functools.partial(_scan_table, catalog),
        )

        # Upload only (idempotent: deterministic keys); the barrier
        # registers and checkpoints.
        n_dense = n_sparse = 0
        dense_uri: Optional[str] = None
        sparse_uri: Optional[str] = None
        if fold.n_live:
            n_dense = fold.n_live
            dense_uri = _upload_parquet(
                lambda path: _write_dense_root(
                    path, fold, agg_run_id, c_idx, window_id
                ),
                s3_client,
                _tier_s3_key(
                    DENSE_REF_TABLE, agg_run_id, c_idx, out_tier, window_id
                ),
            )

            positions, dp, sc = fold.at(
                _ref_site_positions(conn, c_idx, (lo, hi))
            )
            n_sparse = int(positions.size)
            if n_sparse:
                arrow = _staging_table(
                    spec,
                    _ref_sparse_result(c_idx, positions, dp, sc),
                    agg_run_id,
                    out_tier,
                    window_id,
                )
                sparse_uri = _upload_tier_parquet(
                    arrow,
                    s3_client,
                    spec.s3_prefix,
                    agg_run_id,
                    c_idx,
                    out_tier,
                    window_id,
                )

        files: UnitFiles = (
            (DENSE_REF_TABLE, dense_uri),
            (spec.staging_table, sparse_uri),
        )
        return (
            "done",
            metric_name,
            c_idx,
            out_tier,
            window_id,
            n_dense,
            n_sparse,
            n_parent,
            files,
        )
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ── tier planning + orchestration ────────────────────────────────────────────


def _plan_tiers(member_ids: List[int], fan_in: int) -> List[List[List[int]]]:
    """tiers[k] = list of groups for the reduce reading in_tier=k, writing k+1.

    Tier-0 groups partition the batch_ids; each higher tier groups the previous
    tier's group_ids (0..n-1). Stops when a tier has a single group (the root).
    Degenerates to one tier / one group when len(member_ids) <= fan_in.
    """
    # fan_in must be >= 2: with fan_in == 1 each group holds one input, so the
    # group count never shrinks below the input count and the tree never
    # converges (infinite loop). Clamp defensively; main() also validates.
    fan_in = max(2, int(fan_in))
    tiers: List[List[List[int]]] = []
    current = list(member_ids)
    while True:
        groups = [
            current[i : i + fan_in] for i in range(0, len(current), fan_in)
        ]
        tiers.append(groups)
        if len(groups) <= 1:
            break
        current = list(range(len(groups)))
    return tiers


def _batch_covers(region: Optional[str], chromosome: str) -> bool:
    """True iff a batch ingested with ``--region region`` can hold partials on
    ``chromosome``: NULL is a whole-genome batch; otherwise ingest restricts
    sharding to the region's chromosome part (``region.split(":", 1)[0]``)."""
    return region is None or region.split(":", 1)[0] == chromosome


def _covering_batches(
    batches: List[Tuple[int, Optional[str]]], contigs: Dict[int, str]
) -> Dict[int, List[int]]:
    """chromosome index -> the completed batch_ids whose ingest region covers
    it (``batches`` is (batch_id, region) in batch_id order)."""
    return {
        c_idx: [b for b, region in batches if _batch_covers(region, chrom)]
        for c_idx, chrom in contigs.items()
    }


def _plan_tiers_by_chrom(
    batches: List[Tuple[int, Optional[str]]],
    contigs: Dict[int, str],
    fan_in: int,
) -> Dict[int, List[List[List[int]]]]:
    """Per-chromosome tier plans: ``plan[c] = _plan_tiers(batches covering c)``.

    The per-chromosome Nextflow fan-out makes every ingest batch cover one
    chromosome, so a global plan over all batch_ids added tiers no chromosome
    needed (425 batches at fan_in 100 -> 2 tiers, though each chromosome had
    17). A chromosome no batch covers gets ``[]``: no reduce units and nothing
    to read at finalize. ``batches`` is (batch_id, region) in batch_id order.
    """
    return {
        c_idx: _plan_tiers(members, fan_in) if members else []
        for c_idx, members in _covering_batches(batches, contigs).items()
    }


def _completed_batch_regions(conn: Any) -> List[Tuple[int, Optional[str]]]:
    """(batch_id, region) of every completed ingest batch, in batch_id order."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id, region FROM ingestion_progress "
            "WHERE status='complete' ORDER BY batch_id"
        )
        return [(b, r) for b, r in cur.fetchall()]


def _removal_batch_ids(conn: Any) -> List[int]:
    """The completed `mosaic remove` batches (kind 'removal'): their
    partials are the removed samples' contribution, negated, and are summed
    like any other batch's (only the banner tells them apart)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id FROM ingestion_progress "
            "WHERE status='complete' AND kind='removal' ORDER BY batch_id"
        )
        return [int(r[0]) for r in cur.fetchall()]


def _uploaded_batch_count(conn: Any) -> int:
    """Ingest batches `mosaic ingest --defer-register` left ``uploaded``:
    their data files are in S3 but not registered with Iceberg, so this run
    does not see them (it reads ``complete`` batches only)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM ingestion_progress WHERE status='uploaded'"
        )
        return int(cur.fetchone()[0])


# The prune sentinel: tier_completion (agg_run_id, "prune", 0, -1, 0),
# committed with the staging_pruned marks of _prune_staging's first step. The
# metric name cannot collide with a reduce or finalize unit (those use the
# SPECS names) and every tier_completion reader filters by metric.
_PRUNE_METRIC = "prune"
_PRUNE_MARKER = (0, -1, 0)


@dataclass(frozen=True)
class RunRow:
    """One aggregation_progress row, as the run resolver needs it."""

    agg_run_id: int
    batch_set_hash: str
    batch_ids: Tuple[int, ...]
    fan_in: int
    ref_shard_size: Optional[int]
    status: str
    parent_run_id: Optional[int]
    staging_pruned: bool
    # The run committed its prune sentinel (tier_completion metric "prune"):
    # every unit and finalize of it is checkpointed and it has marked every
    # other run, its own parent included, staging_pruned.
    prune_started: bool = False


@dataclass(frozen=True)
class RunDecision:
    """What `mosaic aggregate` does for the current completed-batch set.

    ``state`` is ``"current"`` (the latest run for the set is complete:
    nothing to do; ``run`` is that run), ``"resume"`` (``run`` is the
    in-progress / failed run to continue) or ``"fresh"`` (insert a new run;
    ``run`` is None). ``parent`` is the complete run the new or resumed run
    folds onto (None = a full run over every completed batch).
    """

    state: str
    run: Optional[RunRow]
    parent: Optional[RunRow]


def _load_runs(conn: Any) -> List[RunRow]:
    """Every aggregation_progress row, in agg_run_id order (one per run, so
    a small table)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.agg_run_id, p.batch_set_hash, p.batch_ids, p.fan_in, "
            "p.ref_shard_size, p.status, p.parent_run_id, p.staging_pruned, "
            "EXISTS (SELECT 1 FROM tier_completion t WHERE "
            "t.agg_run_id = p.agg_run_id AND t.metric = %s) "
            "FROM aggregation_progress p ORDER BY p.agg_run_id",
            (_PRUNE_METRIC,),
        )
        return [
            RunRow(
                agg_run_id=int(r[0]),
                batch_set_hash=r[1],
                batch_ids=tuple(int(b) for b in (r[2] or ())),
                fan_in=int(r[3]),
                ref_shard_size=r[4],
                status=r[5],
                parent_run_id=r[6],
                staging_pruned=bool(r[7]),
                prune_started=bool(r[8]),
            )
            for r in cur.fetchall()
        ]


def _choose_parent(
    runs: Sequence[RunRow], completed: Sequence[int]
) -> Optional[RunRow]:
    """The run an incremental run folds onto: the latest (highest
    agg_run_id) run that is ``complete``, whose staging is not pruned (its
    roots still exist), and whose batch_ids are a PROPER subset of the
    completed batches (so the delta is non-empty; a complete run for the
    whole set would have made the set up to date). None = a full run.

    Normally at most one run qualifies: every completion prunes all other
    runs' staging.
    """
    done = set(completed)
    best: Optional[RunRow] = None
    for r in runs:
        if r.status != "complete" or r.staging_pruned:
            continue
        if not set(r.batch_ids) < done:
            continue
        if best is None or r.agg_run_id > best.agg_run_id:
            best = r
    return best


def _decide_run(
    runs: Sequence[RunRow],
    batch_set_hash: str,
    completed: Sequence[int],
    fan_in: int,
    ref_shard_size: int,
    rebuild: bool,
) -> RunDecision:
    """Decide between up to date, resume and a new (full or incremental) run.

    --rebuild: always a new full run (no parent). Otherwise look at the
    latest run for this batch set: ``complete`` (any fan_in) -> up to date.
    Else pick the parent (_choose_parent). An in_progress / failed latest
    run is resumed iff its fan_in and ref_shard_size equal this
    invocation's (they fix the unit keys: tier-tree groups, window ids), its
    staging was not pruned (its checkpoints must still have their staging
    rows behind them), and either its parent_run_id equals the chosen
    parent's (the parent fixes what the fold and window units read) or it
    had begun its prune (``prune_started``) and no candidate parent exists.
    A prune-started run marked its own parent staging_pruned, so the parent
    no longer qualifies; resuming it is safe, because every unit and
    finalize of the run is checkpointed and it has nothing left to read
    from the parent: the resume skips every unit, redoes the deletes
    (idempotent) and marks the run complete. Without that record, a crash
    in the last seconds of a run would throw it away for a full rebuild.
    But the run marked EVERY run then existing staging_pruned, so an
    unpruned complete candidate can only be a run that completed after the
    crash, and whose finalizes swapped its partitions in after this run's.
    Resuming would skip this run's checkpointed finalizes, leave PostgreSQL
    holding the later run's partitions and still mark this run complete
    for its own batch set: a wrong cohort_generation. So a candidate wins:
    a new run onto it. A resume carries the run's STORED parent (pruned or not),
    which fixes its delta and so its plan and prune filter; if that row is
    missing, the run is not resumed. Otherwise a new run with the chosen
    parent.
    """
    if rebuild:
        return RunDecision("fresh", None, None)
    mine = [r for r in runs if r.batch_set_hash == batch_set_hash]
    latest = max(mine, key=lambda r: r.agg_run_id) if mine else None
    if latest is not None and latest.status == "complete":
        return RunDecision("current", latest, None)
    parent = _choose_parent(runs, completed)
    parent_id = parent.agg_run_id if parent is not None else None
    if (
        latest is not None
        and latest.status in ("in_progress", "failed")
        and latest.fan_in == fan_in
        and latest.ref_shard_size == ref_shard_size
        and not latest.staging_pruned
        and (
            latest.parent_run_id == parent_id
            or (latest.prune_started and parent_id is None)
        )
    ):
        if latest.parent_run_id is None:
            return RunDecision("resume", latest, None)
        stored = next(
            (r for r in runs if r.agg_run_id == latest.parent_run_id), None
        )
        if stored is not None:
            return RunDecision("resume", latest, stored)
    return RunDecision("fresh", None, parent)


def _resolve_run(
    conn: Any,
    batch_set_hash: str,
    completed: List[int],
    fan_in: int,
    ref_shard_size: int,
    rebuild: bool,
) -> Tuple[int, str, Optional[int], List[int]]:
    """Return (agg_run_id, state, parent_run_id, delta_batch_ids) for the
    completed-batch set (see _decide_run).

    ``state`` is ``"current"`` (up to date: nothing to do), ``"resume"``
    (the run is set back to in_progress) or ``"fresh"`` (a new run is
    inserted). A run with a parent reduces only ``delta_batch_ids =
    completed - parent.batch_ids`` and folds them onto the parent's roots; a
    full run has parent None and delta = every completed batch.
    """
    decision = _decide_run(
        _load_runs(conn),
        batch_set_hash,
        completed,
        fan_in,
        ref_shard_size,
        rebuild,
    )
    parent = decision.parent
    parent_id = parent.agg_run_id if parent is not None else None
    if parent is None:
        delta = list(completed)
    else:
        prior = set(parent.batch_ids)
        delta = [b for b in completed if b not in prior]
    with conn.cursor() as cur:
        if decision.state == "current":
            assert decision.run is not None
            return decision.run.agg_run_id, "current", None, []
        if decision.state == "resume":
            assert decision.run is not None
            cur.execute(
                "UPDATE aggregation_progress SET status='in_progress', "
                "updated_at=now() WHERE agg_run_id=%s",
                (decision.run.agg_run_id,),
            )
            return decision.run.agg_run_id, "resume", parent_id, delta
        cur.execute(
            "INSERT INTO aggregation_progress (batch_set_hash, batch_ids, "
            "fan_in, ref_shard_size, status, parent_run_id, delta_batch_ids) "
            "VALUES (%s,%s,%s,%s,'in_progress',%s,%s) RETURNING agg_run_id",
            (
                batch_set_hash,
                completed,
                fan_in,
                ref_shard_size,
                parent_id,
                delta,
            ),
        )
        return cur.fetchone()[0], "fresh", parent_id, delta


def _parent_root_tiers(
    conn: Any, parent_run_id: Optional[int]
) -> Dict[Tuple[str, int], int]:
    """(metric, chromosome) -> the parent run's root tier for the tier-tree
    metrics: max(tier) over its checkpointed units with tier >= 1 (a full
    parent's tree root, or an incremental parent's fold tier). A
    (metric, chromosome) with no unit is absent: the parent had no rows
    there, so the fold skips that source."""
    if parent_run_id is None:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT metric, chromosome_index, max(tier) FROM tier_completion "
            "WHERE agg_run_id=%s AND tier >= 1 AND metric = ANY(%s) "
            "GROUP BY metric, chromosome_index",
            (int(parent_run_id), list(PRE_METRICS)),
        )
        return {(m, int(c)): int(t) for m, c, t in cur.fetchall()}


def _drain_units(
    executor: Any, fn: Callable, tasks: List[Tuple[Any, ...]], agg_run_id: int
) -> Tuple[List[Tuple[Tuple[Any, ...], UnitFiles]], int, List[Tuple]]:
    """Run a barrier's reduce units and drain the map. Returns (done, the
    number skipped, every result): ``done`` is (checkpoint key, files) per
    unit that ran in this attempt. A result is (status, metric, c_idx, tier,
    group, ..., files): files last, the key from its first five fields.
    executor.map re-raises the first worker exception (incl.
    BrokenProcessPool if a worker dies) when its result is consumed."""
    done: List[Tuple[Tuple[Any, ...], UnitFiles]] = []
    skipped = 0
    results = list(executor.map(fn, tasks))
    for r in results:
        if r[0] == "skipped":
            skipped += 1
        else:
            done.append(((agg_run_id, r[1], r[2], r[3], r[4]), r[-1]))
    return done, skipped, results


def _registered_note(counts: Tuple[int, int]) -> str:
    """Log text for a barrier's registration (files, commits)."""
    return f"registered {counts[0]} file(s) in {counts[1]} commit(s)"


def _run_metric_tiers(
    metric: str,
    agg_run_id: int,
    plan: Dict[int, List[List[List[int]]]],
    executor: Any,
    fresh: bool,
    conn: Any,
    catalog: Any,
) -> None:
    """Run every tier of one tier-tree metric, with a barrier between tiers.

    ``plan`` is the per-chromosome tier plan (over this run's delta batches).
    The barrier stays global: tier k of every chromosome runs, then k+1; a
    chromosome with fewer tiers simply contributes no units at the higher k.

    At each barrier the main process registers the tier's uploaded files
    (one add_files on the staging table) and then commits the tier's
    checkpoints in one transaction on ``conn`` (_finish_barrier). ``fresh``
    (a brand-new agg_run_id, no prior attempt) skips the registered-URI scan
    -- see register_barrier.
    """
    spec = SPECS[metric]
    n_tiers = max((len(t) for t in plan.values()), default=0)
    for k in range(n_tiers):
        in_tier, out_tier = k, k + 1
        t0 = time.time()
        tasks: List[Tuple[Any, ...]] = []
        n_chroms = 0
        for c_idx, tiers in plan.items():
            if k >= len(tiers):
                continue
            n_chroms += 1
            for gid, members in enumerate(tiers[k]):
                source = (
                    _batch_source(spec, c_idx, members)
                    if in_tier == 0
                    else _staging_source(
                        spec, c_idx, agg_run_id, in_tier, members
                    )
                )
                tasks.append(
                    (metric, c_idx, out_tier, gid, [source], agg_run_id)
                )
        # The map is fully drained here, preserving the per-tier barrier.
        done, skipped, _ = _drain_units(
            executor, _reduce_group, tasks, agg_run_id
        )
        counts = _finish_barrier(conn, catalog, done, fresh)
        print(
            f"  {metric} tier {in_tier}->{out_tier}: "
            f"{len(tasks)} unit(s) over {n_chroms} chrom "
            f"({len(done)} done, {skipped} skipped, "
            f"{time.time() - t0:.1f}s; {_registered_note(counts)})",
            file=sys.stderr,
        )


def _run_fold(
    metric: str,
    agg_run_id: int,
    parent_run_id: int,
    plan: Dict[int, List[List[List[int]]]],
    fold_chroms: List[int],
    parent_roots: Dict[Tuple[str, int], int],
    executor: Any,
    fresh: bool,
    conn: Any,
    catalog: Any,
) -> None:
    """Run the fold tier of one tier-tree metric in an incremental run: one
    unit per chromosome in ``fold_chroms`` at tier T + 1, group 0 (T =
    len(plan[c]), the delta root; 0 when no delta batch covers c), summing
    the delta root and the parent's root. It runs for every chromosome any
    completed batch covers, including those with an empty delta, so this
    run's root tier exists wherever the finalize reads (copying an unchanged
    chromosome's root is the simple rule). One barrier: the units' files
    are registered and their checkpoints committed together
    (_finish_barrier)."""
    spec = SPECS[metric]
    t0 = time.time()
    tasks: List[Tuple[Any, ...]] = []
    for c_idx in fold_chroms:
        delta_root = len(plan[c_idx])
        sources = _fold_sources(
            spec,
            c_idx,
            agg_run_id,
            delta_root,
            parent_run_id,
            parent_roots.get((metric, c_idx)),
        )
        tasks.append((metric, c_idx, delta_root + 1, 0, sources, agg_run_id))
    done, skipped, _ = _drain_units(executor, _reduce_group, tasks, agg_run_id)
    counts = _finish_barrier(conn, catalog, done, fresh)
    print(
        f"  {metric} fold onto parent {parent_run_id}: {len(tasks)} unit(s) "
        f"({len(done)} done, {skipped} skipped, {time.time() - t0:.1f}s; "
        f"{_registered_note(counts)})",
        file=sys.stderr,
    )


def _ref_window_tasks(
    agg_run_id: int,
    ref_chroms: List[int],
    delta_members: Dict[int, List[int]],
    chrom_windows: Dict[int, List[Tuple[int, int]]],
    parent_run_id: Optional[int],
) -> List[Tuple[Any, ...]]:
    """One ref_blocks window unit per (chromosome in ``ref_chroms``, window).
    A window unit reads every delta batch covering its chromosome (ref_blocks
    has no tier tree, so --fan-in does not apply to it) plus, in an
    incremental run, the parent's dense root rows in its range -- so a
    chromosome with an empty delta still gets units that carry the parent's
    root forward."""
    return [
        (
            "ref_blocks",
            c_idx,
            wi,
            win,
            delta_members[c_idx],
            parent_run_id,
            agg_run_id,
        )
        for c_idx in ref_chroms
        for wi, win in enumerate(chrom_windows[c_idx])
    ]


def _run_ref_windows(
    tasks: List[Tuple[Any, ...]],
    executor: Any,
    agg_run_id: int,
    fresh: bool,
    conn: Any,
    catalog: Any,
) -> None:
    """Run the ref_blocks window units (a barrier: the map is drained before
    the ref_blocks finalize reads their sparse roots). The main process then
    registers every window's dense and sparse root files (one add_files on
    tier_ref_blocks_dense, one on tier_ref_blocks) and commits the windows'
    checkpoints in one transaction (_finish_barrier)."""
    t0 = time.time()
    done, skipped, results = _drain_units(
        executor, _reduce_ref_window, tasks, agg_run_id
    )
    dense = sum(r[5] for r in results)
    sparse = sum(r[6] for r in results)
    parent = sum(r[7] for r in results)
    counts = _finish_barrier(conn, catalog, done, fresh)
    n_chroms = len({t[1] for t in tasks})
    parent_note = (
        f"parent dense root {parent} rows read; "
        if any(t[5] is not None for t in tasks)
        else ""
    )
    print(
        f"  ref_blocks windows: {len(tasks)} unit(s) over {n_chroms} chrom "
        f"({len(done)} done, {skipped} skipped; {parent_note}dense root "
        f"{dense} rows, sparse root {sparse} rows; {time.time() - t0:.1f}s; "
        f"{_registered_note(counts)})",
        file=sys.stderr,
    )


# Max final rows a finalize unit holds at once. Caps peak memory to ~one Arrow
# chunk + its COPY text, regardless of chromosome size (genome-wide the result
# is tens of millions of rows -- materializing it all was the aggregate OOM).
# Smaller = lower peak but more agg_* mirror files (one per chunk; still one
# mirror commit per metric); larger = fewer files but higher peak. Override
# via MOSAIC_FINALIZE_CHUNK_ROWS (tests set it small to exercise the
# multi-chunk path on small fixtures).
_FINALIZE_CHUNK_ROWS = max(
    1, int(os.environ.get("MOSAIC_FINALIZE_CHUNK_ROWS", "1000000"))
)

# Per-statement timeout for the finalize transactions (the COPYs, the primary
# key build, the swap): a pathological statement aborts and the run is marked
# failed (resumable) instead of hanging for days. work_mem and
# maintenance_work_mem (the primary-key sort) are set to
# MOSAIC_FINALIZE_WORK_MEM. Both env-overridable.
_FINALIZE_STATEMENT_TIMEOUT = os.environ.get(
    "MOSAIC_FINALIZE_STATEMENT_TIMEOUT", "30min"
)
_FINALIZE_WORK_MEM = os.environ.get("MOSAIC_FINALIZE_WORK_MEM", "256MB")

# How long the swap transaction waits to acquire its locks (the parent's
# ACCESS EXCLUSIVE for DETACH; a long reader of the aggregate table holds it
# off). Fails the unit (resumable) rather than waiting forever.
_SWAP_LOCK_TIMEOUT = "5min"


def _finalize_marker(c_idx: int) -> Tuple[int, int, int]:
    """tier_completion key (chromosome_index, tier, group_id) of a (metric,
    chromosome) finalize checkpoint. No reduce unit uses a negative tier."""
    return (int(c_idx), -1, 0)


def _final_rows(spec: MetricSpec, batch: Any) -> Any:
    """Root-tier staging rows -> a null-free chunk in the agg_* schema (keys
    and values in the PG column order)."""
    return _unflatten(
        spec, _fill_values(spec, _flatten(spec, batch)), sort=False
    )


def _iceberg_commit_with_retry(
    table: Any, write: Callable[[], None], max_retries: int = 10
) -> None:
    """Run one Iceberg write (``write()``), retrying on a concurrent-commit
    conflict.

    Modeled on ingest._add_files_with_retry: the catalog rejects all but one
    of a set of concurrent commits (CommitFailedException: nothing of the
    rejected one was committed). Refresh the table metadata, back off and
    re-run the write, which then applies on top of the winner's snapshot.
    Since F2 the aggregate's own commits (the mirror replace, the prune and
    --rebuild deletes) are made by the main process alone, so this guards
    against another writer only.
    """
    for attempt in range(max_retries):
        try:
            write()
            return
        except _COMMIT_FAILED_EXC:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt + random.random()
            print(
                f"  Iceberg commit conflict on {table.name()[-1]} (attempt "
                f"{attempt + 1}/{max_retries}), retrying in {wait:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
            table.refresh()


def _mirror_prefix(agg_table: str, agg_run_id: int, c_idx: int) -> str:
    """S3 key prefix of a finalize unit's agg_* mirror chunk files. Under
    ``<agg table>/`` like the staging keys under ``<staging table>/``, so
    `mosaic clean`, `mosaic maintain --orphans` and
    scripts/measure_s3_usage.py (prefixes from init_db.ICEBERG_TABLES)
    cover them; the trailing slash keeps c1 from matching c10."""
    return f"{agg_table}/{int(agg_run_id)}/c{int(c_idx)}/"


def _mirror_s3_key(
    agg_table: str, agg_run_id: int, c_idx: int, chunk: int
) -> str:
    """Deterministic S3 key of one finalize chunk's mirror file:
    ``<agg table>/<agg_run_id>/c<c_idx>/<chunk index>.parquet``. A resumed
    attempt of the run reads the same root files in the same order, so it
    overwrites the same keys with the same bytes."""
    return (
        f"{_mirror_prefix(agg_table, agg_run_id, c_idx)}{int(chunk)}.parquet"
    )


def _list_mirror_uris(
    s3_client: Any, agg_table: str, agg_run_id: int, c_idx: int
) -> List[str]:
    """The mirror chunk files an earlier attempt of this run uploaded for
    (agg_table, c_idx), as s3:// URIs sorted by chunk index: what a skipped
    finalize unit (its swap committed) returns, so the barrier's mirror
    commit holds its chromosome even when the crash fell between the swap
    and that commit."""
    prefix = _mirror_prefix(agg_table, agg_run_id, c_idx)
    found: List[Tuple[int, str]] = []
    paginator = s3_client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            name = obj["Key"][len(prefix) :]
            stem = name[: -len(".parquet")]
            if name.endswith(".parquet") and stem.isdigit():
                found.append((int(stem), obj["Key"]))
    return [f"s3://{S3_BUCKET}/{key}" for _, key in sorted(found)]


class _MirrorWriter:
    """Uploads a finalize unit's chunks of the agg_* mirror, one Parquet
    file per chunk at _mirror_s3_key, and collects their URIs. Registration
    is the finalize barrier's (_replace_mirror): agg_* is a write-only
    mirror (nothing reads its rows), so it lagging the PG swap by one
    barrier changes nothing a reader sees."""

    def __init__(
        self, s3_client: Any, agg_table: str, agg_run_id: int, c_idx: int
    ):
        self.s3_client = s3_client
        self.agg_table = agg_table
        self.agg_run_id = int(agg_run_id)
        self.c_idx = int(c_idx)
        self.uris: List[str] = []

    def write(self, chunk: Any) -> None:
        key = _mirror_s3_key(
            self.agg_table, self.agg_run_id, self.c_idx, len(self.uris)
        )
        self.uris.append(
            _upload_parquet(
                lambda path: pq.write_table(chunk, path), self.s3_client, key
            )
        )


def _replace_mirror(catalog: Any, agg_table: str, uris: List[str]) -> int:
    """Make the agg_* mirror hold exactly this run's finalize chunk files:
    one pyiceberg transaction (one catalog commit) that deletes every row
    (AlwaysTrue: the mirror is unpartitioned and every file matches, so
    whole files are dropped and nothing is rewritten) and adds ``uris``.
    Every chromosome has a finalize unit in every run (_run_finalize), so
    this run's files are the whole mirror. After the delete-all no URI is
    live, a re-registration by an earlier attempt included, so the add needs
    no scan and no duplicate check. Returns the snapshots the commit added
    (2, or 1 when the mirror was empty)."""
    table = catalog.load_table(f"{ICEBERG_NAMESPACE}.{agg_table}")
    before = len(table.metadata.snapshots)

    def write() -> None:
        with warnings.catch_warnings():
            # The first run's mirror is empty: the delete matches nothing.
            warnings.filterwarnings(
                "ignore", message="Delete operation did not match any records"
            )
            with table.transaction() as txn:
                if table.current_snapshot() is not None:
                    txn.delete(AlwaysTrue())
                if uris:
                    txn.add_files(
                        file_paths=list(uris), check_duplicate_files=False
                    )

    _iceberg_commit_with_retry(table, write)
    table.refresh()
    return len(table.metadata.snapshots) - before


def _flush_finalize_chunk(
    cur: Any,
    spec: MetricSpec,
    target: str,
    chunk: Any,
    mirror: _MirrorWriter,
) -> int:
    """COPY one Arrow chunk's PostgreSQL rows (``spec.pg_rows``: dead rows
    dropped, gtc's integer part) into the generation table ``target``
    (frozen: it was truncated in this transaction) and upload the whole
    chunk as a mirror file. Returns the rows copied."""
    if chunk.num_rows == 0:
        return 0
    pg_chunk = spec.pg_rows(chunk)
    if pg_chunk.num_rows:
        spec.pg_copy(cur, target, pg_chunk, freeze=True)
    mirror.write(chunk)
    return int(pg_chunk.num_rows)


def _load_generation(
    cur: Any,
    catalog: Any,
    spec: MetricSpec,
    target: str,
    c_idx: int,
    agg_run_id: int,
    root_tier: int,
    mirror: _MirrorWriter,
) -> Tuple[int, int]:
    """Stream chromosome ``c_idx``'s root-tier staging rows into ``target``
    and the mirror's chunk files, in chunks of <= _FINALIZE_CHUNK_ROWS.
    Returns (root rows, rows copied to ``target``): the difference is the
    dead rows ``spec.pg_rows`` dropped (a site or allele only removed
    samples' records had).

    The root is tier ``root_tier`` (0 = no batch covers the chromosome:
    nothing to read, the partition is attached empty), read one data file at
    a time (_iter_scan_record_batches). Read ALL groups at the root tier (no
    group_id filter): a tier-tree root has one group (group_id=0), but the
    ref_blocks window units write one sparse-root group per window at tier 1.
    Those windows are disjoint position ranges, so reading all of them is the
    full result. A chromosome with no root row uploads no file: the
    barrier's delete-all already leaves it no row in the mirror (and
    pyiceberg's add_files warns on a zero-row file's missing statistics).
    """
    total = copied = 0
    pending: List[Any] = []
    pending_rows = 0
    if root_tier > 0:
        staging = catalog.load_table(
            f"{ICEBERG_NAMESPACE}.{spec.staging_table}"
        )
        root_filter = And(
            EqualTo("agg_run_id", int(agg_run_id)),
            And(
                # pyiceberg's pydantic stubs reject positional args.
                EqualTo(  # type: ignore[misc, call-arg]
                    "chromosome_index", int(c_idx)
                ),
                EqualTo("tier", int(root_tier)),
            ),
        )
        fields = spec.key_fields + spec.value_fields
        for batch in _iter_scan_record_batches(staging, root_filter, fields):
            rows = _final_rows(spec, batch)
            total += rows.num_rows
            pending.append(rows)
            pending_rows += rows.num_rows
            # Flush in exact _FINALIZE_CHUNK_ROWS slices so a single large
            # scan batch cannot exceed the cap (hard bound on rows held).
            while pending_rows >= _FINALIZE_CHUNK_ROWS:
                buffered = pa.concat_tables(pending)
                copied += _flush_finalize_chunk(
                    cur,
                    spec,
                    target,
                    buffered.slice(0, _FINALIZE_CHUNK_ROWS),
                    mirror,
                )
                rest = buffered.slice(_FINALIZE_CHUNK_ROWS)
                pending = [rest] if rest.num_rows else []
                pending_rows = rest.num_rows
    if pending:
        copied += _flush_finalize_chunk(
            cur, spec, target, pa.concat_tables(pending), mirror
        )
    return total, copied


def _finalize_done(
    conn: Any, agg_run_id: int, metric: str, c_idx: int
) -> bool:
    """True iff this run already swapped in ``metric``'s partition for
    chromosome ``c_idx`` (its finalize checkpoint row exists)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT 1 FROM tier_completion WHERE agg_run_id=%s AND metric=%s "
            "AND chromosome_index=%s AND tier=%s AND group_id=%s",
            (agg_run_id, metric) + _finalize_marker(c_idx),
        )
        done = cur.fetchone() is not None
    conn.commit()
    return done


def _attached_partition(cur: Any, table: str, c_idx: int) -> Optional[str]:
    """Name of the partition of ``table`` currently attached for chromosome
    ``c_idx`` (None if there is none), looked up through pg_inherits and the
    partition bound -- never by name, so a generation from any earlier run (or
    init's r0) is found."""
    cur.execute(
        """
        SELECT c.relname
        FROM pg_inherits i
        JOIN pg_class c ON c.oid = i.inhrelid
        WHERE i.inhparent = %s::regclass
          AND pg_get_expr(c.relpartbound, c.oid) = %s
        """,
        (table, f"FOR VALUES IN ({int(c_idx)})"),
    )
    rows = cur.fetchall()
    if len(rows) > 1:
        raise RuntimeError(
            f"{table} has {len(rows)} partitions for chromosome_index "
            f"{c_idx}: {[r[0] for r in rows]}"
        )
    return rows[0][0] if rows else None


def _finalize_chrom(task: Tuple) -> Tuple:
    """Finalize one (metric, chromosome): build its next generation table and
    swap it in as the chromosome's partition. Top-level + picklable for the
    spawn ProcessPoolExecutor; runs on its own PG connection and catalog.

    Returns ``(status, metric, c_idx, rows, mirror URIs)``; the mirror
    files are registered by the finalize barrier (_run_finalize ->
    _replace_mirror), not here.

    1. Skip if the unit's checkpoint row exists (the swap already committed):
       return the mirror files the earlier attempt uploaded, listed from the
       unit's deterministic prefix (_list_mirror_uris), so a crash between
       the swap and the barrier's mirror commit loses nothing.
    2. Transaction: drop a leftover generation table from a crashed attempt
       and create an empty one shaped like the parent (LIKE ... INCLUDING
       DEFAULTS INCLUDING CONSTRAINTS: the parent's columns, NOT NULLs, and
       CHECKs, which ATTACH requires by name) plus a CHECK pinning it to this
       chromosome, with autovacuum off: a generation is write-once (loaded,
       then only ever dropped whole), so vacuum has nothing to do on it, and
       an autovacuum of the attached partition would hold off the next
       swap's DETACH. Committed at once: LIKE holds a lock on the parent that
       would otherwise block every other unit's swap for this whole build.
    3. Transaction: TRUNCATE the (empty) table, then COPY the root tier in
       chunks WITH (FREEZE) -- the TRUNCATE in the same transaction is what
       FREEZE requires -- so the rows are loaded frozen and all-visible (no
       later vacuum work; index-only scans from the first read), each chunk
       also uploaded as an agg_* mirror file (_MirrorWriter, deterministic
       keys); then one primary-key btree build, the
       metric's pg_indexes (the alleles GiST on del_range), and ANALYZE.
       Committed, so a crash before the swap leaves the old partition
       attached and this table detached.
    4. The swap, one short transaction: DETACH the attached partition (found
       via pg_inherits), ATTACH the new table, drop its CHECK, DROP the old
       table, insert the checkpoint row, commit. The CHECK plus the NOT NULL
       partition key imply the partition bound, so ATTACH skips its
       validation scan, and the existing primary key (and any pg_indexes
       index) is adopted as the partition's index (no rebuild).
    """
    metric_name, c_idx, agg_run_id, root_tier = task
    spec = SPECS[metric_name]
    if "catalog" not in _WORKER:
        _init_worker()
    catalog = _WORKER["catalog"]
    s3_client = _WORKER["s3"]
    parent = spec.pg_table
    gen = generation_table_name(parent, c_idx, agg_run_id)

    conn = psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]
    try:
        if _finalize_done(conn, agg_run_id, metric_name, c_idx):
            uris = _list_mirror_uris(
                s3_client, spec.agg_table, agg_run_id, c_idx
            )
            return ("skipped", metric_name, c_idx, 0, uris)
        t0 = time.time()
        mirror = _MirrorWriter(s3_client, spec.agg_table, agg_run_id, c_idx)

        with conn.cursor() as cur:
            if _attached_partition(cur, parent, c_idx) == gen:
                # Unreachable through this module: the attach and the
                # checkpoint row commit together. Refuse rather than drop the
                # live partition.
                raise RuntimeError(
                    f"{gen} is attached to {parent} but has no finalize "
                    f"checkpoint for run {agg_run_id}"
                )
            cur.execute(f"DROP TABLE IF EXISTS {gen}")
            cur.execute(
                f"CREATE TABLE {gen} (LIKE {parent} "
                "INCLUDING DEFAULTS INCLUDING CONSTRAINTS) "
                "WITH (autovacuum_enabled = false)"
            )
            cur.execute(
                f"ALTER TABLE {gen} ADD CONSTRAINT {gen}_chrom "
                f"CHECK (chromosome_index = {int(c_idx)})"
            )
        conn.commit()

        with conn.cursor() as cur:
            # SET takes no bind params; the values are operator config.
            cur.execute(
                f"SET LOCAL statement_timeout = '{_FINALIZE_STATEMENT_TIMEOUT}'"
            )
            cur.execute(f"SET LOCAL work_mem = '{_FINALIZE_WORK_MEM}'")
            cur.execute(
                f"SET LOCAL maintenance_work_mem = '{_FINALIZE_WORK_MEM}'"
            )
            # Lets every COPY below load frozen (see _flush_finalize_chunk).
            cur.execute(f"TRUNCATE {gen}")
            root_rows, rows = _load_generation(
                cur, catalog, spec, gen, c_idx, agg_run_id, root_tier, mirror
            )
            cur.execute(
                f"ALTER TABLE {gen} ADD PRIMARY KEY "
                f"({', '.join(spec.key_fields)})"
            )
            for index_def in spec.pg_indexes:
                cur.execute(f"CREATE INDEX ON {gen} {index_def}")
            cur.execute(f"ANALYZE {gen}")
        conn.commit()
        t_built = time.time()

        with conn.cursor() as cur:
            cur.execute(f"SET LOCAL lock_timeout = '{_SWAP_LOCK_TIMEOUT}'")
            cur.execute(
                f"SET LOCAL statement_timeout = '{_FINALIZE_STATEMENT_TIMEOUT}'"
            )
            # DETACH takes this lock anyway; taking it first makes the
            # partition lookup below race-free. ONLY: the parent, not every
            # partition (DETACH locks just the one it detaches).
            cur.execute(f"LOCK TABLE ONLY {parent} IN ACCESS EXCLUSIVE MODE")
            old = _attached_partition(cur, parent, c_idx)
            if old is not None:
                cur.execute(f"ALTER TABLE {parent} DETACH PARTITION {old}")
            cur.execute(
                f"ALTER TABLE {parent} ATTACH PARTITION {gen} "
                f"FOR VALUES IN ({int(c_idx)})"
            )
            cur.execute(f"ALTER TABLE {gen} DROP CONSTRAINT {gen}_chrom")
            if old is not None:
                cur.execute(f"DROP TABLE {old}")
            cur.execute(
                "INSERT INTO tier_completion (agg_run_id, metric, "
                "chromosome_index, tier, group_id) VALUES (%s,%s,%s,%s,%s) "
                "ON CONFLICT DO NOTHING",
                (agg_run_id, metric_name) + _finalize_marker(c_idx),
            )
        # Test-only: crash before / after the swap commit. No-op in production.
        _maybe_crash_finalize(metric_name, c_idx, "pre_commit")
        conn.commit()
        _maybe_crash_finalize(metric_name, c_idx, "post_commit")
        t_done = time.time()
        print(
            f"  {metric_name} finalize c{c_idx}: {rows} rows -> {gen} "
            f"({root_rows - rows} dead root row(s) dropped; "
            f"replaced {old}; build {t_built - t0:.1f}s, "
            f"swap {t_done - t_built:.2f}s)",
            file=sys.stderr,
        )
        return ("done", metric_name, c_idx, rows, mirror.uris)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _run_finalize(
    metric: str,
    agg_run_id: int,
    root_tiers: Dict[int, int],
    executor: Any,
    catalog: Any,
) -> None:
    """Finalize one metric: one _finalize_chrom unit per chromosome, run
    through the executor, with a barrier (the map is drained) before the next
    metric. ``root_tiers`` maps every chromosome to the tier its root is
    read from (0 = no batch covers it). Every chromosome gets a unit -- one
    that no batch covers attaches an empty partition -- so a stale generation
    never survives a run.

    After the barrier the main process replaces the metric's agg_* mirror
    with every unit's chunk files (skipped units included: they list theirs)
    in one commit (_replace_mirror). The mirror is write-only, so it lagging
    the PG swaps by this one barrier changes nothing a reader sees."""
    spec = SPECS[metric]
    t0 = time.time()
    tasks = [
        (metric, c_idx, agg_run_id, root_tier)
        for c_idx, root_tier in root_tiers.items()
    ]
    done = skipped = total = 0
    mirror_uris: List[str] = []
    for status, _, _, rows, uris in executor.map(_finalize_chrom, tasks):
        mirror_uris.extend(uris)
        if status == "skipped":
            skipped += 1
        else:
            done += 1
            total += rows
    t_mirror = time.time()
    n_snaps = _replace_mirror(catalog, spec.agg_table, mirror_uris)
    mirror_note = (
        f"{spec.agg_table} replaced by {len(mirror_uris)} file(s) "
        f"({n_snaps} snapshot(s), {time.time() - t_mirror:.1f}s)"
    )
    if skipped and not done:
        print(
            f"  {metric} finalize: skipped ({skipped} chromosome unit(s) "
            f"checkpointed by an earlier attempt of this run); {mirror_note}",
            file=sys.stderr,
        )
        return
    print(
        f"  {metric}: {total} final rows over {len(tasks)} chrom "
        f"({done} done, {skipped} skipped, {time.time() - t0:.1f}s); "
        f"{mirror_note}",
        file=sys.stderr,
    )


def _reclaim_prior_staging(
    conn: Any, catalog: Any, batch_set_hash: str
) -> None:
    """Delete prior runs' staging for this batch set (used by --rebuild).

    tier_* Iceberg rows (including the dense ref_blocks roots) and the
    tier_completion / aggregation_progress rows are tagged by agg_run_id. A
    completing run prunes other runs' staging rows (_prune_staging) but
    keeps their tier_completion / aggregation_progress rows, and a failed
    run's staging stays until some run completes. On an explicit --rebuild,
    reclaim every prior run recorded for this batch_set_hash before starting
    the new one -- including any of their PG generation tables left DETACHED
    by a finalize that crashed before its swap. (A generation that is
    attached stays: it is the live partition until the new run's finalize
    replaces it.)
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT agg_run_id FROM aggregation_progress "
            "WHERE batch_set_hash=%s",
            (batch_set_hash,),
        )
        old_ids = [int(r[0]) for r in cur.fetchall()]
    if not old_ids:
        return
    print(
        f"--rebuild: reclaiming {len(old_ids)} prior staging run(s) "
        "for this batch set",
        file=sys.stderr,
    )
    staging_tables = [spec.staging_table for spec in SPECS.values()]
    for name in staging_tables + [DENSE_REF_TABLE]:
        table = catalog.load_table(f"{ICEBERG_NAMESPACE}.{name}")
        if table.current_snapshot() is not None:
            table.delete(In("agg_run_id", old_ids))
    with conn.cursor() as cur:
        reclaimed = set(old_ids)
        for name, run_id in detached_generation_tables(cur):
            if run_id in reclaimed:
                print(f"  dropping detached {name}", file=sys.stderr)
                cur.execute(f"DROP TABLE IF EXISTS {name}")
        cur.execute(
            "DELETE FROM tier_completion WHERE agg_run_id = ANY(%s)",
            (old_ids,),
        )
        cur.execute(
            "DELETE FROM aggregation_progress WHERE batch_set_hash=%s",
            (batch_set_hash,),
        )
    conn.commit()


def _nonroot_tiers(tree_roots: Dict[int, int]) -> Dict[int, List[int]]:
    """root tier -> the chromosomes with that root, for every root tier >= 2:
    a tier-tree chromosome whose root is tier r has non-root staging at the
    tiers below r (staging tiers start at 1, so a root at tier 1 has
    none)."""
    out: Dict[int, List[int]] = {}
    for c_idx, root in sorted(tree_roots.items()):
        if root >= 2:
            out.setdefault(int(root), []).append(int(c_idx))
    return out


def _prune_filter(agg_run_id: int, nonroot: Dict[int, List[int]]) -> Any:
    """The Iceberg delete filter that prunes a staging table when run
    ``agg_run_id`` completes: every other run's rows, plus this run's rows
    below each chromosome's root tier (``nonroot`` from _nonroot_tiers; pass
    {} for the ref_blocks tables, whose staging is all root).

    Every staging data file is one unit's output (one run, chromosome, tier
    and group; one window of the dense root), so its column bounds decide
    it outright: pyiceberg drops a file whose bounds prove every row
    matches, keeps one whose bounds prove none can, and only otherwise reads
    it whole to rewrite it. "Another run" is therefore ``agg_run_id < this
    OR agg_run_id > this``, not ``!= this``: pyiceberg's inclusive
    evaluator answers "might match" for any ``!=``, which would make every
    delete read this run's kept root files whole (the dense root: one row
    per covered position) only to rewrite nothing."""
    expr: Any = Or(
        LessThan("agg_run_id", int(agg_run_id)),  # type: ignore[misc, call-arg]
        GreaterThan(  # type: ignore[misc, call-arg]
            "agg_run_id", int(agg_run_id)
        ),
    )
    for root, chroms in sorted(nonroot.items()):
        expr = Or(
            expr,
            And(
                EqualTo(  # type: ignore[misc, call-arg]
                    "agg_run_id", int(agg_run_id)
                ),
                And(
                    In(  # type: ignore[misc, call-arg]
                        "chromosome_index", [int(c) for c in chroms]
                    ),
                    LessThan("tier", int(root)),  # type: ignore[misc, call-arg]
                ),
            ),
        )
    return expr


def _staging_totals(table: Any) -> Tuple[int, int, int]:
    """(data files, rows, bytes) of a staging table's current snapshot, from
    its summary (no scan)."""
    snap = table.current_snapshot()
    if snap is None or snap.summary is None:
        return 0, 0, 0
    return tuple(  # type: ignore[return-value]
        int(snap.summary.get(k) or 0)
        for k in ("total-data-files", "total-records", "total-files-size")
    )


def _prune_staging(
    conn: Any, catalog: Any, agg_run_id: int, tree_roots: Dict[int, int]
) -> None:
    """Retention at completion, after every finalize unit has checkpointed:
    keep only this run's roots, which the next run needs as its parent.

    1. One transaction, committed before any delete: mark every other run
       ``staging_pruned`` and insert this run's prune sentinel
       (tier_completion ``(run, "prune", 0, -1, 0)``). A run whose staging
       is (being) deleted must never be chosen as a parent or resumed, even
       if this step is interrupted; the sentinel records that this run
       pruned its own parent, so _decide_run still resumes it although that
       parent no longer qualifies (as long as no newer complete run has
       appeared to be a parent). If the run crashes after this commit it
       is marked failed, and its resume (every unit and finalize
       checkpointed, so skipped) redoes the deletes and completes it; a
       crash before it leaves an ordinary failed run.
    2. Delete every other run's rows from tier_site_info, tier_allele_counts,
       tier_ref_blocks and tier_ref_blocks_dense, and this run's tier-tree
       rows below each chromosome's root (``tree_roots``: chromosome -> the
       root tier its site_info / allele_counts finalize read). Only the
       newest complete run's roots are ever read again (as the next parent);
       a rebuild reads the batch_* partials, never old staging. Aggregate
       runs are serialized by design (docs/ARCHITECTURE.md, Concurrency).

    pyiceberg drops whole data files here (see _prune_filter), a metadata
    operation; the S3 objects remain until `mosaic maintain` expires the
    snapshots that name them and deletes them (not done here).
    """
    t0 = time.time()
    # Test-only: crash before / after the prune sentinel commits (as
    # MOSAIC_TEST_FAIL_UNIT="prune:*:-1:0:<phase>"). No-op in production.
    _maybe_crash_unit(_PRUNE_METRIC, 0, -1, 0, "pre_register")
    _maybe_crash_unit(_PRUNE_METRIC, 0, -1, 0, "pre_ckpt")
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE aggregation_progress SET staging_pruned = true, "
            "updated_at = now() WHERE agg_run_id <> %s AND NOT staging_pruned "
            "RETURNING agg_run_id",
            (int(agg_run_id),),
        )
        marked = sorted(int(r[0]) for r in cur.fetchall())
    # Commits the marks and the sentinel together.
    _commit_checkpoint(conn, (agg_run_id, _PRUNE_METRIC) + _PRUNE_MARKER)
    _maybe_crash_unit(_PRUNE_METRIC, 0, -1, 0, "post_ckpt")

    nonroot = _nonroot_tiers(tree_roots)
    filters = {
        SPECS[m].staging_table: _prune_filter(agg_run_id, nonroot)
        for m in PRE_METRICS
    }
    filters[SPECS["ref_blocks"].staging_table] = _prune_filter(agg_run_id, {})
    filters[DENSE_REF_TABLE] = _prune_filter(agg_run_id, {})
    for name, delete_filter in filters.items():
        table = catalog.load_table(f"{ICEBERG_NAMESPACE}.{name}")
        if table.current_snapshot() is None:
            continue
        before = _staging_totals(table)
        with warnings.catch_warnings():
            # Nothing to prune (a first run, or a resumed prune).
            warnings.filterwarnings(
                "ignore", message="Delete operation did not match any records"
            )
            _iceberg_commit_with_retry(
                table, functools.partial(table.delete, delete_filter)
            )
        table.refresh()
        after = _staging_totals(table)
        print(
            f"  prune {name}: {before[1]} -> {after[1]} rows, {before[0]} -> "
            f"{after[0]} files, {before[2]} -> {after[2]} bytes",
            file=sys.stderr,
        )
    print(
        f"  prune: other run(s) {_fmt_ids(marked)} marked staging_pruned; "
        f"{time.time() - t0:.1f}s",
        file=sys.stderr,
    )


def _fmt_ids(ids: List[int], limit: int = 20) -> str:
    """Compact batch-id list for the plan log."""
    if len(ids) <= limit:
        return "[" + ",".join(str(i) for i in ids) + "]"
    return f"[{ids[0]}..{ids[-1]}]"


def _check_schema(conn: Any) -> None:
    """Refuse a database initialized before the aggregate tables were
    partitioned by chromosome (site_info a plain table), before the allele
    catalog became a derived table (alleles a plain, append-once table with
    an id column), or before aggregation_progress gained the run lineage
    columns or ingestion_progress.kind (sample removal). There is no
    migration path: the finalize swaps partitions, so
    it cannot run against the old shape, and a database without the lineage
    columns also lacks the dense ref_blocks staging table `mosaic init` now
    creates. SchemaError is init_db's (ingest's lake guard raises it too)."""
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT t, to_regclass(t) IS NOT NULL,
                   EXISTS (SELECT 1 FROM pg_partitioned_table p
                           WHERE p.partrelid = to_regclass(t))
            FROM unnest(%s::text[]) AS t
            """,
            (list(AGGREGATE_PG_TABLES),),
        )
        rows = cur.fetchall()
        cur.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'aggregation_progress'"
        )
        run_columns = {c for (c,) in cur.fetchall()}
        cur.execute(
            "SELECT 1 FROM information_schema.columns "
            "WHERE table_schema = current_schema() "
            "AND table_name = 'ingestion_progress' AND column_name = 'kind'"
        )
        has_kind = cur.fetchone() is not None
    conn.commit()
    problems = [
        f"{t} is missing" if not exists else f"{t} is not partitioned"
        for t, exists, partitioned in rows
        if not partitioned
    ]
    lacking = [c for c in AGGREGATION_RUN_COLUMNS if c not in run_columns]
    if lacking:
        problems.append(f"aggregation_progress lacks {', '.join(lacking)}")
    if not has_kind:
        problems.append("ingestion_progress lacks kind")
    if problems:
        raise SchemaError(
            "the PostgreSQL schema predates this `mosaic aggregate` ("
            + "; ".join(problems)
            + "). There is no migration path: reset with `mosaic clean`, "
            "then re-run `mosaic init` and `mosaic ingest`."
        )


def run_aggregation(
    fan_in: int,
    threads: int,
    rebuild: bool,
    ref_shard_size: int = DEFAULT_REF_SHARD_SIZE,
) -> bool:
    """Aggregate the completed batches. Returns False when there was nothing
    to do (no completed batches, or the batch set is already up to date),
    True after a run completed."""
    ref_shard_size = max(0, int(ref_shard_size))
    conn = get_db_connection()
    try:
        _check_schema(conn)
        catalog = get_catalog()
        # The partials, staging and mirrors must carry the removal columns
        # (n_records, n_listed): no migration path.
        check_removal_lake_schema(catalog)
    except Exception:
        conn.close()
        raise
    contigs = _load_contig_names(conn)
    chrom_lengths = _load_chrom_lengths(conn)
    chrom_windows = {
        c: _position_windows(chrom_lengths.get(c, 0), ref_shard_size)
        for c in contigs
    }
    # Aggregating the complete batches is still correct: a later `mosaic
    # register` + `mosaic aggregate` folds the rest in incrementally.
    n_uploaded = _uploaded_batch_count(conn)
    if n_uploaded:
        print(
            f"WARNING: {n_uploaded} batch(es) uploaded but not registered; "
            "run mosaic register",
            file=sys.stderr,
        )
    batches = _completed_batch_regions(conn)
    completed = [b for b, _ in batches]
    if not completed:
        print("No completed batches to aggregate.", file=sys.stderr)
        conn.close()
        return False
    batch_set_hash = hashlib.sha256(
        ",".join(str(b) for b in completed).encode()
    ).hexdigest()[:24]

    if rebuild:
        _reclaim_prior_staging(conn, catalog, batch_set_hash)

    agg_run_id, state, parent_run_id, delta_ids = _resolve_run(
        conn, batch_set_hash, completed, fan_in, ref_shard_size, rebuild
    )
    conn.commit()
    if state == "current":
        print(f"up to date (agg_run_id={agg_run_id})", file=sys.stderr)
        conn.close()
        return False
    fresh = state == "fresh"
    # No staging DDL: tier_* tables are created once by init_db and every row is
    # tagged with this agg_run_id, so a fresh run's reads never see other runs'
    # rows (and a resume re-reads exactly its own). Avoids drop/create entirely.

    # members: every completed batch covering each chromosome (which
    # chromosomes have data at all); delta_members / plan: only the batches
    # this run reduces (all of them in a full run). A chromosome some
    # completed batch covers gets a root in this run: the delta tree's root
    # (full run), or the fold onto the parent's root (incremental run).
    delta_set = set(delta_ids)
    delta_batches = [(b, r) for b, r in batches if b in delta_set]
    members = _covering_batches(batches, contigs)
    delta_members = _covering_batches(delta_batches, contigs)
    plan = _plan_tiers_by_chrom(delta_batches, contigs, fan_in)
    covered = [c for c in contigs if members[c]]
    parent_roots = _parent_root_tiers(conn, parent_run_id)
    conn.commit()
    n_tiers = max((len(t) for t in plan.values()), default=0)
    ref_tasks = _ref_window_tasks(
        agg_run_id, covered, delta_members, chrom_windows, parent_run_id
    )
    lineage = (
        "full"
        if parent_run_id is None
        else f"parent={parent_run_id} delta={len(delta_ids)}"
    )
    n_removals = len(set(delta_ids) & set(_removal_batch_ids(conn)))
    conn.commit()
    if n_removals:
        lineage += f" removals={n_removals}"
    print(
        f"agg_run_id={agg_run_id} {state} {lineage} "
        f"batches={len(completed)} fan_in={fan_in} tiers={n_tiers} "
        f"threads={threads} ref_shard_size={ref_shard_size} "
        f"ref_windows={len(ref_tasks)}",
        file=sys.stderr,
    )
    for c_idx in covered:
        fold = "" if parent_run_id is None else f" fold={len(plan[c_idx]) + 1}"
        print(
            f"  plan {contigs[c_idx]} (c_idx {c_idx}): "
            f"batches={_fmt_ids(delta_members[c_idx])} "
            f"tiers={len(plan[c_idx])}{fold} "
            f"ref_windows={len(chrom_windows[c_idx])}",
            file=sys.stderr,
        )
    n_empty = len(contigs) - len(covered)
    if n_empty:
        print(
            f"  plan: {n_empty} chromosome(s) covered by no batch "
            "(no reduce units)",
            file=sys.stderr,
        )
    # The tier the finalize reads each chromosome's root from (0 = no batch
    # covers it: the partition is attached empty). An incremental run's root
    # is its fold tier, one above the delta root.
    tree_roots = {
        c: (
            len(plan[c]) + 1
            if parent_run_id is not None and members[c]
            else len(plan[c])
        )
        for c in contigs
    }
    ref_roots = {c: _REF_ROOT_TIER if members[c] else 0 for c in contigs}

    # Run the reduce in a spawn-based ProcessPoolExecutor. spawn (not the default
    # fork) starts each worker as a fresh interpreter, so a worker never inherits
    # the main's threads / boto3 / catalog state -- fork-from-a-multithreaded-
    # process is what made repopulated Pool workers SIGSEGV while creating a boto3
    # client. The executor also raises BrokenProcessPool if a worker dies abruptly
    # (OOM SIGKILL, segfault) instead of hanging forever, so a failure is prompt
    # and the run (marked 'failed' below) stays resumable via tier_completion.
    mp_ctx = multiprocessing.get_context("spawn")
    try:
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=threads,
            mp_context=mp_ctx,
            initializer=_init_worker,
        ) as executor:
            for metric in PRE_METRICS:
                _run_metric_tiers(
                    metric, agg_run_id, plan, executor, fresh, conn, catalog
                )
                if parent_run_id is not None:
                    _run_fold(
                        metric,
                        agg_run_id,
                        parent_run_id,
                        plan,
                        covered,
                        parent_roots,
                        executor,
                        fresh,
                        conn,
                        catalog,
                    )
            # site_info to PG before ref_blocks (the window units' sparse
            # roots read the variant-site positions from PG site_info).
            _run_finalize(
                "site_info", agg_run_id, tree_roots, executor, catalog
            )
            _run_ref_windows(
                ref_tasks, executor, agg_run_id, fresh, conn, catalog
            )
            _run_finalize(
                "ref_blocks", agg_run_id, ref_roots, executor, catalog
            )
            _run_finalize(
                "allele_counts", agg_run_id, tree_roots, executor, catalog
            )
            # Every unit and finalize is checkpointed: keep only this run's
            # roots, then mark the run complete.
            _prune_staging(conn, catalog, agg_run_id, tree_roots)
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE aggregation_progress SET status='complete', "
                    "updated_at=now() WHERE agg_run_id=%s",
                    (agg_run_id,),
                )
            conn.commit()
    except Exception:
        try:
            # Roll back any open transaction on conn first so this status
            # UPDATE isn't itself refused.
            conn.rollback()
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE aggregation_progress SET status='failed', "
                    "updated_at=now() WHERE agg_run_id=%s",
                    (agg_run_id,),
                )
            conn.commit()
        except Exception:
            pass
        raise
    finally:
        conn.close()
    return True


def _fan_in_type(value: str) -> int:
    n = int(value)
    if n < 2:
        raise argparse.ArgumentTypeError(
            "must be >= 2 (1 never converges the tree)"
        )
    return n


def _ref_shard_size_type(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return n


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--fan-in",
        type=_fan_in_type,
        default=100,
        dest="fan_in",
        help="Max inputs reduced per group/tier of the site_info and "
        "allele_counts tier trees (default 100, min 2), per chromosome. >= a "
        "chromosome's batch count collapses it to a single pass; smaller "
        "forces more tiers. ref_blocks has no tier tree and ignores it.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Parallel reduce workers (default: CPU count).",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Start a fresh FULL run (every batch's partials re-reduced, no "
        "parent) instead of resuming, extending the latest complete run "
        "incrementally, or reporting the batch set up to date; reclaim prior "
        "runs' staging (tier_* rows + checkpoints) for this batch set.",
    )
    parser.add_argument(
        "--ref-shard-size",
        type=_ref_shard_size_type,
        default=DEFAULT_REF_SHARD_SIZE,
        dest="ref_shard_size",
        help="Position-window size (bp) of the ref_blocks reduce (default "
        f"{DEFAULT_REF_SHARD_SIZE:,}): one unit per (chromosome, window) sums "
        "every covering batch's dense partials into arrays of ~17 bytes per "
        "base (~340 MB per worker at the default). 0 = one window per "
        "chromosome (discouraged: ~4 GB per worker on chr1). Smaller windows "
        "mean more units and more staging commits. Output is identical for "
        "any value.",
    )


def run(args: Any) -> None:
    threads = args.threads if args.threads > 0 else multiprocessing.cpu_count()

    print("=" * 60)
    print("AGGREGATING PER-BATCH PARTIALS -> POSTGRESQL (tier tree)")
    print("=" * 60)

    try:
        if run_aggregation(
            args.fan_in, threads, args.rebuild, args.ref_shard_size
        ):
            print("\nAggregation complete.")
    except SchemaError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error during aggregation: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate per-batch Iceberg partials (a materialized tier "
            "tree for site_info and allele_counts, position windows for "
            "ref_blocks) and finalize the cross-sample aggregates to "
            "PostgreSQL. Incremental by default: new batches are folded "
            "onto the latest complete run's retained roots. Run after "
            "ingestion, before `mosaic index`."
        )
    )
    configure_parser(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
