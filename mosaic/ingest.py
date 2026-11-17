# Copyright (c) 2026 Sentieon Inc. All rights reserved
import sys
import os
import argparse
import heapq
import multiprocessing
import traceback
import hashlib
import json
import random
import shutil
import subprocess
import threading
from array import array
from concurrent.futures import Future, ThreadPoolExecutor
import tempfile
import time
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    List,
    Optional,
    Set,
    Tuple,
)

import psycopg2
import psycopg2.extras

# PyIceberg / Parquet / S3 imports
from pyiceberg.catalog import load_catalog

try:
    from pyiceberg.exceptions import (
        CommitFailedException as _COMMIT_FAILED_EXC,
    )
except ImportError:
    _COMMIT_FAILED_EXC = Exception  # type: ignore[misc,assignment]
import boto3  # type: ignore[import-untyped]
import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

from .config import (
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
    ICEBERG_NAMESPACE,
    PARQUET_COMPRESSION,
    PARQUET_DATA_PAGE_SIZE,
    PARQUET_ROW_GROUP_ROWS,
    PARQUET_SITE_ROW_GROUP_ROWS,
    S3_BUCKET,
    build_s3_client_kwargs,
)
from .genotypes import BATCH_GENOTYPES_TABLE
from .init_db import (
    SAMPLE_QC_COLUMNS,
    SchemaError,
    check_ingest_lake_schema,
    ensure_batch_genotypes_table,
    ensure_ingest_schema,
)
from .ref_banding import RefBandSpec
from .tombstones import removed_sample_names

try:
    from vcflib import vcf  # type: ignore[import-untyped]
    import vcflib as _vcflib  # type: ignore[import-untyped]
except ImportError:
    print(
        "Error: Could not import vcflib. Install it with: pip install vcflib",
        file=sys.stderr,
    )
    sys.exit(1)

# Parquet write settings — smaller row groups and data pages allow Iceberg
# predicate pushdown to skip more data during selective queries.
# PARQUET_DATA_PAGE_SIZE / PARQUET_COMPRESSION (overridable via env) live in
# config.py; row-group size is a fixed constant.
PARQUET_ROW_GROUP_SIZE = 64 * 1024
PARQUET_KWARGS: Dict[str, Any] = {
    "row_group_size": PARQUET_ROW_GROUP_SIZE,
    "compression": PARQUET_COMPRESSION,
    "data_page_size": PARQUET_DATA_PAGE_SIZE,
    "write_page_index": True,
    "sorting_columns": [
        pq.SortingColumn(1),
        pq.SortingColumn(2),
        pq.SortingColumn(0),
    ],
    "write_statistics": True,
}


def get_db_connection() -> Any:
    return psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]


def _maybe_crash_shard(chrom: str, start_pos: int) -> None:
    """Test-only fault injection; a no-op unless ``MOSAIC_TEST_FAIL_SHARD`` is set.

    ``MOSAIC_TEST_FAIL_SHARD="<chrom>:<threshold_start>"`` makes the main
    process raise *immediately after* it commits the ``shard_completion``
    row of the first shard (in shard order) whose 0-based ``start_pos`` is
    ``>= threshold_start`` (_ShardCheckpointer) -- i.e. that shard's
    checkpoint is durable, but the batch-level Iceberg upload (which happens
    once, after every shard returns) has not run. That is exactly the
    window between a shard's checkpoint and the batch upload, so resume
    tests can reproduce it
    deterministically. Inert in production (the env var is never set there).
    """
    spec = os.environ.get("MOSAIC_TEST_FAIL_SHARD")
    if not spec:
        return
    want_chrom, _, want_start = spec.partition(":")
    if want_chrom == str(chrom) and start_pos >= int(want_start):
        raise RuntimeError(
            f"MOSAIC_TEST_FAIL_SHARD: injected crash after the checkpoint "
            f"commit of shard {chrom}:{start_pos} (threshold {spec})"
        )


class _ShardCheckpointer:
    """The shard results' reduce function (deliver_rust_results calls it
    in shard order, the protocol of vcflib's Sharder.run): collects the
    shard results and commits each shard's ``shard_completion`` row from the
    main process, on its own connection, as the result arrives. The worker
    opens no PostgreSQL connection.

    A row records that the shard was processed by this batch; the shard's
    Iceberg data still waits for the batch-level upload, so a resume
    reprocesses checkpointed shards (the worker reprocesses every shard). ``committed`` holds
    the shards an earlier attempt checkpointed: their row exists, so it is
    not re-inserted. ``seconds`` accumulates the checkpoint commits' time.
    """

    def __init__(
        self, conn: Any, batch_id: int, committed: FrozenSet[Tuple[str, int]]
    ) -> None:
        self.conn = conn
        self.batch_id = batch_id
        self.committed = committed
        self.inserted = 0
        self.seconds = 0.0

    def __call__(
        self,
        results: Optional[List[Dict[str, Any]]],
        rv: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        if rv is None:  # the initial reduce_fun(None) call
            return []
        assert results is not None
        key = (rv["shard_chrom"], rv["shard_start"])
        if key not in self.committed:
            t0 = time.perf_counter()
            with self.conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO shard_completion (batch_id, chrom, start_pos) "
                    "VALUES (%s, %s, %s) ON CONFLICT DO NOTHING",
                    (self.batch_id, key[0], key[1]),
                )
            self.conn.commit()
            self.seconds += time.perf_counter() - t0
            self.inserted += 1
        # Test-only: crash with this shard checkpointed and no Iceberg data
        # uploaded (the window between a shard's checkpoint and the batch
        # upload). No-op in production.
        _maybe_crash_shard(*key)
        results.append(rv)
        return results


def validate_contigs(conn: Any, contigs: Dict[str, Any]) -> Dict[str, int]:
    """
    Validate that VCF contigs match database contigs.
    Returns mapping of contig name to DB id for contigs that exist in both VCF and DB.
    """
    ctg_to_id = {}
    with conn.cursor() as cur:
        cur.execute("SELECT id, chromosome, length FROM contigs ORDER BY id")
        db_contigs = cur.fetchall()

        if not db_contigs:
            print(
                "Error: Contigs table is empty. Please run 'init' command first.",
                file=sys.stderr,
            )
            sys.exit(1)

        db_contig_dict = {row[1]: (row[0], row[2]) for row in db_contigs}

        for name, data in contigs.items():
            if name not in db_contig_dict:
                continue

            db_id, db_length = db_contig_dict[name]
            vcf_length = int(data.get("length", 0))

            if db_length != vcf_length:
                print(
                    f"Error: Contig length mismatch for {name}. DB: {db_length}, VCF: {vcf_length}",
                    file=sys.stderr,
                )
                sys.exit(1)

            ctg_to_id[name] = db_id

        if not ctg_to_id:
            print(
                "Error: No matching chromosomes found between VCF and database.",
                file=sys.stderr,
            )
            print(
                f"  Database has: {', '.join(db_contig_dict.keys())}",
                file=sys.stderr,
            )
            print(f"  VCF has: {', '.join(contigs.keys())}", file=sys.stderr)
            sys.exit(1)

        skipped = set(contigs.keys()) - set(ctg_to_id.keys())
        if skipped:
            print(
                f"  Note: Skipping {len(skipped)} chromosome(s) not in database: "
                f"{', '.join(sorted(skipped)[:5])}{'...' if len(skipped) > 5 else ''}",
                file=sys.stderr,
            )
        print(
            f"  Will process {len(ctg_to_id)} chromosome(s): {', '.join(sorted(ctg_to_id.keys()))}",
            file=sys.stderr,
        )

    return ctg_to_id


def register_samples(conn: Any, vcf_files: List[Any]) -> List[Optional[int]]:
    """
    Register each gVCF's sample in the samples table.

    Returns a list of sample_ids (one per element of vcf_files, None if the
    VCF has no sample column).
    """
    sample_ids: List[Optional[int]] = []
    with conn.cursor() as cur:
        for vf in vcf_files:
            sample_name = vf.samples[0] if vf.samples else None
            if sample_name is None:
                sample_ids.append(None)
                continue
            cur.execute(
                "SELECT id FROM samples WHERE sample_name = %s", (sample_name,)
            )
            row = cur.fetchone()
            if row:
                sample_ids.append(row[0])
            else:
                cur.execute(
                    "INSERT INTO samples (sample_name, batch_id) VALUES (%s, 0) RETURNING id",
                    (sample_name,),
                )
                sample_ids.append(cur.fetchone()[0])
    conn.commit()
    return sample_ids


def write_sample_qc(
    cur: Any,
    batch_id: int,
    batch_qc: Dict[Tuple[int, int], List[int]],
) -> None:
    """Upsert a batch's sample_qc rows, keyed (batch_id, sample_id, chrom).

    ON CONFLICT ... DO UPDATE overwrites: a resumed batch recomputes every
    shard, so its totals replace (never add to) an earlier attempt's.
    """
    if not batch_qc:
        return
    cols = ", ".join(SAMPLE_QC_COLUMNS)
    updates = ", ".join(f"{c} = EXCLUDED.{c}" for c in SAMPLE_QC_COLUMNS)
    rows = [
        (batch_id, sid, c_idx, *counters)
        for (sid, c_idx), counters in sorted(batch_qc.items())
    ]
    psycopg2.extras.execute_values(
        cur,
        f"""
        INSERT INTO sample_qc (batch_id, sample_id, chromosome_index, {cols})
        VALUES %s
        ON CONFLICT (batch_id, sample_id, chromosome_index)
        DO UPDATE SET {updates}
        """,
        rows,
    )


def _get_registered_uris(table: Any) -> Set[str]:
    """Return the set of data-file URIs in the Iceberg table's current snapshot.

    This plans the *entire* table (cost O(files already registered)), so it is
    invoked only on the resume path — a fresh batch/run's URIs provably cannot
    already be registered (the S3 key folds in the SERIAL batch_id/agg_run_id),
    so the caller passes an empty set instead of scanning.
    """
    if table.current_snapshot() is None:
        return set()
    return {task.file.file_path for task in table.scan().plan_files()}


def _add_files_with_retry(
    table: Any,
    s3_uris: List[str],
    max_retries: int = 10,
    check_duplicate_files: bool = True,
) -> None:
    """Register s3_uris with the Iceberg table, retrying on concurrent snapshot conflicts."""
    if not s3_uris:
        return
    for attempt in range(max_retries):
        try:
            # check_duplicate_files scans the table's existing files (O(files
            # already registered)) to catch a data file being registered twice
            # — which would double-count aggregates. The caller passes False
            # only for a *fresh* batch/run, whose URIs provably cannot pre-exist
            # (the S3 key folds in the SERIAL batch_id / agg_run_id, unique to
            # this run); that skips the O(files) scan and is the core of the
            # ingest registration scaling fix. We
            # still force the check on any conflict *retry* (attempt > 0): a
            # retry re-adds the same URIs, and if the prior attempt actually
            # committed on the catalog but the client saw a retriable error
            # (ambiguous commit), an unchecked re-add would silently duplicate
            # the data file. On resume the caller passes True from attempt 0 so
            # already-registered URIs from a crashed prior attempt fail loudly.
            dup_check = check_duplicate_files or attempt > 0
            table.add_files(
                file_paths=s3_uris, check_duplicate_files=dup_check
            )
            return
        except _COMMIT_FAILED_EXC:
            if attempt == max_retries - 1:
                raise
            wait = 2**attempt + random.random()
            print(
                f"  Iceberg commit conflict (attempt {attempt + 1}/{max_retries}), "
                f"retrying in {wait:.1f}s...",
                file=sys.stderr,
            )
            time.sleep(wait)
            # Refresh table metadata so the next attempt sees the branch/snapshot
            # created by the concurrent worker (fixes "branch main was created
            # concurrently" errors when the table has no prior snapshots).
            table.refresh()


def _s3_client() -> Any:
    """A boto3 S3 client for the data-file uploads."""
    return boto3.client(  # type: ignore[call-overload]
        "s3", **build_s3_client_kwargs()
    )


def _init_iceberg_writer(
    sample_names: List[str],
    is_resume: bool,
    s3_client: Any = None,
) -> Dict[str, Any]:
    """Set up S3 and Iceberg connections once; return a reusable writer state dict.

    On a fresh batch (``is_resume`` False) the per-table registered-URI sets are
    left empty: this batch's SERIAL ``batch_id`` is folded into every data
    file's S3 key, so none of its URIs can already be registered and the 6
    full-table ``_get_registered_uris`` scans are provably redundant. Skipping
    them removes the per-batch O(files) cost that made ingest O(N^2). Only a
    resume (which reuses its ``batch_id`` and may have registered some of its
    own URIs before crashing) needs the authoritative scan.

    ``s3_client``: one created earlier (the Rust engine's uploads begin
    during the worker run), else a new one.
    """
    if s3_client is None:
        s3_client = _s3_client()
    catalog = load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)
    rb_table = catalog.load_table(f"{ICEBERG_NAMESPACE}.sample_ref_blocks")
    si_table = catalog.load_table(f"{ICEBERG_NAMESPACE}.sample_site_info")
    brb_table = catalog.load_table(f"{ICEBERG_NAMESPACE}.batch_ref_blocks")
    bsi_table = catalog.load_table(f"{ICEBERG_NAMESPACE}.batch_site_info")
    bac_table = catalog.load_table(f"{ICEBERG_NAMESPACE}.batch_allele_counts")
    # Created on demand so a catalog initialized before the table existed
    # keeps ingesting (`mosaic init` refuses to re-run on a populated DB).
    bgt_table = ensure_batch_genotypes_table(catalog)

    def _registered(table: Any) -> Set[str]:
        return _get_registered_uris(table) if is_resume else set()

    return {
        "sample_names": sample_names,
        "s3_client": s3_client,
        "rb_table": rb_table,
        "si_table": si_table,
        "brb_table": brb_table,
        "bsi_table": bsi_table,
        "bac_table": bac_table,
        "bgt_table": bgt_table,
        "rb_registered": _registered(rb_table),
        "si_registered": _registered(si_table),
        "brb_registered": _registered(brb_table),
        "bsi_registered": _registered(bsi_table),
        "bac_registered": _registered(bac_table),
        "bgt_registered": _registered(bgt_table),
    }


def register_iceberg_uris(
    iceberg_table: Any,
    registered_uris: Set[str],
    obj_uris: List[str],
    check_duplicate_files: bool = True,
) -> None:
    """Register s3 uris with an iceberg table"""
    _add_files_with_retry(
        iceberg_table,
        [uri for uri in obj_uris if uri not in registered_uris],
        check_duplicate_files=check_duplicate_files,
    )


def register_batch_uploads(
    uploads: List[Tuple[str, Any, Set[str], List[str]]],
    is_resume: bool,
) -> float:
    """Register a batch's uploaded data files with Iceberg, one add_files
    commit per table: ``uploads`` is (short name, Iceberg table, the URIs it
    already holds, this batch's URIs) per table, in commit order. The path
    `mosaic ingest` (inline registration) and `mosaic remove` share.

    A fresh batch (a new SERIAL batch_id folded into every S3 key) passes an
    empty registered set and skips pyiceberg's duplicate check: its URIs
    cannot be registered yet. A
    resumed batch passes the table's scanned URIs (_get_registered_uris),
    so a file an earlier attempt registered is not added twice, and keeps
    check_duplicate_files as the backstop. Returns the seconds spent."""
    t0 = time.perf_counter()
    for _name, table, registered, uris in uploads:
        register_iceberg_uris(
            table, registered, uris, check_duplicate_files=is_resume
        )
    return time.perf_counter() - t0


# The Iceberg tables an ingest batch uploads data files to (short names, no
# namespace), in registration order, with their _init_iceberg_writer key.
# These names are what `mosaic ingest --defer-register` records in
# PostgreSQL ingest_files and what `mosaic register` loads.
INGEST_ICEBERG_TABLES: Tuple[Tuple[str, str], ...] = (
    ("sample_ref_blocks", "rb"),
    ("sample_site_info", "si"),
    ("batch_ref_blocks", "brb"),
    ("batch_site_info", "bsi"),
    ("batch_allele_counts", "bac"),
    (BATCH_GENOTYPES_TABLE, "bgt"),
)


def record_ingest_files(
    cur: Any, batch_id: int, uploads: List[Tuple[str, List[str]]]
) -> int:
    """Record a deferred batch's uploaded data files, one ingest_files row
    per (table, uri), pending (registered_at NULL). ON CONFLICT DO NOTHING:
    a resumed batch re-uploads to the same deterministic keys. Runs in the
    caller's batch-completion transaction. Returns the rows offered."""
    rows = [(name, uri, batch_id) for name, uris in uploads for uri in uris]
    if rows:
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO ingest_files (iceberg_table, uri, batch_id) "
            "VALUES %s ON CONFLICT DO NOTHING",
            rows,
        )
    return len(rows)


class RefBlockSegments:
    """One position window's batch_ref_blocks contributions: the inclusive
    (start, end, dp, sample_counts) ranges of every sample's ref blocks
    (and deletion-split parts) clipped to the window, in compact int64
    arrays (32 bytes per range). They overlap freely across samples;
    sum_ref_block_segments adds them up per position at the flush."""

    __slots__ = ("c_idx", "starts", "ends", "dp", "sc")

    def __init__(self, c_idx: int) -> None:
        self.c_idx = c_idx
        self.starts = array("q")
        self.ends = array("q")
        self.dp = array("q")
        self.sc = array("q")

    def add(self, start: int, end: int, dp: int, sc: int) -> None:
        self.starts.append(start)
        self.ends.append(end)
        self.dp.append(dp)
        self.sc.append(sc)

    @classmethod
    def from_arrays(
        cls,
        c_idx: int,
        starts: Any,
        ends: Any,
        dp: Any,
        sc: Any,
    ) -> "RefBlockSegments":
        """Segments from parallel integer arrays (inclusive ranges and their
        per-position dp / sample_counts): `mosaic remove` re-derives a
        sample set's ranges from the lake this way."""
        out = cls(c_idx)
        for name, values in (
            ("starts", starts),
            ("ends", ends),
            ("dp", dp),
            ("sc", sc),
        ):
            getattr(out, name).frombytes(
                np.ascontiguousarray(values, dtype=np.int64).tobytes()
            )
        if not (
            len(out.starts) == len(out.ends) == len(out.dp) == len(out.sc)
        ):
            raise ValueError("segment arrays differ in length")
        return out

    def __len__(self) -> int:
        return len(self.starts)


def sum_ref_block_segments(
    segments: RefBlockSegments,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-position sums of overlapping inclusive ranges: (positions, dp,
    sample_counts), ascending, for every position whose sum is not (0, 0)
    (the positions the partial has always kept).

    A difference array over the ranges' span (np.bincount of +value at each
    start and -value at each end + 1, then a cumulative sum), so the work is
    O(ranges + span) and the memory ~25 bytes per position of the span (a
    position window), never a Python object per position. The float64
    bincount sums are exact integers (|sum| < 2**53).
    """
    n_seg = len(segments)
    if n_seg == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, empty
    starts = np.frombuffer(segments.starts, dtype=np.int64)
    ends = np.frombuffer(segments.ends, dtype=np.int64)
    lo = int(starts.min())
    span = int(ends.max()) - lo + 1
    lo_idx = starts - lo
    hi_idx = ends - lo + 1
    sums = []
    for values in (segments.dp, segments.sc):
        w = np.frombuffer(values, dtype=np.int64).astype(np.float64)
        d = np.bincount(lo_idx, weights=w, minlength=span + 1)
        d -= np.bincount(hi_idx, weights=w, minlength=span + 1)
        np.cumsum(d, out=d)
        sums.append(d[:span])
    covered = (sums[0] != 0) | (sums[1] != 0)
    positions = np.flatnonzero(covered) + lo
    dp_sum = sums[0][covered].astype(np.int64)
    sc_sum = sums[1][covered].astype(np.int64)
    return positions, dp_sum, sc_sum


# The site_info / allele_counts partials as ingest (and `mosaic remove`,
# negated) writes them; the Iceberg schemas are init_db's. gtc is the sum of
# the records' fixed-point GTC (round(g * 2**GTC_FIXED_BITS) as int64:
# the worker's sites.rs, remove.py's np.rint), exact and order-independent, so a removal's negative
# partial nets out exactly; the finalize writes its integer part. n_records /
# n_listed: the batch's per-sample rows at the site / listing the allele;
# the finalize drops a site / an allele whose net count is 0.
BATCH_SITE_INFO_ARROW_SCHEMA = pa.schema(
    [
        pa.field("batch_id", pa.int32(), nullable=False),
        pa.field("chromosome_index", pa.int32(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("qual", pa.float64(), nullable=True),
        pa.field("dp", pa.int64(), nullable=True),
        pa.field("mqsq_1", pa.float64(), nullable=True),
        pa.field("mqsq_2", pa.int64(), nullable=True),
        pa.field("sb", pa.list_(pa.int64()), nullable=True),
        pa.field("gtc", pa.list_(pa.int64()), nullable=True),
        pa.field("n_records", pa.int64(), nullable=True),
    ]
)
BATCH_ALLELE_COUNTS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("batch_id", pa.int32(), nullable=False),
        pa.field("chromosome_index", pa.int32(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("allele", pa.string(), nullable=False),
        pa.field("count", pa.int64(), nullable=True),
        pa.field("n_listed", pa.int64(), nullable=True),
    ]
)


BATCH_REF_BLOCKS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("batch_id", pa.int32(), nullable=False),
        pa.field("chromosome_index", pa.int32(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("dp", pa.int32(), nullable=True),
        pa.field("sample_counts", pa.int32(), nullable=True),
    ]
)


def _whole_sample_key_hash(
    s3_prefix: str,
    sample_names: List[str],
    batch_id: int,
    region: Optional[str],
    c_idx: int,
    idx: int,
) -> str:
    """The key hash of a batch's ``idx``-th whole-sample data file of
    chromosome ``c_idx`` of a per-sample table (the worker's combine;
    earlier also the Python combine's):
    deterministic in the batch, the chromosome and the file's index within
    it, so a resumed batch (same local files, same packing) overwrites the
    same keys."""
    return hashlib.sha256(
        (
            f"{s3_prefix}:{','.join(sorted(sample_names))}:batch{batch_id}"
            f":region{region or ''}:c{c_idx}@agg{idx}"
        ).encode()
    ).hexdigest()[:24]


def _combined_key_hash(
    s3_prefix: str,
    sample_names: List[str],
    batch_id: int,
    region: Optional[str],
    agg_idx: int,
) -> str:
    """The key hash of a batch's ``agg_idx``-th combined data file of one
    table (_aggregate_and_upload_parquet's concatenation): deterministic in
    the batch, so a resumed batch overwrites the same keys."""
    return hashlib.sha256(
        (
            f"{s3_prefix}:{','.join(sorted(sample_names))}"
            f":batch{batch_id}:region{region or ''}@agg{agg_idx}"
        ).encode()
    ).hexdigest()[:24]


def combined_s3_uri(
    s3_prefix: str,
    sample_names: List[str],
    batch_id: int,
    region: Optional[str],
    agg_idx: int = 0,
    bucket: str = S3_BUCKET,
) -> str:
    """The s3:// URI of a batch's ``agg_idx``-th concatenated data file of
    the table at ``s3_prefix`` (the per-batch partials; not the per-sample
    tables' whole-sample files). `mosaic remove` checks its first one to
    tell a table an earlier attempt already registered."""
    key_hash = _combined_key_hash(
        s3_prefix, sample_names, batch_id, region, agg_idx
    )
    return f"s3://{bucket}/{s3_prefix}/{key_hash}/data.parquet"


def _aggregate_and_upload_parquet(
    local_paths: List[str],
    s3_prefix: str,
    s3_client: Any,
    bucket: str,
    max_size_bytes: int,
    sample_names: List[str],
    region: Optional[str],
    batch_id: int,
) -> List[str]:
    """Read small parquet files, concatenate into larger ones, upload to S3.

    Returns list of S3 URIs. Input local_paths are already in genomic order;
    a new combined file opens when the next local file would take the
    current one over ``max_size_bytes``. Since K6 its one caller is `mosaic
    remove` (its negative partials); the ingest worker combines its own
    files (``rust/mosaic-ingest-worker/src/combine.rs``, the same keys:
    ``_combined_key_hash``).

    The S3 key includes ``batch_id`` (and ``region`` for legibility) so that
    the same sample set ingested under different chromosome scopes produces
    distinct keys.  ``batch_id`` is globally unique per ``ingestion_progress``
    row and 1:1 with (sample_set, region): without it the per-chromosome
    fan-out (chr1 task, chr2 task, ... same samples) would compute identical
    keys and silently overwrite each other's data files on S3.  Because a
    resumed batch reuses its ``batch_id``, re-runs stay idempotent (same key,
    overwritten in place) while distinct chromosomes stay separate.
    """
    if not local_paths:
        return []

    s3_uris = []
    current_tables: List[pa.Table] = []
    current_size = 0
    agg_idx = 0

    def _flush_aggregated():
        nonlocal current_tables, current_size, agg_idx
        if not current_tables:
            return
        combined = pa.concat_tables(current_tables)
        key_hash = _combined_key_hash(
            s3_prefix, sample_names, batch_id, region, agg_idx
        )
        # Unique local staging path (mkstemp) so two concurrent runs of the
        # *same* batch — e.g. a resume launched while the original is still
        # draining — don't clobber each other's local file. The S3 *key* below
        # stays deterministic (keyed by key_hash) for resume idempotency; only
        # the transient local file needs to be unique.
        fd, local_file = tempfile.mkstemp(
            prefix=f"{s3_prefix}_{key_hash}_", suffix=".parquet"
        )
        os.close(fd)
        pq.write_table(combined, local_file, **PARQUET_KWARGS)
        try:
            s3_key = f"{s3_prefix}/{key_hash}/data.parquet"
            s3_client.upload_file(local_file, bucket, s3_key)
            s3_uri = f"s3://{bucket}/{s3_key}"
            print(
                f"  Uploaded aggregated {len(combined)} rows to {s3_uri}",
                file=sys.stderr,
            )
            s3_uris.append(s3_uri)
        finally:
            os.remove(local_file)
        current_tables = []
        current_size = 0
        agg_idx += 1

    for path in local_paths:
        table = pq.read_table(path)
        file_size = os.path.getsize(path)
        if current_size > 0 and current_size + file_size > max_size_bytes:
            _flush_aggregated()
        current_tables.append(table)
        current_size += file_size

    _flush_aggregated()

    # Clean up local files
    for path in local_paths:
        try:
            os.remove(path)
        except OSError:
            pass

    return s3_uris


def _ref_gq_bands_arg(text: str) -> RefBandSpec:
    """argparse type of --ref-gq-bands: the parsed (normalized) spec, or an
    argparse error for a malformed one."""
    try:
        return RefBandSpec.parse(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def _ref_gq_bands_text(args: argparse.Namespace) -> Optional[str]:
    """The run's normalized --ref-gq-bands spec (None: no banding). A str
    (a namespace built by hand) is parsed the same way."""
    spec = getattr(args, "ref_gq_bands", None)
    if spec is None:
        return None
    if isinstance(spec, str):
        spec = RefBandSpec.parse(spec)
    return spec.text


# The shard worker: the mosaic-ingest-worker binary (rust/), one process
# per batch, which runs the shards on --threads threads, its results
# delivered to _ShardCheckpointer in genomic order, and then combines the
# local files into the data files. The earlier Python worker it replaced
# has been removed. INGEST_ENGINE is what
# ingestion_progress.engine records for every batch; a stored batch of
# another engine (NULL reads as python) is refused on resume (cmd_ingest).
INGEST_ENGINE = "rust"
INGEST_WORKER_ENV = "MOSAIC_INGEST_WORKER"
INGEST_WORKER_BINARY = "mosaic-ingest-worker"


def find_ingest_worker() -> Optional[str]:
    """The Rust worker binary: $MOSAIC_INGEST_WORKER when set (None if it
    is not an executable file), else mosaic-ingest-worker on PATH."""
    path = os.environ.get(INGEST_WORKER_ENV)
    if path:
        return (
            path if os.path.isfile(path) and os.access(path, os.X_OK) else None
        )
    return shutil.which(INGEST_WORKER_BINARY)


def rust_manifest(
    batch_id: int,
    gvcfs: List[str],
    vcf_files: List[Any],
    sample_ids: List[Optional[int]],
    contig_map: Dict[str, int],
    slot_sample_ids: List[int],
    shards: List[Tuple[str, int, int]],
    pos_batch_size: int,
    ref_gq_bands: Optional[str],
    shard_tmpdir: str,
    committed: FrozenSet[Tuple[str, int]],
    sample_names: List[str],
    combine: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """The worker's manifest: the batch's inputs,
    sample ids, contigs, shards and settings, nothing else. ``shards`` is
    the flattened Sharder.cut list, in genomic order (each entry is one
    shard and one result line). ``combine`` (rust_combine_spec):
    the worker also combines the local files after the shards; absent, it
    ends after them."""
    manifest: Dict[str, Any] = {
        "batch_id": batch_id,
        "inputs": [
            {
                "path": path,
                "sample_id": sid,
                "sample_name": (
                    vf.samples[0] if vf.samples else os.path.basename(path)
                ),
            }
            for path, vf, sid in zip(gvcfs, vcf_files, sample_ids)
        ],
        "contig_map": contig_map,
        "slot_sample_ids": slot_sample_ids,
        "shards": [
            {"chrom": c, "start": s, "end": e, "index": i}
            for i, (c, s, e) in enumerate(shards)
        ],
        "pos_batch_size": pos_batch_size,
        "ref_gq_bands": ref_gq_bands,
        "shard_tmpdir": shard_tmpdir,
        "committed_shards": [[c, p] for c, p in sorted(committed)],
        "sample_names": sorted(sample_names),
    }
    if combine is not None:
        manifest["combine"] = combine
    return manifest


# The worker's combine: the gate's default
# (finished combined files awaiting their upload and acknowledgement) and
# the driver's upload threads.
RUST_COMBINE_PENDING_FILES = 8
RUST_UPLOAD_THREADS = 4
# The tables the worker combines, in registration order
# (INGEST_ICEBERG_TABLES), and the two whose files hold whole samples of one
# chromosome.
RUST_COMBINE_TABLES = tuple(name for name, _ in INGEST_ICEBERG_TABLES)
WHOLE_SAMPLE_TABLES = frozenset(("sample_ref_blocks", "sample_site_info"))
# A shard result's local-path key per combined table.
RUST_TABLE_PATH_KEYS = {
    "sample_ref_blocks": "rb_paths",
    "sample_site_info": "si_paths",
    "batch_ref_blocks": "brb_paths",
    "batch_site_info": "bsi_paths",
    "batch_allele_counts": "bac_paths",
    BATCH_GENOTYPES_TABLE: "bgt_paths",
}


def rust_combine_spec(
    max_size_bytes: int, pending_files: int = RUST_COMBINE_PENDING_FILES
) -> Dict[str, Any]:
    """The manifest's ``combine`` object: the packing / concatenation
    target (--aggregate-parquet-size), the per-sample tables' rows per row
    group, and the gate."""
    return {
        "max_size_bytes": max_size_bytes,
        "rows_per_group": {
            "sample_ref_blocks": PARQUET_ROW_GROUP_ROWS,
            "sample_site_info": PARQUET_SITE_ROW_GROUP_ROWS,
        },
        "pending_files": pending_files,
    }


def parse_rust_result(line: str) -> Tuple[int, Dict[str, Any]]:
    """One result line of the worker -> (shard index, the shard's result
    dict: sample_qc and foreign_del_spans as dicts keyed by tuples). Raises ValueError for a malformed line.
    """
    try:
        obj = json.loads(line)
        kind = obj.pop("kind", "shard")
        if kind != "shard":
            raise ValueError(f"kind {kind!r}")
        idx = obj.pop("index")
        if not isinstance(idx, int):
            raise ValueError(f"shard index {idx!r}")
        obj["sample_qc"] = {
            (int(sid), int(c)): [int(x) for x in counters]
            for sid, c, counters in obj["sample_qc"]
        }
        obj["foreign_del_spans"] = {
            (int(sid), int(c), int(pos)): [int(x) for x in spans]
            for sid, c, pos, spans in obj["foreign_del_spans"]
        }
        for key in (
            "shard_chrom",
            "shard_start",
            "rb_paths",
            "si_paths",
            "brb_paths",
            "bsi_paths",
            "bac_paths",
            "bgt_paths",
            "variant_count",
            "ref_block_count",
            "group_count",
            "parse_stats",
        ):
            if key not in obj:
                raise ValueError(f"no {key!r}")
    except (ValueError, KeyError, TypeError, AttributeError) as e:
        raise ValueError(
            f"malformed mosaic-ingest-worker result line ({e}): "
            f"{line[:200]!r}"
        ) from None
    return idx, obj


def _is_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool)


def parse_rust_file_line(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Check a ``file`` line of the worker's combine (the dict without
    ``kind``): table, chromosome_index (an int for a per-sample table,
    else null), index, path, rows, sample_ids ([first, last] or null),
    n_samples. Raises ValueError."""
    keys = {
        "table",
        "chromosome_index",
        "index",
        "path",
        "rows",
        "sample_ids",
        "n_samples",
    }
    if set(obj) != keys:
        raise ValueError(f"fields {sorted(obj)}")
    table = obj["table"]
    if table not in RUST_COMBINE_TABLES:
        raise ValueError(f"table {table!r}")
    if not _is_int(obj["index"]) or obj["index"] < 0:
        raise ValueError(f"index {obj['index']!r}")
    if not _is_int(obj["rows"]) or obj["rows"] < 0:
        raise ValueError(f"rows {obj['rows']!r}")
    if not isinstance(obj["path"], str) or not obj["path"]:
        raise ValueError(f"path {obj['path']!r}")
    if table in WHOLE_SAMPLE_TABLES:
        ids = obj["sample_ids"]
        if (
            not _is_int(obj["chromosome_index"])
            or not isinstance(ids, list)
            or len(ids) != 2
            or not all(_is_int(x) for x in ids)
            or not _is_int(obj["n_samples"])
            or obj["n_samples"] < 1
        ):
            raise ValueError(
                f"{table}: chromosome_index / sample_ids / n_samples "
                f"{obj['chromosome_index']!r} / {ids!r} / "
                f"{obj['n_samples']!r}"
            )
    elif (
        obj["chromosome_index"] is not None
        or obj["sample_ids"] is not None
        or obj["n_samples"] is not None
    ):
        raise ValueError(f"{table}: per-sample fields set")
    return obj


def parse_rust_summary_line(obj: Dict[str, Any]) -> Dict[str, Any]:
    """Check the combine's ``combined`` summary line (without ``kind``):
    files per table, spans_applied, seconds. Raises ValueError."""
    if set(obj) != {"files", "spans_applied", "seconds"}:
        raise ValueError(f"fields {sorted(obj)}")
    files = obj["files"]
    if not isinstance(files, dict) or any(
        t not in RUST_COMBINE_TABLES or not _is_int(n) or n < 0
        for t, n in files.items()
    ):
        raise ValueError(f"files {files!r}")
    if not _is_int(obj["spans_applied"]) or obj["spans_applied"] < 0:
        raise ValueError(f"spans_applied {obj['spans_applied']!r}")
    if not isinstance(obj["seconds"], (int, float)):
        raise ValueError(f"seconds {obj['seconds']!r}")
    return obj


def parse_rust_line(line: str) -> Tuple[str, Any]:
    """One stdout line of the worker -> (kind, payload): ``("shard",
    (index, result))`` (parse_rust_result; a line without ``kind`` is a
    shard line, R3's protocol), ``("file", dict)``, ``("combined",
    dict)``. Raises ValueError for a malformed line."""
    try:
        obj = json.loads(line)
        if not isinstance(obj, dict):
            raise ValueError("not an object")
        kind = obj.get("kind", "shard")
    except (ValueError, TypeError) as e:
        raise ValueError(
            f"malformed mosaic-ingest-worker result line ({e}): "
            f"{line[:200]!r}"
        ) from None
    if kind == "shard":
        return "shard", parse_rust_result(line)
    obj.pop("kind")
    try:
        if kind == "file":
            return "file", parse_rust_file_line(obj)
        if kind == "combined":
            return "combined", parse_rust_summary_line(obj)
        raise ValueError(f"kind {kind!r}")
    except (ValueError, KeyError, TypeError) as e:
        raise ValueError(
            f"malformed mosaic-ingest-worker {kind} line ({e}): "
            f"{line[:200]!r}"
        ) from None


class RustStreamEnded(RuntimeError):
    """The worker's stdout ended before the protocol was complete."""


class RustShardsMissing(RustStreamEnded):
    """The worker's result stream ended before every shard was reported."""


class RustCombineIncomplete(RustStreamEnded):
    """The worker's stdout ended before the combine's summary line."""


def rust_combined_s3_key(
    fl: Dict[str, Any],
    sample_names: List[str],
    batch_id: int,
    region: Optional[str],
) -> str:
    """The S3 key of a file the worker combined: the Python combine's key
    for the same (table, chromosome, index) (_whole_sample_key_hash for the
    per-sample tables, _combined_key_hash for the others)."""
    table = fl["table"]
    if table in WHOLE_SAMPLE_TABLES:
        h = _whole_sample_key_hash(
            table,
            sample_names,
            batch_id,
            region,
            fl["chromosome_index"],
            fl["index"],
        )
    else:
        h = _combined_key_hash(
            table, sample_names, batch_id, region, fl["index"]
        )
    return f"{table}/{h}/data.parquet"


class RustCombine:
    """The driver's side of the worker's combine: each ``file`` line's
    file is uploaded by a pool of
    ``upload_threads`` threads to the key the Python combine would use, its
    ``Uploaded ...`` line printed, the local file deleted and the file
    acknowledged on the worker's stdin (``ack``, set by run_rust_shards; the
    worker's gate holds new files back until it is). An upload error calls
    ``on_error`` (run_rust_shards kills the worker) and fails the batch.
    After the worker's exit 0, ``finish`` waits for the uploads and
    ``check_complete`` checks the file lines against the summary and the
    shards' local files; ``uris_by_table`` gives each table's URIs in
    (chromosome, index) order for the registration."""

    def __init__(
        self,
        s3_client: Any,
        bucket: str,
        sample_names: List[str],
        batch_id: int,
        region: Optional[str],
        upload_threads: int = RUST_UPLOAD_THREADS,
    ) -> None:
        self.s3_client = s3_client
        self.bucket = bucket
        self.sample_names = sample_names
        self.batch_id = batch_id
        self.region = region
        self.files: Dict[Tuple[str, Optional[int], int], Dict[str, Any]] = {}
        self.uris: Dict[Tuple[str, Optional[int], int], str] = {}
        self.summary: Optional[Dict[str, Any]] = None
        self.error: Optional[BaseException] = None
        self.upload_seconds = 0.0
        self.ack: Optional[Callable[[str], None]] = None
        self.on_error: Optional[Callable[[], None]] = None
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=upload_threads, thread_name_prefix="mosaic-upload"
        )
        self._futures: List[Future] = []

    def file(self, fl: Dict[str, Any]) -> None:
        """A ``file`` line (main thread): queue its upload."""
        key = (fl["table"], fl["chromosome_index"], fl["index"])
        if self.summary is not None:
            raise RuntimeError(
                f"mosaic-ingest-worker reported combined file {key} after "
                "its summary line"
            )
        if key in self.files:
            raise RuntimeError(
                f"mosaic-ingest-worker reported combined file {key} twice"
            )
        self.files[key] = fl
        self._futures.append(self._pool.submit(self._upload, key, fl))

    def combined(self, summary: Dict[str, Any]) -> None:
        """The ``combined`` summary line (main thread)."""
        if self.summary is not None:
            raise RuntimeError(
                "mosaic-ingest-worker printed a second summary line"
            )
        self.summary = summary

    def _upload(
        self, key: Tuple[str, Optional[int], int], fl: Dict[str, Any]
    ) -> None:
        try:
            t0 = time.perf_counter()
            s3_key = rust_combined_s3_key(
                fl, self.sample_names, self.batch_id, self.region
            )
            self.s3_client.upload_file(fl["path"], self.bucket, s3_key)
            uri = f"s3://{self.bucket}/{s3_key}"
            if fl["table"] in WHOLE_SAMPLE_TABLES:
                first, last = fl["sample_ids"]
                msg = (
                    f"  Uploaded {fl['rows']} rows of {fl['n_samples']} "
                    f"sample(s) {first}..{last} (chromosome_index "
                    f"{fl['chromosome_index']}) to {uri}"
                )
            else:
                msg = f"  Uploaded aggregated {fl['rows']} rows to {uri}"
            os.remove(fl["path"])
            with self._lock:
                self.uris[key] = uri
                self.upload_seconds += time.perf_counter() - t0
                print(msg, file=sys.stderr, flush=True)
            if self.ack is not None:
                self.ack(fl["path"])
        except BaseException as e:
            with self._lock:
                if self.error is None:
                    self.error = e
            if self.on_error is not None:
                self.on_error()
            raise

    def raise_error(self) -> None:
        """Raise the first upload error, if any."""
        if self.error is not None:
            raise RuntimeError(
                f"upload of a combined file failed: {self.error}"
            ) from self.error

    def abort(self) -> None:
        """Drop the queued uploads and wait for the running ones."""
        self._pool.shutdown(wait=True, cancel_futures=True)

    def finish(self) -> None:
        """Wait for every upload; raise the first error."""
        self._pool.shutdown(wait=True)
        self.raise_error()

    def check_complete(self, results: List[Dict[str, Any]]) -> None:
        """After the worker's exit 0: the summary was printed, every
        (table, chromosome, index) was reported once with contiguous
        indices from 0 per (table, chromosome), the summary's counts equal
        the file lines', and a table has files iff the shards wrote local
        files for it."""
        if self.summary is None:
            raise RuntimeError(
                "mosaic-ingest-worker exited without its combine summary"
            )
        counts: Dict[str, int] = {t: 0 for t in RUST_COMBINE_TABLES}
        by_part: Dict[Tuple[str, Optional[int]], List[int]] = {}
        for table, c_idx, idx in self.files:
            counts[table] += 1
            by_part.setdefault((table, c_idx), []).append(idx)
        for (table, c_idx), idxs in sorted(
            by_part.items(),
            key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1]),
        ):
            if sorted(idxs) != list(range(len(idxs))):
                raise RuntimeError(
                    f"mosaic-ingest-worker combined {table} (chromosome "
                    f"{c_idx}) file indices {sorted(idxs)[:10]}: not 0.."
                    f"{len(idxs) - 1}"
                )
        for table in RUST_COMBINE_TABLES:
            said = self.summary["files"].get(table, 0)
            if said != counts[table]:
                raise RuntimeError(
                    f"mosaic-ingest-worker's summary counts {said} "
                    f"{table} file(s); it reported {counts[table]}"
                )
            local = sum(len(r[RUST_TABLE_PATH_KEYS[table]]) for r in results)
            if bool(local) != bool(counts[table]):
                raise RuntimeError(
                    f"mosaic-ingest-worker combined {local} local {table} "
                    f"file(s) into {counts[table]} data file(s)"
                )
        if len(self.uris) != len(self.files):
            raise RuntimeError(
                f"{len(self.files) - len(self.uris)} combined file(s) not "
                "uploaded"
            )

    def uris_by_table(self) -> Dict[str, List[str]]:
        """Each table's S3 URIs in (chromosome, index) order."""
        out: Dict[str, List[str]] = {t: [] for t in RUST_COMBINE_TABLES}
        for table, c_idx, idx in sorted(
            self.uris, key=lambda k: (k[0], -1 if k[1] is None else k[1], k[2])
        ):
            out[table].append(self.uris[(table, c_idx, idx)])
        return out


def deliver_rust_results(
    lines: Any,
    n_shards: int,
    reduce_fun: Callable[..., Any],
    combine: Optional[RustCombine] = None,
) -> Any:
    """Feed the worker's result lines (any order) to ``reduce_fun`` (the
    _ShardCheckpointer) in shard-index order, as Sharder.run does:
    ``results = reduce_fun(None)``, then ``results = reduce_fun(results,
    rv)`` per shard. With ``combine`` (R5) the ``file`` lines that follow
    every shard's go to ``combine.file`` (the uploads) and the last,
    ``combined``, to ``combine.combined``. Raises RuntimeError for a
    malformed line, an index out of range or repeated, a line out of
    protocol order, or a shard (RustShardsMissing) or the summary
    (RustCombineIncomplete) never reported."""
    results = reduce_fun(None)
    heap: List[Tuple[int, int, Dict[str, Any]]] = []
    seen: Set[int] = set()
    nxt = 0
    for n, line in enumerate(lines):
        if not line.strip():
            continue
        try:
            kind, payload = parse_rust_line(line)
        except ValueError as e:
            raise RuntimeError(str(e)) from None
        if kind != "shard":
            if combine is None:
                raise RuntimeError(
                    f"mosaic-ingest-worker printed a {kind} line, but no "
                    "combine was asked for"
                )
            if nxt != n_shards:
                raise RuntimeError(
                    f"mosaic-ingest-worker printed a {kind} line after "
                    f"{len(seen)} of {n_shards} shard(s)"
                )
            if combine.summary is not None:
                raise RuntimeError(
                    f"mosaic-ingest-worker printed a {kind} line after its "
                    "combine summary"
                )
            if kind == "file":
                combine.file(payload)
            else:
                combine.combined(payload)
            continue
        if combine is not None and combine.files:
            raise RuntimeError(
                "mosaic-ingest-worker printed a shard line after a "
                "combined file"
            )
        idx, rv = payload
        if not 0 <= idx < n_shards or idx in seen:
            raise RuntimeError(
                f"mosaic-ingest-worker reported shard index {idx} "
                f"({'repeated' if idx in seen else 'out of range'}; "
                f"{n_shards} shard(s))"
            )
        seen.add(idx)
        heapq.heappush(heap, (idx, n, rv))
        while heap and heap[0][0] == nxt:
            _, _, rv = heapq.heappop(heap)
            results = reduce_fun(results, rv)
            nxt += 1
    if nxt != n_shards:
        missing = sorted(set(range(n_shards)) - seen)
        raise RustShardsMissing(
            f"mosaic-ingest-worker reported {len(seen)} of {n_shards} "
            f"shard(s); missing shard index(es) {missing[:10]}"
        )
    if combine is not None and combine.summary is None:
        raise RustCombineIncomplete(
            "mosaic-ingest-worker's output ended before its combine summary "
            f"line ({len(combine.files)} combined file(s) reported)"
        )
    return results


def run_rust_shards(
    worker: str,
    manifest_path: str,
    threads: int,
    n_shards: int,
    reduce_fun: Callable[..., Any],
    combine: Optional[RustCombine] = None,
) -> Any:
    """Run the worker on a manifest (stdout: the result lines, in the
    order its ``threads`` threads finish the shards; stderr: its log,
    passed through) and deliver its results in genomic order. A
    non-zero exit, a malformed line or a missing shard raises RuntimeError;
    the process is killed if delivery fails (e.g. MOSAIC_TEST_FAIL_SHARD).

    With ``combine`` (the manifest has a ``combine`` object, R5) the
    worker's stdin is a pipe: ``combine`` uploads each combined file and
    acknowledges it there (one line, one writer lock); an upload error
    kills the worker and fails the call. After the exit 0 the uploads are
    awaited and the combine checked complete (RustCombine.check_complete).
    """
    proc = subprocess.Popen(
        [worker, "--manifest", manifest_path, "--threads", str(threads)],
        stdout=subprocess.PIPE,
        stdin=subprocess.PIPE if combine is not None else None,
        text=True,
    )
    assert proc.stdout is not None
    if combine is not None:
        assert proc.stdin is not None
        stdin = proc.stdin
        ack_lock = threading.Lock()

        def _ack(path: str) -> None:
            with ack_lock:
                try:
                    stdin.write(path + "\n")
                    stdin.flush()
                except (OSError, ValueError):
                    # The worker is gone (it exits after its summary line,
                    # or died: its exit status says so).
                    pass

        def _kill() -> None:
            if proc.poll() is None:
                proc.kill()

        combine.ack = _ack
        combine.on_error = _kill
    try:
        try:
            results = deliver_rust_results(
                proc.stdout, n_shards, reduce_fun, combine
            )
        except RustStreamEnded:
            # The stream ended: the worker exited (its error is on stderr),
            # or an upload error killed it. Its stdin is closed first: a
            # worker still alive (its stdout cut, waiting at the gate) sees
            # the driver's end and exits.
            if proc.stdin is not None:
                try:
                    proc.stdin.close()
                except OSError:
                    pass
            rc = proc.wait()
            if combine is not None:
                combine.abort()
                combine.raise_error()
            if rc != 0:
                raise RuntimeError(
                    f"mosaic-ingest-worker exited with status {rc}"
                ) from None
            raise
        except BaseException:
            if proc.poll() is None:
                proc.kill()
            proc.wait()
            if combine is not None:
                combine.abort()
            raise
        rc = proc.wait()
        if rc != 0:
            if combine is not None:
                combine.abort()
                combine.raise_error()
            raise RuntimeError(f"mosaic-ingest-worker exited with status {rc}")
        if combine is not None:
            combine.finish()
            combine.check_complete(results)
    finally:
        if proc.stdin is not None:
            try:
                proc.stdin.close()
            except OSError:
                pass
    return results


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest multiple gVCF files into the database."""
    ref_gq_bands = _ref_gq_bands_text(args)
    # The one shard worker: the earlier Python worker is removed; the
    # banner and ingestion_progress.engine keep naming the engine
    # (provenance).
    engine = INGEST_ENGINE
    print(
        f"Ingesting {len(args.gvcfs)} gVCF file(s) "
        f"(ref_gq_bands={ref_gq_bands or 'none'}, engine={engine})...",
        file=sys.stderr,
    )

    # Open all VCF files
    vcf_files = []
    for path in args.gvcfs:
        if not os.path.exists(path):
            print(f"Error: File {path} not found.", file=sys.stderr)
            sys.exit(1)
        vf = vcf.VCF(path)
        vcf_files.append(vf)
        name = vf.samples[0] if vf.samples else os.path.basename(path)
        print(
            f"  [{len(vcf_files)-1}] {path} -> sample: {name}", file=sys.stderr
        )

    try:
        conn = get_db_connection()

        # Refuse a lake whose sample_site_info holds PG allele ids (before
        # anything is written: no migration path).
        try:
            check_ingest_lake_schema(
                load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)
            )
        except SchemaError as e:
            print(f"Error: {e}", file=sys.stderr)
            sys.exit(1)

        # Bring a database initialized before batch_genotypes / sample_qc /
        # ingest_files / samples.status / ingestion_progress.kind up to date
        # (idempotent; serialized across concurrent ingests).
        ensure_ingest_schema(conn)

        # Refuse a tombstoned sample before any work (`mosaic remove` sets
        # the tombstone first): its contribution may still be being
        # subtracted, and re-adding a removed sample is not supported.
        input_names = [vf.samples[0] for vf in vcf_files if vf.samples]
        removed = removed_sample_names(conn, input_names)
        if removed:
            print(
                f"Error: sample(s) {', '.join(removed)} were removed "
                "(samples.status = 'removed'); re-adding a removed sample is "
                "not supported yet. Nothing was ingested.",
                file=sys.stderr,
            )
            sys.exit(1)

        # The worker binary: refuse before anything of the batch is
        # written.
        worker = find_ingest_worker()
        if worker is None:
            where = os.environ.get(INGEST_WORKER_ENV)
            print(
                "Error: the mosaic-ingest-worker binary "
                + (
                    f"{where!r} (${INGEST_WORKER_ENV}) is not an "
                    "executable file"
                    if where
                    else f"is not on PATH and ${INGEST_WORKER_ENV} is "
                    "not set"
                )
                + ". Build it (cd rust && cargo build --release) and put "
                "rust/target/release/mosaic-ingest-worker on PATH or in "
                f"${INGEST_WORKER_ENV}. Nothing was ingested.",
                file=sys.stderr,
            )
            sys.exit(1)

        # Validate contigs for each VCF against the DB and build unified contig_map
        contig_map: Dict[str, int] = {}
        for i, vf in enumerate(vcf_files):
            if not hasattr(vf, "contigs") or not vf.contigs:
                print(
                    f"Error: {args.gvcfs[i]} has no contig headers.",
                    file=sys.stderr,
                )
                sys.exit(1)
            print(f"Validating contigs for file {i}:", file=sys.stderr)
            cm = validate_contigs(conn, vf.contigs)
            for chrom, db_id in cm.items():
                if chrom in contig_map and contig_map[chrom] != db_id:
                    print(
                        f"Error: Inconsistent contig ID for {chrom} across input VCFs.",
                        file=sys.stderr,
                    )
                    sys.exit(1)
            contig_map.update(cm)

        # Register samples and get their DB IDs
        sample_ids = register_samples(conn, vcf_files)
        print(
            f"Registered {sum(s is not None for s in sample_ids)} sample(s): "
            f"{[vcf_files[k].samples[0] for k, s in enumerate(sample_ids) if s is not None]}",
            file=sys.stderr,
        )

        # Collect sample names for deterministic Iceberg S3 key generation.
        all_sample_names = [
            vcf_files[k].samples[0]
            for k in range(len(vcf_files))
            if sample_ids[k] is not None and vcf_files[k].samples
        ]

        # Genotype-matrix slot order: ascending sample_id, persisted as
        # ingestion_progress.sample_ids (the batch_genotypes reader maps
        # slot k -> sample_ids[k]).
        slot_sample_ids = sorted({s for s in sample_ids if s is not None})
        if len(slot_sample_ids) < sum(s is not None for s in sample_ids):
            print(
                "WARNING: a sample appears more than once in this batch; it "
                "gets one genotype-matrix slot (first record wins).",
                file=sys.stderr,
            )

        # --- Checkpoint: check for existing in-progress/complete batch ---
        # Key is (sample_names, region). `region` is whatever was passed
        # via --region (NULL = whole genome). Per-chromosome workflow
        # fan-out (one INGEST_BATCH task per chr) relies on this: running
        # the same samples against chr11 and chr22 produces two distinct
        # batches instead of the second being skipped as "already done".
        # `IS NOT DISTINCT FROM` so NULL = NULL evaluates TRUE. Only
        # kind 'ingest' rows: a `mosaic remove` batch (kind 'removal') lists
        # the removed samples' names with region NULL.
        region = getattr(args, "region", None)
        defer_register = bool(getattr(args, "defer_register", False))
        with conn.cursor() as cur:
            # Check for an already-completed (or uploaded, awaiting `mosaic
            # register`) batch first: either way there is nothing to ingest.
            cur.execute(
                """
                SELECT batch_id, status
                FROM ingestion_progress
                WHERE sample_names = %s
                  AND region IS NOT DISTINCT FROM %s
                  AND kind = 'ingest'
                  AND status IN ('complete', 'uploaded')
                ORDER BY batch_id DESC LIMIT 1
            """,
                (sorted(all_sample_names), region),
            )
            row = cur.fetchone()
            if row:
                if row[1] == "uploaded":
                    # Its Parquet is in S3 and listed in ingest_files; the
                    # registrar is the only path from uploaded to complete.
                    print(
                        f"Batch {row[0]} is uploaded for these samples "
                        f"(region={region!r}) and awaits `mosaic register`. "
                        "Skipping ingestion.",
                        file=sys.stderr,
                    )
                else:
                    print(
                        f"Batch {row[0]} already completed for these samples "
                        f"(region={region!r}). Skipping ingestion.",
                        file=sys.stderr,
                    )
                conn.close()
                return

            # Check for in-progress/failed batch
            cur.execute(
                """
                SELECT batch_id, status, sample_ids, ref_gq_bands, engine
                FROM ingestion_progress
                WHERE sample_names = %s
                  AND region IS NOT DISTINCT FROM %s
                  AND kind = 'ingest'
                  AND status IN ('in_progress', 'failed')
                ORDER BY batch_id DESC LIMIT 1
            """,
                (sorted(all_sample_names), region),
            )
            row = cur.fetchone()
            if row and row[3] != ref_gq_bands:
                # A resume must regenerate the Parquet of the earlier
                # attempt (its committed shards are reprocessed): the same
                # ref-block banding, or the batch would mix two settings.
                print(
                    f"Error: batch {row[0]} (was '{row[1]}', "
                    f"region={region!r}) was begun with "
                    f"ref_gq_bands={row[3] or 'none'}; this run has "
                    f"ref_gq_bands={ref_gq_bands or 'none'}. Resume it "
                    + (
                        f"with --ref-gq-bands {row[3]}"
                        if row[3]
                        else "without --ref-gq-bands"
                    )
                    + ". Nothing was ingested.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if row and (row[4] or "python") != engine:
                # A batch begun under the earlier Python worker (now
                # removed): the engines' local files differ in size, which
                # decides the per-sample packing and the partials' file
                # boundaries, so this worker cannot finish it into the same
                # keys. NULL (an older row) reads as python.
                print(
                    f"Error: batch {row[0]} (was '{row[1]}', "
                    f"region={region!r}) was begun with "
                    f"engine={row[4] or 'python'}, the Python ingest "
                    "worker, which this version of MOSAIC no longer has. "
                    "Finish it with a MOSAIC version that still has "
                    "`mosaic ingest --engine python`, or start over "
                    "(mosaic clean, mosaic init, and ingest again). Nothing "
                    "was ingested.",
                    file=sys.stderr,
                )
                sys.exit(1)
            if row:
                # Resume: reuse the existing batch_id. A prior attempt may have
                # registered some of this batch's (deterministic) URIs before
                # crashing, so the registration path must scan/dedup (is_resume).
                is_resume = True
                resume_batch_id = row[0]
                prev_status = row[1]
                if row[2] is not None and list(row[2]) != slot_sample_ids:
                    raise RuntimeError(
                        f"batch {resume_batch_id}: persisted sample_ids "
                        f"{list(row[2])} differ from this run's "
                        f"{slot_sample_ids}; the genotype-matrix slot order "
                        f"would change on resume"
                    )
                # sample_ids is backfilled for a batch begun before the
                # column existed.
                cur.execute(
                    """
                    UPDATE ingestion_progress SET status = 'in_progress', updated_at = now(),
                        sample_ids = %s
                    WHERE batch_id = %s
                """,
                    (slot_sample_ids, resume_batch_id),
                )
                print(
                    f"Resuming batch {resume_batch_id} (was '{prev_status}', "
                    f"region={region!r})",
                    file=sys.stderr,
                )
            else:
                # Fresh batch: a brand-new SERIAL batch_id, so none of its URIs
                # can be registered yet -> the registration path skips the
                # full-table scans and duplicate check (the O(N^2) fix).
                is_resume = False
                cur.execute(
                    """
                    INSERT INTO ingestion_progress
                        (sample_names, region, status, sample_ids,
                         ref_gq_bands, engine)
                    VALUES (%s, %s, 'in_progress', %s, %s, %s)
                    RETURNING batch_id
                """,
                    (
                        sorted(all_sample_names),
                        region,
                        slot_sample_ids,
                        ref_gq_bands,
                        engine,
                    ),
                )
                resume_batch_id = cur.fetchone()[0]
            # Shards an earlier attempt of this batch checkpointed (none for
            # a fresh batch): reprocessed for their Iceberg data, without a
            # second shard_completion row.
            committed: FrozenSet[Tuple[str, int]] = frozenset()
            if is_resume:
                cur.execute(
                    "SELECT chrom, start_pos FROM shard_completion "
                    "WHERE batch_id = %s",
                    (resume_batch_id,),
                )
                committed = frozenset(
                    (str(c), int(p)) for c, p in cur.fetchall()
                )
        conn.commit()

        # Create shards from contig lengths.
        #
        # When --region names a single chromosome (the per-chromosome fan-out
        # case), restrict sharding to that chromosome. Each batch's gVCFs are
        # data-sliced to the region by the upstream `bcftools view -r`, but they
        # retain the full whole-genome header, so without this filter the
        # shard list holds every DB contig: thousands of empty shards (one
        # per non-target shard-size window), each run by the worker and
        # checkpointed here (with the earlier Python pool each empty shard
        # also re-pickled every input, a measured slowdown).
        region_chrom = region.split(":", 1)[0] if region else None
        if region_chrom is not None and region_chrom not in contig_map:
            print(
                f"Error: --region chromosome {region_chrom!r} is not among the "
                f"contigs shared by the gVCF headers and the database "
                f"({', '.join(sorted(contig_map))}).",
                file=sys.stderr,
            )
            sys.exit(1)
        contig_lengths = [
            (c, 0, int(d["length"]))
            for c, d in vcf_files[0].contigs.items()
            if c in contig_map and (region_chrom is None or c == region_chrom)
        ]
        if region_chrom is not None:
            print(
                f"Restricting sharding to region chromosome {region_chrom} "
                f"(--region {region}).",
                file=sys.stderr,
            )

        # vcflib's Sharder only cuts the shard list; the worker binary runs
        # the shards on --threads threads.
        shards = list(
            _vcflib.Sharder.cut(  # type: ignore[attr-defined]
                contig_lengths, args.shard_size
            )
        )
        print(
            f"Created {len(shards)} shard(s) with size {args.shard_size} bp "
            f"using {args.threads} thread(s)",
            file=sys.stderr,
        )

        # Create temp directory for parquet files
        shard_tmpdir = tempfile.mkdtemp(prefix="mosaic_ingest_")

        # Run sharded ingestion. The workers open no PostgreSQL connection;
        # each shard's checkpoint row is committed here, in the main process,
        # as its result arrives (_ShardCheckpointer).
        checkpointer = _ShardCheckpointer(conn, resume_batch_id, committed)
        # The worker also combines the local files after the shards
        # (R5): the uploads start during its run, so the S3 client is
        # created first.
        s3_client = _s3_client()
        rust_combine = RustCombine(
            s3_client,
            S3_BUCKET,
            all_sample_names,
            resume_batch_id,
            region,
        )
        # The Rust engine: one worker process for the batch, its
        # results delivered to the same checkpointer in shard order.
        flat_shards = [cse for shd in shards for cse in shd]
        manifest_path = os.path.join(shard_tmpdir, "manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(
                rust_manifest(
                    resume_batch_id,
                    list(args.gvcfs),
                    vcf_files,
                    sample_ids,
                    contig_map,
                    slot_sample_ids,
                    flat_shards,
                    args.pos_batch_size,
                    ref_gq_bands,
                    shard_tmpdir,
                    committed,
                    all_sample_names,
                    combine=rust_combine_spec(args.aggregate_parquet_size),
                ),
                fh,
            )
        results = run_rust_shards(
            worker,
            manifest_path,
            args.threads,
            len(flat_shards),
            checkpointer,
            combine=rust_combine,
        )
        print(
            f"Checkpointed {checkpointer.inserted} shard(s) in the main "
            f"process ({checkpointer.seconds:.3f}s); {len(committed)} "
            "checkpointed by an earlier attempt.",
            file=sys.stderr,
        )

        # Aggregate results
        total_variants = sum(r["variant_count"] for r in results)
        total_ref_blocks = sum(r["ref_block_count"] for r in results)
        total_groups = sum(r["group_count"] for r in results)
        print(
            f"Done. {total_groups} groups processed, "
            f"{total_variants} variant sites, "
            f"{total_ref_blocks} ref-blocks ingested.",
            file=sys.stderr,
        )

        # Roll up the per-shard bad-input counters. Anything non-zero means
        # some input was silently handled (dropped record / field counted as
        # 0); surface a batch-level total so it cannot pass a validation run
        # unnoticed.
        batch_parse_stats: Dict[str, int] = {}
        for r in results:
            for key, val in r.get("parse_stats", {}).items():
                batch_parse_stats[key] = batch_parse_stats.get(key, 0) + val
        if batch_parse_stats:
            print(
                "WARNING: bad input handled silently this batch: "
                f"{batch_parse_stats.get('records_dropped', 0)} records "
                "dropped, "
                f"{batch_parse_stats.get('site_info_parse_errors', 0)} "
                "site_info fields and "
                f"{batch_parse_stats.get('gt_parse_errors', 0)} genotype "
                "fields unparseable (counted as 0), "
                f"{batch_parse_stats.get('gt_matrix_conflicts', 0)} "
                "genotype-matrix slot conflicts.",
                file=sys.stderr,
            )

        # Per-sample QC: sum the shards' counters per (sample, chromosome).
        batch_qc: Dict[Tuple[int, int], List[int]] = {}
        for r in results:
            for key, counters in r.get("sample_qc", {}).items():
                acc = batch_qc.get(key)
                if acc is None:
                    batch_qc[key] = list(counters)
                else:
                    for i, val in enumerate(counters):
                        acc[i] += val

        # Initialise Iceberg writer for aggregation + registration. A deferred
        # batch registers nothing here, so it needs no registered-URI scan
        # even on resume: `mosaic register` scans each table once and marks
        # any URI an earlier inline attempt already registered.
        writer = _init_iceberg_writer(
            all_sample_names, is_resume and not defer_register, s3_client
        )

        # Deletion-split spans a shard saw for a deletion row an earlier
        # shard wrote: added to that row as its local file is combined.
        foreign_del_spans: Dict[Tuple[int, int, int], List[int]] = {}
        for r in results:
            for key, spans in r.get("foreign_del_spans", {}).items():
                foreign_del_spans.setdefault(key, []).extend(spans)

        # The worker combined and the uploader uploaded every file;
        # the patch of the cross-shard
        # spans was the worker's, which counted the rows it matched.
        assert rust_combine.summary is not None
        uris = rust_combine.uris_by_table()
        spans_applied = rust_combine.summary["spans_applied"]
        if spans_applied != len(foreign_del_spans):
            raise RuntimeError(
                f"mosaic-ingest-worker added cross-shard deletion-split "
                f"spans to {spans_applied} deletion row(s); the shards "
                f"reported {len(foreign_del_spans)}"
            )
        print(
            f"The worker combined the local files into "
            f"{sum(len(u) for u in uris.values())} data file(s) in "
            f"{rust_combine.summary['seconds']:.2f}s "
            f"({', '.join(f'{t} {len(u)}' for t, u in uris.items())}); "
            f"uploaded on {RUST_UPLOAD_THREADS} thread(s) "
            f"({rust_combine.upload_seconds:.2f}s summed).",
            file=sys.stderr,
        )
        if spans_applied:
            print(
                f"  Added cross-shard deletion-split spans to "
                f"{spans_applied} deletion row(s).",
                file=sys.stderr,
            )
        rb_uris = uris["sample_ref_blocks"]
        si_uris = uris["sample_site_info"]
        brb_uris = uris["batch_ref_blocks"]
        bsi_uris = uris["batch_site_info"]
        bac_uris = uris["batch_allele_counts"]
        bgt_uris = uris[BATCH_GENOTYPES_TABLE]

        # (table, writer key, uris) in INGEST_ICEBERG_TABLES order.
        uris_by_key = {
            "rb": rb_uris,
            "si": si_uris,
            "brb": brb_uris,
            "bsi": bsi_uris,
            "bac": bac_uris,
            "bgt": bgt_uris,
        }
        uploads = [
            (name, key, uris_by_key[key])
            for name, key in INGEST_ICEBERG_TABLES
        ]
        total_uris = sum(len(uris) for _, _, uris in uploads)
        if defer_register:
            # The six per-task commits are the shared-table contention the
            # registrar removes: record the file list (below, with the
            # batch's completion) and leave the batch 'uploaded'.
            print(
                f"Deferring the registration of {total_uris} URIs "
                "(--defer-register): recorded in ingest_files for "
                "`mosaic register`",
                file=sys.stderr,
            )
        else:
            # Register with Iceberg
            print(
                f"Registering {total_uris} URIs with the Iceberg catalog",
                file=sys.stderr,
            )
            t_reg = register_batch_uploads(
                [
                    (
                        name,
                        writer[f"{key}_table"],
                        writer[f"{key}_registered"],
                        uris,
                    )
                    for name, key, uris in uploads
                ],
                is_resume,
            )
            print(
                f"Iceberg write complete ({t_reg:.2f}s registering).",
                file=sys.stderr,
            )

        # Mark batch as complete (or, deferred, uploaded with its file list
        # in ingest_files), write its per-sample QC, and clean up
        # shard_completion rows, in one transaction. Every shard is
        # reprocessed on resume, so the QC totals are always complete; the
        # upsert overwrites any earlier attempt's rows instead of adding.
        final_status = "uploaded" if defer_register else "complete"
        n_files = 0
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE ingestion_progress SET status = %s, updated_at = now()
                WHERE batch_id = %s
            """,
                (final_status, resume_batch_id),
            )
            if defer_register:
                n_files = record_ingest_files(
                    cur,
                    resume_batch_id,
                    [(name, uris) for name, _, uris in uploads],
                )
            write_sample_qc(cur, resume_batch_id, batch_qc)
            cur.execute(
                "DELETE FROM shard_completion WHERE batch_id = %s",
                (resume_batch_id,),
            )
        conn.commit()
        print(
            f"Wrote {len(batch_qc)} sample_qc row(s).",
            file=sys.stderr,
        )
        if defer_register:
            print(
                f"Batch {resume_batch_id} uploaded: {n_files} data file(s) "
                "recorded in ingest_files; run `mosaic register` to "
                "register them and complete the batch.",
                file=sys.stderr,
            )
        shutil.rmtree(shard_tmpdir, ignore_errors=True)

    except Exception as e:
        traceback.print_exc()
        print(f"Error: {e}", file=sys.stderr)
        if "conn" in locals():
            conn.rollback()
            if "resume_batch_id" in locals():
                try:
                    cur = conn.cursor()
                    cur.execute(
                        """
                        UPDATE ingestion_progress SET status = 'failed', updated_at = now()
                        WHERE batch_id = %s
                    """,
                        (resume_batch_id,),
                    )
                    conn.commit()
                except Exception:
                    pass
        sys.exit(1)
    finally:
        if "conn" in locals():
            conn.close()


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("gvcfs", nargs="+", help="Input gVCF files to ingest")
    parser.add_argument(
        "--shard-size",
        type=int,
        default=10_000_000,
        help="Genomic region size in bp per parallel shard "
        "(default: 10,000,000 = 10 Mbp)",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=multiprocessing.cpu_count(),
        help="Number of shards processed concurrently: threads of the one "
        "mosaic-ingest-worker process, which also combine the local files "
        "(default: number of CPUs)",
    )
    parser.add_argument(
        "--aggregate-parquet-size",
        type=int,
        default=268_435_456,
        help="Target size in bytes for aggregated parquet files "
        "(default: 268435456 = 256 MB)",
    )
    parser.add_argument(
        "--pos-batch-size",
        type=int,
        default=10_000_000,
        help="Genomic window size in bp per Iceberg parquet file "
        "(0 = all positions in one file, default: 10,000,000)",
    )
    parser.add_argument(
        "--region",
        type=str,
        default=None,
        help="Optional region string (e.g. 'chr21' or 'chr21:1-30000000'). "
        "Recorded against the checkpoint row in `ingestion_progress`, so the "
        "same sample set can be ingested under multiple chromosome scopes "
        "without colliding in the resume logic. When it names a chromosome, "
        "sharding is also restricted to that chromosome: input gVCFs are "
        "data-sliced to the region by the upstream download but keep their "
        "full whole-genome header, so this avoids enumerating empty shards "
        "across all other contigs. The range part (after ':'), if any, is "
        "used only for the checkpoint label, not as a sharder bound.",
    )
    parser.add_argument(
        "--defer-register",
        action="store_true",
        dest="defer_register",
        help="Upload the Parquet data files but do not register them with "
        "Iceberg: record them in the PostgreSQL ingest_files table and leave "
        "the batch 'uploaded' (invisible to `mosaic aggregate` and the "
        "readers) until `mosaic register` registers them, many files per "
        "commit, and completes it. Removes the per-task commits to the "
        "shared Iceberg tables (the Nextflow workflow's default). Without "
        "it the batch registers its files itself and completes.",
    )
    parser.add_argument(
        "--ref-gq-bands",
        type=_ref_gq_bands_arg,
        default=None,
        dest="ref_gq_bands",
        metavar="SPEC",
        help="Ref-block banding (default: off, every ref block stored as "
        "the gVCF has it). SPEC is ascending GQ band bounds in [1, 99], "
        "comma-separated (GATK --gvcf-gq-bands semantics: '20,60' is the "
        "bands [0,20), [20,60), [60,99]), or 'all' (one band). A sample's "
        "adjacent ref blocks of the same GT class (hom-ref / no-call) and "
        "GQ band in the same --pos-batch-size window are stored as one "
        "sample_ref_blocks row: dp their length-weighted mean (rounded), "
        "AD / PL of the lowest-GQ block. Changes PG ref_blocks.dp (so INFO/DP, QD) and the "
        "FORMAT DP / AD / PL of ref-filled samples; not AN, the genotype "
        "matrix or sample_qc. Recorded in ingestion_progress.ref_gq_bands; "
        "a resume must use the same value.",
    )


def run(args: Any) -> None:
    cmd_ingest(args)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest one or more gVCF files into PostgreSQL + Iceberg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s sample1.gvcf.gz sample2.gvcf.gz
  %(prog)s *.gvcf.gz
        """,
    )
    configure_parser(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
