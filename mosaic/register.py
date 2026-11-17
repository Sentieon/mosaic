# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""`mosaic register`: the Iceberg registrar for deferred ingest batches.

`mosaic ingest --defer-register` uploads a batch's Parquet data files, records
them in the PostgreSQL ``ingest_files`` table and leaves the batch
``uploaded`` (invisible to `mosaic aggregate`, ``cohort_generation`` and the
readers, which see ``complete`` batches only). This command registers the
pending files (``registered_at IS NULL``) with their Iceberg tables and
completes the batches, so the shared tables take one commit per
``--files-per-commit`` files per registrar run instead of six commits per
ingest task.

Per table: scan its registered data-file URIs once (``_get_registered_uris``);
pending URIs already registered are leftovers of a crash between an Iceberg
commit and its PostgreSQL mark, so they are marked without being re-added; the
rest are added in chunks (``_add_files_with_retry`` with the duplicate check
off: the scan proved them absent, and the helper forces the check on any
retry). **Iceberg first, PostgreSQL second:** after each commit one
PostgreSQL transaction marks the chunk's rows and completes every batch of the
chunk with no pending row left in any table. Registrars are serialized by a
session advisory lock (one committer per table), which `mosaic maintain`
also takes: the Iceberg metadata-writer lock the two share. The registrar
does not expire snapshots (`mosaic maintain` does).
"""

import argparse
import os
import sys
import time
import traceback
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import psycopg2

from pyiceberg.catalog import load_catalog

from .config import (
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
    ICEBERG_NAMESPACE,
)
from .genotypes import BATCH_GENOTYPES_TABLE
from .ingest import (
    INGEST_ICEBERG_TABLES,
    _add_files_with_retry,
    _get_registered_uris,
)
from .init_db import ensure_batch_genotypes_table

# pg_try_advisory_lock key of the Iceberg metadata-writer lock: it serializes
# registrars, and `mosaic maintain` takes it too, so a registrar and a
# maintainer never run at once ("MOSR"; init_db's schema lock is "MOSA",
# 0x4D4F5341).
REGISTER_LOCK_KEY = 0x4D4F5352

DEFAULT_FILES_PER_COMMIT = 500

# The tables a deferred ingest records files for (ingest_files.iceberg_table).
REGISTERABLE_TABLES = frozenset(name for name, _ in INGEST_ICEBERG_TABLES)

_HOOK_ENV = "MOSAIC_TEST_FAIL_REGISTER"
_HOOK_PHASES = ("pre_commit", "post_commit")


@dataclass(frozen=True)
class PendingFile:
    """One pending ingest_files row."""

    iceberg_table: str
    uri: str
    batch_id: int


def group_by_table(
    files: Iterable[PendingFile],
) -> List[Tuple[str, List[PendingFile]]]:
    """(table, files) in the tables' first-seen order, each table's files in
    input order (the pending query orders by (iceberg_table, batch_id,
    uri))."""
    out: Dict[str, List[PendingFile]] = {}
    for f in files:
        out.setdefault(f.iceberg_table, []).append(f)
    return list(out.items())


def chunk_files(
    files: Sequence[PendingFile], files_per_commit: int
) -> List[List[PendingFile]]:
    """Consecutive chunks of at most ``files_per_commit`` files, order kept:
    one Iceberg commit each."""
    if files_per_commit < 1:
        raise ValueError(f"files_per_commit must be >= 1: {files_per_commit}")
    return [
        list(files[i : i + files_per_commit])
        for i in range(0, len(files), files_per_commit)
    ]


class CompletionTracker:
    """Pending rows per batch over every table. A batch completes when its
    last pending row (in any table) is marked, not before."""

    def __init__(self, files: Iterable[PendingFile]) -> None:
        self.pending: Counter = Counter(f.batch_id for f in files)

    def mark(self, files: Iterable[PendingFile]) -> List[int]:
        """Mark ``files`` registered; return the batches (sorted) whose last
        pending row this marks."""
        done = set()
        for f in files:
            left = self.pending[f.batch_id] - 1
            if left < 0:
                raise ValueError(
                    f"batch {f.batch_id}: {f.iceberg_table} {f.uri} marked "
                    "with no pending row left"
                )
            self.pending[f.batch_id] = left
            if left == 0:
                done.add(f.batch_id)
        return sorted(done)


def parse_fail_register(spec: Optional[str]) -> Optional[Tuple[str, str]]:
    """Parse ``MOSAIC_TEST_FAIL_REGISTER="<iceberg_table>:<phase>"`` (phase
    ``pre_commit`` or ``post_commit``) into (table, phase); None when unset.
    A malformed spec raises ValueError rather than silently doing nothing (a
    test relying on it would otherwise pass vacuously)."""
    if not spec:
        return None
    table, sep, phase = spec.rpartition(":")
    if not sep or not table or phase not in _HOOK_PHASES:
        raise ValueError(
            f"{_HOOK_ENV}={spec!r}: expected "
            "<iceberg_table>:<pre_commit|post_commit>"
        )
    return table, phase


def _maybe_fail_register(
    hook: Optional[Tuple[str, str]], table: str, phase: str
) -> None:
    """Test-only fault injection, inert unless MOSAIC_TEST_FAIL_REGISTER is
    set: raise just before (``pre_commit``: nothing of the table registered)
    or just after (``post_commit``: the files are registered in Iceberg but
    their ingest_files rows are not marked and the batch is not complete) the
    table's first add_files commit. The caller invokes it for the first
    commit only."""
    if hook == (table, phase):
        raise RuntimeError(
            f"{_HOOK_ENV}: injected crash {table} @{phase} of its first "
            "commit"
        )


def files_per_commit_type(value: str) -> int:
    """argparse type of --files-per-commit (an integer >= 1)."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def get_db_connection() -> Any:
    return psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]


def _load_table(catalog: Any, name: str) -> Any:
    if name == BATCH_GENOTYPES_TABLE:
        return ensure_batch_genotypes_table(catalog)
    return catalog.load_table(f"{ICEBERG_NAMESPACE}.{name}")


def _complete_batches(cur: Any, batch_ids: Sequence[int]) -> List[int]:
    """Set ``complete`` on those of ``batch_ids`` that are ``uploaded`` with
    no pending ingest_files row left (the authoritative check; the caller's
    tracker only nominates them). Returns the batches completed."""
    if not batch_ids:
        return []
    cur.execute(
        """
        UPDATE ingestion_progress p
        SET status = 'complete', updated_at = now()
        WHERE p.batch_id = ANY(%s)
          AND p.status = 'uploaded'
          AND NOT EXISTS (
              SELECT 1 FROM ingest_files f
              WHERE f.batch_id = p.batch_id AND f.registered_at IS NULL)
        RETURNING p.batch_id
        """,
        (list(batch_ids),),
    )
    return sorted(b for (b,) in cur.fetchall())


def _mark_registered(
    conn: Any,
    table: str,
    files: Sequence[PendingFile],
    tracker: CompletionTracker,
) -> List[int]:
    """One PostgreSQL transaction: mark ``files`` registered and complete
    every batch whose last pending row they are. Returns those batches."""
    nominated = tracker.mark(files)
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ingest_files SET registered_at = now() "
            "WHERE iceberg_table = %s AND uri = ANY(%s) "
            "AND registered_at IS NULL",
            (table, [f.uri for f in files]),
        )
        completed = _complete_batches(cur, nominated)
    conn.commit()
    return completed


def _complete_empty_batches(conn: Any) -> List[int]:
    """Complete every ``uploaded`` batch that has no pending row: a deferred
    batch that uploaded no data file (nothing to register) would otherwise
    stay ``uploaded`` forever."""
    with conn.cursor() as cur:
        cur.execute("""
            UPDATE ingestion_progress p
            SET status = 'complete', updated_at = now()
            WHERE p.status = 'uploaded'
              AND NOT EXISTS (
                  SELECT 1 FROM ingest_files f
                  WHERE f.batch_id = p.batch_id AND f.registered_at IS NULL)
            RETURNING p.batch_id
            """)
        completed = sorted(b for (b,) in cur.fetchall())
    conn.commit()
    return completed


def _load_pending(conn: Any) -> List[PendingFile]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT iceberg_table, uri, batch_id FROM ingest_files "
            "WHERE registered_at IS NULL "
            "ORDER BY iceberg_table, batch_id, uri"
        )
        rows = [PendingFile(t, u, int(b)) for t, u, b in cur.fetchall()]
    conn.commit()
    return rows


def _print_completed(completed: Sequence[int]) -> None:
    ids = ", ".join(str(b) for b in sorted(completed)) or "none"
    print(
        f"batches completed: {len(completed)} ({ids})",
        file=sys.stderr,
    )


def run_register(files_per_commit: int) -> int:
    """Register every pending ingest_files row. Returns the exit code: 0 on
    success (including nothing to register), 1 when another registrar holds
    the lock. Raises on any other failure (Iceberg first, PostgreSQL second,
    so a re-run resumes)."""
    if files_per_commit < 1:
        raise ValueError(f"files_per_commit must be >= 1: {files_per_commit}")
    hook = parse_fail_register(os.environ.get(_HOOK_ENV))
    t_start = time.perf_counter()
    conn = get_db_connection()
    locked = False
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT pg_try_advisory_lock(%s)", (REGISTER_LOCK_KEY,)
            )
            locked = cur.fetchone()[0]
            cur.execute("SELECT to_regclass('ingest_files') IS NOT NULL")
            have_table = cur.fetchone()[0]
        conn.commit()
        if not locked:
            print(
                "Error: another `mosaic register` or `mosaic maintain` "
                "holds the registrar lock (the Iceberg metadata-writer lock, "
                f"pg advisory lock {REGISTER_LOCK_KEY:#x}); they are "
                "serialized. Nothing registered.",
                file=sys.stderr,
            )
            return 1
        if not have_table:
            # A database initialized before deferred registration, never
            # touched by a D2 ingest: nothing can be pending.
            print("nothing to register", file=sys.stderr)
            return 0

        completed = _complete_empty_batches(conn)
        pending = _load_pending(conn)
        if not pending:
            print("nothing to register", file=sys.stderr)
            if completed:
                _print_completed(completed)
            return 0
        unknown = sorted(
            {f.iceberg_table for f in pending} - REGISTERABLE_TABLES
        )
        if unknown:
            raise ValueError(
                f"ingest_files names tables a batch does not write: {unknown}"
            )

        plan = group_by_table(pending)
        tracker = CompletionTracker(pending)
        n_batches = len({f.batch_id for f in pending})
        print(
            f"mosaic register: {len(pending)} pending file(s) in "
            f"{len(plan)} table(s) from {n_batches} batch(es); "
            f"files_per_commit={files_per_commit}",
            file=sys.stderr,
        )
        catalog = load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)
        totals = {"scan": 0.0, "commit": 0.0, "mark": 0.0}
        for name, files in plan:
            t0 = time.perf_counter()
            table = _load_table(catalog, name)
            registered = _get_registered_uris(table)
            scan_s = time.perf_counter() - t0
            leftovers = [f for f in files if f.uri in registered]
            todo = [f for f in files if f.uri not in registered]
            mark_s = 0.0
            if leftovers:
                t0 = time.perf_counter()
                completed += _mark_registered(conn, name, leftovers, tracker)
                mark_s += time.perf_counter() - t0
            commits = 0
            commit_s = 0.0
            for chunk in chunk_files(todo, files_per_commit):
                if commits == 0:
                    _maybe_fail_register(hook, name, "pre_commit")
                t0 = time.perf_counter()
                _add_files_with_retry(
                    table,
                    [f.uri for f in chunk],
                    check_duplicate_files=False,
                )
                commit_s += time.perf_counter() - t0
                commits += 1
                if commits == 1:
                    _maybe_fail_register(hook, name, "post_commit")
                t0 = time.perf_counter()
                completed += _mark_registered(conn, name, chunk, tracker)
                mark_s += time.perf_counter() - t0
            totals["scan"] += scan_s
            totals["commit"] += commit_s
            totals["mark"] += mark_s
            print(
                f"  {name}: pending={len(files)} "
                f"leftovers_marked={len(leftovers)} commits={commits} "
                f"registered={len(todo)} scan={scan_s:.2f}s "
                f"commit={commit_s:.2f}s mark={mark_s:.2f}s "
                f"seconds={scan_s + commit_s + mark_s:.2f}",
                file=sys.stderr,
            )
        print(
            f"registered in {time.perf_counter() - t_start:.2f}s "
            f"(scan={totals['scan']:.2f}s commit={totals['commit']:.2f}s "
            f"mark={totals['mark']:.2f}s)",
            file=sys.stderr,
        )
        _print_completed(completed)
        return 0
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


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--files-per-commit",
        type=files_per_commit_type,
        default=DEFAULT_FILES_PER_COMMIT,
        dest="files_per_commit",
        help="Data files registered per Iceberg commit (default "
        f"{DEFAULT_FILES_PER_COMMIT}, min 1). Each commit of a table "
        "registers up to this many of its pending files; the batches whose "
        "files are all registered are completed after each commit.",
    )


def run(args: Any) -> None:
    try:
        rc = run_register(args.files_per_commit)
    except Exception as e:
        traceback.print_exc()
        print(f"Error during registration: {e}", file=sys.stderr)
        sys.exit(1)
    if rc:
        sys.exit(rc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Register the data files of deferred ingest batches (`mosaic "
            "ingest --defer-register`) with Iceberg, many files per commit, "
            "and complete the batches. Run after the ingest fan-out, before "
            "`mosaic aggregate`."
        )
    )
    configure_parser(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
