# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""`mosaic maintain`: Iceberg maintenance of the namespace's tables.

Nothing else in MOSAIC expires a snapshot or deletes a file: every commit
leaves a manifest list (and usually a manifest) behind, a fast append carries
every existing manifest forward, and the aggregate's completion prune
(``_prune_staging``) only records its whole-file deletes, so the pruned
staging stays in S3. This command bounds that growth. Per Iceberg table of
``init_db.ICEBERG_TABLES``:

1. **Properties.** Set ``init_db.maintained_properties`` on a table that
   lacks them (a lake initialized before them): ``ICEBERG_TABLE_PROPERTIES``
   (the committer then deletes the ``metadata.json`` versions that fall out
   of the metadata log) and, on the two per-sample tables,
   ``SAMPLE_TABLE_CODEC_PROPERTIES`` (brotli at level 5: a file a
   ``mosaic remove`` rewrite writes is then one the Rust joint caller can
   read).
2. **Manifest rewrite.** A merge-append of no files, in one transaction with
   the three manifest-merge properties staged and then restored: the current
   snapshot's data manifests are packed into ``--manifest-target-size-bytes``
   bins and every bin of two or more is merged (pyiceberg 0.11 has no
   ``rewrite_manifests``; ``_MergeAppendFiles`` reads the merge properties
   from the transaction's staged metadata). Skipped when the current
   snapshot has fewer than two data manifests.
3. **Snapshot expiry and file reclaim.** A snapshot is retained when it is
   the current one, one of the ``--retain-last`` newest, younger than
   ``--older-than-hours``, or a branch head / tag; every other snapshot is
   expired. Before the expire commit the reclaim set is computed from the
   metadata (``compute_reclaim``): the expired snapshots' manifest lists,
   their manifests and every file they name, plus every file the retained
   snapshots name only as DELETED, minus everything a retained snapshot
   still references. Then the expire commit, then the deletes (grouped by
   bucket, ``delete_objects`` in chunks of 1,000). A crash between the two
   leaves unreferenced files, never a dangling reference.
4. **Orphan sweep** (``--orphans``, opt-in). Lists the upload prefixes
   ``s3://<MOSAIC_S3_BUCKET>/<table>/`` and every table location, and
   deletes each object that no snapshot of any table in any namespace of the
   catalog references, that is not a pending ``ingest_files`` URI and that
   is older than ``--orphan-grace-hours``. It assumes the swept prefixes
   belong to this catalog.

Every S3 path is parsed from its ``s3://`` URI: the table metadata lives
under the catalog's warehouse, which can be another bucket than
``MOSAIC_S3_BUCKET``.

``mosaic maintain`` takes the registrar's session advisory lock
(``register.REGISTER_LOCK_KEY``, the Iceberg metadata-writer lock the two
commands share) and refuses to run while an aggregation run is
``in_progress`` (``--force`` overrides; a SIGKILLed aggregate stays
``in_progress``). ``--dry-run`` takes neither, commits nothing and deletes
nothing. No MOSAIC reader time-travels: every scan plans from the current
snapshot, so with the pipeline serialized (the workflow's ``MAINTAIN`` step
runs after ``AGGREGATE``) ``--older-than-hours 0`` is safe; the default of
120 h protects a reader started by hand that planned from an older snapshot.
"""

import argparse
import math
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import boto3  # type: ignore[import-untyped]
import psycopg2

from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NoSuchTableError
from pyiceberg.manifest import ManifestContent, ManifestEntryStatus
from pyiceberg.table import TableProperties
from pyiceberg.table.update.snapshot import ExpireSnapshots

from .config import (
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
    ICEBERG_NAMESPACE,
    S3_BUCKET,
    build_s3_client_kwargs,
)
from .init_db import ICEBERG_TABLES, maintained_properties
from .register import REGISTER_LOCK_KEY

DEFAULT_RETAIN_LAST = 1
# Iceberg's history.expire.max-snapshot-age-ms default (five days).
DEFAULT_OLDER_THAN_HOURS = 120.0
MIN_MANIFEST_TARGET_SIZE_BYTES = 1 << 20
DEFAULT_MANIFEST_TARGET_SIZE_BYTES = 8 << 20  # pyiceberg's default
DEFAULT_ORPHAN_GRACE_HOURS = 72.0
# S3 DeleteObjects takes at most 1,000 keys per request.
DELETE_CHUNK = 1000
# A dry-run sweep names this many of the objects it would delete.
ORPHAN_LIST_LIMIT = 20
# Manifest lists and manifests read concurrently.
_READ_THREADS = 8

# The table properties _MergeAppendFiles reads; staged for the rewrite and
# restored before its commit.
_MERGE_PROPERTIES = (
    TableProperties.MANIFEST_MERGE_ENABLED,
    TableProperties.MANIFEST_MIN_MERGE_COUNT,
    TableProperties.MANIFEST_TARGET_SIZE_BYTES,
)

_HOOK_ENV = "MOSAIC_TEST_FAIL_MAINTAIN"
_HOOK_PHASES = ("pre_expire", "post_expire")

_S3_SCHEMES = ("s3", "s3a", "s3n")

# Reclaim kinds, in report order.
KINDS = ("data", "manifests", "lists")

ObjectKey = Tuple[str, str]  # (bucket, key)


# ── options ──────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class MaintainOptions:
    retain_last: int = DEFAULT_RETAIN_LAST
    older_than_hours: float = DEFAULT_OLDER_THAN_HOURS
    rewrite_manifests: bool = True
    manifest_target_size_bytes: int = DEFAULT_MANIFEST_TARGET_SIZE_BYTES
    orphans: bool = False
    orphan_grace_hours: float = DEFAULT_ORPHAN_GRACE_HOURS
    dry_run: bool = False
    force: bool = False


def retain_last_type(value: str) -> int:
    """argparse type of --retain-last (an integer >= 1)."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}")
    if n < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return n


def hours_type(value: str) -> float:
    """argparse type of --older-than-hours / --orphan-grace-hours (a finite
    number >= 0)."""
    try:
        h = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not a number: {value!r}")
    if not math.isfinite(h) or h < 0:
        raise argparse.ArgumentTypeError("must be a finite number >= 0")
    return h


def target_size_type(value: str) -> int:
    """argparse type of --manifest-target-size-bytes (an integer >= 1 MiB)."""
    try:
        n = int(value)
    except ValueError:
        raise argparse.ArgumentTypeError(f"not an integer: {value!r}")
    if n < MIN_MANIFEST_TARGET_SIZE_BYTES:
        raise argparse.ArgumentTypeError(
            f"must be >= {MIN_MANIFEST_TARGET_SIZE_BYTES} (1 MiB)"
        )
    return n


# ── the MOSAIC_TEST_FAIL_MAINTAIN hook ───────────────────────────────────────


def parse_fail_maintain(spec: Optional[str]) -> Optional[Tuple[str, str]]:
    """Parse ``MOSAIC_TEST_FAIL_MAINTAIN="<iceberg_table>:<phase>"`` (phase
    ``pre_expire`` or ``post_expire``, the table one of ICEBERG_TABLES) into
    (table, phase); None when unset. A malformed spec raises ValueError
    rather than silently doing nothing (a test relying on it would otherwise
    pass vacuously)."""
    if not spec:
        return None
    table, sep, phase = spec.rpartition(":")
    if not sep or table not in ICEBERG_TABLES or phase not in _HOOK_PHASES:
        raise ValueError(
            f"{_HOOK_ENV}={spec!r}: expected "
            "<iceberg_table>:<pre_expire|post_expire> with the table one of "
            f"{', '.join(ICEBERG_TABLES)}"
        )
    return table, phase


def _maybe_fail_maintain(
    hook: Optional[Tuple[str, str]], table: str, phase: str
) -> None:
    """Test-only fault injection, inert unless MOSAIC_TEST_FAIL_MAINTAIN is
    set: raise just before the table's expire commit (``pre_expire``:
    nothing expired, nothing deleted) or just after it (``post_expire``: the
    expired snapshots are gone from the metadata, their files still in S3)."""
    if hook == (table, phase):
        raise RuntimeError(f"{_HOOK_ENV}: injected crash {table} @{phase}")


# ── retention policy ─────────────────────────────────────────────────────────


@dataclass(frozen=True)
class SnapshotStamp:
    """A snapshot's id and commit time (``timestamp_ms``)."""

    snapshot_id: int
    timestamp_ms: int


def plan_retention(
    snapshots: Sequence[SnapshotStamp],
    current_id: Optional[int],
    protected: Iterable[int],
    retain_last: int,
    older_than_hours: float,
    now_ms: int,
) -> Tuple[Set[int], Set[int]]:
    """(retained, expired) snapshot ids. A snapshot is retained when it is
    the current one, one of the ``retain_last`` newest by ``timestamp_ms``
    (ties: the later in the metadata's commit order is newer), younger than
    ``older_than_hours`` (0: none is), or protected (a branch head or tag);
    every other snapshot is expired."""
    if retain_last < 1:
        raise ValueError(f"retain_last must be >= 1: {retain_last}")
    if not math.isfinite(older_than_hours) or older_than_hours < 0:
        raise ValueError(f"older_than_hours must be >= 0: {older_than_hours}")
    ids = {s.snapshot_id for s in snapshots}
    retained: Set[int] = set()
    if current_id is not None and current_id in ids:
        retained.add(current_id)
    newest = sorted(
        range(len(snapshots)),
        key=lambda i: (snapshots[i].timestamp_ms, i),
        reverse=True,
    )
    retained.update(snapshots[i].snapshot_id for i in newest[:retain_last])
    max_age_ms = older_than_hours * 3_600_000
    retained.update(
        s.snapshot_id
        for s in snapshots
        if now_ms - s.timestamp_ms < max_age_ms
    )
    retained.update(p for p in protected if p in ids)
    return retained, ids - retained


# ── the reclaim set ──────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FileRef:
    """One manifest entry: the data (or delete) file it names, its size and
    whether the entry is live (ADDED / EXISTING) or DELETED."""

    path: str
    size: int
    live: bool


@dataclass
class SnapshotGraph:
    """A table's snapshot -> manifest list -> manifests -> entries graph."""

    lists: Dict[int, str] = field(default_factory=dict)
    manifests: Dict[int, List[str]] = field(default_factory=dict)
    entries: Dict[str, List[FileRef]] = field(default_factory=dict)
    manifest_sizes: Dict[str, int] = field(default_factory=dict)


def compute_reclaim(
    graph: SnapshotGraph, retained: Set[int], expired: Set[int]
) -> Dict[str, str]:
    """The files to delete once the ``expired`` snapshots are gone, as
    {path: kind} (kind in KINDS). With ``lists(S)``, ``manifests(S)``,
    ``named(S)`` (every file their entries name, any status) and ``live(S)``
    (ADDED / EXISTING entries only) over a snapshot set S:

        keep    = lists(R) | manifests(R) | live(R)
        reclaim = (lists(E) | manifests(E) | named(E)
                   | (named(R) - live(R))) - keep

    The ``named(R) - live(R)`` term departs from Iceberg's own expiry (which
    keeps a file until the snapshot that recorded its deletion expires): no
    MOSAIC reader time-travels, so a file only DELETED in retained snapshots
    (the staging the aggregate just pruned) goes now. Nothing ``R``
    references is ever in the set; an empty ``E`` gives an empty set."""
    if not expired:
        return {}

    def lists(snaps: Set[int]) -> Set[str]:
        return {graph.lists[s] for s in snaps}

    def manifests(snaps: Set[int]) -> Set[str]:
        return {m for s in snaps for m in graph.manifests.get(s, [])}

    def named(mans: Set[str], live_only: bool = False) -> Set[str]:
        return {
            ref.path
            for m in mans
            for ref in graph.entries.get(m, [])
            if ref.live or not live_only
        }

    m_retained = manifests(retained)
    m_expired = manifests(expired)
    live_retained = named(m_retained, live_only=True)
    keep = lists(retained) | m_retained | live_retained
    out: Dict[str, str] = {}
    for path in lists(expired) - keep:
        out[path] = "lists"
    for path in m_expired - keep:
        out.setdefault(path, "manifests")
    data = named(m_expired) | (named(m_retained) - live_retained)
    for path in data - keep:
        out.setdefault(path, "data")
    return out


def _read_entries(io: Any, manifest: Any) -> List[FileRef]:
    return [
        FileRef(
            entry.data_file.file_path,
            int(entry.data_file.file_size_in_bytes or 0),
            entry.status != ManifestEntryStatus.DELETED,
        )
        for entry in manifest.fetch_manifest_entry(io, discard_deleted=False)
    ]


def load_graph(
    table: Any, cache: Dict[str, List[FileRef]], pool: ThreadPoolExecutor
) -> SnapshotGraph:
    """Read every snapshot's manifest list once and every distinct manifest
    once (manifests are shared across snapshots: a fast append carries every
    existing manifest forward; ``cache`` also shares them across calls, which
    is safe because a manifest file is immutable)."""
    io = table.io
    snapshots = list(table.metadata.snapshots)
    per_snapshot = list(pool.map(lambda s: s.manifests(io), snapshots))
    graph = SnapshotGraph()
    unique: Dict[str, Any] = {}
    for snap, mans in zip(snapshots, per_snapshot):
        graph.lists[snap.snapshot_id] = snap.manifest_list
        graph.manifests[snap.snapshot_id] = [m.manifest_path for m in mans]
        for m in mans:
            unique.setdefault(m.manifest_path, m)
    todo = [m for path, m in unique.items() if path not in cache]
    for m, refs in zip(todo, pool.map(lambda m: _read_entries(io, m), todo)):
        cache[m.manifest_path] = refs
    for path, m in unique.items():
        graph.entries[path] = cache[path]
        graph.manifest_sizes[path] = int(m.manifest_length or 0)
    return graph


# ── S3 ───────────────────────────────────────────────────────────────────────


def parse_s3_uri(uri: str) -> ObjectKey:
    """``s3://bucket/key`` (or s3a / s3n) -> (bucket, key). Anything else
    raises ValueError."""
    scheme, sep, rest = uri.partition("://")
    bucket, _, key = rest.partition("/")
    if not sep or scheme not in _S3_SCHEMES or not bucket or not key:
        raise ValueError(f"not an s3://bucket/key URI: {uri!r}")
    return bucket, key


def _try_parse_s3_uri(uri: str) -> Optional[ObjectKey]:
    try:
        return parse_s3_uri(uri)
    except ValueError:
        return None


def group_by_bucket(uris: Iterable[str]) -> Dict[str, List[str]]:
    """{bucket: sorted distinct keys} of ``uris`` (each parsed with
    parse_s3_uri: the metadata bucket can differ from the data bucket)."""
    out: Dict[str, Set[str]] = {}
    for uri in uris:
        bucket, key = parse_s3_uri(uri)
        out.setdefault(bucket, set()).add(key)
    return {b: sorted(keys) for b, keys in sorted(out.items())}


def chunked(items: Sequence[Any], size: int = DELETE_CHUNK) -> List[List[Any]]:
    """Consecutive chunks of at most ``size`` items, order kept."""
    if size < 1:
        raise ValueError(f"chunk size must be >= 1: {size}")
    return [list(items[i : i + size]) for i in range(0, len(items), size)]


def delete_uris(s3: Any, uris: Iterable[str]) -> int:
    """Delete ``uris`` (grouped by bucket, ``delete_objects`` in chunks of
    DELETE_CHUNK). A key that no longer exists is not an error; any other
    per-key error raises. Returns the keys requested."""
    n = 0
    for bucket, keys in group_by_bucket(uris).items():
        for chunk in chunked(keys):
            resp = s3.delete_objects(
                Bucket=bucket,
                Delete={
                    "Objects": [{"Key": k} for k in chunk],
                    "Quiet": True,
                },
            )
            errors = [
                e
                for e in resp.get("Errors") or []
                if e.get("Code") != "NoSuchKey"
            ]
            if errors:
                raise RuntimeError(
                    f"S3 delete_objects on bucket {bucket}: {len(errors)} "
                    f"error(s), first: {errors[0]}"
                )
            n += len(chunk)
    return n


def list_prefix(
    s3: Any, bucket: str, prefix: str
) -> Dict[ObjectKey, Tuple[datetime, int]]:
    """{(bucket, key): (LastModified, Size)} of every object under
    ``s3://bucket/prefix``."""
    out: Dict[ObjectKey, Tuple[datetime, int]] = {}
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            out[(bucket, obj["Key"])] = (obj["LastModified"], obj["Size"])
    return out


def _dir_prefix(uri: str) -> ObjectKey:
    """(bucket, key prefix ending in '/') of a table location URI."""
    bucket, key = parse_s3_uri(uri.rstrip("/") + "/")
    return bucket, key


# ── per table ────────────────────────────────────────────────────────────────


@dataclass
class TableResult:
    name: str
    present: bool = False
    properties_set: bool = False
    rewrite: bool = False
    manifests_before: int = 0
    manifests_after: int = 0
    snapshots_before: int = 0
    snapshots_after: int = 0
    expired: int = 0
    reclaim_files: Dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in KINDS}
    )
    reclaim_bytes: Dict[str, int] = field(
        default_factory=lambda: {k: 0 for k in KINDS}
    )
    rewrite_secs: float = 0.0
    expire_secs: float = 0.0
    secs: float = 0.0
    table: Any = None

    @property
    def nothing_to_do(self) -> bool:
        return not (self.properties_set or self.rewrite or self.expired)


def _rewrite_manifests(table: Any, target_size_bytes: int) -> None:
    """Merge the current snapshot's data manifests: a merge-append of no
    files in one transaction, with the merge properties staged before it and
    restored after it (set back to the table's values, removed where the
    table had none), so the committed table keeps its own properties.
    _MergeAppendFiles reads them from the transaction's staged metadata
    (verified on pyiceberg 0.11.0; the check below fails loudly if a later
    version stops doing so, instead of committing an unmerged snapshot)."""
    before = {k: table.properties.get(k) for k in _MERGE_PROPERTIES}
    txn = table.transaction()
    txn.set_properties(
        {
            TableProperties.MANIFEST_MERGE_ENABLED: "true",
            TableProperties.MANIFEST_MIN_MERGE_COUNT: "2",
            TableProperties.MANIFEST_TARGET_SIZE_BYTES: str(target_size_bytes),
        }
    )
    with txn.update_snapshot().merge_append() as producer:
        # No files: the manifests are merged, nothing else changes.
        if not (
            producer._merge_enabled
            and producer._min_count_to_merge == 2
            and producer._target_size_bytes == target_size_bytes
        ):
            raise RuntimeError(
                "pyiceberg's merge-append did not pick up the staged "
                "manifest-merge properties; the manifest rewrite needs "
                "pyiceberg 0.11.0's behavior"
            )
    restore = {k: v for k, v in before.items() if v is not None}
    if restore:
        txn.set_properties(restore)
    absent = [k for k, v in before.items() if v is None]
    if absent:
        txn.remove_properties(*absent)
    txn.commit_transaction()


def _expire(table: Any, expired: Iterable[int]) -> None:
    """Remove the ``expired`` snapshots in one commit. Staged as one
    single-id remove-snapshots update per snapshot on one transaction rather
    than ``table.maintenance.expire_snapshots().by_ids(...)``: the local REST
    server (tabulario/iceberg-rest) rejects a remove-snapshots update with
    more than one id ("Expected one value"), which by_ids would send. The
    server, and pyiceberg's own metadata update (Glue), also drop the removed
    snapshots' snapshot-log entries. Metadata only: no file is deleted."""
    txn = table.transaction()
    for snapshot_id in sorted(expired):
        ExpireSnapshots(transaction=txn).by_id(snapshot_id).commit()
    txn.commit_transaction()


def _reclaim_sizes(
    s3: Any, table: Any, graph: SnapshotGraph, plan: Dict[str, str]
) -> Dict[str, int]:
    """Byte size of each reclaim path: data files and manifests from the
    metadata, manifest lists from one listing of the table's metadata
    directory (the metadata records no list size; a list already gone counts
    0 bytes)."""
    sizes: Dict[str, int] = {}
    data_sizes = {
        ref.path: ref.size for refs in graph.entries.values() for ref in refs
    }
    lists = []
    for path, kind in plan.items():
        if kind == "manifests":
            sizes[path] = graph.manifest_sizes.get(path, 0)
        elif kind == "data":
            sizes[path] = data_sizes.get(path, 0)
        else:
            lists.append(path)
    if lists:
        meta_dir = table.metadata_location.rsplit("/", 1)[0]
        listing = list_prefix(s3, *_dir_prefix(meta_dir))
        for path in lists:
            parsed = _try_parse_s3_uri(path)
            sizes[path] = listing.get(parsed, (None, 0))[1] if parsed else 0
    return sizes


def _fmt_reclaim(res: TableResult, dry_run: bool) -> str:
    verb = "would reclaim" if dry_run else "reclaimed"
    files = res.reclaim_files
    return (
        f"{verb} {sum(files.values())} files, "
        f"{sum(res.reclaim_bytes.values())} bytes (data {files['data']}, "
        f"manifests {files['manifests']}, lists {files['lists']})"
    )


def maintain_table(
    catalog: Any,
    name: str,
    opts: MaintainOptions,
    s3: Any,
    hook: Optional[Tuple[str, str]],
    cache: Dict[str, List[FileRef]],
    pool: ThreadPoolExecutor,
) -> TableResult:
    """Steps 1-3 (properties, manifest rewrite, expiry + reclaim) for one
    table; prints its line. A table absent from the catalog is skipped."""
    t_start = time.perf_counter()
    res = TableResult(name)
    try:
        table = catalog.load_table(f"{ICEBERG_NAMESPACE}.{name}")
    except NoSuchTableError:
        print(f"  {name}: not found; skipped", file=sys.stderr)
        return res
    res.present = True
    res.table = table

    # 1. Properties.
    missing = {
        k: v
        for k, v in maintained_properties(name).items()
        if table.properties.get(k) != v
    }
    if missing:
        res.properties_set = True
        if not opts.dry_run:
            table.transaction().set_properties(missing).commit_transaction()

    # 2. Manifest rewrite, before the expiry so that the pre-rewrite snapshot
    # (and its small manifests) can be expired in the same run.
    res.snapshots_before = len(table.metadata.snapshots)
    snap = table.current_snapshot()
    current = snap.manifests(table.io) if snap is not None else []
    res.manifests_before = res.manifests_after = len(current)
    n_data = sum(1 for m in current if m.content == ManifestContent.DATA)
    res.rewrite = opts.rewrite_manifests and n_data >= 2
    if res.rewrite and not opts.dry_run:
        t0 = time.perf_counter()
        _rewrite_manifests(table, opts.manifest_target_size_bytes)
        res.rewrite_secs = time.perf_counter() - t0
        res.manifests_after = len(table.current_snapshot().manifests(table.io))

    # 3. Expiry and reclaim.
    t0 = time.perf_counter()
    meta = table.metadata
    retained, expired = plan_retention(
        [SnapshotStamp(s.snapshot_id, s.timestamp_ms) for s in meta.snapshots],
        meta.current_snapshot_id,
        [ref.snapshot_id for ref in meta.refs.values()],
        opts.retain_last,
        opts.older_than_hours,
        int(time.time() * 1000),
    )
    res.expired = len(expired)
    if expired:
        graph = load_graph(table, cache, pool)
        plan = compute_reclaim(graph, retained, expired)
        sizes = _reclaim_sizes(s3, table, graph, plan)
        for path, kind in plan.items():
            res.reclaim_files[kind] += 1
            res.reclaim_bytes[kind] += sizes[path]
        if not opts.dry_run:
            _maybe_fail_maintain(hook, name, "pre_expire")
            _expire(table, expired)
            _maybe_fail_maintain(hook, name, "post_expire")
            delete_uris(s3, plan)
            table.refresh()
    res.expire_secs = time.perf_counter() - t0
    res.snapshots_after = (
        len(table.metadata.snapshots) - len(expired)
        if opts.dry_run
        else len(table.metadata.snapshots)
    )
    res.secs = time.perf_counter() - t_start

    if res.nothing_to_do:
        print(f"  {name}: nothing to do", file=sys.stderr)
        return res
    if opts.dry_run:
        props = "would be set" if res.properties_set else "ok"
        manifests = f"{res.manifests_before}" + (
            " (would rewrite)" if res.rewrite else ""
        )
    else:
        props = "set" if res.properties_set else "ok"
        manifests = f"{res.manifests_before} -> {res.manifests_after}"
    print(
        f"  {name}: properties {props}; manifests {manifests}; snapshots "
        f"{res.snapshots_before} -> {res.snapshots_after}; "
        f"{_fmt_reclaim(res, opts.dry_run)}; {res.secs:.2f}s",
        file=sys.stderr,
    )
    return res


# ── orphan sweep ─────────────────────────────────────────────────────────────


ORPHAN_CATEGORIES = ("outside", "referenced", "pending", "young", "orphan")


def classify_objects(
    objects: Mapping[ObjectKey, Tuple[datetime, int]],
    prefixes: Sequence[ObjectKey],
    referenced: Set[ObjectKey],
    pending: Set[ObjectKey],
    now: datetime,
    grace_hours: float,
) -> Dict[str, List[ObjectKey]]:
    """Sort listed objects into ORPHAN_CATEGORIES, first match wins:
    ``outside`` the swept (bucket, prefix) pairs, ``referenced`` by a
    snapshot or table metadata, ``pending`` (an unregistered ingest_files
    URI), ``young`` (LastModified within ``grace_hours`` of ``now``: the
    window between an upload and the commit that references it); the rest
    are ``orphan`` (to delete). Each list is sorted."""
    grace = timedelta(hours=grace_hours)
    out: Dict[str, List[ObjectKey]] = {c: [] for c in ORPHAN_CATEGORIES}
    for obj in sorted(objects):
        bucket, key = obj
        if not any(bucket == b and key.startswith(p) for b, p in prefixes):
            out["outside"].append(obj)
        elif obj in referenced:
            out["referenced"].append(obj)
        elif obj in pending:
            out["pending"].append(obj)
        elif now - objects[obj][0] < grace:
            out["young"].append(obj)
        else:
            out["orphan"].append(obj)
    return out


def all_namespaces(catalog: Any) -> List[Tuple[str, ...]]:
    """Every namespace of the catalog, nested ones included."""
    seen: Set[Tuple[str, ...]] = set()
    stack = [tuple(ns) for ns in catalog.list_namespaces()]
    while stack:
        ns = stack.pop()
        if ns in seen:
            continue
        seen.add(ns)
        for child in catalog.list_namespaces(ns):
            child = tuple(child)
            if len(child) > len(ns) and child[: len(ns)] == ns:
                stack.append(child)
    return sorted(seen)


def table_references(
    table: Any, cache: Dict[str, List[FileRef]], pool: ThreadPoolExecutor
) -> Set[ObjectKey]:
    """Every S3 object a table's metadata references: its current and logged
    metadata.json versions, statistics files, and for every snapshot its
    manifest list, manifests and every file they name (any status)."""
    meta = table.metadata
    paths = {table.metadata_location}
    paths.update(e.metadata_file for e in meta.metadata_log)
    paths.update(s.statistics_path for s in meta.statistics)
    paths.update(s.statistics_path for s in meta.partition_statistics)
    graph = load_graph(table, cache, pool)
    paths.update(graph.lists.values())
    for mans in graph.manifests.values():
        paths.update(mans)
    for refs in graph.entries.values():
        paths.update(ref.path for ref in refs)
    return {p for p in map(_try_parse_s3_uri, paths) if p is not None}


def _pending_uris(conn: Any) -> Set[ObjectKey]:
    """The unregistered ingest_files URIs (none before deferred
    registration existed)."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('ingest_files') IS NOT NULL")
        if not cur.fetchone()[0]:
            conn.commit()
            return set()
        cur.execute("SELECT uri FROM ingest_files WHERE registered_at IS NULL")
        uris = [u for (u,) in cur.fetchall()]
    conn.commit()
    return {p for p in map(_try_parse_s3_uri, uris) if p is not None}


@dataclass
class SweepResult:
    listed: int = 0
    counts: Dict[str, int] = field(default_factory=dict)
    deleted: int = 0
    deleted_bytes: int = 0
    list_secs: float = 0.0
    reference_secs: float = 0.0
    secs: float = 0.0
    skipped: bool = False


def sweep_orphans(
    catalog: Any,
    conn: Any,
    s3: Any,
    tables: Mapping[str, Any],
    opts: MaintainOptions,
    cache: Dict[str, List[FileRef]],
    pool: ThreadPoolExecutor,
) -> SweepResult:
    """Step 4. Candidates are the objects under the upload prefixes
    ``s3://<MOSAIC_S3_BUCKET>/<table>/`` of every ICEBERG_TABLES name and
    under the location of every table of the namespace (``tables``). The
    listing runs first, then the pending rows and the references are read,
    so an object a commit references by then is seen as referenced. The
    references come from every table in every namespace of the catalog (the
    upload prefixes are not namespaced); if any table cannot be read, the
    sweep is skipped."""
    t_start = time.perf_counter()
    res = SweepResult()
    prefixes: List[ObjectKey] = [(S3_BUCKET, f"{t}/") for t in ICEBERG_TABLES]
    for table in tables.values():
        prefixes.append(_dir_prefix(table.location()))
    prefixes = sorted(set(prefixes))

    t0 = time.perf_counter()
    objects: Dict[ObjectKey, Tuple[datetime, int]] = {}
    for bucket, prefix in prefixes:
        objects.update(list_prefix(s3, bucket, prefix))
    res.list_secs = time.perf_counter() - t0
    res.listed = len(objects)
    pending = _pending_uris(conn)

    t0 = time.perf_counter()
    referenced: Set[ObjectKey] = set()
    current = "the namespace list"
    try:
        for ns in all_namespaces(catalog):
            current = ".".join(ns)
            for ident in catalog.list_tables(ns):
                current = ".".join(ident)
                referenced |= table_references(
                    catalog.load_table(ident), cache, pool
                )
    except Exception as e:
        res.skipped = True
        res.secs = time.perf_counter() - t_start
        print(
            f"  orphans: WARNING: could not read {current} ({e}); sweep "
            "skipped, nothing deleted",
            file=sys.stderr,
        )
        return res
    res.reference_secs = time.perf_counter() - t0

    classes = classify_objects(
        objects,
        prefixes,
        referenced,
        pending,
        datetime.now(timezone.utc),
        opts.orphan_grace_hours,
    )
    res.counts = {c: len(v) for c, v in classes.items()}
    orphans = classes["orphan"]
    res.deleted = len(orphans)
    res.deleted_bytes = sum(objects[o][1] for o in orphans)
    uris = [f"s3://{b}/{k}" for b, k in orphans]
    if opts.dry_run:
        for uri in uris[:ORPHAN_LIST_LIMIT]:
            print(f"    would delete {uri}", file=sys.stderr)
        if len(uris) > ORPHAN_LIST_LIMIT:
            print(
                f"    ... and {len(uris) - ORPHAN_LIST_LIMIT} more",
                file=sys.stderr,
            )
    else:
        delete_uris(s3, uris)
    res.secs = time.perf_counter() - t_start
    verb = "would delete" if opts.dry_run else "deleted"
    print(
        f"  orphans: listed {res.listed}, referenced "
        f"{res.counts['referenced']}, pending {res.counts['pending']}, young "
        f"{res.counts['young']}, {verb} {res.deleted} ({res.deleted_bytes} "
        f"bytes); {res.secs:.2f}s (listing {res.list_secs:.2f}s, references "
        f"{res.reference_secs:.2f}s)",
        file=sys.stderr,
    )
    return res


# ── the command ──────────────────────────────────────────────────────────────


def get_db_connection() -> Any:
    return psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]


def _in_progress_runs(conn: Any) -> List[int]:
    """agg_run_ids of the ``in_progress`` aggregation runs."""
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('aggregation_progress') IS NOT NULL")
        if not cur.fetchone()[0]:
            conn.commit()
            return []
        cur.execute(
            "SELECT agg_run_id FROM aggregation_progress "
            "WHERE status = 'in_progress' ORDER BY agg_run_id"
        )
        runs = [int(r) for (r,) in cur.fetchall()]
    conn.commit()
    return runs


def _print_total(
    results: Sequence[TableResult],
    sweep: Optional[SweepResult],
    opts: MaintainOptions,
    secs: float,
) -> None:
    present = [r for r in results if r.present]
    files = {k: sum(r.reclaim_files[k] for r in present) for k in KINDS}
    nbytes = {k: sum(r.reclaim_bytes[k] for r in present) for k in KINDS}
    verb = "would reclaim" if opts.dry_run else "reclaimed"
    split = ", ".join(f"{k} {files[k]} files {nbytes[k]} bytes" for k in KINDS)
    sweep_s = sweep.secs if sweep is not None else 0.0
    print(
        f"total: {len(present)} table(s); snapshots "
        f"{sum(r.snapshots_before for r in present)} -> "
        f"{sum(r.snapshots_after for r in present)}; manifests "
        f"{sum(r.manifests_before for r in present)} -> "
        f"{sum(r.manifests_after for r in present)}; {verb} "
        f"{sum(files.values())} files, {sum(nbytes.values())} bytes "
        f"({split}); rewrite {sum(r.rewrite_secs for r in present):.2f}s, "
        f"expire+reclaim {sum(r.expire_secs for r in present):.2f}s, sweep "
        f"{sweep_s:.2f}s; {secs:.2f}s",
        file=sys.stderr,
    )


def run_maintain(opts: MaintainOptions) -> int:
    """Maintain every table of ICEBERG_TABLES (then sweep, with
    ``opts.orphans``). Returns the exit code: 0 on success (including nothing
    to do), 1 when the lock is held or an aggregation run is in progress
    (without ``opts.force``). Raises on any other failure; a re-run resumes
    (expiry is computed afresh from the metadata, and a crash between an
    expire commit and its deletes leaves files ``--orphans`` reclaims)."""
    if opts.retain_last < 1:
        raise ValueError(f"retain_last must be >= 1: {opts.retain_last}")
    if opts.manifest_target_size_bytes < MIN_MANIFEST_TARGET_SIZE_BYTES:
        raise ValueError(
            "manifest_target_size_bytes must be >= "
            f"{MIN_MANIFEST_TARGET_SIZE_BYTES}"
        )
    hook = parse_fail_maintain(os.environ.get(_HOOK_ENV))
    t_start = time.perf_counter()
    conn = get_db_connection()
    locked = False
    try:
        if not opts.dry_run:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT pg_try_advisory_lock(%s)", (REGISTER_LOCK_KEY,)
                )
                locked = cur.fetchone()[0]
            conn.commit()
            if not locked:
                print(
                    "Error: another `mosaic register` or `mosaic maintain` "
                    "holds the Iceberg metadata-writer lock (pg advisory "
                    f"lock {REGISTER_LOCK_KEY:#x}). Nothing done.",
                    file=sys.stderr,
                )
                return 1
            runs = _in_progress_runs(conn)
            if runs and not opts.force:
                print(
                    f"Error: aggregation run(s) {runs} in progress "
                    "(aggregation_progress.status = 'in_progress'): a "
                    "running `mosaic aggregate` commits to the staging "
                    "tables. Re-run when it has finished, or with --force "
                    "if it is dead (a killed aggregate stays in_progress). "
                    "Nothing done.",
                    file=sys.stderr,
                )
                return 1
            if runs:
                print(
                    f"WARNING: aggregation run(s) {runs} in progress; "
                    "continuing (--force)",
                    file=sys.stderr,
                )

        print(
            f"mosaic maintain: namespace {ICEBERG_NAMESPACE}, "
            f"{len(ICEBERG_TABLES)} tables; retain_last={opts.retain_last} "
            f"older_than_hours={opts.older_than_hours:g} rewrite_manifests="
            f"{opts.rewrite_manifests} target_size_bytes="
            f"{opts.manifest_target_size_bytes} orphans={opts.orphans}"
            + (
                f" orphan_grace_hours={opts.orphan_grace_hours:g}"
                if opts.orphans
                else ""
            )
            + (
                " DRY RUN (nothing committed, nothing deleted)"
                if opts.dry_run
                else ""
            ),
            file=sys.stderr,
        )
        catalog = load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)
        s3 = boto3.client(  # type: ignore[call-overload]
            "s3", **build_s3_client_kwargs()
        )
        cache: Dict[str, List[FileRef]] = {}
        results: List[TableResult] = []
        sweep: Optional[SweepResult] = None
        with ThreadPoolExecutor(max_workers=_READ_THREADS) as pool:
            for name in ICEBERG_TABLES:
                results.append(
                    maintain_table(catalog, name, opts, s3, hook, cache, pool)
                )
            if opts.orphans:
                tables = {r.name: r.table for r in results if r.present}
                sweep = sweep_orphans(
                    catalog, conn, s3, tables, opts, cache, pool
                )
        _print_total(results, sweep, opts, time.perf_counter() - t_start)
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
        "--retain-last",
        type=retain_last_type,
        default=DEFAULT_RETAIN_LAST,
        dest="retain_last",
        help="Retain each table's N newest snapshots (default "
        f"{DEFAULT_RETAIN_LAST}, min 1). The current snapshot, branch heads "
        "and tags are always retained.",
    )
    parser.add_argument(
        "--older-than-hours",
        type=hours_type,
        default=DEFAULT_OLDER_THAN_HOURS,
        dest="older_than_hours",
        help="Also retain every snapshot younger than this (default "
        f"{DEFAULT_OLDER_THAN_HOURS:g}, Iceberg's five days). 0 retains by "
        "count only: safe when no reader is running (the workflow passes 0); "
        "the default protects a reader that planned its scans from an older "
        "snapshot.",
    )
    parser.add_argument(
        "--no-rewrite-manifests",
        action="store_false",
        dest="rewrite_manifests",
        help="Skip the manifest rewrite (merging each table's current data "
        "manifests).",
    )
    parser.add_argument(
        "--manifest-target-size-bytes",
        type=target_size_type,
        default=DEFAULT_MANIFEST_TARGET_SIZE_BYTES,
        dest="manifest_target_size_bytes",
        help="Target size of a merged manifest (default "
        f"{DEFAULT_MANIFEST_TARGET_SIZE_BYTES}, min "
        f"{MIN_MANIFEST_TARGET_SIZE_BYTES}).",
    )
    parser.add_argument(
        "--orphans",
        action="store_true",
        help="Also delete the objects under the upload prefixes "
        "(s3://<MOSAIC_S3_BUCKET>/<table>/) and the table locations that no "
        "snapshot of any table in any namespace of the catalog references, "
        "that are not pending in ingest_files and that are older than "
        "--orphan-grace-hours. Assumes those prefixes belong to this "
        "catalog.",
    )
    parser.add_argument(
        "--orphan-grace-hours",
        type=hours_type,
        default=DEFAULT_ORPHAN_GRACE_HOURS,
        dest="orphan_grace_hours",
        help="With --orphans, keep every object modified within this many "
        f"hours (default {DEFAULT_ORPHAN_GRACE_HOURS:g}): an upload not yet "
        "committed.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        dest="dry_run",
        help="Commit nothing and delete nothing; report what would be done "
        "(the manifest rewrite is not simulated). Takes neither the lock "
        "nor the in-progress-aggregation guard.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Run although aggregation_progress has an in_progress run (one "
        "that was killed).",
    )


def options_from_args(args: Any) -> MaintainOptions:
    return MaintainOptions(
        retain_last=args.retain_last,
        older_than_hours=args.older_than_hours,
        rewrite_manifests=args.rewrite_manifests,
        manifest_target_size_bytes=args.manifest_target_size_bytes,
        orphans=args.orphans,
        orphan_grace_hours=args.orphan_grace_hours,
        dry_run=args.dry_run,
        force=args.force,
    )


def run(args: Any) -> None:
    try:
        rc = run_maintain(options_from_args(args))
    except Exception as e:
        traceback.print_exc()
        print(f"Error during maintenance: {e}", file=sys.stderr)
        sys.exit(1)
    if rc:
        sys.exit(rc)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Iceberg maintenance: set the table properties, rewrite the "
            "manifests, expire snapshots and delete the files only they "
            "needed, and optionally sweep orphan files. Run after `mosaic "
            "aggregate`."
        )
    )
    configure_parser(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
