# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Shared test harness for the MOSAIC pipeline.

Provides:
  * Isolation — forces a dedicated PostgreSQL database (``jc_test``), Iceberg
    namespace (``jc_test``) and S3/MinIO bucket (``jc-test-warehouse``) via the
    ``MOSAIC_*`` env vars, so the suite never touches the developer's own
    pipeline data. The data-file S3 prefixes are NOT namespaced (``clean_db.py``
    deletes ``sample_ref_blocks/`` etc. wholesale), which is why a separate bucket
    — not just a separate namespace — is required for true isolation.
  * ``reset_state()`` — a fast surgical reset (drop PG tables + drop the test
    Iceberg tables + wipe the test bucket) that does NOT bounce Docker, so the
    dev stack and any open connections stay alive (``clean_db.py`` would
    ``docker-compose down -v``, wiping everything).
  * Subprocess wrappers (``run_init`` / ``run_ingest`` / ``run_register`` /
    ``run_aggregate`` / ``run_maintain`` / ``run_joint_call``) that thread the isolated env + optional fault-injection
    env into each pipeline script, exactly as the real per-chromosome fan-out does.
  * A snapshot/compare oracle. The clean run is the truth: PG aggregates are
    compared keyed by genomic coordinate + allele string (the derived
    ``alleles`` catalog has no id column), and per-sample Iceberg data is
    compared by row count + the decoded joint-called VCF.
"""

import os

# ── Forced isolation — MUST run before `config` is imported (it reads env at
#    import time). Direct assignment (not setdefault) so tests can never be
#    pointed at the dev stores by a stray env var. The MinIO endpoint/creds keep
#    their dev defaults but stay overridable for non-localhost setups.
os.environ["MOSAIC_DB_NAME"] = "jc_test"
os.environ["MOSAIC_ICEBERG_NAMESPACE"] = "jc_test"
os.environ["MOSAIC_S3_BUCKET"] = "jc-test-warehouse"
os.environ.setdefault("MOSAIC_S3_ENDPOINT", "http://localhost:9000")

import datetime  # noqa: E402
import gzip  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
import socket  # noqa: E402
import struct  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from typing import (  # noqa: E402
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
)

import boto3  # type: ignore[import-untyped]  # noqa: E402
import numpy as np  # noqa: E402
import psycopg2  # noqa: E402
from pyiceberg.catalog import load_catalog  # noqa: E402
from pyiceberg.expressions import In  # noqa: E402

from mosaic.config import (  # noqa: E402
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
)
from mosaic.init_db import GTC_FIXED_BITS  # noqa: E402
from mosaic.init_db import ICEBERG_TABLES as _ICEBERG_TABLES  # noqa: E402

# ── Paths / constants ────────────────────────────────────────────────────────
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TEST_DB_NAME = "jc_test"
TEST_NAMESPACE = "jc_test"
TEST_BUCKET = "jc-test-warehouse"
# Bucket where the REST catalog stores table metadata (its configured warehouse).
CATALOG_WAREHOUSE_BUCKET = "warehouse"

# The two real chr21 test samples (chr21:1-30M), reused everywhere.
GVCFS = [
    os.path.join(
        REPO_ROOT,
        "test_data/input_gvcfs4/"
        "HG00245.final.hard-filtered_chr21_1_30000000_no-ambig.gvcf.gz",
    ),
    os.path.join(
        REPO_ROOT,
        "test_data/input_gvcfs4/"
        "HG00420.final.hard-filtered_chr21_1_30000000_no-ambig.gvcf.gz",
    ),
]

# The 13 pipeline Iceberg tables (init_db.ICEBERG_TABLES, which `mosaic
# init`, `mosaic clean` and `mosaic maintain` use).
ICEBERG_TABLES = list(_ICEBERG_TABLES)

# ── low-level connections ────────────────────────────────────────────────────


def s3_client() -> Any:
    kwargs = {
        "region_name": os.environ.get("MOSAIC_S3_REGION", "us-east-1"),
        "endpoint_url": os.environ.get(
            "MOSAIC_S3_ENDPOINT", "http://localhost:9000"
        ),
        "aws_access_key_id": os.environ.get(
            "MOSAIC_S3_ACCESS_KEY_ID", "admin"
        ),
        "aws_secret_access_key": os.environ.get(
            "MOSAIC_S3_SECRET_ACCESS_KEY", "password"
        ),
    }
    return boto3.client("s3", **kwargs)  # type: ignore[call-overload]


def get_pg_conn() -> Any:
    """Connection to the isolated test database (jc_test).

    autocommit=True is essential: read helpers (snapshot_pg, pg_counts, …) open a
    transaction per SELECT, taking ACCESS SHARE locks. Left open (psycopg2's default
    is autocommit OFF), those locks block a pipeline subprocess's TRUNCATE (which
    needs ACCESS EXCLUSIVE) indefinitely — the subprocess waits on the lock while the
    test waits on the subprocess: a deadlock. autocommit releases each read's locks
    immediately.
    """
    conn = psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]
    conn.autocommit = True
    return conn


def _admin_pg_conn() -> Any:
    """Connection to the default `postgres` db (for CREATE DATABASE jc_test)."""
    cfg = dict(DB_CONFIG)
    cfg["dbname"] = "postgres"
    conn = psycopg2.connect(**cfg)  # type: ignore[call-overload]
    conn.autocommit = True
    return conn


def get_catalog() -> Any:
    return load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)


# ── infrastructure: stack health + one-time provisioning ─────────────────────


def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def stack_up() -> bool:
    """True iff Postgres (5432), Iceberg REST (8181) and MinIO (9000) answer."""
    return (
        _port_open("localhost", 5432)
        and _port_open("localhost", 8181)
        and _port_open("localhost", 9000)
    )


def ensure_test_infra() -> None:
    """Create the jc_test database and jc-test-warehouse bucket if missing."""
    conn = _admin_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM pg_database WHERE datname=%s", (TEST_DB_NAME,)
            )
            if not cur.fetchone():
                cur.execute(f"CREATE DATABASE {TEST_DB_NAME}")
    finally:
        conn.close()

    s3 = s3_client()
    names = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    if TEST_BUCKET not in names:
        s3.create_bucket(Bucket=TEST_BUCKET)


# ── surgical reset (no Docker bounce) ────────────────────────────────────────


def _wipe_bucket(s3: Any, bucket: str, prefix: str = "") -> int:
    deleted = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        objs = page.get("Contents", [])
        if objs:
            s3.delete_objects(
                Bucket=bucket,
                Delete={"Objects": [{"Key": o["Key"]} for o in objs]},
            )
            deleted += len(objs)
    return deleted


def reset_state() -> None:
    """Drop the test PG tables + Iceberg tables + wipe the test bucket.

    Leaves the Docker stack, the jc_test database and the jc-test-warehouse bucket
    in place — only their *contents* are cleared. A subsequent ``run_init`` then
    recreates every table fresh (PG init uses CREATE TABLE IF NOT EXISTS and
    init_iceberg_tables skips existing tables, so the drop is what guarantees a
    clean slate).
    """
    ensure_test_infra()

    # PostgreSQL: drop every table in the test DB (mirrors clean_db.py).
    conn = get_pg_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT tablename FROM pg_catalog.pg_tables "
                "WHERE schemaname NOT IN ('pg_catalog', 'information_schema')"
            )
            for (table_name,) in cur.fetchall():
                cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE")
        conn.commit()
    finally:
        conn.close()

    # Iceberg: drop the test-namespace tables (catalog metadata)...
    try:
        catalog = get_catalog()
        for t in ICEBERG_TABLES:
            try:
                catalog.drop_table(f"{TEST_NAMESPACE}.{t}")
            except Exception:
                pass
    except Exception:
        pass

    # ...then wipe the data-file bucket and the catalog metadata prefix. The
    # local REST catalog puts a table at s3://warehouse/<namespace>/<table>
    # (`<namespace>.db/` is the Hive-style layout of other catalogs); a
    # dropped table's metadata stays there, so without the wipe every
    # reset would leave the previous tables' files under the new tables'
    # locations (what `mosaic maintain --orphans` sweeps).
    s3 = s3_client()
    _wipe_bucket(s3, TEST_BUCKET)
    for prefix in (f"{TEST_NAMESPACE}/", f"{TEST_NAMESPACE}.db/"):
        try:
            _wipe_bucket(s3, CATALOG_WAREHOUSE_BUCKET, prefix=prefix)
        except Exception:
            pass


# ── subprocess wrappers ──────────────────────────────────────────────────────


# The Rust ingest engine's binary as `cargo build --release` leaves it.
RUST_WORKER_BUILD = os.path.join(
    REPO_ROOT, "rust", "target", "release", "mosaic-ingest-worker"
)
RUST_WORKER_SKIP_REASON = (
    "the Rust ingest worker is not built: cd rust && cargo build --release "
    "(or set MOSAIC_INGEST_WORKER)"
)


def rust_worker_path() -> Optional[str]:
    """$MOSAIC_INGEST_WORKER, else the release build if it exists, else
    mosaic-ingest-worker on PATH (None: none)."""
    path = os.environ.get("MOSAIC_INGEST_WORKER")
    if path:
        return path if os.access(path, os.X_OK) else None
    if os.access(RUST_WORKER_BUILD, os.X_OK):
        return RUST_WORKER_BUILD
    return shutil.which("mosaic-ingest-worker")


def rust_worker_available() -> bool:
    return rust_worker_path() is not None


def ingest_worker_env(env: Dict[str, str]) -> None:
    """``MOSAIC_INGEST_WORKER`` set in ``env`` (in place) to the release
    build when it exists and the variable is unset. ``mosaic ingest`` has
    one shard worker (the earlier Python worker is removed), so there is no
    fallback: without a binary
    the tests that run ingest skip (``require_ingest_worker``)."""
    if "MOSAIC_INGEST_WORKER" not in env and os.access(
        RUST_WORKER_BUILD, os.X_OK
    ):
        env["MOSAIC_INGEST_WORKER"] = RUST_WORKER_BUILD


def require_ingest_worker() -> None:
    """Skip the calling test with ``RUST_WORKER_SKIP_REASON`` when no
    ingest worker binary is available. Decided from the real environment
    (``rust_worker_path()``: ``$MOSAIC_INGEST_WORKER``, the release build,
    PATH), never from a test's ``extra_env``: a test may pass an unusable
    worker on purpose (``test_missing_binary_writes_nothing``)."""
    if rust_worker_path() is None:
        import pytest

        pytest.skip(RUST_WORKER_SKIP_REASON)


# The Rust joint caller as `cargo build --release` leaves it.
RUST_JOINT_CALL_BUILD = os.path.join(
    REPO_ROOT, "rust", "target", "release", "mosaic-joint-call"
)
RUST_JOINT_CALL_SKIP_REASON = (
    "the Rust joint caller is not built: cd rust && cargo build --release "
    "(or set MOSAIC_JOINT_CALL_WORKER)"
)


def joint_call_worker_path() -> Optional[str]:
    """$MOSAIC_JOINT_CALL_WORKER, else the release build if it exists, else
    mosaic-joint-call on PATH (None: none)."""
    path = os.environ.get("MOSAIC_JOINT_CALL_WORKER")
    if path:
        return path if os.access(path, os.X_OK) else None
    if os.access(RUST_JOINT_CALL_BUILD, os.X_OK):
        return RUST_JOINT_CALL_BUILD
    return shutil.which("mosaic-joint-call")


def joint_call_worker_available() -> bool:
    return joint_call_worker_path() is not None


def joint_call_worker_env(env: Dict[str, str]) -> None:
    """``MOSAIC_JOINT_CALL_WORKER`` set in ``env`` (in place) to the release
    build when it exists and the variable is unset. ``joint_call.py`` has
    one engine (the earlier Python engine is removed), so there is no
    fallback: without a binary the tests that run the joint caller skip
    (``require_joint_call_worker``)."""
    if "MOSAIC_JOINT_CALL_WORKER" not in env and os.access(
        RUST_JOINT_CALL_BUILD, os.X_OK
    ):
        env["MOSAIC_JOINT_CALL_WORKER"] = RUST_JOINT_CALL_BUILD


def require_joint_call_worker() -> None:
    """Skip the calling test with ``RUST_JOINT_CALL_SKIP_REASON`` when no
    joint-call binary is available. Decided from the real environment
    (``joint_call_worker_path()``), never from a test's ``extra_env``: a
    test may pass an unusable worker on purpose."""
    if joint_call_worker_path() is None:
        import pytest

        pytest.skip(RUST_JOINT_CALL_SKIP_REASON)


def test_env(extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """os.environ + PYTHONPATH(repo:vcflib) + forced MOSAIC_* isolation +
    the ingest worker binary (``ingest_worker_env``) + the joint-call binary
    (``joint_call_worker_env``) + extra."""
    env = os.environ.copy()
    ingest_worker_env(env)
    joint_call_worker_env(env)
    env["PYTHONPATH"] = os.pathsep.join(
        [REPO_ROOT, env.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)
    env["MOSAIC_DB_NAME"] = TEST_DB_NAME
    env["MOSAIC_ICEBERG_NAMESPACE"] = TEST_NAMESPACE
    env["MOSAIC_S3_BUCKET"] = TEST_BUCKET
    env.setdefault("MOSAIC_S3_ENDPOINT", "http://localhost:9000")
    if extra:
        env.update(extra)
    return env


def _run(
    args: List[str],
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
    timeout: Optional[float] = 600,
) -> subprocess.CompletedProcess:
    proc = subprocess.run(
        [sys.executable, *args],
        cwd=REPO_ROOT,
        env=test_env(extra_env),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    if check and proc.returncode != 0:
        raise AssertionError(
            f"command failed ({proc.returncode}): {args}\n"
            f"--- stdout ---\n{proc.stdout[-4000:]}\n"
            f"--- stderr ---\n{proc.stderr[-4000:]}"
        )
    return proc


def run_init(
    gvcf: str, chromosomes: str = "chr21", check: bool = True
) -> subprocess.CompletedProcess:
    return _run(
        ["-m", "mosaic.cli", "init", gvcf, "--chromosomes", chromosomes],
        check=check,
    )


def run_ingest(
    gvcfs: List[str],
    region: Optional[str] = None,
    shard_size: Optional[int] = None,
    threads: int = 2,
    pos_batch_size: Optional[int] = None,
    aggregate_parquet_size: Optional[int] = None,
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
    defer_register: bool = False,
    ref_gq_bands: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """``mosaic ingest``; skips the calling test when no ingest worker
    binary is available (``require_ingest_worker``)."""
    require_ingest_worker()
    args = ["-m", "mosaic.cli", "ingest", *gvcfs, "--threads", str(threads)]
    if region:
        args += ["--region", region]
    if defer_register:
        args += ["--defer-register"]
    if ref_gq_bands is not None:
        args += ["--ref-gq-bands", ref_gq_bands]
    if shard_size:
        args += ["--shard-size", str(shard_size)]
    if pos_batch_size is not None:
        args += ["--pos-batch-size", str(pos_batch_size)]
    if aggregate_parquet_size is not None:
        args += ["--aggregate-parquet-size", str(aggregate_parquet_size)]
    return _run(args, extra_env=extra_env, check=check)


def run_register(
    files_per_commit: Optional[int] = None,
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """`mosaic register` (registers deferred batches' files, completes them)."""
    args = ["-m", "mosaic.cli", "register"]
    if files_per_commit is not None:
        args += ["--files-per-commit", str(files_per_commit)]
    return _run(args, extra_env=extra_env, check=check)


def run_aggregate(
    fan_in: Optional[int] = None,
    threads: int = 2,
    rebuild: bool = False,
    ref_shard_size: Optional[int] = None,
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
    timeout: Optional[float] = 600,
) -> subprocess.CompletedProcess:
    args = ["-m", "mosaic.cli", "aggregate", "--threads", str(threads)]
    if fan_in is not None:
        args += ["--fan-in", str(fan_in)]
    if rebuild:
        args += ["--rebuild"]
    if ref_shard_size is not None:
        args += ["--ref-shard-size", str(ref_shard_size)]
    return _run(args, extra_env=extra_env, check=check, timeout=timeout)


def run_maintain(
    older_than_hours: Optional[float] = None,
    retain_last: Optional[int] = None,
    rewrite_manifests: bool = True,
    manifest_target_size_bytes: Optional[int] = None,
    orphans: bool = False,
    orphan_grace_hours: Optional[float] = None,
    dry_run: bool = False,
    force: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """`mosaic maintain` (Iceberg maintenance of the jc_test tables)."""
    args = ["-m", "mosaic.cli", "maintain"]
    if older_than_hours is not None:
        args += ["--older-than-hours", str(older_than_hours)]
    if retain_last is not None:
        args += ["--retain-last", str(retain_last)]
    if not rewrite_manifests:
        args += ["--no-rewrite-manifests"]
    if manifest_target_size_bytes is not None:
        args += [
            "--manifest-target-size-bytes",
            str(manifest_target_size_bytes),
        ]
    if orphans:
        args += ["--orphans"]
    if orphan_grace_hours is not None:
        args += ["--orphan-grace-hours", str(orphan_grace_hours)]
    if dry_run:
        args += ["--dry-run"]
    if force:
        args += ["--force"]
    return _run(args, extra_env=extra_env, check=check)


def run_remove(
    names: List[str],
    threads: int = 2,
    shard_size: Optional[int] = None,
    span_margin: Optional[int] = None,
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """`mosaic remove` of the named samples (jc_test stores)."""
    args = ["-m", "mosaic.cli", "remove", *names, "--threads", str(threads)]
    if shard_size is not None:
        args += ["--shard-size", str(shard_size)]
    if span_margin is not None:
        args += ["--span-margin", str(span_margin)]
    return _run(args, extra_env=extra_env, check=check)


def removal_rows(conn: Any) -> List[Tuple[int, List[str], int, str, str]]:
    """(removal_id, sample_names, batch_id, status, the batch's status) of
    every removal_progress row, in removal_id order."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT r.removal_id, r.sample_names, r.batch_id, r.status, "
            "p.status FROM removal_progress r JOIN ingestion_progress p "
            "ON p.batch_id = r.batch_id ORDER BY r.removal_id"
        )
        return [
            (int(a), list(b), int(c), d, e) for a, b, c, d, e in cur.fetchall()
        ]


def compare_pg_snapshots(
    a: Dict, b: Dict, rtol: float = 1e-6
) -> Dict[str, int]:
    """Assert two PG snapshots (snapshot_pg) are equal: every key and every
    integer exactly (gtc too: it is summed in fixed point, E3), and qual /
    mqsq_1 within ``rtol`` (REAL: two float64 sums in another order, e.g. a
    removal's subtraction, can round to neighbouring float32 values).
    Returns the row counts."""
    for key in ("site_info_int", "ref_blocks", "alleles"):
        if a[key] != b[key]:
            diff = [
                k
                for k in set(a[key]) | set(b[key])
                if a[key].get(k) != b[key].get(k)
            ]
            raise AssertionError(
                f"PG {key} differs at {len(diff)} keys: "
                + ", ".join(
                    f"{k}: {a[key].get(k)} vs {b[key].get(k)}"
                    for k in sorted(diff)[:5]
                )
            )
    msg = _floats_close(a["site_info_float"], b["site_info_float"], rtol, 0.0)
    assert msg is None, msg
    return {
        "site_info": len(a["site_info_int"]),
        "alleles": len(a["alleles"]),
        "ref_blocks": len(a["ref_blocks"]),
    }


def gtc_fixed(g: float) -> int:
    """A record's GTC element in fixed point, the rule ingest and `mosaic
    remove` apply (init_db.GTC_FIXED_BITS): round(g * 2**32), half to
    even."""
    return int(round(float(g) * float(1 << GTC_FIXED_BITS)))


def run_index(check: bool = True) -> subprocess.CompletedProcess:
    return _run(["-m", "mosaic.cli", "index"], check=check)


def run_joint_call(
    out_vcf: str,
    check: bool = True,
    allow_stale: bool = False,
    region: Optional[str] = None,
    window: Optional[int] = None,
    threads: Optional[int] = None,
    extra_env: Optional[Dict[str, str]] = None,
    unit_buffer_bytes: Optional[int] = None,
    spill_dir: Optional[str] = None,
    max_alt_alleles: Optional[int] = None,
    samples: Optional[Sequence[str]] = None,
    samples_file: Optional[str] = None,
    trim_alt_alleles: bool = False,
    min_ac: Optional[int] = None,
    from_export: Optional[str] = None,
) -> subprocess.CompletedProcess:
    """``joint_call.py`` (the driver of the Rust joint caller); skips the
    calling test when no joint-call binary is available
    (``require_joint_call_worker``). ``max_alt_alleles`` None passes no
    ``--max-alt-alleles`` (the driver's default cap, 6, applies; K3).
    ``samples`` passes ``--samples`` (the names joined by commas, as
    given) and ``samples_file`` ``--samples-file`` (K4);
    ``trim_alt_alleles`` ``--trim-alt-alleles`` and ``min_ac`` (not None)
    ``--min-ac`` (K5); ``from_export`` ``--from-export DIR`` (the cohort
    export)."""
    require_joint_call_worker()
    args = ["joint_call.py", out_vcf]
    if from_export is not None:
        args += ["--from-export", from_export]
    if allow_stale:
        args += ["--allow-stale"]
    if region is not None:
        args += ["--region", region]
    if window is not None:
        args += ["--window", str(window)]
    if threads is not None:
        args += ["--threads", str(threads)]
    if unit_buffer_bytes is not None:
        args += ["--unit-buffer-bytes", str(unit_buffer_bytes)]
    if spill_dir is not None:
        args += ["--spill-dir", spill_dir]
    if max_alt_alleles is not None:
        args += ["--max-alt-alleles", str(max_alt_alleles)]
    if samples is not None:
        args += ["--samples", ",".join(samples)]
    if samples_file is not None:
        args += ["--samples-file", samples_file]
    if trim_alt_alleles:
        args += ["--trim-alt-alleles"]
    if min_ac is not None:
        args += ["--min-ac", str(min_ac)]
    return _run(args, extra_env=extra_env, check=check)


def run_cohort_state(
    allow_stale: bool = False,
    samples: Optional[Sequence[str]] = None,
    samples_file: Optional[str] = None,
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """``joint_call.py --cohort-state`` (M1): the state line on stdout."""
    args = ["joint_call.py", "--cohort-state"]
    if allow_stale:
        args += ["--allow-stale"]
    if samples is not None:
        args += ["--samples", ",".join(samples)]
    if samples_file is not None:
        args += ["--samples-file", samples_file]
    return _run(args, extra_env=extra_env, check=check)


def run_export_cohort(
    directory: str,
    allow_stale: bool = False,
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """``joint_call.py --export-cohort DIR`` (M1)."""
    args = ["joint_call.py", "--export-cohort", directory]
    if allow_stale:
        args += ["--allow-stale"]
    return _run(args, extra_env=extra_env, check=check)


def run_export_contig(
    directory: str,
    contig: str,
    extra_env: Optional[Dict[str, str]] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """``joint_call.py --export-contig DIR --region CONTIG`` (M1)."""
    args = ["joint_call.py", "--export-contig", directory, "--region", contig]
    return _run(args, extra_env=extra_env, check=check)


_JC_FILES_READ = re.compile(
    r"data files read: sample_site_info (\d+), sample_ref_blocks (\d+)"
)


def joint_call_files_read(stderr: str) -> Tuple[int, int]:
    """From the driver's ``mosaic-joint-call: ...`` line (K4): the data
    files the binary read (the planned ones), per table."""
    m = _JC_FILES_READ.search(stderr)
    assert m is not None, f"no data files read in:\n{stderr[-2000:]}"
    return int(m.group(1)), int(m.group(2))


def spill_dirs_beside(out_vcf: str) -> List[str]:
    """The ``<output>.spill-*`` directories beside a joint-call output (K2:
    none is ever left after a run)."""
    d = os.path.dirname(os.path.abspath(out_vcf)) or "."
    base = os.path.basename(out_vcf) + ".spill-"
    return sorted(n for n in os.listdir(d) if n.startswith(base))


_JC_SPILLED = re.compile(
    r"held at most (\d+) unit\(s\), (\d+) bytes; spilled (\d+) unit\(s\), "
    r"(\d+) bytes \(copy ([0-9.]+) s\)"
)


def joint_call_spilled(stderr: str) -> Tuple[int, int, int, int, float]:
    """From the driver's ``mosaic-joint-call: ...`` line: the held units
    and bytes at most, the spilled units and bytes, the copy seconds."""
    m = _JC_SPILLED.search(stderr)
    assert m is not None, f"no held / spilled figures in:\n{stderr[-2000:]}"
    return (
        int(m.group(1)),
        int(m.group(2)),
        int(m.group(3)),
        int(m.group(4)),
        float(m.group(5)),
    )


_JC_CAP = re.compile(
    r"^ALT cap \(--max-alt-alleles (\d+)\): (\d+) records capped, (\d+) "
    r"genotypes written \./\., (\d+) ALT observations on dropped alleles, "
    r"(\d+) kept-allele ALT observations lost with those genotypes$",
    re.M,
)
CAP_COUNT_KEYS = (
    "records_capped",
    "genotypes_no_called",
    "dropped_alt_observations",
    "kept_alt_observations_lost",
)


def joint_call_cap_counts(
    stderr: str,
) -> Optional[Tuple[int, Dict[str, int]]]:
    """From the driver's ALT cap closing line (K3): the cap and its four
    counts (``vcf_project.project``'s keys); None when the run had no cap
    (no such line)."""
    m = _JC_CAP.search(stderr)
    if m is None:
        return None
    g = [int(x) for x in m.groups()]
    return g[0], dict(zip(CAP_COUNT_KEYS, g[1:]))


_JC_TRIM = re.compile(
    r"^ALT trim \(--trim-alt-alleles\): (\d+) records trimmed, (\d+) ALT "
    r"alleles with AC 0 dropped$",
    re.M,
)
_JC_MIN_AC = re.compile(
    r"^Minimum AC \(--min-ac (\d+)\): (\d+) records left out \(fewer ALT "
    r"observations in their written genotypes\)$",
    re.M,
)


def joint_call_trim_counts(stderr: str) -> Optional[Dict[str, int]]:
    """From the driver's trim closing line (K5): ``vcf_project.project_trim``'s
    counts; None when the run had no ``--trim-alt-alleles``."""
    m = _JC_TRIM.search(stderr)
    if m is None:
        return None
    return {
        "records_trimmed": int(m.group(1)),
        "alts_dropped": int(m.group(2)),
    }


def joint_call_min_ac_counts(
    stderr: str,
) -> Optional[Tuple[int, Dict[str, int]]]:
    """From the driver's ``--min-ac`` closing line (K5): K and
    ``vcf_project.project_min_ac``'s counts; None when the run had none."""
    m = _JC_MIN_AC.search(stderr)
    if m is None:
        return None
    return int(m.group(1)), {"records_left_out": int(m.group(2))}


def popen_pipeline(
    subcommand: str,
    args: List[str],
    extra_env: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    """Launch a `mosaic` subcommand in its OWN process group (start_new_session)
    so a crash test can SIGKILL the whole group, not just the parent.
    ``ingest`` skips the calling test without a worker binary
    (``require_ingest_worker``)."""
    if subcommand == "ingest":
        require_ingest_worker()
    return subprocess.Popen(
        [sys.executable, "-m", "mosaic.cli", subcommand, *args],
        cwd=REPO_ROOT,
        env=test_env(extra_env),
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


# ── gVCF subsetting (bcftools) ───────────────────────────────────────────────


def subset_gvcf(src: str, region: str, dst: str) -> None:
    subprocess.run(
        [
            "bcftools",
            "view",
            "--write-index=tbi",
            "-O",
            "z",
            "-o",
            dst,
            "-r",
            region,
            src,
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def stage_batches(
    regions: List[str],
    work_dir: str,
    gvcfs: Optional[List[str]] = None,
) -> None:
    """reset + init + ingest one batch per region.

    Each ``--region`` produces a distinct ingestion batch (the batch key is
    ``(sample_names, region)``), so N regions stage N batches over the same samples
    — the input an aggregation tier tree needs.
    """
    require_ingest_worker()
    gvcfs = gvcfs or GVCFS
    reset_state()
    run_init(gvcfs[0], "chr21")
    for i, region in enumerate(regions):
        subset = []
        for j, src in enumerate(gvcfs):
            dst = os.path.join(work_dir, f"b{i}_s{j}.g.vcf.gz")
            subset_gvcf(src, region, dst)
            subset.append(dst)
        run_ingest(subset, region=region, threads=2)


# ── snapshot / compare oracle ────────────────────────────────────────────────


def snapshot_pg(conn: Any) -> Dict[str, Dict]:
    """Snapshot the cross-sample PG aggregates, keyed by genomic coordinate +
    allele string (id-independent, so comparable across independent runs).

    Returns dicts split into exact-integer and float (qual/mqsq_1) parts.
    """
    si_int: Dict[Tuple[int, int], Tuple] = {}
    si_float: Dict[Tuple[int, int], Tuple[float, float]] = {}
    rb: Dict[Tuple[int, int], Tuple[int, int]] = {}
    al: Dict[Tuple[int, int, str], Tuple[int, Optional[str]]] = {}
    with conn.cursor() as cur:
        cur.execute(
            "SELECT chromosome_index, position, dp, mqsq_2, sb, gtc, "
            "qual, mqsq_1 FROM site_info"
        )
        for ci, pos, dp, m2, sb, gtc, qual, m1 in cur.fetchall():
            si_int[(ci, pos)] = (
                int(dp or 0),
                int(m2 or 0),
                tuple(sb or []),
                tuple(gtc or []),
            )
            si_float[(ci, pos)] = (float(qual or 0.0), float(m1 or 0.0))
        cur.execute(
            "SELECT chromosome_index, position, dp, sample_counts FROM ref_blocks"
        )
        for ci, pos, dp, sc in cur.fetchall():
            rb[(ci, pos)] = (int(dp or 0), int(sc or 0))
        # The allele catalog: every listed allele with its summed count
        # (zero included) and derived del_range.
        cur.execute(
            "SELECT chromosome_index, position, allele, count, "
            "del_range::text FROM alleles"
        )
        for ci, pos, allele, cnt, dr in cur.fetchall():
            al[(ci, pos, allele)] = (int(cnt), dr)
    return {
        "site_info_int": si_int,
        "site_info_float": si_float,
        "ref_blocks": rb,
        "alleles": al,
    }


def _floats_close(
    a: Dict, b: Dict, rtol: float = 1e-6, atol: float = 1e-2
) -> Optional[str]:
    if set(a) != set(b):
        return f"float keys differ ({len(set(a) ^ set(b))} mismatched)"
    for k, av in a.items():
        bv = b[k]
        for x, y in zip(av, bv):
            if abs(x - y) > atol + rtol * abs(y):
                return f"float mismatch at {k}: {av} vs {bv}"
    return None


def assert_pg_integer_equal(a: Dict, b: Dict) -> None:
    """Assert the bit-exact integer aggregates match (ignores qual/mqsq_1).

    Use for cross-fan-in / cross-tier comparisons where the REAL columns may
    legitimately differ at the last ULP.
    """
    for key in ("site_info_int", "ref_blocks", "alleles"):
        if a[key] != b[key]:
            only_a = {
                k: a[key][k] for k in a[key] if a[key].get(k) != b[key].get(k)
            }
            sample = dict(list(only_a.items())[:5])
            raise AssertionError(
                f"PG integer aggregate '{key}' differs "
                f"(|a|={len(a[key])}, |b|={len(b[key])}); first diffs: {sample}"
            )


def assert_pg_bitexact(a: Dict, b: Dict) -> None:
    """Assert integer aggregates are bit-exact AND qual/mqsq_1 within tolerance."""
    assert_pg_integer_equal(a, b)
    msg = _floats_close(a["site_info_float"], b["site_info_float"])
    if msg:
        raise AssertionError(f"PG site_info float columns differ: {msg}")


def oracle_from_partials(conn: Any, catalog: Any) -> Dict[str, Dict]:
    """Recompute the three cross-sample aggregates from the ``batch_*`` partials
    of the completed batches (ingest and removal batches alike: a removal's
    partials are negative) in pure Python -- the pre-vectorization algorithm
    (``x or 0`` sums in Python ints / floats, dict per key, ref_blocks kept only
    at variant-site positions) -- independent of ``mosaic.aggregate``.

    Returns {"alleles": {(c, pos, allele): count}, "site_info": {(c, pos):
    [qual, dp, mqsq_1, mqsq_2, sb, gtc]}, "site_info_all": the same for every
    root key (the agg_site_info mirror's rows), "ref_blocks": {(c, pos): (dp,
    sc)}}. ``alleles`` has a key for every allele a live record lists (net
    n_listed != 0), count 0 included: the catalog's keys. ``site_info`` has
    the sites with a live record (net n_records != 0); its gtc is the
    fixed-point sum (PostgreSQL holds its integer part, sum >>
    GTC_FIXED_BITS). ``ref_blocks``: the live sites whose net (dp,
    sample_counts) is not (0, 0).
    """
    with conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id FROM ingestion_progress WHERE status='complete'"
        )
        batch_ids = [b for (b,) in cur.fetchall()]

    def scan(table: str) -> Dict[str, List[Any]]:
        tbl = catalog.load_table(f"{TEST_NAMESPACE}.{table}")
        arrow = tbl.scan(row_filter=In("batch_id", batch_ids)).to_arrow()
        return {
            name: arrow.column(name).to_pylist() for name in arrow.schema.names
        }

    counts: Dict[Tuple[int, int, str], List[int]] = {}
    cols = scan("batch_allele_counts")
    for ci, pos, allele, cnt, listed in zip(
        cols["chromosome_index"],
        cols["position"],
        cols["allele"],
        cols["count"],
        cols["n_listed"],
    ):
        a = counts.setdefault((ci, pos, allele), [0, 0])
        a[0] += int(cnt or 0)
        a[1] += int(listed or 0)
    alleles = {k: c for k, (c, listed) in counts.items() if listed != 0}

    site_all: Dict[Tuple[int, int], List[Any]] = {}
    n_records: Dict[Tuple[int, int], int] = {}
    cols = scan("batch_site_info")
    for ci, pos, qual, dp, m1, m2, sb, gtc, nrec in zip(
        cols["chromosome_index"],
        cols["position"],
        cols["qual"],
        cols["dp"],
        cols["mqsq_1"],
        cols["mqsq_2"],
        cols["sb"],
        cols["gtc"],
        cols["n_records"],
    ):
        a = site_all.setdefault((ci, pos), [0.0, 0, 0.0, 0, [0] * 4, [0] * 3])
        a[0] += float(qual or 0.0)
        a[1] += int(dp or 0)
        a[2] += float(m1 or 0.0)
        a[3] += int(m2 or 0)
        sb = sb or [0] * 4
        gtc = gtc or [0] * 3
        for j in range(4):
            a[4][j] += int(sb[j] or 0)
        for j in range(3):
            a[5][j] += int(gtc[j] or 0)
        n_records[(ci, pos)] = n_records.get((ci, pos), 0) + int(nrec or 0)
    site = {k: v for k, v in site_all.items() if n_records[k] != 0}

    dense: Dict[Tuple[int, int], Tuple[int, int]] = {}
    cols = scan("batch_ref_blocks")
    for ci, pos, dp, sc in zip(
        cols["chromosome_index"],
        cols["position"],
        cols["dp"],
        cols["sample_counts"],
    ):
        if (ci, pos) not in site:
            continue  # the leaf filter: variant-site positions only
        old = dense.get((ci, pos), (0, 0))
        dense[(ci, pos)] = (old[0] + int(dp or 0), old[1] + int(sc or 0))
    ref = {k: v for k, v in dense.items() if v != (0, 0)}
    return {
        "alleles": alleles,
        "site_info": site,
        "site_info_all": site_all,
        "ref_blocks": ref,
    }


def _rel_close(a: float, b: float, rtol: float) -> bool:
    return abs(a - b) <= rtol * max(abs(a), abs(b))


def assert_matches_oracle(
    conn: Any, catalog: Any, rtol: float = 1e-9
) -> Dict[str, int]:
    """Compare PostgreSQL (and the agg_site_info mirror) to
    ``oracle_from_partials``. Integer columns must be exact. qual / mqsq_1 are
    float64 sums: the agg_site_info mirror (float64) must be within relative
    ``rtol``; PG stores them as REAL (psycopg2 returns PG's shortest float4 text
    parsed as a double, e.g. 133.01), so the PG value rounded back to float32 must
    be within ``rtol`` (or one float32 ULP) of the oracle rounded to float32.
    gtc: PostgreSQL holds the integer part of the fixed-point sum, exactly.
    Returns row counts (for sanity asserts)."""
    truth = oracle_from_partials(conn, catalog)
    pg = snapshot_pg(conn)

    # The alleles catalog: exactly the oracle's keys (zero-count alleles
    # included), the summed counts, and del_range derived from the string.
    from mosaic.aggregate import allele_del_range

    pg_counts = {k: v[0] for k, v in pg["alleles"].items()}
    missing = set(truth["alleles"]) - set(pg_counts)
    assert not missing, f"oracle alleles absent from PG: {list(missing)[:5]}"
    extra = set(pg_counts) - set(truth["alleles"])
    assert not extra, f"PG alleles no partial lists: {list(extra)[:5]}"
    bad = {
        k: (c, truth["alleles"][k])
        for k, c in pg_counts.items()
        if c != truth["alleles"][k]
    }
    assert (
        not bad
    ), f"allele counts differ (pg, oracle): {list(bad.items())[:5]}"
    for (ci, pos, allele), (_, dr) in pg["alleles"].items():
        span = allele_del_range(pos, allele)
        want = None if span is None else f"[{span[0]},{span[1]})"
        assert dr == want, ((ci, pos, allele), dr, want)

    # site_info integer columns exact; REAL columns compared as float32.
    assert set(pg["site_info_int"]) == set(truth["site_info"]), (
        f"site_info keys differ ({len(pg['site_info_int'])} pg vs "
        f"{len(truth['site_info'])} oracle)"
    )
    for k, (qual, dp, m1, m2, sb, gtc) in truth["site_info"].items():
        want_gtc = tuple(int(g) >> GTC_FIXED_BITS for g in gtc)
        assert pg["site_info_int"][k] == (dp, m2, tuple(sb), want_gtc), k
        pg_q, pg_m1 = pg["site_info_float"][k]
        for got, want in ((pg_q, qual), (pg_m1, m1)):
            got32 = np.float32(got)
            want32 = np.float32(want)
            # Within rtol, or one float32 ULP: two float64 sums in another
            # order (a removal's negative partial, another tier order) can
            # round to neighbouring REALs.
            assert _rel_close(float(got32), float(want32), rtol) or abs(
                got32 - want32
            ) <= np.spacing(abs(want32)), (k, float(got32), float(want32))

    # agg_site_info mirror: every root row (dead sites too), float64 and the
    # fixed-point gtc sums, compared directly.
    agg = (
        catalog.load_table(f"{TEST_NAMESPACE}.agg_site_info")
        .scan()
        .to_arrow()
        .to_pylist()
    )
    assert len(agg) == len(truth["site_info_all"])
    for r in agg:
        qual, dp, m1, m2, sb, gtc = truth["site_info_all"][
            (r["chromosome_index"], r["position"])
        ]
        assert (r["dp"], r["mqsq_2"], r["sb"], r["gtc"]) == (dp, m2, sb, gtc)
        assert _rel_close(r["qual"], qual, rtol) or abs(qual) < 1e-6, (
            r,
            qual,
        )
        assert _rel_close(r["mqsq_1"], m1, rtol) or abs(m1) < 1e-6, (r, m1)

    # ref_blocks exact (sparse: variant sites with coverage).
    assert pg["ref_blocks"] == truth["ref_blocks"], (
        f"ref_blocks differ ({len(pg['ref_blocks'])} pg vs "
        f"{len(truth['ref_blocks'])} oracle)"
    )
    return {
        "alleles": len(truth["alleles"]),
        "site_info": len(truth["site_info"]),
        "ref_blocks": len(truth["ref_blocks"]),
    }


def count_iceberg(
    catalog: Any, tables: Optional[List[str]] = None
) -> Dict[str, int]:
    """Row counts for the given (default: per-sample) Iceberg tables."""
    if tables is None:
        tables = ["sample_ref_blocks", "sample_site_info"]
    out: Dict[str, int] = {}
    for t in tables:
        try:
            tbl = catalog.load_table(f"{TEST_NAMESPACE}.{t}")
            out[t] = tbl.scan().to_arrow().num_rows
        except Exception:
            out[t] = -1
    return out


# The six Iceberg tables an ingest batch writes data files to (the tables
# `mosaic ingest --defer-register` records in ingest_files).
INGEST_TABLES = [
    "sample_ref_blocks",
    "sample_site_info",
    "batch_ref_blocks",
    "batch_site_info",
    "batch_allele_counts",
    "batch_genotypes",
]


def iceberg_data_files(
    catalog: Any, tables: Optional[List[str]] = None
) -> Dict[str, List[str]]:
    """{table: sorted data-file URIs of its current snapshot} (default: the
    ingest tables). A list, not a set, so a file registered twice shows."""
    out: Dict[str, List[str]] = {}
    for t in tables or INGEST_TABLES:
        tbl = catalog.load_table(f"{TEST_NAMESPACE}.{t}")
        if tbl.current_snapshot() is None:
            out[t] = []
        else:
            out[t] = sorted(
                task.file.file_path for task in tbl.scan().plan_files()
            )
    return out


def iceberg_file_etags(
    catalog: Any, tables: Optional[List[str]] = None
) -> Dict[str, List[Tuple[str, str]]]:
    """{table: sorted (data-file URI, S3 ETag)} of the current snapshot
    (default: the ingest tables). The ETag is the object's content digest,
    so equal lists mean the same files, byte for byte (pyarrow writes no
    timestamp into a Parquet file)."""
    s3 = s3_client()
    out: Dict[str, List[Tuple[str, str]]] = {}
    for t, uris in iceberg_data_files(catalog, tables).items():
        rows = []
        for uri in uris:
            bucket, key = uri[len("s3://") :].split("/", 1)
            rows.append((uri, s3.head_object(Bucket=bucket, Key=key)["ETag"]))
        out[t] = rows
    return out


def _hash_array(h: Any, arr: Any) -> None:
    """Feed ``arr`` (a pyarrow Array) to the hash ``h`` in a canonical,
    layout-independent form: the null mask, then the values -- integers as
    their bytes, floats as their IEEE 754 bits (so -0.0, NaN and every ULP
    count), strings / binaries with their lengths, lists as their value
    lengths and then their values, recursively."""
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.compute as pc  # type: ignore[import-untyped]

    t = arr.type
    h.update(str(t).encode())
    h.update(np.asarray(arr.is_null().to_numpy(zero_copy_only=False)))
    if pa.types.is_list(t) or pa.types.is_large_list(t):
        lens = pc.list_value_length(arr).fill_null(-1)
        h.update(np.asarray(lens.to_numpy(zero_copy_only=False), np.int64))
        _hash_array(h, arr.flatten())
    elif pa.types.is_floating(t):
        v = np.asarray(arr.fill_null(0).to_numpy(zero_copy_only=False))
        h.update(v.astype(np.float64).view(np.uint64))
    elif pa.types.is_integer(t) or pa.types.is_boolean(t):
        v = arr.fill_null(0).to_numpy(zero_copy_only=False)
        h.update(np.asarray(v, np.int64))
    elif pa.types.is_dictionary(t):
        _hash_array(h, arr.dictionary_decode())
    else:  # string / binary
        for x in arr.to_pylist():
            b = b"" if x is None else (x.encode() if isinstance(x, str) else x)
            h.update(len(b).to_bytes(8, "little"))
            h.update(b)


def parquet_file_signature(path: str) -> Dict[str, Any]:
    """What a data file holds, independent of its encoder (the
    cross-engine contract, row for row rather than byte for byte): the
    Arrow schema as pyarrow prints it (``str``: the list child names
    count), the row count, each row group's row count, the sort
    declaration, per row group the columns carrying statistics and those
    with a column index (the page index), and a digest of every column's
    values read row group by row group (floats bit-exact, nulls kept)."""
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    pf = pq.ParquetFile(path)
    md = pf.metadata
    names = pf.schema_arrow.names
    digests = {n: hashlib.sha256() for n in names}
    groups, sorting, stats, index = [], set(), set(), set()
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        groups.append(rg.num_rows)
        sorting.add(
            tuple(
                (c.column_index, c.descending, c.nulls_first)
                for c in rg.sorting_columns
            )
        )
        cols = [rg.column(k) for k in range(rg.num_columns)]
        stats.add(tuple(c.path_in_schema for c in cols if c.is_stats_set))
        index.add(tuple(c.path_in_schema for c in cols if c.has_column_index))
        table = pf.read_row_group(g)
        for n in names:
            col = table.column(n)
            for chunk in col.chunks:
                _hash_array(digests[n], chunk)
    return {
        "schema": str(pf.schema_arrow),
        "num_rows": md.num_rows,
        "row_groups": groups,
        "sorting_columns": sorted(sorting),
        "statistics_columns": sorted(stats),
        "column_index_columns": sorted(index),
        "columns": {n: d.hexdigest() for n, d in digests.items()},
    }


def compare_file_signatures(
    a: Dict[str, Any], b: Dict[str, Any], name: str = ""
) -> List[str]:
    """The differences between two ``parquet_file_signature`` results (an
    empty list: the same rows, row groups, schema, sort declaration and
    statistics columns)."""
    diffs = []
    for key in a:
        if key == "columns":
            for col in a[key]:
                if a[key][col] != b[key].get(col):
                    diffs.append(f"{name}: column {col} differs")
        elif a[key] != b.get(key):
            va, vb = a[key], b.get(key)
            if isinstance(va, list) and len(va) > 12:
                va, vb = f"{len(va)} items {va[:6]}", vb and (
                    f"{len(vb)} items {vb[:6]}"
                )
            diffs.append(f"{name}: {key} {va!r} != {vb!r}")
    return diffs


def iceberg_file_signatures(
    catalog: Any, workdir: str, tables: Optional[List[str]] = None
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """{table: {data-file URI: parquet_file_signature}} of the current
    snapshot (default: the ingest tables), each file downloaded into
    ``workdir`` and removed after."""
    s3 = s3_client()
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for t, uris in iceberg_data_files(catalog, tables).items():
        out[t] = {}
        for uri in uris:
            bucket, key = uri[len("s3://") :].split("/", 1)
            local = os.path.join(workdir, key.replace("/", "_"))
            s3.download_file(bucket, key, local)
            try:
                out[t][uri] = parquet_file_signature(local)
            finally:
                os.remove(local)
    return out


def compare_iceberg_signatures(
    a: Dict[str, Dict[str, Dict[str, Any]]],
    b: Dict[str, Dict[str, Dict[str, Any]]],
) -> List[str]:
    """The differences between two ``iceberg_file_signatures`` results:
    the same tables and URIs, and per URI ``compare_file_signatures``."""
    diffs = []
    for t in sorted(set(a) | set(b)):
        ua, ub = sorted(a.get(t, {})), sorted(b.get(t, {}))
        if ua != ub:
            diffs.append(f"{t}: URIs {ua} != {ub}")
            continue
        for uri in ua:
            diffs += compare_file_signatures(a[t][uri], b[t][uri], uri)
    return diffs


def iceberg_snapshot_counts(
    catalog: Any, tables: Optional[List[str]] = None
) -> Dict[str, int]:
    """{table: number of snapshots in its metadata} (default: the ingest
    tables). Until `mosaic maintain` expires some, this counts the
    commits."""
    return {
        t: len(catalog.load_table(f"{TEST_NAMESPACE}.{t}").metadata.snapshots)
        for t in (tables or INGEST_TABLES)
    }


def iceberg_snapshot_ids(
    catalog: Any, tables: Optional[List[str]] = None
) -> Dict[str, set]:
    """{table: ids of the snapshots in its metadata} (default: every
    table)."""
    return {
        t: {
            s.snapshot_id
            for s in catalog.load_table(
                f"{TEST_NAMESPACE}.{t}"
            ).metadata.snapshots
        }
        for t in (tables or ICEBERG_TABLES)
    }


def manifest_counts(
    catalog: Any, tables: Optional[List[str]] = None
) -> Dict[str, int]:
    """{table: manifests in its current snapshot's manifest list} (0 for a
    table with no snapshot; default: the ingest tables)."""
    out: Dict[str, int] = {}
    for t in tables or INGEST_TABLES:
        tbl = catalog.load_table(f"{TEST_NAMESPACE}.{t}")
        snap = tbl.current_snapshot()
        out[t] = 0 if snap is None else len(snap.manifests(tbl.io))
    return out


def table_references(catalog: Any, table: str) -> Dict[str, set]:
    """Every S3 URI a table's metadata names, read directly from pyiceberg
    (independent of mosaic.maintain), as {"lists", "manifests", "live",
    "deleted", "metadata"}: the manifest lists and manifests of every
    snapshot, the files their entries name live (ADDED / EXISTING) or only
    DELETED, and the current + logged metadata.json versions."""
    from pyiceberg.manifest import ManifestEntryStatus

    tbl = catalog.load_table(f"{TEST_NAMESPACE}.{table}")
    out: Dict[str, set] = {
        k: set() for k in ("lists", "manifests", "live", "deleted", "metadata")
    }
    out["metadata"].add(tbl.metadata_location)
    out["metadata"].update(e.metadata_file for e in tbl.metadata.metadata_log)
    seen = set()
    for snap in tbl.metadata.snapshots:
        out["lists"].add(snap.manifest_list)
        for m in snap.manifests(tbl.io):
            out["manifests"].add(m.manifest_path)
            if m.manifest_path in seen:
                continue
            seen.add(m.manifest_path)
            for e in m.fetch_manifest_entry(tbl.io, discard_deleted=False):
                live = e.status != ManifestEntryStatus.DELETED
                out["live" if live else "deleted"].add(e.data_file.file_path)
    out["deleted"] -= out["live"]
    return out


def _uri_listing(bucket: str, prefix: str) -> Dict[str, Tuple[str, Any]]:
    """{s3://bucket/key: (etag, last_modified)} under bucket/prefix."""
    return {
        f"s3://{bucket}/{key}": v
        for key, v in bucket_listing(bucket, prefix).items()
    }


def location_listing(
    catalog: Any, table: str, sub: str = ""
) -> Dict[str, Tuple[str, Any]]:
    """{s3 URI: (etag, last_modified)} of every object under the table's
    location (``sub="metadata/"``: its metadata directory). The location is
    in the catalog's warehouse bucket, not the test data bucket."""
    location = catalog.load_table(f"{TEST_NAMESPACE}.{table}").location()
    bucket, _, key = location.split("://", 1)[1].partition("/")
    return _uri_listing(bucket, key.rstrip("/") + "/" + sub)


def lake_objects(catalog: Any, table: str) -> Dict[str, Tuple[str, Any]]:
    """{s3 URI: (etag, last_modified)} of every object that belongs to a
    table: its upload prefix in the test bucket (s3://jc-test-warehouse/
    <table>/) and its location."""
    out = _uri_listing(TEST_BUCKET, f"{table}/")
    out.update(location_listing(catalog, table))
    return out


def assert_lake_exact(
    catalog: Any, tables: Optional[List[str]] = None
) -> None:
    """After `mosaic maintain --older-than-hours 0`: each table (default:
    all) holds exactly what its remaining snapshots reference -- their
    manifest lists, manifests and live files, plus its metadata.json
    versions -- and nothing else (no file only DELETED, nothing
    unreferenced)."""
    for t in tables or ICEBERG_TABLES:
        refs = table_references(catalog, t)
        want = refs["lists"] | refs["manifests"] | refs["live"]
        want |= refs["metadata"]
        have = set(lake_objects(catalog, t))
        assert have == want, (
            f"{t}: unreferenced objects {sorted(have - want)[:5]}, missing "
            f"objects {sorted(want - have)[:5]}"
        )


def ingest_files_rows(conn: Any) -> List[Tuple[str, str, int, bool]]:
    """Every ingest_files row as (iceberg_table, uri, batch_id, registered),
    sorted."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT iceberg_table, uri, batch_id, registered_at IS NOT NULL "
            "FROM ingest_files ORDER BY iceberg_table, batch_id, uri"
        )
        return list(cur.fetchall())


def bucket_listing(
    bucket: str = TEST_BUCKET, prefix: str = ""
) -> Dict[str, Tuple[str, Any]]:
    """{key: (etag, last_modified)} of every object in the test bucket (or
    under ``prefix`` of ``bucket``): a re-upload changes an object's
    last_modified."""
    out: Dict[str, Tuple[str, Any]] = {}
    paginator = s3_client().get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for o in page.get("Contents", []):
            out[o["Key"]] = (o["ETag"], o["LastModified"])
    return out


def snapshot_vcf(path: str) -> str:
    """sha256 of the sorted variant records (no header), id-independent."""
    proc = subprocess.run(
        ["bcftools", "view", "-H", path],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = sorted(proc.stdout.splitlines())
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


# ── Joint-call goldens ───────────────────────────────────────────────────────
#
# A frozen digest of the joint caller's decompressed VCF text, header
# included, per fixture scenario: the reference for the joint caller's
# output since the Python engine's removal. The six scenarios of the
# first recording were recorded where the Python and Rust engines agreed
# byte for byte. The digests are tracked (GOLDEN_DIGESTS), the VCFs are
# local copies under the git-ignored test_data/ (GOLDEN_VCF_DIR), used to
# show the first differing lines on a mismatch.
#
# Record mode: MOSAIC_JOINT_CALL_GOLDEN_RECORD=1 (not the e2e test's
# MOSAIC_GOLDEN_RECORD) and MOSAIC_JOINT_CALL_GOLDEN_REASON=<why the output
# changes> (required: the re-recording rule, a golden is re-recorded only
# when an intended output change is being made). An
# entry whose digest, counts and closing lines are unchanged is left
# exactly as it is; a changed digest restamps the entry's ``recorded``
# block; a change of the closing lines alone keeps ``recorded`` and adds
# ``closing_lines_recorded``.

GOLDEN_DIGESTS = os.path.join(
    REPO_ROOT, "tests", "golden", "joint_call_digests.json"
)
GOLDEN_VCF_DIR = os.path.join(REPO_ROOT, "test_data", "golden_joint_call")
GOLDEN_RECORD_ENV = "MOSAIC_JOINT_CALL_GOLDEN_RECORD"
GOLDEN_REASON_ENV = "MOSAIC_JOINT_CALL_GOLDEN_REASON"
GOLDEN_FORMAT = 2
GOLDEN_DIFF_LINES = 5
GOLDEN_NOTE = (
    "Digests (SHA-256, records, bytes) of the joint caller's decompressed "
    "VCF text, header included, per scenario of "
    "tests/integration/test_rust_joint_call.py; closing_lines are the run's "
    "'Completed!' and catalog cross-check stderr lines; each scenario's "
    "'recorded' says when, where and against what its text was recorded, "
    "and 'closing_lines_recorded' (when present) when only its closing "
    f"lines were re-recorded. Recorded with {GOLDEN_RECORD_ENV}=1 and "
    f"{GOLDEN_REASON_ENV}=<why> (jc_testlib.assert_golden); the VCFs are "
    "local copies under test_data/golden_joint_call/."
)
# What a recording without a cross-engine comparison is checked against.
RUST_ONLY_REFERENCE = "the Rust engine's output"
# Scenarios recorded by this process: in record mode the first output of a
# scenario is recorded and every later one (another window or thread
# count) is checked against it.
_GOLDEN_RECORDED: set = set()


def golden_recording() -> bool:
    return _golden_flag(GOLDEN_RECORD_ENV)


def vcf_text_bytes(path: str) -> bytes:
    """The decompressed bytes of a VCF (gzip / BGZF by its magic, else the
    file as is)."""
    with open(path, "rb") as fh:
        magic = fh.read(2)
    if magic == b"\x1f\x8b":
        with gzip.open(path, "rb") as fh:
            return fh.read()
    with open(path, "rb") as fh:
        return fh.read()


def vcf_text_digest(path: str) -> Dict[str, Any]:
    """SHA-256 of the decompressed text, header included, its record count
    (the lines not starting ``#``) and its byte count."""
    data = vcf_text_bytes(path)
    records = sum(
        1 for line in data.split(b"\n") if line and not line.startswith(b"#")
    )
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "records": records,
        "bytes": len(data),
    }


def closing_lines(stderr: str) -> List[str]:
    """A joint-call run's ``Completed!`` line and the catalog cross-check
    lines after it."""
    lines = stderr.splitlines()
    for k, x in enumerate(lines):
        if x.startswith("Completed! Total"):
            return lines[k:]
    raise AssertionError(
        "no 'Completed! Total' line in the joint call's stderr"
    )


# The golden machinery shared by the joint-call goldens (above) and the
# ingest signatures (below, K6): the digest file's load and atomic write,
# the record-mode switch, the required reason and a recording's provenance.


def _golden_flag(env: str) -> bool:
    return os.environ.get(env, "") not in ("", "0")


def _load_golden_file(path: str, fmt: int) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {"format": fmt, "scenarios": {}}
    with open(path) as fh:
        return json.load(fh)


def _write_golden_file(
    path: str, data: Dict[str, Any], fmt: int, note: str
) -> None:
    data["format"] = fmt
    data["note"] = note
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)
        fh.write("\n")
    os.replace(tmp, path)


def _golden_reason(
    scenario: str, kind: str, record_env: str, reason_env: str, rule: str
) -> str:
    """The record mode's required reason (``reason_env``), else an
    AssertionError naming the rule."""
    reason = os.environ.get(reason_env, "").strip()
    if not reason:
        raise AssertionError(
            f"{kind} {scenario}: record mode ({record_env}=1) needs "
            f"{reason_env}=<why this output changes> (the "
            f"re-recording rule: {rule}); nothing recorded"
        )
    return reason


def _golden_provenance(reference: str, reason: str) -> Dict[str, Any]:
    """A recording's ``recorded`` block: commit, tracked changes, date,
    platform, and what the output was checked against with the reason."""
    return {
        "commit": _git("rev-parse", "--short", "HEAD"),
        "tracked_changes": bool(_git("status", "--porcelain", "-uno")),
        "date": _now(),
        "platform": f"{platform.system()} {platform.release()} "
        f"{platform.machine()}",
        "reference": f"{reference}; {reason}",
    }


def _load_goldens() -> Dict[str, Any]:
    return _load_golden_file(GOLDEN_DIGESTS, GOLDEN_FORMAT)


def _git(*args: str) -> str:
    proc = subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True
    )
    return proc.stdout.strip() if proc.returncode == 0 else "unknown"


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


DIGEST_KEYS = ("sha256", "records", "bytes")


def _record_golden(
    scenario: str,
    vcf_path: str,
    digest: Dict[str, Any],
    closing: Optional[List[str]],
    reference: Optional[str],
) -> None:
    reason = _golden_reason(
        scenario,
        "golden",
        GOLDEN_RECORD_ENV,
        GOLDEN_REASON_ENV,
        "re-record only when an intended output change is being made",
    )
    goldens = _load_goldens()
    scenarios = goldens.setdefault("scenarios", {})
    old = scenarios.get(scenario)
    same_digest = old is not None and all(
        old.get(k) == digest[k] for k in DIGEST_KEYS
    )
    same_closing = closing is None or (
        old is not None and old.get("closing_lines") == closing
    )
    copy = os.path.join(GOLDEN_VCF_DIR, f"{scenario}.vcf.gz")
    if same_digest and same_closing:
        state = "unchanged"
    elif same_digest:
        assert old is not None
        old["closing_lines"] = closing
        old["closing_lines_recorded"] = {
            "commit": _git("rev-parse", "--short", "HEAD"),
            "date": _now(),
            "reason": reason,
        }
        state = "closing lines re-recorded"
    else:
        ref = reference or RUST_ONLY_REFERENCE
        entry: Dict[str, Any] = dict(digest)
        if closing is not None:
            entry["closing_lines"] = closing
        entry["recorded"] = _golden_provenance(ref, reason)
        scenarios[scenario] = entry
        state = "new" if old is None else f"recorded, was {old.get('sha256')}"
    if state != "unchanged":
        _write_golden_file(GOLDEN_DIGESTS, goldens, GOLDEN_FORMAT, GOLDEN_NOTE)
    if not same_digest or not os.path.exists(copy):
        os.makedirs(GOLDEN_VCF_DIR, exist_ok=True)
        with open(vcf_path, "rb") as fh:
            compressed = fh.read(2) == b"\x1f\x8b"
        if compressed:
            shutil.copyfile(vcf_path, copy)
        else:  # a plain-text output: the copy compressed like the others
            with open(vcf_path, "rb") as src, gzip.open(copy, "wb") as dst:
                shutil.copyfileobj(src, dst)
    print(
        f"golden {scenario}: sha256={digest['sha256']} "
        f"records={digest['records']} bytes={digest['bytes']} ({state})"
    )
    _GOLDEN_RECORDED.add(scenario)


def _first_differences(golden_copy: str, vcf_path: str) -> List[str]:
    a = vcf_text_bytes(golden_copy).decode(errors="replace").split("\n")
    b = vcf_text_bytes(vcf_path).decode(errors="replace").split("\n")
    out: List[str] = []
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "<end of golden>"
        y = b[i] if i < len(b) else "<end of output>"
        if x != y:
            out.append(
                f"  line {i + 1}:\n    golden: {x[:300]}\n    output: "
                f"{y[:300]}"
            )
            if len(out) >= GOLDEN_DIFF_LINES:
                break
    return out


def assert_golden(
    scenario: str,
    vcf_path: str,
    stderr: Optional[str] = None,
    reference: Optional[str] = None,
) -> Dict[str, Any]:
    """Check a joint-call output against the scenario's recorded digest
    (``GOLDEN_DIGESTS``), and, when ``stderr`` is given and the golden has
    them, its closing lines. A scenario with no recorded digest fails.

    Record mode (MOSAIC_JOINT_CALL_GOLDEN_RECORD=1, with
    MOSAIC_JOINT_CALL_GOLDEN_REASON required): the scenario's first call
    in this process records it (see the section comment) and copies the
    VCF to ``GOLDEN_VCF_DIR``; ``reference`` says what the text was checked
    against (default: nothing but the Rust engine's own output). Returns
    the output's digest."""
    digest = vcf_text_digest(vcf_path)
    closing = closing_lines(stderr) if stderr is not None else None
    if golden_recording() and scenario not in _GOLDEN_RECORDED:
        _record_golden(scenario, vcf_path, digest, closing, reference)
        return digest
    want = _load_goldens().get("scenarios", {}).get(scenario)
    if want is None:
        raise AssertionError(
            f"golden {scenario}: no digest recorded in {GOLDEN_DIGESTS}; "
            f"record it with {GOLDEN_RECORD_ENV}=1 and "
            f"{GOLDEN_REASON_ENV}=<why> (the re-recording rule: "
            "re-record only when an intended output change is being made)"
        )
    got = {k: digest[k] for k in DIGEST_KEYS}
    exp = {k: want.get(k) for k in DIGEST_KEYS}
    if got != exp:
        msg = [
            f"golden {scenario}: the output {vcf_path} differs from the "
            "recorded digest",
            f"  recorded: {exp}",
            f"  output:   {got}",
        ]
        copy = os.path.join(GOLDEN_VCF_DIR, f"{scenario}.vcf.gz")
        if os.path.exists(copy):
            msg.append(f"  first differing lines against {copy}:")
            msg += _first_differences(copy, vcf_path)
        else:
            msg.append(f"  (no local golden copy at {copy})")
        raise AssertionError("\n".join(msg))
    if closing is not None and "closing_lines" in want:
        assert closing == want["closing_lines"], (
            f"golden {scenario}: the closing lines differ\n  recorded: "
            f"{want['closing_lines']}\n  output:   {closing}"
        )
    return digest


# ── Ingest signatures ────────────────────────────────────────────────────────
#
# A frozen digest of what tests/integration/test_rust_engine.py (and the
# R5 combine test of tests/unit/test_rust_engine_driver.py) compared
# between the Python and the Rust ingest workers, per case: the one-shard
# row oracle's local files (names, schemas, slice indexes, every column's
# values with floats as bits) and result dicts, the three 2-sample batches'
# data files (``parquet_file_signature``: rows, row groups, schema, sort
# declaration, statistics columns; not bytes, which the brotli library
# decides), sample_qc and banner lines, and the per-shard lines. The
# reference for the Rust worker once the Python worker is gone: each
# case was first recorded in the same test right after the two workers'
# comparison passed. The digests are tracked (INGEST_GOLDEN_DIGESTS); a
# local copy per case (the case's detail as JSON and, for the row oracle,
# the worker's local files) lives under the git-ignored
# test_data/golden_ingest/ (INGEST_GOLDEN_DIR), used to show the first
# differences on a mismatch.
#
# Floats are digested as their IEEE 754 bits, so a digest is per platform
# (R4 saw one gtc double 1 ULP apart between macOS and Linux libm `pow`):
# each recording names its platform.
#
# Record mode: MOSAIC_INGEST_GOLDEN_RECORD=1 with
# MOSAIC_INGEST_GOLDEN_REASON=<why> (required), under the joint-call
# goldens' rules: an unchanged entry is left exactly as it is, a changed
# one is restamped, a case without a digest fails.

INGEST_GOLDEN_DIGESTS = os.path.join(
    REPO_ROOT, "tests", "golden", "ingest_signatures.json"
)
INGEST_GOLDEN_DIR = os.path.join(REPO_ROOT, "test_data", "golden_ingest")
INGEST_GOLDEN_RECORD_ENV = "MOSAIC_INGEST_GOLDEN_RECORD"
INGEST_GOLDEN_REASON_ENV = "MOSAIC_INGEST_GOLDEN_REASON"
INGEST_GOLDEN_FORMAT = 1
INGEST_GOLDEN_DIFF_LINES = 10
INGEST_GOLDEN_RULE = (
    "re-record only when an intended output change is being made"
)
INGEST_GOLDEN_NOTE = (
    "SHA-256 of a canonical serialization (jc_testlib.canonical_digest: "
    "JSON with sorted keys, floats as IEEE 754 bits, bytes as hex) of what "
    "the Python and Rust ingest workers' comparison tests compare, per "
    "case, with a few readable counts; each case's 'recorded' says when, "
    "where (the platform: floats are bit-exact, so a digest is per "
    "platform) and against what it was recorded. Recorded with "
    f"{INGEST_GOLDEN_RECORD_ENV}=1 and {INGEST_GOLDEN_REASON_ENV}=<why> "
    "(jc_testlib.assert_ingest_signature); the local copies are under "
    "test_data/golden_ingest/."
)
# What a recording made now stamps as its reference. The 26 cases recorded
# in K6 say "the Python ingest worker and the Rust ingest worker compared
# in the same test, equal" and keep that provenance while their digests
# stand; since the Python worker went there is no second worker, so a
# re-recording (made only for an intended ingest output change) is the
# worker's own output, and its required reason says why it changed.
INGEST_WORKER_REFERENCE = (
    "the Rust ingest worker's output, with no second implementation to "
    "compare with (the Python worker was removed in K6)"
)
_INGEST_GOLDEN_RECORDED: set = set()


def ingest_golden_recording() -> bool:
    return _golden_flag(INGEST_GOLDEN_RECORD_ENV)


def canonical(x: Any) -> Any:
    """``x`` as JSON-able data, deterministically: floats (Python or numpy)
    as ``f64:<hex bits>`` (so -0.0, NaN and every ULP count), bytes as
    ``bytes:<hex>``, tuples and lists as lists, sets sorted, dict keys
    that are not strings as their canonical JSON text."""
    if isinstance(x, bool) or x is None or isinstance(x, str):
        return x
    if isinstance(x, (float, np.floating)):
        return "f64:" + struct.pack("<d", float(x)).hex()
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (bytes, bytearray)):
        return "bytes:" + bytes(x).hex()
    if isinstance(x, (list, tuple)):
        return [canonical(v) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted((canonical(v) for v in x), key=lambda v: json.dumps(v))
    if isinstance(x, dict):
        out = {}
        for k, v in x.items():
            key = k if isinstance(k, str) else json.dumps(canonical(k))
            out[key] = canonical(v)
        return out
    raise TypeError(f"canonical: unsupported {type(x).__name__}: {x!r}")


def canonical_digest(x: Any) -> str:
    """SHA-256 of ``canonical(x)`` as compact JSON with sorted keys."""
    text = json.dumps(canonical(x), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def table_column_digests(table: Any) -> Dict[str, str]:
    """{column: SHA-256 of its values} of a pyarrow Table, by
    ``_hash_array`` (layout independent, floats as bits, nulls kept)."""
    out = {}
    for name in table.schema.names:
        h = hashlib.sha256()
        for chunk in table.column(name).chunks:
            _hash_array(h, chunk)
        out[name] = h.hexdigest()
    return out


def _ingest_case_dir(case: str) -> str:
    return os.path.join(INGEST_GOLDEN_DIR, case.replace("/", "__"))


def _save_ingest_copy(
    case: str, detail: Any, files: Optional[Dict[str, str]]
) -> None:
    d = _ingest_case_dir(case)
    if os.path.isdir(d):
        shutil.rmtree(d)
    os.makedirs(d)
    with gzip.open(os.path.join(d, "detail.json.gz"), "wt") as fh:
        json.dump(canonical(detail), fh, sort_keys=True)
    for name, path in (files or {}).items():
        shutil.copyfile(path, os.path.join(d, name))


def load_ingest_copy(case: str) -> Optional[Tuple[Any, str]]:
    """(the recorded detail, canonical; the copy's directory) of a case's
    local copy, or None."""
    d = _ingest_case_dir(case)
    path = os.path.join(d, "detail.json.gz")
    if not os.path.exists(path):
        return None
    with gzip.open(path, "rt") as fh:
        return json.load(fh), d


def _first_detail_differences(
    a: Any, b: Any, where: str = "", out: Optional[List[str]] = None
) -> List[str]:
    """Where two canonical structures differ (the first
    INGEST_GOLDEN_DIFF_LINES places, depth first; a list's first differing
    index)."""
    if out is None:
        out = []
    if len(out) >= INGEST_GOLDEN_DIFF_LINES or a == b:
        return out
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(
                    f"  {where}/{k}: only in the "
                    f"{'output' if k not in a else 'golden'}"
                )
            else:
                _first_detail_differences(a[k], b[k], f"{where}/{k}", out)
            if len(out) >= INGEST_GOLDEN_DIFF_LINES:
                break
    elif isinstance(a, list) and isinstance(b, list):
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                if isinstance(x, (dict, list)) and isinstance(y, type(x)):
                    _first_detail_differences(x, y, f"{where}[{i}]", out)
                else:
                    out.append(
                        f"  {where}[{i}]: golden {str(x)[:200]} != output "
                        f"{str(y)[:200]}"
                    )
                return out
        out.append(
            f"  {where}: {len(a)} items in the golden, {len(b)} in "
            "the output"
        )
    else:
        out.append(
            f"  {where}: golden {str(a)[:200]} != output " f"{str(b)[:200]}"
        )
    return out


def assert_ingest_signature(
    case: str,
    detail: Any,
    counts: Dict[str, Any],
    files: Optional[Dict[str, str]] = None,
    reference: Optional[str] = None,
    compare: Optional[Callable[[Any, str], List[str]]] = None,
) -> str:
    """Check ``detail`` (what the test compares, any structure
    ``canonical`` takes) against the case's recorded digest
    (INGEST_GOLDEN_DIGESTS) and ``counts`` against its recorded counts. A
    case with no recorded digest fails. On a mismatch, the first
    differences against the local copy when one exists: ``compare(golden
    detail, copy dir)`` when given (e.g. rows of the copied local files),
    else the canonical details side by side.

    Record mode (MOSAIC_INGEST_GOLDEN_RECORD=1 with
    MOSAIC_INGEST_GOLDEN_REASON): the case's first call in this process
    records it -- call it only after the test's own cross-worker
    comparison passed -- and keeps a local copy (``detail`` and
    ``files`` {name: path}) under INGEST_GOLDEN_DIR; an entry whose digest
    and counts are unchanged is left as it is. Returns the digest."""
    digest = canonical_digest(detail)
    counts = canonical(counts)
    store = _load_golden_file(INGEST_GOLDEN_DIGESTS, INGEST_GOLDEN_FORMAT)
    want = store.get("scenarios", {}).get(case)
    if ingest_golden_recording() and case not in _INGEST_GOLDEN_RECORDED:
        reason = _golden_reason(
            case,
            "ingest signature",
            INGEST_GOLDEN_RECORD_ENV,
            INGEST_GOLDEN_REASON_ENV,
            INGEST_GOLDEN_RULE,
        )
        same = (
            want is not None
            and want.get("sha256") == digest
            and want.get("counts") == counts
        )
        if same:
            state = "unchanged"
        else:
            store.setdefault("scenarios", {})[case] = {
                "sha256": digest,
                "counts": counts,
                "recorded": _golden_provenance(
                    reference or INGEST_WORKER_REFERENCE, reason
                ),
            }
            _write_golden_file(
                INGEST_GOLDEN_DIGESTS,
                store,
                INGEST_GOLDEN_FORMAT,
                INGEST_GOLDEN_NOTE,
            )
            state = (
                "new" if want is None else f"recorded, was {want['sha256']}"
            )
        if not same or load_ingest_copy(case) is None:
            _save_ingest_copy(case, detail, files)
        print(f"ingest signature {case}: sha256={digest} {counts} ({state})")
        _INGEST_GOLDEN_RECORDED.add(case)
        return digest
    if want is None:
        raise AssertionError(
            f"ingest signature {case}: no digest recorded in "
            f"{INGEST_GOLDEN_DIGESTS}; record it with "
            f"{INGEST_GOLDEN_RECORD_ENV}=1 and {INGEST_GOLDEN_REASON_ENV}="
            f"<why> (the re-recording rule: {INGEST_GOLDEN_RULE})"
        )
    if want.get("sha256") != digest or want.get("counts") != counts:
        msg = [
            f"ingest signature {case}: the output differs from the "
            "recorded digest",
            f"  recorded: {want.get('sha256')} {want.get('counts')}",
            f"  output:   {digest} {counts}",
        ]
        copy = load_ingest_copy(case)
        if copy is None:
            msg.append(f"  (no local copy at {_ingest_case_dir(case)})")
        else:
            golden, cdir = copy
            msg.append(f"  first differences against {cdir}:")
            lines = (
                compare(golden, cdir)[:INGEST_GOLDEN_DIFF_LINES]
                if compare is not None
                else []
            )
            # A difference the callback does not see (the result dict, a
            # count): the canonical details side by side.
            msg += lines or _first_detail_differences(
                golden, canonical(detail)
            )
        raise AssertionError("\n".join(msg))
    return digest


def genotype_matrix_cells(m: Any) -> Dict[Tuple[int, int], Tuple]:
    """Flatten a GenotypeMatrix to {(position, sample_id): (a1, a2, gq)}.

    Alleles are decoded to their strings (None = no call, "?" = UNRESOLVED)
    so matrices whose per-site allele order differs (other batch layouts)
    compare cell by cell.
    """
    out: Dict[Tuple[int, int], Tuple] = {}
    for i, pos in enumerate(m.positions.tolist()):
        alleles = m.alleles[i]
        g1 = m.gt_1[i].tolist()
        g2 = m.gt_2[i].tolist()
        gq = m.gq[i].tolist()
        for j, sid in enumerate(m.sample_ids.tolist()):
            cell = []
            for g in (g1[j], g2[j]):
                cell.append(None if g == -1 else "?" if g < 0 else alleles[g])
            out[(int(pos), int(sid))] = (cell[0], cell[1], gq[j])
    return out


def sample_qc_rows(conn: Any) -> Dict[Tuple[int, int, int], Tuple]:
    """{(batch_id, sample_id, chromosome_index): counters} from sample_qc."""
    from mosaic.init_db import SAMPLE_QC_COLUMNS

    with conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id, sample_id, chromosome_index, "
            + ", ".join(SAMPLE_QC_COLUMNS)
            + " FROM sample_qc"
        )
        return {tuple(r[:3]): tuple(r[3:]) for r in cur.fetchall()}


# ── PG checkpoint-state probes (for crash/resume assertions) ─────────────────


def pg_counts(conn: Any) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with conn.cursor() as cur:
        for tbl in ("site_info", "alleles", "ref_blocks"):
            cur.execute(f"SELECT count(*) FROM {tbl}")
            out[tbl] = cur.fetchone()[0]
    return out


def shard_completion_count(conn: Any) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM shard_completion")
        return cur.fetchone()[0]


def batch_statuses(conn: Any) -> List[Tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id, status FROM ingestion_progress ORDER BY batch_id"
        )
        return list(cur.fetchall())


def tier_completion_count(conn: Any) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT count(*) FROM tier_completion")
        return cur.fetchone()[0]


def tier_completion_has(
    conn: Any,
    metric: str,
    tier: int,
    group_id: int,
    agg_run_id: Optional[int] = None,
) -> bool:
    """True iff a checkpoint row exists for this reduce unit (any chromosome).

    Filters by ``agg_run_id`` — defaulting to the most recent run — because the
    checkpoint tables accumulate rows across runs (each ``agg_run_id`` is isolated),
    so an earlier baseline run's row must not be mistaken for the crash run's.
    """
    with conn.cursor() as cur:
        if agg_run_id is None:
            cur.execute("SELECT max(agg_run_id) FROM aggregation_progress")
            agg_run_id = cur.fetchone()[0]
        cur.execute(
            "SELECT count(*) FROM tier_completion WHERE agg_run_id=%s AND "
            "metric=%s AND tier=%s AND group_id=%s",
            (agg_run_id, metric, tier, group_id),
        )
        return cur.fetchone()[0] > 0


def finalize_markers(
    conn: Any, metric: str, agg_run_id: Optional[int] = None
) -> set:
    """Chromosome indices whose ``metric`` finalize checkpoint row exists for
    the run (default: the most recent). A (metric, chromosome) finalize unit's
    sentinel is the tier_completion key (chromosome_index=c, tier=-1,
    group_id=0), committed with its partition swap."""
    with conn.cursor() as cur:
        if agg_run_id is None:
            cur.execute("SELECT max(agg_run_id) FROM aggregation_progress")
            agg_run_id = cur.fetchone()[0]
        cur.execute(
            "SELECT chromosome_index FROM tier_completion WHERE "
            "agg_run_id=%s AND metric=%s AND tier=-1 AND group_id=0",
            (agg_run_id, metric),
        )
        return {c for (c,) in cur.fetchall()}


def finalize_marker_exists(
    conn: Any, metric: str, agg_run_id: Optional[int] = None
) -> bool:
    """True iff ``metric``'s finalize is checkpointed for EVERY contig in the
    run (default: the most recent) -- see ``finalize_markers``."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM contigs")
        contigs = {c for (c,) in cur.fetchall()}
    done = finalize_markers(conn, metric, agg_run_id)
    return bool(contigs) and contigs <= done


def clear_pg_aggregates(conn: Any) -> None:
    """Empty the finalized PG aggregate tables (site_info / ref_blocks /
    alleles) without touching the Iceberg partials or the checkpoint
    tables (TRUNCATE of a partitioned parent empties every partition).

    Used so a resume test cannot false-pass: aggregate.py finalizes to PG only at
    the very end, so a crash mid-reduce leaves the PREVIOUS run's values in place.
    Clearing first means the resumed run must genuinely reconstruct them.
    """
    with conn.cursor() as cur:
        cur.execute("TRUNCATE site_info")
        cur.execute("TRUNCATE ref_blocks")
        cur.execute("TRUNCATE alleles")
    conn.commit()


# The PG tables `mosaic aggregate` rebuilds partition by partition.
AGGREGATE_PG_TABLES = ("site_info", "ref_blocks", "alleles")


def partition_layout(conn: Any) -> Dict[str, Dict[int, str]]:
    """{table: {chromosome_index: attached partition name}} for the three
    aggregate tables, read from pg_inherits + the partition bound (the same
    lookup the finalize uses), never from the names."""
    out: Dict[str, Dict[int, str]] = {}
    with conn.cursor() as cur:
        for table in AGGREGATE_PG_TABLES:
            cur.execute(
                "SELECT c.relname, pg_get_expr(c.relpartbound, c.oid) "
                "FROM pg_inherits i JOIN pg_class c ON c.oid = i.inhrelid "
                "WHERE i.inhparent = %s::regclass",
                (table,),
            )
            parts: Dict[int, str] = {}
            for name, bound in cur.fetchall():
                m = re.fullmatch(r"FOR VALUES IN \((\d+)\)", bound)
                assert m, (table, name, bound)
                c_idx = int(m.group(1))
                assert c_idx not in parts, (table, c_idx, name, parts)
                parts[c_idx] = name
            out[table] = parts
    return out


def generation_tables(conn: Any) -> Dict[str, bool]:
    """{name: attached?} of every ``{table}_c<i>_r<run>`` generation table."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT relname, relispartition FROM pg_class "
            "WHERE relkind = 'r' AND relname ~ "
            "'^(site_info|ref_blocks|alleles)_c[0-9]+_r[0-9]+$'"
        )
        return dict(cur.fetchall())


def latest_agg_run_id(conn: Any) -> Optional[int]:
    """The most recent aggregation run (None before any)."""
    with conn.cursor() as cur:
        cur.execute("SELECT max(agg_run_id) FROM aggregation_progress")
        return cur.fetchone()[0]


def assert_single_generation(conn: Any, agg_run_id: int) -> None:
    """Exactly one partition per contig per aggregate table, every one the
    generation ``_r{agg_run_id}``, and no detached generation tables."""
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM contigs")
        contigs = sorted(c for (c,) in cur.fetchall())
    layout = partition_layout(conn)
    for table in AGGREGATE_PG_TABLES:
        assert sorted(layout[table]) == contigs, (table, layout[table])
        for c_idx, name in layout[table].items():
            assert name == f"{table}_c{c_idx}_r{agg_run_id}", (table, name)
    detached = [
        n for n, attached in generation_tables(conn).items() if not attached
    ]
    assert not detached, f"detached generation tables left: {detached}"


def cohort_generation(conn: Any) -> Optional[Dict[str, Any]]:
    """The cohort_generation view's row (the latest aggregation run), or
    None before any run."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT agg_run_id, batch_ids, parent_run_id, status, updated_at "
            "FROM cohort_generation"
        )
        row = cur.fetchone()
        if row is None:
            return None
        return dict(zip([d[0] for d in cur.description], row))


def dense_root(catalog: Any, agg_run_id: int) -> Any:
    """The run's dense ref_blocks root (tier_ref_blocks_dense rows) as an
    Arrow table sorted by (chromosome_index, position)."""
    from pyiceberg.expressions import EqualTo

    table = catalog.load_table(f"{TEST_NAMESPACE}.tier_ref_blocks_dense")
    rows = table.scan(row_filter=EqualTo("agg_run_id", agg_run_id)).to_arrow()
    return rows.sort_by(
        [("chromosome_index", "ascending"), ("position", "ascending")]
    )


# The aggregate staging tables (tier_* + the dense ref_blocks root).
STAGING_TABLES = [
    "tier_site_info",
    "tier_allele_counts",
    "tier_ref_blocks",
    "tier_ref_blocks_dense",
]


# The agg_* mirrors (written by the finalize barriers).
AGG_TABLES = ["agg_site_info", "agg_allele_counts", "agg_ref_blocks"]

# Every Iceberg table `mosaic aggregate` writes data files to.
AGGREGATE_TABLES = STAGING_TABLES + AGG_TABLES


def aggregate_file_etags(catalog: Any) -> Dict[str, List[Tuple[str, str]]]:
    """iceberg_file_etags of the staging tables and the agg_* mirrors: the
    live data files of every table the aggregate writes, by URI and S3
    ETag. Their keys fold in the agg_run_id, so two runs compare equal only
    under the same id (see reset_aggregate_runs)."""
    return iceberg_file_etags(catalog, AGGREGATE_TABLES)


def reset_aggregate_runs(conn: Any, catalog: Any) -> None:
    """Return the aggregate state to "after ingest": forget every run
    (clear_lineage), empty the agg_* mirrors, put every aggregate partition
    back to an empty ``_r0`` generation (as `mosaic init` leaves it; detached
    generation tables dropped) and restart the agg_run_id sequence, so the
    next runs get the same ids as the runs after a previous reset. A run's
    staging and mirror keys (and the staging rows' agg_run_id) fold in its
    id, so a resumed run can then be compared with a clean run file by file
    (aggregate_file_etags)."""
    from pyiceberg.expressions import AlwaysTrue

    from mosaic.init_db import detached_generation_tables

    clear_lineage(conn, catalog)
    for name in AGG_TABLES:
        table = catalog.load_table(f"{TEST_NAMESPACE}.{name}")
        if table.current_snapshot() is not None:
            table.delete(AlwaysTrue())
    layout = partition_layout(conn)
    with conn.cursor() as cur:
        for name, _run in detached_generation_tables(cur):
            cur.execute(f"DROP TABLE IF EXISTS {name}")
        for table, parts in layout.items():
            for c_idx, part in sorted(parts.items()):
                r0 = f"{table}_c{int(c_idx)}_r0"
                if part == r0:
                    cur.execute(f"TRUNCATE {part}")
                    continue
                cur.execute(f"DROP TABLE {part}")
                cur.execute(
                    f"CREATE TABLE {r0} PARTITION OF {table} "
                    f"FOR VALUES IN ({int(c_idx)})"
                )
        cur.execute(
            "SELECT setval(pg_get_serial_sequence('aggregation_progress', "
            "'agg_run_id'), 1, false)"
        )
    conn.commit()


def agg_run(conn: Any, agg_run_id: int) -> Dict[str, Any]:
    """One aggregation_progress row as a dict."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT * FROM aggregation_progress WHERE agg_run_id=%s",
            (agg_run_id,),
        )
        row = cur.fetchone()
        assert row is not None, f"no aggregation run {agg_run_id}"
        return dict(zip([d[0] for d in cur.description], row))


def set_completed_batches(conn: Any, batch_ids: List[int]) -> None:
    """Make exactly ``batch_ids`` the completed ingest batches: the others
    are set to status 'held', which `mosaic aggregate` (status 'complete'
    only) does not see. Lets one staged fixture play "batches arrive over
    time" without re-ingesting: every aggregate read is filtered by batch_id,
    so a held batch's partials are invisible. Only ever GROW the set within
    one lineage (call clear_lineage before shrinking it): a run is keyed by
    its batch set, so returning to an older set would find that set's old
    complete run and report it up to date."""
    with conn.cursor() as cur:
        cur.execute(
            "UPDATE ingestion_progress SET status = CASE WHEN "
            "batch_id = ANY(%s) THEN 'complete' ELSE 'held' END",
            (list(batch_ids),),
        )


def clear_lineage(conn: Any, catalog: Any) -> None:
    """Forget every aggregation run: delete all staging rows (tier_* and
    the dense root) and every aggregation_progress / tier_completion row, so
    the next plain `mosaic aggregate` is a full run with no parent. The PG
    aggregate partitions stay attached (the next finalize swaps them by
    pg_inherits, whatever their names); new runs get new agg_run_ids."""
    from pyiceberg.expressions import AlwaysTrue

    for name in STAGING_TABLES:
        table = catalog.load_table(f"{TEST_NAMESPACE}.{name}")
        if table.current_snapshot() is not None:
            table.delete(AlwaysTrue())
    with conn.cursor() as cur:
        cur.execute("DELETE FROM tier_completion")
        cur.execute("DELETE FROM aggregation_progress")


def staging_rows(catalog: Any, name: str) -> Any:
    """Every row of one staging table (Arrow), for retention assertions."""
    return catalog.load_table(f"{TEST_NAMESPACE}.{name}").scan().to_arrow()


def agg_statuses(conn: Any) -> List[Tuple[int, str]]:
    with conn.cursor() as cur:
        cur.execute(
            "SELECT agg_run_id, status FROM aggregation_progress "
            "ORDER BY agg_run_id"
        )
        return list(cur.fetchall())


def poll_until(
    predicate, timeout: float = 120, interval: float = 0.25
) -> bool:
    """Poll `predicate()` until truthy or timeout. Returns the final truthiness.

    Opens a fresh PG connection per call inside the predicate as needed; used by
    the signal-kill tests to wait for a partial checkpoint state mid-run.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


# ── the lake oracle: batch partials re-derived from the per-sample tables ───
#
# The removal substrate: every batch_site_info, batch_allele_counts and
# batch_ref_blocks row of a completed batch is recomputed from
# sample_site_info and sample_ref_blocks ALONE (nothing from the gVCFs,
# nothing from mosaic.ingest), which is what sample removal needs. The
# acceptance test of the per-record contribution columns and the regression
# oracle for the whole-sample layout and sample removal. With
# `mosaic remove` batches present, the batches that held a removed
# sample are checked together with the removal batches: their NET partials
# against the recompute over the live rows (the "net" mode).


def _parse_region(conn: Any, region: Optional[str]) -> Tuple[Any, int, int]:
    """(chromosome_index or None, lo, hi) of a batch's --region (NULL =
    every chromosome; a bare chromosome = all of it)."""
    if not region:
        return None, 0, 2**31 - 1
    chrom, _, rng = region.partition(":")
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM contigs WHERE chromosome = %s", (chrom,))
        c_idx = cur.fetchone()[0]
    if not rng:
        return c_idx, 0, 2**31 - 1
    lo, _, hi = rng.replace(",", "").partition("-")
    return c_idx, int(lo), int(hi)


def completed_batches(conn: Any) -> Dict[int, Dict[str, Any]]:
    """{batch_id: {"sample_ids", "chrom", "lo", "hi"}} of the completed
    ingest batches (ingestion_progress.sample_ids and region; `mosaic remove`
    batches, kind 'removal', are not ingest batches)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT batch_id, sample_ids, region FROM ingestion_progress "
            "WHERE status = 'complete' AND kind = 'ingest' ORDER BY batch_id"
        )
        rows = cur.fetchall()
    out: Dict[int, Dict[str, Any]] = {}
    for batch_id, sample_ids, region in rows:
        c_idx, lo, hi = _parse_region(conn, region)
        out[batch_id] = {
            "sample_ids": list(sample_ids),
            "chrom": c_idx,
            "lo": lo,
            "hi": hi,
        }
    return out


def completed_removals(conn: Any) -> Dict[int, List[int]]:
    """{batch_id: removed sample ids} of the completed `mosaic remove`
    batches (kind 'removal'): their partials are those samples' share,
    negated."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT p.batch_id, array_agg(s.id ORDER BY s.id) "
            "FROM ingestion_progress p JOIN samples s "
            "ON s.sample_name = ANY(p.sample_names) "
            "WHERE p.status = 'complete' AND p.kind = 'removal' "
            "GROUP BY p.batch_id ORDER BY p.batch_id"
        )
        return {int(b): [int(i) for i in ids] for b, ids in cur.fetchall()}


def _assign_batches(
    batches: Dict[int, Dict[str, Any]],
    sample_id: np.ndarray,
    chrom: np.ndarray,
    position: np.ndarray,
) -> np.ndarray:
    """The batch of each per-sample row (sample_site_info /
    sample_ref_blocks carry no batch_id): the batch holding the row's sample
    and chromosome; of several (one sample set ingested over several
    regions), the one whose region is nearest the row's position (a subset
    gVCF also keeps records that overlap its region's edge). -1 = none.
    Assumes a sample's batches cover disjoint, separated regions."""
    out = np.full(len(sample_id), -1, dtype=np.int64)
    best = np.full(len(sample_id), np.inf)
    for batch_id, b in batches.items():
        ok = np.isin(sample_id, b["sample_ids"])
        if b["chrom"] is not None:
            ok &= chrom == b["chrom"]
        dist = np.maximum(b["lo"] - position, 0) + np.maximum(
            position - b["hi"], 0
        )
        dist = np.where(ok, dist.astype(np.float64), np.inf)
        better = dist < best
        out[better] = batch_id
        best[better] = dist[better]
    return out


def _lake_table(catalog: Any, table: str) -> Any:
    return catalog.load_table(f"{TEST_NAMESPACE}.{table}").scan().to_arrow()


def _ulps(a: float, b: float) -> float:
    """|a - b| in units of the last place of the larger magnitude."""
    if a == b:
        return 0.0
    import math

    return abs(a - b) / math.ulp(max(abs(a), abs(b)))


def _net_partial_rows(
    rows: List[Dict[str, Any]], key: Callable[[Dict[str, Any]], Any]
) -> Dict[Any, Dict[str, Any]]:
    """Sum partial rows (one or several batches: ingest and removal) per
    key: numbers and lists element-wise; the floats with math.fsum."""
    import math

    groups: Dict[Any, List[Dict[str, Any]]] = {}
    for r in rows:
        groups.setdefault(key(r), []).append(r)
    out: Dict[Any, Dict[str, Any]] = {}
    for k, rs in groups.items():
        if len(rs) == 1:
            out[k] = rs[0]
            continue
        net: Dict[str, Any] = {}
        for name, v in rs[0].items():
            vals = [r[name] for r in rs]
            if isinstance(v, list):
                net[name] = [
                    (math.fsum if isinstance(v[0], float) else sum)(
                        x[j] for x in vals
                    )
                    for j in range(len(v))
                ]
            elif isinstance(v, float):
                net[name] = math.fsum(vals)
            elif name in ("batch_id", "chromosome_index", "position") or (
                isinstance(v, str)
            ):
                net[name] = v
            else:
                net[name] = sum(vals)
        out[k] = net
    return out


def _check_site_info(
    rows: Dict[Tuple[int, int], List[Dict[str, Any]]],
    partial: Dict[Tuple[int, int], Dict[str, Any]],
    stats: Dict[str, int],
    net: bool = False,
) -> None:
    """batch_site_info vs the per-site sums of the rows' contribution
    columns: integers exact (n_records = the site's rows; gtc = the sum of
    each row's fixed-point gtc, gtc_fixed, exactly); qual / mqsq_1 equal to
    the exact sum (fsum) for one or two rows, else within n - 1 ULPs (ingest
    sums in record order). ``net``: the partial is the net of several
    batches (ingest + removal), qual / mqsq_1 within a relative 1e-9; a site
    no live row has must net to n_records 0 (and every other column to 0,
    gtc exactly)."""
    import math

    if net:
        dead = {k: p for k, p in partial.items() if p["n_records"] == 0}
        for k, p in dead.items():
            assert p["dp"] == 0 and p["mqsq_2"] == 0, (k, p)
            assert p["sb"] == [0, 0, 0, 0], (k, p)
            for col in ("qual", "mqsq_1"):
                assert abs(p[col]) <= 1e-6, (k, col, p)
            assert p["gtc"] == [0, 0, 0], (k, p)
        stats["dead_sites"] += len(dead)
        partial = {k: p for k, p in partial.items() if k not in dead}
    assert set(rows) == set(partial), (
        f"site keys differ: {len(set(rows) - set(partial))} only in the "
        f"lake, {len(set(partial) - set(rows))} only in batch_site_info"
    )
    for key, rs in rows.items():
        p = partial[key]
        n = len(rs)
        assert p["n_records"] == n, (key, p["n_records"], n)
        assert p["dp"] == sum(r["site_dp"] for r in rs), (key, p, rs)
        assert p["mqsq_2"] == sum(r["mqsq_2"] for r in rs), (key, p)
        assert p["sb"] == [sum(r["sb"][j] for r in rs) for j in range(4)], (
            key,
            p,
        )
        assert p["gtc"] == [
            sum(gtc_fixed(r["gtc"][j]) for r in rs) for j in range(3)
        ], (key, p["gtc"], [r["gtc"] for r in rs])
        pairs = [(p[c], math.fsum(r[c] for r in rs), c) for c in ("qual",)]
        pairs += [(p["mqsq_1"], math.fsum(r["mqsq_1"] for r in rs), "mqsq_1")]
        for got, want, col in pairs:
            if net:
                ok = abs(got - want) <= 1e-9 * max(1.0, abs(want))
                assert ok, (key, col, got, want)
                stats[f"{col}_net"] += 1
                continue
            ulps = _ulps(got, want)
            # 0.0 + a (+ b) is correctly rounded: exact up to two rows.
            assert ulps <= (0 if n <= 2 else n - 1), (key, col, got, want)
            stats[f"{col}_exact" if ulps == 0 else f"{col}_ulp"] += 1
        # FORMAT/DP is not the site's dp contribution (INFO/DP is).
        if p["dp"] != sum(r["dp"] or 0 for r in rs):
            stats["format_dp_differs"] += 1
        stats["sites"] += 1


def _check_allele_counts(
    rows: Dict[Tuple[int, int], List[Dict[str, Any]]],
    partial: Dict[Tuple[int, int, str], Dict[str, Any]],
    stats: Dict[str, int],
    net: bool = False,
) -> None:
    """batch_allele_counts vs the rows: each row lists its record's own
    alleles (E3); count = the gt_1 / gt_2 hits through each row's own list,
    n_listed = the rows listing the allele; every allele a row lists (zero
    counts too) has a partial row. ``net``: an allele no live row lists must
    net to n_listed 0 and count 0, and is dropped from the comparison."""
    want: Dict[Tuple[int, int, str], List[int]] = {}
    for (c, pos), rs in rows.items():
        for r in rs:
            for a in set(r["alleles"]):
                want.setdefault((c, pos, a), [0, 0])[1] += 1
            for g in (r["gt_1"], r["gt_2"]):
                if g is not None:
                    want[(c, pos, r["alleles"][g])][0] += 1
    have = {k: [p["count"], p["n_listed"]] for k, p in partial.items()}
    if net:
        dead = {k: v for k, v in have.items() if v[1] == 0}
        assert all(v == [0, 0] for v in dead.values()), list(dead.items())[:5]
        stats["dead_alleles"] += len(dead)
        have = {k: v for k, v in have.items() if k not in dead}
    assert set(want) == set(have), (
        f"allele keys differ: {sorted(set(want) - set(have))[:5]} only "
        f"in the lake, {sorted(set(have) - set(want))[:5]} only in "
        "batch_allele_counts"
    )
    bad = [(k, want[k], have[k]) for k in want if want[k] != have[k]]
    assert (
        not bad
    ), f"allele (count, n_listed) differ (lake, partial): {bad[:5]}"
    stats["alleles"] += len(want)
    stats["zero_count_alleles"] += sum(1 for v in want.values() if v[0] == 0)


def _dense_ref_sums(
    starts: np.ndarray,
    ends: np.ndarray,
    dp: np.ndarray,
    sc: np.ndarray,
    lo: int,
    hi: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-position (dp, sample_counts) sums over [lo, hi] of inclusive
    ranges (clipped), positions whose sum is (0, 0) dropped. np.add.at on a
    difference array (independent of mosaic.ingest's bincount)."""
    s = np.maximum(starts, lo)
    e = np.minimum(ends, hi)
    keep = s <= e
    s, e, dp, sc = s[keep], e[keep], dp[keep], sc[keep]
    span = hi - lo + 1
    sums = []
    for w in (dp, sc):
        diff = np.zeros(span + 1, dtype=np.int64)
        np.add.at(diff, s - lo, w)
        np.add.at(diff, e - lo + 1, -w)
        sums.append(np.cumsum(diff[:span]))
    covered = (sums[0] != 0) | (sums[1] != 0)
    return (
        np.flatnonzero(covered) + lo,
        sums[0][covered],
        sums[1][covered],
    )


def _net_dense(
    c: np.ndarray, pos: np.ndarray, dp: np.ndarray, sc: np.ndarray
) -> Tuple[np.ndarray, ...]:
    """Dense partial rows of several batches summed per (chromosome,
    position), net (0, 0) positions dropped; sorted."""
    if len(pos) == 0:
        return c, pos, dp, sc
    key = c.astype(np.int64) * (1 << 32) + pos.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    d = np.zeros(len(uniq), dtype=np.int64)
    s = np.zeros(len(uniq), dtype=np.int64)
    np.add.at(d, inv, dp)
    np.add.at(s, inv, sc)
    keep = (d != 0) | (s != 0)
    uniq = uniq[keep]
    return (uniq >> 32).astype(np.int64), uniq & 0xFFFFFFFF, d[keep], s[keep]


def _check_ref_blocks(
    catalog: Any,
    batch_ids: List[int],
    seg: Dict[str, np.ndarray],
    stats: Dict[str, int],
    chunk: int = 5_000_000,
) -> None:
    """batch_ref_blocks of one batch (or the net of several: ingest +
    removal) vs the dense sum of the live samples' contributions: every
    sample_ref_blocks row (dp, sample_counts) plus every deletion row's
    del_ref_spans (site_dp, 0). Exact, per position, over every covered
    position (in chunks of ``chunk`` positions); a position whose net is
    (0, 0) has no row."""
    from pyiceberg.expressions import In as _In

    got = (
        catalog.load_table(f"{TEST_NAMESPACE}.batch_ref_blocks")
        .scan(row_filter=_In("batch_id", [int(b) for b in batch_ids]))
        .to_arrow()
    )
    pairs = np.stack(
        [
            got.column("batch_id").to_numpy().astype(np.int64),
            got.column("chromosome_index").to_numpy().astype(np.int64),
            got.column("position").to_numpy().astype(np.int64),
        ]
    )
    assert (
        pairs.shape[1] == np.unique(pairs, axis=1).shape[1]
    ), "duplicate (batch, chromosome, position) rows"
    g_c, g_pos, g_dp, g_sc = _net_dense(
        got.column("chromosome_index").to_numpy().astype(np.int64),
        got.column("position").to_numpy().astype(np.int64),
        got.column("dp").to_numpy(zero_copy_only=False).astype(np.int64),
        got.column("sample_counts")
        .to_numpy(zero_copy_only=False)
        .astype(np.int64),
    )
    for c in sorted(set(seg["c"].tolist()) | set(g_c.tolist())):
        m = seg["c"] == c
        starts, ends = seg["start"][m], seg["end"][m]
        dp, sc = seg["dp"][m], seg["sc"][m]
        gm = g_c == c
        gp, gd, gs = g_pos[gm], g_dp[gm], g_sc[gm]
        lo = int(
            min(
                starts.min() if len(starts) else 2**31,
                gp.min() if len(gp) else 2**31,
            )
        )
        hi = int(
            max(ends.max() if len(ends) else 0, gp.max() if len(gp) else 0)
        )
        for a in range(lo, hi + 1, chunk):
            b = min(a + chunk - 1, hi)
            pos, wdp, wsc = _dense_ref_sums(starts, ends, dp, sc, a, b)
            i, j = np.searchsorted(gp, [a, b + 1])
            if not (
                np.array_equal(pos, gp[i:j])
                and np.array_equal(wdp, gd[i:j])
                and np.array_equal(wsc, gs[i:j])
            ):
                want = dict(zip(pos.tolist(), zip(wdp.tolist(), wsc.tolist())))
                have = dict(
                    zip(
                        gp[i:j].tolist(),
                        zip(gd[i:j].tolist(), gs[i:j].tolist()),
                    )
                )
                diff = sorted(
                    q
                    for q in set(want) | set(have)
                    if want.get(q) != have.get(q)
                )
                raise AssertionError(
                    f"batches {batch_ids} c{c}: batch_ref_blocks differs from "
                    f"the lake at {len(diff)} positions, first: "
                    + ", ".join(
                        f"{q}: lake {want.get(q)} partial {have.get(q)}"
                        for q in diff[:5]
                    )
                )
            stats["ref_positions"] += len(pos)


def _ref_segments(
    srb: Any,
    mask: np.ndarray,
    rows: Dict[Tuple[int, int], List[Dict[str, Any]]],
    stats: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """The ref-block contributions of a set of per-sample rows as inclusive
    ranges: the sample_ref_blocks rows selected by ``mask`` (dp,
    sample_counts) and the deletion rows' del_ref_spans (site_dp, 0)."""
    c = [srb.column("chromosome_index").to_numpy()[mask]]
    start = [srb.column("position_start").to_numpy()[mask].astype(np.int64)]
    end = [srb.column("position_end").to_numpy()[mask].astype(np.int64)]
    dp = [
        srb.column("dp").to_numpy(zero_copy_only=False)[mask].astype(np.int64)
    ]
    sc = [
        srb.column("sample_counts")
        .to_numpy(zero_copy_only=False)[mask]
        .astype(np.int64)
    ]
    sp_c, sp_s, sp_e, sp_dp = [], [], [], []
    for rs in rows.values():
        for r in rs:
            spans = r["del_ref_spans"]
            if not spans:
                continue
            assert r["del_start"] is not None, r  # a deletion carrier
            stats["deletion_rows_with_spans"] += 1
            for s, e in zip(spans[0::2], spans[1::2]):
                assert r["position"] < s <= e, r
                sp_c.append(r["chromosome_index"])
                sp_s.append(s)
                sp_e.append(e)
                sp_dp.append(r["site_dp"])
                stats["split_positions"] += e - s + 1
    c.append(np.array(sp_c, dtype=c[0].dtype))
    start.append(np.array(sp_s, dtype=np.int64))
    end.append(np.array(sp_e, dtype=np.int64))
    dp.append(np.array(sp_dp, dtype=np.int64))
    sc.append(np.zeros(len(sp_c), dtype=np.int64))
    return {
        "c": np.concatenate(c),
        "start": np.concatenate(start),
        "end": np.concatenate(end),
        "dp": np.concatenate(dp),
        "sc": np.concatenate(sc),
    }


def assert_partials_derivable_from_lake(
    conn: Any, catalog: Any
) -> Dict[str, int]:
    """For every completed ingest batch, recompute its batch_site_info,
    batch_allele_counts and batch_ref_blocks rows from sample_site_info and
    sample_ref_blocks alone (rows attributed to batches by sample and
    region, _assign_batches) and assert they equal the partials.

    With completed `mosaic remove` batches: the ingest batches that held a
    removed sample, together with every removal batch, form one group whose
    NET partials (summed per key, net-zero keys checked to be all zero and
    dropped) must equal the recompute over the group's live rows (a removed
    sample's rows excluded: they may still exist before the removal's delete
    step); the other batches are checked one by one as before.
    stats["plain_batches"] / stats["net_batches"] say which mode ran.
    Returns counters (sites, alleles, ref_positions, deletion rows with
    spans, ...) for sanity asserts and the notes."""
    from collections import Counter

    batches = completed_batches(conn)
    removals = completed_removals(conn)
    removed = sorted({s for ids in removals.values() for s in ids})
    stats: Counter = Counter()

    ssi = _lake_table(catalog, "sample_site_info")
    srb = _lake_table(catalog, "sample_ref_blocks")
    si_live = ~np.isin(ssi.column("sample_id").to_numpy(), removed)
    rb_live = ~np.isin(srb.column("sample_id").to_numpy(), removed)
    si_batch = _assign_batches(
        batches,
        ssi.column("sample_id").to_numpy(),
        ssi.column("chromosome_index").to_numpy(),
        ssi.column("position").to_numpy().astype(np.int64),
    )
    rb_batch = _assign_batches(
        batches,
        srb.column("sample_id").to_numpy(),
        srb.column("chromosome_index").to_numpy(),
        srb.column("position_start").to_numpy().astype(np.int64),
    )
    assert (si_batch >= 0).all() and (rb_batch >= 0).all(), "orphan rows"

    # The groups: each untouched batch alone; the touched ones (holding a
    # removed sample) with every removal batch.
    touched = [
        b
        for b, info in batches.items()
        if set(info["sample_ids"]) & set(removed)
    ]
    groups: List[Tuple[List[int], List[int], bool]] = [
        ([b], [b], False) for b in batches if b not in touched
    ]
    if removals:
        groups.append((touched, touched + sorted(removals), True))

    bsi = _lake_table(catalog, "batch_site_info").to_pylist()
    bac = _lake_table(catalog, "batch_allele_counts").to_pylist()
    ssi_rows = ssi.to_pylist()
    for ingest_ids, partial_ids, net in groups:
        mask = np.isin(si_batch, ingest_ids) & si_live
        rows: Dict[Tuple[int, int], List[Dict[str, Any]]] = {}
        for i in np.flatnonzero(mask).tolist():
            r = ssi_rows[i]
            rows.setdefault((r["chromosome_index"], r["position"]), []).append(
                r
            )
        ids = set(partial_ids)
        _check_site_info(
            rows,
            _net_partial_rows(
                [r for r in bsi if r["batch_id"] in ids],
                lambda r: (r["chromosome_index"], r["position"]),
            ),
            stats,
            net,
        )
        _check_allele_counts(
            rows,
            _net_partial_rows(
                [r for r in bac if r["batch_id"] in ids],
                lambda r: (r["chromosome_index"], r["position"], r["allele"]),
            ),
            stats,
            net,
        )
        seg = _ref_segments(
            srb, np.isin(rb_batch, ingest_ids) & rb_live, rows, stats
        )
        _check_ref_blocks(catalog, partial_ids, seg, stats)
        stats["net_batches" if net else "plain_batches"] += len(partial_ids)
        stats["batches"] += len(ingest_ids)
    return dict(stats)
