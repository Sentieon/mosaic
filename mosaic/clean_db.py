# Copyright (c) 2026 Sentieon Inc. All rights reserved
import argparse
from typing import Any, Dict, List

import psycopg2
import subprocess
import sys
import time
import os
from pyiceberg.exceptions import NoSuchTableError

from .config import (
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
    ICEBERG_NAMESPACE,
    S3_BUCKET,
    build_s3_client_kwargs,
)
from .init_db import ICEBERG_TABLES, detached_generation_tables

# PostgreSQL tables created by `mosaic init` (see init_db.init_db_schema).
# Listed explicitly so `clean` only drops MOSAIC state and never co-tenant
# tables in a shared database. Parents only — DROP ... CASCADE removes the
# attached {table}_c*_r* LIST partitions and the foreign-key constraints
# automatically; detached generation tables left by an interrupted aggregate
# finalize are dropped separately (_drop_detached_generations). Children
# (FK -> contigs) are listed before contigs for tidiness; CASCADE makes the
# order non-load-bearing. Keep in sync with init_db.py.
MOSAIC_PG_TABLES = [
    "alleles",
    "site_info",
    "ref_blocks",
    # Not created any more: the counts table of a database initialized before
    # the derived alleles catalog, which needs this reset.
    "allele_counts",
    "samples",
    "ingestion_progress",
    "shard_completion",
    "ingest_files",
    "aggregation_progress",  # CASCADE also drops the cohort_generation view
    "tier_completion",
    "sample_qc",  # CASCADE also drops the sample_qc_summary view
    "contigs",
]


def _drop_detached_generations(cur: Any) -> List[str]:
    """Drop the detached aggregate generation tables (names matching
    init_db.GENERATION_TABLE_RE, e.g. ``site_info_c3_r12``, ``alleles_c3_r12``
    or a pre-catalog database's ``allele_counts_c3_r12``) that an
    interrupted `mosaic aggregate` finalize can leave behind. Attached ones go
    with their parent's DROP ... CASCADE, so run this after the parents.
    Scoped by those name prefixes only. Returns failure descriptions."""
    failures: List[str] = []
    try:
        leftovers = detached_generation_tables(cur)
    except Exception as e:
        print(f"  Error listing generation tables: {e}", file=sys.stderr)
        return [f"list generation tables: {e}"]
    for name, _ in leftovers:
        try:
            print(f"Dropping detached generation table: {name}")
            cur.execute(f"DROP TABLE IF EXISTS {name}")
        except Exception as e:
            print(f"  Error dropping {name}: {e}", file=sys.stderr)
            failures.append(f"drop {name}: {e}")
    return failures


def _verify_iceberg_clean(catalog: Any) -> None:
    """Print a summary of whether Iceberg tables are empty/absent."""
    try:
        namespaces = catalog.list_namespaces()
        print(f"  Namespaces: {namespaces}")

        if (ICEBERG_NAMESPACE,) in namespaces:
            tables = catalog.list_tables(ICEBERG_NAMESPACE)
            print(f"  Tables in {ICEBERG_NAMESPACE}: {tables}")

            for table_name in [
                f"{ICEBERG_NAMESPACE}.{t}" for t in ICEBERG_TABLES
            ]:
                try:
                    table = catalog.load_table(table_name)
                    df = table.scan(limit=1).to_arrow()
                    if len(df) > 0:
                        print(
                            f"  WARNING: {table_name} still has data ({len(df)} rows in first batch)!"
                        )
                    else:
                        print(f"  OK: {table_name} is empty")
                except NoSuchTableError:
                    # The cloud path's expected state: the clean dropped it
                    # (the namespace survives).
                    print(f"  OK: {table_name} dropped")
                except Exception as e:
                    print(f"  Note: Could not check {table_name}: {e}")
        else:
            print(
                f"  OK: {ICEBERG_NAMESPACE} namespace does not exist (clean slate)"
            )

    except Exception as e:
        print(f"  Note: Could not list namespaces: {e}")
        print("  This is expected if Iceberg catalog is freshly initialized")


def _reset_iceberg_local() -> List[str]:
    """Reset local Iceberg + MinIO by bouncing the Docker containers.

    Returns a list of failure descriptions (empty == success), so the caller
    can fail the overall ``clean`` rather than reporting success on a stack
    that was never actually reset.
    """
    compose_file = None
    compose_dir = None

    for filename, directory in [
        ("docker-compose.yml", "."),
        ("docker-compose.yaml", "."),
        ("iceberg_test/docker-compose.yml", "iceberg_test"),
        ("iceberg_test/docker-compose.yaml", "iceberg_test"),
    ]:
        if os.path.exists(filename):
            compose_file = os.path.basename(filename)
            compose_dir = directory
            break

    if not compose_file:
        print(
            "Error: docker-compose.yml not found; Iceberg/MinIO not reset.",
            file=sys.stderr,
        )
        print("Manually run: docker-compose down -v && docker-compose up -d")
        return ["docker-compose.yml not found; Iceberg/MinIO not reset"]

    print(f"Found {compose_file} in {compose_dir}")
    print("Stopping containers and removing volumes...")

    cwd = compose_dir if compose_dir != "." else None

    result = subprocess.run(
        ["docker-compose", "-f", compose_file, "down", "-v"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        print(f"Error stopping containers: {result.stderr}", file=sys.stderr)
        print("You may need to manually run: docker-compose down -v")
        return [f"docker-compose down -v failed: {result.stderr.strip()}"]

    print("Containers stopped and volumes removed. Starting containers...")

    result = subprocess.run(
        ["docker-compose", "-f", compose_file, "up", "-d"],
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    if result.returncode != 0:
        print(f"Error starting containers: {result.stderr}", file=sys.stderr)
        print("You may need to manually run: docker-compose up -d")
        return [f"docker-compose up -d failed: {result.stderr.strip()}"]

    print(
        "Containers started. Waiting for services to be ready (10 seconds)..."
    )
    time.sleep(10)

    print("\nVerifying Iceberg tables are clean...")
    try:
        from pyiceberg.catalog import load_catalog

        catalog = load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)
        _verify_iceberg_clean(catalog)
    except ImportError:
        print("  Skipping Iceberg verification (pyiceberg not available)")
    except Exception as e:
        print(f"  Note: Could not verify Iceberg tables: {e}")

    return []


def _reset_iceberg_cloud() -> List[str]:
    """Reset cloud Iceberg tables (Glue + S3) by dropping tables and deleting
    S3 data files: the upload prefixes ``s3://<bucket>/<table>/`` and each
    dropped table's own location.

    Returns a list of failure descriptions (empty == success). Table-drop
    misses are treated as benign notes (a table that does not exist yet is not
    a failure), but S3 deletion errors are real failures the caller surfaces.
    """
    import boto3  # type: ignore[import-untyped]
    from pyiceberg.catalog import load_catalog

    failures: List[str] = []
    catalog = load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)

    # Drop tables from the Iceberg catalog, recording each dropped table's
    # own location first: dropping does not delete it.
    locations = _drop_tables_recording_locations(catalog)

    # Delete S3 data files under the known prefixes for this pipeline: ingest
    # and the aggregate's staging upload their data files directly to
    # s3://<bucket>/<table>/..., one prefix per Iceberg table (ICEBERG_TABLES;
    # the same upload prefixes `mosaic maintain --orphans` sweeps).
    bucket = S3_BUCKET
    # Same S3 client convention as ingest/aggregate (build_s3_client_kwargs):
    # keyed off the catalog type, so deletion targets exactly where ingest
    # wrote. This is the cloud path (catalog type is never 'rest' here), so it
    # returns region only and lets boto3 resolve IAM / task-role credentials.
    s3_client = boto3.client(  # type: ignore[call-overload]
        "s3", **build_s3_client_kwargs()
    )

    for prefix in [f"{t}/" for t in ICEBERG_TABLES]:
        deleted = 0
        try:
            paginator = s3_client.get_paginator("list_objects_v2")
            for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
                objects = page.get("Contents", [])
                if objects:
                    s3_client.delete_objects(
                        Bucket=bucket,
                        Delete={
                            "Objects": [{"Key": obj["Key"]} for obj in objects]
                        },
                    )
                    deleted += len(objects)
            print(
                f"  Deleted {deleted} S3 objects under s3://{bucket}/{prefix}"
            )
        except Exception as e:
            print(
                f"  Error deleting s3://{bucket}/{prefix}: {e}",
                file=sys.stderr,
            )
            failures.append(f"s3 delete {prefix}: {e}")

    # Each dropped table's own location under the warehouse
    # (s3://<warehouse>/<namespace>.db/<table>/ under Glue): its metadata and
    # the files pyiceberg wrote there (a delete's rewritten data files).
    failures.extend(_delete_table_locations(s3_client, locations))

    print("\nVerifying Iceberg tables are clean...")
    _verify_iceberg_clean(catalog)

    return failures


def _drop_tables_recording_locations(catalog: Any) -> Dict[str, str]:
    """Drop every table of ICEBERG_TABLES in ICEBERG_NAMESPACE; return
    ``{table: location()}`` of the tables dropped. A table that cannot be
    loaded or dropped (one that does not exist yet) is a note, not a
    failure, and its location is not recorded (a live table's files are
    never deleted)."""
    locations: Dict[str, str] = {}
    for table_short in ICEBERG_TABLES:
        table_name = f"{ICEBERG_NAMESPACE}.{table_short}"
        try:
            location = catalog.load_table(table_name).location()
            catalog.drop_table(table_name)
            print(f"  Dropped Iceberg table: {table_name}")
        except Exception as e:
            print(f"  Note: Could not drop {table_name}: {e}")
            continue
        locations[table_short] = location
    return locations


def _delete_table_locations(
    s3_client: Any, locations: Dict[str, str]
) -> List[str]:
    """Delete every object under each ``{table: location}`` (any bucket the
    location names; ``delete_objects`` in chunks of 1,000) and print the
    count per table. Returns failure descriptions: a location that cannot
    be parsed, listed or deleted is one."""
    from .maintain import _dir_prefix, delete_uris, list_prefix

    failures: List[str] = []
    for table_short, location in locations.items():
        try:
            bucket, prefix = _dir_prefix(location)
            keys = list_prefix(s3_client, bucket, prefix)
            deleted = delete_uris(
                s3_client, (f"s3://{b}/{k}" for b, k in keys)
            )
            print(
                f"  Deleted {deleted} S3 objects under the {table_short} "
                f"location s3://{bucket}/{prefix}"
            )
        except Exception as e:
            print(
                f"  Error deleting the {table_short} location {location}: "
                f"{e}",
                file=sys.stderr,
            )
            failures.append(f"s3 delete {table_short} location: {e}")
    return failures


def clean_db(assume_yes: bool = False) -> None:
    is_local = ICEBERG_CATALOG_NAME == "rest"

    # Cloud mode is destructive to shared infrastructure (an RDS database, a
    # Glue catalog, and a possibly-shared S3 bucket), so require an explicit
    # --yes. Local mode runs against the disposable docker-compose stack and
    # stays zero-config.
    if not is_local and not assume_yes:
        print(
            "Refusing to run a CLOUD clean without --yes. This will, against "
            "the configured cloud backend:\n"
            f"  - DROP MOSAIC tables in PostgreSQL database "
            f"{DB_CONFIG['dbname']!r} on host {DB_CONFIG['host']!r}\n"
            f"  - DROP the Iceberg/Glue tables in namespace "
            f"{ICEBERG_NAMESPACE!r}\n"
            f"  - DELETE all S3 objects under the MOSAIC prefixes in bucket "
            f"{S3_BUCKET!r}, and under each dropped table's location\n"
            "Re-run with `mosaic clean --yes` to confirm.",
            file=sys.stderr,
        )
        sys.exit(2)

    failures: List[str] = []

    # --- PostgreSQL -----------------------------------------------------------
    conn = None
    try:
        print("=" * 60)
        print("CLEANING POSTGRESQL DATABASE")
        print("=" * 60)

        conn = psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]
        # Autocommit so one failed DROP does not poison the rest of the batch.
        conn.autocommit = True
        cur = conn.cursor()

        # Scope the drop to MOSAIC's own tables (DROP ... CASCADE handles the
        # attached partitions and FK constraints). Never enumerate and drop
        # every table in the database — that would wipe co-tenant tables in a
        # shared (e.g. RDS) instance.
        for table_name in MOSAIC_PG_TABLES:
            try:
                print(f"Dropping table: {table_name}")
                cur.execute(f"DROP TABLE IF EXISTS {table_name} CASCADE;")
            except Exception as e:
                print(f"  Error dropping {table_name}: {e}", file=sys.stderr)
                failures.append(f"drop {table_name}: {e}")

        failures.extend(_drop_detached_generations(cur))

        print("PostgreSQL database cleanup complete.\n")

    except Exception as e:
        print(f"Error during PostgreSQL cleanup: {e}", file=sys.stderr)
        failures.append(f"postgres: {e}")
    finally:
        if conn:
            conn.close()

    # --- Iceberg --------------------------------------------------------------
    print("=" * 60)
    print(
        f"RESETTING ICEBERG TABLES ({'local Docker' if is_local else 'cloud'})"
    )
    print("=" * 60)

    try:
        if is_local:
            failures.extend(_reset_iceberg_local())
        else:
            failures.extend(_reset_iceberg_cloud())
    except FileNotFoundError:
        print("Error: docker-compose command not found.", file=sys.stderr)
        print(
            "Please install docker-compose or manually reset Iceberg tables."
        )
        failures.append("docker-compose command not found")
    except Exception as e:
        print(f"Error during Iceberg reset: {e}", file=sys.stderr)
        failures.append(f"iceberg reset: {e}")

    print("\n" + "=" * 60)
    if failures:
        print("CLEANUP FAILED")
        for f in failures:
            print(f"  - {f}")
        print("=" * 60)
        sys.exit(1)
    print("CLEANUP COMPLETE")
    print("=" * 60)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """Reset wipes MOSAIC pipeline state (PostgreSQL + Iceberg)."""
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Confirm a destructive cloud reset. Required when "
        "MOSAIC_ICEBERG_TYPE is not 'rest' (drops the Glue tables, deletes "
        "the S3 data files, and drops MOSAIC tables in the configured "
        "database). Ignored for the local Docker stack.",
    )


def run(args: Any) -> None:
    clean_db(assume_yes=getattr(args, "yes", False))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reset PostgreSQL + Iceberg pipeline state."
    )
    configure_parser(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
