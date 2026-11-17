# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""
Maintain the PostgreSQL secondary indexes.

`mosaic aggregate` builds every index of the derived tables itself (each
generation table's primary key, and the alleles GiST on del_range, before the
partition is attached), and `mosaic init` declares the alleles GiST on the
partitioned parent. This step drops indexes earlier versions created and
re-creates the parent GiST if it is missing; on a current database it has
nothing to do. The workflow still runs it after aggregation:
  mosaic index
"""

import argparse
import sys
import traceback
from typing import Any, List

import psycopg2

from .config import DB_CONFIG
from .init_db import ALLELES_DEL_RANGE_INDEX, ALLELES_DEL_RANGE_INDEX_DEF


def get_db_connection() -> Any:
    return psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]


# Indexes earlier versions created that are now redundant or harmful:
# alleles_pos_index duplicated a prefix of alleles' key, and
# ref_block_pos_index has the same columns as the ref_blocks primary key. A
# non-unique index on a partitioned parent would also make every ATTACH
# PARTITION build a matching index inside the finalize's swap.
REDUNDANT_INDEXES = ("alleles_pos_index", "ref_block_pos_index")


def _index_exists(cur: Any, name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", (name,))
    return bool(cur.fetchone()[0])


def create_indexes(conn: Any) -> List[str]:
    """Drop the legacy indexes that exist and create the parent alleles GiST
    if it is missing (idempotent). Returns the actions taken (empty: nothing
    was needed)."""
    actions: List[str] = []
    with conn.cursor() as cur:
        # Bound how long each CREATE / DROP INDEX waits to ACQUIRE its lock
        # (not how long the build itself runs). Without this, a conflicting
        # lock would block indefinitely — under Nextflow's 10-day process
        # time that silently strands the final pipeline step. Once the lock
        # is held the build runs to completion regardless of this setting.
        cur.execute("SET lock_timeout = '5min'")
        for name in REDUNDANT_INDEXES:
            if _index_exists(cur, name):
                print(f"  Dropping redundant index: {name}")
                cur.execute(f"DROP INDEX IF EXISTS {name}")
                actions.append(f"dropped {name}")
        if not _index_exists(cur, ALLELES_DEL_RANGE_INDEX):
            # On the partitioned parent: adopts each partition's matching
            # GiST (the finalize builds one per generation) or builds it.
            print(f"  Creating index: {ALLELES_DEL_RANGE_INDEX}...")
            cur.execute(
                f"CREATE INDEX IF NOT EXISTS {ALLELES_DEL_RANGE_INDEX} "
                f"ON alleles {ALLELES_DEL_RANGE_INDEX_DEF}"
            )
            actions.append(f"created {ALLELES_DEL_RANGE_INDEX}")
    conn.commit()
    if not actions:
        print(
            "  Nothing to do: the aggregate tables' indexes are built by "
            "`mosaic aggregate` and the alleles GiST exists."
        )
    return actions


def configure_parser(parser: argparse.ArgumentParser) -> None:
    """No options; index creation is parameterless."""


def run(args: Any) -> None:
    print("=" * 60)
    print("CHECKING DATABASE INDEXES")
    print("=" * 60)

    try:
        conn = get_db_connection()
        create_indexes(conn)
        conn.close()
        print("\nIndexes are in place.")
    except Exception as e:
        print(f"Error creating indexes: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    configure_parser(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
