#!/usr/bin/env python3
# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Microbenchmark: Iceberg registration cost vs. table file count.

Confirms the mechanism behind ingest's super-linear scaling: the per-batch
registration work is O(files already in the table), so as batches accumulate
(file count ∝ N) the per-batch cost grows ∝ N and total ingest is O(N²). Two
operations are timed against a table pre-loaded to increasing file counts:

  * ``plan_files_s``  — ``list(table.scan().plan_files())``, i.e. the body of
    ``mosaic.ingest._get_registered_uris`` (the pre-registration full scan).
  * ``add_dup_s``     — ``add_files(check_duplicate_files=True)`` of one new
    file (the resume / pre-fix path — its internal dedup scan is O(files)).

and, to show the fix is O(1) in file count:

  * ``add_nodup_s``   — ``add_files(check_duplicate_files=False)`` of one new
    file (the fresh-batch path this change takes).

Expected: ``plan_files_s`` and ``add_dup_s`` grow ~linearly with
``file_count``; ``add_nodup_s`` stays ~flat.

Reuses the isolated ``jc_test`` catalog / ``jc-test-warehouse`` bucket (via
``tests.jc_testlib``), so it never touches dev pipeline data. Requires the
Docker stack up (Postgres / Iceberg REST / MinIO). Emits a TSV to stdout.

Usage:
    python scripts/bench_iceberg_registration.py --counts 500 1000 2000 4000
"""

import argparse
import os
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

# jc_testlib forces the MOSAIC_* env onto the isolated test stores *before*
# mosaic.config is imported, so importing it first keeps this benchmark off the
# dev warehouse.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tests import jc_testlib  # noqa: E402
from mosaic.config import ICEBERG_NAMESPACE, S3_BUCKET  # noqa: E402

BENCH_TABLE = f"{ICEBERG_NAMESPACE}.bench_registration"
BENCH_PREFIX = "bench_registration"
_ARROW = pa.table({"id": pa.array([0], type=pa.int64())})


def _drop_bench(catalog: Any, s3: Any) -> None:
    try:
        catalog.drop_table(BENCH_TABLE)
    except Exception:
        pass
    # Wipe every data file this benchmark uploaded.
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(
        Bucket=S3_BUCKET, Prefix=f"{BENCH_PREFIX}/"
    ):
        keys = [{"Key": o["Key"]} for o in page.get("Contents", [])]
        if keys:
            s3.delete_objects(Bucket=S3_BUCKET, Delete={"Objects": keys})


def _local_parquet() -> str:
    fd, path = tempfile.mkstemp(prefix="bench_reg_", suffix=".parquet")
    os.close(fd)
    pq.write_table(_ARROW, path)
    return path


def _upload_n(s3: Any, local_file: str, n: int, start: int) -> List[str]:
    """Upload the same tiny parquet under n distinct keys; return s3:// URIs."""

    def _put(i: int) -> str:
        key = f"{BENCH_PREFIX}/f{start + i:08d}/data.parquet"
        s3.upload_file(local_file, S3_BUCKET, key)
        return f"s3://{S3_BUCKET}/{key}"

    with ThreadPoolExecutor(max_workers=16) as ex:
        return list(ex.map(_put, range(n)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--counts",
        type=int,
        nargs="+",
        default=[500, 1000, 2000, 4000],
        help="cumulative file-count points to measure at",
    )
    args = ap.parse_args()
    counts = sorted(args.counts)

    if not jc_testlib.stack_up():
        sys.exit(
            "Docker stack not up (need Postgres:5432, REST:8181, MinIO:9000). "
            "Run: cd iceberg_test && docker-compose up -d"
        )
    jc_testlib.ensure_test_infra()
    catalog = jc_testlib.get_catalog()
    s3 = jc_testlib.s3_client()

    _drop_bench(catalog, s3)
    catalog.create_table(BENCH_TABLE, schema=_ARROW.schema)

    local_file = _local_parquet()
    print("file_count\tplan_files_s\tadd_dup_s\tadd_nodup_s")
    try:
        loaded = 0
        for target in counts:
            # Bulk-load up to `target` files (fast: check_duplicate_files=False).
            if target > loaded:
                uris = _upload_n(s3, local_file, target - loaded, loaded)
                tbl = catalog.load_table(BENCH_TABLE)
                tbl.add_files(file_paths=uris, check_duplicate_files=False)
                loaded = target

            tbl = catalog.load_table(BENCH_TABLE)

            t0 = time.perf_counter()
            files = list(tbl.scan().plan_files())
            plan_s = time.perf_counter() - t0
            assert len(files) == loaded, (len(files), loaded)

            # One-file add WITH the duplicate scan (resume / pre-fix path).
            probe = _upload_n(s3, local_file, 1, loaded)
            loaded += 1
            t0 = time.perf_counter()
            tbl.add_files(file_paths=probe, check_duplicate_files=True)
            add_dup_s = time.perf_counter() - t0

            # One-file add WITHOUT the duplicate scan (fresh-batch path).
            tbl = catalog.load_table(BENCH_TABLE)
            probe = _upload_n(s3, local_file, 1, loaded)
            loaded += 1
            t0 = time.perf_counter()
            tbl.add_files(file_paths=probe, check_duplicate_files=False)
            add_nodup_s = time.perf_counter() - t0

            print(
                f"{target}\t{plan_s:.4f}\t{add_dup_s:.4f}\t{add_nodup_s:.4f}",
                flush=True,
            )
    finally:
        os.remove(local_file)
        _drop_bench(catalog, s3)


if __name__ == "__main__":
    main()
