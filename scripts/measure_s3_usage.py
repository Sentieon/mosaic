#!/usr/bin/env python3
# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Measure the total size of MOSAIC objects in an S3 bucket.

For benchmarking: sums the size of the Parquet data files that the pipeline
uploads to S3, scoped to the *known* MOSAIC prefixes so unrelated objects in a
shared bucket are excluded.

Ingestion (`mosaic ingest`) and aggregation (`mosaic aggregate`) upload
their data files directly to bucket-root prefixes, one per Iceberg table
(`mosaic.init_db.ICEBERG_TABLES`, the list `mosaic clean` deletes and
`mosaic maintain --orphans` sweeps):

    s3://<bucket>/sample_ref_blocks/<hash>/data.parquet
    s3://<bucket>/batch_genotypes/<hash>/data.parquet
    s3://<bucket>/tier_ref_blocks_dense/<hash>/data.parquet
    ...

The prefixes are derived from that list, grouped by the table name's prefix:
`sample_*` / `batch_*` are written by ingest, `agg_*` / `tier_*` by the
aggregate. The data files are NOT namespaced, which is why a per-prefix scan
(rather than a whole-bucket scan) is the right way to isolate MOSAIC's
footprint from other data in the bucket.

This counts the upload prefixes only. Iceberg table *metadata* (manifests,
snapshots, metadata.json) and the files pyiceberg itself writes (a delete's
rewritten data files) live under each table's location in the catalog
warehouse, not these prefixes, and are excluded.

Usage:
    # Run with the same MOSAIC_* environment as the pipeline so the S3
    # endpoint/region/credentials match (e.g. MOSAIC_ICEBERG_TYPE=glue for AWS).
    python scripts/measure_s3_usage.py
    python scripts/measure_s3_usage.py --group ingest
    python scripts/measure_s3_usage.py --bucket my-bucket --json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import boto3

# Reuse the pipeline's own S3 client convention (endpoint/region/credentials)
# so this script talks to exactly the backend ingest/aggregate write to, and
# its table list, so a new table is measured without an edit here.
from mosaic.config import build_s3_client_kwargs
from mosaic.init_db import ICEBERG_TABLES

DEFAULT_BUCKET = os.environ.get("MOSAIC_S3_BUCKET", "warehouse")

# The stage that writes a table, by its name's prefix.
GROUP_OF_TABLE_PREFIX: Dict[str, str] = {
    "sample_": "ingest",
    "batch_": "ingest",
    "agg_": "aggregate",
    "tier_": "aggregate",
}


def prefix_groups(tables=ICEBERG_TABLES) -> Dict[str, List[str]]:
    """{group: ["<table>/", ...]} over ``tables``, in their order; a table
    whose name has none of GROUP_OF_TABLE_PREFIX's prefixes raises."""
    groups: Dict[str, List[str]] = {"ingest": [], "aggregate": []}
    for t in tables:
        g = next(
            (g for p, g in GROUP_OF_TABLE_PREFIX.items() if t.startswith(p)),
            None,
        )
        if g is None:
            raise ValueError(f"no --group for Iceberg table {t!r}")
        groups[g].append(f"{t}/")
    return groups


# The MOSAIC-owned data-file prefixes, grouped by the stage that writes them.
PREFIX_GROUPS: Dict[str, List[str]] = prefix_groups()


def human_bytes(n: int) -> str:
    """Format a byte count as a binary-unit string (KiB/MiB/GiB/...)."""
    step = 1024.0
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < step or unit == "PiB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= step
    return f"{size:.2f} PiB"  # unreachable; keeps mypy happy


def measure_prefix(s3, bucket: str, prefix: str) -> Tuple[int, int]:
    """Return (object_count, total_bytes) under a single prefix."""
    count = 0
    total = 0
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            count += 1
            total += obj["Size"]
    return count, total


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure the total size of MOSAIC data files in an S3 bucket, "
            "scoped to the known pipeline prefixes."
        )
    )
    p.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket to scan (default: {DEFAULT_BUCKET}).",
    )
    p.add_argument(
        "--group",
        choices=["ingest", "aggregate", "all"],
        default="all",
        help=(
            "Which stage's prefixes to measure. 'ingest' = per-sample + "
            "per-batch files written by `mosaic ingest`; 'aggregate' = "
            "agg_*/tier_* written by `mosaic aggregate`; 'all' = both "
            "(default). The per-group subtotals are always shown."
        ),
    )
    p.add_argument(
        "--prefixes",
        help=(
            "Comma-separated prefix override. Bypasses --group and scans "
            "exactly these prefixes (e.g. 'batch_ref_blocks/,batch_site_info/')."
        ),
    )
    p.add_argument(
        "--endpoint-url",
        help=(
            "Override the S3 endpoint (else taken from the MOSAIC_* env / the "
            "pipeline config). Use for MinIO, omit for AWS S3."
        ),
    )
    p.add_argument(
        "--region",
        help="Override the AWS region (else MOSAIC_S3_REGION / us-east-1).",
    )
    p.add_argument(
        "--profile",
        help="AWS profile name to use for credentials (optional).",
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a formatted table.",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    # Build the prefix list and remember each prefix's group for subtotals.
    if args.prefixes:
        prefixes = [s for s in args.prefixes.split(",") if s]
        group_of = {pfx: "custom" for pfx in prefixes}
        groups_present = ["custom"]
    elif args.group == "all":
        prefixes = PREFIX_GROUPS["ingest"] + PREFIX_GROUPS["aggregate"]
        group_of = {p: "ingest" for p in PREFIX_GROUPS["ingest"]}
        group_of.update({p: "aggregate" for p in PREFIX_GROUPS["aggregate"]})
        groups_present = ["ingest", "aggregate"]
    else:
        prefixes = PREFIX_GROUPS[args.group]
        group_of = {p: args.group for p in prefixes}
        groups_present = [args.group]

    kwargs = build_s3_client_kwargs()
    if args.region:
        kwargs["region_name"] = args.region
    if args.endpoint_url:
        kwargs["endpoint_url"] = args.endpoint_url

    session = (
        boto3.Session(profile_name=args.profile)
        if args.profile
        else boto3.Session()
    )
    s3 = session.client("s3", **kwargs)

    endpoint = kwargs.get("endpoint_url", "AWS default (s3.amazonaws.com)")
    if not args.json:
        print(
            f"Scanning s3://{args.bucket} via {endpoint} "
            f"(region {kwargs.get('region_name', 'default')})\n",
            file=sys.stderr,
        )

    per_prefix: List[Dict[str, object]] = []
    for prefix in prefixes:
        count, total = measure_prefix(s3, args.bucket, prefix)
        per_prefix.append(
            {
                "prefix": prefix,
                "group": group_of[prefix],
                "objects": count,
                "bytes": total,
            }
        )

    grand_objects = sum(int(r["objects"]) for r in per_prefix)
    grand_bytes = sum(int(r["bytes"]) for r in per_prefix)

    if args.json:
        out = {
            "bucket": args.bucket,
            "group": args.group if not args.prefixes else "custom",
            "prefixes": per_prefix,
            "subtotals": {
                g: {
                    "objects": sum(
                        int(r["objects"])
                        for r in per_prefix
                        if r["group"] == g
                    ),
                    "bytes": sum(
                        int(r["bytes"]) for r in per_prefix if r["group"] == g
                    ),
                }
                for g in groups_present
            },
            "total": {"objects": grand_objects, "bytes": grand_bytes},
        }
        print(json.dumps(out, indent=2))
        return 0

    # Formatted table.
    name_w = max((len(str(r["prefix"])) for r in per_prefix), default=10)
    name_w = max(name_w, len("PREFIX"))
    header = f"{'PREFIX':<{name_w}}  {'OBJECTS':>10}  {'SIZE':>14}"
    print(header)
    print("-" * len(header))
    for g in groups_present:
        rows = [r for r in per_prefix if r["group"] == g]
        if len(groups_present) > 1:
            print(f"[{g}]")
        for r in rows:
            print(
                f"{str(r['prefix']):<{name_w}}  {int(r['objects']):>10,}  "
                f"{human_bytes(int(r['bytes'])):>14}"
            )
        if len(groups_present) > 1:
            sub_obj = sum(int(r["objects"]) for r in rows)
            sub_bytes = sum(int(r["bytes"]) for r in rows)
            print(
                f"{'  subtotal':<{name_w}}  {sub_obj:>10,}  "
                f"{human_bytes(sub_bytes):>14}"
            )
    print("-" * len(header))
    print(
        f"{'TOTAL':<{name_w}}  {grand_objects:>10,}  "
        f"{human_bytes(grand_bytes):>14}"
    )
    print(f"\nTotal: {grand_bytes:,} bytes ({human_bytes(grand_bytes)})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
