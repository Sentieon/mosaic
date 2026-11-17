# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Generate the joint caller's end-to-end slice fixture: a REGION of a
store
(the six chr21 samples as three 2-sample batches, fully aggregated) as
local files the Rust test reads with no store at all.

Writes into OUT:
  * ``sites.parquet`` / ``alleles.parquet``: the driver's two exports for
    the region (``joint_call.export_query``, the ``--region`` SQL);
  * ``sample_site_info.parquet``: the lake's rows of the contig from the
    region's deletion lookback (its first position spanning into the
    region) to its end, the columns the engine reads, sorted
    ``(position, sample_id)`` in 8,192-row groups;
  * ``sample_ref_blocks.parquet``: the blocks overlapping the region,
    sorted ``(position_start, sample_id)`` in 65,536-row groups;
  * ``expected.vcf.gz``: ``joint_call.py``'s output for ``--region`` (its
    ``.tbi`` removed, if any). The checked-in file was recorded from the
    Python engine and is kept as recorded since that engine's removal:
    ``joint_call.py`` now runs the Rust joint caller, so writing this file
    again records the Rust engine's own output as its expectation. That
    is a re-recording: do it only for a change meant to change the
    output, and say so in the change's notes;
  * ``slice.json``: the samples (id, name, header order), the removed ids,
    the contig and the region.

Run with the project's Python environment and PYTHONPATH at the repo
root, under the
``MOSAIC_*`` of the store:
    python rust/tools/gen_jc_slice.py \
        rust/mosaic-joint-call/tests/data/jc_slice chr21:9002902-10002901

(the checked-in slice: it starts inside the deletion -CTTTTTTTTTTT at
9,002,901, which spans E3's worked example site 9,002,912, so the unit's
lookback loads it).
"""

import argparse
import json
import os
import subprocess
import sys

import psycopg2
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from pyiceberg.catalog import load_catalog
from pyiceberg.expressions import And, EqualTo, GreaterThanOrEqual
from pyiceberg.expressions import LessThanOrEqual

import joint_call as jc
from mosaic.config import (
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
    ICEBERG_NAMESPACE,
)
from mosaic.tombstones import (
    active_sample_names,
    active_samples,
    removed_sample_ids,
)

SITE_COLUMNS = [
    "sample_id",
    "position",
    "alleles",
    "gt_1",
    "gt_2",
    "dp",
    "ad",
    "f1r2",
    "f2r1",
    "pl",
]
BLOCK_COLUMNS = [
    "sample_id",
    "position_start",
    "position_end",
    "ad_1",
    "ad_2",
    "dp",
    "pl_1",
    "pl_2",
    "pl_3",
    "sample_counts",
]


def plain_types(t: pa.Table) -> pa.Table:
    """large_string / large_list (pyiceberg's reads) as string / list."""
    fields = []
    for f in t.schema:
        typ = f.type
        if pa.types.is_large_list(typ):
            v = typ.value_type
            v = pa.string() if pa.types.is_large_string(v) else v
            typ = pa.list_(v)
        elif pa.types.is_large_string(typ):
            typ = pa.string()
        fields.append(pa.field(f.name, typ, f.nullable))
    return t.cast(pa.schema(fields))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("out")
    ap.add_argument("region")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    contig, lo, hi = jc.parse_region(args.region)
    assert lo is not None and hi is not None
    conn = psycopg2.connect(**DB_CONFIG)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, length FROM contigs WHERE chromosome = %s", (contig,)
        )
        c_idx, length = cur.fetchone()
    params = {"c": c_idx, "lo": lo, "hi": hi}
    sites = os.path.join(args.out, "sites.parquet")
    alleles = os.path.join(args.out, "alleles.parquet")
    jc.export_query(
        conn, jc.SITES_EXPORT_SQL, params, jc.SITES_EXPORT_SCHEMA, sites
    )
    jc.export_query(
        conn, jc.ALLELES_EXPORT_SQL, params, jc.ALLELES_EXPORT_SCHEMA, alleles
    )
    lb = min([lo] + pq.read_table(alleles).column("position").to_pylist())

    catalog = load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)
    removed = removed_sample_ids(conn)

    def scan(table: str, row_filter, sort, columns):
        t = catalog.load_table(f"{ICEBERG_NAMESPACE}.{table}")
        a = t.scan(
            row_filter=And(EqualTo("chromosome_index", c_idx), row_filter),
            selected_fields=tuple(columns),
        ).to_arrow()
        a = plain_types(a.select(columns))
        if removed:
            a = a.filter(
                pc.invert(pc.is_in(a["sample_id"], pa.array(removed)))
            )
        return a.sort_by([(k, "ascending") for k in sort])

    ssi = scan(
        "sample_site_info",
        And(
            GreaterThanOrEqual("position", lb),
            LessThanOrEqual("position", hi),
        ),
        ["position", "sample_id"],
        SITE_COLUMNS,
    )
    srb = scan(
        "sample_ref_blocks",
        And(
            LessThanOrEqual("position_start", hi),
            GreaterThanOrEqual("position_end", lo),
        ),
        ["position_start", "sample_id"],
        BLOCK_COLUMNS,
    )
    pq.write_table(
        ssi,
        os.path.join(args.out, "sample_site_info.parquet"),
        compression="brotli",
        compression_level=11,
        row_group_size=8192,
    )
    pq.write_table(
        srb,
        os.path.join(args.out, "sample_ref_blocks.parquet"),
        compression="brotli",
        compression_level=11,
        row_group_size=65536,
    )
    expected = os.path.join(args.out, "expected.vcf.gz")
    subprocess.run(
        [
            sys.executable,
            jc.__file__,
            expected,
            "--region",
            args.region,
        ],
        check=True,
    )
    if os.path.exists(expected + ".tbi"):
        os.remove(expected + ".tbi")
    ids = active_samples(conn)
    by_name = {n: i for i, n in ids.items()}
    with open(os.path.join(args.out, "slice.json"), "w") as fh:
        json.dump(
            {
                "samples": [
                    {"id": by_name[n], "name": n}
                    for n in active_sample_names(conn)
                ],
                "removed_ids": removed,
                "contig": {"index": c_idx, "name": contig, "length": length},
                "region": {"contig": contig, "start": lo, "end": hi},
            },
            fh,
            indent=1,
        )
    print(
        f"{args.region}: {ssi.num_rows} sample_site_info rows (from {lb}), "
        f"{srb.num_rows} sample_ref_blocks rows",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
