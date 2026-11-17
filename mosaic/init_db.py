# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""
Initialize the PostgreSQL tables and Iceberg tables.

Run once before ingesting any samples:
  mosaic init <sample.gvcf.gz> [--chromosomes chr21,chr22]

After init, run `mosaic index` to create indexes (recommended after bulk
ingestion).
"""

import sys
import os
import re
import argparse
import traceback
from typing import Any, Dict, List, Optional, Set, Tuple

import psycopg2

from pyiceberg.catalog import load_catalog
from pyiceberg.exceptions import NoSuchTableError, TableAlreadyExistsError
from pyiceberg.partitioning import PartitionField, PartitionSpec
from pyiceberg.schema import Schema
from pyiceberg.table import TableProperties
from pyiceberg.table.sorting import (
    NullOrder,
    SortDirection,
    SortField,
    SortOrder,
)
from pyiceberg.transforms import IdentityTransform
from pyiceberg.types import (
    BinaryType,
    NestedField,
    IntegerType,
    LongType,
    DoubleType,
    StringType,
    ListType,
)

from .config import (
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
    ICEBERG_NAMESPACE,
    PARQUET_ROW_GROUP_ROWS,
    PARQUET_SITE_ROW_GROUP_ROWS,
)
from .genotypes import BATCH_GENOTYPES_TABLE

# sample_qc counter columns (BIGINT), in table order. One row per
# (batch_id, sample_id, chromosome_index); the ingest worker's shards count
# them and cmd_ingest writes them when the batch completes. Definitions are
# in rust/mosaic-ingest-worker/src/qc.rs (qc_variant_record, qc_ref_block).
SAMPLE_QC_COLUMNS = (
    "n_var_records",
    "n_no_call",
    "n_hom_ref",
    "n_het",
    "n_hom_alt",
    "n_haploid",
    "n_snv",
    "n_mnp",
    "n_ins",
    "n_del",
    "n_ti",
    "n_tv",
    "n_var_filtered",
    "dp_sum_var",
    "n_dp_var",
    "rb_bases",
    "rb_dp_sum",
    "rb_bases_gq20",
    "rb_bases_no_call",
)

# sample_site_info columns holding the record's own site_info contribution:
# extract_site_info's QUAL, DP (INFO/DP, else FORMAT/DP), MQSQ pair and GTC
# (with the row's sb, which also comes from it). Their sum over a batch's rows
# at a site is the site's batch_site_info row, so a sample's share can be
# re-derived from the lake (sample removal) without its gVCF. `mosaic ingest`
# refuses a lake whose sample_site_info lacks any of them.
SAMPLE_SITE_CONTRIBUTION_COLUMNS: Tuple[str, ...] = (
    "qual",
    "site_dp",
    "mqsq_1",
    "mqsq_2",
    "gtc",
)

# sample_site_info.del_ref_spans: on a deletion carrier's row, the flattened
# inclusive (start, end) spans of the sample's own ref blocks the deletion
# overlapped (the deletion split: batch_ref_blocks counts them with the row's
# site_dp and sample_counts 0; sample_ref_blocks stores only the rest of each
# block). sample_ref_blocks.sample_counts: the block's per-position
# sample_counts contribution (1 hom-ref, 0 otherwise). Together with the
# contribution columns, what `mosaic ingest` requires of the lake.
DEL_REF_SPANS_COLUMN = "del_ref_spans"
SAMPLE_REF_BLOCKS_CONTRIBUTION_COLUMNS: Tuple[str, ...] = ("sample_counts",)

# The per-sample tables' layout (the whole-sample layout):
# partitioned by identity(chromosome_index), with the sort order (position,
# sample_id) declared on the table. Each data file holds whole samples of one
# batch and one chromosome, packed in sample_id order to
# `--aggregate-parquet-size` (the ingest worker's combine, combine.rs), so
# a sample_id delete (copy-on-write: pyiceberg cannot write merge-on-read
# deletes) rewrites only the files that hold the sample and a sample-subset
# scan prunes whole files; inside a file rows are sorted (position,
# sample_id), so a row group covers a short span of the chromosome for a
# position-window reader. `mosaic ingest` refuses a lake whose per-sample
# tables are not partitioned.
SAMPLE_TABLE_SORT_COLUMNS: Dict[str, Tuple[str, str]] = {
    "sample_ref_blocks": ("position_start", "sample_id"),
    "sample_site_info": ("position", "sample_id"),
}

# samples.status: the tombstone. 'active' unless the sample was removed;
# readers (mosaic/tombstones.py) skip removed samples and `mosaic ingest`
# refuses one. `mosaic remove` sets 'removed' (its first transaction).
SAMPLE_STATUS_ACTIVE = "active"
SAMPLE_STATUS_REMOVED = "removed"
SAMPLES_STATUS_COLUMN_DDL = (
    "status TEXT NOT NULL DEFAULT 'active' "
    "CONSTRAINT samples_status_check CHECK (status IN ('active', 'removed'))"
)

# ingestion_progress.kind: 'ingest' (a batch of ingested samples) or
# 'removal' (`mosaic remove`'s negative batch: the removed samples' re-derived
# contribution, negated, in the batch_* partials; region NULL, sample_ids
# NULL). `mosaic aggregate` sums both kinds alike; the cohort_generation gate
# counts both (a removal not yet aggregated is stale).
BATCH_KIND_INGEST = "ingest"
BATCH_KIND_REMOVAL = "removal"
INGESTION_KIND_COLUMN_DDL = (
    "kind TEXT NOT NULL DEFAULT 'ingest' "
    "CONSTRAINT ingestion_progress_kind_check "
    "CHECK (kind IN ('ingest', 'removal'))"
)

# ingestion_progress.ref_gq_bands: the batch's `mosaic ingest
# --ref-gq-bands` spec, normalized ("20,60" / "all"); NULL = no ref-block
# banding (and a `mosaic remove` batch). A resume must use the same value.
INGESTION_REF_GQ_BANDS_COLUMN_DDL = "ref_gq_bands TEXT"

# ingestion_progress.engine: the shard worker that began the batch. Every
# batch is now 'rust' (the earlier Python worker and `mosaic ingest
# --engine` are removed); 'python', or NULL (a
# batch begun before the column, and every `mosaic remove` batch), is a
# batch of the Python worker, which `mosaic ingest` refuses to resume: the
# two writers' local files differ in size, and the per-sample packing and
# the partials' concatenation decide file membership from those sizes.
INGESTION_ENGINE_COLUMN_DDL = "engine TEXT"

# removal_progress.status: in_progress (the samples are tombstoned; the
# negative batch is not yet registered), registered (the negative batch is
# complete; the samples' per-sample rows may still exist), complete (the rows
# are deleted), failed (resumable: `mosaic remove` with the same samples).
REMOVAL_STATUSES = ("in_progress", "registered", "complete", "failed")

# E3 (sample removal): the site_info / allele_counts partials, their tier
# staging and their agg_* mirrors carry the counts the finalize uses to drop
# dead rows: n_records (the batch's per-sample rows at the site) and
# n_listed (those listing the allele). `mosaic ingest`, `mosaic aggregate`
# and `mosaic remove` refuse a lake without them (removal_lake_problems).
REMOVAL_PARTIAL_COLUMNS: Dict[str, Tuple[str, ...]] = {
    "batch_site_info": ("n_records",),
    "tier_site_info": ("n_records",),
    "agg_site_info": ("n_records",),
    "batch_allele_counts": ("n_listed",),
    "tier_allele_counts": ("n_listed",),
    "agg_allele_counts": ("n_listed",),
}

# gtc is summed in fixed point (E3): each record's GTC (a double on its
# sample_site_info row; PL-derived, it is fractional) is quantized once to
# q = round(gtc * 2**GTC_FIXED_BITS) as int64 (round half to even), and
# batch_site_info / tier_site_info / agg_site_info hold sums of q: an integer
# sum is exact and order-independent, so a removal's negative partial nets a
# sample out exactly and an incremental run equals --rebuild. The finalize
# writes the integer part, sum >> GTC_FIXED_BITS, to PG site_info.gtc.
# int64 holds sum(gtc) < 2**31 per site element (~2 billion samples).
GTC_FIXED_BITS = 32

# Serializes ensure_ingest_schema across concurrent ingests (the Nextflow
# fan-out starts many at once; concurrent CREATE TABLE IF NOT EXISTS can race).
_SCHEMA_LOCK_KEY = 0x4D4F5341  # "MOSA"

# The PostgreSQL tables `mosaic aggregate` rebuilds. Each is LIST-partitioned
# by chromosome_index, and each partition is a *generation* table named
# {table}_c{chromosome_index}_r{agg_run_id} (r0 = the empty partition `mosaic
# init` creates). The finalize builds the next generation as a standalone
# table and swaps it in with DETACH/ATTACH PARTITION; the partition currently
# attached is found through pg_inherits, never by name. `alleles` is the
# derived allele catalog (identities + summed counts + del_range), built by
# the allele_counts metric's finalize.
AGGREGATE_PG_TABLES = ("site_info", "ref_blocks", "alleles")

# Also matches `allele_counts`, the counts table of a database initialized
# before the derived allele catalog, so `mosaic clean` (the reset such a
# database needs) drops its detached generation tables too.
GENERATION_TABLE_RE = re.compile(
    r"^(site_info|ref_blocks|alleles|allele_counts)_c([0-9]+)_r([0-9]+)$"
)

# aggregation_progress columns added for run lineage (and the ref_blocks
# window size that keys a resume); `mosaic aggregate` refuses a database
# initialized without them (no migration path).
AGGREGATION_RUN_COLUMNS = (
    "parent_run_id",
    "delta_batch_ids",
    "staging_pruned",
    "ref_shard_size",
)

# The GiST index on the allele catalog's deletion spans: declared on the
# partitioned parent by `mosaic init` (and `mosaic index`), and built on each
# alleles generation table before it is attached, which adopts it (the same
# definition, predicate included). Partial: only deletion alleles have a
# span (~4% of the catalog's rows on six chr21 samples).
ALLELES_DEL_RANGE_INDEX = "alleles_delrange_index"
ALLELES_DEL_RANGE_INDEX_DEF = (
    "USING GIST (del_range) WHERE del_range IS NOT NULL"
)

# Iceberg staging table of the dense ref_blocks root: one row per covered
# position, summed over every batch of the run, per (run, chromosome,
# window). Also its S3 data-file prefix.
DENSE_REF_TABLE = "tier_ref_blocks_dense"

# Every Iceberg table of the namespace (short names), in creation order: the
# per-sample tables, the per-batch partials, the agg_* mirrors and the
# aggregate staging. `mosaic init` creates them, `mosaic clean` drops them
# (and deletes their data-file prefixes, s3://<bucket>/<name>/), `mosaic
# maintain` maintains them. The single list all three use.
ICEBERG_TABLES: Tuple[str, ...] = (
    "sample_ref_blocks",
    "sample_site_info",
    "batch_ref_blocks",
    "batch_site_info",
    "batch_allele_counts",
    BATCH_GENOTYPES_TABLE,
    "agg_ref_blocks",
    "agg_site_info",
    "agg_allele_counts",
    "tier_ref_blocks",
    "tier_site_info",
    "tier_allele_counts",
    DENSE_REF_TABLE,
)

# Properties every MOSAIC Iceberg table is created with; `mosaic maintain`
# sets them on a table that lacks them (a lake initialized before them).
# write.metadata.delete-after-commit.enabled: after each commit the committer
# deletes the metadata.json versions that fell out of the table's metadata log
# (write.metadata.previous-versions-max, left at its default of 100), so the
# metadata prefix keeps ~101 versions instead of one per commit forever.
ICEBERG_TABLE_PROPERTIES: Dict[str, str] = {
    TableProperties.METADATA_DELETE_AFTER_COMMIT_ENABLED: "true",
}

# Per-table properties `mosaic init` adds to ICEBERG_TABLE_PROPERTIES (E3):
# the per-sample tables' row-group cap. pyiceberg writes a file it rewrites
# (a copy-on-write `mosaic remove` delete) in row groups of
# write.parquet.row-group-limit rows (default 1,048,576), so without it a
# rewritten file's row groups span far more of the chromosome than ingest's
# and a position-window reader decodes far more (joint_call.py on the
# six-sample fixture after removing three: 74.9 s without it, 50.8 s with
# it, 49.4 s on a fresh ingest of the three left). Set
# from the same config values ingest's writer reads
# (MOSAIC_PARQUET_ROW_GROUP_ROWS / MOSAIC_PARQUET_SITE_ROW_GROUP_ROWS): set
# them identically for `mosaic init` and `mosaic ingest`, or not at all.
# `mosaic maintain` does not check the row-group cap (a config value), only
# ICEBERG_TABLE_PROPERTIES and SAMPLE_TABLE_CODEC_PROPERTIES
# (maintained_properties).
#
# The per-sample tables' codec: pyiceberg writes a file it rewrites with its
# default codec,
# zstd, and the Rust joint caller's parquet build enables brotli only (the
# ingest worker's codec), so it could not read a file `mosaic remove`
# rewrote. brotli at level 5 is the worker's combine's codec and level.
SAMPLE_TABLE_CODEC_PROPERTIES: Dict[str, str] = {
    TableProperties.PARQUET_COMPRESSION: "brotli",
    TableProperties.PARQUET_COMPRESSION_LEVEL: "5",
}
TABLE_EXTRA_PROPERTIES: Dict[str, Dict[str, str]] = {
    "sample_ref_blocks": {
        TableProperties.PARQUET_ROW_GROUP_LIMIT: str(PARQUET_ROW_GROUP_ROWS),
        **SAMPLE_TABLE_CODEC_PROPERTIES,
    },
    "sample_site_info": {
        TableProperties.PARQUET_ROW_GROUP_LIMIT: str(
            PARQUET_SITE_ROW_GROUP_ROWS
        ),
        **SAMPLE_TABLE_CODEC_PROPERTIES,
    },
}


def table_properties(table_short: str) -> Dict[str, str]:
    """The properties `mosaic init` creates ``table_short`` with:
    ICEBERG_TABLE_PROPERTIES plus its TABLE_EXTRA_PROPERTIES."""
    return {
        **ICEBERG_TABLE_PROPERTIES,
        **TABLE_EXTRA_PROPERTIES.get(table_short, {}),
    }


def maintained_properties(table_short: str) -> Dict[str, str]:
    """The properties `mosaic maintain` sets on ``table_short`` when it
    lacks them (a lake initialized before them): ICEBERG_TABLE_PROPERTIES,
    plus SAMPLE_TABLE_CODEC_PROPERTIES on the two per-sample tables."""
    extra = (
        SAMPLE_TABLE_CODEC_PROPERTIES
        if table_short in TABLE_EXTRA_PROPERTIES
        else {}
    )
    return {**ICEBERG_TABLE_PROPERTIES, **extra}


def generation_table_name(table: str, c_idx: int, agg_run_id: int) -> str:
    """Name of generation ``agg_run_id`` of ``table``'s partition for
    chromosome ``c_idx`` (matches GENERATION_TABLE_RE)."""
    if table not in AGGREGATE_PG_TABLES:
        raise ValueError(f"not an aggregate table: {table!r}")
    return f"{table}_c{int(c_idx)}_r{int(agg_run_id)}"


def detached_generation_tables(cur: Any) -> List[Tuple[str, int]]:
    """(name, agg_run_id) of every generation table in the current schema
    that is NOT attached as a partition: the leftovers of a finalize that
    built its table but never swapped it in. Sorted by name."""
    cur.execute(
        """
        SELECT c.relname FROM pg_class c
        WHERE c.relnamespace =
              (SELECT oid FROM pg_namespace WHERE nspname = current_schema())
          AND c.relkind = 'r'
          AND NOT c.relispartition
          AND c.relname ~ %s
        """,
        (GENERATION_TABLE_RE.pattern,),
    )
    out: List[Tuple[str, int]] = []
    for (name,) in cur.fetchall():
        m = GENERATION_TABLE_RE.match(name)
        if m:
            out.append((name, int(m.group(3))))
    return sorted(out)


try:
    from vcflib import vcf  # type: ignore[import-untyped]
except ImportError:
    print(
        "Error: Could not import vcflib. Install it with: pip install vcflib",
        file=sys.stderr,
    )
    sys.exit(1)


def get_db_connection() -> Any:
    return psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]


def get_iceberg_catalog() -> Any:
    return load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)


def init_db_schema(conn: Any) -> None:
    """Create PostgreSQL tables (without indexes)."""
    with conn.cursor() as cur:
        # samples first: the sample_qc_summary view reads its status.
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS samples (
                id SERIAL PRIMARY KEY,
                sample_name TEXT UNIQUE,
                batch_id INTEGER DEFAULT 0,
                {SAMPLES_STATUS_COLUMN_DDL}
            );
        """)
        # A samples table from before the tombstone gets the column.
        _add_samples_status(cur)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS contigs (
                id SERIAL PRIMARY KEY,
                chromosome TEXT UNIQUE NOT NULL,
                length INTEGER NOT NULL
            );
        """)

        # The derived aggregate tables are LIST-partitioned by chromosome:
        # `mosaic aggregate` builds a fresh table per chromosome and swaps it
        # in with DETACH/ATTACH PARTITION (build-then-attach). They carry no
        # FOREIGN KEY to contigs: an FK on a partitioned parent is inherited
        # and validated by a scan of every attached partition.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS site_info (
                chromosome_index INTEGER,
                position INTEGER,
                qual REAL DEFAULT 0.0,
                dp BIGINT DEFAULT 0,
                mqsq_1 REAL DEFAULT 0.0,
                mqsq_2 BIGINT DEFAULT 0,
                sb BIGINT[] CONSTRAINT sq_length CHECK (array_length(sb, 1) = 4),
                gtc BIGINT[] CONSTRAINT gtc_length CHECK (array_length(gtc, 1) = 3),
                PRIMARY KEY (chromosome_index, position)
            ) PARTITION BY LIST (chromosome_index);
        """)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS ref_blocks (
                chromosome_index INTEGER,
                position INTEGER,
                dp BIGINT DEFAULT 0,
                sample_counts INTEGER DEFAULT 0,
                PRIMARY KEY (chromosome_index, position)
            ) PARTITION BY LIST (chromosome_index);
        """)

        # The allele catalog: every allele any aggregated batch listed at a
        # site (count 0 included), with its summed count and, for a deletion
        # allele, the span it deletes (derived from the allele string by the
        # finalize). Ingest writes nothing here; the per-sample Iceberg rows
        # carry their own allele strings. The GiST index on del_range is
        # declared on the parent so each finalize's generation table, which
        # builds the same index before it is attached, has it adopted.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS alleles (
                chromosome_index INTEGER NOT NULL,
                position INTEGER NOT NULL,
                allele TEXT NOT NULL,
                del_range INT4RANGE,
                count BIGINT NOT NULL,
                PRIMARY KEY (chromosome_index, position, allele)
            ) PARTITION BY LIST (chromosome_index);
        """)
        cur.execute(f"""
            CREATE INDEX IF NOT EXISTS {ALLELES_DEL_RANGE_INDEX}
            ON alleles {ALLELES_DEL_RANGE_INDEX_DEF};
        """)

        # sample_ids: the batch's genotype-matrix slot order (ascending
        # sample_id); batch_genotypes slot k belongs to sample_ids[k].
        # status: in_progress / failed (resumable), uploaded (`mosaic ingest
        # --defer-register`: Parquet in S3 and listed in ingest_files, not
        # registered with Iceberg; `mosaic register` completes it), complete
        # (registered; the only status aggregate and the readers see).
        cur.execute("""
            CREATE TABLE IF NOT EXISTS ingestion_progress (
                batch_id SERIAL PRIMARY KEY,
                sample_names TEXT[] NOT NULL,
                region TEXT,
                status TEXT DEFAULT 'in_progress',
                created_at TIMESTAMP DEFAULT now(),
                updated_at TIMESTAMP DEFAULT now(),
                sample_ids INTEGER[]
            );
        """)
        # Idempotent migration: pick up `region` on already-initialized DBs
        # so we don't need a full clean_db + init cycle to enable
        # per-chromosome checkpointing. NULL = whole-genome run (the
        # pre-existing semantics), so existing rows stay valid.
        cur.execute("""
            ALTER TABLE ingestion_progress
            ADD COLUMN IF NOT EXISTS region TEXT;
        """)
        _create_ingest_qc_schema(cur)
        _create_ingest_files_schema(cur)
        _create_removal_schema(cur)
        _add_ref_gq_bands(cur)
        _add_ingest_engine(cur)

        cur.execute("""
            CREATE TABLE IF NOT EXISTS shard_completion (
                batch_id INTEGER NOT NULL,
                chrom TEXT NOT NULL,
                start_pos INTEGER NOT NULL,
                PRIMARY KEY (batch_id, chrom, start_pos)
            );
        """)

        # Aggregation run registry (mirrors ingestion_progress): one row per
        # aggregate.py run, keyed for resume by the set of completed batches it
        # reduces (batch_set_hash), the fan-in (which fixes the tree shape) and
        # ref_shard_size (which fixes the ref_blocks window units). Lineage:
        # a run that extends a parent run reduces only delta_batch_ids on top
        # of the parent's retained roots; a full run has parent_run_id NULL
        # and delta_batch_ids = batch_ids. staging_pruned marks a run whose
        # staging has been deleted (it can no longer be a parent).
        cur.execute("""
            CREATE TABLE IF NOT EXISTS aggregation_progress (
                agg_run_id SERIAL PRIMARY KEY,
                batch_set_hash TEXT NOT NULL,
                batch_ids INTEGER[] NOT NULL,
                fan_in INTEGER NOT NULL,
                status TEXT DEFAULT 'in_progress',
                created_at TIMESTAMP DEFAULT now(),
                updated_at TIMESTAMP DEFAULT now(),
                parent_run_id INTEGER NULL,
                delta_batch_ids INTEGER[],
                staging_pruned BOOLEAN NOT NULL DEFAULT false,
                ref_shard_size INTEGER
            );
        """)

        # The cohort generation the PG aggregates hold: the latest aggregation
        # run. A reader (joint_call.py) checks it is `complete` and covers
        # every completed ingest batch; a failed run can leave some
        # chromosomes at its generation and others at the previous one.
        cur.execute("""
            CREATE OR REPLACE VIEW cohort_generation AS
            SELECT agg_run_id, batch_ids, parent_run_id, status, updated_at
            FROM aggregation_progress
            ORDER BY agg_run_id DESC
            LIMIT 1;
        """)

        # Per-(metric, chromosome, tier, group) checkpoint (mirrors
        # shard_completion). A row exists iff that reduce unit's staging output
        # is durable in Iceberg (written before the row is committed), so a
        # checkpointed unit is safely skipped on resume.
        cur.execute("""
            CREATE TABLE IF NOT EXISTS tier_completion (
                agg_run_id INTEGER NOT NULL,
                metric TEXT NOT NULL,
                chromosome_index INTEGER NOT NULL,
                tier INTEGER NOT NULL,
                group_id INTEGER NOT NULL,
                created_at TIMESTAMP DEFAULT now(),
                PRIMARY KEY (agg_run_id, metric, chromosome_index, tier, group_id)
            );
        """)

    conn.commit()


def _add_samples_status(cur: Any) -> None:
    """samples.status (the tombstone), added to a table that predates it
    (idempotent; existing rows become 'active')."""
    cur.execute(
        "ALTER TABLE samples ADD COLUMN IF NOT EXISTS "
        f"{SAMPLES_STATUS_COLUMN_DDL}"
    )


def _create_ingest_qc_schema(cur: Any) -> None:
    """ingestion_progress.sample_ids, the sample_qc table and its summary
    view (all idempotent)."""
    cur.execute("""
        ALTER TABLE ingestion_progress
        ADD COLUMN IF NOT EXISTS sample_ids INTEGER[];
    """)
    counters = ",\n".join(
        f"                {c} BIGINT NOT NULL DEFAULT 0"
        for c in SAMPLE_QC_COLUMNS
    )
    cur.execute(f"""
            CREATE TABLE IF NOT EXISTS sample_qc (
                batch_id INTEGER NOT NULL,
                sample_id INTEGER NOT NULL,
                chromosome_index INTEGER NOT NULL,
{counters},
                PRIMARY KEY (batch_id, sample_id, chromosome_index)
            );
        """)
    sums = ",\n".join(
        f"            SUM({c})::BIGINT AS {c}" for c in SAMPLE_QC_COLUMNS
    )
    # Active samples only: a removed sample's sample_qc rows stay (they
    # record what each batch ingested) but leave the summary.
    cur.execute(f"""
        CREATE OR REPLACE VIEW sample_qc_summary AS
        SELECT
            sample_id,
{sums},
            SUM(dp_sum_var)::DOUBLE PRECISION
                / NULLIF(SUM(n_dp_var), 0) AS mean_dp_var,
            SUM(n_ti)::DOUBLE PRECISION / NULLIF(SUM(n_tv), 0) AS ti_tv,
            SUM(n_het)::DOUBLE PRECISION
                / NULLIF(SUM(n_hom_alt), 0) AS het_hom,
            SUM(rb_dp_sum)::DOUBLE PRECISION
                / NULLIF(SUM(rb_bases), 0) AS mean_rb_dp
        FROM sample_qc
        WHERE sample_id IN (
            SELECT id FROM samples WHERE status = '{SAMPLE_STATUS_ACTIVE}')
        GROUP BY sample_id;
    """)


def _create_ingest_files_schema(cur: Any) -> None:
    """The deferred-registration file list (idempotent). `mosaic ingest
    --defer-register` records one row per Parquet data file it uploaded, in
    the transaction that sets its batch 'uploaded'; `mosaic register` adds
    the pending files (registered_at NULL) to their Iceberg tables, marks
    them and completes the batches. iceberg_table is the short table name
    (no namespace)."""
    cur.execute("""
        CREATE TABLE IF NOT EXISTS ingest_files (
            iceberg_table TEXT NOT NULL,
            uri TEXT NOT NULL,
            batch_id INTEGER NOT NULL,
            created_at TIMESTAMP DEFAULT now(),
            registered_at TIMESTAMP,
            PRIMARY KEY (iceberg_table, uri)
        );
    """)
    cur.execute("""
        CREATE INDEX IF NOT EXISTS ingest_files_batch_idx
        ON ingest_files (batch_id);
    """)


def _create_removal_schema(cur: Any) -> None:
    """ingestion_progress.kind and the removal_progress table (idempotent).

    removal_progress is the audit record of `mosaic remove`: one row per
    removal (the samples, their ids and the negative batch's batch_id), with
    its status (REMOVAL_STATUSES). The removal's negative partials are an
    ordinary ingestion_progress batch of kind 'removal'."""
    cur.execute(
        "ALTER TABLE ingestion_progress ADD COLUMN IF NOT EXISTS "
        f"{INGESTION_KIND_COLUMN_DDL}"
    )
    statuses = ", ".join(f"'{st}'" for st in REMOVAL_STATUSES)
    cur.execute(f"""
        CREATE TABLE IF NOT EXISTS removal_progress (
            removal_id SERIAL PRIMARY KEY,
            sample_ids INTEGER[] NOT NULL,
            sample_names TEXT[] NOT NULL,
            batch_id INTEGER NOT NULL,
            status TEXT NOT NULL
                CONSTRAINT removal_progress_status_check
                CHECK (status IN ({statuses})),
            created_at TIMESTAMP DEFAULT now(),
            updated_at TIMESTAMP DEFAULT now()
        );
    """)


def _add_ref_gq_bands(cur: Any) -> None:
    """ingestion_progress.ref_gq_bands (idempotent; NULL for the existing
    rows: no banding)."""
    cur.execute(
        "ALTER TABLE ingestion_progress ADD COLUMN IF NOT EXISTS "
        f"{INGESTION_REF_GQ_BANDS_COLUMN_DDL}"
    )


def _add_ingest_engine(cur: Any) -> None:
    """ingestion_progress.engine (idempotent; NULL for the existing rows:
    read as 'python')."""
    cur.execute(
        "ALTER TABLE ingestion_progress ADD COLUMN IF NOT EXISTS "
        f"{INGESTION_ENGINE_COLUMN_DDL}"
    )


def ensure_ingest_schema(conn: Any) -> None:
    """Bring a database initialized before sample_ids / sample_qc /
    ingest_files / samples.status / ingestion_progress.kind /
    removal_progress / ingestion_progress.ref_gq_bands /
    ingestion_progress.engine existed up to date,
    so `mosaic ingest` and `mosaic remove` (which both call it first) keep
    working without `mosaic init` (which refuses to run on a populated
    database).

    A cheap catalog probe skips the DDL when everything is present, so the
    common case takes no ACCESS EXCLUSIVE lock on ingestion_progress. The
    DDL itself runs under a transaction-scoped advisory lock because the
    per-chromosome fan-out starts many ingests at once and concurrent
    CREATE TABLE IF NOT EXISTS can fail on a catalog unique index.
    """
    with conn.cursor() as cur:
        cur.execute("""
            SELECT to_regclass('sample_qc') IS NOT NULL
               AND to_regclass('sample_qc_summary') IS NOT NULL
               AND to_regclass('ingest_files') IS NOT NULL
               AND EXISTS (
                   SELECT 1 FROM information_schema.columns
                   WHERE table_schema = current_schema()
                     AND table_name = 'ingestion_progress'
                     AND column_name = 'sample_ids')
               AND EXISTS (
                   SELECT 1 FROM information_schema.columns
                   WHERE table_schema = current_schema()
                     AND table_name = 'samples'
                     AND column_name = 'status')
               AND to_regclass('removal_progress') IS NOT NULL
               AND EXISTS (
                   SELECT 1 FROM information_schema.columns
                   WHERE table_schema = current_schema()
                     AND table_name = 'ingestion_progress'
                     AND column_name = 'kind')
               AND EXISTS (
                   SELECT 1 FROM information_schema.columns
                   WHERE table_schema = current_schema()
                     AND table_name = 'ingestion_progress'
                     AND column_name = 'ref_gq_bands')
               AND EXISTS (
                   SELECT 1 FROM information_schema.columns
                   WHERE table_schema = current_schema()
                     AND table_name = 'ingestion_progress'
                     AND column_name = 'engine')
        """)
        if cur.fetchone()[0]:
            conn.commit()
            return
        cur.execute("SELECT pg_advisory_xact_lock(%s)", (_SCHEMA_LOCK_KEY,))
        _add_samples_status(cur)
        _create_ingest_qc_schema(cur)
        _create_ingest_files_schema(cur)
        _create_removal_schema(cur)
        _add_ref_gq_bands(cur)
        _add_ingest_engine(cur)
    conn.commit()


def init_contigs_from_vcf(
    conn: Any,
    vcf_path: str,
    chromosome_filter: Optional[Set[str]] = None,
) -> None:
    """
    Populate the contigs table from the gVCF header.

    Args:
        conn: Database connection
        vcf_path: Path to gVCF file
        chromosome_filter: Optional set of chromosome names to include
    """
    print(f"Reading contig information from {vcf_path}...")

    variant_file = vcf.VCF(vcf_path)

    if not hasattr(variant_file, "contigs") or not variant_file.contigs:
        print("Error: input VCF does not have contig headers", file=sys.stderr)
        sys.exit(1)

    contigs_to_use = {}
    for name, data in variant_file.contigs.items():
        if chromosome_filter is None or name in chromosome_filter:
            contigs_to_use[name] = data

    if chromosome_filter and len(contigs_to_use) != len(chromosome_filter):
        missing = set(chromosome_filter) - set(contigs_to_use.keys())
        print(
            f"Warning: The following chromosomes were not found in VCF: {missing}"
        )

    if not contigs_to_use:
        print(
            "Error: No contigs to initialize (all filtered out?)",
            file=sys.stderr,
        )
        sys.exit(1)

    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM contigs")
        count = cur.fetchone()[0]
        if count > 0:
            print(
                f"Error: Contigs table already contains {count} entries. "
                "Run clean_db.py to reset before re-initializing.",
                file=sys.stderr,
            )
            sys.exit(1)

        print(
            f"Initializing contigs table with {len(contigs_to_use)} chromosomes..."
        )
        for i, (name, data) in enumerate(contigs_to_use.items()):
            length = data.get("length", 0)
            cur.execute(
                "INSERT INTO contigs (id, chromosome, length) VALUES (%s, %s, %s)",
                (i, name, length),
            )
            print(f"  Added: {name} (length={length})")

            # The initial (empty) generation of each aggregate partition;
            # `mosaic aggregate` swaps in generation r{agg_run_id}.
            for table in AGGREGATE_PG_TABLES:
                part = generation_table_name(table, i, 0)
                cur.execute(
                    f"CREATE TABLE {part} PARTITION OF {table} "
                    f"FOR VALUES IN ({int(i)})"
                )
            print(f"  Added partitions for: {name}")

    conn.commit()
    print(
        f"Successfully initialized contigs table with {len(contigs_to_use)} chromosomes."
    )


def init_iceberg_namespace(catalog: Any) -> None:
    """Create the Iceberg namespace, ignoring 'already exists' errors."""
    try:
        catalog.create_namespace(ICEBERG_NAMESPACE)
        print(f"Created Iceberg namespace: {ICEBERG_NAMESPACE}")
    except Exception as e:
        if "already exists" in str(e).lower():
            print(f"Iceberg namespace already exists: {ICEBERG_NAMESPACE}")
        else:
            raise


def tier_table_schemas() -> List[Tuple[str, Schema]]:
    """Iceberg schemas for the materialized tier-tree staging tables.

    Created once by init_db; aggregate.py appends to them (never drops them) and
    scopes every read/write by `agg_run_id`, so concurrent/sequential runs are
    isolated without DDL or row deletes. Each row carries its agg_run_id, tier
    level, and group_id, plus the metric's key and summed value columns (as
    the batch_* partials: gtc in fixed point, GTC_FIXED_BITS, and n_records
    / n_listed).

    tier_ref_blocks_dense is the dense ref_blocks root a run's window units
    write: one row per covered position (not only variant sites), tagged with
    the run, chromosome and window, so a later run can add new batches to the
    older batches' coverage at sites the older run did not know.
    """
    tier_ref_blocks_dense_schema = Schema(
        NestedField(1, "agg_run_id", IntegerType(), required=True),
        NestedField(2, "chromosome_index", IntegerType(), required=True),
        NestedField(3, "window_id", IntegerType(), required=True),
        NestedField(4, "position", IntegerType(), required=True),
        NestedField(5, "dp", LongType(), required=False),
        NestedField(6, "sample_counts", LongType(), required=False),
    )
    tier_ref_blocks_schema = Schema(
        NestedField(1, "agg_run_id", IntegerType(), required=True),
        NestedField(2, "tier", IntegerType(), required=True),
        NestedField(3, "group_id", IntegerType(), required=True),
        NestedField(4, "chromosome_index", IntegerType(), required=True),
        NestedField(5, "position", IntegerType(), required=True),
        NestedField(6, "dp", LongType(), required=False),
        NestedField(7, "sample_counts", LongType(), required=False),
    )
    tier_site_info_schema = Schema(
        NestedField(1, "agg_run_id", IntegerType(), required=True),
        NestedField(2, "tier", IntegerType(), required=True),
        NestedField(3, "group_id", IntegerType(), required=True),
        NestedField(4, "chromosome_index", IntegerType(), required=True),
        NestedField(5, "position", IntegerType(), required=True),
        NestedField(6, "qual", DoubleType(), required=False),
        NestedField(7, "dp", LongType(), required=False),
        NestedField(8, "mqsq_1", DoubleType(), required=False),
        NestedField(9, "mqsq_2", LongType(), required=False),
        NestedField(
            10,
            "sb",
            ListType(
                element_id=41,
                element_type=LongType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            11,
            "gtc",
            ListType(
                element_id=42,
                element_type=LongType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(12, "n_records", LongType(), required=False),
    )
    tier_allele_counts_schema = Schema(
        NestedField(1, "agg_run_id", IntegerType(), required=True),
        NestedField(2, "tier", IntegerType(), required=True),
        NestedField(3, "group_id", IntegerType(), required=True),
        NestedField(4, "chromosome_index", IntegerType(), required=True),
        NestedField(5, "position", IntegerType(), required=True),
        NestedField(6, "allele", StringType(), required=True),
        NestedField(7, "count", LongType(), required=False),
        NestedField(8, "n_listed", LongType(), required=False),
    )
    return [
        ("tier_ref_blocks", tier_ref_blocks_schema),
        ("tier_site_info", tier_site_info_schema),
        ("tier_allele_counts", tier_allele_counts_schema),
        (DENSE_REF_TABLE, tier_ref_blocks_dense_schema),
    ]


def batch_genotypes_schema() -> Schema:
    """Per-batch genotype matrix: one row per (batch, variant site the batch
    records). ``gt_1``/``gt_2`` hold each sample slot's local allele index
    (-1 = no call) in one signed byte, or in two (little-endian int16) at a
    site whose codes exceed 127, the width ``len(blob) // n_samples``;
    ``gq`` one unsigned byte per slot (0-99, 255 = unknown); slot k is
    ``ingestion_progress.sample_ids[k]``. Binary blobs,
    not list<int>: Iceberg has no int8, Parquet would widen to INT32, and a
    blob reshapes to a numpy matrix without copying."""
    return Schema(
        NestedField(1, "batch_id", IntegerType(), required=True),
        NestedField(2, "chromosome_index", IntegerType(), required=True),
        NestedField(3, "position", IntegerType(), required=True),
        NestedField(4, "n_samples", IntegerType(), required=True),
        NestedField(
            5,
            "alleles",
            ListType(
                element_id=51,
                element_type=StringType(),
                element_required=False,
            ),
            required=True,
        ),
        NestedField(6, "gt_1", BinaryType(), required=True),
        NestedField(7, "gt_2", BinaryType(), required=True),
        NestedField(8, "gq", BinaryType(), required=True),
    )


def chromosome_partition_spec(schema: Schema) -> PartitionSpec:
    """identity(chromosome_index) over ``schema``'s chromosome_index field:
    a window read touches one partition. pyiceberg's add_files infers the
    value from each file's statistics, so ingest writes one chromosome per
    data file. batch_genotypes and the two per-sample tables use it."""
    return PartitionSpec(
        PartitionField(
            source_id=schema.find_field("chromosome_index").field_id,
            field_id=1000,
            transform=IdentityTransform(),
            name="chromosome_index",
        )
    )


def batch_genotypes_partition_spec() -> PartitionSpec:
    """batch_genotypes' partition spec: identity(chromosome_index)."""
    return chromosome_partition_spec(batch_genotypes_schema())


def sample_table_sort_order(table_short: str, schema: Schema) -> SortOrder:
    """The sort order declared on a per-sample table:
    SAMPLE_TABLE_SORT_COLUMNS, ascending. Informational: pyiceberg's
    add_files records no sort order id on the files it adds, and ingest
    writes the order itself."""
    return SortOrder(
        *[
            SortField(
                source_id=schema.find_field(name).field_id,
                transform=IdentityTransform(),
                direction=SortDirection.ASC,
                null_order=NullOrder.NULLS_FIRST,
            )
            for name in SAMPLE_TABLE_SORT_COLUMNS[table_short]
        ]
    )


def is_chromosome_partitioned(table: Any) -> bool:
    """Whether an Iceberg table's current partition spec has an identity
    field on chromosome_index."""
    source_id = table.schema().find_field("chromosome_index").field_id
    return any(
        f.source_id == source_id and isinstance(f.transform, IdentityTransform)
        for f in table.spec().fields
    )


def unpartitioned_sample_tables(catalog: Any) -> List[str]:
    """The per-sample tables (SAMPLE_TABLE_SORT_COLUMNS) not partitioned by
    chromosome_index: a lake initialized before the whole-sample layout,
    whose data files each hold every sample of a batch. Empty when the lake
    is current."""
    return [
        short
        for short in SAMPLE_TABLE_SORT_COLUMNS
        if not is_chromosome_partitioned(
            catalog.load_table(f"{ICEBERG_NAMESPACE}.{short}")
        )
    ]


class SchemaError(RuntimeError):
    """The PostgreSQL schema or the Iceberg lake predates this version of
    MOSAIC (there is no migration path)."""


def missing_lake_columns(catalog: Any) -> List[str]:
    """The per-sample contribution columns (``table.column``) the lake's
    sample_site_info / sample_ref_blocks lack: a lake initialized before
    they existed. Empty when the lake is current."""
    required = {
        "sample_site_info": SAMPLE_SITE_CONTRIBUTION_COLUMNS
        + (DEL_REF_SPANS_COLUMN,),
        "sample_ref_blocks": SAMPLE_REF_BLOCKS_CONTRIBUTION_COLUMNS,
    }
    missing: List[str] = []
    for short, columns in required.items():
        schema = catalog.load_table(f"{ICEBERG_NAMESPACE}.{short}").schema()
        names = {f.name for f in schema.fields}
        missing += [f"{short}.{c}" for c in columns if c not in names]
    return missing


def removal_lake_problems(catalog: Any) -> List[str]:
    """What the lake's site_info / allele_counts partials, staging and
    mirrors lack for sample removal (REMOVAL_PARTIAL_COLUMNS): a lake
    initialized before E3 (its gtc is list<long> too, but under the old
    meaning, each batch's truncated sum; the missing columns tell it
    apart). Empty when the lake is current."""
    problems: List[str] = []
    for short, columns in REMOVAL_PARTIAL_COLUMNS.items():
        schema = catalog.load_table(f"{ICEBERG_NAMESPACE}.{short}").schema()
        names = {f.name for f in schema.fields}
        problems += [f"{short} lacks {c}" for c in columns if c not in names]
    return problems


def check_removal_lake_schema(catalog: Any) -> None:
    """Refuse a lake whose partials cannot carry a removal (see
    removal_lake_problems), with the usual remedy. `mosaic aggregate` runs
    it at startup; check_ingest_lake_schema (ingest, remove) runs it last."""
    problems = removal_lake_problems(catalog)
    if problems:
        raise SchemaError(
            "the Iceberg lake predates sample removal ("
            + "; ".join(problems)
            + ": the partials cannot net a removed sample out exactly). "
            "There is no migration path: reset with `mosaic clean`, then "
            "re-run `mosaic init` and `mosaic ingest`."
        )


def check_ingest_lake_schema(catalog: Any) -> None:
    """Refuse a lake `mosaic ingest` cannot append to. Raises SchemaError
    (the same remedy `mosaic aggregate` prints for an old database):

    * a lake initialized before sample_site_info carried allele strings: its
      ``alleles`` column is a list of PostgreSQL allele ids there, and
      Iceberg cannot evolve list<int> to list<string>;
    * a lake initialized before the per-sample tables carried each record's
      own aggregate contribution (missing_lake_columns): its existing rows
      have none, so a sample in it could not be removed from the lake alone;
    * a lake initialized before the whole-sample layout
      (unpartitioned_sample_tables): its per-sample data files each hold
      every sample of a batch, so deleting one sample would rewrite them all;
    * a lake initialized before sample removal (check_removal_lake_schema):
      its partials lack n_records / n_listed and hold gtc as each batch's
      truncated sum (and its sample_site_info rows carry their batch's
      merged allele lists).
    """
    table = catalog.load_table(f"{ICEBERG_NAMESPACE}.sample_site_info")
    field_type = table.schema().find_field("alleles").field_type
    element = getattr(field_type, "element_type", None)
    if not isinstance(element, StringType):
        raise SchemaError(
            "the Iceberg lake predates this `mosaic ingest` "
            f"(sample_site_info.alleles is {field_type}, not list<string>: "
            "it holds PostgreSQL allele ids, not allele strings). There is "
            "no migration path: reset with `mosaic clean`, then re-run "
            "`mosaic init` and `mosaic ingest`."
        )
    missing = missing_lake_columns(catalog)
    if missing:
        raise SchemaError(
            "the Iceberg lake predates this `mosaic ingest` (it lacks "
            f"{', '.join(missing)}: the per-sample rows do not carry their "
            "own aggregate contribution). There is no migration path: reset "
            "with `mosaic clean`, then re-run `mosaic init` and `mosaic "
            "ingest`."
        )
    flat = unpartitioned_sample_tables(catalog)
    if flat:
        raise SchemaError(
            "the Iceberg lake predates this `mosaic ingest` "
            f"({', '.join(flat)}: not partitioned by chromosome_index, so "
            "each data file holds every sample of a batch and deleting one "
            "sample would rewrite them all). There is no migration path: "
            "reset with `mosaic clean`, then re-run `mosaic init` and "
            "`mosaic ingest`."
        )
    check_removal_lake_schema(catalog)


def ensure_batch_genotypes_table(catalog: Any) -> Any:
    """Load batch_genotypes, creating it if this catalog predates it."""
    name = f"{ICEBERG_NAMESPACE}.{BATCH_GENOTYPES_TABLE}"
    try:
        return catalog.load_table(name)
    except NoSuchTableError:
        pass
    try:
        return catalog.create_table(
            name,
            schema=batch_genotypes_schema(),
            partition_spec=batch_genotypes_partition_spec(),
            properties=ICEBERG_TABLE_PROPERTIES,
        )
    except TableAlreadyExistsError:  # a concurrent ingest created it
        return catalog.load_table(name)


def sample_ref_blocks_schema() -> Schema:
    """Per-sample hom-ref evidence: one row per (sample, ref block) with
    the block's range (the part a sample's own deletion did not overlap,
    see sample_site_info.del_ref_spans) and its FORMAT values.
    sample_counts is the block's batch_ref_blocks contribution per
    position: 1 for a hom-ref block (GT 0/0, 0|0 or 0), 0 otherwise (a
    ./. block); dp is its contribution too."""
    return Schema(
        NestedField(1, "sample_id", IntegerType(), required=True),
        NestedField(2, "chromosome_index", IntegerType(), required=True),
        NestedField(3, "position_start", IntegerType(), required=True),
        NestedField(4, "position_end", IntegerType(), required=True),
        NestedField(5, "ad_1", IntegerType(), required=False),
        NestedField(6, "ad_2", IntegerType(), required=False),
        NestedField(7, "dp", IntegerType(), required=False),
        NestedField(8, "pl_1", IntegerType(), required=False),
        NestedField(9, "pl_2", IntegerType(), required=False),
        NestedField(10, "pl_3", IntegerType(), required=False),
        NestedField(11, "sample_counts", IntegerType(), required=False),
    )


def sample_site_info_schema() -> Schema:
    """Per-sample variant evidence: one row per (sample, variant record at a
    site). Fields 15-19 (SAMPLE_SITE_CONTRIBUTION_COLUMNS) are the record's
    own site_info contribution, the same types as batch_site_info's except
    gtc: extract_site_info's GTC is fractional when derived from PL, so the
    row keeps the double (batch_site_info.gtc is the batch's sum of the
    records' fixed-point quantizations, GTC_FIXED_BITS)."""
    return Schema(
        NestedField(1, "sample_id", IntegerType(), required=True),
        NestedField(2, "chromosome_index", IntegerType(), required=True),
        NestedField(3, "position", IntegerType(), required=True),
        NestedField(4, "del_start", IntegerType(), required=False),
        NestedField(5, "del_end", IntegerType(), required=False),
        # The record's own allele list (ref base first, then its ALTs in
        # the record's order, prefix-encoded as in batch_genotypes, <NON_REF>
        # as *), and the sample's GT as local indices into it (NULL = no
        # call / no second allele). ad, f1r2, f2r1 and pl are in that order,
        # as the record carries them. Nothing on a row depends on the other
        # samples of its batch (E3: a removal leaves the remaining samples'
        # rows as a fresh ingest of them would write them).
        NestedField(
            6,
            "alleles",
            ListType(
                element_id=21,
                element_type=StringType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(7, "gt_1", IntegerType(), required=False),
        NestedField(8, "gt_2", IntegerType(), required=False),
        NestedField(9, "dp", IntegerType(), required=False),
        NestedField(
            10,
            "ad",
            ListType(
                element_id=22,
                element_type=IntegerType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            11,
            "f1r2",
            ListType(
                element_id=23,
                element_type=IntegerType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            12,
            "f2r1",
            ListType(
                element_id=24,
                element_type=IntegerType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            13,
            "sb",
            ListType(
                element_id=25,
                element_type=IntegerType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            14,
            "pl",
            ListType(
                element_id=26,
                element_type=IntegerType(),
                element_required=False,
            ),
            required=False,
        ),
        # The record's own site_info contribution (extract_site_info; sb
        # above is its SB). site_dp is INFO/DP, else FORMAT/DP (dp above is
        # FORMAT/DP).
        NestedField(15, "qual", DoubleType(), required=False),
        NestedField(16, "site_dp", LongType(), required=False),
        NestedField(17, "mqsq_1", DoubleType(), required=False),
        NestedField(18, "mqsq_2", LongType(), required=False),
        NestedField(
            19,
            "gtc",
            ListType(
                element_id=27,
                element_type=DoubleType(),
                element_required=False,
            ),
            required=False,
        ),
        # Only on a deletion carrier's row: the flattened (start, end)
        # inclusive spans of the sample's own ref blocks that this deletion
        # overlapped. sample_ref_blocks stores only the rest of each block;
        # batch_ref_blocks counted these spans with this row's site_dp and
        # sample_counts 0 (the deletion split). Removal input only:
        # joint_call.py never reads it.
        NestedField(
            20,
            DEL_REF_SPANS_COLUMN,
            ListType(
                element_id=28,
                element_type=IntegerType(),
                element_required=False,
            ),
            required=False,
        ),
    )


def iceberg_table_schemas() -> Dict[str, Schema]:
    """The schema of every Iceberg table of ICEBERG_TABLES, by short name:
    what `mosaic init` creates (and what the lake guards compare against)."""
    # ── Per-batch aggregate partials (append-only, tier-0) ──────────────
    # Each ingest batch appends its own contribution here instead of doing
    # an additive in-place UPSERT into the PostgreSQL aggregate tables. The
    # aggregation step (aggregate.py) sums these and finalizes to PG.
    # Dense per-position (not ranges): input gVCF ref-blocks average ~20-30 bp,
    # so exploding them avoids the downstream stab-sum and makes ref_blocks
    # aggregate the same way as site_info / allele_counts (SUM by chrom/pos).
    batch_ref_blocks_schema = Schema(
        NestedField(1, "batch_id", IntegerType(), required=True),
        NestedField(2, "chromosome_index", IntegerType(), required=True),
        NestedField(3, "position", IntegerType(), required=True),
        NestedField(4, "dp", IntegerType(), required=False),
        NestedField(5, "sample_counts", IntegerType(), required=False),
    )

    batch_site_info_schema = Schema(
        NestedField(1, "batch_id", IntegerType(), required=True),
        NestedField(2, "chromosome_index", IntegerType(), required=True),
        NestedField(3, "position", IntegerType(), required=True),
        NestedField(4, "qual", DoubleType(), required=False),
        NestedField(5, "dp", LongType(), required=False),
        NestedField(6, "mqsq_1", DoubleType(), required=False),
        NestedField(7, "mqsq_2", LongType(), required=False),
        NestedField(
            8,
            "sb",
            ListType(
                element_id=31,
                element_type=LongType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            9,
            "gtc",
            ListType(
                element_id=32,
                element_type=LongType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(10, "n_records", LongType(), required=False),
    )

    batch_allele_counts_schema = Schema(
        NestedField(1, "batch_id", IntegerType(), required=True),
        NestedField(2, "chromosome_index", IntegerType(), required=True),
        NestedField(3, "position", IntegerType(), required=True),
        NestedField(4, "allele", StringType(), required=True),
        NestedField(5, "count", LongType(), required=False),
        NestedField(6, "n_listed", LongType(), required=False),
    )

    # ── Materialized aggregates (hierarchical reduction lands here) ──────
    # Cross-batch sums: int64 for dp/sample_counts/count (can exceed int32).
    agg_ref_blocks_schema = Schema(
        NestedField(1, "chromosome_index", IntegerType(), required=True),
        NestedField(2, "position", IntegerType(), required=True),
        NestedField(3, "dp", LongType(), required=False),
        NestedField(4, "sample_counts", LongType(), required=False),
    )

    agg_site_info_schema = Schema(
        NestedField(1, "chromosome_index", IntegerType(), required=True),
        NestedField(2, "position", IntegerType(), required=True),
        NestedField(3, "qual", DoubleType(), required=False),
        NestedField(4, "dp", LongType(), required=False),
        NestedField(5, "mqsq_1", DoubleType(), required=False),
        NestedField(6, "mqsq_2", LongType(), required=False),
        NestedField(
            7,
            "sb",
            ListType(
                element_id=31,
                element_type=LongType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(
            8,
            "gtc",
            ListType(
                element_id=32,
                element_type=LongType(),
                element_required=False,
            ),
            required=False,
        ),
        NestedField(9, "n_records", LongType(), required=False),
    )

    agg_allele_counts_schema = Schema(
        NestedField(1, "chromosome_index", IntegerType(), required=True),
        NestedField(2, "position", IntegerType(), required=True),
        NestedField(3, "allele", StringType(), required=True),
        NestedField(4, "count", LongType(), required=False),
        NestedField(5, "n_listed", LongType(), required=False),
    )

    schemas = dict(
        [
            ("sample_ref_blocks", sample_ref_blocks_schema()),
            ("sample_site_info", sample_site_info_schema()),
            ("batch_ref_blocks", batch_ref_blocks_schema),
            ("batch_site_info", batch_site_info_schema),
            ("batch_allele_counts", batch_allele_counts_schema),
            (BATCH_GENOTYPES_TABLE, batch_genotypes_schema()),
            ("agg_ref_blocks", agg_ref_blocks_schema),
            ("agg_site_info", agg_site_info_schema),
            ("agg_allele_counts", agg_allele_counts_schema),
        ]
        + tier_table_schemas()
    )
    assert sorted(schemas) == sorted(ICEBERG_TABLES), sorted(schemas)
    return schemas


def init_iceberg_tables(catalog: Any) -> None:
    """Create every Iceberg table of ICEBERG_TABLES (an existing one is
    left as it is), each with ICEBERG_TABLE_PROPERTIES."""
    print("\nCreating Iceberg tables...")
    schemas = iceberg_table_schemas()
    for table_short in ICEBERG_TABLES:
        table_name = f"{ICEBERG_NAMESPACE}.{table_short}"
        kwargs: Dict[str, Any] = {
            "schema": schemas[table_short],
            "properties": table_properties(table_short),
        }
        if table_short == BATCH_GENOTYPES_TABLE:
            # Partitioned by chromosome_index (as the per-sample tables
            # below; the others are unpartitioned).
            kwargs["partition_spec"] = batch_genotypes_partition_spec()
        elif table_short in SAMPLE_TABLE_SORT_COLUMNS:
            # Whole samples per data file: partitioned by
            # chromosome_index, sorted (position, sample_id).
            schema = schemas[table_short]
            kwargs["partition_spec"] = chromosome_partition_spec(schema)
            kwargs["sort_order"] = sample_table_sort_order(table_short, schema)
        try:
            table = catalog.create_table(table_name, **kwargs)
            print(f"  Created table: {table_name}")
            print(f"  Location: {table.metadata.location}")
        except TableAlreadyExistsError:
            print(f"  Table already exists: {table_name}")
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"  Table already exists: {table_name}")
            else:
                print(
                    f"  Failed to create table {table_name}: {e}",
                    file=sys.stderr,
                )
                raise

    print("Iceberg tables created successfully.\n")


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "gvcf", help="Path to gVCF file (used to extract contig information)"
    )
    parser.add_argument(
        "-c",
        "--chromosomes",
        help="Comma-separated list of chromosomes to include "
        "(default: all chromosomes in VCF header)",
    )


def run(args: Any) -> None:
    if not os.path.exists(args.gvcf):
        print(f"Error: File {args.gvcf} not found.", file=sys.stderr)
        sys.exit(1)

    chromosome_filter = None
    if args.chromosomes:
        chromosome_filter = set(args.chromosomes.split(","))
        print(f"Chromosome filter: {chromosome_filter}")

    # --- PostgreSQL -----------------------------------------------------------
    try:
        conn = get_db_connection()
        print("Creating database tables...")
        init_db_schema(conn)
        print("Database tables created successfully.")
        init_contigs_from_vcf(conn, args.gvcf, chromosome_filter)
        conn.close()
        print("PostgreSQL initialization complete.")
    except Exception as e:
        print(f"PostgreSQL error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    # --- Iceberg --------------------------------------------------------------
    try:
        print("\nInitializing Iceberg tables...")
        catalog = get_iceberg_catalog()
        init_iceberg_namespace(catalog)
        init_iceberg_tables(catalog)
        print("Iceberg initialization complete.")
    except Exception as e:
        print(f"Iceberg error: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "=" * 60)
    print("Initialization complete!")
    print(
        "Run `mosaic index` after bulk ingestion to create database indexes."
    )
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize PostgreSQL + Iceberg tables for the pipeline."
    )
    configure_parser(parser)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
