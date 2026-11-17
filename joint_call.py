#!/usr/bin/env python3
# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Joint-call the cohort into a VCF: the driver of ``mosaic-joint-call``.

The script checks that the PostgreSQL aggregates are the ingested cohort's
(the ``cohort_generation`` gate; ``--allow-stale`` overrides), prints the
tombstone line, resolves ``--region``, exports each contig's ``site_info`` +
``ref_blocks`` rows and its ``alleles`` rows to local Parquet, plans the
per-sample tables' data files through pyiceberg, writes a JSON manifest and
runs the Rust joint caller (``mosaic-joint-call``, from
``$MOSAIC_JOINT_CALL_WORKER`` or PATH), which reads the lake and writes the
whole VCF; then it prints the closing lines from the binary's summary.
``--samples`` / ``--samples-file`` (the sample subset) joint-call a
subset of the active samples: each record is the
whole-cohort record projected onto the subset's columns, and only the data
files whose ``sample_id`` bounds can hold a subset sample are read.
``--trim-alt-alleles`` and ``--min-ac K`` drop the ALTs no written
genotype names and the records with fewer than K written ALT observations;
the options apply in the order subset, trim, cap (``--max-alt-alleles``),
min-ac, each to the record the previous one wrote.

The cohort export stands
in for PostgreSQL: ``--cohort-state`` runs the checks and prints the cohort
state as one JSON line; ``--export-cohort DIR`` writes ``DIR/cohort.json``
(the state, the contigs, the active and removed samples);
``--export-contig DIR --region CONTIG`` writes that contig's two exports over
the whole contig and its sidecar, in one snapshot that checks the database's
state is ``cohort.json``'s; ``--from-export DIR`` joint-calls from it with no
PostgreSQL connection, writing what the PostgreSQL path writes byte for byte.

The Python calling engine this script used to hold (and its ``--engine`` /
``--step-size`` options) has been removed; the frozen goldens of
tests/golden/joint_call_digests.json, recorded while the two engines agreed
byte for byte, are the output's reference since.
"""

import sys
import argparse
import hashlib
import json
import os
import psycopg2
import secrets
import shutil
import subprocess
import tempfile
import time
from typing import Any, Dict, List, Optional, Tuple

import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq

# PyIceberg imports
from pyiceberg.catalog import load_catalog
from pyiceberg.conversions import from_bytes
from pyiceberg.expressions import EqualTo

from mosaic.config import (
    DB_CONFIG,
    ICEBERG_CATALOG_NAME,
    ICEBERG_CONFIG,
    ICEBERG_NAMESPACE,
    build_s3_client_kwargs,
)
from mosaic.tombstones import (
    active_sample_names,
    active_samples,
    missing_status_remedy,
    removed_sample_ids,
    removed_sample_names,
    samples_status_missing,
)

# ========== Configuration ==========

VCF_HEADERS = [
    "##fileformat=VCFv4.2",
    '##FILTER=<ID=PASS,Description="All filters passed">',
    '##FORMAT=<ID=AD,Number=R,Type=Integer,Description="Allelic depths for the ref and alt alleles in the order listed">',
    '##FORMAT=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth (reads with MQ=255 or with bad mates are filtered)">',
    '##FORMAT=<ID=F1R2,Number=R,Type=Integer,Description="Count of reads in F1R2 pair orientation supporting each allele">',
    '##FORMAT=<ID=F2R1,Number=R,Type=Integer,Description="Count of reads in F2R1 pair orientation supporting each allele">',
    '##FORMAT=<ID=GQ,Number=1,Type=Integer,Description="Genotype quality">',
    '##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">',
    '##FORMAT=<ID=PL,Number=G,Type=Integer,Description="The phred-scaled genotype likelihoods rounded to the closest integer">',
    '##INFO=<ID=AC,Number=A,Type=Integer,Description="Allele count in genotypes, for each ALT allele, in the same order as listed">',
    '##INFO=<ID=AF,Number=A,Type=Float,Description="Allele frequency, for each ALT allele, in the same order as listed">',
    '##INFO=<ID=AN,Number=1,Type=Integer,Description="Total number of alleles in called genotypes">',
    '##INFO=<ID=DP,Number=1,Type=Integer,Description="Approximate read depth (informative and non-informative); some reads may have been filtered based on mapq etc.">',
    '##INFO=<ID=FS,Number=1,Type=Float,Description="Phred-scaled p-value using Fisher\'s exact test to detect strand bias">',
    '##INFO=<ID=InbreedingCoeff,Number=1,Type=Float,Description="Inbreeding coefficient as estimated from the genotype likelihoods per-sample when compared against the Hardy-Weinberg expectation">',
    '##INFO=<ID=MQ,Number=1,Type=Float,Description="RMS mapping quality">',
    '##INFO=<ID=QD,Number=1,Type=Float,Description="Variant Confidence/Quality by Depth">',
    '##INFO=<ID=SOR,Number=1,Type=Float,Description="Symmetric Odds Ratio of 2x2 contingency table to detect strand bias">',
]

# The ALT cap: --max-alt-alleles'
# default, and its header line, written after VCF_HEADERS (right before the
# first ##contig line) only when a cap is set. tests/vcf_project.py holds
# the same text by value.
DEFAULT_MAX_ALT_ALLELES = 6
MAX_ALT_ALLELES_HEADER = "##mosaic_max_alt_alleles={}"

# K5: --trim-alt-alleles' and --min-ac's header lines, each written only
# when its option is set, in the order trim, cap, min-ac (right before the
# first ##contig line). tests/vcf_project.py holds the same text by value.
TRIM_ALT_ALLELES_HEADER = "##mosaic_trim_alt_alleles=true"
MIN_AC_HEADER = "##mosaic_min_ac={}"


# The closing line of a subset run (K4): the binary does not run the
# catalog cross-check, whose counts (alleles.count) are the whole cohort's.
SUBSET_CROSS_CHECK_LINE = (
    "Catalog cross-check: skipped under --samples / --samples-file (the "
    "catalog's allele counts, alleles.count, are the whole cohort's)"
)


# ========== --samples / --samples-file ==========


def subset_names(args: Any) -> Optional[List[str]]:
    """The names ``--samples`` (comma-separated) or ``--samples-file`` (one
    per line, blank lines ignored) give, in the order given, each stripped
    of surrounding white space; None when neither option is given. An
    unreadable file: OSError."""
    if args.samples is not None:
        return [x.strip() for x in args.samples.split(",") if x.strip()]
    if args.samples_file is not None:
        with open(args.samples_file) as fh:
            return [x.strip() for x in fh if x.strip()]
    return None


def resolve_subset(
    names: List[str], active: List[str], removed: List[str]
) -> List[str]:
    """The subset's header columns: the active samples (``active``, in the
    header's order) that ``names`` lists. ValueError naming the offenders
    for an empty list, a name given more than once, a removed sample
    (``removed``: ``samples.status = 'removed'``) or an unknown name."""
    if not names:
        raise ValueError("an empty list of samples")
    problems = []
    seen = set()
    twice = []
    for n in names:
        if n in seen and n not in twice:
            twice.append(n)
        seen.add(n)
    if twice:
        problems.append("given more than once: " + ", ".join(twice))
    gone = [n for n in dict.fromkeys(names) if n in set(removed)]
    if gone:
        problems.append(
            "removed (samples.status = 'removed'): " + ", ".join(gone)
        )
    known = set(active) | set(removed)
    unknown = [n for n in dict.fromkeys(names) if n not in known]
    if unknown:
        problems.append("unknown: " + ", ".join(unknown))
    if problems:
        raise ValueError("; ".join(problems))
    return [n for n in active if n in seen]


# ========== --region ==========


def parse_region(spec: str) -> Tuple[str, Optional[int], Optional[int]]:
    """``contig[:start-end]`` (1-based, inclusive) -> (contig, start, end);
    start and end are None for a whole contig. ValueError when malformed."""
    if not spec or spec != spec.strip():
        raise ValueError(f"malformed region {spec!r}")
    contig, sep, rng = spec.rpartition(":")
    if not sep:
        return spec, None, None
    lo, dash, hi = rng.partition("-")
    if not contig or not dash or not lo.isdigit() or not hi.isdigit():
        raise ValueError(
            f"malformed region {spec!r} (expected contig or "
            "contig:start-end)"
        )
    start, end = int(lo), int(hi)
    if start < 1 or end < start:
        raise ValueError(f"malformed region {spec!r} (need 1 <= start <= end)")
    return contig, start, end


def resolve_region(
    region: Tuple[str, Optional[int], Optional[int]],
    contigs_data: List[Tuple[int, str, int]],
) -> Tuple[int, str, int, int, int]:
    """The region against ``contigs`` -> (chrom_idx, contig, length, start,
    end), the end clipped to the contig's length. ValueError for a contig
    not in ``contigs`` or a start past its end."""
    contig, start, end = region
    for chrom_idx, name, length in contigs_data:
        if name == contig:
            lo = 1 if start is None else start
            hi = length if end is None else min(end, length)
            if lo > length:
                raise ValueError(
                    f"region start {lo} is past the end of {contig} "
                    f"(length {length})"
                )
            return chrom_idx, name, length, lo, hi
    raise ValueError(f"contig {contig!r} is not in the contigs table")


# ========== Catalog cross-check ==========


class CatalogCrossCheck:
    """The closing lines of the catalog cross-check: the binary counts the
    records whose catalog non-ref allele counts (``alleles.count``; the
    ``*`` ALT aside) differ from the AC the record writes (from its
    genotypes), per reason, and the driver prints them.

    Differences are expected only where the aggregates are not the active
    cohort's: a removal (or any batch) not yet aggregated (``stale``, under
    ``--allow-stale``: the run's reason, which every differing record then
    takes). Otherwise a record's reason is ``unresolved`` when a sample's
    own call at the record is written ``./.`` although its row has both GT
    alleles (one of them, e.g. its <NON_REF>, is no VCF allele of the
    record, while the catalog counts the other), else ``other``. Tombstoned
    samples are no reason of their own: before the aggregate that applies
    their removal the run is stale, and after it the catalog no longer
    counts them.
    """

    UNRESOLVED = (
        "a sample's called genotype written ./. (an allele of it, e.g. "
        "<NON_REF>, is not among the record's alleles)"
    )

    def __init__(self, stale: bool = False) -> None:
        self.reason: Optional[str] = (
            "stale aggregates (--allow-stale)" if stale else None
        )
        # The records compared (a record the ALT cap restricted is not:
        # ``capped``, K3; a no-called carrier lowers the emitted AC of a
        # kept allele it also carried).
        self.records = 0
        self.capped = 0
        self.differing: Dict[str, int] = {}
        self.first: List[str] = []

    def lines(self) -> List[str]:
        n = sum(self.differing.values())
        # Under the ALT cap (K3), the records it restricted were not
        # compared; without one (or when none was capped) the text is the
        # goldens'.
        not_compared = (
            f"{self.capped} records capped by --max-alt-alleles not compared"
        )
        if n == 0:
            if self.capped:
                return [
                    "Catalog cross-check: the catalog's non-ref allele "
                    "counts (alleles.count) equal the emitted AC at every "
                    f"record compared ({self.records} records; "
                    f"{not_compared})"
                ]
            return [
                "Catalog cross-check: the catalog's non-ref allele counts "
                "(alleles.count) equal the emitted AC at every record "
                f"({self.records} records)"
            ]
        head = (
            f"Catalog cross-check: {n} of {self.records} records have "
            "catalog non-ref allele counts (alleles.count) that differ "
            "from the emitted AC"
        )
        out = [f"{head} ({not_compared})" if self.capped else head]
        for reason, cnt in sorted(self.differing.items()):
            out.append(f"  {reason}: {cnt}")
        out.append(
            f"  first {len(self.first)} differing site(s): "
            + ", ".join(self.first)
        )
        return out


# ========== The driver of the Rust joint caller ==========
#
# After the gate and the tombstone line, the driver exports the PostgreSQL side per contig
# to local Parquet (`sites`, `alleles`), plans the per-sample tables' data
# files per contig through pyiceberg, writes a JSON manifest and runs
# `mosaic-joint-call`, which reads the lake and writes the whole VCF; then
# it prints the closing lines from the binary's one JSON summary line.

JOINT_CALL_WORKER_ENV = "MOSAIC_JOINT_CALL_WORKER"
JOINT_CALL_WORKER = "mosaic-joint-call"
DEFAULT_WINDOW = 10_000_000
# The binary's --unit-buffer-bytes default (K2): a unit that is not the head
# buffers its output in memory up to this many bytes, then spills it.
DEFAULT_UNIT_BUFFER_BYTES = 64 << 20
# Rows per row group of an export: the unit's reads select row groups by
# their position statistics.
EXPORT_ROW_GROUP_ROWS = 65_536

# The `sites` export: a site's row over a contig (the removed Python
# engine's `fetch_batch_sites` row).
SITES_EXPORT_SCHEMA = pa.schema(
    [
        ("position", pa.int32()),
        ("qual", pa.float64()),
        ("dp", pa.int64()),
        ("mqsq_1", pa.float64()),
        ("mqsq_2", pa.int64()),
        ("sb_1", pa.int64()),
        ("sb_2", pa.int64()),
        ("sb_3", pa.int64()),
        ("sb_4", pa.int64()),
        ("gtc_1", pa.int64()),
        ("gtc_2", pa.int64()),
        ("gtc_3", pa.int64()),
        ("rb_dp", pa.int64()),
        ("rb_sample_counts", pa.int64()),
    ]
)
SITES_EXPORT_SQL = """
    SELECT si.position, si.qual, si.dp, si.mqsq_1, si.mqsq_2,
           si.sb[1], si.sb[2], si.sb[3], si.sb[4],
           si.gtc[1], si.gtc[2], si.gtc[3],
           COALESCE(rb.dp, 0), COALESCE(rb.sample_counts, 0)
    FROM site_info si
    JOIN contigs c ON si.chromosome_index = c.id
    LEFT JOIN ref_blocks rb ON si.chromosome_index = rb.chromosome_index
                            AND si.position = rb.position
    WHERE si.chromosome_index = %(c)s
      AND si.position >= %(lo)s
      AND si.position <= %(hi)s
    ORDER BY si.position
"""

# The `alleles` export: the catalog rows (the removed Python engine's
# `fetch_batch_alleles` plus its `DeletionContext` columns), over [lo, hi] and the positions below lo that hold a deletion
# spanning into it (the unit lookback of a --region run; none for a whole
# contig, lo = 1).
ALLELES_EXPORT_SCHEMA = pa.schema(
    [
        ("position", pa.int32()),
        ("allele", pa.string()),
        ("count", pa.int64()),
        ("del_start", pa.int32()),
        ("del_end", pa.int32()),
    ]
)
ALLELES_EXPORT_SQL = """
    SELECT a.position, a.allele, a.count,
           lower(a.del_range), upper(a.del_range)
    FROM alleles a
    WHERE a.chromosome_index = %(c)s
      AND ((a.position >= %(lo)s AND a.position <= %(hi)s)
           OR a.position IN (
               SELECT d.position FROM alleles d
               WHERE d.chromosome_index = %(c)s
                 AND d.del_range IS NOT NULL
                 AND d.del_range && int4range(%(lo)s, %(hi)s, '[]')
                 AND d.position < %(lo)s))
    ORDER BY a.position, a.allele COLLATE "C"
"""


def spill_dir_path(
    output: str, tmpdir: str, spill_dir: Optional[str], token: str
) -> str:
    """The spill directory the driver names for the binary (K2): always one
    of its own naming, never the user's directory itself, since the binary
    removes it. Under ``--spill-dir DIR``:
    ``DIR/mosaic-joint-call-spill-<token>``; for a ``/dev/...`` output (or
    ``-``): ``<tmpdir>/spill``; else ``<output>.spill-<token>`` beside the
    output (the filesystem sized for it). Not created here."""
    if spill_dir is not None:
        return os.path.join(
            os.path.abspath(spill_dir), f"mosaic-joint-call-spill-{token}"
        )
    if output == "-" or output.startswith("/dev/"):
        return os.path.join(tmpdir, "spill")
    return f"{os.path.abspath(output)}.spill-{token}"


def find_joint_call_worker() -> Optional[str]:
    """$MOSAIC_JOINT_CALL_WORKER (when executable), else
    ``mosaic-joint-call`` on PATH; None when neither."""
    path = os.environ.get(JOINT_CALL_WORKER_ENV)
    if path:
        return path if os.access(path, os.X_OK) else None
    return shutil.which(JOINT_CALL_WORKER)


def missing_worker_remedy() -> str:
    path = os.environ.get(JOINT_CALL_WORKER_ENV)
    found = (
        f"${JOINT_CALL_WORKER_ENV}={path} is not an executable file"
        if path
        else f"{JOINT_CALL_WORKER} is not on PATH and "
        f"${JOINT_CALL_WORKER_ENV} is unset"
    )
    return (
        f"the Rust joint caller is not found ({found}). "
        "Build it with `cd rust && cargo build --release` and set "
        f"{JOINT_CALL_WORKER_ENV}=rust/target/release/{JOINT_CALL_WORKER} "
        f"(or put {JOINT_CALL_WORKER} on PATH)."
    )


def cap_line(cap: Dict[str, Any]) -> str:
    """The closing line of the ALT cap's counts (K3), from the summary's
    ``cap`` object."""
    return (
        f"ALT cap (--max-alt-alleles {cap['max_alt_alleles']}): "
        f"{cap['records_capped']} records capped, "
        f"{cap['genotypes_no_called']} genotypes written ./., "
        f"{cap['dropped_alt_observations']} ALT observations on dropped "
        f"alleles, {cap['kept_alt_observations_lost']} kept-allele ALT "
        "observations lost with those genotypes"
    )


def trim_line(trim: Dict[str, Any]) -> str:
    """The closing line of the trim's counts (K5), from the summary's
    ``trim`` object."""
    return (
        "ALT trim (--trim-alt-alleles): "
        f"{trim['records_trimmed']} records trimmed, "
        f"{trim['alts_dropped']} ALT alleles with AC 0 dropped"
    )


def min_ac_line(min_ac: Dict[str, Any]) -> str:
    """The closing line of --min-ac's count (K5), from the summary's
    ``min_ac`` object."""
    return (
        f"Minimum AC (--min-ac {min_ac['min_ac']}): "
        f"{min_ac['records_left_out']} records left out (fewer ALT "
        "observations in their written genotypes)"
    )


def option_lines(summary: Dict[str, Any]) -> List[str]:
    """The options' closing lines, after the cross-check's, in the order
    the options apply: the trim (K5), the cap (K3), --min-ac (K5); each
    only when its option was set."""
    out = []
    if summary.get("trim") is not None:
        out.append(trim_line(summary["trim"]))
    if summary.get("cap") is not None:
        out.append(cap_line(summary["cap"]))
    if summary.get("min_ac") is not None:
        out.append(min_ac_line(summary["min_ac"]))
    return out


def vcf_header_lines(
    contigs: List[Tuple[str, int]],
    samples: List[str],
    max_alt_alleles: int = 0,
    trim_alt_alleles: bool = False,
    min_ac: int = 0,
) -> List[str]:
    """The header lines: VCF_HEADERS, the contig lines and the #CHROM line
    (as vcflib's ``emit_header`` wrote them for the removed Python engine,
    which the goldens froze); between VCF_HEADERS and the contig lines, one
    line per option set, in the order the options apply: under
    --trim-alt-alleles (K5) TRIM_ALT_ALLELES_HEADER, under an ALT cap (K3)
    MAX_ALT_ALLELES_HEADER, under --min-ac (K5) MIN_AC_HEADER."""
    lines = list(VCF_HEADERS)
    if trim_alt_alleles:
        lines.append(TRIM_ALT_ALLELES_HEADER)
    if max_alt_alleles > 0:
        lines.append(MAX_ALT_ALLELES_HEADER.format(max_alt_alleles))
    if min_ac > 0:
        lines.append(MIN_AC_HEADER.format(min_ac))
    for chrom, length in contigs:
        lines.append(f"##contig=<ID={chrom},length={length}>")
    cols = "#CHROM,POS,ID,REF,ALT,QUAL,FILTER,INFO".split(",")
    if samples:
        cols.append("FORMAT")
        cols += samples
    lines.append("\t".join(cols))
    return lines


def export_query(
    conn: Any,
    sql: str,
    params: Dict[str, Any],
    schema: Any,
    path: str,
) -> int:
    """``COPY (sql) TO STDOUT (FORMAT csv)`` into a temporary CSV beside
    ``path``, then streamed through ``pyarrow.csv`` with ``schema``'s types
    into a brotli Parquet file of EXPORT_ROW_GROUP_ROWS-row groups (with
    statistics); never the whole result in memory. Returns the rows."""
    csv_path = path + ".csv"
    with conn.cursor() as cur:
        copy = cur.mogrify(sql, params).decode()
        with open(csv_path, "wb") as fh:
            cur.copy_expert(f"COPY ({copy}) TO STDOUT (FORMAT csv)", fh)
    rows = 0
    try:
        with pq.ParquetWriter(path, schema, compression="brotli") as writer:
            if os.path.getsize(csv_path) == 0:
                writer.write_table(schema.empty_table())
                return 0
            reader = pacsv.open_csv(
                csv_path,
                read_options=pacsv.ReadOptions(
                    column_names=schema.names, block_size=4 << 20
                ),
                convert_options=pacsv.ConvertOptions(
                    column_types={f.name: f.type for f in schema},
                    null_values=[""],
                    strings_can_be_null=False,
                    quoted_strings_can_be_null=False,
                ),
            )
            pending: List[Any] = []
            n_pending = 0
            for batch in reader:
                pending.append(batch)
                n_pending += batch.num_rows
                if n_pending >= EXPORT_ROW_GROUP_ROWS:
                    t = pa.Table.from_batches(pending, schema)
                    full = (n_pending // EXPORT_ROW_GROUP_ROWS) * (
                        EXPORT_ROW_GROUP_ROWS
                    )
                    writer.write_table(
                        t.slice(0, full), row_group_size=EXPORT_ROW_GROUP_ROWS
                    )
                    rows += full
                    rest = t.slice(full)
                    pending = rest.to_batches() if rest.num_rows else []
                    n_pending = rest.num_rows
            if n_pending:
                t = pa.Table.from_batches(pending, schema)
                writer.write_table(t, row_group_size=EXPORT_ROW_GROUP_ROWS)
                rows += n_pending
    finally:
        os.remove(csv_path)
    return rows


def sample_id_bounds(
    data_file: Any, field_id: int, field_type: Any
) -> Tuple[Optional[int], Optional[int]]:
    """A data file's ``sample_id`` lower and upper bounds from its Iceberg
    manifest entry, decoded with the column's type (Iceberg's single-value
    serialization); None where the entry has none."""
    out: List[Optional[int]] = []
    for bounds in (data_file.lower_bounds, data_file.upper_bounds):
        raw = (bounds or {}).get(field_id)
        out.append(None if raw is None else int(from_bytes(field_type, raw)))
    return out[0], out[1]


def file_may_hold(
    bounds: Tuple[Optional[int], Optional[int]], ids: List[int]
) -> bool:
    """Whether a data file whose ``sample_id`` bounds are ``bounds`` can
    hold one of ``ids`` (sorted): some id within [lower, upper]; a missing
    bound does not exclude (a file without bounds is kept)."""
    lo, hi = bounds
    return any(
        (lo is None or lo <= i) and (hi is None or i <= hi) for i in ids
    )


def plan_data_files(
    table: Any, chrom_idx: int, subset_ids: Optional[List[int]] = None
) -> Tuple[List[Dict[str, Any]], int]:
    """The data files of a per-sample table's contig partition
    (``plan_files``) and how many the partition holds; a task with delete
    files is an error (pyiceberg 0.11 writes none and the engine reads
    none). Under a subset (K4, ``subset_ids``) only the files whose
    ``sample_id`` bounds can hold a subset id (``file_may_hold``) are kept:
    conservative (a file between two subset ids is kept), and the binary's
    mask drops the other samples' rows. No ``In`` filter is pushed down."""
    out = []
    total = 0
    field = table.schema().find_field("sample_id")
    ids = sorted(subset_ids) if subset_ids is not None else None
    for task in table.scan(
        row_filter=EqualTo("chromosome_index", int(chrom_idx))
    ).plan_files():
        if task.delete_files:
            raise RuntimeError(
                f"{task.file.file_path} has {len(task.delete_files)} delete "
                "file(s); the joint caller reads no deletes"
            )
        total += 1
        if ids is not None and not file_may_hold(
            sample_id_bounds(task.file, field.field_id, field.field_type), ids
        ):
            continue
        out.append(
            {
                "uri": task.file.file_path,
                "record_count": int(task.file.record_count),
                "file_size_in_bytes": int(task.file.file_size_in_bytes),
            }
        )
    return out, total


def s3_settings() -> Dict[str, Any]:
    """The manifest's ``s3``: MOSAIC_S3_* under the ``rest`` catalog
    (MinIO: endpoint, region, credentials, path style), else the AWS
    credential chain (``from_env``) with MOSAIC_S3_REGION."""
    kw = build_s3_client_kwargs()
    if ICEBERG_CATALOG_NAME != "rest":
        return {"from_env": True, "region": kw["region_name"]}
    endpoint = kw["endpoint_url"]
    if "://" not in endpoint:
        endpoint = "http://" + endpoint
    path_style = os.environ.get("MOSAIC_S3_PATH_STYLE", "true")
    return {
        "endpoint": endpoint,
        "region": kw["region_name"],
        "access_key_id": kw["aws_access_key_id"],
        "secret_access_key": kw["aws_secret_access_key"],
        "path_style": path_style.strip().lower() in ("1", "true", "yes"),
    }


def write_empty_export(path: str, schema: Any) -> None:
    """An export of no rows (``export_query``'s file for an empty
    result)."""
    with pq.ParquetWriter(path, schema, compression="brotli") as writer:
        writer.write_table(schema.empty_table())


class PgCohort:
    """The cohort as PostgreSQL gives it, for ``run_rust_engine``: the
    contigs, the sample ids, the removed ids, the stale reason, and each
    contig's two exports, made into the driver's temporary directory over
    the region (``SITES_EXPORT_SQL``, ``ALLELES_EXPORT_SQL``)."""

    def __init__(
        self,
        conn: Any,
        contigs: List[Tuple[int, str, int]],
        removed_ids: List[int],
        stale: Optional[str],
    ) -> None:
        self.conn = conn
        self.contigs = contigs
        self.removed_ids = removed_ids
        self.stale = stale

    def sample_ids(self) -> Dict[str, int]:
        """{name: id} of the active samples."""
        return {
            name: int(sid) for sid, name in active_samples(self.conn).items()
        }

    def contig_exports(
        self, chrom_idx: int, lo: int, hi: int, tmpdir: str
    ) -> Tuple[str, str, int, str]:
        """(sites path, alleles path, the sites rows, the line's text)."""
        params = {"c": int(chrom_idx), "lo": int(lo), "hi": int(hi)}
        sites_path = os.path.join(tmpdir, f"sites.c{chrom_idx}.parquet")
        alleles_path = os.path.join(tmpdir, f"alleles.c{chrom_idx}.parquet")
        n_sites = export_query(
            self.conn,
            SITES_EXPORT_SQL,
            params,
            SITES_EXPORT_SCHEMA,
            sites_path,
        )
        n_alleles = export_query(
            self.conn,
            ALLELES_EXPORT_SQL,
            params,
            ALLELES_EXPORT_SCHEMA,
            alleles_path,
        )
        text = f"exported {n_sites} sites and {n_alleles} alleles rows"
        return sites_path, alleles_path, n_sites, text


class ExportCohort:
    """The cohort as a cohort export gives it (``--from-export DIR``):
    everything ``PgCohort`` gives, from ``cohort.json`` and the contig
    exports, which were checked before (``check_contig_exports``). The
    exports cover whole contigs; the binary reads the region's rows and
    finds a unit's deletion lookback itself. A contig without sites has no
    files: two empty exports are written into the driver's temporary
    directory, as the PostgreSQL path's empty export."""

    def __init__(self, directory: str, cohort: Dict[str, Any]) -> None:
        self.directory = os.path.abspath(directory)
        self.contigs = [
            (int(c["index"]), str(c["name"]), int(c["length"]))
            for c in cohort["contigs"]
        ]
        self.has_sites = {
            int(c["index"]): bool(c["has_sites"]) for c in cohort["contigs"]
        }
        self.ids = {str(s["name"]): int(s["id"]) for s in cohort["samples"]}
        self.removed_ids = sorted(int(s["id"]) for s in cohort["removed"])
        self.stale = cohort["state"]["stale"]
        # The sites rows of each checked contig export.
        self.rows: Dict[int, Dict[str, int]] = {}

    def sample_ids(self) -> Dict[str, int]:
        return dict(self.ids)

    def contig_exports(
        self, chrom_idx: int, lo: int, hi: int, tmpdir: str
    ) -> Tuple[str, str, int, str]:
        if not self.has_sites[chrom_idx]:
            sites_path = os.path.join(tmpdir, f"sites.c{chrom_idx}.parquet")
            alleles_path = os.path.join(
                tmpdir, f"alleles.c{chrom_idx}.parquet"
            )
            write_empty_export(sites_path, SITES_EXPORT_SCHEMA)
            write_empty_export(alleles_path, ALLELES_EXPORT_SCHEMA)
            return sites_path, alleles_path, 0, "no sites (none exported)"
        names = contig_export_names(chrom_idx)
        rows = self.rows[chrom_idx]
        text = (
            f"{rows['sites']} sites and {rows['alleles']} alleles rows from "
            f"the export {self.directory}"
        )
        return (
            os.path.join(self.directory, names["sites"]),
            os.path.join(self.directory, names["alleles"]),
            int(rows["sites"]),
            text,
        )


def run_rust_engine(
    args: Any,
    source: Any,
    get_catalog: Any,
    samples: List[str],
    region: Optional[Tuple[int, str, int, int, int]],
    subset: bool = False,
) -> None:
    """The exports, the manifest, the binary, the closing lines. Exits 1
    when the binary is missing (before any export) or fails (its partial
    output deleted). ``source`` is a ``PgCohort`` or an ``ExportCohort``:
    the contigs, the sample ids, the removed ids, the stale reason and the
    contigs' exports; ``get_catalog()`` gives the Iceberg catalog (called
    after the worker lookup). ``samples`` are the header's columns: every
    active sample, or under ``subset`` (K4) the subset in the header's
    order, whose data files alone are planned and whose run skips the
    catalog cross-check."""
    contigs_data = source.contigs
    removed = source.removed_ids
    stale = source.stale
    worker = find_joint_call_worker()
    if worker is None:
        print(f"Error: {missing_worker_remedy()}", file=sys.stderr)
        sys.exit(1)
    threads = args.threads
    window = args.window
    cap = args.max_alt_alleles
    trim = bool(args.trim_alt_alleles)
    min_ac = args.min_ac
    # K5's options are named only when set (the line is K4's otherwise).
    k5 = (" trim_alt_alleles=true" if trim else "") + (
        f" min_ac={min_ac}" if min_ac > 0 else ""
    )
    print(
        f"engine=rust threads={threads} window={window} "
        f"max_alt_alleles={cap}{k5}",
        file=sys.stderr,
    )

    id_of = source.sample_ids()
    subset_ids = [int(id_of[n]) for n in samples] if subset else None
    output = args.output
    pass_fds: Tuple[int, ...] = ()
    to_delete: Optional[str] = output
    if output in ("/dev/stdout", "-"):
        # The binary's stdout carries its summary line: it writes the VCF
        # to the driver's stdout through an inherited descriptor.
        sys.stdout.flush()
        fd = os.dup(sys.stdout.fileno())
        pass_fds = (fd,)
        output = f"/dev/fd/{fd}"
        to_delete = None
    elif output.startswith("/dev/"):
        to_delete = None
    elif output.endswith(".gz") and os.path.exists(output + ".tbi"):
        # The joint caller writes no tabix index: an index
        # left beside the path by an earlier run would not match.
        os.remove(output + ".tbi")

    tmpdir = tempfile.mkdtemp(prefix="mosaic-joint-call-")
    spill_path = spill_dir_path(
        args.output, tmpdir, args.spill_dir, secrets.token_hex(4)
    )
    try:
        iceberg_catalog = get_catalog()
        site_info_table = iceberg_catalog.load_table(
            f"{ICEBERG_NAMESPACE}.sample_site_info"
        )
        ref_blocks_table = iceberg_catalog.load_table(
            f"{ICEBERG_NAMESPACE}.sample_ref_blocks"
        )
        manifest_contigs = []
        for chrom_idx, chrom_name, chrom_length in contigs_data:
            if region is not None and region[0] != chrom_idx:
                continue
            lo, hi = (
                (1, chrom_length) if region is None else (region[3], region[4])
            )
            t0 = time.perf_counter()
            sites_path, alleles_path, n_sites, exported = (
                source.contig_exports(chrom_idx, lo, hi, tmpdir)
            )
            # A contig without sites writes no record: its data files are
            # not needed (nor planned).
            ssi, ssi_all = (
                plan_data_files(site_info_table, chrom_idx, subset_ids)
                if n_sites
                else ([], 0)
            )
            srb, srb_all = (
                plan_data_files(ref_blocks_table, chrom_idx, subset_ids)
                if n_sites
                else ([], 0)
            )
            # Under a subset: the files planned of the partition's.
            of = (f" of {ssi_all}", f" of {srb_all}") if subset else ("", "")
            print(
                f"{chrom_name}: {exported}; "
                f"{len(ssi)}{of[0]} sample_site_info and "
                f"{len(srb)}{of[1]} sample_ref_blocks data files "
                f"({time.perf_counter() - t0:.2f} s)",
                file=sys.stderr,
            )
            manifest_contigs.append(
                {
                    "index": int(chrom_idx),
                    "name": chrom_name,
                    "length": int(chrom_length),
                    "sites": sites_path,
                    "alleles": alleles_path,
                    "sample_site_info": ssi,
                    "sample_ref_blocks": srb,
                }
            )
        manifest = {
            "output": output,
            "header_lines": vcf_header_lines(
                [(name, length) for _, name, length in contigs_data],
                samples,
                cap,
                trim,
                min_ac,
            ),
            "samples": [{"id": int(id_of[n]), "name": n} for n in samples],
            "removed_ids": [int(r) for r in removed],
            "contigs": manifest_contigs,
            "s3": s3_settings(),
            "window": int(window),
            "region": (
                None
                if region is None
                else {
                    "contig": region[1],
                    "start": int(region[3]),
                    "end": int(region[4]),
                }
            ),
            "tmpdir": tmpdir,
            "spill_dir": spill_path,
            "stale": bool(stale),
        }
        if subset:
            # K4: the binary skips the catalog cross-check. Written only
            # under a subset, so a whole-cohort manifest is the one before.
            manifest["subset"] = True
        manifest_path = os.path.join(tmpdir, "manifest.json")
        with open(manifest_path, "w") as fh:
            json.dump(manifest, fh)

        cmd = [
            worker,
            "--manifest",
            manifest_path,
            "--threads",
            str(threads),
            "--window",
            str(window),
            "--unit-buffer-bytes",
            str(args.unit_buffer_bytes),
            "--max-alt-alleles",
            str(cap),
        ]
        # K5: forwarded only when set (the binary's defaults are off).
        if trim:
            cmd.append("--trim-alt-alleles")
        if min_ac > 0:
            cmd += ["--min-ac", str(min_ac)]
        sys.stderr.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, pass_fds=pass_fds)
        assert proc.stdout is not None
        stdout = proc.stdout.read()
        rc = proc.wait()
        summary = None
        if rc == 0:
            lines = [x for x in stdout.decode().splitlines() if x.strip()]
            try:
                summary = json.loads(lines[-1]) if lines else None
            except ValueError:
                summary = None
            if not isinstance(summary, dict) or summary.get("kind") != (
                "summary"
            ):
                summary = None
        if summary is None:
            if to_delete is not None and os.path.exists(to_delete):
                os.remove(to_delete)
            if rc < 0:
                why = f"was killed by signal {-rc}"
            elif rc != 0:
                why = f"exited with code {rc}"
            else:
                why = "printed no summary line"
            print(
                f"Error: the Rust joint caller ({worker}) {why}",
                file=sys.stderr,
            )
            sys.exit(1)
    finally:
        for fd in pass_fds:
            os.close(fd)
        # The binary removes its spill directory itself; again here, in
        # every case (a binary killed by a signal cannot).
        shutil.rmtree(spill_path, ignore_errors=True)
        shutil.rmtree(tmpdir, ignore_errors=True)

    sec = summary["seconds"]
    ph = sec["phases"]
    tables = summary["tables"]
    held = summary["held"]
    print(
        f"{JOINT_CALL_WORKER}: {summary['units']} unit(s), "
        f"{summary.get('samples', len(samples))} sample(s), "
        f"{sec['total']:.2f} s (footers {sec['footers']:.2f}, units "
        f"{sec['units']:.2f}: exports {ph['exports']:.2f}, "
        f"sample_site_info {ph['site_rows']:.2f}, sample_ref_blocks "
        f"{ph['ref_blocks']:.2f}, calls {ph['calls']:.2f}, format "
        f"{ph['format']:.2f}, write {ph['write']:.2f}; writer "
        f"{sec['writer']:.2f}); held at most {held['peak_units']} "
        f"unit(s), {held['peak_bytes']} bytes; spilled "
        f"{held['spilled_units']} unit(s), {held['spilled_bytes']} bytes "
        f"(copy {sec['copy']:.2f} s); bytes fetched: "
        + ", ".join(
            f"{t} {tables[t]['bytes_fetched']} of {tables[t]['file_bytes']}"
            for t in ("sample_site_info", "sample_ref_blocks")
        )
        + "; data files read: "
        + ", ".join(
            f"{t} {tables[t]['files']}"
            for t in ("sample_site_info", "sample_ref_blocks")
        ),
        file=sys.stderr,
    )
    print(
        f"\nCompleted! Total processed: {summary['records']} variants",
        file=sys.stderr,
    )
    cap_counts = summary.get("cap")
    if summary.get("subset"):
        # K4: the binary did not run the cross-check (its counts are the
        # whole cohort's).
        print(SUBSET_CROSS_CHECK_LINE, file=sys.stderr)
        for line in option_lines(summary):
            print(line, file=sys.stderr)
        return
    cc = CatalogCrossCheck(stale=bool(stale))
    reasons = {
        "stale": cc.reason,
        "unresolved": cc.UNRESOLVED,
        "other": "other",
    }
    check = summary["cross_check"]
    cc.records = int(check["records"])
    cc.differing = {
        str(reasons.get(k) or k): int(v) for k, v in check["differing"].items()
    }
    cc.first = list(check["first"])
    if cap_counts is not None:
        cc.capped = int(cap_counts["records_capped"])
    for line in cc.lines():
        print(line, file=sys.stderr)
    for line in option_lines(summary):
        print(line, file=sys.stderr)


# ========== Cohort generation check ==========


def _fmt_batch_ids(ids: List[int], limit: int = 10) -> str:
    if len(ids) <= limit:
        return "[" + ",".join(str(i) for i in ids) + "]"
    return f"[{','.join(str(i) for i in ids[:limit])},... ({len(ids)} total)]"


def check_cohort_generation(conn: Any) -> Optional[str]:
    """Why the PostgreSQL aggregates are not the ingested cohort's, or None.

    Reads the ``cohort_generation`` view (the latest aggregation run) and the
    completed ingest batches. A run that is not complete may have swapped in
    only some chromosomes' partitions (the others still hold the previous
    generation), and a run over a different batch set leaves batches out:
    either way the VCF would silently mix cohorts.
    """
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('cohort_generation') IS NOT NULL")
        if not cur.fetchone()[0]:
            return (
                "the database has no cohort_generation view (initialized "
                "before it existed; re-run `mosaic init`)"
            )
        cur.execute(
            "SELECT agg_run_id, batch_ids, status FROM cohort_generation"
        )
        row = cur.fetchone()
        cur.execute(
            "SELECT batch_id FROM ingestion_progress "
            "WHERE status = 'complete' ORDER BY batch_id"
        )
        completed = [b for (b,) in cur.fetchall()]
    if row is None:
        return "no aggregation run exists: run `mosaic aggregate` first"
    agg_run_id, batch_ids, status = row
    if status != "complete":
        return (
            f"the latest aggregation run (agg_run_id={agg_run_id}) is "
            f"{status}: resume it with `mosaic aggregate`"
        )
    aggregated = sorted(batch_ids or [])
    if aggregated != completed:
        missing = sorted(set(completed) - set(aggregated))
        extra = sorted(set(aggregated) - set(completed))
        return (
            f"the latest aggregation run (agg_run_id={agg_run_id}) does not "
            f"cover the completed ingest batches (not aggregated: "
            f"{_fmt_batch_ids(missing)}; aggregated but not complete: "
            f"{_fmt_batch_ids(extra)}): run `mosaic aggregate`"
        )
    return None


# ========== The cohort export ==========
#
# A directory that stands in for PostgreSQL in a joint call. `cohort.json`
# holds the cohort state,
# the contigs, the active samples in the header's order and the removed
# samples; per contig, by index, `sites.c<i>.parquet` and
# `alleles.c<i>.parquet` (the two exports over the whole contig) and, written
# last, `export.c<i>.json` (the contig's state and row counts: its
# completion marker). `--from-export DIR` joint-calls from it with no
# PostgreSQL connection; its output is the PostgreSQL path's, byte for byte.

EXPORT_FORMAT = 1
COHORT_FILE = "cohort.json"
# The keys of the cohort state, in the order the docs list them (the line
# is serialized with sorted keys).
COHORT_STATE_KEYS = (
    "format",
    "agg_run_id",
    "status",
    "aggregated_at",
    "completed_batches",
    "active_samples",
    "removed_samples",
    "samples_sha256",
    "stale",
)


class ExportError(Exception):
    """An export that cannot serve the call or the contig export: a
    refusal (exit 2), the same on a retry."""


def contig_export_names(chrom_idx: int) -> Dict[str, str]:
    """A contig export's three file names, by contig index (a contig name
    can hold characters a file name cannot)."""
    return {
        "sites": f"sites.c{int(chrom_idx)}.parquet",
        "alleles": f"alleles.c{int(chrom_idx)}.parquet",
        "export": f"export.c{int(chrom_idx)}.json",
    }


def begin_snapshot(conn: Any) -> None:
    """Every query until the connection's commit or close in one
    ``REPEATABLE READ``, read-only transaction (one snapshot)."""
    conn.set_session(
        isolation_level=psycopg2.extensions.ISOLATION_LEVEL_REPEATABLE_READ,
        readonly=True,
    )


def samples_digest(rows: List[Tuple[int, str, str]]) -> str:
    """SHA-256 of the canonical text of the ``samples`` rows (id, name,
    status), ordered by id: compact JSON, UTF-8."""
    canon = [[int(i), str(n), str(s)] for i, n, s in sorted(rows)]
    text = json.dumps(canon, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def cohort_state(conn: Any, stale: Optional[str]) -> Dict[str, Any]:
    """The cohort state: the ``cohort_generation``
    view's row (``agg_run_id``, ``status``, ``aggregated_at``: its
    ``updated_at`` as the database holds it, a ``timestamp`` with no time
    zone, rendered by the query; all three None without the view or a
    run), the completed ingest batches, the active and removed sample
    counts, the samples' digest (``samples_digest``) and ``stale`` (the
    gate's reason the caller let through, or None). No time of the call,
    nothing about listed names: the same text until the cohort changes."""
    agg_run_id = status = aggregated_at = None
    with conn.cursor() as cur:
        cur.execute("SELECT to_regclass('cohort_generation') IS NOT NULL")
        if cur.fetchone()[0]:
            cur.execute(
                "SELECT agg_run_id, status, to_char(updated_at, "
                "'YYYY-MM-DD\"T\"HH24:MI:SS.US') FROM cohort_generation"
            )
            row = cur.fetchone()
            if row is not None:
                agg_run_id = int(row[0])
                status = str(row[1])
                aggregated_at = row[2]
        cur.execute(
            "SELECT count(*) FROM ingestion_progress WHERE status = 'complete'"
        )
        completed = int(cur.fetchone()[0])
        cur.execute("SELECT id, sample_name, status FROM samples ORDER BY id")
        rows = [(int(i), n, s) for i, n, s in cur.fetchall()]
    return {
        "format": EXPORT_FORMAT,
        "agg_run_id": agg_run_id,
        "status": status,
        "aggregated_at": aggregated_at,
        "completed_batches": completed,
        "active_samples": sum(1 for r in rows if r[2] == "active"),
        "removed_samples": sum(1 for r in rows if r[2] == "removed"),
        "samples_sha256": samples_digest(rows),
        "stale": stale,
    }


def state_line(state: Dict[str, Any]) -> str:
    """The cohort state as ``--cohort-state`` prints it: one line, sorted
    keys, compact separators."""
    return json.dumps(state, sort_keys=True, separators=(",", ":"))


def state_differences(a: Dict[str, Any], b: Dict[str, Any]) -> List[str]:
    """The keys whose values differ between two states, sorted."""
    return sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))


def write_json_atomic(path: str, obj: Any) -> None:
    """``obj`` as deterministic JSON (sorted keys, one key per line) at
    ``path``: written to a temporary name beside it and renamed; nothing
    left behind on a failure."""
    tmp = f"{path}.tmp-{secrets.token_hex(4)}"
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, sort_keys=True, indent=1)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def removed_samples(conn: Any) -> List[Tuple[int, str]]:
    """(id, name) of the tombstoned samples, by id."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, sample_name FROM samples WHERE status = 'removed' "
            "ORDER BY id"
        )
        return [(int(i), str(n)) for i, n in cur.fetchall()]


def cohort_document(
    state: Dict[str, Any],
    contigs: List[Tuple[int, str, int, bool]],
    samples: List[Tuple[int, str]],
    removed: List[Tuple[int, str]],
) -> Dict[str, Any]:
    """``cohort.json``'s content: the state, the contigs (index, name,
    length, whether ``site_info`` has a row of it), the active samples in
    the header's order and the removed samples by id."""
    return {
        "format": EXPORT_FORMAT,
        "state": state,
        "contigs": [
            {
                "index": int(i),
                "name": n,
                "length": int(length),
                "has_sites": bool(h),
            }
            for i, n, length, h in contigs
        ],
        "samples": [{"id": int(i), "name": n} for i, n in samples],
        "removed": [{"id": int(i), "name": n} for i, n in removed],
    }


def _load_json(path: str, what: str) -> Dict[str, Any]:
    try:
        with open(path) as fh:
            obj = json.load(fh)
    except FileNotFoundError:
        raise ExportError(f"{what} {path} does not exist")
    except (OSError, ValueError) as e:
        raise ExportError(f"{what} {path} cannot be read: {e}")
    if not isinstance(obj, dict):
        raise ExportError(f"{what} {path} is not a JSON object")
    if obj.get("format") != EXPORT_FORMAT:
        raise ExportError(
            f"{what} {path} has format {obj.get('format')!r}; this driver "
            f"reads format {EXPORT_FORMAT}"
        )
    return obj


def load_cohort_file(directory: str) -> Dict[str, Any]:
    """``DIR/cohort.json``, checked for its format and keys
    (ExportError otherwise)."""
    path = os.path.join(directory, COHORT_FILE)
    obj = _load_json(path, "the cohort export")
    for key in ("state", "contigs", "samples", "removed"):
        if key not in obj:
            raise ExportError(f"the cohort export {path} has no {key!r}")
    if not isinstance(obj["state"], dict) or (
        obj["state"].get("format") != EXPORT_FORMAT
    ):
        raise ExportError(
            f"the cohort export {path} holds a cohort state of another "
            "format"
        )
    return obj


def export_contig_entry(
    cohort: Dict[str, Any], name: str, directory: str
) -> Dict[str, Any]:
    """The contig ``name`` of ``cohort.json`` (ExportError when it has
    none)."""
    for c in cohort["contigs"]:
        if c["name"] == name:
            return c
    raise ExportError(
        f"contig {name!r} is not in the cohort export "
        f"{os.path.join(directory, COHORT_FILE)}"
    )


def parquet_rows(path: str) -> int:
    """A Parquet file's row count from its footer (ExportError when it
    cannot be read)."""
    try:
        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception as e:  # pyarrow raises several types
        raise ExportError(f"{path} cannot be read as Parquet: {e}")


def check_contig_exports(
    directory: str,
    cohort: Dict[str, Any],
    contigs: List[Dict[str, Any]],
) -> Dict[int, Dict[str, int]]:
    """For each of ``contigs`` (``cohort.json`` entries) that has sites:
    its three files present, its sidecar of this format, of this contig
    and of ``cohort.json``'s state, and each Parquet file's footer row
    count the sidecar's. Returns {index: {"sites": n, "alleles": n}};
    ExportError naming every problem otherwise. Reads only."""
    out: Dict[int, Dict[str, int]] = {}
    problems: List[str] = []
    for c in contigs:
        if not c["has_sites"]:
            continue
        idx, name = int(c["index"]), str(c["name"])
        names = contig_export_names(idx)
        paths = {k: os.path.join(directory, v) for k, v in names.items()}
        remedy = f"run --export-contig {directory} --region {name}"
        missing = [names[k] for k in names if not os.path.exists(paths[k])]
        if missing:
            problems.append(
                f"{name} has sites but the export holds no "
                f"{', '.join(missing)} ({remedy})"
            )
            continue
        side = _load_json(paths["export"], "the contig export")
        want_contig = {"index": idx, "name": name, "length": c["length"]}
        if side.get("contig") != want_contig:
            problems.append(
                f"{paths['export']} is the export of {side.get('contig')}, "
                f"not of {want_contig} ({remedy})"
            )
            continue
        diff = state_differences(side.get("state") or {}, cohort["state"])
        if diff:
            problems.append(
                f"{paths['export']} was made from another cohort state than "
                f"{COHORT_FILE}'s (differing: {', '.join(diff)}); {remedy}, "
                "or make a new export with --export-cohort if the cohort "
                "has changed"
            )
            continue
        rows = side.get("rows") or {}
        counted = {}
        for k in ("sites", "alleles"):
            n = parquet_rows(paths[k])
            counted[k] = n
            if rows.get(k) != n:
                problems.append(
                    f"{paths[k]} holds {n} rows, {names['export']} says "
                    f"{rows.get(k)} ({remedy})"
                )
        out[idx] = counted
    if problems:
        raise ExportError("; ".join(problems))
    return out


def export_contig_files(
    conn: Any, directory: str, contig: Dict[str, Any], state: Dict[str, Any]
) -> Dict[str, int]:
    """The contig's two exports over the whole contig into ``directory``
    (``export_query``), each under a temporary name then renamed, and last
    its sidecar (the completion marker; an earlier one removed first).
    Nothing temporary is left on a failure. Returns the rows."""
    idx = int(contig["index"])
    names = contig_export_names(idx)
    paths = {k: os.path.join(directory, v) for k, v in names.items()}
    token = secrets.token_hex(4)
    tmp = {k: f"{paths[k]}.tmp-{token}" for k in ("sites", "alleles")}
    params = {"c": idx, "lo": 1, "hi": int(contig["length"])}
    try:
        rows = {
            "sites": export_query(
                conn,
                SITES_EXPORT_SQL,
                params,
                SITES_EXPORT_SCHEMA,
                tmp["sites"],
            ),
            "alleles": export_query(
                conn,
                ALLELES_EXPORT_SQL,
                params,
                ALLELES_EXPORT_SCHEMA,
                tmp["alleles"],
            ),
        }
        if os.path.exists(paths["export"]):
            os.remove(paths["export"])
        for k in ("sites", "alleles"):
            os.replace(tmp[k], paths[k])
    finally:
        for p in tmp.values():
            if os.path.exists(p):
                os.remove(p)
    write_json_atomic(
        paths["export"],
        {
            "format": EXPORT_FORMAT,
            "contig": {
                "index": idx,
                "name": contig["name"],
                "length": int(contig["length"]),
            },
            "state": state,
            "rows": rows,
        },
    )
    return rows


# ========== Main Entry Point ==========


def _region_arg(value: str) -> Tuple[str, Optional[int], Optional[int]]:
    """argparse ``type`` for ``--region`` (a malformed value: exit 2)."""
    try:
        return parse_region(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e))


def build_parser() -> argparse.ArgumentParser:
    """The command line's parser."""
    parser = argparse.ArgumentParser(
        description="Joint call variants from Iceberg/PostgreSQL storage "
        f"(the driver of {JOINT_CALL_WORKER})"
    )
    # The default (stdout) is applied by parse_args after the checks, so
    # the modes that write no VCF can refuse an output path.
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output VCF file (default: stdout); a .gz path is BGZF, with "
        "no .tbi (an existing one beside it is removed).",
    )
    parser.add_argument(
        "--allow-stale",
        action="store_true",
        help="Joint-call even when the latest aggregation run (the "
        "cohort_generation view) is not complete or does not cover every "
        "completed ingest batch.",
    )
    parser.add_argument(
        "--region",
        type=_region_arg,
        default=None,
        metavar="CONTIG[:START-END]",
        help="Joint-call only this region: a contig of the contigs table, "
        "optionally with START-END (1-based, inclusive; END is clipped to "
        "the contig's length). The header is unchanged (every ##contig "
        "line, every active sample), so the records equal the whole run's "
        "records in the region. Default: every contig.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=os.cpu_count() or 1,
        help="Units in flight, the output written in order (default: the "
        "CPU count).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=DEFAULT_WINDOW,
        help="The unit width in bp (default: 10000000; 0 = one unit per "
        "contig). The output does not depend on it.",
    )
    parser.add_argument(
        "--unit-buffer-bytes",
        type=int,
        default=DEFAULT_UNIT_BUFFER_BYTES,
        metavar="N",
        help="A unit that is not the first unit not yet written buffers its "
        "compressed output in memory up to N bytes, then spills it to a "
        "file of its own (default: 67108864; 0 or 1: every chunk spills). "
        "The first unit not yet written streams to the output. The output "
        "does not depend on it.",
    )
    parser.add_argument(
        "--max-alt-alleles",
        type=int,
        default=DEFAULT_MAX_ALT_ALLELES,
        metavar="N",
        help="A record with more than N ALT alleles (* among them) keeps "
        "the N with the largest allele count (AC of its genotypes; ties to "
        "the earlier ALT) and is written as a projection of the uncapped "
        "record: a genotype carrying a dropped ALT is ./., AD / F1R2 / "
        "F2R1 / PL are restricted to the kept alleles, AC / AN / AF are "
        "recomputed; the header names the cap (default: 6; 0 = no cap).",
    )
    parser.add_argument(
        "--trim-alt-alleles",
        action="store_true",
        help="An ALT allele no written genotype names (AC 0) leaves the "
        "record, before the cap: the other ALTs keep their order, the GTs "
        "are re-indexed, AD / F1R2 / F2R1 / PL are restricted to the kept "
        "alleles, AC / AF lose its entry (AN and the other INFO fields are "
        "unchanged); a record left without ALTs is written with ALT '.'. "
        "The header names it (default: off).",
    )
    parser.add_argument(
        "--min-ac",
        type=int,
        default=0,
        metavar="K",
        help="Write only the records whose written genotypes hold at least "
        "K ALT observations (the sum of the record's INFO/AC, after "
        "--samples, --trim-alt-alleles and --max-alt-alleles); the header "
        "names it (default: 0, off).",
    )
    who = parser.add_mutually_exclusive_group()
    who.add_argument(
        "--samples",
        default=None,
        metavar="NAME[,NAME...]",
        help="Joint-call only these active samples (comma-separated "
        "names). Each record is the whole-cohort record projected onto "
        "their columns, in the header's order: * leaves the ALT list when "
        "none of them carries it, a FORMAT key none of them has is "
        "dropped, AC / AN / AF are theirs; QUAL and the other INFO fields "
        "stay the cohort's, and every cohort site is still a record. Only "
        "the data files that can hold one of them are read; the catalog "
        "cross-check is skipped. An unknown, removed or repeated name, or "
        "an empty list: exit 2.",
    )
    who.add_argument(
        "--samples-file",
        default=None,
        metavar="FILE",
        help="As --samples, one name per line (blank lines ignored).",
    )
    parser.add_argument(
        "--spill-dir",
        default=None,
        metavar="DIR",
        help="Spill under DIR, in a directory of the driver's own "
        "naming that it removes when it ends (default: beside the output "
        "path, <output>.spill-<token>, or the driver's temporary directory "
        "for a /dev/... output).",
    )
    # The cohort export: a directory that stands in for PostgreSQL in a
    # joint call.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--cohort-state",
        action="store_true",
        help="Run the driver's checks (samples.status present, the "
        "--samples / --samples-file names, the cohort_generation gate) and "
        "print the cohort state as one JSON line on stdout: the "
        "aggregation run, its status and time, the completed batches, the "
        "sample counts and digest, the stale reason (null unless "
        "--allow-stale let a stale cohort through). Writes no VCF.",
    )
    mode.add_argument(
        "--export-cohort",
        default=None,
        metavar="DIR",
        help="Write DIR/cohort.json: the cohort state, the contigs, the "
        "active and removed samples. A stale cohort: exit 2 unless "
        "--allow-stale, which records the reason. Writes no VCF.",
    )
    mode.add_argument(
        "--export-contig",
        default=None,
        metavar="DIR",
        help="With --region CONTIG (a whole contig): write that contig's "
        "sites and alleles exports beside DIR/cohort.json, after checking "
        "that the database's cohort state is the one cohort.json holds "
        "(another: exit 2). Writes no VCF.",
    )
    mode.add_argument(
        "--from-export",
        default=None,
        metavar="DIR",
        help="Joint-call from an export made by --export-cohort and "
        "--export-contig, with no PostgreSQL connection; the output is the "
        "one a joint call from PostgreSQL writes. Every call option works "
        "as without it, but --allow-stale (it belongs to the export).",
    )
    return parser


def parse_args(
    argv: Optional[List[str]] = None,
) -> Tuple[argparse.ArgumentParser, argparse.Namespace]:
    """``build_parser``'s parser and its arguments, validated (an invalid
    --threads, --window, --unit-buffer-bytes, --max-alt-alleles or
    --min-ac, or an
    unknown option such as the removed --engine / --step-size: exit 2); the
    parser for main's later ``--region`` error."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.threads < 1:
        parser.error("--threads must be at least 1")
    if args.window < 0:
        parser.error("--window must be 0 or more")
    if args.unit_buffer_bytes < 0:
        parser.error("--unit-buffer-bytes must be 0 or more")
    if args.max_alt_alleles < 0:
        parser.error("--max-alt-alleles must be 0 or more")
    if args.min_ac < 0:
        parser.error("--min-ac must be 0 or more")
    _check_mode_options(parser, args)
    if not writes_no_vcf(args) and args.output is None:
        args.output = "/dev/stdout"
    return parser, args


def writes_no_vcf(args: argparse.Namespace) -> bool:
    """Whether ``args`` select one of the three modes that read PostgreSQL
    and write no VCF (``--cohort-state``, ``--export-cohort``,
    ``--export-contig``)."""
    return bool(
        args.cohort_state
        or args.export_cohort is not None
        or args.export_contig is not None
    )


def _check_mode_options(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """The options each mode of the cohort export refuses: exit 2
    through ``parser.error``."""
    if args.cohort_state:
        mode = "--cohort-state"
        refused = ["output", "region", "spill_dir"]
    elif args.export_cohort is not None:
        mode = "--export-cohort"
        refused = ["output", "region", "samples", "samples_file", "spill_dir"]
    elif args.export_contig is not None:
        mode = "--export-contig"
        refused = [
            "output",
            "samples",
            "samples_file",
            "spill_dir",
            "allow_stale",
        ]
        if args.region is None:
            parser.error("--export-contig needs --region CONTIG")
        if args.region[1] is not None:
            parser.error(
                "--export-contig exports a whole contig: --region CONTIG, "
                "with no range"
            )
    elif args.from_export is not None:
        mode = "--from-export"
        refused = ["allow_stale"]
    else:
        return
    names = {
        "output": "an output path",
        "region": "--region",
        "samples": "--samples",
        "samples_file": "--samples-file",
        "spill_dir": "--spill-dir",
        "allow_stale": "--allow-stale",
    }
    given = [
        names[k] for k in refused if getattr(args, k) not in (None, False)
    ]
    if given:
        why = (
            " (it belongs to the export: --export-cohort)"
            if mode == "--from-export"
            else ""
        )
        parser.error(f"{mode} does not take {', '.join(given)}{why}")


def _subset_option(args: argparse.Namespace) -> str:
    return "--samples" if args.samples is not None else "--samples-file"


def resolve_subset_or_exit(
    parser: argparse.ArgumentParser,
    args: argparse.Namespace,
    active: List[str],
    removed_of: Any,
) -> Optional[List[str]]:
    """``--samples`` / ``--samples-file`` (K4) against the active names
    (header order): the subset's columns, or None without the options;
    exit 2 naming the offenders. ``removed_of(names)`` gives those of the
    names that are tombstoned."""
    try:
        names = subset_names(args)
    except OSError as e:
        parser.error(f"argument --samples-file: {e}")
    if names is None:
        return None
    try:
        return resolve_subset(names, active, removed_of(names))
    except ValueError as e:
        parser.error(f"argument {_subset_option(args)}: {e}")
    return None  # unreachable: parser.error exits


def gate_or_exit(args: argparse.Namespace, stale: Optional[str]) -> None:
    """The cohort_generation gate's refusal (exit 2) unless
    ``--allow-stale``, which prints its warning."""
    if stale and not args.allow_stale:
        print(f"Error: {stale}", file=sys.stderr)
        print(
            "Refusing to joint-call stale aggregates; pass --allow-stale "
            "to override.",
            file=sys.stderr,
        )
        sys.exit(2)
    if stale:
        print(f"Warning (--allow-stale): {stale}", file=sys.stderr)


def print_sample_lines(
    samples: List[str], n_active: int, subset: bool, n_removed: int
) -> None:
    """The "Sample subset:" line (K4) and the tombstone line."""
    if subset:
        print(
            f"Sample subset: {len(samples)} of {n_active} active samples: "
            + ", ".join(samples),
            file=sys.stderr,
        )
    if n_removed:
        print(
            f"Excluding {n_removed} tombstoned sample(s) "
            "(samples.status = 'removed') from the header, the FORMAT "
            "columns, the per-sample reads and AC / AN / AF (from the "
            "emitted genotypes). The other INFO fields (DP, QD, MQ, FS, "
            "SOR, InbreedingCoeff), from the PostgreSQL aggregates, "
            "include them until the `mosaic aggregate` that applies "
            "their removal.",
            file=sys.stderr,
        )


def status_or_exit(conn: Any) -> None:
    """A database from before the sample tombstone (samples.status): exit
    1 with the remedy."""
    if samples_status_missing(conn):
        print(f"Error: {missing_status_remedy()}", file=sys.stderr)
        sys.exit(1)


def call_from_postgres(
    parser: argparse.ArgumentParser, args: argparse.Namespace, conn: Any
) -> None:
    """A joint call from PostgreSQL (no mode option)."""
    # --region against the contigs table (an unknown contig: exit 2,
    # before the gate and before anything is written).
    with conn.cursor() as cur:
        cur.execute("SELECT id, chromosome, length FROM contigs ORDER BY id")
        all_contigs = cur.fetchall()
    region = None
    if args.region is not None:
        try:
            region = resolve_region(args.region, all_contigs)
        except ValueError as e:
            parser.error(f"argument --region: {e}")

    status_or_exit(conn)

    # --samples / --samples-file (K4): every name an active sample, given
    # once (exit 2 naming the offenders, before the gate).
    samples = active_sample_names(conn)
    chosen = resolve_subset_or_exit(
        parser, args, samples, lambda n: removed_sample_names(conn, n)
    )
    subset = chosen is not None
    if chosen is not None:
        samples = chosen

    # Refuse to emit a VCF from aggregates that are not the ingested
    # cohort's (before anything is written to the output).
    stale = check_cohort_generation(conn)
    gate_or_exit(args, stale)

    iceberg_catalog = load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG)

    # The active samples, or the subset (a tombstoned sample has no
    # column).
    removed = removed_sample_ids(conn)
    print_sample_lines(
        samples,
        len(active_sample_names(conn)) if subset else 0,
        subset,
        len(removed),
    )
    run_rust_engine(
        args,
        PgCohort(conn, all_contigs, removed, stale),
        lambda: iceberg_catalog,
        samples,
        region,
        subset,
    )


def print_cohort_state(
    parser: argparse.ArgumentParser, args: argparse.Namespace, conn: Any
) -> None:
    """``--cohort-state``: the driver's checks, then the state line on
    stdout (one line, nothing else)."""
    begin_snapshot(conn)
    status_or_exit(conn)
    resolve_subset_or_exit(
        parser,
        args,
        active_sample_names(conn),
        lambda n: removed_sample_names(conn, n),
    )
    stale = check_cohort_generation(conn)
    gate_or_exit(args, stale)
    line = state_line(cohort_state(conn, stale))
    sys.stdout.write(line + "\n")
    sys.stdout.flush()


def export_cohort(
    parser: argparse.ArgumentParser, args: argparse.Namespace, conn: Any
) -> None:
    """``--export-cohort DIR``: ``DIR/cohort.json`` from one snapshot."""
    begin_snapshot(conn)
    status_or_exit(conn)
    stale = check_cohort_generation(conn)
    gate_or_exit(args, stale)
    state = cohort_state(conn, stale)
    with conn.cursor() as cur:
        cur.execute("""
            SELECT c.id, c.chromosome, c.length,
                   EXISTS (SELECT 1 FROM site_info s
                           WHERE s.chromosome_index = c.id)
            FROM contigs c ORDER BY c.id
        """)
        contigs = [(int(i), n, int(ln), bool(h)) for i, n, ln, h in cur]
    ids = active_samples(conn)
    by_name = {name: sid for sid, name in ids.items()}
    samples = [(by_name[n], n) for n in active_sample_names(conn)]
    removed = removed_samples(conn)
    conn.rollback()  # the read-only snapshot ends
    doc = cohort_document(state, contigs, samples, removed)
    directory = args.export_cohort
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, COHORT_FILE)
    write_json_atomic(path, doc)
    print(
        f"Wrote {path}: agg_run_id={state['agg_run_id']}, "
        f"{len(contigs)} contig(s) ({sum(1 for c in contigs if c[3])} with "
        f"sites), {len(samples)} active and {len(removed)} removed "
        "sample(s)",
        file=sys.stderr,
    )


def export_contig_prechecks(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """``--export-contig DIR --region CONTIG``'s checks before the
    connection: ``DIR/cohort.json`` readable, of this format, holding the
    contig (exit 2 otherwise). Returns (cohort.json, the contig)."""
    try:
        cohort = load_cohort_file(args.export_contig)
        contig = export_contig_entry(
            cohort, args.region[0], args.export_contig
        )
    except ExportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    return cohort, contig


def export_contig(
    args: argparse.Namespace,
    conn: Any,
    cohort: Dict[str, Any],
    contig: Dict[str, Any],
) -> None:
    """``--export-contig DIR --region CONTIG``: in one snapshot, the
    database's cohort state against ``cohort.json``'s (another: exit 2,
    nothing written), then the contig's exports."""
    begin_snapshot(conn)
    status_or_exit(conn)
    state = cohort_state(conn, check_cohort_generation(conn))
    diff = state_differences(state, cohort["state"])
    if diff:
        print(
            "Error: the database's cohort state differs from "
            f"{os.path.join(args.export_contig, COHORT_FILE)}'s (differing: "
            f"{', '.join(diff)}): the cohort has changed since the export; "
            "make a new export with --export-cohort NEW_DIR",
            file=sys.stderr,
        )
        sys.exit(2)
    t0 = time.perf_counter()
    rows = export_contig_files(conn, args.export_contig, contig, state)
    conn.rollback()  # the read-only snapshot ends
    print(
        f"{contig['name']}: exported {rows['sites']} sites and "
        f"{rows['alleles']} alleles rows to {args.export_contig} "
        f"({time.perf_counter() - t0:.2f} s)",
        file=sys.stderr,
    )


def call_from_export(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """``--from-export DIR``: a joint call with no PostgreSQL connection.
    Every refusal (exit 2) comes before the worker lookup and any catalog
    or S3 access; nothing in DIR is created, changed or removed."""
    directory = os.path.abspath(args.from_export)
    try:
        cohort = load_cohort_file(directory)
    except ExportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)
    source = ExportCohort(directory, cohort)

    region = None
    if args.region is not None:
        try:
            region = resolve_region(args.region, source.contigs)
        except ValueError as e:
            parser.error(f"argument --region: {e}")

    active = [str(s["name"]) for s in cohort["samples"]]
    removed_names = [str(s["name"]) for s in cohort["removed"]]
    samples = active
    chosen = resolve_subset_or_exit(
        parser, args, active, lambda n: removed_names
    )
    subset = chosen is not None
    if chosen is not None:
        samples = chosen

    in_scope = [
        c
        for c in cohort["contigs"]
        if region is None or int(c["index"]) == region[0]
    ]
    try:
        source.rows = check_contig_exports(directory, cohort, in_scope)
    except ExportError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(2)

    if source.stale:
        print(
            "Warning (the export was made under --allow-stale): "
            f"{source.stale}",
            file=sys.stderr,
        )

    print_sample_lines(samples, len(active), subset, len(source.removed_ids))
    # The Iceberg catalog after the worker lookup (run_rust_engine).
    run_rust_engine(
        args,
        source,
        lambda: load_catalog(ICEBERG_CATALOG_NAME, **ICEBERG_CONFIG),
        samples,
        region,
        subset,
    )


def _guarded(body: Any) -> None:
    """``body()`` with the driver's failure rule: an unexpected exception
    prints its error and traceback and exits 1."""
    try:
        body()
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


def main() -> None:
    parser, args = parse_args()

    if args.from_export is not None:
        _guarded(lambda: call_from_export(parser, args))
        return
    pre: Optional[Tuple[Dict[str, Any], Dict[str, Any]]] = None
    if args.export_contig is not None:
        pre = export_contig_prechecks(parser, args)

    conn = None

    def body() -> None:
        nonlocal conn
        conn = psycopg2.connect(**DB_CONFIG)  # type: ignore[call-overload]
        if args.cohort_state:
            print_cohort_state(parser, args, conn)
        elif args.export_cohort is not None:
            export_cohort(parser, args, conn)
        elif pre is not None:
            export_contig(args, conn, *pre)
        else:
            call_from_postgres(parser, args, conn)

    try:
        _guarded(body)
    finally:
        if conn is not None:
            conn.close()


if __name__ == "__main__":
    main()
