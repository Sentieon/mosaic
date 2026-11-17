# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The test harness's reader of the Rust ingest worker's local files and
its copies of the per-sample Arrow schemas.

Until the earlier Python ingest worker was removed these lived in ``mosaic/ingest.py`` beside the Python worker that
wrote the files and the Python combine that read them; since the Python
worker's removal nothing in production reads a sample-slice container
(the Rust worker writes and combines them: ``slices.rs``, ``combine.rs``),
so the reader moved here, unchanged, for the tests that read the worker's
local files (``worker_harness.local_rows``, the row-oracle signatures of
``test_rust_engine.py``).

The container format (one standalone one-row-group Parquet blob per
(chromosome_index, sample_id) slice of a window, a JSON trailer with each
blob's offset, length, compressed bytes and rows, then a 20-byte tail with
the magic and the version) is owned by the Rust worker's ``slices.rs``;
a plain Parquet file, a truncated container or another version is
refused.

The three Arrow schemas are the shapes the tests build per-sample and
genotype-matrix rows with (``test_remove.py``, the reader tests of
``test_genotype_builder.py``); ``tests/unit/test_harness_schemas.py``
checks that their field names and types equal the Iceberg tables'
(``mosaic.init_db``), so they cannot drift.
"""

import json
import os
import struct
from typing import Any, List, NamedTuple, Optional, Tuple

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.parquet as pq  # type: ignore[import-untyped]

SAMPLE_SITE_INFO_ARROW_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.int32(), nullable=False),
        pa.field("chromosome_index", pa.int32(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("del_start", pa.int32(), nullable=True),
        pa.field("del_end", pa.int32(), nullable=True),
        pa.field("alleles", pa.list_(pa.string()), nullable=True),
        pa.field("gt_1", pa.int32(), nullable=True),
        pa.field("gt_2", pa.int32(), nullable=True),
        pa.field("dp", pa.int32(), nullable=True),
        pa.field("ad", pa.list_(pa.int32()), nullable=True),
        pa.field("f1r2", pa.list_(pa.int32()), nullable=True),
        pa.field("f2r1", pa.list_(pa.int32()), nullable=True),
        pa.field("sb", pa.list_(pa.int32()), nullable=True),
        pa.field("pl", pa.list_(pa.int32()), nullable=True),
        pa.field("qual", pa.float64(), nullable=True),
        pa.field("site_dp", pa.int64(), nullable=True),
        pa.field("mqsq_1", pa.float64(), nullable=True),
        pa.field("mqsq_2", pa.int64(), nullable=True),
        pa.field("gtc", pa.list_(pa.float64()), nullable=True),
        pa.field("del_ref_spans", pa.list_(pa.int32()), nullable=True),
    ]
)

# sample_ref_blocks as ingest writes it (init_db.sample_ref_blocks_schema):
# sample_counts is the block's per-position batch_ref_blocks contribution.
SAMPLE_REF_BLOCKS_ARROW_SCHEMA = pa.schema(
    [
        pa.field("sample_id", pa.int32(), nullable=False),
        pa.field("chromosome_index", pa.int32(), nullable=False),
        pa.field("position_start", pa.int32(), nullable=False),
        pa.field("position_end", pa.int32(), nullable=False),
        pa.field("ad_1", pa.int32(), nullable=True),
        pa.field("ad_2", pa.int32(), nullable=True),
        pa.field("dp", pa.int32(), nullable=True),
        pa.field("pl_1", pa.int32(), nullable=True),
        pa.field("pl_2", pa.int32(), nullable=True),
        pa.field("pl_3", pa.int32(), nullable=True),
        pa.field("sample_counts", pa.int32(), nullable=True),
    ]
)


BATCH_GENOTYPES_ARROW_SCHEMA = pa.schema(
    [
        pa.field("batch_id", pa.int32(), nullable=False),
        pa.field("chromosome_index", pa.int32(), nullable=False),
        pa.field("position", pa.int32(), nullable=False),
        pa.field("n_samples", pa.int32(), nullable=False),
        pa.field("alleles", pa.list_(pa.string()), nullable=False),
        pa.field("gt_1", pa.binary(), nullable=False),
        pa.field("gt_2", pa.binary(), nullable=False),
        pa.field("gq", pa.binary(), nullable=False),
    ]
)


# ── Sample-slice containers: the per-sample tables' local files ─────────────
SAMPLE_SLICE_MAGIC = b"MOSAICSS"
SAMPLE_SLICE_VERSION = 1
SAMPLE_SLICE_TAIL = struct.Struct("<QI8s")


class SampleSliceError(ValueError):
    """A local per-sample file is not a well-formed sample-slice container
    (a plain Parquet file, a truncated container, a version this code does
    not read) or a blob does not match its trailer entry."""


class SampleSlice(NamedTuple):
    """One blob of a sample-slice container: the rows of one
    (chromosome_index, sample_id) slice of one window."""

    path: str
    chromosome_index: int
    sample_id: int
    offset: int
    length: int
    nbytes: int
    num_rows: int


def read_sample_slice_index(path: str) -> List[SampleSlice]:
    """Read a sample-slice container's trailer (its only metadata read):
    the slices in blob order. Raises SampleSliceError for anything but a
    complete container of SAMPLE_SLICE_VERSION: a plain Parquet file, a
    truncated container, blobs that do not tile the file, keys out of
    order."""
    size = os.path.getsize(path)
    tail_size = SAMPLE_SLICE_TAIL.size
    with open(path, "rb") as f:
        if size < tail_size:
            raise SampleSliceError(
                f"{path}: {size} bytes, too short for a sample-slice "
                "container (truncated?)"
            )
        f.seek(size - tail_size)
        trailer_len, version, magic = SAMPLE_SLICE_TAIL.unpack(
            f.read(tail_size)
        )
        if magic != SAMPLE_SLICE_MAGIC:
            raise SampleSliceError(
                f"{path}: not a sample-slice container (no "
                f"{SAMPLE_SLICE_MAGIC.decode()} tail: a plain Parquet file "
                "or a truncated container)"
            )
        if version != SAMPLE_SLICE_VERSION:
            raise SampleSliceError(
                f"{path}: sample-slice container version {version}, this "
                f"code reads version {SAMPLE_SLICE_VERSION}"
            )
        body = size - tail_size - trailer_len
        if trailer_len > size - tail_size:
            raise SampleSliceError(f"{path}: trailer overruns the file")
        f.seek(body)
        try:
            entries = json.loads(f.read(trailer_len))["slices"]
        except (ValueError, KeyError, TypeError) as e:
            raise SampleSliceError(f"{path}: unreadable trailer ({e})")
    slices = [SampleSlice(path, *(int(x) for x in e)) for e in entries]
    expect = 0
    prev: Optional[Tuple[int, int]] = None
    for sl in slices:
        key = (sl.chromosome_index, sl.sample_id)
        if sl.offset != expect or sl.length <= 0:
            raise SampleSliceError(
                f"{path}: blob {key} at {sl.offset} (+{sl.length}); "
                f"expected offset {expect}"
            )
        if prev is not None and key <= prev:
            raise SampleSliceError(
                f"{path}: blob {key} after {prev}: not in "
                "(chromosome_index, sample_id) order"
            )
        expect += sl.length
        prev = key
    if expect != body:
        raise SampleSliceError(
            f"{path}: blobs end at byte {expect}, trailer starts at {body}"
        )
    return slices


def read_sample_slice(sl: SampleSlice) -> Any:
    """Read and decode one blob. Checks that its row count matches the
    trailer and that, in every row group, the chromosome_index and
    sample_id statistics give min == max == the trailer's key."""
    with open(sl.path, "rb") as f:
        f.seek(sl.offset)
        data = f.read(sl.length)
    if len(data) != sl.length:
        raise SampleSliceError(
            f"{sl.path}: blob at {sl.offset} short read ({len(data)} of "
            f"{sl.length} bytes)"
        )
    pf = pq.ParquetFile(pa.BufferReader(data))
    md = pf.metadata
    want = {"chromosome_index": sl.chromosome_index, "sample_id": sl.sample_id}
    for g in range(md.num_row_groups):
        rg = md.row_group(g)
        for k in range(rg.num_columns):
            col = rg.column(k)
            if col.path_in_schema not in want:
                continue
            st = col.statistics
            expected = want[col.path_in_schema]
            if (
                st is None
                or not st.has_min_max
                or st.min != expected
                or st.max != expected
            ):
                raise SampleSliceError(
                    f"{sl.path}: blob at {sl.offset}: {col.path_in_schema} "
                    f"statistics {None if st is None else (st.min, st.max)}"
                    f", trailer says {expected}"
                )
    table = pf.read()
    if table.num_rows != sl.num_rows:
        raise SampleSliceError(
            f"{sl.path}: blob at {sl.offset} has {table.num_rows} rows, "
            f"trailer says {sl.num_rows}"
        )
    return table


def read_sample_slices(path: str) -> Any:
    """Every row of a sample-slice container, in blob order (tests and
    debugging; the combine reads blob by blob)."""
    tables = [read_sample_slice(sl) for sl in read_sample_slice_index(path)]
    if not tables:
        raise SampleSliceError(f"{path}: an empty container")
    return pa.concat_tables(tables)
