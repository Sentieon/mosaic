# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Read the per-batch genotype matrix (``batch_genotypes``) for a window.

Ingest writes one ``batch_genotypes`` row per (batch, variant site the batch
records): the production writer is the Rust ingest worker
(``rust/mosaic-ingest-worker/src/matrix.rs``; the only one).
``encode_gt_blob`` below is
the format's encoder beside its decoder (the reader's tests build their
inputs with it). Each row carries a local allele list and three fixed-length byte
vectors over the batch's sample slots: ``gt_1``, ``gt_2`` hold each slot's
local allele index in one or two bytes per slot, by the site's allele count
(signed int8 when every code of the row fits in [-1, 127], else little-endian
signed int16; a row's width is ``len(blob) // n_samples``), and ``gq`` one
uint8 per slot. Slot ``k`` of a batch is
``ingestion_progress.sample_ids[k]`` (ascending ``sample_id``). Hom-ref and
no-call are already resolved for every sample of the batch at every site the
batch records, so the matrix never needs ``sample_ref_blocks``.

``read_genotype_matrix`` assembles the matrix across batches for one genomic
window: columns are the batches' slots concatenated in ascending ``batch_id``
order, each site's allele list is the union of the batches' local lists (by
allele string), and a batch with no row at a site leaves its columns
``UNRESOLVED`` (it saw no variant record there, so the cell is hom-ref or
missing and cannot be told apart without its ref blocks). The matrix's
``gt_1`` / ``gt_2`` are int16 indices into the union list: a site's union
across many batches can exceed the 128 alleles int8 indexes.

A removed sample (``samples.status = 'removed'``, set by ``mosaic remove``)
keeps its slot and bytes in ``batch_genotypes`` (scrubbing them would rewrite
every row of each batch it was in); ``read_genotype_matrix`` drops its column
unless ``include_removed=True``. A site's allele list may still hold an
allele only a removed sample carried (the batch's row lists it).

This is the initial consumer API; it will change.
"""

from array import array
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa  # type: ignore[import-untyped]
from pyiceberg.expressions import (
    And,
    EqualTo,
    GreaterThanOrEqual,
    In,
    LessThanOrEqual,
)

from .config import ICEBERG_NAMESPACE

# Cell codes shared with ingest.
GT_NO_CALL = -1  # stored as byte 0xFF
GQ_UNKNOWN = 255  # no FORMAT/GQ and no PL
GQ_CAP = 99
UNRESOLVED = -2  # reader only: the column's batch has no row at this site
STAR_ALLELE = "*"  # spanning deletion / <NON_REF>; always last when present

BATCH_GENOTYPES_TABLE = "batch_genotypes"


# A row's allele list may hold at most this many alleles: its codes, and the
# reader's union indices, must fit a signed 16-bit slot.
MAX_SITE_ALLELES = 32767
GT_NARROW_MAX = 127  # the largest code a one-byte slot holds
GT_WIDE_DTYPE = np.dtype("<i2")


def encode_gt_blob(codes: Sequence[int], wide: bool = False) -> bytes:
    """Pack per-slot genotype codes into one signed byte each (-1..127), or
    with ``wide`` into two bytes each, little-endian signed 16-bit
    (-1..32767). A code out of range raises OverflowError."""
    if not wide:
        return array("b", codes).tobytes()
    return np.asarray(codes, dtype=GT_WIDE_DTYPE).tobytes()


def gt_blob_width(blob_len: int, n_samples: int) -> int:
    """Bytes per slot of a ``gt_1`` / ``gt_2`` blob over ``n_samples``
    slots: 1 or 2; any other length raises ValueError."""
    if n_samples > 0 and blob_len in (n_samples, 2 * n_samples):
        return blob_len // n_samples
    raise ValueError(
        f"genotype blob of {blob_len} bytes for {n_samples} slots "
        f"(expected 1 or 2 bytes per slot)"
    )


def decode_gt_blob(blob: bytes, n_samples: int) -> np.ndarray:
    """Inverse of ``encode_gt_blob`` at either width (inferred from
    ``len(blob) // n_samples``): a zero-copy int8 or int16 view."""
    if gt_blob_width(len(blob), n_samples) == 1:
        return np.frombuffer(blob, dtype=np.int8)
    return np.frombuffer(blob, dtype=GT_WIDE_DTYPE)


def decode_gq_blob(blob: bytes) -> np.ndarray:
    """A zero-copy uint8 view of a ``gq`` blob (255 = unknown)."""
    return np.frombuffer(blob, dtype=np.uint8)


@dataclass
class GenotypeMatrix:
    chromosome_index: int
    positions: np.ndarray  # int32, (n_sites,), ascending
    alleles: List[List[str]]  # per site: union, ref first, '*' last
    sample_ids: np.ndarray  # int32, (n_samples,), batch slot order
    batch_ids: np.ndarray  # int32, (n_samples,), each column's batch
    gt_1: np.ndarray  # int16, (n_sites, n_samples); indices into alleles[i]
    gt_2: np.ndarray  # int16
    gq: np.ndarray  # uint8


def _blob_offsets(col: Any) -> Tuple[Any, np.ndarray]:
    """(the combined Arrow binary array, its ``len + 1`` value offsets)."""
    if isinstance(col, pa.ChunkedArray):
        col = col.combine_chunks()
    n = len(col)
    if n == 0:
        return col, np.zeros(1, dtype=np.int64)
    off_dtype = np.int64 if pa.types.is_large_binary(col.type) else np.int32
    offsets = np.frombuffer(col.buffers()[1], dtype=off_dtype)[
        col.offset : col.offset + n + 1
    ]
    return col, offsets


def _blob_matrix(col: Any, width: int, dtype: Any) -> np.ndarray:
    """View an Arrow binary column whose values all hold ``width`` values of
    ``dtype`` as a ``(len(col), width)`` matrix without copying the data
    buffer."""
    dtype = np.dtype(dtype)
    col, offsets = _blob_offsets(col)
    n = len(col)
    if n == 0:
        return np.empty((0, width), dtype=dtype)
    lo, hi = int(offsets[0]), int(offsets[-1])
    if hi - lo != n * width * dtype.itemsize:
        raise ValueError(
            f"genotype blobs are not all {width * dtype.itemsize} bytes "
            f"({hi - lo} bytes for {n} rows)"
        )
    data = np.frombuffer(col.buffers()[2], dtype=np.uint8)[lo:hi]
    return data.view(dtype).reshape(n, width)


def _gt_matrix(col: Any, width: int) -> np.ndarray:
    """Decode a ``gt_1`` / ``gt_2`` column over ``width`` slots into an
    int16 ``(len(col), width)`` matrix, each row by its own blob width (1 or
    2 bytes per slot: the rows of one batch differ by site)."""
    col, offsets = _blob_offsets(col)
    lengths = np.diff(offsets.astype(np.int64))
    wide = lengths == 2 * width
    bad = np.flatnonzero(~wide & (lengths != width))
    if len(bad):
        gt_blob_width(int(lengths[bad[0]]), width)  # raises
    if not wide.any():
        return _blob_matrix(col, width, np.int8).astype(np.int16)
    if wide.all():
        return _blob_matrix(col, width, GT_WIDE_DTYPE).astype(np.int16)
    out = np.empty((len(col), width), dtype=np.int16)
    out[~wide] = _blob_matrix(col.filter(pa.array(~wide)), width, np.int8)
    out[wide] = _blob_matrix(col.filter(pa.array(wide)), width, GT_WIDE_DTYPE)
    return out


def batch_slot_orders(
    conn: Any,
    chromosome_index: int,
    batch_ids: Optional[Sequence[int]] = None,
) -> List[Tuple[int, List[int]]]:
    """Return ``[(batch_id, sample_ids), ...]`` in ascending batch order.

    Without ``batch_ids``: every complete ingest batch whose ``region``
    covers the chromosome (``NULL`` = whole genome, else ``chrN`` /
    ``chrN:a-b``); a `mosaic remove` batch (kind 'removal', no matrix rows,
    ``sample_ids`` NULL) is never one.
    The per-chromosome fan-out ingests the same samples once per chromosome,
    so selecting on region keeps a sample from appearing as several all-
    UNRESOLVED columns. Batches ingested before ``sample_ids`` existed have
    no matrix rows and are skipped (or rejected when named explicitly).
    """
    with conn.cursor() as cur:
        if batch_ids is not None:
            cur.execute(
                "SELECT batch_id, sample_ids, region FROM ingestion_progress "
                "WHERE batch_id = ANY(%s) ORDER BY batch_id",
                (list(batch_ids),),
            )
            rows = cur.fetchall()
            missing = set(batch_ids) - {r[0] for r in rows}
            no_slots = [r[0] for r in rows if r[1] is None]
            if missing or no_slots:
                raise ValueError(
                    f"batches without a slot order: "
                    f"{sorted(missing | set(no_slots))}"
                )
            return [(r[0], list(r[1])) for r in rows]
        cur.execute(
            "SELECT chromosome FROM contigs WHERE id = %s",
            (chromosome_index,),
        )
        row = cur.fetchone()
        chrom = row[0] if row else None
        cur.execute(
            "SELECT batch_id, sample_ids, region FROM ingestion_progress "
            "WHERE status = 'complete' AND kind = 'ingest' "
            "AND sample_ids IS NOT NULL ORDER BY batch_id"
        )
        out = []
        for bid, sids, region in cur.fetchall():
            if region is None or region.split(":", 1)[0] == chrom:
                out.append((bid, list(sids)))
        return out


def read_genotype_matrix(
    catalog: Any,
    conn: Any,
    chromosome_index: int,
    start: int,
    end: int,
    *,
    batch_ids: Optional[Sequence[int]] = None,
    include_removed: bool = False,
) -> GenotypeMatrix:
    """Assemble the genotype matrix for ``[start, end]`` (1-based,
    inclusive) on one chromosome, across batches.

    Sites are the union of the batches' rows. Each row's local allele
    indices are remapped into the site's union allele list by string (a
    small per-row lookup applied with ``np.take_along_axis``); columns of a
    batch with no row at a site stay ``UNRESOLVED`` with ``gq`` 255. The
    columns of removed samples (``samples.status``) are dropped unless
    ``include_removed``.
    """
    m = _read_genotype_matrix(
        catalog, conn, chromosome_index, start, end, batch_ids
    )
    if include_removed:
        return m
    with conn.cursor() as cur:
        cur.execute("SELECT id FROM samples WHERE status = 'removed'")
        removed = [int(r[0]) for r in cur.fetchall()]
    if not removed:
        return m
    keep = ~np.isin(m.sample_ids, removed)
    return GenotypeMatrix(
        chromosome_index=m.chromosome_index,
        positions=m.positions,
        alleles=m.alleles,
        sample_ids=m.sample_ids[keep],
        batch_ids=m.batch_ids[keep],
        gt_1=m.gt_1[:, keep],
        gt_2=m.gt_2[:, keep],
        gq=m.gq[:, keep],
    )


def _read_genotype_matrix(
    catalog: Any,
    conn: Any,
    chromosome_index: int,
    start: int,
    end: int,
    batch_ids: Optional[Sequence[int]],
) -> GenotypeMatrix:
    """read_genotype_matrix with every column (removed samples' too)."""
    slots = batch_slot_orders(conn, chromosome_index, batch_ids)
    widths = {bid: len(sids) for bid, sids in slots}
    offset: Dict[int, int] = {}
    col = 0
    for bid, sids in slots:
        offset[bid] = col
        col += len(sids)
    n_samples = col
    sample_ids = np.array(
        [s for _, sids in slots for s in sids], dtype=np.int32
    )
    col_batch = np.array(
        [bid for bid, sids in slots for _ in sids], dtype=np.int32
    )

    # PyIceberg's expression stubs reject positional arguments (mypy);
    # the calls are correct at runtime.
    row_filter: Any = And(
        EqualTo("chromosome_index", int(chromosome_index)),  # type: ignore
        GreaterThanOrEqual("position", int(start)),  # type: ignore
        LessThanOrEqual("position", int(end)),  # type: ignore
    )
    if slots:
        row_filter = And(
            row_filter, In("batch_id", list(widths))  # type: ignore
        )
    tbl = catalog.load_table(f"{ICEBERG_NAMESPACE}.{BATCH_GENOTYPES_TABLE}")
    arrow = tbl.scan(row_filter=row_filter).to_arrow() if slots else None
    if arrow is None or arrow.num_rows == 0:
        return GenotypeMatrix(
            chromosome_index=chromosome_index,
            positions=np.empty(0, dtype=np.int32),
            alleles=[],
            sample_ids=sample_ids,
            batch_ids=col_batch,
            gt_1=np.empty((0, n_samples), dtype=np.int16),
            gt_2=np.empty((0, n_samples), dtype=np.int16),
            gq=np.empty((0, n_samples), dtype=np.uint8),
        )
    arrow = arrow.sort_by(
        [("position", "ascending"), ("batch_id", "ascending")]
    )

    row_pos = arrow.column("position").to_numpy()
    row_batch = arrow.column("batch_id").to_numpy()
    row_alleles = arrow.column("alleles").to_pylist()
    row_n = arrow.column("n_samples").to_numpy()
    positions = np.unique(row_pos).astype(np.int32)
    row_site = np.searchsorted(positions, row_pos)

    # Union allele list per site: ref first, alts in first-seen order
    # (ascending batch), '*' last if any batch uses it.
    site_alleles: List[List[str]] = [[] for _ in range(len(positions))]
    site_star = [False] * len(positions)
    for i, als in zip(row_site, row_alleles):
        cur = site_alleles[i]
        if not cur:
            cur.append(als[0])
        for a in als[1:]:
            if a == STAR_ALLELE:
                site_star[i] = True
            elif a not in cur:
                cur.append(a)
    site_index: List[Dict[str, int]] = []
    for i, cur in enumerate(site_alleles):
        if site_star[i]:
            cur.append(STAR_ALLELE)
        if len(cur) > MAX_SITE_ALLELES:
            raise ValueError(
                f"site {chromosome_index}:{int(positions[i])}: the batches' "
                f"allele union holds {len(cur)} alleles, more than the "
                f"{MAX_SITE_ALLELES} an int16 genotype index can address"
            )
        site_index.append({a: j for j, a in enumerate(cur)})

    n_sites = len(positions)
    gt_1 = np.full((n_sites, n_samples), UNRESOLVED, dtype=np.int16)
    gt_2 = np.full((n_sites, n_samples), UNRESOLVED, dtype=np.int16)
    gq = np.full((n_sites, n_samples), GQ_UNKNOWN, dtype=np.uint8)

    for bid, width in widths.items():
        sel = np.flatnonzero(row_batch == bid)
        if len(sel) == 0:
            continue
        if np.any(row_n[sel] != width):
            raise ValueError(
                f"batch {bid}: n_samples in batch_genotypes differs from "
                f"ingestion_progress.sample_ids ({width})"
            )
        sub = arrow.take(pa.array(sel))
        sites = row_site[sel]
        # Per-row lookup local index -> union index.
        max_local = max(len(row_alleles[r]) for r in sel)
        lut = np.zeros((len(sel), max_local), dtype=np.int32)
        for j, r in enumerate(sel):
            idx = site_index[row_site[r]]
            for li, a in enumerate(row_alleles[r]):
                lut[j, li] = idx[a]
        lo = offset[bid]
        for name, out in (("gt_1", gt_1), ("gt_2", gt_2)):
            local = _gt_matrix(sub.column(name), width)
            mapped = np.take_along_axis(
                lut, np.maximum(local, 0).astype(np.intp), axis=1
            )
            out[sites, lo : lo + width] = np.where(local < 0, local, mapped)
        gq[sites, lo : lo + width] = _blob_matrix(
            sub.column("gq"), width, np.uint8
        )

    return GenotypeMatrix(
        chromosome_index=chromosome_index,
        positions=positions,
        alleles=site_alleles,
        sample_ids=sample_ids,
        batch_ids=col_batch,
        gt_1=gt_1,
        gt_2=gt_2,
        gq=gq,
    )


def dosage(m: GenotypeMatrix, site_index: int, allele: str) -> np.ndarray:
    """Copies of ``allele`` per sample at one site (int8), from the int16
    ``gt_1`` / ``gt_2`` of ``m``.

    ``-1`` where the first genotype slot is a no-call or ``UNRESOLVED``. A
    haploid or half call ``(a, -1)`` counts its one called allele.
    """
    alleles = m.alleles[site_index]
    a = alleles.index(allele) if allele in alleles else -100
    g1 = m.gt_1[site_index]
    g2 = m.gt_2[site_index]
    d = (g1 == a).astype(np.int8) + (g2 == a).astype(np.int8)
    d[g1 < 0] = -1
    return d
