# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The sample tombstone (``samples.status``) and the read-side filter.

A sample whose ``samples`` row has ``status = 'removed'`` is tombstoned:
readers of the per-sample evidence (``joint_call.py``,
``scripts/gwas_benchmark.py``) list only active samples and drop a removed
sample's ``sample_site_info`` / ``sample_ref_blocks`` rows, and ``mosaic
ingest`` refuses to ingest it again. ``mosaic remove`` sets it in its first
transaction, before anything else: until the following ``mosaic aggregate``
folds the removal's negative partials in, the cross-sample aggregates (PG
``site_info`` / ``alleles`` / ``ref_blocks``, so a VCF's INFO fields) still
count the sample, and ``joint_call.py`` refuses that stale cohort unless
``--allow-stale``. The sample keeps its slot and bytes in
``batch_genotypes``; ``mosaic/genotypes.py``'s reader drops its column.

The filter (``tombstone_filter``) drops the removed samples' rows from an
Arrow table after the scan, and pushes ``NotIn("sample_id", removed)`` down
to the scan only for a small removed set (``NOT_IN_PUSHDOWN_MAX``): an
expression of a million ids is not a plan.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import pyarrow as pa  # type: ignore[import-untyped]
import pyarrow.compute as pc  # type: ignore[import-untyped]
from pyiceberg.expressions import And, BooleanExpression, In, NotIn

from .init_db import SAMPLE_STATUS_ACTIVE, SAMPLE_STATUS_REMOVED

# Largest removed set pushed down to an Iceberg scan as NotIn("sample_id",
# ...); a larger one is only masked after the scan.
NOT_IN_PUSHDOWN_MAX = 1000

ArrowMask = Callable[[Any], Any]


def samples_status_missing(conn: Any) -> bool:
    """True for a database whose samples table predates the tombstone (no
    status column; `mosaic init` and `mosaic ingest` add it)."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT NOT EXISTS (
                SELECT 1 FROM information_schema.columns
                WHERE table_schema = current_schema()
                  AND table_name = 'samples'
                  AND column_name = 'status')
        """)
        return bool(cur.fetchone()[0])


def missing_status_remedy() -> str:
    """The error a reader prints for a database without samples.status."""
    return (
        "the samples table has no status column (a database initialized "
        "before the sample tombstone). Re-run `mosaic init` on it: it adds "
        "the column (every sample 'active'), then stops at the populated "
        "contigs table, which is expected."
    )


def active_samples(conn: Any) -> Dict[int, str]:
    """{sample_id: sample_name} of the active (not tombstoned) samples."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id, sample_name FROM samples WHERE status = %s",
            (SAMPLE_STATUS_ACTIVE,),
        )
        return {int(sid): name for sid, name in cur.fetchall()}


def active_sample_names(conn: Any) -> List[str]:
    """The active samples' names in PostgreSQL's ``ORDER BY sample_name``
    order (a VCF header's sample columns)."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT sample_name FROM samples WHERE status = %s "
            "ORDER BY sample_name",
            (SAMPLE_STATUS_ACTIVE,),
        )
        return [name for (name,) in cur.fetchall()]


def removed_sample_ids(conn: Any) -> List[int]:
    """The tombstoned sample ids, ascending."""
    with conn.cursor() as cur:
        cur.execute(
            "SELECT id FROM samples WHERE status = %s ORDER BY id",
            (SAMPLE_STATUS_REMOVED,),
        )
        return [int(sid) for (sid,) in cur.fetchall()]


def removed_sample_names(conn: Any, names: Sequence[str]) -> List[str]:
    """Those of ``names`` whose samples row is tombstoned, sorted."""
    if not names:
        return []
    with conn.cursor() as cur:
        cur.execute(
            "SELECT sample_name FROM samples "
            "WHERE sample_name = ANY(%s) AND status = %s "
            "ORDER BY sample_name",
            (list(names), SAMPLE_STATUS_REMOVED),
        )
        return [name for (name,) in cur.fetchall()]


def tombstone_filter(
    removed_ids: Sequence[int],
) -> Tuple[Optional[BooleanExpression], Optional[ArrowMask]]:
    """The read-side filter for the removed samples: ``(row_filter,
    mask)``.

    * no removed sample: ``(None, None)``, nothing to do;
    * 1..NOT_IN_PUSHDOWN_MAX removed: ``NotIn("sample_id", removed)`` to AND
      into the scan's row filter, and the mask;
    * more: ``(None, mask)``, the mask alone.

    ``mask(table)`` returns the Arrow table (or record batch) without the
    rows whose ``sample_id`` is removed; it is applied whether or not the
    filter was pushed down (pyiceberg's pushdown prunes files and row
    groups, and is not relied on for exactness).
    """
    removed = sorted({int(s) for s in removed_ids})
    if not removed:
        return None, None
    value_set = pa.array(removed, type=pa.int64())

    def mask(table: Any) -> Any:
        if table.num_rows == 0:
            return table
        ids = pc.cast(table.column("sample_id"), pa.int64())
        keep = pc.invert(pc.is_in(ids, value_set=value_set))
        return table.filter(keep)

    if len(removed) <= NOT_IN_PUSHDOWN_MAX:
        return NotIn(term="sample_id", literals=removed), mask
    return None, mask


def sample_filter(
    sample_ids: Sequence[int],
) -> Tuple[Optional[BooleanExpression], ArrowMask]:
    """The positive selection of a sample set (`mosaic remove` reads the
    removed samples' own rows): ``(row_filter, mask)`` by the same rule as
    tombstone_filter, ``In("sample_id", ids)`` pushed down for at most
    NOT_IN_PUSHDOWN_MAX ids, and ``mask(table)`` (keep the set's rows)
    applied either way."""
    ids = sorted({int(s) for s in sample_ids})
    value_set = pa.array(ids, type=pa.int64())

    def mask(table: Any) -> Any:
        if table.num_rows == 0:
            return table
        keep = pc.is_in(
            pc.cast(table.column("sample_id"), pa.int64()), value_set=value_set
        )
        return table.filter(keep)

    if 0 < len(ids) <= NOT_IN_PUSHDOWN_MAX:
        return In(term="sample_id", literals=ids), mask
    return None, mask


def and_filter(
    row_filter: BooleanExpression, extra: Optional[BooleanExpression]
) -> BooleanExpression:
    """``row_filter AND extra`` (``row_filter`` when there is no extra)."""
    return row_filter if extra is None else And(row_filter, extra)
