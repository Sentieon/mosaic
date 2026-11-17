# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the removal substrate (the per-record contribution
columns, the tombstone) that survive the earlier Python ingest worker.

* The tombstone filter: no removed sample -> nothing; a small set -> NotIn
  pushdown + mask; a large set -> mask only.
* The ``samples.status`` DDL.
* The schemas (Iceberg and the harness's Arrow copies) and the lake guard's
  column check.

The ingest-side tests this file held earlier tested the Python ingest
worker: each
row's ``extract_site_info`` contribution (``rust/mosaic-ingest-worker/src/
sites.rs``: ``row_carries_extract_site_info``, ``site_dp_is_info_dp_...``,
``rows_sum_to_the_site_info``, ``sb_is_the_summed_value``), the ref-block
contribution and the deletion split (``ref_blocks.rs`` ``k6_*``), the local
writer (``localfiles.rs::local_sample_ref_blocks_has_sample_counts``; the
contribution columns on the worker binary,
``tests/unit/test_worker_bin_files.py``) and the cross-shard span patch
(``combine.rs::patcher_the_python_case``).
"""

import pyarrow as pa
import pytest
from pyiceberg.expressions import NotIn
from pyiceberg.schema import Schema
from pyiceberg.types import DoubleType, IntegerType, ListType, LongType

from mosaic.init_db import (
    DEL_REF_SPANS_COLUMN,
    SAMPLE_REF_BLOCKS_CONTRIBUTION_COLUMNS,
    SAMPLE_SITE_CONTRIBUTION_COLUMNS,
    SAMPLES_STATUS_COLUMN_DDL,
    SchemaError,
    check_ingest_lake_schema,
    iceberg_table_schemas,
    chromosome_partition_spec,
    missing_lake_columns,
    sample_ref_blocks_schema,
    sample_site_info_schema,
)
from mosaic.tombstones import (
    NOT_IN_PUSHDOWN_MAX,
    and_filter,
    tombstone_filter,
)
from sample_slices import (
    SAMPLE_REF_BLOCKS_ARROW_SCHEMA,
    SAMPLE_SITE_INFO_ARROW_SCHEMA,
)

pytestmark = pytest.mark.unit


# ── the tombstone filter ─────────────────────────────────────────────────────


def _ids_table(ids):
    return pa.table(
        {
            "sample_id": pa.array(ids, pa.int32()),
            "position": pa.array(range(len(ids)), pa.int32()),
        }
    )


def test_no_removed_sample_means_no_filter_and_no_mask():
    assert tombstone_filter([]) == (None, None)


def test_small_removed_set_pushes_not_in_down_and_masks():
    row_filter, mask = tombstone_filter([7, 3, 7])
    assert row_filter == NotIn("sample_id", [3, 7])
    table = _ids_table([1, 3, 5, 7, 3, 9])
    out = mask(table)
    assert out.column("sample_id").to_pylist() == [1, 5, 9]
    assert out.column("position").to_pylist() == [0, 2, 5]  # order kept
    batch = table.to_batches()[0]
    assert mask(batch).column("sample_id").to_pylist() == [1, 5, 9]
    assert mask(_ids_table([])).num_rows == 0


def test_pushdown_limit():
    at = list(range(NOT_IN_PUSHDOWN_MAX))
    assert isinstance(tombstone_filter(at)[0], NotIn)
    row_filter, mask = tombstone_filter(list(range(NOT_IN_PUSHDOWN_MAX + 1)))
    assert row_filter is None and mask is not None
    out = mask(_ids_table([5, NOT_IN_PUSHDOWN_MAX, NOT_IN_PUSHDOWN_MAX + 1]))
    assert out.column("sample_id").to_pylist() == [NOT_IN_PUSHDOWN_MAX + 1]


def test_and_filter():
    base = NotIn("position", [1])
    assert and_filter(base, None) is base
    assert "sample_id" in repr(and_filter(base, NotIn("sample_id", [2])))


def test_samples_status_ddl():
    assert "DEFAULT 'active'" in SAMPLES_STATUS_COLUMN_DDL
    assert "CHECK (status IN ('active', 'removed'))" in (
        SAMPLES_STATUS_COLUMN_DDL
    )


# ── schemas and the lake guard ───────────────────────────────────────────────


def test_iceberg_schemas_carry_the_contribution():
    ssi = sample_site_info_schema()
    types = {f.name: f.field_type for f in ssi.fields}
    assert isinstance(types["qual"], DoubleType)
    assert isinstance(types["site_dp"], LongType)
    assert isinstance(types["mqsq_1"], DoubleType)
    assert isinstance(types["mqsq_2"], LongType)
    assert isinstance(types["gtc"], ListType)
    assert isinstance(types["gtc"].element_type, DoubleType)
    assert isinstance(types[DEL_REF_SPANS_COLUMN].element_type, IntegerType)
    assert [f.name for f in ssi.fields] == SAMPLE_SITE_INFO_ARROW_SCHEMA.names
    srb = sample_ref_blocks_schema()
    assert [f.name for f in srb.fields] == SAMPLE_REF_BLOCKS_ARROW_SCHEMA.names
    # (Until K6 an import-time assert of mosaic/ingest.py.)
    assert (
        tuple(SAMPLE_SITE_INFO_ARROW_SCHEMA.names[-6:-1])
        == SAMPLE_SITE_CONTRIBUTION_COLUMNS
    )
    assert set(SAMPLE_SITE_CONTRIBUTION_COLUMNS) <= set(types)
    assert set(SAMPLE_REF_BLOCKS_CONTRIBUTION_COLUMNS) <= {
        f.name for f in srb.fields
    }


class _FakeTable:
    def __init__(self, schema):
        self._schema = schema

    def schema(self):
        return self._schema

    def spec(self):
        # The whole-sample layout (E2): partitioned by chromosome_index.
        return chromosome_partition_spec(self._schema)


class _FakeCatalog:
    """The current lake (init_db.iceberg_table_schemas), with the tables in
    ``schemas`` replaced."""

    def __init__(self, schemas):
        self.schemas = {**iceberg_table_schemas(), **schemas}

    def load_table(self, name):
        return _FakeTable(self.schemas[name.split(".", 1)[1]])


def _without(schema, names):
    return Schema(*[f for f in schema.fields if f.name not in names])


def test_lake_guard_accepts_the_current_lake():
    cat = _FakeCatalog(
        {
            "sample_site_info": sample_site_info_schema(),
            "sample_ref_blocks": sample_ref_blocks_schema(),
        }
    )
    assert missing_lake_columns(cat) == []
    check_ingest_lake_schema(cat)


def test_lake_guard_refuses_a_lake_without_the_contribution():
    old = set(SAMPLE_SITE_CONTRIBUTION_COLUMNS) | {DEL_REF_SPANS_COLUMN}
    cat = _FakeCatalog(
        {
            "sample_site_info": _without(sample_site_info_schema(), old),
            "sample_ref_blocks": _without(
                sample_ref_blocks_schema(), {"sample_counts"}
            ),
        }
    )
    missing = missing_lake_columns(cat)
    assert missing == [
        f"sample_site_info.{c}"
        for c in (
            *SAMPLE_SITE_CONTRIBUTION_COLUMNS,
            DEL_REF_SPANS_COLUMN,
        )
    ] + ["sample_ref_blocks.sample_counts"]
    with pytest.raises(SchemaError) as e:
        check_ingest_lake_schema(cat)
    msg = str(e.value)
    assert "sample_site_info.site_dp" in msg
    assert "no migration path" in msg and "mosaic init" in msg
