# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the whole-sample per-sample layout, the Iceberg side:
the per-sample tables'
partition spec and sort order, and the lake guard refusing an unpartitioned
per-sample table.

The tests of the local files and the combine this file held earlier (the
sample-slice container and its
refusals, the local writer, ``pack_sample_files``, the whole-sample output
writer and the combine's keys) tested the Python ingest worker and its
combine; they are ``cargo test``s of the Rust worker with the same inputs
and expected values (``rust/mosaic-ingest-worker/src/slices.rs``,
``pqwrite.rs``, ``localfiles.rs``, ``combine.rs``) and, for the keys,
``tests/unit/test_worker_bin_files.py``.
"""

import pytest
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.table.sorting import SortDirection
from pyiceberg.transforms import IdentityTransform

from mosaic.init_db import (
    SAMPLE_TABLE_SORT_COLUMNS,
    SchemaError,
    batch_genotypes_partition_spec,
    check_ingest_lake_schema,
    iceberg_table_schemas,
    chromosome_partition_spec,
    is_chromosome_partitioned,
    sample_ref_blocks_schema,
    sample_site_info_schema,
    sample_table_sort_order,
    unpartitioned_sample_tables,
)

pytestmark = pytest.mark.unit


# ── the Iceberg side ─────────────────────────────────────────────────────────


def test_per_sample_partition_spec_and_sort_order():
    for short, schema in (
        ("sample_ref_blocks", sample_ref_blocks_schema()),
        ("sample_site_info", sample_site_info_schema()),
    ):
        (field,) = chromosome_partition_spec(schema).fields
        assert (
            field.source_id == schema.find_field("chromosome_index").field_id
        )
        assert isinstance(field.transform, IdentityTransform)
        order = sample_table_sort_order(short, schema)
        assert [
            schema.find_column_name(f.source_id) for f in order.fields
        ] == list(SAMPLE_TABLE_SORT_COLUMNS[short])
        assert SAMPLE_TABLE_SORT_COLUMNS[short][1] == "sample_id"
        assert all(f.direction == SortDirection.ASC for f in order.fields)
    # batch_genotypes keeps its spec.
    (bgt,) = batch_genotypes_partition_spec().fields
    assert (bgt.source_id, bgt.field_id, bgt.name) == (
        2,
        1000,
        "chromosome_index",
    )


class _Table:
    def __init__(self, schema, spec):
        self._schema, self._spec = schema, spec

    def schema(self):
        return self._schema

    def spec(self):
        return self._spec


class _Catalog:
    def __init__(self, flat=()):
        self.flat = set(flat)

    def load_table(self, name):
        short = name.split(".", 1)[1]
        schema = iceberg_table_schemas()[short]
        spec = (
            UNPARTITIONED_PARTITION_SPEC
            if short in self.flat
            else chromosome_partition_spec(schema)
        )
        return _Table(schema, spec)


def test_lake_guard_refuses_an_unpartitioned_per_sample_table():
    assert unpartitioned_sample_tables(_Catalog()) == []
    check_ingest_lake_schema(_Catalog())
    cat = _Catalog(flat=["sample_ref_blocks"])
    assert not is_chromosome_partitioned(
        cat.load_table("ns.sample_ref_blocks")
    )
    assert unpartitioned_sample_tables(cat) == ["sample_ref_blocks"]
    with pytest.raises(SchemaError) as e:
        check_ingest_lake_schema(cat)
    msg = str(e.value)
    assert "sample_ref_blocks: not partitioned by chromosome_index" in msg
    assert "no migration path" in msg and "mosaic clean" in msg
