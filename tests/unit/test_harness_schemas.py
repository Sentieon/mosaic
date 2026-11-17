# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The harness's Arrow schemas (``tests/sample_slices.py``, re-homed from
``mosaic/ingest.py`` in K6) against the Iceberg tables' schemas of
``mosaic.init_db``: the same field names in order, the same types (Iceberg's
``large_*`` read as the plain Arrow type, list element names ignored) and
the same nullability, so the test-side copies cannot drift from the
lake."""

import pyarrow as pa  # type: ignore[import-untyped]
import pytest

import sample_slices as S
from mosaic.init_db import iceberg_table_schemas

pytestmark = pytest.mark.unit


def _norm(t):
    if pa.types.is_large_list(t) or pa.types.is_list(t):
        return ("list", _norm(t.value_type))
    if pa.types.is_large_string(t) or pa.types.is_string(t):
        return "string"
    if pa.types.is_large_binary(t) or pa.types.is_binary(t):
        return "binary"
    return str(t)


def _fields(schema):
    return [(f.name, _norm(f.type), f.nullable) for f in schema]


@pytest.mark.parametrize(
    "table,schema",
    [
        ("sample_site_info", S.SAMPLE_SITE_INFO_ARROW_SCHEMA),
        ("sample_ref_blocks", S.SAMPLE_REF_BLOCKS_ARROW_SCHEMA),
        ("batch_genotypes", S.BATCH_GENOTYPES_ARROW_SCHEMA),
    ],
)
def test_harness_schema_equals_the_iceberg_schema(table, schema):
    iceberg = iceberg_table_schemas()[table].as_arrow()
    assert _fields(schema) == _fields(iceberg)
