# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Tier-tree associativity: the aggregate result is independent of --fan-in.

Each metric is a uniform SUM-by-key reduction, so a deep tree (--fan-in 2, multiple
tiers) and a single pass (--fan-in >= #batches) must produce identical integer
aggregates. Only qual/mqsq_1 (REAL, summed in float64) may differ at the last ULP
across tier orders, so this asserts integer-exact equality and tolerates the floats.

This is the integration-level proof of the property that underwrites restart
correctness (a resumed partial tree == a clean tree); the unit-level proof is in
tests/unit/test_reducers.py.
"""

import pytest

import jc_testlib as L

pytestmark = [pytest.mark.integration, pytest.mark.restart]

REGIONS = [
    "chr21:10000000-10500000",
    "chr21:20000000-20500000",
    "chr21:25000000-25500000",
]


@pytest.fixture(scope="module")
def staged_batches(docker_stack, tmp_path_factory):
    work = tmp_path_factory.mktemp("fanin_stage")
    L.stage_batches(REGIONS, str(work))
    return REGIONS


def test_fanin_2_equals_single_pass(staged_batches):
    conn = L.get_pg_conn()
    try:
        # Deep tree: fan_in=2 over 3 batches -> 2 tiers.
        L.run_aggregate(fan_in=2, threads=2, rebuild=True)
        deep = L.snapshot_pg(conn)

        # Single pass: fan_in >= #batches -> 1 tier. The finalize TRUNCATE+COPY
        # fully overwrites PG, so no manual clear is needed between runs.
        L.run_aggregate(fan_in=100, threads=2, rebuild=True)
        flat = L.snapshot_pg(conn)

        # Integer columns must be bit-exact; floats may differ at the last ULP.
        L.assert_pg_integer_equal(deep, flat)
    finally:
        conn.close()
