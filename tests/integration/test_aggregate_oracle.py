# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Reference oracle for `mosaic aggregate`.

The vectorized (Arrow) reduce and finalize must produce the same PostgreSQL
aggregates as the original per-row algorithm. After each aggregate run the test
recomputes all three aggregates from the ``batch_*`` partials in pure Python
(``jc_testlib.oracle_from_partials``: pyiceberg scan -> dict sums, the old
algorithm, no ``mosaic.aggregate`` code) and compares
(``jc_testlib.assert_matches_oracle``): integer columns exact; qual / mqsq_1
within relative 1e-9 (PG REAL against the oracle rounded to float32, and the
float64 agg_site_info mirror against the oracle directly).

Runs over the tier shapes and the knobs that change the code path: a two-tier
tree (--fan-in 2), a single pass (--fan-in 100), a position-sharded ref_blocks
reduce, and tiny reduce-buffer / finalize-chunk bounds that force many buffer
flushes and many COPY chunks.
"""

import pytest

import jc_testlib as L

pytestmark = pytest.mark.integration

REGIONS = [
    "chr21:10000000-10500000",
    "chr21:20000000-20500000",
    "chr21:25000000-25500000",
]


@pytest.fixture(scope="module")
def staged_batches(docker_stack, tmp_path_factory):
    work = tmp_path_factory.mktemp("oracle_stage")
    L.stage_batches(REGIONS, str(work))
    return REGIONS


@pytest.mark.parametrize(
    "fan_in,ref_shard_size,extra_env",
    [
        pytest.param(2, None, None, id="two_tiers"),
        pytest.param(100, None, None, id="single_pass"),
        pytest.param(100, 5_000_000, None, id="ref_position_sharded"),
        pytest.param(
            2,
            None,
            {
                "MOSAIC_REDUCE_BUFFER_ROWS": "1000",
                "MOSAIC_FINALIZE_CHUNK_ROWS": "50",
            },
            id="tiny_buffers",
        ),
    ],
)
def test_aggregate_matches_python_oracle(
    staged_batches, fan_in, ref_shard_size, extra_env
):
    conn = L.get_pg_conn()
    try:
        L.clear_pg_aggregates(conn)
        L.run_aggregate(
            fan_in=fan_in,
            threads=2,
            rebuild=True,
            ref_shard_size=ref_shard_size,
            extra_env=extra_env,
        )
        counts = L.assert_matches_oracle(conn, L.get_catalog())
        # Guard against a vacuous pass: the fixture has real data in all three
        # aggregates, and more rows than the tiny buffer / chunk bounds.
        assert counts["site_info"] > 1000
        assert counts["ref_blocks"] > 1000
        assert counts["alleles"] > 1000
    finally:
        conn.close()
