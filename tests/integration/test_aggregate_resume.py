# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Crash + resume correctness for the tier-tree aggregation.

aggregate.py's reduce units only upload their staging output (deterministic S3
keys); after each barrier (a tier) the main process registers the barrier's
files in Iceberg FIRST, then commits the barrier's tier_completion checkpoints
SECOND, in one transaction. MOSAIC_TEST_FAIL_UNIT names a unit and raises at
its barrier:
  * pre_register (uploaded, nothing registered) and pre_ckpt (registered, not
    checkpointed) must resume with no double-count -- the unit is recomputed,
    its file overwritten at the same key and registered once;
  * post_ckpt (checkpointed) must resume by SKIPPING that unit.

Oracle: a crash+resume must finalize PG bit-exact to a clean single-pass run,
and leave the same staging and agg_* mirror data files, by URI and S3 ETag
(``reset_aggregate_runs`` before each run restarts the agg_run_id sequence, so
the clean and the resumed run share the id their keys fold in). The PG
aggregates are cleared before the crash so the resumed run must genuinely
reconstruct them (aggregate.py finalizes to PG only at the very end, so a crash
otherwise leaves the prior run's values in place — a false-pass trap).

With 3 batches and --fan-in 2 the tier tree is [[b1,b2],[b3]] -> [[g0,g1]], so the
unit "site_info:*:1:0" (tier 0->1, group 0) genuinely exists and the checkpoint
skip-path is exercised.

Each (metric, chromosome) finalize unit is checkpointed too: its partition swap
and a sentinel tier_completion row commit in one transaction
(MOSAIC_TEST_FAIL_FINALIZE crashes just before or just after that commit), so a
resume skips a committed swap and rebuilds one that rolled back. A skipped
finalize unit lists the mirror files its earlier attempt uploaded, so the
mirror the barrier commits is the clean run's.
"""

import os

import pytest

import jc_testlib as L

pytestmark = [pytest.mark.integration, pytest.mark.restart]

REGIONS = [
    "chr21:10000000-10500000",
    "chr21:20000000-20500000",
    "chr21:25000000-25500000",
]
FAN_IN = 2
# tier 0->1, group 0 (reduces the first two batches); chromosome wildcarded.
TARGET_METRIC = "site_info"
TARGET_TIER = 1
TARGET_GROUP = 0


@pytest.fixture(scope="module")
def staged_batches(docker_stack, tmp_path_factory):
    """Ingest 3 non-overlapping single-shard batches once for the whole module.

    Init TWO contigs (chr20 + chr21) but ingest only chr21 data, so every
    aggregate here runs the finalize over >1 chromosome -- exercising the
    per-chromosome finalize units (and an empty chromosome, whose partition is
    attached empty) that a chr21-only fixture would never reach. chr20 stays
    empty; the bit-exact oracles still hold (baseline and resumed runs share
    this init)."""
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr20,chr21")
    work = tmp_path_factory.mktemp("agg_stage")
    for i, region in enumerate(REGIONS):
        gvcfs = []
        for j, src in enumerate(L.GVCFS):
            dst = os.path.join(str(work), f"b{i}_s{j}.g.vcf.gz")
            L.subset_gvcf(src, region, dst)
            gvcfs.append(dst)
        L.run_ingest(gvcfs, region=region, threads=2)
    conn = L.get_pg_conn()
    try:
        assert len(L.batch_statuses(conn)) == 3
        assert all(s == "complete" for _, s in L.batch_statuses(conn))
    finally:
        conn.close()
    return REGIONS


def _baseline(conn):
    """Clean run at FAN_IN from a reset aggregate state; capture the
    finalized PG aggregates and the aggregate's data files (URI, ETag)."""
    L.reset_aggregate_runs(conn, L.get_catalog())
    L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
    return L.snapshot_pg(conn), L.aggregate_file_etags(L.get_catalog())


def _assert_same_files(want):
    assert L.aggregate_file_etags(L.get_catalog()) == want


def _registered_rows(catalog, agg_run_id, tier, group_id):
    """Rows of run ``agg_run_id``'s (tier, group) in tier_site_info."""
    from pyiceberg.expressions import And, EqualTo

    table = catalog.load_table(f"{L.TEST_NAMESPACE}.tier_site_info")
    rows = table.scan(
        row_filter=And(
            EqualTo("agg_run_id", agg_run_id),
            And(EqualTo("tier", tier), EqualTo("group_id", group_id)),
        )
    ).to_arrow()
    return rows.num_rows


@pytest.mark.parametrize("phase", ["pre_register", "pre_ckpt", "post_ckpt"])
def test_aggregate_resume_tier_barrier(staged_batches, phase):
    """Crash at the tier-0 barrier of site_info around unit (tier 1, group
    0): before its registration, between registration and checkpoint, or
    after the checkpoint; the resume is bit-exact with a clean run and
    leaves the same staging and mirror files."""
    conn = L.get_pg_conn()
    try:
        base, files = _baseline(conn)
        L.reset_aggregate_runs(conn, L.get_catalog())

        proc = L.run_aggregate(
            fan_in=FAN_IN,
            threads=2,
            rebuild=True,
            extra_env={
                "MOSAIC_TEST_FAIL_UNIT": f"{TARGET_METRIC}:*:{TARGET_TIER}:"
                f"{TARGET_GROUP}:{phase}"
            },
            check=False,
        )
        assert proc.returncode != 0, f"aggregate should have crashed ({phase})"
        assert f"@{phase}" in proc.stderr
        assert any(s == "failed" for _, s in L.agg_statuses(conn))
        crashed = L.latest_agg_run_id(conn)
        # Checkpointed only after post_ckpt; registered from pre_ckpt on.
        assert L.tier_completion_has(
            conn, TARGET_METRIC, TARGET_TIER, TARGET_GROUP
        ) == (phase == "post_ckpt")
        n_rows = _registered_rows(
            L.get_catalog(), crashed, TARGET_TIER, TARGET_GROUP
        )
        assert (n_rows > 0) == (phase != "pre_register"), n_rows

        # Resume the failed run and finalize.
        proc = L.run_aggregate(fan_in=FAN_IN, threads=2)
        assert f"agg_run_id={crashed} resume" in proc.stderr
        assert L.latest_agg_run_id(conn) == crashed
        L.assert_pg_bitexact(L.snapshot_pg(conn), base)
        _assert_same_files(files)
    finally:
        conn.close()


def test_aggregate_worker_death_fails_fast(staged_batches):
    """An abruptly-killed reduce worker (stand-in for an OOM SIGKILL / segfault)
    must fail the run PROMPTLY, not hang.

    The old multiprocessing.Pool could not detect a worker that died via signal
    and hung forever in imap_unordered; the spawn ProcessPoolExecutor surfaces it
    as BrokenProcessPool. MOSAIC_TEST_KILL_WORKER makes the matching tier-0 unit
    os._exit(), and we assert the process exits non-zero quickly (a generous 120s
    cap, far below a true hang) and the run is marked 'failed' (resumable).
    """
    import time

    conn = L.get_pg_conn()
    try:
        L.clear_pg_aggregates(conn)
        t0 = time.time()
        proc = L.run_aggregate(
            fan_in=FAN_IN,
            threads=2,
            rebuild=True,
            extra_env={"MOSAIC_TEST_KILL_WORKER": "allele_counts:*:1:0"},
            check=False,
            timeout=120,
        )
        elapsed = time.time() - t0
        assert (
            proc.returncode != 0
        ), "aggregate should fail when a worker dies, not succeed"
        assert elapsed < 120, "aggregate hung instead of failing fast"
        assert any(
            s == "failed" for _, s in L.agg_statuses(conn)
        ), "the killed run should be marked 'failed' (resumable)"
    finally:
        conn.close()


def test_ref_blocks_position_sharding_equivalence(staged_batches):
    """The ref_blocks window units must produce PG aggregates independent of
    the window size.

    --ref-shard-size 0 runs one window unit per chromosome; a 5 Mbp window
    spreads this fixture's regions (chr21:10-10.5M, 20-20.5M, 25-25.5M) across
    distinct non-empty windows, and their union must equal the single-window
    result.
    """
    import re

    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=100, threads=2, rebuild=True, ref_shard_size=0)
        unsharded = L.snapshot_pg(conn)

        proc = L.run_aggregate(
            fan_in=100, threads=2, rebuild=True, ref_shard_size=5_000_000
        )
        m = re.search(r"ref_windows=(\d+)", proc.stderr)
        assert (
            m and int(m.group(1)) > 1
        ), "position-sharding did not engage (expected >1 window)"
        L.assert_pg_bitexact(L.snapshot_pg(conn), unsharded)
    finally:
        conn.close()


def test_finalize_chunking_equivalence(staged_batches):
    """The streamed, chunked finalize must produce PG aggregates identical to a
    single-chunk finalize.

    _finalize_chrom does not materialize a chromosome's whole result: it
    streams the root tier and flushes in <= MOSAIC_FINALIZE_CHUNK_ROWS slices,
    uploading each chunk as one agg_* mirror file (registered by the
    finalize barrier) and COPYing it into the new generation table. Local
    fixtures have far fewer than the 1M-row default, so normal runs never
    exercise the multi-chunk path.
    Force a tiny chunk size and assert the result is bit-exact vs the default.
    """
    CHUNK = 50
    conn = L.get_pg_conn()
    try:
        L.run_aggregate(fan_in=FAN_IN, threads=2, rebuild=True)
        single_chunk = L.snapshot_pg(conn)
        # Guard: the fixture must exceed the tiny chunk so >1 chunk really runs
        # (else this would silently degrade to the single-chunk path).
        assert (
            len(single_chunk["site_info_int"]) > CHUNK
        ), "fixture too small to exercise multi-chunk finalize"

        L.run_aggregate(
            fan_in=FAN_IN,
            threads=2,
            rebuild=True,
            extra_env={"MOSAIC_FINALIZE_CHUNK_ROWS": str(CHUNK)},
        )
        L.assert_pg_bitexact(L.snapshot_pg(conn), single_chunk)
        # The mirror holds every chunk: one file per chunk, the same rows.
        catalog = L.get_catalog()
        mirror = catalog.load_table(f"{L.TEST_NAMESPACE}.agg_site_info")
        n_files = len(
            L.iceberg_data_files(catalog, ["agg_site_info"])["agg_site_info"]
        )
        n_rows = mirror.scan().to_arrow().num_rows
        assert n_rows >= len(single_chunk["site_info_int"])
        assert n_files >= n_rows // CHUNK, (n_files, n_rows)
    finally:
        conn.close()


def test_finalize_checkpoint_post_commit_skipped_on_resume(staged_batches):
    """A crash just AFTER the site_info swap commits leaves its checkpoint
    (and its PG rows) durable; the resume skips that finalize, rebuilds the
    rest, and ends bit-exact with a clean run. ``*`` crashes every site_info
    unit; both chromosomes' units run at once (2 contigs, --threads 2), so
    both commit before the run fails."""
    conn = L.get_pg_conn()
    try:
        base, files = _baseline(conn)
        L.reset_aggregate_runs(conn, L.get_catalog())

        proc = L.run_aggregate(
            fan_in=FAN_IN,
            threads=2,
            rebuild=True,
            extra_env={"MOSAIC_TEST_FAIL_FINALIZE": "site_info:*:post_commit"},
            check=False,
        )
        assert proc.returncode != 0, "aggregate should have crashed"
        assert "MOSAIC_TEST_FAIL_FINALIZE" in proc.stderr
        assert any(s == "failed" for _, s in L.agg_statuses(conn))
        # site_info's finalize committed together with its checkpoint row...
        assert L.finalize_marker_exists(conn, "site_info")
        assert L.pg_counts(conn)["site_info"] == len(base["site_info_int"])
        # ...and nothing after it ran.
        assert L.finalize_markers(conn, "ref_blocks") == set()
        assert L.finalize_markers(conn, "allele_counts") == set()
        assert L.pg_counts(conn)["ref_blocks"] == 0

        # The swaps committed, the mirror commit (after the barrier) not.
        assert L.iceberg_data_files(L.get_catalog(), ["agg_site_info"]) == {
            "agg_site_info": []
        }

        proc = L.run_aggregate(fan_in=FAN_IN, threads=2)
        assert "site_info finalize: skipped" in proc.stderr
        assert "ref_blocks: " in proc.stderr  # the later finalizes ran
        for metric in ("site_info", "ref_blocks", "allele_counts"):
            assert L.finalize_marker_exists(conn, metric)
        L.assert_pg_bitexact(L.snapshot_pg(conn), base)
        # The skipped units listed their uploaded mirror files.
        _assert_same_files(files)
    finally:
        conn.close()


def test_finalize_checkpoint_pre_commit_reruns(staged_batches):
    """A crash just BEFORE the site_info swap commits rolls back the swap AND
    its checkpoint row together; the resume re-runs the finalize."""
    conn = L.get_pg_conn()
    try:
        base, files = _baseline(conn)
        L.reset_aggregate_runs(conn, L.get_catalog())

        proc = L.run_aggregate(
            fan_in=FAN_IN,
            threads=2,
            rebuild=True,
            extra_env={"MOSAIC_TEST_FAIL_FINALIZE": "site_info:*:pre_commit"},
            check=False,
        )
        assert proc.returncode != 0, "aggregate should have crashed"
        assert "MOSAIC_TEST_FAIL_FINALIZE" in proc.stderr
        assert L.finalize_markers(conn, "site_info") == set()
        assert L.pg_counts(conn)["site_info"] == 0  # rolled back

        proc = L.run_aggregate(fan_in=FAN_IN, threads=2)
        assert "site_info finalize: skipped" not in proc.stderr
        assert (
            f"site_info: {len(base['site_info_int'])} final rows"
            in proc.stderr
        )
        assert L.finalize_marker_exists(conn, "site_info")
        L.assert_pg_bitexact(L.snapshot_pg(conn), base)
        _assert_same_files(files)
    finally:
        conn.close()
