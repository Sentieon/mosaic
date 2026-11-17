# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Incremental `mosaic aggregate`: fold new batches onto the parent run.

Without --rebuild, a run whose completed-batch set extends the latest complete
run (its parent) reduces only the delta batches: per chromosome, the
site_info / allele_counts tier tree over the delta plus one fold unit (tier
T + 1, group 0) summing the delta root and the parent's root, and ref_blocks
window units that add the parent's dense root (by position range) before the
delta batches' partials. It must produce the same PostgreSQL content as a full
--rebuild. When a run completes it deletes its own lower tiers and every other
run's staging, and marks those runs staging_pruned.

Fixture: three batches of two DIFFERENT samples each over the SAME two chr21
windows (10.0-10.3 Mbp and 25.0-25.3 Mbp), init chr20 + chr21 (chr20 is
covered by no batch). Overlapping positions are what make the sparse-root trap
testable: a site only b3 reveals needs b1/b2's hom-ref coverage at that
position, which only the parent's DENSE root still holds. "Batches arriving"
is played by ``set_completed_batches`` (the others are held back), so one
staged fixture serves every lineage; ``clear_lineage`` (or
``reset_aggregate_runs``, which also restarts the agg_run_id sequence) starts
each test with no run.

Incremental runs use --ref-shard-size CHILD_REF while their parents use the
default 20 Mbp, so the window units read the parent's dense root across
windows that do not line up (by position range, never by window_id).
"""

import os
import re

import pytest

import jc_testlib as L

pytestmark = [pytest.mark.integration, pytest.mark.restart]

WINDOWS = "chr21:10000000-10300000,chr21:25000000-25300000"
SAMPLES = [
    L.GVCFS,
    [
        os.path.join(
            L.REPO_ROOT,
            f"test_data/input_gvcfs5/{s}.chr21.final.hard-filtered.gvcf.gz",
        )
        for s in ("HG00096", "HG00097")
    ],
    [
        os.path.join(
            L.REPO_ROOT,
            f"test_data/downloads/{s}/{s}.chr21.final.hard-filtered.gvcf.gz",
        )
        for s in ("HG00101", "HG00102")
    ],
]
ALL = [1, 2, 3]
CHILD_REF = 12_000_000  # windows [1,12M] [12M+1,24M] [24M+1,36M] [36M+1,..]
C_EMPTY, C_DATA = 0, 1  # chr20 (no batch), chr21


@pytest.fixture(scope="module")
def staged(docker_stack, tmp_path_factory):
    for paths in SAMPLES:
        for path in paths:
            if not os.path.exists(path):
                pytest.skip(f"missing test input {path}")
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr20,chr21")
    work = str(tmp_path_factory.mktemp("incremental_stage"))
    for i, sources in enumerate(SAMPLES):
        gvcfs = []
        for j, src in enumerate(sources):
            dst = os.path.join(work, f"b{i + 1}_s{j}.g.vcf.gz")
            L.subset_gvcf(src, WINDOWS, dst)
            gvcfs.append(dst)
        L.run_ingest(gvcfs, region="chr21", threads=2)
    conn = L.get_pg_conn()
    try:
        assert L.batch_statuses(conn) == [(b, "complete") for b in ALL]
    finally:
        conn.close()
    return ALL


@pytest.fixture(scope="module")
def rebuild_all(staged):
    """The truth: a full --rebuild over all three batches."""
    conn = L.get_pg_conn()
    try:
        L.clear_lineage(conn, L.get_catalog())
        L.set_completed_batches(conn, ALL)
        proc = L.run_aggregate(fan_in=100, threads=2, rebuild=True)
        assert " full " in proc.stderr
        run = L.latest_agg_run_id(conn)
        return {"pg": L.snapshot_pg(conn), "dense": _dense(run)}
    finally:
        conn.close()


@pytest.fixture(scope="module")
def incremental(rebuild_all):
    """An uninterrupted incremental run: full on {b1, b2}, then +{b3}."""
    conn = L.get_pg_conn()
    try:
        parent, child, _ = _full_then_incremental(conn)
        return {
            "pg": L.snapshot_pg(conn),
            "dense": _dense(child),
            "parent": parent,
            "child": child,
            "files": L.aggregate_file_etags(L.get_catalog()),
        }
    finally:
        conn.close()


def _dense(agg_run_id):
    """A run's dense root without its run and window columns (so roots of
    other runs and other window sizes compare equal)."""
    return L.dense_root(L.get_catalog(), agg_run_id).drop_columns(
        ["agg_run_id", "window_id"]
    )


def _full_then_incremental(conn, extra_env=None, check=True, clear_pg=False):
    """reset_aggregate_runs (so the parent and the child get the same
    agg_run_ids every time, and their staging and mirror files the same
    keys); full on {b1, b2}; then +{b3} (CHILD_REF windows) with
    ``extra_env``. Returns (parent run, child run, the child's process).

    ``clear_pg`` empties the PG aggregates between the two runs, so a crashed
    child followed by its resume must genuinely rebuild every partition it
    had not swapped in yet (a crash otherwise leaves the parent's values in
    place -- the false-pass trap of test_aggregate_resume)."""
    L.reset_aggregate_runs(conn, L.get_catalog())
    L.set_completed_batches(conn, [1, 2])
    proc = L.run_aggregate(fan_in=100, threads=2)
    parent = L.latest_agg_run_id(conn)
    assert f"agg_run_id={parent} fresh full " in proc.stderr
    if clear_pg:
        L.clear_pg_aggregates(conn)
    L.set_completed_batches(conn, ALL)
    proc = L.run_aggregate(
        fan_in=100,
        threads=2,
        ref_shard_size=CHILD_REF,
        extra_env=extra_env,
        check=check,
    )
    child = L.latest_agg_run_id(conn)
    assert child > parent
    return parent, child, proc


def _ref_by_batch(catalog, batch_ids):
    """{(c, pos): (dp, sample_counts)} summed over ``batch_ids``'s dense
    batch_ref_blocks partials."""
    from pyiceberg.expressions import In

    rows = (
        catalog.load_table(f"{L.TEST_NAMESPACE}.batch_ref_blocks")
        .scan(row_filter=In("batch_id", batch_ids))
        .to_arrow()
        .group_by(["chromosome_index", "position"])
        .aggregate([("dp", "sum"), ("sample_counts", "sum")])
    )
    return {
        (c, p): (int(dp or 0), int(sc or 0))
        for c, p, dp, sc in zip(
            rows["chromosome_index"].to_pylist(),
            rows["position"].to_pylist(),
            rows["dp_sum"].to_pylist(),
            rows["sample_counts_sum"].to_pylist(),
        )
    }


def _sites_by_batch(catalog, batch_id):
    from pyiceberg.expressions import EqualTo

    rows = (
        catalog.load_table(f"{L.TEST_NAMESPACE}.batch_site_info")
        .scan(row_filter=EqualTo("batch_id", batch_id))
        .to_arrow()
    )
    return set(
        zip(
            rows["chromosome_index"].to_pylist(),
            rows["position"].to_pylist(),
        )
    )


def test_incremental_equals_rebuild(incremental, rebuild_all):
    """Full on {b1, b2} then +{b3} == a full --rebuild on all three, and the
    sites only b3 reveals carry b1/b2's coverage (the sparse-root trap)."""
    conn = L.get_pg_conn()
    catalog = L.get_catalog()
    try:
        parent, child = incremental["parent"], incremental["child"]
        row = L.agg_run(conn, child)
        assert row["parent_run_id"] == parent
        assert row["delta_batch_ids"] == [3]
        assert row["batch_ids"] == ALL
        assert row["status"] == "complete"
        assert row["ref_shard_size"] == CHILD_REF

        L.assert_pg_bitexact(incremental["pg"], rebuild_all["pg"])
        # The dense root is the full cumulative sum, cut into other windows.
        assert incremental["dense"].equals(rebuild_all["dense"])

        # Sites present in b3's partials but in neither b1's nor b2's.
        b3_only = _sites_by_batch(catalog, 3) - (
            _sites_by_batch(catalog, 1) | _sites_by_batch(catalog, 2)
        )
        older = _ref_by_batch(catalog, [1, 2])
        newer = _ref_by_batch(catalog, [3])
        pg = incremental["pg"]["ref_blocks"]
        truth = rebuild_all["pg"]["ref_blocks"]
        checked = with_older = 0
        for key in b3_only:
            if key not in truth:
                continue  # no ref coverage at all at this site
            checked += 1
            assert pg.get(key) == truth[key], key
            # Non-zero wherever the rebuild's are...
            for got, want in zip(pg[key], truth[key]):
                assert (got != 0) == (want != 0), (key, pg[key], truth[key])
            # ...and exactly b1/b2's coverage plus b3's own.
            o, n = older.get(key, (0, 0)), newer.get(key, (0, 0))
            assert pg[key] == (o[0] + n[0], o[1] + n[1]), key
            if o[1] > 0:
                with_older += 1
        assert checked > 50, checked
        # Most b3-only sites are hom-ref in the four older samples: a parent
        # sparse root would have lost exactly this coverage.
        assert with_older > checked // 2, (with_older, checked)

        L.assert_matches_oracle(conn, catalog)
    finally:
        conn.close()


def test_incremental_banner_and_plan(rebuild_all):
    conn = L.get_pg_conn()
    try:
        parent, child, proc = _full_then_incremental(conn)
        assert f"agg_run_id={child} fresh parent={parent} delta=1 " in (
            proc.stderr
        )
        # chr21: the delta tree is one tier over b3, folded at tier 2.
        assert "plan chr21 (c_idx 1): batches=[3] tiers=1 fold=2" in (
            proc.stderr
        )
        assert f"site_info fold onto parent {parent}: 1 unit(s)" in (
            proc.stderr
        )
        assert "ref_windows=4" in proc.stderr
        # A plain re-run is up to date.
        proc = L.run_aggregate(fan_in=100, threads=2)
        assert f"up to date (agg_run_id={child})" in proc.stderr
    finally:
        conn.close()


def test_chain_of_incremental_runs(rebuild_all):
    """Full on {b1}; +{b2}; +{b3} == the rebuild; each run folds onto the
    previous one (never the pruned grandparent)."""
    conn = L.get_pg_conn()
    try:
        L.clear_lineage(conn, L.get_catalog())
        runs = []
        for n in (1, 2, 3):
            L.set_completed_batches(conn, ALL[:n])
            proc = L.run_aggregate(fan_in=2, threads=2)
            run = L.latest_agg_run_id(conn)
            if runs:
                assert (
                    f"agg_run_id={run} fresh parent={runs[-1]} delta=1 "
                    in (proc.stderr)
                ), proc.stderr[
                    -2000:
                ]
                assert L.agg_run(conn, run)["parent_run_id"] == runs[-1]
                # The parent is now pruned; the grandparent already was.
                assert L.agg_run(conn, runs[-1])["staging_pruned"] is True
            else:
                assert f"agg_run_id={run} fresh full " in proc.stderr
            runs.append(run)
        assert L.agg_run(conn, runs[-1])["staging_pruned"] is False
        L.assert_pg_bitexact(L.snapshot_pg(conn), rebuild_all["pg"])
        assert _dense(runs[-1]).equals(rebuild_all["dense"])
    finally:
        conn.close()


def _run_ids(table):
    return set(table["agg_run_id"].to_pylist())


def test_pruning_at_completion(rebuild_all):
    """After an incremental run completes: the parent's staging (tier_* and
    dense root) is gone and it is marked staging_pruned; the run's own
    non-root tiers are gone and its roots remain; the deletes dropped whole
    files (no rewrite). A run whose only candidate parent is pruned is a
    full run."""
    conn = L.get_pg_conn()
    catalog = L.get_catalog()
    try:
        L.clear_lineage(conn, catalog)
        L.set_completed_batches(conn, [1])
        L.run_aggregate(fan_in=2, threads=2)
        parent = L.latest_agg_run_id(conn)
        for name in L.STAGING_TABLES:
            assert _run_ids(L.staging_rows(catalog, name)) == {parent}, name

        # +{b2, b3} at fan-in 2: the delta tree is one tier (tier 1), the
        # fold is tier 2.
        L.set_completed_batches(conn, ALL)
        proc = L.run_aggregate(fan_in=2, threads=2, ref_shard_size=CHILD_REF)
        child = L.latest_agg_run_id(conn)
        assert f"parent={parent} delta=2 " in proc.stderr
        assert re.search(r"prune tier_site_info: \d+ -> \d+ rows", proc.stderr)

        for metric in ("site_info", "allele_counts"):
            rows = L.staging_rows(catalog, f"tier_{metric}")
            assert _run_ids(rows) == {child}, metric
            assert set(rows["tier"].to_pylist()) == {2}, metric
            assert set(rows["chromosome_index"].to_pylist()) == {C_DATA}
        ref = L.staging_rows(catalog, "tier_ref_blocks")
        assert _run_ids(ref) == {child}
        assert set(ref["tier"].to_pylist()) == {1}
        dense = L.staging_rows(catalog, "tier_ref_blocks_dense")
        assert _run_ids(dense) == {child}
        assert dense.num_rows == rebuild_all["dense"].num_rows

        assert L.agg_run(conn, parent)["staging_pruned"] is True
        assert L.agg_run(conn, child)["staging_pruned"] is False
        # Every prune snapshot is a whole-file delete (copy-on-write would
        # have written an "overwrite" snapshot with added files).
        for name in L.STAGING_TABLES:
            table = catalog.load_table(f"{L.TEST_NAMESPACE}.{name}")
            summary = table.current_snapshot().summary
            assert summary.operation.value == "delete", (name, summary)
            assert not int(summary.get("added-data-files") or 0), name
        L.assert_pg_bitexact(L.snapshot_pg(conn), rebuild_all["pg"])

        # The pruned parent is never chosen again: with the child marked
        # failed by hand, the only complete subset run is the pruned one,
        # so a run over the same batches is a full run.
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE aggregation_progress SET status='failed' "
                "WHERE agg_run_id=%s",
                (child,),
            )
        proc = L.run_aggregate(fan_in=100, threads=2)
        full = L.latest_agg_run_id(conn)
        assert f"agg_run_id={full} fresh full " in proc.stderr
        assert L.agg_run(conn, full)["parent_run_id"] is None
        L.assert_pg_bitexact(L.snapshot_pg(conn), rebuild_all["pg"])
    finally:
        conn.close()


def test_rebuild_after_incremental_is_full(incremental, rebuild_all):
    conn = L.get_pg_conn()
    try:
        parent, child, _ = _full_then_incremental(conn)
        proc = L.run_aggregate(fan_in=100, threads=2, rebuild=True)
        run = L.latest_agg_run_id(conn)
        assert run > child
        assert f"agg_run_id={run} fresh full " in proc.stderr
        row = L.agg_run(conn, run)
        assert row["parent_run_id"] is None
        assert row["delta_batch_ids"] == row["batch_ids"] == ALL
        runs = dict(L.agg_statuses(conn))
        assert child not in runs  # reclaimed: same batch set
        assert L.agg_run(conn, parent)["staging_pruned"] is True
        L.assert_pg_bitexact(L.snapshot_pg(conn), incremental["pg"])
        assert _dense(run).equals(rebuild_all["dense"])
    finally:
        conn.close()


@pytest.mark.parametrize("phase", ["pre_register", "pre_ckpt", "post_ckpt"])
@pytest.mark.parametrize(
    "unit,tier",
    [("site_info", 2), ("ref_blocks", 1)],
    ids=["fold_unit", "window_unit"],
)
def test_incremental_crash_resume(incremental, unit, tier, phase, tmp_path):
    """Crash an incremental run at the barrier of its site_info fold unit
    (tier 2) or of its ref_blocks window units (named by the first): before
    the barrier registers, between registration and checkpoint, or after
    the checkpoint transaction; the failed run keeps its parent's staging,
    the cohort_generation view shows it failed with its parent, joint
    calling refuses; the resume folds onto the same parent and is bit-exact
    with the uninterrupted run, with the same staging and mirror files."""
    conn = L.get_pg_conn()
    try:
        parent, child, proc = _full_then_incremental(
            conn,
            extra_env={"MOSAIC_TEST_FAIL_UNIT": f"{unit}:*:{tier}:0:{phase}"},
            check=False,
            clear_pg=True,
        )
        assert proc.returncode != 0, "aggregate should have crashed"
        assert "MOSAIC_TEST_FAIL_UNIT" in proc.stderr
        assert L.tier_completion_has(conn, unit, tier, 0, child) == (
            phase == "post_ckpt"
        )
        gen = L.cohort_generation(conn)
        assert (gen["agg_run_id"], gen["status"]) == (child, "failed")
        assert gen["parent_run_id"] == parent
        # Pruning happens only at completion: the parent is intact.
        assert L.agg_run(conn, parent)["staging_pruned"] is False
        dense = L.staging_rows(L.get_catalog(), "tier_ref_blocks_dense")
        assert parent in _run_ids(dense)
        if unit == "site_info" and phase == "pre_register":
            out = str(tmp_path / "stale.vcf.gz")
            jc = L.run_joint_call(out, check=False)
            assert jc.returncode == 2, jc.stderr[-2000:]
            assert f"(agg_run_id={child}) is failed" in jc.stderr

        proc = L.run_aggregate(fan_in=100, threads=2, ref_shard_size=CHILD_REF)
        assert f"agg_run_id={child} resume parent={parent} delta=1 " in (
            proc.stderr
        ), proc.stderr[-2000:]
        if unit == "site_info":
            m = re.search(
                rf"site_info fold onto parent {parent}: \d+ unit\(s\) "
                r"\((\d+) done, (\d+) skipped",
                proc.stderr,
            )
            assert m, proc.stderr[-2000:]
            done, skipped = int(m.group(1)), int(m.group(2))
        else:
            m = re.search(
                r"ref_blocks windows: \d+ unit\(s\) over 1 chrom "
                r"\((\d+) done, (\d+) skipped",
                proc.stderr,
            )
            assert m, proc.stderr[-2000:]
            done, skipped = int(m.group(1)), int(m.group(2))
        if phase == "post_ckpt":
            assert skipped >= 1
        else:
            assert done >= 1
        assert L.cohort_generation(conn)["status"] == "complete"
        assert L.snapshot_pg(conn) == incremental["pg"]
        assert _dense(child).equals(incremental["dense"])
        assert L.agg_run(conn, parent)["staging_pruned"] is True
        for name in L.STAGING_TABLES:
            rows = L.staging_rows(L.get_catalog(), name)
            assert _run_ids(rows) == {child}, name
        assert (parent, child) == (
            incremental["parent"],
            incremental["child"],
        )
        assert L.aggregate_file_etags(L.get_catalog()) == incremental["files"]
    finally:
        conn.close()


def _prune_sentinel(conn, agg_run_id):
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM tier_completion WHERE agg_run_id=%s AND "
            "metric='prune' AND chromosome_index=0 AND tier=-1 AND "
            "group_id=0",
            (agg_run_id,),
        )
        return cur.fetchone()[0] == 1


@pytest.mark.parametrize("phase", ["pre_register", "pre_ckpt", "post_ckpt"])
def test_incremental_prune_crash_resume(incremental, phase):
    """Crash an incremental run at its completion prune, just before /
    after the prune sentinel (and the staging_pruned marks on the other
    runs) commits (the prune registers nothing: pre_register raises where
    pre_ckpt does). Every unit and finalize is checkpointed by then. After
    post_ckpt the parent IS marked pruned, so it no longer qualifies as a
    parent -- the run's own prune record is what lets the plain re-run
    resume it (onto its stored parent) instead of starting a full run. The
    resume skips every unit, redoes the deletes and completes, equal to the
    uninterrupted run."""
    conn = L.get_pg_conn()
    catalog = L.get_catalog()
    try:
        parent, child, proc = _full_then_incremental(
            conn,
            extra_env={"MOSAIC_TEST_FAIL_UNIT": f"prune:*:-1:0:{phase}"},
            check=False,
        )
        assert proc.returncode != 0, "aggregate should have crashed"
        assert "MOSAIC_TEST_FAIL_UNIT: injected crash prune" in proc.stderr
        assert dict(L.agg_statuses(conn))[child] == "failed"
        post = phase == "post_ckpt"
        assert _prune_sentinel(conn, child) is post
        assert L.agg_run(conn, parent)["staging_pruned"] is post
        # Nothing was deleted yet: the parent's roots are still there.
        for name in L.STAGING_TABLES:
            ids = _run_ids(L.staging_rows(catalog, name))
            assert {parent, child} <= ids, (name, ids)
        for metric in ("site_info", "ref_blocks", "allele_counts"):
            assert L.finalize_marker_exists(conn, metric, child)

        proc = L.run_aggregate(fan_in=100, threads=2, ref_shard_size=CHILD_REF)
        assert f"agg_run_id={child} resume parent={parent} delta=1 " in (
            proc.stderr
        ), proc.stderr[-2000:]
        for metric in ("allele_counts", "site_info"):
            m = re.search(
                rf"{metric} fold onto parent {parent}: (\d+) unit\(s\) "
                r"\((\d+) done, (\d+) skipped",
                proc.stderr,
            )
            assert m, proc.stderr[-2000:]
            assert int(m.group(2)) == 0 and int(m.group(3)) == int(m.group(1))
        m = re.search(
            r"ref_blocks windows: (\d+) unit\(s\) over 1 chrom "
            r"\((\d+) done, (\d+) skipped",
            proc.stderr,
        )
        assert m, proc.stderr[-2000:]
        assert int(m.group(2)) == 0 and int(m.group(3)) == int(m.group(1))
        for metric in ("site_info", "ref_blocks", "allele_counts"):
            assert f"{metric} finalize: skipped" in proc.stderr
        assert re.search(
            r"prune tier_ref_blocks_dense: \d+ -> \d+ rows", proc.stderr
        )

        assert L.cohort_generation(conn)["status"] == "complete"
        assert _prune_sentinel(conn, child)
        assert L.agg_run(conn, parent)["staging_pruned"] is True
        assert L.snapshot_pg(conn) == incremental["pg"]
        assert _dense(child).equals(incremental["dense"])
        for name in L.STAGING_TABLES:
            assert _run_ids(L.staging_rows(catalog, name)) == {child}, name
        assert L.aggregate_file_etags(catalog) == incremental["files"]
    finally:
        conn.close()
