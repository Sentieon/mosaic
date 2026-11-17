# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for `mosaic maintain` (mosaic/maintain.py) that need no Docker.

The retention policy and the reclaim set are pure functions over a snapshot
graph; the S3 helpers are checked against a fake client; the manifest rewrite
and the expire commit run on a real pyiceberg table in a local directory
whose catalog commit is simulated (the updates applied in memory), so the
pyiceberg-dependent pieces are covered without a catalog. The command itself
(lock, guard, hook, the reclaim against S3, the orphan sweep) is covered by
tests/integration/test_iceberg_maintain.py.
"""

import argparse
from datetime import datetime, timedelta, timezone
from unittest import mock

import pyarrow as pa
import pytest
from pyiceberg.io import load_file_io
from pyiceberg.partitioning import UNPARTITIONED_PARTITION_SPEC
from pyiceberg.schema import Schema
from pyiceberg.table import Table, TableProperties
from pyiceberg.table.metadata import new_table_metadata
from pyiceberg.table.sorting import UNSORTED_SORT_ORDER
from pyiceberg.table.update import (
    RemovePropertiesUpdate,
    RemoveSnapshotsUpdate,
    SetPropertiesUpdate,
    update_table_metadata,
)
from pyiceberg.types import IntegerType, NestedField

from mosaic import cli, init_db
from mosaic.config import PARQUET_ROW_GROUP_ROWS, PARQUET_SITE_ROW_GROUP_ROWS
from mosaic.genotypes import BATCH_GENOTYPES_TABLE
from mosaic.init_db import ICEBERG_TABLE_PROPERTIES, ICEBERG_TABLES
from mosaic.maintain import (
    DEFAULT_MANIFEST_TARGET_SIZE_BYTES,
    DELETE_CHUNK,
    FileRef,
    SnapshotGraph,
    SnapshotStamp,
    _expire,
    _maybe_fail_maintain,
    _rewrite_manifests,
    chunked,
    classify_objects,
    compute_reclaim,
    delete_uris,
    group_by_bucket,
    hours_type,
    load_graph,
    parse_fail_maintain,
    parse_s3_uri,
    plan_retention,
    retain_last_type,
    target_size_type,
)

pytestmark = pytest.mark.unit

HOUR_MS = 3_600_000
NOW_MS = 1_000 * HOUR_MS


def _stamps(*ages_hours):
    """Snapshots 1..n in commit order, aged ``ages_hours`` before NOW_MS."""
    return [
        SnapshotStamp(i + 1, NOW_MS - int(age * HOUR_MS))
        for i, age in enumerate(ages_hours)
    ]


# ── retention policy ─────────────────────────────────────────────────────────


def test_current_is_always_retained():
    snaps = _stamps(300, 200, 100)
    # The current snapshot is not the newest (a rollback): kept anyway.
    retained, expired = plan_retention(snaps, 1, [], 1, 0, NOW_MS)
    assert retained == {1, 3} and expired == {2}


def test_retain_last_keeps_the_newest_n():
    snaps = _stamps(500, 400, 300, 200, 100)
    retained, expired = plan_retention(snaps, 5, [], 3, 0, NOW_MS)
    assert retained == {3, 4, 5} and expired == {1, 2}


def test_older_than_zero_retains_by_count_only():
    snaps = _stamps(0.01, 0.001, 0)
    retained, expired = plan_retention(snaps, 3, [], 1, 0, NOW_MS)
    assert retained == {3} and expired == {1, 2}


def test_older_than_retains_the_young():
    snaps = _stamps(200, 121, 119, 1)
    retained, expired = plan_retention(snaps, 4, [], 1, 120, NOW_MS)
    assert retained == {3, 4} and expired == {1, 2}


def test_protected_refs_are_retained():
    snaps = _stamps(300, 200, 100)
    # A tag on snapshot 1; an id no snapshot has is ignored.
    retained, expired = plan_retention(snaps, 3, [1, 99], 1, 0, NOW_MS)
    assert retained == {1, 3} and expired == {2}


def test_newest_is_by_timestamp_not_commit_order():
    # Snapshot 2 is the newest by timestamp although committed before 3.
    snaps = [
        SnapshotStamp(1, NOW_MS - 3 * HOUR_MS),
        SnapshotStamp(2, NOW_MS - 1 * HOUR_MS),
        SnapshotStamp(3, NOW_MS - 2 * HOUR_MS),
    ]
    retained, _ = plan_retention(snaps, 3, [], 2, 0, NOW_MS)
    assert retained == {2, 3}
    retained, _ = plan_retention(snaps, None, [], 1, 0, NOW_MS)
    assert retained == {2}


def test_equal_timestamps_break_ties_by_commit_order():
    snaps = [SnapshotStamp(i, NOW_MS) for i in (7, 8, 9)]
    retained, expired = plan_retention(snaps, None, [], 1, 0, NOW_MS)
    assert retained == {9} and expired == {7, 8}


def test_no_snapshots():
    assert plan_retention([], None, [], 1, 0, NOW_MS) == (set(), set())


@pytest.mark.parametrize("retain_last, hours", [(0, 0), (1, -1)])
def test_retention_refuses_bad_parameters(retain_last, hours):
    with pytest.raises(ValueError):
        plan_retention(_stamps(1), 1, [], retain_last, hours, NOW_MS)


# ── the reclaim set ──────────────────────────────────────────────────────────


def _graph(snapshots):
    """``snapshots``: {snapshot id: {manifest path: [(file, live)]}}; each
    file is 10 bytes, each manifest 100."""
    g = SnapshotGraph()
    for sid, mans in snapshots.items():
        g.lists[sid] = f"list-{sid}"
        g.manifests[sid] = list(mans)
        for path, entries in mans.items():
            g.entries[path] = [FileRef(f, 10, live) for f, live in entries]
            g.manifest_sizes[path] = 100
    return g


def test_shared_manifest_is_kept_and_expired_list_is_reclaimed():
    # A fast append carries m1 forward from snapshot 1 into snapshot 2.
    g = _graph(
        {
            1: {"m1": [("a", True)]},
            2: {"m1": [("a", True)], "m2": [("b", True)]},
        }
    )
    assert compute_reclaim(g, {2}, {1}) == {"list-1": "lists"}


def test_expired_manifest_and_its_files_are_reclaimed():
    # Snapshot 2 rewrote m1 (live a, c) into m3 and deleted c.
    g = _graph(
        {
            1: {"m1": [("a", True), ("c", True)]},
            2: {"m3": [("a", True), ("c", False)]},
        }
    )
    assert compute_reclaim(g, {2}, {1}) == {
        "list-1": "lists",
        "m1": "manifests",
        "c": "data",
    }


def test_file_deleted_only_in_a_retained_snapshot_is_reclaimed():
    # The prune case: snapshot 2 (retained, current) records c as DELETED;
    # no retained snapshot has it live, so it goes although Iceberg's own
    # expiry would keep it until snapshot 2 expires.
    g = _graph(
        {
            1: {"m1": [("a", True)]},
            2: {"m1": [("a", True)], "m2": [("c", False)]},
        }
    )
    assert compute_reclaim(g, {2}, {1}) == {"list-1": "lists", "c": "data"}


def test_file_live_in_any_retained_snapshot_is_kept():
    # c is DELETED in retained snapshot 3 but live in retained snapshot 2,
    # and live in expired snapshot 1.
    g = _graph(
        {
            1: {"m1": [("c", True)]},
            2: {"m2": [("c", True)]},
            3: {"m3": [("c", False)]},
        }
    )
    plan = compute_reclaim(g, {2, 3}, {1})
    assert "c" not in plan
    assert plan == {"list-1": "lists", "m1": "manifests"}


def test_nothing_retained_is_ever_reclaimed():
    g = _graph(
        {
            1: {"m1": [("a", True), ("b", False)], "m4": [("d", True)]},
            2: {"m1": [("a", True), ("b", False)], "m2": [("b", False)]},
            3: {"m3": [("a", True), ("e", True)], "m4": [("d", True)]},
        }
    )
    retained, expired = {3}, {1, 2}
    plan = compute_reclaim(g, retained, expired)
    kept = {g.lists[s] for s in retained}
    kept |= {m for s in retained for m in g.manifests[s]}
    kept |= {
        r.path
        for s in retained
        for m in g.manifests[s]
        for r in g.entries[m]
        if r.live
    }
    assert not set(plan) & kept
    assert plan == {
        "list-1": "lists",
        "list-2": "lists",
        "m1": "manifests",
        "m2": "manifests",
        "b": "data",
    }


def test_nothing_expired_reclaims_nothing():
    # Even a file only DELETED in the retained snapshot stays: with nothing
    # to expire the table is skipped (and a crashed run's leftovers are the
    # orphan sweep's).
    g = _graph({1: {"m1": [("a", True), ("c", False)]}})
    assert compute_reclaim(g, {1}, set()) == {}


# ── S3 URIs, grouping, chunking, deletes ─────────────────────────────────────


@pytest.mark.parametrize(
    "uri, want",
    [
        (
            "s3://warehouse/jc_test/t/metadata/x.avro",
            ("warehouse", "jc_test/t/metadata/x.avro"),
        ),
        ("s3a://b/k", ("b", "k")),
        ("s3n://b/a/b/c", ("b", "a/b/c")),
    ],
)
def test_parse_s3_uri(uri, want):
    assert parse_s3_uri(uri) == want


@pytest.mark.parametrize(
    "uri", ["", "s3://", "s3://bucket", "s3://bucket/", "file:///tmp/x", "b/k"]
)
def test_parse_s3_uri_refuses_the_rest(uri):
    with pytest.raises(ValueError):
        parse_s3_uri(uri)


def test_group_by_bucket_parses_every_uri():
    uris = [
        "s3://jc-test-warehouse/tier_site_info/h1/data.parquet",
        "s3://warehouse/jc_test/tier_site_info/metadata/snap-1.avro",
        "s3://jc-test-warehouse/tier_site_info/h0/data.parquet",
        "s3://warehouse/jc_test/tier_site_info/metadata/m0.avro",
        "s3://jc-test-warehouse/tier_site_info/h0/data.parquet",
    ]
    assert group_by_bucket(uris) == {
        "jc-test-warehouse": [
            "tier_site_info/h0/data.parquet",
            "tier_site_info/h1/data.parquet",
        ],
        "warehouse": [
            "jc_test/tier_site_info/metadata/m0.avro",
            "jc_test/tier_site_info/metadata/snap-1.avro",
        ],
    }


def test_chunked_by_1000():
    keys = [f"k{i}" for i in range(2_345)]
    chunks = chunked(keys)
    assert DELETE_CHUNK == 1_000
    assert [len(c) for c in chunks] == [1_000, 1_000, 345]
    assert [k for c in chunks for k in c] == keys
    assert chunked([], 1_000) == []
    with pytest.raises(ValueError):
        chunked(keys, 0)


def test_delete_uris_chunks_per_bucket_and_tolerates_missing_keys():
    s3 = mock.MagicMock()
    s3.delete_objects.return_value = {
        "Errors": [{"Key": "x", "Code": "NoSuchKey"}]
    }
    uris = [f"s3://a/k{i:04d}" for i in range(1_500)] + ["s3://b/one"]
    assert delete_uris(s3, uris) == 1_501
    calls = [
        (c.kwargs["Bucket"], len(c.kwargs["Delete"]["Objects"]))
        for c in s3.delete_objects.call_args_list
    ]
    assert calls == [("a", 1_000), ("a", 500), ("b", 1)]


def test_delete_uris_raises_on_other_errors():
    s3 = mock.MagicMock()
    s3.delete_objects.return_value = {
        "Errors": [{"Key": "k", "Code": "AccessDenied"}]
    }
    with pytest.raises(RuntimeError, match="AccessDenied"):
        delete_uris(s3, ["s3://a/k"])


def test_delete_nothing_makes_no_request():
    s3 = mock.MagicMock()
    assert delete_uris(s3, []) == 0
    s3.delete_objects.assert_not_called()


# ── the orphan filter ────────────────────────────────────────────────────────


def test_classify_objects():
    now = datetime(2026, 10, 28, 12, tzinfo=timezone.utc)
    old = now - timedelta(hours=100)
    young = now - timedelta(hours=1)
    objects = {
        ("d", "sample_ref_blocks/h1/data.parquet"): (old, 10),
        ("d", "sample_ref_blocks/h2/data.parquet"): (old, 20),
        ("d", "sample_ref_blocks/h3/data.parquet"): (young, 30),
        ("d", "sample_ref_blocks/h4/data.parquet"): (old, 40),
        ("m", "ns/t/metadata/zz.metadata.json"): (old, 50),
        ("m", "ns/t/metadata/v1.metadata.json"): (young, 60),
        # Outside the swept prefixes: never touched, even when old.
        ("d", "sample_ref_blocks_x/h5/data.parquet"): (old, 70),
        ("m", "other/t/metadata/v1.metadata.json"): (old, 80),
    }
    prefixes = [("d", "sample_ref_blocks/"), ("m", "ns/t/")]
    referenced = {
        ("d", "sample_ref_blocks/h1/data.parquet"),
        ("m", "ns/t/metadata/v1.metadata.json"),
    }
    pending = {
        ("d", "sample_ref_blocks/h2/data.parquet"),
        ("d", "sample_ref_blocks/h1/data.parquet"),
    }
    out = classify_objects(objects, prefixes, referenced, pending, now, 72)
    assert out == {
        "outside": [
            ("d", "sample_ref_blocks_x/h5/data.parquet"),
            ("m", "other/t/metadata/v1.metadata.json"),
        ],
        "referenced": [
            ("d", "sample_ref_blocks/h1/data.parquet"),
            ("m", "ns/t/metadata/v1.metadata.json"),
        ],
        "pending": [("d", "sample_ref_blocks/h2/data.parquet")],
        "young": [("d", "sample_ref_blocks/h3/data.parquet")],
        "orphan": [
            ("d", "sample_ref_blocks/h4/data.parquet"),
            ("m", "ns/t/metadata/zz.metadata.json"),
        ],
    }
    # Grace 0: nothing is young.
    out = classify_objects(objects, prefixes, referenced, pending, now, 0)
    assert out["young"] == []
    assert ("d", "sample_ref_blocks/h3/data.parquet") in out["orphan"]


# ── the MOSAIC_TEST_FAIL_MAINTAIN hook ───────────────────────────────────────


@pytest.mark.parametrize("spec", [None, ""])
def test_hook_unset_is_inert(spec):
    assert parse_fail_maintain(spec) is None
    _maybe_fail_maintain(None, "tier_site_info", "pre_expire")


@pytest.mark.parametrize(
    "spec, want",
    [
        ("tier_site_info:pre_expire", ("tier_site_info", "pre_expire")),
        ("batch_genotypes:post_expire", ("batch_genotypes", "post_expire")),
    ],
)
def test_hook_parses(spec, want):
    assert parse_fail_maintain(spec) == want


@pytest.mark.parametrize(
    "spec",
    [
        "tier_site_info",
        "tier_site_info:",
        ":pre_expire",
        "tier_site_info:pre_commit",
        "no_such_table:pre_expire",
        "jc_test.tier_site_info:post_expire",
    ],
)
def test_malformed_hook_is_refused(spec):
    with pytest.raises(ValueError):
        parse_fail_maintain(spec)


def test_hook_fires_only_on_its_table_and_phase():
    hook = parse_fail_maintain("tier_site_info:post_expire")
    _maybe_fail_maintain(hook, "tier_site_info", "pre_expire")
    _maybe_fail_maintain(hook, "tier_allele_counts", "post_expire")
    with pytest.raises(RuntimeError, match="tier_site_info @post_expire"):
        _maybe_fail_maintain(hook, "tier_site_info", "post_expire")


# ── options ──────────────────────────────────────────────────────────────────


def test_option_types():
    assert retain_last_type("3") == 3
    assert hours_type("0") == 0.0 and hours_type("1.5") == 1.5
    assert target_size_type(str(1 << 20)) == 1 << 20
    for f, bad in [
        (retain_last_type, ["0", "-1", "x", "1.5"]),
        (hours_type, ["-1", "-0.5", "x", "nan", "inf"]),
        (target_size_type, [str((1 << 20) - 1), "0", "x"]),
    ]:
        for value in bad:
            with pytest.raises(argparse.ArgumentTypeError):
                f(value)


def test_maintain_subcommand_parses_and_defaults():
    parser = cli.build_parser()
    args = parser.parse_args(["maintain"])
    assert (args.retain_last, args.older_than_hours) == (1, 120.0)
    assert args.rewrite_manifests and not args.orphans
    assert (
        args.manifest_target_size_bytes == DEFAULT_MANIFEST_TARGET_SIZE_BYTES
    )
    assert args.orphan_grace_hours == 72.0
    assert not args.dry_run and not args.force
    args = parser.parse_args(
        [
            "maintain",
            "--older-than-hours",
            "0",
            "--no-rewrite-manifests",
            "--orphans",
            "--orphan-grace-hours",
            "0",
            "--dry-run",
            "--force",
        ]
    )
    assert args.older_than_hours == 0 and args.orphan_grace_hours == 0
    assert not args.rewrite_manifests and args.orphans
    assert args.dry_run and args.force


@pytest.mark.parametrize(
    "argv",
    [
        ["--retain-last", "0"],
        ["--older-than-hours", "-1"],
        ["--orphan-grace-hours", "-1"],
        ["--manifest-target-size-bytes", "1048575"],
    ],
)
def test_maintain_refuses_bad_options(argv):
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["maintain", *argv])


# ── the table list and properties ────────────────────────────────────────────


def test_iceberg_tables_is_the_single_list():
    assert len(ICEBERG_TABLES) == len(set(ICEBERG_TABLES)) == 13
    assert BATCH_GENOTYPES_TABLE in ICEBERG_TABLES
    assert ICEBERG_TABLE_PROPERTIES == {
        "write.metadata.delete-after-commit.enabled": "true"
    }


def test_init_creates_every_table_with_the_properties():
    catalog = mock.MagicMock()
    init_db.init_iceberg_tables(catalog)
    calls = catalog.create_table.call_args_list
    names = [c.args[0].split(".", 1)[1] for c in calls]
    assert names == list(ICEBERG_TABLES)
    for c in calls:
        short = c.args[0].split(".", 1)[1]
        props = c.kwargs["properties"]
        # ICEBERG_TABLE_PROPERTIES on every table (what `mosaic maintain`
        # checks); the per-sample tables also get the row-group cap (E3).
        assert {
            k: v for k, v in props.items() if k in ICEBERG_TABLE_PROPERTIES
        } == ICEBERG_TABLE_PROPERTIES
        if short in init_db.SAMPLE_TABLE_SORT_COLUMNS:
            cap = (
                PARQUET_ROW_GROUP_ROWS
                if short == "sample_ref_blocks"
                else PARQUET_SITE_ROW_GROUP_ROWS
            )
            assert props == {
                **ICEBERG_TABLE_PROPERTIES,
                "write.parquet.row-group-limit": str(cap),
                # The per-sample codec: a copy-on-write rewrite
                # (mosaic remove) writes brotli, which the Rust joint
                # caller reads.
                "write.parquet.compression-codec": "brotli",
                "write.parquet.compression-level": "5",
            }
            # mosaic maintain sets the codec on an older lake.
            assert init_db.maintained_properties(short) == {
                **ICEBERG_TABLE_PROPERTIES,
                "write.parquet.compression-codec": "brotli",
                "write.parquet.compression-level": "5",
            }
        else:
            assert props == ICEBERG_TABLE_PROPERTIES
            assert (
                init_db.maintained_properties(short)
                == ICEBERG_TABLE_PROPERTIES
            )
        # batch_genotypes and the (whole-sample) per-sample tables are
        # partitioned by chromosome_index; only the latter declare a sort
        # order.
        per_sample = short in init_db.SAMPLE_TABLE_SORT_COLUMNS
        partitioned = short == BATCH_GENOTYPES_TABLE or per_sample
        assert ("partition_spec" in c.kwargs) == partitioned
        assert ("sort_order" in c.kwargs) == per_sample


# ── pyiceberg integration on a local table (catalog commit simulated) ───────


@pytest.fixture
def local_table(tmp_path, monkeypatch):
    """A real pyiceberg table under ``tmp_path`` whose commits are applied
    in memory (requirements validated), as a catalog would."""

    def commit(self, updates, requirements):
        for r in requirements:
            r.validate(self.metadata)
        self.metadata = update_table_metadata(self.metadata, updates)
        commits.append(updates)

    commits = []
    monkeypatch.setattr(Table, "_do_commit", commit)
    location = f"file://{tmp_path}/t"
    metadata = new_table_metadata(
        Schema(NestedField(1, "x", IntegerType(), required=False)),
        UNPARTITIONED_PARTITION_SPEC,
        UNSORTED_SORT_ORDER,
        location,
        {TableProperties.MANIFEST_MIN_MERGE_COUNT: "50"},
    )
    table = Table(
        identifier=("ns", "t"),
        metadata=metadata,
        metadata_location=f"{location}/metadata/v0.metadata.json",
        io=load_file_io(location=location),
        catalog=mock.MagicMock(
            **{"supports_server_side_planning.return_value": False}
        ),
    )
    for i in range(3):  # three fast appends: three manifests
        table.append(pa.table({"x": pa.array([i, i + 10], pa.int32())}))
    table.commits = commits
    return table


def _data_files(table):
    return sorted(t.file.file_path for t in table.scan().plan_files())


def test_rewrite_merges_manifests_and_restores_properties(local_table):
    t = local_table
    files = _data_files(t)
    props = dict(t.properties)
    assert len(t.current_snapshot().manifests(t.io)) == 3
    _rewrite_manifests(t, DEFAULT_MANIFEST_TARGET_SIZE_BYTES)
    assert len(t.current_snapshot().manifests(t.io)) == 1
    assert _data_files(t) == files
    assert t.scan().to_arrow().num_rows == 6
    assert len(t.metadata.snapshots) == 4
    # The merge properties were staged, then set back (the one the table
    # had) or removed (the two it lacked), in the same commit.
    assert dict(t.properties) == props
    (updates,) = t.commits[-1:]
    kinds = [type(u) for u in updates]
    assert kinds[0] is SetPropertiesUpdate
    assert kinds[-2:] == [SetPropertiesUpdate, RemovePropertiesUpdate]
    assert updates[0].updates == {
        TableProperties.MANIFEST_MERGE_ENABLED: "true",
        TableProperties.MANIFEST_MIN_MERGE_COUNT: "2",
        TableProperties.MANIFEST_TARGET_SIZE_BYTES: str(
            DEFAULT_MANIFEST_TARGET_SIZE_BYTES
        ),
    }
    assert updates[-2].updates == {
        TableProperties.MANIFEST_MIN_MERGE_COUNT: "50"
    }
    assert set(updates[-1].removals) == {
        TableProperties.MANIFEST_MERGE_ENABLED,
        TableProperties.MANIFEST_TARGET_SIZE_BYTES,
    }


def test_expire_is_one_commit_of_single_id_updates(local_table):
    t = local_table
    _rewrite_manifests(t, DEFAULT_MANIFEST_TARGET_SIZE_BYTES)
    current = t.metadata.current_snapshot_id
    expired = {s.snapshot_id for s in t.metadata.snapshots} - {current}
    graph = load_graph(t, {}, mock.MagicMock(map=map))
    plan = compute_reclaim(graph, {current}, expired)
    n_commits = len(t.commits)
    _expire(t, expired)
    assert len(t.commits) == n_commits + 1
    (updates,) = t.commits[-1:]
    # One id per update: the local REST server refuses a multi-id one.
    assert [u.snapshot_ids for u in updates] == [[s] for s in sorted(expired)]
    assert all(isinstance(u, RemoveSnapshotsUpdate) for u in updates)
    assert [s.snapshot_id for s in t.metadata.snapshots] == [current]
    assert [e.snapshot_id for e in t.metadata.snapshot_log] == [current]
    # The three appends' lists and manifests go, no data file does.
    kinds = sorted(plan.values())
    assert kinds == ["lists"] * 3 + ["manifests"] * 3
    assert not set(plan) & set(_data_files(t))
    assert t.scan().to_arrow().num_rows == 6
