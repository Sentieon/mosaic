# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""`mosaic clean` (cloud mode) deleting each dropped table's own location,
and `scripts/measure_s3_usage.py`'s prefixes from ``ICEBERG_TABLES``.
Fakes for the catalog and S3; no Docker."""

import datetime
import importlib.util
import os

import pytest
from pyiceberg.exceptions import NoSuchTableError

from mosaic import clean_db
from mosaic.config import ICEBERG_NAMESPACE
from mosaic.init_db import ICEBERG_TABLES

pytestmark = pytest.mark.unit

REPO = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
T0 = datetime.datetime(2026, 11, 5, tzinfo=datetime.timezone.utc)


class FakeS3:
    """list_objects_v2 pages of 1,000 keys and delete_objects over a dict
    of buckets; ``unlistable`` buckets raise on listing."""

    def __init__(self, objects, unlistable=()):
        self.objects = {b: set(keys) for b, keys in objects.items()}
        self.unlistable = set(unlistable)
        self.delete_calls = []

    def get_paginator(self, name):
        assert name == "list_objects_v2"
        s3 = self

        class P:
            def paginate(self, Bucket, Prefix):
                if Bucket in s3.unlistable:
                    raise RuntimeError(f"AccessDenied on {Bucket}")
                keys = sorted(
                    k
                    for k in s3.objects.get(Bucket, ())
                    if k.startswith(Prefix)
                )
                for i in range(0, max(len(keys), 1), 1000):
                    yield {
                        "Contents": [
                            {"Key": k, "LastModified": T0, "Size": 1}
                            for k in keys[i : i + 1000]
                        ]
                    }

        return P()

    def delete_objects(self, Bucket, Delete):
        keys = [o["Key"] for o in Delete["Objects"]]
        assert len(keys) <= 1000
        self.delete_calls.append((Bucket, len(keys)))
        self.objects[Bucket] -= set(keys)
        return {}


class FakeCatalog:
    def __init__(self, locations):
        self.locations = dict(locations)  # short name -> location
        self.dropped = []

    def load_table(self, name):
        short = name.split(".", 1)[1]
        if short not in self.locations:
            raise LookupError(f"no table {name}")
        loc = self.locations[short]

        class T:
            def location(self):
                return loc

        return T()

    def drop_table(self, name):
        self.dropped.append(name)
        del self.locations[name.split(".", 1)[1]]


def test_drop_records_locations_of_dropped_tables_only():
    cat = FakeCatalog(
        {
            "sample_ref_blocks": "s3://wh/mosaic.db/sample_ref_blocks",
            "batch_genotypes": "s3://wh/mosaic.db/batch_genotypes/",
        }
    )
    locs = clean_db._drop_tables_recording_locations(cat)
    assert locs == {
        "sample_ref_blocks": "s3://wh/mosaic.db/sample_ref_blocks",
        "batch_genotypes": "s3://wh/mosaic.db/batch_genotypes/",
    }
    assert sorted(cat.dropped) == sorted(
        f"{ICEBERG_NAMESPACE}.{t}" for t in locs
    )
    assert len(ICEBERG_TABLES) == 13


def test_delete_table_locations_every_object_any_bucket(capsys):
    keys = [
        f"mosaic.db/sample_site_info/data/f{i}.parquet" for i in range(2500)
    ]
    s3 = FakeS3(
        {
            "wh": set(keys)
            | {
                "mosaic.db/sample_site_info_other/keep.parquet",
                "mosaic.db/batch_genotypes/metadata/v1.metadata.json",
                "sample_site_info/abc/data.parquet",
            },
            "other": {"loc/agg_site_info/metadata/v1.metadata.json"},
        }
    )
    failures = clean_db._delete_table_locations(
        s3,
        {
            "sample_site_info": "s3://wh/mosaic.db/sample_site_info",
            "agg_site_info": "s3://other/loc/agg_site_info/",
            "tier_site_info": "s3://wh/mosaic.db/tier_site_info",  # empty
        },
    )
    assert failures == []
    # Exactly the location's objects (the trailing '/' keeps the sibling
    # sample_site_info_other/ and the bucket-root upload prefix).
    assert s3.objects["wh"] == {
        "mosaic.db/sample_site_info_other/keep.parquet",
        "mosaic.db/batch_genotypes/metadata/v1.metadata.json",
        "sample_site_info/abc/data.parquet",
    }
    assert s3.objects["other"] == set()
    assert s3.delete_calls == [
        ("wh", 1000),
        ("wh", 1000),
        ("wh", 500),
        ("other", 1),
    ]
    out = capsys.readouterr().out
    assert "Deleted 2500 S3 objects under the sample_site_info location" in out
    assert "Deleted 0 S3 objects under the tier_site_info location" in out


def test_delete_table_locations_unlistable_is_a_failure():
    s3 = FakeS3({"wh": {"mosaic.db/agg_site_info/m.json"}}, unlistable={"no"})
    failures = clean_db._delete_table_locations(
        s3,
        {
            "agg_site_info": "s3://wh/mosaic.db/agg_site_info",
            "tier_site_info": "s3://no/mosaic.db/tier_site_info",
            "batch_site_info": "file:///tmp/batch_site_info",
        },
    )
    assert s3.objects["wh"] == set()  # the others still deleted
    assert len(failures) == 2
    assert "tier_site_info" in failures[0] and "AccessDenied" in failures[0]
    assert "batch_site_info" in failures[1]


def _load_measure_script():
    path = os.path.join(REPO, "scripts", "measure_s3_usage.py")
    spec = importlib.util.spec_from_file_location("measure_s3_usage", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_measure_s3_usage_prefixes_are_the_tables():
    m = _load_measure_script()
    groups = m.PREFIX_GROUPS
    assert sorted(groups["ingest"] + groups["aggregate"]) == sorted(
        f"{t}/" for t in ICEBERG_TABLES
    )
    assert "batch_genotypes/" in groups["ingest"]
    assert "tier_ref_blocks_dense/" in groups["aggregate"]
    assert all(p.startswith(("sample_", "batch_")) for p in groups["ingest"])
    assert all(p.startswith(("agg_", "tier_")) for p in groups["aggregate"])
    with pytest.raises(ValueError, match="new_table"):
        m.prefix_groups(("sample_site_info", "new_table"))


# ── _verify_iceberg_clean ────────────────────────────────────────────────────


class VerifyCatalog:
    """A catalog whose namespace survives the clean; ``tables`` maps a short
    table name to its row count (0: empty) or to the exception that
    ``load_table`` raises."""

    def __init__(self, tables):
        self.tables = tables

    def list_namespaces(self):
        return [(ICEBERG_NAMESPACE,)]

    def list_tables(self, namespace):
        return []

    def load_table(self, name):
        state = self.tables[name.split(".", 1)[1]]
        if isinstance(state, Exception):
            raise state

        class Arrow:
            def __len__(self):
                return state

        class Scan:
            def to_arrow(self):
                return Arrow()

        class T:
            def scan(self, limit=None):
                return Scan()

        return T()


def _verify(capsys, special):
    """Every table dropped (NoSuchTableError) but ``special``'s."""
    tables = {
        t: NoSuchTableError(f"Table does not exist: {t}")
        for t in ICEBERG_TABLES
    }
    tables.update(special)
    clean_db._verify_iceberg_clean(VerifyCatalog(tables))
    return capsys.readouterr().out


def test_verify_dropped_table_is_ok(capsys):
    out = _verify(capsys, {})
    for t in ICEBERG_TABLES:
        assert f"  OK: {ICEBERG_NAMESPACE}.{t} dropped" in out
    assert "Could not check" not in out
    assert "Table does not exist" not in out


def test_verify_other_error_keeps_the_note(capsys):
    out = _verify(capsys, {"agg_site_info": RuntimeError("catalog down")})
    assert (
        f"  Note: Could not check {ICEBERG_NAMESPACE}.agg_site_info: "
        "catalog down" in out
    )
    assert out.count("Could not check") == 1
    assert f"{ICEBERG_NAMESPACE}.agg_site_info dropped" not in out


def test_verify_empty_and_nonempty_tables_keep_their_lines(capsys):
    out = _verify(capsys, {"sample_site_info": 0, "batch_genotypes": 3})
    assert f"  OK: {ICEBERG_NAMESPACE}.sample_site_info is empty" in out
    assert (
        f"  WARNING: {ICEBERG_NAMESPACE}.batch_genotypes still has data "
        "(3 rows in first batch)!" in out
    )
    assert "Could not check" not in out
