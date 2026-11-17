# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for the pure helpers of the build-then-attach finalize.

The partition lookup (pg_inherits + partition bound), the swap transaction and
the leftover cleanup need PostgreSQL and are covered by
tests/integration/test_aggregate_swap.py. Here: the generation-table naming
shared by init, aggregate and clean, and the MOSAIC_TEST_FAIL_FINALIZE hook
format ``<metric>:<c_idx|*>:<pre_commit|post_commit>``.
"""

import pytest

from mosaic.aggregate import _finalize_marker, _maybe_crash_finalize
from mosaic.init_db import (
    AGGREGATE_PG_TABLES,
    GENERATION_TABLE_RE,
    generation_table_name,
)

pytestmark = pytest.mark.unit


# ── generation-table names ───────────────────────────────────────────────────


@pytest.mark.parametrize("table", AGGREGATE_PG_TABLES)
def test_generation_name_round_trips_through_the_regex(table):
    name = generation_table_name(table, 24, 1234)
    assert name == f"{table}_c24_r1234"
    m = GENERATION_TABLE_RE.match(name)
    assert m and m.groups() == (table, "24", "1234")


def test_init_partition_is_generation_zero():
    assert generation_table_name("site_info", 0, 0) == "site_info_c0_r0"


@pytest.mark.parametrize("table", ["allele_counts", "samples"])
def test_generation_name_rejects_other_tables(table):
    # allele_counts: the pre-catalog counts table, no longer rebuilt.
    with pytest.raises(ValueError):
        generation_table_name(table, 1, 1)


def test_alleles_is_an_aggregate_table():
    assert "alleles" in AGGREGATE_PG_TABLES
    assert "allele_counts" not in AGGREGATE_PG_TABLES


def test_legacy_allele_counts_generations_still_match():
    """`mosaic clean` drops a pre-catalog database's detached
    allele_counts generation tables through the same regex."""
    m = GENERATION_TABLE_RE.match("allele_counts_c3_r12")
    assert m and m.groups() == ("allele_counts", "3", "12")


@pytest.mark.parametrize("table", AGGREGATE_PG_TABLES)
def test_generation_name_fits_a_postgres_identifier(table):
    # NAMEDATALEN-1 = 63; the swap also adds a "_chrom" CHECK on the name,
    # and the alleles GiST is auto-named {gen}_del_range_idx.
    name = generation_table_name(table, 9999, 2**31 - 1)
    assert len(f"{name}_chrom") <= 63
    assert len(f"{name}_pkey") <= 63
    assert len(f"{name}_del_range_idx") <= 63


@pytest.mark.parametrize(
    "name",
    [
        "site_info",  # the partitioned parent
        "site_info_c1",  # no generation
        "site_info_c1_r",  # no run id
        "samples_c1_r2",  # not an aggregate table
        "ref_blocks_1",  # the pre-phase-B partition naming
        "site_info_c1_r2_chrom",  # a constraint name
        "xsite_info_c1_r2",
    ],
)
def test_generation_regex_rejects_non_generation_names(name):
    assert GENERATION_TABLE_RE.match(name) is None


def test_finalize_marker_is_per_chromosome_tier_minus_one():
    assert _finalize_marker(3) == (3, -1, 0)
    assert _finalize_marker(0) != _finalize_marker(1)


# ── MOSAIC_TEST_FAIL_FINALIZE ────────────────────────────────────────────────


def test_hook_inert_when_unset(monkeypatch):
    monkeypatch.delenv("MOSAIC_TEST_FAIL_FINALIZE", raising=False)
    _maybe_crash_finalize("site_info", 1, "pre_commit")
    _maybe_crash_finalize("site_info", 1, "post_commit")


@pytest.mark.parametrize(
    "spec", ["site_info:1:pre_commit", "site_info:*:pre_commit"]
)
def test_hook_fires_on_match(monkeypatch, spec):
    monkeypatch.setenv("MOSAIC_TEST_FAIL_FINALIZE", spec)
    with pytest.raises(RuntimeError, match="MOSAIC_TEST_FAIL_FINALIZE"):
        _maybe_crash_finalize("site_info", 1, "pre_commit")


@pytest.mark.parametrize(
    "metric,c_idx,phase",
    [
        ("ref_blocks", 1, "pre_commit"),  # other metric
        ("site_info", 2, "pre_commit"),  # other chromosome
        ("site_info", 1, "post_commit"),  # other phase
    ],
)
def test_hook_ignores_other_units(monkeypatch, metric, c_idx, phase):
    monkeypatch.setenv("MOSAIC_TEST_FAIL_FINALIZE", "site_info:1:pre_commit")
    _maybe_crash_finalize(metric, c_idx, phase)


@pytest.mark.parametrize(
    "spec",
    [
        "site_info:pre_commit",  # the phase-A two-field format
        "site_info:1:pre_commit:extra",
        "",
    ],
)
def test_hook_ignores_malformed_specs(monkeypatch, spec):
    monkeypatch.setenv("MOSAIC_TEST_FAIL_FINALIZE", spec)
    _maybe_crash_finalize("site_info", 1, "pre_commit")
