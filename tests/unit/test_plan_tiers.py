# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for aggregate._plan_tiers — the reduce-tree shape — and the
per-chromosome planner (_plan_tiers_by_chrom).

tiers[k] = list of groups for the reduce that reads in_tier=k and writes k+1.
Tier 0 partitions the batch_ids; each higher tier groups the previous tier's
group_ids; the tree stops at a single root group.
"""

import pytest

from mosaic.aggregate import _batch_covers, _plan_tiers, _plan_tiers_by_chrom

pytestmark = pytest.mark.unit


def _group_counts(tiers):
    return [len(groups) for groups in tiers]


def test_degenerate_single_pass_when_batches_le_fan_in():
    # 3 batches, fan_in 100 -> one tier, one group (the whole batch set).
    tiers = _plan_tiers([1, 2, 3], fan_in=100)
    assert len(tiers) == 1
    assert tiers == [[[1, 2, 3]]]


def test_exact_fan_in_is_single_group():
    tiers = _plan_tiers([1, 2], fan_in=2)
    assert tiers == [[[1, 2]]]


def test_multi_tier_shape():
    # 5 batches, fan_in 2 -> tier group counts 3 -> 2 -> 1.
    tiers = _plan_tiers([1, 2, 3, 4, 5], fan_in=2)
    assert _group_counts(tiers) == [3, 2, 1]
    # Tier 0 partitions the actual batch_ids...
    assert tiers[0] == [[1, 2], [3, 4], [5]]
    # ...higher tiers group the previous tier's group_ids (0..n-1).
    assert tiers[1] == [[0, 1], [2]]
    assert tiers[2] == [[0, 1]]
    # Root tier is a single group.
    assert len(tiers[-1]) == 1


def test_fan_in_one_is_clamped_and_terminates():
    # fan_in=1 would never converge; _plan_tiers clamps to >=2.
    tiers = _plan_tiers([1, 2, 3, 4], fan_in=1)
    assert len(tiers[-1]) == 1
    assert _group_counts(tiers) == [2, 1]  # same as fan_in=2


def test_single_batch():
    assert _plan_tiers([7], fan_in=2) == [[[7]]]


# ── per-chromosome planner ───────────────────────────────────────────────────

CONTIGS = {1: "chr2", 2: "chr20", 3: "chr21", 4: "chr22"}


@pytest.mark.parametrize(
    "region,chromosome,covers",
    [
        (None, "chr21", True),  # whole-genome batch covers every contig
        ("chr21", "chr21", True),
        ("chr21:1-15000000", "chr21", True),
        ("chr20", "chr21", False),
        ("chr21", "chr2", False),  # no prefix match: chr2 != chr21
        ("chr2:1-100", "chr21", False),
    ],
)
def test_batch_covers(region, chromosome, covers):
    assert _batch_covers(region, chromosome) is covers


def test_per_chrom_plan_mixed_regions():
    batches = [
        (1, None),  # whole genome
        (2, "chr21"),
        (3, "chr21:1-15000000"),
        (4, "chr20"),
    ]
    plan = _plan_tiers_by_chrom(batches, CONTIGS, fan_in=2)
    # chr2 / chr22: only the whole-genome batch.
    assert plan[1] == [[[1]]]
    assert plan[4] == [[[1]]]
    # chr20: the whole-genome batch + its own.
    assert plan[2] == [[[1, 4]]]
    # chr21: three batches at fan_in 2 -> two tiers, as _plan_tiers would.
    assert plan[3] == _plan_tiers([1, 2, 3], fan_in=2)
    assert _group_counts(plan[3]) == [2, 1]


def test_per_chrom_plan_chromosome_without_batches_is_empty():
    batches = [(5, "chr21:10000000-10500000"), (6, "chr21")]
    plan = _plan_tiers_by_chrom(batches, CONTIGS, fan_in=100)
    assert plan[3] == [[[5, 6]]]
    assert plan[1] == [] and plan[2] == [] and plan[4] == []


def test_per_chrom_plan_resume_fixture_matches_global_plan():
    """The resume tests' fixture (3 chr21 batches, fan_in 2, init chr20+chr21)
    plans exactly as the old global plan for chr21 and has no chr20 units."""
    batches = [
        (1, "chr21:10000000-10500000"),
        (2, "chr21:20000000-20500000"),
        (3, "chr21:25000000-25500000"),
    ]
    plan = _plan_tiers_by_chrom(batches, {1: "chr20", 2: "chr21"}, fan_in=2)
    assert plan[2] == _plan_tiers([1, 2, 3], fan_in=2)
    assert plan[1] == []


def test_per_chrom_plan_collapses_per_chromosome_fanout():
    """The July shape: 17 batches per chromosome x 25 chromosomes at fan_in 100
    is one tier per chromosome (the global plan over 425 batches had two)."""
    names = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY", "chrM"]
    contigs = {i + 1: n for i, n in enumerate(names)}
    batches = []
    bid = 0
    for _ in range(17):
        for n in names:
            bid += 1
            batches.append((bid, n))
    assert len(_plan_tiers([b for b, _ in batches], 100)) == 2
    plan = _plan_tiers_by_chrom(batches, contigs, fan_in=100)
    assert all(len(t) == 1 and len(t[0][0]) == 17 for t in plan.values())
