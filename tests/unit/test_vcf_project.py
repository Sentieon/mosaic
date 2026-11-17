# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for ``tests/vcf_project.py``, the ALT cap's and the sample
subset's contracts as projections of the uncapped VCF text (one PL rule
for both: a PL is shifted only when its source held a 0), on crafted records
with hand-derived expectations."""

import pytest

import vcf_project as P

pytestmark = pytest.mark.unit

HEAD = [
    "##fileformat=VCFv4.2",
    '##INFO=<ID=AC,Number=A,Type=Integer,Description="x">',
    "##contig=<ID=chr21,length=1000>",
    "##contig=<ID=chr22,length=1000>",
]


def _vcf(samples, *records, head=HEAD):
    cols = "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO"
    if samples:
        cols += "\tFORMAT\t" + "\t".join(samples)
    return "\n".join(list(head) + [cols] + list(records)) + "\n"


def _rec(pos, ref, alt, info, fmt, *calls, qual="50.00"):
    f = ["chr21", str(pos), ".", ref, alt, qual, ".", info]
    if fmt:
        f += [fmt, *calls]
    return "\t".join(f)


def _records(text):
    return [x for x in text.split("\n") if x and not x.startswith("#")]


def _header(text):
    return [x for x in text.split("\n") if x.startswith("#")]


CAP_LINE_2 = "##mosaic_max_alt_alleles=2"

# REF A, ALTs C, G, T, *: S1 1/2, S2 3/3, S3 0/2, S4 4/4 -> AC 1, 2, 2, 2.
FOUR = _rec(
    100,
    "A",
    "C,G,T,*",
    "AC=1,2,2,2;AF=0.125,0.25,0.25,0.25;AN=8;DP=40;QD=1.25",
    "GT:DP",
    "1/2:10",
    "3/3:11",
    "0/2:12",
    "4/4:13",
)


def test_zero_cap_is_the_identity_header_included():
    text = _vcf(["S1", "S2", "S3", "S4"], FOUR)
    out, counts = P.project(text, 0)
    assert out == text
    assert set(counts.values()) == {0}


def test_negative_cap_refused():
    with pytest.raises(ValueError):
        P.project(_vcf([]), -1)


def test_at_or_under_the_cap_untouched_but_the_header_line():
    text = _vcf(["S1", "S2", "S3", "S4"], FOUR)
    for n in (4, 5, 10):
        out, counts = P.project(text, n)
        assert _records(out) == [FOUR]
        assert counts["records_capped"] == 0
        assert P.MAX_ALT_ALLELES_HEADER.format(n) in _header(out)


def test_header_line_before_the_first_contig_line_or_chrom():
    out, _ = P.project(_vcf(["S1"]), 2)
    h = _header(out)
    assert h[:3] == HEAD[:2] + [CAP_LINE_2]
    assert h[3:5] == HEAD[2:]
    assert h[5].startswith("#CHROM")
    # No ##contig line: right before #CHROM.
    out, _ = P.project(_vcf(["S1"], head=HEAD[:2]), 2)
    assert _header(out)[:3] == HEAD[:2] + [CAP_LINE_2]
    assert _header(out)[3].startswith("#CHROM")
    assert P.MAX_ALT_ALLELES_HEADER == "##mosaic_max_alt_alleles={}"


def test_ranking_tie_to_the_earlier_alt_and_star_dropped():
    # AC: C 1, G 2, T 2, * 2. N = 2: G and T (the tie among G, T, * goes
    # to the earlier ALTs); C and * dropped.
    out, counts = P.project(_vcf(["S1", "S2", "S3", "S4"], FOUR), 2)
    (rec,) = _records(out)
    f = rec.split("\t")
    assert f[4] == "G,T"
    # S1 1/2 (C dropped) ./.; S2 3/3 -> 2/2; S3 0/2 -> 0/1; S4 4/4 (*
    # dropped) ./.; DP unchanged.
    assert f[9:] == ["./.:10", "2/2:11", "0/1:12", "./.:13"]
    # AC G 1, T 2; AN 4; DP / QD / QUAL unchanged, in order.
    assert f[7] == "AC=1,2;AF=0.25,0.5;AN=4;DP=40;QD=1.25"
    assert f[5] == "50.00"
    assert counts == {
        "records_capped": 1,
        "genotypes_no_called": 2,  # S1, S4
        "dropped_alt_observations": 3,  # C 1 + * 2
        "kept_alt_observations_lost": 1,  # S1's G
    }


def test_star_competes_by_its_count_and_is_kept():
    out, counts = P.project(_vcf(["S1", "S2", "S3", "S4"], FOUR), 3)
    f = _records(out)[0].split("\t")
    assert f[4] == "G,T,*"
    assert f[9:] == ["./.:10", "2/2:11", "0/1:12", "3/3:13"]
    assert f[7] == "AC=1,2,2;AF=0.167,0.333,0.333;AN=6;DP=40;QD=1.25"
    assert counts == {
        "records_capped": 1,
        "genotypes_no_called": 1,
        "dropped_alt_observations": 1,
        "kept_alt_observations_lost": 1,
    }


# REF A, ALTs C, G, T: S1 0/1, S2 2/2, S3 0/3 -> AC 1, 2, 1; N = 2 keeps C
# (tied with T, earlier) and G. PL over 4 alleles: (0,0) (0,1) (1,1) (0,2)
# (1,2) (2,2) (0,3) (1,3) (2,3) (3,3); the kept genotypes are the first
# six.
THREE = _rec(
    200,
    "A",
    "C,G,T",
    "AC=1,2,1;AF=0.167,0.333,0.167;AN=6",
    "GT:AD:DP:F1R2:F2R1:PL",
    "0/1:3,4,5,6:18:1,2,3,4:2,2,2,0:10,0,20,30,40,50,60,70,80,90",
    "2/2:.:7:.:.:.",
    "0/3:9,0,1,8:18:4,0,0,4:5,0,1,4:5,7,9,11,13,15,0,17,19,21",
)


def test_per_allele_fields_pl_selection_and_subtraction():
    out, counts = P.project(_vcf(["S1", "S2", "S3"], THREE), 2)
    f = _records(out)[0].split("\t")
    assert f[4] == "C,G"
    assert f[8] == "GT:AD:DP:F1R2:F2R1:PL"
    # S1: AD / F1R2 / F2R1 REF + C + G; its kept PL holds a 0: unchanged.
    assert f[9] == "0/1:3,4,5:18:1,2,3:2,2,2:10,0,20,30,40,50"
    # S2: the `.` values stay `.`.
    assert f[10] == "2/2:.:7:.:.:."
    # S3: T dropped, ./.; its kept PL all > 0: minus 5.
    assert f[11] == "./.:9,0,1:18:4,0,0:5,0,1:0,2,4,6,8,10"
    assert f[7] == "AC=1,2;AF=0.25,0.5;AN=4"
    assert counts == {
        "records_capped": 1,
        "genotypes_no_called": 1,
        "dropped_alt_observations": 1,
        "kept_alt_observations_lost": 0,
    }


def test_an_zero_after_the_cap():
    rec = _rec(300, "A", "C,G", "AC=1,1;AF=0.5,0.5;AN=2", "GT", "1/2")
    out, counts = P.project(_vcf(["S1"], rec), 1)
    f = _records(out)[0].split("\t")
    assert (f[4], f[7], f[9]) == ("C", "AC=0;AF=0.0;AN=0", "./.")
    assert counts["kept_alt_observations_lost"] == 1


def test_no_call_and_ref_genotypes_kept():
    rec = _rec(
        400, "A", "C,G", "AC=0,1;AF=0.0,0.25;AN=4", "GT", "./.", "0/0", "0/2"
    )
    out, counts = P.project(_vcf(["S1", "S2", "S3"], rec), 1)
    f = _records(out)[0].split("\t")
    assert f[4] == "G"
    assert f[9:] == ["./.", "0/0", "0/1"]
    assert f[7] == "AC=1;AF=0.25;AN=4"
    # The ./. was not called before: not counted.
    assert counts["genotypes_no_called"] == 0
    assert counts["dropped_alt_observations"] == 0


def test_alt_dot_record_untouched():
    rec = _rec(500, "A", ".", "AN=2", "GT", "0/0")
    out, counts = P.project(_vcf(["S1"], rec), 1)
    assert _records(out) == [rec]
    assert counts["records_capped"] == 0


def test_ac_disagreeing_with_the_gts_is_an_error():
    rec = _rec(600, "A", "C,G", "AC=2,0;AF=0.5,0.0;AN=4", "GT", "0/1", "0/2")
    with pytest.raises(AssertionError, match="INFO/AC"):
        P.project(_vcf(["S1", "S2"], rec), 1)


def test_unknown_format_key_refused():
    rec = _rec(700, "A", "C,G", "AC=1,0;AF=0.5,0.0;AN=2", "GT:GQ", "0/1:5")
    with pytest.raises(ValueError, match="GQ"):
        P.project(_vcf(["S1"], rec), 1)


def test_counts_sum_over_records_and_others_unchanged():
    small = _rec(50, "A", "C", "AC=1;AF=0.5;AN=2", "GT", "0/1")
    other = _rec(60, "A", "C,G", "AC=1,0;AF=0.5,0.0;AN=2", "GT", "0/1")
    text = _vcf(["S1", "S2", "S3"], small, THREE, other)
    out, counts = P.project(text, 2)
    recs = _records(out)
    assert recs[0] == small and recs[2] == other
    assert counts["records_capped"] == 1
    out3, c3 = P.project(text, 3)
    assert out3 == text.replace(
        HEAD[2], "##mosaic_max_alt_alleles=3\n" + HEAD[2]
    )
    assert c3["records_capped"] == 0


def test_helpers():
    assert P.kept_alts([1, 2, 2, 2], 2) == [1, 2]
    assert P.kept_alts([0, 0, 5], 1) == [2]
    assert P.kept_alts([3, 3], 1) == [0]
    assert [P.gt_index(a, b) for a, b in ((0, 0), (0, 1), (1, 1), (2, 0))] == [
        0,
        1,
        2,
        3,
    ]
    assert P.af_text(1, 3) == "0.333"
    assert P.af_text(0, 0) == "0.0"
    assert P.af_text(2, 2) == "1.0"
    assert P.gt_alleles("./.") == [] and P.gt_alleles("0|2") == [0, 2]


# ── K4: the one PL rule (decision 1) ────────────────────────────────────────


def test_pl_rule_three_cases():
    # A source with a 0 whose kept entries are all > 0: shifted.
    assert P.project_pl([5, 0, 9, 7], [5, 9, 7]) == [0, 4, 2]
    # A source with a 0 that stays: unchanged.
    assert P.project_pl([0, 3, 9, 7], [0, 9]) == [0, 9]
    # An all-positive source (a ref-block fill): never shifted.
    assert P.project_pl([5, 15, 30, 15, 30, 30], [5, 15, 30]) == [5, 15, 30]


def test_cap_leaves_an_all_positive_fill_unshifted():
    # S1 0/1, S2 0/2 (AC 1, 1, 0 -> N = 2 keeps C, G); S3 a fill whose PL
    # holds no 0: its kept entries stay as they are (K3 shifted them).
    rec = _rec(
        800,
        "A",
        "C,G,T",
        "AC=1,1,0;AF=0.167,0.167,0.0;AN=6",
        "GT:PL",
        "0/1:9,0,9,9,9,9,9,9,9,9",
        "0/2:9,9,9,0,9,9,9,9,9,9",
        "0/0:5,15,30,15,30,30,15,30,30,30",
    )
    out, _ = P.project(_vcf(["S1", "S2", "S3"], rec), 2)
    f = _records(out)[0].split("\t")
    assert f[11] == "0/0:5,15,30,15,30,30"


# ── K4: project_samples ─────────────────────────────────────────────────────

SIX = ["S1", "S2", "S3", "S4"]

# REF A, ALTs C, *: S1 0/1, S2 0/2 (a star call), S3 0/0 (a fill), S4 ./.
STAR = _rec(
    900,
    "A",
    "C,*",
    "AC=1,1;AF=0.167,0.167;AN=6;DP=40;QD=1.25",
    "GT:AD:DP:F1R2:F2R1:PL",
    "0/1:3,4,0:7:1,2,0:2,2,0:20,0,30,25,35,45",
    "0/2:5,0,6:11:2,0,3:3,0,3:40,45,90,0,50,7",
    "0/0:9,1,1:10:.:.:4,12,30,12,30,30",
    "./.:.:.:.:.:.",
)


def test_subset_columns_in_header_order():
    text = _vcf(SIX, STAR)
    out = P.project_samples(text, ["S3", "S1"])
    assert _header(out)[-1].split("\t")[9:] == ["S1", "S3"]
    assert _header(out)[:-1] == _header(text)[:-1]


def test_subset_star_dropped_when_no_column_names_it():
    out = P.project_samples(_vcf(SIX, STAR), ["S1", "S3"])
    f = _records(out)[0].split("\t")
    assert f[4] == "C"
    # S1: AD / F1R2 / F2R1 lose *'s entry; PL its genotypes (20,0,30
    # holds a 0: unchanged). S3: the fill's PL holds no 0: 4,12,30 as is.
    # F1R2 / F2R1 kept (S1 carries them; S3 writes `.`).
    assert f[8] == "GT:AD:DP:F1R2:F2R1:PL"
    assert f[9:] == ["0/1:3,4:7:1,2:2,2:20,0,30", "0/0:9,1:10:.:.:4,12,30"]
    assert f[7] == "AC=1;AF=0.25;AN=4;DP=40;QD=1.25"
    assert f[5] == "50.00"


def test_subset_star_kept_when_a_column_names_it():
    out = P.project_samples(_vcf(SIX, STAR), ["S2", "S4"])
    f = _records(out)[0].split("\t")
    assert f[4] == "C,*"
    assert f[9:] == [
        "0/2:5,0,6:11:2,0,3:3,0,3:40,45,90,0,50,7",
        "./.:.:.:.:.:.",
    ]
    assert f[7] == "AC=0,1;AF=0.0,0.5;AN=2;DP=40;QD=1.25"


def test_subset_star_drop_shifts_a_pl_that_held_its_0_on_star():
    # S2 named *; in a subset without it, another sample whose PL had its
    # 0 at a * genotype loses it and is shifted.
    rec = _rec(
        950,
        "A",
        "C,*",
        "AC=0,2;AF=0.0,0.5;AN=4",
        "GT:PL",
        "0/2:40,45,90,0,50,7",
        "./.:30,35,40,10,20,0",
    )
    out = P.project_samples(_vcf(["S1", "S2"], rec), ["S2"])
    f = _records(out)[0].split("\t")
    assert (f[4], f[9]) == ("C", "./.:0,5,10")
    assert f[7] == "AC=0;AF=0.0;AN=0"


def test_subset_star_only_record():
    rec = _rec(
        960,
        "A",
        "*",
        "AC=1;AF=0.25;AN=4",
        "GT:AD:DP:PL",
        "0/1:4,3:7:30,0,40",
        "0/0:9,0:9:6,20,30",
    )
    out = P.project_samples(_vcf(["S1", "S2"], rec), ["S2"])
    f = _records(out)[0].split("\t")
    assert f[4] == "."
    # No AC or AF key on a record without ALTs (K5).
    assert f[7] == "AN=2"
    assert f[8:] == ["GT:AD:DP:PL", "0/0:9:9:6"]


def test_subset_format_key_dropped():
    out = P.project_samples(_vcf(SIX, STAR), ["S3", "S4"])
    f = _records(out)[0].split("\t")
    # F1R2 / F2R1: `.` in both columns, dropped; * named by none: dropped.
    assert f[8] == "GT:AD:DP:PL"
    assert f[9:] == ["0/0:9,1:10:4,12,30", "./.:.:.:."]
    out = P.project_samples(_vcf(SIX, STAR), ["S4"])
    f = _records(out)[0].split("\t")
    assert f[8:] == ["GT", "./."]
    assert f[7] == "AC=0;AF=0.0;AN=0;DP=40;QD=1.25"


def test_subset_of_every_sample_is_the_identity():
    other = _rec(
        50, "A", "C", "AC=1;AF=0.167;AN=6", "GT", "0/1", "0/0", "0/0", "./."
    )
    text = _vcf(SIX, other, STAR)
    assert P.project_samples(text, SIX) == text
    assert P.project_samples(text, list(reversed(SIX))) == text


def test_subset_then_cap():
    # S1 1/2, S2 3/3, S3 0/2, S4 4/4 over C, G, T, *; the subset S1, S3:
    # * leaves (named by S4 only), AC C 1, G 2, T 0; then N = 1 keeps G.
    sub = P.project_samples(_vcf(SIX, FOUR), ["S1", "S3"])
    f = _records(sub)[0].split("\t")
    assert (f[4], f[7]) == (
        "C,G,T",
        "AC=1,2,0;AF=0.25,0.5,0.0;AN=4;DP=40;QD=1.25",
    )
    out, counts = P.project(sub, 1)
    f = _records(out)[0].split("\t")
    assert f[4] == "G"
    assert f[9:] == ["./.:10", "0/1:12"]
    assert f[7] == "AC=1;AF=0.5;AN=2;DP=40;QD=1.25"
    assert counts == {
        "records_capped": 1,
        "genotypes_no_called": 1,
        "dropped_alt_observations": 1,
        "kept_alt_observations_lost": 1,
    }


def test_subset_refusals():
    text = _vcf(SIX, STAR)
    for names in ([], ["S1", "S1"], ["S9"]):
        with pytest.raises(ValueError):
            P.project_samples(text, names)
    bad = _rec(970, "A", "*,C", "AC=0,0;AF=0.0,0.0;AN=0", "GT", "./.")
    with pytest.raises(ValueError, match="last ALT"):
        P.project_samples(_vcf(["S1"], bad), ["S1"])


# ── K5: project_trim and project_min_ac ────────────────────────────────────

TRIM_LINE = "##mosaic_trim_alt_alleles=true"

# REF A, ALTs C, G, T: S1 0/1, S2 3/3, S3 ./. (its PL's 0 at 0/G), S4 a
# fill (no 0) -> AC 1, 0, 2: G, in the middle, leaves. PL over 4 alleles:
# (0,0) (0,1) (1,1) (0,2) (1,2) (2,2) (0,3) (1,3) (2,3) (3,3); the kept
# genotypes (over A, C, T) are indices 0, 1, 2, 6, 7, 9.
MID = _rec(
    1000,
    "A",
    "C,G,T",
    "AC=1,0,2;AF=0.167,0.0,0.333;AN=6;DP=40;QD=1.25",
    "GT:AD:DP:F1R2:F2R1:PL",
    "0/1:5,4,1,0:10:2,2,0,0:3,2,1,0:30,0,40,35,45,50,60,70,80,90",
    "3/3:0,0,0,9:9:.:.:90,80,70,60,50,40,20,10,5,0",
    "./.:3,0,4,0:7:.:.:20,25,30,0,10,15,40,45,50,55",
    "0/0:7,1,1,1:8:.:.:5,15,30,15,30,30,15,30,30,30",
)


def test_trim_an_alt_in_the_middle():
    text = _vcf(SIX, MID)
    out, counts = P.project_trim(text)
    f = _records(out)[0].split("\t")
    assert f[4] == "C,T"
    # AC / AF lose G's entry; AN, DP, QD, QUAL unchanged, in order.
    assert f[7] == "AC=1,2;AF=0.167,0.333;AN=6;DP=40;QD=1.25"
    assert f[5] == "50.00"
    assert f[8] == "GT:AD:DP:F1R2:F2R1:PL"
    assert f[9:] == [
        # The kept PL holds a 0: unchanged.
        "0/1:5,4,0:10:2,2,0:3,2,0:30,0,40,60,70,90",
        # T re-indexed 3 -> 2; `.` values stay `.`.
        "2/2:0,0,9:9:.:.:90,80,70,20,10,0",
        # The source's 0 was at a dropped genotype: the kept entries less 20.
        "./.:3,0,0:7:.:.:0,5,10,20,25,35",
        # A fill with no 0: as it is.
        "0/0:7,1,1:8:.:.:5,15,30,15,30,30",
    ]
    assert counts == {"records_trimmed": 1, "alts_dropped": 1}


def test_trim_several_alts_and_the_last():
    # REF A, ALTs C, G, T, *: only G carried (S1 2/2, S2 0/2).
    rec = _rec(
        1100,
        "A",
        "C,G,T,*",
        "AC=0,3,0,0;AF=0.0,0.75,0.0,0.0;AN=4",
        "GT:AD",
        "2/2:0,1,8,0,0",
        "0/2:4,0,3,1,0",
    )
    out, counts = P.project_trim(_vcf(["S1", "S2"], rec))
    f = _records(out)[0].split("\t")
    assert (f[4], f[7]) == ("G", "AC=3;AF=0.75;AN=4")
    assert f[9:] == ["1/1:0,8", "0/1:4,3"]
    assert counts == {"records_trimmed": 1, "alts_dropped": 3}


def test_trim_every_alt_gives_alt_dot():
    rec = _rec(
        1200,
        "A",
        "C,G",
        "AC=0,0;AF=0.0,0.0;AN=2;DP=9",
        "GT:AD:DP:PL",
        "0/0:6,1,0:7:0,10,20,30,40,50",
        # Its PL's 0 is at a dropped genotype: [10] less 10.
        "./.:.:.:10,0,20,30,40,50",
        # A fill with no 0: [5] as it is.
        "./.:4,0,0:4:5,15,30,15,30,30",
    )
    out, counts = P.project_trim(_vcf(["S1", "S2", "S3"], rec))
    f = _records(out)[0].split("\t")
    # No AC or AF key; AN and DP keep their places.
    assert (f[4], f[7]) == (".", "AN=2;DP=9")
    assert f[9:] == ["0/0:6:7:0", "./.:.:.:0", "./.:4:4:5"]
    assert counts == {"records_trimmed": 1, "alts_dropped": 2}


def test_trim_leaves_a_record_with_nothing_to_trim_and_alt_dot():
    full = _rec(50, "A", "C", "AC=1;AF=0.5;AN=2", "GT", "0/1")
    dot = _rec(60, "A", ".", "AN=2", "GT", "0/0")
    text = _vcf(["S1"], full, dot)
    out, counts = P.project_trim(text)
    assert _records(out) == [full, dot]
    assert counts == {"records_trimmed": 0, "alts_dropped": 0}
    assert out == text.replace(HEAD[2], TRIM_LINE + "\n" + HEAD[2])


def test_trim_refuses_an_ac_disagreeing_with_the_gts():
    rec = _rec(1300, "A", "C,G", "AC=1,1;AF=0.5,0.5;AN=2", "GT", "0/1")
    with pytest.raises(AssertionError, match="INFO/AC"):
        P.project_trim(_vcf(["S1"], rec))


# AC 5, 0, 0, 3 over C, G, T, TA: the cap alone at 3 keeps C, G (the tie
# among the AC-0 ALTs to the earlier) and TA; the trim first leaves C, TA,
# under the cap.
AC5003 = _rec(
    1400,
    "A",
    "C,G,T,TA",
    "AC=5,0,0,3;AF=0.5,0.0,0.0,0.3;AN=10",
    "GT",
    "1/1",
    "1/4",
    "1/1",
    "0/4",
    "4/0",
)


def test_trim_then_cap_differs_from_the_cap_alone():
    names = ["S1", "S2", "S3", "S4", "S5"]
    text = _vcf(names, AC5003)
    capped, cc = P.project(text, 3)
    f = _records(capped)[0].split("\t")
    assert (f[4], f[9:]) == ("C,G,TA", ["1/1", "1/3", "1/1", "0/3", "3/0"])
    assert cc["records_capped"] == 1
    assert cc["genotypes_no_called"] == cc["dropped_alt_observations"] == 0
    trimmed, tc = P.project_trim(text)
    both, bc = P.project(trimmed, 3)
    f = _records(both)[0].split("\t")
    assert (f[4], f[7]) == ("C,TA", "AC=5,3;AF=0.5,0.3;AN=10")
    assert f[9:] == ["1/1", "1/2", "1/1", "0/2", "2/0"]
    assert tc == {"records_trimmed": 1, "alts_dropped": 2}
    assert bc["records_capped"] == 0
    # The trim then a cap of 1: ranked on the trimmed record.
    one, oc = P.project(trimmed, 1)
    f = _records(one)[0].split("\t")
    assert (f[4], f[9:]) == ("C", ["1/1", "./.", "1/1", "./.", "./."])
    assert oc == {
        "records_capped": 1,
        "genotypes_no_called": 3,
        "dropped_alt_observations": 3,
        "kept_alt_observations_lost": 1,
    }


def test_min_ac_zero_one_and_above():
    zero = _rec(10, "A", "C", "AC=0;AF=0.0;AN=4", "GT", "0/0", "./.")
    one = _rec(20, "A", "C", "AC=1;AF=0.25;AN=4", "GT", "0/1", "0/0")
    three = _rec(30, "A", "C,G", "AC=2,1;AF=0.5,0.25;AN=4", "GT", "1/1", "0/2")
    dot = _rec(40, "A", ".", "AN=4", "GT", "0/0", "0/0")
    text = _vcf(["S1", "S2"], zero, one, three, dot)
    out, c = P.project_min_ac(text, 0)
    assert (out, c) == (text, {"records_left_out": 0})
    out, c = P.project_min_ac(text, 1)
    assert _records(out) == [one, three]
    assert c == {"records_left_out": 2}
    assert P.MIN_AC_HEADER.format(1) in _header(out)
    out, c = P.project_min_ac(text, 2)
    assert _records(out) == [three]
    assert c == {"records_left_out": 3}
    out, c = P.project_min_ac(text, 4)
    assert _records(out) == [] and c == {"records_left_out": 4}
    with pytest.raises(ValueError):
        P.project_min_ac(text, -1)
    assert [P.alt_observations(r) for r in (zero, one, three, dot)] == [
        0,
        1,
        3,
        0,
    ]


def test_min_ac_counts_the_written_genotypes_after_the_cap():
    # S1 1/2 is the only carrier (AC 1, 1): N = 1 keeps C, no-calls S1, so
    # the capped record has no ALT observation and --min-ac 1 leaves it
    # out; uncapped it stays.
    rec = _rec(
        1500, "A", "C,G", "AC=1,1;AF=0.25,0.25;AN=4", "GT", "1/2", "0/0"
    )
    text = _vcf(["S1", "S2"], rec)
    assert _records(P.project_min_ac(text, 1)[0]) == [rec]
    capped, _ = P.project(text, 1)
    out, c = P.project_min_ac(capped, 1)
    assert _records(out) == [] and c == {"records_left_out": 1}


def test_header_lines_order_trim_cap_min_ac():
    text = _vcf(["S1", "S2", "S3", "S4"], FOUR)
    cap = "##mosaic_max_alt_alleles=2"
    mac = "##mosaic_min_ac=1"
    assert P.TRIM_ALT_ALLELES_HEADER == TRIM_LINE
    assert P.MIN_AC_HEADER.format(1) == mac
    for steps, lines in (
        (("trim",), [TRIM_LINE]),
        (("min",), [mac]),
        (("trim", "cap"), [TRIM_LINE, cap]),
        (("trim", "min"), [TRIM_LINE, mac]),
        (("cap", "min"), [cap, mac]),
        (("trim", "cap", "min"), [TRIM_LINE, cap, mac]),
    ):
        out = text
        if "trim" in steps:
            out = P.project_trim(out)[0]
        if "cap" in steps:
            out = P.project(out, 2)[0]
        if "min" in steps:
            out = P.project_min_ac(out, 1)[0]
        h = _header(out)
        assert h[:2] == HEAD[:2], steps
        assert h[2 : 2 + len(lines)] == lines, steps
        assert h[2 + len(lines) :][:2] == HEAD[2:], steps


def test_trim_counts_sum_over_records():
    two = _rec(
        70,
        "A",
        "C,G",
        "AC=0,1;AF=0.0,0.125;AN=8",
        "GT",
        *(["0/2", "0/0", "0/0", "./."]),
    )
    text = _vcf(SIX, MID, two, STAR)
    _, counts = P.project_trim(text)
    assert counts == {"records_trimmed": 2, "alts_dropped": 2}


def test_full_chain_with_a_subset():
    # The subset S1, S3 of FOUR (C, G, T, *): * leaves (S4 only), AC 1, 2,
    # 0; the trim drops T; the cap of 1 keeps G, no-calling S1 (1/2);
    # --min-ac 2 leaves the record out (one G observation left), 1 keeps
    # it.
    sub = P.project_samples(_vcf(SIX, FOUR), ["S1", "S3"])
    trimmed, tc = P.project_trim(sub)
    f = _records(trimmed)[0].split("\t")
    assert (f[4], f[9:]) == ("C,G", ["1/2:10", "0/2:12"])
    assert f[7] == "AC=1,2;AF=0.25,0.5;AN=4;DP=40;QD=1.25"
    assert tc == {"records_trimmed": 1, "alts_dropped": 1}
    capped, cc = P.project(trimmed, 1)
    f = _records(capped)[0].split("\t")
    assert (f[4], f[9:]) == ("G", ["./.:10", "0/1:12"])
    assert cc["records_capped"] == 1
    out, mc = P.project_min_ac(capped, 2)
    assert _records(out) == [] and mc == {"records_left_out": 1}
    out, mc = P.project_min_ac(capped, 1)
    assert _records(out) == [_records(capped)[0]]
    assert mc == {"records_left_out": 0}
    h = _header(out)
    assert h[2:5] == [
        TRIM_LINE,
        "##mosaic_max_alt_alleles=1",
        "##mosaic_min_ac=1",
    ]


def test_a_record_without_alts_never_carries_ac_or_af():
    # A source record without ALTs (no AC / AF): every projection leaves it
    # without them; the subset recomputes AN only.
    dot = _rec(1600, "A", ".", "AN=4;DP=9", "GT:AD", "0/0:5", "./.:.")
    text = _vcf(["S1", "S2"], dot)
    assert _records(P.project_trim(text)[0]) == [dot]
    assert _records(P.project(text, 1)[0]) == [dot]
    sub = _records(P.project_samples(text, ["S2"]))[0].split("\t")
    assert (sub[4], sub[7]) == (".", "AN=0;DP=9")
    # The subset leaving a *-only record: AC and AF go, DP and QD stay in
    # their places.
    star = _rec(
        1700, "A", "*", "AC=1;AF=0.25;AN=4;DP=8;QD=1.0", "GT", "0/1", "0/0"
    )
    f = _records(P.project_samples(_vcf(["S1", "S2"], star), ["S2"]))[0]
    assert f.split("\t")[7] == "AN=2;DP=8;QD=1.0"
