# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for ``tests/vcf_project_stream.py``: the streamed projection
chain equals ``tests/vcf_project.py``'s whole-text chain byte for byte,
whatever the chunk size, with the same counts, and its closing lines are
the driver's (``joint_call.py``'s ``trim_line``, ``cap_line`` and
``min_ac_line``). The tool exists for cohort-scale checks, where a whole
VCF text does not fit in memory."""

import gzip
import io
import os
import subprocess
import sys

import pytest

import joint_call as J
import vcf_project as P
import vcf_project_stream as S

pytestmark = pytest.mark.unit

HEAD = [
    "##fileformat=VCFv4.2",
    '##INFO=<ID=AC,Number=A,Type=Integer,Description="x">',
    "##contig=<ID=chr21,length=1000>",
    "##contig=<ID=chr22,length=1000>",
    "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\tS1\tS2\tS3\tS4",
]


def _rec(pos, alt, info, fmt, *calls):
    f = ["chr21", str(pos), ".", "A", alt, "50.00", ".", info, fmt, *calls]
    return "\t".join(f)


# Seven records: over a cap of 2 with a carrier of a dropped ALT; an AC-0
# ALT in the middle; a `*` only S4 carries; a record no sample carries an
# ALT of; a PL whose 0 sits at a dropped genotype; a singleton; a
# multi-allelic record at the cap.
RECORDS = [
    _rec(
        100,
        "C,G,T,*",
        "AC=1,2,2,2;AF=0.125,0.25,0.25,0.25;AN=8;DP=40",
        "GT:DP",
        "1/2:10",
        "3/3:11",
        "0/2:12",
        "4/4:13",
    ),
    _rec(
        110,
        "C,G,T",
        "AC=1,0,3;AF=0.125,0.0,0.375;AN=8;DP=40",
        "GT:AD:DP:PL",
        "0/1:5,4,0,0:9:30,0,40,60,70,80,90,95,96,99",
        "3/3:0,0,0,9:9:90,91,92,93,94,95,60,61,62,0",
        "0/3:4,0,0,5:9:20,50,60,70,71,72,0,55,65,80",
        "0/0:9,0,0,0:9:0,27,270,27,271,272,27,273,274,275",
    ),
    _rec(
        120,
        "C,*",
        "AC=1,2;AF=0.125,0.25;AN=8;DP=40",
        "GT:AD:DP:PL",
        "0/1:5,4,0:9:30,0,40,60,70,80",
        "0/0:9,0,0:9:0,27,270,27,271,272",
        "0/0:8,0,0:8:0,24,240,24,241,242",
        "2/2:0,0,7:7:70,71,72,21,22,0",
    ),
    _rec(
        130,
        "C",
        "AC=0;AF=0.0;AN=6;DP=30",
        "GT:DP",
        "0/0:10",
        "0/0:11",
        "./.:.",
        "0/0:9",
    ),
    _rec(
        140,
        "C,G,T",
        "AC=1,1,3;AF=0.125,0.125,0.375;AN=8;DP=40",
        "GT:PL",
        "1/2:50,60,70,61,0,71,80,81,82,83",
        "3/3:90,91,92,93,94,95,60,61,62,0",
        "0/3:20,50,60,70,71,72,0,55,65,80",
        "0/0:0,27,270,27,271,272,27,273,274,275",
    ),
    _rec(
        150,
        "G",
        "AC=1;AF=0.125;AN=8;DP=40",
        "GT:DP",
        "0/0:10",
        "0/1:11",
        "0/0:12",
        "0/0:13",
    ),
    _rec(
        160,
        "C,G",
        "AC=2,3;AF=0.25,0.375;AN=8;DP=40",
        "GT:DP",
        "1/2:10",
        "1/2:11",
        "0/2:12",
        "0/0:13",
    ),
]
TEXT = "\n".join(HEAD + RECORDS) + "\n"

# (names, trim, cap, min_ac)
CASES = [
    (None, False, 0, 0),
    (None, False, 2, 0),
    (None, True, 0, 0),
    (None, False, 0, 2),
    (None, True, 2, 1),
    (["S1", "S3"], False, 0, 0),
    (["S3", "S1"], True, 1, 1),
    (["S4"], True, 2, 1),
    (["S2"], False, 1, 3),
]


def _whole(text, names, trim, cap, min_ac):
    """The whole-text chain and its counts per option set."""
    counts = {}
    if names is not None:
        text = P.project_samples(text, names)
    if trim:
        text, counts["trim"] = P.project_trim(text)
    if cap > 0:
        text, counts["cap"] = P.project(text, cap)
    if min_ac > 0:
        text, counts["min_ac"] = P.project_min_ac(text, min_ac)
    return text, counts


def _stream(text, names, trim, cap, min_ac, chunk):
    dst = io.BytesIO()
    records, counts = S.stream(
        io.BytesIO(text.encode()), dst, names, trim, cap, min_ac, chunk
    )
    return dst.getvalue().decode(), records, counts


@pytest.mark.parametrize("case", CASES)
@pytest.mark.parametrize("chunk", [1, 2, 3, 1000])
def test_stream_equals_the_whole_text_chain(case, chunk):
    names, trim, cap, min_ac = case
    want, want_counts = _whole(TEXT, names, trim, cap, min_ac)
    got, records, counts = _stream(TEXT, names, trim, cap, min_ac, chunk)
    assert got == want
    assert counts == want_counts
    assert records == sum(
        1 for x in want.split("\n") if x and not x.startswith("#")
    )


def test_the_cases_exercise_every_step():
    """The fixture is not vacuous: the cap, the trim and --min-ac each
    change it, alone and under a subset."""
    _, c = _whole(TEXT, None, True, 2, 1)
    assert c["trim"] == {"records_trimmed": 2, "alts_dropped": 2}
    assert c["cap"]["records_capped"] == 2
    assert c["cap"]["genotypes_no_called"] > 0
    assert c["min_ac"]["records_left_out"] == 1
    _, c = _whole(TEXT, ["S4"], True, 2, 1)
    assert c["min_ac"]["records_left_out"] == 5


def test_a_header_only_input_and_an_input_without_a_final_newline():
    head = "\n".join(HEAD) + "\n"
    want, _ = _whole(head, ["S2"], True, 3, 1)
    got, records, counts = _stream(head, ["S2"], True, 3, 1, 10)
    assert (got, records) == (want, 0)
    assert counts["trim"] == {"records_trimmed": 0, "alts_dropped": 0}
    assert counts["cap"]["records_capped"] == 0
    assert counts["min_ac"] == {"records_left_out": 0}
    want, _ = _whole(TEXT, None, False, 2, 0)
    got, records, _ = _stream(TEXT.rstrip("\n"), None, False, 2, 0, 4)
    assert (got, records) == (want, len(RECORDS))


def test_refusals():
    with pytest.raises(ValueError):
        _stream(TEXT, None, False, 0, 0, 0)
    with pytest.raises(ValueError):
        _stream(RECORDS[0] + "\n", None, False, 0, 0, 5)
    with pytest.raises(ValueError):
        _stream("##fileformat=VCFv4.2\n", None, False, 0, 0, 5)
    with pytest.raises(ValueError):
        _stream(TEXT, ["nobody"], False, 0, 0, 5)


def test_closing_lines_are_the_drivers():
    _, records, counts = _stream(TEXT, None, True, 2, 1, 2)
    lines = S.closing_lines(records, 2, 1, counts)
    assert lines == [
        f"Completed! Total processed: {records} variants",
        J.trim_line(counts["trim"]),
        J.cap_line({"max_alt_alleles": 2, **counts["cap"]}),
        J.min_ac_line({"min_ac": 1, **counts["min_ac"]}),
    ]
    # No option: the one line.
    _, records, counts = _stream(TEXT, None, False, 0, 0, 2)
    assert S.closing_lines(records, 0, 0, counts) == [
        f"Completed! Total processed: {len(RECORDS)} variants"
    ]


def _repo():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def _run(argv, text):
    return subprocess.run(
        [sys.executable, os.path.join("tests", "vcf_project_stream.py")]
        + argv,
        cwd=_repo(),
        input=text.encode(),
        capture_output=True,
    )


def test_the_command(tmp_path):
    names = tmp_path / "names.txt"
    names.write_text("S3\n\nS1\n")
    proc = _run(
        [
            "--samples-file",
            str(names),
            "--trim-alt-alleles",
            "--max-alt-alleles",
            "1",
            "--min-ac",
            "1",
            "--chunk-records",
            "2",
        ],
        TEXT,
    )
    assert proc.returncode == 0, proc.stderr
    want, counts = _whole(TEXT, ["S3", "S1"], True, 1, 1)
    assert proc.stdout.decode() == want
    records = sum(1 for x in want.split("\n") if x and not x.startswith("#"))
    assert proc.stderr.decode().splitlines() == S.closing_lines(
        records, 1, 1, counts
    )
    # --samples, and the default cap of 0 (the input unchanged).
    proc = _run(["--samples", "S1,S2,S3,S4"], TEXT)
    assert proc.returncode == 0
    assert proc.stdout.decode() == P.project_samples(
        TEXT, ["S1", "S2", "S3", "S4"]
    )
    proc = _run([], TEXT)
    assert proc.stdout.decode() == TEXT
    assert proc.stderr.decode().splitlines() == [
        f"Completed! Total processed: {len(RECORDS)} variants"
    ]


def test_the_command_refuses_negative_options():
    for argv in (["--max-alt-alleles=-1"], ["--min-ac=-1"]):
        proc = _run(argv, TEXT)
        assert proc.returncode == 2
        assert proc.stdout == b""


GOLDEN = os.path.join(
    _repo(), "test_data", "golden_joint_call", "slice_chr21_10M_12M.vcf.gz"
)


@pytest.mark.skipif(
    not os.path.exists(GOLDEN), reason="no local golden copy (test_data/)"
)
def test_stream_equals_the_chain_on_the_golden_slice():
    """On real joint-call output (the six-sample slice's uncapped golden):
    the full chain under a subset, and the default cap alone."""
    with gzip.open(GOLDEN, "rb") as fh:
        text = fh.read().decode()
    header = [x for x in text.split("\n") if x.startswith("#CHROM")][0]
    samples = header.split("\t")[9:]
    for names, trim, cap, min_ac in (
        (None, False, 6, 0),
        ([samples[4], samples[1]], True, 3, 1),
    ):
        want, want_counts = _whole(text, names, trim, cap, min_ac)
        got, _, counts = _stream(text, names, trim, cap, min_ac, 97)
        assert got == want
        assert counts == want_counts
