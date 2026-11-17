# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Unit tests for joint_call.py, the driver of the Rust joint caller:

* ``--region`` parsing and resolution;
* the ``##INFO`` header line for InbreedingCoeff (the header lines the
  driver hands the binary);
* the catalog cross-check's closing lines (the binary counts; the driver
  prints): the reason strings, ``stale`` the only run reason (K1);
* the removed options (``--engine``, ``--step-size``) are argparse errors;
* K4's ``--samples`` / ``--samples-file``: the names, the header's order,
  the refusals, the data files kept by their ``sample_id`` bounds.

The calling rules these tests covered for the Python engine (AC / AN / AF,
the rounding, FS / SOR / InbreedingCoeff, the null FORMAT values, the
ref-block fill, the cross-check's counting) are ``cargo test``s of
``rust/mosaic-joint-call`` since the Python engine's removal
(``call.rs``, ``format.rs``, ``pyfmt.rs``, ``run.rs``).
"""

import argparse
import json
import os
import subprocess
import sys

import pytest

import joint_call as J
from joint_call import (
    VCF_HEADERS,
    CatalogCrossCheck,
    parse_region,
    resolve_region,
)

pytestmark = pytest.mark.unit


# ── --region ────────────────────────────────────────────────────────────────

CONTIGS = [(1, "chr20", 64_444_167), (2, "chr21", 46_709_983)]


def test_parse_region():
    assert parse_region("chr21") == ("chr21", None, None)
    assert parse_region("chr21:10000000-12000000") == (
        "chr21",
        10_000_000,
        12_000_000,
    )
    assert parse_region("chr21:5-5") == ("chr21", 5, 5)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        " chr21",
        "chr21:",
        "chr21:5",
        "chr21:5-",
        "chr21:-5",
        "chr21:0-5",
        "chr21:6-5",
        "chr21:a-b",
        ":1-5",
        "chr21:1,000-2,000",
    ],
)
def test_parse_region_refuses_malformed(bad):
    with pytest.raises(ValueError):
        parse_region(bad)
    with pytest.raises(argparse.ArgumentTypeError):
        J._region_arg(bad)


def test_resolve_region():
    assert resolve_region(parse_region("chr21"), CONTIGS) == (
        2,
        "chr21",
        46_709_983,
        1,
        46_709_983,
    )
    assert resolve_region(parse_region("chr20:10-20"), CONTIGS) == (
        1,
        "chr20",
        64_444_167,
        10,
        20,
    )
    # The end is clipped to the contig.
    assert resolve_region(parse_region("chr21:46000000-99000000"), CONTIGS)[
        3:
    ] == (46_000_000, 46_709_983)
    with pytest.raises(ValueError, match="not in the contigs table"):
        resolve_region(parse_region("chr22"), CONTIGS)
    with pytest.raises(ValueError, match="past the end"):
        resolve_region(parse_region("chr21:47000000-48000000"), CONTIGS)


def _repo():
    return os.path.dirname(os.path.dirname(os.path.dirname(__file__)))


def test_malformed_region_is_an_argparse_error():
    """Exit 2 with argparse's message, before any connection."""
    proc = subprocess.run(
        [sys.executable, "joint_call.py", "out.vcf", "--region", "chr21:9-1"],
        cwd=_repo(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2
    assert "argument --region" in proc.stderr


# ── the removed options ──────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "argv", [["--engine", "python"], ["--step-size", "5"]]
)
def test_removed_options_are_argparse_errors(argv, capsys):
    with pytest.raises(SystemExit) as e:
        J.parse_args(["out.vcf.gz", *argv])
    assert e.value.code == 2
    assert f"unrecognized arguments: {' '.join(argv)}" in (
        capsys.readouterr().err
    )
    # The same from the command line, before any connection.
    proc = subprocess.run(
        [sys.executable, "joint_call.py", "out.vcf", *argv],
        cwd=_repo(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stderr
    assert "unrecognized arguments" in proc.stderr


def test_parser_keeps_the_driver_options():
    parser, args = J.parse_args(["out.vcf.gz"])
    assert isinstance(parser, argparse.ArgumentParser)
    assert args.window == J.DEFAULT_WINDOW and args.threads >= 1
    assert not hasattr(args, "engine") and not hasattr(args, "step_size")


# ── bounded output memory ────────────────────────────────────────────────────


def test_unit_buffer_bytes_and_spill_dir_options():
    _, args = J.parse_args(["out.vcf.gz"])
    assert args.unit_buffer_bytes == 67_108_864
    assert J.DEFAULT_UNIT_BUFFER_BYTES == 64 << 20
    assert args.spill_dir is None
    _, args = J.parse_args(
        ["out.vcf.gz", "--unit-buffer-bytes", "0", "--spill-dir", "/s"]
    )
    assert (args.unit_buffer_bytes, args.spill_dir) == (0, "/s")
    _, args = J.parse_args(["out.vcf.gz", "--unit-buffer-bytes", "1"])
    assert args.unit_buffer_bytes == 1


@pytest.mark.parametrize("bad", ["-1", "x", "1.5"])
def test_negative_unit_buffer_bytes_is_exit_2(bad, capsys):
    with pytest.raises(SystemExit) as e:
        J.parse_args(["out.vcf.gz", f"--unit-buffer-bytes={bad}"])
    assert e.value.code == 2
    assert "--unit-buffer-bytes" in capsys.readouterr().err


def test_negative_unit_buffer_bytes_from_the_command_line():
    """Exit 2 before any connection."""
    proc = subprocess.run(
        [sys.executable, "joint_call.py", "out.vcf", "--unit-buffer-bytes=-5"],
        cwd=_repo(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stderr
    assert "--unit-buffer-bytes must be 0 or more" in proc.stderr


def test_spill_dir_path_rule(tmp_path):
    """The driver always names the spill directory itself (the binary
    removes it): never the user's directory."""
    tmpdir = str(tmp_path / "drv")
    # A regular output path: beside it, absolute.
    out = str(tmp_path / "out.vcf.gz")
    assert J.spill_dir_path(out, tmpdir, None, "ab12") == out + ".spill-ab12"
    rel = J.spill_dir_path("sub/out.vcf", tmpdir, None, "ab12")
    assert rel == os.path.abspath("sub/out.vcf") + ".spill-ab12"
    # /dev/stdout, '-' and any other /dev/... path: the driver's temp dir.
    for dev in ("/dev/stdout", "-", "/dev/null", "/dev/fd/3"):
        assert J.spill_dir_path(dev, tmpdir, None, "ab12") == os.path.join(
            tmpdir, "spill"
        )
    # --spill-dir DIR: a directory of its own under DIR, never DIR.
    user = str(tmp_path / "user")
    for o in (out, "/dev/stdout"):
        got = J.spill_dir_path(o, tmpdir, user, "ab12")
        assert got == os.path.join(user, "mosaic-joint-call-spill-ab12")
        assert os.path.dirname(got) == user and got != user
    # A trailing slash or a relative DIR changes nothing of that.
    got = J.spill_dir_path(out, tmpdir, user + "/", "cd34")
    assert got == os.path.join(user, "mosaic-joint-call-spill-cd34")
    got = J.spill_dir_path(out, tmpdir, "rel", "cd34")
    assert got == os.path.join(
        os.path.abspath("rel"), "mosaic-joint-call-spill-cd34"
    )
    # Nothing is created.
    assert not os.path.exists(tmpdir) and not os.path.exists(user)


# ── header ─────────────────────────────────────────────────────────────────


def test_inbreeding_coeff_header_line():
    lines = [h for h in VCF_HEADERS if h.startswith("##INFO=<ID=")]
    ids = [h[len("##INFO=<ID=") :].split(",")[0] for h in lines]
    assert "InbreedingCoeff" in ids
    line = lines[ids.index("InbreedingCoeff")]
    assert line == (
        '##INFO=<ID=InbreedingCoeff,Number=1,Type=Float,Description="'
        "Inbreeding coefficient as estimated from the genotype likelihoods "
        "per-sample when compared against the Hardy-Weinberg expectation"
        '">'
    )
    # Every INFO key the joint caller writes has a header line.
    for key in ("AC", "AF", "AN", "DP", "FS", "MQ", "QD", "SOR"):
        assert key in ids


# ── the catalog cross-check's closing lines ─────────────────────────────────


def test_cross_check_lines_and_reasons():
    """The driver's half of the old ``test_cross_check_counts_and_reasons``
    (the counting is ``run.rs::cross_check_tests``): the reason strings and
    the lines. Since K1 the run's reason is ``stale`` or none."""
    assert CatalogCrossCheck().reason is None
    assert (
        CatalogCrossCheck(stale=True).reason
        == "stale aggregates (--allow-stale)"
    )
    cc = CatalogCrossCheck()
    cc.records = 3
    cc.differing = {"other": 1, CatalogCrossCheck.UNRESOLVED: 1}
    cc.first = ["chr21:2", "chr21:3"]
    assert cc.lines() == [
        "Catalog cross-check: 2 of 3 records have catalog non-ref allele "
        "counts (alleles.count) that differ from the emitted AC",
        f"  {CatalogCrossCheck.UNRESOLVED}: 1",
        "  other: 1",
        "  first 2 differing site(s): chr21:2, chr21:3",
    ]
    clean = CatalogCrossCheck()
    clean.records = 1
    assert clean.lines() == [
        "Catalog cross-check: the catalog's non-ref allele counts "
        "(alleles.count) equal the emitted AC at every record (1 records)"
    ]
    # check() went with the Python engine (K1); no tombstoned reason.
    assert not hasattr(CatalogCrossCheck, "check")
    with pytest.raises(TypeError):
        CatalogCrossCheck(n_tombstoned=2)  # type: ignore[call-arg]


# ── the ALT cap ──────────────────────────────────────────────────────────────


def test_max_alt_alleles_option():
    _, args = J.parse_args(["out.vcf.gz"])
    assert args.max_alt_alleles == J.DEFAULT_MAX_ALT_ALLELES == 6
    for v in ("0", "1", "250"):
        _, args = J.parse_args(["out.vcf.gz", "--max-alt-alleles", v])
        assert args.max_alt_alleles == int(v)


@pytest.mark.parametrize("bad", ["-1", "x", "1.5", ""])
def test_bad_max_alt_alleles_is_exit_2(bad, capsys):
    with pytest.raises(SystemExit) as e:
        J.parse_args(["out.vcf.gz", f"--max-alt-alleles={bad}"])
    assert e.value.code == 2
    assert "--max-alt-alleles" in capsys.readouterr().err


def test_negative_max_alt_alleles_from_the_command_line():
    """Exit 2 before any connection."""
    proc = subprocess.run(
        [sys.executable, "joint_call.py", "out.vcf", "--max-alt-alleles=-1"],
        cwd=_repo(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stderr
    assert "--max-alt-alleles must be 0 or more" in proc.stderr


def test_cap_header_line_where_the_projection_puts_it():
    """The header line names the cap, after VCF_HEADERS and before the
    contig lines, only under a cap; tests/vcf_project.py (which does not
    import joint_call.py) holds the same text and inserts it at the same
    place."""
    import vcf_project

    assert J.MAX_ALT_ALLELES_HEADER == vcf_project.MAX_ALT_ALLELES_HEADER
    contigs = [("chr21", 100), ("chr22", 200)]
    plain = J.vcf_header_lines(contigs, ["S1", "S2"])
    assert J.vcf_header_lines(contigs, ["S1", "S2"], 0) == plain
    assert not any(x.startswith("##mosaic_max_alt_alleles") for x in plain)
    capped = J.vcf_header_lines(contigs, ["S1", "S2"], 6)
    k = len(VCF_HEADERS)
    assert capped[:k] == VCF_HEADERS
    assert capped[k] == "##mosaic_max_alt_alleles=6"
    assert capped[k + 1 :] == plain[k:]
    # The projection of the uncapped header is the capped header.
    text = "\n".join(plain) + "\n"
    out, _ = vcf_project.project(text, 6)
    assert out == "\n".join(capped) + "\n"
    # No contig: before #CHROM.
    assert J.vcf_header_lines([], [], 3)[k] == "##mosaic_max_alt_alleles=3"


def test_cap_closing_lines():
    cap = {
        "max_alt_alleles": 6,
        "records_capped": 311,
        "genotypes_no_called": 60,
        "dropped_alt_observations": 63,
        "kept_alt_observations_lost": 48,
    }
    assert J.cap_line(cap) == (
        "ALT cap (--max-alt-alleles 6): 311 records capped, 60 genotypes "
        "written ./., 63 ALT observations on dropped alleles, 48 kept-allele "
        "ALT observations lost with those genotypes"
    )
    # The cross-check says the capped records were not compared.
    clean = CatalogCrossCheck()
    clean.records = 132071
    clean.capped = 311
    assert clean.lines() == [
        "Catalog cross-check: the catalog's non-ref allele counts "
        "(alleles.count) equal the emitted AC at every record compared "
        "(132071 records; 311 records capped by --max-alt-alleles not "
        "compared)"
    ]
    cc = CatalogCrossCheck()
    cc.records = 10
    cc.capped = 2
    cc.differing = {"other": 1}
    cc.first = ["chr21:2"]
    assert cc.lines()[0] == (
        "Catalog cross-check: 1 of 10 records have catalog non-ref allele "
        "counts (alleles.count) that differ from the emitted AC (2 records "
        "capped by --max-alt-alleles not compared)"
    )
    # None capped: the uncapped text.
    cc.capped = 0
    assert cc.lines()[0].endswith("differ from the emitted AC")


# ── --samples / --samples-file (K4) ─────────────────────────────────────────


def test_samples_options_parse_and_are_exclusive(capsys):
    _, a = J.parse_args(["out.vcf"])
    assert (a.samples, a.samples_file) == (None, None)
    assert J.subset_names(a) is None
    _, a = J.parse_args(["out.vcf", "--samples", "B, A,,C"])
    assert J.subset_names(a) == ["B", "A", "C"]
    with pytest.raises(SystemExit) as e:
        J.parse_args(["out.vcf", "--samples", "A", "--samples-file", "f"])
    assert e.value.code == 2
    assert "not allowed with" in capsys.readouterr().err


def test_samples_file_one_name_per_line(tmp_path):
    f = tmp_path / "names.txt"
    f.write_text("HG2\n\n  HG1  \n\nHG3")
    _, a = J.parse_args(["out.vcf", "--samples-file", str(f)])
    assert J.subset_names(a) == ["HG2", "HG1", "HG3"]
    _, a = J.parse_args(["out.vcf", "--samples-file", str(tmp_path / "x")])
    with pytest.raises(OSError):
        J.subset_names(a)


def test_resolve_subset_header_order_and_refusals():
    active = ["HG1", "HG2", "HG3", "HG4"]
    # The header's order, whatever the order given.
    assert J.resolve_subset(["HG3", "HG1"], active, []) == ["HG1", "HG3"]
    assert J.resolve_subset(active[::-1], active, []) == active
    with pytest.raises(ValueError, match="empty"):
        J.resolve_subset([], active, [])
    with pytest.raises(ValueError) as e:
        J.resolve_subset(["HG1", "HG9", "HG1", "HG5", "HG8"], active, ["HG5"])
    msg = str(e.value)
    assert "given more than once: HG1" in msg
    assert "removed (samples.status = 'removed'): HG5" in msg
    assert "unknown: HG9, HG8" in msg


def test_file_may_hold_by_sample_id_bounds():
    assert J.file_may_hold((3, 7), [1, 5])
    assert J.file_may_hold((3, 7), [3]) and J.file_may_hold((3, 7), [7])
    assert not J.file_may_hold((3, 7), [1, 2, 8])
    # A file between two subset ids is not held by either.
    assert not J.file_may_hold((4, 6), [3, 7])
    # A missing bound does not exclude.
    assert J.file_may_hold((None, None), [1])
    assert J.file_may_hold((None, 2), [1]) and not J.file_may_hold(
        (None, 2), [3]
    )
    assert J.file_may_hold((5, None), [9]) and not J.file_may_hold(
        (5, None), [4]
    )


def test_sample_id_bounds_decoded_from_the_manifest_entry():
    from pyiceberg.conversions import to_bytes
    from pyiceberg.types import IntegerType

    class DataFile:
        lower_bounds = {7: to_bytes(IntegerType(), 300)}
        upper_bounds = {7: to_bytes(IntegerType(), 70000)}

    assert J.sample_id_bounds(DataFile(), 7, IntegerType()) == (300, 70000)
    # Another field's bounds only, or none at all: None.
    assert J.sample_id_bounds(DataFile(), 8, IntegerType()) == (None, None)
    DataFile.lower_bounds = None
    assert J.sample_id_bounds(DataFile(), 7, IntegerType()) == (None, 70000)


def test_subset_cross_check_line():
    assert J.SUBSET_CROSS_CHECK_LINE.startswith("Catalog cross-check: skipped")


# ── --trim-alt-alleles and --min-ac (K5) ────────────────────────────────────


def test_trim_and_min_ac_options_off_by_default():
    _, a = J.parse_args(["out.vcf"])
    assert (a.trim_alt_alleles, a.min_ac) == (False, 0)
    _, a = J.parse_args(["out.vcf", "--trim-alt-alleles", "--min-ac", "3"])
    assert (a.trim_alt_alleles, a.min_ac) == (True, 3)


@pytest.mark.parametrize("bad", ["-1", "x", "1.5", "", "1:nref"])
def test_bad_min_ac_is_exit_2(bad, capsys):
    with pytest.raises(SystemExit) as e:
        J.parse_args(["out.vcf.gz", f"--min-ac={bad}"])
    assert e.value.code == 2
    assert "--min-ac" in capsys.readouterr().err


def test_trim_alt_alleles_takes_no_value(capsys):
    with pytest.raises(SystemExit) as e:
        J.parse_args(["out.vcf.gz", "--trim-alt-alleles=1"])
    assert e.value.code == 2


def test_negative_min_ac_from_the_command_line():
    """Exit 2 before any connection."""
    proc = subprocess.run(
        [sys.executable, "joint_call.py", "out.vcf", "--min-ac=-2"],
        cwd=_repo(),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 2, proc.stderr
    assert "--min-ac must be 0 or more" in proc.stderr


def test_option_header_lines_where_the_projections_put_them():
    """One line per option set, in the order trim, cap, min-ac, after
    VCF_HEADERS and before the contig lines; tests/vcf_project.py holds the
    same text and the chain of its projections puts it at the same
    place."""
    import vcf_project as P

    assert J.TRIM_ALT_ALLELES_HEADER == P.TRIM_ALT_ALLELES_HEADER
    assert J.MIN_AC_HEADER == P.MIN_AC_HEADER
    contigs = [("chr21", 100), ("chr22", 200)]
    plain = J.vcf_header_lines(contigs, ["S1", "S2"])
    assert J.vcf_header_lines(contigs, ["S1", "S2"], 0, False, 0) == plain
    text = "\n".join(plain) + "\n"
    k = len(VCF_HEADERS)
    for trim, cap, min_ac in (
        (True, 0, 0),
        (False, 0, 2),
        (True, 6, 0),
        (True, 0, 1),
        (False, 3, 1),
        (True, 6, 1),
    ):
        lines = J.vcf_header_lines(contigs, ["S1", "S2"], cap, trim, min_ac)
        want = (
            (["##mosaic_trim_alt_alleles=true"] if trim else [])
            + ([f"##mosaic_max_alt_alleles={cap}"] if cap else [])
            + ([f"##mosaic_min_ac={min_ac}"] if min_ac else [])
        )
        assert lines[:k] == VCF_HEADERS
        assert lines[k : k + len(want)] == want
        assert lines[k + len(want) :] == plain[k:]
        out = P.project_trim(text)[0] if trim else text
        out = P.project(out, cap)[0]
        out = P.project_min_ac(out, min_ac)[0]
        assert out == "\n".join(lines) + "\n", (trim, cap, min_ac)


def test_option_closing_lines():
    assert J.trim_line({"records_trimmed": 7, "alts_dropped": 9}) == (
        "ALT trim (--trim-alt-alleles): 7 records trimmed, 9 ALT alleles "
        "with AC 0 dropped"
    )
    assert J.min_ac_line({"min_ac": 2, "records_left_out": 5}) == (
        "Minimum AC (--min-ac 2): 5 records left out (fewer ALT "
        "observations in their written genotypes)"
    )
    cap = {
        "max_alt_alleles": 6,
        "records_capped": 1,
        "genotypes_no_called": 0,
        "dropped_alt_observations": 0,
        "kept_alt_observations_lost": 0,
    }
    # Each only when its option was set, in the order trim, cap, min-ac.
    assert J.option_lines({"records": 3}) == []
    assert J.option_lines({"cap": cap}) == [J.cap_line(cap)]
    summary = {
        "min_ac": {"min_ac": 1, "records_left_out": 0},
        "cap": cap,
        "trim": {"records_trimmed": 0, "alts_dropped": 0},
    }
    assert J.option_lines(summary) == [
        J.trim_line(summary["trim"]),
        J.cap_line(cap),
        J.min_ac_line(summary["min_ac"]),
    ]


# ── the cohort export ────────────────────────────────────────────────────────

# A port nothing listens on: a run that tried PostgreSQL or the Iceberg
# catalog would fail with exit 1, not refuse with exit 2.
CLOSED = "1"
NO_STORES = {
    "MOSAIC_DB_PORT": CLOSED,
    "MOSAIC_ICEBERG_URI": f"http://127.0.0.1:{CLOSED}",
    "MOSAIC_S3_ENDPOINT": f"http://127.0.0.1:{CLOSED}",
    "MOSAIC_JOINT_CALL_WORKER": "/nonexistent/mosaic-joint-call",
}


def _cli(argv, env=None):
    full = dict(os.environ)
    full.update(NO_STORES)
    full.update(env or {})
    return subprocess.run(
        [sys.executable, "joint_call.py", *argv],
        cwd=_repo(),
        capture_output=True,
        text=True,
        env=full,
        timeout=120,
    )


def test_output_default_resolved_after_the_checks():
    _, a = J.parse_args([])
    assert a.output == "/dev/stdout"
    _, a = J.parse_args(["out.vcf"])
    assert a.output == "out.vcf"
    _, a = J.parse_args(["--from-export", "d"])
    assert a.output == "/dev/stdout"
    for argv in (
        ["--cohort-state"],
        ["--export-cohort", "d"],
        ["--export-contig", "d", "--region", "chr21"],
    ):
        _, a = J.parse_args(argv)
        assert a.output is None and J.writes_no_vcf(a)
    _, a = J.parse_args(["--from-export", "d", "out.vcf", "--samples", "A"])
    assert (a.from_export, a.output, a.samples) == ("d", "out.vcf", "A")
    assert not J.writes_no_vcf(a)


@pytest.mark.parametrize(
    "argv,message",
    [
        (["--cohort-state", "o.vcf"], "--cohort-state does not take an "),
        (["--cohort-state", "--region", "chr21"], "does not take --region"),
        (["--cohort-state", "--spill-dir", "s"], "does not take --spill-dir"),
        (["--export-cohort", "d", "o.vcf"], "does not take an output path"),
        (["--export-cohort", "d", "--region", "chr21"], "take --region"),
        (["--export-cohort", "d", "--samples", "A"], "take --samples"),
        (["--export-cohort", "d", "--samples-file", "f"], "--samples-file"),
        (["--export-cohort", "d", "--spill-dir", "s"], "take --spill-dir"),
        (["--export-contig", "d"], "--export-contig needs --region CONTIG"),
        (
            ["--export-contig", "d", "--region", "chr21:1-5"],
            "exports a whole contig",
        ),
        (
            ["--export-contig", "d", "--region", "chr21", "o.vcf"],
            "does not take an output path",
        ),
        (
            ["--export-contig", "d", "--region", "chr21", "--samples", "A"],
            "does not take --samples",
        ),
        (
            ["--export-contig", "d", "--region", "chr21", "--allow-stale"],
            "does not take --allow-stale",
        ),
        (
            ["--export-contig", "d", "--region", "chr21", "--spill-dir", "s"],
            "does not take --spill-dir",
        ),
        (["--from-export", "d", "--allow-stale"], "it belongs to the export"),
        (["--from-export", "d", "--cohort-state"], "not allowed with"),
        (["--export-cohort", "d", "--export-contig", "e"], "not allowed"),
    ],
)
def test_mode_refusals_are_argparse_errors(argv, message, capsys):
    with pytest.raises(SystemExit) as e:
        J.parse_args(argv)
    assert e.value.code == 2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv,message",
    [
        (["--cohort-state", "o.vcf"], "--cohort-state does not take"),
        (["--export-cohort", "d", "--samples", "A"], "--export-cohort does"),
        (["--export-contig", "d"], "--export-contig needs --region"),
        (["--from-export", "d", "--allow-stale"], "--from-export does not"),
    ],
)
def test_mode_refusals_from_the_command_line(argv, message):
    proc = _cli(argv)
    assert proc.returncode == 2, proc.stderr
    assert message in proc.stderr


class _FakeCursor:
    """A cursor answering the cohort state's queries."""

    def __init__(self, db):
        self.db = db
        self.last = None

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def execute(self, sql, params=None):
        self.db["queries"].append(sql)
        self.last = sql

    def fetchone(self):
        if "to_regclass" in self.last:
            return (self.db["view"],)
        if "FROM cohort_generation" in self.last:
            return self.db["run"]
        if "count(*)" in self.last:
            return (self.db["completed"],)
        raise AssertionError(self.last)

    def fetchall(self):
        assert "FROM samples" in self.last
        return list(self.db["samples"])

    def mogrify(self, sql, params):
        return (sql % {k: repr(v) for k, v in params.items()}).encode()

    def copy_expert(self, sql, fh):
        key = "sites" if "site_info" in sql else "alleles"
        fh.write(self.db["csv"][key].encode())


class _FakeConn:
    def __init__(self, **db):
        db.setdefault("queries", [])
        self.db = db

    def cursor(self):
        return _FakeCursor(self.db)


SAMPLES = [(2, "HG2", "active"), (1, "HG1", "active"), (3, "HG3", "removed")]


def _fake(**kw):
    db = {
        "view": True,
        "run": (7, "complete", "2026-11-12T10:11:12.123456"),
        "completed": 3,
        "samples": SAMPLES,
        "csv": {"sites": "", "alleles": ""},
    }
    db.update(kw)
    return _FakeConn(**db)


def test_cohort_state_keys_and_line():
    state = J.cohort_state(_fake(), None)
    assert tuple(sorted(state)) == tuple(sorted(J.COHORT_STATE_KEYS))
    assert state["agg_run_id"] == 7 and state["status"] == "complete"
    assert state["aggregated_at"] == "2026-11-12T10:11:12.123456"
    assert state["completed_batches"] == 3
    assert (state["active_samples"], state["removed_samples"]) == (2, 1)
    assert state["stale"] is None and state["format"] == 1
    # The digest is of the rows ordered by id, whatever order they come.
    assert state["samples_sha256"] == J.samples_digest(sorted(SAMPLES))
    line = J.state_line(state)
    assert "\n" not in line and " " not in line
    assert line == json.dumps(state, sort_keys=True, separators=(",", ":"))
    keys = list(json.loads(line))
    assert keys == sorted(keys)
    # No key for the time of the call or the names listed.
    for k in keys:
        assert "time" not in k and "listed" not in k and "names" not in k
    # A changed sample, or a stale reason, changes the line.
    moved = J.cohort_state(
        _fake(samples=[(1, "HG1", "active"), (2, "HG2", "removed")]), None
    )
    assert J.state_differences(state, moved) == [
        "active_samples",
        "samples_sha256",
    ]
    assert J.cohort_state(_fake(), "why")["stale"] == "why"


def test_cohort_state_without_a_view_or_a_run():
    for conn in (_fake(view=False), _fake(run=None)):
        s = J.cohort_state(conn, "no run")
        assert (s["agg_run_id"], s["status"], s["aggregated_at"]) == (
            None,
            None,
            None,
        )


CONTIGS_DOC = [(1, "chr21", 1000, True), (2, "chr22", 2000, False)]


def _crafted(tmp_path, state=None, sites_rows=2, alleles_rows=3):
    """A crafted export: cohort.json and chr21's three files."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    d = tmp_path / "export"
    d.mkdir()
    state = state or J.cohort_state(_fake(), None)
    doc = J.cohort_document(
        state, CONTIGS_DOC, [(1, "HG1"), (2, "HG2")], [(3, "HG3")]
    )
    J.write_json_atomic(str(d / "cohort.json"), doc)
    sites = pa.table(
        {
            f.name: pa.array([0] * sites_rows, f.type)
            for f in J.SITES_EXPORT_SCHEMA
        }
    )
    pq.write_table(sites, str(d / "sites.c1.parquet"))
    alleles = pa.table(
        {
            "position": pa.array([1] * alleles_rows, pa.int32()),
            "allele": pa.array(["A"] * alleles_rows),
            "count": pa.array([0] * alleles_rows, pa.int64()),
            "del_start": pa.array([None] * alleles_rows, pa.int32()),
            "del_end": pa.array([None] * alleles_rows, pa.int32()),
        }
    )
    pq.write_table(alleles, str(d / "alleles.c1.parquet"))
    J.write_json_atomic(
        str(d / "export.c1.json"),
        {
            "format": 1,
            "contig": {"index": 1, "name": "chr21", "length": 1000},
            "state": state,
            "rows": {"sites": 2, "alleles": 3},
        },
    )
    return d


def _listing(d):
    return {
        p.name: (p.stat().st_size, p.stat().st_mtime_ns)
        for p in sorted(d.iterdir())
    }


def test_cohort_json_written_and_read_back(tmp_path):
    d = _crafted(tmp_path)
    text = (d / "cohort.json").read_bytes()
    doc = J.load_cohort_file(str(d))
    assert doc["contigs"][1] == {
        "index": 2,
        "name": "chr22",
        "length": 2000,
        "has_sites": False,
    }
    assert doc["samples"] == [
        {"id": 1, "name": "HG1"},
        {"id": 2, "name": "HG2"},
    ]
    assert doc["removed"] == [{"id": 3, "name": "HG3"}]
    # Deterministic bytes, no temporary file left.
    J.write_json_atomic(str(d / "cohort.json"), doc)
    assert (d / "cohort.json").read_bytes() == text
    assert not [p for p in os.listdir(d) if ".tmp-" in p]
    rows = J.check_contig_exports(str(d), doc, doc["contigs"])
    assert rows == {1: {"sites": 2, "alleles": 3}}
    src = J.ExportCohort(str(d), doc)
    assert src.contigs == [(1, "chr21", 1000), (2, "chr22", 2000)]
    assert src.sample_ids() == {"HG1": 1, "HG2": 2}
    assert src.removed_ids == [3] and src.stale is None


def test_export_contig_files_writes_the_sidecar_last(tmp_path):
    d = tmp_path / "e"
    d.mkdir()
    state = J.cohort_state(_fake(), None)
    conn = _fake(
        csv={
            "sites": "5,1.5,3,2.0,1,1,2,3,4,0,1,0,7,2\n"
            "9,2.5,4,3.0,2,1,1,1,1,0,0,1,8,2\n",
            "alleles": "5,A,0,,\n5,xT,1,,\n9,-AT,1,9,11\n",
        }
    )
    contig = {"index": 1, "name": "chr21", "length": 1000}
    rows = J.export_contig_files(conn, str(d), contig, state)
    assert rows == {"sites": 2, "alleles": 3}
    assert sorted(os.listdir(d)) == [
        "alleles.c1.parquet",
        "export.c1.json",
        "sites.c1.parquet",
    ]
    side = json.loads((d / "export.c1.json").read_text())
    assert side == {
        "format": 1,
        "contig": contig,
        "state": state,
        "rows": rows,
    }
    doc = J.cohort_document(state, CONTIGS_DOC, [], [])
    J.write_json_atomic(str(d / "cohort.json"), doc)
    assert J.check_contig_exports(str(d), doc, doc["contigs"]) == {1: rows}


def test_export_contig_files_failure_leaves_nothing(tmp_path):
    d = tmp_path / "e"
    d.mkdir()
    conn = _fake(csv={"sites": "not,a,number\n", "alleles": ""})
    with pytest.raises(Exception):
        J.export_contig_files(
            conn,
            str(d),
            {"index": 1, "name": "chr21", "length": 9},
            J.cohort_state(_fake(), None),
        )
    assert os.listdir(d) == []


def test_from_export_runs_past_the_checks_to_the_worker(tmp_path):
    """The crafted export passes every check: the run gets to the worker
    lookup (exit 1, the remedy) without touching a store."""
    d = _crafted(tmp_path)
    before = _listing(d)
    proc = _cli(["--from-export", str(d), str(tmp_path / "o.vcf")])
    assert proc.returncode == 1, proc.stderr
    assert "the Rust joint caller is not found" in proc.stderr
    assert _listing(d) == before


def _edit_json(path, fn):
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj))


@pytest.mark.parametrize(
    "case",
    [
        "format",
        "state_format",
        "unknown",
        "twice",
        "removed",
        "empty",
        "empty_file",
        "missing_sites",
        "missing_sidecar",
        "sidecar_state",
        "sidecar_contig",
        "rows",
        "region",
        "allow_stale",
        "no_cohort",
    ],
)
def test_from_export_refusals_before_any_store(tmp_path, case):
    d = _crafted(tmp_path)
    out = str(tmp_path / "o.vcf")
    argv = ["--from-export", str(d), out]
    message = None
    if case == "format":
        _edit_json(d / "cohort.json", lambda o: o.update(format=2))
        message = "has format 2; this driver reads format 1"
    elif case == "state_format":
        _edit_json(d / "cohort.json", lambda o: o["state"].update(format=9))
        message = "a cohort state of another format"
    elif case == "unknown":
        argv += ["--samples", "HG1,NOPE"]
        message = "argument --samples: unknown: NOPE"
    elif case == "twice":
        argv += ["--samples", "HG1,HG1"]
        message = "argument --samples: given more than once: HG1"
    elif case == "removed":
        argv += ["--samples", "HG1,HG3"]
        message = (
            "argument --samples: removed (samples.status = 'removed'): HG3"
        )
    elif case == "empty":
        argv += ["--samples", ","]
        message = "argument --samples: an empty list of samples"
    elif case == "empty_file":
        names = tmp_path / "names"
        names.write_text("\n\n")
        argv += ["--samples-file", str(names)]
        message = "argument --samples-file: an empty list of samples"
    elif case == "missing_sites":
        os.remove(d / "sites.c1.parquet")
        message = (
            "chr21 has sites but the export holds no sites.c1.parquet "
            f"(run --export-contig {d} --region chr21)"
        )
    elif case == "missing_sidecar":
        os.remove(d / "export.c1.json")
        message = "holds no export.c1.json"
    elif case == "sidecar_state":
        _edit_json(
            d / "export.c1.json", lambda o: o["state"].update(agg_run_id=6)
        )
        message = "another cohort state than cohort.json's (differing: "
    elif case == "sidecar_contig":
        _edit_json(
            d / "export.c1.json", lambda o: o["contig"].update(length=5)
        )
        message = "export.c1.json is the export of"
    elif case == "rows":
        _edit_json(d / "export.c1.json", lambda o: o["rows"].update(alleles=4))
        message = "alleles.c1.parquet holds 3 rows, export.c1.json says 4"
    elif case == "region":
        argv += ["--region", "chrX"]
        message = "argument --region: contig 'chrX' is not in the contigs"
    elif case == "allow_stale":
        argv += ["--allow-stale"]
        message = "--from-export does not take --allow-stale"
    elif case == "no_cohort":
        os.remove(d / "cohort.json")
        message = "cohort.json does not exist"
    before = _listing(d)
    proc = _cli(argv)
    assert proc.returncode == 2, (case, proc.stderr)
    assert message in proc.stderr, (case, proc.stderr)
    assert "not found" not in proc.stderr  # before the worker lookup
    assert not os.path.exists(out)
    assert _listing(d) == before


def test_from_export_contig_without_sites_needs_no_files(tmp_path):
    """--region chr22 (no sites): no file of it is needed, and chr21's
    are not looked at; the run gets to the worker lookup."""
    d = _crafted(tmp_path)
    for f in ("sites.c1.parquet", "alleles.c1.parquet", "export.c1.json"):
        os.remove(d / f)
    proc = _cli(["--from-export", str(d), "--region", "chr22"])
    assert proc.returncode == 1, proc.stderr
    assert "the Rust joint caller is not found" in proc.stderr
    # The whole genome needs chr21's.
    proc = _cli(["--from-export", str(d)])
    assert proc.returncode == 2 and "holds no sites.c1.parquet" in (
        proc.stderr
    )


@pytest.mark.parametrize("case", ["no_cohort", "format", "contig"])
def test_export_contig_refusals_before_the_connection(tmp_path, case):
    d = _crafted(tmp_path)
    region = "chr21"
    if case == "no_cohort":
        os.remove(d / "cohort.json")
        message = "cohort.json does not exist"
    elif case == "format":
        _edit_json(d / "cohort.json", lambda o: o.update(format=0))
        message = "has format 0"
    else:
        region = "chr9"
        message = "contig 'chr9' is not in the cohort export"
    before = _listing(d)
    proc = _cli(["--export-contig", str(d), "--region", region])
    assert proc.returncode == 2, proc.stderr
    assert message in proc.stderr
    assert _listing(d) == before


def test_pg_modes_reach_the_connection():
    """With the database port closed, each PostgreSQL mode fails (exit 1)
    at the connection: the control for the refusals above."""
    for argv in (["--cohort-state"], ["--export-cohort", "/nonexistent"]):
        proc = _cli(argv)
        assert proc.returncode == 1, proc.stderr
        assert "Error:" in proc.stderr
