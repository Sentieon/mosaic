# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The joint caller (``joint_call.py``, the driver of the Rust
``mosaic-joint-call``) against its frozen goldens:

  1. every output, at each ``--window`` (10,000,000 / 1,000,000 / 0 or
     100,000) and ``--threads`` (2; 1 and 4 at ``--window 1000000`` on the
     fresh cohort), equals its scenario's golden
     (``jc_testlib.assert_golden``: the SHA-256, record and byte counts of
     the decompressed text, header included, and the ``Completed!`` and
     catalog cross-check closing lines, in
     tests/golden/joint_call_digests.json), with no ``nan`` token. The
     scenarios: the six chr21 samples as three 2-sample batches; a
     ``--region`` slice; ``--allow-stale`` after ``mosaic remove`` of one
     sample and before the aggregate that applies it; after that aggregate
     (the removed sample's batch's rewritten files are brotli); a
     ``--ref-gq-bands 20,60`` batch whose shard boundary falls inside a
     deletion's span (the cross-shard split), alone and with an HG00096 +
     HG00097 batch (their ``./.`` blocks; when test_data/input_gvcfs5 is
     present); the golden e2e region. The goldens were recorded while the
     cross-engine contract held (the Python engine's text and closing
     lines byte for byte, the contract this module checked until the
     Python engine's removal); a golden is re-recorded only for an
     intended output change (MOSAIC_JOINT_CALL_GOLDEN_RECORD=1 with
     MOSAIC_JOINT_CALL_GOLDEN_REASON);
  2. the gate (exit 2 and its message), ``--allow-stale``'s warning and the
     tombstone line;

plus the refusals: the binary missing (exit 1 with the remedy, no output),
a worker error (a malformed manifest, an unreadable export) leaving no
output and exit 1, and a zstd data file (a rewrite made before the codec
property) refused naming its codec; and J3's crash paths: the test seam
``MOSAIC_TEST_FAIL_JOINT_CALL_UNIT`` at ``--threads 4`` (exit 1 naming the
unit, no output, the driver's temp directory gone), a malformed seam value
(exit 1 before any unit) and a SIGKILL of the binary mid-run (exit 1, no
output). Bounded output memory: every golden run also
at ``--window 1000000`` × ``--threads`` 1 / 2 / 4 × the default
``--unit-buffer-bytes`` and 1 (no ``<output>.spill-*`` directory left
after any run); the seam under ``--unit-buffer-bytes 1``; a SIGKILL of the
binary once its spill directory exists (no output, no spill directory, no
temp directory); ``--spill-dir``. A sample subset (``--samples`` /
``--samples-file``) on the fresh cohort, stale after the removal and after
the aggregate that applies it: each subset's output equals
``vcf_project.project_samples`` of the same state's whole-cohort uncapped
output byte for byte (header included), and under a cap
``project(project_samples(...), N)`` with the driver's cap counts; the
data files the run reads (a subset of one batch: that batch's only);
``bcftools view -s`` of the whole output gives the same GT columns, AN and
AC; the refusals (exit 2 naming the names, before the gate).
``--trim-alt-alleles`` and ``--min-ac`` alone and together, with and
without a subset and under caps 0, 3 and the default, on the fresh cohort,
stale after the removal and after the aggregate that applies it: each
output equals the projection chain ``project_min_ac(project(project_trim(
project_samples(whole, S)), N), K)`` of the same state's whole-cohort
uncapped output byte for byte, with the driver's counts the projections';
the cross-check lines under the trim alone are the golden's; ``bcftools
view [-s S] --min-ac 1 --trim-alt-alleles`` of the whole output gives the
same sites, ALTs, GT columns, AC and AN.
"""

import gzip
import json
import os
import re
import signal
import stat
import subprocess
import sys
import time
from typing import Dict, List, Optional, Tuple

import pyarrow.parquet as pq
import pytest

import jc_testlib as L
import vcf_project as P

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        not L.joint_call_worker_available(),
        reason=L.RUST_JOINT_CALL_SKIP_REASON,
    ),
]

TD = os.path.join(L.REPO_ROOT, "test_data")
DOWNLOADS_CHR21 = [
    os.path.join(TD, f"downloads/{s}/{s}.chr21.final.hard-filtered.gvcf.gz")
    for s in ("HG00099", "HG00100", "HG00101", "HG00102")
]
HG00096 = os.path.join(
    TD, "input_gvcfs5/HG00096.chr21.final.hard-filtered.gvcf.gz"
)
HG00097 = os.path.join(
    TD, "input_gvcfs5/HG00097.chr21.final.hard-filtered.gvcf.gz"
)
SLICE = "chr21:10000000-12000000"
GOLDEN_REGION = "chr21:10000000-12000000"
# A shard boundary at 0-based 10,001,908 (4 x 2,500,477): inside the span of
# the two input_gvcfs4 samples' deletion GAAA>G at 10,001,907
# (test_removal_substrate_lake.py's cross-shard fixture).
CROSS_SHARD_SIZE = 2_500_477
CROSS_REGION = "chr21:10000000-10300000"
NO_CALL_REGION = "chr21:5000000-10300000"


# ── helpers ──────────────────────────────────────────────────────────────────


def _text(path: str) -> str:
    with gzip.open(path, "rt") as fh:
        return fh.read()


# Bounded output memory (K2): every golden scenario at
# threads 1 / 2 / 4, with the default --unit-buffer-bytes and with 1 (every
# chunk of a unit that is not the head spills), at a window that gives
# several units.
K2_WINDOW = 1_000_000
K2_THREADS = (1, 2, 4)
K2_BOUNDS: Tuple[Optional[int], ...] = (None, 1)


def _runs(
    work: str,
    tag: str,
    golden: str,
    region: Optional[str] = None,
    allow_stale: bool = False,
    windows: Tuple[int, ...] = (10_000_000,),
    threads: Tuple[Tuple[int, int], ...] = (),
) -> Dict[str, object]:
    """The joint caller at each window (2 threads), plus each ``(window,
    threads)`` of ``threads``, plus K2's runs (``K2_WINDOW`` at each of
    ``K2_THREADS`` and ``K2_BOUNDS``), every output against the ``golden``
    scenario (``jc_testlib.assert_golden``, the closing lines included),
    and no ``<output>.spill-*`` directory left beside any output. Returns
    per run the counts (records, ``nan`` tokens) and the stderr (keyed by
    the window at 2 threads, else ``<window>_t<threads>``, ``_b<bound>``
    appended under a bound other than the default)."""
    base = [(w, 2) for w in windows] + [
        x for x in threads if x not in [(w, 2) for w in windows]
    ]
    runs: List[Tuple[int, int, Optional[int]]] = [
        (w, t, None) for w, t in base
    ]
    for b in K2_BOUNDS:
        for t in K2_THREADS:
            if (K2_WINDOW, t, b) not in runs:
                runs.append((K2_WINDOW, t, b))
    out: Dict[str, object] = {}
    for w, t, b in runs:
        key = f"{w}" if t == 2 else f"{w}_t{t}"
        if b is not None:
            key += f"_b{b}"
        rs = os.path.join(work, f"{tag}.rs{key}.vcf.gz")
        proc = L.run_joint_call(
            rs,
            region=region,
            allow_stale=allow_stale,
            window=w,
            threads=t,
            unit_buffer_bytes=b,
            max_alt_alleles=0,
        )
        assert not os.path.exists(rs + ".tbi")
        assert L.spill_dirs_beside(rs) == []
        assert (
            f"engine=rust threads={t} window={w} max_alt_alleles=0"
            in proc.stderr
        )
        held_units, held_bytes, sp_units, sp_bytes, copy = (
            L.joint_call_spilled(proc.stderr)
        )
        print(
            f"K2 {tag} window={w} threads={t} bound={b}: held at most "
            f"{held_units} unit(s), {held_bytes} bytes; spilled {sp_units} "
            f"unit(s), {sp_bytes} bytes; copy {copy:.2f} s"
        )
        if t == 1:
            # One thread: every unit is the head from its start.
            assert (held_units, held_bytes, sp_units, sp_bytes) == (0, 0, 0, 0)
        if b is None:
            # Under the default bound (64 MiB) no fixture unit spills.
            assert (sp_units, sp_bytes) == (0, 0)
        text = _text(rs)
        counts = {
            "records": sum(
                1 for x in text.split("\n") if x and not x.startswith("#")
            ),
            "nan": len(
                re.findall(r"(?<![A-Za-z])nan(?![A-Za-z])", text, re.I)
            ),
        }
        print(f"{tag} window={w} threads={t}: {counts}")
        assert counts["nan"] == 0
        L.assert_golden(golden, rs, proc.stderr)
        out[f"counts_{key}"] = counts
        out[f"stderr_{key}"] = proc.stderr
        if b is not None:
            os.remove(rs)
    # K3: the capped outputs against the projection of this scenario's
    # uncapped text (the first run's, its golden's).
    first = runs[0]
    base = f"{first[0]}" if first[1] == 2 else f"{first[0]}_t{first[1]}"
    out["cap"] = _cap_runs(
        work,
        tag,
        _text(os.path.join(work, f"{tag}.rs{base}.vcf.gz")),
        region=region,
        allow_stale=allow_stale,
    )
    return out


# The ALT cap (K3): the caps checked against the
# projection of each golden scenario's uncapped output, at two settings
# (so K2's ordered writer runs under the cap), plus the default.
CAPS = (1, 3, 6, 10)
CAP_SETTINGS = ((10_000_000, 2), (1_000_000, 4))


def _first_differing_lines(want: str, got: str, limit: int = 5) -> str:
    a, b = want.split("\n"), got.split("\n")
    out = []
    for i in range(max(len(a), len(b))):
        x = a[i] if i < len(a) else "<end of projection>"
        y = b[i] if i < len(b) else "<end of output>"
        if x != y:
            out.append(
                f"  line {i + 1}:\n    projection: {x[:300]}\n    output:     "
                f"{y[:300]}"
            )
            if len(out) >= limit:
                break
    return "\n".join(out)


def _cap_runs(
    work: str,
    tag: str,
    uncapped: str,
    region: Optional[str] = None,
    allow_stale: bool = False,
) -> Dict[int, Dict[str, int]]:
    """For each cap of CAPS at each of CAP_SETTINGS: the output's
    decompressed text equals ``vcf_project.project(uncapped, N)`` byte for
    byte and the driver's cap line its counts; the cross-check line says
    the capped records were not compared. Then a run with no
    ``--max-alt-alleles`` (the default) equals the N = 6 projection, its
    header naming 6. Returns the projection's counts per cap."""
    counts: Dict[int, Dict[str, int]] = {}
    for n in CAPS:
        want, c = P.project(uncapped, n)
        counts[n] = c
        for w, t in CAP_SETTINGS:
            rs = os.path.join(work, f"{tag}.cap{n}.w{w}.t{t}.vcf.gz")
            proc = L.run_joint_call(
                rs,
                region=region,
                allow_stale=allow_stale,
                window=w,
                threads=t,
                max_alt_alleles=n,
            )
            got = _text(rs)
            assert got == want, (
                f"{tag} cap {n} window {w} threads {t}: the output differs "
                "from the projection of the uncapped output\n"
                + _first_differing_lines(want, got)
            )
            assert L.joint_call_cap_counts(proc.stderr) == (n, c), proc.stderr[
                -2000:
            ]
            cross = [
                x
                for x in L.closing_lines(proc.stderr)
                if x.startswith("Catalog cross-check:")
            ]
            assert len(cross) == 1
            not_compared = (
                f"{c['records_capped']} records capped by --max-alt-alleles "
                "not compared"
            )
            assert (not_compared in cross[0]) == (c["records_capped"] > 0)
            os.remove(rs)
        print(f"K3 {tag} cap {n}: {c}")
    rs = os.path.join(work, f"{tag}.cap_default.vcf.gz")
    proc = L.run_joint_call(rs, region=region, allow_stale=allow_stale)
    got = _text(rs)
    assert got == P.project(uncapped, 6)[0]
    assert "\n##mosaic_max_alt_alleles=6\n" in got
    assert "max_alt_alleles=6" in proc.stderr
    assert L.joint_call_cap_counts(proc.stderr) == (6, counts[6])
    os.remove(rs)
    return counts


# The sample subset (K4): the six samples, in the
# header's order, and the batches they were ingested in.
SIX = ["HG00099", "HG00100", "HG00101", "HG00102", "HG00245", "HG00420"]
BATCH_OF = {
    "HG00245": 0,
    "HG00420": 0,
    "HG00099": 1,
    "HG00100": 1,
    "HG00101": 2,
    "HG00102": 2,
}


def _whole(work: str, tag: str, **kw) -> Tuple[str, str]:
    """The whole cohort's uncapped output (text, stderr) at 10 Mbp units
    and 2 threads."""
    rs = os.path.join(work, f"{tag}.whole.vcf.gz")
    proc = L.run_joint_call(
        rs, window=10_000_000, threads=2, max_alt_alleles=0, **kw
    )
    return _text(rs), proc.stderr


def _subset_runs(
    work: str,
    tag: str,
    whole: str,
    names: List[str],
    settings: Tuple[Tuple[int, int], ...] = ((10_000_000, 2),),
    caps: Tuple[Optional[int], ...] = (0,),
    keep: bool = False,
    **kw,
) -> Dict[Tuple[int, int, Optional[int]], str]:
    """``--samples names`` at each (window, threads) of ``settings`` and
    each cap of ``caps`` (None: no ``--max-alt-alleles``, the default 6):
    the output's decompressed text equals ``project(project_samples(whole,
    names), N)`` byte for byte (N = 0: the subset projection alone), the
    driver's cap counts equal the projection's, and the cross-check line
    says it was skipped. Returns each run's stderr; the outputs are removed
    unless ``keep``."""
    sub = P.project_samples(whole, names)
    out: Dict[Tuple[int, int, Optional[int]], str] = {}
    for n in caps:
        want, counts = P.project(sub, 6 if n is None else n)
        for w, t in settings:
            rs = os.path.join(
                work, f"{tag}.sub{len(names)}.n{n}.w{w}.t{t}.vcf.gz"
            )
            proc = L.run_joint_call(
                rs,
                window=w,
                threads=t,
                max_alt_alleles=n,
                samples=names,
                **kw,
            )
            got = _text(rs)
            assert got == want, (
                f"{tag} subset {names} cap {n} window {w} threads {t}: the "
                "output differs from the projection of the whole-cohort "
                "output\n" + _first_differing_lines(want, got)
            )
            closing = L.closing_lines(proc.stderr)
            assert P_SKIPPED in closing, closing
            assert not any(
                "equal the emitted AC" in x or "differ from the emitted" in x
                for x in closing
            )
            if n == 0:
                assert L.joint_call_cap_counts(proc.stderr) is None
            else:
                assert L.joint_call_cap_counts(proc.stderr) == (
                    6 if n is None else n,
                    counts,
                ), proc.stderr[-2000:]
            assert f"Sample subset: {len(names)} of " in proc.stderr
            out[(w, t, n)] = proc.stderr
            if not keep:
                os.remove(rs)
        print(f"K4 {tag} subset {names} cap {n}: {counts}")
    return out


# The driver's closing line of a subset run (joint_call.py's
# SUBSET_CROSS_CHECK_LINE, by value).
P_SKIPPED = (
    "Catalog cross-check: skipped under --samples / --samples-file (the "
    "catalog's allele counts, alleles.count, are the whole cohort's)"
)


def _bcftools_view_s_agrees(whole_path: str, sub_path: str, names) -> int:
    """``bcftools view -s names`` of the whole uncapped output (which
    recomputes AC / AN for the subset) against the subset's output: the
    same records, GT columns and AN at every record, and the same AC for
    every ALT the subset record lists; an ALT it does not list (the ``*``
    the subset drops) has AC 0 there. Returns the records compared."""
    fmt = "%CHROM\t%POS\t%ALT\t%INFO/AN\t%INFO/AC[\t%GT]\n"
    view = subprocess.run(
        ["bcftools", "view", "-s", ",".join(names), whole_path],
        capture_output=True,
        check=True,
    )
    ref = subprocess.run(
        ["bcftools", "query", "-f", fmt, "-"],
        input=view.stdout,
        capture_output=True,
        check=True,
    ).stdout.decode()
    got = subprocess.run(
        ["bcftools", "query", "-f", fmt, sub_path],
        capture_output=True,
        check=True,
    ).stdout.decode()
    a, b = ref.splitlines(), got.splitlines()
    assert len(a) == len(b) and len(a) > 0
    for x, y in zip(a, b):
        fx, fy = x.split("\t"), y.split("\t")
        assert fx[:2] == fy[:2], (x, y)
        assert fx[3] == fy[3], (x, y)  # AN
        assert fx[5:] == fy[5:], (x, y)  # the GT columns
        ac_x = dict(zip(fx[2].split(","), fx[4].split(",")))
        alts_y = [] if fy[2] == "." else fy[2].split(",")
        ac_y = dict(zip(alts_y, fy[4].split(","))) if alts_y else {}
        for alt, c in ac_x.items():
            assert ac_y.get(alt, "0") == c, (alt, x, y)
        assert set(ac_y) <= set(ac_x), (x, y)
    return len(a)


# The trim and --min-ac (K5): (--trim-alt-alleles,
# --min-ac) combinations, each under the caps of K5_CAPS (None: no
# --max-alt-alleles, the default 6).
K5_OPTIONS = ((True, 0), (False, 1), (False, 2), (True, 1))
K5_CAPS: Tuple[Optional[int], ...] = (0, 3, None)


def _k5_runs(
    work: str,
    tag: str,
    whole: str,
    names: Optional[List[str]] = None,
    options: Tuple[Tuple[bool, int], ...] = K5_OPTIONS,
    caps: Tuple[Optional[int], ...] = K5_CAPS,
    settings: Tuple[Tuple[int, int], ...] = ((10_000_000, 2),),
    **kw,
) -> Dict[Tuple[bool, int, Optional[int], int, int], str]:
    """For each (trim, K) of ``options``, cap of ``caps`` and (window,
    threads) of ``settings``: ``joint_call.py [--samples names]
    [--trim-alt-alleles] [--min-ac K]`` (K 0: no option) at the cap; the
    output's decompressed text equals the projection chain
    ``project_min_ac(project(project_trim(project_samples(whole, names)),
    N), K)`` (each step only when set) byte for byte, header included; the
    driver's trim, cap and --min-ac counts equal the projections' (a line
    only when its option is set); ``Completed!`` counts the records
    written; under a subset the cross-check is skipped. Returns each run's
    stderr."""
    base = whole if names is None else P.project_samples(whole, names)
    trimmed = P.project_trim(base)
    out: Dict[Tuple[bool, int, Optional[int], int, int], str] = {}
    for trim, k in options:
        src, tc = trimmed if trim else (base, None)
        for cap in caps:
            n = 6 if cap is None else cap
            capped, cc = P.project(src, n)
            want, mc = P.project_min_ac(capped, k)
            records = sum(
                1 for x in want.split("\n") if x and not x.startswith("#")
            )
            for w, t in settings:
                rs = os.path.join(
                    work,
                    f"{tag}.k5.s{len(names or [])}.t{int(trim)}.k{k}.n{cap}"
                    f".w{w}.t{t}.vcf.gz",
                )
                proc = L.run_joint_call(
                    rs,
                    window=w,
                    threads=t,
                    max_alt_alleles=cap,
                    samples=names,
                    trim_alt_alleles=trim,
                    min_ac=k if k else None,
                    **kw,
                )
                got = _text(rs)
                what = (
                    f"{tag} subset {names} trim {trim} min-ac {k} cap {cap} "
                    f"window {w} threads {t}"
                )
                assert got == want, (
                    f"{what}: the output differs from the projection chain "
                    "of the whole-cohort output\n"
                    + _first_differing_lines(want, got)
                )
                err = proc.stderr
                assert L.joint_call_trim_counts(err) == tc, (what, err[-2000:])
                assert L.joint_call_cap_counts(err) == (
                    (n, cc) if n > 0 else None
                ), (what, err[-2000:])
                assert L.joint_call_min_ac_counts(err) == (
                    (k, mc) if k > 0 else None
                ), (what, err[-2000:])
                assert (
                    f"Completed! Total processed: {records} variants" in err
                ), (what, err[-2000:])
                assert ("trim_alt_alleles=true" in err) == trim
                assert (f" min_ac={k}" in err) == (k > 0)
                closing = L.closing_lines(err)
                assert (P_SKIPPED in closing) == (names is not None), closing
                out[(trim, k, cap, w, t)] = err
                os.remove(rs)
            print(
                f"K5 {tag} subset {names} trim {trim} min-ac {k} cap {cap}: "
                f"{records} records; trim {tc}; cap {cc}; min-ac {mc}"
            )
    return out


def _bcftools_view_trim_min_ac_agrees(
    whole_path: str,
    path: str,
    names: Optional[List[str]],
    min_ac: Optional[int] = 1,
) -> Tuple[int, int]:
    """``bcftools view [-s names] [--min-ac K] --trim-alt-alleles`` of the
    whole uncapped output against ``joint_call.py [--min-ac K]
    --trim-alt-alleles [--samples names] --max-alt-alleles 0``: the same
    sites, ALTs, AN, AC (``.`` when absent: a record without ALTs carries
    none, on both sides) and GT columns at every record (PL is not
    compared: bcftools cuts it without the shift); bcftools' stderr holds no
    ``[W::`` / ``[E::`` line. Returns the records compared and those
    without ALTs."""
    fmt = "%CHROM\t%POS\t%REF\t%ALT\t%INFO/AN\t%INFO/AC[\t%GT]\n"
    view = ["bcftools", "view", "--trim-alt-alleles"]
    if min_ac is not None:
        view += ["--min-ac", str(min_ac)]
    if names is not None:
        # The header's order (bcftools keeps the order -s gives; the joint
        # caller writes the cohort header's).
        view += ["-s", ",".join(sorted(names, key=SIX.index))]
    viewed = subprocess.run(
        view + [whole_path], capture_output=True, check=True
    )
    assert not _htslib_complaints(viewed.stderr.decode()), viewed.stderr
    ref = subprocess.run(
        ["bcftools", "query", "-f", fmt, "-"],
        input=viewed.stdout,
        capture_output=True,
        check=True,
    ).stdout.decode()
    got = subprocess.run(
        ["bcftools", "query", "-f", fmt, path],
        capture_output=True,
        check=True,
    ).stdout.decode()
    a, b = ref.splitlines(), got.splitlines()
    assert len(a) > 0
    bad = [(x, y) for x, y in zip(a, b) if x != y]
    assert len(a) == len(b) and not bad, (
        f"bcftools {len(a)} records, joint_call.py {len(b)}; first "
        f"differences: {bad[:5]}"
    )
    return len(a), sum(1 for x in b if x.split("\t")[3] == ".")


def _htslib_complaints(stderr: str) -> List[str]:
    """htslib's warning and error lines (``[W::`` / ``[E::``)."""
    return [x for x in stderr.splitlines() if x.startswith(("[W::", "[E::"))]


def _alt_less(text: str) -> int:
    """The records without ALTs (ALT ``.``) of a VCF text."""
    return sum(
        1
        for x in text.split("\n")
        if x and not x.startswith("#") and x.split("\t", 5)[4] == "."
    )


def _sample_files_codecs(table: str) -> Dict[str, set]:
    """Each data file of a per-sample table and its column codecs."""
    catalog = L.get_catalog()
    t = catalog.load_table(f"{L.TEST_NAMESPACE}.{table}")
    out = {}
    s3 = L.s3_client()
    for task in t.scan().plan_files():
        uri = task.file.file_path
        bucket, key = uri[len("s3://") :].split("/", 1)
        body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
        import io

        md = pq.ParquetFile(io.BytesIO(body)).metadata
        codecs = {
            md.row_group(g).column(c).compression
            for g in range(md.num_row_groups)
            for c in range(md.num_columns)
        }
        out[uri] = codecs
    return out


def _wrapper(path: str, body: str) -> str:
    """An executable shell script standing in for the binary."""
    with open(path, "w") as fh:
        fh.write("#!/bin/bash\n" + body + "\n")
    os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)
    return path


# ── the six-sample cohort ────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def cohort(docker_stack, tmp_path_factory):
    missing = [p for p in DOWNLOADS_CHR21 if not os.path.exists(p)]
    if missing:
        pytest.skip(f"chr21 downloads not present: {missing}")
    work = str(tmp_path_factory.mktemp("rust_jc"))
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    for batch in (L.GVCFS, DOWNLOADS_CHR21[:2], DOWNLOADS_CHR21[2:]):
        L.run_ingest(batch, region="chr21", threads=2)
    L.run_aggregate()
    return {"work": work}


def test_fresh_cohort_byte_for_byte(cohort):
    """The fresh cohort against its golden at every window and thread
    setting, with its closing lines and the cap counts."""
    r = _runs(
        cohort["work"],
        "fresh",
        "fresh_cohort_chr21",
        windows=(10_000_000, 1_000_000, 0),
        threads=((1_000_000, 1), (1_000_000, 4)),
    )
    assert r["counts_10000000"]["records"] > 130_000
    # K3: the records over each cap (the fixture's counts at 3, 6, 10).
    assert {n: c["records_capped"] for n, c in r["cap"].items()} == {
        1: 9626,
        3: 1882,
        6: 311,
        10: 40,
    }
    assert "Completed! Total processed: 132382 variants" in (
        r["stderr_10000000"]
    )


def test_region_slice_byte_for_byte(cohort):
    r = _runs(cohort["work"], "slice", "slice_chr21_10M_12M", region=SLICE)
    assert r["counts_10000000"]["records"] > 1000


def test_subsets_are_projections(cohort):
    """K4 on the fresh cohort: one sample; two of different batches; two
    of one batch; all but one; every sample (the whole cohort's text): each
    equals ``project_samples`` of the whole uncapped output; one subset
    also at 1 Mbp x 4 threads and the whole contig x 1 thread; under the
    caps 1 / 3 / 6 and the default; ``--samples-file`` gives the same."""
    work = cohort["work"]
    whole, _ = _whole(work, "k4fresh")
    two_batches = ["HG00245", "HG00100"]  # given out of header order
    _subset_runs(
        work,
        "k4fresh",
        whole,
        two_batches,
        settings=((10_000_000, 2), (1_000_000, 4), (0, 1)),
        caps=(0, 1, 3, 6, None),
    )
    for names in (
        ["HG00099"],
        ["HG00101", "HG00102"],
        [x for x in SIX if x != "HG00420"],
    ):
        _subset_runs(work, "k4fresh", whole, names)
    # Every sample: still a subset run (the cross-check skipped), the
    # whole cohort's text.
    assert P.project_samples(whole, SIX) == whole
    _subset_runs(work, "k4fresh", whole, SIX[::-1])
    # --samples-file: the same text as --samples.
    names_file = os.path.join(work, "k4.names")
    with open(names_file, "w") as fh:
        fh.write("HG00245\n\nHG00100\n")
    rs = os.path.join(work, "k4fresh.file.vcf.gz")
    L.run_joint_call(rs, max_alt_alleles=0, samples_file=names_file)
    assert _text(rs) == P.project_samples(whole, two_batches)


def test_subset_reads_only_its_files_and_agrees_with_bcftools(cohort):
    """K4: the data files the run reads (the binary's summary, the
    driver's line): the whole cohort all of them (one per batch and table
    here), a subset of one batch that batch's only, of two batches two;
    the bytes fetched fall with them. ``bcftools view -s`` of the whole
    output agrees on the GT columns, AN and AC."""
    work = cohort["work"]
    catalog = L.get_catalog()
    n_files = {
        t: len(
            list(
                catalog.load_table(f"{L.TEST_NAMESPACE}.{t}")
                .scan()
                .plan_files()
            )
        )
        for t in ("sample_site_info", "sample_ref_blocks")
    }
    # The fixture: one data file per batch and table.
    assert n_files == {"sample_site_info": 3, "sample_ref_blocks": 3}
    whole_path = os.path.join(work, "k4files.whole.vcf.gz")
    proc = L.run_joint_call(whole_path, max_alt_alleles=0)
    assert L.joint_call_files_read(proc.stderr) == (3, 3)
    whole = _text(whole_path)
    for names, files in (
        (["HG00101", "HG00102"], 1),
        (["HG00101"], 1),
        (["HG00099", "HG00420"], 2),
        (["HG00100", "HG00102", "HG00245"], 3),
    ):
        assert len({BATCH_OF[n] for n in names}) == files
        err = _subset_runs(work, "k4files", whole, names, keep=True)
        stderr = err[(10_000_000, 2, 0)]
        assert L.joint_call_files_read(stderr) == (files, files), stderr
        assert (
            f"; {files} of 3 sample_site_info and {files} of 3 "
            "sample_ref_blocks data files"
        ) in stderr
        sub = os.path.join(
            work, f"k4files.sub{len(names)}.n0.w10000000.t2.vcf.gz"
        )
        n = _bcftools_view_s_agrees(whole_path, sub, names)
        print(f"K4 bcftools view -s {names}: {n} records agree")
        os.remove(sub)


def test_trim_and_min_ac_are_projections(cohort):
    """K5 on the fresh cohort: --trim-alt-alleles, --min-ac 1, --min-ac 2
    and both, each under caps 0, 3 and the default, for the whole cohort,
    one sample and two samples of different batches, against the
    projection chain of the whole uncapped output; one combination also at
    1 Mbp x 4 threads and the whole contig x 1 thread. Under the trim alone
    at cap 0 the closing lines are the golden's plus the trim's line."""
    work = cohort["work"]
    whole, _ = _whole(work, "k5fresh")
    err = _k5_runs(work, "k5fresh", whole)
    _k5_runs(
        work,
        "k5fresh",
        whole,
        options=((True, 1),),
        caps=(3,),
        settings=((1_000_000, 4), (0, 1)),
    )
    for names in (["HG00099"], ["HG00245", "HG00100"]):
        _k5_runs(work, "k5fresh", whole, names=names)
    # The cross-check under the trim alone (cap 0): a trimmed record is
    # compared on its untrimmed alleles, so the lines are the golden's.
    with open(L.GOLDEN_DIGESTS) as fh:
        golden = json.load(fh)["scenarios"]["fresh_cohort_chr21"]
    closing = L.closing_lines(err[(True, 0, 0, 10_000_000, 2)])
    assert closing[-1].startswith("ALT trim (--trim-alt-alleles): ")
    assert closing[:-1] == golden["closing_lines"], closing


def test_trim_and_min_ac_agree_with_bcftools(cohort):
    """K5 against ``bcftools view [-s S] --min-ac 1 --trim-alt-alleles`` of
    the whole uncapped output, for the whole cohort and two subsets: the
    same sites, ALTs, GT columns, AC and AN."""
    work = cohort["work"]
    whole_path = os.path.join(work, "k5bcf.whole.vcf.gz")
    L.run_joint_call(whole_path, max_alt_alleles=0)
    for names in (None, ["HG00099"], ["HG00245", "HG00100"]):
        path = os.path.join(work, f"k5bcf.{len(names or [])}.vcf.gz")
        L.run_joint_call(
            path,
            max_alt_alleles=0,
            samples=names,
            trim_alt_alleles=True,
            min_ac=1,
        )
        n, dots = _bcftools_view_trim_min_ac_agrees(whole_path, path, names)
        print(f"K5 bcftools view --min-ac 1 --trim-alt-alleles {names}: {n}")
        assert dots == 0
        os.remove(path)


def test_trim_agrees_with_bcftools_without_min_ac(cohort):
    """K5: ``bcftools view [-s S] --trim-alt-alleles`` of the whole
    uncapped output against ``--trim-alt-alleles [--samples S]
    --max-alt-alleles 0``, for the whole cohort and two subsets, at every
    record including those left without ALTs (no AC on either side): the
    same sites, ALTs, AN, AC and GT columns. Prints the ALT-less records of
    each trim output and of each subset's own output at cap 0 (its
    ``*``-only records), which carry no AC and no AF."""
    work = cohort["work"]
    whole_path = os.path.join(work, "k5bcf0.whole.vcf.gz")
    L.run_joint_call(whole_path, max_alt_alleles=0)
    whole = _text(whole_path)
    assert _alt_less(whole) == 0
    for names in (None, ["HG00099"], ["HG00245", "HG00100"]):
        path = os.path.join(work, f"k5bcf0.{len(names or [])}.vcf.gz")
        L.run_joint_call(
            path, max_alt_alleles=0, samples=names, trim_alt_alleles=True
        )
        text = _text(path)
        n, dots = _bcftools_view_trim_min_ac_agrees(
            whole_path, path, names, min_ac=None
        )
        assert dots == _alt_less(text)
        assert not re.search(r"\tAC=[;\t]|;AF=[;\t]|\tAF=[;\t]", text)
        sub = P.project_samples(whole, names) if names else whole
        print(
            f"K5 bcftools view --trim-alt-alleles {names}: {n} records agree; "
            f"{dots} without ALTs after the trim; {_alt_less(sub)} "
            "without ALTs in the subset's own output at cap 0"
        )
        os.remove(path)


def test_bcftools_reads_the_trimmed_subset_cleanly(cohort):
    """K5: on ``--trim-alt-alleles --samples HG00245,HG00100`` output
    (which holds records without ALTs), ``bcftools view
    --trim-alt-alleles``, ``bcftools view -s HG00100`` and ``bcftools
    stats`` exit 0 with no htslib warning or error, and bcftools' trim
    leaves AN unchanged at every record."""
    work = cohort["work"]
    path = os.path.join(work, "k5clean.vcf.gz")
    L.run_joint_call(
        path,
        max_alt_alleles=0,
        samples=["HG00245", "HG00100"],
        trim_alt_alleles=True,
    )
    assert _alt_less(_text(path)) > 0
    outs = {}
    for name, cmd in (
        ("trim", ["bcftools", "view", "--trim-alt-alleles", path]),
        ("view_s", ["bcftools", "view", "-s", "HG00100", path]),
        ("stats", ["bcftools", "stats", path]),
    ):
        p = subprocess.run(cmd, capture_output=True)
        assert p.returncode == 0, (name, p.stderr)
        assert not _htslib_complaints(p.stderr.decode()), (name, p.stderr)
        outs[name] = p.stdout
    an = ["bcftools", "query", "-f", "%POS\t%INFO/AN\n"]
    before = subprocess.run(
        an + [path], capture_output=True, check=True
    ).stdout
    after = subprocess.run(
        an + ["-"], input=outs["trim"], capture_output=True, check=True
    ).stdout
    assert before == after and before


def test_subset_refusals(cohort, tmp_path):
    """K4: an unknown name, a duplicate, an empty list, an empty file and
    both options at once: exit 2 with the names, no output, no gate."""
    out = str(tmp_path / "refused.vcf.gz")
    empty = str(tmp_path / "empty.names")
    with open(empty, "w") as fh:
        fh.write("\n\n")
    for kw, message in (
        ({"samples": ["HG00099", "NOPE1", "NOPE2"]}, "unknown: NOPE1, NOPE2"),
        ({"samples": ["HG00099", "HG00099"]}, "given more than once: HG00099"),
        ({"samples": [""]}, "an empty list of samples"),
        ({"samples_file": empty}, "an empty list of samples"),
        (
            {"samples": ["HG00099"], "samples_file": empty},
            "not allowed with argument",
        ),
        (
            {"samples_file": str(tmp_path / "none.names")},
            "argument --samples-file",
        ),
    ):
        proc = L.run_joint_call(out, check=False, **kw)
        assert proc.returncode == 2, (kw, proc.stderr[-2000:])
        assert message in proc.stderr, (kw, proc.stderr[-2000:])
        assert "exported" not in proc.stderr
        assert not os.path.exists(out)


def test_binary_missing_refused(cohort, tmp_path):
    out = str(tmp_path / "missing.vcf.gz")
    proc = L.run_joint_call(
        out,
        check=False,
        extra_env={"MOSAIC_JOINT_CALL_WORKER": str(tmp_path / "nope")},
    )
    assert proc.returncode == 1
    assert "the Rust joint caller is not found" in proc.stderr
    assert "cargo build --release" in proc.stderr
    assert "exported" not in proc.stderr  # before any export
    assert not os.path.exists(out)


def test_worker_errors_leave_no_output(cohort, tmp_path):
    binary = L.joint_call_worker_path()
    # A malformed manifest: an unknown field.
    bad_manifest = _wrapper(
        str(tmp_path / "bad_manifest.sh"),
        f"""sed -i.bak 's/"stale"/"bogus": 1, "stale"/' "$2"\n"""
        f'exec "{binary}" "$@"',
    )
    # An unreadable export: the sites file removed before the run.
    no_export = _wrapper(
        str(tmp_path / "no_export.sh"),
        'rm -f "$(dirname "$2")"/sites.c*.parquet\n' f'exec "{binary}" "$@"',
    )
    for wrapper, message in (
        (bad_manifest, "unknown field `bogus`"),
        (no_export, "sites.c"),
    ):
        out = str(tmp_path / "err.vcf.gz")
        proc = L.run_joint_call(
            out,
            region=SLICE,
            check=False,
            extra_env={"MOSAIC_JOINT_CALL_WORKER": wrapper},
        )
        assert proc.returncode == 1, proc.stderr[-2000:]
        assert message in proc.stderr, proc.stderr[-2000:]
        assert "Error: the Rust joint caller" in proc.stderr
        assert not os.path.exists(out)


FAIL_UNIT_ENV = "MOSAIC_TEST_FAIL_JOINT_CALL_UNIT"


def _tmp_env(tmp_path, name: str) -> Tuple[str, Dict[str, str]]:
    """A fresh directory for the driver's temp directory (``TMPDIR``), so
    a test can check the driver removed its own."""
    d = str(tmp_path / name)
    os.makedirs(d)
    return d, {"TMPDIR": d}


@pytest.mark.parametrize("bound", [None, 1], ids=["default", "bound1"])
def test_unit_error_seam_at_four_threads(cohort, tmp_path, bound):
    """J3: a unit's error ends the run at once: exit 1, the unit named, no
    summary, no output, the driver's temp directory gone; K2: also under
    ``--unit-buffer-bytes 1`` (the units past the head spill), no spill
    directory left."""
    tmp, env = _tmp_env(tmp_path, "tmp_seam")
    out = str(tmp_path / "seam.vcf.gz")
    env[FAIL_UNIT_ENV] = "chr21:5000001"
    proc = L.run_joint_call(
        out,
        window=1_000_000,
        threads=4,
        check=False,
        extra_env=env,
        unit_buffer_bytes=bound,
    )
    assert proc.returncode == 1, proc.stderr[-2000:]
    assert (
        "Error: unit chr21:5000001-6000000: "
        f"{FAIL_UNIT_ENV}: injected failure of unit chr21:5000001"
    ) in proc.stderr, proc.stderr[-2000:]
    assert "Error: the Rust joint caller" in proc.stderr
    assert "exited with code 1" in proc.stderr
    assert "Completed!" not in proc.stderr
    assert not os.path.exists(out)
    assert L.spill_dirs_beside(out) == []
    assert os.listdir(tmp) == []


def test_malformed_seam_refused_before_any_unit(cohort, tmp_path):
    tmp, env = _tmp_env(tmp_path, "tmp_bad_seam")
    for value in ("chr21", "chr21:0", ":5", "chr21:x"):
        out = str(tmp_path / "bad_seam.vcf.gz")
        env[FAIL_UNIT_ENV] = value
        proc = L.run_joint_call(out, check=False, extra_env=env)
        assert proc.returncode == 1, (value, proc.stderr[-2000:])
        assert f"{FAIL_UNIT_ENV}=" in proc.stderr, proc.stderr[-2000:]
        assert "expected <contig>:<start>" in proc.stderr
        assert "Unit 1/" not in proc.stderr
        assert not os.path.exists(out)
        assert os.listdir(tmp) == []


def _children(pid: int) -> List[Tuple[int, str]]:
    r = subprocess.run(
        ["ps", "-A", "-o", "pid=,ppid=,comm="],
        capture_output=True,
        text=True,
        check=True,
    )
    out = []
    for line in r.stdout.splitlines():
        parts = line.split(None, 2)
        if len(parts) == 3 and int(parts[1]) == pid:
            out.append((int(parts[0]), parts[2]))
    return out


def test_sigkill_of_the_binary_mid_run(cohort, tmp_path):
    """J3: a SIGKILL of `mosaic-joint-call` once its output exists: the
    driver exits 1 naming the signal and leaves no output."""
    tmp, env = _tmp_env(tmp_path, "tmp_kill")
    out = str(tmp_path / "killed.vcf.gz")
    proc = subprocess.Popen(
        [
            sys.executable,
            "joint_call.py",
            out,
            "--threads",
            "1",
            "--window",
            "1000000",
        ],
        cwd=L.REPO_ROOT,
        env=L.test_env(env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    killed = None
    deadline = time.time() + 120
    while killed is None and time.time() < deadline:
        if proc.poll() is not None:
            break
        kids = [
            p for p, comm in _children(proc.pid) if "mosaic-joint-call" in comm
        ]
        if kids and os.path.exists(out):
            os.kill(kids[0], signal.SIGKILL)
            killed = kids[0]
        else:
            time.sleep(0.02)
    _, stderr = proc.communicate(timeout=120)
    assert killed is not None, stderr[-2000:]
    assert proc.returncode == 1, stderr[-2000:]
    assert "was killed by signal 9" in stderr, stderr[-2000:]
    assert "Completed!" not in stderr
    assert not os.path.exists(out)
    assert os.listdir(tmp) == []


def test_sigkill_of_the_binary_while_spilling(cohort, tmp_path):
    """K2: a SIGKILL of `mosaic-joint-call` under ``--unit-buffer-bytes 1``
    at 4 threads and 100 kb units, once its spill directory exists beside
    the output: the driver exits 1 naming the signal and leaves no output,
    no spill directory and no temp directory."""
    tmp, env = _tmp_env(tmp_path, "tmp_kill_spill")
    out = str(tmp_path / "killed_spill.vcf.gz")
    proc = subprocess.Popen(
        [
            sys.executable,
            "joint_call.py",
            out,
            "--threads",
            "4",
            "--window",
            "100000",
            "--unit-buffer-bytes",
            "1",
        ],
        cwd=L.REPO_ROOT,
        env=L.test_env(env),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )
    killed = None
    spill_seen: List[str] = []
    deadline = time.time() + 120
    while killed is None and time.time() < deadline:
        if proc.poll() is not None:
            break
        spill_seen = L.spill_dirs_beside(out)
        kids = [
            p for p, comm in _children(proc.pid) if "mosaic-joint-call" in comm
        ]
        if kids and spill_seen:
            os.kill(kids[0], signal.SIGKILL)
            killed = kids[0]
        else:
            time.sleep(0.005)
    _, stderr = proc.communicate(timeout=120)
    assert killed is not None, (
        "the run ended (or timed out) before a spill directory appeared "
        "beside the output; nothing was killed:\n" + stderr[-2000:]
    )
    print(f"killed {killed} with {spill_seen} present")
    assert proc.returncode == 1, stderr[-2000:]
    assert "was killed by signal 9" in stderr, stderr[-2000:]
    assert "Completed!" not in stderr
    assert not os.path.exists(out)
    assert L.spill_dirs_beside(out) == []
    assert os.listdir(tmp) == []


def test_spill_dir_option(cohort, tmp_path):
    """K2: ``--spill-dir DIR`` spills in a directory of the driver's naming
    under DIR and removes it; DIR itself stays (empty). The output equals
    the slice's golden."""
    spill = tmp_path / "spill_root"
    spill.mkdir()
    out = str(tmp_path / "spill_dir.vcf.gz")
    proc = L.run_joint_call(
        out,
        region=SLICE,
        window=100_000,
        threads=4,
        unit_buffer_bytes=1,
        spill_dir=str(spill),
        max_alt_alleles=0,
    )
    print("K2 --spill-dir:", L.joint_call_spilled(proc.stderr))
    L.assert_golden("slice_chr21_10M_12M", out, proc.stderr)
    assert spill.is_dir() and os.listdir(spill) == []
    assert L.spill_dirs_beside(out) == []


def test_gate_and_stale_after_removal(cohort):
    """After `mosaic remove` and before the aggregate, the gate refuses
    (exit 2, its message, no output); `--allow-stale` writes the stale
    golden with the warning and the tombstone line."""
    L.run_remove(["HG00100"])
    work = cohort["work"]
    out = os.path.join(work, "gate.vcf.gz")
    proc = L.run_joint_call(out, check=False)
    assert proc.returncode == 2, proc.stderr[-2000:]
    assert not os.path.exists(out)
    assert re.search(
        r"Error: the latest aggregation run \(agg_run_id=\d+\) does not "
        r"cover the completed ingest batches \(not aggregated: \[\d+\]; "
        r"aggregated but not complete: \[\]\): run `mosaic aggregate`",
        proc.stderr,
    ), proc.stderr[-2000:]
    assert (
        "Refusing to joint-call stale aggregates; pass --allow-stale to "
        "override."
    ) in proc.stderr
    # K4: a removed sample is refused (exit 2, before the gate: the store
    # is stale and the message is the subset's, not the gate's).
    proc = L.run_joint_call(out, check=False, samples=["HG00099", "HG00100"])
    assert proc.returncode == 2, proc.stderr[-2000:]
    assert (
        "argument --samples: removed (samples.status = 'removed'): HG00100"
    ) in proc.stderr, proc.stderr[-2000:]
    assert "Refusing to joint-call" not in proc.stderr
    assert not os.path.exists(out)
    r = _runs(work, "stale", "stale_after_remove_HG00100", allow_stale=True)
    err = r["stderr_10000000"]
    assert "Warning (--allow-stale):" in err
    assert "Excluding 1 tombstoned sample(s)" in err
    assert "stale aggregates (--allow-stale)" in err
    # K4: subsets of the stale cohort (HG00100's batch-mate among them)
    # against the projection of its whole output.
    whole = _text(os.path.join(work, "stale.rs10000000.vcf.gz"))
    for names in (["HG00099", "HG00245"], ["HG00099"]):
        _subset_runs(
            work, "k4stale", whole, names, caps=(0, 3), allow_stale=True
        )
    # K5: both options under a cap of 3, on the stale cohort.
    _k5_runs(
        work,
        "k5stale",
        whole,
        options=((True, 1),),
        caps=(3,),
        allow_stale=True,
    )


def test_after_removal_byte_for_byte(cohort):
    """The golden after the aggregate that applies the removal: the
    removed sample's batch's rewritten files are brotli and the joint caller
    reads them."""
    L.run_aggregate()
    for table in ("sample_site_info", "sample_ref_blocks"):
        codecs = _sample_files_codecs(table)
        assert codecs and all(c == {"BROTLI"} for c in codecs.values()), (
            table,
            codecs,
        )
    r = _runs(cohort["work"], "removed", "removed_HG00100_aggregated")
    assert "Excluding 1 tombstoned sample(s)" in r["stderr_10000000"]
    # K4: subsets after the aggregate that applies the removal.
    whole = _text(os.path.join(cohort["work"], "removed.rs10000000.vcf.gz"))
    for names in (["HG00099", "HG00245"], ["HG00101", "HG00102"]):
        _subset_runs(cohort["work"], "k4removed", whole, names, caps=(0, 3))
    # K5: both options under a subset after the aggregate.
    _k5_runs(
        cohort["work"],
        "k5removed",
        whole,
        names=["HG00099", "HG00245"],
        options=((True, 1),),
        caps=(0,),
    )


def test_zstd_file_refused(cohort):
    """A per-sample file a removal rewrote in zstd (a lake from before the
    codec property): the joint caller refuses it naming the file and the
    codec. (The earlier Python engine read such a file; nothing does
    now.)"""
    catalog = L.get_catalog()
    t = catalog.load_table(f"{L.TEST_NAMESPACE}.sample_site_info")
    with t.transaction() as txn:
        txn.set_properties(
            {
                "write.parquet.compression-codec": "zstd",
                "write.parquet.compression-level": "3",
            }
        )
    L.run_remove(["HG00101"])
    L.run_aggregate()
    zstd = [
        uri
        for uri, c in _sample_files_codecs("sample_site_info").items()
        if "ZSTD" in c
    ]
    assert zstd
    work = cohort["work"]
    out = os.path.join(work, "zstd.vcf.gz")
    proc = L.run_joint_call(out, check=False)
    assert proc.returncode == 1
    assert "compressed with ZSTD" in proc.stderr, proc.stderr[-2000:]
    assert any(uri in proc.stderr for uri in zstd)
    assert not os.path.exists(out)


# ── other cohorts (each resets the stores) ───────────────────────────────────


@pytest.mark.parametrize("no_call", [False, True], ids=["only", "with_g5"])
def test_banding_cross_shard_and_no_call_blocks(
    docker_stack, tmp_path, no_call
):
    """A `--ref-gq-bands 20,60` batch of the input_gvcfs4 samples over
    chr21:10.0-10.3M whose shard boundary falls inside their deletion
    GAAA>G at 10,001,907 (the cross-shard deletion split): alone
    (``banded_cross_shard_only``), and with HG00096 + HG00097 over
    chr21:5.0-10.3M (their ./. blocks; ``..._with_hg00096_hg00097``, skipped
    when test_data/input_gvcfs5 lacks them)."""
    if no_call and not (os.path.exists(HG00096) and os.path.exists(HG00097)):
        pytest.skip(f"HG00096 / HG00097 not present: {HG00096}, {HG00097}")
    work = str(tmp_path)
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    banded = []
    for j, src in enumerate(L.GVCFS):
        dst = os.path.join(work, f"band_s{j}.g.vcf.gz")
        L.subset_gvcf(src, CROSS_REGION, dst)
        banded.append(dst)
    proc = L.run_ingest(
        banded,
        region=CROSS_REGION,
        shard_size=CROSS_SHARD_SIZE,
        ref_gq_bands="20,60",
    )
    assert "Added cross-shard deletion-split spans to 2" in proc.stderr
    if no_call:
        subsets = []
        for j, src in enumerate((HG00096, HG00097)):
            dst = os.path.join(work, f"g5_s{j}.g.vcf.gz")
            L.subset_gvcf(src, NO_CALL_REGION, dst)
            subsets.append(dst)
        L.run_ingest(subsets, region=NO_CALL_REGION, shard_size=1_000_000)
    L.run_aggregate()
    # The scenario is named by the inputs it ran with.
    golden = (
        "banded_cross_shard_with_hg00096_hg00097"
        if no_call
        else "banded_cross_shard_only"
    )
    r = _runs(work, "mixed", golden, windows=(10_000_000, 100_000))
    assert r["counts_10000000"]["records"] > (500 if no_call else 100)
    if no_call:
        # Some HG00096 / HG00097 calls are written ./. from a no-call block.
        text = _text(os.path.join(work, "mixed.rs10000000.vcf.gz"))
        assert re.search(r"\t\./\.:[0-9]", text)


def test_golden_region(docker_stack, tmp_path):
    """The golden e2e's input (the input_gvcfs4 samples over
    chr21:10-12M, one batch)."""
    work = str(tmp_path)
    L.reset_state()
    gvcfs = []
    for j, src in enumerate(L.GVCFS):
        dst = os.path.join(work, f"s{j}.g.vcf.gz")
        L.subset_gvcf(src, GOLDEN_REGION, dst)
        gvcfs.append(dst)
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(gvcfs, region=GOLDEN_REGION, threads=2)
    L.run_aggregate(fan_in=100, threads=2)
    r = _runs(
        work, "golden", "golden_region_chr21_10M_12M", windows=(10_000_000, 0)
    )
    assert r["counts_10000000"]["records"] > 3000
