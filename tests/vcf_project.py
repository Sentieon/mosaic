# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The ALT cap and the sample subset as projections of VCF text:
``project(text, n)``
computes, from the uncapped joint-call output alone, what ``joint_call.py
--max-alt-alleles n`` must write, and ``project_samples(text, names)``,
from the whole-cohort uncapped output alone, what ``joint_call.py --samples
NAMES --max-alt-alleles 0`` must write. No engine code is used (nor
``joint_call.py``): this module is the test side's independent reading of
the contracts. A subset under a cap is ``project(project_samples(whole,
S), n)``.

The one PL rule of both: a projected PL is the kept
genotypes' entries (VCF genotype order), and when the source PL held a 0
and the kept entries are all > 0, their minimum is subtracted
(``project_pl``). A ``.`` stays ``.``.

The cap, with N the cap (0 = none) and n a record's ALT count (``*`` is an
ALT like any other):

- n <= N, or N = 0: the record is unchanged.
- n > N: kept = the N ALTs with the largest AC (the counts of the record's
  own GT columns, which must equal its INFO/AC), ties to the earlier ALT,
  in their original order. ALT: the kept ALTs. GT: ``./.`` when either
  allele is a dropped ALT, else re-indexed (``./.`` stays). AD, F1R2,
  F2R1: the entries of REF and the kept ALTs. PL: the entries of the
  genotypes over REF and the kept ALTs, under the PL rule. DP unchanged.
  INFO: AC, AN, AF recomputed from the projected GTs (AN the called
  alleles, ``AF = round(AC / AN, 3)`` as ``str(float)``, ``0.0`` at AN =
  0), every other key and QUAL unchanged, in the same order.
- The header gains MAX_ALT_ALLELES_HEADER when N > 0, right before the
  first ``##contig`` line (or the ``#CHROM`` line when there is none):
  where the driver writes it.

The subset S (names of the ``#CHROM`` line; an unknown or repeated name is
an error):

- The header: every line unchanged but ``#CHROM``, whose sample columns
  are S's in the header's order (not the order given).
- Every record stays, its sample columns cut to S's. ``*`` (always the
  last ALT) leaves the ALT list iff no column of S names it in its GT; no
  GT index moves. AD, F1R2, F2R1 then lose its entry and PL its genotypes
  (the PL rule). A record whose only ALT was ``*`` gets ALT ``.`` and no AC
  or AF key (as bcftools writes a record without ALTs; K5).
- A FORMAT key (not GT) is dropped when every column of S has ``.`` for it.
- AC, AN, AF from S's GTs (as above); QUAL and every other INFO key the
  cohort's.

The trim and --min-ac add two projections, applied in the order subset, trim, cap,
min-ac, each on the previous one's output: ``project_min_ac(project(
project_trim(project_samples(whole, S)), N), K)``, each step optional.

The trim (``project_trim``, ``joint_call.py --trim-alt-alleles``): in each
record an ALT whose AC among its GT columns is 0 leaves it (which must
equal INFO/AC). The other ALTs keep their order; the GTs are re-indexed
(none names a dropped ALT, so none becomes ``./.``); AD, F1R2, F2R1 keep
the entries of REF and the kept ALTs and PL the kept genotypes' entries
under the PL rule; a ``.`` stays ``.``. AC and AF lose the dropped
entries; AN, QUAL and every other INFO key are unchanged. A record left
without ALTs gets ALT ``.`` and loses its AC and AF keys (every other key
keeps its place), as bcftools writes such a record; a source record that
has no ALT has no AC or AF and is left as it is. The header gains
TRIM_ALT_ALLELES_HEADER right before the first ``##contig`` line.

The minimum AC (``project_min_ac``, ``joint_call.py --min-ac K``): a
record is kept iff the ALT observations of its GT columns (the called
alleles other than REF, i.e. the sum of its INFO/AC) are at least K. K = 0:
the text unchanged; K > 0: MIN_AC_HEADER right before the first
``##contig`` line.
"""

import re
from typing import Dict, List, Sequence, Tuple

# The cap's header line (the driver's ``MAX_ALT_ALLELES_HEADER`` in
# joint_call.py: the same text, by value, not by import).
MAX_ALT_ALLELES_HEADER = "##mosaic_max_alt_alleles={}"
# K5's header lines (joint_call.py's TRIM_ALT_ALLELES_HEADER and
# MIN_AC_HEADER, by value).
TRIM_ALT_ALLELES_HEADER = "##mosaic_trim_alt_alleles=true"
MIN_AC_HEADER = "##mosaic_min_ac={}"

# The FORMAT keys the joint caller writes; any other key in a capped
# record is an error (the contract does not say how to project it).
_PER_ALLELE = ("AD", "F1R2", "F2R1")
_UNCHANGED = ("DP",)

_GT_SPLIT = re.compile(r"[/|]")


def _empty_counts() -> Dict[str, int]:
    return {
        "records_capped": 0,
        "genotypes_no_called": 0,
        "dropped_alt_observations": 0,
        "kept_alt_observations_lost": 0,
    }


def gt_alleles(gt: str) -> List[int]:
    """The called allele indices of a GT (``.`` skipped)."""
    return [int(a) for a in _GT_SPLIT.split(gt) if a not in (".", "")]


def gt_index(a: int, b: int) -> int:
    """The PL index of the diploid genotype (a, b) (VCF order)."""
    lo, hi = (a, b) if a <= b else (b, a)
    return hi * (hi + 1) // 2 + lo


def af_text(ac: int, an: int) -> str:
    """AF as the joint caller prints it: ``str(round(AC / AN, 3) + 0.0)``,
    ``0.0`` at AN = 0."""
    if an == 0:
        return "0.0"
    return str(round(ac / an, 3) + 0.0)


def project_pl(source: List[int], kept: List[int]) -> List[int]:
    """The one PL rule (K4's decision 1): ``kept`` (the kept genotypes'
    entries of ``source``) less their minimum when ``source`` held a 0 and
    every kept entry is > 0; else ``kept`` as it is."""
    if 0 in source and kept and all(p > 0 for p in kept):
        m = min(kept)
        return [p - m for p in kept]
    return list(kept)


def kept_alts(ac: List[int], n: int) -> List[int]:
    """The 0-based ALT indices kept: the n with the largest AC, ties to
    the earlier ALT, in their original order."""
    order = sorted(range(len(ac)), key=lambda i: (-ac[i], i))
    return sorted(order[:n])


def _project_record(
    line: str, n: int, counts: Dict[str, int]
) -> Tuple[str, bool]:
    f = line.split("\t")
    alt = f[4]
    alts = [] if alt == "." else alt.split(",")
    if len(alts) <= n:
        return line, False
    if len(f) < 10:
        raise ValueError(f"a record over the cap without samples: {line}")
    info = f[7].split(";")
    info_ac = None
    for kv in info:
        if kv.startswith("AC="):
            info_ac = [int(x) for x in kv[3:].split(",")]
    fmt = f[8].split(":")
    if fmt[0] != "GT":
        raise ValueError(f"FORMAT does not start with GT: {line}")
    for key in fmt[1:]:
        if key not in _PER_ALLELE + _UNCHANGED + ("PL",):
            raise ValueError(f"FORMAT key {key} has no projection: {line}")
    samples = [c.split(":") for c in f[9:]]
    # The uncapped AC from the GT columns: the ranking's input.
    ac = [0] * len(alts)
    for s in samples:
        for a in gt_alleles(s[0]):
            if a > 0:
                ac[a - 1] += 1
    if info_ac is not None and info_ac != ac:
        raise AssertionError(
            f"{f[0]}:{f[1]}: AC from the GTs {ac} != INFO/AC {info_ac}"
        )
    kept = kept_alts(ac, n)
    kept_set = set(kept)
    # Old VCF allele index -> new (REF stays 0).
    new_of = {0: 0}
    for j, k in enumerate(kept):
        new_of[k + 1] = j + 1
    keep_alleles = [0] + [k + 1 for k in kept]
    counts["records_capped"] += 1
    counts["dropped_alt_observations"] += sum(
        c for i, c in enumerate(ac) if i not in kept_set
    )
    out_samples = []
    new_ac = [0] * len(kept)
    an = 0
    for s in samples:
        vals = list(s)
        gt = vals[0]
        called = gt_alleles(gt)
        if any(a not in new_of for a in called):
            counts["genotypes_no_called"] += 1
            counts["kept_alt_observations_lost"] += sum(
                1 for a in called if a > 0 and a in new_of
            )
            vals[0] = "./."
        elif called:
            parts = re.split(r"([/|])", gt)
            vals[0] = "".join(
                p if p in ("/", "|", ".") else str(new_of[int(p)])
                for p in parts
            )
            for a in called:
                an += 1
                if a > 0:
                    new_ac[new_of[a] - 1] += 1
        for k, key in enumerate(fmt[1:], start=1):
            if k >= len(vals) or vals[k] == ".":
                continue
            if key in _PER_ALLELE:
                v = vals[k].split(",")
                vals[k] = ",".join(v[i] for i in keep_alleles)
            elif key == "PL":
                v = [int(x) for x in vals[k].split(",")]
                pl: List[int] = []
                for jj in range(len(keep_alleles)):
                    for ii in range(jj + 1):
                        pl.append(
                            v[gt_index(keep_alleles[ii], keep_alleles[jj])]
                        )
                vals[k] = ",".join(str(p) for p in project_pl(v, pl))
        out_samples.append(":".join(vals))
    new_info = []
    for kv in info:
        if kv.startswith("AC="):
            kv = "AC=" + ",".join(str(c) for c in new_ac)
        elif kv.startswith("AN="):
            kv = f"AN={an}"
        elif kv.startswith("AF="):
            kv = "AF=" + ",".join(af_text(c, an) for c in new_ac)
        new_info.append(kv)
    f[4] = ",".join(alts[k] for k in kept)
    f[7] = ";".join(new_info)
    return "\t".join(f[:9] + out_samples), True


def project(text: str, n: int) -> Tuple[str, Dict[str, int]]:
    """The VCF text ``joint_call.py --max-alt-alleles n`` must write, from
    the uncapped text, and the cap's counts: the records capped, the
    genotypes written ``./.`` by the cap (called before, ``./.`` after),
    the ALT observations on the dropped alleles (their uncapped AC summed)
    and the kept-allele ALT observations lost with those genotypes. n = 0:
    the text unchanged and every count 0."""
    if n < 0:
        raise ValueError(f"cap {n} is negative")
    counts = _empty_counts()
    if n == 0:
        return text, counts
    lines = text.split("\n")
    out: List[str] = []
    header_done = False
    for line in lines:
        if line.startswith("#"):
            if not header_done and (
                line.startswith("##contig=") or line.startswith("#CHROM")
            ):
                out.append(MAX_ALT_ALLELES_HEADER.format(n))
                header_done = True
            out.append(line)
        elif line:
            out.append(_project_record(line, n, counts)[0])
        else:
            out.append(line)
    return "\n".join(out), counts


def _counts_info(info: List[str], gts: List[str], n_alts: int) -> str:
    """INFO with AC / AN / AF recomputed from ``gts`` over ``n_alts`` ALTs
    (none: no AC and no AF key; K5), every other key unchanged."""
    ac = [0] * n_alts
    an = 0
    for gt in gts:
        for a in gt_alleles(gt):
            an += 1
            if a > 0:
                ac[a - 1] += 1
    out = []
    for kv in info:
        if n_alts == 0 and (kv.startswith("AC=") or kv.startswith("AF=")):
            continue
        if kv.startswith("AC="):
            kv = "AC=" + ",".join(str(c) for c in ac)
        elif kv.startswith("AN="):
            kv = f"AN={an}"
        elif kv.startswith("AF="):
            kv = "AF=" + ",".join(af_text(c, an) for c in ac)
        out.append(kv)
    return ";".join(out)


def _subset_record(line: str, cols: List[int]) -> str:
    f = line.split("\t")
    if len(f) < 10:
        raise ValueError(f"a record without samples: {line}")
    alts = [] if f[4] == "." else f[4].split(",")
    if "*" in alts[:-1]:
        raise ValueError(f"* is not the last ALT: {line}")
    fmt = f[8].split(":")
    if fmt[0] != "GT":
        raise ValueError(f"FORMAT does not start with GT: {line}")
    for key in fmt[1:]:
        if key not in _PER_ALLELE + _UNCHANGED + ("PL",):
            raise ValueError(f"FORMAT key {key} has no projection: {line}")
    samples = [f[9 + c].split(":") for c in cols]
    n_alleles = len(alts) + 1
    drop_star = bool(alts) and alts[-1] == "*"
    if drop_star:
        star = len(alts)
        drop_star = not any(star in gt_alleles(s[0]) for s in samples)
    if drop_star:
        alts = alts[:-1]
        keep = n_alleles - 1
        n_gt = keep * (keep + 1) // 2
        for s in samples:
            for k, key in enumerate(fmt[1:], start=1):
                if k >= len(s) or s[k] == ".":
                    continue
                v = s[k].split(",")
                if key in _PER_ALLELE:
                    s[k] = ",".join(v[:keep])
                elif key == "PL":
                    src = [int(x) for x in v]
                    s[k] = ",".join(
                        str(p) for p in project_pl(src, src[:n_gt])
                    )
    # A FORMAT key every column of S has `.` for is dropped.
    keys = [0] + [
        k
        for k in range(1, len(fmt))
        if any(k < len(s) and s[k] != "." for s in samples)
    ]
    f[4] = ",".join(alts) if alts else "."
    f[7] = _counts_info(f[7].split(";"), [s[0] for s in samples], len(alts))
    f[8] = ":".join(fmt[k] for k in keys)
    out = [":".join(s[k] for k in keys) for s in samples]
    return "\t".join(f[:9] + out)


def _insert_header_line(text: str, header_line: str, record) -> str:
    """``text`` with ``header_line`` right before the first ``##contig``
    line (or the ``#CHROM`` line when there is none) and each record line
    passed through ``record`` (None: the record leaves the text)."""
    out: List[str] = []
    header_done = False
    for line in text.split("\n"):
        if line.startswith("#"):
            if not header_done and (
                line.startswith("##contig=") or line.startswith("#CHROM")
            ):
                out.append(header_line)
                header_done = True
            out.append(line)
        elif line:
            new = record(line)
            if new is not None:
                out.append(new)
        else:
            out.append(line)
    return "\n".join(out)


def _gt_ac(samples: List[List[str]], n_alts: int) -> List[int]:
    """AC per ALT from the GT columns."""
    ac = [0] * n_alts
    for s in samples:
        for a in gt_alleles(s[0]):
            if a > 0:
                ac[a - 1] += 1
    return ac


def _trim_record(line: str, counts: Dict[str, int]) -> str:
    f = line.split("\t")
    alts = [] if f[4] == "." else f[4].split(",")
    if not alts:
        return line
    if len(f) < 10:
        raise ValueError(f"a record without samples: {line}")
    fmt = f[8].split(":")
    if fmt[0] != "GT":
        raise ValueError(f"FORMAT does not start with GT: {line}")
    for key in fmt[1:]:
        if key not in _PER_ALLELE + _UNCHANGED + ("PL",):
            raise ValueError(f"FORMAT key {key} has no projection: {line}")
    info = f[7].split(";")
    samples = [c.split(":") for c in f[9:]]
    ac = _gt_ac(samples, len(alts))
    for kv in info:
        if kv.startswith("AC=") and [int(x) for x in kv[3:].split(",")] != ac:
            raise AssertionError(
                f"{f[0]}:{f[1]}: AC from the GTs {ac} != INFO/{kv}"
            )
    kept = [i for i, c in enumerate(ac) if c > 0]
    if len(kept) == len(alts):
        return line
    counts["records_trimmed"] += 1
    counts["alts_dropped"] += len(alts) - len(kept)
    new_of = {0: 0}
    for j, k in enumerate(kept):
        new_of[k + 1] = j + 1
    keep_alleles = [0] + [k + 1 for k in kept]
    out_samples = []
    for s in samples:
        vals = list(s)
        gt = vals[0]
        if gt_alleles(gt):
            # No GT names a dropped ALT (its AC is 0): re-indexed.
            parts = re.split(r"([/|])", gt)
            vals[0] = "".join(
                p if p in ("/", "|", ".") else str(new_of[int(p)])
                for p in parts
            )
        for k, key in enumerate(fmt[1:], start=1):
            if k >= len(vals) or vals[k] == ".":
                continue
            if key in _PER_ALLELE:
                v = vals[k].split(",")
                vals[k] = ",".join(v[i] for i in keep_alleles)
            elif key == "PL":
                src = [int(x) for x in vals[k].split(",")]
                pl = [
                    src[gt_index(keep_alleles[ii], keep_alleles[jj])]
                    for jj in range(len(keep_alleles))
                    for ii in range(jj + 1)
                ]
                vals[k] = ",".join(str(p) for p in project_pl(src, pl))
        out_samples.append(":".join(vals))
    new_info = []
    for kv in info:
        if not kept and (kv.startswith("AC=") or kv.startswith("AF=")):
            # A record left without ALTs carries no AC and no AF (K5).
            continue
        if kv.startswith("AC=") or kv.startswith("AF="):
            v = kv[3:].split(",")
            kv = kv[:3] + ",".join(v[i] for i in kept)
        new_info.append(kv)
    f[4] = ",".join(alts[k] for k in kept) if kept else "."
    f[7] = ";".join(new_info)
    return "\t".join(f[:9] + out_samples)


def project_trim(text: str) -> Tuple[str, Dict[str, int]]:
    """The VCF text ``joint_call.py --trim-alt-alleles`` must write, from
    the untrimmed text (the module's docstring), and the trim's counts: the
    records it changed (``records_trimmed``) and the ALTs it dropped
    (``alts_dropped``)."""
    counts = {"records_trimmed": 0, "alts_dropped": 0}
    out = _insert_header_line(
        text, TRIM_ALT_ALLELES_HEADER, lambda x: _trim_record(x, counts)
    )
    return out, counts


def alt_observations(line: str) -> int:
    """A record's ALT observations: its GT columns' called alleles other
    than REF (the sum of its INFO/AC)."""
    f = line.split("\t")
    return sum(1 for c in f[9:] for a in gt_alleles(c.split(":")[0]) if a > 0)


def project_min_ac(text: str, k: int) -> Tuple[str, Dict[str, int]]:
    """The VCF text ``joint_call.py --min-ac k`` must write, from the text
    without it: the records whose ALT observations (``alt_observations``)
    are at least k, and the records left out (``records_left_out``). k = 0:
    the text unchanged; a negative k is a ValueError."""
    if k < 0:
        raise ValueError(f"--min-ac {k} is negative")
    counts = {"records_left_out": 0}
    if k == 0:
        return text, counts

    def keep(line: str):
        if alt_observations(line) >= k:
            return line
        counts["records_left_out"] += 1
        return None

    return _insert_header_line(text, MIN_AC_HEADER.format(k), keep), counts


def project_samples(text: str, names: Sequence[str]) -> str:
    """The VCF text ``joint_call.py --samples NAMES --max-alt-alleles 0``
    must write, from the whole-cohort uncapped text (the module's
    docstring). An empty, unknown or repeated name is a ValueError."""
    names = list(names)
    if not names:
        raise ValueError("an empty subset")
    seen = set()
    for n in names:
        if n in seen:
            raise ValueError(f"sample {n} given twice")
        seen.add(n)
    out: List[str] = []
    cols: List[int] = []
    for line in text.split("\n"):
        if line.startswith("#CHROM"):
            f = line.split("\t")
            header = f[9:]
            unknown = [n for n in names if n not in header]
            if unknown:
                raise ValueError(f"unknown sample(s): {unknown}")
            cols = [i for i, n in enumerate(header) if n in seen]
            out.append("\t".join(f[:9] + [header[i] for i in cols]))
        elif line.startswith("#") or not line:
            out.append(line)
        else:
            if not cols:
                raise ValueError("a record before the #CHROM line")
            out.append(_subset_record(line, cols))
    return "\n".join(out)
