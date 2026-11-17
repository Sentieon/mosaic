# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Stream a VCF through the projection chain of ``tests/vcf_project.py``.

``vcf_project``'s functions take a whole VCF text. At cohort scale that
text does not fit in memory (a 1 Mbp slice of 3,202 samples is several
GB, and one record can be hundreds of MB), so this wrapper feeds it the
header once and then the records in chunks, through the module's public
functions only. The result is the text the whole-text chain

    project_min_ac(project(project_trim(project_samples(whole, S)), N), K)

gives (each step only when its option is set), which is what
``joint_call.py [--samples S] [--trim-alt-alleles] --max-alt-alleles N
[--min-ac K]`` must write, from the whole-cohort uncapped output
(``joint_call.py --max-alt-alleles 0``). No engine code is used.

    gzip -dc whole.vcf.gz | python tests/vcf_project_stream.py \\
        [--samples NAME[,NAME...] | --samples-file FILE] \\
        [--trim-alt-alleles] [--max-alt-alleles N] [--min-ac K] > want.vcf

The cap's default here is 0 (no cap): the input is the uncapped output,
and the expected output of a default ``joint_call.py`` run is
``--max-alt-alleles 6``. Stderr gets the closing lines the driver prints
for the same options, so the two can be compared as text:
``Completed! Total processed: R variants`` and, per option set, the
trim's, the cap's and --min-ac's lines.
"""

import argparse
import os
import sys
from typing import Any, BinaryIO, Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import vcf_project as P  # noqa: E402

# The records fed to the projections at a time. A chunk's memory is its
# text several times over; at 3,202 samples a record is about 130 kB.
DEFAULT_CHUNK_RECORDS = 200


def _chain(
    text: str,
    names: Optional[Sequence[str]],
    trim: bool,
    cap: int,
    min_ac: int,
    counts: Dict[str, Dict[str, int]],
) -> str:
    """``text`` through the chain, each step's counts added to ``counts``
    (``trim``, ``cap``, ``min_ac``: a key only when its option is set)."""
    if names is not None:
        text = P.project_samples(text, names)
    if trim:
        text, c = P.project_trim(text)
        _add(counts, "trim", c)
    if cap > 0:
        text, c = P.project(text, cap)
        _add(counts, "cap", c)
    if min_ac > 0:
        text, c = P.project_min_ac(text, min_ac)
        _add(counts, "min_ac", c)
    return text


def _add(
    counts: Dict[str, Dict[str, int]], key: str, new: Dict[str, int]
) -> None:
    have = counts.setdefault(key, {})
    for k, v in new.items():
        have[k] = have.get(k, 0) + v


def closing_lines(
    records: int, cap: int, min_ac: int, counts: Dict[str, Dict[str, int]]
) -> List[str]:
    """The driver's closing lines for these options, by value (the text
    of ``joint_call.py``'s ``trim_line``, ``cap_line`` and ``min_ac_line``,
    in the order the options apply; the cross-check's lines are not the
    projection's to give)."""
    out = [f"Completed! Total processed: {records} variants"]
    if "trim" in counts:
        t = counts["trim"]
        out.append(
            "ALT trim (--trim-alt-alleles): "
            f"{t['records_trimmed']} records trimmed, "
            f"{t['alts_dropped']} ALT alleles with AC 0 dropped"
        )
    if "cap" in counts:
        c = counts["cap"]
        out.append(
            f"ALT cap (--max-alt-alleles {cap}): "
            f"{c['records_capped']} records capped, "
            f"{c['genotypes_no_called']} genotypes written ./., "
            f"{c['dropped_alt_observations']} ALT observations on dropped "
            f"alleles, {c['kept_alt_observations_lost']} kept-allele ALT "
            "observations lost with those genotypes"
        )
    if "min_ac" in counts:
        m = counts["min_ac"]
        out.append(
            f"Minimum AC (--min-ac {min_ac}): "
            f"{m['records_left_out']} records left out (fewer ALT "
            "observations in their written genotypes)"
        )
    return out


def stream(
    src: BinaryIO,
    dst: BinaryIO,
    names: Optional[Sequence[str]] = None,
    trim: bool = False,
    cap: int = 0,
    min_ac: int = 0,
    chunk_records: int = DEFAULT_CHUNK_RECORDS,
) -> Tuple[int, Dict[str, Dict[str, int]]]:
    """``src`` (a VCF, uncompressed) through the chain into ``dst``.
    Returns the records written and the counts per option set. The header
    goes through the chain once, alone; each chunk of records goes through
    it behind the ``#CHROM`` line (which ``project_samples`` needs), and
    the header lines of the chunk's result are dropped."""
    if chunk_records < 1:
        raise ValueError(f"chunk_records {chunk_records} is below 1")
    counts: Dict[str, Dict[str, int]] = {}
    # A key is present whenever its option is set, records or not.
    if trim:
        _add(counts, "trim", {"records_trimmed": 0, "alts_dropped": 0})
    if cap > 0:
        _add(counts, "cap", P.project("", cap)[1])
    if min_ac > 0:
        _add(counts, "min_ac", {"records_left_out": 0})
    header: List[str] = []
    chrom_line: Optional[str] = None
    chunk: List[str] = []
    records = 0

    def flush() -> int:
        if not chunk:
            return 0
        assert chrom_line is not None
        text = "\n".join([chrom_line] + chunk) + "\n"
        out = _chain(text, names, trim, cap, min_ac, counts)
        kept = [x for x in out.split("\n") if x and not x.startswith("#")]
        if kept:
            dst.write(("\n".join(kept) + "\n").encode())
        chunk.clear()
        return len(kept)

    for raw in src:
        line = raw.decode().rstrip("\n")
        if chrom_line is None:
            if not line.startswith("#"):
                raise ValueError("a record before the #CHROM line")
            header.append(line)
            if line.startswith("#CHROM"):
                chrom_line = line
                text = "\n".join(header) + "\n"
                # The header alone: the option lines and the subset's
                # #CHROM line, no counts (there is no record).
                dst.write(_chain(text, names, trim, cap, min_ac, {}).encode())
            continue
        if not line:
            continue
        chunk.append(line)
        if len(chunk) >= chunk_records:
            records += flush()
    records += flush()
    if chrom_line is None:
        raise ValueError("no #CHROM line")
    return records, counts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Stream the whole-cohort uncapped joint-call VCF (stdin) "
            "through tests/vcf_project.py's projection chain (subset, "
            "trim, cap, min-ac) to stdout; the driver's closing lines for "
            "the same options go to stderr."
        )
    )
    g = p.add_mutually_exclusive_group()
    g.add_argument("--samples", metavar="NAME[,NAME...]")
    g.add_argument("--samples-file", metavar="FILE")
    p.add_argument("--trim-alt-alleles", action="store_true")
    p.add_argument(
        "--max-alt-alleles",
        type=int,
        default=0,
        metavar="N",
        help="the cap to project to (default: 0, none)",
    )
    p.add_argument("--min-ac", type=int, default=0, metavar="K")
    p.add_argument(
        "--chunk-records",
        type=int,
        default=DEFAULT_CHUNK_RECORDS,
        metavar="R",
        help=f"records per chunk (default: {DEFAULT_CHUNK_RECORDS})",
    )
    return p


def subset_names(args: Any) -> Optional[List[str]]:
    """As ``joint_call.py`` reads them: comma-separated, or one per line
    (blank lines ignored), each stripped; None without either option."""
    if args.samples is not None:
        return [x.strip() for x in args.samples.split(",") if x.strip()]
    if args.samples_file is not None:
        with open(args.samples_file) as fh:
            return [x.strip() for x in fh if x.strip()]
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.max_alt_alleles < 0 or args.min_ac < 0:
        print(
            "Error: a negative --max-alt-alleles or --min-ac", file=sys.stderr
        )
        return 2
    records, counts = stream(
        sys.stdin.buffer,
        sys.stdout.buffer,
        subset_names(args),
        args.trim_alt_alleles,
        args.max_alt_alleles,
        args.min_ac,
        args.chunk_records,
    )
    sys.stdout.buffer.flush()
    for line in closing_lines(
        records, args.max_alt_alleles, args.min_ac, counts
    ):
        print(line, file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
