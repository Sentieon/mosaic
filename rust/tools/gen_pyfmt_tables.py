# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Generate the expectation tables of the joint caller's ``pyfmt.rs``.

Writes ``pyfmt_str.tsv`` (``repr(x)``), ``pyfmt_round.tsv``
(``repr(round(x, 3) + 0.0)``) and ``pyfmt_2f.tsv`` (``'%4.2f' % x``) into
OUTDIR, one ``<hex bits of x>\\t<Python's text>`` line per distinct x,
sorted by bits. The inputs, the same for the three tables:

- every distinct QUAL, QD, MQ, FS, SOR, AF and InbreedingCoeff value of a
  joint-called VCF (the decompressed text, parsed with ``float``), at most
  ``--per-field`` of each field, evenly spaced over its sorted values;
- with ``--site-info``, the unrounded values the Python engine rounds,
  recomputed from PostgreSQL's ``site_info`` / ``ref_blocks`` (QUAL, QD,
  MQ, FS, SOR, InbreedingCoeff, through the removed Python engine's own
  formulas: ``odds_ratio`` is kept here since the engine's removal),
  and AF from the VCF's AC / AN, likewise at most ``--per-field`` each;
- exact decimal ties, powers of two, the exponent boundaries, integral
  and negative values, ``-0.0`` and values in ``(-0.0005, 0)``.

Run with the project's Python environment and PYTHONPATH at the repo
root, under the
``MOSAIC_DB_*`` of the store that was joint-called:
    python rust/tools/gen_pyfmt_tables.py VCF OUTDIR [--site-info]
"""

import argparse
import gzip
import math
import os
import struct
from typing import Dict, Iterable, List, Set

FIELDS = ("QD", "MQ", "FS", "SOR", "AF", "InbreedingCoeff")


def bits(x: float) -> str:
    return "%016x" % struct.unpack("<Q", struct.pack("<d", x))[0]


def odds_ratio(table: List[int]) -> float:
    """The removed Python engine's symmetric odds ratio (SOR before its
    rounding), on the table with 1 added to each cell."""
    t0 = table[0] + 1
    t1 = table[1] + 1
    t2 = table[2] + 1
    t3 = table[3] + 1
    ratio = (t0 * t3) / (t1 * t2) + (t1 * t2) / (t0 * t3)
    ref = min(t0, t1) / max(t0, t1)
    alt = min(t2, t3) / max(t2, t3)
    return math.log(ratio * ref / alt)


def vcf_values(path: str) -> Dict[str, Set[float]]:
    """The distinct printed values per field, and the raw AC / AN ratios
    (under ``AF_raw``)."""
    out: Dict[str, Set[float]] = {k: set() for k in ("QUAL", "AF_raw")}
    for k in FIELDS:
        out[k] = set()
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            cols = line.split("\t", 8)
            if cols[5] != ".":
                out["QUAL"].add(float(cols[5]))
            info = dict(
                kv.split("=", 1) for kv in cols[7].split(";") if "=" in kv
            )
            for k in FIELDS:
                if k in info:
                    for v in info[k].split(","):
                        out[k].add(float(v))
            if "AC" in info and "AN" in info and int(info["AN"]) > 0:
                an = int(info["AN"])
                for c in info["AC"].split(","):
                    out["AF_raw"].add(int(c) / an)
    return out


def site_info_values() -> Dict[str, Set[float]]:
    """The unrounded INFO values ``calculate_info_fields`` rounds."""
    import psycopg2
    from scipy.stats import fisher_exact

    from mosaic.config import DB_CONFIG

    out: Dict[str, Set[float]] = {
        k: set() for k in ("QUAL", "QD", "MQ", "FS", "SOR", "IC")
    }
    fs_cache: Dict[tuple, float] = {}
    conn = psycopg2.connect(**DB_CONFIG)
    with conn, conn.cursor() as cur:
        cur.execute(
            "SELECT s.qual, s.dp, s.mqsq_1, s.mqsq_2, s.sb, s.gtc, "
            "COALESCE(r.dp, 0) FROM site_info s LEFT JOIN ref_blocks r "
            "ON r.chromosome_index = s.chromosome_index "
            "AND r.position = s.position"
        )
        for qual, dp, mq1, mq2, sb, gtc, rb_dp in cur:
            total_dp = (dp or 0) + rb_dp
            if qual is not None:
                out["QUAL"].add(float(qual))
                if total_dp > 0:
                    out["QD"].add(qual / total_dp)
            if mq2:
                out["MQ"].add(math.sqrt((mq1 or 0) / mq2))
            if sb:
                key = tuple(sb)
                if key not in fs_cache:
                    _, pv = fisher_exact([[sb[0], sb[1]], [sb[2], sb[3]]])
                    fs_cache[key] = -10.0 * math.log10(max(pv, 1e-30))
                out["FS"].add(fs_cache[key])
                out["SOR"].add(odds_ratio(sb))
            if gtc:
                denom = gtc[0] + gtc[1] + gtc[2]
                if denom > 0:
                    p = (gtc[0] + gtc[1] * 0.5) / denom
                    s = denom * p * (1 - p) * 2.0
                    if s > 0:
                        out["IC"].add(1 - gtc[1] / s)
    conn.close()
    return out


def spread(values: Iterable[float], n: int) -> List[float]:
    """At most ``n`` of the sorted distinct values, evenly spaced."""
    v = sorted(set(values))
    if len(v) <= n:
        return v
    step = len(v) / n
    return [v[int(i * step)] for i in range(n)]


def synthetic() -> List[float]:
    xs: List[float] = [
        0.0,
        -0.0,
        0.0625,
        2.675,
        1.005,
        0.125,
        0.375,
        2.5,
        0.5,
        1.5,
        1e-5,
        9.9999e-5,
        1e-4,
        1e15,
        9.99e15,
        1e16,
        1.5e20,
        123456789012345.6,
        5e-324,
        1.7976931348623157e308,
        0.1,
        0.2,
        0.3,
        1 / 3,
        2 / 3,
        math.pi,
        math.e,
    ]
    for k in range(-50, 51):
        xs += [k + 0.0005, k - 0.0005, k + 0.005, k - 0.005, float(k)]
        xs += [k + 0.0015, k + 0.0025, k + 0.125, k + 0.375]
    for j in range(-300, 301):
        xs.append(j / 1024)
        xs.append(j / 16)
        xs.append(j / 8)
    for e in range(-1074, 1024, 23):
        xs += [2.0**e, -(2.0**e)]
    for e in range(-30, 31):
        xs += [10.0**e, -(10.0**e), 1.5 * 10.0**e, 9.99 * 10.0**e]
    # Values that round to zero from below.
    for i in range(1, 200):
        xs.append(-i * 0.0005 / 200)
    xs += [-0.0005, -0.00049999999999999, -1e-300, -5e-324]
    return xs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("vcf")
    ap.add_argument("outdir")
    ap.add_argument("--site-info", action="store_true")
    ap.add_argument("--per-field", type=int, default=250)
    args = ap.parse_args()

    xs: Set[float] = set()
    counts = {}
    for k, v in vcf_values(args.vcf).items():
        sel = spread(v, args.per_field)
        counts[k] = len(sel)
        xs.update(sel)
    if args.site_info:
        for k, v in site_info_values().items():
            sel = spread(v, args.per_field)
            counts[k + "_raw"] = len(sel)
            xs.update(sel)
    syn = synthetic()
    counts["synthetic"] = len(set(syn))
    xs.update(syn)
    # -0.0 == 0.0 in a set: keep both explicitly, by bits.
    by_bits = {bits(x): x for x in xs}
    by_bits[bits(-0.0)] = -0.0
    rows = sorted(by_bits.items())
    os.makedirs(args.outdir, exist_ok=True)
    tables = {
        "pyfmt_str.tsv": repr,
        "pyfmt_round.tsv": lambda x: repr(round(x, 3) + 0.0),
        "pyfmt_2f.tsv": lambda x: "%4.2f" % x,
    }
    header = "# generated by rust/tools/gen_pyfmt_tables.py; inputs: " + (
        ", ".join(f"{k} {n}" for k, n in sorted(counts.items()))
    )
    for name, fn in tables.items():
        with open(os.path.join(args.outdir, name), "w") as fh:
            fh.write(header + "\n")
            for b, x in rows:
                fh.write(f"{b}\t{fn(x)}\n")
    print(f"{len(rows)} values; {header}")


if __name__ == "__main__":
    main()
