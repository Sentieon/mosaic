#!/usr/bin/env python3
# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Correctness sweep: joint-call output must be invariant to (shard_size, pos_batch_size).

For every (shard_size, pos_batch_size) combination, run the full pipeline
    reset -> init -> ingest(shard_size, pos_batch_size) -> aggregate -> joint_call
and take the md5 of the joint-called VCF. All combinations within a region MUST
produce the same md5; any difference is a correctness edge case in the ingest
sharding / position-window flushing.

The md5 is taken over the DECOMPRESSED, SORTED variant records (header stripped)
-- the same id-independent digest the test harness uses (jc_testlib.snapshot_vcf).
A raw `md5sum file.vcf.gz` is unreliable here: bgzip block framing and any header
fields are not part of the variant content, so two byte-different files can carry
identical variants. The raw-file md5 is still reported, for transparency.

A bonus check compares the PG aggregate snapshot (site_info / ref_blocks /
alleles, by genomic coordinate) across combos -- a stricter, earlier signal than
the joint-call output alone.

Requires the local Docker stack (Postgres :5432 / Iceberg REST :8181 /
MinIO :9000) and runs entirely against the isolated jc_test stores (jc_testlib
forces the MOSAIC_* isolation on import, so this never touches dev data). Run it
with the runtime env that has the dependency stack, e.g.

    python scripts/check_pos_shard_invariance.py

Examples:
    # default 2-region x 4-shard x 6-pos_batch_size matrix
    python scripts/check_pos_shard_invariance.py
    # custom, quick smoke
    python scripts/check_pos_shard_invariance.py \
        --regions chr21:10000000-10100000 \
        --shard-sizes 50000,200000 --pos-batch-sizes 0,9973
"""

import argparse
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict

# jc_testlib lives under tests/ and forces the jc_test isolation env on import.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "tests"))
import jc_testlib as L  # noqa: E402


def vcf_content_md5(path):
    """md5 over sorted, header-stripped VCF records (id/compression independent)."""
    proc = subprocess.run(
        ["bcftools", "view", "-H", path],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = sorted(proc.stdout.splitlines())
    return hashlib.md5("\n".join(lines).encode()).hexdigest()


def raw_md5(path):
    """md5 of the raw .vcf.gz bytes (FYI only -- not used for the verdict)."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _slug(region):
    return region.replace(":", "_").replace("-", "_")


def run_combo(gvcfs, region, shard_size, pos_batch_size, out_vcf):
    """One full pipeline at a (shard_size, pos_batch_size); return (content_md5,
    raw_md5, pg_snapshot)."""
    L.reset_state()
    L.run_init(L.GVCFS[0], "chr21")
    L.run_ingest(
        gvcfs,
        region=region,
        shard_size=shard_size,
        pos_batch_size=pos_batch_size,
        threads=2,
    )
    L.run_aggregate(fan_in=100, threads=2)
    L.run_joint_call(out_vcf)
    conn = L.get_pg_conn()
    try:
        pg = L.snapshot_pg(conn)
    finally:
        conn.close()
    return vcf_content_md5(out_vcf), raw_md5(out_vcf), pg


def sweep_region(region, gvcfs, shard_sizes, pos_batch_sizes, workdir):
    """Run the full shard x pos_batch_size matrix for one region.

    Returns (ok, rows) where rows is a list of dicts. ok is True iff every combo
    produced an identical joint-call content md5 AND an identical PG snapshot.
    """
    combos = [(ss, pbs) for ss in shard_sizes for pbs in pos_batch_sizes]
    print(
        f"\n##### REGION {region}  "
        f"({len(shard_sizes)} shard_size x {len(pos_batch_sizes)} "
        f"pos_batch_size = {len(combos)} combos) #####",
        flush=True,
    )
    rows = []
    ref_pg = None
    for i, (ss, pbs) in enumerate(combos, 1):
        out = os.path.join(
            workdir, f"jc_{_slug(region)}_ss{ss}_pbs{pbs}.vcf.gz"
        )
        t0 = time.time()
        try:
            cmd5, rmd5, pg = run_combo(gvcfs, region, ss, pbs, out)
            err = None
        except Exception as e:  # keep sweeping; record the failure
            cmd5 = rmd5 = None
            pg = None
            err = f"{type(e).__name__}: {str(e)[:200]}"
        secs = time.time() - t0

        pg_ok = None
        if pg is not None:
            if ref_pg is None:
                ref_pg = pg
                pg_ok = True
            else:
                try:
                    L.assert_pg_bitexact(pg, ref_pg)
                    pg_ok = True
                except AssertionError:
                    pg_ok = False

        rows.append(
            {
                "shard_size": ss,
                "pos_batch_size": pbs,
                "content_md5": cmd5,
                "raw_md5": rmd5,
                "pg_ok": pg_ok,
                "secs": secs,
                "err": err,
            }
        )
        status = (
            f"content_md5={cmd5[:12]} raw={rmd5[:8]} pg_ok={pg_ok}"
            if err is None
            else f"ERROR {err}"
        )
        print(
            f"[{i:>2}/{len(combos)}] shard={ss:>8} pbs={pbs:>8}  "
            f"{status}  {secs:5.1f}s",
            flush=True,
        )

    md5s = {r["content_md5"] for r in rows if r["content_md5"]}
    errored = [r for r in rows if r["err"]]
    pg_bad = [r for r in rows if r["pg_ok"] is False]
    ok = len(md5s) == 1 and not errored and not pg_bad

    print(f"  -> distinct joint-call content md5s: {len(md5s)}", flush=True)
    if len(md5s) > 1:
        by = defaultdict(list)
        for r in rows:
            if r["content_md5"]:
                by[r["content_md5"]].append(
                    (r["shard_size"], r["pos_batch_size"])
                )
        for md5, combs in by.items():
            print(f"     md5 {md5[:16]}: {combs}", flush=True)
    if pg_bad:
        print(
            "  -> PG snapshot DIFFERS for: "
            + str([(r["shard_size"], r["pos_batch_size"]) for r in pg_bad]),
            flush=True,
        )
    if errored:
        print(
            "  -> combos that ERRORED: "
            + str([(r["shard_size"], r["pos_batch_size"]) for r in errored]),
            flush=True,
        )
    print(f"  -> REGION {region}: {'PASS' if ok else 'FAIL'}", flush=True)
    return ok, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--regions",
        default="chr21:10000000-10500000,chr21:15000000-15500000",
        help="comma-separated regions; each gets the full matrix",
    )
    ap.add_argument(
        "--shard-sizes",
        default="37000,100000,250000,600000",
        help="comma-separated --shard-size values",
    )
    ap.add_argument(
        "--pos-batch-sizes",
        default="0,997,9973,50000,250000,600000",
        help="comma-separated --pos-batch-size values (0 = single window)",
    )
    ap.add_argument(
        "--samples",
        type=int,
        default=2,
        help="number of test gVCFs to use (max = len(jc_testlib.GVCFS))",
    )
    ap.add_argument("--workdir", default=None)
    args = ap.parse_args()

    if not L.stack_up():
        print(
            "ERROR: Docker stack not up (Postgres:5432 / Iceberg:8181 / "
            "MinIO:9000). Start it: cd iceberg_test && docker-compose up -d",
            file=sys.stderr,
        )
        return 2

    regions = [r.strip() for r in args.regions.split(",") if r.strip()]
    shard_sizes = [int(x) for x in args.shard_sizes.split(",")]
    pos_batch_sizes = [int(x) for x in args.pos_batch_sizes.split(",")]
    srcs = L.GVCFS[: args.samples]

    workdir = args.workdir or tempfile.mkdtemp(prefix="pos_shard_check_")
    os.makedirs(workdir, exist_ok=True)

    print(
        f"samples={len(srcs)}  shard_sizes={shard_sizes}  "
        f"pos_batch_sizes={pos_batch_sizes}\n"
        f"regions={regions}\nworkdir={workdir}",
        flush=True,
    )

    overall_ok = True
    t_start = time.time()
    for region in regions:
        # Subset the input gVCFs to the region once; reused across all combos.
        gvcfs = []
        for j, src in enumerate(srcs):
            dst = os.path.join(workdir, f"{_slug(region)}_s{j}.g.vcf.gz")
            L.subset_gvcf(src, region, dst)
            gvcfs.append(dst)
        ok, _rows = sweep_region(
            region, gvcfs, shard_sizes, pos_batch_sizes, workdir
        )
        overall_ok = overall_ok and ok

    elapsed = time.time() - t_start
    print(
        f"\n================ OVERALL: {'PASS' if overall_ok else 'FAIL'} "
        f"({elapsed/60:.1f} min) ================",
        flush=True,
    )
    if overall_ok:
        print(
            "Every (shard_size, pos_batch_size) combo produced identical "
            "joint-call output (and PG aggregates) within each region.",
            flush=True,
        )
    return 0 if overall_ok else 1


if __name__ == "__main__":
    sys.exit(main())
