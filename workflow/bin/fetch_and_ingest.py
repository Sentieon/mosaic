#!/usr/bin/env python3
# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""
Download one batch of 1000 Genomes DRAGEN gVCFs from S3, strip unused
annotations, and ingest them via `mosaic ingest`.

Invoked by the Nextflow INGEST_BATCH process (see workflow/main.nf). Also callable
by hand for debugging.

Usage:
    fetch_and_ingest.py [options] sample1 sample2 ...

Example (local, chr21 only):
    fetch_and_ingest.py --batch-id 0 --workdir . \\
        --region chr21 --threads 4 HG00096 HG00097

Example (AWS, whole genome):
    fetch_and_ingest.py --batch-id 42 --workdir . --threads 8 \\
        HG00096 HG00097 ... (up to ~200 samples)
"""

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

# Sibling helper in this bin/ dir (Nextflow stages bin/ onto PATH and the
# Docker image bakes both scripts together). Single shared implementation of
# the Secrets Manager fetch, used here for ingest and by the AGGREGATE / INDEX
# process wrappers via with_db_password.py's CLI entrypoint.
from with_db_password import fetch_db_password_into_env

S3_URL_TEMPLATE = (
    "s3://1000genomes-dragen-v4-2-7/data/individuals/hg38_alt_masked_graph_v3"
    "/{sample}/{sample}.final.hard-filtered.gvcf.gz"
)

# Annotations used by `mosaic ingest` — everything else is stripped
# to reduce gVCF size on disk.
KEEP_INFO = [
    "INFO/DP",
    "INFO/END",
    "INFO/MQ",
    "INFO/MQSQ",
    "INFO/RAW_MQ",
    "INFO/RAW_MQandDP",
    "INFO/GTC",
    "INFO/SB",
]
KEEP_FORMAT = [
    "FORMAT/GT",
    "FORMAT/AD",
    "FORMAT/DP",
    "FORMAT/MIN_DP",
    "FORMAT/SB",
    "FORMAT/PL",
    "FORMAT/F1R2",
    "FORMAT/F2R1",
    "FORMAT/GQ",
]


def log(msg):
    print(msg, flush=True)


def err(msg):
    print(msg, file=sys.stderr, flush=True)


def download_sample(sample, out_path, region, max_retries=5):
    """Stream one sample's gVCF from S3 through bcftools, strip unused
    annotations, and write to out_path + .tbi. Returns file size in bytes.
    """
    gvcf_url = S3_URL_TEMPLATE.format(sample=sample)
    keep_expr = ",".join(
        [f"^{a}" for a in KEEP_INFO] + [f"^{a}" for a in KEEP_FORMAT]
    )
    # Use `-r <region>` for indexed-jump downloads. Same .tbi-driven code
    # path that triggered an htslib long-virtual-offset bug in an
    # earlier multi-region download — but that bug only fires when many
    # regions are concatenated into one stream crossing ~3 GB of virtual
    # offset. Nextflow now fans out per chromosome, so each download is a
    # single chr (~150 MB compressed for the 1KG DRAGEN gVCFs), well below
    # the threshold. `-r` only fetches the relevant slice of the gVCF —
    # much cheaper than `-t`, which streams the whole file.
    # NOTE: if `region` is ever a multi-chromosome comma list again, swap
    # back to `-t` to sidestep the htslib bug.
    view_cmd = ["bcftools", "view"]
    if region:
        view_cmd += ["-r", region]
    view_cmd.append(gvcf_url)
    annotate_cmd = [
        "bcftools",
        "annotate",
        "-x",
        keep_expr,
        "-O",
        "z",
        "--write-index=tbi",
        "-o",
        out_path,
    ]

    for attempt in range(1, max_retries + 1):
        log(
            f"[RUN] Downloading "
            f"{'region ' + region + ' of ' if region else ''}"
            f"{sample} (attempt {attempt}/{max_retries})"
        )
        # Wire `bcftools view | bcftools annotate` manually rather than via
        # `shell=True` with a `|`. The shell pipeline's exit code is just
        # the last command's, so a mid-stream failure in `view` (e.g. the
        # BGZF read error path) gets masked when `annotate` exits cleanly
        # on the truncated input — silent data loss. Two Popens with
        # independent return-code checks surface both sides.
        view_proc = subprocess.Popen(view_cmd, stdout=subprocess.PIPE)
        annotate_proc = subprocess.Popen(annotate_cmd, stdin=view_proc.stdout)
        # Drop our copy of the read end so `view` receives SIGPIPE if
        # `annotate` dies early.
        assert view_proc.stdout is not None
        view_proc.stdout.close()
        # Wait on the downstream consumer first, then the upstream
        # producer — annotate exits on EOF from the pipe, so view should
        # already be done (or about to be) by the time annotate returns.
        annotate_rc = annotate_proc.wait()
        view_rc = view_proc.wait()

        if view_rc == 0 and annotate_rc == 0:
            size = os.path.getsize(out_path)
            log(f"  {sample}: {size:,} bytes")
            return size

        err(
            f"[WARN] Download failed for {sample} "
            f"(view exit {view_rc}, annotate exit {annotate_rc})"
        )
        # Clean up partial files before retry
        for f in (out_path, out_path + ".tbi"):
            if os.path.exists(f):
                os.remove(f)
        if attempt < max_retries:
            wait = 2**attempt
            err(f"  Retrying in {wait}s...")
            time.sleep(wait)

    err(f"[ERROR] All {max_retries} download attempts failed for {sample}")
    sys.exit(1)


def ingest(
    gvcf_paths,
    pos_batch_size,
    shard_size,
    threads,
    region=None,
    defer_register=False,
    ref_gq_bands=None,
):
    """Invoke `mosaic ingest` as a subprocess with the local gVCFs.

    `region`, if set, is forwarded to the ingester so it can scope the
    `ingestion_progress` checkpoint row by (sample_names, region). The
    per-chromosome Nextflow fan-out relies on this — without it, running
    the same samples for chr11 then chr22 would have the second run
    short-circuit as "batch already complete".

    `defer_register` forwards `--defer-register`: the batch uploads its
    Parquet and records the file list in PostgreSQL (status 'uploaded')
    instead of committing to the shared Iceberg tables; the workflow's
    REGISTER step (`mosaic register`) registers them and completes it.

    `ref_gq_bands`, if set, forwards `--ref-gq-bands` (ref-block banding;
    `mosaic ingest` validates the spec). Unset = off.
    """
    cmd = [
        sys.executable,
        "-m",
        "mosaic.cli",
        "ingest",
        "--pos-batch-size",
        str(pos_batch_size),
        "--shard-size",
        str(shard_size),
        "--threads",
        str(threads),
    ]
    if region:
        cmd += ["--region", region]
    if defer_register:
        cmd += ["--defer-register"]
    if ref_gq_bands is not None:
        cmd += ["--ref-gq-bands", ref_gq_bands]
    cmd += gvcf_paths
    log(f"[RUN] {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        err(
            f"[ERROR] `mosaic ingest` failed with exit code "
            f"{result.returncode}"
        )
        sys.exit(result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Download a batch of 1000 Genomes DRAGEN gVCFs from S3 and "
            "ingest into the joint caller pipeline."
        )
    )
    parser.add_argument(
        "samples",
        nargs="+",
        help="Sample names (e.g. HG00096) to download and ingest",
    )
    parser.add_argument(
        "--batch-id",
        default="0",
        help=(
            "Identifier used to disambiguate filenames for concurrent "
            "batches sharing a workdir (default: 0)"
        ),
    )
    parser.add_argument(
        "--workdir",
        default=".",
        help="Directory for downloaded gVCF files (default: cwd)",
    )
    parser.add_argument(
        "--region",
        default=None,
        help=(
            "Optional region filter passed to `bcftools view` "
            "(e.g. 'chr21'). Default: whole genome."
        ),
    )
    parser.add_argument(
        "--pos-batch-size",
        type=int,
        default=150_000,
        help="Passed through to `mosaic ingest` --pos-batch-size",
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=500_000,
        help="Passed through to `mosaic ingest` --shard-size",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help=(
            "Parallel workers for downloads and ingestion "
            "(default: CPU count)"
        ),
    )
    parser.add_argument(
        "--defer-register",
        action="store_true",
        help=(
            "Passed through to `mosaic ingest --defer-register`: upload the "
            "Parquet and record it for `mosaic register` instead of "
            "registering it with Iceberg (the batch is left 'uploaded')"
        ),
    )
    parser.add_argument(
        "--ref-gq-bands",
        default=None,
        help=(
            "Passed through to `mosaic ingest --ref-gq-bands` when set "
            "(ref-block banding, e.g. '20,60' or 'all'). Default: off"
        ),
    )
    args = parser.parse_args()

    workdir = os.path.abspath(args.workdir)
    os.makedirs(workdir, exist_ok=True)
    threads = args.threads if args.threads is not None else os.cpu_count()

    # Populate MOSAIC_DB_PASSWORD from AWS Secrets Manager if we're on AWS
    # and the password isn't already in the environment.
    fetch_db_password_into_env()

    # Deterministic per-batch filenames so concurrent workers sharing a
    # workdir don't collide. Format: batch_<id>_<sample>.gvcf.gz
    sample_paths = {
        s: os.path.join(workdir, f"batch_{args.batch_id}_{s}.gvcf.gz")
        for s in args.samples
    }

    log(
        f"[INFO] Batch {args.batch_id}: downloading "
        f"{len(args.samples)} gVCF(s) with {threads} thread(s)"
        + (f" (region={args.region})" if args.region else "")
    )

    total_bytes = 0
    with ThreadPoolExecutor(max_workers=threads) as pool:
        futures = {
            pool.submit(download_sample, sample, path, args.region): sample
            for sample, path in sample_paths.items()
        }
        for future in as_completed(futures):
            total_bytes += future.result()  # raises on failure

    log(
        f"[INFO] Total gVCF size on disk: {total_bytes:,} bytes "
        f"({total_bytes / (1024 * 1024):.1f} MB) "
        f"across {len(args.samples)} sample(s)"
    )

    # Preserve original CLI sample order when invoking the ingester
    gvcf_paths = [sample_paths[s] for s in args.samples]

    log(
        f"[INFO] Starting ingestion of {len(args.samples)} gVCFs "
        f"(batch {args.batch_id})"
    )
    start = time.time()
    try:
        ingest(
            gvcf_paths,
            args.pos_batch_size,
            args.shard_size,
            threads,
            region=args.region,
            defer_register=args.defer_register,
            ref_gq_bands=args.ref_gq_bands,
        )
    finally:
        # Clean up local gVCFs whether ingestion succeeded or failed —
        # ingest_multi_gvcfs.py's own shard-level checkpointing in
        # Postgres is what lets us resume safely; the downloaded files
        # don't need to stick around.
        for path in gvcf_paths:
            for f in (path, path + ".tbi"):
                if os.path.exists(f):
                    try:
                        os.remove(f)
                    except OSError as e:
                        err(f"[WARN] Could not remove {f}: {e}")

    elapsed = time.time() - start
    log(
        f"[DONE] Batch {args.batch_id}: ingested "
        f"{len(args.samples)} samples in {elapsed:.1f}s"
    )


if __name__ == "__main__":
    main()
