#!/usr/bin/env python3
# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""The tasks of the joint-call workflow (workflow/joint_call.nf), one
subcommand per process.

Each subcommand runs the joint-call driver, ``joint_call.py``, found as
``--driver``, else ``$MOSAIC_JOINT_CALL_DRIVER``, else ``joint_call.py`` on
PATH (the image's /opt/mosaic/bin), else the repository root above this
script (a run from a checkout); it runs as ``[sys.executable, <driver>,
...]``. Everything written under the output prefix (``--out``: a local
directory or ``s3://bucket/prefix``) is written by ``publish``, under
``<out>/gen<agg_run_id>/``, the generation read from the export's
``cohort.json``.

    state [--samples-file F]
        The database password (Secrets Manager on AWS), then the driver's
        ``--cohort-state``: its one state line is this command's stdout,
        nothing else.
    export-cohort --dir D --out OUT
        The password, ``--export-cohort D``, then ``D/cohort.json``
        published to ``OUT/gen<id>/export/``.
    export-contig --dir D --contig NAME --out OUT
        The password, ``--export-contig D --region NAME``, then the
        contig's three files published to ``OUT/gen<id>/export/``.
    call --set S --contig NAME --contig-index I --export D --out OUT ...
        One set and one chromosome, with no password and no PostgreSQL:
        the driver's ``--from-export D --region NAME`` into a local
        ``.vcf.gz`` (TMPDIR and ``--spill-dir`` under the work directory),
        ``bcftools index -t`` and the index's record count against the
        driver's ``Completed!`` line, the names file, the VCF and the index
        published to ``OUT/gen<id>/<S>/``, then ``<S>.<chrom>.summary.json``
        written and published last (the completion marker). The exit status
        is the driver's when it fails (a refusal, 2, stays 2); a failure
        leaves no summary, local or published.
    summarize --out OUT SUMMARY.json...
        ``joint_call_summary.tsv`` (one row per set and chromosome)
        published to ``OUT/gen<id>/``.
"""

import argparse
import contextlib
import csv
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

# Sibling helper in this bin/ dir (Nextflow puts bin/ on PATH; the image
# carries both in /opt/mosaic/bin).
from with_db_password import fetch_db_password_into_env

DRIVER = "joint_call.py"
DRIVER_ENV = "MOSAIC_JOINT_CALL_DRIVER"
COHORT_FILE = "cohort.json"
SUMMARY_FORMAT = 1
SUMMARY_TSV = "joint_call_summary.tsv"
# The S3 multipart part size (boto3 raises it by itself to stay under
# 10,000 parts for a larger object).
MULTIPART_CHUNK_BYTES = 64 << 20
# A chromosome's name is used in a file name when it is plain.
PLAIN_NAME = re.compile(r"[A-Za-z0-9._-]+")
COMPLETED = re.compile(r"Completed! Total processed: (\d+) variants")
WORKER_LINE = "mosaic-joint-call:"
TSV_COLUMNS = [
    "set",
    "chromosome",
    "contig_index",
    "samples",
    "records",
    "vcf_bytes",
    "driver_seconds",
    "index_seconds",
    "publish_seconds",
    "vcf",
]


class TaskError(Exception):
    """A failure of the task (exit 1) with its message."""


def err(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


# ── the driver ───────────────────────────────────────────────────────────


def repo_driver() -> str:
    """``joint_call.py`` at the repository root above this script
    (workflow/bin/ -> the root)."""
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), DRIVER)


def find_driver(explicit: Optional[str]) -> str:
    """``explicit``, else ``$MOSAIC_JOINT_CALL_DRIVER``, else
    ``joint_call.py`` on PATH, else the repository root's; TaskError with
    the remedy when the one chosen (or every one) is missing."""
    for given, what in (
        (explicit, "--driver"),
        (os.environ.get(DRIVER_ENV), f"${DRIVER_ENV}"),
    ):
        if given:
            if os.path.isfile(given):
                return os.path.abspath(given)
            raise TaskError(
                f"the joint-call driver {what}={given} is not a file; point "
                f"{what} at joint_call.py"
            )
    on_path = shutil.which(DRIVER)
    if on_path:
        return on_path
    local = repo_driver()
    if os.path.isfile(local):
        return local
    raise TaskError(
        f"the joint-call driver is not found: no --driver, ${DRIVER_ENV} "
        f"unset, {DRIVER} not on PATH and not at {local}. Set "
        f"{DRIVER_ENV}=<checkout>/{DRIVER} (or put {DRIVER} on PATH: the "
        "image carries it in /opt/mosaic/bin)"
    )


def run_driver(
    cmd: List[str],
    env: Optional[Dict[str, str]] = None,
    capture_stdout: bool = False,
) -> Tuple[int, str, List[str]]:
    """Run the driver: its stderr passed through to this process's stderr
    line by line and kept; its stdout kept when ``capture_stdout`` (else
    inherited). Returns (exit status, stdout, stderr lines)."""
    sys.stdout.flush()
    sys.stderr.flush()
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE if capture_stdout else None,
        stderr=subprocess.PIPE,
        env=env,
        text=True,
        errors="replace",
    )
    lines: List[str] = []
    assert proc.stderr is not None
    for line in proc.stderr:
        sys.stderr.write(line)
        sys.stderr.flush()
        lines.append(line.rstrip("\n"))
    out = proc.stdout.read() if proc.stdout is not None else ""
    rc = proc.wait()
    return rc, out, lines


def exit_status(rc: int) -> int:
    """The driver's exit status as this command's: as is, or 128 + the
    signal for a driver killed by one."""
    return 128 - rc if rc < 0 else rc


def driver_or_exit(cmd: List[str], **kw: Any) -> Tuple[str, List[str]]:
    """``run_driver``; on a failure, exit with the driver's status."""
    rc, out, lines = run_driver(cmd, **kw)
    if rc != 0:
        err(f"[ERROR] the joint-call driver exited with status {rc}")
        sys.exit(exit_status(rc))
    return out, lines


# ── the export ───────────────────────────────────────────────────────────


def contig_export_names(chrom_idx: int) -> Dict[str, str]:
    """A contig export's three files (joint_call.py's
    ``contig_export_names``), the sidecar last: the order they are
    published in."""
    return {
        "sites": f"sites.c{int(chrom_idx)}.parquet",
        "alleles": f"alleles.c{int(chrom_idx)}.parquet",
        "export": f"export.c{int(chrom_idx)}.json",
    }


def load_cohort(directory: str) -> Dict[str, Any]:
    path = os.path.join(directory, COHORT_FILE)
    try:
        with open(path) as fh:
            doc = json.load(fh)
    except (OSError, ValueError) as e:
        raise TaskError(f"cannot read {path}: {e}")
    if not isinstance(doc, dict) or not isinstance(doc.get("state"), dict):
        raise TaskError(f"{path} is not a cohort export")
    return doc


def generation(cohort: Dict[str, Any]) -> int:
    """The cohort generation (``agg_run_id``) of ``cohort.json``."""
    gen = cohort["state"].get("agg_run_id")
    if not isinstance(gen, int) or isinstance(gen, bool):
        raise TaskError(
            f"the cohort export has no aggregation run (agg_run_id {gen!r})"
        )
    return gen


def gen_folder(gen: int) -> str:
    return f"gen{int(gen)}"


def chrom_file_name(name: str, index: int) -> str:
    """The chromosome's name when it is plain (letters, digits, ``.``,
    ``_``, ``-``), else ``c<index>``."""
    if PLAIN_NAME.fullmatch(name) and name not in (".", ".."):
        return name
    return f"c{int(index)}"


# ── publishing ───────────────────────────────────────────────────────────


_S3 = None


def s3_client() -> Any:
    """One boto3 S3 client, from ``mosaic.config.build_s3_client_kwargs()``
    (MinIO and AWS through the same path)."""
    global _S3
    if _S3 is None:
        import boto3  # type: ignore[import-untyped]

        from mosaic.config import build_s3_client_kwargs

        _S3 = boto3.client(  # type: ignore[call-overload]
            "s3", **build_s3_client_kwargs()
        )
    return _S3


def split_s3(uri: str) -> Tuple[str, str]:
    """``s3://bucket/prefix`` -> (bucket, prefix without slashes at its
    ends)."""
    rest = uri[len("s3://") :]
    bucket, _, prefix = rest.partition("/")
    if not bucket:
        raise TaskError(f"no bucket in {uri}")
    return bucket, prefix.strip("/")


def check_out(out: str) -> str:
    """The output prefix: ``s3://bucket[/prefix]`` or an absolute local
    path (made absolute here); another scheme: TaskError."""
    if out.startswith("s3://"):
        split_s3(out)
        return out
    if "://" in out:
        raise TaskError(
            f"--out {out}: only a local directory or s3://bucket/prefix"
        )
    return os.path.abspath(out)


def dest_of(out: str, rel: str) -> str:
    """The published path of ``rel`` under ``out``."""
    if out.startswith("s3://"):
        bucket, prefix = split_s3(out)
        key = f"{prefix}/{rel}" if prefix else rel
        return f"s3://{bucket}/{key}"
    return os.path.join(out, rel)


def publish(local: str, out: str, rel: str, move: bool = False) -> str:
    """Write ``local`` to ``<out>/<rel>`` and check its size there: a local
    directory (created; the file moved when ``move``, else copied, through
    a temporary name and a rename) or S3 (boto3 ``upload_file``, 64 MiB
    parts, the size from ``head_object``). A size that differs: the
    published object is removed and TaskError raised. Returns the
    published path."""
    size = os.path.getsize(local)
    dest = dest_of(out, rel)
    if dest.startswith("s3://"):
        from boto3.s3.transfer import TransferConfig  # type: ignore

        bucket, key = split_s3(dest)
        client = s3_client()
        client.upload_file(
            local,
            bucket,
            key,
            Config=TransferConfig(
                multipart_threshold=MULTIPART_CHUNK_BYTES,
                multipart_chunksize=MULTIPART_CHUNK_BYTES,
            ),
        )
        got = int(client.head_object(Bucket=bucket, Key=key)["ContentLength"])
        if got != size:
            client.delete_object(Bucket=bucket, Key=key)
            raise TaskError(f"{dest} holds {got} bytes, {local} {size}")
        if move:
            os.remove(local)
        return dest
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = f"{dest}.tmp-{secrets.token_hex(4)}"
    try:
        if move:
            shutil.move(local, tmp)
        else:
            shutil.copyfile(local, tmp)
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)
    got = os.path.getsize(dest)
    if got != size:
        os.remove(dest)
        raise TaskError(f"{dest} holds {got} bytes, {local} {size}")
    return dest


def unpublish(out: str, rel: str) -> None:
    """Remove ``<out>/<rel>`` if it exists (an earlier run's summary, before
    its VCF is overwritten)."""
    dest = dest_of(out, rel)
    if dest.startswith("s3://"):
        bucket, key = split_s3(dest)
        s3_client().delete_object(Bucket=bucket, Key=key)
    elif os.path.exists(dest):
        os.remove(dest)


# ── the subcommands ──────────────────────────────────────────────────────


def cmd_state(args: argparse.Namespace) -> None:
    driver = find_driver(args.driver)
    # The password fetch logs on stdout, which carries the state line only.
    with contextlib.redirect_stdout(sys.stderr):
        fetch_db_password_into_env()
    cmd = [sys.executable, driver, "--cohort-state"]
    if args.samples_file:
        cmd += ["--samples-file", args.samples_file]
    out, _ = driver_or_exit(cmd, capture_stdout=True)
    lines = [x for x in out.splitlines() if x.strip()]
    if len(lines) != 1:
        raise TaskError(
            f"the driver's --cohort-state printed {len(lines)} lines, not one"
        )
    sys.stdout.write(lines[0].strip() + "\n")
    sys.stdout.flush()


def cmd_export_cohort(args: argparse.Namespace) -> None:
    driver = find_driver(args.driver)
    out = check_out(args.out)
    fetch_db_password_into_env()
    driver_or_exit([sys.executable, driver, "--export-cohort", args.dir])
    gen = generation(load_cohort(args.dir))
    dest = publish(
        os.path.join(args.dir, COHORT_FILE),
        out,
        f"{gen_folder(gen)}/export/{COHORT_FILE}",
    )
    err(f"Published {dest}")


def cmd_export_contig(args: argparse.Namespace) -> None:
    driver = find_driver(args.driver)
    out = check_out(args.out)
    cohort = load_cohort(args.dir)
    gen = generation(cohort)
    entry = next(
        (c for c in cohort["contigs"] if c["name"] == args.contig), None
    )
    if entry is None:
        raise TaskError(f"contig {args.contig!r} is not in the export")
    fetch_db_password_into_env()
    driver_or_exit(
        [
            sys.executable,
            driver,
            "--export-contig",
            args.dir,
            "--region",
            args.contig,
        ]
    )
    for name in contig_export_names(int(entry["index"])).values():
        dest = publish(
            os.path.join(args.dir, name),
            out,
            f"{gen_folder(gen)}/export/{name}",
        )
        err(f"Published {dest}")


def call_command(
    args: argparse.Namespace, driver: str, vcf: str, spill: str
) -> List[str]:
    """The driver's command line for one set and one chromosome."""
    cmd = [
        sys.executable,
        driver,
        vcf,
        "--from-export",
        args.export,
        "--region",
        args.contig,
        "--spill-dir",
        spill,
    ]
    if args.samples_file:
        cmd += ["--samples-file", args.samples_file]
    if args.threads is not None:
        cmd += ["--threads", str(args.threads)]
    if args.window is not None:
        cmd += ["--window", str(args.window)]
    if args.max_alt_alleles is not None:
        cmd += ["--max-alt-alleles", str(args.max_alt_alleles)]
    if args.trim_alt_alleles:
        cmd += ["--trim-alt-alleles"]
    if args.min_ac is not None:
        cmd += ["--min-ac", str(args.min_ac)]
    return cmd


def parse_driver_stderr(lines: List[str]) -> Dict[str, Any]:
    """From the driver's stderr: the records of its ``Completed!`` line, its
    ``mosaic-joint-call:`` line and its closing lines (the ``Completed!``
    line and every line after it). TaskError without a ``Completed!``
    line."""
    worker = next((x for x in lines if x.startswith(WORKER_LINE)), None)
    for k, line in enumerate(lines):
        m = COMPLETED.fullmatch(line.strip())
        if m:
            return {
                "records": int(m.group(1)),
                "worker_line": worker,
                "closing_lines": lines[k:],
            }
    raise TaskError("the driver printed no 'Completed! Total processed' line")


def index_records(vcf: str, threads: Optional[int] = None) -> int:
    """``bcftools index -t`` of the VCF (on ``threads`` threads when given:
    they decompress the BGZF blocks), then the index's record count
    (``bcftools index -n``)."""
    cmd = ["bcftools", "index", "-f", "-t"]
    if threads is not None:
        cmd += ["--threads", str(threads)]
    subprocess.run(cmd + [vcf], check=True)
    proc = subprocess.run(
        ["bcftools", "index", "-n", vcf],
        check=True,
        capture_output=True,
        text=True,
    )
    try:
        return int(proc.stdout.strip())
    except ValueError:
        raise TaskError(
            f"bcftools index -n {vcf} printed {proc.stdout.strip()!r}"
        )


def cmd_call(args: argparse.Namespace) -> None:
    driver = find_driver(args.driver)
    out = check_out(args.out)
    cohort = load_cohort(args.export)
    gen = generation(cohort)
    workdir = os.path.abspath(args.workdir)
    chrom = chrom_file_name(args.contig, args.contig_index)
    base = f"{args.set}.{chrom}"
    vcf = os.path.join(workdir, f"{base}.vcf.gz")
    tbi = f"{vcf}.tbi"
    summary_name = f"{base}.summary.json"
    summary_path = os.path.join(workdir, summary_name)
    folder = f"{gen_folder(gen)}/{args.set}"
    tmp = os.path.join(workdir, "tmp")
    spill = os.path.join(workdir, "spill")
    for p in (summary_path, vcf, tbi):
        if os.path.exists(p):
            os.remove(p)
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(spill, exist_ok=True)
    env = dict(os.environ, TMPDIR=tmp)
    names: Optional[List[str]] = None
    if args.samples_file:
        with open(args.samples_file) as fh:
            names = [x.strip() for x in fh if x.strip()]
    n_samples = len(names) if names is not None else len(cohort["samples"])
    seconds: Dict[str, float] = {}
    try:
        t0 = time.perf_counter()
        rc, _, lines = run_driver(
            call_command(args, driver, vcf, spill), env=env
        )
        seconds["driver"] = round(time.perf_counter() - t0, 3)
        if rc != 0:
            err(f"[ERROR] the joint-call driver exited with status {rc}")
            sys.exit(exit_status(rc))
        parsed = parse_driver_stderr(lines)
        if not os.path.isfile(vcf):
            raise TaskError(f"the driver wrote no {vcf}")
        vcf_bytes = os.path.getsize(vcf)
        t0 = time.perf_counter()
        indexed = not args.no_index
        if indexed:
            n = index_records(vcf, args.threads)
            if n != parsed["records"]:
                raise TaskError(
                    f"the index of {vcf} counts {n} records, the driver's "
                    f"Completed! line {parsed['records']}"
                )
        seconds["index"] = round(time.perf_counter() - t0, 3)

        t0 = time.perf_counter()
        unpublish(out, f"{folder}/{summary_name}")
        published: Dict[str, str] = {}
        if args.samples_file:
            published["samples"] = publish(
                args.samples_file,
                out,
                f"{folder}/{os.path.basename(args.samples_file)}",
            )
        published["vcf"] = publish(
            vcf, out, f"{folder}/{base}.vcf.gz", move=True
        )
        if indexed:
            published["tbi"] = publish(tbi, out, f"{folder}/{base}.vcf.gz.tbi")
        published["summary"] = dest_of(out, f"{folder}/{summary_name}")
        seconds["publish"] = round(time.perf_counter() - t0, 3)
        summary = {
            "format": SUMMARY_FORMAT,
            "set": args.set,
            "chromosome": args.contig,
            "contig_index": int(args.contig_index),
            "samples": n_samples,
            "generation": gen,
            "state": cohort["state"],
            "options": {
                "threads": args.threads,
                "window": args.window,
                "max_alt_alleles": args.max_alt_alleles,
                "trim_alt_alleles": bool(args.trim_alt_alleles),
                "min_ac": args.min_ac,
                "index": indexed,
            },
            "records": parsed["records"],
            "vcf_bytes": vcf_bytes,
            "worker_line": parsed["worker_line"],
            "closing_lines": parsed["closing_lines"],
            "seconds": seconds,
            "published": published,
        }
        with open(summary_path, "w") as fh:
            json.dump(summary, fh, indent=1, sort_keys=True)
            fh.write("\n")
        publish(summary_path, out, f"{folder}/{summary_name}")
        err(f"Published {published['vcf']} and its summary")
    except BaseException:
        if os.path.exists(summary_path):
            os.remove(summary_path)
        raise
    finally:
        for p in (vcf, tbi):
            if os.path.exists(p):
                os.remove(p)
        shutil.rmtree(spill, ignore_errors=True)
        shutil.rmtree(tmp, ignore_errors=True)


def summary_rows(paths: List[str]) -> Tuple[int, List[Dict[str, Any]]]:
    """The summaries' generation and their rows, by set then contig index;
    TaskError for summaries of more than one generation (or none)."""
    docs = []
    for p in paths:
        with open(p) as fh:
            docs.append(json.load(fh))
    gens = sorted({int(d["generation"]) for d in docs})
    if len(gens) != 1:
        raise TaskError(
            "the summaries are of "
            + (f"generations {gens}" if gens else "no generation")
            + ": one run's summaries are of one generation"
        )
    rows = []
    for d in sorted(docs, key=lambda d: (d["set"], int(d["contig_index"]))):
        sec = d.get("seconds", {})
        rows.append(
            {
                "set": d["set"],
                "chromosome": d["chromosome"],
                "contig_index": int(d["contig_index"]),
                "samples": d["samples"],
                "records": d["records"],
                "vcf_bytes": d["vcf_bytes"],
                "driver_seconds": sec.get("driver"),
                "index_seconds": sec.get("index"),
                "publish_seconds": sec.get("publish"),
                "vcf": d["published"]["vcf"],
            }
        )
    return gens[0], rows


def cmd_summarize(args: argparse.Namespace) -> None:
    out = check_out(args.out)
    gen, rows = summary_rows(args.summaries)
    path = os.path.abspath(SUMMARY_TSV)
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(
            fh, fieldnames=TSV_COLUMNS, delimiter="\t", lineterminator="\n"
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)
    dest = publish(path, out, f"{gen_folder(gen)}/{SUMMARY_TSV}")
    err(f"Published {dest}: {len(rows)} set x chromosome row(s)")


# ── the command line ─────────────────────────────────────────────────────


def _plain(value: str) -> str:
    if not PLAIN_NAME.fullmatch(value) or value in (".", ".."):
        raise argparse.ArgumentTypeError(
            f"{value!r}: letters, digits, '.', '_' and '-' only"
        )
    return value


def _nonneg(value: str) -> int:
    n = int(value)
    if n < 0:
        raise argparse.ArgumentTypeError(f"{value}: must be 0 or more")
    return n


def _positive(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError(f"{value}: must be at least 1")
    return n


def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--driver",
        default=None,
        help=f"joint_call.py (default: ${DRIVER_ENV}, else {DRIVER} on "
        "PATH, else the repository root above this script)",
    )
    parser = argparse.ArgumentParser(
        description="The tasks of the joint-call workflow "
        "(workflow/joint_call.nf)."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser(
        "state", parents=[common], help="the cohort state line (stdout)"
    )
    p.add_argument("--samples-file", default=None)
    p.set_defaults(func=cmd_state)

    p = sub.add_parser(
        "export-cohort", parents=[common], help="DIR/cohort.json, published"
    )
    p.add_argument("--dir", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export_cohort)

    p = sub.add_parser(
        "export-contig", parents=[common], help="a contig's export, published"
    )
    p.add_argument("--dir", required=True)
    p.add_argument("--contig", required=True)
    p.add_argument("--out", required=True)
    p.set_defaults(func=cmd_export_contig)

    p = sub.add_parser(
        "call", parents=[common], help="one set and one chromosome"
    )
    p.add_argument("--set", required=True, type=_plain)
    p.add_argument("--contig", required=True)
    p.add_argument("--contig-index", required=True, type=_nonneg)
    p.add_argument("--export", required=True, help="the export directory")
    p.add_argument("--samples-file", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--workdir", default=".")
    p.add_argument("--threads", type=_positive, default=None)
    p.add_argument("--window", type=_nonneg, default=None)
    p.add_argument("--max-alt-alleles", type=_nonneg, default=None)
    p.add_argument("--trim-alt-alleles", action="store_true")
    p.add_argument("--min-ac", type=_nonneg, default=None)
    p.add_argument(
        "--no-index",
        action="store_true",
        help="write no .tbi (and so no record-count check)",
    )
    p.set_defaults(func=cmd_call)

    p = sub.add_parser(
        "summarize", parents=[common], help="joint_call_summary.tsv"
    )
    p.add_argument("--out", required=True)
    p.add_argument("summaries", nargs="+")
    p.set_defaults(func=cmd_summarize)
    return parser


def main(argv: Optional[List[str]] = None) -> None:
    args = build_parser().parse_args(argv)
    try:
        args.func(args)
    except TaskError as e:
        err(f"[ERROR] {e}")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        err(f"[ERROR] {' '.join(map(str, e.cmd))} exited with {e.returncode}")
        sys.exit(1)


if __name__ == "__main__":
    main()
