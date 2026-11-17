#!/usr/bin/env python3
# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""Measure S3 ``BytesDownloaded`` for a benchmark run via CloudWatch.

After running ``scripts/gwas_benchmark.py`` against **AWS S3**, this reports how
many bytes the run pulled out of the bucket, summing the CloudWatch S3 *request
metric* ``BytesDownloaded`` (Namespace ``AWS/S3``) over the run's window via
``GetMetricData``. Pair it with ``scripts/measure_s3_usage.py`` (which sizes
what is *stored*) to compare bytes read against bytes resident for a given
sample-batching strategy.

Three ways to set the time window (StartTime / EndTime) — the crux of getting a
clean number:

1. ``--wrap -- <gwas_benchmark args>``   (recommended)
     Run ``gwas_benchmark.py`` as a child process and bracket it automatically:
     StartTime is captured just before launch, EndTime just after it exits, each
     padded by ``--pad`` minutes. The most accurate window for a single run, and
     the one mode that needs no clock bookkeeping by hand.

2. ``--lookback MINUTES``
     EndTime = now, StartTime = now - MINUTES. Use right after a manual run.
     This is the default (15 min) when no mode is given.

3. ``--start ISO --end ISO``
     Explicit timestamps, e.g. ``2026-06-23T18:00:00`` (UTC assumed if no offset
     is given). Full manual control, and the way to *re-query* a past run once
     the metrics have settled — every mode prints the exact ``--start/--end`` it
     used so you can reproduce it.

Important CloudWatch caveats:

* **Request metrics must be enabled on the bucket.** ``BytesDownloaded`` is an
  opt-in, paid S3 *request* metric (a bucket "metrics configuration" with a
  filter). ``--filter-id`` must match that filter's name — it defaults to
  ``s3-io-filter`` (this project's filter). A mismatched filter id is the most
  common cause of an empty result.
* **No Unit filter by default.** The query is issued without a CloudWatch Unit,
  because filtering on a Unit that doesn't match what S3 stored makes the result
  silently empty — a classic ``get_metric_statistics`` foot-gun. Pass
  ``--unit Bytes`` only if you have a specific reason to.
* **Reporting latency — the main correctness risk.** Datapoints lag the actual
  requests by several minutes (S3 advertises "within 15 min") and arrive one
  1-minute bucket at a time, so the running total *plateaus between arrivals*.
  Reading too early — or mistaking a plateau for the final value — yields a
  partial number, and two runs can disagree. ``--wait-minutes N`` polls up to N
  minutes and only trusts a reading once it has held steady (total *and*
  bucket count) for ``--settle-minutes``, riding out the gaps; if N elapses
  first the result is reported UNCONFIRMED. For a dependable number use a
  generous budget (e.g. ``--wait-minutes 20 --settle-minutes 3``) or re-query
  with explicit ``--start/--end`` >= ~15 min after the run, when every bucket
  is present.
* **AWS only.** MinIO (local dev, ``MOSAIC_S3_ENDPOINT`` set) does not publish
  CloudWatch metrics. The CloudWatch client here always talks to AWS in
  ``--region`` (default ``MOSAIC_S3_REGION``) and never to the S3 endpoint
  override, so do not point this at a MinIO run.
* **Whole-bucket scope.** The metric counts GETs for the bucket (or the
  FilterId's scope), so isolate the benchmark in time — no other readers — for a
  number attributable to the run.

Usage:
    # Wrap a run end-to-end (AWS env: MOSAIC_ICEBERG_TYPE=glue, MOSAIC_S3_BUCKET,
    # MOSAIC_S3_REGION, ... matching the benchmark), waiting up to 20 min for the
    # metric to hold steady for 3 min before trusting it:
    python scripts/s3_bytes_downloaded.py --wait-minutes 20 --settle-minutes 3 \
        --wrap -- --af-cutoff 0.01 --samples HG00245 HG00420

    # Look back over the last 10 minutes after a manual run:
    python scripts/s3_bytes_downloaded.py --lookback 10

    # Explicit window, machine-readable output:
    python scripts/s3_bytes_downloaded.py \
        --start 2026-06-23T18:00:00 --end 2026-06-23T18:05:00 --json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import boto3

DEFAULT_BUCKET = os.environ.get("MOSAIC_S3_BUCKET", "warehouse")
DEFAULT_REGION = os.environ.get("MOSAIC_S3_REGION", "us-east-1")
# This project's S3 request-metrics filter id (the bucket "metrics
# configuration" name). Must match what is configured on the bucket.
DEFAULT_FILTER_ID = "s3-io-filter"

# Path to the benchmark this tool brackets, resolved next to this script so
# --wrap works regardless of the caller's CWD.
BENCHMARK = os.path.join(os.path.dirname(__file__), "gwas_benchmark.py")

# Soft cap on datapoints per window: coarsen the Period so output stays
# manageable (get_metric_data itself paginates via NextToken, so this is a
# readability bound, not a hard API limit).
MAX_DATAPOINTS = 1440


def human_bytes(n: float) -> str:
    """Format a byte count as a binary-unit string (KiB/MiB/GiB/...)."""
    step = 1024.0
    size = float(n)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB", "PiB"):
        if size < step or unit == "PiB":
            return f"{size:.2f} {unit}" if unit != "B" else f"{int(size)} B"
        size /= step
    return f"{size:.2f} PiB"  # unreachable; keeps mypy happy


def parse_iso(value: str) -> datetime:
    """Parse an ISO-8601 timestamp into an aware UTC datetime.

    A naive timestamp (no offset) is interpreted as UTC, matching the labels
    CloudWatch reports and the `datetime.now(timezone.utc)` the other modes use.
    """
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def floor_minute(dt: datetime) -> datetime:
    return dt.replace(second=0, microsecond=0)


def ceil_minute(dt: datetime) -> datetime:
    floored = floor_minute(dt)
    return floored if floored == dt else floored + timedelta(minutes=1)


def safe_period(start: datetime, end: datetime, requested: int) -> int:
    """Round `requested` up to a 60s multiple and widen it if the window would
    exceed CloudWatch's 1440-datapoint cap.

    A 5-minute benchmark stays at 60s granularity; a multi-day --start/--end
    window auto-coarsens instead of erroring.
    """
    period = max(60, (requested // 60) * 60)
    span = max(1.0, (end - start).total_seconds())
    while span / period > MAX_DATAPOINTS:
        period += 60
    return period


def resolve_window(args: argparse.Namespace) -> Tuple[datetime, datetime, str]:
    """Resolve (StartTime, EndTime, mode_label) from the chosen mode.

    --wrap is handled by the caller (it needs to launch the child first); this
    covers the explicit and lookback windows. Times are returned padded and
    snapped to minute boundaries so they line up with the 1-minute metric
    buckets.
    """
    pad = timedelta(minutes=args.pad)
    if args.start or args.end:
        if not (args.start and args.end):
            sys.exit("Error: --start and --end must be given together.")
        start = parse_iso(args.start)
        end = parse_iso(args.end)
        label = "explicit"
    else:
        # Lookback (the zero-config default).
        minutes = args.lookback if args.lookback is not None else 15
        end = datetime.now(timezone.utc)
        start = end - timedelta(minutes=minutes)
        label = f"lookback {minutes}m"
    if end <= start:
        sys.exit(f"Error: EndTime ({end}) must be after StartTime ({start}).")
    return floor_minute(start - pad), ceil_minute(end + pad), label


def run_benchmark(
    passthrough: List[str], pad: timedelta
) -> Tuple[datetime, datetime]:
    """Run gwas_benchmark.py, returning the padded (start, end) bracketing it.

    Uses this interpreter (sys.executable) and inherits the environment, so the
    same MOSAIC_* config that selects the S3 backend flows through to the child.
    """
    if not os.path.exists(BENCHMARK):
        sys.exit(f"Error: benchmark not found at {BENCHMARK}")
    cmd = [sys.executable, BENCHMARK, *passthrough]
    print(f"Running: {' '.join(cmd)}\n", file=sys.stderr)
    start = datetime.now(timezone.utc)
    # Let stdout/stderr stream straight through to the user's terminal.
    result = subprocess.run(cmd)
    end = datetime.now(timezone.utc)
    if result.returncode != 0:
        print(
            f"Warning: benchmark exited non-zero ({result.returncode}); "
            "the reported bytes may not reflect a complete run.",
            file=sys.stderr,
        )
    return floor_minute(start - pad), ceil_minute(end + pad)


def query_window(
    cw: Any,
    bucket: str,
    filter_id: str,
    metric: str,
    start: datetime,
    end: datetime,
    period: int,
    unit: Optional[str] = None,
) -> Dict[str, Any]:
    """Sum `metric` over [start, end] via get_metric_data.

    Returns {"total", "points", "status", "messages"}, where `points` is a
    chronological list of {"timestamp", "value"} and `total` is the sum of the
    per-period Sum values (total bytes / request count over the window).

    get_metric_data is used in preference to get_metric_statistics because it
    does *not* require the stored Unit to be matched (a mismatched Unit makes
    get_metric_statistics return silently-empty results), it paginates via
    NextToken, and it returns a per-query StatusCode / Messages we surface for
    diagnostics. `unit` is None by default; pass it only to deliberately filter
    on a unit.
    """
    metric_stat: Dict[str, Any] = {
        "Metric": {
            "Namespace": "AWS/S3",
            "MetricName": metric,
            "Dimensions": [
                {"Name": "BucketName", "Value": bucket},
                {"Name": "FilterId", "Value": filter_id},
            ],
        },
        "Period": period,
        "Stat": "Sum",
    }
    if unit:
        metric_stat["Unit"] = unit
    query = {"Id": "m1", "MetricStat": metric_stat, "ReturnData": True}

    timestamps: List[datetime] = []
    values: List[float] = []
    status: Optional[str] = None
    messages: List[str] = []
    next_token: Optional[str] = None
    while True:
        kwargs: Dict[str, Any] = {
            "MetricDataQueries": [query],
            "StartTime": start,
            "EndTime": end,
            # Oldest-first, so the parallel Timestamps/Values arrays (and our
            # paged accumulation of them) are already chronological.
            "ScanBy": "TimestampAscending",
        }
        if next_token:
            kwargs["NextToken"] = next_token
        response = cw.get_metric_data(**kwargs)
        result = response["MetricDataResults"][0]
        timestamps.extend(result.get("Timestamps", []))
        values.extend(result.get("Values", []))
        status = result.get("StatusCode", status)
        for msg in result.get("Messages", []) + response.get("Messages", []):
            text = msg.get("Value", "")
            if text and text not in messages:
                messages.append(text)
        next_token = response.get("NextToken")
        if not next_token:
            break

    points = [{"timestamp": t, "value": v} for t, v in zip(timestamps, values)]
    return {
        "total": math.fsum(values),
        "points": points,
        "status": status,
        "messages": messages,
    }


def collect(
    cw: Any,
    args: argparse.Namespace,
    start: datetime,
    end: datetime,
) -> Tuple[Dict[str, Any], int]:
    """Query once, or poll until the value settles when --wait-minutes is set.

    S3 request metrics populate one 1-minute bucket at a time with multi-minute
    lag, so the running total *plateaus between bucket arrivals*. A naive
    "unchanged since the last poll" check mistakes such a plateau for the final
    value (and two runs can stop on different plateaus). Instead we require the
    reading to hold steady for a full ``--settle-minutes`` confirmation window
    before trusting it, keyed on BOTH the total and the datapoint count so a
    newly-arrived bucket — even a near-zero one — resets the timer.

    Adds two keys to the returned result: ``settled`` (True only if it confirmed
    stable; False if --wait-minutes elapsed first or it was a single query) and
    ``stable_seconds`` (how long the reading was unchanged at return). Returns
    (result, period).
    """
    period = safe_period(start, end, args.period)
    settle_secs = args.settle_minutes * 60
    waiting = args.wait_minutes > 0
    deadline = time.monotonic() + args.wait_minutes * 60
    last_key: Optional[Tuple[float, int]] = None
    last_change = time.monotonic()
    while True:
        result = query_window(
            cw,
            args.bucket,
            args.filter_id,
            args.metric,
            start,
            end,
            period,
            args.unit,
        )
        total = result["total"]
        npts = len(result["points"])
        now = time.monotonic()
        key = (total, npts)
        if key != last_key:
            last_key = key
            last_change = now
        stable_for = now - last_change
        result["stable_seconds"] = stable_for

        if not waiting:
            # Single query: no confirmation possible, so never claim settled.
            result["settled"] = False
            return result, period
        confirmed = total > 0 and stable_for >= settle_secs
        if confirmed or now >= deadline:
            result["settled"] = bool(confirmed)
            return result, period
        if not args.json:
            seen = (
                "no data yet"
                if total == 0
                else (
                    f"{total:,.0f} bytes across {npts} min-bucket(s), "
                    f"stable {int(stable_for)}s/{settle_secs}s"
                )
            )
            print(
                f"  metrics still settling ({seen}); "
                f"re-checking in {args.poll_seconds}s...",
                file=sys.stderr,
            )
        time.sleep(args.poll_seconds)


def make_cloudwatch_client(args: argparse.Namespace) -> Any:
    """Build a CloudWatch client in the bucket's region.

    Deliberately independent of the pipeline's S3 client kwargs: CloudWatch is
    always an AWS endpoint, never the MinIO/S3 endpoint override, so we pass only
    region (and an optional profile) and let boto3 resolve credentials from the
    environment / IAM role.
    """
    session = (
        boto3.Session(profile_name=args.profile)
        if args.profile
        else boto3.Session()
    )
    return session.client("cloudwatch", region_name=args.region)


def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Measure S3 BytesDownloaded for a benchmark run via CloudWatch "
            "request metrics. Choose a time window with --wrap, --lookback, or "
            "--start/--end."
        )
    )
    mode = p.add_argument_group(
        "time window (pick one; default: --lookback 15)"
    )
    mode.add_argument(
        "--wrap",
        action="store_true",
        help=(
            "Run gwas_benchmark.py and bracket it automatically. Pass the "
            "benchmark's own args after a literal '--'."
        ),
    )
    mode.add_argument(
        "--lookback",
        type=int,
        metavar="MINUTES",
        help="Window = the last MINUTES (EndTime=now). Default mode (15 min).",
    )
    mode.add_argument(
        "--start",
        help="Explicit StartTime, ISO-8601 (UTC assumed if no offset).",
    )
    mode.add_argument(
        "--end",
        help="Explicit EndTime, ISO-8601 (UTC assumed if no offset).",
    )
    mode.add_argument(
        "--pad",
        type=int,
        default=1,
        metavar="MINUTES",
        help=(
            "Minutes to widen each end of the window (snapped to minute "
            "buckets) so boundary requests are captured. Default: 1."
        ),
    )

    p.add_argument(
        "--bucket",
        default=DEFAULT_BUCKET,
        help=f"S3 bucket name (default: {DEFAULT_BUCKET}).",
    )
    p.add_argument(
        "--filter-id",
        default=DEFAULT_FILTER_ID,
        help=(
            "S3 request-metrics filter id configured on the bucket "
            f"(default: {DEFAULT_FILTER_ID}). A wrong filter id is the most "
            "common cause of an empty result."
        ),
    )
    p.add_argument(
        "--metric",
        default="BytesDownloaded",
        help=(
            "AWS/S3 request metric to sum (default: BytesDownloaded; e.g. "
            "BytesUploaded, GetRequests)."
        ),
    )
    p.add_argument(
        "--unit",
        default=None,
        help=(
            "Optional CloudWatch Unit filter (e.g. Bytes). Omitted by default: "
            "filtering on a Unit that doesn't match what S3 stored is a common "
            "cause of empty results."
        ),
    )
    p.add_argument(
        "--region",
        default=DEFAULT_REGION,
        help=(
            "AWS region the bucket's metrics live in "
            f"(default: {DEFAULT_REGION})."
        ),
    )
    p.add_argument(
        "--profile",
        help="AWS profile name for credentials (optional).",
    )
    p.add_argument(
        "--period",
        type=int,
        default=60,
        metavar="SECONDS",
        help=(
            "Aggregation granularity, rounded up to a 60s multiple and widened "
            "if needed to stay under CloudWatch's 1440-datapoint cap "
            "(default: 60)."
        ),
    )
    p.add_argument(
        "--wait-minutes",
        type=int,
        default=0,
        metavar="MINUTES",
        help=(
            "Max minutes to poll for the metric to settle. 0 = single query "
            "(default). S3 request metrics lag up to ~15 min, so for a "
            "confirmed reading use ~20 here with --settle-minutes."
        ),
    )
    p.add_argument(
        "--settle-minutes",
        type=int,
        default=3,
        metavar="MINUTES",
        help=(
            "Reading must stay unchanged this long to count as final "
            "(default: 3). Rides out the gaps as 1-min buckets trickle in; "
            "must be < --wait-minutes to ever confirm."
        ),
    )
    p.add_argument(
        "--poll-seconds",
        type=int,
        default=60,
        help=(
            "Seconds between polls when --wait-minutes is set (default: 60, "
            "matching the 1-minute metric granularity)."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON instead of a formatted report.",
    )
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    # Split off the benchmark's args (everything after a literal '--') before
    # argparse, so --wrap can forward them verbatim.
    if "--" in argv:
        i = argv.index("--")
        our_args, bench_args = argv[:i], argv[i + 1 :]
    else:
        our_args, bench_args = argv, []
    args = parse_args(our_args)

    if bench_args and not args.wrap:
        sys.exit("Error: arguments after '--' are only used with --wrap.")
    # Output formatting only: byte metrics get human-readable sizes, count
    # metrics (e.g. GetRequests) print as plain integers. This is independent
    # of the optional --unit *query* filter.
    is_bytes = "Byte" in args.metric

    if args.wait_minutes > 0 and args.settle_minutes >= args.wait_minutes:
        print(
            f"Warning: --settle-minutes ({args.settle_minutes}) >= "
            f"--wait-minutes ({args.wait_minutes}); the reading can never "
            "confirm stable and will always be reported unconfirmed.",
            file=sys.stderr,
        )

    # ---- Resolve the time window from the selected mode -------------------
    if args.wrap:
        if args.lookback is not None or args.start or args.end:
            sys.exit(
                "Error: --wrap sets the window itself; do not combine it with "
                "--lookback / --start / --end."
            )
        start, end = run_benchmark(bench_args, timedelta(minutes=args.pad))
        mode_label = "wrap"
    else:
        start, end, mode_label = resolve_window(args)

    cw = make_cloudwatch_client(args)

    unit_note = f" Unit={args.unit}" if args.unit else ""
    if not args.json:
        print(
            f"\nMetric : AWS/S3 {args.metric}{unit_note}\n"
            f"Bucket : {args.bucket}  (FilterId={args.filter_id})\n"
            f"Region : {args.region}\n"
            f"Window : {start.isoformat()}  ->  {end.isoformat()}  "
            f"[{mode_label}]\n"
            f"Re-run : --start {start.strftime('%Y-%m-%dT%H:%M:%S')} "
            f"--end {end.strftime('%Y-%m-%dT%H:%M:%S')}\n",
            file=sys.stderr,
        )

    try:
        result, period = collect(cw, args, start, end)
    except Exception as exc:  # boto/cred/permission errors -> clean message
        sys.exit(f"CloudWatch query failed: {type(exc).__name__}: {exc}")

    total = result["total"]
    points = result["points"]

    if args.json:
        print(
            json.dumps(
                {
                    "metric": args.metric,
                    "unit_filter": args.unit,
                    "bucket": args.bucket,
                    "filter_id": args.filter_id,
                    "region": args.region,
                    "mode": mode_label,
                    "start": start.isoformat(),
                    "end": end.isoformat(),
                    "period_seconds": period,
                    "settled": result.get("settled", False),
                    "stable_seconds": round(
                        result.get("stable_seconds", 0.0), 1
                    ),
                    "status": result["status"],
                    "messages": result["messages"],
                    "total": total,
                    "datapoints": [
                        {
                            "timestamp": p["timestamp"].isoformat(),
                            "value": p["value"],
                        }
                        for p in points
                    ],
                },
                indent=2,
            )
        )
        return 0

    if points:
        for p in points:
            value = (
                human_bytes(p["value"]) if is_bytes else f"{int(p['value']):,}"
            )
            print(f"  {p['timestamp'].isoformat()}  |  {value}")
        summary = (
            f"{int(total):,} bytes ({human_bytes(total)})"
            if is_bytes
            else f"{int(total):,}"
        )
        print(f"\nTotal {args.metric}: {summary}")
    else:
        print(
            "No datapoints (get_metric_data StatusCode="
            f"{result['status']!r}). Likely causes: (1) wrong --filter-id "
            f"(using {args.filter_id!r}) or request metrics not enabled on the "
            "bucket; (2) metrics not yet reported — retry with --wait-minutes, "
            "or re-query later with the --start/--end shown above; (3) pointed "
            "at MinIO, which emits no CloudWatch metrics."
        )

    # Surface anything CloudWatch flagged (e.g. PartialData while metrics are
    # still settling, or a permissions/throttling message).
    if result["status"] and result["status"] != "Complete":
        print(f"StatusCode: {result['status']}", file=sys.stderr)
    for msg in result["messages"]:
        print(f"  message: {msg}", file=sys.stderr)

    # Settle verdict — the crux of the "two runs disagree" problem: say plainly
    # whether this number is trustworthy or still populating.
    stable = int(result.get("stable_seconds", 0))
    if not points:
        pass  # the no-datapoints message above already explains it
    elif args.wait_minutes <= 0:
        print(
            "\nNote: single query (no --wait-minutes). S3 request metrics lag "
            "up to ~15 min and arrive one minute-bucket at a time, so this is "
            "likely PARTIAL. Re-run with e.g. --wait-minutes 20 "
            "--settle-minutes 3, or re-query later with the --start/--end "
            "above.",
            file=sys.stderr,
        )
    elif not result.get("settled"):
        print(
            f"\nWARNING: value NOT confirmed — hit --wait-minutes "
            f"({args.wait_minutes}) before it held steady for "
            f"--settle-minutes ({args.settle_minutes}); last changed {stable}s "
            "ago. It is probably still populating. Raise --wait-minutes, or "
            "re-query later with the --start/--end above.",
            file=sys.stderr,
        )
    else:
        print(
            f"\nConfirmed: held steady for {stable}s "
            f"(>= --settle-minutes {args.settle_minutes}).",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
