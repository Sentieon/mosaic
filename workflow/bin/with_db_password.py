#!/usr/bin/env python3
# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""
Populate MOSAIC_DB_PASSWORD from AWS Secrets Manager, then exec a command.

Workflow-layer AWS glue. The `mosaic` CLI stays cloud-agnostic -- it just
reads MOSAIC_DB_PASSWORD from the environment (see mosaic/config.py). The
Nextflow process scripts wrap their `mosaic` invocations with this helper so
that on AWS the RDS password is pulled from Secrets Manager at task start
(keeping the secret out of Nextflow's config dump / task env), while local
runs -- where MOSAIC_DB_PASSWORD_SECRET is unset -- are a no-op.

`fetch_and_ingest.py` imports ``fetch_db_password_into_env`` for the ingest
task; the AGGREGATE / INDEX processes wrap their commands with this script:

    with_db_password.py python -m mosaic.cli aggregate --fan-in 100
"""

import json
import os
import sys

# Secrets Manager secret name for the RDS DB password (AWS deployment).
# Set MOSAIC_DB_PASSWORD_SECRET to enable the fetch; when it is unset (local
# dev) the fetch is skipped entirely, even if AWS creds/region are present.
# When it IS set, fetching is mandatory (failure is fatal).
DB_PASSWORD_SECRET = os.environ.get("MOSAIC_DB_PASSWORD_SECRET")


def _err(msg):
    print(msg, file=sys.stderr, flush=True)


def fetch_db_password_into_env():
    """Populate MOSAIC_DB_PASSWORD from AWS Secrets Manager when configured.

    Gated on MOSAIC_DB_PASSWORD_SECRET being set: unset means local dev (skip,
    even when AWS_REGION is present for other AWS access). When the secret name
    IS set, the password must come from Secrets Manager, so any failure is
    fatal -- otherwise the run would silently fall back to config.py's
    local-dev default and fail RDS auth several frames later, burning every
    Nextflow retry on a confusing error. No-op if MOSAIC_DB_PASSWORD is already
    set.
    """
    if os.environ.get("MOSAIC_DB_PASSWORD"):
        return
    if not DB_PASSWORD_SECRET:
        return  # local dev -- no secret configured; config.py default is used

    region = os.environ.get("AWS_REGION") or os.environ.get(
        "AWS_DEFAULT_REGION"
    )
    try:
        import boto3  # type: ignore
    except ImportError:
        _err(
            "[ERROR] MOSAIC_DB_PASSWORD_SECRET is set but boto3 is "
            "unavailable to fetch it"
        )
        sys.exit(1)
    try:
        client = boto3.client("secretsmanager", region_name=region)
        resp = client.get_secret_value(SecretId=DB_PASSWORD_SECRET)
        secret_string = resp.get("SecretString")
    except Exception as e:
        _err(
            f"[ERROR] Failed to fetch {DB_PASSWORD_SECRET} from Secrets "
            f"Manager: {e}"
        )
        sys.exit(1)
    if not secret_string:
        _err(f"[ERROR] Secret {DB_PASSWORD_SECRET} has no SecretString")
        sys.exit(1)

    # Accept either a plaintext secret or a JSON object with a "password"
    # key (the shape produced by the RDS-managed / console key-value flows).
    password = secret_string
    try:
        obj = json.loads(secret_string)
    except (ValueError, TypeError):
        obj = None
    if isinstance(obj, dict):
        password = obj.get("password") or obj.get("MOSAIC_DB_PASSWORD")
        if not password:
            _err(
                f"[ERROR] Secret {DB_PASSWORD_SECRET} is JSON but has no "
                "'password' key"
            )
            sys.exit(1)

    os.environ["MOSAIC_DB_PASSWORD"] = password
    print(
        f"[INFO] Loaded MOSAIC_DB_PASSWORD from Secrets Manager "
        f"({DB_PASSWORD_SECRET})",
        flush=True,
    )


def main():
    if len(sys.argv) < 2:
        _err("usage: with_db_password.py <command> [args...]")
        sys.exit(2)
    fetch_db_password_into_env()
    # Replace this process with the wrapped command, which inherits the
    # MOSAIC_DB_PASSWORD we just exported (if any).
    os.execvp(sys.argv[1], sys.argv[1:])


if __name__ == "__main__":
    main()
