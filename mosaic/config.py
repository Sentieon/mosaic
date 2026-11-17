# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""
Shared configuration for all pipeline scripts.

All settings are read from environment variables with sensible defaults
for a local Docker-based development setup (PostgreSQL + Iceberg REST
catalog + MinIO).  See README.md for the full variable reference.
"""

import logging
import os
from typing import Any, Dict

# ── PostgreSQL ───────────────────────────────────────────────────────
DB_CONFIG: Dict[str, str] = {
    "dbname": os.environ.get("MOSAIC_DB_NAME", "postgres"),
    "user": os.environ.get("MOSAIC_DB_USER", "postgres"),
    "password": os.environ.get("MOSAIC_DB_PASSWORD", "mysecretpassword"),
    "host": os.environ.get("MOSAIC_DB_HOST", "localhost"),
    "port": os.environ.get("MOSAIC_DB_PORT", "5432"),
}
_sslmode = os.environ.get("MOSAIC_DB_SSLMODE")
if _sslmode:
    DB_CONFIG["sslmode"] = _sslmode
    # The CA bundle is only *required* for the verifying modes; `require`
    # encrypts without verifying the server cert. Read it with .get() so a
    # missing MOSAIC_DB_SSLCERT does not raise an opaque KeyError at import
    # time on every subcommand.
    _sslcert = os.environ.get("MOSAIC_DB_SSLCERT")
    if _sslcert:
        DB_CONFIG["sslrootcert"] = _sslcert
    elif _sslmode in ("verify-ca", "verify-full"):
        raise RuntimeError(
            f"MOSAIC_DB_SSLMODE={_sslmode!r} requires MOSAIC_DB_SSLCERT to "
            "point at a CA bundle (e.g. the RDS global-bundle.pem). Set "
            "MOSAIC_DB_SSLCERT, or use MOSAIC_DB_SSLMODE=require to encrypt "
            "without CA verification."
        )


# ── Apache Iceberg / S3 ─────────────────────────────────────────────
def _build_iceberg_config() -> Dict[str, str]:
    cfg: Dict[str, str] = {}
    iceberg_type = os.environ.get("MOSAIC_ICEBERG_TYPE", "rest")
    if iceberg_type == "rest":
        cfg["uri"] = os.environ.get(
            "MOSAIC_ICEBERG_URI", "http://localhost:8181"
        )
        cfg["s3.endpoint"] = os.environ.get(
            "MOSAIC_S3_ENDPOINT", "http://minio:9000"
        )
        cfg["s3.access-key-id"] = os.environ.get(
            "MOSAIC_S3_ACCESS_KEY_ID", "admin"
        )
        cfg["s3.secret-access-key"] = os.environ.get(
            "MOSAIC_S3_SECRET_ACCESS_KEY", "password"
        )
        cfg["s3.path-style-access"] = os.environ.get(
            "MOSAIC_S3_PATH_STYLE", "true"
        )
    else:
        cfg["type"] = iceberg_type
    cfg["s3.region"] = os.environ.get("MOSAIC_S3_REGION", "us-east-1")
    warehouse = os.environ.get("MOSAIC_WAREHOUSE")
    if warehouse:
        cfg["warehouse"] = warehouse
    return cfg


ICEBERG_CATALOG_NAME: str = os.environ.get("MOSAIC_ICEBERG_TYPE", "rest")
ICEBERG_CONFIG: Dict[str, str] = _build_iceberg_config()
ICEBERG_NAMESPACE: str = os.environ.get("MOSAIC_ICEBERG_NAMESPACE", "mosaic")
S3_BUCKET: str = os.environ.get("MOSAIC_S3_BUCKET", "warehouse")


class _RestMetadataCleanupFilter(logging.Filter):
    """Drop pyiceberg's "Failed to delete metadata file" warnings under the
    REST catalog.

    Every MOSAIC table sets ``write.metadata.delete-after-commit.enabled``
    (`mosaic init`, `mosaic maintain`). After each commit both the REST
    server and the committer (pyiceberg's ``Table._do_commit``) delete the
    ``metadata.json`` versions that fell out of the metadata log. The server
    deletes them inside the commit, so pyiceberg's own delete then finds each
    file gone (FileNotFoundError) and logs one warning per version: noise, not
    a failure. Under Glue pyiceberg is the only deleter and its warnings are
    kept. A version a genuine failure left behind is unreferenced, and `mosaic
    maintain --orphans` reclaims it.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        return not str(record.msg).startswith("Failed to delete metadata file")


if ICEBERG_CATALOG_NAME == "rest":
    logging.getLogger("pyiceberg.catalog").addFilter(
        _RestMetadataCleanupFilter()
    )


def build_s3_client_kwargs() -> Dict[str, Any]:
    """Build the kwargs for a boto3 ``s3`` client.

    For the local ``rest`` catalog (MinIO) this supplies the custom endpoint
    and dev credentials; for any other catalog type (e.g. Glue on AWS) it
    returns only the region and lets boto3 resolve credentials from the IAM
    role / instance metadata. Keying off the catalog type (not the presence of
    ``AWS_ACCESS_KEY_ID``) is what makes this correct on AWS Batch, where
    credentials come from the ECS task role rather than env vars.

    Distinct from ``ICEBERG_CONFIG``: those are pyiceberg keys
    (``s3.endpoint`` / ``s3.access-key-id`` ...) and default the endpoint to
    ``minio:9000`` (resolved inside the Docker network), whereas a boto3 client
    runs host-side and uses ``localhost:9000``.
    """
    kwargs: Dict[str, Any] = {
        "region_name": os.environ.get("MOSAIC_S3_REGION", "us-east-1")
    }
    if ICEBERG_CATALOG_NAME == "rest":
        kwargs["endpoint_url"] = os.environ.get(
            "MOSAIC_S3_ENDPOINT", "http://localhost:9000"
        )
        kwargs["aws_access_key_id"] = os.environ.get(
            "MOSAIC_S3_ACCESS_KEY_ID", "admin"
        )
        kwargs["aws_secret_access_key"] = os.environ.get(
            "MOSAIC_S3_SECRET_ACCESS_KEY", "password"
        )
    return kwargs


# ── Parquet write tuning (Iceberg data files) ───────────────────────
# Overridable via env so the write path can be benchmarked/tuned without
# code edits; defaults preserve production behavior. Module-level, so each
# process (a spawned worker of `mosaic aggregate` or `mosaic remove`
# included) reads the current environment at import.
PARQUET_DATA_PAGE_SIZE: int = int(
    os.environ.get("MOSAIC_PARQUET_PAGE_SIZE", 128 * 1024)
)  # 128 KB default (smaller than the 1 MB pyarrow default)
PARQUET_COMPRESSION: str = os.environ.get(
    "MOSAIC_PARQUET_COMPRESSION", "brotli"
)
# Rows per row group of the per-sample tables' data files. Each file holds
# whole samples of one batch and one chromosome, rows sorted by position then
# sample_id, so a row group spans about rows / (samples in the file x rows per
# kb) of the chromosome: what a position-window reader (joint_call.py)
# decodes beyond its window. The tables' row densities differ (one chr21
# sample: ~54 ref-block rows and ~1.6 site rows per kb), hence one cap each.
# sample_ref_blocks: the partials' 65,536 (PARQUET_ROW_GROUP_SIZE in
# ingest.py).
PARQUET_ROW_GROUP_ROWS: int = int(
    os.environ.get("MOSAIC_PARQUET_ROW_GROUP_ROWS", 64 * 1024)
)
# sample_site_info.
PARQUET_SITE_ROW_GROUP_ROWS: int = int(
    os.environ.get("MOSAIC_PARQUET_SITE_ROW_GROUP_ROWS", 8 * 1024)
)
