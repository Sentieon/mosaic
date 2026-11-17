# Copyright (c) 2026 Sentieon Inc. All rights reserved
"""MOSAIC — a scalable gVCF ingestion/storage pipeline.

Ingests per-sample gVCF files into a hybrid store: metadata and variant
sites in PostgreSQL, per-sample genotype data as Parquet in S3-compatible
object storage managed by an Apache Iceberg catalog.
"""

__version__ = "0.1.0"
