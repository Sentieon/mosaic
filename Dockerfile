# Copyright (c) 2026 Sentieon Inc. All rights reserved
# Image for MOSAIC pipeline workers on AWS Batch: the ingestion workflow
# (workflow/main.nf: ingest, register, aggregate, index, maintain) and the
# joint-call workflow (workflow/joint_call.nf: joint_call.py, run by
# workflow/bin/joint_call_task.py).
#
# Stages:
#   rust-build: the Rust workspace (rust/): the ingest worker
#      (`mosaic-ingest-worker`), the one engine `mosaic ingest` runs,
#      and the joint caller (`mosaic-joint-call`), with the
#      library crate both use (`mosaic-core`). The rust image's version is
#      the one pinned in rust/rust-toolchain.toml; only the workspace files,
#      the lock and the crates are copied in, and `cargo build --release
#      --locked` builds them.
#   the runtime image below, which copies the binaries to
#      /usr/local/bin/mosaic-ingest-worker and
#      /usr/local/bin/mosaic-joint-call (layer 7).
#
# Layers of the runtime image (top-to-bottom for cache efficiency):
#   1. Python base from AWS public ECR (avoids Docker Hub throttling)
#   2. System deps: build toolchain, libpq (psycopg2 builds from
#      source), htslib runtime deps for bcftools, curl/unzip
#      (AWS CLI install), tabix from apt
#   3. bcftools built from source (Debian's apt bcftools is too old —
#      we want >=1.23 for `--write-index=tbi` and S3 perf fixes)
#   4. AWS CLI v2 — required by Nextflow's awsbatch executor for
#      S3 staging; path must match aws.batch.cliPath in awsbatch.config.
#      The zip matches the image's architecture (x86_64 or aarch64), so
#      the image also builds natively on an arm64 machine; the production
#      image (AWS Batch) is linux/amd64: build it on an x86_64 host or
#      with `docker build --platform linux/amd64`.
#   5. Python deps (requirements.txt) — cached when deps don't change
#   6. Package source only (mosaic/, pyproject.toml, README.md, LICENSE,
#      bin/: the workflow helpers and joint_call.py) + `pip install .`
#      (vcflib is pulled from PyPI) — not the whole repo, so tests/, conf/,
#      the Nextflow files, etc. stay out
#   7. The Rust binaries from the rust-build stage, on PATH
#
# Build + push:
#   aws ecr get-login-password --region us-east-1 \
#     | docker login --username AWS \
#         --password-stdin <ACCT>.dkr.ecr.us-east-1.amazonaws.com
#   docker build -t mosaic:latest .
#   docker tag mosaic:latest \
#     <ACCT>.dkr.ecr.us-east-1.amazonaws.com/mosaic:latest
#   docker push <ACCT>.dkr.ecr.us-east-1.amazonaws.com/mosaic:latest

# ── the Rust binaries ───────────────────────────────────────────────────────
# Keep the tag's version equal to rust/rust-toolchain.toml's channel (the
# RUN below fails otherwise). RUSTUP_TOOLCHAIN selects the image's own
# toolchain: the toolchain file also names rustfmt and clippy, which the
# slim image lacks, and honoring it would download them into the build.
FROM rust:1.99.1-slim-bookworm AS rust-build
ENV RUSTUP_TOOLCHAIN=1.99.1
WORKDIR /build
COPY rust/Cargo.toml rust/Cargo.lock rust/rust-toolchain.toml ./
COPY rust/mosaic-core/ ./mosaic-core/
COPY rust/mosaic-ingest-worker/ ./mosaic-ingest-worker/
COPY rust/mosaic-joint-call/ ./mosaic-joint-call/
RUN grep -qx "channel = \"$(rustc --version | cut -d' ' -f2)\"" \
         rust-toolchain.toml \
    && cargo build --release --locked \
    && strip target/release/mosaic-ingest-worker \
             target/release/mosaic-joint-call

# ── the runtime image ───────────────────────────────────────────────────────
FROM public.ecr.aws/docker/library/python:3.12-slim-bookworm

# System deps. bcftools is built from source below; here we install
# only the build toolchain + htslib runtime deps (zlib/bz2/lzma/curl/
# openssl) that ./configure links against, plus libpq for psycopg2 and
# tabix from apt (its CLI is stable across versions and good enough
# for any direct invocation outside bcftools).
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
         tabix \
         curl \
         unzip \
         ca-certificates \
         build-essential \
         libpq-dev \
         postgresql-client \
         zlib1g-dev \
         libbz2-dev \
         liblzma-dev \
         libcurl4-openssl-dev \
         libssl-dev \
         git \
         procps \
    && rm -rf /var/lib/apt/lists/*

# Install bcftools from source (Debian apt ships an older version).
ARG BCFTOOLS_VERSION=1.23.1
ARG BCFTOOLS_SHA256=01899a46f9420cdc1385d52fcfc84cce2806f9c996b787081a90d7dfc85eafa3
RUN mkdir -p /opt/bcftools/ \
    && curl -fL -o /tmp/bcftools.tar.bz2 \
         "https://github.com/samtools/bcftools/releases/download/${BCFTOOLS_VERSION}/bcftools-${BCFTOOLS_VERSION}.tar.bz2" \
    && echo "${BCFTOOLS_SHA256}  /tmp/bcftools.tar.bz2" | sha256sum -c - \
    && tar -C /opt/bcftools/ -jxf /tmp/bcftools.tar.bz2 \
    && rm /tmp/bcftools.tar.bz2 \
    && cd /opt/bcftools/bcftools-${BCFTOOLS_VERSION}/ \
    && ./configure \
    && make install \
    && strip /usr/local/bin/bcftools

# AWS CLI v2 — needed by the Nextflow awsbatch executor for S3 staging.
# The zip for the image's architecture (x86_64 / aarch64).
RUN curl -sSf "https://awscli.amazonaws.com/awscli-exe-linux-$(uname -m).zip" \
         -o /tmp/awscliv2.zip \
    && unzip -q /tmp/awscliv2.zip -d /tmp \
    && /tmp/aws/install \
    && rm -rf /tmp/aws /tmp/awscliv2.zip

# RDS global CA bundle — required when MOSAIC_DB_SSLMODE=require so
# psycopg2 can verify the RDS cert. Baked into the image (independent
# of whether the host has the file locally).
RUN curl -sSf -o /opt/global-bundle.pem \
         "https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem"

# Install Python deps before copying the repo so pip cache is reused
# across source-only changes. (vcflib is pulled from PyPI here.)
WORKDIR /opt/mosaic
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the package build + runtime need — not the whole repo, so
# tests/ and the rest of the Nextflow project (workflow/) never enter the
# image. The build reads README.md (project.readme) and LICENSE
# (project.license-files), so both must sit alongside pyproject.toml. The
# workflow/bin/ helpers (fetch_and_ingest.py and joint_call_task.py, which
# both import with_db_password.py) are the AWS glue and must stay co-located
# on PATH; mosaic/ carries ingest, aggregate, and index. joint_call.py, the
# driver of the joint caller (not part of the `mosaic` CLI),
# goes beside them, so the image holds the driver that matches its
# mosaic-joint-call binary; joint_call_task.py finds it on PATH. Then install
# the package (deps already satisfied above) so the `mosaic` console command
# is on PATH.
COPY pyproject.toml README.md LICENSE ./
COPY mosaic/ ./mosaic/
COPY workflow/bin/fetch_and_ingest.py workflow/bin/with_db_password.py \
     workflow/bin/joint_call_task.py joint_call.py ./bin/
RUN pip install --no-cache-dir --no-deps .

# The Rust ingest worker (`mosaic-ingest-worker`): `mosaic ingest`'s one
# shard worker (the earlier Python worker has been removed), which it
# finds on PATH when
# MOSAIC_INGEST_WORKER is unset.
COPY --from=rust-build /build/target/release/mosaic-ingest-worker \
     /usr/local/bin/mosaic-ingest-worker
# The Rust joint caller (`mosaic-joint-call`): the one engine of
# `joint_call.py` (its driver, in /opt/mosaic/bin above; the earlier Python
# engine has been removed),
# which finds it on PATH when MOSAIC_JOINT_CALL_WORKER is unset.
COPY --from=rust-build /build/target/release/mosaic-joint-call \
     /usr/local/bin/mosaic-joint-call

# bin/ holds the workflow helpers (Nextflow also adds it to PATH per task).
ENV PYTHONUNBUFFERED=1 \
    PATH=/opt/mosaic/bin:$PATH

# No ENTRYPOINT — Nextflow supplies its own command per task.
