# Sentieon MOSAIC

MOSAIC is a scalable pipeline that ingests per-sample gVCF files into a hybrid
PostgreSQL + Apache Iceberg store. Metadata and variant sites live in
PostgreSQL; per-sample genotype data is stored as Parquet files in
S3-compatible object storage and managed by an Iceberg catalog. The result is a
cross-sample store with population-level aggregates, ready for downstream
analysis at scale. The design, its invariants and the implementation status are described in
[`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md); benchmark results on up to 3,202 whole
genomes are in [`docs/benchmarks.md`](docs/benchmarks.md).

## Prerequisites

- Python 3.10+
- PostgreSQL 10+ (uses declarative table partitioning; 12+ recommended)
- Apache Iceberg REST catalog (or AWS Glue)
- S3-compatible object storage (MinIO for local dev, AWS S3, etc.)
- For `mosaic ingest`: the `mosaic-ingest-worker` binary, built from `rust/` with a
  Rust toolchain ([`rustup`](https://rustup.rs); the version is pinned in
  `rust/rust-toolchain.toml`), on `PATH` or named by `MOSAIC_INGEST_WORKER`. It is
  required: `mosaic ingest` has no other shard worker (without it, it exits 1 before
  any work). The Docker image builds it in its own stage.
- For `joint_call.py`: the `mosaic-joint-call` binary, which the same `rust/`
  workspace builds into `rust/target/release/` beside the ingest worker;
  `joint_call.py` finds it through `MOSAIC_JOINT_CALL_WORKER` or `PATH`. It is
  required: `joint_call.py` is its driver and has no engine of its own to fall
  back to (see **6. Joint calling**)

## Installation

```bash
pip install .
```

This installs the `mosaic` command and its dependencies, including
[`vcflib`](https://github.com/Sentieon/vcflib) from PyPI. For a fully pinned
environment, use the lockfile:

```bash
pip install -r requirements.txt && pip install --no-deps .
```

Then build the Rust ingest worker and joint caller and put them on `PATH` (or
export `MOSAIC_INGEST_WORKER` and `MOSAIC_JOINT_CALL_WORKER` with their paths):

```bash
cd rust && cargo build --release && cd ..
export PATH="$PWD/rust/target/release:$PATH"   # or MOSAIC_INGEST_WORKER=$PWD/rust/target/release/mosaic-ingest-worker
```

The Python package does not depend on them (`pip install` is unchanged), but the
commands do: `mosaic ingest` needs `mosaic-ingest-worker` and `joint_call.py` needs
`mosaic-joint-call` (each exits 1 with the remedy when its binary is missing).

A local Docker stack (PostgreSQL + Iceberg REST + MinIO) for development lives
in `iceberg_test/`:

```bash
cd iceberg_test && docker-compose up -d
```

## Pipeline Overview

A single `mosaic` command with subcommands:

```
mosaic init ──> mosaic ingest ──> [mosaic register] ──> mosaic aggregate ──> mosaic index
                                                                        └──> mosaic maintain
removing samples:   mosaic remove ──> mosaic aggregate ──> mosaic index + mosaic maintain
```

| Subcommand | Purpose |
|---|---|
| `mosaic init <gvcf> [--chromosomes ...]` | Create PostgreSQL + Iceberg tables and populate contigs (run once) |
| `mosaic ingest <gvcfs...>` | Ingest one or more gVCF files (sharded, in the `mosaic-ingest-worker` binary) |
| `mosaic register` | Register the files of batches ingested with `--defer-register` with Iceberg, many files per commit, and complete those batches (a no-op otherwise) |
| `mosaic aggregate` | Reduce the per-batch Iceberg partials into the PostgreSQL aggregates |
| `mosaic remove <samples...>` | Remove samples: tombstone them, write their contribution as a negative batch for the next `mosaic aggregate`, and delete their per-sample rows (an initial version) |
| `mosaic index` | Check PostgreSQL indexes: re-create the `alleles` GiST if missing (run after aggregation) |
| `mosaic maintain` | Iceberg maintenance: rewrite each table's manifests, expire old snapshots and delete the files only they needed, optionally sweep orphan files (run after aggregation) |
| `mosaic clean` | Reset PostgreSQL + Iceberg state |

## Configuration

All configuration is read from environment variables. The defaults connect to
the local Docker stack in `iceberg_test/`, so **no variables need to be set for
local development**. The default credentials (`mysecretpassword` / `admin` /
`password`) are the zero-config local-dev values baked into
`iceberg_test/docker-compose.yml` — they are not secrets.

| Variable | Default | Purpose |
|---|---|---|
| `MOSAIC_DB_NAME` | `postgres` | Database name |
| `MOSAIC_DB_USER` | `postgres` | Database user |
| `MOSAIC_DB_PASSWORD` | `mysecretpassword` | Database password (local-dev default) |
| `MOSAIC_DB_HOST` | `localhost` | Database host |
| `MOSAIC_DB_PORT` | `5432` | Database port |
| `MOSAIC_DB_SSLMODE` | _(unset)_ | Set to `require` for RDS |
| `MOSAIC_DB_SSLCERT` | _(unset)_ | Path to SSL CA cert (required when `MOSAIC_DB_SSLMODE` is set) |
| `MOSAIC_ICEBERG_TYPE` | `rest` | Catalog type (`rest` or `glue`) |
| `MOSAIC_ICEBERG_URI` | `http://localhost:8181` | REST catalog URI (ignored for Glue) |
| `MOSAIC_S3_ENDPOINT` | _(unset)_ | Custom S3 endpoint for MinIO; omit for AWS S3. When unset under the `rest` catalog the effective default is `minio:9000` (pyiceberg) / `localhost:9000` (boto3). |
| `MOSAIC_S3_ACCESS_KEY_ID` | `admin` | S3 access key (local-dev default) |
| `MOSAIC_S3_SECRET_ACCESS_KEY` | `password` | S3 secret key (local-dev default) |
| `MOSAIC_S3_PATH_STYLE` | `true` | Path-style S3 access (for MinIO; omit for AWS S3) |
| `MOSAIC_S3_REGION` | `us-east-1` | AWS region |
| `MOSAIC_S3_BUCKET` | `warehouse` | S3 bucket for Parquet data files |
| `MOSAIC_WAREHOUSE` | _(unset)_ | Iceberg warehouse URI (e.g. `s3://my-warehouse/`) |
| `MOSAIC_ICEBERG_NAMESPACE` | `mosaic` | Iceberg namespace |
| `MOSAIC_PARQUET_ROW_GROUP_ROWS` | `65536` | Rows per row group in `sample_ref_blocks` data files (read by `mosaic ingest`'s writer and by `mosaic init`, which sets the table's `write.parquet.row-group-limit` for pyiceberg's rewrites: set it identically for both, or not at all) |
| `MOSAIC_PARQUET_SITE_ROW_GROUP_ROWS` | `8192` | The same for `sample_site_info` |
| `MOSAIC_INGEST_WORKER` | _(unset)_ | Path of the `mosaic-ingest-worker` binary `mosaic ingest` runs; unset, `mosaic-ingest-worker` on `PATH` |
| `MOSAIC_JOINT_CALL_WORKER` | _(unset)_ | Path of the `mosaic-joint-call` binary `joint_call.py` runs; unset, `mosaic-joint-call` on `PATH` |

### AWS deployment example

```bash
# DB configuration
export MOSAIC_DB_HOST=my-db.xxxxxxxx.us-east-1.rds.amazonaws.com
export MOSAIC_DB_PASSWORD=<from-secrets-manager>
export MOSAIC_DB_SSLMODE=require
export MOSAIC_DB_SSLCERT=/path/to/global-bundle.pem

# Iceberg / S3 configuration
export MOSAIC_ICEBERG_TYPE=glue
export MOSAIC_S3_REGION=us-east-1
export MOSAIC_S3_BUCKET=my-warehouse-bucket
export MOSAIC_WAREHOUSE=s3://my-warehouse-bucket/
# MOSAIC_S3_ENDPOINT intentionally unset — uses AWS S3 natively.
# AWS credentials come from the instance/task role.
```

Size the RDS volume for the aggregate's finalize, or enable storage autoscaling
(`MaxAllocatedStorage`) so a full disk grows instead of killing the run. Genome-wide
at 3,202 samples the database is 69.8 GB (65 GiB) after the aggregate, which ran on a 100 GiB
volume that also held four smaller test databases (16.6 GB) and ended with 15.6 GB
free.
`workflow/conf/awsbatch.config` and `workflow/nextflow.config` carry the settings
those runs used (`maxForks = 30`, `pos_batch_size = 100000`), except
`aggregate_threads`, which defaults to the task's cpus. **At 3,202 samples
genome-wide pass `--aggregate_threads 2`** (or 1): at the default 4 threads the
aggregate runs out of memory in a 12 GiB task (see **`AGGREGATE` memory and
storage** below).

## Usage

### 1. Initialize

Creates PostgreSQL tables (without indexes) and Iceberg tables, and populates
the `contigs` table from a gVCF header.

```bash
mosaic init sample.gvcf.gz                       # all chromosomes
mosaic init sample.gvcf.gz --chromosomes chr21,chr22
```

### 2. Ingest

Ingests one or more gVCF files in the Rust worker (`mosaic-ingest-worker`). Each genomic
shard is processed independently; per-sample genotype data and per-batch
aggregate partials are written as Parquet, uploaded to S3, and registered with
Iceberg.

```bash
mosaic ingest sample1.gvcf.gz sample2.gvcf.gz

mosaic ingest *.gvcf.gz --threads 8 --shard-size 500000 --pos-batch-size 150000
```

| Option | Default | Description |
|---|---|---|
| `--threads` | CPU count | Parallel shards, and then parallel combined files: threads of the one worker process |
| `--shard-size` | 10,000,000 | Genomic region size (bp) per shard |
| `--pos-batch-size` | 10,000,000 | Genomic window size (bp) per Parquet file |
| `--aggregate-parquet-size` | 268,435,456 | Target aggregated Parquet file size (bytes). The per-sample tables pack whole samples up to it: a sample larger than the target gets a file of its own |
| `--region` | _(unset)_ | Scope the run (and its checkpoint) to a region, e.g. `chr21` |
| `--defer-register` | off | Upload the Parquet but leave its Iceberg registration to `mosaic register` (see below) |
| `--ref-gq-bands` | off | Ref-block banding: store a sample's adjacent reference blocks of one GT class and GQ band as one row (see below). A comma-separated list of ascending GQ bounds in [1, 99] (`20,60`) or `all` |

Ingestion supports **checkpoint/resume**: if a run is interrupted, re-running
with the same sample set resumes from the last completed shard.
`--pos-batch-size` controls memory use (smaller = lower memory). **The CLI
defaults (10 Mbp shard and window) are deliberately coarse** — the Nextflow
path and `workflow/bin/fetch_and_ingest.py` override them to `--shard-size 500000
--pos-batch-size 150000`, which we benchmarked at 200 gVCFs per batch. A bare
`mosaic ingest` with the defaults uses far more memory and can OOM on a large
batch; lower `--pos-batch-size` if that happens.

**The worker.** The shard work (reading the gVCFs, merging them by position and
writing the local Parquet) runs in the Rust worker: one `mosaic-ingest-worker`
process per batch, its shards on `--threads` threads. After its last shard the same
process combines the local files into the data files (the per-sample tables'
whole-sample files, the partials' concatenation) on `--threads` threads; `mosaic
ingest` uploads each file as it is finished (4 upload threads) and registers them.
The test suite checks the worker's output against frozen signatures (see
**Testing**).

### Deferred registration (`--defer-register` + `mosaic register`)

By default each `mosaic ingest` ends by committing its data files to the six
shared Iceberg tables itself. With many concurrent ingests those commits
contend, and each one rewrites the table's `metadata.json`. With
`--defer-register` the batch uploads its Parquet as usual, records the file
list in the PostgreSQL table `ingest_files` and ends in status `uploaded`;
`mosaic register` later registers every pending file with one committer per
table (`--files-per-commit`, default 500, files per commit) and marks the
batches `complete`. Until then `mosaic aggregate` and the readers do not see an
`uploaded` batch (`mosaic aggregate` prints a warning); re-running `mosaic ingest`
on it does nothing. The registrar registers to Iceberg first and records that in
PostgreSQL second, so an interrupted `mosaic register` is simply re-run. Only one
registrar runs at a time (a PostgreSQL advisory lock). The Nextflow workflow uses
this path. Registration does not expire Iceberg snapshots; `mosaic maintain` does.

```bash
mosaic ingest --defer-register batch1/*.gvcf.gz
mosaic ingest --defer-register batch2/*.gvcf.gz
mosaic register                  # one commit per table for both batches
```

### Ref-block banding (`--ref-gq-bands`)

Many tools band gVCFs finely: on the six chr21 test samples the median reference block is
2–3 bp long, and the per-sample reference blocks are most of the lake's bytes (80%
genome-wide at 100 samples). `--ref-gq-bands` merges them at ingest. It is **off by
default**: without it every reference block is stored as the gVCF has it.

- **What merges.** A sample's reference blocks that are adjacent (no gap, no variant record
  between them), of the same GT class (hom-ref or no-call), whose GQ falls in the same band,
  and that lie in the same `--pos-batch-size` window, become one `sample_ref_blocks` row.
  The spec lists the bands' exclusive upper bounds, as GATK's `--gvcf-gq-bands` does: `20,60`
  is the bands [0, 20), [20, 60) and [60, 99]; `all` is one band. A block without GQ (no
  FORMAT/GQ and no PL) is in a band of its own. Under banding a block that crosses a window
  edge is cut there.
- **The merged values.** `dp` is the length-weighted mean of the blocks' depths (each
  the gVCF block's MIN_DP), rounded half up: the depth summed over the run's positions is
  kept up to that rounding, and a run whose blocks share one depth keeps it exactly. AD and
  PL are those of the block with the lowest GQ (ties: the first), so no position is
  reported better than it was;
  `sample_counts` is the same for all of them. The row's contribution to
  `batch_ref_blocks` is emitted from the same values, so the partials stay derivable from
  the per-sample rows and `mosaic remove` stays exact.
- **What changes in the output.** PG `ref_blocks.dp`, so INFO/DP and QD at variant sites
  (up or down), and the FORMAT DP, AD and PL that `joint_call.py`
  fills in for a sample with a reference block at the site. **What does not:** GT, AC, AN
  and every other INFO field, `site_info`, `alleles`, the genotype matrix
  (`batch_genotypes`) and `sample_qc`, which read the raw records.
- **Provenance.** The batch records its normalized spec in
  `ingestion_progress.ref_gq_bands` (NULL: no banding). A resumed batch must use the same
  value (`mosaic ingest` exits 1 otherwise, naming the stored one). Batches ingested with
  different settings can be aggregated together.

Measured on the six chr21 samples as three 2-sample batches (`--region chr21 --threads 2`,
CLI defaults):

| | off | `20` | `20,60` | `all` |
|---|---|---|---|---|
| `sample_ref_blocks` rows (all six samples) | 12,591,956 | 1,838,511 (6.8× fewer) | 5,002,492 (2.5×) | 473,893 (26.6×) |
| `sample_ref_blocks` bytes | 112.3 MB | 17.6 MB (6.4× smaller) | 46.1 MB (2.4×) | 4.5 MB (24.7×) |
| the six ingest tables' bytes | 393.8 MB | 285.8 MB | 318.9 MB | 271.0 MB |
| `mosaic remove` of one sample: wall (ref-block rows it reads) | 16.2–16.9 s (2,966,248) | 13.6 s (362,684) | 14.4 s (1,199,826) | 13.6 s (97,733) |
| VCF records changed (of 132,382) | — | 93,233 (70%) | 89,997 (68%) | 99,981 (76%) |
| INFO/DP changed at: up / down (median absolute change) | — | 63% of records: 40,761 / 42,995 (4 = 2.4%) | 58%: 39,027 / 37,713 (3 = 1.7%) | 70%: 43,242 / 48,851 (4 = 2.8%) |
| INFO/DP summed over all records | — | −0.14% | −0.07% | −0.72% |
| ref-filled calls with GQ < 20 (GQ from the PL) | 14.1% | 14.1% | 14.1% | 59.5% |
| n=2 vs GVCFtyper, `formatkeys=GT,PL,DP infokeys=DP,AC,AF,AN` | `+/- 128 = 56644` | `+/- 23830 = 32942` | `+/- 21867 = 34905` | `+/- 28005 = 28767` |
| n=2 vs GVCFtyper, `formatkeys=GT infokeys=AC,AF,AN` | `+/- 117 = 56655` | `+/- 117 = 56655` | `+/- 117 = 56655` | `+/- 117 = 56655` |

GT, AC, AN and every INFO field but DP and QD are identical at every record under every
setting, so the genotype-level comparison does not move; the DP and PL comparisons do,
since GVCFtyper reports the gVCF's own blocks and vcfdiff counts any change (at `20,60`
the PL alone differs at 19,512 records, DP alone at 16,499). The mean moves INFO/DP both
ways by a few percent: the largest relative changes are at low-depth sites (at `20,60`,
90% of the changed sites move by at most 5.3%). The raw blocks of these
samples have a median length of 2–3 bp; `20,60` merges them 2.4–2.7×, `all` 19–32×. What
bounds a merged run is the variant records between blocks, not the windows: at the
workflow's 150 kb windows the row counts are within 0.4% of these.

Measured on AWS at `20`:
- **Storage:** on chr21 × 3,202 the ingest tables are 75% smaller (`sample_ref_blocks`
  8.8× fewer bytes). Genome-wide at 3,202 samples the whole lake is 1.50 TB, 11.6% of
  the 12.98 TB of source gVCFs, and the per-sample tables are 0.39 GB per sample against
  4.05 GB of gVCF. Unbanded, the ingest tables take 2.17 GB per sample genome-wide (at
  100 samples).
- **Calls:** on chr21 × 3,202, the joint calls of 17 sets of 200 samples, banded against
  unbanded, are identical in GT, AC, AN, AF, QUAL, FS, MQ, SOR and InbreedingCoeff at
  all 5.32 billion genotypes. Only INFO/DP and QD and the FORMAT DP / AD / PL of hom-ref
  and no-call fills differ.

### 3. Aggregate

After all ingest batches are registered, reduce the per-batch Iceberg partials
into the cross-sample PostgreSQL aggregates — a materialized tier-tree reduce
with its own checkpoint/resume.

```bash
mosaic aggregate                 # resumes an interrupted run if present
mosaic aggregate --fan-in 100 --threads 8
mosaic aggregate --rebuild       # force a fresh run instead of resuming
```

### Removing samples (`mosaic remove`)

`mosaic remove` takes samples out of the cohort; the following `mosaic aggregate`
applies it, after which the PostgreSQL aggregates and the joint-called output are
those of a fresh ingest of the remaining samples. It is an initial version: its
exactness is tested on six chr21 samples, it has run on AWS at chr21 × 100
samples (two samples removed in 31 s, the aggregate that applied it 30 s), and it is
not part of the Nextflow workflow.

```bash
mosaic remove HG00100 HG00101          # tombstone, negative batch, delete rows
mosaic aggregate                       # folds the negative batch in (incremental)
mosaic index
mosaic maintain --older-than-hours 0   # the deleted rows' files leave S3 here
```

What it does, in order (each step resumable: re-run `mosaic remove` with the same
samples after a failure):

1. **Tombstones the samples** (`samples.status = 'removed'`): from then on the
   readers hide their rows and `mosaic ingest` refuses them. Until the following
   `mosaic aggregate` the aggregates still count them, so `joint_call.py` refuses the
   stale cohort.
2. **Re-derives their exact contribution** from their own per-sample rows (each row
   stores it) and writes it, negated, as one ingest batch of kind `removal`
   (`batch_site_info`, `batch_allele_counts`, `batch_ref_blocks`), registered with
   Iceberg like an ingest batch. Nothing is subtracted in place: the next `mosaic
   aggregate` sums it like any added batch and drops the sites, alleles and
   reference positions only the removed samples had.
3. **Deletes their rows** from `sample_site_info` and `sample_ref_blocks` (a
   copy-on-write delete that rewrites only the files holding them). The bytes leave
   S3 when `mosaic maintain` expires the snapshots that still name the old files.

It refuses, changing nothing, on an unknown or already removed sample, a sample in an
ingest batch that is not `complete` (an `uploaded` one needs `mosaic register`
first), another unfinished removal, and a run while `mosaic register` or `mosaic
maintain` holds the shared lock. It does not scrub `batch_genotypes`: a removed
sample keeps its slot and bytes there, and the reader (`mosaic/genotypes.py`) hides
its column. A removed sample cannot be re-added.

| Option | Default | Description |
|---|---|---|
| `--threads` | CPU count | Parallel derive workers |
| `--shard-size` | 10,000,000 | Genomic window (bp) of one derive unit |
| `--aggregate-parquet-size` | 268,435,456 | Target size (bytes) of the negative batch's data files |
| `--span-margin` | 100,000 | How far before a window the derive reads deletion rows whose split spans reach into it; a longer span is an error |

### 4. Iceberg maintenance

Apart from `mosaic clean`, no other command deletes an Iceberg file: every commit
leaves a manifest list (and usually a manifest) behind, and the staging that
`mosaic aggregate` prunes when a run completes is only recorded as deleted, so its
Parquet stays in S3. `mosaic maintain` bounds that growth. For each Iceberg table
it:

1. sets `write.metadata.delete-after-commit.enabled` if the table lacks it
   (`mosaic init` sets it on every table), so each commit deletes the
   `metadata.json` versions beyond the last 100;
2. rewrites the manifests: the current snapshot's data manifests are merged into
   files of up to `--manifest-target-size-bytes` (default 8 MiB);
3. expires the snapshots the retention policy drops and deletes every file only
   they needed, plus the files that a retained snapshot names only as deleted
   (the pruned staging). A snapshot is retained when it is the current one, one
   of the `--retain-last` newest (default 1), younger than `--older-than-hours`
   (default 120), or a branch head or tag;
4. with `--orphans`, deletes the objects under the upload prefixes
   (`s3://<MOSAIC_S3_BUCKET>/<table>/`) and the table locations that no snapshot
   of any table in any namespace of the catalog references, that are not
   pending in `ingest_files`, and that are older than `--orphan-grace-hours`
   (default 72). The sweep assumes those prefixes belong to this catalog.

`--older-than-hours 0` keeps only each table's current snapshot. That is safe
when no reader is running, and the Nextflow workflow passes it. The default keeps
five days of snapshots for a reader started by hand that planned its scans from an
older snapshot. `mosaic maintain` shares `mosaic register`'s advisory lock, and
refuses to run while an aggregation run is `in_progress` (`--force` overrides, for
a run that was killed). `--dry-run` commits nothing and deletes nothing. Data
files are not compacted.

```bash
mosaic maintain --dry-run                        # report what would be done
mosaic maintain --older-than-hours 0             # no reader is running
mosaic maintain --older-than-hours 0 --orphans   # also sweep orphan files
```

### 5. Create indexes

Run after aggregation. This creates the secondary indexes; the tables' primary-key and
unique constraints already exist from `mosaic init`. Creating the secondary indexes after
ingestion, rather than before, avoids per-row index maintenance during the bulk loads.

```bash
mosaic index
```

### 6. Joint calling (`joint_call.py`)

`joint_call.py` writes a multi-sample VCF of every active sample (or of a subset,
`--samples`) over every ingested contig from the PostgreSQL aggregates and the per-sample Iceberg tables. It is also the
regression oracle of the e2e tier; it is not part of the `mosaic` command. It refuses (exit 2) a cohort whose latest
aggregation run is not complete or does not cover every completed ingest batch, unless
`--allow-stale`. Its output has been compared with Sentieon GVCFtyper on one 2-sample chr21 test
only (chr21:1-30M; the figures are in the banding table above), and it does not handle
haploid calls (see the known limitation below).

`joint_call.py` is the driver of `mosaic-joint-call`, the Rust joint caller: it runs
the gate, exports the PostgreSQL side, plans the per-sample data files and runs the
binary (`$MOSAIC_JOINT_CALL_WORKER`, else `PATH`; missing: exit 1 with the remedy,
before any work), which writes the VCF.

```bash
python joint_call.py cohort.vcf.gz --threads 4
python joint_call.py slice.vcf.gz --region chr21:10000000-12000000  # a slice
python joint_call.py two.vcf.gz --samples HG00100,HG00245          # a sample subset
python joint_call.py own.vcf.gz --samples HG00100,HG00245 \
    --trim-alt-alleles --min-ac 1                                  # its own sites and alleles
```

| Option | Default | Description |
|---|---|---|
| `output` | stdout | The VCF; BGZF-compressed when the path ends in `.gz` |
| `--region` | the whole scope | `CONTIG[:START-END]`, 1-based, inclusive; the records equal the whole run's in the slice |
| `--allow-stale` | off | Joint-call a cohort the gate would refuse |
| `--threads` | the CPU count | Units run concurrently; the output is written in genomic order |
| `--window` | 10,000,000 | The unit width in bp, `0` = one unit per contig; the output does not depend on it |
| `--unit-buffer-bytes` | 67,108,864 (64 MiB) | A unit that is not the first unit not yet written keeps its compressed output in memory up to this many bytes, then spills it to a file of its own (`0` or `1`: every chunk spills); the output does not depend on it |
| `--spill-dir` | beside the output | Spill under this directory instead, in a directory of the driver's own naming (`<DIR>/mosaic-joint-call-spill-<token>`) |
| `--max-alt-alleles` | 6 | The ALT cap: a record with more ALT alleles keeps this many (below); `0` = no cap |
| `--samples` | every active sample | `NAME[,NAME...]`: joint-call only these active samples (below) |
| `--samples-file` | — | As `--samples`, one name per line (blank lines ignored); not with `--samples` |
| `--trim-alt-alleles` | off | An ALT allele no written genotype names (AC 0) leaves the record (below) |
| `--min-ac` | 0 (off) | `K`: write only the records whose written genotypes hold at least K ALT observations (below) |
| `--cohort-state` | — | Print the cohort state as one JSON line and exit; no VCF (below) |
| `--export-cohort` | — | `DIR`: write `DIR/cohort.json`; no VCF (below) |
| `--export-contig` | — | `DIR` with `--region CONTIG`: write that contig's exports into `DIR`; no VCF (below) |
| `--from-export` | — | `DIR`: joint-call from the export, with no PostgreSQL connection (below) |

The joint caller reads each data file once per unit, by position, as a k-way merge
across the files, and its units run on `--threads` threads, each compressing its own
output in 4 MiB text chunks. The **head** unit (the first unit not yet written) writes
its chunks to the output as it goes; every other unit keeps them in memory up to
`--unit-buffer-bytes` and then moves them to a spill file of its own and appends there;
when the head ends, the units after it that already finished are copied to the output in
order and the next unit still running becomes the head. A unit starts only while it is
fewer than `2 × threads` units past the head, so at most `2 × threads − 1` units hold a
buffer (each at most the bound plus one chunk) or a spill file, and at one thread nothing
is buffered or spilled. Spill files go in `<output>.spill-<token>` beside a regular output
path (the filesystem sized for the output; at most the in-flight units' compressed bytes),
in the driver's temporary directory for a `/dev/...` output (stdout included), or under
`--spill-dir`; the directory is created at the first spill and removed when the run ends,
whether it succeeds or fails. The compressed output is the same bytes at every
`--threads` and every bound. It writes **no tabix index** beside a `.gz` output; index it
with `tabix -p vcf` or `bcftools index -t`. On the six chr21 test samples the whole run
takes about 5 s at one thread (2.7 s at four).

**The ALT cap (`--max-alt-alleles N`, default 6, `0` = none).** A record with more than N
ALT alleles (`*` counts as one) is written as a projection of the uncapped record: it
keeps the N ALTs with the largest AC (the counts of its genotypes; ties to the earlier
ALT), in their original order; a genotype that carries a dropped ALT is written `./.`;
AD, F1R2 and F2R1 keep the entries of REF and the kept ALTs; PL keeps the entries of the
genotypes over those alleles and, when the uncapped PL held a 0 and all the kept entries
are above 0, has their minimum subtracted (a ref-block fill, whose PL may hold no 0, is
never shifted); DP is unchanged; AC, AN and AF are recomputed from the written genotypes, and
QUAL and the other INFO fields (DP, QD, MQ, FS, SOR, InbreedingCoeff, the site's) are
unchanged. A record with at most N ALTs is the uncapped record exactly. A genotype the cap
writes `./.` keeps its AD, DP and PL over the kept alleles, and that PL is normalized over
the kept genotypes like any call's, so it no longer points at the dropped call (a het
carrier of a dropped allele usually has its 0 at the hom-ref genotype). Under a cap the
header holds `##mosaic_max_alt_alleles=N` (before the `##contig` lines), the catalog
cross-check leaves the capped records out (it says how many), and a closing line reports
the records capped, the genotypes written `./.`, the ALT observations on the dropped
alleles and the kept-allele observations lost with those genotypes. On the six chr21 test
samples the default caps 311 of 132,382 records and writes 60 genotypes `./.`;
`--max-alt-alleles 0` gives the uncapped output.

**A sample subset (`--samples A,B` or `--samples-file FILE`).** The run joint-calls only
the named active samples, and each record is the whole cohort's record projected onto
their columns: the columns are the subset's in the cohort header's order (whatever order
is given); every cohort site is still a record, and QUAL and the site's INFO fields (DP,
QD, MQ, FS, SOR, InbreedingCoeff) stay the cohort's; `*` leaves the ALT list when no
sample of the subset carries it (AD, F1R2 and F2R1 then lose its entry, PL its genotypes
under the rule above; a record whose only ALT was `*` gets ALT `.` and no AC or AF); a FORMAT key no
sample of the subset carries is dropped; AC, AN and AF are the subset's (`bcftools view
-s` of the whole cohort's output gives the same genotypes, AC and AN). Under the ALT cap a
subset record is capped like any record, ranked by the subset's own AC. The header is the
cohort's but for the `#CHROM` line. Only the per-sample data files whose `sample_id`
bounds can hold a subset sample are read (with the whole-sample layout, a subset of one
batch reads that batch's files only), while the PostgreSQL export and the per-site work
still cover every cohort site. The catalog cross-check is skipped (its counts are the
whole cohort's) and a closing line says so. An unknown, removed or repeated name, or an
empty list, exits 2 naming them, before the gate. Naming every active sample gives the
whole cohort's text.

**Trimming ALTs and a minimum AC (`--trim-alt-alleles`, `--min-ac K`; both off by
default; the names are `bcftools view`'s).** The options apply in a fixed order, each to
the record the previous one wrote: the subset (`--samples`), then the trim, then the ALT
cap, then `--min-ac`. **`--trim-alt-alleles`** drops every ALT allele that no written
genotype names (AC 0): the other ALTs keep their order, the GTs are re-indexed (no
genotype changes: none names a dropped allele), AD, F1R2 and F2R1 keep the entries of REF
and the kept ALTs, PL the entries of the genotypes over them under the rule above, AC and
AF lose the dropped entries, and AN, QUAL and the other INFO fields are unchanged. A record
left without ALTs is written with ALT `.` and no AC or AF key, as bcftools writes such a
record (unless `--min-ac` leaves it
out); a ref-block fill there keeps its 0/0 PL, shifted to 0 only when the fill held a 0 at
another genotype. The cap then ranks the trimmed record, so it no longer keeps an ALT of AC
0 to fill its N. **`--min-ac K`** writes a record only when the ALT observations of its
written genotypes (the sum of its INFO/AC, after the subset, the trim and the cap) are at
least K, so every record of the output satisfies it on its own INFO/AC; `--min-ac 1` with
`--samples` keeps the sites variable in the subset. A record the cap leaves without its
only carrier (written `./.`) is left out at `--min-ac 1`. The records left out are skipped
before their calls are built, but their sites are still read (the run still walks every
cohort site). Each option set adds a header line before the `##contig` lines, in the order
the options apply: `##mosaic_trim_alt_alleles=true`, `##mosaic_max_alt_alleles=N`,
`##mosaic_min_ac=K`; and a closing line after the cross-check's, in the same order:
`ALT trim (--trim-alt-alleles): R records trimmed, A ALT alleles with AC 0 dropped`, the
cap's line, `Minimum AC (--min-ac K): L records left out (fewer ALT observations in their
written genotypes)`. The cap's counts are over the trimmed record when both are set, and
the trim's and the cap's counts include the records `--min-ac` leaves out. The catalog
cross-check compares a trimmed record on its untrimmed alleles, so the trim alone does not
change its lines; it does not count a record `--min-ac` leaves out. At
`--max-alt-alleles 0`, `bcftools view [-s S] [--min-ac 1] --trim-alt-alleles` of the
whole cohort's output gives the same sites, ALTs, genotypes, AC and AN as `[--min-ac 1]
--trim-alt-alleles [--samples S]` (bcftools cuts PL without the shift), and bcftools
reads the output, its records without ALTs included, with no warning. On the six chr21 test
samples (132,382 records) the trim changes 4,469 records and drops 6,799 ALTs, and under
the default cap 47 records are still capped (311 without the trim); `--min-ac 1` leaves
out 3 records of the whole cohort and 45,479 for a two-sample subset; the time is
unchanged (2.2–2.3 s at four threads). Against GVCFtyper on two samples the native trim in
place of `bcftools view --trim-alt-alleles` gives the same concordance as the bcftools
trim (`+/- 128 = 56644`, the same with both trims; with no trim at all, `+/- 1614`).

**A cohort export (`--cohort-state`, `--export-cohort DIR`, `--export-contig DIR --region
CONTIG`, `--from-export DIR`).** A joint call reads PostgreSQL on every run; an export
reads it once and then serves any number of calls that open no PostgreSQL connection.
`--cohort-state` runs the driver's
checks (the gate, the `--samples` names) and prints the cohort state as one JSON line on
stdout: the aggregation run, its status and time, the completed batches, the sample counts
and a digest of the samples, and the stale reason; the line is the same until the cohort
changes. `--export-cohort DIR` writes `DIR/cohort.json` (the state, the contigs, the active
and removed samples; a stale cohort only with `--allow-stale`, which records the reason).
`--export-contig DIR --region CONTIG` writes that contig's sites and alleles exports over the
whole contig (`sites.c<index>.parquet`, `alleles.c<index>.parquet`, then
`export.c<index>.json`, its completion marker), after checking in the same transaction that
the database's cohort state is `cohort.json`'s (another: exit 2, make a new export).
`--from-export DIR` is a joint call with every option above but `--allow-stale`: the contigs,
the samples and the exports come from the directory, which it does not change; it refuses
(exit 2, before any store is read) a directory of another format, a contig in scope with
sites whose files are missing or of another cohort state, and a bad sample name. The
per-sample Iceberg tables are still read from the lake. The output equals the PostgreSQL
path's byte for byte, closing lines included, for every option set.

```bash
python joint_call.py --cohort-state                    # one JSON line on stdout
python joint_call.py --export-cohort /scratch/exp
python joint_call.py --export-contig /scratch/exp --region chr21
python joint_call.py --from-export /scratch/exp two.chr21.vcf.gz \
    --region chr21 --samples-file two.names            # no PostgreSQL connection
```

**Memory and disk at cohort scale.** `--unit-buffer-bytes` bounds the output a run holds,
and the ALT cap bounds a site's calls and a record's text: every call is built over the
kept alleles only, so a capped record's PL has at most (N + 1)(N + 2) / 2 values per
sample. Measured on AWS (S3 and RDS) at 3,202 samples on chr21 (the 1000 Genomes DRAGEN
gVCFs): the defaults fit a 14 GiB memory limit, so no `--threads` or `--window` setting is
needed for memory at that scale. A larger chromosome needs fewer threads: the whole
cohort of chr1 × 3,202 does not fit 12 GiB at 4 threads and fits at 2 (below). Memory
follows the samples in a call, so a smaller sample set fits easily: a set of 200 samples
on chr1 × 3,202 takes about 5.5 min at 1.7 GB, a set of 800 25 min at 3.9 GB.

| Call (the default cap and window) | `--threads` | Wall | Peak RSS | Output (`.gz`) |
|---|---|---|---|---|
| chr21 × 3,202 samples, 1,662,044 records | 4 | 30 min 19 s | 6.4 GB | 28.8 GB |
| the same | 1 | 78 min 57 s | 2.4 GB | the same bytes |
| the genome × 100 samples, 18,071,827 records | 4 | 29 min 27 s | 4.0 GB | 13.0 GB |
| chr1 × 3,202 samples (banded at `20`), 9,657,568 records | 4 | out of memory in 12 GiB | — | — |
| the same | 2 | 3.6 h | 8.2 GB | 173 GB |

Memory grows with the units in flight: at 3,202 samples about 1.3 GB per unit over about
1 GB held once (the reader keeps a row group and a decoded batch per data file, 386 files
there). The uncapped output (`--max-alt-alleles 0`) needs far more: at 3,202 samples it
completed at one thread with a peak of 13.8 GB (12.8 GiB of the 14), reached at a 605-ALT
site whose record is 2.13 GB of text (`bcftools +fill-tags` stops at such a record).

**The spill copy (expected).** With more than one thread, a unit whose compressed output
exceeds `--unit-buffer-bytes` has usually written most of it to its spill file before the
units ahead of it are done; when its turn comes, one thread copies the spill file into the
output. About (threads − 1) / threads of a large output is therefore written twice. At
3,202 samples and 4 threads, 21.2 GB of the 28.8 GB output was spilled, the spill directory
reached 25.8 GB, and the copies took 292 s of the 30 min. This is the cost of an output
whose bytes do not depend on `--threads`. Leave free space about the size of the output
beside it, or point `--spill-dir` at a filesystem that has it. Nothing is spilled at one
thread, or while every unit's output stays under the bound (the genome call of 100 samples
spilled nothing).

**Known limitation: haploid calls (chrX outside the PARs, chrY).** The joint caller writes
diploid genotypes only and does not yet handle a gVCF's haploid genotypes, which callers
commonly write for a male sample on chrX outside the pseudoautosomal regions and on chrY:

- a haploid ALT call is written `./.`. Its AD is kept, and its PL, which has one value per
  allele, is spread over the diploid genotype slots with `990` fills;
- a haploid reference block is filled `0/0`.

So on those regions a record's AC and AF miss the haploid ALT calls, and its AN counts each
haploid reference call as two alleles. The catalog cross-check reports such a record as
`other`: in the genome-wide call of 100 samples above, 399,172 of chrX's 613,607 records
and 17,434 of chrY's 30,192. Do not rely on the joint-called genotypes, AC, AN or AF of
chrX outside the PARs or of chrY for a cohort with haploid calls. The autosomes and the
PARs are not affected, and a diploid sample's own genotypes on chrX are written as its gVCF
carries them. The ingested data is not affected either: the PostgreSQL `alleles` counts
include the haploid calls, a haploid call's `sample_site_info` row has `gt_1` set and
`gt_2` NULL, and the genotype matrix (`batch_genotypes`) and `sample_qc.n_haploid` keep the
ploidy. `sample_ref_blocks` stores no ploidy for a reference block, which a fix has to
address.

### Resetting

`mosaic clean` drops MOSAIC's PostgreSQL tables and resets the Iceberg tables.
In local mode (`MOSAIC_ICEBERG_TYPE=rest`) it restarts the Docker stack; in
cloud mode it drops the Iceberg tables and deletes the S3 data files: the
upload prefixes and each dropped table's own location in the warehouse. Because
the cloud reset is destructive to shared infrastructure (RDS + Glue + S3), it
requires `--yes`:

```bash
mosaic clean            # local Docker stack
mosaic clean --yes      # cloud (Glue + S3 + RDS) — confirmation required
```

Only MOSAIC's own tables are dropped, so it is safe to point at a shared
database; a partial reset exits non-zero rather than reporting success.

## Running via Nextflow

The Nextflow project lives in `workflow/` (`main.nf`, `nextflow.config`, the
`conf/` profiles, and the `bin/` helper). `workflow/main.nf` orchestrates the
full **ingest → register → aggregate → {index, maintain}** flow as a
pipeline: it fans a sample list out into parallel `INGEST_BATCH` tasks (each a
`mosaic ingest --defer-register` by default), then — once every ingest finishes —
runs a single `mosaic register` and `mosaic aggregate`, and after it `mosaic index`
and `mosaic maintain --older-than-hours 0` (independent of each other).
`mosaic init` remains a manual one-time step (it needs a reference gVCF header and
the desired chromosome scope). Two profiles are provided: `local` (host machine
against the `iceberg_test/` Docker stack) and `awsbatch` (AWS Batch jobs). A second
workflow, `workflow/joint_call.nf`, joint-calls the ingested cohort after that
(below).

```bash
# One-time schema setup (manual), then run the pipeline
mosaic init <reference-sample.gvcf.gz> --chromosomes chr21
nextflow run workflow/main.nf -profile local --samples samples.txt
```

| Parameter | Default | Purpose |
|---|---|---|
| `--samples` | _(required)_ | Newline-delimited sample-name file (an `s3://` URL on awsbatch) |
| `--batch_size` | 200 | Samples per `INGEST_BATCH` task |
| `--region` | _(unset)_ | Comma-separated chromosomes to fan out over, e.g. `chr1,chr2` |
| `--defer_register` | `true` | `INGEST_BATCH` runs `mosaic ingest --defer-register`, and `REGISTER` registers every batch's files. `false`: each task registers its own files and `REGISTER` reports nothing to register |
| `--register_files_per_commit` | 500 | `mosaic register --files-per-commit` (min 1) |
| `--fan_in` | 100 | Batches merged per tier in `mosaic aggregate` (min 2); `>= #batches` is a single pass, smaller forces more tiers |
| `--shard_size` | 500,000 | Passed through to `mosaic ingest` |
| `--pos_batch_size` | 100,000 | Passed through to `mosaic ingest` (the value the 3,202-sample run completed with) |
| `--ref_gq_bands` | _(unset: off)_ | `mosaic ingest --ref-gq-bands`, passed through when set (ref-block banding, e.g. `20,60` or `all`; see above). Keep it unchanged between a failed run and its `-resume` |
| `--ref_shard_size` | 20,000,000 | `mosaic aggregate --ref-shard-size` (bp): the `ref_blocks` window; one unit per (chromosome, window) holds ~17 bytes per window base (~340 MB per thread at the default). `0` = one window per chromosome (~4 GB on chr1), discouraged. Output is identical regardless |
| `--aggregate_threads` | `null` (task cpus) | `mosaic aggregate --threads` (reduce concurrency). On 3,202 samples genome-wide 4 threads run out of memory in a 12 GiB task and 2 complete (64 min, 12.84 GB peak): set 2, or 1 on a larger cohort. Only the main process commits to Iceberg (once per barrier and table), so threads do not contend on commits |
| `--aggregate_cpus` / `--aggregate_memory` | 4 / `12.GB` | `AGGREGATE` task resources (awsbatch `withName`). Raise memory for a very large cohort **only** alongside a compute environment with matching instances |
| `--maintain` | `true` | Run the `MAINTAIN` step (`mosaic maintain`) after `AGGREGATE`; `false` skips it |
| `--maintain_older_than_hours` | 0 | `mosaic maintain --older-than-hours`. 0 keeps only each table's current snapshot, so the staging the aggregate just pruned is reclaimed in the same run; safe because the pipeline runs no reader meanwhile |
| `--maintain_orphans` | `false` | Also pass `--orphans` (sweep the upload prefixes and table locations for unreferenced objects) |
| `--maintain_orphan_grace_hours` | 72 | `mosaic maintain --orphan-grace-hours`, with `--maintain_orphans` |

Each `mosaic ingest` invocation is one batch, so the total number of batches is
`#sample-batches × #regions`. When that exceeds `--fan_in`, `aggregate` performs
a **multi-tier** merge.

**`AGGREGATE` memory and storage.** Memory is bounded in code, but not by a constant:
- **finalize:** streams the whole cross-sample result to PG + `agg_*` in row chunks;
- **reduce:** each worker's accumulator is bounded by output keys (variant sites or alleles on
  one chromosome), not by input rows.

Per-worker memory therefore still grows with the number of distinct sites per chromosome,
and peak memory is roughly `--aggregate_threads ×` that.

**Settings for the full 1000 Genomes cohort.** Measured on AWS in a 4-cpu / 12 GiB task:

| Cohort | `--aggregate_threads` | Wall | Peak RSS |
|---|---|---|---|
| chr21 × 3,202 | 4 | 80 s | 4.0 GB |
| genome × 100 | 4 | 6 min | 5.6 GB |
| chr1 × 3,202 (banded at `20`) | 4 | 6.4 min | 10.0 GB |
| genome × 3,202 (banded at `20`) | 4 | out of memory within about a minute | — |
| genome × 3,202 (banded at `20`) | 2 | 64 min | 12.84 GB, about 45 MB under the limit |

So at 3,202 samples genome-wide set `--aggregate_threads 2`; beyond that, 1 thread
(unmeasured) or larger instances with `--aggregate_memory`. An aggregate killed for memory
leaves the store consistent: rerun with `-resume` and fewer threads.

The finalize builds each chromosome's new partitions before dropping the old ones, so it
also needs transient PostgreSQL storage: genome-wide at 3,202 samples the database is
69.8 GB (65 GiB) and the run fit a 100 GiB RDS volume. Plan the RDS size ahead, or enable storage
autoscaling.

**`--ref_shard_size`.** Keep the 20 Mbp default. `0` (one window per chromosome) needs
~4 GB of arrays per thread on chr1; much smaller windows multiply the Iceberg staging
commits, which contend on one table and cause commit-conflict retries at genome scale.

### Joint calling (`workflow/joint_call.nf`)

A second workflow beside `main.nf`, sharing `nextflow.config`, the two profiles and
`workflow/bin/`, joint-calls an ingested, aggregated cohort as **one VCF per sample set
and chromosome** with `joint_call.py` (section 6), after the ingestion run has finished.
PostgreSQL is read once per run, by
its first three steps; the call tasks open no PostgreSQL connection:

1. `COHORT_STATE`: `joint_call.py --cohort-state` over the whole sample list, on every
   run. A stale cohort or a bad sample name stops the run here (exit 2).
2. `EXPORT_COHORT`: `--export-cohort` (`cohort.json`: the generation, the contigs, the
   samples).
3. `EXPORT_CONTIG`, one per chromosome with sites: `--export-contig`, at most
   `--joint_call_export_forks` at a time.
4. `JOINT_CALL`, one per set and chromosome: `--from-export ... --region <chromosome>
   --samples-file <set>` to the task's disk, `bcftools index -t` and a check of the
   index's record count against the driver's, then the outputs are uploaded.
5. `JOINT_CALL_SUMMARY`: `joint_call_summary.tsv`.

Each task runs `workflow/bin/joint_call_task.py`, which finds the driver as
`$MOSAIC_JOINT_CALL_DRIVER`, else `joint_call.py` on `PATH` (the image carries it in
`/opt/mosaic/bin`), else the repository root (a local run from a checkout); the driver
finds `mosaic-joint-call` as section 6 says.

```bash
nextflow run workflow/joint_call.nf -profile local \
    --samples samples.txt --batch_size 200 --region chr21,chr22
nextflow run workflow/joint_call.nf -profile awsbatch \
    --samples s3://.../samples.txt --batch_size 200
```

**The sets.** `--samples` is read as `main.nf` reads it (blank lines and lines starting
with `#` skipped) and cut into consecutive sets of `--batch_size`: `set_0001`, `set_0002`,
…, the last one holding the remainder. `--batch_size` is this run's own value: a set may
hold part of an ingest batch, several of them or the whole list (the joint caller plans a
set's data files wherever its samples were ingested). Without `--samples` there is one
set, `cohort`, of every active sample, called without a names file (so the catalog
cross-check runs). **The chromosomes** are those of the cohort that have sites; `--region
a,b,c` (whole chromosomes only) narrows them, and an entry that is not a contig of the
cohort stops the run before any contig export. A chromosome without sites gets no export
and no task, and the log says so.

**The outputs**, all written by the helper (to a local directory or `s3://`):

```
<joint_call_out>/gen<agg_run_id>/
    export/cohort.json, sites.c<index>.parquet, alleles.c<index>.parquet, export.c<index>.json
    set_0001/set_0001.samples.txt
    set_0001/set_0001.chr21.vcf.gz
    set_0001/set_0001.chr21.vcf.gz.tbi
    set_0001/set_0001.chr21.summary.json
    ...
    joint_call_summary.tsv
```

A file is named by the chromosome when its name is plain (letters, digits, `.`, `_`,
`-`), else by `c<index>`. A task's `summary.json` is written last, after the VCF, its
index and the names file, and is its completion marker: a VCF without one is not a
finished output (a failed task leaves none). It holds the set, the chromosome, the sample
count, the options, the generation and cohort state, the record count, the VCF's bytes,
the driver's `mosaic-joint-call:` line and closing lines, the seconds per step (driver,
index, publish) and the published paths. `joint_call_summary.tsv` has one row per set and
chromosome (set, chromosome, contig index, samples, records, VCF bytes, the seconds per
step, the VCF's path), written when every task is done. `export/` is a copy of the export
the tasks used. The task logs go to `--log_dir` as for `main.nf`.

| Parameter | Default | Purpose |
|---|---|---|
| `--samples` | _(unset: the whole cohort)_ | The sample list; cut into sets of `--batch_size` |
| `--batch_size` | 200 | Samples per set (min 1) |
| `--region` | _(unset: every chromosome with sites)_ | Comma-separated chromosome names (no ranges) |
| `--joint_call_out` | `./joint_calls` (local), `s3://<s3_bucket>/joint-calls/` (awsbatch) | The output prefix; a local path is made absolute |
| `--joint_call_max_alt_alleles` | _(unset: the driver's 6)_ | `--max-alt-alleles`; `0` = no cap |
| `--joint_call_trim_alt_alleles` | `false` | `--trim-alt-alleles` |
| `--joint_call_min_ac` | _(unset: off)_ | `--min-ac` |
| `--joint_call_window` | _(unset: 10 Mbp)_ | `--window` |
| `--joint_call_threads` | _(unset: the task's cpus)_ | `--threads` |
| `--joint_call_index` | `true` | Write each VCF's `.tbi` and check its record count |
| `--joint_call_export_forks` | 2 | Chromosome exports in flight: the only load on PostgreSQL |
| `--joint_call_max_forks` | _(unset: the profile's limit, 30 on awsbatch)_ | `JOINT_CALL` tasks in flight |
| `--joint_call_cpus` / `--joint_call_memory` | 4 / `12.GB` | A `JOINT_CALL` task's resources |

Give these on the command line (`--name value`). In a `-c` file a `params` block is read
after the configuration has computed what it derives from parameters, so there it does not
change the awsbatch default of `joint_call_out` (which keeps the profile's placeholder
bucket: set `joint_call_out` itself) nor a call task's cpus, memory and forks (set the
process settings instead, `process { withLabel: jc_call { memory = 24.GB } }`); the
parameters the workflow script reads (the sets, the chromosomes, the driver's options,
`joint_call_threads`) work either way. The `awsbatch` profile has run from chr21 × 100
to genome × 3,202: as 17 sets of 200 samples, genome × 3,202 is 425 tasks in 54 min of
wall (81 CPU-hours, at most 6.2 min and 1.9 GB per task, 1.11 TB of VCF), and the 25
contig exports kept RDS at ≤ 47% CPU. Every VCF holds its contig's record count, and the
sampled ones equal a direct `joint_call.py` call byte for byte. A whole-cohort set of
chr1 × 3,202 needs `--joint_call_threads 2` in a 12 GiB task.

The workflow refuses, before any task, a `--batch_size` below 1, a negative cap or
`--min-ac`, an empty sample list and a `--region` entry that is empty or a range. An exit
status 2 of the driver (a refusal: a stale cohort, a bad name, a changed cohort under
`--export-contig`) ends the run after one submission of that task; any other failure
retries as the other processes do (the default `errorStrategy` retries up to three times
whatever `maxRetries` says: to bound retries in an override file, set `errorStrategy` as
well as `maxRetries`). The local profile runs one task of each process at a
time. A set's memory grows with the data files it reads (samples × chromosome size,
divided by ref-block banding): for a large set lower `--joint_call_threads` or raise
`--joint_call_memory` together with a compute environment that has such instances.

**`-resume`.** `COHORT_STATE` runs every time, and its one-line state is the value that
keys every task after it: on an unchanged cohort `-resume` runs `COHORT_STATE` alone and
reuses the exports and every finished call; a different `--batch_size` (or sample list)
reuses the exports and calls the new sets; after a new aggregate everything runs again,
into the new generation's folder. The cache does not look at the outputs: a file deleted by
hand is not rebuilt by `-resume`, and a cached task does not publish again. A different
option set or `--batch_size` on the same generation writes over the same set folders (and a
set of an earlier `--batch_size` that the new one does not produce stays there): give each
its own `--joint_call_out`.

**Do not run `mosaic remove`, an ingest or an aggregate while a joint-call run is in
flight.** The cohort is checked at the start of a run only: the call tasks read the export
and the lake, and a change under them (a removal deletes a sample's per-sample rows while
the export still lists the sample) is not detected, so the tasks before and after it would
not call one cohort and nothing would report it.

The workflow writes the joint caller's output as section 6 describes it, its known
limitation on haploid calls included.

### Full local multi-tier test

Downloads a few 1000 Genomes DRAGEN gVCFs from the public S3 bucket
(`workflow/bin/fetch_and_ingest.py`) but runs ingest/register/aggregate/index/maintain
against the local Docker stack. Four single-sample batches with `--fan_in 2` force a
2-tier merge; the run takes a few minutes. `mosaic clean` in local mode restarts the
Docker stack and wipes everything on it, so run this only against a stack whose data you
do not need. `mosaic init` needs the header of any gVCF of the same reference (GRCh38,
with `chr`-prefixed contigs), for example one of the four samples' gVCFs:

```bash
cd iceberg_test && docker-compose up -d && cd ..
mosaic clean
mosaic init <one-1000G-DRAGEN-sample.gvcf.gz> --chromosomes chr21
printf 'HG00096\nHG00097\nHG00099\nHG00100\n' > samples.txt
nextflow run workflow/main.nf -profile local \
    --samples samples.txt --region chr21 --batch_size 1 --fan_in 2
```

## Storage Schema

### PostgreSQL tables

| Table | Purpose |
|---|---|
| `contigs` | Chromosome names and lengths |
| `alleles` | The allele catalog, built by `mosaic aggregate`: every allele any batch listed at a site, its summed count, and `del_range` for deletions (derived from the allele string) |
| `site_info` | Aggregated per-site statistics (QUAL, DP, MQSQ, SB, GTC) |
| `ref_blocks` | Aggregated reference-block DP and sample counts |
| `samples` | Sample names, IDs and `status`: `active`, or `removed` (the tombstone `mosaic remove` sets first). A removed sample is left out of the per-sample reads of `joint_call.py`, `scripts/gwas_benchmark.py` and the genotype-matrix reader, and `mosaic ingest` refuses it; the aggregates count it until the `mosaic aggregate` that applies its removal |
| `ingestion_progress` / `shard_completion` | Ingest checkpoint/resume tracking (batch status `in_progress`, `failed`, `uploaded` or `complete`; `kind` is `ingest`, or `removal` for the negative batch of a `mosaic remove`; `ref_gq_bands` is the batch's `--ref-gq-bands` spec, NULL without banding) |
| `removal_progress` | One row per `mosaic remove`: the samples, the negative batch's `batch_id`, `status` (`in_progress`, `registered`, `complete`, `failed`) |
| `sample_qc` / `sample_qc_summary` | Per-(batch, sample, chromosome) QC counters; the summary view lists active samples only |
| `ingest_files` | The data files of `--defer-register` batches, pending until `mosaic register` registers them |
| `aggregation_progress` / `tier_completion` | Aggregation checkpoint/resume tracking |

### Iceberg tables (Parquet in S3)

**`sample_ref_blocks`** — per-sample reference blocks:
`sample_id`, `chromosome_index`, `position_start`, `position_end`,
`ad_1`, `ad_2`, `dp`, `pl_1`, `pl_2`, `pl_3`, `sample_counts` (1 for a
hom-ref block, 0 for a `./.` block). With `--ref-gq-bands` a row is a merged run of
adjacent blocks (see "Ref-block banding" above)

**`sample_site_info`** — per-sample variant genotypes:
`sample_id`, `chromosome_index`, `position`, `del_start`, `del_end`,
`alleles` (the record's own allele strings: its reference base, then its ALT
alleles in the record's order), `gt_1`, `gt_2` (local indices into `alleles`),
`dp`, `ad`, `f1r2`, `f2r1`, `sb`, `pl` (lists, as the record carries them), and the
record's own contribution to `site_info`: `qual`, `site_dp` (INFO/DP, else
FORMAT/DP; `dp` is FORMAT/DP), `mqsq_1`, `mqsq_2`, `gtc` (a list of three
doubles). On a deletion carrier's row, `del_ref_spans` lists the (start, end)
spans of the sample's own reference blocks that the deletion overlaps (see
below; `joint_call.py` does not read it).

**Layout of the two per-sample tables.** Both are partitioned by `chromosome_index`.
Each data file holds whole samples of one ingest batch and one chromosome, packed in
`sample_id` order up to `--aggregate-parquet-size`; a file never splits a sample. Inside a
file, rows are sorted by position, then `sample_id` (the sort order declared on both
tables), in row groups of `MOSAIC_PARQUET_ROW_GROUP_ROWS` / `MOSAIC_PARQUET_SITE_ROW_GROUP_ROWS`
rows. So:
- deleting one sample's rows (a `sample_id` filter) rewrites only the files that hold it,
  one or a few per chromosome per batch, instead of every file of its batch; a file that
  holds only that sample is dropped without a rewrite. pyiceberg 0.11 cannot write
  merge-on-read deletes (it falls back to copy-on-write with a warning), so this packing is
  what makes such a delete cheap;
- a scan for a subset of samples skips the files of the other samples;
- a scan of a position window skips the row groups outside the window, but plans every
  file of the chromosome, since each file spans it.

`mosaic remove` deletes a removed sample's rows this way. The files such a delete
rewrites are written by pyiceberg, in row groups of the table's
`write.parquet.row-group-limit`, which `mosaic init` sets to the same caps, so a
position-window scan reads a rewritten file as cheaply as an ingested one.

Ingestion also writes append-only per-batch partials (`batch_ref_blocks`,
`batch_site_info`, `batch_allele_counts`, with `gtc` summed in fixed point
(2^-32 units, int64) and the record and listing counts `n_records` /
`n_listed`); `mosaic remove` writes the
same tables with negative values. `mosaic aggregate` reduces them
through staging tables (`tier_*`) into the materialized aggregates (`agg_*`) and
finalizes to PostgreSQL. Every table is created with
`write.metadata.delete-after-commit.enabled`; `mosaic maintain` expires old
snapshots and deletes their files.

**The partials are a sum of the per-sample rows.** Each batch's partial rows
are exactly the sum, over the batch's samples, of what their per-sample rows
store, and the test suite checks this by recomputing every partial row from the
two per-sample tables:
- `batch_site_info` is the sum of the contribution columns (each row's `gtc`
  quantized to 2^-32 units first); `n_records` is the number of rows at the
  site.
- `batch_allele_counts` is the count of the `gt_1` / `gt_2` hits; `n_listed` is
  the number of rows whose own allele list holds the allele.
- `batch_ref_blocks`, per position, is the sum of the `sample_ref_blocks` rows'
  `dp` and `sample_counts`, plus each `del_ref_spans` span with its row's
  `site_dp` and a sample count of 0. A reference block that a sample's own
  deletion overlaps counts the deletion's depth there.

So a sample's contribution to every aggregate can be re-derived from its
Iceberg rows without its gVCF, which is what `mosaic remove` does. The finalize
of `mosaic aggregate` drops a site whose net `n_records` is 0 and an allele whose
net `n_listed` is 0, and writes the integer part of the fixed-point `gtc` sum to
PostgreSQL. All of these are integer sums, so the result is exact whatever the
order of summation (a removal, an incremental run or `--rebuild`).

### Allele encoding

Alleles are stored using a prefix notation:

| Type | Format | Example |
|---|---|---|
| Reference | first base | `A` |
| SNP/MNP | `x` + alt | `xT` |
| Insertion | `+` + inserted bases | `+AT` |
| Deletion | `-` + deleted bases | `-ATG` |
| Non-ref symbolic | `*` | `*` |

## Testing

Two test layers run from a clone with no extra data: the Python unit tests and the Rust
workspace's tests.

```bash
./run_tests.sh --fast          # Python unit tests (no Docker, no test data)
cd rust && cargo test          # Rust unit tests (plus: cargo fmt --check, cargo clippy --all-targets -- -D warnings)
```

`run_tests.sh` runs pytest under `python3`; set `MOSAIC_TEST_PYTHON` to another
interpreter if the dependencies are installed elsewhere. Unit tests that run the Rust
binaries use the release builds under `rust/target/release/` when they exist and skip
otherwise.

The other tiers are for maintainers:

```bash
./run_tests.sh --integration   # integration tests (needs the Docker stack)
./run_tests.sh --e2e           # full pipeline vs. a recorded vcfdiff baseline (Docker, slow)
./run_tests.sh --all           # unit -> restart -> other integration -> e2e
```

They need the Docker stack in `iceberg_test/` (`cd iceberg_test && docker-compose up -d`)
and chr21 gVCF test data under `test_data/` (1000 Genomes DRAGEN gVCFs and slices derived
from them), and the e2e tier also a `vcfdiff.py` comparison script at the repository root.
Neither the test data nor that script is distributed with the repository, so tests that
need them do not run from a clone. The tests use their own PostgreSQL database
(`jc_test`), Iceberg namespace (`jc_test`) and S3 bucket (`jc-test-warehouse`), so they do
not touch pipeline data on the same stack.

The ingest worker's and the joint caller's outputs on the test data are checked against
frozen digests: `tests/golden/ingest_signatures.json` (per case a SHA-256 over the
worker's rows, floats as bits) and `tests/golden/joint_call_digests.json` (per scenario
the SHA-256, record and byte counts of the decompressed VCF and its closing lines). A
digest is re-recorded only when a change is meant to alter the output:
`MOSAIC_INGEST_GOLDEN_RECORD=1` with `MOSAIC_INGEST_GOLDEN_REASON="<why>"` (on
`tests/integration/test_rust_engine.py`), or `MOSAIC_JOINT_CALL_GOLDEN_RECORD=1` with
`MOSAIC_JOINT_CALL_GOLDEN_REASON="<why>"` (on `tests/integration/test_rust_joint_call.py`).

## License

Licensed under the BSD 2-Clause License — see [LICENSE](LICENSE).
