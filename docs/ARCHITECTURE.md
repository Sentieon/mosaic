# MOSAIC architecture

This is the design document of MOSAIC: how it is built, what each stage guarantees, and what
is not built. Setup, configuration and the command reference are in [`../README.md`](../README.md);
measurements are in [`benchmarks.md`](benchmarks.md).

## 1. Overview

MOSAIC ingests per-sample gVCFs into two stores and keeps cross-sample aggregates over
them. Its readers are the joint caller (`joint_call.py` and the Rust binary
`mosaic-joint-call`), `scripts/gwas_benchmark.py`, the genotype-matrix reader in
`mosaic/genotypes.py`, and ad-hoc SQL.

The two stores are split by access pattern:

- PostgreSQL is the variant catalog: contigs, samples, the allele catalog and per-site
  population statistics. It is indexed and small, and answers common queries such as global
  allele frequency or site QC alone.
- Apache Iceberg tables of Parquet files on S3 form the lake: the per-sample evidence, the
  per-batch partial sums and the aggregate's staging. It holds high-cardinality, append-only
  data, read with predicate pushdown, for per-sample or per-position detail.

```
  gVCFs ─▶ mosaic ingest ─▶ ┌ Iceberg lake (Parquet on S3) ──────────────────────────────┐
                            │ per-sample   sample_site_info  sample_ref_blocks           │
                            │ per-batch    batch_site_info  batch_allele_counts          │
                            │              batch_ref_blocks  batch_genotypes             │
                            │ aggregate    tier_* staging  agg_* mirrors                 │
                            └────────────────────────────────────────────────────────────┘
                                   │ mosaic aggregate: SUM by key
                                   ▼
                            ┌ PostgreSQL ────────────────────────────────────────────────┐
                            │ contigs  samples  alleles  site_info  ref_blocks           │
                            │ sample_qc  control and checkpoint tables                   │
                            └────────────────────────────────────────────────────────────┘
                                   │                         │
                                   ▼                         ▼
                   joint_call.py, gwas_benchmark.py, SQL   per-sample reads from the lake
```

Each stage is a subcommand of the `mosaic` command (`mosaic/cli.py`), one module each:

| Subcommand | Module | Role |
|---|---|---|
| `mosaic init` | `mosaic/init_db.py` | create the PostgreSQL schema and the Iceberg tables; load `contigs` |
| `mosaic ingest` | `mosaic/ingest.py` | write one batch's per-sample evidence and partials |
| `mosaic register` | `mosaic/register.py` | register deferred batches' files with Iceberg |
| `mosaic aggregate` | `mosaic/aggregate.py` | sum the partials into the PostgreSQL aggregates |
| `mosaic remove` | `mosaic/remove.py` | remove samples as a negative batch |
| `mosaic index` | `mosaic/index_db.py` | re-create the `alleles` GiST if it is missing |
| `mosaic maintain` | `mosaic/maintain.py` | expire Iceberg snapshots and reclaim files |
| `mosaic clean` | `mosaic/clean_db.py` | drop MOSAIC's tables and data |

`joint_call.py` is a separate script, not part of the `mosaic` command. Two Nextflow
workflows run the stages on a batch system: `workflow/main.nf` (ingest, register,
aggregate, index, maintain) and `workflow/joint_call.nf` (joint calling per sample set and
chromosome).

## 2. Design goals and the central invariant

Correctness: the joint-called output is the same whether a cohort was built by one ingest
or by many additions and removals. Every aggregate equals the sum over the samples in the
cohort, with no double counting, no stale contribution and no orphaned site.

Fault tolerance: every stage is idempotent and crash-safe. A killed process, a lost
instance or a full disk leaves state from which a re-run converges to the result of an
uninterrupted run, and never a silent difference between the two stores.

Scalability: no operation serializes on a shared mutable row, and the cost of adding or
removing samples is proportional to those samples. Ingest and removal meet this; the
aggregate reads only the new partials but rewrites every chromosome's PostgreSQL rows.

The design follows from one invariant. For every aggregate column `A` and position `P`:

```
A(P) = Σ contribution_s(P)    over the samples s in the cohort
```

From it:

- Per-sample evidence is append-only and attributable to one sample. It is the only source
  of FORMAT fields and holds each record's exact contribution to the sums.
- Each batch writes partials that are sums of its samples' contributions. Partials are
  append-only, so concurrent writers never contend on a row.
- The PostgreSQL aggregates are derived: sums over the partials, rebuildable at any time
  from the lake.
- Adding samples and removing them are the same operation with opposite signs. A removal
  writes its samples' contribution negated as one more batch.

## 3. Data model

### 3.1 Reference data

| Table | Store | Content |
|---|---|---|
| `contigs` | PostgreSQL | contig id, name, length; loaded by `mosaic init`, which refuses a non-empty table |
| `samples` | PostgreSQL | `id`, unique `sample_name`, `status` (`active` or `removed`) |

`status = 'removed'` is the **tombstone** that `mosaic remove` sets: readers skip the
sample and `mosaic ingest` refuses it. The `batch_id` column of `samples` is unused, and no
table maps a sample to its batches; removal does not need one.

### 3.2 Per-sample evidence

Two Iceberg tables hold the per-sample evidence; every row belongs to one sample:

| Table | Grain | Columns |
|---|---|---|
| `sample_site_info` | one row per sample and variant record | `sample_id`, `chromosome_index`, `position`, `del_start`, `del_end`, `alleles`, `gt_1`, `gt_2`, `dp`, `ad`, `f1r2`, `f2r1`, `sb`, `pl`, `qual`, `site_dp`, `mqsq_1`, `mqsq_2`, `gtc`, `del_ref_spans` |
| `sample_ref_blocks` | one row per sample and reference block | `sample_id`, `chromosome_index`, `position_start`, `position_end`, `dp`, `ad_1`, `ad_2`, `pl_1`, `pl_2`, `pl_3`, `sample_counts` |

A `sample_site_info` row holds its record's own allele list: the reference base, then the
record's ALTs in the record's order. `gt_1` and `gt_2` are indices into that list, NULL for
no call, and `ad`, `f1r2`, `f2r1` and `pl` are as the record carries them. A row therefore
depends only on its own record, and one sample's rows are the same whatever batch it was
ingested in. Alleles use a prefix encoding after the shared suffix of REF and ALT is
trimmed: the reference is its first base, a substitution is `x` plus the ALT, an insertion
`+` plus the ALT, a deletion `-` plus the deleted reference including its anchor base, and
`<NON_REF>` is `*`.

Each row also stores its record's contribution to the batch partials: `qual`, `site_dp`
(INFO/DP, else FORMAT/DP), `mqsq_1`, `mqsq_2`, `sb` and `gtc` as a double. A deletion
carrier's row stores `del_ref_spans`: the spans of the sample's own reference blocks that
the deletion overlaps, which `sample_ref_blocks` does not hold. In `sample_ref_blocks`,
`dp` is the block's FORMAT/MIN_DP, else INFO/DP, and `sample_counts` is 1 for a hom-ref
block and 0 for a no-call block.

The workflow (`workflow/bin/fetch_and_ingest.py`) strips each gVCF to INFO DP, END, MQ,
MQSQ, RAW_MQ, RAW_MQandDP, GTC, SB and FORMAT GT, AD, DP, MIN_DP, SB, PL, F1R2, F2R1, GQ.
GQ feeds `batch_genotypes`, `sample_qc` and banding and is not stored per sample; FILTER is
counted in `sample_qc` only.

#### File layout

Both tables are partitioned by `chromosome_index`. A data file holds whole samples of one
batch and one chromosome, packed in `sample_id` order up to `--aggregate-parquet-size`
(default 256 MiB), and a sample larger than the target gets a file of its own. Inside a
file rows are sorted `(position, sample_id)`, in row groups of 65,536 rows for
`sample_ref_blocks` and 8,192 for `sample_site_info`. This layout serves three reads:

- A copy-on-write delete of one sample rewrites only the files that hold it and drops a
  file that holds it alone. pyiceberg 0.11 cannot write merge-on-read deletes, so the cost
  of a delete is set by the file packing.
- A sample-subset scan prunes files by their `sample_id` bounds.
- A position scan prunes row groups by their position statistics. Rows are position-major
  inside a file because a sample-major order would make a row group span one sample's whole
  chromosome.

Every file of a chromosome spans the chromosome, so a reader that plans each position window
separately opens every file per window; the joint caller instead reads each footer once and
streams row groups by position. `mosaic init` sets `write.parquet.row-group-limit` to the
same row-group sizes and the codec to brotli level 5 on both tables, so pyiceberg's
copy-on-write rewrites keep ingest's row groups and stay readable by the joint caller, whose
Parquet reader accepts brotli only.

#### Ref-block banding

`mosaic ingest --ref-gq-bands <spec>`, off by default, merges a sample's reference blocks
to shrink `sample_ref_blocks`, which is most of the lake. The spec is ascending GQ bounds in
[1, 99] (`20,60` gives bands [0,20), [20,60), [60,99]) or `all` for one band. Blocks merge
when they are adjacent, of the same `sample_counts` class, in the same GQ band and in the
same position window. The merged `dp` is the length-weighted mean of the parts, rounded half
up; AD and PL are those of the lowest-GQ part. The merged row and its `batch_ref_blocks`
range are emitted from the same values, so the partials remain sums of the stored rows.

Banding changes `sample_ref_blocks` and what derives from it: `ref_blocks.dp`, hence INFO/DP
and QD, and the FORMAT DP, AD and PL of reference-filled genotypes. It does not change
`sample_counts`, hence AN, nor GT, the other INFO fields, `batch_genotypes` or `sample_qc`.
A batch records its normalized spec in `ingestion_progress.ref_gq_bands`, a resume must use
the same spec, and a cohort may mix settings.

### 3.3 Per-batch partials

A batch is one `ingestion_progress` row: a sample set and a region. Ingest writes three
partials per batch, each the sum of what the batch's per-sample rows store:

| Table | Grain | Columns |
|---|---|---|
| `batch_site_info` | batch, site | `qual`, `dp`, `mqsq_1`, `mqsq_2`, `sb[4]`, `gtc[3]`, `n_records` |
| `batch_allele_counts` | batch, site, listed allele | `count`, `n_listed` |
| `batch_ref_blocks` | batch, covered position | `dp`, `sample_counts` |

- `batch_site_info` sums the rows' `qual`, `site_dp`, `mqsq_*`, `sb` and `gtc`, and
  `n_records` counts the rows at the site.
- `batch_allele_counts` has a row for every allele any row of the site lists, with count 0
  kept. `count` is the rows' `gt_1` and `gt_2` hits; `n_listed` is the rows whose own list
  holds the allele.
- `batch_ref_blocks` is dense: one row per covered position, the sum of the reference
  blocks' `dp` and `sample_counts` plus each `del_ref_spans` span with its row's `site_dp`
  and a sample count of 0. A position whose two sums are both 0 has no row.

`gtc` is summed in fixed point. Each record's double is quantized once to
`round(gtc × 2^32)` as int64, half to even, and the partials hold the integer sums. An
integer sum does not depend on the order of reduction, so a removal nets out exactly and an
incremental aggregate equals a full one. `qual` and `mqsq_1` are summed in float64.

A `mosaic remove` batch (`ingestion_progress.kind = 'removal'`) writes the same three tables
with its samples' contribution negated.

The test suite recomputes every partial row from the per-sample tables alone
(`assert_partials_derivable_from_lake` in `tests/jc_testlib.py`): integers exactly, `qual`
and `mqsq_1` to the last ULPs.

Two more products of ingest are read-side outputs that the aggregate does not use:

- `batch_genotypes`, partitioned by chromosome: per batch and variant site, the alleles and
  `gt_1`, `gt_2`, `gq` as blobs over the batch's sample slots, with hom-ref, no-call and `*`
  resolved from the batch's reference blocks and deletions. A GT blob has one byte per slot
  when every code fits in [-1, 127], else two; -1 is no call. `read_genotype_matrix` in
  `mosaic/genotypes.py` reads it and drops removed samples' columns.
- `sample_qc` in PostgreSQL: per batch, sample and chromosome, counters of variant types,
  zygosity, Ti/Tv, depth, callable reference bases and filters. The view
  `sample_qc_summary` sums them per active sample.

### 3.4 Derived aggregates

`mosaic aggregate` sums the partials into three PostgreSQL tables, each LIST-partitioned by
`chromosome_index` with one partition per contig and no foreign key to `contigs`:

| Table | Key | Columns | Feeds |
|---|---|---|---|
| `alleles` | chromosome, position, allele | `count`, `del_range` | the ALT list, spanning deletions, the catalog cross-check |
| `site_info` | chromosome, position | `qual`, `dp`, `mqsq_1`, `mqsq_2`, `sb[4]`, `gtc[3]` | INFO DP, QD, MQ, FS, SOR, InbreedingCoeff |
| `ref_blocks` | chromosome, position | `dp`, `sample_counts` | INFO DP, at variant sites only |

`alleles` is the allele catalog: every allele any batch lists, count 0 kept, with no
surrogate id. Ingest writes nothing to it. `del_range`, GiST-indexed, is the span
`[position, position + len(allele) - 1)` of a `-` allele and NULL otherwise; a complex
substitution such as `ATG>CA` encodes as `-ATG` and gets the same span as `ATG>A`.

`ref_blocks` is sparse: one row per variant site. At thousands of samples the summed depth
changes at nearly every base, so ranges would compress nothing, and the aggregate is read
only at variant sites. The depth at every covered position is kept in the lake as each
run's dense root (below).

INFO DP is `site_info.dp + ref_blocks.dp`; QD is `qual / DP`; MQ is `√(mqsq_1 / mqsq_2)`;
FS and SOR come from `sb`; InbreedingCoeff from `gtc` when its sum is at least 10. AC, AN
and AF are not taken from the aggregates (see Joint calling).

Integer columns are exact under any summation order and under subtraction; `gtc` holds
the integer part of the fixed-point sum. `qual` and `mqsq_1` are `REAL` and can differ in
the last float32 ULP between reduction orders, below VCF output precision.

In Iceberg, the `tier_*` tables hold the reduction's intermediate and root sums, tagged with
the run id, and the write-only `agg_*` tables mirror the PostgreSQL result with every root
row and the fixed-point `gtc`.

### 3.5 Control state

| Table | Role |
|---|---|
| `ingestion_progress` | one row per batch: `batch_id`, `sample_names`, `region`, `status`, `kind`, `sample_ids` (the `batch_genotypes` slot order), `ref_gq_bands` |
| `shard_completion` | one row per finished shard of an unfinished batch |
| `ingest_files` | the data files of a deferred batch, with `registered_at` NULL until registered |
| `aggregation_progress` | one row per aggregate run: the batch set and its hash, `fan_in`, `ref_shard_size`, `status`, `parent_run_id`, `delta_batch_ids`, `staging_pruned` |
| `tier_completion` | one row per finished aggregate unit: run, metric, chromosome, tier, group |
| `removal_progress` | one row per removal: the samples, the negative batch, `status` |
| `cohort_generation` | a view of the latest `aggregation_progress` row |

A batch's `status` is `in_progress`, `failed`, `uploaded` (files in S3 and listed in
`ingest_files`, not registered) or `complete`. The aggregate, the gate and the readers see
`complete` batches only.

## 4. Pipeline stages

| Stage | Command | Idempotent by | Unit of commit | Purpose |
|---|---|---|---|---|
| Initialize | `mosaic init` | not idempotent: refuses a loaded `contigs` | per statement | schema, Iceberg tables, contigs |
| Ingest | `mosaic ingest` | batch key and deterministic keys | shard checkpoint; batch registration or `ingest_files` | evidence and partials of one batch |
| Register | `mosaic register` | Iceberg first, PostgreSQL mark second | `--files-per-commit` files, then one transaction | register deferred files, complete batches |
| Aggregate | `mosaic aggregate` | run key and per-unit checkpoints | per barrier; per chromosome swap | PostgreSQL aggregates from the partials |
| Index | `mosaic index` | `IF NOT EXISTS` | per index | the `alleles` parent GiST |
| Maintain | `mosaic maintain` | a recomputed retention set | per table | expire snapshots, reclaim files |
| Remove | `mosaic remove` | `removal_progress` status | T0, registration, T1, deletes, T2 | negative batch, per-sample delete |
| Joint call | `joint_call.py` | read-only | none | VCF from the aggregates and the lake |
| Reset | `mosaic clean` | yes | none | drop everything; the cloud path needs `--yes` |

### 4.1 Ingest

`mosaic/ingest.py` is the driver of the Rust shard worker `mosaic-ingest-worker`, one
process per batch. The driver checks the schemas, registers the samples, creates or resumes
the batch row, cuts the region into shards (`--shard-size`, default 10 Mbp) and writes a
JSON manifest. The worker runs the shards on `--threads` threads and prints a JSON line per
finished shard. It opens no database connection: ingest writes nothing to PostgreSQL but
`samples`, its control rows (`ingestion_progress`, `shard_completion`, and `ingest_files`
under `--defer-register`) and `sample_qc`.

Per shard, the worker merges the samples' records by position and writes local Parquet:
per-sample rows, partials and `batch_genotypes`. A record belongs to the shard that holds
its position; reference blocks are clipped to the shard. Each sample's reference block is
emitted once, when the merge reaches it, split around the sample's own deletions, and added
as a range to its position window (`--pos-batch-size`, default 10 Mbp). A flushed window
sums its ranges per position into the dense `batch_ref_blocks` rows, so the partial equals
the per-sample sum and memory is bounded by the window.

After the last shard the worker combines the local files into data files. The per-sample
tables' local files are containers of one Parquet blob per window, chromosome and sample;
the combine packs whole samples per chromosome into files, merges their blobs by
`(position, sample_id)`, and patches in deletion spans another shard saw. The partials and
`batch_genotypes` are concatenated in genomic order. Each file is written by one thread from
an order the packing fixes, so the bytes do not depend on the thread count. The driver
uploads each finished file on 4 threads to a deterministic key, deletes it and acknowledges
it on the worker's stdin; the worker holds at most 8 unacknowledged files, which bounds
local disk.

By default the driver then makes one `add_files` commit per table and marks the batch
`complete`. With `--defer-register` it records the files in `ingest_files` and marks the
batch `uploaded` in one transaction, for `mosaic register`. The Nextflow workflow defers.

The driver commits a shard's `shard_completion` row when its result line arrives. On
resume, a shard without a row runs fresh and a shard with one is reprocessed to regenerate
its Parquet, which the batch upload needs; the output is a function of the input, so the
files are identical and go to the same keys. A fresh batch registers without scanning the
table, because its keys contain a new `batch_id`; a resumed batch skips URIs the table holds
and keeps pyiceberg's duplicate check. Re-running ingest on an `uploaded` batch does
nothing.

### 4.2 Register

`mosaic register` is the one Iceberg committer for deferred batches. Holding the shared
advisory lock, it scans each table's registered URIs once, marks a pending file the table
holds without adding it, and adds the rest in commits of `--files-per-commit` files, 500 by
default. After each commit one PostgreSQL transaction marks the files and completes every
batch with nothing left pending. A crash between the two leaves files that the next run
finds registered and marks.

### 4.3 Aggregate

`mosaic aggregate` reduces the partials of the `complete` batches into the PostgreSQL
aggregates. Every metric is a sum by key: `site_info` and `ref_blocks` by chromosome and
position, `allele_counts` by chromosome, position and allele. Partials are signed, so
ingest and removal batches are summed alike.

#### Tier tree

`site_info` and `allele_counts` reduce as a tree, planned per chromosome over the batches
whose region covers it. Tier 0 is the partials; groups of `--fan-in` inputs (default 100)
are summed into one file of the matching `tier_*` table, and tiers repeat until one root
remains. With at most `fan_in` batches a chromosome reduces in one pass. Each unit reads
its inputs one data file at a time and sums them with an Arrow `group_by` into a buffer of
bounded rows (`MOSAIC_REDUCE_BUFFER_ROWS`, default 4 million).

#### Reference depth by window

`ref_blocks` has no tree. One unit per chromosome and `--ref-shard-size` window (default
20 Mbp) sums every covering batch's dense rows into two int64 arrays indexed by position,
16 bytes per base. It uploads the window's dense root, every position whose net
`(dp, sample_counts)` is not `(0, 0)`, to `tier_ref_blocks_dense`, and its sparse root, the
dense root at the sites of PostgreSQL `site_info`, to `tier_ref_blocks`. `--fan-in` does not
apply. Because the sparse root reads `site_info`, the order of a run is: the
`allele_counts` and `site_info` reduces, the `site_info` finalize, the reference windows,
the `ref_blocks` finalize, the `allele_counts` finalize.

#### Registration and checkpoints

Units run in a spawn-based process pool of `--threads` workers; a worker that dies fails the
run instead of hanging it. The workers make no Iceberg commit: a unit uploads its output to
a key derived from the run id and the unit, and returns the URIs. After each
barrier (a tier, a fold, the reference windows) the main process makes one `add_files` per
staging table for all of the barrier's files, then one PostgreSQL transaction with the
barrier's `tier_completion` rows. A checkpoint therefore implies that its file is
registered. A unit whose checkpoint exists is skipped on resume; any other unit is
recomputed and uploads the same bytes to the same key.

#### Finalize

One unit per metric and chromosome builds a new partition and swaps it in. It copies the
root's rows in chunks of `MOSAIC_FINALIZE_CHUNK_ROWS` (default 1 million) into a fresh table
`{table}_c{c}_r{agg_run_id}` created without autovacuum and loaded with `COPY ... FREEZE`,
builds its primary key once (and, for `alleles`, the GiST on `del_range`), and analyzes it.
One short transaction then detaches the attached partition, attaches the new one, drops the
old one and commits the unit's checkpoint. A crash before that transaction leaves the old
partition in place; readers never see a half-built chromosome. The build runs under a
statement timeout (`MOSAIC_FINALIZE_STATEMENT_TIMEOUT`, default 30 min) and the swap under a
5 min lock timeout. Until its swap a new partition exists beside the one it replaces, so the
database needs free space for the chromosomes in flight. Dead rows are dropped as the rows
are copied:

- a site whose net `n_records` is 0;
- an allele whose net `n_listed` is 0;
- for `ref_blocks`, a position whose net `(dp, sample_counts)` is `(0, 0)` has no root row.

With only positive partials no row is dropped; after a removal these rules remove the
sites, alleles and positions that only the removed samples had. Each chunk is also uploaded
as an `agg_*` mirror file; after a metric's finalize barrier the main process replaces that
mirror in one Iceberg transaction.

#### Incremental runs

Without `--rebuild`, a run takes as its parent the latest complete, unpruned run whose
batches are a proper subset of the completed batches, and reduces only the delta batches.
Per chromosome a fold unit sums the delta's root with the parent's root, and each reference
window adds the parent's dense root rows in its range. The result equals a full run:
integers exactly, `qual` and `mqsq_1` to the last ULP. `--rebuild` starts a full run with no
parent. If the latest run over the current batch set is complete, the command prints
`up to date` and exits 0.

When a run completes it marks every other run `staging_pruned` and commits a prune sentinel
in the same transaction, then deletes those runs' staging and its own non-root tiers. Only
the newest complete run's roots remain, to serve as the next parent. These are whole-file
Iceberg deletes; the objects leave S3 when `mosaic maintain` expires the snapshots that name
them.

#### Resume

A run is keyed by the hash of its batch set, `fan_in`, `ref_shard_size` and its parent. A
failed run with the same key resumes, skipping checkpointed units and swaps. A pruned run is
never resumed or chosen as a parent; a run that had begun its own prune resumes onto its
stored parent. A fresh run skips the staging tables' URI scan and duplicate check; a resume
performs them once per barrier per table.

#### Memory

A tier unit holds one input file, its buffer and its result's distinct keys, so a
chromosome's root unit holds that chromosome's whole result. A reference window holds 16
bytes per base, a finalize one chunk. Peak memory is the sum over the units in flight, so
`--threads` trades time for memory; at the largest cohort measured, memory is the limit.

### 4.4 Index and maintain

The finalize builds every index of the aggregate tables. `mosaic index` re-creates the
parent GiST on `alleles.del_range` if it is missing and otherwise does nothing; the
workflow runs it after the aggregate.

`mosaic maintain`, per Iceberg table, sets missing table properties, merges the current
snapshot's manifests, expires the snapshots outside the retention policy (`--retain-last`,
`--older-than-hours`, branches and tags) and deletes the files only they needed, plus files
the retained snapshots name only as deleted, such as pruned staging. Deletes follow the
expire commit, so a crash leaves unreferenced files, never a dangling reference.
`--orphans` also deletes unreferenced objects under the tables' prefixes that are older than
a grace period and not pending in `ingest_files`. It holds the shared advisory lock and
refuses to run during an aggregate run unless `--force`. Tables are created with
`write.metadata.delete-after-commit.enabled`, which keeps 100 metadata versions. The
workflow runs `mosaic maintain --older-than-hours 0` after the aggregate.

### 4.5 Remove

`mosaic remove <samples>` is an initial version, not part of the Nextflow workflow. It
writes the samples' contribution, negated, as one batch, which the next `mosaic aggregate`
folds in like an added one. Under the joint-call gate an in-place decrement would take
effect no sooner, and a batch reuses the exactly-once mechanics that exist for ingest.

It holds the shared advisory lock for its whole run and refuses, changing nothing, on an
unknown or already removed sample, on a sample in an ingest batch that is not `complete`,
and while a different removal is unfinished. Then:

1. T0, one transaction: `samples.status = 'removed'`, the removal batch in
   `ingestion_progress` (`kind = 'removal'`) and the `removal_progress` row. From here
   readers hide the samples and ingest refuses them.
2. Derive: one unit per chromosome and `--shard-size` window, in a process pool with no
   database connection, reads the samples' rows and computes their partials as ingest
   computes them from the same records, negated. Deletion rows are read from `--span-margin`
   (100 kb) before the
   window; a longer span is an error. The main process uploads the partials to
   deterministic keys and registers them, one commit per table, as a resumed ingest batch
   does.
3. T1: the batch `complete`, the removal `registered`.
4. Delete the samples' rows from both per-sample tables, one copy-on-write commit per table.
   A file holding only removed samples is dropped; a shared file is rewritten.
5. T2: the removal `complete`.

A failure marks the removal `failed`; the same command resumes it at the derive, or after
T1 at the delete. From T1 the joint-call gate refuses the cohort until the next `mosaic
aggregate`, which drops what only the removed samples had. Its result equals a fresh ingest
of the remaining samples: integer columns exact, `qual` and `mqsq_1` within a float32 ULP,
the joint-called VCF byte-identical. The integration tests check this on a six-sample
cohort. The deleted rows' files leave S3 at the next `mosaic maintain`. A removed sample
keeps its slot and bytes in `batch_genotypes`, where the reader hides its column, and cannot
be re-added.

### 4.6 Joint calling

`joint_call.py` is the driver of the Rust binary `mosaic-joint-call`. The driver checks the
cohort, resolves the samples and region, exports from PostgreSQL, plans the data files
through the Iceberg catalog and writes a manifest. The binary reads the lake, calls and
writes the VCF, BGZF without an index for a `.gz` output. On a failure the driver deletes
the partial output.

#### Gate

The driver exits 2 when there is no aggregate run, the latest run is not `complete`, or its
batch set differs from the completed batches; `--allow-stale` overrides. A completed removal
batch counts like any batch, so the gate refuses the cohort from a removal's T1 until the
next aggregate.

#### Inputs and reader

Per contig the driver exports the sites (`site_info` left-joined to `ref_blocks`) and the
`alleles` rows, with deletions that span into the range from before it. A site is written
iff it has a `site_info` row. Removed samples' ids go in the manifest and the reader masks
their rows. The binary works in units of `--window` bases, 10 Mbp by default. It reads each
data file's footer once; a unit opens every file of its contig, selects row groups by their
position statistics and merges the files' rows by `(position, sample_id)` in a heap,
holding about one row group per file.

#### Ordered output

Units run on `--threads` threads, each compressing its text in 4 MiB chunks, and the output
is written in unit order. The head unit, the lowest-numbered unit not fully written, writes
its chunks to the output directly. Every other unit buffers its chunks in memory up to
`--unit-buffer-bytes`, 64 MiB by default, then spills them to a file of its own. When the
head finishes, the following finished units are copied in order and the next running unit
becomes the head. A unit starts only while it is fewer than `2 × threads` units past the
head, so at most `2 × threads − 1` units hold output. The VCF text does not depend on
`--threads`, `--window` or the buffer size, and the compressed bytes of a `.gz` output do not
depend on `--threads` or the buffer size. With more than one thread about
(threads − 1) / threads of a large output passes through spill files, so it needs free disk
of about the output's size beside it or under `--spill-dir`.

#### Output fields

- FORMAT GT, AD, DP, PL, F1R2 and F2R1 come from the sample's own row. A sample without
  one is filled from its reference block, `0/0` when `sample_counts` is 1 and `./.` for a
  no-call block, with the block's DP, AD and PL.
- A sample whose own deletion spans the site gets a star call, which takes precedence over
  a reference fill. Its `*` and star allele are found by allele string in its record's own
  list; a genotype the record cannot price gets PL 990.
- AC, AN and AF come from the written genotypes, with `bcftools +fill-tags` semantics: AN
  counts called alleles, AC each ALT with `*` an ALT, AF is `AC / AN` to three decimals. A
  record without ALTs carries no AC and no AF.
- QUAL and the other INFO fields come from the aggregates (see Derived aggregates). A
  removed sample is excluded from AC, AN and AF from its tombstone on, and from the other
  INFO fields after the aggregate that applies its removal.

After the VCF the driver prints a catalog cross-check: the records whose written AC differs
from `alleles.count`, by reason. Differences are expected under `--allow-stale` and where a
sample's own call names an allele the record cannot express, written `./.`; capped records
are not compared.

#### Options as projections

Four options change the output. Each projects the record the step before it wrote, in the
order subset, trim, cap, minimum AC:

- `--samples` or `--samples-file`: the record restricted to these samples' columns, in the
  header's order. Every cohort site remains a record; `*` leaves the ALT list when no
  subset sample carries it; a FORMAT key none of them has is dropped; AC, AN and AF are the
  subset's; QUAL and the other INFO fields are the cohort's. Only the data files whose
  `sample_id` bounds can hold a subset sample are read, and the catalog cross-check is
  skipped.
- `--trim-alt-alleles`: ALTs with AC 0 leave the record and the GTs are re-indexed.
- `--max-alt-alleles N`, default 6, 0 for none: a record with more than N ALTs keeps the N
  with the largest AC, a tie going to the lower index, in their order. A genotype carrying a
  dropped ALT is written `./.`, AD, F1R2, F2R1 and PL keep the kept alleles' entries, and
  AC, AN and AF are recomputed. The ALTs are ranked before any call is built, and calls are
  built over the kept alleles only, which bounds a site's calls and a record's text.
- `--min-ac K`: a record is written only if its written genotypes hold at least K ALT
  observations.

A projected PL keeps the kept genotypes' entries and is shifted by their minimum only when
the source PL held a 0 and the kept entries are all positive. Each option adds a header
line and a closing line with its counts. `tests/vcf_project.py` implements the projections
on VCF text, and the tests compare the engine's output with them byte for byte.

#### Cohort export and the joint-call workflow

The PostgreSQL side of a joint call can be written once and reused. `--cohort-state`
prints the cohort state as one JSON line (the aggregate run and its time, the completed
batches, a digest of the samples, the stale reason), identical across calls on an unchanged
cohort. `--export-cohort DIR` writes it with the contigs and samples to `cohort.json`.
`--export-contig DIR --region CONTIG` writes the contig's whole sites and alleles exports,
after checking in one `REPEATABLE READ` snapshot that the database's state is
`cohort.json`'s. `--from-export DIR` joint-calls from the directory with no PostgreSQL
connection, writing the PostgreSQL path's output byte for byte; the lake is read as usual.

`workflow/joint_call.nf` writes one VCF per sample set and chromosome with these modes:
`COHORT_STATE` on every run, whose output keys the cache of every later task, then
`EXPORT_COHORT`, `EXPORT_CONTIG` per chromosome with sites, `JOINT_CALL` per set and
chromosome, and `JOINT_CALL_SUMMARY`. The sets are `--samples` cut by the run's
`--batch_size`, independent of the ingest batches, or the whole cohort. A call task indexes
its VCF, checks the index's record count and writes `summary.json` last, under
`<joint_call_out>/gen<agg_run_id>/`. The exports are the only PostgreSQL load, and nothing
checks the cohort at the end of a run.

#### Haploid calls

The calling logic is diploid. A `sample_site_info` row whose `gt_2` is NULL resolves to no
genotype pair, so a haploid ALT call is written `./.`, and a reference block with
`sample_counts` 1 is filled `0/0` because `sample_ref_blocks` stores no ploidy. Callers
commonly write a male's calls haploid on chrX outside the PARs and on chrY, so there the
genotypes, AC, AN and AF are wrong for male samples. The cross-check reports those records
as `other`. The lake and the PostgreSQL allele counts are not affected: `alleles.count`
counts a haploid call once, `batch_genotypes` holds it as `(a, -1)`, and `sample_qc`
counts it.

## 5. Cross-cutting rules

### 5.1 Fault tolerance

The ordering rule: make the data durable first, write the checkpoint second. A checkpoint
that claims more than it owns can lose data silently. Ingest's `shard_completion` row does
not imply that the shard's files are in Iceberg, so a resume reprocesses committed shards;
the aggregate's and the registrar's checkpoints are written after the Iceberg commit, so
they do imply it. Every derived table can be rebuilt from the partials, so a crash leaves
at worst work to redo.

| Stage | Checkpoint | Resume key | Idempotence |
|---|---|---|---|
| Ingest | `shard_completion` row per shard; batch `complete` or `uploaded` | sorted sample names and region | committed shards reprocessed; deterministic keys; a resume filters registered URIs |
| Register | `ingest_files.registered_at`, after each Iceberg commit | pending `ingest_files` rows | a URI the table holds is marked, not added |
| Aggregate reduce | `tier_completion` per unit, after the barrier's Iceberg commits | batch-set hash, `fan_in`, `ref_shard_size`, parent | checkpointed units skipped; others recomputed to the same keys |
| Aggregate finalize | `tier_completion` with tier -1, in the swap transaction | as above | a committed swap skipped; an uncommitted one rebuilt |
| Remove | `removal_progress.status` and the removal batch's status | the same sample set | T0 skipped on resume; derive redone before T1; deletes idempotent |

A batch's partial keys are written once: shards and windows are disjoint, and every file
key contains the batch id. The aggregate relies on this and does not dedup rows.

### 5.2 Concurrency

- Ingest batches may run concurrently with disjoint sample sets, or with the same samples
  on different regions. Deferred, they make no Iceberg commit; inline, each commits to six
  shared tables and `add_files` retries on conflict.
- `mosaic register`, `mosaic maintain` and `mosaic remove` share one PostgreSQL session
  advisory lock, the Iceberg metadata-writer lock; a second holder exits 1. A registrar may
  overlap deferred ingests and registers what was pending when it started.
- `mosaic remove` and `mosaic ingest` touch disjoint rows, because ingest refuses a
  tombstoned sample.
- Aggregate runs are serialized by design. A run's batch set is fixed at its start and it
  reads no per-sample table, so a batch that completes during the run, from an ingest or a
  removal, is left for the next run.
- An aggregate must not overlap a joint call, which would read partitions while they are
  swapped. The workflow runs the aggregate after every ingest task and before any joint
  call; the code does not enforce that order, except that `mosaic maintain` refuses to run
  during an aggregate.
- Two ingests registering the same new sample names simultaneously, such as the per-chromosome
  tasks of one sample batch, can collide in `register_samples` (`mosaic/ingest.py`, a select
  then an insert): one exits 1 before any work, and a retry passes. This race is not fixed.
- No `mosaic remove`, ingest or aggregate may run during a joint-call workflow run, because
  nothing checks the cohort at its end.

### 5.3 Output gating

Joint calling runs only against a fully aggregated cohort (see Joint calling, Gate). The
gate is a startup check, not a pinned generation: an aggregate that starts swapping
partitions during a joint call is not detected. The cohort export narrows the window,
because each contig export checks the state recorded in `cohort.json`.

The gate does not cover a removal between its T0 and its T1, or one that failed before T1:
the tombstone hides the samples' columns while the aggregates count them, and the removal
batch is not `complete`, so the gate passes. Do not joint-call while a removal is
unfinished.

### 5.4 Maintenance and audit

Each Iceberg commit adds a snapshot and manifests, and a whole-file delete only records the
deletion; `mosaic maintain` bounds the growth, and deferred registration cuts ingest's
commits to one per table per registrar run below 500 pending files. A tagged snapshot is
never expired and can serve as a pre-removal audit anchor, subject to any deletion deadline.
`ingestion_progress`, `aggregation_progress` and `removal_progress` record what was added,
aggregated and removed, and when.

## 6. Scaling characteristics

| Operation | Cost is proportional to |
|---|---|
| Ingest | the batch's samples × the region; independent of the existing cohort |
| Ingest memory | samples per batch × shard and window size |
| Register | the files pending, in commits of 500 |
| Aggregate reads | the delta's partials plus the parent's roots; all partials under `--rebuild` |
| Aggregate finalize | the whole cohort's sites and alleles: every chromosome is rewritten |
| Aggregate memory | a chromosome's distinct keys at the root, plus 16 bytes per base per reference window, times the units in flight |
| Removal | the removed samples × the genome; its aggregate costs an incremental run |
| PostgreSQL size | distinct sites and alleles, which grow sub-linearly with samples |
| Lake size | samples × genome, linear |
| Joint call time | records × samples in the call |
| Joint call memory | samples in the call × units in flight, plus the data files per contig |

Measured on AWS with all 3,202 genomes of the 1000 Genomes Project, reference blocks banded
at GQ 20 unless stated (details in [`benchmarks.md`](benchmarks.md)):

- Ingest: 1,048.7 core-hours (0.33 per genome) in 425 tasks.
- Ingest cost per sample per Mb: between 0.38 and 0.43 core-seconds from 100 samples of one
  chromosome to 3,202 whole genomes.
- Aggregate: 63.7 min in one task at 2 threads; peak memory 12.84 GB of a 12 GiB task. At 4
  threads it was killed by the memory limit.
- Lake: 1,501.4 GB, 11.6% of the 12,979 GB of source gVCFs.
- PostgreSQL: 69.8 GB; 126,001,716 sites.
- PostgreSQL on chr21, unbanded: 0.15 GB at 100 samples, 0.94 GB at 3,202.
- Joint call, 17 sets of 200 samples × 25 chromosomes: 425 VCFs in 54.1 min.
- Whole-cohort joint call of chr1: 2 threads in a 12 GiB task (8.24 GB, 3.6 h).

Nothing above 3,202 samples has been measured.

## 7. Status and open items

| Component | Status | Code |
|---|---|---|
| Per-sample evidence and its file layout | implemented | `rust/mosaic-ingest-worker`, `mosaic/init_db.py` |
| Per-batch partials | implemented | `rust/mosaic-ingest-worker` |
| Ingest driver, checkpoints and resume | implemented | `mosaic/ingest.py` |
| Ref-block banding | implemented, off by default | `mosaic/ref_banding.py`, `rust/mosaic-ingest-worker/src/banding.rs` |
| Deferred registration | implemented | `mosaic/register.py` |
| Aggregate: tier tree, reference windows, finalize, incremental runs, resume | implemented | `mosaic/aggregate.py` |
| Iceberg maintenance | implemented | `mosaic/maintain.py` |
| Sample removal | initial version | `mosaic/remove.py`, `mosaic/tombstones.py` |
| Genotype matrix and its reader | initial version | `rust/mosaic-ingest-worker/src/matrix.rs`, `mosaic/genotypes.py` |
| Per-sample QC | implemented | `rust/mosaic-ingest-worker/src/qc.rs` |
| Output gate | implemented as a startup check | `joint_call.py` |
| Joint calling | implemented, diploid only | `joint_call.py`, `rust/mosaic-joint-call` |
| Cohort export and joint-call workflow | implemented | `joint_call.py`, `workflow/joint_call.nf`, `workflow/bin/joint_call_task.py` |

Designed or known, not built:

- The site-row reference fold: `ref_blocks.dp` and `sample_counts` as `site_info` columns,
  so every INFO field is one PostgreSQL row.
- A scheduled rebuild into shadow tables, diffed against the incremental aggregates.
- A generation pinned for the length of a joint call.
- A gate that refuses the cohort while a removal is unfinished before its T1.
- Data-file compaction after copy-on-write deletes, which leave smaller files.
- Merge-on-read deletes, unavailable in pyiceberg 0.11.
- Scrubbing a removed sample's slots from `batch_genotypes`.
- A fast path that removes a whole batch by negating its own partials.
- Re-adding a removed sample.
- Haploid genotypes in joint calling: a ploidy for reference blocks at read time and
  haploid GT and PL in the writer.
- A fix for the sample-registration race in `register_samples`.
- Genotype accuracy at cohort scale: the only comparison with another caller covers two
  samples on part of chr21.
- An exact global AN in `scripts/gwas_benchmark.py`, which uses twice the sample count.
