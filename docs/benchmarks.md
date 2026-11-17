# MOSAIC benchmarks

Results of the cloud scale tests of November 2026 on the current code: ingest, aggregation,
storage and joint calling of up to 3,202 whole genomes from the 1000 Genomes Project, on AWS
Batch. Every figure states its cohort, its region and whether the lake was banded. Sample
removal (`mosaic remove`) is an initial version. Nothing above 3,202 samples was measured.

Headline, all 3,202 genomes on all 25 chromosomes (chr1 to chr22, X, Y, M), reference blocks
banded at GQ 20 (13 to 14 Nov 2026):

| Stage | Result |
|---|---|
| Ingest | 1,048.7 core-hours (0.33 per genome) in 425 tasks; 9.3 h from the first task's start to the last task's end, at most 30 tasks at once; no task failed |
| Aggregate | 63.7 min in one task at 2 threads; peak memory 12.84 GB of a 12 GiB (12.88 GB) task |
| Joint call (17 sets of 200 samples × 25 chromosomes) | 425 VCFs in 54.1 min; 80.99 core-hours; 1,111.7 GB of VCF |
| Lake (Iceberg / Parquet) | 1,501.4 GB, 11.6% of the 12,979 GB of source gVCFs (not like for like: see Storage) |
| PostgreSQL | 69.8 GB; 126,001,716 sites, 408,139,676 alleles |

The main limits: the aggregate's memory grows with the cohort and fit a 12 GiB task only at 2
threads at this scale; a whole-cohort joint call of chr1 needs 2 threads in 12 GiB; the joint
caller writes diploid genotypes only (wrong on chrX outside the PARs and on chrY for male
samples); genotype accuracy at cohort scale is not measured.

## Setup

**Data.** The 1000 Genomes DRAGEN v4.2.7 gVCFs (3,202 samples, hg38) from the AWS Open Data
bucket, 12,979 GB in all (mean 4.05 GB per sample, range 1.56 to 7.25 GB). Each ingest task
downloads its samples' region with `bcftools` and keeps only INFO DP, END, MQ, MQSQ, RAW_MQ,
RAW_MQandDP, GTC, SB and FORMAT GT, AD, DP, MIN_DP, SB, PL, F1R2, F2R1, GQ
(`workflow/bin/fetch_and_ingest.py`), then runs `mosaic ingest` on the slices.

**Cohorts** (13 to 14 Nov 2026, one ingest task per batch and chromosome):

| Cohort | Samples | Region | Samples per batch | Ingest tasks | Reference blocks |
|---|---|---|---|---|---|
| chr21 × 100 | 100 | chr21 (46.7 Mb) | 20 | 5 | unbanded |
| chr21 × 3,202 | 3,202 | chr21 | 200 | 17 | unbanded |
| genome × 100 | 100 | 25 chromosomes (3,088.1 Mb) | 50 | 50 | unbanded |
| chr1 × 3,202 | 3,202 | chr1 (249.0 Mb) | 200 | 17 | banded at GQ 20 |
| genome × 3,202 | 3,202 | 25 chromosomes | 200 | 425 | banded at GQ 20 |

With 200 samples per batch, the 17th batch holds the last 2 samples. Banding
(`--ref_gq_bands 20`, off by default) merges a sample's adjacent reference blocks whose GQ falls
in the same band; it is lossy for reference-block depth and likelihoods (see Storage and
Correctness).

**Infrastructure.** AWS Batch on on-demand m8a.xlarge instances (4 vCPU, 16 GiB), at most 256
vCPU, each instance with a 1000 GB volume. Every task: 4 cpus, 12 GiB. At most 30 ingest tasks at
once. PostgreSQL 18.3 on RDS `db.m8gd.large` with a 100 GiB gp3 volume (storage autoscaling to
400 GiB enabled), one database per cohort. Iceberg tables in S3 with the AWS Glue catalog.
Nextflow 26.10.6; the checks on the orchestration node used bcftools / htslib 1.25.

**Workflow settings.** The defaults of `workflow/nextflow.config` except where stated: shards
of 500 kb, position windows of 100 kb, aggregate fan-in 100, reference-depth windows of
20 Mb, deferred registration (one `REGISTER` step), `mosaic maintain` after the aggregate.
The aggregate ran at the task's 4 threads, except at genome × 3,202 (2 threads, see Aggregation).

**Joint calls on one node** (11 to 12 Nov 2026). The whole-chromosome calls of chr21 × 3,202
and genome × 100 in the joint-calling section were run directly with `joint_call.py` on an
r8a.xlarge instance (4 vCPU, 32 GiB, no swap), each under a 14 GiB memory limit
(`systemd-run`), with bcftools 1.23.1 / htslib 1.24 for the checks. The cohorts were ingested
the same way as above and their aggregates are identical (see Correctness).

**Units.** GB = 10^9 bytes; GiB only for configured limits. Core-hours = a task's wall time ×
its 4 cpus, from the Nextflow traces. Memory is a task's peak resident memory from the trace;
for the node calls it is `/usr/bin/time -v`'s maximum, converted to GB. Core-seconds per
sample per Mb = core-hours × 3,600 / samples / Mb of reference sequence in the region.

## Ingestion

| Cohort | Tasks | Core-hours | Core-s per sample per Mb | Wall, first start to last end | Slowest task | Peak memory | Retries |
|---|---|---|---|---|---|---|---|
| chr21 × 100 | 5 | 0.54 | 0.413 | 2.0 min | 99 s | 0.72 GB | 0 |
| chr21 × 3,202 | 17 | 16.53 | 0.398 | 18.5 min | 975 s | 3.91 GB | 0 |
| genome × 100 | 50 | 36.70 | 0.428 | 23.3 min | 1,361 s | 2.26 GB | 0 |
| chr1 × 3,202 (banded) | 17 | 86.37 | 0.390 | 87.2 min | 5,056 s | 4.42 GB | 0 |
| genome × 3,202 (banded) | 425 | 1,048.71 | 0.382 | 9.3 h | 5,047 s | 7.40 GB | 0 |

The cost per sample per Mb stays between 0.38 and 0.43 core-seconds from 100 samples of one
chromosome to 3,202 whole genomes: ingest cost is linear in samples × sequence. A task's work
depends on its batch (samples) and chromosome, not on the size of the cohort already ingested.
Three downloads were retried inside their tasks (chr1 × 3,202: 2, genome × 3,202: 1); every
retry passed.

**What an ingest task includes.** The download and slicing of its gVCFs from the public bucket
takes 57 to 60% of a task's wall unbanded and 69% banded (medians per cohort); the rest is
`mosaic ingest` (the Rust shard worker, the combine of its local files, the upload to S3).
Banding shortens the ingest stage itself: on chr21 with the same 3,202 samples and batches,
its median is 371 s unbanded and 239 s banded.

**Genome × 3,202 by chromosome** (17 tasks per chromosome, the last of them a 2-sample
batch):

| Chromosome | Core-hours | Median task | Peak memory | Core-s per sample per Mb |
|---|---|---|---|---|
| chr1 | 85.34 | 4,751 s | 4.35 GB | 0.385 |
| chr2 | 86.19 | 4,830 s | 3.60 GB | 0.400 |
| chr6 | 62.31 | 3,445 s | 7.40 GB | 0.410 |
| chr21 | 13.95 | 785 s | 2.97 GB | 0.336 |
| chrX | 33.19 | 1,838 s | 4.40 GB | 0.239 |
| chrY | 1.30 | 72 s | 2.03 GB | 0.025 |
| chrM | 0.62 | 35 s | 0.69 GB | n/a (16.6 kb: per-task overhead) |

The autosomes cost 0.33 to 0.52 core-seconds per sample per Mb; chrX and chrY less (less gVCF
data per Mb). chr6 is the only chromosome whose tasks exceed 4.5 GB: its 16 tasks of 200
samples peak at 6.72 to 7.40 GB (the cause was not traced). Within a chromosome the slowest
200-sample batch takes 1.04 to 1.16 times the median one.

## Aggregation

One `AGGREGATE` task sums the per-batch partials into PostgreSQL. `REGISTER` (the Iceberg
registration of the ingest files) runs before it.

| Cohort | Register | Aggregate | Threads | Aggregate peak memory | Iceberg commit conflicts |
|---|---|---|---|---|---|
| chr21 × 100 | 9.1 s | 29.2 s | 4 | 2.83 GB | 0 |
| chr21 × 3,202 | 22.8 s | 79.8 s | 4 | 3.96 GB | 0 |
| genome × 100 | 48.5 s | 334.5 s | 4 | 5.54 GB | 0 |
| chr1 × 3,202 (banded) | 35.6 s | 383.8 s | 4 | 9.98 GB | 0 |
| genome × 3,202 (banded) | 409.6 s | 3,822.5 s | 2 | 12.84 GB | 0 |

**The memory limit at genome × 3,202.** At 4 threads the aggregate was killed by the 12 GiB
limit four times, 40 to 70 s into each attempt. At 2 threads it completed in 63.7 min with a
peak of 12,840,271,872 bytes, 44.6 MB under 12 GiB. chr1 alone at 3,202 samples already took
9.98 GB at 4 threads. For a cohort of this size set the aggregate to 2 threads in a 12 GiB task
(the `aggregate_threads` parameter), or give it more memory; 1 thread was not measured at this
scale.

Where the time goes at genome × 3,202 (2 threads, from the aggregate's log):

| Step | Seconds |
|---|---|
| allele counts: reduce | 706.3 |
| site statistics: reduce | 290.8 |
| site statistics: write to PostgreSQL | 276.5 |
| reference depth at sites (position windows) | 1,931.3 |
| reference depth: write to PostgreSQL | 116.6 |
| allele catalog: write to PostgreSQL | 497.3 |
| clean-up of staging | 1.3 |
| total | 3,820.1 |

Half the time goes to summing the per-position reference depth of every batch. `mosaic index`
then took under a second and `mosaic maintain` 16.0 s at genome × 3,202.

**The PostgreSQL result.** Sizes in GB of 10^9 bytes (PostgreSQL reports them in units of
1,024, rounded; genome × 3,202's 69.8 GB is the 65 GiB it printed):

| Cohort | Sites (`site_info`) | Alleles (`alleles`) | Reference-depth sites (`ref_blocks`) | Database |
|---|---|---|---|---|
| chr21 × 100 | 259,642 | 851,134 | 251,091 | 0.15 GB |
| chr21 × 3,202 | 1,662,044 | 5,430,611 | 1,658,838 | 0.94 GB |
| genome × 100 | 18,071,827 | 59,234,070 | 17,491,612 | 10.1 GB |
| chr1 × 3,202 (banded) | 9,657,568 | 31,372,573 | 9,649,230 | 5.38 GB |
| genome × 3,202 (banded) | 126,001,716 | 408,139,676 | 125,904,051 | 69.8 GB |

At genome × 3,202 the three aggregate tables are `alleles` 35.4 GB, `site_info` 24.7 GB and
`ref_blocks` 9.4 GB. The `alleles` catalog keeps every allele any batch listed, zero counts
included. After that aggregate the RDS volume had about 15.6 GB free of its 100 GiB; the
same volume also held the four smaller cohorts' databases (16.6 GB together).

## Joint calling

### The workflow: one VCF per sample set and chromosome

`workflow/joint_call.nf` exports each chromosome's sites and alleles from PostgreSQL once, then
runs one call task per sample set and chromosome (4 cpus, 12 GiB, 4 threads unless stated, the
default ALT cap of 6). Launch wall is from launch to exit and includes the exports; core-hours
cover every task of the launch.

| Lake | Launch | Tasks | Launch wall | Largest task | Peak memory | VCF | Core-hours |
|---|---|---|---|---|---|---|---|
| chr21 × 100 | 5 sets of 20 | 5 (1 reused from an earlier launch) | 155 s | 10.7 s | 0.40 GB | 0.24 GB | 0.05 |
| chr21 × 100 | whole cohort, one task | 1 | 156 s | 25.2 s | 0.74 GB | 0.18 GB | 0.03 |
| chr21 × 3,202 | 17 sets of 200 | 17 | 346 s | 99.5 s | 1.02 GB | 23.71 GB | 1.71 |
| chr21 × 3,202 | one set of 800 | 1 | 656 s | 371.6 s | 2.06 GB | 6.45 GB | 0.43 |
| chr21 × 3,202 | whole cohort, one task | 1 | 1,986 s | 1,871.7 s | 6.63 GB | 28.85 GB | 2.09 |
| genome × 100 | 2 sets of 50 × 25 chromosomes | 50 | 606 s | 76.1 s | 1.32 GB | 14.12 GB | 2.33 |
| genome × 100 | whole cohort × 25 chromosomes | 25 | 667 s | 150.5 s | 2.09 GB | 13.02 GB | 2.27 |
| chr1 × 3,202 (banded) | 17 sets of 200 | 17 | 795 s | 343.0 s | 1.65 GB | 86.75 GB | 5.93 |
| chr1 × 3,202 (banded) | one set of 800 | 1 | 1,766 s | 1,510.4 s | 3.91 GB | 31.20 GB | 1.83 |
| chr1 × 3,202 (banded) | whole cohort, 4 threads | 1 | 536 s | killed twice, about 60 s in | over 12 GiB | n/a | 0.34 |
| chr1 × 3,202 (banded) | whole cohort, 2 threads | 1 | 13,176 s | 12,929.3 s | 8.24 GB | 173.17 GB | 14.52 |
| genome × 3,202 (banded) | 17 sets of 200 × 25 chromosomes | 425 | 3,246 s | 371.4 s | 1.90 GB | 1,111.74 GB | 80.99 |

No task failed except the 4-thread whole-cohort call of chr1. In the 425-task launch the call
tasks took 79.86 core-hours and the 25 exports the rest. The median call task ran 161 s
(medians by chromosome: chr1 326 s, chr2 332 s, chr21 63 s, chrY 12 s; the largest task, on
chr2, 371 s), with 22 tasks in flight on average.
Indexing (`bcftools index -t`) and the upload are 9.4% of the call tasks' wall. A task's local
disk need is its VCF plus the spill files (below): at most 10.1 GB for 200 samples × one
chromosome (chr2), 54.5 GB for 800 samples × chr1, 50.1 GB for the whole cohort on chr21 and
259.5 GB for the whole cohort on chr1.

**The exports' load on PostgreSQL** (genome × 3,202, two exports at a time): 25 chromosomes, a
median of 38 s each, the slowest 84 s (chr2); at most 3 PostgreSQL connections; RDS CPU at
most 46.6%, freeable memory at least 5.0 GB; 4.41 GB of export files in all. A call task reads
the export and the lake and opens no PostgreSQL connection.

### One process on one node

Whole-chromosome calls with `joint_call.py` on one node under a 14 GiB limit (11 to 12 Nov):

| Call | Threads | ALT cap | Wall | Peak memory | Output (BGZF) | Spilled |
|---|---|---|---|---|---|---|
| chr21 × 3,202, whole cohort | 4 | 6 | 30 min 19 s | 6.38 GB | 28.85 GB | 21.2 GB (4 units) |
| chr21 × 3,202, whole cohort | 1 | 6 | 78 min 57 s | 2.40 GB | 28.85 GB, byte-identical | none |
| chr21 × 3,202, whole cohort | 1 | none | 96 min 4 s | 13.8 GB (12.8 GiB) | 30.17 GB | none |
| chr21 × 3,202, first 100 samples (one ingest batch) | 4 | 6 | 58 s | 0.50 GB | 0.71 GB | 0.48 GB |
| chr21 × 3,202, 100 samples spread over all 17 batches | 4 | 6 | 2 min 35 s | 3.59 GB | 0.76 GB | 0.51 GB |
| genome × 100, whole cohort, 25 chromosomes | 4 | 6 | 29 min 27 s | 3.97 GB | 13.02 GB | none |

**Threads.** At 4 threads the whole chr21 cohort runs 2.6 times faster than at 1 thread, in
2.7 times the memory. Its 10 Mb units each take 1.19 to 1.26 times their one-thread time when
four run at once. The two outputs are byte-identical, and so is the same call run two days later
as a Batch task (6.63 GB there, a figure that includes the driver and `bcftools index`).

**The ALT cap** (`--max-alt-alleles`, default 6). A record with more ALT alleles keeps the 6
with the largest allele count; a genotype that carries a dropped allele is written `./.`. On
the whole chr21 cohort it restricted 15,271 of 1,662,044 records (0.92%) and wrote 1,000,385
genotypes `./.` (0.019%). The capped text is 61.6% of the uncapped (157.0 against 254.7 GB),
the compressed output 95.6%. Uncapped, the largest record is a 2.13 GB line (605 ALT alleles);
capped, 443,555 bytes. The uncapped run at 1 thread peaked at 12.8 GiB of the 14 GiB limit,
at the unit holding that site. In the 425-task genome × 3,202 launch the cap restricted
17,376,804 records and wrote 56,967,528 genotypes `./.` (0.0141% of the genotypes).

**Sample subsets.** A subset call writes every cohort site and reads only the data files that
can hold its samples, so its cost follows how the samples were batched: 2 of 69 site files and
10 of 317 reference-block files for 100 samples ingested together, 65 and 100 for 100 samples
spread over all batches, against 30 min 19 s for the whole cohort.

**The spill copy.** With more than one thread, a unit that finishes before the units ahead of
it buffers up to 64 MiB in memory and then spills to a file beside the output; the file is
copied into the output when the unit's turn comes. About (threads − 1) / threads of a large
output passes through spill files: the whole chr21 cohort at 4 threads spilled 21.2 of 28.8 GB
and wrote 50.1 GB to disk in all; the copying added up to 292 s on the node and 153 s in the
Batch task; the spill directory reached 25.8 GB. The whole chr1 cohort at 2 threads spilled
86.35 of 173.17 GB, with 603 s of copying. Leave free disk of about the output's size beside it.

### What memory and time follow

Memory follows the samples in the call and the units in flight. The largest task's peak at 4
threads: on chr21 × 3,202, 1.02 GB for 200 samples, 2.06 GB for 800, 6.63 GB for all 3,202; on
chr1 × 3,202 (banded), 1.65 GB for 200, 3.91 GB for 800, and the whole cohort over 12 GiB (8.24
GB at 2 threads). On the node, each 10 Mb unit in flight added about 1.3 GB at chr21 × 3,202
((6.38 − 2.40) / 3). Time follows the genotypes written (records × samples), median seconds
of task wall per billion genotypes:

| Launch | Seconds per billion genotypes |
|---|---|
| chr21 × 3,202, 17 sets of 200 | 284 |
| chr21 × 3,202, whole cohort | 352 |
| chr1 × 3,202 (banded), 17 sets of 200 | 168 |
| chr1 × 3,202 (banded), one set of 800 | 195 |
| chr1 × 3,202 (banded), whole cohort, 2 threads | 418 |
| genome × 3,202 (banded), 425 tasks | 184 |
| genome × 100, whole cohort × 25 chromosomes | 1,026 |

### Haploid calls

The joint caller writes diploid genotypes only. On chrX outside the PARs and on chrY a haploid
ALT call is written `./.` and a haploid reference block `0/0`, so the genotypes, AC, AN and AF
there are wrong for male samples. The ingested data and the PostgreSQL allele counts are not
affected. The chrX and chrY VCFs of these runs were written and counted, but sex-chromosome
joint calls are not validated.

## Storage

**Genome × 3,202 (banded at GQ 20)**, after the aggregate and `mosaic maintain`:

| Part | GB | Of the gVCFs |
|---|---|---|
| Per-sample tables (`sample_site_info` 638.9, `sample_ref_blocks` 611.9) | 1,250.8 | 9.6% |
| Per-batch partials and genotype matrix (`batch_ref_blocks` 149.2, `batch_genotypes` 69.8, `batch_site_info` 17.4, `batch_allele_counts` 5.8) | 242.2 | 1.9% |
| Aggregate copies (`agg_*`) | 8.4 | 0.06% |
| Lake, all Parquet data | 1,501.4 | 11.6% |
| Iceberg metadata (13 tables) | 0.021 | about 0 |
| PostgreSQL database | 69.8 | 0.54% |
| Source gVCFs | 12,979 | |

The lake total excludes 13.7 GB of the aggregate's staging tables (`tier_*`, kept for the next
incremental aggregate). Per sample: 0.47 GB of Parquet, of which 0.39 GB in the per-sample
tables, against 4.05 GB of gVCF. **This is not a like-for-like comparison, for two reasons:**
the lake keeps fewer fields than a gVCF (above, Setup), and its reference blocks are banded at
GQ 20. No unbanded genome × 3,202 lake exists to separate the two.

**Banding's measured effect** (chr21 × 3,202, the same samples and batches, unbanded against
the banded genome × 3,202 lake's chr21):

| Table | Unbanded | Banded at GQ 20 | Banded / unbanded |
|---|---|---|---|
| `sample_ref_blocks` | 72.11 GB, 8,570,430,600 rows | 8.19 GB, 1,235,237,130 rows | 11.4% of bytes, 14.4% of rows |
| `batch_ref_blocks` | 2.53 GB | 1.90 GB | 75.1% |
| the other four ingest tables | 10.88 GB | 10.88 GB | equal |
| the six ingest tables | 85.52 GB | 20.97 GB | 24.5% |

**All five lakes** (Parquet data without staging; PostgreSQL in GB of 10^9 bytes):

| Cohort | Lake | Lake per sample | PostgreSQL |
|---|---|---|---|
| chr21 × 100 | 3.13 GB | 31.3 MB | 0.15 GB |
| chr21 × 3,202 | 85.6 GB | 26.7 MB | 0.94 GB |
| genome × 100 | 219.4 GB | 2.19 GB | 10.1 GB |
| chr1 × 3,202 (banded) | 129.3 GB | 40.4 MB | 5.38 GB |
| genome × 3,202 (banded) | 1,501.4 GB | 0.47 GB | 69.8 GB |

**Joint-called VCFs** (BGZF): the 425 per-set VCFs of genome × 3,202 hold 1,111.7 GB, 2.76
bytes per genotype; the whole chr21 cohort at 3,202 samples is 28.85 GB (5.42 bytes per
genotype), the whole chr1 cohort 173.17 GB (5.60), the whole genome × 100 cohort 13.02 GB.

## Read-side queries at chr21 × 3,202

Measured 8 Nov 2026 on the unbanded chr21 × 3,202 cohort (median of three runs):

| Query | Reads | Time |
|---|---|---|
| Site QC table for all 1,662,044 sites (per site: QUAL, depth, the mapping-quality, strand and genotype-count sums, the reference sample count and the ALT allele count) | PostgreSQL | 7.28 s |
| Per-sample QC summary, 3,202 samples | PostgreSQL | 0.034 s |
| Global allele frequency from PostgreSQL plus a 10-sample recount from the lake (`scripts/gwas_benchmark.py --af-cutoff 0.01`) | both | 8.97 s |
| Genotype matrix, chr21:10-20 Mb (361,379 sites × 3,202 samples) | lake | 10.8 s, about 9.2 GB peak memory |

The joint calls of 100-sample subsets in the table above are read-side measurements too: 58 s
when the samples were ingested together, 2 min 35 s when they are spread over all batches.

## Correctness and reproducibility checks

What was checked, and at what scale. Every comparison below is exact unless stated.

| Check | Scale | Result |
|---|---|---|
| Aggregates of three independent runs (7 to 8, 11 to 12 and 13 to 14 Nov) | chr21 × 100, chr21 × 3,202, genome × 100 | checksums of `site_info`, `alleles` and `ref_blocks` identical (all columns but one float32 mapping-quality sum) |
| Allele counts in PostgreSQL against a recount from the lake's per-sample rows (7 to 8 Nov) | chr21 × 100; chr21 × 3,202 (254,015,841 rows); chr1 of genome × 100 (37,939,478 rows) | equal |
| A full rebuild of the aggregate (`mosaic aggregate --rebuild`) against the run's result (7 to 8 Nov) | chr21 × 100, chr21 × 3,202 | identical |
| Aggregates of two lakes of the same samples: unbanded against banded (chr21 × 3,202), and two banded lakes (chr1 × 3,202) | chr21 and chr1 at 3,202 samples | `site_info` equal except one float32 rounding step in one row (chr21) and 11 rows (chr1); `alleles`, reference sample counts and per-sample QC equal; reference depth equal on chr1 and different on chr21, as banding should |
| Records per VCF equal PostgreSQL's sites | all 425 VCFs of genome × 3,202; whole cohorts of chr21 × 3,202, chr1 × 3,202, genome × 100 | equal |
| Same output across thread counts | whole chr21 cohort at 3,202 (1 and 4 threads); chr21 × 100 (1 and 4 threads, minimal buffer) | byte-identical |
| Same output across independent runs | whole chr21 cohort at 3,202: node call (11 Nov) against Batch task (13 to 14 Nov) | byte-identical, 28,846,384,963 bytes |
| Workflow task against a direct call of the driver | 3 set/chromosome pairs of genome × 3,202; 2 sets on chr1 × 3,202; chr22 and chrM of genome × 100 | byte-identical VCF and closing lines |
| Call from the cohort export against a call from PostgreSQL | chr21 × 100 | equal |
| `bcftools +fill-tags` AC / AN and `bcftools stats` | whole cohorts: chr21 × 3,202 (1,662,044 records), genome × 100 (18,071,827), chr21 × 100 | AC and AN agree at every record; stats complete; 0 htslib warnings |
| Options against an independent projection of the uncapped VCF text (`tests/vcf_project_stream.py`): ALT cap, trim, minimum AC, subsets | every option at chr21 × 100; 11 option sets on a 1 Mb slice of chr21 × 3,202; chr22 of genome × 100 | byte-identical; `bcftools view --trim-alt-alleles` (with and without `-s`) agrees |
| Banded against unbanded joint calls | chr21 × 3,202, 17 sets of 200: 28,254,748 records, 5,321,864,888 genotypes | identical CHROM, POS, REF, ALT, QUAL, FILTER, AC, AN, AF, FS, MQ, SOR, InbreedingCoeff, GT, F1R2, F2R1; differ: INFO/DP in 99.5% of records, QD in 17.3%; PL in 84.3% of genotypes (all `0/0`), AD in 83.8% and DP in 72.9% (`0/0` and `./.`) |
| Written AC against PostgreSQL's allele counts (the caller's cross-check) | whole cohorts: chr21 × 3,202, chr1 × 3,202 | 12 of 1,646,773 compared records differ on chr21 and 150 of 9,574,648 on chr1 (capped records are not compared); 43 of chr1's and all of chr21's are a sample's own call that the record cannot express, written `./.`; chr1's other 107 are not characterized |
| Against Sentieon GVCFtyper | 2 samples, chr21:1-30M, a local test | 56,644 of 56,772 records identical comparing FORMAT GT, PL, DP and INFO DP, AC, AF, AN; 128 differ; none missing or extra |

**Where the evidence stops.** Genotype accuracy against another caller exists only for 2
samples; no truth set exists above 5 samples, so genotype accuracy at cohort scale is not
measured. The banding comparison shows that banding at GQ 20 changes the depth-derived fields
of hom-ref and no-call fills and none of the fields the calls are made from; how far INFO/DP
moves was not measured at this scale. On the whole genome × 100 cohort (the node call) the
cross-check reports, beside 248 records of the inexpressible-call kind, 416,721 records whose
written AC differs from the catalog: 399,172 on chrX and 17,434 on chrY (the haploid
limitation), 103 on chr1 and 12 on chrM (not characterized).

## Adding and removing samples

Sample removal (`mosaic remove`) is an initial version and is not part of the Nextflow
workflow. Adding samples re-reads no existing gVCF: the aggregate folds only the new batches'
partials onto the previous result.

**In the cloud, chr21 × 100 (11 Nov 2026).** `mosaic remove` of 2 of the 100 samples took
30.7 s. A joint call before the next aggregate is refused (the cohort is stale). The
incremental aggregate that applies the removal took 30.0 s (a full aggregate of this cohort:
30 s) and dropped 1,194 sites and 4,106 alleles that only the removed samples' records
held. The joint call
of the 98 samples afterwards has 258,448 records and passes `bcftools +fill-tags` and `stats`;
an 8-sample subset of it equals its projection. `mosaic maintain` then reclaimed the 379 MB of
replaced files.

**Locally, six chr21 samples in three batches (small scale, not a benchmark).** The
integration tests (`tests/integration/`) check that removing samples and aggregating gives the
PostgreSQL contents of a fresh ingest of the remaining samples (integer columns exact, float32
columns within one rounding step) and a byte-identical joint-called VCF, and that an
incremental aggregate after adding a batch equals a full rebuild over all batches (the same
tolerance).

Removal and incremental addition above 100 samples were not measured.

## Cost estimates

No run recorded dollar costs. The estimates below are core-hours and GB times AWS list prices
for us-east-1, looked up 2026-11-15, on-demand. Substitute your own rates.

| Item | List price | As used here |
|---|---|---|
| EC2 m8a.xlarge (4 vCPU), Linux | $0.2434 per hour | one 4-cpu task per instance: $0.06085 per core-hour |
| gp3 volume | $0.08 per GB-month | the instance volumes (below) |
| RDS `db.m8gd.large`, PostgreSQL, Single-AZ | $0.1976 per hour | the database instance |
| RDS gp3 storage | $0.115 per GB-month | the database volume |
| S3 Standard | $0.023 per GB-month | the lake, the VCFs, the source gVCFs |

**Compute for genome × 3,202** (banded):

| Stage | Core-hours | Compute (estimate) |
|---|---|---|
| Ingest (425 tasks) | 1,048.71 | $63.81 ($0.020 per genome) |
| Register | 0.46 | $0.03 |
| Aggregate, including the four killed attempts | 4.49 | $0.27 |
| Index and maintain | 0.02 | under $0.01 |
| Joint call, 425 VCFs | 80.99 | $4.93 |
| Total | 1,134.67 | $69.04 |

Ingest compute for the other cohorts: chr21 × 100 $0.03, chr21 × 3,202 $1.01, genome × 100
$2.23, chr1 × 3,202 $5.26. The database instance for the 11.4 h of active run time (ingest
span, register, aggregate with its failed attempts and the joint-call launch, idle gaps
excluded) is $2.26, plus $144 per month if it runs continuously and $11.50 per month for a
100 GiB volume. **Storage per month:** the genome × 3,202 lake $34.53 in S3, the 425 VCFs
$25.57, against $299 for the source gVCFs.

Compute and storage only: instance boot and idle time, data transfer, Glue, CloudWatch and S3
requests are not counted. **Instance volumes:** the runs attached a 1000 GB gp3 volume to every
instance ($0.110 per instance-hour, 45% on top of the instance price; $28.73 over the ingest's
262 instance-hours), far more than any task needs. Measured local disk per kind of task: an
ingest task of 200 samples on the largest chromosome downloads 54.4 GB of gVCF slices (its
local Parquet files were not measured); a joint-call task needs its VCF plus spill files, at
most 10.1 GB for 200 samples × one chromosome, 54.5 GB for 800 samples × chr1, 50.1 GB for the
whole chr21 cohort and 259.5 GB for the whole chr1 cohort.

## Limits and what was not measured

- **Nothing above 3,202 samples.** The 1000 Genomes cohort is the largest data set used.
- **Aggregate memory grows with the cohort.** At genome × 3,202 it fit a 12 GiB task only at 2
  threads, 44.6 MB under the limit; 4 threads were killed. 1 thread was not measured at this
  scale. A larger cohort needs fewer threads or more memory.
- **Whole-cohort joint calls.** The whole-cohort genome call at 3,202 samples was not run. The
  whole cohort on chr1 needs 2 threads in a 12 GiB task (8.24 GB, 3.6 h).
- **Haploid calls** on chrX outside the PARs and on chrY are written as diploid (see Joint
  calling); sex-chromosome joint calls are not validated.
- **No genotype accuracy check at cohort scale.** The only comparison with another caller is
  the 2-sample chr21 test.
- **The joint-call workflow on AWS Batch** has run once at each scale, on 13 to 14 Nov 2026.
- **Sample removal** was measured at chr21 × 100 in the cloud and on six samples locally only.
- **Banding** was measured at GQ 20 only; its effect on INFO/DP at 3,202 samples was counted
  (which records differ) but not sized.

See the [README](../README.md) for the commands and options, and
[ARCHITECTURE.md](ARCHITECTURE.md) for the design.
