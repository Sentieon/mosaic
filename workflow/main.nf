// Copyright (c) 2026 Sentieon Inc. All rights reserved
// Nextflow ingestion pipeline for MOSAIC.
//
// Fans a sample-name list out into batches of `params.batch_size`, takes
// the cross-product with the chromosomes in `params.region`, and runs
// `bin/fetch_and_ingest.py` on each (chromosome × batch) pair. That
// helper downloads the gVCFs from s3://1000genomes-dragen-v4-2-7,
// strips unused annotations via bcftools, and invokes `mosaic ingest`.
//
// The per-chromosome fan-out is what keeps individual tasks small enough
// to finish on a single spot instance.
//
// With `params.defer_register` (the default) each INGEST_BATCH task uploads
// its Parquet and records the file list in PostgreSQL (`mosaic ingest
// --defer-register`, batch status 'uploaded') instead of committing to the
// six shared Iceberg tables. After ALL INGEST_BATCH tasks finish, REGISTER
// runs `mosaic register` once: one committer per table, many files per
// commit (`params.register_files_per_commit`), which completes the batches.
// Then `mosaic aggregate` runs once — reducing the per-batch Iceberg partials
// and finalizing ref_blocks/site_info/allele_counts to PostgreSQL via a
// tier-tree merge whose fan-in (batches merged per tier) is `params.fan_in` —
// then, both after AGGREGATE and independent of each other, `mosaic index`
// (INDEX) and `mosaic maintain` (MAINTAIN, unless `params.maintain` is
// false): Iceberg maintenance, which rewrites each table's manifests,
// expires the old snapshots and deletes the files only they needed, among
// them the staging the aggregate's completion just pruned. The flow is
// ingest -> register -> aggregate -> {index, maintain}.
// `mosaic init` is the one exception that stays a manual one-shot command run
// before the pipeline: it needs a reference gVCF header and the desired
// chromosome scope.
//
// Usage:
//   nextflow run workflow/main.nf -profile local   --samples samples.txt --fan_in 100
//   nextflow run workflow/main.nf -profile awsbatch --samples s3://.../samples.txt \
//       --region chr11,chr12,...,chrY --fan_in 100
//
// See README.md for full parameter docs.

nextflow.enable.dsl = 2


process INGEST_BATCH {
    tag "${batch_id}/${chrom ?: 'wg'}/${samples.size()}samples"

    publishDir "${params.log_dir}", mode: 'copy', pattern: '*.log'

    input:
        tuple val(batch_id), val(chrom), val(samples)

    output:
        tuple val(batch_id),
              path("batch_${batch_id}.done"),
              path("batch_${batch_id}.log")

    script:
        def region_arg  = chrom                 ? "--region ${chrom}" : ''
        def threads_arg = params.ingest_threads ? "--threads ${params.ingest_threads}" \
                                                : "--threads ${task.cpus}"
        // Deferred registration: the batch ends 'uploaded'; REGISTER
        // registers its files and completes it.
        def defer_arg   = params.defer_register ? '--defer-register' : ''
        // Ref-block banding: passed through only when set (default off).
        def bands_arg   = params.ref_gq_bands != null \
                              ? "--ref-gq-bands '${params.ref_gq_bands}'" : ''
        def sample_args = samples.join(' ')
        """
        set -euo pipefail

        fetch_and_ingest.py \\
            --batch-id ${batch_id} \\
            --workdir . \\
            --shard-size ${params.shard_size} \\
            --pos-batch-size ${params.pos_batch_size} \\
            ${threads_arg} ${region_arg} ${defer_arg} ${bands_arg} \\
            ${sample_args} 2>&1 | tee batch_${batch_id}.log

        touch batch_${batch_id}.done
        """
}


process REGISTER {
    tag "files_per_commit=${params.register_files_per_commit}"

    publishDir "${params.log_dir}", mode: 'copy', pattern: '*.log'

    input:
        // The collected batch_*.done markers from every INGEST_BATCH task.
        // Used purely as an ordering barrier — register reads the pending
        // file list from PostgreSQL (ingest_files), not from these files.
        path done_markers

    output:
        path 'register.done', emit: done
        path 'register.log'

    script:
        // With params.defer_register = false the ingests registered their
        // own files: `mosaic register` then reports nothing to register.
        """
        set -euo pipefail

        # with_db_password.py populates MOSAIC_DB_PASSWORD from Secrets Manager
        # on AWS (no-op locally), then execs the cloud-agnostic mosaic command.
        with_db_password.py python -m mosaic.cli register \\
            --files-per-commit ${params.register_files_per_commit} 2>&1 | tee register.log

        touch register.done
        """
}


process AGGREGATE {
    tag "fan_in=${params.fan_in}"

    publishDir "${params.log_dir}", mode: 'copy', pattern: '*.log'

    input:
        // The register.done marker — forces AGGREGATE to run strictly after
        // REGISTER has completed the ingest batches (aggregate reads only
        // 'complete' batches; an 'uploaded' one would be skipped).
        path reg_done

    output:
        path 'aggregate.done', emit: done
        path 'aggregate.log'

    script:
        // Aggregate reduce concurrency: params.aggregate_threads (if set)
        // else the task's cpu count. Lower it to cap the ref_blocks reduce's
        // memory multiplier without shrinking --ref-shard-size.
        def agg_threads = params.aggregate_threads ?: task.cpus
        """
        set -euo pipefail

        # with_db_password.py populates MOSAIC_DB_PASSWORD from Secrets Manager
        # on AWS (no-op locally), then execs the cloud-agnostic mosaic command.
        with_db_password.py python -m mosaic.cli aggregate \\
            --fan-in ${params.fan_in} \\
            --threads ${agg_threads} \\
            --ref-shard-size ${params.ref_shard_size} 2>&1 | tee aggregate.log

        touch aggregate.done
        """
}


process MAINTAIN {
    tag "older_than_hours=${params.maintain_older_than_hours}"

    publishDir "${params.log_dir}", mode: 'copy', pattern: '*.log'

    input:
        // The aggregate.done marker — forces MAINTAIN to run strictly after
        // AGGREGATE (whose completion pruned the previous runs' staging).
        // INDEX takes the same marker; neither waits for the other.
        path agg_done

    output:
        path 'maintain.done'
        path 'maintain.log'

    script:
        // params.maintain_older_than_hours = 0 retains each table's current
        // snapshot only: nothing reads an older one while the pipeline runs.
        def orphan_args = params.maintain_orphans \
            ? "--orphans --orphan-grace-hours ${params.maintain_orphan_grace_hours}" \
            : ''
        """
        set -euo pipefail

        # with_db_password.py populates MOSAIC_DB_PASSWORD from Secrets Manager
        # on AWS (no-op locally), then execs the cloud-agnostic mosaic command.
        with_db_password.py python -m mosaic.cli maintain \\
            --older-than-hours ${params.maintain_older_than_hours} ${orphan_args} 2>&1 | tee maintain.log

        touch maintain.done
        """
}


process INDEX {
    tag 'index'

    publishDir "${params.log_dir}", mode: 'copy', pattern: '*.log'

    input:
        // The aggregate.done marker — forces INDEX to run strictly after
        // AGGREGATE has finalized the PostgreSQL aggregate tables.
        path agg_done

    output:
        path 'index.done'
        path 'index.log'

    script:
        """
        set -euo pipefail

        # with_db_password.py populates MOSAIC_DB_PASSWORD from Secrets Manager
        # on AWS (no-op locally), then execs the cloud-agnostic mosaic command.
        with_db_password.py python -m mosaic.cli index 2>&1 | tee index.log

        touch index.done
        """
}


workflow {
    // `--ingest_engine` selected the earlier Python ingest worker, now
    // removed; Nextflow accepts an
    // unknown parameter silently, so refuse it here.
    if (params.containsKey('ingest_engine')) {
        error "--ingest_engine is no longer a parameter: the Python ingest " +
              "worker is removed and mosaic ingest always runs the Rust " +
              "worker (mosaic-ingest-worker). Drop the parameter."
    }
    if (!params.samples) {
        error "Missing required parameter: --samples <path-to-sample-list>"
    }

    // (batch_hash, samples) — deterministic hash derived from the
    // sample list so Nextflow `-resume` hits the cache on rerun.
    def sample_batches = Channel
        .fromPath(params.samples)
        .splitText()
        .map { it.trim() }
        .filter { it && !it.startsWith('#') }
        .collate(params.batch_size)
        .map { batch ->
            tuple(batch.join('|').hashCode().abs().toString(), batch)
        }

    // Build the per-task channel as (batch_id, chrom, samples).
    //  - If `params.region` is set, cross-product chromosomes × sample
    //    batches and encode the chromosome into batch_id. Both the
    //    Nextflow cache key and the downstream DB checkpoint row in
    //    `ingestion_progress` will then treat (samples, chr11) and
    //    (samples, chr22) as distinct work.
    //  - If `params.region` is unset, emit one task per batch with
    //    chrom='' (whole-genome run; original single-task-per-batch
    //    behavior). Empty string instead of null because nulls in
    //    Nextflow channels and process inputs are fragile.
    def tasks
    if (params.region) {
        def chroms = Channel.fromList(
            params.region.split(',').collect { it.trim() }
        )
        tasks = chroms
            .combine(sample_batches)
            .map { chrom, batch_hash, samples ->
                tuple("${batch_hash}_${chrom}", chrom, samples)
            }
    } else {
        tasks = sample_batches
            .map { batch_hash, samples ->
                tuple(batch_hash, '', samples)
            }
    }

    if (params.fan_in < 2) {
        error "params.fan_in must be >= 2 (got ${params.fan_in})"
    }

    if (params.register_files_per_commit < 1) {
        error "params.register_files_per_commit must be >= 1 (got ${params.register_files_per_commit})"
    }

    if (params.maintain_older_than_hours < 0 || params.maintain_orphan_grace_hours < 0) {
        error "params.maintain_older_than_hours and params.maintain_orphan_grace_hours must be >= 0"
    }

    def ingested = INGEST_BATCH(tasks)

    // Barrier: collect the .done marker from EVERY ingest task into one list,
    // so REGISTER runs exactly once, after all ingests finish. A process with
    // a single collect()ed input does not execute until its upstream channel
    // closes — which only happens after the last INGEST_BATCH task emits.
    def reg = REGISTER(ingested.map { batch_id, done, logf -> done }.collect())

    // AGGREGATE reduces the batches REGISTER completed.
    def agg = AGGREGATE(reg.done)

    // index builds the PostgreSQL indexes on the aggregate tables that
    // AGGREGATE just finalized (TRUNCATE+COPY), so it must run after it.
    INDEX(agg.done)

    // Iceberg maintenance, also after AGGREGATE and independent of INDEX
    // (both consume the same marker; neither waits for the other).
    if (params.maintain) {
        MAINTAIN(agg.done)
    }
}


workflow.onComplete {
    def status = workflow.success ? 'SUCCESS' : 'FAILURE'
    log.info "Pipeline finished: ${status}  duration=${workflow.duration}"
}
