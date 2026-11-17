// Copyright (c) 2026 Sentieon Inc. All rights reserved
// Nextflow joint-call workflow for MOSAIC.
//
// Joint-calls an ingested, aggregated cohort as one VCF per sample set and
// chromosome, after the ingestion workflow (main.nf) has finished. It shares
// nextflow.config, the two profiles and bin/ with main.nf. PostgreSQL is read
// by the first three steps only:
//
//   COHORT_STATE   `joint_call.py --cohort-state` over the whole sample list
//                  (the gate and the names), every run (cache false). Its
//                  one-line state is the value that keys the cache of every
//                  task after it: -resume reuses the exports and the calls of
//                  an unchanged cohort and redoes everything after a new
//                  aggregate.
//   EXPORT_COHORT  `--export-cohort`: cohort.json (the generation, the
//                  contigs, the samples).
//   EXPORT_CONTIG  `--export-contig` per chromosome with sites (the contig's
//                  sites and alleles, the whole contig), at most
//                  params.joint_call_export_forks at a time: the only load on
//                  PostgreSQL.
//   JOINT_CALL     one set x one chromosome, `--from-export` (no PostgreSQL
//                  connection), then `bcftools index -t` with a record-count
//                  check, the outputs published and a summary.json last.
//   JOINT_CALL_SUMMARY  joint_call_summary.tsv, one row per set x chromosome.
//
// The sets: --samples cut into sets of --batch_size (set_0001, set_0002, ...,
// the last one short; this run's own value, not tied to the ingest batches),
// or without --samples one set, `cohort`, of every active sample. The
// chromosomes: those of cohort.json with sites, narrowed by --region a,b,c.
// Every task runs workflow/bin/joint_call_task.py, which writes everything
// under <joint_call_out>/gen<agg_run_id>/ (a local directory or s3://).
//
// Nothing checks the cohort at the end of a run: do not run `mosaic remove`,
// an ingest or an aggregate while a joint-call run is in flight.
//
// Usage:
//   nextflow run workflow/joint_call.nf -profile local --samples samples.txt \
//       --batch_size 200 --region chr21
//   nextflow run workflow/joint_call.nf -profile awsbatch \
//       --samples s3://.../samples.txt --batch_size 200
//
// See README "Running via Nextflow" for the parameters.

nextflow.enable.dsl = 2


// A string as one single-quoted shell word.
def shq(s) {
    "'" + s.toString().replace("'", "'\\''") + "'"
}

// A chromosome's name in a file name: as is when plain, else c<index>
// (joint_call_task.py's chrom_file_name).
def chromTag(name, idx) {
    def n = name.toString()
    (n ==~ /[A-Za-z0-9._-]+/ && n != '.' && n != '..') ? n : "c${idx}"
}

// params.joint_call_out as the helper's --out: s3:// as is, a local path
// made absolute (against the launch directory). The process scripts call it
// as outPrefix(params.joint_call_out): a params reference in a script is part
// of the task's cache key, a value a function reads by itself is not, and a
// new joint_call_out must re-run (and so re-publish) every task.
def outPrefix(value) {
    def o = value
    if (o == null || o instanceof Boolean || !o.toString()) {
        error "--joint_call_out needs a value (a local directory or s3://bucket/prefix)"
    }
    o = o.toString()
    if (o.startsWith('s3://')) {
        return o
    }
    if (o.contains('://')) {
        error "--joint_call_out ${o}: a local directory or s3://bucket/prefix only"
    }
    return file(o).toAbsolutePath().normalize().toString()
}

// A heredoc writing names, one per line, to path (no argument-length limit).
def namesHeredoc(names, path) {
    "cat > ${path} <<'MOSAIC_SAMPLE_NAMES'\n${names.join('\n')}\nMOSAIC_SAMPLE_NAMES"
}

// The sample list, read as main.nf reads it: lines trimmed, blank lines and
// lines starting with # skipped.
def readSampleList(path) {
    file(path.toString(), checkIfExists: true)
        .readLines()
        .collect { it.trim() }
        .findAll { it && !it.startsWith('#') }
}

// --region a,b,c: whole chromosomes only (null without --region).
def parseRegions(region) {
    if (region == null || region == false || region == '') {
        return null
    }
    if (region == true) {
        error "--region needs a value: chromosome names, comma-separated"
    }
    def entries = region.toString().split(',', -1).collect { it.trim() }
    if (entries.any { !it }) {
        error "--region ${region}: an empty entry"
    }
    def ranges = entries.findAll { it ==~ /.+:[0-9]+-[0-9]+/ }
    if (ranges) {
        error "--region: ${ranges.join(', ')} is a range; the joint-call " +
              "workflow calls whole chromosomes (one VCF per set and chromosome)"
    }
    return entries.unique()
}

// An integer parameter that must be >= min when it is set.
def checkInt(name, min, required) {
    def v = params[name]
    if (v == null && !required) {
        return
    }
    if (!(v instanceof Integer) || v < min) {
        error "--${name} must be an integer >= ${min} (got ${v})"
    }
}

// The chromosomes to call from cohort.json: (index, name) of each contig
// with sites, narrowed by --region; a --region entry that is not a contig of
// the cohort stops the run here, before any contig export.
def selectContigs(cohort_json, regions) {
    def doc = new groovy.json.JsonSlurper().parseText(cohort_json.text)
    def byName = doc.contigs.collectEntries { [(it.name): it] }
    if (regions) {
        def unknown = regions.findAll { !byName.containsKey(it) }
        if (unknown) {
            // Raised inside an operator, where Nextflow reports only the
            // exception's type: the message is logged first.
            def msg = "--region: ${unknown.join(', ')} is not a contig of the " +
                      "cohort (cohort.json holds ${doc.contigs*.name.join(', ')})"
            log.error msg
            error msg
        }
    }
    def chosen = regions ? regions.collect { byName[it] } : doc.contigs
    def out = []
    chosen.each { c ->
        if (c.has_sites) {
            out << tuple(c.index as Integer, c.name.toString())
        } else {
            log.info "${c.name}: no sites in the cohort; no export and no call"
        }
    }
    if (!out) {
        log.info "No chromosome with sites to call"
    }
    return out
}


process COHORT_STATE {
    label 'jc_driver'
    tag "${names ? names.size() + ' samples' : 'cohort'}"
    // Every run: its text keys the cache of the tasks after it.
    cache false

    publishDir "${params.log_dir}", mode: 'copy', pattern: '*.log'

    input:
        val names

    output:
        path 'state.txt', emit: state
        path 'cohort_state.log'

    script:
        def write_names = names ? namesHeredoc(names, 'cohort_state.samples.txt') : ''
        def samples_arg = names ? '--samples-file cohort_state.samples.txt' : ''
        """
        set -euo pipefail
${write_names}
        # The state line is the helper's stdout; its stderr is the log.
        joint_call_task.py state ${samples_arg} 2>&1 >state.txt | tee cohort_state.log
        """
}


process EXPORT_COHORT {
    label 'jc_driver'
    tag 'cohort.json'

    publishDir "${params.log_dir}", mode: 'copy', pattern: '*.log'

    input:
        // The cohort state's text: unused by the script, it is the cache
        // key (as a value, not a path: a fresh file would defeat -resume).
        val state

    output:
        path 'export/cohort.json', emit: cohort
        path 'export_cohort.log'

    script:
        """
        set -euo pipefail

        joint_call_task.py export-cohort --dir export --out ${shq(outPrefix(params.joint_call_out))} \\
            2>&1 | tee export_cohort.log
        """
}


process EXPORT_CONTIG {
    label 'jc_driver'
    label 'jc_export_contig'
    tag "${chrom}"

    publishDir "${params.log_dir}", mode: 'copy', pattern: '*.log'

    input:
        tuple val(idx), val(chrom), path(cohort_json)

    output:
        tuple val(idx), val(chrom), path("export_c${idx}"), emit: export
        path 'export_*.log'

    script:
        // The contig's own directory: cohort.json and its three files.
        """
        set -euo pipefail

        mkdir -p export_c${idx}
        cp -L ${cohort_json} export_c${idx}/cohort.json
        joint_call_task.py export-contig --dir export_c${idx} \\
            --contig ${shq(chrom)} --out ${shq(outPrefix(params.joint_call_out))} \\
            2>&1 | tee export_${chromTag(chrom, idx)}.log
        """
}


process JOINT_CALL {
    label 'jc_driver'
    label 'jc_call'
    tag "${set_name}/${chrom}"

    publishDir "${params.log_dir}", mode: 'copy', pattern: '*.log'

    input:
        // The names as a value (the cache keys on them); [] for the
        // whole-cohort set, called without a names file.
        tuple val(set_name), val(names), val(idx), val(chrom), path(export_dir)

    output:
        path '*.summary.json', emit: summary
        path 'joint_call_*.log'

    script:
        def threads = params.joint_call_threads ?: task.cpus
        def write_names = names ? namesHeredoc(names, "${set_name}.samples.txt") : ''
        def opts = []
        if (names) {
            opts << "--samples-file ${set_name}.samples.txt"
        }
        if (params.joint_call_max_alt_alleles != null) {
            opts << "--max-alt-alleles ${params.joint_call_max_alt_alleles}"
        }
        if (params.joint_call_trim_alt_alleles) {
            opts << '--trim-alt-alleles'
        }
        if (params.joint_call_min_ac != null) {
            opts << "--min-ac ${params.joint_call_min_ac}"
        }
        if (params.joint_call_window != null) {
            opts << "--window ${params.joint_call_window}"
        }
        if (!params.joint_call_index) {
            opts << '--no-index'
        }
        """
        set -euo pipefail
${write_names}
        joint_call_task.py call --set ${set_name} \\
            --contig ${shq(chrom)} --contig-index ${idx} \\
            --export ${export_dir} --out ${shq(outPrefix(params.joint_call_out))} \\
            --threads ${threads} ${opts.join(' ')} \\
            2>&1 | tee joint_call_${set_name}.${chromTag(chrom, idx)}.log
        """
}


process JOINT_CALL_SUMMARY {
    tag 'joint_call_summary.tsv'

    publishDir "${params.log_dir}", mode: 'copy', pattern: '*.log'

    input:
        path summaries

    output:
        path 'joint_call_summary.tsv'
        path 'joint_call_summary.log'

    script:
        """
        set -euo pipefail

        joint_call_task.py summarize --out ${shq(outPrefix(params.joint_call_out))} ${summaries} \\
            2>&1 | tee joint_call_summary.log
        """
}


workflow {
    // Refusals before any task.
    checkInt('batch_size', 1, true)
    checkInt('joint_call_max_alt_alleles', 0, false)
    checkInt('joint_call_min_ac', 0, false)
    def regions = parseRegions(params.region)
    def out = outPrefix(params.joint_call_out)

    // The sets: (name, names). Without --samples one set, `cohort`, of
    // every active sample ([]: called without a names file).
    def all_names = []
    def sets
    if (params.samples) {
        all_names = readSampleList(params.samples)
        if (!all_names) {
            error "--samples ${params.samples}: the list holds no sample name"
        }
        sets = all_names.collate(params.batch_size as Integer)
            .withIndex()
            .collect { batch, i -> tuple(String.format('set_%04d', i + 1), batch) }
    } else {
        sets = [tuple('cohort', [])]
    }
    log.info "Joint-calling ${params.samples ? all_names.size() + ' samples in ' + sets.size() + ' set(s) of up to ' + params.batch_size : 'the whole cohort as one set'}" +
             " over ${regions ? regions.join(', ') : 'every chromosome with sites'}; outputs under ${out.replaceAll('/+$', '')}/gen<agg_run_id>/"

    def state = COHORT_STATE(Channel.value(all_names)).state
        .map { it.text.trim() }
    def cohort = EXPORT_COHORT(state).cohort
    def contigs = cohort.flatMap { selectContigs(it, regions) }
    def exports = EXPORT_CONTIG(contigs.combine(cohort)).export
    def calls = JOINT_CALL(Channel.fromList(sets).combine(exports))
    JOINT_CALL_SUMMARY(calls.summary.collect())
}


workflow.onComplete {
    def status = workflow.success ? 'SUCCESS' : 'FAILURE'
    log.info "Pipeline finished: ${status}  duration=${workflow.duration}"
}
