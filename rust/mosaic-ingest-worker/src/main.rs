// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! `mosaic-ingest-worker`: `mosaic ingest`'s shard worker. `--manifest`
//! alone runs the batch's shards on `--threads` threads and prints one
//! JSON result line per shard on stdout, then, when the manifest has a
//! `combine` object, combines the local files into the data files the
//! driver uploads (the combine in the worker); the `--dump-groups` /
//! `--count-records` diagnostic modes and the Parquet spike stay.

mod banding;
mod combine;
mod dump;
mod failhook;
mod grouper;
mod header;
mod localfiles;
mod manifest;
mod matrix;
mod partials;
mod pqwrite;
mod pyops;
mod qc;
mod reader;
mod record;
mod ref_blocks;
mod shard;
mod sites;
mod slices;
mod spike;
mod value;
mod windows;

use std::io::{BufWriter, Write};
use std::ops::ControlFlow;
use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use clap::Parser;

use crate::dump::ShardRange;
use crate::reader::Input;

/// The binary's one test seam (`failhook::parse_fail_hook`: a shard or, R5,
/// a combined file).
const FAIL_HOOK_ENV: &str = "MOSAIC_TEST_FAIL_RUST_SHARD";

#[derive(Parser, Debug)]
#[command(version, about = "MOSAIC ingest shard worker (Rust engine)")]
struct Cli {
    /// The batch manifest; its inputs and shards.
    #[arg(long)]
    manifest: Option<PathBuf>,
    /// Worker threads: shards run concurrently, one per thread (the
    /// `--dump-groups` / `--count-records` modes run one thread).
    #[arg(long, default_value_t = 1)]
    threads: usize,
    /// Inputs in stream order, instead of a manifest.
    #[arg(long, num_args = 1..)]
    inputs: Vec<PathBuf>,
    /// One shard, `chrom:start-end` (0-based start, exclusive end).
    #[arg(long)]
    region: Option<String>,
    /// Cut these chromosomes (length from the first input's header) into
    /// `--shard-size` shards, as `Sharder.cut` does for one contig.
    #[arg(long)]
    chrom: Vec<String>,
    #[arg(long)]
    shard_size: Option<i64>,
    /// Print one JSON line per grouper yield.
    #[arg(long)]
    dump_groups: bool,
    /// Read the shards' records with no grouper; print the count.
    #[arg(long)]
    count_records: bool,
    /// Write output here instead of stdout.
    #[arg(long, short)]
    output: Option<PathBuf>,
    /// Write the Parquet round-trip spike files into this directory.
    #[arg(long)]
    spike: Option<PathBuf>,
    /// Worker mode: print each shard's seconds per bucket on stderr.
    #[arg(long)]
    timing: bool,
}

/// Worker mode (`--manifest` alone): run the shards on `threads` threads
/// (`queue::run_queue`: taken in manifest order, each thread with its own
/// reader per input, re-seeked per shard; the inputs' headers and indexes
/// loaded once and shared) and print one JSON line per finished shard on
/// stdout, in completion order, from this thread only, each written whole
/// and flushed after the shard's local files are closed. A shard's error
/// (or panic) ends the process at once: the message on stderr, exit 1.
///
/// With a `combine` object (R5) the worker then combines the local files
/// on `threads` threads (`combine::run_combine`): one `file` line per
/// combined file as it closes, then the `combined` summary line. The
/// driver acknowledges each uploaded file with its path on a stdin line
/// (`combine::Gate`); stdin's end before the summary line ends the process
/// (the driver is gone): exit 1.
fn run_worker(path: &std::path::Path, threads: usize, timing: bool) -> Result<()> {
    let m = manifest::Manifest::load(path)?;
    let hook = failhook::parse_fail_hook(std::env::var(FAIL_HOOK_ENV).ok().as_deref())?;
    let fail = match &hook {
        Some(failhook::FailHook::Shard(c, s)) => Some((c.clone(), *s)),
        _ => None,
    };
    let fail_combine = match hook {
        Some(failhook::FailHook::Combine(t, i)) => Some((t, i)),
        _ => None,
    };
    // The acknowledgements: read from the start, so a driver that dies
    // during the shards ends the worker too.
    let gate = Arc::new(combine::Gate::new(
        m.combine.as_ref().map_or(1, |c| c.pending_files),
    ));
    let summary_written = Arc::new(AtomicBool::new(false));
    if m.combine.is_some() {
        let (g, done) = (gate.clone(), summary_written.clone());
        std::thread::spawn(move || {
            let e = combine::read_acks(std::io::stdin().lock(), &g);
            if let Err(e) = e
                && !done.load(Ordering::SeqCst)
            {
                eprintln!("Error: {e:?}");
                std::process::exit(1);
            }
        });
    }
    let batch = shard::Batch::from_manifest(&m)?;
    let inputs: Vec<Arc<Input>> = m
        .inputs
        .iter()
        .map(|i| Input::open(std::path::Path::new(&i.path)))
        .collect::<Result<_>>()?;
    let stdout = std::io::stdout();
    let mut tot = shard::Timing::default();
    let mut lines: Vec<Option<serde_json::Value>> = vec![None; m.shards.len()];
    let mut write_err: Option<anyhow::Error> = None;
    let work = |readers: &mut Option<Vec<Option<reader::ShardReader>>>,
                i: usize|
     -> Result<shard::ShardResult> {
        let s = &m.shards[i];
        let ctx = || format!("shard {}:{}-{}", s.chrom, s.start, s.end);
        if fail
            .as_ref()
            .is_some_and(|(c, st): &(String, i64)| *c == s.chrom && *st == s.start)
        {
            return Err(anyhow::anyhow!(
                "{FAIL_HOOK_ENV}: injected failure of shard {}:{}",
                s.chrom,
                s.start
            ))
            .with_context(ctx);
        }
        let rs = match readers.take() {
            Some(rs) => rs,
            None => inputs
                .iter()
                .map(|i| reader::ShardReader::new(i.clone()).map(Some))
                .collect::<Result<_>>()
                .with_context(ctx)?,
        };
        let (res, back) = shard::run_shard(&batch, rs, s).with_context(ctx)?;
        *readers = Some(back);
        Ok(res)
    };
    let flow = mosaic_core::queue::run_queue(m.shards.len(), threads, work, |i, r| {
        let res = match r {
            Ok(res) => res,
            Err(e) => {
                // Exit now: the other threads' shards are abandoned (the
                // batch fails; the Python side keeps shard_tmpdir).
                eprintln!("Error: {e:?}");
                std::process::exit(1);
            }
        };
        let s = &m.shards[i];
        let written = (|| -> Result<()> {
            let mut line = serde_json::to_vec(&res.json)?;
            line.push(b'\n');
            let mut out = stdout.lock();
            out.write_all(&line)?;
            out.flush()?;
            Ok(())
        })();
        if let Err(e) = written {
            write_err = Some(e.context("write a result line to stdout"));
            return ControlFlow::Break(());
        }
        if m.combine.is_some() {
            lines[i] = Some(res.json);
        }
        if timing {
            let t = res.timing;
            eprintln!(
                "  timing shard {}:{}-{}: total {:.3}s read+group {:.3}s logic {:.3}s \
                 dense-brb-write {:.3}s other-write {:.3}s",
                s.chrom,
                s.start,
                s.end,
                t.total,
                t.read_group,
                t.logic(),
                t.dense_write,
                t.other_write
            );
            tot.total += t.total;
            tot.read_group += t.read_group;
            tot.dense_write += t.dense_write;
            tot.other_write += t.other_write;
        }
        ControlFlow::Continue(())
    });
    if let Some(e) = write_err {
        return Err(e);
    }
    debug_assert!(flow.is_continue());
    if timing {
        eprintln!(
            "  timing total: {} shard(s) {:.3}s read+group {:.3}s logic {:.3}s \
             dense-brb-write {:.3}s other-write {:.3}s",
            m.shards.len(),
            tot.total,
            tot.read_group,
            tot.logic(),
            tot.dense_write,
            tot.other_write
        );
    }
    if let Some(spec) = &m.combine {
        run_combine_phase(&m, spec, &lines, threads, fail_combine, &gate)?;
        summary_written.store(true, Ordering::SeqCst);
    }
    Ok(())
}

/// The combine phase (R5): the local paths and the cross-shard spans of
/// the shards' results, in shard order, combined; each line written whole
/// and flushed by this (the main) thread.
fn run_combine_phase(
    m: &manifest::Manifest,
    spec: &manifest::CombineSpec,
    lines: &[Option<serde_json::Value>],
    threads: usize,
    fail: Option<(String, usize)>,
    gate: &combine::Gate,
) -> Result<()> {
    let mut inputs = combine::Inputs::default();
    let mut foreign = combine::Foreign::default();
    for (i, l) in lines.iter().enumerate() {
        let l = l
            .as_ref()
            .with_context(|| format!("no result of shard {i}"))?;
        for (k, t) in combine::Table::ALL.iter().enumerate() {
            for p in l[t.path_key()].as_array().into_iter().flatten() {
                inputs.paths[k].push(p.as_str().context("a local path")?.to_string());
            }
        }
        for e in l["foreign_del_spans"].as_array().into_iter().flatten() {
            let n = |j: usize| e[j].as_i64().context("a foreign_del_spans key");
            let spans: Vec<i64> = e[3]
                .as_array()
                .context("foreign_del_spans spans")?
                .iter()
                .map(|x| x.as_i64().context("a span"))
                .collect::<Result<_>>()?;
            foreign.add((n(0)?, n(1)?, n(2)?), &spans);
        }
    }
    let mut opts = combine::Options::new(spec, &m.shard_tmpdir, threads);
    opts.fail = fail;
    opts.fail_fast = true;
    let stdout = std::io::stdout();
    let mut emit = |j: &serde_json::Value| -> Result<()> {
        let mut line = serde_json::to_vec(j)?;
        line.push(b'\n');
        let mut out = stdout.lock();
        out.write_all(&line)?;
        out.flush()?;
        Ok(())
    };
    let n_inputs: usize = inputs.paths.iter().map(Vec::len).sum();
    eprintln!(
        "  Combining {n_inputs} local file(s) into data files on {threads} thread(s) \
         (at most {} bytes per file, {} pending upload(s))...",
        spec.max_size_bytes, spec.pending_files
    );
    let s = combine::run_combine(&inputs, &foreign, &opts, gate, &mut emit)?;
    eprintln!(
        "  Combined {} data file(s) in {:.2}s ({}).",
        s.files.values().sum::<usize>(),
        s.seconds,
        s.files
            .iter()
            .map(|(t, n)| format!("{t} {n}"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    Ok(())
}

fn parse_region(r: &str) -> Result<ShardRange> {
    let (chrom, span) = r
        .rsplit_once(':')
        .with_context(|| format!("--region {r}: expected chrom:start-end"))?;
    let (s, e) = span
        .split_once('-')
        .with_context(|| format!("--region {r}: expected chrom:start-end"))?;
    let s: i64 = s.replace(',', "").parse()?;
    let e: i64 = e.replace(',', "").parse()?;
    if s < 0 || e <= s {
        bail!("--region {r}: empty range");
    }
    Ok((chrom.to_string(), s, e))
}

fn main() -> Result<()> {
    let cli = Cli::parse();
    if let Some(dir) = &cli.spike {
        return spike::run(dir);
    }
    if let Some(m) = &cli.manifest
        && !cli.dump_groups
        && !cli.count_records
    {
        return run_worker(m, cli.threads, cli.timing);
    }
    let (paths, shards): (Vec<PathBuf>, Vec<ShardRange>) = if let Some(m) = &cli.manifest {
        let m = manifest::Manifest::load(m)?;
        (
            m.inputs.iter().map(|i| PathBuf::from(&i.path)).collect(),
            m.shards
                .iter()
                .map(|s| (s.chrom.clone(), s.start, s.end))
                .collect(),
        )
    } else {
        if cli.inputs.is_empty() {
            bail!("give --manifest, --inputs or --spike");
        }
        (cli.inputs.clone(), Vec::new())
    };
    let inputs: Vec<Arc<Input>> = paths
        .iter()
        .map(|p| Input::open(p))
        .collect::<Result<_>>()?;
    let mut shards = shards;
    if let Some(r) = &cli.region {
        shards.push(parse_region(r)?);
    }
    for c in &cli.chrom {
        let size = cli.shard_size.context("--chrom needs --shard-size")?;
        if size <= 0 {
            bail!("--shard-size must be positive");
        }
        let len = inputs[0]
            .header
            .contig_length(c)
            .with_context(|| format!("{c}: no length in the first input's header"))?;
        let mut s = 0;
        while s < len {
            let e = (s + size).min(len);
            shards.push((c.clone(), s, e));
            s = e;
        }
    }
    if shards.is_empty() {
        bail!("no shards: give --region, --chrom with --shard-size, or a manifest");
    }
    let mut out: Box<dyn Write> = match &cli.output {
        Some(p) => Box::new(BufWriter::with_capacity(
            1 << 20,
            std::fs::File::create(p).with_context(|| format!("create {}", p.display()))?,
        )),
        None => Box::new(BufWriter::with_capacity(1 << 20, std::io::stdout().lock())),
    };
    if cli.dump_groups {
        let (t, secs) = dump::timed(|| dump::dump_groups(&inputs, &shards, &mut *out))?;
        out.flush()?;
        eprintln!(
            "dump: shards={} yields={} records_dropped={} seconds={secs:.3}",
            t.shards, t.yields, t.records_dropped
        );
    } else if cli.count_records {
        let (t, secs) = dump::timed(|| dump::count_records(&inputs, &shards))?;
        writeln!(
            out,
            "records={} records_dropped={} shards={} seconds={secs:.3} us_per_record={:.3}",
            t.records,
            t.records_dropped,
            t.shards,
            if t.records > 0 {
                secs * 1e6 / t.records as f64
            } else {
                0.0
            }
        )?;
        out.flush()?;
    } else {
        bail!("give --manifest (the worker), --dump-groups, --count-records or --spike");
    }
    let _ = cli.threads;
    Ok(())
}
