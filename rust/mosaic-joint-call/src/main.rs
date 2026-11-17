// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! `mosaic-joint-call`: the joint caller `joint_call.py` drives: the
//! manifest, the streaming reader of the per-sample tables
//! (`--count-rows` exercises it alone), the numeric primitives (`pyfmt`,
//! `fisher`), and the calling run (the exports, the units, the site and
//! sample logic, the writer; the units on `--threads` threads, the crash
//! paths, the test seam; the ordered writer's bounded memory,
//! `--unit-buffer-bytes`).

mod call;
mod count;
mod exports;
mod failhook;
mod fisher;
mod format;
mod manifest;
mod ordered;
mod pyfmt;
mod reader;
mod rows;
mod run;
mod store;
#[cfg(test)]
mod testlake;
mod writer;

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use clap::Parser;

/// The default unit width.
const DEFAULT_WINDOW: i64 = 10_000_000;

/// The default ALT cap (`--max-alt-alleles`): GATK's default.
const DEFAULT_MAX_ALT_ALLELES: usize = 6;

#[derive(Parser, Debug)]
#[command(version, about = "MOSAIC joint caller (Rust engine)")]
struct Cli {
    /// The run manifest.
    #[arg(long)]
    manifest: PathBuf,
    /// Units in flight (default: the CPU count).
    #[arg(long)]
    threads: Option<usize>,
    /// The unit width in bp (0 = the whole contig); overrides the
    /// manifest's `window`; default 10,000,000.
    #[arg(long)]
    window: Option<i64>,
    /// A unit that is not the head (the first unit not fully written)
    /// buffers its output in memory up to this many bytes, then spills it
    /// to a file of its own in the manifest's `spill_dir` (K2; 0 or 1:
    /// every chunk spills). `--count-rows` ignores it.
    #[arg(long, default_value_t = ordered::DEFAULT_UNIT_BUFFER_BYTES)]
    unit_buffer_bytes: u64,
    /// A record with more ALT alleles (`*` among them) keeps the N with
    /// the largest allele count and is written as a projection of the
    /// uncapped record (K3: a genotype carrying a dropped ALT `./.`, AD /
    /// PL over the kept alleles, AC / AN / AF recomputed); 0 = no cap.
    /// `--count-rows` ignores it.
    #[arg(long, value_name = "N", default_value_t = DEFAULT_MAX_ALT_ALLELES)]
    max_alt_alleles: usize,
    /// An ALT no written genotype names (AC 0) leaves the record, before
    /// the cap (K5: the other ALTs keep their order, the GTs are
    /// re-indexed, AD / F1R2 / F2R1 / PL are restricted to the kept
    /// alleles, AC / AF lose its entry; a record left without ALTs is
    /// written with ALT `.`). `--count-rows` ignores it.
    #[arg(long)]
    trim_alt_alleles: bool,
    /// A record whose written ALT observations (the sum of its INFO/AC,
    /// after the subset, the trim and the cap) are below K is not written
    /// (K5); 0 = off. `--count-rows` ignores it.
    #[arg(long, value_name = "K", default_value_t = 0)]
    min_ac: u64,
    /// Read both per-sample tables unit by unit and count their rows
    /// (J1's reader spike) instead of calling.
    #[arg(long)]
    count_rows: bool,
}

fn run(cli: Cli) -> Result<()> {
    // The test seam, parsed before anything else (a malformed value fails
    // before any unit).
    let fail_unit =
        failhook::parse_fail_unit(std::env::var(failhook::FAIL_HOOK_ENV).ok().as_deref())?;
    let m = manifest::Manifest::load(&cli.manifest)?;
    let threads = cli
        .threads
        .unwrap_or_else(|| std::thread::available_parallelism().map_or(1, |n| n.get()))
        .max(1);
    let window = cli.window.or(m.window).unwrap_or(DEFAULT_WINDOW);
    if window < 0 {
        bail!("--window {window} is negative");
    }
    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(threads.max(2))
        .enable_all()
        .build()
        .context("the tokio runtime")?;
    if cli.count_rows {
        count::run(&m, threads, window, &rt)
    } else {
        let opts = run::RunOpts {
            fail_unit,
            exit_on_unit_error: true,
            unit_buffer_bytes: cli.unit_buffer_bytes,
            max_alt_alleles: cli.max_alt_alleles,
            trim_alt_alleles: cli.trim_alt_alleles,
            min_ac: cli.min_ac,
            ..run::RunOpts::default()
        };
        run::run_with(&m, threads, window, &rt, &opts).map(|_| ())
    }
}

fn main() {
    if let Err(e) = run(Cli::parse()) {
        eprintln!("Error: {e:#}");
        std::process::exit(1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unit_buffer_bytes_parses_with_its_default() {
        let c = Cli::try_parse_from(["x", "--manifest", "m.json"]).unwrap();
        assert_eq!(c.unit_buffer_bytes, 67_108_864);
        let c =
            Cli::try_parse_from(["x", "--manifest", "m.json", "--unit-buffer-bytes", "1"]).unwrap();
        assert_eq!(c.unit_buffer_bytes, 1);
        for bad in ["-1", "x", ""] {
            assert!(
                Cli::try_parse_from(["x", "--manifest", "m", "--unit-buffer-bytes", bad]).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn max_alt_alleles_parses_with_its_default() {
        let c = Cli::try_parse_from(["x", "--manifest", "m.json"]).unwrap();
        assert_eq!(c.max_alt_alleles, 6);
        for (v, want) in [("0", 0), ("1", 1), ("250", 250)] {
            let c = Cli::try_parse_from(["x", "--manifest", "m", "--max-alt-alleles", v]).unwrap();
            assert_eq!(c.max_alt_alleles, want);
        }
        for bad in ["-1", "x", "", "1.5"] {
            assert!(
                Cli::try_parse_from(["x", "--manifest", "m", "--max-alt-alleles", bad]).is_err(),
                "{bad}"
            );
        }
    }

    #[test]
    fn trim_alt_alleles_and_min_ac_parse_off_by_default() {
        let c = Cli::try_parse_from(["x", "--manifest", "m.json"]).unwrap();
        assert!(!c.trim_alt_alleles);
        assert_eq!(c.min_ac, 0);
        let c = Cli::try_parse_from([
            "x",
            "--manifest",
            "m",
            "--trim-alt-alleles",
            "--min-ac",
            "3",
        ])
        .unwrap();
        assert!(c.trim_alt_alleles);
        assert_eq!(c.min_ac, 3);
        for bad in ["-1", "x", "", "1.5", "2:nref"] {
            assert!(
                Cli::try_parse_from(["x", "--manifest", "m", "--min-ac", bad]).is_err(),
                "{bad}"
            );
        }
        // A flag: it takes no value.
        assert!(Cli::try_parse_from(["x", "--manifest", "m", "--trim-alt-alleles=1"]).is_err());
    }
}
