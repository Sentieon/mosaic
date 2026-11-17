// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The worker's test seam, `MOSAIC_TEST_FAIL_RUST_SHARD`: its parser.
//! Moved out of `queue.rs` when the queue moved to `mosaic-core`: the
//! worker is its only user.

use anyhow::{Result, anyhow};

/// The worker's test seam, `MOSAIC_TEST_FAIL_RUST_SHARD` (test only).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FailHook {
    /// `<chrom>:<start>`: the shard whose chromosome and 0-based start
    /// match fails, before it reads or writes anything.
    Shard(String, i64),
    /// `combine:<table>:<index>` (R5): the combine's output file of that
    /// Iceberg table with that index (its index within its chromosome for
    /// the per-sample tables, so the file of that index of every
    /// chromosome) fails before it is written.
    Combine(String, usize),
}

/// The tables a combine hook may name.
pub const COMBINE_TABLES: [&str; 6] = [
    "sample_ref_blocks",
    "sample_site_info",
    "batch_ref_blocks",
    "batch_site_info",
    "batch_allele_counts",
    "batch_genotypes",
];

/// Parse `MOSAIC_TEST_FAIL_RUST_SHARD`: `<chrom>:<start>` (split at the
/// last colon, so a contig name may hold one) or `combine:<table>:<index>`
/// (a value starting `combine:`). Unset or empty: None. A malformed value
/// is an error.
pub fn parse_fail_hook(value: Option<&str>) -> Result<Option<FailHook>> {
    let Some(v) = value.filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    if let Some(rest) = v.strip_prefix("combine:") {
        let bad = || {
            anyhow!(
                "MOSAIC_TEST_FAIL_RUST_SHARD={v:?}: expected combine:<table>:<index> with \
                 <table> one of {}",
                COMBINE_TABLES.join(", ")
            )
        };
        let (table, index) = rest.split_once(':').ok_or_else(bad)?;
        if !COMBINE_TABLES.contains(&table) {
            return Err(bad());
        }
        let index: usize = index.parse().map_err(|_| bad())?;
        return Ok(Some(FailHook::Combine(table.to_string(), index)));
    }
    let bad = || anyhow!("MOSAIC_TEST_FAIL_RUST_SHARD={v:?}: expected <chrom>:<start>");
    let (chrom, start) = v.rsplit_once(':').ok_or_else(bad)?;
    if chrom.is_empty() {
        return Err(bad());
    }
    let start: i64 = start.parse().map_err(|_| bad())?;
    if start < 0 {
        return Err(bad());
    }
    Ok(Some(FailHook::Shard(chrom.to_string(), start)))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fail_hook_parser() {
        assert_eq!(parse_fail_hook(None).unwrap(), None);
        assert_eq!(parse_fail_hook(Some("")).unwrap(), None);
        assert_eq!(
            parse_fail_hook(Some("chr21:2000000")).unwrap(),
            Some(FailHook::Shard("chr21".to_string(), 2_000_000))
        );
        assert_eq!(
            parse_fail_hook(Some("HLA-A*01:01:0")).unwrap(),
            Some(FailHook::Shard("HLA-A*01:01".to_string(), 0))
        );
        for bad in ["chr21", ":5", "chr21:", "chr21:x", "chr21:-1", "chr21:1.5"] {
            assert!(parse_fail_hook(Some(bad)).is_err(), "{bad}");
        }
        assert_eq!(
            parse_fail_hook(Some("combine:sample_site_info:3")).unwrap(),
            Some(FailHook::Combine("sample_site_info".to_string(), 3))
        );
        assert_eq!(
            parse_fail_hook(Some("combine:batch_genotypes:0")).unwrap(),
            Some(FailHook::Combine("batch_genotypes".to_string(), 0))
        );
        for bad in [
            "combine:",
            "combine:sample_site_info",
            "combine:sample_site_info:",
            "combine:sample_site_info:-1",
            "combine:sample_site_info:x",
            "combine:samples:0",
            "combine::0",
            "combine:sample_site_info:0:1",
        ] {
            assert!(parse_fail_hook(Some(bad)).is_err(), "{bad}");
        }
    }
}
