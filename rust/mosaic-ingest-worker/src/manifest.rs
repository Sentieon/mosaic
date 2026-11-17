// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The batch manifest `cmd_ingest` writes for the worker: everything a
//! shard needs, nothing else.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

/// One input stream `k`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ManifestInput {
    pub path: String,
    /// `samples.id`, or null (`sample_ids[k]` is None).
    pub sample_id: Option<i64>,
    pub sample_name: String,
}

/// One shard: 0-based `start`, exclusive `end`, its index in genomic order.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Shard {
    pub chrom: String,
    pub start: i64,
    pub end: i64,
    pub index: usize,
}

/// Rows per row group of the two per-sample tables' combined files
/// (`PARQUET_ROW_GROUP_ROWS`, `PARQUET_SITE_ROW_GROUP_ROWS`).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct RowsPerGroup {
    pub sample_ref_blocks: usize,
    pub sample_site_info: usize,
}

/// The combine after the shards: present, the worker
/// combines the batch's local files into the data files the driver
/// uploads; absent, it ends after the shards.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CombineSpec {
    /// `--aggregate-parquet-size`: the packing / concatenation target.
    pub max_size_bytes: u64,
    pub rows_per_group: RowsPerGroup,
    /// The gate: a new output file starts only while fewer than this many
    /// finished files are unacknowledged on stdin.
    pub pending_files: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    pub batch_id: i64,
    /// In stream order.
    pub inputs: Vec<ManifestInput>,
    /// Chromosome name -> `contigs` index.
    pub contig_map: HashMap<String, i64>,
    /// The genotype-matrix slot order (ascending `sample_id`).
    pub slot_sample_ids: Vec<i64>,
    /// In genomic order.
    pub shards: Vec<Shard>,
    pub pos_batch_size: i64,
    /// The normalized `--ref-gq-bands` spec, or null.
    pub ref_gq_bands: Option<String>,
    pub shard_tmpdir: String,
    /// `(chrom, start)` of the shards an earlier attempt checkpointed.
    pub committed_shards: Vec<(String, i64)>,
    /// The sorted sample names `_batch_key` folds in.
    pub sample_names: Vec<String>,
    /// The combine (R5); absent: the worker ends after the shards.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub combine: Option<CombineSpec>,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Manifest> {
        let m: Manifest = mosaic_core::manifest::load_json(path)?;
        m.validate()?;
        Ok(m)
    }

    pub fn validate(&self) -> Result<()> {
        if self.inputs.is_empty() {
            bail!("manifest: no inputs");
        }
        for (i, s) in self.shards.iter().enumerate() {
            if s.index != i {
                bail!("manifest: shard {i} has index {}", s.index);
            }
            if s.start < 0 || s.end <= s.start {
                bail!("manifest: shard {i} has range [{}, {})", s.start, s.end);
            }
        }
        if let Some(c) = &self.combine {
            if c.pending_files < 1 {
                bail!("manifest: combine.pending_files must be at least 1");
            }
            if c.rows_per_group.sample_ref_blocks < 1 || c.rows_per_group.sample_site_info < 1 {
                bail!("manifest: combine.rows_per_group must be at least 1");
            }
        }
        let mut sorted = self.sample_names.clone();
        sorted.sort();
        if sorted != self.sample_names {
            bail!("manifest: sample_names are not sorted");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"{
        "batch_id": 7,
        "inputs": [
            {"path": "a.gvcf.gz", "sample_id": 3, "sample_name": "HG1"},
            {"path": "b.gvcf.gz", "sample_id": null, "sample_name": "HG2"}
        ],
        "contig_map": {"chr21": 20, "chr22": 21},
        "slot_sample_ids": [3],
        "shards": [
            {"chrom": "chr21", "start": 0, "end": 10000000, "index": 0},
            {"chrom": "chr21", "start": 10000000, "end": 20000000, "index": 1}
        ],
        "pos_batch_size": 100000,
        "ref_gq_bands": "20,60",
        "shard_tmpdir": "/tmp/x",
        "committed_shards": [["chr21", 0]],
        "sample_names": ["HG1", "HG2"]
    }"#;

    #[test]
    fn round_trip() {
        let m: Manifest = serde_json::from_str(EXAMPLE).unwrap();
        m.validate().unwrap();
        assert_eq!(m.inputs[1].sample_id, None);
        assert_eq!(m.committed_shards, vec![("chr21".to_string(), 0)]);
        assert_eq!(m.contig_map["chr22"], 21);
        let again: Manifest = serde_json::from_str(&serde_json::to_string(&m).unwrap()).unwrap();
        assert_eq!(again, m);
        assert_eq!(m.combine, None);
    }

    #[test]
    fn combine_object() {
        let with = EXAMPLE.replace(
            "\"sample_names\": [\"HG1\", \"HG2\"]",
            "\"sample_names\": [\"HG1\", \"HG2\"], \"combine\": {\"max_size_bytes\": 1000, \
             \"rows_per_group\": {\"sample_ref_blocks\": 65536, \"sample_site_info\": 8192}, \
             \"pending_files\": 8}",
        );
        let m: Manifest = serde_json::from_str(&with).unwrap();
        m.validate().unwrap();
        let c = m.combine.as_ref().unwrap();
        assert_eq!(c.max_size_bytes, 1000);
        assert_eq!(c.rows_per_group.sample_site_info, 8192);
        assert_eq!(c.pending_files, 8);
        for bad in [
            with.replace("\"pending_files\": 8", "\"pending_files\": 0"),
            with.replace("\"sample_site_info\": 8192", "\"sample_site_info\": 0"),
        ] {
            let m: Manifest = serde_json::from_str(&bad).unwrap();
            assert!(m.validate().is_err());
        }
        let extra = with.replace("\"pending_files\": 8", "\"pending_files\": 8, \"x\": 1");
        assert!(serde_json::from_str::<Manifest>(&extra).is_err());
    }

    #[test]
    fn refused() {
        let bad = EXAMPLE.replace("\"batch_id\": 7,", "\"batch_id\": 7, \"extra\": 1,");
        assert!(serde_json::from_str::<Manifest>(&bad).is_err());
        let mut m: Manifest = serde_json::from_str(EXAMPLE).unwrap();
        m.shards[1].index = 5;
        assert!(m.validate().is_err());
        let mut m: Manifest = serde_json::from_str(EXAMPLE).unwrap();
        m.sample_names.reverse();
        assert!(m.validate().is_err());
    }
}
