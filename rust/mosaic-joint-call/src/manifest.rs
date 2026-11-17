// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The run manifest `joint_call.py` writes. `--count-rows` reads all of
//! it but uses only what it needs: the samples,
//! the removed ids, the contigs with their data files, the S3 settings,
//! `window` and `region`. `output`, `header_lines`, the exports and
//! `tmpdir` are optional here, so a `--count-rows` manifest may omit them
//! (the calling run requires the ones it needs).

use std::collections::HashSet;
use std::path::Path;

use anyhow::{Result, bail};
use serde::Deserialize;

/// One header column: `samples.id` and `sample_name`, in header order.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Sample {
    pub id: i32,
    pub name: String,
}

/// One Iceberg data file of a per-sample table (`plan_files`).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DataFile {
    /// `s3://bucket/key` or `file:///absolute/path`.
    pub uri: String,
    pub record_count: u64,
    pub file_size_in_bytes: u64,
    /// The task's delete files: pyiceberg 0.11 writes none, and the engine
    /// refuses a plan that has any.
    #[serde(default)]
    pub delete_files: Vec<String>,
}

/// One contig, in `contigs.id` order, with its exports and data files.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Contig {
    /// `contigs.id`: the lake's `chromosome_index`.
    pub index: i32,
    pub name: String,
    pub length: i64,
    /// The `sites.c<idx>.parquet` export.
    #[serde(default)]
    pub sites: Option<String>,
    /// The `alleles.c<idx>.parquet` export.
    #[serde(default)]
    pub alleles: Option<String>,
    pub sample_site_info: Vec<DataFile>,
    pub sample_ref_blocks: Vec<DataFile>,
}

/// The object store settings: static (MinIO / the `rest` catalog's
/// `MOSAIC_S3_*`) or `{"from_env": true}` (the AWS credential chain).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct S3 {
    #[serde(default)]
    pub from_env: bool,
    #[serde(default)]
    pub endpoint: Option<String>,
    #[serde(default)]
    pub region: Option<String>,
    #[serde(default)]
    pub access_key_id: Option<String>,
    #[serde(default)]
    pub secret_access_key: Option<String>,
    #[serde(default)]
    pub path_style: bool,
}

/// `--region contig[:start-end]` as the driver resolved it (1-based,
/// inclusive, END clipped to the contig's length).
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Region {
    pub contig: String,
    pub start: i64,
    pub end: i64,
}

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Manifest {
    #[serde(default)]
    pub output: Option<String>,
    #[serde(default)]
    pub header_lines: Option<Vec<String>>,
    pub samples: Vec<Sample>,
    #[serde(default)]
    pub removed_ids: Vec<i32>,
    pub contigs: Vec<Contig>,
    #[serde(default)]
    pub s3: Option<S3>,
    /// The unit width; `--window` overrides it.
    #[serde(default)]
    pub window: Option<i64>,
    #[serde(default)]
    pub region: Option<Region>,
    #[serde(default)]
    pub tmpdir: Option<String>,
    /// Where a unit that is not the head spills its output once its
    /// buffer exceeds `--unit-buffer-bytes` (K2): created at the first
    /// spill, removed when the run ends. Absent: a directory named for the
    /// process under the system temp dir.
    #[serde(default)]
    pub spill_dir: Option<String>,
    /// The run is stale (`--allow-stale` past the gate): the catalog
    /// cross-check's reason for every differing record (J0's
    /// `CatalogCrossCheck`); else a non-empty `removed_ids` is.
    #[serde(default)]
    pub stale: bool,
    /// `samples` is a subset of the active samples (K4, `joint_call.py
    /// --samples`): the catalog cross-check is not run (its counts are the
    /// whole cohort's). Absent: false.
    #[serde(default)]
    pub subset: bool,
}

impl Manifest {
    pub fn load(path: &Path) -> Result<Manifest> {
        let m: Manifest = mosaic_core::manifest::load_json(path)?;
        m.validate()?;
        Ok(m)
    }

    /// What the calling run needs beyond `validate` (J2): the output path,
    /// the header lines, and every called contig's two exports.
    pub fn validate_for_call(&self) -> Result<()> {
        if self.output.as_deref().is_none_or(str::is_empty) {
            bail!("manifest: no output path");
        }
        let Some(h) = &self.header_lines else {
            bail!("manifest: no header_lines");
        };
        if !h.last().is_some_and(|l| l.starts_with("#CHROM")) {
            bail!("manifest: header_lines does not end with the #CHROM line");
        }
        for c in &self.contigs {
            if self.region.as_ref().is_some_and(|r| r.contig != c.name) {
                continue;
            }
            if c.sites.is_none() || c.alleles.is_none() {
                bail!(
                    "manifest: contig {} lacks its sites / alleles export",
                    c.name
                );
            }
        }
        Ok(())
    }

    pub fn validate(&self) -> Result<()> {
        let mut ids = HashSet::new();
        for s in &self.samples {
            if !ids.insert(s.id) {
                bail!("manifest: sample id {} listed twice", s.id);
            }
        }
        for r in &self.removed_ids {
            if ids.contains(r) {
                bail!("manifest: removed id {r} is also an active sample");
            }
        }
        let mut indexes = HashSet::new();
        for c in &self.contigs {
            if !indexes.insert(c.index) {
                bail!("manifest: contig index {} listed twice", c.index);
            }
            if c.length < 1 {
                bail!("manifest: contig {} has length {}", c.name, c.length);
            }
            for f in c.sample_site_info.iter().chain(&c.sample_ref_blocks) {
                if !f.delete_files.is_empty() {
                    bail!(
                        "manifest: {} has {} delete file(s); the engine reads no deletes",
                        f.uri,
                        f.delete_files.len()
                    );
                }
                if !(f.uri.starts_with("s3://") || f.uri.starts_with("file://")) {
                    bail!("manifest: {}: not an s3:// or file:// URI", f.uri);
                }
            }
        }
        if let Some(w) = self.window
            && w < 0
        {
            bail!("manifest: window {w} is negative");
        }
        if let Some(r) = &self.region {
            let Some(c) = self.contigs.iter().find(|c| c.name == r.contig) else {
                bail!("manifest: region contig {} is not in contigs", r.contig);
            };
            if r.start < 1 || r.end < r.start || r.end > c.length {
                bail!(
                    "manifest: region {}:{}-{} is not inside [1, {}]",
                    r.contig,
                    r.start,
                    r.end,
                    c.length
                );
            }
        }
        if let Some(s3) = &self.s3
            && !s3.from_env
            && (s3.region.is_none() || s3.access_key_id.is_none() || s3.secret_access_key.is_none())
        {
            bail!("manifest: s3 needs region, access_key_id and secret_access_key, or from_env");
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE: &str = r#"{
        "samples": [{"id": 3, "name": "HG1"}, {"id": 1, "name": "HG2"}],
        "removed_ids": [2],
        "contigs": [{"index": 0, "name": "chr21", "length": 1000,
            "sample_site_info": [{"uri": "s3://b/k.parquet", "record_count": 5,
                                  "file_size_in_bytes": 100}],
            "sample_ref_blocks": [{"uri": "file:///tmp/x.parquet", "record_count": 5,
                                   "file_size_in_bytes": 100}]}],
        "s3": {"endpoint": "http://localhost:9000", "region": "us-east-1",
               "access_key_id": "a", "secret_access_key": "b", "path_style": true},
        "window": 500,
        "region": {"contig": "chr21", "start": 10, "end": 900},
        "spill_dir": "/tmp/out.vcf.gz.spill-0a1b"
    }"#;

    #[test]
    fn parse_and_validate() {
        let m: Manifest = serde_json::from_str(EXAMPLE).unwrap();
        m.validate().unwrap();
        assert_eq!(m.samples[1].id, 1);
        assert_eq!(m.contigs[0].sample_site_info[0].record_count, 5);
        assert!(m.output.is_none() && m.header_lines.is_none());
        assert_eq!(m.spill_dir.as_deref(), Some("/tmp/out.vcf.gz.spill-0a1b"));
        // `spill_dir` is optional.
        let mut v: serde_json::Value = serde_json::from_str(EXAMPLE).unwrap();
        v.as_object_mut().unwrap().remove("spill_dir").unwrap();
        let m: Manifest = serde_json::from_value(v).unwrap();
        m.validate().unwrap();
        assert!(m.spill_dir.is_none());
        let env: S3 = serde_json::from_str(r#"{"from_env": true}"#).unwrap();
        assert!(env.from_env);
        // `subset` (K4) is optional, default false.
        assert!(!serde_json::from_str::<Manifest>(EXAMPLE).unwrap().subset);
        for flag in [true, false] {
            let text = EXAMPLE.replace(
                r#""window": 500,"#,
                &format!(r#""window": 500, "subset": {flag},"#),
            );
            let m: Manifest = serde_json::from_str(&text).unwrap();
            m.validate().unwrap();
            assert_eq!(m.subset, flag);
        }
        let bad = EXAMPLE.replace(r#""window": 500,"#, r#""window": 500, "subset": "yes","#);
        assert!(serde_json::from_str::<Manifest>(&bad).is_err());
    }

    #[test]
    fn refused() {
        for (from, to) in [
            (r#""window": 500,"#, r#""window": 500, "x": 1,"#),
            // The ALT cap is the binary's option (`--max-alt-alleles`,
            // K3), always passed by the driver, not a manifest field.
            (
                r#""window": 500,"#,
                r#""window": 500, "max_alt_alleles": 6,"#,
            ),
            (r#""spill_dir""#, r#""spill_directory""#),
            (r#""removed_ids": [2]"#, r#""removed_ids": [3]"#),
            (r#""end": 900"#, r#""end": 1001"#),
            (r#""contig": "chr21""#, r#""contig": "chr22""#),
            (r#""window": 500"#, r#""window": -1"#),
            (r#""access_key_id": "a", "#, ""),
            ("s3://b/k.parquet", "http://b/k.parquet"),
            (
                r#""file_size_in_bytes": 100}],
            "sample_ref_blocks""#,
                r#""file_size_in_bytes": 100, "delete_files": ["s3://b/d"]}],
            "sample_ref_blocks""#,
            ),
        ] {
            let text = EXAMPLE.replace(from, to);
            assert_ne!(text, EXAMPLE, "{from}");
            let r = serde_json::from_str::<Manifest>(&text)
                .map_err(anyhow::Error::from)
                .and_then(|m| m.validate());
            assert!(r.is_err(), "{to}");
        }
    }
}
