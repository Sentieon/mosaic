// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The driver's PostgreSQL exports, one pair of local Parquet
//! files per contig, both sorted by `position`: `sites.c<idx>.parquet`
//! (the Python engine's `fetch_batch_sites` row) and
//! `alleles.c<idx>.parquet` (`fetch_batch_alleles` plus the
//! `DeletionContext` columns). A unit reads them by position: the row groups
//! whose `position` statistics overlap it (and, for the deletion lookback,
//! those whose `del_end` maximum passes the unit's start), decoded one batch
//! at a time; neither export is held whole.

use std::collections::BTreeSet;
use std::fs::File;

use anyhow::{Context, Result, anyhow, bail};
use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{DataType, Float64Type, Int32Type, Int64Type};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{
    ArrowReaderMetadata, ArrowReaderOptions, ParquetRecordBatchReader,
    ParquetRecordBatchReaderBuilder,
};
use parquet::file::statistics::Statistics;

/// Rows per decoded batch of an export.
const EXPORT_BATCH_ROWS: usize = 8192;

/// One `sites` row: a `site_info` row with its `ref_blocks` row
/// (`COALESCE(..., 0)` in the query).
#[derive(Debug, Clone, PartialEq)]
pub struct Site {
    pub position: i32,
    pub qual: Option<f64>,
    pub dp: Option<i64>,
    pub mqsq_1: Option<f64>,
    pub mqsq_2: Option<i64>,
    /// None when the array is null (the four columns null).
    pub sb: Option<[i64; 4]>,
    pub gtc: Option<[i64; 3]>,
    pub rb_dp: i64,
    pub rb_sample_counts: i64,
}

/// One `alleles` row: the catalog's allele at a position, its count, and
/// `[del_start, del_end)` for a deletion allele.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CatalogAllele {
    pub position: i32,
    pub allele: String,
    pub count: i64,
    pub del: Option<(i32, i32)>,
}

/// A row group's statistics on `position` (and `del_end`).
#[derive(Debug, Clone, Copy)]
struct GroupStats {
    pos_min: i64,
    pos_max: i64,
    /// The largest `del_end` (None: the column is all null there);
    /// `i64::MAX` when the file carries no statistics for it.
    del_end_max: Option<i64>,
}

/// One export file, its footer read.
pub struct Export {
    path: String,
    meta: ArrowReaderMetadata,
    groups: Vec<GroupStats>,
}

fn int_stat(s: Option<&Statistics>, max: bool) -> Option<i64> {
    match s? {
        Statistics::Int32(v) => if max { v.max_opt() } else { v.min_opt() }.map(|x| i64::from(*x)),
        Statistics::Int64(v) => if max { v.max_opt() } else { v.min_opt() }.copied(),
        _ => None,
    }
}

/// The `sites` export's columns and types (the driver's
/// `SITES_EXPORT_SCHEMA`).
pub const SITES_COLUMNS: &[(&str, DataType)] = &[
    ("position", DataType::Int32),
    ("qual", DataType::Float64),
    ("dp", DataType::Int64),
    ("mqsq_1", DataType::Float64),
    ("mqsq_2", DataType::Int64),
    ("sb_1", DataType::Int64),
    ("sb_2", DataType::Int64),
    ("sb_3", DataType::Int64),
    ("sb_4", DataType::Int64),
    ("gtc_1", DataType::Int64),
    ("gtc_2", DataType::Int64),
    ("gtc_3", DataType::Int64),
    ("rb_dp", DataType::Int64),
    ("rb_sample_counts", DataType::Int64),
];

/// The `alleles` export's columns and types (`ALLELES_EXPORT_SCHEMA`).
pub const ALLELES_COLUMNS: &[(&str, DataType)] = &[
    ("position", DataType::Int32),
    ("allele", DataType::Utf8),
    ("count", DataType::Int64),
    ("del_start", DataType::Int32),
    ("del_end", DataType::Int32),
];

impl Export {
    /// Open the `sites` export.
    pub fn open_sites(path: &str) -> Result<Export> {
        Export::open(path, SITES_COLUMNS, false)
    }

    /// Open the `alleles` export (with its `del_end` statistics).
    pub fn open_alleles(path: &str) -> Result<Export> {
        Export::open(path, ALLELES_COLUMNS, true)
    }

    /// Open `path`, check its columns' types, and keep its row groups'
    /// statistics; `del_end`: that column's maxima too.
    fn open(path: &str, columns: &[(&str, DataType)], del_end: bool) -> Result<Export> {
        let f = File::open(path).with_context(|| format!("{path}: open"))?;
        let opts = ArrowReaderOptions::new().with_skip_arrow_metadata(true);
        let meta = ArrowReaderMetadata::load(&f, opts)
            .with_context(|| format!("{path}: read the footer"))?;
        for (name, t) in columns {
            let (_, field) = meta
                .schema()
                .column_with_name(name)
                .ok_or_else(|| anyhow!("{path}: no column {name}"))?;
            if field.data_type() != t {
                bail!(
                    "{path}: column {name} is {}, expected {t}",
                    field.data_type()
                );
            }
        }
        let leaf = |name: &str| {
            meta.parquet_schema()
                .columns()
                .iter()
                .position(|c| c.path().string() == name)
                .ok_or_else(|| anyhow!("{path}: no column {name}"))
        };
        let p = leaf("position")?;
        let d = if del_end {
            Some(leaf("del_end")?)
        } else {
            None
        };
        let mut groups = Vec::new();
        for (g, rg) in meta.metadata().row_groups().iter().enumerate() {
            if rg.num_rows() == 0 {
                // An empty row group (pyarrow writes one for an empty
                // export): never selected.
                groups.push(GroupStats {
                    pos_min: i64::MAX,
                    pos_max: i64::MIN,
                    del_end_max: None,
                });
                continue;
            }
            let pc = rg.column(p).statistics();
            let (Some(pos_min), Some(pos_max)) = (int_stat(pc, false), int_stat(pc, true)) else {
                bail!("{path}: row group {g} has no position statistics");
            };
            let del_end_max = match d {
                None => None,
                Some(d) => match rg.column(d).statistics() {
                    None => Some(i64::MAX),
                    Some(s) => match int_stat(Some(s), true) {
                        Some(m) => Some(m),
                        // All null in the group: no deletion there.
                        None if s.null_count_opt() == Some(rg.num_rows() as u64) => None,
                        None => Some(i64::MAX),
                    },
                },
            };
            groups.push(GroupStats {
                pos_min,
                pos_max,
                del_end_max,
            });
        }
        Ok(Export {
            path: path.to_string(),
            meta,
            groups,
        })
    }

    fn reader(&self, groups: Vec<usize>) -> Result<ParquetRecordBatchReader> {
        let f = File::open(&self.path).with_context(|| format!("{}: open", self.path))?;
        ParquetRecordBatchReaderBuilder::new_with_metadata(f, self.meta.clone())
            .with_row_groups(groups)
            .with_projection(ProjectionMask::all())
            .with_batch_size(EXPORT_BATCH_ROWS)
            .build()
            .with_context(|| format!("{}: read", self.path))
    }

    /// The row groups whose positions overlap `[lo, hi]`.
    fn overlapping(&self, lo: i64, hi: i64) -> Vec<usize> {
        (0..self.groups.len())
            .filter(|&g| self.groups[g].pos_min <= hi && self.groups[g].pos_max >= lo)
            .collect()
    }
}

fn col<'a>(b: &'a RecordBatch, path: &str, name: &str) -> Result<&'a dyn Array> {
    b.column_by_name(name)
        .map(|c| c.as_ref())
        .ok_or_else(|| anyhow!("{path}: no column {name}"))
}

fn i64_opt(a: &dyn Array, r: usize) -> Option<i64> {
    if a.is_null(r) {
        None
    } else {
        Some(a.as_primitive::<Int64Type>().value(r))
    }
}

fn f64_opt(a: &dyn Array, r: usize) -> Option<f64> {
    if a.is_null(r) {
        None
    } else {
        Some(a.as_primitive::<Float64Type>().value(r))
    }
}

fn i32_opt(a: &dyn Array, r: usize) -> Option<i32> {
    if a.is_null(r) {
        None
    } else {
        Some(a.as_primitive::<Int32Type>().value(r))
    }
}

/// The `sites` rows of `[lo, hi]`, in position order, a batch at a time.
pub struct SiteCursor {
    path: String,
    reader: ParquetRecordBatchReader,
    rows: std::vec::IntoIter<Site>,
    lo: i64,
    hi: i64,
    done: bool,
}

impl SiteCursor {
    pub fn new(export: &Export, lo: i64, hi: i64) -> Result<SiteCursor> {
        Ok(SiteCursor {
            path: export.path.clone(),
            reader: export.reader(export.overlapping(lo, hi))?,
            rows: Vec::new().into_iter(),
            lo,
            hi,
            done: false,
        })
    }

    fn decode(&self, b: &RecordBatch) -> Result<Vec<Site>> {
        let p = &self.path;
        let pos = col(b, p, "position")?.as_primitive::<Int32Type>();
        let qual = col(b, p, "qual")?;
        let dp = col(b, p, "dp")?;
        let m1 = col(b, p, "mqsq_1")?;
        let m2 = col(b, p, "mqsq_2")?;
        let sb: Vec<&dyn Array> = (1..=4)
            .map(|i| col(b, p, &format!("sb_{i}")))
            .collect::<Result<_>>()?;
        let gtc: Vec<&dyn Array> = (1..=3)
            .map(|i| col(b, p, &format!("gtc_{i}")))
            .collect::<Result<_>>()?;
        let rb_dp = col(b, p, "rb_dp")?;
        let rb_sc = col(b, p, "rb_sample_counts")?;
        let mut out = Vec::with_capacity(b.num_rows());
        for r in 0..b.num_rows() {
            let position = pos.value(r);
            if i64::from(position) < self.lo || i64::from(position) > self.hi {
                continue;
            }
            // A null array is four (three) null columns; a null element of
            // a present array (not written by the aggregate) reads as 0.
            let sb = (!sb[0].is_null(r)).then(|| {
                let v = |i: usize| i64_opt(sb[i], r).unwrap_or(0);
                [v(0), v(1), v(2), v(3)]
            });
            let gtc = (!gtc[0].is_null(r)).then(|| {
                let v = |i: usize| i64_opt(gtc[i], r).unwrap_or(0);
                [v(0), v(1), v(2)]
            });
            out.push(Site {
                position,
                qual: f64_opt(qual, r),
                dp: i64_opt(dp, r),
                mqsq_1: f64_opt(m1, r),
                mqsq_2: i64_opt(m2, r),
                sb,
                gtc,
                rb_dp: i64_opt(rb_dp, r).unwrap_or(0),
                rb_sample_counts: i64_opt(rb_sc, r).unwrap_or(0),
            });
        }
        Ok(out)
    }

    pub fn next_site(&mut self) -> Result<Option<Site>> {
        loop {
            if let Some(s) = self.rows.next() {
                return Ok(Some(s));
            }
            if self.done {
                return Ok(None);
            }
            match self.reader.next() {
                None => self.done = true,
                Some(b) => {
                    let b = b.with_context(|| format!("{}: decode", self.path))?;
                    self.rows = self.decode(&b)?.into_iter();
                }
            }
        }
    }
}

fn decode_alleles(
    path: &str,
    b: &RecordBatch,
    mut keep: impl FnMut(i32, Option<(i32, i32)>) -> bool,
) -> Result<Vec<CatalogAllele>> {
    let pos = col(b, path, "position")?.as_primitive::<Int32Type>();
    let allele = col(b, path, "allele")?.as_string::<i32>();
    let count = col(b, path, "count")?;
    let ds = col(b, path, "del_start")?;
    let de = col(b, path, "del_end")?;
    let mut out = Vec::new();
    for r in 0..b.num_rows() {
        let p = pos.value(r);
        let del = match (i32_opt(ds, r), i32_opt(de, r)) {
            (Some(s), Some(e)) => Some((s, e)),
            _ => None,
        };
        if !keep(p, del) {
            continue;
        }
        if allele.is_null(r) {
            bail!("{path}: a null allele at {p}");
        }
        out.push(CatalogAllele {
            position: p,
            allele: allele.value(r).to_string(),
            count: i64_opt(count, r).unwrap_or(0),
            del,
        });
    }
    Ok(out)
}

/// The `alleles` rows of `[lo, hi]` in export order, a position's rows at
/// a time.
pub struct AlleleCursor {
    path: String,
    reader: ParquetRecordBatchReader,
    rows: std::collections::VecDeque<CatalogAllele>,
    lo: i64,
    hi: i64,
    done: bool,
}

impl AlleleCursor {
    pub fn new(export: &Export, lo: i64, hi: i64) -> Result<AlleleCursor> {
        Ok(AlleleCursor {
            path: export.path.clone(),
            reader: export.reader(export.overlapping(lo, hi))?,
            rows: Default::default(),
            lo,
            hi,
            done: false,
        })
    }

    fn fill(&mut self) -> Result<bool> {
        while self.rows.is_empty() {
            if self.done {
                return Ok(false);
            }
            match self.reader.next() {
                None => self.done = true,
                Some(b) => {
                    let b = b.with_context(|| format!("{}: decode", self.path))?;
                    let (lo, hi) = (self.lo, self.hi);
                    self.rows.extend(decode_alleles(&self.path, &b, |p, _| {
                        i64::from(p) >= lo && i64::from(p) <= hi
                    })?);
                }
            }
        }
        Ok(true)
    }

    /// The rows of the next position at or below `max_pos` (all of them:
    /// a position's rows may span batches and row groups), or None.
    pub fn next_group(&mut self, max_pos: i64) -> Result<Option<Vec<CatalogAllele>>> {
        if !self.fill()? {
            return Ok(None);
        }
        let p = self.rows[0].position;
        if i64::from(p) > max_pos {
            return Ok(None);
        }
        let mut group = Vec::new();
        loop {
            while self.rows.front().is_some_and(|r| r.position == p) {
                group.push(self.rows.pop_front().expect("a front row"));
            }
            if !self.rows.is_empty() || !self.fill()? || self.rows[0].position != p {
                break;
            }
        }
        Ok(Some(group))
    }
}

/// The deletion lookback of a unit starting at `ws`: every
/// catalog row at a position below `ws` that holds a deletion allele with
/// `del_end > ws` (the deletions that may span a site of the unit), in
/// export order. The deletions themselves are the rows with `del` set and
/// `del_end > ws`; the other rows give each such site's non-ref set.
pub fn lookback(export: &Export, ws: i64) -> Result<Vec<CatalogAllele>> {
    // 1. The row groups that may hold such a deletion: positions below ws
    // and a del_end maximum above it.
    let g1: Vec<usize> = (0..export.groups.len())
        .filter(|&g| {
            let s = &export.groups[g];
            s.pos_min < ws && s.del_end_max.is_some_and(|m| m > ws)
        })
        .collect();
    if g1.is_empty() {
        return Ok(Vec::new());
    }
    let mut positions = BTreeSet::new();
    for b in export.reader(g1)? {
        let b = b.with_context(|| format!("{}: decode", export.path))?;
        for r in decode_alleles(&export.path, &b, |p, del| {
            i64::from(p) < ws && del.is_some_and(|(_, e)| i64::from(e) > ws)
        })? {
            positions.insert(r.position);
        }
    }
    if positions.is_empty() {
        return Ok(Vec::new());
    }
    // 2. Every row at those positions (a site's rows may span row groups).
    let g2: Vec<usize> = (0..export.groups.len())
        .filter(|&g| {
            let s = &export.groups[g];
            positions
                .range(s.pos_min.max(i64::from(i32::MIN)) as i32..)
                .next()
                .is_some_and(|&p| i64::from(p) <= s.pos_max)
        })
        .collect();
    let mut out = Vec::new();
    for b in export.reader(g2)? {
        let b = b.with_context(|| format!("{}: decode", export.path))?;
        out.extend(decode_alleles(&export.path, &b, |p, _| {
            positions.contains(&p)
        })?);
    }
    Ok(out)
}
