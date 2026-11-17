// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The combine in the worker: after the last shard, the batch's local
//! files become the data files the driver uploads. What it does is
//! `_aggregate_and_upload_parquet` (`mosaic/ingest.py`) without the
//! upload: the whole-sample combine of the two per-sample tables (whole
//! samples packed per chromosome to the byte target, `pack_sample_files`;
//! each file written window by window, `_write_whole_sample_file`; the
//! cross-shard deletion spans patched in, `_DelSpanPatcher`: names kept
//! from the Python combine it was originally a port of) and the
//! concatenation of the partials and `batch_genotypes`.
//!
//! The output files are planned up front in the main thread (the packing
//! and the concatenation decide each file's inputs from the trailers' and
//! the local files' sizes), so every file is written by one thread from a
//! fixed input order: the same local files give the same bytes at every
//! `--threads`. The files are written on `threads` threads
//! (`queue::run_queue`); the main thread reports each one as it closes
//! (`emit`), deletes a table's inputs after its last file, and checks the
//! cross-shard deletion spans. The gate (`Gate`) holds a new file back
//! while `pending_files` finished files await the driver's
//! acknowledgement.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::BufRead;
use std::ops::ControlFlow;
use std::sync::{Arc, Condvar, Mutex};
use std::time::Instant;

use anyhow::{Context, Result, anyhow, bail};
use arrow::array::{
    Array, Int32Array, Int32Builder, ListArray, ListBuilder, RecordBatch, UInt32Array,
};
use arrow::compute::{concat_batches, take_record_batch};
use arrow::datatypes::{DataType, Field, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::properties::WriterProperties;
use serde_json::{Value as Json, json};

use crate::localfiles::join;
use crate::manifest::CombineSpec;
use crate::pqwrite::{
    COMBINE_BROTLI_LEVEL, PLAIN_SORT, ROW_GROUP_SIZE, SampleSlice, bgt_writer_props,
    writer_props_at,
};
use crate::ref_blocks::merge_spans;
use crate::slices::{read_index, read_slice};
use mosaic_core::queue::run_queue;
use mosaic_core::schemas;

/// The six Iceberg tables an ingest batch uploads, in the driver's
/// registration order (`INGEST_ICEBERG_TABLES`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum Table {
    SampleRefBlocks,
    SampleSiteInfo,
    BatchRefBlocks,
    BatchSiteInfo,
    BatchAlleleCounts,
    BatchGenotypes,
}

impl Table {
    pub const ALL: [Table; 6] = [
        Table::SampleRefBlocks,
        Table::SampleSiteInfo,
        Table::BatchRefBlocks,
        Table::BatchSiteInfo,
        Table::BatchAlleleCounts,
        Table::BatchGenotypes,
    ];

    /// The Iceberg table name (the file lines' `table`).
    pub fn name(self) -> &'static str {
        match self {
            Table::SampleRefBlocks => "sample_ref_blocks",
            Table::SampleSiteInfo => "sample_site_info",
            Table::BatchRefBlocks => "batch_ref_blocks",
            Table::BatchSiteInfo => "batch_site_info",
            Table::BatchAlleleCounts => "batch_allele_counts",
            Table::BatchGenotypes => "batch_genotypes",
        }
    }

    /// The shard result's key of the table's local paths.
    pub fn path_key(self) -> &'static str {
        match self {
            Table::SampleRefBlocks => "rb_paths",
            Table::SampleSiteInfo => "si_paths",
            Table::BatchRefBlocks => "brb_paths",
            Table::BatchSiteInfo => "bsi_paths",
            Table::BatchAlleleCounts => "bac_paths",
            Table::BatchGenotypes => "bgt_paths",
        }
    }

    pub fn schema(self) -> SchemaRef {
        match self {
            Table::SampleRefBlocks => schemas::sample_ref_blocks(),
            Table::SampleSiteInfo => schemas::sample_site_info(),
            Table::BatchRefBlocks => schemas::batch_ref_blocks(),
            Table::BatchSiteInfo => schemas::batch_site_info(),
            Table::BatchAlleleCounts => schemas::batch_allele_counts(),
            Table::BatchGenotypes => schemas::batch_genotypes(),
        }
    }

    /// The per-sample tables: whole samples per file.
    pub fn whole_samples(self) -> bool {
        matches!(self, Table::SampleRefBlocks | Table::SampleSiteInfo)
    }

    fn idx(self) -> usize {
        self as usize
    }
}

/// A (sample_id, chromosome_index, position) key of a foreign span.
pub type SpanKey = (i64, i64, i64);

/// The cross-shard deletion-split spans (`_DelSpanPatcher`): the spans one
/// shard saw for a deletion row another shard wrote, added to that row of
/// `sample_site_info` as it is combined.
#[derive(Debug, Default)]
pub struct Foreign {
    spans: HashMap<SpanKey, Vec<i64>>,
    /// (sample_id, chromosome_index) -> positions: a blob is one slice.
    by_slice: HashMap<(i64, i64), HashSet<i64>>,
}

impl Foreign {
    /// From the shards' `foreign_del_spans`, in shard order (the spans of
    /// one key are concatenated; `merge_spans` sorts them).
    pub fn add(&mut self, key: SpanKey, spans: &[i64]) {
        self.spans.entry(key).or_default().extend_from_slice(spans);
        self.by_slice
            .entry((key.0, key.1))
            .or_default()
            .insert(key.2);
    }

    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }

    pub fn keys(&self) -> impl Iterator<Item = &SpanKey> {
        self.spans.keys()
    }

    /// `_DelSpanPatcher.__call__` on one blob of `sample_site_info`: for
    /// each foreign key, the sample's deletion-carrier row at the position
    /// (of two, the one with the larger `del_end`; ties: the first) gets
    /// `merge_spans(own + foreign)`. The keys matched go to `applied`.
    pub fn patch(&self, batch: RecordBatch, applied: &mut HashSet<SpanKey>) -> Result<RecordBatch> {
        if self.spans.is_empty() || batch.num_rows() == 0 {
            return Ok(batch);
        }
        let i32_col = |name: &str| -> Result<&Int32Array> {
            batch
                .column_by_name(name)
                .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
                .with_context(|| format!("sample_site_info: no int32 column {name}"))
        };
        let sid = i32_col("sample_id")?;
        let chrom = i32_col("chromosome_index")?;
        let pos = i32_col("position")?;
        let del_end = i32_col("del_end")?;
        let mut best: BTreeMap<SpanKey, (usize, i32)> = BTreeMap::new();
        for i in 0..batch.num_rows() {
            let (s, c, p) = (
                sid.value(i) as i64,
                chrom.value(i) as i64,
                pos.value(i) as i64,
            );
            let Some(ps) = self.by_slice.get(&(s, c)) else {
                continue;
            };
            if !ps.contains(&p) || del_end.is_null(i) {
                continue;
            }
            let key = (s, c, p);
            let e = del_end.value(i);
            match best.get(&key) {
                Some(&(_, b)) if e <= b => {}
                _ => {
                    best.insert(key, (i, e));
                }
            }
        }
        if best.is_empty() {
            return Ok(batch);
        }
        let f = batch.schema().index_of("del_ref_spans")?;
        let old = batch
            .column(f)
            .as_any()
            .downcast_ref::<ListArray>()
            .context("sample_site_info: del_ref_spans is not a list")?;
        let mut repl: HashMap<usize, Vec<i32>> = HashMap::new();
        for (key, (i, _)) in &best {
            let mut flat: Vec<i64> = Vec::new();
            if !old.is_null(*i) {
                let v = old.value(*i);
                let v = v
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .context("del_ref_spans items")?;
                flat.extend(v.iter().map(|x| x.unwrap_or_default() as i64));
            }
            flat.extend_from_slice(&self.spans[key]);
            let merged = merge_spans(&flat).unwrap_or_default();
            repl.insert(
                *i,
                merged
                    .into_iter()
                    .map(|x| i32::try_from(x).map_err(|_| anyhow!("span {x} does not fit int32")))
                    .collect::<Result<_>>()?,
            );
            applied.insert(*key);
        }
        let mut b = ListBuilder::new(Int32Builder::new()).with_field(Arc::new(Field::new(
            schemas::LIST_ITEM,
            DataType::Int32,
            true,
        )));
        for i in 0..old.len() {
            if let Some(v) = repl.get(&i) {
                b.values().append_slice(v);
                b.append(true);
            } else if old.is_null(i) {
                b.append(false);
            } else {
                let v = old.value(i);
                let v = v
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .context("del_ref_spans items")?;
                for x in v.iter() {
                    b.values().append_option(x);
                }
                b.append(true);
            }
        }
        let mut cols = batch.columns().to_vec();
        cols[f] = Arc::new(b.finish());
        Ok(RecordBatch::try_new(batch.schema(), cols)?)
    }
}

/// `pack_sample_files`: whole samples into output files, per chromosome,
/// `[(chromosome_index, [sample_id, ...]), ...]` in ascending chromosome
/// and file order. A sample's weight is the sum of its slices' `nbytes`;
/// samples are taken in ascending (chromosome_index, sample_id) and a new
/// file opens when the next sample would exceed `max_size_bytes` and the
/// current file is not empty, or the chromosome changes, so a sample over
/// the target gets a file of its own and a file never splits a sample.
pub fn pack_sample_files<'a>(
    slices: impl IntoIterator<Item = &'a SampleSlice>,
    max_size_bytes: u64,
) -> Vec<(i32, Vec<i32>)> {
    let mut weight: BTreeMap<(i32, i32), i128> = BTreeMap::new();
    for sl in slices {
        *weight
            .entry((sl.chromosome_index, sl.sample_id))
            .or_default() += sl.nbytes as i128;
    }
    let mut files: Vec<(i32, Vec<i32>)> = Vec::new();
    let mut cur_chrom: Option<i32> = None;
    let mut cur: Vec<i32> = Vec::new();
    let mut cur_bytes: i128 = 0;
    for (&(c, s), &w) in &weight {
        if !cur.is_empty() && (Some(c) != cur_chrom || cur_bytes + w > max_size_bytes as i128) {
            files.push((cur_chrom.unwrap_or(c), std::mem::take(&mut cur)));
            cur_bytes = 0;
        }
        cur_chrom = Some(c);
        cur.push(s);
        cur_bytes += w;
    }
    if let Some(c) = cur_chrom
        && !cur.is_empty()
    {
        files.push((c, cur));
    }
    files
}

/// A local file's size and its `chromosome_index` range (None: no rows).
pub type FileInfo = (u64, Option<(i32, i32)>);

/// The concatenation's file boundaries (`_aggregate_and_upload_parquet`
/// without `whole_samples`): in the given (genomic) order, a new output
/// file when the current one's input bytes plus the next file's would
/// exceed `max_size_bytes` and it is not empty, and, with
/// `split_on_chromosome`, whenever the chromosome changes. `info(path)`
/// gives a local file's size and, for the split, its chromosome range
/// (None: no rows). A local file holding two chromosomes is an error.
pub fn plan_concat(
    paths: &[String],
    max_size_bytes: u64,
    split_on_chromosome: bool,
    info: &dyn Fn(&str) -> Result<FileInfo>,
) -> Result<Vec<Vec<String>>> {
    let mut out: Vec<Vec<String>> = Vec::new();
    let mut cur: Vec<String> = Vec::new();
    let mut cur_size: u64 = 0;
    let mut cur_chrom: Option<i32> = None;
    for path in paths {
        let (size, range) = info(path)?;
        if split_on_chromosome && let Some((lo, hi)) = range {
            if lo != hi {
                bail!("{path} holds chromosomes {lo}..{hi}; a partitioned data file must hold one");
            }
            if !cur.is_empty() && Some(lo) != cur_chrom {
                out.push(std::mem::take(&mut cur));
                cur_size = 0;
            }
            cur_chrom = Some(lo);
        }
        if cur_size > 0 && cur_size + size > max_size_bytes {
            out.push(std::mem::take(&mut cur));
            cur_size = 0;
        }
        cur.push(path.clone());
        cur_size += size;
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    Ok(out)
}

/// A local file's size and its `chromosome_index` range (None: no rows),
/// read from the column itself.
fn local_file_info(path: &str, want_range: bool) -> Result<FileInfo> {
    let size = std::fs::metadata(path)
        .with_context(|| format!("stat {path}"))?
        .len();
    if !want_range {
        return Ok((size, None));
    }
    let f = File::open(path).with_context(|| format!("open {path}"))?;
    let b = ParquetRecordBatchReaderBuilder::try_new(f).with_context(|| path.to_string())?;
    let mask = ProjectionMask::roots(b.parquet_schema(), [1]);
    let reader = b
        .with_projection(mask)
        .with_batch_size(ROW_GROUP_SIZE)
        .build()?;
    let mut range: Option<(i32, i32)> = None;
    for batch in reader {
        let batch = batch?;
        let c = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .with_context(|| format!("{path}: chromosome_index is not int32"))?;
        for v in c.iter().flatten() {
            range = Some(match range {
                None => (v, v),
                Some((lo, hi)) => (lo.min(v), hi.max(v)),
            });
        }
    }
    Ok((size, range))
}

/// The timings of a whole-sample file (`_CombineStats`).
#[derive(Debug, Default, Clone, Copy)]
pub struct Stats {
    pub blobs: usize,
    pub windows: usize,
    pub read_s: f64,
    pub transform_s: f64,
    pub sort_s: f64,
    pub write_s: f64,
    pub row_groups: usize,
}

impl Stats {
    fn add(&mut self, o: &Stats) {
        self.blobs += o.blobs;
        self.windows += o.windows;
        self.read_s += o.read_s;
        self.transform_s += o.transform_s;
        self.sort_s += o.sort_s;
        self.write_s += o.write_s;
        self.row_groups += o.row_groups;
    }
}

/// One container's slices of a whole-sample file, in sample_id order.
type Window = (usize, Vec<SampleSlice>);

enum Body {
    Whole {
        c_idx: i32,
        sample_ids: Vec<i32>,
        windows: Vec<Window>,
    },
    Concat {
        inputs: Vec<String>,
    },
}

/// One output file.
struct Job {
    table: Table,
    index: usize,
    path: String,
    body: Body,
}

/// What writing a job gives back to the main thread.
struct Written {
    rows: usize,
    stats: Stats,
    applied: HashSet<SpanKey>,
}

/// The gate: a new output file starts only while fewer than
/// `limit` finished files are unacknowledged; the driver acknowledges a
/// file (its path on one stdin line) once it is uploaded.
pub struct Gate {
    limit: usize,
    st: Mutex<GateState>,
    cv: Condvar,
}

#[derive(Default)]
struct GateState {
    pending: HashSet<String>,
    closed: Option<String>,
    /// The most finished-but-unacknowledged files seen when a file
    /// started (tests).
    max_at_start: usize,
}

impl Gate {
    pub fn new(limit: usize) -> Gate {
        Gate {
            limit: limit.max(1),
            st: Mutex::new(GateState::default()),
            cv: Condvar::new(),
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, GateState> {
        self.st.lock().unwrap_or_else(|p| p.into_inner())
    }

    /// Wait until a new file may start; an error once the gate is closed.
    pub fn acquire(&self) -> Result<()> {
        let mut g = self.lock();
        while g.closed.is_none() && g.pending.len() >= self.limit {
            g = self.cv.wait(g).unwrap_or_else(|p| p.into_inner());
        }
        if let Some(r) = &g.closed {
            bail!("{r}");
        }
        g.max_at_start = g.max_at_start.max(g.pending.len());
        Ok(())
    }

    /// A file closed: it waits for its acknowledgement.
    pub fn finished(&self, path: &str) {
        self.lock().pending.insert(path.to_string());
    }

    /// The driver's acknowledgement of `path`; an error for a path that is
    /// not a finished, unacknowledged file.
    pub fn ack(&self, path: &str) -> Result<()> {
        let mut g = self.lock();
        if !g.pending.remove(path) {
            bail!("acknowledgement of {path:?}, which is not a combined file awaiting one");
        }
        self.cv.notify_all();
        Ok(())
    }

    /// Fail every waiting and later `acquire` with `reason`.
    pub fn close(&self, reason: &str) {
        let mut g = self.lock();
        if g.closed.is_none() {
            g.closed = Some(reason.to_string());
        }
        self.cv.notify_all();
    }

    #[cfg(test)]
    fn pending(&self) -> usize {
        self.lock().pending.len()
    }

    #[cfg(test)]
    fn max_at_start(&self) -> usize {
        self.lock().max_at_start
    }
}

/// Read the driver's acknowledgements (one path per line) into the gate
/// until stdin ends; the end is always an error (the driver is gone), as
/// is an acknowledgement of an unknown path. The caller decides whether it
/// still matters (after the summary line it does not).
pub fn read_acks<R: BufRead>(r: R, gate: &Gate) -> Result<()> {
    for line in r.lines() {
        let line = line.context("read an acknowledgement from stdin")?;
        let path = line.trim_end_matches('\r');
        if path.is_empty() {
            continue;
        }
        gate.ack(path)?;
    }
    let msg = "stdin closed before the combine finished: the driver is gone";
    gate.close(msg);
    bail!("{msg}")
}

/// The combine's inputs: per table, the local paths in shard order.
#[derive(Debug, Default, Clone)]
pub struct Inputs {
    pub paths: [Vec<String>; 6],
}

/// The result of a whole combine (its summary line).
#[derive(Debug)]
pub struct Summary {
    pub files: BTreeMap<&'static str, usize>,
    pub spans_applied: usize,
    pub seconds: f64,
}

impl Summary {
    pub fn json(&self) -> Json {
        json!({
            "kind": "combined",
            "files": self.files,
            "spans_applied": self.spans_applied,
            "seconds": self.seconds,
        })
    }
}

/// The combine's options.
pub struct Options<'a> {
    pub spec: &'a CombineSpec,
    pub tmpdir: &'a str,
    pub threads: usize,
    /// The brotli quality of the combined files (`COMBINE_BROTLI_LEVEL`).
    pub level: u32,
    /// `MOSAIC_TEST_FAIL_RUST_SHARD=combine:<table>:<index>`.
    pub fail: Option<(String, usize)>,
    /// A failed file ends the process at once (`Error: ...`, exit 1), the
    /// other threads' files abandoned (the worker); false: the gate is
    /// closed and the error returned once the threads stop (tests).
    pub fail_fast: bool,
}

impl Options<'_> {
    pub fn new<'a>(spec: &'a CombineSpec, tmpdir: &'a str, threads: usize) -> Options<'a> {
        Options {
            spec,
            tmpdir,
            threads,
            level: COMBINE_BROTLI_LEVEL,
            fail: None,
            fail_fast: false,
        }
    }

    fn rows_per_group(&self, t: Table) -> usize {
        match t {
            Table::SampleRefBlocks => self.spec.rows_per_group.sample_ref_blocks,
            Table::SampleSiteInfo => self.spec.rows_per_group.sample_site_info,
            _ => ROW_GROUP_SIZE,
        }
    }
}

/// The per-sample table's index: its containers and their slices.
struct WholeIndex {
    containers: Vec<String>,
    slices: Vec<Vec<SampleSlice>>,
    seconds: f64,
}

/// Plan every output file: the per-sample tables packed from their
/// trailers, the others concatenated from their sizes.
fn plan(inputs: &Inputs, opts: &Options) -> Result<(Vec<Job>, BTreeMap<Table, WholeIndex>)> {
    let max = opts.spec.max_size_bytes;
    let mut jobs = Vec::new();
    let mut indexes = BTreeMap::new();
    for t in Table::ALL {
        let paths = &inputs.paths[t.idx()];
        if paths.is_empty() {
            continue;
        }
        if t.whole_samples() {
            let t0 = Instant::now();
            let slices: Vec<Vec<SampleSlice>> =
                paths.iter().map(|p| read_index(p)).collect::<Result<_>>()?;
            let mut by_sample: HashMap<(i32, i32), Vec<(usize, SampleSlice)>> = HashMap::new();
            for (k, ss) in slices.iter().enumerate() {
                for sl in ss {
                    by_sample
                        .entry((sl.chromosome_index, sl.sample_id))
                        .or_default()
                        .push((k, sl.clone()));
                }
            }
            let files = pack_sample_files(slices.iter().flatten(), max);
            let mut per_chrom: HashMap<i32, usize> = HashMap::new();
            for (c, sample_ids) in files {
                let index = {
                    let n = per_chrom.entry(c).or_default();
                    *n += 1;
                    *n - 1
                };
                // The file's slices by container, in genomic order, each
                // container's in sample_id order.
                let mut members: Vec<(usize, i32, SampleSlice)> = sample_ids
                    .iter()
                    .flat_map(|s| {
                        by_sample[&(c, *s)]
                            .iter()
                            .map(|(k, sl)| (*k, *s, sl.clone()))
                    })
                    .collect();
                members.sort_by_key(|m| (m.0, m.1));
                let mut windows: Vec<Window> = Vec::new();
                for (k, _s, sl) in members {
                    match windows.last_mut() {
                        Some((last, v)) if *last == k => v.push(sl),
                        _ => windows.push((k, vec![sl])),
                    }
                }
                jobs.push(Job {
                    table: t,
                    index,
                    path: join(opts.tmpdir, &format!("{}_c{c}_{index}.parquet", t.name())),
                    body: Body::Whole {
                        c_idx: c,
                        sample_ids,
                        windows,
                    },
                });
            }
            indexes.insert(
                t,
                WholeIndex {
                    containers: paths.clone(),
                    slices,
                    seconds: t0.elapsed().as_secs_f64(),
                },
            );
        } else {
            let split = t == Table::BatchGenotypes;
            let groups = plan_concat(paths, max, split, &|p| local_file_info(p, split))?;
            for (index, inputs) in groups.into_iter().enumerate() {
                jobs.push(Job {
                    table: t,
                    index,
                    path: join(opts.tmpdir, &format!("{}_{index}.parquet", t.name())),
                    body: Body::Concat { inputs },
                });
            }
        }
    }
    Ok((jobs, indexes))
}

fn create_new(path: &str) -> Result<File> {
    OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .with_context(|| format!("create {path}"))
}

/// `_write_whole_sample_file`: window by window, the file's samples'
/// blobs decoded (and, for `sample_site_info`, patched), concatenated in
/// sample_id order and stably sorted by (position, sample_id) -- a
/// sample's two rows at one position keep their order -- then written in
/// row groups of exactly `rows_per_group` rows that span windows and
/// samples (the last shorter).
fn write_whole(
    job: &Job,
    windows: &[Window],
    index: &WholeIndex,
    props: WriterProperties,
    foreign: Option<&Foreign>,
) -> Result<Written> {
    let schema = job.table.schema();
    let mut st = Stats::default();
    let mut applied = HashSet::new();
    let mut w = ArrowWriter::try_new(create_new(&job.path)?, schema.clone(), Some(props))?;
    let mut rows = 0;
    for (k, slices) in windows {
        let cpath = &index.containers[*k];
        let f = File::open(cpath).with_context(|| format!("open {cpath}"))?;
        let mut parts = Vec::with_capacity(slices.len());
        for sl in slices {
            let t0 = Instant::now();
            let b = read_slice(&f, cpath, sl, &schema)?;
            let t1 = Instant::now();
            st.read_s += (t1 - t0).as_secs_f64();
            st.blobs += 1;
            let b = match foreign {
                Some(fr) => {
                    let b = fr.patch(b, &mut applied)?;
                    st.transform_s += t1.elapsed().as_secs_f64();
                    b
                }
                None => b,
            };
            parts.push(b);
        }
        let t0 = Instant::now();
        let chunk = if parts.len() == 1 {
            parts
                .pop()
                .unwrap_or_else(|| RecordBatch::new_empty(schema.clone()))
        } else {
            sort_by_position_sample(&concat_batches(&schema, &parts)?)?
        };
        st.sort_s += t0.elapsed().as_secs_f64();
        st.windows += 1;
        rows += chunk.num_rows();
        let t0 = Instant::now();
        w.write(&chunk)?;
        st.write_s += t0.elapsed().as_secs_f64();
    }
    let t0 = Instant::now();
    let md = w.close()?;
    st.write_s += t0.elapsed().as_secs_f64();
    st.row_groups = md.row_groups().len();
    Ok(Written {
        rows,
        stats: st,
        applied,
    })
}

/// A stable sort by (position, sample_id): column 2 is `position_start`
/// / `position`, column 0 `sample_id`, in both per-sample schemas.
pub fn sort_by_position_sample(b: &RecordBatch) -> Result<RecordBatch> {
    let col = |k: usize| -> Result<&Int32Array> {
        b.column(k)
            .as_any()
            .downcast_ref::<Int32Array>()
            .with_context(|| format!("column {k} is not int32"))
    };
    let pos = col(2)?.values();
    let sid = col(0)?.values();
    let mut idx: Vec<u32> = (0..b.num_rows() as u32).collect();
    idx.sort_by_key(|&i| (pos[i as usize], sid[i as usize]));
    Ok(take_record_batch(b, &UInt32Array::from(idx))?)
}

/// The concatenation of one output file: its inputs read one row group's
/// worth of rows at a time, re-encoded in row groups of 65,536 rows that
/// span the inputs.
fn write_concat(job: &Job, inputs: &[String], props: WriterProperties) -> Result<Written> {
    let schema = job.table.schema();
    let mut w = ArrowWriter::try_new(create_new(&job.path)?, schema.clone(), Some(props))?;
    let mut rows = 0;
    let t0 = Instant::now();
    for p in inputs {
        let f = File::open(p).with_context(|| format!("open {p}"))?;
        let reader = ParquetRecordBatchReaderBuilder::try_new(f)
            .with_context(|| p.to_string())?
            .with_batch_size(ROW_GROUP_SIZE)
            .build()?;
        for b in reader {
            let b = b.with_context(|| p.to_string())?;
            let b = RecordBatch::try_new(schema.clone(), b.columns().to_vec())
                .with_context(|| format!("{p}: not the {} schema", job.table.name()))?;
            rows += b.num_rows();
            w.write(&b)?;
        }
    }
    let md = w.close()?;
    Ok(Written {
        rows,
        stats: Stats {
            write_s: t0.elapsed().as_secs_f64(),
            row_groups: md.row_groups().len(),
            ..Stats::default()
        },
        applied: HashSet::new(),
    })
}

fn props_for(t: Table, opts: &Options) -> Result<WriterProperties> {
    if t == Table::BatchGenotypes {
        bgt_writer_props(opts.level)
    } else {
        writer_props_at(&PLAIN_SORT, opts.rows_per_group(t), opts.level)
    }
}

fn write_job(
    job: &Job,
    opts: &Options,
    indexes: &BTreeMap<Table, WholeIndex>,
    foreign: &Foreign,
) -> Result<Written> {
    if let Some((t, i)) = &opts.fail
        && t == job.table.name()
        && *i == job.index
    {
        bail!(
            "MOSAIC_TEST_FAIL_RUST_SHARD: injected failure of combine {} file {}",
            t,
            i
        );
    }
    let props = props_for(job.table, opts)?;
    match &job.body {
        Body::Whole { windows, .. } => {
            let fr = (job.table == Table::SampleSiteInfo && !foreign.is_empty()).then_some(foreign);
            write_whole(job, windows, &indexes[&job.table], props, fr)
        }
        Body::Concat { inputs } => write_concat(job, inputs, props),
    }
}

/// The file line of a finished output file.
fn file_line(job: &Job, rows: usize) -> Json {
    match &job.body {
        Body::Whole {
            c_idx, sample_ids, ..
        } => json!({
            "kind": "file",
            "table": job.table.name(),
            "chromosome_index": c_idx,
            "index": job.index,
            "path": job.path,
            "rows": rows,
            "sample_ids": [sample_ids.first(), sample_ids.last()],
            "n_samples": sample_ids.len(),
        }),
        Body::Concat { .. } => json!({
            "kind": "file",
            "table": job.table.name(),
            "chromosome_index": Json::Null,
            "index": job.index,
            "path": job.path,
            "rows": rows,
            "sample_ids": Json::Null,
            "n_samples": Json::Null,
        }),
    }
}

/// The message of the unmatched cross-shard spans (the driver's).
fn unmatched_error(foreign: &Foreign, applied: &HashSet<SpanKey>) -> Option<anyhow::Error> {
    let mut un: Vec<&SpanKey> = foreign.keys().filter(|k| !applied.contains(k)).collect();
    if un.is_empty() {
        return None;
    }
    un.sort();
    let shown: Vec<String> = un
        .iter()
        .take(5)
        .map(|(s, c, p)| format!("({s}, {c}, {p})"))
        .collect();
    Some(anyhow!(
        "{} deletion row(s) for cross-shard deletion-split spans not found in sample_site_info: [{}]",
        un.len(),
        shown.join(", ")
    ))
}

/// Run the combine: plan, write on `opts.threads` threads (each file
/// waits at `gate`), `emit` each file line in completion order and then
/// the summary line (main thread only). A table's local inputs are
/// deleted after its last file closes. Returns the summary.
pub fn run_combine(
    inputs: &Inputs,
    foreign: &Foreign,
    opts: &Options,
    gate: &Gate,
    emit: &mut dyn FnMut(&Json) -> Result<()>,
) -> Result<Summary> {
    let t_start = Instant::now();
    let (jobs, indexes) = plan(inputs, opts)?;
    let mut remaining = [0usize; 6];
    for j in &jobs {
        remaining[j.table.idx()] += 1;
    }
    let mut files: BTreeMap<&'static str, usize> =
        Table::ALL.iter().map(|t| (t.name(), 0)).collect();
    let mut stats: [Stats; 6] = Default::default();
    let mut applied: HashSet<SpanKey> = HashSet::new();
    let mut err: Option<anyhow::Error> = None;
    let fail = |e: anyhow::Error, err: &mut Option<anyhow::Error>| -> ControlFlow<()> {
        if opts.fail_fast {
            eprintln!("Error: {e:?}");
            std::process::exit(1);
        }
        gate.close("the combine failed");
        *err = Some(e);
        ControlFlow::Break(())
    };
    // The cross-shard spans are checked after sample_site_info's last
    // file (here, when it has none).
    if remaining[Table::SampleSiteInfo.idx()] == 0
        && let Some(e) = unmatched_error(foreign, &applied)
    {
        return Err(e);
    }
    let work = |_: &mut (), i: usize| -> Result<Written> {
        let job = &jobs[i];
        let ctx = || format!("combine {} file {}", job.table.name(), job.index);
        gate.acquire().with_context(ctx)?;
        let w = write_job(job, opts, &indexes, foreign).with_context(ctx)?;
        gate.finished(&job.path);
        Ok(w)
    };
    let flow = run_queue(jobs.len(), opts.threads, work, |i, r| {
        let job = &jobs[i];
        let w = match r {
            Ok(w) => w,
            Err(e) => return fail(e, &mut err),
        };
        if let Err(e) = emit(&file_line(job, w.rows)) {
            return fail(e.context("write a file line to stdout"), &mut err);
        }
        let t = job.table;
        if let Some(n) = files.get_mut(t.name()) {
            *n += 1;
        }
        stats[t.idx()].add(&w.stats);
        applied.extend(w.applied);
        remaining[t.idx()] -= 1;
        if remaining[t.idx()] == 0 {
            // The table's last file closed: its local inputs go.
            for p in &inputs.paths[t.idx()] {
                let _ = std::fs::remove_file(p);
            }
            if let Some(ix) = indexes.get(&t) {
                let st = &stats[t.idx()];
                let pairs: HashSet<(i32, i32)> = ix
                    .slices
                    .iter()
                    .flatten()
                    .map(|s| (s.chromosome_index, s.sample_id))
                    .collect();
                eprintln!(
                    "  {}: whole-sample combine of {} local file(s), {} slice(s), {} \
                     (chromosome, sample) pair(s) -> {} file(s) of at most {} bytes, {} row \
                     group(s) of at most {} rows in {:.2}s (index {:.3}s, blob read+decode \
                     {:.2}s, transform {:.2}s, window sort {:.2}s, encode+write {:.2}s)",
                    t.name(),
                    ix.containers.len(),
                    ix.slices.iter().map(Vec::len).sum::<usize>(),
                    pairs.len(),
                    files[t.name()],
                    opts.spec.max_size_bytes,
                    st.row_groups,
                    opts.rows_per_group(t),
                    t_start.elapsed().as_secs_f64(),
                    ix.seconds,
                    st.read_s,
                    st.transform_s,
                    st.sort_s,
                    st.write_s
                );
            }
            if t == Table::SampleSiteInfo
                && let Some(e) = unmatched_error(foreign, &applied)
            {
                return fail(e, &mut err);
            }
        }
        ControlFlow::Continue(())
    });
    if let Some(e) = err {
        return Err(e);
    }
    debug_assert!(flow.is_continue());
    let summary = Summary {
        files,
        spans_applied: applied.len(),
        seconds: t_start.elapsed().as_secs_f64(),
    };
    emit(&summary.json()).context("write the summary line to stdout")?;
    Ok(summary)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::manifest::RowsPerGroup;
    use crate::pqwrite::{write_parquet_file, write_sample_slices};
    use arrow::array::{Float64Array, Int64Array, StringArray};
    use parquet::file::reader::{FileReader, SerializedFileReader};

    fn tmpdir(name: &str) -> String {
        let d =
            std::env::temp_dir().join(format!("mosaic_iw_combine_{}_{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d.to_string_lossy().into_owned()
    }

    fn sl(c: i32, s: i32, nbytes: i64) -> SampleSlice {
        SampleSlice {
            chromosome_index: c,
            sample_id: s,
            offset: 0,
            length: 1,
            nbytes,
            num_rows: 1,
        }
    }

    // ── pack_sample_files (test_sample_major_layout.py's cases) ─────────

    #[test]
    fn pack_whole_samples_to_the_target() {
        // Two slices of sample 1 (30 + 30), sample 2 (50), sample 3 (10).
        let s = [sl(20, 1, 30), sl(20, 2, 50), sl(20, 1, 30), sl(20, 3, 10)];
        assert_eq!(
            pack_sample_files(&s, 100),
            vec![(20, vec![1]), (20, vec![2, 3])]
        );
        assert_eq!(pack_sample_files(&s, 1000), vec![(20, vec![1, 2, 3])]);
        // Exactly at the target: kept together.
        assert_eq!(
            pack_sample_files(&s, 110),
            vec![(20, vec![1, 2]), (20, vec![3])]
        );
    }

    #[test]
    fn pack_oversize_sample_alone_and_chromosomes_apart() {
        let s = [sl(20, 1, 10), sl(20, 2, 500), sl(20, 3, 10), sl(21, 1, 10)];
        assert_eq!(
            pack_sample_files(&s, 100),
            vec![(20, vec![1]), (20, vec![2]), (20, vec![3]), (21, vec![1])]
        );
        // A one-byte target: one file per sample.
        assert_eq!(
            pack_sample_files(&s, 1),
            vec![(20, vec![1]), (20, vec![2]), (20, vec![3]), (21, vec![1])]
        );
        assert!(pack_sample_files(&[], 100).is_empty());
    }

    #[test]
    fn pack_order_independent() {
        let mut s = vec![
            sl(21, 4, 7),
            sl(20, 3, 40),
            sl(20, 1, 30),
            sl(20, 2, 50),
            sl(20, 1, 1),
        ];
        let a = pack_sample_files(&s, 90);
        s.reverse();
        assert_eq!(pack_sample_files(&s, 90), a);
        assert_eq!(a, vec![(20, vec![1, 2]), (20, vec![3]), (21, vec![4])]);
    }

    // ── the concatenation's plan ────────────────────────────────────────

    #[test]
    fn concat_size_rule_split_and_index() {
        let paths: Vec<String> = ["a", "b", "c", "d", "e"]
            .iter()
            .map(|s| s.to_string())
            .collect();
        let info = |p: &str| -> Result<FileInfo> {
            Ok(match p {
                "a" => (40, Some((20, 20))),
                "b" => (40, Some((20, 20))),
                "c" => (40, Some((20, 20))),
                "d" => (10, Some((21, 21))),
                _ => (500, Some((21, 21))),
            })
        };
        // The size rule alone: a+b (80), c+d (50), e (oversize alone).
        let g = plan_concat(&paths, 100, false, &info).unwrap();
        assert_eq!(g, vec![vec!["a", "b"], vec!["c", "d"], vec!["e"]]);
        // With the chromosome split: d starts a file (index counts across).
        let g = plan_concat(&paths, 100, true, &info).unwrap();
        assert_eq!(g, vec![vec!["a", "b"], vec!["c"], vec!["d"], vec!["e"]]);
        // A two-chromosome local file is refused.
        let bad = |p: &str| -> Result<FileInfo> {
            Ok((1, Some(if p == "b" { (20, 21) } else { (20, 20) })))
        };
        let e = plan_concat(&paths, 100, true, &bad).unwrap_err();
        assert!(format!("{e}").contains("b holds chromosomes 20..21"), "{e}");
        // Without the split the two-chromosome file is not looked at.
        assert!(plan_concat(&paths, 100, false, &bad).is_ok());
        // A file without rows does not change the chromosome.
        let empty =
            |p: &str| -> Result<FileInfo> { Ok((1, if p == "b" { None } else { Some((20, 20)) })) };
        assert_eq!(plan_concat(&paths, 100, true, &empty).unwrap().len(), 1);
    }

    // ── the window merge, the row groups, the patcher ───────────────────

    /// (sample_id, position, del_end, del_ref_spans) of a test row.
    type SiRow = (i32, i32, Option<i32>, Option<Vec<i32>>);

    fn si_batch(rows: &[SiRow]) -> RecordBatch {
        // (sample_id, position, del_end, del_ref_spans); chromosome 20.
        let schema = schemas::sample_site_info();
        let n = rows.len();
        let mut cols: Vec<Arc<dyn Array>> = Vec::new();
        for f in schema.fields() {
            let a: Arc<dyn Array> = match f.name().as_str() {
                "sample_id" => Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.0))),
                "chromosome_index" => Arc::new(Int32Array::from(vec![20; n])),
                "position" => Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.1))),
                "del_end" => Arc::new(Int32Array::from_iter(rows.iter().map(|r| r.2))),
                "del_ref_spans" => {
                    let mut b = ListBuilder::new(Int32Builder::new()).with_field(Arc::new(
                        Field::new(schemas::LIST_ITEM, DataType::Int32, true),
                    ));
                    for r in rows {
                        match &r.3 {
                            Some(v) => {
                                b.values().append_slice(v);
                                b.append(true);
                            }
                            None => b.append(false),
                        }
                    }
                    Arc::new(b.finish())
                }
                "qual" | "mqsq_1" => Arc::new(Float64Array::from(vec![1.5; n])),
                "site_dp" | "mqsq_2" => Arc::new(Int64Array::from(vec![3; n])),
                _ => arrow::array::new_null_array(f.data_type(), n),
            };
            cols.push(a);
        }
        RecordBatch::try_new(schema, cols).unwrap()
    }

    fn spans_of(b: &RecordBatch) -> Vec<Option<Vec<i32>>> {
        let l = b
            .column_by_name("del_ref_spans")
            .unwrap()
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        (0..l.len())
            .map(|i| {
                (!l.is_null(i)).then(|| {
                    let v = l.value(i);
                    v.as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values()
                        .to_vec()
                })
            })
            .collect()
    }

    #[test]
    fn patcher_longer_deletion_wins_other_sample_untouched() {
        // test_removal_substrate.py's case: sample 1 has two deletion rows
        // at 100 (del_end 110 and 120); sample 2 a deletion at 100 too but
        // no foreign key.
        let mut fr = Foreign::default();
        fr.add((1, 20, 100), &[115, 118]);
        fr.add((1, 20, 100), &[111, 112]);
        let b = si_batch(&[
            (1, 100, Some(110), Some(vec![101, 103])),
            (1, 100, Some(120), Some(vec![101, 104])),
            (1, 101, None, None),
        ]);
        let mut applied = HashSet::new();
        let out = fr.patch(b, &mut applied).unwrap();
        assert_eq!(
            spans_of(&out),
            vec![
                Some(vec![101, 103]),
                Some(vec![101, 104, 111, 112, 115, 118]),
                None
            ]
        );
        assert_eq!(applied, HashSet::from([(1, 20, 100)]));
        let other = si_batch(&[(2, 100, Some(120), None)]);
        let mut applied = HashSet::new();
        let out = fr.patch(other.clone(), &mut applied).unwrap();
        assert_eq!(out, other);
        assert!(applied.is_empty());
        // Null own spans: the foreign ones alone, merged.
        let b = si_batch(&[(1, 100, Some(120), None)]);
        let out = fr.patch(b, &mut HashSet::new()).unwrap();
        assert_eq!(spans_of(&out), vec![Some(vec![111, 112, 115, 118])]);
        // A row without del_end is not a deletion carrier.
        let b = si_batch(&[(1, 100, None, None)]);
        let mut applied = HashSet::new();
        fr.patch(b, &mut applied).unwrap();
        assert!(applied.is_empty());
        let un = unmatched_error(&fr, &applied).unwrap();
        assert_eq!(
            format!("{un}"),
            "1 deletion row(s) for cross-shard deletion-split spans not found in \
             sample_site_info: [(1, 20, 100)]"
        );
    }

    #[test]
    fn patcher_the_python_case() {
        // test_removal_substrate.py's
        // test_del_span_patcher_adds_foreign_spans_to_the_deletion_row:
        // the adjacent foreign span merges into the longer deletion's.
        let mut fr = Foreign::default();
        fr.add((5, 20, 100), &[103, 103]);
        let b = si_batch(&[
            (5, 100, Some(101), Some(vec![101, 101])),
            (5, 100, Some(103), Some(vec![101, 102])),
            (6, 100, None, None),
            (5, 200, None, None),
        ]);
        let mut applied = HashSet::new();
        let out = fr.patch(b.clone(), &mut applied).unwrap();
        assert_eq!(out.schema(), b.schema());
        assert_eq!(
            spans_of(&out),
            vec![Some(vec![101, 101]), Some(vec![101, 103]), None, None]
        );
        assert_eq!(applied, HashSet::from([(5, 20, 100)]));
        let none = Foreign::default();
        assert_eq!(none.patch(b.clone(), &mut HashSet::new()).unwrap(), b);
    }

    #[test]
    fn window_merge_keeps_a_samples_rows_in_order() {
        // Three samples' blobs of one window, concatenated in sample_id
        // order; sample 2 has two rows at 10 (their order kept).
        let a = si_batch(&[(1, 5, None, None), (1, 10, None, None), (1, 30, None, None)]);
        let b = si_batch(&[
            (2, 10, Some(2), None),
            (2, 10, Some(1), None),
            (2, 20, None, None),
        ]);
        let c = si_batch(&[(3, 5, None, None), (3, 10, None, None)]);
        let cat = concat_batches(&a.schema(), &[a, b, c]).unwrap();
        let s = sort_by_position_sample(&cat).unwrap();
        let sid = s
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        let pos = s
            .column(2)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec();
        let de = s.column(4).as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(pos, vec![5, 5, 10, 10, 10, 10, 20, 30]);
        assert_eq!(sid, vec![1, 3, 1, 2, 2, 3, 2, 1]);
        assert_eq!((de.value(3), de.value(4)), (2, 1));
    }

    fn rb_rows(sample: i32, positions: impl Iterator<Item = i32>) -> RecordBatch {
        let schema = schemas::sample_ref_blocks();
        let p: Vec<i32> = positions.collect();
        let n = p.len();
        let mut cols: Vec<Arc<dyn Array>> = vec![
            Arc::new(Int32Array::from(vec![sample; n])),
            Arc::new(Int32Array::from(vec![20; n])),
            Arc::new(Int32Array::from(p.clone())),
            Arc::new(Int32Array::from_iter_values(p.iter().map(|x| x + 1))),
        ];
        for _ in 4..schema.fields().len() {
            cols.push(Arc::new(Int32Array::from(vec![Some(sample); n])));
        }
        RecordBatch::try_new(schema, cols).unwrap()
    }

    /// Three windows of two samples each as containers.
    fn rb_containers(dir: &str) -> Vec<String> {
        let mut out = Vec::new();
        for w in 0..3 {
            let lo = w * 1000;
            let b = concat_batches(
                &schemas::sample_ref_blocks(),
                &[
                    rb_rows(1, (lo..lo + 7).map(|x| x * 3)),
                    rb_rows(2, (lo..lo + 5).map(|x| x * 3)),
                ],
            )
            .unwrap();
            let p = join(dir, &format!("ref_blocks_w{w}.slices"));
            write_sample_slices(&b, std::path::Path::new(&p)).unwrap();
            out.push(p);
        }
        out
    }

    fn spec(max: u64, rbg: usize, pending: usize) -> CombineSpec {
        CombineSpec {
            max_size_bytes: max,
            rows_per_group: RowsPerGroup {
                sample_ref_blocks: rbg,
                sample_site_info: rbg,
            },
            pending_files: pending,
        }
    }

    fn row_groups(path: &str) -> Vec<i64> {
        let r = SerializedFileReader::new(File::open(path).unwrap()).unwrap();
        r.metadata()
            .row_groups()
            .iter()
            .map(|g| g.num_rows())
            .collect()
    }

    #[test]
    fn row_groups_of_exactly_the_cap_spanning_windows_and_samples() {
        let dir = tmpdir("rg");
        let mut inputs = Inputs::default();
        inputs.paths[0] = rb_containers(&dir);
        let sp = spec(1 << 30, 5, 100);
        let opts = Options::new(&sp, &dir, 2);
        let gate = Gate::new(100);
        let mut lines = Vec::new();
        let s = run_combine(&inputs, &Foreign::default(), &opts, &gate, &mut |j| {
            lines.push(j.clone());
            Ok(())
        })
        .unwrap();
        assert_eq!(s.files["sample_ref_blocks"], 1);
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["rows"], 36);
        assert_eq!(lines[0]["sample_ids"], json!([1, 2]));
        assert_eq!(lines[0]["chromosome_index"], 20);
        assert_eq!(lines[1]["kind"], "combined");
        let out = lines[0]["path"].as_str().unwrap();
        // 36 rows: 7 groups of 5, the last 1.
        assert_eq!(row_groups(out), vec![5, 5, 5, 5, 5, 5, 5, 1]);
        // Rows sorted (position, sample_id) across windows.
        let r = ParquetRecordBatchReaderBuilder::try_new(File::open(out).unwrap())
            .unwrap()
            .build()
            .unwrap();
        let mut keys = Vec::new();
        for b in r {
            let b = b.unwrap();
            let s = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            let p = b.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
            keys.extend(p.values().iter().zip(s.values()).map(|(a, b)| (*a, *b)));
        }
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
        // The inputs are deleted after the table's last file.
        assert!(
            inputs.paths[0]
                .iter()
                .all(|p| !std::path::Path::new(p).exists())
        );
        // One file per sample at a one-byte target, the chromosome index
        // counting per chromosome.
        let dir = tmpdir("rg1");
        inputs.paths[0] = rb_containers(&dir);
        let sp = spec(1, 5, 100);
        let opts = Options::new(&sp, &dir, 3);
        let mut lines = Vec::new();
        run_combine(
            &inputs,
            &Foreign::default(),
            &opts,
            &Gate::new(100),
            &mut |j| {
                lines.push(j.clone());
                Ok(())
            },
        )
        .unwrap();
        let mut got: Vec<(i64, i64, i64)> = lines[..2]
            .iter()
            .map(|l| {
                (
                    l["index"].as_i64().unwrap(),
                    l["sample_ids"][0].as_i64().unwrap(),
                    l["rows"].as_i64().unwrap(),
                )
            })
            .collect();
        got.sort();
        assert_eq!(got, vec![(0, 1, 21), (1, 2, 15)]);
    }

    fn bac_file(path: &str, chrom: i32, n: usize) {
        let schema = schemas::batch_allele_counts();
        let b = RecordBatch::try_new(
            schema,
            vec![
                Arc::new(Int32Array::from(vec![7; n])),
                Arc::new(Int32Array::from(vec![chrom; n])),
                Arc::new(Int32Array::from_iter_values(0..n as i32)),
                Arc::new(StringArray::from_iter_values(
                    (0..n).map(|i| format!("x{i}")),
                )),
                Arc::new(Int64Array::from(vec![Some(1); n])),
                Arc::new(Int64Array::from(vec![Some(1); n])),
            ],
        )
        .unwrap();
        write_parquet_file(&b, std::path::Path::new(path)).unwrap();
    }

    #[test]
    fn concat_rows_span_inputs_in_row_groups_of_65536() {
        let dir = tmpdir("cat");
        let mut inputs = Inputs::default();
        for (k, n) in [40_000usize, 40_000, 100].iter().enumerate() {
            let p = join(&dir, &format!("batch_allele_counts_chr21_{k}.parquet"));
            bac_file(&p, 20, *n);
            inputs.paths[4].push(p);
        }
        let sp = spec(1 << 40, 5, 100);
        let opts = Options::new(&sp, &dir, 1);
        let mut lines = Vec::new();
        run_combine(
            &inputs,
            &Foreign::default(),
            &opts,
            &Gate::new(100),
            &mut |j| {
                lines.push(j.clone());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[0]["rows"], 80_100);
        assert_eq!(lines[0]["chromosome_index"], Json::Null);
        assert_eq!(
            row_groups(lines[0]["path"].as_str().unwrap()),
            vec![65_536, 14_564]
        );
    }

    #[test]
    fn the_gate_holds_files_until_acknowledged() {
        // pending_files 1: a file starts only once every finished file is
        // acknowledged; an acknowledger thread acks each line after a delay.
        let dir = tmpdir("gate");
        let mut inputs = Inputs::default();
        for k in 0..6 {
            let p = join(&dir, &format!("batch_allele_counts_chr21_{k}.parquet"));
            bac_file(&p, 20, 10);
            inputs.paths[4].push(p);
        }
        let sp = spec(1, 5, 1);
        let opts = Options::new(&sp, &dir, 4);
        let gate = Arc::new(Gate::new(1));
        let (tx, rx) = std::sync::mpsc::channel::<String>();
        let g2 = gate.clone();
        let acker = std::thread::spawn(move || {
            for p in rx {
                std::thread::sleep(std::time::Duration::from_millis(20));
                g2.ack(&p).unwrap();
            }
        });
        let mut n = 0;
        let s = run_combine(&inputs, &Foreign::default(), &opts, &gate, &mut |j| {
            if j["kind"] == "file" {
                n += 1;
                tx.send(j["path"].as_str().unwrap().to_string()).unwrap();
            }
            Ok(())
        })
        .unwrap();
        drop(tx);
        acker.join().unwrap();
        assert_eq!(n, 6);
        assert_eq!(s.files["batch_allele_counts"], 6);
        assert_eq!(gate.max_at_start(), 0);
        assert_eq!(gate.pending(), 0);
        // An acknowledgement of an unknown path is an error.
        assert!(gate.ack("/nope").is_err());
    }

    #[test]
    fn stdin_eof_is_an_error_and_closes_the_gate() {
        let gate = Gate::new(1);
        gate.finished("/a");
        let e = read_acks(std::io::Cursor::new("/a\n\n"), &gate).unwrap_err();
        assert!(format!("{e}").contains("the driver is gone"), "{e}");
        assert_eq!(gate.pending(), 0);
        assert!(gate.acquire().is_err());
        let gate = Gate::new(1);
        let e = read_acks(std::io::Cursor::new("/b\n"), &gate).unwrap_err();
        assert!(format!("{e}").contains("not a combined file"), "{e}");
    }

    #[test]
    fn the_combine_seam_fails_its_file() {
        let dir = tmpdir("seam");
        let mut inputs = Inputs::default();
        for k in 0..3 {
            let p = join(&dir, &format!("batch_allele_counts_chr21_{k}.parquet"));
            bac_file(&p, 20, 10);
            inputs.paths[4].push(p);
        }
        let sp = spec(1, 5, 100);
        let mut opts = Options::new(&sp, &dir, 2);
        opts.fail = Some(("batch_allele_counts".to_string(), 1));
        let e = run_combine(
            &inputs,
            &Foreign::default(),
            &opts,
            &Gate::new(100),
            &mut |_| Ok(()),
        )
        .unwrap_err();
        let msg = format!("{e:?}");
        assert!(msg.contains("combine batch_allele_counts file 1"), "{msg}");
        assert!(
            msg.contains("injected failure of combine batch_allele_counts file 1"),
            "{msg}"
        );
        // The file was not written.
        assert!(!std::path::Path::new(&join(&dir, "batch_allele_counts_1.parquet")).exists());
    }

    #[test]
    fn unmatched_spans_fail_after_the_table() {
        let dir = tmpdir("unmatched");
        let mut fr = Foreign::default();
        fr.add((9, 20, 5), &[6, 7]);
        // No sample_site_info at all: an error before anything is written.
        let sp = spec(1, 5, 100);
        let opts = Options::new(&sp, &dir, 1);
        let e = run_combine(&Inputs::default(), &fr, &opts, &Gate::new(100), &mut |_| {
            Ok(())
        })
        .unwrap_err();
        assert!(format!("{e}").starts_with("1 deletion row(s)"), "{e}");
    }

    // ── K6: test_sample_major_layout.py's packing and combine cases ─────

    /// `_slices(weights)`: one SampleSlice per part of each (chromosome,
    /// sample)'s weights.
    fn weighted(weights: &[((i32, i32), Vec<i64>)]) -> Vec<SampleSlice> {
        weights
            .iter()
            .flat_map(|&((c, s), ref parts)| parts.iter().map(move |&w| sl(c, s, w)))
            .collect()
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_pack_whole_samples_to_the_byte_target (K6).
    #[test]
    fn pack_whole_samples_to_the_byte_target_python_case() {
        let weights = vec![
            ((0, 1), vec![30, 30]), // 60
            ((0, 2), vec![40]),     // 40: 60 + 40 = 100 fits exactly
            ((0, 3), vec![10]),     // would make 110
            ((0, 4), vec![250]),    // larger than the target: a file of its own
            ((0, 5), vec![20]),
            ((0, 6), vec![70]), // 20 + 70 = 90
            ((1, 1), vec![5]),  // another chromosome: a new file
            ((1, 9), vec![5]),
        ];
        let files = pack_sample_files(&weighted(&weights), 100);
        assert_eq!(
            files,
            vec![
                (0, vec![1, 2]),
                (0, vec![3]),
                (0, vec![4]),
                (0, vec![5, 6]),
                (1, vec![1, 9]),
            ]
        );
        // Every (chromosome, sample) is in exactly one file.
        let mut placed: Vec<(i32, i32)> = files
            .iter()
            .flat_map(|(c, ss)| ss.iter().map(move |s| (*c, *s)))
            .collect();
        placed.sort();
        let mut want: Vec<(i32, i32)> = weights.iter().map(|w| w.0).collect();
        want.sort();
        assert_eq!(placed, want);
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_pack_is_independent_of_the_slice_order (K6).
    #[test]
    fn pack_is_independent_of_the_slice_order_python_case() {
        let weights: Vec<((i32, i32), Vec<i64>)> = [0, 2]
            .iter()
            .flat_map(|&c| (0..20).map(move |s| ((c, s), vec![(s as i64) * 7 % 13 + 1, 3])))
            .collect();
        let mut slices = weighted(&weights);
        // pack_sample_files(slices, 25) as the Python combine computed it
        // (frozen before its deletion).
        let per_chrom = |c: i32| {
            [
                vec![0, 1, 2],
                vec![3, 4],
                vec![5, 6],
                vec![7, 8],
                vec![9, 10],
                vec![11],
                vec![12, 13, 14],
                vec![15, 16, 17],
                vec![18, 19],
            ]
            .into_iter()
            .map(move |v| (c, v))
        };
        let want: Vec<(i32, Vec<i32>)> = per_chrom(0).chain(per_chrom(2)).collect();
        assert_eq!(pack_sample_files(&slices, 25), want);
        // Five reorderings (the Python test shuffled with five seeds): the
        // reverse and four rotations-with-reversal of the order.
        for k in 0..5 {
            slices.reverse();
            slices.rotate_left(7 * k + 3);
            assert_eq!(pack_sample_files(&slices, 25), want, "order {k}");
        }
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_pack_oversize_first_sample_and_tiny_target (K6).
    #[test]
    fn pack_oversize_first_sample_and_tiny_target_python_case() {
        let s = weighted(&[((0, 1), vec![500]), ((0, 2), vec![1]), ((0, 3), vec![1])]);
        assert_eq!(
            pack_sample_files(&s, 100),
            vec![(0, vec![1]), (0, vec![2, 3])]
        );
        // A 1-byte target: one file per sample.
        assert_eq!(
            pack_sample_files(&s, 1),
            vec![(0, vec![1]), (0, vec![2]), (0, vec![3])]
        );
    }

    /// `_windows(tmp_path, n_windows=3, samples, per=4, chrom)`: three
    /// containers (genomic order) of `per` rows per sample at positions
    /// 1000 w + 10 k, dp = the sample id.
    fn python_windows(dir: &str, samples: &[i32], chrom: i32) -> Vec<String> {
        let schema = schemas::sample_ref_blocks();
        (0..3)
            .map(|w| {
                let parts: Vec<RecordBatch> = samples
                    .iter()
                    .map(|&s| {
                        let b = rb_rows(s, (0..4).map(|k| 1000 * w + 10 * k));
                        let mut cols = b.columns().to_vec();
                        cols[1] = Arc::new(Int32Array::from(vec![chrom; b.num_rows()]));
                        RecordBatch::try_new(schema.clone(), cols).unwrap()
                    })
                    .collect();
                let p = join(dir, &format!("c{chrom}_w{w}.slices"));
                write_sample_slices(
                    &concat_batches(&schema, &parts).unwrap(),
                    std::path::Path::new(&p),
                )
                .unwrap();
                p
            })
            .collect()
    }

    fn i32_column(b: &RecordBatch, name: &str) -> Vec<i32> {
        b.column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    fn read_file(path: &str) -> RecordBatch {
        let r = ParquetRecordBatchReaderBuilder::try_new(File::open(path).unwrap()).unwrap();
        let schema = r.schema().clone();
        let parts: Vec<RecordBatch> = r.build().unwrap().map(|b| b.unwrap()).collect();
        concat_batches(&schema, &parts).unwrap()
    }

    fn pos_sample_keys(b: &RecordBatch, pos: &str) -> Vec<(i32, i32)> {
        i32_column(b, pos)
            .into_iter()
            .zip(i32_column(b, "sample_id"))
            .collect()
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_whole_sample_file_is_sorted_by_position_then_sample (K6).
    #[test]
    fn whole_sample_file_is_sorted_by_position_then_sample_python_case() {
        let dir = tmpdir("k6_whole");
        let mut inputs = Inputs::default();
        inputs.paths[Table::SampleRefBlocks.idx()] = python_windows(&dir, &[1, 2, 3], 0);
        let sp = spec(1 << 30, 5, 100);
        let opts = Options::new(&sp, &dir, 1);
        let (jobs, indexes) = plan(&inputs, &opts).unwrap();
        assert_eq!(jobs.len(), 1);
        let Body::Whole { windows, .. } = &jobs[0].body else {
            panic!("not a whole-sample job")
        };
        let w = write_whole(
            &jobs[0],
            windows,
            &indexes[&Table::SampleRefBlocks],
            props_for(Table::SampleRefBlocks, &opts).unwrap(),
            None,
        )
        .unwrap();
        // Each blob (one sample of one window) read once.
        assert_eq!(w.rows, 36);
        assert_eq!((w.stats.blobs, w.stats.windows), (9, 3));
        // 36 rows at a cap of 5: row groups span samples and windows.
        assert_eq!(row_groups(&jobs[0].path), vec![5, 5, 5, 5, 5, 5, 5, 1]);
        let r = SerializedFileReader::new(File::open(&jobs[0].path).unwrap()).unwrap();
        let rg0 = r.metadata().row_group(0);
        match rg0.column(0).statistics() {
            Some(parquet::file::statistics::Statistics::Int32(v)) => {
                assert_eq!((v.min_opt(), v.max_opt()), (Some(&1), Some(&3)))
            }
            other => panic!("sample_id statistics {other:?}"),
        }
        // The sort declaration: PARQUET_KWARGS' (chromosome_index,
        // position, sample_id), ascending.
        let decl: Vec<(i32, bool)> = rg0
            .sorting_columns()
            .unwrap()
            .iter()
            .map(|c| (c.column_idx, c.descending))
            .collect();
        assert_eq!(decl, vec![(1, false), (2, false), (0, false)]);
        let keys = pos_sample_keys(&read_file(&jobs[0].path), "position_start");
        let mut sorted = keys.clone();
        sorted.sort();
        assert_eq!(keys, sorted);
        assert_eq!(keys[..4], [(0, 1), (0, 2), (0, 3), (10, 1)]);
    }

    /// A `sample_site_info` batch of (sample_id, position, gt_1) rows.
    fn si_gt(rows: &[(i32, i32, i32)]) -> RecordBatch {
        let b = si_batch(
            &rows
                .iter()
                .map(|r| (r.0, r.1, None, None))
                .collect::<Vec<SiRow>>(),
        );
        let mut cols = b.columns().to_vec();
        let k = b.schema().index_of("gt_1").unwrap();
        cols[k] = Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.2)));
        RecordBatch::try_new(b.schema(), cols).unwrap()
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_whole_sample_file_keeps_a_samples_same_position_rows_in_order (K6).
    #[test]
    fn whole_sample_file_keeps_a_samples_same_position_rows_in_order_python_case() {
        let dir = tmpdir("k6_same_pos");
        // Sample 4 has two rows at position 50 (gt_1 1 then 0), sample 1
        // one; sample 2 only in the second window.
        let w0 = join(&dir, "w0.slices");
        write_sample_slices(
            &si_gt(&[(1, 50, 0), (1, 70, 0), (4, 50, 1), (4, 50, 0)]),
            std::path::Path::new(&w0),
        )
        .unwrap();
        let w1 = join(&dir, "w1.slices");
        write_sample_slices(
            &si_gt(&[(2, 120, 1), (4, 110, 1)]),
            std::path::Path::new(&w1),
        )
        .unwrap();
        let mut inputs = Inputs::default();
        inputs.paths[Table::SampleSiteInfo.idx()] = vec![w0, w1];
        let sp = spec(1 << 30, 8192, 100);
        let opts = Options::new(&sp, &dir, 1);
        let mut lines = Vec::new();
        run_combine(
            &inputs,
            &Foreign::default(),
            &opts,
            &Gate::new(100),
            &mut |j| {
                lines.push(j.clone());
                Ok(())
            },
        )
        .unwrap();
        assert_eq!(lines.len(), 2);
        let t = read_file(lines[0]["path"].as_str().unwrap());
        assert_eq!(
            pos_sample_keys(&t, "position"),
            vec![(50, 1), (50, 4), (50, 4), (70, 1), (110, 4), (120, 2)]
        );
        assert_eq!(i32_column(&t, "gt_1"), vec![0, 1, 0, 0, 1, 1]);
    }

    fn combine_lines(inputs: &Inputs, dir: &str, max: u64) -> Result<Vec<Json>> {
        let sp = spec(max, 5, 100);
        let opts = Options::new(&sp, dir, 2);
        let mut lines = Vec::new();
        run_combine(
            inputs,
            &Foreign::default(),
            &opts,
            &Gate::new(100),
            &mut |j| {
                lines.push(j.clone());
                Ok(())
            },
        )?;
        Ok(lines)
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_combine_packs_whole_samples_per_chromosome (K6).
    #[test]
    fn combine_packs_whole_samples_per_chromosome_python_case() {
        let dir = tmpdir("k6_pack");
        let mut paths = python_windows(&dir, &[1, 2, 3], 0);
        paths.extend(python_windows(&dir, &[2, 5], 1));
        let index: Vec<SampleSlice> = paths.iter().flat_map(|p| read_index(p).unwrap()).collect();
        let weight = |c: i32, s: i32| -> i64 {
            index
                .iter()
                .filter(|x| (x.chromosome_index, x.sample_id) == (c, s))
                .map(|x| x.nbytes)
                .sum()
        };
        // A target that fits two samples of chromosome 0, not three.
        let target = (weight(0, 1) + weight(0, 2)) as u64;
        let want = pack_sample_files(&index, target);
        assert_eq!(want, vec![(0, vec![1, 2]), (0, vec![3]), (1, vec![2, 5])]);
        let total_rows: usize = index.iter().map(|x| x.num_rows).sum();
        let mut inputs = Inputs::default();
        inputs.paths[Table::SampleRefBlocks.idx()] = paths;
        let lines = combine_lines(&inputs, &dir, target).unwrap();
        let files: Vec<&Json> = lines.iter().filter(|l| l["kind"] == "file").collect();
        assert_eq!(files.len(), 3);
        let mut contents = Vec::new();
        let mut rows = 0;
        for l in files {
            let t = read_file(l["path"].as_str().unwrap());
            let mut chroms = i32_column(&t, "chromosome_index");
            chroms.dedup();
            assert_eq!(chroms.len(), 1, "one chromosome per file");
            assert_eq!(l["chromosome_index"], chroms[0]);
            let keys = pos_sample_keys(&t, "position_start");
            let mut sorted = keys.clone();
            sorted.sort();
            assert_eq!(keys, sorted);
            let mut samples = i32_column(&t, "sample_id");
            samples.sort();
            samples.dedup();
            contents.push((chroms[0], samples));
            rows += t.num_rows();
        }
        contents.sort();
        assert_eq!(contents, want);
        // Every row, once.
        assert_eq!(rows, total_rows);
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_combine_keys_are_deterministic_and_fold_chromosome_and_index
    // (K6): the worker's half, the same bytes for the same local files and
    // one (chromosome, index) per file; the keys are the driver's
    // (rust_combined_s3_key, tests/unit/test_worker_bin_files.py).
    #[test]
    fn combine_is_deterministic_one_file_per_sample_at_a_one_byte_target() {
        let mut runs: Vec<BTreeMap<(i64, i64), Vec<u8>>> = Vec::new();
        for run in 0..2 {
            let dir = tmpdir(&format!("k6_det{run}"));
            let mut paths = python_windows(&dir, &[1, 2, 3], 0);
            paths.extend(python_windows(&dir, &[1, 2, 3], 1));
            let mut inputs = Inputs::default();
            inputs.paths[Table::SampleRefBlocks.idx()] = paths;
            let lines = combine_lines(&inputs, &dir, 1).unwrap();
            let mut files = BTreeMap::new();
            for l in lines.iter().filter(|l| l["kind"] == "file") {
                let key = (
                    l["chromosome_index"].as_i64().unwrap(),
                    l["index"].as_i64().unwrap(),
                );
                assert_eq!(l["n_samples"], 1);
                let bytes = std::fs::read(l["path"].as_str().unwrap()).unwrap();
                assert!(files.insert(key, bytes).is_none(), "{key:?} twice");
            }
            // 2 chromosomes x 3 samples, the index counting per chromosome.
            assert_eq!(
                files.keys().copied().collect::<Vec<_>>(),
                vec![(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
            );
            runs.push(files);
        }
        assert_eq!(runs[0], runs[1]); // byte for byte
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_combine_refuses_a_plain_parquet_local_file (K6).
    #[test]
    fn combine_refuses_a_plain_parquet_local_file() {
        let dir = tmpdir("k6_plain");
        let p = join(&dir, "plain.parquet");
        write_parquet_file(&rb_rows(3, 0..5), std::path::Path::new(&p)).unwrap();
        let mut inputs = Inputs::default();
        inputs.paths[Table::SampleRefBlocks.idx()] = vec![p];
        let e = combine_lines(&inputs, &dir, 1 << 20).unwrap_err();
        assert!(
            format!("{e}").contains("not a sample-slice container"),
            "{e}"
        );
    }
}
