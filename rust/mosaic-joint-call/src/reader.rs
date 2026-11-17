// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The streaming reader of the per-sample tables: each data
//! file's footer read once, its row groups selected per unit by their own
//! statistics, decoded in batches by the parquet crate's async reader (one
//! task per file, one row group fetched ahead of the merge), the files'
//! rows merged by `(key, sample_id)` (the files' own sort order) through a
//! binary heap, and a tombstoned sample's rows dropped as they are decoded.

use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashSet};
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use anyhow::{Context, Result, anyhow, bail};
use arrow::array::{Array, AsArray, Int32Array, RecordBatch};
use arrow::datatypes::{DataType, Int32Type};
use bytes::Bytes;
use futures::StreamExt;
use futures::future::BoxFuture;
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt};
use parquet::arrow::ProjectionMask;
use parquet::arrow::arrow_reader::{ArrowReaderMetadata, ArrowReaderOptions};
use parquet::arrow::async_reader::{AsyncFileReader, ParquetRecordBatchStreamBuilder};
use parquet::basic::Compression;
use parquet::errors::ParquetError;
use parquet::file::metadata::{ParquetMetaData, ParquetMetaDataReader};
use parquet::file::statistics::Statistics;
use tokio::sync::mpsc;

use crate::manifest::DataFile;
use crate::store::{Location, Stores};

/// Rows per decoded batch.
pub const BATCH_ROWS: usize = 8192;
/// The footer read's first suffix request (one request for most footers).
const FOOTER_HINT: usize = 64 * 1024;
/// Decoded batches buffered per file ahead of the merge: when the last
/// batch of a row group is queued, the file's task already fetches the
/// next row group.
const BATCHES_AHEAD: usize = 2;

/// The two per-sample tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Table {
    SiteInfo,
    RefBlocks,
}

impl Table {
    pub const ALL: [Table; 2] = [Table::SiteInfo, Table::RefBlocks];

    pub fn name(self) -> &'static str {
        match self {
            Table::SiteInfo => "sample_site_info",
            Table::RefBlocks => "sample_ref_blocks",
        }
    }

    /// The columns the engine reads.
    pub fn columns(self) -> &'static [&'static str] {
        match self {
            Table::SiteInfo => &[
                "sample_id",
                "position",
                "alleles",
                "gt_1",
                "gt_2",
                "dp",
                "ad",
                "f1r2",
                "f2r1",
                "pl",
            ],
            Table::RefBlocks => &[
                "sample_id",
                "position_start",
                "position_end",
                "ad_1",
                "ad_2",
                "dp",
                "pl_1",
                "pl_2",
                "pl_3",
                // The no-call fill: a `./.` block is written `./.`.
                "sample_counts",
            ],
        }
    }

    /// The merge key: the files' sort column (with `sample_id`).
    pub fn key(self) -> &'static str {
        match self {
            Table::SiteInfo => "position",
            Table::RefBlocks => "position_start",
        }
    }

    /// The row's last position: a block's end; a site's own position.
    pub fn end(self) -> &'static str {
        match self {
            Table::SiteInfo => "position",
            Table::RefBlocks => "position_end",
        }
    }

    fn schema(self) -> arrow::datatypes::SchemaRef {
        match self {
            Table::SiteInfo => mosaic_core::schemas::sample_site_info(),
            Table::RefBlocks => mosaic_core::schemas::sample_ref_blocks(),
        }
    }

    /// Does a row `[key, end]` lie in the unit `[ws, we]`? A site at a
    /// position inside it; a block overlapping it.
    pub fn in_unit(self, key: i32, end: i32, ws: i64, we: i64) -> bool {
        i64::from(key) <= we && i64::from(end) >= ws
    }
}

/// Bytes moved through a set of readers.
#[derive(Debug, Default)]
pub struct Counters {
    /// The byte ranges the parquet reader asked for.
    pub requested: AtomicU64,
    /// The bytes the store fetched for them: the ranges of one call merged
    /// as object_store 0.14 merges them (gaps up to the store's coalesce
    /// gap included).
    pub fetched: AtomicU64,
    /// The requests the store made.
    pub requests: AtomicU64,
}

impl Counters {
    pub fn get(&self) -> (u64, u64, u64) {
        (
            self.requested.load(Ordering::Relaxed),
            self.fetched.load(Ordering::Relaxed),
            self.requests.load(Ordering::Relaxed),
        )
    }
}

/// object_store 0.14's `merge_ranges` (crate-private there): the sorted
/// ranges covering `ranges`, those less than `gap` apart merged.
fn merge_ranges(ranges: &[Range<u64>], gap: u64) -> Vec<Range<u64>> {
    let mut r = ranges.to_vec();
    r.sort_unstable_by_key(|x| x.start);
    let mut out: Vec<Range<u64>> = Vec::with_capacity(r.len());
    for x in r {
        match out.last_mut() {
            Some(last) if x.start.checked_sub(last.end).is_none_or(|d| d <= gap) => {
                last.end = last.end.max(x.end);
            }
            _ => out.push(x),
        }
    }
    out
}

/// A data file through object_store, every fetch counted: the parquet
/// crate's `AsyncFileReader` implemented directly over the store (its
/// `ParquetObjectReader` is deprecated since parquet 59.2 in favor of
/// this), one `get_range` per single range and one `get_ranges` per row
/// group's column chunks (which the S3 store coalesces).
struct CountingReader {
    store: Arc<dyn ObjectStore>,
    path: Path,
    size: u64,
    counters: Arc<Counters>,
    gap: u64,
}

impl CountingReader {
    fn new(loc: &Location, size: u64, counters: Arc<Counters>) -> CountingReader {
        CountingReader {
            store: loc.store.clone(),
            path: loc.path.clone(),
            size,
            counters,
            gap: loc.coalesce_gap,
        }
    }
}

fn external(e: object_store::Error) -> ParquetError {
    ParquetError::External(Box::new(e))
}

impl AsyncFileReader for CountingReader {
    fn get_bytes(&mut self, range: Range<u64>) -> BoxFuture<'_, parquet::errors::Result<Bytes>> {
        let n = range.end - range.start;
        self.counters.requested.fetch_add(n, Ordering::Relaxed);
        self.counters.fetched.fetch_add(n, Ordering::Relaxed);
        self.counters.requests.fetch_add(1, Ordering::Relaxed);
        Box::pin(async move {
            self.store
                .get_range(&self.path, range)
                .await
                .map_err(external)
        })
    }

    fn get_byte_ranges(
        &mut self,
        ranges: Vec<Range<u64>>,
    ) -> BoxFuture<'_, parquet::errors::Result<Vec<Bytes>>> {
        let requested: u64 = ranges.iter().map(|r| r.end - r.start).sum();
        let merged = merge_ranges(&ranges, self.gap);
        let fetched: u64 = merged.iter().map(|r| r.end - r.start).sum();
        self.counters
            .requested
            .fetch_add(requested, Ordering::Relaxed);
        self.counters.fetched.fetch_add(fetched, Ordering::Relaxed);
        self.counters
            .requests
            .fetch_add(merged.len() as u64, Ordering::Relaxed);
        Box::pin(async move {
            self.store
                .get_ranges(&self.path, &ranges)
                .await
                .map_err(external)
        })
    }

    fn get_metadata<'a>(
        &'a mut self,
        _options: Option<&'a ArrowReaderOptions>,
    ) -> BoxFuture<'a, parquet::errors::Result<Arc<ParquetMetaData>>> {
        // The footer through `get_bytes` (counted). The engine reads it once
        // per file (`OpenFile::open`) and hands it to each stream, so the
        // streams never call this.
        Box::pin(async move {
            let size = self.size;
            let md = ParquetMetaDataReader::new()
                .with_prefetch_hint(Some(FOOTER_HINT))
                .load_and_finish(self, size)
                .await?;
            Ok(Arc::new(md))
        })
    }
}

/// One row group's statistics on the merge key and the row's end.
#[derive(Debug, Clone, Copy)]
struct RowGroupStats {
    key_min: i32,
    end_max: i32,
    rows: u64,
}

/// One data file, its footer read.
pub struct OpenFile {
    pub uri: String,
    pub size: u64,
    loc: Location,
    meta: ArrowReaderMetadata,
    projection: ProjectionMask,
    row_groups: Vec<RowGroupStats>,
    /// The bytes the footer read requested / fetched.
    pub footer: (u64, u64, u64),
}

/// Two Arrow types equal but for the list child field's name, nullability
/// and metadata (a file another writer made may name it otherwise).
fn same_type(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::List(x), DataType::List(y)) => same_type(x.data_type(), y.data_type()),
        _ => a == b,
    }
}

fn int32_stat(s: Option<&Statistics>, max: bool) -> Option<i32> {
    match s? {
        Statistics::Int32(v) => {
            if max {
                v.max_opt().copied()
            } else {
                v.min_opt().copied()
            }
        }
        _ => None,
    }
}

impl OpenFile {
    /// Read the footer of `df` (one suffix request when it fits the hint),
    /// check it against the manifest and the table's schema, and keep the
    /// row groups' statistics on the key and end columns.
    pub async fn open(stores: &Stores, table: Table, df: &DataFile) -> Result<OpenFile> {
        let uri = &df.uri;
        let loc = stores.resolve(uri)?;
        let counters = Arc::new(Counters::default());
        let mut reader = CountingReader::new(&loc, df.file_size_in_bytes, counters.clone());
        let md = reader
            .get_metadata(None)
            .await
            .with_context(|| format!("{uri}: read the footer"))?;
        let footer = counters.get();
        let rows = md.file_metadata().num_rows();
        if rows < 0 || rows as u64 != df.record_count {
            bail!(
                "{uri}: {rows} rows in the footer, {} in the manifest",
                df.record_count
            );
        }
        // The Arrow types from the Parquet schema alone, not from a writer's
        // embedded `ARROW:schema` (pyarrow / pyiceberg may record large or
        // dictionary types; the engine reads the lake's plain types).
        let opts = ArrowReaderOptions::new().with_skip_arrow_metadata(true);
        let meta = ArrowReaderMetadata::try_new(md, opts)
            .with_context(|| format!("{uri}: the Arrow schema"))?;
        // The columns the engine reads, with the types the lake declares.
        let want = table.schema();
        let have = meta.schema();
        let mut roots = Vec::new();
        for name in table.columns() {
            let w = want.field_with_name(name).expect("a declared column");
            let (i, h) = have
                .column_with_name(name)
                .ok_or_else(|| anyhow!("{uri}: no column {name}"))?;
            if !same_type(h.data_type(), w.data_type()) {
                bail!(
                    "{uri}: column {name} is {}, the table declares {}",
                    h.data_type(),
                    w.data_type()
                );
            }
            roots.push(i);
        }
        let projection = ProjectionMask::roots(meta.parquet_schema(), roots);
        // The codecs this build decodes (the parquet crate's `brotli` feature
        // only, the lake's codec): a file in another codec is refused here,
        // naming it, rather than at its first row group.
        for (g, rg) in meta.metadata().row_groups().iter().enumerate() {
            for (i, c) in rg.columns().iter().enumerate() {
                if !projection.leaf_included(i) {
                    continue;
                }
                let codec = c.compression();
                if !matches!(codec, Compression::BROTLI(_) | Compression::UNCOMPRESSED) {
                    bail!(
                        "{uri}: column {} of row group {g} is compressed with {codec}; the \
                         joint caller reads brotli (the lake's codec) and uncompressed only \
                         (a file a `mosaic remove` rewrite wrote before the per-sample \
                         tables' write.parquet.compression-codec=brotli property; the \
                         Python engine that read such files was removed in K1: re-ingest \
                         the samples into a lake made by a later `mosaic init`)",
                        c.column_path().string()
                    );
                }
            }
        }
        // The row groups' statistics on the key and end columns.
        let leaf = |name: &str| {
            meta.parquet_schema()
                .columns()
                .iter()
                .position(|c| c.path().string() == name)
                .ok_or_else(|| anyhow!("{uri}: no leaf column {name}"))
        };
        let (k, e) = (leaf(table.key())?, leaf(table.end())?);
        let mut row_groups = Vec::new();
        for (g, rg) in meta.metadata().row_groups().iter().enumerate() {
            let key_min = int32_stat(rg.column(k).statistics(), false);
            let end_max = int32_stat(rg.column(e).statistics(), true);
            let (Some(key_min), Some(end_max)) = (key_min, end_max) else {
                bail!(
                    "{uri}: row group {g} has no min / max statistics on {} / {}",
                    table.key(),
                    table.end()
                );
            };
            row_groups.push(RowGroupStats {
                key_min,
                end_max,
                rows: rg.num_rows() as u64,
            });
        }
        Ok(OpenFile {
            uri: uri.clone(),
            size: df.file_size_in_bytes,
            loc,
            meta,
            projection,
            row_groups,
            footer,
        })
    }

    /// The row groups that may hold a row of the unit `[ws, we]`: key min
    /// ≤ `we` and end max ≥ `ws` (for `sample_site_info` the end is the
    /// position itself, so this is `position` min ≤ we and max ≥ ws).
    pub fn select(&self, ws: i64, we: i64) -> Vec<usize> {
        self.row_groups
            .iter()
            .enumerate()
            .filter(|(_, g)| i64::from(g.key_min) <= we && i64::from(g.end_max) >= ws)
            .map(|(i, _)| i)
            .collect()
    }

    fn rows_in(&self, groups: &[usize]) -> u64 {
        groups.iter().map(|&g| self.row_groups[g].rows).sum()
    }
}

/// Decode `groups` of `file` and send the batches to the merge.
async fn stream_file(
    file: Arc<OpenFile>,
    groups: Vec<usize>,
    counters: Arc<Counters>,
    tx: mpsc::Sender<Result<RecordBatch>>,
) {
    let run = async {
        let reader = CountingReader::new(&file.loc, file.size, counters);
        let mut stream =
            ParquetRecordBatchStreamBuilder::new_with_metadata(reader, file.meta.clone())
                .with_projection(file.projection.clone())
                .with_row_groups(groups)
                .with_batch_size(BATCH_ROWS)
                .build()?;
        while let Some(b) = stream.next().await {
            if tx.send(b.map_err(anyhow::Error::from)).await.is_err() {
                return Ok(()); // the merge is gone
            }
        }
        anyhow::Ok(())
    };
    if let Err(e) = run.await {
        let _ = tx.send(Err(e.context(format!("{}: read", file.uri)))).await;
    }
}

/// Which per-sample rows a merge drops (masks) by their `sample_id`.
#[derive(Debug, Clone)]
pub enum Mask {
    /// The calling run's (since the sample subset): a row is kept
    /// iff its sample has a header column. For the whole cohort that is
    /// the tombstone mask (a removed sample has no column); under a subset
    /// it drops every sample outside it.
    Keep(Arc<HashSet<i32>>),
    /// `--count-rows`' tombstone mask: a removed sample's rows dropped.
    Drop(Arc<HashSet<i32>>),
}

impl Mask {
    /// Whether a row of `sample_id` is dropped.
    pub fn masks(&self, sample_id: i32) -> bool {
        match self {
            Mask::Keep(s) => !s.contains(&sample_id),
            Mask::Drop(s) => s.contains(&sample_id),
        }
    }
}

/// One file's position in the merge.
struct Cursor {
    uri: String,
    rx: mpsc::Receiver<Result<RecordBatch>>,
    task: tokio::task::JoinHandle<()>,
    batch: Option<RecordBatch>,
    sample: Int32Array,
    key: Int32Array,
    end: Int32Array,
    row: usize,
    /// Batches received so far (a batch's identity for decode caches).
    seq: u64,
    last: Option<(i32, i32)>,
    expect_rows: u64,
    decoded: u64,
    masked: u64,
}

fn int32_col(b: &RecordBatch, name: &str) -> Result<Int32Array> {
    let c = b
        .column_by_name(name)
        .ok_or_else(|| anyhow!("no column {name} in a decoded batch"))?;
    if c.null_count() > 0 {
        bail!("column {name} holds a null");
    }
    Ok(c.as_primitive::<Int32Type>().clone())
}

impl Cursor {
    /// Move to the next unmasked row (the first one on a fresh cursor);
    /// false at the end of the file's selected row groups.
    async fn advance(&mut self, table: Table, mask: &Mask) -> Result<bool> {
        loop {
            if self.batch.is_some() {
                self.row += 1;
            }
            while self.batch.as_ref().is_none_or(|b| self.row >= b.num_rows()) {
                match self.rx.recv().await {
                    Some(b) => {
                        let b = b?;
                        self.sample = int32_col(&b, "sample_id")?;
                        self.key = int32_col(&b, table.key())?;
                        self.end = int32_col(&b, table.end())?;
                        self.decoded += b.num_rows() as u64;
                        self.batch = Some(b);
                        self.row = 0;
                        self.seq += 1;
                    }
                    None => {
                        (&mut self.task)
                            .await
                            .map_err(|e| anyhow!("{}: reader task: {e}", self.uri))?;
                        if self.decoded != self.expect_rows {
                            bail!(
                                "{}: decoded {} rows of the {} the selected row groups hold",
                                self.uri,
                                self.decoded,
                                self.expect_rows
                            );
                        }
                        self.batch = None;
                        return Ok(false);
                    }
                }
            }
            let k = (self.key.value(self.row), self.sample.value(self.row));
            if self.last.is_some_and(|l| l > k) {
                bail!(
                    "{}: rows not sorted by ({}, sample_id): {:?} after {:?}",
                    self.uri,
                    table.key(),
                    k,
                    self.last.unwrap()
                );
            }
            self.last = Some(k);
            if mask.masks(k.1) {
                self.masked += 1;
                continue;
            }
            return Ok(true);
        }
    }
}

/// A merged row: its key, sample and end, and where it is (the file's
/// current batch, valid until the next call to `Merge::next`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Row {
    pub key: i32,
    pub sample_id: i32,
    pub end: i32,
    pub file: usize,
    pub row: usize,
    /// The file's batch number: (file, seq) names the batch.
    pub seq: u64,
}

/// The k-way merge of one table's files over one unit.
pub struct Merge {
    table: Table,
    mask: Mask,
    cursors: Vec<Cursor>,
    heap: BinaryHeap<Reverse<(i32, i32, usize)>>,
    pending: Option<usize>,
    primed: bool,
    pub row_groups: u64,
}

impl Merge {
    /// Start a reader task per file with selected row groups.
    pub fn start(
        table: Table,
        files: &[Arc<OpenFile>],
        ws: i64,
        we: i64,
        mask: Mask,
        counters: Arc<Counters>,
    ) -> Merge {
        let mut cursors = Vec::new();
        let mut row_groups = 0;
        for f in files {
            let groups = f.select(ws, we);
            if groups.is_empty() {
                continue;
            }
            row_groups += groups.len() as u64;
            let expect_rows = f.rows_in(&groups);
            let (tx, rx) = mpsc::channel(BATCHES_AHEAD);
            let task = tokio::spawn(stream_file(f.clone(), groups, counters.clone(), tx));
            cursors.push(Cursor {
                uri: f.uri.clone(),
                rx,
                task,
                batch: None,
                sample: Int32Array::from(Vec::<i32>::new()),
                key: Int32Array::from(Vec::<i32>::new()),
                end: Int32Array::from(Vec::<i32>::new()),
                row: 0,
                seq: 0,
                last: None,
                expect_rows,
                decoded: 0,
                masked: 0,
            });
        }
        Merge {
            table,
            mask,
            cursors,
            heap: BinaryHeap::new(),
            pending: None,
            primed: false,
            row_groups,
        }
    }

    /// Prime every cursor (first call) or advance the one last returned.
    async fn refill(&mut self) -> Result<()> {
        let to_advance: Vec<usize> = if !self.primed {
            self.primed = true;
            (0..self.cursors.len()).collect()
        } else {
            self.pending.take().into_iter().collect()
        };
        for i in to_advance {
            let c = &mut self.cursors[i];
            if c.advance(self.table, &self.mask).await? {
                let r = c.row;
                self.heap
                    .push(Reverse((c.key.value(r), c.sample.value(r), i)));
            }
        }
        Ok(())
    }

    /// The next row in `(key, sample_id)` order, ties by file.
    pub async fn next(&mut self) -> Result<Option<Row>> {
        self.refill().await?;
        let Some(Reverse((key, sample_id, file))) = self.heap.pop() else {
            return Ok(None);
        };
        self.pending = Some(file);
        let c = &self.cursors[file];
        Ok(Some(Row {
            key,
            sample_id,
            end: c.end.value(c.row),
            file,
            row: c.row,
            seq: c.seq,
        }))
    }

    /// The batch `row` points into (the site loop reads the row there).
    pub fn batch(&self, row: &Row) -> &RecordBatch {
        self.cursors[row.file]
            .batch
            .as_ref()
            .expect("a row's batch is current")
    }

    /// Rows decoded and rows masked so far, over the files.
    pub fn decoded_masked(&self) -> (u64, u64) {
        self.cursors
            .iter()
            .fold((0, 0), |(d, m), c| (d + c.decoded, m + c.masked))
    }
}
