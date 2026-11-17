// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! Local Parquet writing with the settings of the Python writer it
//! replaced (`PARQUET_KWARGS` in `mosaic/ingest.py`, and the sample-slice
//! blobs' `SAMPLE_SLICE_PARQUET_KWARGS` of the removed Python worker),
//! except the brotli level (`BROTLI_LEVEL`), and the sample-slice container
//! (`write_sample_slices`).

use std::fs::File;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result, bail};
use arrow::array::{Array, Int32Array, RecordBatch};
use parquet::arrow::ArrowWriter;
use parquet::basic::{BrotliLevel, Compression};
use parquet::file::metadata::{ParquetMetaData, SortingColumn};
use parquet::file::properties::{EnabledStatistics, WriterProperties};
use parquet::schema::types::ColumnPath;

/// The brotli quality of the local files. pyarrow writes them at its
/// default 8 (`pa.Codec.default_compression_level("brotli")` in pyarrow
/// 23; `parquet`'s own default is 1), but the `brotli` crate allocates and
/// zeroes a fresh hasher table per column chunk it compresses, 2^15 x
/// 2^(quality - 1) x 4 bytes at qualities 5-9 (16 MB at 8, 1 MB at 5 with
/// its 2^14 buckets), where the C library pyarrow calls zeroes only its
/// 64 KB of counters; a container blob of a few rows pays that per column.
/// Qualities 5-8 use the same hasher family and compress these files to
/// the same bytes within 1% (level 4 switches hasher: +40%), so 5 keeps
/// the packing weights (`pack_sample_files` on the trailer's `nbytes`,
/// `_aggregate_and_upload_parquet` on file sizes) within the
/// engine-to-engine difference level 8 had, at 4-10x less write time.
/// The S3 files are written at `COMBINE_BROTLI_LEVEL` (the worker's
/// combine).
pub const BROTLI_LEVEL: u32 = 5;
/// The brotli quality of the combined data files the worker uploads.
/// The Python combine it replaced wrote them with pyarrow at 8; the rule
/// was 5 if every table's bytes are within 1.5% of pyarrow's at 5, else
/// 8, and at 5 every table measured within +0.75% while 8 doubled the
/// combine's time.
pub const COMBINE_BROTLI_LEVEL: u32 = 5;
/// `PARQUET_DATA_PAGE_SIZE` (config.py default, 128 KiB).
pub const DATA_PAGE_SIZE: usize = 128 * 1024;
/// `PARQUET_ROW_GROUP_SIZE`.
pub const ROW_GROUP_SIZE: usize = 64 * 1024;

pub const SAMPLE_SLICE_MAGIC: &[u8; 8] = b"MOSAICSS";
pub const SAMPLE_SLICE_VERSION: u32 = 1;

fn sorting(cols: &[i32]) -> Vec<SortingColumn> {
    cols.iter()
        .map(|&c| SortingColumn {
            column_idx: c,
            descending: false,
            nulls_first: false,
        })
        .collect()
}

/// The writer properties: brotli at `BROTLI_LEVEL`, statistics with a
/// page index, the 128 KiB data pages, `sorting_columns` as the Python
/// kwargs declare them (column indices), `row_group` rows per row group.
pub fn writer_props(sort_cols: &[i32], row_group: usize) -> Result<WriterProperties> {
    writer_props_at(sort_cols, row_group, BROTLI_LEVEL)
}

/// `writer_props` at brotli quality `level`.
pub fn writer_props_at(
    sort_cols: &[i32],
    row_group: usize,
    level: u32,
) -> Result<WriterProperties> {
    Ok(WriterProperties::builder()
        .set_compression(Compression::BROTLI(BrotliLevel::try_new(level)?))
        .set_data_page_size_limit(DATA_PAGE_SIZE)
        .set_statistics_enabled(EnabledStatistics::Page)
        .set_sorting_columns(Some(sorting(sort_cols)))
        .set_max_row_group_row_count(Some(row_group.max(1)))
        .build())
}

/// `BGT_PARQUET_KWARGS`: `PARQUET_KWARGS` at brotli quality `level` with
/// statistics (and so the page index) on the four scalar key columns
/// only (`batch_id`, `chromosome_index`, `position`, `n_samples`).
pub fn bgt_writer_props(level: u32) -> Result<WriterProperties> {
    let mut props = writer_props_at(&PLAIN_SORT, ROW_GROUP_SIZE, level)?.into_builder();
    for c in [
        vec!["alleles", "list", mosaic_core::schemas::LIST_ITEM],
        vec!["gt_1"],
        vec!["gt_2"],
        vec!["gq"],
    ] {
        props = props.set_column_statistics_enabled(
            ColumnPath::new(c.into_iter().map(String::from).collect()),
            EnabledStatistics::None,
        );
    }
    Ok(props.build())
}

/// `PARQUET_KWARGS`' sort declaration: columns 1, 2, 0.
pub const PLAIN_SORT: [i32; 3] = [1, 2, 0];
/// `SAMPLE_SLICE_PARQUET_KWARGS`' sort declaration: columns 1, 0, 2.
pub const SLICE_SORT: [i32; 3] = [1, 0, 2];

/// Write one Parquet file (`pq.write_table(table, path, **PARQUET_KWARGS)`).
pub fn write_parquet_file(batch: &RecordBatch, path: &Path) -> Result<ParquetMetaData> {
    let file = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let props = writer_props(&PLAIN_SORT, ROW_GROUP_SIZE)?;
    let mut w = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    w.write(batch)?;
    Ok(w.close()?)
}

/// One blob of a container, as `SampleSlice` (without the path).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SampleSlice {
    pub chromosome_index: i32,
    pub sample_id: i32,
    pub offset: u64,
    pub length: u64,
    pub nbytes: i64,
    pub num_rows: usize,
}

fn i32_col<'a>(batch: &'a RecordBatch, name: &str) -> Result<&'a Int32Array> {
    batch
        .column_by_name(name)
        .with_context(|| format!("no column {name}"))?
        .as_any()
        .downcast_ref::<Int32Array>()
        .with_context(|| format!("column {name} is not int32"))
}

/// `write_sample_slices`: `batch` (rows sorted by chromosome_index,
/// sample_id, then position) as a container at `path`: one standalone
/// one-row-group Parquet blob per (chromosome_index, sample_id) run, the
/// JSON trailer, the 20-byte tail. Errors on keys not strictly ascending
/// or positions decreasing within a slice, as the Python writer does.
pub fn write_sample_slices(batch: &RecordBatch, path: &Path) -> Result<Vec<SampleSlice>> {
    let n = batch.num_rows();
    let chrom = i32_col(batch, "chromosome_index")?;
    let sample = i32_col(batch, "sample_id")?;
    let pos_name = if batch.schema().index_of("position_start").is_ok() {
        "position_start"
    } else {
        "position"
    };
    let pos = i32_col(batch, pos_name)?;
    let mut runs: Vec<(i32, i32, usize, usize)> = Vec::new();
    let mut i = 0;
    while i < n {
        let (c, s) = (chrom.value(i), sample.value(i));
        let mut j = i + 1;
        while j < n && chrom.value(j) == c && sample.value(j) == s {
            if pos.value(j) < pos.value(j - 1) {
                bail!("{}: positions decrease within a slice", path.display());
            }
            j += 1;
        }
        if let Some(&(pc, ps, _, _)) = runs.last()
            && (c, s) <= (pc, ps)
        {
            bail!(
                "{}: rows are not sorted by (chromosome_index, sample_id)",
                path.display()
            );
        }
        runs.push((c, s, i, j));
        i = j;
    }
    let mut out = File::create(path).with_context(|| format!("create {}", path.display()))?;
    let mut slices = Vec::with_capacity(runs.len());
    let mut offset = 0u64;
    for (c, s, i, j) in runs {
        let part = batch.slice(i, j - i);
        let props = writer_props(&SLICE_SORT, j - i)?;
        let mut buf: Vec<u8> = Vec::new();
        let mut w = ArrowWriter::try_new(&mut buf, batch.schema(), Some(props))?;
        w.write(&part)?;
        let md = w.close()?;
        let nbytes: i64 = md
            .row_groups()
            .iter()
            .flat_map(|rg| rg.columns().iter())
            .map(|cc| cc.compressed_size())
            .sum();
        out.write_all(&buf)?;
        slices.push(SampleSlice {
            chromosome_index: c,
            sample_id: s,
            offset,
            length: buf.len() as u64,
            nbytes,
            num_rows: j - i,
        });
        offset += buf.len() as u64;
    }
    let entries: Vec<serde_json::Value> = slices
        .iter()
        .map(|sl| {
            serde_json::json!([
                sl.chromosome_index,
                sl.sample_id,
                sl.offset,
                sl.length,
                sl.nbytes,
                sl.num_rows
            ])
        })
        .collect();
    let trailer = serde_json::to_vec(&serde_json::json!({ "slices": entries }))?;
    out.write_all(&trailer)?;
    out.write_all(&(trailer.len() as u64).to_le_bytes())?;
    out.write_all(&SAMPLE_SLICE_VERSION.to_le_bytes())?;
    out.write_all(SAMPLE_SLICE_MAGIC)?;
    Ok(slices)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unsorted_keys_refused() {
        let schema = mosaic_core::schemas::sample_ref_blocks();
        let cols: Vec<std::sync::Arc<dyn Array>> = (0..schema.fields().len())
            .map(|k| {
                let v = match k {
                    0 => vec![2, 1],
                    _ => vec![0, 0],
                };
                std::sync::Arc::new(Int32Array::from(v)) as std::sync::Arc<dyn Array>
            })
            .collect();
        let batch = RecordBatch::try_new(schema, cols).unwrap();
        let dir = std::env::temp_dir().join("mosaic_iw_unsorted.slices");
        assert!(write_sample_slices(&batch, &dir).is_err());
        let _ = std::fs::remove_file(dir);
    }

    /// A `sample_ref_blocks` batch of (sample_id, chromosome_index,
    /// position_start) rows (`_rb_row`'s keys; the other columns 1).
    fn rb(rows: &[(i32, i32, i32)]) -> RecordBatch {
        let schema = mosaic_core::schemas::sample_ref_blocks();
        let mut cols: Vec<std::sync::Arc<dyn Array>> = vec![
            std::sync::Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.0))),
            std::sync::Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.1))),
            std::sync::Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.2))),
            std::sync::Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.2 + 4))),
        ];
        for _ in 4..schema.fields().len() {
            cols.push(std::sync::Arc::new(Int32Array::from(vec![1; rows.len()])));
        }
        RecordBatch::try_new(schema, cols).unwrap()
    }

    fn tmp(name: &str) -> std::path::PathBuf {
        let d = std::env::temp_dir().join(format!("mosaic_iw_pqwrite_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name)
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_container_write_enforces_the_key_order (K6).
    #[test]
    fn container_write_enforces_the_key_order() {
        for keys in [
            vec![(0, 7), (0, 3)],         // samples descending
            vec![(1, 3), (0, 3)],         // chromosomes descending
            vec![(0, 3), (0, 7), (0, 3)], // a key recurs
        ] {
            let rows: Vec<(i32, i32, i32)> = keys.iter().map(|&(c, s)| (s, c, 100)).collect();
            let e = write_sample_slices(&rb(&rows), &tmp("k6_keys.slices")).unwrap_err();
            assert!(format!("{e}").contains("not sorted"), "{keys:?}: {e}");
        }
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_container_write_enforces_position_order_within_a_slice (K6).
    #[test]
    fn container_write_enforces_position_order_within_a_slice() {
        let p = tmp("k6_pos.slices");
        let e = write_sample_slices(&rb(&[(3, 0, 200), (3, 0, 100)]), &p).unwrap_err();
        assert!(format!("{e}").contains("positions decrease"), "{e}");
        // A new slice may start below the previous one's last position.
        write_sample_slices(&rb(&[(3, 0, 200), (4, 0, 100)]), &p).unwrap();
        assert_eq!(
            crate::slices::read_index(p.to_str().unwrap())
                .unwrap()
                .len(),
            2
        );
    }
}
