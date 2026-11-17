// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! Reading the sample-slice containers (for the combine): the same
//! reader and checks as `read_sample_slice_index` and `read_sample_slice`
//! of `tests/sample_slices.py` (whose module docstring has the byte
//! layout; the writer is `pqwrite::write_sample_slices`).

use std::fs::File;

use anyhow::{Context, Result, anyhow, bail};
use arrow::array::RecordBatch;
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use parquet::file::reader::ChunkReader;
use parquet::file::statistics::Statistics;

use crate::pqwrite::{SAMPLE_SLICE_MAGIC, SAMPLE_SLICE_VERSION, SampleSlice};

/// The 20-byte tail: u64 trailer length, u32 version, 8-byte magic.
pub const TAIL_SIZE: u64 = 20;

/// `read_sample_slice_index`: a container's trailer (its only metadata
/// read: the tail, then the trailer), the slices in blob order. Refuses
/// anything but a complete container of `SAMPLE_SLICE_VERSION`: a plain
/// Parquet file, a truncated container, blobs that do not tile the file,
/// keys out of order.
pub fn read_index(path: &str) -> Result<Vec<SampleSlice>> {
    let f = File::open(path).with_context(|| format!("open {path}"))?;
    let size = f.metadata()?.len();
    index_from(path, size, &|off, len| {
        Ok(f.get_bytes(off, len)
            .with_context(|| format!("{path}: read {len} bytes at {off}"))?
            .to_vec())
    })
}

/// `read_index` over a container's bytes (tests).
#[cfg(test)]
pub fn parse_index(path: &str, data: &[u8]) -> Result<Vec<SampleSlice>> {
    index_from(path, data.len() as u64, &|off, len| {
        Ok(data[off as usize..off as usize + len].to_vec())
    })
}

fn index_from(
    path: &str,
    size: u64,
    read: &dyn Fn(u64, usize) -> Result<Vec<u8>>,
) -> Result<Vec<SampleSlice>> {
    if size < TAIL_SIZE {
        bail!("{path}: {size} bytes, too short for a sample-slice container (truncated?)");
    }
    let tail = read(size - TAIL_SIZE, TAIL_SIZE as usize)?;
    let trailer_len = u64::from_le_bytes(tail[0..8].try_into()?);
    let version = u32::from_le_bytes(tail[8..12].try_into()?);
    if &tail[12..20] != SAMPLE_SLICE_MAGIC {
        bail!(
            "{path}: not a sample-slice container (no MOSAICSS tail: a plain Parquet file or a \
             truncated container)"
        );
    }
    if version != SAMPLE_SLICE_VERSION {
        bail!(
            "{path}: sample-slice container version {version}, this code reads version \
             {SAMPLE_SLICE_VERSION}"
        );
    }
    if trailer_len > size - TAIL_SIZE {
        bail!("{path}: trailer overruns the file");
    }
    let body = size - TAIL_SIZE - trailer_len;
    let trailer = read(body, trailer_len as usize)?;
    let v: serde_json::Value = serde_json::from_slice(&trailer)
        .map_err(|e| anyhow!("{path}: unreadable trailer ({e})"))?;
    let entries = v
        .get("slices")
        .and_then(|s| s.as_array())
        .ok_or_else(|| anyhow!("{path}: unreadable trailer (no slices)"))?;
    let mut slices = Vec::with_capacity(entries.len());
    for e in entries {
        let xs: Vec<i64> = e
            .as_array()
            .filter(|a| a.len() == 6)
            .and_then(|a| a.iter().map(|x| x.as_i64()).collect::<Option<Vec<i64>>>())
            .ok_or_else(|| anyhow!("{path}: unreadable trailer (entry {e})"))?;
        slices.push(SampleSlice {
            chromosome_index: i32::try_from(xs[0])?,
            sample_id: i32::try_from(xs[1])?,
            offset: u64::try_from(xs[2]).map_err(|_| anyhow!("{path}: negative offset"))?,
            length: u64::try_from(xs[3]).unwrap_or(0),
            nbytes: xs[4],
            num_rows: usize::try_from(xs[5]).map_err(|_| anyhow!("{path}: negative rows"))?,
        });
    }
    let mut expect = 0u64;
    let mut prev: Option<(i32, i32)> = None;
    for sl in &slices {
        let key = (sl.chromosome_index, sl.sample_id);
        if sl.offset != expect || sl.length == 0 {
            bail!(
                "{path}: blob {key:?} at {} (+{}); expected offset {expect}",
                sl.offset,
                sl.length
            );
        }
        if let Some(p) = prev
            && key <= p
        {
            bail!("{path}: blob {key:?} after {p:?}: not in (chromosome_index, sample_id) order");
        }
        expect += sl.length;
        prev = Some(key);
    }
    if expect != body {
        bail!("{path}: blobs end at byte {expect}, trailer starts at {body}");
    }
    Ok(slices)
}

/// `read_sample_slice`: read and decode one blob of the container open as
/// `file` (at `path`), as a batch with `schema`. Checks that its row count
/// matches the trailer and that, in every row group, the
/// `chromosome_index` and `sample_id` statistics give min == max == the
/// trailer's key.
pub fn read_slice(
    file: &File,
    path: &str,
    sl: &SampleSlice,
    schema: &SchemaRef,
) -> Result<RecordBatch> {
    let bytes = file
        .get_bytes(sl.offset, sl.length as usize)
        .map_err(|e| anyhow!("{path}: blob at {} short read ({e})", sl.offset))?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(bytes)
        .with_context(|| format!("{path}: blob at {}", sl.offset))?;
    for rg in builder.metadata().row_groups() {
        for col in rg.columns() {
            let name = col.column_path().string();
            let expected = match name.as_str() {
                "chromosome_index" => sl.chromosome_index,
                "sample_id" => sl.sample_id,
                _ => continue,
            };
            let mm = match col.statistics() {
                Some(Statistics::Int32(v)) => match (v.min_opt(), v.max_opt()) {
                    (Some(&a), Some(&b)) => Some((a, b)),
                    _ => None,
                },
                _ => None,
            };
            if mm != Some((expected, expected)) {
                bail!(
                    "{path}: blob at {}: {name} statistics {mm:?}, trailer says {expected}",
                    sl.offset
                );
            }
        }
    }
    let reader = builder
        .with_batch_size(sl.num_rows.max(1))
        .build()
        .with_context(|| format!("{path}: blob at {}", sl.offset))?;
    let mut batches = Vec::new();
    for b in reader {
        let b = b.with_context(|| format!("{path}: blob at {}", sl.offset))?;
        batches.push(RecordBatch::try_new(schema.clone(), b.columns().to_vec())?);
    }
    let batch = if batches.len() == 1 {
        batches
            .pop()
            .unwrap_or_else(|| RecordBatch::new_empty(schema.clone()))
    } else {
        concat_batches(schema, &batches)?
    };
    if batch.num_rows() != sl.num_rows {
        bail!(
            "{path}: blob at {} has {} rows, trailer says {}",
            sl.offset,
            batch.num_rows(),
            sl.num_rows
        );
    }
    Ok(batch)
}

/// Every row of a container (tests).
#[cfg(test)]
pub fn read_all(path: &str, schema: &SchemaRef) -> Result<RecordBatch> {
    let idx = read_index(path)?;
    let f = File::open(path)?;
    let parts: Vec<RecordBatch> = idx
        .iter()
        .map(|sl| read_slice(&f, path, sl, schema))
        .collect::<Result<_>>()?;
    Ok(concat_batches(schema, &parts)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pqwrite::write_sample_slices;
    use arrow::array::{Array, Int32Array};
    use mosaic_core::schemas;
    use std::io::Write;
    use std::sync::Arc;

    fn rb_batch(rows: &[(i32, i32, i32)]) -> RecordBatch {
        // (sample_id, chromosome_index, position_start)
        let schema = schemas::sample_ref_blocks();
        let n = rows.len();
        let mut cols: Vec<Arc<dyn Array>> = vec![
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.0))),
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.1))),
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.2))),
            Arc::new(Int32Array::from_iter_values(rows.iter().map(|r| r.2 + 5))),
        ];
        for _ in 4..schema.fields().len() {
            cols.push(Arc::new(Int32Array::from(vec![Some(1); n])));
        }
        RecordBatch::try_new(schema, cols).unwrap()
    }

    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("mosaic_iw_slices_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        d.join(name).to_string_lossy().into_owned()
    }

    #[test]
    fn round_trip_and_index() {
        let p = tmp("rt.slices");
        let b = rb_batch(&[(1, 20, 10), (1, 20, 30), (2, 20, 5), (3, 21, 7)]);
        let written = write_sample_slices(&b, std::path::Path::new(&p)).unwrap();
        let idx = read_index(&p).unwrap();
        assert_eq!(idx, written);
        assert_eq!(
            idx.iter()
                .map(|s| (s.chromosome_index, s.sample_id, s.num_rows))
                .collect::<Vec<_>>(),
            vec![(20, 1, 2), (20, 2, 1), (21, 3, 1)]
        );
        let back = read_all(&p, &schemas::sample_ref_blocks()).unwrap();
        assert_eq!(back, b);
    }

    #[test]
    fn truncated_plain_and_other_version_refused() {
        let p = tmp("trunc.slices");
        let b = rb_batch(&[(1, 20, 10), (2, 20, 5)]);
        write_sample_slices(&b, std::path::Path::new(&p)).unwrap();
        let data = std::fs::read(&p).unwrap();
        // Truncated: the tail cut off.
        let e = parse_index(&p, &data[..data.len() - 3]).unwrap_err();
        assert!(
            format!("{e}").contains("not a sample-slice container"),
            "{e}"
        );
        let e = parse_index(&p, &data[..10]).unwrap_err();
        assert!(format!("{e}").contains("too short"), "{e}");
        // Another version.
        let mut v2 = data.clone();
        let n = v2.len();
        v2[n - 12..n - 8].copy_from_slice(&2u32.to_le_bytes());
        let e = parse_index(&p, &v2).unwrap_err();
        assert!(format!("{e}").contains("version 2"), "{e}");
        // A plain Parquet file.
        let pq = tmp("plain.parquet");
        crate::pqwrite::write_parquet_file(&b, std::path::Path::new(&pq)).unwrap();
        let e = read_index(&pq).unwrap_err();
        assert!(
            format!("{e}").contains("not a sample-slice container"),
            "{e}"
        );
        // A trailer that does not tile the file: bytes inserted before it.
        let idx = parse_index(&p, &data).unwrap();
        let body: usize = idx.iter().map(|s| s.length as usize).sum();
        let mut gap = data[..body].to_vec();
        gap.extend_from_slice(b"xx");
        gap.extend_from_slice(&data[body..]);
        let e = parse_index(&p, &gap).unwrap_err();
        assert!(format!("{e}").contains("trailer starts at"), "{e}");
    }

    #[test]
    fn blob_statistics_must_match_the_trailer() {
        // A container whose trailer names another sample than its blob.
        let p = tmp("liar.slices");
        let b = rb_batch(&[(4, 20, 10)]);
        let sl = write_sample_slices(&b, std::path::Path::new(&p)).unwrap();
        let data = std::fs::read(&p).unwrap();
        let blob = &data[..sl[0].length as usize];
        let trailer = serde_json::to_vec(&serde_json::json!({
            "slices": [[20, 5, 0, sl[0].length, sl[0].nbytes, 1]]
        }))
        .unwrap();
        let liar = tmp("liar2.slices");
        let mut f = File::create(&liar).unwrap();
        f.write_all(blob).unwrap();
        f.write_all(&trailer).unwrap();
        f.write_all(&(trailer.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(SAMPLE_SLICE_MAGIC).unwrap();
        drop(f);
        let idx = read_index(&liar).unwrap();
        let fh = File::open(&liar).unwrap();
        let e = read_slice(&fh, &liar, &idx[0], &schemas::sample_ref_blocks()).unwrap_err();
        assert!(format!("{e}").contains("sample_id statistics"), "{e}");
        // The row count.
        let trailer = serde_json::to_vec(&serde_json::json!({
            "slices": [[20, 4, 0, sl[0].length, sl[0].nbytes, 2]]
        }))
        .unwrap();
        let mut f = File::create(&liar).unwrap();
        f.write_all(blob).unwrap();
        f.write_all(&trailer).unwrap();
        f.write_all(&(trailer.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&1u32.to_le_bytes()).unwrap();
        f.write_all(SAMPLE_SLICE_MAGIC).unwrap();
        drop(f);
        let idx = read_index(&liar).unwrap();
        let fh = File::open(&liar).unwrap();
        let e = read_slice(&fh, &liar, &idx[0], &schemas::sample_ref_blocks()).unwrap_err();
        assert!(format!("{e}").contains("trailer says 2"), "{e}");
    }

    // ── K6: test_sample_major_layout.py's container cases, value for value ──

    /// `_sorted_rows()`: three (chromosome, sample) slices, sorted, with
    /// 5 / 2 / 4 rows at positions 100, 110, ...
    fn python_sorted_rows() -> RecordBatch {
        let mut rows = Vec::new();
        for (c, s, n) in [(0, 3, 5), (0, 7, 2), (1, 3, 4)] {
            for k in 0..n {
                rows.push((s, c, 100 + 10 * k));
            }
        }
        rb_batch(&rows)
    }

    /// A container of `python_sorted_rows()` and its bytes (`_container`).
    fn python_container(name: &str) -> (String, Vec<u8>) {
        let p = tmp(name);
        write_sample_slices(&python_sorted_rows(), std::path::Path::new(&p)).unwrap();
        let data = std::fs::read(&p).unwrap();
        (p, data)
    }

    /// `_rewrite_trailer`: the container with its trailer's slice entries
    /// edited (the blobs and the tail's version and magic kept).
    fn rewrite_trailer(p: &str, data: &[u8], edit: impl Fn(&mut Vec<serde_json::Value>)) {
        let n = data.len();
        let tail = &data[n - TAIL_SIZE as usize..];
        let trailer_len = u64::from_le_bytes(tail[0..8].try_into().unwrap()) as usize;
        let body = n - TAIL_SIZE as usize - trailer_len;
        let mut trailer: serde_json::Value =
            serde_json::from_slice(&data[body..body + trailer_len]).unwrap();
        edit(trailer["slices"].as_array_mut().unwrap());
        let new = serde_json::to_vec(&trailer).unwrap();
        let mut f = File::create(p).unwrap();
        f.write_all(&data[..body]).unwrap();
        f.write_all(&new).unwrap();
        f.write_all(&(new.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&tail[8..]).unwrap();
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_container_round_trip_and_trailer_index (K6).
    #[test]
    fn container_round_trip_and_trailer_against_the_blobs() {
        let (p, data) = python_container("k6_rt.slices");
        let b = python_sorted_rows();
        let schema = schemas::sample_ref_blocks();
        let idx = read_index(&p).unwrap();
        assert_eq!(
            idx,
            write_sample_slices(&b, std::path::Path::new(&tmp("k6_rt2.slices"))).unwrap()
        );
        assert_eq!(
            idx.iter()
                .map(|s| (s.chromosome_index, s.sample_id, s.num_rows))
                .collect::<Vec<_>>(),
            vec![(0, 3, 5), (0, 7, 2), (1, 3, 4)]
        );
        assert_eq!(read_all(&p, &schema).unwrap(), b);
        // The tail: trailer length, version, magic.
        let n = data.len();
        let tail_len = u64::from_le_bytes(data[n - 20..n - 12].try_into().unwrap());
        let version = u32::from_le_bytes(data[n - 12..n - 8].try_into().unwrap());
        assert_eq!(version, SAMPLE_SLICE_VERSION);
        assert_eq!(&data[n - 8..], SAMPLE_SLICE_MAGIC);
        // The blobs tile the body from byte 0 up to the trailer.
        assert_eq!(idx[0].offset, 0);
        for w in idx.windows(2) {
            assert_eq!(w[1].offset, w[0].offset + w[0].length);
        }
        let last = idx.last().unwrap();
        assert_eq!(last.offset + last.length, n as u64 - TAIL_SIZE - tail_len);
        // Each blob a standalone one-row-group Parquet file of the stated
        // rows and compressed column-chunk bytes, decoding as read_slice.
        let fh = File::open(&p).unwrap();
        for (k, sl) in idx.iter().enumerate() {
            let blob = &data[sl.offset as usize..(sl.offset + sl.length) as usize];
            assert_eq!(&blob[..4], b"PAR1");
            assert_eq!(&blob[blob.len() - 4..], b"PAR1");
            let bp = tmp(&format!("k6_blob{k}.parquet"));
            std::fs::write(&bp, blob).unwrap();
            let builder =
                ParquetRecordBatchReaderBuilder::try_new(File::open(&bp).unwrap()).unwrap();
            let md = builder.metadata().clone();
            assert_eq!(md.num_row_groups(), 1);
            assert_eq!(md.file_metadata().num_rows() as usize, sl.num_rows);
            let nbytes: i64 = md
                .row_group(0)
                .columns()
                .iter()
                .map(|c| c.compressed_size())
                .sum();
            assert_eq!(sl.nbytes, nbytes);
            let parts: Vec<RecordBatch> = builder
                .build()
                .unwrap()
                .map(|b| {
                    RecordBatch::try_new(schema.clone(), b.unwrap().columns().to_vec()).unwrap()
                })
                .collect();
            assert_eq!(
                read_slice(&fh, &p, sl, &schema).unwrap(),
                concat_batches(&schema, &parts).unwrap()
            );
        }
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_container_of_site_info_rows (K6).
    #[test]
    fn container_of_site_info_rows() {
        use arrow::array::{ListBuilder, StringBuilder};
        let schema = schemas::sample_site_info();
        let mut cols: Vec<Arc<dyn Array>> = Vec::new();
        for f in schema.fields() {
            let a: Arc<dyn Array> = match f.name().as_str() {
                "sample_id" => Arc::new(Int32Array::from(vec![1, 1])),
                "chromosome_index" => Arc::new(Int32Array::from(vec![0, 0])),
                "position" => Arc::new(Int32Array::from(vec![5, 9])),
                "gt_1" => Arc::new(Int32Array::from(vec![0, 1])),
                "alleles" => {
                    let mut b = ListBuilder::new(StringBuilder::new()).with_field(Arc::new(
                        arrow::datatypes::Field::new(
                            schemas::LIST_ITEM,
                            arrow::datatypes::DataType::Utf8,
                            true,
                        ),
                    ));
                    for row in [["A", "xT"], ["C", "-CA"]] {
                        for a in row {
                            b.values().append_value(a);
                        }
                        b.append(true);
                    }
                    Arc::new(b.finish())
                }
                _ => arrow::array::new_null_array(f.data_type(), 2),
            };
            cols.push(a);
        }
        let table = RecordBatch::try_new(schema.clone(), cols).unwrap();
        let p = tmp("k6_si.slices");
        let sl = write_sample_slices(&table, std::path::Path::new(&p)).unwrap();
        assert_eq!(sl.len(), 1);
        let fh = File::open(&p).unwrap();
        assert_eq!(read_slice(&fh, &p, &sl[0], &schema).unwrap(), table);
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_truncated_container_is_refused,
    // ::test_short_file_is_refused, ::test_plain_parquet_file_is_refused and
    // ::test_other_version_is_refused (K6).
    #[test]
    fn truncated_short_plain_and_other_version_refused_python_cases() {
        let (p, data) = python_container("k6_trunc.slices");
        for cut in [1usize, TAIL_SIZE as usize, 100] {
            let e = parse_index(&p, &data[..data.len() - cut]).unwrap_err();
            assert!(
                format!("{e}").contains("not a sample-slice container"),
                "cut {cut}: {e}"
            );
        }
        let e = parse_index(&p, b"PAR1").unwrap_err();
        assert!(format!("{e}").contains("too short"), "{e}");
        let plain = tmp("k6_plain.parquet");
        crate::pqwrite::write_parquet_file(&python_sorted_rows(), std::path::Path::new(&plain))
            .unwrap();
        let e = read_index(&plain).unwrap_err();
        assert!(format!("{e}").contains("a plain Parquet file"), "{e}");
        let mut other = data.clone();
        let n = other.len();
        other[n - 12..n - 8].copy_from_slice(&(SAMPLE_SLICE_VERSION + 1).to_le_bytes());
        let e = parse_index(&p, &other).unwrap_err();
        assert!(format!("{e}").contains("version"), "{e}");
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_trailer_that_does_not_tile_the_blobs_is_refused (K6).
    #[test]
    fn trailer_without_its_middle_entry_is_refused() {
        let (p, data) = python_container("k6_tile.slices");
        rewrite_trailer(&p, &data, |e| {
            e.remove(1);
        });
        let e = read_index(&p).unwrap_err();
        assert!(format!("{e}").contains("expected offset"), "{e}");
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_trailer_out_of_order_is_refused (K6).
    #[test]
    fn trailer_out_of_order_is_refused() {
        let (p, data) = python_container("k6_order.slices");
        rewrite_trailer(&p, &data, |e| {
            let a = e[0][1].clone();
            e[0][1] = e[1][1].clone();
            e[1][1] = a;
        });
        let e = read_index(&p).unwrap_err();
        assert!(format!("{e}").contains("not in"), "{e}");
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_blob_statistics_must_match_the_trailer (K6).
    #[test]
    fn relabelled_blob_refused_when_read() {
        let (p, data) = python_container("k6_relabel.slices");
        rewrite_trailer(&p, &data, |e| {
            e[1][1] = serde_json::json!(8); // sample 7's blob claimed as sample 8
        });
        let idx = read_index(&p).unwrap(); // still in order: 3, 8 | 3
        assert_eq!(
            idx.iter()
                .map(|s| (s.chromosome_index, s.sample_id))
                .collect::<Vec<_>>(),
            vec![(0, 3), (0, 8), (1, 3)]
        );
        let fh = File::open(&p).unwrap();
        let schema = schemas::sample_ref_blocks();
        read_slice(&fh, &p, &idx[0], &schema).unwrap();
        let e = read_slice(&fh, &p, &idx[1], &schema).unwrap_err();
        assert!(format!("{e}").contains("sample_id statistics"), "{e}");
    }
}
