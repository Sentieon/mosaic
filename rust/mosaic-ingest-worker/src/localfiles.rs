// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The shard's local files: the per-sample sample-slice containers
//! (`_write_parquet_local`), the dense `batch_ref_blocks` windows
//! (`_write_brb_window_local`), the whole-shard partials
//! (`_write_batch_partials_local`) and `batch_genotypes`
//! (`_write_batch_genotypes_local`; the names are the Python worker's
//! writers this was ported from), with their rows' sort keys and
//! null-versus-empty rules. An integer that does not fit its int32 column
//! is an error, as `pa.Table.from_pydict` raises.

use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use anyhow::{Context, Result, anyhow};
use arrow::array::{
    ArrayRef, BinaryBuilder, Float64Array, Float64Builder, Int32Array, Int32Builder, Int64Array,
    Int64Builder, ListBuilder, RecordBatch, StringArray, StringBuilder,
};
use arrow::datatypes::{DataType, Field};
use parquet::arrow::ArrowWriter;
use sha2::{Digest, Sha256};

use crate::matrix::BgtRow;
use crate::partials::{BacRow, BsiRow, SiOut};
use crate::pqwrite::{BROTLI_LEVEL, bgt_writer_props, write_parquet_file, write_sample_slices};
use crate::ref_blocks::RbRow;
use crate::windows::{Segments, sum_segments};
use mosaic_core::schemas;

/// `os.path.join(dir, name)`.
pub fn join(dir: &str, name: &str) -> String {
    if dir.is_empty() || dir.ends_with('/') {
        format!("{dir}{name}")
    } else {
        format!("{dir}/{name}")
    }
}

/// `_batch_key(sample_names, c_idx, start_pos)`: 24 hex characters of
/// SHA-256 over `c{c_idx}:{sorted names}@{start}`.
pub fn batch_key(sorted_names: &[String], c_idx: i64, start: i64) -> String {
    let content = format!("c{c_idx}:{}@{start}", sorted_names.join(","));
    let d = Sha256::digest(content.as_bytes());
    let mut s = String::with_capacity(64);
    for b in d.iter() {
        s.push_str(&format!("{b:02x}"));
    }
    s.truncate(24);
    s
}

fn i32_of(x: i64, what: &str) -> Result<i32> {
    i32::try_from(x).map_err(|_| anyhow!("{what} = {x} does not fit int32"))
}

fn i32s(v: impl Iterator<Item = Result<i32>>) -> Result<ArrayRef> {
    Ok(Arc::new(Int32Array::from(v.collect::<Result<Vec<i32>>>()?)))
}

fn opt_i32s(v: impl Iterator<Item = Result<Option<i32>>>) -> Result<ArrayRef> {
    Ok(Arc::new(Int32Array::from(
        v.collect::<Result<Vec<Option<i32>>>>()?,
    )))
}

fn item(t: DataType) -> Arc<Field> {
    Arc::new(Field::new(mosaic_core::schemas::LIST_ITEM, t, true))
}

/// A list<int32> column; `None` rows are null.
fn list_i32<'a>(rows: impl Iterator<Item = Option<&'a [i64]>>, what: &str) -> Result<ArrayRef> {
    let mut b = ListBuilder::new(Int32Builder::new()).with_field(item(DataType::Int32));
    for r in rows {
        match r {
            Some(xs) => {
                for &x in xs {
                    b.values().append_value(i32_of(x, what)?);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Ok(Arc::new(b.finish()))
}

fn list_i64<'a>(rows: impl Iterator<Item = &'a [i64]>) -> ArrayRef {
    let mut b = ListBuilder::new(Int64Builder::new()).with_field(item(DataType::Int64));
    for xs in rows {
        for &x in xs {
            b.values().append_value(x);
        }
        b.append(true);
    }
    Arc::new(b.finish())
}

fn list_f64<'a>(rows: impl Iterator<Item = &'a [f64]>) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new()).with_field(item(DataType::Float64));
    for xs in rows {
        for &x in xs {
            b.values().append_value(x);
        }
        b.append(true);
    }
    Arc::new(b.finish())
}

fn list_str<'a>(rows: impl Iterator<Item = &'a [String]>) -> ArrayRef {
    let mut b = ListBuilder::new(StringBuilder::new()).with_field(item(DataType::Utf8));
    for xs in rows {
        for x in xs {
            b.values().append_value(x);
        }
        b.append(true);
    }
    Arc::new(b.finish())
}

/// A `sample_ref_blocks` container from a window's rows (sorted here,
/// stably, by (chromosome_index, sample_id, position_start)).
pub fn write_rb_container(rows: &mut [RbRow], path: &str) -> Result<()> {
    rows.sort_by_key(|r| (r.c_idx, r.sample_id, r.start));
    let w = "sample_ref_blocks";
    let col = |f: fn(&RbRow) -> i64| i32s(rows.iter().map(|r| i32_of(f(r), w)));
    let batch = RecordBatch::try_new(
        schemas::sample_ref_blocks(),
        vec![
            col(|r| r.sample_id)?,
            col(|r| r.c_idx)?,
            col(|r| r.start)?,
            col(|r| r.end)?,
            col(|r| r.ad_1)?,
            col(|r| r.ad_2)?,
            col(|r| r.dp)?,
            col(|r| r.pl_1)?,
            col(|r| r.pl_2)?,
            col(|r| r.pl_3)?,
            col(|r| r.sc)?,
        ],
    )?;
    write_sample_slices(&batch, Path::new(path))?;
    Ok(())
}

/// A `sample_site_info` container from a window's rows (sorted here,
/// stably, by (chromosome_index, sample_id, position)).
pub fn write_si_container(rows: &mut [SiOut], path: &str) -> Result<()> {
    rows.sort_by_key(|r| (r.c_idx, r.sample_id, r.pos));
    let w = "sample_site_info";
    let opt = |f: fn(&SiOut) -> Option<i64>| {
        opt_i32s(rows.iter().map(|r| f(r).map(|x| i32_of(x, w)).transpose()))
    };
    let req = |f: fn(&SiOut) -> i64| i32s(rows.iter().map(|r| i32_of(f(r), w)));
    let batch = RecordBatch::try_new(
        schemas::sample_site_info(),
        vec![
            req(|r| r.sample_id)?,
            req(|r| r.c_idx)?,
            req(|r| r.pos)?,
            opt(|r| r.del_start)?,
            opt(|r| r.del_end)?,
            list_str(rows.iter().map(|r| r.alleles.as_slice())),
            opt(|r| r.gt_1)?,
            opt(|r| r.gt_2)?,
            opt(|r| Some(r.dp))?,
            list_i32(rows.iter().map(|r| Some(r.ad.as_slice())), w)?,
            list_i32(rows.iter().map(|r| Some(r.f1r2.as_slice())), w)?,
            list_i32(rows.iter().map(|r| Some(r.f2r1.as_slice())), w)?,
            list_i32(rows.iter().map(|r| Some(&r.sb[..])), w)?,
            list_i32(rows.iter().map(|r| Some(r.pl.as_slice())), w)?,
            Arc::new(Float64Array::from_iter_values(rows.iter().map(|r| r.qual))),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.site_dp))),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|r| r.mqsq_1),
            )),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.mqsq_2))),
            list_f64(rows.iter().map(|r| &r.gtc[..])),
            list_i32(rows.iter().map(|r| r.del_ref_spans.as_deref()), w)?,
        ],
    )?;
    write_sample_slices(&batch, Path::new(path))?;
    Ok(())
}

/// `_write_brb_window_local`: the window's dense rows, or None when it
/// sums to nothing. Values are cast to int32 as numpy's `astype` does
/// (wrapping).
pub fn write_brb_window(seg: &Segments, batch_id: i64, path: &str) -> Result<Option<String>> {
    let (pos, dp, sc) = sum_segments(seg);
    let n = pos.len();
    if n == 0 {
        return Ok(None);
    }
    let batch = RecordBatch::try_new(
        schemas::batch_ref_blocks(),
        vec![
            Arc::new(Int32Array::from(vec![batch_id as i32; n])),
            Arc::new(Int32Array::from(vec![seg.c_idx as i32; n])),
            Arc::new(Int32Array::from_iter_values(pos.iter().map(|&x| x as i32))),
            Arc::new(Int32Array::from_iter_values(dp.iter().map(|&x| x as i32))),
            Arc::new(Int32Array::from_iter_values(sc.iter().map(|&x| x as i32))),
        ],
    )?;
    write_parquet_file(&batch, Path::new(path))?;
    Ok(Some(path.to_string()))
}

/// `batch_site_info_{shard_id}.parquet`, rows sorted (c_idx, position).
pub fn write_bsi(rows: &mut [BsiRow], batch_id: i64, path: &str) -> Result<()> {
    rows.sort_by_key(|r| (r.c_idx, r.pos));
    let w = "batch_site_info";
    let batch = RecordBatch::try_new(
        schemas::batch_site_info(),
        vec![
            i32s(rows.iter().map(|_| i32_of(batch_id, w)))?,
            i32s(rows.iter().map(|r| i32_of(r.c_idx, w)))?,
            i32s(rows.iter().map(|r| i32_of(r.pos, w)))?,
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|r| r.si.qual),
            )),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.si.dp))),
            Arc::new(Float64Array::from_iter_values(
                rows.iter().map(|r| r.si.mq0),
            )),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.si.mq1))),
            list_i64(rows.iter().map(|r| &r.si.sb[..])),
            list_i64(rows.iter().map(|r| &r.si.gtc[..])),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|r| r.n_records),
            )),
        ],
    )?;
    write_parquet_file(&batch, Path::new(path))?;
    Ok(())
}

/// `batch_allele_counts_{shard_id}.parquet`, rows sorted (c_idx,
/// position, allele) (Python's `str` order is the code-point order, as
/// UTF-8 bytes compare).
pub fn write_bac(rows: &mut [BacRow], batch_id: i64, path: &str) -> Result<()> {
    rows.sort_by(|a, b| (a.c_idx, a.pos, &a.allele).cmp(&(b.c_idx, b.pos, &b.allele)));
    let w = "batch_allele_counts";
    let batch = RecordBatch::try_new(
        schemas::batch_allele_counts(),
        vec![
            i32s(rows.iter().map(|_| i32_of(batch_id, w)))?,
            i32s(rows.iter().map(|r| i32_of(r.c_idx, w)))?,
            i32s(rows.iter().map(|r| i32_of(r.pos, w)))?,
            Arc::new(StringArray::from_iter_values(
                rows.iter().map(|r| r.allele.as_str()),
            )),
            Arc::new(Int64Array::from_iter_values(rows.iter().map(|r| r.count))),
            Arc::new(Int64Array::from_iter_values(
                rows.iter().map(|r| r.n_listed),
            )),
        ],
    )?;
    write_parquet_file(&batch, Path::new(path))?;
    Ok(())
}

/// `batch_genotypes_{shard_id}.parquet` (`BGT_PARQUET_KWARGS`: statistics
/// on the four scalar key columns only), rows sorted (c_idx, position).
pub fn write_bgt(rows: &mut [BgtRow], batch_id: i64, c_idx: i64, path: &str) -> Result<()> {
    rows.sort_by_key(|r| r.pos);
    let w = "batch_genotypes";
    let mut g1 = BinaryBuilder::new();
    let mut g2 = BinaryBuilder::new();
    let mut gq = BinaryBuilder::new();
    for r in rows.iter() {
        g1.append_value(&r.gt_1);
        g2.append_value(&r.gt_2);
        gq.append_value(&r.gq);
    }
    let batch = RecordBatch::try_new(
        schemas::batch_genotypes(),
        vec![
            i32s(rows.iter().map(|_| i32_of(batch_id, w)))?,
            i32s(rows.iter().map(|_| i32_of(c_idx, w)))?,
            i32s(rows.iter().map(|r| i32_of(r.pos, w)))?,
            i32s(rows.iter().map(|r| i32_of(r.n_slots, w)))?,
            list_str(rows.iter().map(|r| r.alleles.as_slice())),
            Arc::new(g1.finish()),
            Arc::new(g2.finish()),
            Arc::new(gq.finish()),
        ],
    )?;
    let props = bgt_writer_props(BROTLI_LEVEL)?;
    let file = File::create(path).with_context(|| format!("create {path}"))?;
    let mut wr = ArrowWriter::try_new(file, batch.schema(), Some(props))?;
    wr.write(&batch)?;
    wr.close()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_and_join() {
        // _batch_key(["HG2", "HG1"], 20, 0) in Python.
        let names = vec!["HG1".to_string(), "HG2".to_string()];
        let want = {
            let d = Sha256::digest(b"c20:HG1,HG2@0");
            d.iter().map(|b| format!("{b:02x}")).collect::<String>()[..24].to_string()
        };
        assert_eq!(batch_key(&names, 20, 0), want);
        assert_eq!(batch_key(&names, 20, 0).len(), 24);
        assert_eq!(join("/tmp/x", "a"), "/tmp/x/a");
        assert_eq!(join("/tmp/x/", "a"), "/tmp/x/a");
    }

    use crate::slices::{read_index, read_slice};
    use arrow::array::Array;

    fn tmp(name: &str) -> String {
        let d = std::env::temp_dir().join(format!("mosaic_iw_localfiles_{}", std::process::id()));
        std::fs::create_dir_all(&d).unwrap();
        join(d.to_str().unwrap(), name)
    }

    /// `_rb_row(sid, c, start)` of test_sample_major_layout.py.
    fn rb_row(sample_id: i64, c_idx: i64, start: i64) -> RbRow {
        RbRow {
            sample_id,
            c_idx,
            start,
            end: start + 4,
            ad_1: 0,
            ad_2: 0,
            dp: 10,
            pl_1: 0,
            pl_2: 3,
            pl_3: 30,
            sc: 1,
        }
    }

    fn i32_values(b: &RecordBatch, name: &str) -> Vec<i32> {
        b.column_by_name(name)
            .unwrap()
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap()
            .values()
            .to_vec()
    }

    fn keys(path: &str) -> Vec<(i32, i32)> {
        read_index(path)
            .unwrap()
            .iter()
            .map(|s| (s.chromosome_index, s.sample_id))
            .collect()
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_local_writer_one_blob_per_sample (K6).
    #[test]
    fn local_writer_one_blob_per_sample() {
        // random.Random(1).shuffle of [(s, p) for s in (5, 2, 9) for p in
        // (300, 100, 200)] in Python.
        let mut rows: Vec<RbRow> = [
            (2, 200),
            (9, 300),
            (9, 100),
            (2, 100),
            (2, 300),
            (5, 300),
            (9, 200),
            (5, 100),
            (5, 200),
        ]
        .iter()
        .map(|&(s, p)| rb_row(s, 0, p))
        .collect();
        let p = tmp("k6_one_blob.slices");
        write_rb_container(&mut rows, &p).unwrap();
        assert_eq!(keys(&p), vec![(0, 2), (0, 5), (0, 9)]);
        let f = File::open(&p).unwrap();
        for sl in read_index(&p).unwrap() {
            // read_slice checks min == max of both keys' statistics.
            let t = read_slice(&f, &p, &sl, &schemas::sample_ref_blocks()).unwrap();
            assert_eq!(i32_values(&t, "position_start"), vec![100, 200, 300]);
            assert_eq!(i32_values(&t, "sample_id"), vec![sl.sample_id; 3]);
        }
    }

    fn si_row(sample_id: i64, pos: i64, gt_1: i64) -> SiOut {
        SiOut {
            sample_id,
            c_idx: 0,
            pos,
            del_start: None,
            del_end: None,
            alleles: vec!["A".into(), "xT".into()],
            gt_1: Some(gt_1),
            gt_2: None,
            dp: 0,
            ad: vec![],
            f1r2: vec![],
            f2r1: vec![],
            sb: [0; 4],
            pl: vec![],
            qual: 0.0,
            site_dp: 0,
            mqsq_1: 0.0,
            mqsq_2: 0,
            gtc: [0.0; 3],
            del_ref_spans: None,
        }
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_local_writer_keeps_a_samples_same_position_rows_in_order (K6).
    #[test]
    fn local_writer_keeps_a_samples_same_position_rows_in_order() {
        // Two site rows of sample 4 at one position (the same-site
        // re-queue): their relative order is the one bulk_upload produced.
        let mut rows = vec![si_row(4, 50, 1), si_row(1, 50, 0), si_row(4, 50, 0)];
        let p = tmp("k6_same_pos.slices");
        write_si_container(&mut rows, &p).unwrap();
        let t = crate::slices::read_all(&p, &schemas::sample_site_info()).unwrap();
        assert_eq!(i32_values(&t, "sample_id"), vec![1, 4, 4]);
        assert_eq!(i32_values(&t, "gt_1"), vec![0, 1, 0]);
    }

    // Replaces tests/unit/test_sample_major_layout.py::test_local_writer_sentinel_window_with_two_chromosomes (K6).
    #[test]
    fn local_writer_window_with_two_chromosomes() {
        // pos_batch_size 0: one window holds every chromosome; blobs per
        // (chromosome, sample).
        let mut rows = vec![rb_row(2, 1, 10), rb_row(1, 0, 10), rb_row(2, 0, 10)];
        let p = tmp("k6_two_chroms.slices");
        write_rb_container(&mut rows, &p).unwrap();
        assert_eq!(keys(&p), vec![(0, 1), (0, 2), (1, 2)]);
    }

    // Replaces tests/unit/test_removal_substrate.py::test_local_sample_ref_blocks_has_sample_counts (K6).
    #[test]
    fn local_sample_ref_blocks_has_sample_counts() {
        let mut rows = vec![RbRow {
            sample_id: 5,
            c_idx: 0,
            start: 10,
            end: 20,
            ad_1: 1,
            ad_2: 2,
            dp: 30,
            pl_1: 0,
            pl_2: 3,
            pl_3: 30,
            sc: 1,
        }];
        let p = tmp("k6_sc.slices");
        write_rb_container(&mut rows, &p).unwrap();
        let t = crate::slices::read_all(&p, &schemas::sample_ref_blocks()).unwrap();
        assert_eq!(t.schema(), schemas::sample_ref_blocks());
        assert_eq!(i32_values(&t, "sample_counts"), vec![1]);
        assert!(!t.column_by_name("sample_counts").unwrap().is_null(0));
    }
}
