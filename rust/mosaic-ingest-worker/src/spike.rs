// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! `--spike <dir>`: a Parquet round-trip check. Writes crafted rows in
//! each schema, for the Python readers (pyarrow, the sample-slice reader)
//! to read back and compare schemas and values.
//!
//! Files: `ref_blocks.slices` (two `sample_ref_blocks` slices),
//! `site_info.slices` (two `sample_site_info` slices, with nulls and empty
//! lists), `batch_site_info.parquet`, `batch_allele_counts.parquet`.

use std::path::Path;
use std::sync::Arc;

use anyhow::Result;
use arrow::array::{
    ArrayRef, Float64Array, Float64Builder, Int32Array, Int32Builder, Int64Array, Int64Builder,
    ListBuilder, RecordBatch, StringArray, StringBuilder,
};
use arrow::datatypes::{DataType, Field};

use crate::pqwrite::{write_parquet_file, write_sample_slices};
use mosaic_core::schemas;

fn i32s(v: &[Option<i32>]) -> ArrayRef {
    Arc::new(Int32Array::from(v.to_vec()))
}

fn item(t: DataType) -> Arc<Field> {
    Arc::new(Field::new(mosaic_core::schemas::LIST_ITEM, t, true))
}

fn list_i32(v: &[Option<Vec<i32>>]) -> ArrayRef {
    let mut b = ListBuilder::new(Int32Builder::new()).with_field(item(DataType::Int32));
    for row in v {
        match row {
            Some(xs) => {
                for &x in xs {
                    b.values().append_value(x);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

fn list_i64(v: &[Option<Vec<i64>>]) -> ArrayRef {
    let mut b = ListBuilder::new(Int64Builder::new()).with_field(item(DataType::Int64));
    for row in v {
        match row {
            Some(xs) => {
                for &x in xs {
                    b.values().append_value(x);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

fn list_f64(v: &[Option<Vec<f64>>]) -> ArrayRef {
    let mut b = ListBuilder::new(Float64Builder::new()).with_field(item(DataType::Float64));
    for row in v {
        match row {
            Some(xs) => {
                for &x in xs {
                    b.values().append_value(x);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

fn list_str(v: &[Option<Vec<&str>>]) -> ArrayRef {
    let mut b = ListBuilder::new(StringBuilder::new()).with_field(item(DataType::Utf8));
    for row in v {
        match row {
            Some(xs) => {
                for &x in xs {
                    b.values().append_value(x);
                }
                b.append(true);
            }
            None => b.append(false),
        }
    }
    Arc::new(b.finish())
}

pub fn run(dir: &Path) -> Result<()> {
    std::fs::create_dir_all(dir)?;

    // sample_ref_blocks: slice (0, 3) with two rows, slice (0, 7) one row.
    let rb = RecordBatch::try_new(
        schemas::sample_ref_blocks(),
        vec![
            i32s(&[Some(3), Some(3), Some(7)]),
            i32s(&[Some(0), Some(0), Some(0)]),
            i32s(&[Some(100), Some(151), Some(90)]),
            i32s(&[Some(150), Some(151), Some(200)]),
            i32s(&[Some(10), None, Some(5)]),
            i32s(&[Some(0), None, Some(1)]),
            i32s(&[Some(12), Some(0), Some(6)]),
            i32s(&[Some(0), None, Some(0)]),
            i32s(&[Some(30), None, Some(15)]),
            i32s(&[Some(450), None, Some(200)]),
            i32s(&[Some(1), Some(0), Some(1)]),
        ],
    )?;
    let s = write_sample_slices(&rb, &dir.join("ref_blocks.slices"))?;
    eprintln!("ref_blocks.slices: {s:?}");

    // sample_site_info: slice (0, 3) two rows, slice (0, 7) one row.
    let si = RecordBatch::try_new(
        schemas::sample_site_info(),
        vec![
            i32s(&[Some(3), Some(3), Some(7)]),
            i32s(&[Some(0), Some(0), Some(0)]),
            i32s(&[Some(120), Some(130), Some(120)]),
            i32s(&[None, Some(130), None]),
            i32s(&[None, Some(133), None]),
            list_str(&[
                Some(vec!["A", "xG", "*"]),
                Some(vec!["C", "-CTA", "*"]),
                Some(vec!["A", "xT"]),
            ]),
            i32s(&[Some(0), None, Some(1)]),
            i32s(&[Some(1), None, Some(1)]),
            i32s(&[Some(20), None, Some(8)]),
            list_i32(&[Some(vec![10, 10, 0]), Some(vec![]), Some(vec![0, 8])]),
            list_i32(&[Some(vec![]), Some(vec![]), Some(vec![0, 4])]),
            list_i32(&[Some(vec![]), Some(vec![]), Some(vec![0, 4])]),
            list_i32(&[
                Some(vec![1, 2, 3, 4]),
                Some(vec![0, 0, 0, 0]),
                Some(vec![0, 0, 4, 4]),
            ]),
            list_i32(&[
                Some(vec![100, 0, 200, 300, 400, 500]),
                Some(vec![]),
                Some(vec![200, 24, 0]),
            ]),
            Arc::new(Float64Array::from(vec![Some(50.5), None, Some(1e-300)])),
            Arc::new(Int64Array::from(vec![Some(21), Some(0), Some(9)])),
            Arc::new(Float64Array::from(vec![
                Some(72000.0),
                Some(0.0),
                Some(1.25),
            ])),
            Arc::new(Int64Array::from(vec![Some(20), Some(0), Some(9)])),
            list_f64(&[
                Some(vec![0.0, 1.0, 0.0]),
                Some(vec![0.1, 0.2, 0.7]),
                Some(vec![0.0, 0.0, 0.9999999999]),
            ]),
            list_i32(&[None, Some(vec![131, 133]), Some(vec![])]),
        ],
    )?;
    let s = write_sample_slices(&si, &dir.join("site_info.slices"))?;
    eprintln!("site_info.slices: {s:?}");

    let bsi = RecordBatch::try_new(
        schemas::batch_site_info(),
        vec![
            i32s(&[Some(5), Some(5)]),
            i32s(&[Some(0), Some(0)]),
            i32s(&[Some(120), Some(130)]),
            Arc::new(Float64Array::from(vec![Some(51.5), None])),
            Arc::new(Int64Array::from(vec![Some(41), None])),
            Arc::new(Float64Array::from(vec![Some(3600.5), None])),
            Arc::new(Int64Array::from(vec![Some(30), None])),
            list_i64(&[Some(vec![1, 2, 3, 4]), None]),
            list_i64(&[Some(vec![0, 4294967296, 1 << 40]), None]),
            Arc::new(Int64Array::from(vec![Some(2), None])),
        ],
    )?;
    write_parquet_file(&bsi, &dir.join("batch_site_info.parquet"))?;

    let bac = RecordBatch::try_new(
        schemas::batch_allele_counts(),
        vec![
            i32s(&[Some(5), Some(5), Some(5)]),
            i32s(&[Some(0), Some(0), Some(0)]),
            i32s(&[Some(120), Some(120), Some(130)]),
            Arc::new(StringArray::from(vec!["A", "xG", "-CTA"])),
            Arc::new(Int64Array::from(vec![Some(1), Some(1), Some(0)])),
            Arc::new(Int64Array::from(vec![Some(2), Some(1), Some(1)])),
        ],
    )?;
    write_parquet_file(&bac, &dir.join("batch_allele_counts.parquet"))?;
    eprintln!("spike written to {}", dir.display());
    Ok(())
}
