// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The per-sample rows the calling logic uses, decoded from the merged
//! batches of the streaming reader (`reader.rs`): a `sample_site_info` row
//! (the Python engine's `sample_gts` / `del_site_samples` tuple, with the
//! row's own allele strings instead of window ids) and a
//! `sample_ref_blocks` row.

use anyhow::{Result, anyhow};
use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::Int32Type;

/// A `sample_site_info` row as the calling logic reads it. A null list
/// element stays `None` here; the call functions apply the Python
/// engine's fills (`_fill_nulls`).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct SiteRow {
    pub sample_id: i32,
    /// The record's own allele list (ref, its ALTs, `*` for <NON_REF>).
    pub alleles: Option<Vec<String>>,
    /// Local indices into `alleles` (None: no call).
    pub gt_1: Option<i32>,
    pub gt_2: Option<i32>,
    pub dp: Option<i32>,
    pub ad: Option<Vec<Option<i32>>>,
    pub f1r2: Option<Vec<Option<i32>>>,
    pub f2r1: Option<Vec<Option<i32>>>,
    pub pl: Option<Vec<Option<i32>>>,
}

impl SiteRow {
    /// The allele string a local GT index names (`_local_gt_id`: None for
    /// a no call, a missing list or an index outside it).
    pub fn gt_allele(&self, gt: Option<i32>) -> Option<&str> {
        let a = self.alleles.as_ref()?;
        let i = usize::try_from(gt?).ok()?;
        a.get(i).map(String::as_str)
    }
}

/// A `sample_ref_blocks` row (`[start, end]`, 1-based inclusive).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Block {
    pub sample_id: i32,
    pub start: i32,
    pub end: i32,
    pub ad_1: Option<i32>,
    pub ad_2: Option<i32>,
    pub dp: Option<i32>,
    pub pl_1: Option<i32>,
    pub pl_2: Option<i32>,
    pub pl_3: Option<i32>,
    /// 1 a hom-ref block, 0 a `./.` block (E1; null on no E1 lake).
    pub sample_counts: Option<i32>,
}

fn column<'a>(b: &'a RecordBatch, name: &str) -> Result<&'a dyn Array> {
    b.column_by_name(name)
        .map(|c| c.as_ref())
        .ok_or_else(|| anyhow!("no column {name} in a decoded batch"))
}

fn int(b: &RecordBatch, name: &str, r: usize) -> Result<Option<i32>> {
    let a = column(b, name)?;
    Ok((!a.is_null(r)).then(|| a.as_primitive::<Int32Type>().value(r)))
}

fn int_list(b: &RecordBatch, name: &str, r: usize) -> Result<Option<Vec<Option<i32>>>> {
    let a = column(b, name)?.as_list::<i32>();
    if a.is_null(r) {
        return Ok(None);
    }
    let v = a.value(r);
    let v = v.as_primitive::<Int32Type>();
    Ok(Some(
        (0..v.len())
            .map(|i| (!v.is_null(i)).then(|| v.value(i)))
            .collect(),
    ))
}

/// Decode row `r` of a `sample_site_info` batch.
pub fn site_row(b: &RecordBatch, r: usize) -> Result<SiteRow> {
    let alleles = {
        let a = column(b, "alleles")?.as_list::<i32>();
        if a.is_null(r) {
            None
        } else {
            let v = a.value(r);
            let v = v.as_string::<i32>();
            // A null element is no allele string; "" never matches one.
            Some(
                (0..v.len())
                    .map(|i| {
                        if v.is_null(i) {
                            String::new()
                        } else {
                            v.value(i).to_string()
                        }
                    })
                    .collect(),
            )
        }
    };
    Ok(SiteRow {
        sample_id: int(b, "sample_id", r)?.ok_or_else(|| anyhow!("a null sample_id"))?,
        alleles,
        gt_1: int(b, "gt_1", r)?,
        gt_2: int(b, "gt_2", r)?,
        dp: int(b, "dp", r)?,
        ad: int_list(b, "ad", r)?,
        f1r2: int_list(b, "f1r2", r)?,
        f2r1: int_list(b, "f2r1", r)?,
        pl: int_list(b, "pl", r)?,
    })
}

/// A `sample_ref_blocks` batch's columns, typed once per batch.
pub struct BlockCols {
    cols: [arrow::array::Int32Array; 10],
}

const BLOCK_COLUMNS: [&str; 10] = [
    "sample_id",
    "position_start",
    "position_end",
    "ad_1",
    "ad_2",
    "dp",
    "pl_1",
    "pl_2",
    "pl_3",
    "sample_counts",
];

impl BlockCols {
    pub fn new(b: &RecordBatch) -> Result<BlockCols> {
        let mut cols: Vec<arrow::array::Int32Array> = Vec::with_capacity(10);
        for name in BLOCK_COLUMNS {
            cols.push(column(b, name)?.as_primitive::<Int32Type>().clone());
        }
        Ok(BlockCols {
            cols: cols.try_into().expect("ten columns"),
        })
    }

    fn get(&self, i: usize, r: usize) -> Option<i32> {
        let a = &self.cols[i];
        (!a.is_null(r)).then(|| a.value(r))
    }

    /// Decode row `r`.
    pub fn block(&self, r: usize) -> Result<Block> {
        let req = |i: usize| {
            self.get(i, r)
                .ok_or_else(|| anyhow!("a null {}", BLOCK_COLUMNS[i]))
        };
        Ok(Block {
            sample_id: req(0)?,
            start: req(1)?,
            end: req(2)?,
            ad_1: self.get(3, r),
            ad_2: self.get(4, r),
            dp: self.get(5, r),
            pl_1: self.get(6, r),
            pl_2: self.get(7, r),
            pl_3: self.get(8, r),
            sample_counts: self.get(9, r),
        })
    }
}
