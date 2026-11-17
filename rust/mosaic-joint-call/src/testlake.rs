// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! Test support: a crafted local lake (per-sample data files and the two
//! exports as Parquet with statistics, `file://` URIs) and a manifest over
//! it, for the calling run's unit tests.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float64Array, Int32Array, Int64Array, ListBuilder, RecordBatch, StringArray,
    StringBuilder, new_null_array,
};
use arrow::array::{Int32Builder, builder::ArrayBuilder};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use parquet::arrow::ArrowWriter;
use parquet::file::properties::{EnabledStatistics, WriterProperties};

use crate::exports::{ALLELES_COLUMNS, SITES_COLUMNS};
use crate::manifest::{Contig, DataFile, Manifest, Sample};
use crate::rows::{Block, SiteRow};

pub fn tmpdir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("mosaic-jc-{tag}-{}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn write(path: &Path, b: &RecordBatch, rows_per_group: usize) -> DataFile {
    let props = WriterProperties::builder()
        .set_max_row_group_row_count(Some(rows_per_group.max(1)))
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .build();
    let f = std::fs::File::create(path).unwrap();
    let mut w = ArrowWriter::try_new(f, b.schema(), Some(props)).unwrap();
    w.write(b).unwrap();
    w.close().unwrap();
    DataFile {
        uri: format!("file://{}", path.display()),
        record_count: b.num_rows() as u64,
        file_size_in_bytes: std::fs::metadata(path).unwrap().len(),
        delete_files: vec![],
    }
}

fn child(schema: &SchemaRef, name: &str) -> Arc<Field> {
    match schema.field_with_name(name).unwrap().data_type() {
        DataType::List(f) => f.clone(),
        t => panic!("{name} is {t}"),
    }
}

fn int_list(schema: &SchemaRef, name: &str, v: &[Option<Vec<Option<i32>>>]) -> ArrayRef {
    let mut b = ListBuilder::new(Int32Builder::new()).with_field(child(schema, name));
    for x in v {
        match x {
            None => b.append_null(),
            Some(xs) => {
                for e in xs {
                    b.values().append_option(*e);
                }
                b.append(true);
            }
        }
    }
    Arc::new(b.finish())
}

/// A `sample_site_info` data file of `rows` (sorted by the caller).
pub fn site_file(
    dir: &Path,
    name: &str,
    chrom: i32,
    rows: &[(i32, SiteRow)],
    rpg: usize,
) -> DataFile {
    let schema = mosaic_core::schemas::sample_site_info();
    let n = rows.len();
    let mut alleles = ListBuilder::new(StringBuilder::new()).with_field(child(&schema, "alleles"));
    for (_, r) in rows {
        match &r.alleles {
            None => alleles.append_null(),
            Some(a) => {
                for x in a {
                    alleles.values().append_value(x);
                }
                alleles.append(true);
            }
        }
    }
    let _ = alleles.len();
    let alleles: ArrayRef = Arc::new(alleles.finish());
    let col = |f: &Field| -> ArrayRef {
        let i32s = |g: &dyn Fn(&SiteRow) -> Option<i32>| -> ArrayRef {
            Arc::new(Int32Array::from(
                rows.iter().map(|(_, r)| g(r)).collect::<Vec<_>>(),
            ))
        };
        match f.name().as_str() {
            "sample_id" => i32s(&|r| Some(r.sample_id)),
            "chromosome_index" => Arc::new(Int32Array::from(vec![chrom; n])),
            "position" => Arc::new(Int32Array::from(
                rows.iter().map(|(p, _)| *p).collect::<Vec<_>>(),
            )),
            "alleles" => alleles.clone(),
            "gt_1" => i32s(&|r| r.gt_1),
            "gt_2" => i32s(&|r| r.gt_2),
            "dp" => i32s(&|r| r.dp),
            "ad" => int_list(
                &schema,
                "ad",
                &rows.iter().map(|(_, r)| r.ad.clone()).collect::<Vec<_>>(),
            ),
            "f1r2" => int_list(
                &schema,
                "f1r2",
                &rows.iter().map(|(_, r)| r.f1r2.clone()).collect::<Vec<_>>(),
            ),
            "f2r1" => int_list(
                &schema,
                "f2r1",
                &rows.iter().map(|(_, r)| r.f2r1.clone()).collect::<Vec<_>>(),
            ),
            "pl" => int_list(
                &schema,
                "pl",
                &rows.iter().map(|(_, r)| r.pl.clone()).collect::<Vec<_>>(),
            ),
            _ => new_null_array(f.data_type(), n),
        }
    };
    let arrays: Vec<ArrayRef> = schema.fields().iter().map(|f| col(f)).collect();
    let b = RecordBatch::try_new(schema.clone(), arrays).unwrap();
    write(&dir.join(name), &b, rpg)
}

/// A `sample_ref_blocks` data file of `blocks` (sorted by the caller).
pub fn block_file(dir: &Path, name: &str, chrom: i32, blocks: &[Block], rpg: usize) -> DataFile {
    let schema = mosaic_core::schemas::sample_ref_blocks();
    let n = blocks.len();
    let col = |f: &Field| -> ArrayRef {
        let g: fn(&Block) -> Option<i32> = match f.name().as_str() {
            "sample_id" => |b| Some(b.sample_id),
            "position_start" => |b| Some(b.start),
            "position_end" => |b| Some(b.end),
            "ad_1" => |b| b.ad_1,
            "ad_2" => |b| b.ad_2,
            "dp" => |b| b.dp,
            "pl_1" => |b| b.pl_1,
            "pl_2" => |b| b.pl_2,
            "pl_3" => |b| b.pl_3,
            "sample_counts" => |b| b.sample_counts,
            "chromosome_index" => return Arc::new(Int32Array::from(vec![chrom; n])),
            _ => return new_null_array(f.data_type(), n),
        };
        Arc::new(Int32Array::from(blocks.iter().map(g).collect::<Vec<_>>()))
    };
    let arrays: Vec<ArrayRef> = schema.fields().iter().map(|f| col(f)).collect();
    let b = RecordBatch::try_new(schema.clone(), arrays).unwrap();
    write(&dir.join(name), &b, rpg)
}

fn export_schema(cols: &[(&str, DataType)]) -> SchemaRef {
    Arc::new(Schema::new(
        cols.iter()
            .map(|(n, t)| Field::new(*n, t.clone(), true))
            .collect::<Vec<_>>(),
    ))
}

/// A `sites` row: (position, qual, dp, mqsq_1, mqsq_2, sb, gtc, rb_dp,
/// rb_sample_counts).
pub type SiteExportRow = (
    i32,
    Option<f64>,
    Option<i64>,
    Option<f64>,
    Option<i64>,
    Option<[i64; 4]>,
    Option<[i64; 3]>,
    i64,
    i64,
);

pub fn sites_export(path: &Path, rows: &[SiteExportRow], rpg: usize) -> String {
    let schema = export_schema(SITES_COLUMNS);
    let i64s = |g: &dyn Fn(&SiteExportRow) -> Option<i64>| -> ArrayRef {
        Arc::new(Int64Array::from(rows.iter().map(g).collect::<Vec<_>>()))
    };
    let mut arrays: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(Float64Array::from(
            rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        )),
        i64s(&|r| r.2),
        Arc::new(Float64Array::from(
            rows.iter().map(|r| r.3).collect::<Vec<_>>(),
        )),
        i64s(&|r| r.4),
    ];
    for k in 0..4 {
        arrays.push(i64s(&|r| r.5.map(|s| s[k])));
    }
    for k in 0..3 {
        arrays.push(i64s(&|r| r.6.map(|s| s[k])));
    }
    arrays.push(i64s(&|r| Some(r.7)));
    arrays.push(i64s(&|r| Some(r.8)));
    let b = RecordBatch::try_new(schema, arrays).unwrap();
    write(path, &b, rpg);
    path.display().to_string()
}

/// An `alleles` row: (position, allele, count, del).
pub type AlleleExportRow = (i32, &'static str, i64, Option<(i32, i32)>);

pub fn alleles_export(path: &Path, rows: &[AlleleExportRow], rpg: usize) -> String {
    let schema = export_schema(ALLELES_COLUMNS);
    let arrays: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.0).collect::<Vec<_>>(),
        )),
        Arc::new(StringArray::from(
            rows.iter().map(|r| r.1).collect::<Vec<_>>(),
        )),
        Arc::new(Int64Array::from(
            rows.iter().map(|r| r.2).collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.3.map(|d| d.0)).collect::<Vec<_>>(),
        )),
        Arc::new(Int32Array::from(
            rows.iter().map(|r| r.3.map(|d| d.1)).collect::<Vec<_>>(),
        )),
    ];
    let b = RecordBatch::try_new(schema, arrays).unwrap();
    write(path, &b, rpg);
    path.display().to_string()
}

/// A one-contig manifest over local files.
pub fn manifest(
    output: &Path,
    samples: &[(i32, &str)],
    contig: (&str, i64),
    sites: String,
    alleles: String,
    ssi: Vec<DataFile>,
    srb: Vec<DataFile>,
) -> Manifest {
    let names: Vec<&str> = samples.iter().map(|s| s.1).collect();
    Manifest {
        output: Some(output.display().to_string()),
        header_lines: Some(vec![
            "##fileformat=VCFv4.2".to_string(),
            format!("##contig=<ID={},length={}>", contig.0, contig.1),
            format!(
                "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t{}",
                names.join("\t")
            ),
        ]),
        samples: samples
            .iter()
            .map(|(id, name)| Sample {
                id: *id,
                name: name.to_string(),
            })
            .collect(),
        removed_ids: vec![],
        contigs: vec![Contig {
            index: 0,
            name: contig.0.to_string(),
            length: contig.1,
            sites: Some(sites),
            alleles: Some(alleles),
            sample_site_info: ssi,
            sample_ref_blocks: srb,
        }],
        s3: None,
        window: None,
        region: None,
        tmpdir: None,
        spill_dir: None,
        stale: false,
        subset: false,
    }
}
