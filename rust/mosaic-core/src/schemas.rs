// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The Arrow schemas of the local files, exactly as the Python side
//! defines them (`*_ARROW_SCHEMA`: the partials' in `mosaic/ingest.py`,
//! the per-sample and genotype-matrix ones in `tests/sample_slices.py`):
//! field names, types, nullability; list
//! child fields nullable and named `element` (see `LIST_ITEM`).

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema, SchemaRef};

/// The child field name of every list column. pyarrow 23 names a list
/// read from Parquet after the Parquet element, which it writes as
/// `element` (`use_compliant_nested_type`), whatever `ARROW:schema` says;
/// parquet-rs names the Parquet element after the Arrow child. So the Rust
/// files carry `element`: read back, their tables then equal the Python
/// files' exactly, and the combine's output (its `ARROW:schema` included)
/// is the same bytes. With `item` the rows were equal but every combined
/// file holding a list column differed in its footer.
pub const LIST_ITEM: &str = "element";

fn list_of(t: DataType) -> DataType {
    DataType::List(Arc::new(Field::new(LIST_ITEM, t, true)))
}

fn f(name: &str, t: DataType, nullable: bool) -> Field {
    Field::new(name, t, nullable)
}

/// `SAMPLE_REF_BLOCKS_ARROW_SCHEMA`.
pub fn sample_ref_blocks() -> SchemaRef {
    use DataType::Int32;
    Arc::new(Schema::new(vec![
        f("sample_id", Int32, false),
        f("chromosome_index", Int32, false),
        f("position_start", Int32, false),
        f("position_end", Int32, false),
        f("ad_1", Int32, true),
        f("ad_2", Int32, true),
        f("dp", Int32, true),
        f("pl_1", Int32, true),
        f("pl_2", Int32, true),
        f("pl_3", Int32, true),
        f("sample_counts", Int32, true),
    ]))
}

/// `SAMPLE_SITE_INFO_ARROW_SCHEMA`.
pub fn sample_site_info() -> SchemaRef {
    use DataType::{Float64, Int32, Int64, Utf8};
    Arc::new(Schema::new(vec![
        f("sample_id", Int32, false),
        f("chromosome_index", Int32, false),
        f("position", Int32, false),
        f("del_start", Int32, true),
        f("del_end", Int32, true),
        f("alleles", list_of(Utf8), true),
        f("gt_1", Int32, true),
        f("gt_2", Int32, true),
        f("dp", Int32, true),
        f("ad", list_of(Int32), true),
        f("f1r2", list_of(Int32), true),
        f("f2r1", list_of(Int32), true),
        f("sb", list_of(Int32), true),
        f("pl", list_of(Int32), true),
        f("qual", Float64, true),
        f("site_dp", Int64, true),
        f("mqsq_1", Float64, true),
        f("mqsq_2", Int64, true),
        f("gtc", list_of(Float64), true),
        f("del_ref_spans", list_of(Int32), true),
    ]))
}

/// `BATCH_SITE_INFO_ARROW_SCHEMA`.
pub fn batch_site_info() -> SchemaRef {
    use DataType::{Float64, Int32, Int64};
    Arc::new(Schema::new(vec![
        f("batch_id", Int32, false),
        f("chromosome_index", Int32, false),
        f("position", Int32, false),
        f("qual", Float64, true),
        f("dp", Int64, true),
        f("mqsq_1", Float64, true),
        f("mqsq_2", Int64, true),
        f("sb", list_of(Int64), true),
        f("gtc", list_of(Int64), true),
        f("n_records", Int64, true),
    ]))
}

/// `BATCH_ALLELE_COUNTS_ARROW_SCHEMA`.
pub fn batch_allele_counts() -> SchemaRef {
    use DataType::{Int32, Int64, Utf8};
    Arc::new(Schema::new(vec![
        f("batch_id", Int32, false),
        f("chromosome_index", Int32, false),
        f("position", Int32, false),
        f("allele", Utf8, false),
        f("count", Int64, true),
        f("n_listed", Int64, true),
    ]))
}

/// `BATCH_REF_BLOCKS_ARROW_SCHEMA`.
pub fn batch_ref_blocks() -> SchemaRef {
    use DataType::Int32;
    Arc::new(Schema::new(vec![
        f("batch_id", Int32, false),
        f("chromosome_index", Int32, false),
        f("position", Int32, false),
        f("dp", Int32, true),
        f("sample_counts", Int32, true),
    ]))
}

/// `BATCH_GENOTYPES_ARROW_SCHEMA`.
pub fn batch_genotypes() -> SchemaRef {
    use DataType::{Binary, Int32, Utf8};
    Arc::new(Schema::new(vec![
        f("batch_id", Int32, false),
        f("chromosome_index", Int32, false),
        f("position", Int32, false),
        f("n_samples", Int32, false),
        f("alleles", list_of(Utf8), false),
        f("gt_1", Binary, false),
        f("gt_2", Binary, false),
        f("gq", Binary, false),
    ]))
}
