// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! `mosaic-core`: the modules the two Rust engines (the ingest worker
//! and the joint caller) share, moved from the ingest worker: the Arrow schemas of the lake's tables, the allele encoding,
//! the thread queue and the manifest loader.

pub mod alleles;
pub mod manifest;
pub mod queue;
pub mod schemas;
