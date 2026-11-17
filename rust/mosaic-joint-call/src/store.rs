// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The data files' transport: `object_store` over `s3://`
//! (the manifest's endpoint, region, path style and static credentials,
//! HTTP allowed; or, under `{"from_env": true}`, the crate's default AWS
//! chain) and `file://` (the local filesystem).

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result, anyhow, bail};
use object_store::aws::AmazonS3Builder;
use object_store::local::LocalFileSystem;
use object_store::path::Path;
use object_store::{OBJECT_STORE_COALESCE_DEFAULT, ObjectStore};

use crate::manifest::S3;

/// Where a data file lives: its store, its path in it, and the gap below
/// which the store merges two ranges of one `get_ranges` call into one
/// fetch (object_store 0.14: 1 MiB on S3, the default `get_ranges`; 0 on
/// the local filesystem, which merges only touching ranges).
pub struct Location {
    pub store: Arc<dyn ObjectStore>,
    pub path: Path,
    pub coalesce_gap: u64,
}

pub struct Stores {
    s3: Option<S3>,
    buckets: Mutex<HashMap<String, Arc<dyn ObjectStore>>>,
    local: Arc<dyn ObjectStore>,
}

impl Stores {
    pub fn new(s3: Option<S3>) -> Stores {
        Stores {
            s3,
            buckets: Mutex::new(HashMap::new()),
            local: Arc::new(LocalFileSystem::new()),
        }
    }

    /// The store and path of `uri` (`s3://bucket/key`, `file:///path`).
    pub fn resolve(&self, uri: &str) -> Result<Location> {
        if let Some(rest) = uri.strip_prefix("file://") {
            let path =
                Path::from_absolute_path(rest).with_context(|| format!("{uri}: bad path"))?;
            return Ok(Location {
                store: self.local.clone(),
                path,
                coalesce_gap: 0,
            });
        }
        let Some(rest) = uri.strip_prefix("s3://") else {
            bail!("{uri}: not an s3:// or file:// URI");
        };
        let (bucket, key) = rest
            .split_once('/')
            .filter(|(b, k)| !b.is_empty() && !k.is_empty())
            .ok_or_else(|| anyhow!("{uri}: expected s3://bucket/key"))?;
        let path = Path::parse(key).with_context(|| format!("{uri}: bad key"))?;
        let mut buckets = self
            .buckets
            .lock()
            .expect("the store map is never poisoned");
        let store = match buckets.get(bucket) {
            Some(s) => s.clone(),
            None => {
                let s = self.build_s3(bucket)?;
                buckets.insert(bucket.to_string(), s.clone());
                s
            }
        };
        Ok(Location {
            store,
            path,
            coalesce_gap: OBJECT_STORE_COALESCE_DEFAULT,
        })
    }

    fn build_s3(&self, bucket: &str) -> Result<Arc<dyn ObjectStore>> {
        let Some(cfg) = &self.s3 else {
            bail!("an s3:// data file, but the manifest has no s3 settings");
        };
        let mut b = if cfg.from_env {
            AmazonS3Builder::from_env()
        } else {
            AmazonS3Builder::new()
        }
        .with_bucket_name(bucket);
        if let Some(r) = &cfg.region {
            b = b.with_region(r);
        }
        if let Some(e) = &cfg.endpoint {
            b = b.with_endpoint(e).with_allow_http(e.starts_with("http://"));
        }
        if let Some(k) = &cfg.access_key_id {
            b = b.with_access_key_id(k);
        }
        if let Some(k) = &cfg.secret_access_key {
            b = b.with_secret_access_key(k);
        }
        if cfg.path_style {
            b = b.with_virtual_hosted_style_request(false);
        }
        Ok(Arc::new(b.build().with_context(|| {
            format!("S3 store for bucket {bucket}")
        })?))
    }
}
