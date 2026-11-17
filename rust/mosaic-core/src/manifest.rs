// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The manifest loader both binaries use: a JSON file read and parsed into
//! the binary's own manifest type (`#[serde(deny_unknown_fields)]` on
//! their structs refuses a field the binary does not know), with the path
//! in every error. The types and their validation stay with each binary
//! (moved from the worker's `manifest.rs`).

use std::path::Path;

use anyhow::{Context, Result};
use serde::de::DeserializeOwned;

/// Read `path` and parse it as JSON into `T`.
pub fn load_json<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parse {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Deserialize;

    #[derive(Debug, Deserialize, PartialEq)]
    #[serde(deny_unknown_fields)]
    struct M {
        a: i64,
    }

    #[test]
    fn load_and_refuse() {
        let dir = std::env::temp_dir().join(format!("mosaic-core-manifest-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("m.json");
        std::fs::write(&p, r#"{"a": 3}"#).unwrap();
        assert_eq!(load_json::<M>(&p).unwrap(), M { a: 3 });
        std::fs::write(&p, r#"{"a": 3, "b": 1}"#).unwrap();
        let e = format!("{:#}", load_json::<M>(&p).unwrap_err());
        assert!(e.contains("parse") && e.contains("m.json"), "{e}");
        let missing = dir.join("none.json");
        let e = format!("{:#}", load_json::<M>(&missing).unwrap_err());
        assert!(e.contains("read") && e.contains("none.json"), "{e}");
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
