// Copyright (c) 2026 Sentieon Inc. All rights reserved

//! The binary's one test seam, `MOSAIC_TEST_FAIL_JOINT_CALL_UNIT` (J3; the
//! ingest worker's `MOSAIC_TEST_FAIL_RUST_SHARD` pattern): the calling
//! run's unit whose contig name and 1-based start match exactly fails
//! before it reads anything, as an ordinary unit error.

use anyhow::{Result, anyhow};

/// The seam's environment variable.
pub const FAIL_HOOK_ENV: &str = "MOSAIC_TEST_FAIL_JOINT_CALL_UNIT";

/// The unit to fail: its contig name and 1-based start (`ws`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FailUnit {
    pub contig: String,
    pub ws: i64,
}

/// Parse `MOSAIC_TEST_FAIL_JOINT_CALL_UNIT`: `<contig>:<ws>` (split at the
/// last colon, so a contig name may hold one), `ws` a positive integer.
/// Unset or empty: None. A malformed value (no colon, an empty contig, a
/// start that is not a positive integer) is an error.
pub fn parse_fail_unit(value: Option<&str>) -> Result<Option<FailUnit>> {
    let Some(v) = value.filter(|v| !v.is_empty()) else {
        return Ok(None);
    };
    let bad = || anyhow!("{FAIL_HOOK_ENV}={v:?}: expected <contig>:<start>, start >= 1");
    let (contig, ws) = v.rsplit_once(':').ok_or_else(bad)?;
    if contig.is_empty() {
        return Err(bad());
    }
    let ws: i64 = ws.parse().map_err(|_| bad())?;
    if ws < 1 {
        return Err(bad());
    }
    Ok(Some(FailUnit {
        contig: contig.to_string(),
        ws,
    }))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fail_unit_parser() {
        assert_eq!(parse_fail_unit(None).unwrap(), None);
        assert_eq!(parse_fail_unit(Some("")).unwrap(), None);
        let f = |c: &str, ws| {
            Some(FailUnit {
                contig: c.to_string(),
                ws,
            })
        };
        assert_eq!(
            parse_fail_unit(Some("chr21:10000001")).unwrap(),
            f("chr21", 10_000_001)
        );
        assert_eq!(parse_fail_unit(Some("chr21:1")).unwrap(), f("chr21", 1));
        assert_eq!(
            parse_fail_unit(Some("HLA-A*01:01:1")).unwrap(),
            f("HLA-A*01:01", 1)
        );
        for bad in [
            "chr21",
            ":5",
            "chr21:",
            "chr21:x",
            "chr21:0",
            "chr21:-1",
            "chr21:1.5",
            "chr21: 1",
        ] {
            let e = parse_fail_unit(Some(bad)).unwrap_err().to_string();
            assert!(e.contains(FAIL_HOOK_ENV), "{bad}: {e}");
        }
    }
}
